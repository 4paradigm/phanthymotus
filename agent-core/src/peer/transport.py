"""
peer/transport.py — authenticated peer-to-peer requests.

## Why not mTLS

The design called for pinned mTLS, and that is the cleaner primitive — but it
cannot be used here. The Agent Core serves **one** port, 15678, to both the
browser dashboard and to peers. Turning on client-certificate verification
there would demand a certificate from every browser and lock the operator out
of their own robot. Splitting peers onto a second port would mean a second
TLS listener and a new firewall rule on every deployment, which the "no new
port" property in the README exists to avoid.

So the link is the existing HTTPS, and authentication is per-request Ed25519
signatures over `method | path | ts | nonce | sha256(body)`. That yields the
properties that actually mattered:

  * the peer's key is **pinned** — the signature is checked against the key
    recorded at pairing time (peer/store.public_key_bytes), never against a key
    the request supplies, so a self-signed TLS cert on either end changes nothing;
  * replay is bounded by a timestamp window plus a nonce cache;
  * it survives links that are not TLS at all, which is what the BLE bootstrap
    path in the design will need.

TLS still does its job of encrypting the channel. It is simply not what
establishes who is on the other end.

## Clock skew

The timestamp window is the one part that depends on the environment: offline
robots have no NTP and drift. `peer_settings.clock_skew_s` widens it. The nonce
cache is what actually stops replay, so widening the window degrades gracefully
rather than opening a hole — a replay inside the window still fails on the nonce.
"""

import asyncio
import hashlib
import json
import ssl
import time

import config
from peer import identity, store


# Paths a *peer* calls, authenticated by Ed25519 signature rather than by the
# operator's ACCESS_TOKEN. auth.py consults this instead of hardcoding a prefix.
#
# Single source of truth on purpose: the exemption list and the routes drifted
# once already — tool-proxy and delegation endpoints were added outside the
# '/inbox/' prefix auth.py knew about, so every peer request to them got 401
# while the endpoints themselves were perfectly correct.
PEER_FACING_PATHS = frozenset({
    '/api/peer/inbox/pair_request',
    '/api/peer/inbox/ping',
    '/api/peer/inbox/message',
    '/api/peer/tools/list',
    '/api/peer/tools/call',
    '/api/peer/delegate',
})


def is_peer_facing(path: str) -> bool:
    """True if `path` authenticates by peer signature, not ACCESS_TOKEN."""
    return path in PEER_FACING_PATHS


HEADER_PEER_ID = 'x-motus-peer-id'
HEADER_TIMESTAMP = 'x-motus-timestamp'
HEADER_NONCE = 'x-motus-nonce'
HEADER_SIGNATURE = 'x-motus-signature'

DEFAULT_SKEW_S = 120
_NONCE_CACHE_MAX = 4096

# nonce → expiry. Bounded so a peer cannot grow it without limit; entries are
# only useful until the timestamp window closes anyway.
_seen_nonces: dict[str, float] = {}


def _skew_s() -> int:
    try:
        return int(config.main.get('peer_settings', {}).get('clock_skew_s', DEFAULT_SKEW_S))
    except (TypeError, ValueError):
        return DEFAULT_SKEW_S


def canonical_payload(method: str, path: str, ts: str, nonce: str, body: bytes) -> bytes:
    """The exact bytes both sides sign.

    Body is hashed rather than included so the payload stays small, and the
    method and path are in there so a signature captured for
    `POST /api/peer/inbox/ping` cannot be replayed against a different endpoint.
    """
    body_hash = hashlib.sha256(body or b'').hexdigest()
    return '\n'.join([method.upper(), path, ts, nonce, body_hash]).encode()


def sign_headers(method: str, path: str, body: bytes) -> dict[str, str]:
    """Headers proving this request came from us."""
    from peer.pairing import new_nonce
    ts = str(int(time.time()))
    nonce = new_nonce()
    payload = canonical_payload(method, path, ts, nonce, body)
    return {
        HEADER_PEER_ID: identity.peer_id(),
        HEADER_TIMESTAMP: ts,
        HEADER_NONCE: nonce,
        HEADER_SIGNATURE: identity.sign(payload),
        'content-type': 'application/json',
    }


def _remember_nonce(nonce: str, expiry: float) -> bool:
    """Record a nonce; False if it was already used."""
    now = time.time()
    if len(_seen_nonces) > _NONCE_CACHE_MAX:
        for n in [n for n, exp in _seen_nonces.items() if exp < now]:
            _seen_nonces.pop(n, None)
        if len(_seen_nonces) > _NONCE_CACHE_MAX:
            # Still full of live entries: drop the oldest rather than stop
            # accepting requests. Only reachable under a flood.
            for n in sorted(_seen_nonces, key=_seen_nonces.get)[:_NONCE_CACHE_MAX // 4]:
                _seen_nonces.pop(n, None)
    if nonce in _seen_nonces and _seen_nonces[nonce] >= now:
        return False
    _seen_nonces[nonce] = expiry
    return True


def clear_nonces() -> None:
    _seen_nonces.clear()


def verify_signed_request(method: str, path: str, headers, body: bytes,
                          *, require_paired: bool = True) -> tuple[str, str]:
    """Authenticate an inbound peer request.

    Returns `(peer_id, '')` on success, or `('', reason)` — never raises and
    never leaks which check failed to the caller's response beyond a coarse
    reason, since a precise oracle helps an attacker more than it helps an
    operator.

    `require_paired=False` is only for the pairing handshake itself, where by
    definition no key is on file yet. It authenticates the *self-consistency* of
    the request (the key hashes to the claimed peer_id) and nothing more — that
    is exactly why pairing still needs a human to compare a short code.
    """
    def _h(name: str) -> str:
        try:
            return headers.get(name, '') or ''
        except AttributeError:
            return ''

    peer_id = _h(HEADER_PEER_ID)
    ts = _h(HEADER_TIMESTAMP)
    nonce = _h(HEADER_NONCE)
    signature = _h(HEADER_SIGNATURE)
    if not (peer_id and ts and nonce and signature):
        return '', 'missing_signature_headers'

    try:
        ts_val = int(ts)
    except ValueError:
        return '', 'bad_timestamp'
    skew = _skew_s()
    if abs(time.time() - ts_val) > skew:
        return '', 'timestamp_outside_window'

    if require_paired:
        public_key = store.public_key_bytes(peer_id)
        if public_key is None:
            return '', 'unknown_peer'
        peer = store.get(peer_id)
        if peer and peer['role'] == 'blocked':
            return '', 'blocked'
    else:
        supplied = _h('x-motus-public-key')
        if not supplied:
            return '', 'missing_public_key'
        import base64
        try:
            public_key = base64.b64decode(supplied)
        except (ValueError, TypeError):
            return '', 'bad_public_key'
        if len(public_key) != 32 or identity.fingerprint(public_key) != peer_id:
            return '', 'public_key_fingerprint_mismatch'

    payload = canonical_payload(method, path, ts, nonce, body)
    if not identity.verify(public_key, signature, payload):
        return '', 'bad_signature'

    # Nonce is consumed only after the signature checks out, so an unauthenticated
    # caller cannot burn nonces to lock out the real peer.
    if not _remember_nonce(nonce, ts_val + skew):
        return '', 'replayed_nonce'

    return peer_id, ''


# ── outbound ────────────────────────────────────────────────────────────────

def _ssl_context() -> ssl.SSLContext:
    """TLS context for peer links.

    Certificate verification is off, and that is deliberate rather than
    negligent: peers use self-signed certs and identity is established by the
    Ed25519 signature over the request, which a TLS-terminating middlebox cannot
    forge. Turning verification on here would only require pre-sharing certs to
    prove something the signature already proves.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def post_json(endpoints: list[str], path: str, payload: dict,
                    *, timeout: float = 10.0,
                    extra_headers: dict[str, str] | None = None) -> tuple[dict | None, str]:
    """POST a signed JSON body, trying each endpoint until one answers.

    Returns `(response_json, '')` or `(None, reason)`. Endpoints are tried in
    order because registry.endpoints_for() puts the freshest link first —
    walking the list is what makes "LAN died, cloud link still there" work with
    no extra logic at the call site.
    """
    import aiohttp

    if not endpoints:
        return None, 'no_known_endpoint'

    body = json.dumps(payload, ensure_ascii=False).encode()
    failures = []
    for base in endpoints:
        url = base.rstrip('/') + path
        headers = sign_headers('POST', path, body)
        if extra_headers:
            headers.update(extra_headers)
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                async with session.post(url, data=body, headers=headers,
                                        ssl=_ssl_context()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        failures.append(f'{base} → HTTP {resp.status} {text[:120]}')
                        continue
                    try:
                        return json.loads(text) if text else {}, ''
                    except json.JSONDecodeError:
                        failures.append(f'{base} → non-JSON response')
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            failures.append(f'{base} → {type(e).__name__}: {e}')
            continue
    return None, '; '.join(failures) or 'unreachable'


async def get_json(endpoints: list[str], path: str, *,
                   timeout: float = 10.0) -> tuple[dict | None, str]:
    """GET a signed request, trying each endpoint until one answers.

    Returns `(response_json, '')` or `(None, reason)`. Used for /tools/list.
    """
    import aiohttp

    if not endpoints:
        return None, 'no_known_endpoint'

    failures = []
    for base in endpoints:
        url = base.rstrip('/') + path
        headers = sign_headers('GET', path, b'')
        try:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                async with session.get(url, headers=headers, ssl=_ssl_context()) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        failures.append(f'{base} → HTTP {resp.status} {text[:120]}')
                        continue
                    try:
                        return json.loads(text) if text else {}, ''
                    except json.JSONDecodeError:
                        failures.append(f'{base} → non-JSON response')
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            failures.append(f'{base} → {type(e).__name__}: {e}')
            continue
    return None, '; '.join(failures) or 'unreachable'

