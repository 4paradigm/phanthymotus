"""
api/peer.py — peer discovery, pairing and messaging endpoints.

## Two sides

**Dashboard → this Agent Core** — authenticated with ACCESS_TOKEN per auth.py,
exactly like every other dashboard-facing endpoint. These tell the operator what
peers are out there, kick off pairing, list paired peers, and mutate their roles.

**Peer Agent Core → this Agent Core** — authenticated with Ed25519 signatures
per peer/transport.py. The pairing handshake is the only request type that comes
from an unpaired peer, so it runs `verify_signed_request(require_paired=False)`
and relies on the human comparing a short code. Everything else requires the
peer be in the `peers` table, and the signature is checked against the key we
recorded at pairing time — that pin is what makes trust durable.

The `/api/peer/inbox/*` paths are exempt from `auth.py` middleware because they
carry their own authentication. See the rule in auth.py and the CRITICAL comment
in start.py's router ordering.
"""

import base64
import time

import fastapi
from fastapi import Request
from pydantic import BaseModel

from peer import identity, pairing, store, transport
from peer.registry import registry
from peer import tools as peer_tools
from peer import delegation
from channel.adapters import lan
import mcp_client
from subagent.protocol import SubagentSpec


# NOTE: no '/api' here — start.py mounts app_api at '/api', so this router's
# prefix is relative to that. Spelling it '/api/peer' produced '/api/api/peer'
# and every endpoint 404'd while the process looked perfectly healthy.
router = fastapi.APIRouter(prefix='/peer', tags=['peer'])


def _local_display_name() -> str:
    """What to call this agent on the other side's pairing screen.

    Falls back to the hostname: an operator confronted with two bare
    fingerprints has no way to tell which robot is which.
    """
    import socket
    import config
    return (config.main.get('peer_settings', {}).get('display_name')
            or socket.gethostname())


def _peer_label(peer_id: str) -> str:
    """Best known name for a peer — whatever discovery advertised, else the id."""
    advert = registry.get(peer_id)
    if advert is not None and advert.display_name:
        return advert.display_name
    return peer_id[:12]


# ── dashboard-facing (ACCESS_TOKEN) ──────────────────────────────────────────

@router.get('/identity')
async def get_identity():
    """This agent's durable identity. The fingerprint is its peer_id."""
    ident = identity.ensure_identity()
    return {
        'peer_id': ident['peer_id'],
        'public_key': ident['public_key'],
        'created_at': ident['created_at'],
    }


@router.get('/settings')
async def get_settings():
    """Peer settings as the dashboard needs them.

    display_name is echoed resolved (hostname when unset) so the panel can show
    what peers will actually see, rather than a blank field that means
    "something else".
    """
    import config
    s = config.main.get('peer_settings', {}) or {}
    disc = s.get('discovery') or {}
    return {
        'enabled': bool(s.get('enabled', False)),
        'display_name': s.get('display_name', ''),
        'resolved_display_name': _local_display_name(),
        'discovery': {
            'mdns': bool(disc.get('mdns', True)),
            'static': disc.get('static') or [],
            'ble': bool(disc.get('ble', False)),
        },
        'default_role': s.get('default_role', 'viewer'),
        'clock_skew_s': s.get('clock_skew_s', 120),
    }


# Agent Core's port. A hand-typed address is much more often "10.100.121.14" than
# a full URL, and defaulting saves the operator from a broken entry that only
# shows up later as a peer that never appears.
_DEFAULT_PEER_PORT = 15678


def _normalize_static(entries: list) -> list[dict]:
    """Clean up hand-entered peer addresses.

    Accepts `host`, `host:port`, `https://host:port`, or `{url, display_name}`, and
    normalises to `{'url': 'https://host:port', 'display_name': ...}`. Written when
    this list became editable from the dashboard: it used to be stored verbatim, so
    a missing scheme produced an advert nothing could ever reach and no error
    anywhere — the peer simply never showed up.
    """
    from urllib.parse import urlparse

    out, seen = [], set()
    for entry in entries or []:
        if isinstance(entry, str):
            raw, name = entry.strip(), ''
        elif isinstance(entry, dict):
            raw = str(entry.get('url', '')).strip()
            name = str(entry.get('display_name', '')).strip()
        else:
            raise fastapi.HTTPException(400, 'each static entry must be a string or an object')
        if not raw:
            continue

        candidate = raw if '://' in raw else f'https://{raw}'
        parsed = urlparse(candidate)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise fastapi.HTTPException(
                400, f'"{raw}" is not an address — use host, host:port or https://host:port')
        port = parsed.port or _DEFAULT_PEER_PORT
        url = f'{parsed.scheme}://{parsed.hostname}:{port}'
        if url in seen:
            continue
        seen.add(url)
        item = {'url': url}
        if name:
            item['display_name'] = name
        out.append(item)
    return out


class PeerSettingsReq(BaseModel):
    enabled: bool | None = None
    display_name: str | None = None
    mdns: bool | None = None
    static: list | None = None
    ble: bool | None = None
    default_role: str | None = None
    clock_skew_s: int | None = None


@router.post('/settings')
async def save_settings(req: PeerSettingsReq):
    """Update peer settings and apply them without a container restart.

    The discovery layer is torn down and rebuilt in place. Until this existed
    the only way to turn peering on was editing the SQLite config by hand and
    restarting the container, which is not something an operator can be asked
    to do.
    """
    import config
    from channel.acl import ROLES

    s = dict(config.main.get('peer_settings', {}) or {})
    disc = dict(s.get('discovery') or {})

    if req.default_role is not None:
        if req.default_role not in ROLES or req.default_role == 'owner':
            raise fastapi.HTTPException(
                400, f'default_role must be one of viewer/operator/blocked')
        s['default_role'] = req.default_role
    if req.enabled is not None:
        s['enabled'] = bool(req.enabled)
    if req.display_name is not None:
        s['display_name'] = req.display_name.strip()
    if req.clock_skew_s is not None:
        if req.clock_skew_s < 5:
            raise fastapi.HTTPException(400, 'clock_skew_s must be at least 5 seconds')
        s['clock_skew_s'] = int(req.clock_skew_s)
    if req.mdns is not None:
        disc['mdns'] = bool(req.mdns)
    if req.static is not None:
        disc['static'] = _normalize_static(req.static)
    if req.ble is not None:
        disc['ble'] = bool(req.ble)
    s['discovery'] = disc
    config.main['peer_settings'] = s

    # Rebuild discovery in place. stop() is safe when nothing is running.
    try:
        await registry.stop()
        registry.reset()
        await registry.start()
        applied, apply_error = True, ''
    except Exception as e:
        applied, apply_error = False, f'{type(e).__name__}: {e}'
        print(f'[peer] settings applied to config but discovery restart failed: {apply_error}')

    return {
        'settings': await get_settings(),
        'discovery_restarted': applied,
        'error': apply_error,
        'providers': registry.provider_status(),
    }


@router.get('/discovered')
async def list_discovered(include_paired: bool = True):
    """Peers seen by any discovery provider, freshest first.

    `include_paired=false` narrows to unpaired peers, which is the pairing
    screen's default filter. Already-paired peers are still *discovered* (the
    registry keeps tracking them) so the LAN link for a paired peer is always
    current even after a restart.
    """
    return {'peers': registry.discovered(include_paired=include_paired)}


@router.get('/paired')
async def list_paired():
    """Peers we have paired with, what they may do, and whether they can do it now.

    Liveness is attached here rather than left to the UI: pairing is durable and
    reachability is not, and the two were indistinguishable on screen — the only
    dot in that row encodes the *role*, so an `operator` peer showed a green dot
    whether it was running, idle or switched off.

    `agent_running` is a third fact again: a peer with 智能控制 off keeps pushing
    state and answering tool calls, but /delegate returns 503.
    """
    from peer import dds_state as _state
    from peer import liveness as _liveness, naming as _naming
    from peer import mcp_bridge as _bridge
    peers = store.list_peers()
    labels = _naming.labels(peers)
    out = []
    for p in peers:
        live = _liveness.liveness(p)
        out.append({
            **p,
            'label': labels[p['peer_id']],
            'online': live['online'],
            'agent_running': live['agent_running'],
            'contact_age_s': live['contact_age_s'],
            # What this agent can currently call on that peer. Worth surfacing:
            # /api/mcp is built from the configured device list, so the synthetic
            # peer entries never appear there, and a bridge offering nothing looks
            # exactly like one that works.
            'tools_offered': _bridge.offered.get(p['peer_id'], []),
            # Why our last state push to it failed. A 403 here means the other side
            # has us in no peers table — i.e. nobody approved the pairing over
            # there, which otherwise looks identical to a healthy pairing from here.
            'last_push_error': _state.push_errors.get(p['peer_id'], ''),
            # False until the peer proves it has us as well. Confirming here writes
            # only this side's record, so a pairing is genuinely half-done until
            # then — and it used to render as finished.
            'mutual': bool(p.get('mutual_at')),
            # Timestamp for when mutual was last confirmed, or null. Used to tell
            # "never confirmed" from "was mutual, now they unpaired us" — both cause
            # 403s but need different UI text.
            'mutual_at': p.get('mutual_at'),
        })
    return {'peers': out}


def _adopt_static_peer_id(endpoints: list[str], peer_id: str) -> None:
    """Write a proven fingerprint back onto the static entry it came from.

    The static provider re-reads config every minute, so without this it keeps
    re-emitting the provisional `static:<url>` advert: the same machine shows up
    again as an unpaired "manual address" row, `registry.get(real_id)` finds
    nothing, and the peer reads as offline on a link that is in fact working.
    `StaticProvider` already honours a `peer_id` key on the entry — this is the
    step that ever filled it in.
    """
    import config

    s = dict(config.main.get('peer_settings', {}) or {})
    disc = dict(s.get('discovery') or {})
    entries = list(disc.get('static') or [])
    changed = False
    for i, entry in enumerate(entries):
        url = entry if isinstance(entry, str) else (entry or {}).get('url', '')
        if url not in endpoints:
            continue
        item = {'url': url} if isinstance(entry, str) else dict(entry)
        if item.get('peer_id') == peer_id:
            continue
        item['peer_id'] = peer_id
        entries[i] = item
        changed = True
    if not changed:
        return
    disc['static'] = entries
    s['discovery'] = disc
    config.main['peer_settings'] = s
    registry.refresh_provider('static')


def _source_endpoint(req: Request) -> str:
    """The address an authenticated peer request arrived from.

    A fallback for learning where a peer lives when nothing else knows — see
    store.touch(). Not authoritative: behind a proxy or NAT this is the middlebox.
    """
    host = req.client.host if req.client else ''
    return f'https://{host}:{_DEFAULT_PEER_PORT}' if host else ''


def _local_endpoints() -> list[str]:
    """How this agent can be reached, best guess first.

    Sent in a pair request so the far side has an address even when it never
    discovered us (mDNS does not cross subnets). Derived from the same primary-IP
    logic mDNS advertises with, so the two agree.
    """
    try:
        from peer.discovery.mdns import MdnsProvider
        ip = MdnsProvider._primary_ip()
    except Exception:
        ip = ''
    return [f'https://{ip}:{_DEFAULT_PEER_PORT}'] if ip else []


def _inbound_endpoints(req: Request, payload: dict, peer_id: str) -> list[str]:
    """Where to reach the peer that just asked us to pair.

    Three sources, in order: what it told us, what discovery already knows, and
    the address the request came from. The last one is a fallback rather than the
    truth because a proxy or NAT would make it wrong — but having *something* beats
    an empty list, which is what a cross-subnet pairing used to store.
    """
    given = [e for e in (payload.get('endpoints') or []) if isinstance(e, str) and e.strip()]
    known = registry.endpoints_for(peer_id)
    seen, out = set(), []
    for ep in [*given, *known]:
        ep = ep.strip()
        if ep and ep not in seen:
            seen.add(ep)
            out.append(ep)
    if not out and req.client and req.client.host:
        out.append(f'https://{req.client.host}:{_DEFAULT_PEER_PORT}')
    return out


class StartPairingReq(BaseModel):
    peer_id: str


@router.post('/pair/start')
async def start_pairing(req: StartPairingReq):
    """Begin pairing: generate our nonce, request theirs, derive the short code.

    Returns the 6-digit code the operator compares on both screens. The code
    expires in 5 minutes; that timeout is what makes an unattended robot refuse
    to complete a pairing approved before a restart.
    """
    advert = registry.get(req.peer_id)
    if advert is None:
        raise fastapi.HTTPException(404, 'peer not discovered')

    endpoints = registry.endpoints_for(req.peer_id)
    if not endpoints:
        raise fastapi.HTTPException(400, 'peer has no known endpoint')

    local_nonce = pairing.new_nonce()
    local_pubkey = identity.public_key_b64()
    payload = {
        'nonce': local_nonce,
        'public_key': local_pubkey,
        'display_name': _local_display_name(),
        # Our own reachable addresses. The receiver discovered us over mDNS or not
        # at all — across subnets it is the latter, and it then stored no endpoints
        # for us, so the reverse direction had no address to use: state pushes, tool
        # calls and delegation could only ever go one way. Measured between Tianyi
        # and Orin5: 43 pushes arrived one way in four minutes, zero came back, and
        # the peer showed as offline on the side that could not reach.
        'endpoints': _local_endpoints(),
    }
    result, err = await transport.post_json(
        endpoints, '/api/peer/inbox/pair_request', payload,
        extra_headers={'x-motus-public-key': local_pubkey},
        timeout=10.0,
    )
    if result is None:
        raise fastapi.HTTPException(503, f'peer unreachable: {err}')

    remote_nonce = result.get('nonce', '')
    remote_pubkey = result.get('public_key', '')
    if not remote_nonce or not remote_pubkey:
        raise fastapi.HTTPException(502, 'peer response missing nonce or public_key')

    try:
        remote_pubkey_raw = base64.b64decode(remote_pubkey)
    except (ValueError, TypeError):
        raise fastapi.HTTPException(502, 'peer public_key is not valid base64')
    if len(remote_pubkey_raw) != 32:
        raise fastapi.HTTPException(502, 'peer public_key is not 32 bytes')
    remote_peer_id = identity.fingerprint(remote_pubkey_raw)
    if remote_peer_id != req.peer_id:
        # A hand-entered address carries a URL, not a fingerprint, so the static
        # provider advertises a provisional id (`static:<url>`) and this is where
        # the real one arrives. Refusing it made pairing over a manual address
        # impossible — the very case mDNS cannot cover — while static.py's own
        # docstring promised the swap happened here. It never did.
        #
        # A *non*-provisional mismatch stays fatal: that is a discovered peer whose
        # key does not hash to the id it advertised, which is what the check exists
        # to catch.
        from peer.discovery.static import is_provisional
        if not is_provisional(req.peer_id):
            raise fastapi.HTTPException(
                502, f'peer public_key fingerprint mismatch: advertised {req.peer_id}, '
                f'key hashes to {remote_peer_id}'
            )
        # Re-file the advert under the identity it just proved, so everything after
        # this (endpoints_for, the pending row, /pair/confirm) refers to the real
        # peer rather than a URL-shaped placeholder.
        from peer.discovery.base import PeerAdvert
        registry.observe(PeerAdvert(
            peer_id=remote_peer_id,
            display_name=result.get('display_name', '') or advert.display_name,
            endpoints=endpoints,
            source=advert.source,
        ))
        registry.forget(req.peer_id)
        _adopt_static_peer_id(endpoints, remote_peer_id)

    session = pairing.PairingSession(
        peer_id=remote_peer_id,
        peer_public_key=remote_pubkey_raw,
        display_name=result.get('display_name', advert.display_name),
        endpoints=endpoints,
        local_nonce=local_nonce,
        remote_nonce=remote_nonce,
        local_public_key=identity.public_key_raw(),
    )
    pairing.put(session)
    return session.to_dict()


class ConfirmPairingReq(BaseModel):
    peer_id: str
    code: str


@router.post('/pair/confirm')
async def confirm_pairing(req: ConfirmPairingReq):
    """Complete pairing after the human confirms the short code matches.

    Creates the peer record with the pinned public key and a default role.

    Idempotent for an already-paired peer, but the response says so explicitly.
    Once the session is consumed the submitted code cannot be checked against
    anything, and answering a plain 200 made a wrong code look verified — the
    caller had no way to tell "code matched" from "nothing was compared". No
    access is granted either way (this endpoint needs ACCESS_TOKEN and only
    echoes an existing record), but a confirmation step that reports success
    without confirming anything is worth being loud about.
    """
    session = pairing.get(req.peer_id)
    if session is None:
        # Maybe already confirmed, check the store.
        peer = store.get(req.peer_id)
        if peer is not None:
            return {
                'peer': peer,
                'already_paired': True,
                'code_verified': False,
                'note': ('no active pairing session — returned the existing pairing '
                         'unchanged; the submitted code was not checked. To re-verify, '
                         'unpair and pair again.'),
            }
        raise fastapi.HTTPException(404, 'no active pairing session for this peer_id')

    if session.expired:
        pairing.pop(req.peer_id)
        raise fastapi.HTTPException(410, 'pairing session expired')

    if session.code != req.code:
        raise fastapi.HTTPException(403, 'code does not match')

    import config
    default_role = config.main.get('peer_settings', {}).get('default_role', 'viewer')
    peer = store.upsert(
        peer_id=req.peer_id,
        public_key_b64=base64.b64encode(session.peer_public_key).decode(),
        display_name=session.display_name,
        role=default_role,
        endpoints=session.endpoints,
    )
    pairing.pop(req.peer_id)
    print(f'[peer] paired with {req.peer_id[:12]} ({peer["display_name"]}) as {default_role}')
    return {'peer': peer, 'already_paired': False, 'code_verified': True}


class RejectPairingReq(BaseModel):
    peer_id: str


@router.post('/pair/reject')
async def reject_pairing(req: RejectPairingReq):
    """Decline a pairing request outright.

    Without this the only way to refuse was to wait out the five-minute
    expiry, during which the requesting side keeps showing the pairing as
    pending. Refusing should be as explicit an act as approving.
    """
    session = pairing.pop(req.peer_id)
    if session is None:
        raise fastapi.HTTPException(404, 'no active pairing session for this peer_id')
    print(f'[peer] pairing rejected for {req.peer_id[:12]}')
    return {'rejected': True, 'peer_id': req.peer_id}


@router.get('/pair/active')
async def active_pairings():
    """In-flight pairing sessions. For the dashboard to show the operator what
    codes are live."""
    return {'sessions': pairing.active()}


class UpdatePeerReq(BaseModel):
    display_name: str | None = None
    role: str | None = None
    tool_filter: str | None = None


@router.post('/paired/{peer_id}')
async def update_peer(peer_id: str, req: UpdatePeerReq):
    """Mutate a paired peer's display name, role or tool filter."""
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not fields:
        raise fastapi.HTTPException(400, 'no fields to update')
    peer = store.update(peer_id, **fields)
    if peer is None:
        raise fastapi.HTTPException(404, 'peer not found')
    return {'peer': peer}


@router.delete('/paired/{peer_id}')
async def unpair(peer_id: str):
    """Remove a peer. The dashboard shows this as "unpair"."""
    if not store.delete(peer_id):
        raise fastapi.HTTPException(404, 'peer not found')
    return {'deleted': True}


@router.get('/providers')
async def provider_status():
    """Which discovery providers are running, and why any failed."""
    return {'providers': registry.provider_status()}


@router.get('/dds_isolation')
async def dds_isolation_status():
    """DDS 是否真的被隔离在本机。

    值得单独暴露：隔离失效时没有任何外部症状——机器人照常工作，直到某天另一台
    机器人替它回答了指令。运维需要一个能直接问的地方，而不是去翻启动日志。
    """
    import dds_isolation
    return dds_isolation.check_and_report()


@router.get('/dds_topology')
async def dds_topology():
    """每个 peer 各自看得见哪些 ROS2 话题。

    返回 {peer_id: {topics: [...], last_seen: float}}。内容仍是 DDS 状态，但
    自从 DDS 被锁在本机之后，它是各 peer 通过签名 HTTPS 推过来的 —— 前端不受
    影响，结构没变。
    """
    from peer import dds_state
    if not dds_state.is_available():
        return {'available': False, 'peers': {}}
    return {'available': True, 'peers': dds_state.get_peer_topics()}


# ── peer-facing (Ed25519 signature) ──────────────────────────────────────────

@router.post('/inbox/pair_request')
async def inbox_pair_request(req: Request):
    """The other half of the pairing handshake.

    Unpaired by definition, so `require_paired=False` — the consistency check
    (key hashes to peer_id) is what transport.verify_signed_request does, and
    the human comparing a short code on both screens is what stops a MITM.

    Crucially this must *store* a PairingSession, not just answer with a nonce.
    The whole security property of SAS is that a human sees the same six digits
    on both dashboards; a receiver that discards its own nonce can neither
    derive that code nor offer the operator anything to confirm, which reduces
    pairing to "whoever asked first wins".
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=False
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    remote_nonce = payload.get('nonce', '')
    remote_pubkey = payload.get('public_key', '')
    if not remote_nonce or not remote_pubkey:
        raise fastapi.HTTPException(400, 'missing nonce or public_key')

    try:
        remote_pubkey_raw = base64.b64decode(remote_pubkey)
    except (ValueError, TypeError):
        raise fastapi.HTTPException(400, 'public_key is not valid base64')
    if len(remote_pubkey_raw) != 32 or identity.fingerprint(remote_pubkey_raw) != peer_id:
        raise fastapi.HTTPException(400, 'public_key does not match signing identity')

    local_nonce = pairing.new_nonce()
    local_pubkey = identity.public_key_b64()

    # sas_code() sorts its inputs, so this derives the identical code the
    # initiator computed — neither side needs to know who started.
    session = pairing.PairingSession(
        peer_id=peer_id,
        peer_public_key=remote_pubkey_raw,
        display_name=payload.get('display_name', '') or _peer_label(peer_id),
        endpoints=_inbound_endpoints(req, payload, peer_id),
        local_nonce=local_nonce,
        remote_nonce=remote_nonce,
        local_public_key=identity.public_key_raw(),
    )
    pairing.put(session)
    print(f'[peer] pairing requested by {peer_id[:12]} — code {session.code}')

    # Surface it in the activity stream. A pairing that only reaches the log
    # requires the operator to already have the Peers panel open to notice it,
    # which defeats the point of requiring human confirmation.
    try:
        from api.motus_stream import push_event
        await push_event({
            'type': 'peer_pair_request',
            'mcp_id': f'peer:{peer_id[:12]}',
            'payload': {
                'peer_id': peer_id,
                'display_name': session.display_name,
                'code': session.code,
                'expires_in': session.to_dict()['expires_in'],
            },
        })
    except Exception as e:
        # Never fail the handshake because the notification path is broken.
        print(f'[peer] pair request notification failed: {type(e).__name__}: {e}')

    return {
        'nonce': local_nonce,
        'public_key': local_pubkey,
        'display_name': _local_display_name(),
    }


@router.post('/inbox/ping')
async def inbox_ping(req: Request):
    """Health probe from a paired peer."""
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')
    store.touch(peer_id, _source_endpoint(req))
    return {'pong': True, 'timestamp': time.time()}


@router.post('/inbox/state')
async def inbox_state(req: Request):
    """一个已配对 peer 推过来的话题清单。

    只接受状态，绝不接受指令 —— 这条规则比它原来的 DDS 通道活得久。改走签名
    链路顺带补上了一个真实缺陷：原来的 peer DDS 总线没有任何鉴权，同一个
    ROS_DOMAIN_ID 上任何进程都能伪造另一台机器人的话题清单。
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    topics = payload.get('topics')
    if not isinstance(topics, list):
        raise fastapi.HTTPException(400, 'topics must be a list')
    # 只保留字符串，并且设上限：对端是可以任意构造这个字段的，即便它已配对。
    clean = [t for t in topics if isinstance(t, str)][:2000]

    running = payload.get('agent_running')
    if not isinstance(running, bool):
        running = None   # older peers do not send it; unknown ≠ off

    from peer import dds_state
    dds_state.record_peer_topics(peer_id, clean, agent_running=running)
    store.touch(peer_id, _source_endpoint(req))

    # Sync the display name: a rename on one machine should reach all others within
    # one push interval (5s), not stay stale until the pairing is redone.
    name = payload.get('display_name', '')
    if name and isinstance(name, str):
        store.update_display_name(peer_id, name)

    return {'accepted': len(clean)}


@router.post('/inbox/message')
async def inbox_message(req: Request):
    """Inbound peer message, routed to the lan ChannelAdapter."""
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    accepted, err = await lan.deliver(peer_id, payload)
    return {'accepted': accepted, 'reason': err if not accepted else ''}


# ── tool proxy (peer-facing, Ed25519 signature) ──────────────────────────────

@router.get('/tools/list')
async def list_tools(req: Request):
    """Return the subset of tools this peer may call.

    Filtered by role + tool_filter. Schemas are OpenAI function-calling format,
    so a peer can pass them to its own LLM.
    """
    # Signature verification with no body (GET)
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, b'', require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    # Advertise only what the peer could actually invoke: role/tool_filter *and*
    # wired to the decision core. Listing unwired tools would have the peer's LLM
    # plan around capabilities that always 403 at call time.
    #
    # Another peer's tools are excluded. They are in all_schemas() now (the outbound
    # bridge registers them so the local LLM can call them) and re-advertising them
    # makes two agents mirror each other: A offers B's tools to B, B offers them back,
    # and the list grows every refresh round. Observed as "offers 4 tools:
    # tts_70461, tts_3177a, tts_70461, tts_3177a". A tool is ours to offer only if it
    # runs here.
    import canvas_binding
    all_schemas = [s for s in mcp_client.all_schemas()
                   if not canvas_binding.is_peer_tool(s.get('name', ''))
                   and canvas_binding.is_bound(s.get('name', ''))]
    allowed = peer_tools.filter_schemas(peer_id, all_schemas)
    return {'tools': allowed, 'count': len(allowed),
            'acp_meta': _acp_meta_for(allowed)}


def _acp_meta_for(schemas: list[dict]) -> dict:
    """Per-tool ACP facts a caller needs, keyed by the tool's local name.

    `tools` carries OpenAI function-calling schemas, whose `parameters` is not the
    MCP `inputSchema` — so `x-completion` and `x-resource` are not in there, and the
    bridge on the far side had no way to learn either. It filled `tool_meta` with
    `{}`, which made every peer tool undeclared, and undeclared means "exclusive
    against everything": calling one peer tool blocked all local actuation.

    These two are safe to report where `type` was not (see mcp_bridge): the remote
    is not guessing, it is repeating what its own driver declared.
    """
    out = {}
    for schema in schemas:
        name = schema.get('name', '')
        if not name:
            continue
        mcp_id = name.split('__')[1] if name.startswith('mcp__') else ''
        meta = (mcp_client.registry.get(mcp_id, {}).get('tool_meta', {}) or {}).get(name)
        if not meta:
            continue
        resource = meta.get('resource')
        entry = {}
        if meta.get('completion'):
            entry['completion'] = meta['completion']
        if resource:
            # frozenset is not JSON; sorted list keeps it stable across refreshes so
            # a bridge diff does not churn.
            entry['resource'] = sorted(resource)
        if entry:
            out[name] = entry
    return out


class ToolCallReq(BaseModel):
    tool_name: str
    arguments: dict


@router.post('/tools/call')
async def call_tool(req: Request):
    """Execute a tool on behalf of the peer.

    Two gates apply, and both are enforced here:

      1. the peer's role and tool_filter (peer/tools.py), and
      2. the tool being wired to the decision core on the local canvas.

    Gate 2 used to be missing on this path. The docstring claimed
    mcp_client.call_tool enforced it; it does not — that function performs no
    canvas check at all, so an `operator` peer could invoke any tool its
    tool_filter allowed, wired or not. The canvas is the operator's authority
    over what a remote machine may touch, and a proxy that ignores it makes
    that authority decorative.
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    tool_name = payload.get('tool_name', '')
    arguments = payload.get('arguments', {})
    if not tool_name:
        raise fastapi.HTTPException(400, 'tool_name required')

    # Gate 1 — the peer's role and tool_filter
    allowed, perm_reason = peer_tools.check_tool_permission(peer_id, tool_name)
    if not allowed:
        raise fastapi.HTTPException(403, f'tool call denied: {perm_reason}')

    # Gate 2 — the operator must have wired this tool to the decision core
    import canvas_binding
    if not canvas_binding.is_bound(tool_name):
        raise fastapi.HTTPException(
            403,
            f'tool call denied: "{tool_name}" is not wired to the decision core on '
            f'this agent\'s canvas. Connect it there to let a peer request it.'
        )

    # Announce start and end on the activity stream.
    #
    # Without this a peer using this machine's tools is invisible to the person
    # standing next to it: the call never enters the collector, never reaches the
    # LLM's context and never touches conversation history — it only ever appeared
    # in stdout. peer/tools.py claimed "same audit trail" as a human calling
    # through a channel; that was not true.
    #
    # Shaped like ACP's own start/finish pair so the dashboard can pair them by
    # `action_id`. For a tool that returns immediately the two arrive together;
    # for an async one the driver's later /api/acp/complete carries the real end.
    # Only tools that *act* are announced. A viewer polling battery or a camera
    # frame would otherwise flood the stream, and the point of the announcement is
    # that someone standing next to the robot learns a peer made it do something —
    # reading its state is not that.
    notify = not peer_tools.is_read_only(tool_name)

    from api.motus_stream import push_event
    peer_row = store.get(peer_id) or {}
    who = peer_row.get('display_name') or peer_id[:12]
    started = time.time()

    async def _emit(kind: str, extra: dict) -> None:
        if not notify:
            return
        try:
            await push_event({
                'type': kind,
                'mcp_id': f'peer:{peer_id[:12]}',
                'payload': {'peer_id': peer_id, 'peer': who, 'tool': tool_name,
                            'action': arguments.get('action', ''), **extra},
            })
        except Exception as exc:      # 可见性不该影响调用本身
            print(f'[peer] activity event {kind} failed: {type(exc).__name__}: {exc}')

    await _emit('peer_tool_call', {})
    try:
        result = await mcp_client.call_tool(tool_name, arguments)
        store.touch(peer_id, _source_endpoint(req))

        # ACP does not stop at the machine boundary.
        #
        # An async action returns as soon as the driver has *queued* it, so replying
        # here handed the caller "done" while nothing had happened yet. Locally the
        # barrier hides that; across a peer there was no barrier at all, which is why
        # one robot told another to speak and then immediately spoke over it.
        #
        # So hold the response until this action actually finishes, and report the
        # terminal state rather than the acknowledgement. The connection stays open
        # for the action's duration — the same shape /delegate already has, and
        # transport.post_json is cancellable now, so the caller can still abandon it.
        action_id = _action_id_of(result)
        action_status = 'sync'
        if action_id:
            sync_out = await mcp_client.sync([action_id], timeout=_ACTION_WAIT_S)
            action_status = sync_out.get('status', 'unknown')
            if action_status != 'completed':
                print(f'[peer] {who} action {action_id} ended {action_status!r} '
                      f'(tool={tool_name})')

        await _emit('peer_tool_result', {
            'ok': True,
            'elapsed_ms': round((time.time() - started) * 1000),
            'action_id': action_id,
            'action_status': action_status,
        })
        return {'result': result, 'error': None,
                'action_id': action_id, 'action_status': action_status}
    except Exception as e:
        await _emit('peer_tool_result', {
            'ok': False, 'error': str(e),
            'elapsed_ms': round((time.time() - started) * 1000),
        })
        return {'result': None, 'error': str(e)}



# How long a peer's tool call may hold this connection waiting for its action to
# finish. Above the longest x-completion timeout a driver declares (g1 switch_mode
# is 150s) so the driver's own limit is what expires first and we report *its*
# verdict, rather than giving up early and leaving the caller unable to tell a slow
# action from a lost one.
_ACTION_WAIT_S = 180.0


def _action_id_of(result) -> str:
    """The ACP action_id in a tool result, or ''.

    Async tools answer `{"status": "queued", "action_id": ...}`, sometimes wrapped
    in MCP content items and sometimes as a JSON string. Pulled out so the
    start/end events can be paired with the driver's later completion callback.
    """
    import json as _json
    payload = result
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0].get('text', '')
    if isinstance(payload, str):
        try:
            payload = _json.loads(payload)
        except Exception:
            return ''
    return str(payload.get('action_id', '')) if isinstance(payload, dict) else ''

# ── task delegation (peer-facing, Ed25519 signature) ─────────────────────────

class DelegateReq(BaseModel):
    goal: str
    priority: int = 2
    model: str | None = None
    tool_filter: list[str] | None = None
    tool_deny: list[str] | None = None
    max_rounds: int = 10
    timeout_s: float = 300.0
    # How many peers this task has already been handed through. The receiver
    # increments it and refuses past delegation.MAX_HOP_COUNT.
    hop_count: int = 0


@router.post('/delegate')
async def delegate_task(req: Request):
    """Spawn a subagent on behalf of the peer and return the result.

    The delegated spec's tool_filter is intersected with the peer's role
    permissions, and hop_count is incremented. Returns SubagentResult JSON.
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    # Build SubagentSpec from request.
    #
    # hop_count MUST be read from the payload. Omitting it left the field at its
    # default of 0 on every request, so validate_delegation()'s limit could never
    # trigger and the circuit breaker against delegation storms was inert —
    # A→B→C→D… would have chained without bound.
    try:
        hop_count = int(payload.get('hop_count', 0) or 0)
    except (TypeError, ValueError):
        raise fastapi.HTTPException(400, 'hop_count must be an integer')
    if hop_count < 0:
        raise fastapi.HTTPException(400, 'hop_count must not be negative')

    spec = SubagentSpec(
        goal=payload.get('goal', ''),
        priority=payload.get('priority', 2),
        model=payload.get('model'),
        tool_filter=payload.get('tool_filter'),
        tool_deny=payload.get('tool_deny'),
        max_rounds=payload.get('max_rounds', 10),
        timeout_s=payload.get('timeout_s', 300.0),
        hop_count=hop_count,
    )
    if not spec.goal:
        raise fastapi.HTTPException(400, 'goal required')

    # Validate delegation
    allowed, val_reason = delegation.validate_delegation(peer_id, spec)
    if not allowed:
        raise fastapi.HTTPException(403, f'delegation denied: {val_reason}')

    # Prepare augmented spec (increment hop, intersect filter)
    augmented_spec = delegation.prepare_delegated_spec(peer_id, spec)

    # The manager is owned by the agent loop (event/llm.py) and only exists once
    # that has initialised, so it is resolved per-request rather than imported at
    # module load. Importing a manager singleton at import time is what broke
    # startup before: subagent.manager exposes only the class.
    import subagent
    subagent_manager = subagent._manager_instance
    if subagent_manager is None:
        raise fastapi.HTTPException(
            503,
            'agent loop is not running — delegation needs the local subagent '
            'manager, which starts with the main event loop'
        )

    # Spawn and wait
    try:
        result = await subagent_manager.spawn_and_wait(augmented_spec, timeout=spec.timeout_s)
        store.touch(peer_id, _source_endpoint(req))
        return result.to_dict()
    except Exception as e:
        return {
            'agent_id': '',
            'status': 'failed',
            'output': '',
            'error': str(e),
        }


