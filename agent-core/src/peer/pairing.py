"""
peer/pairing.py — short authentication string (SAS) pairing.

The Bluetooth model: both sides derive the same 6-digit code from material
neither side controls alone, a human reads it off both screens, and confirms on
both. An attacker sitting in the middle holds a different key on each side, so
the two screens show different codes and the human refuses.

This is chosen over a CA because it has to work with no network at all (BLE
bootstrap in the field), and because "power the robots on and pair them" is the
actual deployment story. There is no auto-approve: `channel_settings` offers one
for humans joining a chat, but an unattended pairing would defeat the entire
point of a short code.

The derivation MUST be symmetric — both sides compute it independently with no
agreement on who is "initiator". Hence the sorting: inputs go in canonical
order, so A-then-B and B-then-A hash identically.
"""

import base64
import hashlib
import hmac
import os
import time

_CODE_DIGITS = 6
_PAIRING_TTL_S = 300  # a code a human has been staring at for 5 minutes is stale


def new_nonce() -> str:
    return base64.b64encode(os.urandom(16)).decode()


def sas_code(pubkey_a: bytes, pubkey_b: bytes, nonce_a: str, nonce_b: str) -> str:
    """Derive the 6-digit code shown on both dashboards.

    Sorted inputs make this symmetric; keying the HMAC on the (sorted) keys and
    hashing the (sorted) nonces means changing *either* side's key changes the
    code, which is what defeats a man in the middle.
    """
    keys = sorted([pubkey_a, pubkey_b])
    nonces = sorted([nonce_a.encode(), nonce_b.encode()])
    digest = hmac.new(
        key=b'motus-peer-sas\x00' + keys[0] + keys[1],
        msg=nonces[0] + b'\x00' + nonces[1],
        digestmod=hashlib.sha256,
    ).digest()
    value = int.from_bytes(digest[:8], 'big') % (10 ** _CODE_DIGITS)
    return str(value).zfill(_CODE_DIGITS)


class PairingSession:
    """One in-flight pairing, held in memory only.

    Not persisted on purpose: an unconfirmed pairing surviving a restart would
    let a code approved before the restart be confirmed after it, with nobody
    still looking at the screen.
    """

    def __init__(self, peer_id: str, peer_public_key: bytes, display_name: str,
                 endpoints: list[str], local_nonce: str, remote_nonce: str,
                 local_public_key: bytes):
        self.peer_id = peer_id
        self.peer_public_key = peer_public_key
        self.display_name = display_name
        self.endpoints = endpoints
        self.local_nonce = local_nonce
        self.remote_nonce = remote_nonce
        self.code = sas_code(local_public_key, peer_public_key, local_nonce, remote_nonce)
        self.created_at = time.time()

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > _PAIRING_TTL_S

    def to_dict(self) -> dict:
        return {
            'peer_id': self.peer_id,
            'display_name': self.display_name,
            'code': self.code,
            'endpoints': self.endpoints,
            'created_at': self.created_at,
            'expires_in': max(0, _PAIRING_TTL_S - (time.time() - self.created_at)),
        }


_sessions: dict[str, PairingSession] = {}


def put(session: PairingSession) -> None:
    _prune()
    _sessions[session.peer_id] = session


def get(peer_id: str) -> PairingSession | None:
    _prune()
    return _sessions.get(peer_id)


def pop(peer_id: str) -> PairingSession | None:
    _prune()
    return _sessions.pop(peer_id, None)


def active() -> list[dict]:
    _prune()
    return [s.to_dict() for s in _sessions.values()]


def clear() -> None:
    _sessions.clear()


def _prune() -> None:
    for pid in [p for p, s in _sessions.items() if s.expired]:
        _sessions.pop(pid, None)
