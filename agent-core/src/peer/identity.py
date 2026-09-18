"""
peer/identity.py — this Agent Core's long-term cryptographic identity.

An Ed25519 keypair generated on first boot and kept in ConfigDB. The public
key's fingerprint is the `peer_id`, and it is the only durable name a peer has:
IP addresses move, Feishu open_ids are per-platform, but the fingerprint is the
same no matter which discovery provider surfaced the peer. That is what lets
the registry merge one peer found over several paths into a single record.

`ACCESS_TOKEN` (auth.py) is deliberately *not* reused for this. It is one shared
secret per host, so it cannot tell two peers apart, cannot be revoked for one of
them, and leaks the whole fleet at once. It stays what it always was: the
credential for a human operating this dashboard.
"""

import base64
import hashlib
import threading
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

import config


_CONFIG_KEY = 'peer_identity'

# The private key is read on nearly every outbound request; parsing it from the
# DB each time would put a SQLite round-trip in the signing path.
_lock = threading.RLock()
_cached: dict | None = None


def fingerprint(public_key_raw: bytes) -> str:
    """peer_id for a raw 32-byte Ed25519 public key.

    32 hex chars (128 bits) — short enough to show in a UI, far too long to
    collide by accident or to brute-force a match against.
    """
    return hashlib.sha256(public_key_raw).hexdigest()[:32]


def _generate() -> dict:
    private = ed25519.Ed25519PrivateKey.generate()
    raw_priv = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_pub = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        'private_key': base64.b64encode(raw_priv).decode(),
        'public_key': base64.b64encode(raw_pub).decode(),
        'peer_id': fingerprint(raw_pub),
        'created_at': time.time(),
    }


def ensure_identity() -> dict:
    """Load the identity, generating and persisting one on first call.

    Idempotent: repeated calls (including across restarts) return the same
    keypair. Regenerating would silently invalidate every existing pairing,
    so the only way to get a new identity is to delete the row.
    """
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        stored = config.main.get(_CONFIG_KEY)
        if not stored or not stored.get('private_key'):
            stored = _generate()
            config.main[_CONFIG_KEY] = stored
            print(f'[peer] generated identity {stored["peer_id"]}')
        _cached = stored
        return _cached


def reset_cache() -> None:
    """Drop the in-process cache. For tests, and after deleting the identity."""
    global _cached
    with _lock:
        _cached = None


def peer_id() -> str:
    return ensure_identity()['peer_id']


def public_key_raw() -> bytes:
    return base64.b64decode(ensure_identity()['public_key'])


def public_key_b64() -> str:
    return ensure_identity()['public_key']


def _private_key() -> ed25519.Ed25519PrivateKey:
    raw = base64.b64decode(ensure_identity()['private_key'])
    return ed25519.Ed25519PrivateKey.from_private_bytes(raw)


def sign(payload: bytes) -> str:
    """Sign with this agent's identity key. Returns base64."""
    return base64.b64encode(_private_key().sign(payload)).decode()


def verify(public_key_raw_bytes: bytes, signature_b64: str, payload: bytes) -> bool:
    """Verify a peer's signature. Returns False rather than raising — callers
    are request handlers, and a malformed signature is a normal hostile input,
    not an exceptional condition."""
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(public_key_raw_bytes)
        pub.verify(base64.b64decode(signature_b64), payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
