"""Process-local QQ transport counters for support and onboarding.

No message text, OpenIDs, groups, credentials, or attachment paths are stored.
This is observational telemetry, not persisted acceptance or delivery proof.
"""
from __future__ import annotations

from collections import Counter
from threading import RLock
import time

_KNOWN = frozenset({
    'gateway_ready', 'private_inbound', 'group_inbound', 'group_filtered',
    'parse_rejected', 'canvas_blocked', 'acl_blocked', 'inbound_forwarded',
    'media_received', 'voice_transcribed', 'typing_sent',
    'platform_accepted', 'media_sent', 'send_failed', 'gateway_failed',
})
def classify_gateway_failure(reason: str, *, credential_ready: bool = True) -> str:
    """Stable error codes only; never echo raw SDK exceptions or secrets."""
    if not credential_ready:
        return 'CREDENTIAL_UNAVAILABLE'
    message = str(reason or '').lower()
    if any(key in message for key in ('4014', 'intent', 'not authorized')):
        return 'PLATFORM_EVENT_PERMISSION'
    if any(key in message for key in ('credential', 'appsecret', 'invalid token', '401')):
        return 'AUTHENTICATION_FAILED'
    if any(key in message for key in ('dns', 'name resolution', 'connecterror')):
        return 'NETWORK_DNS_FAILURE'
    if any(key in message for key in ('timeout', 'ready failed', 'timed out')):
        return 'GATEWAY_TIMEOUT'
    if not message or 'not started' in message:
        return 'CHANNEL_NOT_RUNNING'
    return 'GATEWAY_UNAVAILABLE'


_LOCK = RLock()
_EVENTS: dict[str, dict] = {}
_MAX_CHANNELS = 32


def record(channel_id: str, stage: str) -> None:
    """Record only predefined aggregate stages; invalid labels are ignored."""
    if (not isinstance(channel_id, str) or not channel_id or
            len(channel_id) > 96 or stage not in _KNOWN):
        return
    with _LOCK:
        if channel_id not in _EVENTS and len(_EVENTS) >= _MAX_CHANNELS:
            oldest = min(_EVENTS, key=lambda key: _EVENTS[key]['updated_at'])
            _EVENTS.pop(oldest, None)
        entry = _EVENTS.setdefault(
            channel_id, {'counts': Counter(), 'recent': {}, 'updated_at': 0.0},
        )
        entry['counts'][stage] += 1
        timestamp = time.time()
        entry['recent'][stage] = timestamp
        entry['updated_at'] = timestamp


def snapshot(channel_id: str) -> dict:
    """Return a JSON-serializable snapshot; only aggregates, never identities."""
    with _LOCK:
        entry = _EVENTS.get(channel_id)
        if not entry:
            return {'counts': {}, 'recent': {}, 'process_local': True}
        return {
            'counts': dict(entry['counts']),
            'recent': dict(entry['recent']),
            'process_local': True,
        }
