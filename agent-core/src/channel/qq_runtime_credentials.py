"""Process-memory-only QQ credentials for first-run setup.

Never serialize credential values to config, database, logs, files or HTTP
responses. A new process requires re-entry or a managed env/file reference.
"""
import secrets
import threading

_lock = threading.Lock()
_values = {}


def put(value: str) -> str:
    if not isinstance(value, str) or not (8 <= len(value.strip()) <= 512):
        raise ValueError('Invalid QQ credential')
    ref = secrets.token_urlsafe(32)
    with _lock:
        _values[ref] = value.strip()
    return ref


def get(ref: str) -> str:
    with _lock:
        return _values.get(ref, '')


def revoke(ref: str) -> None:
    with _lock:
        _values.pop(ref, None)


def available(ref: str) -> bool:
    return bool(get(ref))
