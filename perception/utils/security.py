"""Helpers for keeping credentials out of operational logs."""

from __future__ import annotations

import re


REDACTED = "<redacted>"
_SENSITIVE_PARTS = {
    "authorization",
    "credential",
    "credentials",
    "key",
    "password",
    "secret",
    "token",
}
_SENSITIVE_COMPACT_KEYS = {
    "apikey",
    "accesstoken",
    "authtoken",
    "clientsecret",
    "llmkey",
    "refreshtoken",
}


def redact_sensitive(value):
    """Return a log-only copy with recursively redacted secret fields."""

    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(key) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value


def _is_sensitive_key(key) -> bool:
    normalized = str(key).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if compact in _SENSITIVE_COMPACT_KEYS:
        return True
    parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    return bool(parts & _SENSITIVE_PARTS)
