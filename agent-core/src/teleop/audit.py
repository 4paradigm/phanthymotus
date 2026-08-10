from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import config

MAX_DETAIL_DEPTH = 5
MAX_COLLECTION_ITEMS = 50
MAX_STRING_LENGTH = 512
MAX_KEY_LENGTH = 128
MAX_STORED_EVENTS = 10_000
MAX_PENDING_DELIVERIES = 128
_FAST_PATH_WAIT_SECONDS = 0.02
_delivery_tasks: set[asyncio.Task] = set()
_SENSITIVE_KEY_PARTS = (
    'credential',
    'fence',
    'password',
    'privatekey',
    'secret',
    'token',
)


def _is_sensitive_key(key: object) -> bool:
    normalized = ''.join(character for character in str(key).lower() if character.isalnum())
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _bounded_text(value: object, limit: int = MAX_STRING_LENGTH) -> str:
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 -- sanitizer must tolerate hostile values
        return '<unprintable>'
    return text[:limit]


def _sanitize_value(value: Any, *, depth: int, seen: set[int]) -> Any:
    if depth > MAX_DETAIL_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value[:MAX_STRING_LENGTH]).decode('utf-8', errors='replace')

    container = isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, str)
    ) or isinstance(value, (set, frozenset))
    identity = id(value)
    if container:
        if identity in seen:
            return None
        seen.add(identity)

    try:
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            try:
                items = value.items()
            except Exception:  # noqa: BLE001 -- custom Mapping may fail
                return {}
            for index, pair in enumerate(items):
                if index >= MAX_COLLECTION_ITEMS:
                    break
                try:
                    key, item = pair
                except Exception:  # noqa: BLE001, S112 -- skip malformed pair
                    continue
                if _is_sensitive_key(key):
                    continue
                safe_key = _bounded_text(key, MAX_KEY_LENGTH)
                sanitized[safe_key] = _sanitize_value(
                    item,
                    depth=depth + 1,
                    seen=seen,
                )
            return sanitized

        if isinstance(value, (set, frozenset)):
            try:
                values = sorted(value, key=lambda item: _bounded_text(item))
            except Exception:  # noqa: BLE001 -- custom set values may fail
                values = []
            return [
                _sanitize_value(item, depth=depth + 1, seen=seen)
                for item in values[:MAX_COLLECTION_ITEMS]
            ]

        if isinstance(value, Sequence) and not isinstance(value, str):
            sanitized_items: list[Any] = []
            try:
                for index, item in enumerate(value):
                    if index >= MAX_COLLECTION_ITEMS:
                        break
                    sanitized_items.append(
                        _sanitize_value(item, depth=depth + 1, seen=seen)
                    )
            except Exception:  # noqa: BLE001, S110 -- return bounded prefix
                pass
            return sanitized_items

        return _bounded_text(value)
    except Exception:  # noqa: BLE001 -- sanitizer is deliberately fail-closed
        return None
    finally:
        if container:
            seen.discard(identity)


def _safe_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return bounded JSON data with secret-shaped keys removed at every depth."""

    if not isinstance(details, Mapping):
        return {}
    sanitized = _sanitize_value(details, depth=0, seen=set())
    return sanitized if isinstance(sanitized, dict) else {}


def _ensure_table(conn) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS teleop_audit (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT DEFAULT '',
            robot_id TEXT DEFAULT '',
            principal_id TEXT DEFAULT '',
            source TEXT DEFAULT '',
            decision TEXT DEFAULT '',
            reason TEXT DEFAULT '',
            tool TEXT DEFAULT '',
            action TEXT DEFAULT '',
            details TEXT DEFAULT '{}'
        )
    ''')
    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_teleop_audit_created '
        'ON teleop_audit(created_at)'
    )


async def emit(
    event_type: str,
    *,
    session_id: str = '',
    robot_id: str = '',
    principal_id: str = '',
    source: str = '',
    decision: str = '',
    reason: str = '',
    tool: str = '',
    action: str = '',
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist and broadcast a sanitized event without breaking control flow.

    Audit storage and Activity delivery are observability paths, so either may
    fail without changing the lifecycle decision that has already been made.
    """

    event = {
        'id': str(uuid.uuid4()),
        'created_at': time.time(),
        'event_type': _bounded_text(event_type),
        'session_id': _bounded_text(session_id),
        'robot_id': _bounded_text(robot_id),
        'principal_id': _bounded_text(principal_id),
        'source': _bounded_text(source),
        'decision': _bounded_text(decision),
        'reason': _bounded_text(reason),
        'tool': _bounded_text(tool),
        'action': _bounded_text(action),
        'details': _safe_details(details),
    }

    if len(_delivery_tasks) < MAX_PENDING_DELIVERIES:
        task = asyncio.create_task(
            _deliver_event(event),
            name=f'teleop-audit-{event["id"]}',
        )
        _delivery_tasks.add(task)
        task.add_done_callback(_delivery_tasks.discard)
        # Preserve immediate visibility on a healthy database, but never let
        # observability consume a meaningful fraction of a Driver watchdog.
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=_FAST_PATH_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            pass
    return event


async def _deliver_event(event: Mapping[str, Any]) -> None:
    try:
        await asyncio.to_thread(_persist_event, event)
    except Exception:  # noqa: BLE001, S110 -- audit storage is fail-safe
        pass

    try:
        from api.motus_stream import push_event

        await push_event({
            'type': 'teleop_activity',
            'mcp_id': event['robot_id'],
            'payload': event,
        })
    except Exception:  # noqa: BLE001, S110 -- broadcast is best effort
        pass


async def flush(timeout_seconds: float = 1.0) -> None:
    """Best-effort test/shutdown helper; control paths never need to call it."""

    pending = list(_delivery_tasks)
    if not pending:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=max(0.0, timeout_seconds),
        )
    except asyncio.TimeoutError:
        pass


def _persist_event(event: Mapping[str, Any]) -> None:
    """Persist one already-sanitized event from a worker thread."""

    try:
        with config._get_conn() as conn:
            _ensure_table(conn)
            conn.execute(
                '''INSERT INTO teleop_audit
                   (id, created_at, event_type, session_id, robot_id, principal_id,
                    source, decision, reason, tool, action, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    event['id'],
                    event['created_at'],
                    event['event_type'],
                    event['session_id'],
                    event['robot_id'],
                    event['principal_id'],
                    event['source'],
                    event['decision'],
                    event['reason'],
                    event['tool'],
                    event['action'],
                    json.dumps(
                        event['details'],
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                ),
            )
            conn.execute(
                '''DELETE FROM teleop_audit WHERE id IN (
                       SELECT id FROM teleop_audit
                       ORDER BY created_at DESC, id DESC
                       LIMIT -1 OFFSET ?
                   )''',
                (MAX_STORED_EVENTS,),
            )
            conn.commit()
    except Exception:  # noqa: BLE001, S110 -- audit storage is fail-safe
        pass


def _row_to_event(row) -> dict[str, Any]:
    try:
        decoded = json.loads(row[11] or '{}')
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        decoded = {}
    created_at = row[1] if isinstance(row[1], (int, float)) else 0.0
    if not math.isfinite(created_at):
        created_at = 0.0
    return {
        'id': _bounded_text(row[0]),
        'created_at': created_at,
        'event_type': _bounded_text(row[2]),
        'session_id': _bounded_text(row[3]),
        'robot_id': _bounded_text(row[4]),
        'principal_id': _bounded_text(row[5]),
        'source': _bounded_text(row[6]),
        'decision': _bounded_text(row[7]),
        'reason': _bounded_text(row[8]),
        'tool': _bounded_text(row[9]),
        'action': _bounded_text(row[10]),
        'details': _safe_details(decoded),
    }


def list_events(
    limit: int = 100,
    robot_id: str = '',
    session_id: str = '',
) -> list[dict[str, Any]]:
    try:
        bounded_limit = min(500, max(1, int(limit)))
    except (TypeError, ValueError, OverflowError):
        bounded_limit = 100

    try:
        with config._get_conn() as conn:
            _ensure_table(conn)
            if session_id:
                rows = conn.execute(
                    '''SELECT id, created_at, event_type, session_id, robot_id,
                              principal_id, source, decision, reason, tool, action, details
                       FROM teleop_audit WHERE session_id=?
                       ORDER BY created_at DESC LIMIT ?''',
                    (_bounded_text(session_id), bounded_limit),
                ).fetchall()
            elif robot_id:
                rows = conn.execute(
                    '''SELECT id, created_at, event_type, session_id, robot_id,
                              principal_id, source, decision, reason, tool, action, details
                       FROM teleop_audit WHERE robot_id=?
                       ORDER BY created_at DESC LIMIT ?''',
                    (_bounded_text(robot_id), bounded_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    '''SELECT id, created_at, event_type, session_id, robot_id,
                              principal_id, source, decision, reason, tool, action, details
                       FROM teleop_audit ORDER BY created_at DESC LIMIT ?''',
                    (bounded_limit,),
                ).fetchall()
    except Exception:  # noqa: BLE001 -- reads are an observability path
        return []

    events: list[dict[str, Any]] = []
    for row in rows:
        try:
            events.append(_row_to_event(row))
        except Exception:  # noqa: BLE001, S112 -- skip malformed legacy row
            continue
    return events
