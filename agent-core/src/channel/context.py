"""Server-owned origin of asynchronous work started by a Channel turn."""

import json
from contextvars import ContextVar


current: ContextVar[dict | None] = ContextVar('channel_turn_origin', default=None)


def context_key(origin: dict) -> str:
    scope = origin.get('_channel_scope')
    if (not isinstance(scope, (tuple, list)) or len(scope) != 6
            or not all(isinstance(value, str) and value for value in scope[:4])
            or scope[4] not in ('viewer', 'operator', 'owner')
            or not isinstance(scope[5], bool)):
        return ''
    return json.dumps(scope, ensure_ascii=False, separators=(',', ':'))


def history_context(origin: dict | None) -> str | None:
    """Full-control turns share main memory; restricted turns use their own history."""
    if origin is None:
        return None
    key = context_key(origin)
    if key:
        scope = origin['_channel_scope']
        if scope[4] in ('operator', 'owner') and not scope[5]:
            return None
    return key  # An unknown Channel identity must not fall back to shared memory.


def capture(origin: dict | None = None) -> dict | None:
    """Copy routing fields only; never retain the mutable trigger or its text."""
    origin = current.get() if origin is None else origin
    if not isinstance(origin, dict) or not context_key(origin):
        return None
    session_id = origin.get('_channel_session_id')
    message_ids = origin.get('_channel_message_ids')
    if (not isinstance(session_id, str) or not session_id
            or not isinstance(message_ids, list) or not message_ids
            or not all(isinstance(value, str) and value for value in message_ids)):
        return None
    return {
        '_channel_scope': list(origin['_channel_scope']),
        '_channel_session_id': session_id,
        '_channel_message_ids': list(message_ids),
        '_channel_image_paths': list(origin.get('_channel_image_paths') or []),
    }


def is_active(origin: dict) -> bool:
    """A deleted conversation or revoked authorization cannot be resumed."""
    origin = capture(origin)
    if origin is None:
        return False
    import config
    from channel import acl
    from channel.manager import get_channel_config

    platform, channel_id, _, user_id, role, chat_only = origin['_channel_scope']
    user = acl.get_user(platform, user_id)
    channel = get_channel_config(channel_id)
    if (not user or user['role'] != role or not channel or not channel.get('enabled')
            or channel.get('platform') != platform or bool(channel.get('chat_only')) != chat_only):
        return False
    with config._get_conn() as conn:
        if history_context(origin) is None:
            return conn.execute(
                "SELECT 1 FROM chat_sessions WHERE id=? AND (kind='main' "
                "OR (kind='channel' AND context_key=?))",
                (origin['_channel_session_id'], context_key(origin)),
            ).fetchone() is not None
        return conn.execute(
            "SELECT 1 FROM chat_sessions WHERE id=? AND kind='channel' AND context_key=?",
            (origin['_channel_session_id'], context_key(origin)),
        ).fetchone() is not None
