"""Server-side tool authorization for human Channel principals.

The LLM's instructions and the Canvas are not authorization boundaries.
For batched messages, every human principal must permit the requested tool.
"""
import asyncio
import fnmatch

from channel import acl


def _qq_safe(name: str) -> bool:
    """QQ V1 is text-only; no device execution, audio, or delegated tools.

    An explicit operator role and tool_filter cannot expand this platform cap.
    Audio features require a separate reviewed change and real-device tests.
    """
    return name in ('finish', 'mcp__channel__channel_reply')


def _matches_filter(raw: str, name: str) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    patterns = [p.strip() for p in raw.split(',') if p.strip()]
    bare = name.split('__', 2)[-1] if name.startswith('mcp__') else name
    return any(fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(bare, p)
               for p in patterns)


def allowed_for_principals(principals: list, name: str, snapshot: dict | None = None) -> bool:
    """Explicit allow for all sender identities, fetching current ACL from storage.

    A missing or blocked principal fails closed. Every viewer is read-only even
    when an operator shares the same collector batch.
    """
    if not principals:
        return True  # Non-human turns: existing bot policy remains authoritative.
    from event.llm import _restricted_channel_tool_allowed
    for item in principals:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        platform, user_id = item
        if not isinstance(platform, str) or not isinstance(user_id, str):
            return False
        user = (snapshot.get((platform, user_id)) if snapshot is not None
                else acl.get_user(platform, user_id))
        if user is None or user.get('role') == 'blocked':
            return False
        if user.get('role') not in ('owner', 'operator', 'viewer'):
            return False
        if user['role'] == 'viewer' and not _restricted_channel_tool_allowed(
                name, bot_restricted=False):
            return False
        if not _matches_filter(user.get('tool_filter', ''), name):
            # finish should always be possible to terminate an authorized turn.
            if name != 'finish':
                return False
        if platform == 'qq' and not _qq_safe(name):
            return False
    return True


def read_principal_snapshot(principals: list) -> dict:
    """Read ACL outside the event loop; missing identities fail closed."""
    return {
        (p[0], p[1]): acl.get_user(p[0], p[1])
        for p in principals
        if isinstance(p, (list, tuple)) and len(p) == 2
        and isinstance(p[0], str) and isinstance(p[1], str)
    }


async def snapshot_principals(principals: list) -> dict:
    """Collect permissions without blocking asynchronous event processing."""
    return await asyncio.to_thread(read_principal_snapshot, principals)


async def async_allowed_for_principals(principals: list, name: str) -> bool:
    """Re-check ACL at dispatch for revocation, without blocking event loop."""
    return await asyncio.to_thread(allowed_for_principals, principals, name)
