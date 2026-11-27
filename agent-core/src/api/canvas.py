"""
canvas.py — Canvas layout persistence + per-tool config storage.

Stores the orchestration canvas layout (card positions) and per-tool
configuration in the SQLite config table.

Includes editor lock: only one session can edit at a time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import time
from typing import Any, Optional
from uuid import uuid4

import fastapi
from pydantic import BaseModel

import config

router = fastapi.APIRouter(prefix='/canvas', tags=['canvas'])

_TOOL_CONFIG_PREFIX = 'tool_config:'

# ── Editor Lock State (in-memory, resets on restart) ─────────────────────────
#
# The lock expires on *idleness*: 60s without a real user action frees it, even
# if the holder's tab is still open. An earlier version keyed the lifetime off the
# holder's /ws/motus connection instead and never timed out a live one, which meant
# a tab left open on the canvas held it hostage indefinitely — the bug this replaces.
# Idleness is measured from real actions only (claim, layout write, /keep-edit,
# which the frontend fires from actual pointer/key/wheel input); a poll must never
# renew it or an open tab renews forever, which is how the old TTL was defeated.
#
# WS liveness is still tracked, but now only to release *faster* than the TTL when
# a tab goes away: api/motus_stream reports connect/disconnect here and a disconnect
# releases after a short grace period. It can no longer keep a lock alive. That
# matters because beforeunload/pagehide never fire on a killed process, a reclaimed
# mobile tab or a crashed page, and sendBeacon may be dropped.

_editor_session: Optional[str] = None   # session_id of current editor
_editor_last_seen: float = 0.0          # monotonic timestamp of last real action
_EDITOR_TIMEOUT = 60.0                  # idle seconds before the lock auto-releases

_live_sessions: dict[str, int] = {}     # session_id → open /ws/motus count
_disconnect_tasks: dict[str, asyncio.Task] = {}
_DISCONNECT_GRACE = 8.0                 # seconds to tolerate a WS reconnect flap

_SWEEP_INTERVAL = 10.0                  # background expiry check cadence
_TAKEOVER_TIMEOUT = 15.0

# Keep the latest transition available to clients that missed the WebSocket push.
_editor_reason = ''
_editor_revision = 0
_editor_generation = 0
_editor_instance = uuid4().hex

_layout_rev = 0                         # bumped on every canvas_layout write


@dataclass(frozen=True)
class _Takeover:
    id: str
    requester: str
    deadline: float
    layout: CanvasSnapshot | None = None


_takeover: _Takeover | None = None
_takeover_result: dict | None = None


def _broadcast(event: dict) -> None:
    """Fire-and-forget push to /ws/motus clients.

    Local import breaks the cycle (motus_stream is imported by everything);
    tolerating "no running loop" mirrors apply_tool_config below — a missed
    broadcast degrades cross-tab freshness, not correctness, since clients also
    poll /canvas/edit-status.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    from api.motus_stream import push_event

    loop.create_task(push_event(event))


def _editor_state() -> dict:
    request = _takeover
    return {
        'editor': _editor_session,
        'reason': _editor_reason,
        'editor_instance': _editor_instance,
        'editor_revision': _editor_revision,
        'editor_generation': _editor_generation,
        'takeover': ({
            'id': request.id, 'requester': request.requester,
            'layout_source': 'personal' if request.layout is not None else 'shared',
            'remaining_seconds': max(0, math.ceil(request.deadline - time.monotonic())),
        } if request else None),
        'takeover_result': _takeover_result,
    }


def _broadcast_editor(reason: str) -> None:
    """Publish the lock and takeover together so they cannot disagree in clients."""
    global _editor_reason, _editor_revision
    _editor_reason = reason
    _editor_revision += 1
    _broadcast({'type': 'canvas_editor', 'payload': _editor_state()})


def _set_editor(session_id: str | None, reason: str, *, renew: bool = False) -> None:
    global _editor_session, _editor_last_seen, _editor_generation
    if renew or session_id != _editor_session:
        _editor_generation += 1
    _editor_session = session_id
    _editor_last_seen = time.monotonic() if session_id else 0.0
    _broadcast_editor(reason)


def _requester_present(request: _Takeover) -> bool:
    return request.requester not in _disconnect_tasks


def _settle_takeover(outcome: str, *, grant: bool = False, release_reason: str = '') -> None:
    global _takeover, _takeover_result
    request = _takeover
    if request is None:
        return
    if grant and request.layout is not None:
        if _can_apply_personal_layout():
            _apply_personal_layout(request.layout, request.requester)
        else:
            grant = False
            outcome = 'takeover_layout_locked'
    _takeover = None
    _takeover_result = {
        'id': request.id, 'requester': request.requester, 'outcome': outcome,
        'layout_source': 'personal' if request.layout is not None else 'shared',
    }
    if grant:
        _set_editor(request.requester, outcome)
    elif release_reason:
        _set_editor(None, release_reason)
    else:
        _broadcast_editor(outcome)


def _release_editor(reason: str) -> None:
    if _takeover is None:
        _set_editor(None, reason)
    elif _requester_present(_takeover):
        _settle_takeover('takeover_released', grant=True, release_reason=reason)
    else:
        _settle_takeover('takeover_cancelled', release_reason=reason)


def notify_layout_changed(session_id: str = '') -> None:
    """Tell every client the saved layout changed, so readers can reload.

    `session_id` is the writer — clients use it to ignore the echo of their own
    save. Also called by api/solutions.py, which rewrites canvas_layout directly.
    """
    global _layout_rev
    _layout_rev += 1
    _broadcast({'type': 'canvas_layout',
                'payload': {'editor': session_id, 'rev': _layout_rev}})


def _check_editor_expired():
    """Release the lock if its holder has been idle past the TTL.

    Deliberately does *not* consult _live_sessions: an open tab is not an active
    editor, and letting a live WS veto expiry is exactly what made the lock
    un-releasable. Only real actions push _editor_last_seen forward.
    """
    if not _editor_session:
        return
    now = time.monotonic()
    if _takeover and now >= _takeover.deadline:
        if _requester_present(_takeover):
            _settle_takeover('takeover_timeout', grant=True)
            return
        _settle_takeover('takeover_cancelled')
    if (now - _editor_last_seen) > _EDITOR_TIMEOUT:
        _release_editor('idle')


def _takeover_timeout(request_id: str) -> None:
    if _takeover and _takeover.id == request_id:
        if (remaining := _takeover.deadline - time.monotonic()) > 0:
            asyncio.get_running_loop().call_later(remaining, _takeover_timeout, request_id)
        else:
            _check_editor_expired()


async def _editor_sweeper() -> None:
    """Expire idle locks even when nothing is polling.

    _check_editor_expired is otherwise lazy, and the frontend's status poll lives
    in a setInterval that browsers throttle hard once the tab is backgrounded — so
    the one tab that most needs its lock reaped is the least likely to ask.
    """
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL)
        _check_editor_expired()


def start_editor_sweeper() -> None:
    """Launch the sweeper; called from start.py's lifespan."""
    asyncio.get_running_loop().create_task(_editor_sweeper())


def session_connected(session_id: str) -> None:
    """A /ws/motus connection opened for this session."""
    if not session_id:
        return
    if pending := _disconnect_tasks.pop(session_id, None):
        pending.cancel()
    _live_sessions[session_id] = _live_sessions.get(session_id, 0) + 1


def session_disconnected(session_id: str) -> None:
    """A /ws/motus connection closed; release the lock if nothing reconnects.

    Refcounted because one page may hold several connections and a reconnect can
    overlap the old socket's teardown — decrementing to zero is the only reliable
    "this session is gone" signal.
    """
    if not session_id:
        return
    remaining = _live_sessions.get(session_id, 0) - 1
    if remaining > 0:
        _live_sessions[session_id] = remaining
        return
    _live_sessions.pop(session_id, None)
    if pending := _disconnect_tasks.get(session_id):
        pending.cancel()
    _disconnect_tasks[session_id] = asyncio.get_running_loop().create_task(_grace_release(session_id))


async def _grace_release(session_id: str) -> None:
    """Release `session_id`'s lock unless it reconnects within the grace period."""
    await asyncio.sleep(_DISCONNECT_GRACE)
    _disconnect_tasks.pop(session_id, None)
    if _live_sessions.get(session_id):
        return                          # reconnected — keep editing
    if _takeover and _takeover.requester == session_id:
        _settle_takeover('takeover_cancelled')
    if _editor_session == session_id:
        _release_editor('disconnect')


def current_editor() -> Optional[str]:
    """Session id currently holding the edit lock, or None.

    Used by api/solutions.py: applying a solution rewrites canvas_layout behind
    the lock's back, so it has to refuse while someone is editing — their next
    autosave would otherwise clobber the freshly loaded layout.
    """
    _check_editor_expired()
    return _editor_session


def apply_tool_config(mcp_id: str, tool_name: str, body: Any,
                      instance_id: str = '') -> None:
    """Push a saved config down to the MCP plugin (fire-and-forget).

    Shared by the tool-config endpoints below and by api/solutions.py when a
    solution is applied, so both paths send the exact same `action: config`
    call shape.

    The body is partitioned by declared scope — shared settings are sent
    without instance_id even when saved under an instance key, because plugins
    holding one engine per process reject shared keys addressed to a single
    instance. See split_config_by_scope for why unknown keys are dropped.
    """
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    from tool_config import find_tool, plan_config_calls

    cfg = dict(body) if isinstance(body, dict) else {}
    tool_obj = find_tool(mcp_id, tool_name)
    if instance_id:
        calls = plan_config_calls(tool_obj, {}, cfg, instance_id)
    else:
        calls = plan_config_calls(tool_obj, cfg, {}, '')

    async def _apply():
        for cfg_body, extra_args in calls:
            try:
                req = MCPCallRequest(tool=tool_name,
                                     arguments={'action': 'config', **cfg_body, **extra_args})
                await mcp_call_tool(mcp_id, req)
            except Exception:
                pass

        # Read instance topics only after the config calls have finished; the
        # save endpoint returns before this background task applies them.
        if calls:
            _broadcast({'type': 'tool_config_updated', 'payload': {
                'mcp_id': mcp_id, 'tool': tool_name, 'instance_id': instance_id,
            }})

    try:
        asyncio.create_task(_apply())
    except RuntimeError:
        # No running loop (called from a sync context outside a request). The
        # config row is already persisted, and mcp_manage._restore_saved_configs
        # re-sends it when the device next comes online — so skipping the live
        # push here degrades timing, not correctness.
        pass


def tool_config_key(mcp_id: str, tool_name: str, instance_id: str = '') -> str:
    """ConfigDB key for a tool config row."""
    key = f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'
    return f'{key}:{instance_id}' if instance_id else key



class CanvasSnapshot(BaseModel):
    cards:           list  = []
    connections:     list  = []
    execConnections: list  = []
    transform:       dict  = {}


class DraftConfig(BaseModel):
    mcp_id: str
    tool_name: str
    instance_id: str = ''
    values: dict[str, Any]


class CanvasLayout(CanvasSnapshot):
    draft_configs: list[DraftConfig] = []
    session_id:      Optional[str] = None
    editor_generation: int | None = None
    editor_instance: str | None = None


class TakeoverRequest(BaseModel):
    session_id: str
    layout: CanvasSnapshot | None = None


class TakeoverResponse(BaseModel):
    session_id: str
    request_id: str
    action: str
    layout: CanvasLayout | None = None


def _holds_editor(session_id: str | None, generation: int | None, instance: str | None) -> bool:
    return bool(session_id and session_id == _editor_session
                and generation == _editor_generation and instance == _editor_instance)


# ── Editor Lock Endpoints ────────────────────────────────────────────────────

@router.post('/claim-edit')
async def claim_edit(body: dict = fastapi.Body(...)):
    """Request edit permission. Returns 423 if someone else is editing."""
    _check_editor_expired()

    session_id = body.get('session_id', '')
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
        return fastapi.responses.JSONResponse(
            status_code=400, content={'code': 400, 'message': 'session_id required'})

    if _editor_session and _editor_session != session_id:
        return _editor_conflict('Canvas is locked by another editor.', 423)

    _set_editor(session_id, 'claim')
    return {'code': 200, **_editor_state()}


def _editor_conflict(message: str, code: int = 409):
    return fastapi.responses.JSONResponse(
        status_code=code, content={'code': code, 'message': message, **_editor_state()},
    )


@router.post('/request-takeover')
async def request_takeover(body: TakeoverRequest):
    global _takeover, _takeover_result
    _check_editor_expired()
    session = body.session_id.strip()
    if not session or len(session) > 128:
        return _editor_conflict('Invalid canvas session.', 400)
    if _editor_session == session:
        return {'code': 200, **_editor_state()}
    if body.layout is not None and not _can_apply_personal_layout():
        return _editor_conflict('Stop intelligent control before applying a personal layout.')
    if _editor_session is None:
        if body.layout is not None:
            _apply_personal_layout(body.layout, session)
        _set_editor(session, 'takeover_personal' if body.layout is not None else 'claim')
        return {'code': 200, **_editor_state()}
    if _takeover:
        if _takeover.requester == session:
            return {'code': 200, **_editor_state()}
        return _editor_conflict('Another takeover request is pending.')
    _takeover = _Takeover(uuid4().hex, session, time.monotonic() + _TAKEOVER_TIMEOUT, body.layout)
    _takeover_result = None
    asyncio.get_running_loop().call_later(_TAKEOVER_TIMEOUT, _takeover_timeout, _takeover.id)
    _broadcast_editor('takeover_requested')
    return {'code': 200, **_editor_state()}


@router.post('/respond-takeover')
async def respond_takeover(body: TakeoverResponse):
    _check_editor_expired()
    old_cards = None
    request = _takeover
    if not request or request.id != body.request_id:
        return _editor_conflict('This takeover request has already closed.')
    if body.action == 'cancel':
        if body.session_id != request.requester:
            return _editor_conflict('Only the requester can cancel.', 403)
        _settle_takeover('takeover_cancelled')
    elif body.action in ('approve', 'deny'):
        if body.session_id != _editor_session:
            return _editor_conflict('Only the current editor can respond.', 403)
        if body.action == 'approve':
            if not _requester_present(request):
                _settle_takeover('takeover_cancelled')
                return _editor_conflict('The requester closed the canvas.')
            if body.layout is not None:
                layout = body.layout
                if layout.session_id != body.session_id or not _holds_editor(
                    layout.session_id, layout.editor_generation, layout.editor_instance,
                ):
                    return _editor_conflict('Not the current editor.', 403)
                if request.layout is None:
                    old_cards = _store_layout(layout, body.session_id)
            _settle_takeover('takeover_approved', grant=True)
            if _editor_session != request.requester:
                return _editor_conflict('Stop intelligent control before applying a personal layout.')
        else:
            _settle_takeover('takeover_denied')
    else:
        return _editor_conflict('Unknown takeover response.', 400)
    result = {'code': 200, **_editor_state()}
    if old_cards is not None:
        from api.config import stop_removed_cards
        await stop_removed_cards(old_cards, body.layout.cards)
    return result


@router.post('/release-edit')
async def release_edit(request: fastapi.Request):
    """Release edit permission.

    Body is parsed by hand rather than via Body(...) because the page-close path
    uses navigator.sendBeacon, which sends Content-Type: text/plain — FastAPI only
    JSON-decodes an `application/*json` body, so a declared dict param 422'd and
    the lock was never released.
    """
    try:
        body = json.loads(await request.body() or b'{}')
    except (ValueError, TypeError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    session_id = body.get('session_id', '')
    _check_editor_expired()

    if not _holds_editor(session_id, body.get('editor_generation'), body.get('editor_instance')):
        return _editor_conflict('Not the current editor.')

    _release_editor('release')
    return {'code': 200, **_editor_state()}


@router.post('/keep-edit')
async def keep_edit(body: dict = fastapi.Body(...)):
    """Renew the idle TTL. Sent by the frontend on real user input, throttled.

    This is the only heartbeat — /edit-status deliberately does not renew, because
    it is polled unconditionally and would keep any open tab's lock alive forever.
    """
    global _editor_last_seen
    _check_editor_expired()

    session_id = body.get('session_id', '')
    if not _holds_editor(session_id, body.get('editor_generation'), body.get('editor_instance')):
        # Already expired or taken — a ping cannot resurrect it, the client has to
        # re-claim. Returning the real holder lets it fix its local state in place.
        return _editor_conflict('Not the current editor.')

    _editor_last_seen = time.monotonic()
    return {'code': 200, **_editor_state()}


@router.get('/edit-status')
async def edit_status(session_id: str = ''):
    """Check who is currently editing. Pure query — does not renew the TTL."""
    _check_editor_expired()
    return {'code': 200, **_editor_state()}


# ── Layout Endpoints ─────────────────────────────────────────────────────────

@router.get('/layout')
async def get_layout():
    """Return the saved canvas layout + current editor info."""
    _check_editor_expired()
    data = config.main.get('canvas_layout', {'cards': []})
    return {'code': 200, 'data': data, 'rev': _layout_rev, **_editor_state()}


@router.post('/layout')
async def save_layout(layout: CanvasLayout):
    """Persist the canvas layout. Only the current editor can save."""
    global _editor_last_seen
    _check_editor_expired()

    session_id = layout.session_id
    if not _holds_editor(session_id, layout.editor_generation, layout.editor_instance):
        return _editor_conflict('Not the current editor.', 403)

    _editor_last_seen = time.monotonic()

    old_cards = _store_layout(layout, session_id)
    # An instance removed from the saved layout won't be found by stop-project.
    from api.config import stop_removed_cards
    await stop_removed_cards(old_cards, layout.cards)
    return {'code': 200}


@router.post('/restore-draft')
async def restore_draft(layout: CanvasLayout):
    """Replace the live layout from a browser draft under the current edit lease."""
    _check_editor_expired()
    session_id = layout.session_id
    if not session_id or len(session_id) > 128 or not session_id.strip():
        return _editor_conflict('Invalid canvas session.', 400)
    if _editor_session and _editor_session != session_id:
        return _editor_conflict('Canvas is locked by another editor.', 423)
    if layout.editor_generation != _editor_generation or layout.editor_instance != _editor_instance:
        return _editor_conflict('Canvas edit state changed. Try again.', 409)
    if _takeover:
        return _editor_conflict('Resolve the current takeover request before restoring a draft.')
    if not _can_apply_personal_layout():
        return _editor_conflict('Stop intelligent control before restoring a draft.')
    try:
        settings = _draft_config_updates(layout)
    except ValueError as error:
        return _editor_conflict(str(error), 422)
    _apply_personal_layout(layout, session_id, settings)
    # A save sent before restoring the draft must not overwrite it afterwards.
    _set_editor(session_id, 'draft_restored', renew=True)
    return {'code': 200, **_editor_state()}


def _draft_config_updates(layout: CanvasLayout) -> dict:
    from tool_config import find_tool
    cards = {(card.get('mcpId'), card.get('toolName'), card.get('id')) for card in layout.cards}
    tools = {(mcp_id, tool_name) for mcp_id, tool_name, _ in cards}
    updates = {}
    for entry in layout.draft_configs:
        tool = (entry.mcp_id, entry.tool_name)
        if tool not in tools or (entry.instance_id and (*tool, entry.instance_id) not in cards):
            continue
        schema = find_tool(*tool).get('configSchema') or {}
        props = schema.get('properties') or {}
        for name in entry.values:
            field = props.get(name)
            if (field is None or field.get('x-sensitive', field.get('format') == 'password')
                    or (field.get('scope') == 'instance') != bool(entry.instance_id)):
                raise ValueError(f'Invalid draft setting {entry.tool_name}.{name}. Refresh the card settings.')
        if entry.values:
            key = tool_config_key(*tool, entry.instance_id)
            updates[key] = {**(config.main.get(key) or {}), **entry.values}
    return updates


def _store_layout(layout: CanvasSnapshot, session_id: str, settings: dict | None = None) -> list:
    old_cards = (config.main.get('canvas_layout', {}) or {}).get('cards', [])
    snapshot = layout.model_dump(include=set(CanvasSnapshot.model_fields))
    if settings:
        config.main.update({**settings, 'canvas_layout': snapshot})
    else:
        config.main['canvas_layout'] = snapshot
    notify_layout_changed(session_id)
    return old_cards


def _can_apply_personal_layout() -> bool:
    from api.config import canvas_replacement_allowed
    return canvas_replacement_allowed()


def _apply_personal_layout(layout: CanvasSnapshot, requester: str, settings: dict | None = None) -> None:
    from api.config import stop_removed_cards
    old_cards = _store_layout(layout, requester, settings)
    asyncio.get_running_loop().create_task(stop_removed_cards(old_cards, layout.cards))
    for key, values in (settings or {}).items():
        mcp_id, tool_name, *instance = key.removeprefix(_TOOL_CONFIG_PREFIX).split(':', 2)
        apply_tool_config(mcp_id, tool_name, values, instance[0] if instance else '')


# ── Per-tool config CRUD ─────────────────────────────────────────────────────

@router.get('/tool-config/{mcp_id}/{tool_name}')
async def get_tool_config(mcp_id: str, tool_name: str):
    """Get saved config for a tool."""
    data = config.main.get(tool_config_key(mcp_id, tool_name), None)
    return {'code': 200, 'data': data}


def all_tool_configs() -> dict:
    """Every saved tool config, keyed by "mcp_id:tool_name[:instance_id]"."""
    result = {}
    try:
        conn = config._get_conn()
        rows = conn.execute(
            "SELECT key, value FROM config WHERE key LIKE ?",
            (f'{_TOOL_CONFIG_PREFIX}%',)
        ).fetchall()
        for key, value in rows:
            tool_key = key[len(_TOOL_CONFIG_PREFIX):]  # "mcp_id:tool_name"
            result[tool_key] = json.loads(value)
    except Exception:
        pass
    return result


def delete_all_tool_configs() -> int:
    """Drop every saved tool config. Returns the number of rows removed.

    Only used when a solution replaces the whole canvas: the incoming cards
    bring their own configs, and leftovers would keep pushing stale settings
    (old topics, old device paths) to plugins that the new canvas reuses.
    """
    try:
        conn = config._get_conn()
        cur = conn.execute("DELETE FROM config WHERE key LIKE ?",
                           (f'{_TOOL_CONFIG_PREFIX}%',))
        conn.commit()
        return cur.rowcount or 0
    except Exception:
        return 0


@router.get('/tool-configs')
async def get_all_tool_configs():
    """Batch-get all tool configs."""
    return {'code': 200, 'data': all_tool_configs()}



@router.put('/tool-config/{mcp_id}/{tool_name}')
async def save_tool_config(mcp_id: str, tool_name: str, body: Any = fastapi.Body(...)):
    """Save config for a tool and apply it to the MCP plugin."""
    config.main[tool_config_key(mcp_id, tool_name)] = body
    apply_tool_config(mcp_id, tool_name, body)
    return {'code': 200}


@router.delete('/tool-config/{mcp_id}/{tool_name}')
async def delete_tool_config(mcp_id: str, tool_name: str):
    """Delete config for a tool."""
    try:
        conn = config._get_conn()
        conn.execute("DELETE FROM config WHERE key = ?",
                     (tool_config_key(mcp_id, tool_name),))
        conn.commit()
    except Exception:
        pass
    return {'code': 200}


# ── Per-instance config CRUD ────────────────────────────────────────────────

@router.get('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def get_instance_config(mcp_id: str, tool_name: str, instance_id: str):
    """Get saved config for a specific tool instance."""
    data = config.main.get(tool_config_key(mcp_id, tool_name, instance_id), None)
    return {'code': 200, 'data': data}


@router.put('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def save_instance_config(mcp_id: str, tool_name: str, instance_id: str, body: Any = fastapi.Body(...)):
    """Save config for a specific tool instance and apply it."""
    config.main[tool_config_key(mcp_id, tool_name, instance_id)] = body
    apply_tool_config(mcp_id, tool_name, body, instance_id)
    return {'code': 200}


@router.delete('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def delete_instance_config(mcp_id: str, tool_name: str, instance_id: str):
    """Delete config for a specific tool instance."""
    try:
        conn = config._get_conn()
        conn.execute("DELETE FROM config WHERE key = ?",
                     (tool_config_key(mcp_id, tool_name, instance_id),))
        conn.commit()
    except Exception:
        pass
    return {'code': 200}
