"""
canvas.py — Canvas layout persistence + per-tool config storage.

Stores the orchestration canvas layout (card positions) and per-tool
configuration in the SQLite config table.

Includes editor lock: only one session can edit at a time.
"""

import asyncio
from contextlib import nullcontext
import json
import time
import fastapi
from pydantic import BaseModel
from typing import Any, Optional

import config

router = fastapi.APIRouter(prefix='/canvas', tags=['canvas'])

_TOOL_CONFIG_PREFIX = 'tool_config:'
_LOCO_YAW_FIELDS = frozenset({'wz_limit', 'angular_acceleration'})
_loco_config_locks: dict[str, asyncio.Lock] = {}

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
_DISCONNECT_GRACE = 8.0                 # seconds to tolerate a WS reconnect flap

_SWEEP_INTERVAL = 10.0                  # background expiry check cadence

# Why the lock was last freed. Mirrored into /edit-status so a client that missed
# the canvas_editor push (WS down — the very case the poll exists for) can still
# tell "you idled out" from "someone released it".
_last_release_reason = ''

_layout_rev = 0                         # bumped on every canvas_layout write


def _broadcast(event: dict) -> None:
    """Fire-and-forget push to /ws/motus clients.

    Local import breaks the cycle (motus_stream is imported by everything);
    tolerating "no running loop" mirrors apply_tool_config below — a missed
    broadcast degrades cross-tab freshness, not correctness, since clients also
    poll /canvas/edit-status.
    """
    from api.motus_stream import push_event

    try:
        asyncio.create_task(push_event(event))
    except RuntimeError:
        pass


def _broadcast_editor(reason: str) -> None:
    """Tell every client who holds the lock now (None = free)."""
    global _last_release_reason
    if _editor_session is None:
        _last_release_reason = reason
    _broadcast({'type': 'canvas_editor',
                'payload': {'editor': _editor_session, 'reason': reason}})


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
    global _editor_session, _editor_last_seen
    if not _editor_session:
        return
    if (time.monotonic() - _editor_last_seen) > _EDITOR_TIMEOUT:
        _editor_session = None
        _editor_last_seen = 0.0
        _broadcast_editor('idle')


async def _editor_sweeper() -> None:
    """Expire idle locks even when nothing is polling.

    _check_editor_expired is otherwise lazy, and the frontend's status poll lives
    in a setInterval that browsers throttle hard once the tab is backgrounded — so
    the one tab that most needs its lock reaped is the least likely to ask.
    """
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL)
        try:
            _check_editor_expired()
        except Exception:
            pass                        # a sweep failure must not kill the loop


def start_editor_sweeper() -> None:
    """Launch the sweeper; called from start.py's lifespan."""
    try:
        asyncio.create_task(_editor_sweeper())
    except RuntimeError:
        pass                            # no loop (tests, import-time) — lazy checks still apply


def session_connected(session_id: str) -> None:
    """A /ws/motus connection opened for this session."""
    if not session_id:
        return
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
    try:
        asyncio.create_task(_grace_release(session_id))
    except RuntimeError:
        pass


async def _grace_release(session_id: str) -> None:
    """Release `session_id`'s lock unless it reconnects within the grace period."""
    global _editor_session, _editor_last_seen
    await asyncio.sleep(_DISCONNECT_GRACE)
    if _live_sessions.get(session_id):
        return                          # reconnected — keep editing
    if _editor_session != session_id:
        return
    _editor_session = None
    _editor_last_seen = 0.0
    _broadcast_editor('disconnect')


def current_editor() -> Optional[str]:
    """Session id currently holding the edit lock, or None.

    Used by api/solutions.py: applying a solution rewrites canvas_layout behind
    the lock's back, so it has to refuse while someone is editing — their next
    autosave would otherwise clobber the freshly loaded layout.
    """
    _check_editor_expired()
    return _editor_session


async def _refresh_navi_limits(mcp_id: str, instance_id: str) -> None:
    from api.config import payload_of
    from api.mcp_manage import mcp_call_tool, MCPCallRequest

    layout = config.main.get('canvas_layout') or {}
    cards = layout.get('cards') or []
    links = [link for link in layout.get('connections') or []
             if link.get('format') == 'control/velocity']
    drivers = {card['id'] for card in cards if card.get('id') and
               card.get('mcpId') == mcp_id and card.get('toolName') == 'loco_servo' and
               (not instance_id or card['id'] == instance_id)}
    failures = []
    for card in cards:
        if card.get('toolName') != 'navi' or not card.get('mcpId') or not card.get('id'):
            continue
        destinations = {link.get('toCardId') for link in links
                        if link.get('fromCardId') == card['id']}
        if len(destinations) != 1 or not destinations <= drivers:
            continue
        driver_id = next(iter(destinations))
        info = await mcp_call_tool(mcp_id, MCPCallRequest(
            tool='loco_servo', arguments={'action': 'info', 'instance_id': driver_id},
        ), timeout_s=10.0)
        descriptor = payload_of(info).get('control_interface')
        if not isinstance(descriptor, dict) or descriptor.get('control_interface') != 'motus.control/1':
            detail = info.get('message') or '没有可用的速度声明'
            failures.append(f'读取 loco_servo {driver_id} 失败：{detail}')
            continue
        result = await mcp_call_tool(card['mcpId'], MCPCallRequest(
            tool='navi', arguments={'action': 'config', 'instance_id': card['id'],
                                    'control_interface': descriptor},
        ), timeout_s=10.0)
        applied = payload_of(result)
        if not applied.get('control_interface_updated') and applied.get('status') != 'idle':
            detail = applied.get('message') or result.get('message') or '没有确认'
            failures.append(f"Navi {card['id']} 未采用新声明：{detail}")
    if failures:
        raise RuntimeError('；'.join(failures))


async def _apply_tool_config(mcp_id: str, tool_name: str, body: Any, instance_id: str,
                             *, confirm_save: bool = False) -> str:
    from api.config import payload_of
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    from tool_config import find_tool, plan_config_calls

    cfg = dict(body) if isinstance(body, dict) else {}
    tool_obj = find_tool(mcp_id, tool_name)
    if instance_id:
        calls = plan_config_calls(tool_obj, {}, cfg, instance_id)
    else:
        calls = plan_config_calls(tool_obj, cfg, {}, '')

    sent_yaw_fields = {key for values, _ in calls for key in values} & _LOCO_YAW_FIELDS
    update_yaw = tool_name == 'loco_servo' and bool(sent_yaw_fields)
    if confirm_save:
        missing_yaw = (cfg.keys() & _LOCO_YAW_FIELDS) - sent_yaw_fields
        if missing_yaw:
            raise fastapi.HTTPException(
                status_code=503, detail=f"当前驱动未声明转向配置项 {'、'.join(sorted(missing_yaw))}")
        # S10 exposes one servo. Keep related limits in one physical config
        # transaction even when cached metadata groups their scopes differently.
        settings = {key: value for values, _ in calls for key, value in values.items()}
        calls = [(settings, {'instance_id': instance_id} if instance_id else {})]

    lock = _loco_config_locks.setdefault(mcp_id, asyncio.Lock()) if update_yaw else nullcontext()
    async with lock:
        refresh_limits = False
        for cfg_body, extra_args in calls:
            try:
                req = MCPCallRequest(tool=tool_name,
                                     arguments={'action': 'config', **cfg_body, **extra_args})
                result = (await mcp_call_tool(mcp_id, req, timeout_s=10.0) if confirm_save
                          else await mcp_call_tool(mcp_id, req))
            except Exception as error:
                if confirm_save:
                    detail = error.detail if isinstance(error, fastapi.HTTPException) else str(error)
                    raise fastapi.HTTPException(status_code=503, detail=f'loco_servo 无法确认配置：{detail}')
                continue
            if update_yaw:
                settings = payload_of(result)
                if settings.get('ok') is True:
                    refresh_limits = True
                else:
                    detail = settings.get('message') or result.get('message') or '没有确认'
                    if confirm_save:
                        status = 422 if settings.get('ok') is False else 503
                        raise fastapi.HTTPException(status_code=status, detail=f'loco_servo：{detail}')
                    print(f'[canvas] {tool_name} did not adopt yaw settings: {detail}')
        if confirm_save:
            # The acknowledgement precedes persistence so rejected settings
            # are not picked up by the next project start.
            config.main[tool_config_key(mcp_id, tool_name, instance_id)] = body
        warning = ''
        if refresh_limits:
            try:
                await _refresh_navi_limits(mcp_id, instance_id)
            except Exception as error:
                warning = f'loco_servo 已采用配置，Navi 同步失败：{error}'
                print(f'[canvas] {warning}')

        # Read instance topics only after the config calls have finished.
        if calls:
            _broadcast({'type': 'tool_config_updated', 'payload': {
                'mcp_id': mcp_id, 'tool': tool_name, 'instance_id': instance_id,
            }})
        return warning


def apply_tool_config(mcp_id: str, tool_name: str, body: Any,
                      instance_id: str = '') -> None:
    """Apply a saved config asynchronously for solutions and ordinary cards."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    asyncio.create_task(_apply_tool_config(mcp_id, tool_name, body, instance_id))


def tool_config_key(mcp_id: str, tool_name: str, instance_id: str = '') -> str:
    """ConfigDB key for a tool config row."""
    key = f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'
    return f'{key}:{instance_id}' if instance_id else key



class CanvasLayout(BaseModel):
    cards:           list  = []
    connections:     list  = []
    execConnections: list  = []
    transform:       dict  = {}
    session_id:      Optional[str] = None


# ── Editor Lock Endpoints ────────────────────────────────────────────────────

@router.post('/claim-edit')
async def claim_edit(body: dict = fastapi.Body(...)):
    """Request edit permission. Returns 423 if someone else is editing."""
    global _editor_session, _editor_last_seen
    _check_editor_expired()

    session_id = body.get('session_id', '')
    if not session_id:
        return fastapi.responses.JSONResponse(
            status_code=400, content={'code': 400, 'message': 'session_id required'})

    if _editor_session and _editor_session != session_id:
        return fastapi.responses.JSONResponse(
            status_code=423, content={'code': 423, 'message': 'Canvas is locked by another editor',
                                      'editor': _editor_session})

    _editor_session = session_id
    _editor_last_seen = time.monotonic()
    _broadcast_editor('claim')
    return {'code': 200, 'editor': session_id}


@router.post('/release-edit')
async def release_edit(request: fastapi.Request):
    """Release edit permission.

    Body is parsed by hand rather than via Body(...) because the page-close path
    uses navigator.sendBeacon, which sends Content-Type: text/plain — FastAPI only
    JSON-decodes an `application/*json` body, so a declared dict param 422'd and
    the lock was never released.
    """
    global _editor_session, _editor_last_seen
    _check_editor_expired()

    try:
        body = json.loads(await request.body() or b'{}')
    except (ValueError, TypeError):
        body = {}
    session_id = body.get('session_id', '') if isinstance(body, dict) else ''

    if not session_id or _editor_session != session_id:
        return fastapi.responses.JSONResponse(
            status_code=409, content={'code': 409, 'message': 'Not the current editor',
                                      'editor': _editor_session})

    _editor_session = None
    _editor_last_seen = 0.0
    _broadcast_editor('release')
    return {'code': 200}


@router.post('/keep-edit')
async def keep_edit(body: dict = fastapi.Body(...)):
    """Renew the idle TTL. Sent by the frontend on real user input, throttled.

    This is the only heartbeat — /edit-status deliberately does not renew, because
    it is polled unconditionally and would keep any open tab's lock alive forever.
    """
    global _editor_last_seen
    _check_editor_expired()

    session_id = body.get('session_id', '')
    if not session_id or _editor_session != session_id:
        # Already expired or taken — a ping cannot resurrect it, the client has to
        # re-claim. Returning the real holder lets it fix its local state in place.
        return fastapi.responses.JSONResponse(
            status_code=409, content={'code': 409, 'message': 'Not the current editor',
                                      'editor': _editor_session})

    _editor_last_seen = time.monotonic()
    return {'code': 200, 'editor': _editor_session}


@router.get('/edit-status')
async def edit_status(session_id: str = ''):
    """Check who is currently editing. Pure query — does not renew the TTL."""
    _check_editor_expired()
    return {'code': 200, 'editor': _editor_session,
            'reason': '' if _editor_session else _last_release_reason}


# ── Layout Endpoints ─────────────────────────────────────────────────────────

@router.get('/layout')
async def get_layout():
    """Return the saved canvas layout + current editor info."""
    _check_editor_expired()
    data = config.main.get('canvas_layout', {'cards': []})
    return {'code': 200, 'data': data, 'editor': _editor_session, 'rev': _layout_rev}


@router.post('/layout')
async def save_layout(layout: CanvasLayout):
    """Persist the canvas layout. Only the current editor can save."""
    global _editor_session, _editor_last_seen
    _check_editor_expired()

    session_id = layout.session_id
    if session_id != _editor_session:
        return fastapi.responses.JSONResponse(
            status_code=403, content={'code': 403, 'message': 'Not the current editor',
                                      'editor': _editor_session})

    _editor_last_seen = time.monotonic()

    save_data = layout.dict()
    save_data.pop('session_id', None)
    old_cards = (config.main.get('canvas_layout', {}) or {}).get('cards', [])
    config.main['canvas_layout'] = save_data
    # A card that leaves the layout is unreachable afterwards — stop-project only
    # walks the saved cards — so its plugin instance would keep running forever.
    from api.config import stop_removed_cards
    await stop_removed_cards(old_cards, save_data.get('cards', []))
    notify_layout_changed(session_id or '')
    return {'code': 200}


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
    return await _save_tool_config(mcp_id, tool_name, body)


async def _save_tool_config(mcp_id: str, tool_name: str, body: Any, instance_id: str = ''):
    if tool_name == 'loco_servo' and isinstance(body, dict) and _LOCO_YAW_FIELDS & body.keys():
        warning = await _apply_tool_config(mcp_id, tool_name, body, instance_id, confirm_save=True)
        return {'code': 200, 'warning': warning} if warning else {'code': 200}
    config.main[tool_config_key(mcp_id, tool_name, instance_id)] = body
    apply_tool_config(mcp_id, tool_name, body, instance_id)
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
    return await _save_tool_config(mcp_id, tool_name, body, instance_id)


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
