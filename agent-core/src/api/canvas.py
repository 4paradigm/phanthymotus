"""
canvas.py — Canvas layout persistence + per-tool config storage.

Stores the orchestration canvas layout (card positions) and per-tool
configuration in the SQLite config table.

Includes editor lock: only one session can edit at a time.
"""

import json
import time
import fastapi
from pydantic import BaseModel
from typing import Any, Optional

import config

router = fastapi.APIRouter(prefix='/canvas', tags=['canvas'])

_TOOL_CONFIG_PREFIX = 'tool_config:'
_RESERVED_TOOL_CONFIG_KEYS = frozenset({'action', 'instance_id'})


def _validated_tool_config(body: Any) -> dict:
    """Return a safe persisted config object or reject protocol-owned keys."""

    if not isinstance(body, dict):
        raise fastapi.HTTPException(
            status_code=422,
            detail='Tool config must be a JSON object',
        )
    reserved = sorted(_RESERVED_TOOL_CONFIG_KEYS.intersection(body))
    if reserved:
        raise fastapi.HTTPException(
            status_code=422,
            detail={
                'code': 'tool_config_reserved_key',
                'keys': reserved,
            },
        )
    return dict(body)


async def _apply_saved_tool_config(mcp_id: str, request):
    """Apply a persisted config through the broker-protected MCP path."""

    from api.mcp_manage import mcp_call_tool

    try:
        result = await mcp_call_tool(mcp_id, request)
    except fastapi.HTTPException as error:
        detail = error.detail if isinstance(error.detail, dict) else {}
        if error.status_code == 409 and detail.get('code') == 'teleop_command_blocked':
            return fastapi.responses.JSONResponse(
                status_code=202,
                content={
                    'code': 202,
                    'data': {
                        'saved': True,
                        'applied': False,
                        'deferred': True,
                        'reason': 'teleop_session_active',
                    },
                },
            )
        return fastapi.responses.JSONResponse(
            status_code=502,
            content={
                'code': 502,
                'message': 'Config was saved but could not be applied',
                'data': {
                    'saved': True,
                    'applied': False,
                    'deferred': False,
                    'reason': detail.get('code', 'driver_apply_failed'),
                },
            },
        )
    if isinstance(result, dict) and result.get('code') == 200:
        return {
            'code': 200,
            'data': {'saved': True, 'applied': True, 'deferred': False},
        }
    return fastapi.responses.JSONResponse(
        status_code=502,
        content={
            'code': 502,
            'message': 'Config was saved but Driver apply failed',
            'data': {
                'saved': True,
                'applied': False,
                'deferred': False,
                'reason': 'driver_apply_failed',
            },
        },
    )

# ── Editor Lock State (in-memory, resets on restart) ─────────────────────────

_editor_session: Optional[str] = None   # session_id of current editor
_editor_last_seen: float = 0.0          # monotonic timestamp of last activity
_EDITOR_TIMEOUT = 60.0                  # seconds before auto-release


def _check_editor_expired():
    """Release editor if inactive for too long."""
    global _editor_session, _editor_last_seen
    if _editor_session and (time.monotonic() - _editor_last_seen) > _EDITOR_TIMEOUT:
        _editor_session = None
        _editor_last_seen = 0.0


class CanvasLayout(BaseModel):
    cards:           list  = []
    connections:     list  = []
    execConnections: list  = []
    transform:       dict  = {}
    session_id:      Optional[str] = None
    revision:        Optional[int] = None


def _current_layout_revision() -> int:
    layout = config.main.get('canvas_layout', {})
    revision = layout.get('revision', 0) if isinstance(layout, dict) else 0
    return revision if type(revision) is int and revision >= 0 else 0


def validate_project_start_snapshot(
    session_id: str,
    expected_revision: int,
) -> Optional[dict]:
    """Validate that Start targets the layout visible to the current editor.

    The caller must hold the project transition lock.  Canvas saves reject while
    that lock is held, so a successful validation pins the revision until the
    start transaction has taken its own layout snapshot.
    """

    _check_editor_expired()
    if not session_id or _editor_session != session_id:
        return {
            'status_code': 409,
            'detail': {
                'code': 'project_editor_required',
                'reason': 'only_the_current_canvas_editor_can_start',
            },
            'errors': [],
        }

    current_revision = _current_layout_revision()
    if expected_revision != current_revision:
        return {
            'status_code': 409,
            'detail': {
                'code': 'project_layout_revision_conflict',
                'expected_revision': expected_revision,
                'current_revision': current_revision,
            },
            'errors': [],
        }
    return None


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
    return {'code': 200, 'editor': session_id}


@router.post('/release-edit')
async def release_edit(body: dict = fastapi.Body(...)):
    """Release edit permission."""
    global _editor_session, _editor_last_seen
    session_id = body.get('session_id', '')
    if _editor_session == session_id:
        _editor_session = None
        _editor_last_seen = 0.0
    return {'code': 200}


@router.get('/edit-status')
async def edit_status():
    """Check who is currently editing."""
    _check_editor_expired()
    return {'code': 200, 'editor': _editor_session}


# ── Layout Endpoints ─────────────────────────────────────────────────────────

@router.get('/layout')
async def get_layout():
    """Return the saved canvas layout + current editor info."""
    _check_editor_expired()
    data = config.main.get('canvas_layout', {'cards': []})
    data = dict(data) if isinstance(data, dict) else {'cards': []}
    data['revision'] = _current_layout_revision()
    return {'code': 200, 'data': data, 'editor': _editor_session}


@router.post('/layout')
async def save_layout(layout: CanvasLayout):
    """Persist the canvas layout. Only the current editor can save."""
    global _editor_session, _editor_last_seen
    _check_editor_expired()

    session_id = layout.session_id
    if _editor_session and session_id != _editor_session:
        return fastapi.responses.JSONResponse(
            status_code=403, content={'code': 403, 'message': 'Not the current editor',
                                      'editor': _editor_session})

    # Auto-claim if no editor (backward compat: first save becomes editor)
    if not _editor_session and session_id:
        _editor_session = session_id

    if session_id == _editor_session:
        _editor_last_seen = time.monotonic()

    core = config.main.get('core', {})
    from api import config as config_api
    if core.get('project_running', False) or config_api._project_transition_lock.locked():
        return fastapi.responses.JSONResponse(
            status_code=409,
            content={
                'code': 409,
                'detail': {
                    'code': 'project_topology_locked',
                    'project_state': core.get('project_state', 'transitioning'),
                },
            },
        )

    current_revision = _current_layout_revision()
    if layout.revision is not None and layout.revision != current_revision:
        return fastapi.responses.JSONResponse(
            status_code=409,
            content={
                'code': 409,
                'detail': {
                    'code': 'canvas_revision_conflict',
                    'expected_revision': layout.revision,
                    'current_revision': current_revision,
                },
            },
        )

    save_data = layout.model_dump()
    save_data.pop('session_id', None)
    save_data['revision'] = current_revision + 1
    config.main['canvas_layout'] = save_data
    return {'code': 200, 'data': {'revision': save_data['revision']}}


# ── Per-tool config CRUD ─────────────────────────────────────────────────────

@router.get('/tool-config/{mcp_id}/{tool_name}')
async def get_tool_config(mcp_id: str, tool_name: str):
    """Get saved config for a tool."""
    data = config.main.get(f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}', None)
    return {'code': 200, 'data': data}


@router.get('/tool-configs')
async def get_all_tool_configs():
    """Batch-get all tool configs."""
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
    return {'code': 200, 'data': result}


@router.put('/tool-config/{mcp_id}/{tool_name}')
async def save_tool_config(mcp_id: str, tool_name: str, body: Any = fastapi.Body(...)):
    """Save config for a tool and apply it to the MCP plugin."""
    safe_body = _validated_tool_config(body)
    config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'] = safe_body

    from api.mcp_manage import MCPCallRequest
    req = MCPCallRequest(
        tool=tool_name,
        arguments={**safe_body, 'action': 'config'},
    )
    return await _apply_saved_tool_config(mcp_id, req)


@router.delete('/tool-config/{mcp_id}/{tool_name}')
async def delete_tool_config(mcp_id: str, tool_name: str):
    """Delete config for a tool."""
    try:
        conn = config._get_conn()
        conn.execute("DELETE FROM config WHERE key = ?",
                     (f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}',))
        conn.commit()
    except Exception:
        pass
    return {'code': 200}


# ── Per-instance config CRUD ────────────────────────────────────────────────

@router.get('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def get_instance_config(mcp_id: str, tool_name: str, instance_id: str):
    """Get saved config for a specific tool instance."""
    data = config.main.get(f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}:{instance_id}', None)
    return {'code': 200, 'data': data}


@router.put('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def save_instance_config(mcp_id: str, tool_name: str, instance_id: str, body: Any = fastapi.Body(...)):
    """Save config for a specific tool instance and apply it."""
    safe_body = _validated_tool_config(body)
    config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}:{instance_id}'] = safe_body

    from api.mcp_manage import MCPCallRequest
    req = MCPCallRequest(
        tool=tool_name,
        arguments={
            **safe_body,
            'action': 'config',
            'instance_id': instance_id,
        },
    )
    return await _apply_saved_tool_config(mcp_id, req)


@router.delete('/tool-config/{mcp_id}/{tool_name}/{instance_id}')
async def delete_instance_config(mcp_id: str, tool_name: str, instance_id: str):
    """Delete config for a specific tool instance."""
    try:
        conn = config._get_conn()
        conn.execute("DELETE FROM config WHERE key = ?",
                     (f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}:{instance_id}',))
        conn.commit()
    except Exception:
        pass
    return {'code': 200}
