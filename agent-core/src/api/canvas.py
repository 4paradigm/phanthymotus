"""
canvas.py — Canvas layout persistence + per-tool config storage.

Stores the orchestration canvas layout (card positions) and per-tool
configuration in the SQLite config table.
"""

import asyncio
import json
import fastapi
from pydantic import BaseModel
from typing import Any

import config

router = fastapi.APIRouter(prefix='/canvas', tags=['canvas'])

_TOOL_CONFIG_PREFIX = 'tool_config:'


def _is_inspector_tool(mcp_id: str, tool_name: str) -> bool:
    services = config.main.get('services', {}).get('mcp', [])
    mcp = next((item for item in services if item.get('id') == mcp_id), None) or {}
    if mcp.get('category') == 'inspection':
        return True
    tool = next(
        (item for item in (mcp.get('tools') or []) if isinstance(item, dict) and item.get('name') == tool_name),
        None,
    )
    return bool(tool and tool.get('type') == 'inspector')


async def _validate_inspector_config(
    mcp_id: str,
    tool_name: str,
    body: dict,
    *,
    instance_id: str = '',
) -> dict:
    from api.mcp_manage import mcp_call_tool, MCPCallRequest

    arguments = {'action': 'config', **body}
    if instance_id:
        arguments['instance_id'] = instance_id
    result = await mcp_call_tool(mcp_id, MCPCallRequest(tool=tool_name, arguments=arguments))
    if result.get('code') != 200:
        raise fastapi.HTTPException(status_code=400, detail=result.get('message') or 'Inspector config rejected')
    parsed: dict = {}
    content = result.get('data') or []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get('type') != 'text':
                continue
            try:
                value = json.loads(item.get('text', '{}'))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed = value
                break
    if not parsed.get('adapter_ok', True):
        raise fastapi.HTTPException(
            status_code=400,
            detail=parsed.get('message') or 'Inspector storage backend is not ready',
        )
    return parsed


class CanvasLayout(BaseModel):
    cards:           list  = []
    connections:     list  = []
    execConnections: list  = []
    transform:       dict  = {}


@router.get('/layout')
async def get_layout():
    """Return the saved canvas layout."""
    data = config.main.get('canvas_layout', {'cards': []})
    return {'code': 200, 'data': data}


@router.post('/layout')
async def save_layout(layout: CanvasLayout):
    """Persist the canvas layout to the config store."""
    config.main['canvas_layout'] = layout.dict()
    return {'code': 200}


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
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    if _is_inspector_tool(mcp_id, tool_name):
        applied = await _validate_inspector_config(mcp_id, tool_name, body)
        config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'] = body
        return {'code': 200, 'data': applied}

    config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'] = body

    # Existing non-Inspector tools keep their asynchronous apply behavior.
    async def _apply():
        try:
            req = MCPCallRequest(tool=tool_name, arguments={'action': 'config', **body})
            await mcp_call_tool(mcp_id, req)
        except Exception:
            pass
    asyncio.create_task(_apply())

    return {'code': 200}


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
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    if _is_inspector_tool(mcp_id, tool_name):
        applied = await _validate_inspector_config(
            mcp_id,
            tool_name,
            body,
            instance_id=instance_id,
        )
        config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}:{instance_id}'] = body
        return {'code': 200, 'data': applied}

    config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}:{instance_id}'] = body

    # Existing non-Inspector tools keep their asynchronous apply behavior.
    async def _apply():
        try:
            req = MCPCallRequest(tool=tool_name, arguments={'action': 'config', 'instance_id': instance_id, **body})
            await mcp_call_tool(mcp_id, req)
        except Exception:
            pass
    asyncio.create_task(_apply())
    return {'code': 200}


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
