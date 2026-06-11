"""
canvas.py — Canvas layout persistence + per-tool config storage.

Stores the orchestration canvas layout (card positions) and per-tool
configuration in the SQLite config table.
"""

import json
import fastapi
from pydantic import BaseModel
from typing import Any

import config

router = fastapi.APIRouter(prefix='/canvas', tags=['canvas'])

_TOOL_CONFIG_PREFIX = 'tool_config:'


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
    config.main[f'{_TOOL_CONFIG_PREFIX}{mcp_id}:{tool_name}'] = body

    # Apply config to the MCP plugin
    from api.mcp_manage import mcp_call_tool, MCPCallRequest
    try:
        req = MCPCallRequest(tool=tool_name, arguments={'action': 'config', **body})
        await mcp_call_tool(mcp_id, req)
    except Exception:
        pass  # Config saved even if apply fails (MCP may be offline)

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
