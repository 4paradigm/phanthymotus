"""The canvas gear saves Bumi's shared direction settings through MCP config."""

import asyncio
import sys
import types

from api import canvas
import config


def test_canvas_gear_sends_direction_config_without_instance_id(monkeypatch):
    tool = {"name": "sound_direction", "configSchema": {"properties": {
        "onset_ratio": {"scope": "shared"}}}}
    monkeypatch.setattr(config, "main", {"services": {"mcp": [
        {"id": "mcp-bumi", "tools": [tool]}]}})
    sent = []

    class Request:
        def __init__(self, tool, arguments):
            self.tool = tool
            self.arguments = arguments

    async def call(mcp_id, request):
        sent.append((mcp_id, request.tool, request.arguments))

    monkeypatch.setitem(sys.modules, "api.mcp_manage", types.SimpleNamespace(
        MCPCallRequest=Request, mcp_call_tool=call))

    async def save():
        result = await canvas.save_tool_config(
            "mcp-bumi", "sound_direction", {"onset_ratio": 2.4})
        await asyncio.sleep(0)
        return result

    assert asyncio.run(save()) == {"code": 200}
    assert config.main["tool_config:mcp-bumi:sound_direction"] == {"onset_ratio": 2.4}
    assert sent == [("mcp-bumi", "sound_direction",
                     {"action": "config", "onset_ratio": 2.4})]
