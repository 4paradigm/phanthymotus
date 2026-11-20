"""Regression coverage for Bumi live-stream Core/Driver coordination."""
import asyncio
import json

import aiohttp

from api import canvas, mcp_manage, solutions
import live_stream_auth


def _result(**payload):
    return {"code": 200, "data": [{"type": "text", "text": json.dumps(payload)}]}


def test_solution_never_exports_stream_credentials(monkeypatch):
    value = "rtmp://example.invalid/live/sample"
    monkeypatch.setattr(canvas, "all_tool_configs", lambda: {
        "mcp-qa:live_stream:card-a": {"rtmp_url": value, "label": "safe"}
    })
    monkeypatch.setattr(solutions, "_layout", lambda: {
        "cards": [{"id": "card-a", "mcpId": "mcp-qa", "toolName": "live_stream"}]
    })
    for schema in ({}, {"properties": {"rtmp_url": {
        "format": "password", "x-sensitive": True,
    }}}):
        monkeypatch.setattr(solutions, "_mcp_list", lambda s=schema: [{
            "id": "mcp-qa", "tools": [{"name": "live_stream", "configSchema": s}]
        }])
        packed, cleared = solutions._pack_canvas({"mcp-qa": "d0"}, set())
        assert packed["toolConfigs"]["d0:live_stream:card-a"] == {
            "rtmp_url": "", "label": "safe"
        }
        assert "d0:live_stream:card-a:rtmp_url" in cleared
        assert value not in json.dumps(packed)
        fields = solutions._config_field_paths({"mcp-qa": "d0"})
        assert next(x for x in fields if x["prop"] == "rtmp_url")["sensitive"]


def test_signed_canvas_save_rearm_start(monkeypatch, tmp_path):
    url = "rtmp://example.invalid/live/sample"
    state = {}
    monkeypatch.setattr(canvas.config, "main", state)
    monkeypatch.setattr(canvas, "_broadcast", lambda *_: None)
    monkeypatch.setattr(live_stream_auth, "sign_live_stream_action", lambda action, _url: 'test-proof:' + action)
    import tool_config
    monkeypatch.setattr(tool_config, "find_tool", lambda *_: {"name": "live_stream"})
    monkeypatch.setattr(tool_config, "plan_config_calls", lambda *_: [
        ({"rtmp_url": url}, {"instance_id": "card-a"})
    ])
    actions = []

    async def fake_call(_mcp, req, timeout_s=None):
        actions.append(dict(req.arguments))
        assert req.arguments.get("operator_proof")
        return (_result(ok=True, configured=True, ready_to_go_live=True)
                if req.arguments["action"] == "config" else
                _result(ok=True, state="starting"))

    monkeypatch.setattr(mcp_manage, "mcp_call_tool", fake_call)
    outcome = asyncio.run(canvas._save_live_stream_instance(
        "mcp-qa", "card-a", {"rtmp_url": url}))
    assert outcome["applied"]
    assert state[canvas.tool_config_key("mcp-qa", "live_stream", "card-a")]["rtmp_url"] == url
    assert "rtmp_url" not in outcome
    assert asyncio.run(canvas.authorize_saved_live_stream_start("mcp-qa", "card-a")) == {
        "code": 200, "ready_to_go_live": True
    }
    assert asyncio.run(canvas.go_live_from_canvas("mcp-qa", "card-a")) == {
        "code": 200, "state": "starting"
    }
    assert [x["action"] for x in actions] == ["config", "config", "go_live"]


def test_refused_config_does_not_persist(monkeypatch, tmp_path):
    url = "rtmp://example.invalid/live/sample"
    state = {}
    monkeypatch.setattr(canvas.config, "main", state)
    monkeypatch.setattr(live_stream_auth, "sign_live_stream_action", lambda action, _url: 'test-proof:' + action)
    import tool_config
    monkeypatch.setattr(tool_config, "find_tool", lambda *_: {"name": "live_stream"})
    monkeypatch.setattr(tool_config, "plan_config_calls", lambda *_: [
        ({"rtmp_url": url}, {})
    ])

    async def deny(*_args, **_kwargs):
        return _result(ok=False, error_code="APPROVAL_REQUIRED")

    monkeypatch.setattr(mcp_manage, "mcp_call_tool", deny)
    response = asyncio.run(canvas._save_live_stream_instance(
        "mcp-qa", "card-a", {"rtmp_url": url}))
    assert response.status_code == 422
    assert state == {}
    assert url not in response.body.decode()


def test_signed_go_live_reaches_driver_without_unsigned_config(monkeypatch):
    monkeypatch.setattr(mcp_manage, "_get_mcp_list", lambda: [{
        "id": "mcp-qa", "url": "http://test-driver/mcp",
        "transport": "http", "tools": [{"name": "live_stream"}]
    }])
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __await__(self):
            async def ready():
                return self
            return ready().__await__()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self, **_kwargs):
            return self.payload

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, _url, json, **_kwargs):
            calls.append(json)
            if json["method"] == "initialize":
                return Response({"result": {}})
            return Response({"result": {"content": [
                {"type": "text", "text": '{"ok":true,"state":"starting"}'}
            ]}})

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: Session())
    args = {"action": "go_live", "operator_proof": "test-proof"}
    req = mcp_manage.MCPCallRequest(tool="live_stream", arguments=args)
    result = asyncio.run(mcp_manage.mcp_call_tool("mcp-qa", req, timeout_s=5))
    assert result["code"] == 200, result
    sent = [x["params"]["arguments"] for x in calls
            if x["method"] == "tools/call"]
    assert sent == [args]
