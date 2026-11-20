"""Real loopback MCP transport tests; no ROS or hardware subscription."""

import json
import threading
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest
from plugins.tabletop_roi_goal_verifier.contracts import NAME
from plugins.tabletop_roi_goal_verifier.plugin import TabletopROIGoalVerifierPlugin
from plugins.tabletop_roi_goal_verifier.server import handler, rpc
from test_tabletop_roi_goal_verifier import configuration

urlopen = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


@contextmanager
def running():
    card = TabletopROIGoalVerifierPlugin(configuration())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(card))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield card, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(3)
        server.server_close()
        card.stop()


def test_mcp_discovery_call_and_browser_entry():
    with running() as (card, base):
        assert (
            rpc(base + "/mcp", "initialize")["serverInfo"]["name"]
            == "tabletop-roi-goal-verifier"
        )
        tools = rpc(base + "/mcp", "tools/list")["tools"]
        assert [t["name"] for t in tools] == [NAME]
        assert tools[0]["inputSchema"]["x-workbench"]["entry"] == "ui"
        for action in ("info", "ui", "start", "stop"):
            out = rpc(
                base + "/mcp",
                "tools/call",
                {"name": NAME, "arguments": {"action": action}},
            )
            assert not out["isError"]
        with urlopen(base + "/") as r:
            assert r.status == 200 and b"<canvas" in r.read()
        with urlopen(base + "/health") as r:
            assert json.load(r)["motion_capability"] is False
        assert card.sources.closed


@pytest.mark.parametrize(
    "body,reason",
    [
        ('{"jsonrpc":"2.0","id":1,"id":2}', "DUPLICATE_JSON_KEY"),
        ('{"jsonrpc":"2.0","id":1,"method":"unknown"}', "METHOD_NOT_FOUND"),
        (
            '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"piper_object_goal_verifier"}}',
            "UNKNOWN_TOOL",
        ),
    ],
)
def test_invalid_rpc_has_explicit_error(body, reason):
    with running() as (_, base):
        req = urllib.request.Request(
            base + "/mcp",
            data=body.encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req) as r:
            out = json.load(r)
        assert out["error"]["data"]["reason_code"] == reason


def test_browser_cross_origin_refused():
    with running() as (_, base):
        req = urllib.request.Request(
            base + "/mcp",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://foreign.invalid",
            },
        )
        with urlopen(req) as r:
            out = json.load(r)
        assert out["error"]["data"]["reason_code"] == "INVALID_ORIGIN"
