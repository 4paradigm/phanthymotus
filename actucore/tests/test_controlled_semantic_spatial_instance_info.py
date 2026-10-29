"""Instance-aware navigation info, without ROS processes or robot access."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

# Pytest importlib mode does not add the test directory to sys.path. Resolve
# both paths from this file so invocation from either directory works.
TESTS_DIR = Path(__file__).resolve().parent
ACTUCORE_ROOT = TESTS_DIR.parent
for directory in (ACTUCORE_ROOT, TESTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from test_controlled_semantic_spatial_plugin import FakeComponent, FakeRuntime  # noqa: E402
from plugins.controlled_semantic_spatial.plugin import ControlledSemanticSpatialPlugin  # noqa: E402


class ControlledSemanticSpatialInstanceInfoTest(unittest.TestCase):
    def make_plugin(self, *, runtime=None, mapping=None, planning=None, semantic=None):
        return ControlledSemanticSpatialPlugin(
            {"namespace": "ubuntu"}, "ubuntu", None,
            runtime=runtime or FakeRuntime(),
            mapping_plugin=mapping or FakeComponent("mapping"),
            planning_plugin=planning or FakeComponent("planning"),
            semantic_plugin=semantic or FakeComponent("semantic"),
        )

    @staticmethod
    def call(plugin, action, **args):
        return plugin.dispatch(plugin.PREFIX, {"action": action, **args})

    @staticmethod
    def bindings():
        return [
            {"port": "lidar", "topic": "/custom/lidar"},
            {"port": "imu", "topic": "/custom/imu"},
            {"port": "rgb", "topic": "/custom/rgb"},
        ]

    def assert_actual(self, info):
        self.assertEqual(info["topic_in_scope"], "instance")
        self.assertEqual(info["topic_in_binding_state"], "known")
        self.assertIs(info["instance_match"], True)
        actual = {item["port"]: item for item in info["topic_in"]}
        for binding in self.bindings():
            item = actual[binding["port"]]
            self.assertEqual(item["topic"], binding["topic"])
            self.assertIs(item["connected"], True)
        for port in ("depth_frame", "goal_pose"):
            self.assertEqual(actual[port]["topic"], "")
            self.assertIs(actual[port]["connected"], False)

    def assert_unmatched(self, info, state):
        self.assertEqual(info["state"], state)
        self.assertEqual(info["status"], state)
        self.assertIs(info["instance_match"], False)
        self.assertEqual(info["topic_in_scope"], "instance")
        self.assertTrue(all(item["topic"] == "" for item in info["topic_in"]))
        self.assertTrue(all(item["connected"] is False for item in info["topic_in"]))
        self.assertTrue(info["topic_out"])

    def test_discovery_runtime_wrong_id_stop_and_restart(self):
        plugin = self.make_plugin()
        declared = plugin.get_tools()[0]["topic_in"]
        pre = self.call(plugin, "info", instance_id="card-a")
        self.assert_unmatched(pre, "idle")
        self.assertIsNone(pre["instance_id"])
        start = self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())
        self.assert_actual(start)
        self.assertEqual(start["state"], "ready")
        self.assertEqual(start["instance_id"], "card-a")
        self.assert_actual(self.call(
            plugin, "info", instance_id="card-a",
            input_bindings=[{"port": "lidar", "topic": "/untrusted"}],
        ))
        already = self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())
        self.assertIs(already["already_started"], True)
        self.assert_actual(already)
        # A different card may not claim the existing running instance.
        stale = self.call(plugin, "start", instance_id="card-b", input_bindings=[
            {"port": "lidar", "topic": "/different/lidar"},
            {"port": "imu", "topic": "/different/imu"},
        ])
        self.assertEqual(stale["error_code"], "navigation_instance_conflict")
        self.assertEqual(stale["instance_id"], "card-a")
        self.assert_actual(self.call(plugin, "info", instance_id="card-a"))
        for args in ({}, {"instance_id": ""}, {"instance_id": None}):
            info = self.call(plugin, "info", **args)
            self.assertEqual(info["state"], "ready")
            self.assertEqual(info["topic_in_scope"], "declaration")
            self.assertEqual([(p["port"], p["topic"]) for p in info["topic_in"]],
                             [(p["port"], p["topic"]) for p in declared])
            self.assertTrue(all(p["topic"] for p in info["topic_in"]))
        wrong = self.call(plugin, "info", instance_id="card-b")
        self.assert_unmatched(wrong, "ready")
        self.assertEqual(wrong["instance_id"], "card-a")
        self.assertEqual(self.call(plugin, "stop")["state"], "idle")
        stopped = self.call(plugin, "info", instance_id="card-a")
        self.assert_unmatched(stopped, "idle")
        self.assertIsNone(stopped["instance_id"])
        restarted = self.call(plugin, "start", instance_id="card-b", input_bindings=self.bindings())
        self.assert_actual(restarted)
        self.assert_unmatched(self.call(plugin, "info", instance_id="card-a"), "ready")

    def test_same_format_port_bindings_are_not_swapped(self):
        plugin = self.make_plugin()
        bindings = self.bindings() + [{"port": "depth_frame", "topic": "/custom/depth"}]
        for item in bindings:
            if item["port"] == "rgb":
                item["topic"] = "/custom/depth"
            elif item["port"] == "depth_frame":
                item["topic"] = "/custom/rgb"
        self.call(plugin, "start", instance_id="card-a", input_bindings=bindings)
        info = self.call(plugin, "info", instance_id="card-a")
        actual = {p["port"]: p["topic"] for p in info["topic_in"]}
        self.assertEqual(actual["rgb"], "/custom/depth")
        self.assertEqual(actual["depth_frame"], "/custom/rgb")

    def test_direct_start_without_id_uses_default_instance(self):
        plugin = self.make_plugin()
        start = self.call(plugin, "start", input_bindings=self.bindings())
        self.assertEqual(start["instance_id"], "default")
        self.assert_actual(start)
        self.assert_actual(self.call(plugin, "info", instance_id="default"))
        self.assertEqual(self.call(plugin, "info")["topic_in_scope"], "declaration")

    def test_failed_stop_preserves_global_state_and_accepted_binding(self):
        planning = FakeComponent("planning", stop_result={"state": "error", "status": "error"})
        plugin = self.make_plugin(planning=planning)
        self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())
        failed = self.call(plugin, "stop")
        self.assertEqual(failed["error_code"], "navigation_stop_pending")
        info = self.call(plugin, "info", instance_id="card-a")
        self.assertEqual(info["state"], "stopping")
        self.assert_actual(info)
        self.assert_unmatched(self.call(plugin, "info", instance_id="wrong"), "stopping")
        self.assertEqual(self.call(plugin, "info")["state"], "stopping")
        planning.stop_result = None
        self.assertEqual(self.call(plugin, "stop")["state"], "idle")
        self.assert_unmatched(self.call(plugin, "info", instance_id="card-a"), "idle")

    def test_cleanup_unconfirmed_is_unknown_until_stop_confirmation(self):
        planning = FakeComponent("planning", fail_start=True,
                                 stop_result={"state": "error", "status": "error"})
        plugin = self.make_plugin(planning=planning)
        failed = self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())
        self.assertFalse(failed["cleanup_confirmed"])
        info = self.call(plugin, "info", instance_id="card-a")
        self.assertEqual(info["state"], "stopping")
        self.assertIsNone(info["instance_id"])
        self.assertIs(info["instance_match"], False)
        self.assertEqual(info["topic_in_binding_state"], "unknown")
        self.assertTrue(all(item["topic"] == "" and item["connected"] is None
                            for item in info["topic_in"]))
        discovery = self.call(plugin, "info")
        self.assertEqual(discovery["topic_in_binding_state"], "unknown")
        self.assertTrue(all(item["connected"] is None for item in discovery["topic_in"]))
        self.assertTrue(all(item["topic"] for item in discovery["topic_in"]))
        planning.stop_result = None
        self.assertEqual(self.call(plugin, "stop")["state"], "idle")
        self.assert_unmatched(self.call(plugin, "info", instance_id="card-a"), "idle")

    def test_global_error_remains_visible_to_wrong_instance(self):
        runtime = FakeRuntime()
        plugin = self.make_plugin(runtime=runtime)
        self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())
        runtime.started = False
        actual = self.call(plugin, "info", instance_id="card-a")
        self.assertEqual(actual["state"], "error")
        self.assert_actual(actual)
        wrong = self.call(plugin, "info", instance_id="wrong")
        self.assert_unmatched(wrong, "error")
        self.assertEqual(wrong["error_code"], "navigation_runtime_unavailable")
        self.assertEqual(self.call(plugin, "info")["state"], "error")

    def test_start_in_progress_is_unknown_not_unwired(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingRuntime(FakeRuntime):
            def start(self, **kwargs):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test timeout")
                return super().start(**kwargs)

        plugin = self.make_plugin(runtime=BlockingRuntime())
        results = []
        thread = threading.Thread(target=lambda: results.append(
            self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            info = self.call(plugin, "info", instance_id="card-a")
            self.assertEqual(info["topic_in_binding_state"], "unknown")
            self.assertTrue(all(p["connected"] is None for p in info["topic_in"]))
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assert_actual(results[0])

    def test_heartbeat_uses_real_core_ping_and_preserves_all_ports(self):
        # Real consumer code; only the MCP HTTP transport is replaced by an in-process fake.
        manage_path = ACTUCORE_ROOT.parent / "agent-core" / "src" / "api" / "mcp_manage.py"
        if not manage_path.is_file():
            self.skipTest("Agent Core source unavailable for real heartbeat contract")
        tree = ast.parse(manage_path.read_text(encoding="utf-8"), filename=str(manage_path))
        ping_node = next(node for node in tree.body
                         if isinstance(node, ast.AsyncFunctionDef) and node.name == "_ping_mcp_http")
        # Compile the actual consumer function, without importing Agent Core's unrelated
        # database/config dependencies or requiring an HTTP listener.
        aiohttp = types.SimpleNamespace(ClientTimeout=lambda **kwargs: None, ClientSession=None)
        namespace = {"aiohttp": aiohttp, "asyncio": asyncio, "json": json}
        exec(compile(ast.Module(body=[ping_node], type_ignores=[]), str(manage_path), "exec"), namespace)
        planning = FakeComponent("planning")
        plugin = self.make_plugin(planning=planning)
        declarations = plugin.get_tools()[0]["topic_in"]
        calls = []
        json_module = json

        class Response:
            status = 200
            def __init__(self, result):
                self.result = result
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def json(self, content_type=None):
                return {"result": self.result}

        class Session:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def post(self, url, *, json, headers):
                method = json["method"]
                if method == "initialize":
                    return Response({"serverInfo": {"name": "fake"}})
                if method == "tools/list":
                    return Response({"tools": plugin.get_tools()})
                if method == "tools/call":
                    params = json["params"]
                    calls.append(params["arguments"])
                    result = plugin.dispatch(params["name"], params["arguments"])
                    return Response({"content": [{"type": "text", "text": json_module.dumps(result)}]})
                return Response({"resources": []})

        async def ping():
            with mock.patch.object(aiohttp, "ClientSession", Session):
                return await namespace["_ping_mcp_http"]("http://fake.invalid")

        def check():
            caps = asyncio.run(ping())
            tool = next(t for t in caps["tools"] if t["name"] == plugin.PREFIX)
            self.assertEqual([(p["port"], p["topic"]) for p in tool["topic_in"]],
                             [(p["port"], p["topic"]) for p in declarations])
            self.assertEqual(calls[-1], {"action": "info"})

        check()
        self.call(plugin, "start", instance_id="card-a", input_bindings=self.bindings())
        check()
        planning.stop_result = {"state": "error", "status": "error"}
        self.assertEqual(self.call(plugin, "stop")["error_code"], "navigation_stop_pending")
        check()
        planning.stop_result = None
        self.assertEqual(self.call(plugin, "stop")["state"], "idle")
        check()


if __name__ == "__main__":
    unittest.main()
