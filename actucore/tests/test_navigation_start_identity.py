"""Ownership and wiring checks for repeat navigation starts."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
for directory in (TESTS_DIR.parent, TESTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from test_navigation_plugin import FakeComponent, FakeRuntime  # noqa: E402
from plugins.navigation.plugin import NavigationPlugin  # noqa: E402


class NavigationStartIdentityTest(unittest.TestCase):
    def make_plugin(self, runtime=None, mapping=None):
        self.runtime = runtime or FakeRuntime()
        self.mapping = mapping or FakeComponent("mapping")
        self.planning = FakeComponent("planning")
        self.semantic = FakeComponent("semantic")
        self.plugin = NavigationPlugin(
            {"namespace": "ubuntu"}, "ubuntu", None, runtime=self.runtime,
            mapping_plugin=self.mapping, planning_plugin=self.planning,
            semantic_plugin=self.semantic,
        )
        self.bindings = [
            {"port": "lidar", "topic": "/source/lidar"},
            {"port": "imu", "topic": "/source/imu"},
            {"port": "rgb", "topic": "/camera/color"},
            {"port": "depth_frame", "topic": "/camera/depth"},
        ]
        return self.plugin

    def call(self, action, **args):
        return self.plugin.dispatch(self.plugin.PREFIX, {"action": action, **args})

    def assert_owner(self, owner, expected):
        info = self.call("info", instance_id=owner)
        self.assertEqual(info["instance_id"], owner)
        self.assertTrue(info["instance_match"])
        self.assertEqual(info["state"], "ready")
        self.assertEqual({p["port"]: p["topic"] for p in info["topic_in"] if p["connected"]}, expected)
        self.assertTrue(self.runtime.started)

    def test_multiple_bindings_same_mapping_is_idempotent_not_sorted_assignment(self):
        self.make_plugin()
        start = self.call("start", instance_id="card-a", input_bindings=self.bindings)
        self.assertEqual(start["state"], "ready")
        expected = {b["port"]: b["topic"] for b in self.bindings}
        self.assertEqual(self.runtime.start_kwargs["input_topics"], expected)
        retry = self.call("start", instance_id="card-a", input_bindings=list(reversed(self.bindings)))
        self.assertTrue(retry["already_started"])
        self.assert_owner("card-a", expected)
        swaps = [dict(b) for b in self.bindings]
        swaps[2]["topic"], swaps[3]["topic"] = swaps[3]["topic"], swaps[2]["topic"]
        error = self.call("start", instance_id="card-a", input_bindings=swaps)
        self.assertEqual(error["error_code"], "navigation_wiring_conflict")
        self.assertIn("stop", error["error"])
        self.assert_owner("card-a", expected)
        self.assertEqual(self.runtime.stop_calls, 0)

    def test_missing_empty_invalid_and_changed_inputs_never_reuse_wiring(self):
        self.make_plugin()
        self.call("start", instance_id="a", input_bindings=self.bindings)
        expected = {b["port"]: b["topic"] for b in self.bindings}
        for args in ({}, {"input_bindings": []}, {"input_topics": []},
                     {"input_bindings": [{"port": "lidar", "topic": "/source/lidar"}]},
                     {"input_bindings": [{"port": "lidar", "topic": ""}]},
                     {"input_bindings": "invalid"},
                     {"input_topics": 1, "input_topic": "/source/imu"},
                     {"input_topic": 1}):
            with self.subTest(args=args):
                error = self.call("start", instance_id="a", **args)
                self.assertEqual(error["error_code"], "invalid_canvas_wiring")
                self.assert_owner("a", expected)
        changed = self.bindings[:2]
        error = self.call("start", instance_id="a", input_bindings=changed)
        self.assertEqual(error["error_code"], "navigation_wiring_conflict")
        self.assert_owner("a", expected)

    def test_different_id_and_legacy_default_owner(self):
        self.make_plugin()
        self.assertEqual(self.call("start", input_bindings=self.bindings)["instance_id"], "default")
        for value in (None, "", "default"):
            args = {"input_bindings": self.bindings}
            if value is not None:
                args["instance_id"] = value
            self.assertTrue(self.call("start", **args)["already_started"])
        conflict = self.call("start", instance_id="a", input_bindings=self.bindings)
        self.assertEqual(conflict["error_code"], "navigation_instance_conflict")
        self.assertIn("stop", conflict["error"])
        self.assert_owner("default", {b["port"]: b["topic"] for b in self.bindings})
        self.assertEqual(self.call("stop")["state"], "idle")
        self.assertEqual(self.call("start", instance_id="a", input_bindings=self.bindings)["state"], "ready")
        self.assertEqual(self.call("start", input_bindings=self.bindings)["error_code"], "navigation_instance_conflict")
        self.assert_owner("a", {b["port"]: b["topic"] for b in self.bindings})

    def test_null_and_whitespace_ids_share_legacy_default(self):
        self.make_plugin()
        self.assertEqual(self.call("start", instance_id=None, input_bindings=self.bindings)["instance_id"], "default")
        for value in (None, "", "   ", "default"):
            self.assertTrue(self.call("start", instance_id=value, input_bindings=self.bindings)["already_started"])

    def test_topic_list_order_is_irrelevant_but_changed_selection_is_not(self):
        self.make_plugin()
        defaults = {p["port"]: p["topic"] for p in self.plugin.get_tools()[0]["topic_in"]}
        topics = [defaults["lidar"], defaults["imu"], defaults["rgb"]]
        start = self.call("start", instance_id="a", input_topics=topics, input_topic=topics[0])
        self.assertEqual(start["state"], "ready")
        self.assertEqual(self.runtime.start_kwargs["input_topics"],
                         {port: defaults[port] for port in ("lidar", "imu", "rgb")})
        self.assertTrue(self.call("start", instance_id="a", input_topics=list(reversed(topics)))["already_started"])
        # Bindings have precedence and express the same accepted mapping.
        equivalent = [{"port": port, "topic": defaults[port]} for port in ("rgb", "imu", "lidar")]
        self.assertTrue(self.call("start", instance_id="a", input_bindings=equivalent)["already_started"])
        changed = self.call("start", instance_id="a", input_topics=topics[:2])
        self.assertEqual(changed["error_code"], "navigation_wiring_conflict")
        self.assert_owner("a", {port: defaults[port] for port in ("lidar", "imu", "rgb")})

    @unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("fastapi"),
                         "Agent Core dependencies are not installed")
    def test_agent_core_start_project_with_real_navigation_plugin(self):
        # Exercise the actual Core startup orchestration with a fake MCP
        # transport, including two upstream sources and their live topics.
        self.make_plugin()
        core_src = TESTS_DIR.parent.parent / "agent-core" / "src"
        sys.path.insert(0, str(core_src))
        try:
            with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"DB_PATH": str(Path(temp) / "core.db")}):
                import config
                import mcp_client
                from api import config as config_api
                import topic_actions

                calls = []
                sources = {"lidar-source": "/live/lidar", "imu-source": "/live/imu"}
                class Request:
                    def __init__(self, tool, arguments):
                        self.tool, self.arguments = tool, arguments

                async def call(mcp_id, req, timeout_s=None):
                    args = dict(req.arguments)
                    calls.append((req.tool, args))
                    if req.tool == self.plugin.PREFIX:
                        return {"code": 200, "data": self.plugin.dispatch(req.tool, args)}
                    return {"code": 200, "data": {"state": "ready", "topic_out": [
                        {"topic": sources[args["instance_id"]], "format": "data/json"}]}}

                async def nop(*args, **kwargs):
                    return None

                mcp = types.ModuleType("api.mcp_manage")
                mcp.mcp_call_tool, mcp.MCPCallRequest = call, Request
                stream = types.ModuleType("api.motus_stream")
                stream.push_event = nop
                inspection = types.ModuleType("api.inspection")
                inspection.register_topic_internal = nop
                channel = types.ModuleType("channel.manager")
                channel.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
                channel._get_channel_configs = lambda: []
                layout = {"cards": [
                    {"id": "nav-card", "toolName": self.plugin.PREFIX, "mcpId": "nav", "topicIn": [{"port": "lidar"}, {"port": "imu"}]},
                    {"id": "lidar-source", "toolName": "lidar", "mcpId": "source"},
                    {"id": "imu-source", "toolName": "imu", "mcpId": "source"},
                ], "connections": [
                    {"fromCardId": "lidar-source", "toCardId": "nav-card", "fromPortIdx": 0, "toPortIdx": 0},
                    {"fromCardId": "imu-source", "toCardId": "nav-card", "fromPortIdx": 0, "toPortIdx": 1},
                ]}
                registry = {"nav": {"tool_definitions": [self.plugin.get_tools()[0]]}}
                modules = {"api.mcp_manage": mcp, "api.motus_stream": stream,
                           "api.inspection": inspection, "channel.manager": channel}
                with mock.patch.dict(sys.modules, modules), \
                     mock.patch.object(config, "main", {"canvas_layout": layout, "core": {}, "services": {"mcp": []}}), \
                     mock.patch.object(mcp_client, "registry", registry), \
                     mock.patch.object(topic_actions.manager, "start", new=nop):
                    self.assertTrue(asyncio.run(config_api._do_start_project_impl()))
                    self.assertTrue(asyncio.run(config_api._do_start_project_impl()))
                starts = [args for tool, args in calls if tool == self.plugin.PREFIX and args["action"] == "start"]
                self.assertEqual(len(starts), 2)
                self.assertEqual(starts[0]["instance_id"], "nav-card")
                self.assertEqual(starts[1], starts[0])
                self.assertEqual(starts[0]["input_bindings"], [
                    {"port": "lidar", "topic": "/live/lidar"},
                    {"port": "imu", "topic": "/live/imu"},
                ])
                self.assert_owner("nav-card", {"lidar": "/live/lidar", "imu": "/live/imu"})
        finally:
            sys.path.remove(str(core_src))

    def test_stop_pending_refuses_start_and_retains_owner(self):
        self.make_plugin()
        self.call("start", instance_id="a", input_bindings=self.bindings)
        self.planning.stop_result = {"state": "error", "status": "error"}
        self.assertEqual(self.call("stop")["error_code"], "navigation_stop_pending")
        self.assertEqual(self.call("start", instance_id="a", input_bindings=self.bindings)["error_code"], "navigation_start_cancelled")
        info = self.call("info", instance_id="a")
        self.assertEqual(info["state"], "stopping")
        self.assertEqual(info["instance_id"], "a")
        self.assertTrue(info["instance_match"])
        self.planning.stop_result = None
        self.assertEqual(self.call("stop")["state"], "idle")
        self.assertEqual(self.call("start", instance_id="b", input_bindings=self.bindings)["state"], "ready")

    def test_confirmed_stop_then_new_owner_and_new_wiring(self):
        self.make_plugin()
        self.call("start", instance_id="a", input_bindings=self.bindings)
        changed = [dict(b) for b in self.bindings]
        changed[0]["topic"] = "/new/lidar"
        self.assertEqual(self.call("start", instance_id="b", input_bindings=changed)["error_code"], "navigation_instance_conflict")
        self.assertEqual(self.call("stop")["state"], "idle")
        new = self.call("start", instance_id="b", input_bindings=changed)
        self.assertEqual(new["state"], "ready")
        self.assert_owner("b", {b["port"]: b["topic"] for b in changed})

    def test_stop_signal_during_retry_validation_prevents_idempotent_success(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingMapping(FakeComponent):
            block = False
            def dispatch(self, prefix, args):
                if self.block and args.get("action") == "info":
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test timeout")
                return super().dispatch(prefix, args)

        mapping = BlockingMapping("mapping")
        self.make_plugin(mapping=mapping)
        self.call("start", instance_id="a", input_bindings=self.bindings)
        mapping.block = True
        results = {}
        retry = threading.Thread(target=lambda: results.update(retry=self.call("start", instance_id="a", input_bindings=self.bindings)))
        stop = threading.Thread(target=lambda: results.update(stop=self.call("stop")))
        retry.start()
        try:
            self.assertTrue(entered.wait(2))
            stop.start()
            self.assertTrue(self.plugin._stop_requested.wait(2))
        finally:
            release.set()
            retry.join(3)
            if stop.ident is not None:
                stop.join(3)
        self.assertFalse(retry.is_alive())
        self.assertFalse(stop.is_alive())
        self.assertEqual(results["retry"]["error_code"], "navigation_start_cancelled")
        self.assertEqual(results["stop"]["state"], "idle")
        self.assertIsNone(self.call("info", instance_id="a")["instance_id"])

    def test_concurrent_starts_compare_after_first_commits(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingRuntime(FakeRuntime):
            def start(self, **kwargs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test timeout")
                return super().start(**kwargs)

        self.make_plugin(runtime=BlockingRuntime())
        results = {}
        first = threading.Thread(target=lambda: results.update(first=self.call("start", instance_id="a", input_bindings=self.bindings)))
        second = threading.Thread(target=lambda: results.update(second=self.call("start", instance_id="b", input_bindings=self.bindings)))
        first.start()
        try:
            self.assertTrue(entered.wait(2))
            second.start()
        finally:
            release.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results["first"]["state"], "ready")
        self.assertEqual(results["second"]["error_code"], "navigation_instance_conflict")
        self.assert_owner("a", {b["port"]: b["topic"] for b in self.bindings})


if __name__ == "__main__":
    unittest.main()
