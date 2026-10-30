"""Ownership and wiring checks for repeat navigation starts."""
from __future__ import annotations

import asyncio
import copy
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

from test_controlled_semantic_spatial_plugin import FakeComponent, FakeRuntime, _input_bindings  # noqa: E402
from plugins.controlled_semantic_spatial.plugin import ControlledSemanticSpatialPlugin  # noqa: E402


class ControlledSemanticSpatialStartIdentityTest(unittest.TestCase):
    def make_plugin(self, runtime=None, mapping=None):
        self.runtime = runtime or FakeRuntime()
        self.mapping = mapping or FakeComponent("mapping")
        self.planning = FakeComponent("planning")
        self.semantic = FakeComponent("semantic")
        self.plugin = ControlledSemanticSpatialPlugin(
            {"namespace": "ubuntu"}, "ubuntu", None, runtime=self.runtime,
            mapping_plugin=self.mapping, planning_plugin=self.planning,
            semantic_plugin=self.semantic,
        )
        self.bindings = [
            {"port": "lidar", "topic": "/source/lidar"},
            {"port": "rgb", "topic": "/camera/color"},
            {"port": "imu", "topic": "/source/imu"},
        ]
        self.direct_bindings = _input_bindings(self.bindings)
        return self.plugin

    def expected(self, bindings=None):
        return {b["port"]: b["topic"] for b in (self.bindings if bindings is None else bindings)}

    def call(self, action, **args):
        return self.plugin.dispatch(self.plugin.PREFIX, {"action": action, **args})

    def assert_owner(self, owner, expected):
        info = self.call("info", instance_id=owner)
        self.assertEqual(info["instance_id"], owner)
        self.assertTrue(info["instance_match"])
        self.assertEqual(info["state"], "ready")
        self.assertEqual({p["port"]: p["topic"] for p in info["topic_in"] if p["connected"]},
                         expected)
        self.assertTrue(self.runtime.started)

    def test_multiple_bindings_same_mapping_is_idempotent_not_sorted_assignment(self):
        self.make_plugin()
        start = self.call("start", instance_id="card-a", input_bindings=self.direct_bindings)
        self.assertEqual(start["state"], "ready")
        expected = self.expected()
        self.assertEqual(self.runtime.start_kwargs["input_topics"], expected)
        mapping_start = next(args for _, args in self.mapping.calls if args["action"] == "start")
        self.assertEqual({item["port"]: item["topic"] for item in mapping_start["input_bindings"]},
                         {"lidar": expected["lidar"], "imu": expected["imu"]})
        retry = self.call("start", instance_id="card-a", input_bindings=list(reversed(self.direct_bindings)))
        self.assertTrue(retry["already_started"])
        self.assert_owner("card-a", expected)
        swaps = [{"port": "lidar", "topic": "/camera/color"},
                 {"port": "rgb", "topic": "/source/lidar"},
                 self.bindings[2]]
        error = self.call("start", instance_id="card-a", input_bindings=swaps)
        self.assertEqual(error["error_code"], "navigation_wiring_conflict")
        self.assert_owner("card-a", expected)
        self.assertEqual(self.runtime.stop_calls, 0)

    def test_missing_empty_invalid_and_changed_inputs_never_reuse_wiring(self):
        self.make_plugin()
        self.assertEqual(self.call("start", instance_id="a", input_bindings=self.direct_bindings)["state"], "ready")
        expected = self.expected()
        for args in ({}, {"input_bindings": []}, {"input_topics": []},
                     {"input_bindings": "invalid"},
                     {"input_bindings": self.direct_bindings, "input_topics": 1},
                     {"input_bindings": self.direct_bindings, "input_topic": 1}):
            with self.subTest(args=args):
                error = self.call("start", instance_id="a", **args)
                self.assertEqual(error["error_code"], "invalid_canvas_wiring")
                self.assert_owner("a", expected)
        changed = [b for b in self.direct_bindings if b["port"] != "rgb"]  # RGB is optional, but changes identity.
        error = self.call("start", instance_id="a", input_bindings=changed)
        self.assertEqual(error["error_code"], "navigation_wiring_conflict")
        self.assert_owner("a", expected)

    def test_different_id_and_legacy_default_owner(self):
        self.make_plugin()
        self.assertEqual(self.call("start", input_bindings=self.direct_bindings)["instance_id"], "default")
        for value in (None, "", "default"):
            args = {"input_bindings": self.direct_bindings}
            if value is not None:
                args["instance_id"] = value
            self.assertTrue(self.call("start", **args)["already_started"])
        conflict = self.call("start", instance_id="a", input_bindings=self.direct_bindings)
        self.assertEqual(conflict["error_code"], "navigation_instance_conflict")
        self.assertIn("stop", conflict["error"])
        self.assert_owner("default", self.expected())
        self.assertEqual(self.call("stop")["state"], "idle")
        self.assertEqual(self.call("start", instance_id="a", input_bindings=self.direct_bindings)["state"], "ready")
        self.assertEqual(self.call("start", input_bindings=self.direct_bindings)["error_code"], "navigation_instance_conflict")
        self.assert_owner("a", self.expected())

    def test_null_and_whitespace_ids_share_legacy_default(self):
        self.make_plugin()
        self.assertEqual(self.call("start", instance_id=None, input_bindings=self.direct_bindings)["instance_id"], "default")
        for value in (None, "", "   ", "default"):
            self.assertTrue(self.call("start", instance_id=value, input_bindings=self.direct_bindings)["already_started"])

    def test_topic_only_requires_configured_match_and_is_order_independent(self):
        self.make_plugin()
        topics = [b["topic"] for b in self.bindings]
        self.assertEqual(self.call("start", input_topics=topics)["error_code"], "invalid_canvas_wiring")
        self.assertEqual(self.call("config", lidar_topic=topics[0], rgb_topic=topics[1], imu_topic=topics[2])["state"], "configured")
        self.assertEqual(self.call("start", instance_id="a", input_topics=list(reversed(topics)), input_topic=topics[1])["state"], "ready")
        self.assertTrue(self.call("start", instance_id="a", input_bindings=list(reversed(self.bindings)), input_topics=topics)["already_started"])
        self.assertEqual(self.call("start", instance_id="a", input_bindings=self.bindings, input_topics=topics[:1])["error_code"], "invalid_canvas_wiring")
        self.assert_owner("a", self.expected())

    def test_required_ports_and_ambiguous_wiring_are_rejected(self):
        self.make_plugin()
        defaults = {p["port"]: p["topic"] for p in self.plugin.get_tools()[0]["topic_in"]}
        cases = [
            {"input_topics": [defaults["lidar"]]},
            {"input_topics": [defaults["imu"]]},
            {"input_topics": [defaults["lidar"], defaults["imu"], "/unknown"]},
            {"input_topics": [defaults["lidar"], defaults["lidar"], defaults["imu"]]},
            {"input_bindings": [self.bindings[0]]},
            {"input_bindings": [self.bindings[2]]},
            {"input_bindings": [self.bindings[0], {"port": "imu", "topic": self.bindings[0]["topic"]}]},
            {"input_bindings": [self.bindings[0], self.bindings[2], {"port": "other", "topic": "/other"}]},
            {"input_bindings": [self.bindings[0], self.bindings[2]], "input_topics": [defaults["lidar"], defaults["imu"]]},
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self.call("start", **args)["error_code"], "invalid_canvas_wiring")
                self.assertFalse(self.runtime.started)

    def test_default_topics_without_rgb_are_bound_to_runtime_and_mapper(self):
        self.make_plugin()
        ports = self.plugin.get_tools()[0]["topic_in"]
        self.assertEqual([p["port"] for p in ports], ["lidar", "rgb", "imu"])
        self.assertEqual(ports[2]["format"], "sensor/imu")
        self.assertTrue(ports[2]["required"])
        defaults = {p["port"]: p["topic"] for p in ports}
        expected = {p: defaults[p] for p in ("lidar", "imu")}
        self.assertEqual(self.call("start", instance_id="a", input_topics=list(expected.values()))["state"], "ready")
        self.assertEqual(self.runtime.start_kwargs["input_topics"], expected)
        mapping_start = next(args for _, args in self.mapping.calls if args["action"] == "start")
        self.assertEqual({b["port"]: b["topic"] for b in mapping_start["input_bindings"]}, expected)
        info = self.call("info", instance_id="a")
        self.assertEqual({p["port"] for p in info["topic_in"] if p["connected"]}, {"lidar", "imu"})
        self.assertFalse(next(p for p in info["topic_in"] if p["port"] == "rgb")["connected"])

    @unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("fastapi"),
                         "Agent Core dependencies are not installed")
    def test_agent_core_start_project_with_real_navigation_plugin(self):
        # Exercise the actual Core startup orchestration with a fake MCP
        # transport and real lidar/IMU source declarations.
        self.make_plugin()
        core_src = TESTS_DIR.parent.parent / "agent-core" / "src"
        sys.path.insert(0, str(core_src))
        try:
            with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, {"DB_PATH": str(Path(temp) / "core.db")}):
                import config
                import mcp_client
                from api import config as config_api
                calls = []
                defaults = {p["port"]: p["topic"] for p in self.plugin.get_tools()[0]["topic_in"]}
                sources = {"lidar-source": {"topic": defaults["lidar"], "format": "sensor/pointcloud", "port": "navigation_lidar"},
                           "imu-source": {"topic": defaults["imu"], "format": "sensor/imu", "port": "navigation_imu"}}
                class Request:
                    def __init__(self, tool, arguments):
                        self.tool, self.arguments = tool, arguments

                async def call(mcp_id, req, timeout_s=None):
                    args = dict(req.arguments)
                    calls.append((req.tool, args))
                    if req.tool == self.plugin.PREFIX:
                        return {"code": 200, "data": self.plugin.dispatch(req.tool, args)}
                    return {"code": 200, "data": {"state": "ready", "topic_out": [sources[args["instance_id"]]]}}

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
                    {"id": "nav-card", "toolName": self.plugin.PREFIX, "mcpId": "nav", "topicIn": [{"port": "lidar"}, {"port": "rgb"}, {"port": "imu"}]},
                    {"id": "lidar-source", "toolName": "lidar", "mcpId": "source"},
                    {"id": "imu-source", "toolName": "imu", "mcpId": "source"},
                ], "connections": [
                    {"fromCardId": "lidar-source", "toCardId": "nav-card", "fromPortIdx": 0, "toPortIdx": 0,
                     "fromPort": "navigation_lidar", "toPort": "lidar", "format": sources["lidar-source"]["format"]},
                    {"fromCardId": "imu-source", "toCardId": "nav-card", "fromPortIdx": 0, "toPortIdx": 2,
                     "fromPort": "navigation_imu", "toPort": "imu", "format": sources["imu-source"]["format"]},
                ]}
                registry = {"nav": {"tool_definitions": [self.plugin.get_tools()[0]]}}
                modules = {"api.mcp_manage": mcp, "api.motus_stream": stream,
                           "api.inspection": inspection, "channel.manager": channel}
                with mock.patch.dict(sys.modules, modules), \
                     mock.patch.object(config, "main", {"canvas_layout": layout, "core": {}, "services": {"mcp": [{"id": "nav", "tools": [self.plugin.get_tools()[0]]}]}}), \
                     mock.patch.object(mcp_client, "registry", registry):
                    self.assertTrue(asyncio.run(config_api._do_start_project_impl()))
                    self.assertTrue(asyncio.run(config_api._do_start_project_impl()))
                starts = [args for tool, args in calls if tool == self.plugin.PREFIX and args["action"] == "start"]
                self.assertEqual(len(starts), 2)
                self.assertEqual(starts[0]["instance_id"], "nav-card")
                self.assertEqual(starts[1], starts[0])
                self.assertEqual(set(starts[0]["input_topics"]), {defaults["lidar"], defaults["imu"]})
                self.assertIn(starts[0]["input_topic"], starts[0]["input_topics"])
                self.assertNotIn("input_bindings", starts[0])
                expected = {"lidar": defaults["lidar"], "imu": defaults["imu"]}
                self.assertEqual(self.runtime.start_kwargs["input_topics"], expected)
                self.assert_owner("nav-card", expected)
        finally:
            sys.path.remove(str(core_src))

    def test_stop_pending_refuses_start_and_retains_owner(self):
        self.make_plugin()
        self.call("start", instance_id="a", input_bindings=self.direct_bindings)
        self.planning.stop_result = {"state": "error", "status": "error"}
        self.assertEqual(self.call("stop")["error_code"], "navigation_stop_pending")
        self.assertEqual(self.call("start", instance_id="a", input_bindings=self.direct_bindings)["error_code"], "navigation_start_cancelled")
        info = self.call("info", instance_id="a")
        self.assertEqual(info["state"], "stopping")
        self.assertEqual(info["instance_id"], "a")
        self.assertTrue(info["instance_match"])
        self.planning.stop_result = None
        self.assertEqual(self.call("stop")["state"], "idle")
        self.assertEqual(self.call("start", instance_id="b", input_bindings=self.direct_bindings)["state"], "ready")

    def test_confirmed_stop_then_new_owner_and_new_wiring(self):
        self.make_plugin()
        self.call("start", instance_id="a", input_bindings=self.direct_bindings)
        changed = [dict(b) for b in self.bindings]
        changed[0]["topic"] = "/new/lidar"
        self.assertEqual(self.call("start", instance_id="b", input_bindings=_input_bindings(changed))["error_code"], "navigation_instance_conflict")
        self.assertEqual(self.call("stop")["state"], "idle")
        new = self.call("start", instance_id="b", input_bindings=_input_bindings(changed))
        self.assertEqual(new["state"], "ready")
        self.assert_owner("b", self.expected(changed))

    def test_stop_signal_during_retry_validation_prevents_idempotent_success(self):
        entered, release = threading.Event(), threading.Event()

        self.make_plugin()
        self.call("start", instance_id="a", input_bindings=self.direct_bindings)
        validate = self.plugin._validate_external_wiring

        def blocking_validate(args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test timeout")
            return validate(args)

        self.plugin._validate_external_wiring = blocking_validate
        results = {}
        retry = threading.Thread(target=lambda: results.update(retry=self.call("start", instance_id="a", input_bindings=self.direct_bindings)))
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
        first = threading.Thread(target=lambda: results.update(first=self.call("start", instance_id="a", input_bindings=self.direct_bindings)))
        second = threading.Thread(target=lambda: results.update(second=self.call("start", instance_id="b", input_bindings=self.direct_bindings)))
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
        self.assert_owner("a", self.expected())


if __name__ == "__main__":
    unittest.main()
