from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
import unittest
import urllib.request
from pathlib import Path


PERCEPTION_DIR = Path(__file__).resolve().parents[3]
MAIN_PATH = PERCEPTION_DIR / "main.py"
sys.path.insert(0, str(PERCEPTION_DIR))


def _load_perception_main():
    rclpy = types.ModuleType("rclpy")
    executors = types.ModuleType("rclpy.executors")
    executors.MultiThreadedExecutor = object
    rclpy.executors = executors
    sys.modules.setdefault("rclpy", rclpy)
    sys.modules.setdefault("rclpy.executors", executors)

    spec = importlib.util.spec_from_file_location("vlapi05g1_bundle_main", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Perception Bundle entry point")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeExecutor:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_node(self, node):
        self.added.append(node)
        return True

    def remove_node(self, node):
        self.removed.append(node)
        return True


class _FakeVLAPi05G1Node:
    created = []

    def __init__(
        self,
        *,
        image_topic,
        state_topic,
        output_topic,
        instance_id,
        shared_config,
        instance_config,
    ):
        self.image_topic = image_topic
        self.state_topic = state_topic
        self.output_topic = output_topic
        self.instance_id = instance_id
        self.shared_config = dict(shared_config)
        self.instance_config = dict(instance_config)
        self.shutdown_count = 0
        self.destroy_count = 0
        self.__class__.created.append(self)

    def status(self):
        return {
            "name": "π0.5 G1 VLA 推理",
            "state": "running",
            "instance_id": self.instance_id,
            "topic_in": [
                {"topic": self.image_topic, "format": "image/jpeg"},
                {"topic": self.state_topic, "format": "data/json"},
            ],
            "topic_out": [{"topic": self.output_topic, "format": "data/json"}],
            "config": dict(self.instance_config),
            "in_flight": False,
        }

    def update_instance_config(self, config):
        self.instance_config = dict(config)

    def shutdown_card(self):
        self.shutdown_count += 1

    def destroy_node(self):
        self.destroy_count += 1


class BundleIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _load_perception_main()

    def setUp(self):
        self.executor = _FakeExecutor()
        self.cfg = {
            "identity": {
                "core_display_name": "Perception Stack · Shanghai G1 Test",
                "mcp_server_name": "perception-stack-shanghai-g1-test",
            },
            "plugins": {
                "vlapi05g1": {
                    "enabled": True,
                    "policy_url": "http://127.0.0.1:1/predict",
                    "health_url": "",
                    "request_timeout_s": 0.1,
                }
            },
        }
        _FakeVLAPi05G1Node.created.clear()
        fake_node_module = types.ModuleType("plugins.vlapi05g1.node")
        fake_node_module.VLAPi05G1Node = _FakeVLAPi05G1Node
        self.node_module_name = "plugins.vlapi05g1.node"
        self.previous_node_module = sys.modules.get(self.node_module_name)
        sys.modules[self.node_module_name] = fake_node_module

        with self.assertLogs(self.main.log, level="INFO") as captured:
            self.bundle = self.main.PerceptionBundle(self.cfg, self.executor)
        self.load_logs = captured.output
        self.main._bundle = self.bundle
        self.server = self.main.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            self.main.make_handler(self.cfg["identity"]["mcp_server_name"]),
        )
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/mcp"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.request_id = 0

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2.0)
        self.main._bundle = None
        if self.previous_node_module is None:
            sys.modules.pop(self.node_module_name, None)
        else:
            sys.modules[self.node_module_name] = self.previous_node_module

    def rpc(self, method, params=None):
        self.request_id += 1
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params or {},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(request, timeout=2.0) as response:
            return json.loads(response.read())

    def call_tool(self, arguments):
        response = self.rpc(
            "tools/call",
            {"name": "vlapi05g1", "arguments": arguments},
        )
        if "error" in response:
            return response
        text = response["result"]["content"][0]["text"]
        return json.loads(text)

    def test_bundle_mcp_schema_config_and_idempotent_lifecycle(self):
        self.assertTrue(any("VLAPi05G1Plugin loaded" in line for line in self.load_logs))

        initialize = self.rpc("initialize")
        self.assertEqual(
            "perception-stack-shanghai-g1-test",
            initialize["result"]["serverInfo"]["name"],
        )
        self.assertNotEqual("perception-bundle", initialize["result"]["serverInfo"]["name"])

        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual(1, len(tools))
        tool = tools[0]
        self.assertEqual("vlapi05g1", tool["name"])
        self.assertEqual("processor", tool["type"])
        self.assertTrue(tool["multiInstance"])
        self.assertIn("configSchema", tool)
        self.assertIn("action", tool["inputSchema"]["properties"])
        self.assertEqual(
            2,
            tool["inputSchema"]["properties"]["input_topics"]["minItems"],
        )
        self.assertEqual(2, len(tool["topic_in"]))
        self.assertEqual(1, len(tool["topic_out"]))

        invalid = self.call_tool({"action": "config", "request_timeout_s": 0.0})
        self.assertEqual(-32603, invalid["error"]["code"])

        valid = self.call_tool(
            {
                "action": "config",
                "policy_url": "http://127.0.0.1:18081/predict",
                "request_timeout_s": 1.0,
            }
        )
        self.assertEqual("configured", valid["status"])
        self.assertEqual("shared", valid["scope"])
        self.assertEqual("http://127.0.0.1:18081/health", valid["config"]["health_url"])

        start_args = {
            "action": "start",
            "instance_id": "stage5",
            "image_topic": "/stage5/camera/compressed",
            "state_topic": "/stage5/g1/joints",
            "output_topic": "/stage5/vla/action_proposal",
        }
        first_start = self.call_tool(start_args)
        replayed_config = self.call_tool(
            {
                "action": "config",
                "policy_url": "http://127.0.0.1:18081/predict",
                "health_url": "",
                "request_timeout_s": 1.0,
                "max_image_bytes": 8388608,
                "instance_id": "stage5",
            }
        )
        changed_config = self.call_tool({"action": "config", "request_timeout_s": 2.0})
        second_start = self.call_tool(start_args)
        self.assertEqual("running", first_start["state"])
        self.assertEqual("configured", replayed_config["status"])
        self.assertEqual("none", replayed_config["applied"])
        self.assertFalse(replayed_config["restart_required"])
        self.assertEqual("restart_required", changed_config["error_code"])
        self.assertEqual(first_start, second_start)
        self.assertEqual(1, len(self.executor.added))
        self.assertEqual(1, len(_FakeVLAPi05G1Node.created))

        info = self.call_tool({"action": "info", "instance_id": "stage5"})
        self.assertEqual(2, len(info["topic_in"]))
        self.assertEqual("/stage5/vla/action_proposal", info["topic_out"][0]["topic"])
        self.assertEqual("error", info["last_health"]["status"])

        first_stop = self.call_tool({"action": "stop", "instance_id": "stage5"})
        second_stop = self.call_tool({"action": "stop", "instance_id": "stage5"})
        self.assertEqual("idle", first_stop["state"])
        self.assertEqual(first_stop, second_stop)
        self.assertEqual(1, len(self.executor.removed))
        node = _FakeVLAPi05G1Node.created[0]
        self.assertEqual(1, node.shutdown_count)
        self.assertEqual(1, node.destroy_count)
        self.assertEqual({}, self.bundle._plugins[0]._instances)

    def test_dashboard_input_topics_start_maps_port_order(self):
        started = self.call_tool(
            {
                "action": "start",
                "instance_id": "canvas-card",
                "input_topics": [
                    "/canvas/camera/compressed",
                    "/canvas/g1/joints",
                ],
            }
        )
        self.assertEqual("running", started["state"])
        self.assertEqual(
            [
                {"topic": "/canvas/camera/compressed", "format": "image/jpeg"},
                {"topic": "/canvas/g1/joints", "format": "data/json"},
            ],
            started["topic_in"],
        )
        self.assertEqual(
            "/canvas/camera/compressed/vlapi05g1",
            started["topic_out"][0]["topic"],
        )
        invalid = self.call_tool(
            {
                "action": "start",
                "instance_id": "invalid-canvas-card",
                "input_topics": ["/only/one"],
            }
        )
        self.assertEqual(-32603, invalid["error"]["code"])

        stopped = self.call_tool({"action": "stop", "instance_id": "canvas-card"})
        self.assertEqual("idle", stopped["state"])

    def test_host_identity_is_explicit_distinct_and_not_default(self):
        good = self.main._load_host_identity(self.cfg)
        self.assertEqual(
            ("Perception Stack · Shanghai G1 Test", "perception-stack-shanghai-g1-test"),
            good,
        )
        for identity in (
            None,
            {},
            {"core_display_name": None, "mcp_server_name": "unique"},
            {"core_display_name": "same", "mcp_server_name": "same"},
            {"core_display_name": "Perception Stack", "mcp_server_name": "unique"},
            {"core_display_name": "unique", "mcp_server_name": "perception-bundle"},
        ):
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    self.main._load_host_identity({"identity": identity})


if __name__ == "__main__":
    unittest.main()
