from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path


INSPECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INSPECTION_ROOT))

from main import InspectionBundle, ThreadingHTTPServer, make_handler  # noqa: E402


class InspectionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = InspectionBundle({"plugins": {}})

    def test_two_inspector_cards_are_declared(self) -> None:
        tools = {tool["name"]: tool for tool in self.bundle.get_all_tools()}
        self.assertEqual({"audioinspector", "videoinspector"}, set(tools))
        self.assertEqual("audio/pcm-16k", tools["audioinspector"]["topic_in"][0]["format"])
        self.assertEqual("image/jpeg", tools["videoinspector"]["topic_in"][0]["format"])
        for tool in tools.values():
            self.assertEqual("inspector", tool["type"])
            self.assertTrue(tool["multiInstance"])
            self.assertFalse(tool["agentCallable"])
            self.assertEqual([], tool["topic_out"])

    def test_config_start_info_stop_is_idempotent(self) -> None:
        configured = self.bundle.dispatch("audioinspector", {
            "action": "config",
            "cos_bucket": "test-1250000000",
            "instance_id": "card-1",
            "segment_seconds": 10,
        })
        self.assertEqual("configured", configured["state"])

        first = self.bundle.dispatch("audioinspector", {
            "action": "start", "instance_id": "card-1", "input_topic": "/robot/mic/audio",
        })
        second = self.bundle.dispatch("audioinspector", {
            "action": "start", "instance_id": "card-1", "input_topic": "/robot/mic/audio",
        })
        self.assertEqual("recording", first["state"])
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertFalse(first["storage_ready"])

        stopped = self.bundle.dispatch("audioinspector", {"action": "stop", "instance_id": "card-1"})
        stopped_again = self.bundle.dispatch("audioinspector", {"action": "stop", "instance_id": "card-1"})
        self.assertEqual("idle", stopped["state"])
        self.assertEqual("idle", stopped_again["state"])

    def test_running_instance_rejects_topic_change(self) -> None:
        self.bundle.dispatch("videoinspector", {
            "action": "config", "cos_bucket": "test-1250000000", "instance_id": "camera-1",
        })
        self.bundle.dispatch("videoinspector", {
            "action": "start", "instance_id": "camera-1", "input_topic": "/camera/front/image",
        })
        with self.assertRaisesRegex(ValueError, "already records"):
            self.bundle.dispatch("videoinspector", {
                "action": "start", "instance_id": "camera-1", "input_topic": "/camera/rear/image",
            })

    def test_start_requires_cos_bucket_when_upload_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "cos_bucket"):
            self.bundle.dispatch("audioinspector", {
                "action": "start", "instance_id": "card-2", "input_topic": "/robot/mic/audio",
            })


class InspectionHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = InspectionBundle({"plugins": {}})
        self.server: HTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.bundle, "inspection-test"))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_and_tools_list(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as response:
            health = json.load(response)
        self.assertTrue(health["ok"])
        request = urllib.request.Request(
            f"{self.base_url}/mcp",
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(2, len(payload["result"]["tools"]))


if __name__ == "__main__":
    unittest.main()
