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
            storage_mode = tool["configSchema"]["properties"]["storage_mode"]
            self.assertEqual(["local_ring", "local_and_cos"], storage_mode["enum"])
            self.assertEqual("local_and_cos", storage_mode["default"])
            corrupt_retention = tool["configSchema"]["properties"]["corrupt_retention_hours"]
            self.assertEqual(24, corrupt_retention["default"])
            self.assertEqual("instance", corrupt_retention["scope"])

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

    def test_stop_cleanup_failure_is_not_reported_as_recording(self) -> None:
        self.bundle.dispatch("videoinspector", {
            "action": "config", "cos_bucket": "test-1250000000", "instance_id": "camera-stop-error",
        })
        self.bundle.dispatch("videoinspector", {
            "action": "start", "instance_id": "camera-stop-error", "input_topic": "/camera/front/image",
        })
        plugin = next(item for item in self.bundle._plugins if item.card_id == "videoinspector")

        def fail_stop(_instance, *, for_shutdown):
            self.assertFalse(for_shutdown)
            raise TimeoutError("writer queue did not drain")

        original_stats = plugin._runtime_stats
        plugin._runtime_stats = lambda instance, instance_id: {
            **original_stats(instance, instance_id),
            "last_error": "",
        }
        plugin._stop_runtime = fail_stop
        stopped = self.bundle.dispatch("videoinspector", {
            "action": "stop", "instance_id": "camera-stop-error",
        })

        self.assertEqual("stop_error", stopped["state"])
        self.assertFalse(stopped["recording"])
        self.assertTrue(stopped["resume_required"])
        self.assertIn("queue did not drain", stopped["last_error"])

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

    def test_local_ring_mode_does_not_require_cos(self) -> None:
        configured = self.bundle.dispatch("audioinspector", {
            "action": "config",
            "storage_mode": "local_ring",
            "instance_id": "card-local",
        })
        started = self.bundle.dispatch("audioinspector", {
            "action": "start",
            "instance_id": "card-local",
            "input_topic": "/robot/mic/audio",
        })

        self.assertTrue(configured["adapter_ok"])
        self.assertFalse(configured["upload_ready"])
        self.assertEqual("recording", started["state"])
        self.assertEqual("local_ring", started["storage_mode"])
        self.assertIn("upload_backlog_bytes", started)
        self.assertIn("disk_pressure", started)

    def test_storage_mode_rejects_legacy_switch_conflict(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.bundle.dispatch("audioinspector", {
                "action": "config",
                "storage_mode": "local_ring",
                "upload_enabled": True,
            })

    def test_config_is_validated_atomically(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown config field"):
            self.bundle.dispatch("audioinspector", {
                "action": "config",
                "instance_id": "card-3",
                "cos_bucket": "must-not-stick-1250000000",
                "unknown_field": True,
            })
        with self.assertRaisesRegex(ValueError, "cos_bucket"):
            self.bundle.dispatch("audioinspector", {
                "action": "start", "instance_id": "card-3", "input_topic": "/robot/mic/audio",
            })
        with self.assertRaisesRegex(ValueError, ">= 5"):
            self.bundle.dispatch("audioinspector", {
                "action": "config", "instance_id": "card-3", "segment_seconds": 1,
            })

    def test_recording_instance_rejects_effective_config_change(self) -> None:
        config = {
            "action": "config",
            "cos_bucket": "test-1250000000",
            "instance_id": "card-4",
            "segment_seconds": 10,
        }
        self.bundle.dispatch("audioinspector", config)
        self.bundle.dispatch("audioinspector", {
            "action": "start", "instance_id": "card-4", "input_topic": "/robot/mic/audio",
        })

        same = self.bundle.dispatch("audioinspector", config)
        self.assertEqual("configured", same["state"])
        with self.assertRaisesRegex(ValueError, "stop instance"):
            self.bundle.dispatch("audioinspector", {
                "action": "config", "instance_id": "card-4", "segment_seconds": 11,
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


class InspectionDeployContractTest(unittest.TestCase):
    def test_jetson_devices_and_durable_directories_are_mounted(self) -> None:
        service_path = INSPECTION_ROOT / "deploy" / "service.yml"
        service = service_path.read_text(encoding="utf-8")

        self.assertIn("privileged: true", service)
        self.assertIn("- /dev:/dev", service)
        self.assertIn(
            "/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro",
            service,
        )
        self.assertIn(
            "/usr/lib/aarch64-linux-gnu/tegra-egl:"
            "/usr/lib/aarch64-linux-gnu/tegra-egl:ro",
            service,
        )
        for plugin in ("libgstnvjpeg.so", "libgstnvvidconv.so", "libgstnvvideo4linux2.so"):
            self.assertIn(
                f"/usr/lib/aarch64-linux-gnu/gstreamer-1.0/{plugin}:"
                f"/usr/lib/aarch64-linux-gnu/gstreamer-1.0/{plugin}:ro",
                service,
            )
        self.assertIn(
            "/lib/aarch64-linux-gnu/libgstnvexifmeta.so:"
            "/lib/aarch64-linux-gnu/libgstnvexifmeta.so:ro",
            service,
        )
        self.assertIn("/opt/phanthy-motus/inspection-data:/opt/phanthy-motus/inspection-data", service)
        self.assertIn("/opt/phanthy-motus/inspection-state:/opt/phanthy-motus/inspection-state", service)
        self.assertIn(
            "/opt/phanthy-motus/secrets/phanthymotus:/run/secrets/phanthymotus:ro",
            service,
        )
        self.assertIn(
            "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:"
            "/usr/lib/aarch64-linux-gnu/tegra-egl",
            service,
        )
        self.assertIn('restart: "on-failure:3"', service)
        self.assertNotIn('restart: "unless-stopped"', service)
        self.assertNotIn('restart: "always"', service)
        self.assertNotIn(
            "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:"
            "/usr/lib/aarch64-linux-gnu/gstreamer-1.0:ro",
            service,
        )

    def test_image_installs_gstreamer_python_and_supports_jetson_ros_layout(self) -> None:
        dockerfile = (INSPECTION_ROOT / "Dockerfile").read_text(encoding="utf-8")
        requirements = (INSPECTION_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("python3-gst-1.0", dockerfile)
        self.assertIn("gir1.2-gstreamer-1.0", dockerfile)
        self.assertIn("gstreamer1.0-plugins-base-apps", dockerfile)
        self.assertIn("UBUNTU_PORTS_MIRROR=https://mirrors.ustc.edu.cn/ubuntu-ports", dockerfile)
        self.assertIn("mirrors.tencentyun.com/ubuntu-ports", dockerfile)
        self.assertIn("apt-get update -o Acquire::Retries=3", dockerfile)
        self.assertIn("-maxdepth 1 -type f -size 0 -delete", dockerfile)
        self.assertIn("-maxdepth 1 -type f -size 0 -print -quit", dockerfile)
        self.assertIn("--ignore-installed", dockerfile)
        self.assertIn("--target ${INSPECTION_PYTHON_DIR}", dockerfile)
        self.assertIn("ENV PYTHONPATH=/opt/inspection-python:${PYTHONPATH}", dockerfile)
        self.assertIn("isolated inspection Python dependencies PASS", dockerfile)
        self.assertIn("/opt/ros/humble/install/setup.bash", dockerfile)
        self.assertIn("/opt/ros/humble/setup.bash", dockerfile)
        self.assertIn("pyyaml==6.0.2", requirements)


if __name__ == "__main__":
    unittest.main()
