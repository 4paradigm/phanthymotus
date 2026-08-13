from __future__ import annotations

import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.fast_livo2.plugin import FastLivo2Plugin  # noqa: E402


class ReadyBackend:
    def __init__(self, subscribers: int = 1) -> None:
        self.subscribers = subscribers
        self.info_calls = 0
        self.stop_calls = 0
        self.calls: list[tuple[str, dict]] = []

    def info(self) -> dict:
        self.info_calls += 1
        return {
            "state": "ready",
            "status": "ready",
            "backend": "fast_livo2_ros_topic",
            "bridge_subscribers": self.subscribers,
        }

    def execute(self, action: str, args: dict) -> dict:
        self.calls.append((action, dict(args)))
        if action == "configure_collection":
            return {
                "status": "recording" if args["enabled"] else "disabled",
                "collection": {
                    "enabled": args["enabled"],
                    "state": "recording" if args["enabled"] else "disabled",
                },
            }
        if action == "start_mapping":
            return {"status": "mapping", "map_name": args["map_name"]}
        if action == "load_map":
            return {"status": "map_loaded", "map_name": args["map_name"]}
        if action == "unload_map":
            return {"status": "unloaded", "map_name": args["map_name"]}
        return {"status": "saved", "map_name": args["map_name"]}

    def stop(self) -> None:
        self.stop_calls += 1


class CollectionRejectingBackend(ReadyBackend):
    def execute(self, action: str, args: dict) -> dict:
        if action == "configure_collection" and args["enabled"]:
            self.calls.append((action, dict(args)))
            return {
                "status": "error",
                "error_code": "collection_start_failed",
                "error": "recording directory is not writable",
            }
        return super().execute(action, args)


class CollectionStopRejectingBackend(ReadyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.disable_calls = 0

    def execute(self, action: str, args: dict) -> dict:
        if action == "configure_collection" and not args["enabled"]:
            self.disable_calls += 1
            if self.disable_calls == 1:
                return super().execute(action, args)
            self.calls.append((action, dict(args)))
            return {
                "status": "error",
                "error_code": "collection_stop_failed",
                "error": "recorder is still running",
            }
        return super().execute(action, args)


def _bindings(plugin: FastLivo2Plugin) -> list[dict]:
    return [
        {"port": item["port"], "topic": item["topic"]}
        for item in plugin.get_tools()[0]["topic_in"]
    ]


class FastLivo2PluginTest(unittest.TestCase):
    def test_info_is_lazy_and_start_requires_exact_wiring(self) -> None:
        backend = ReadyBackend()
        plugin = FastLivo2Plugin({}, None, backend=backend)
        self.assertEqual(plugin.dispatch("fast_livo2", {"action": "info"})["state"], "idle")
        self.assertEqual(backend.info_calls, 0)

        missing = plugin.dispatch(
            "fast_livo2",
            {"action": "start", "input_bindings": _bindings(plugin)[:1]},
        )
        self.assertEqual(missing["error_code"], "invalid_canvas_wiring")
        self.assertEqual(backend.info_calls, 0)

        duplicate = plugin.dispatch(
            "fast_livo2",
            {
                "action": "start",
                "input_bindings": [
                    _bindings(plugin)[0],
                    _bindings(plugin)[0],
                    _bindings(plugin)[1],
                ],
            },
        )
        self.assertEqual(duplicate["error_code"], "invalid_canvas_wiring")

        ready = plugin.dispatch(
            "fast_livo2",
            {
                "action": "start",
                "instance_id": "mapping-card",
                "input_bindings": _bindings(plugin),
            },
        )
        self.assertEqual(ready["state"], "ready")
        self.assertTrue(ready["canvas_wired"])
        self.assertTrue(all(item["connected"] for item in ready["topic_in"]))
        self.assertEqual(backend.calls[0][0], "configure_collection")
        self.assertFalse(backend.calls[0][1]["enabled"])

    def test_canvas_stop_finalizes_mapping_before_backend_release(self) -> None:
        backend = ReadyBackend()
        plugin = FastLivo2Plugin({}, None, backend=backend)
        plugin.dispatch(
            "fast_livo2", {"action": "start", "input_bindings": _bindings(plugin)}
        )
        plugin.dispatch(
            "fast_livo2", {"action": "start_mapping", "map_name": "office"}
        )

        stopped = plugin.dispatch("fast_livo2", {"action": "stop"})

        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(stopped["stop_result"]["status"], "saved")
        self.assertEqual(
            [call[0] for call in backend.calls],
            [
                "configure_collection",
                "start_mapping",
                "stop_mapping",
                "configure_collection",
            ],
        )
        self.assertEqual(backend.stop_calls, 1)

    def test_collection_is_config_only_and_starts_with_canvas(self) -> None:
        backend = ReadyBackend()
        plugin = FastLivo2Plugin(
            {
                "collection_enabled": True,
                "collection_directory": (
                    "/opt/phanthy-motus/data/fast_livo2/recordings/acceptance"
                ),
            },
            None,
            backend=backend,
        )

        actions = plugin.get_tools()[0]["inputSchema"]["properties"]["action"]["enum"]
        self.assertNotIn("start_recording", actions)
        ready = plugin.dispatch(
            "fast_livo2",
            {"action": "start", "input_bindings": _bindings(plugin)},
        )

        self.assertEqual(ready["state"], "ready")
        self.assertEqual(backend.calls[0][0], "configure_collection")
        self.assertEqual(
            backend.calls[0][1],
            {
                "enabled": True,
                "directory": (
                    "/opt/phanthy-motus/data/fast_livo2/recordings/acceptance"
                ),
                "namespace": "ubuntu",
            },
        )

        stopped = plugin.dispatch("fast_livo2", {"action": "stop"})
        self.assertEqual(
            stopped["collection_stop_result"]["status"], "disabled"
        )
        self.assertFalse(backend.calls[-1][1]["enabled"])

    def test_collection_directory_is_confined_to_mounted_root(self) -> None:
        for directory in ("relative", "/tmp/recordings", "/opt/../tmp"):
            with self.subTest(directory=directory):
                plugin = FastLivo2Plugin(
                    {
                        "collection_enabled": True,
                        "collection_directory": directory,
                    },
                    None,
                    backend=ReadyBackend(),
                )
                info = plugin.dispatch("fast_livo2", {"action": "info"})
                self.assertEqual(info["error_code"], "invalid_config")

    def test_collection_start_failure_prevents_canvas_ready(self) -> None:
        backend = CollectionRejectingBackend()
        plugin = FastLivo2Plugin(
            {"collection_enabled": True}, None, backend=backend
        )

        result = plugin.dispatch(
            "fast_livo2",
            {"action": "start", "input_bindings": _bindings(plugin)},
        )

        self.assertEqual(result["error_code"], "collection_start_failed")
        self.assertEqual(backend.stop_calls, 1)
        info = plugin.dispatch("fast_livo2", {"action": "info"})
        self.assertFalse(info["canvas_wired"])

    def test_collection_stop_failure_keeps_card_retryable(self) -> None:
        backend = CollectionStopRejectingBackend()
        plugin = FastLivo2Plugin({}, None, backend=backend)
        plugin.dispatch(
            "fast_livo2", {"action": "start", "input_bindings": _bindings(plugin)}
        )

        result = plugin.dispatch("fast_livo2", {"action": "stop"})

        self.assertEqual(result["error_code"], "canvas_stop_failed")
        self.assertTrue(result["canvas_wired"])
        self.assertEqual(backend.stop_calls, 0)
        self.assertTrue(
            plugin.dispatch("fast_livo2", {"action": "info"})["canvas_wired"]
        )

    def test_missing_companion_and_invalid_config_fail_closed(self) -> None:
        unavailable = FastLivo2Plugin(
            {"discovery_timeout_sec": 0.5}, None, backend=ReadyBackend(0)
        )
        result = unavailable.dispatch(
            "fast_livo2",
            {"action": "start", "input_bindings": _bindings(unavailable)},
        )
        self.assertEqual(result["error_code"], "fast_livo2_companion_unavailable")

        invalid = FastLivo2Plugin({"map_max_points": 80_000}, None, backend=ReadyBackend())
        info = invalid.dispatch("fast_livo2", {"action": "info"})
        self.assertEqual(info["error_code"], "invalid_config")

    def test_canvas_stop_unloads_localization_before_backend_release(self) -> None:
        backend = ReadyBackend()
        plugin = FastLivo2Plugin({}, None, backend=backend)
        plugin.dispatch(
            "fast_livo2", {"action": "start", "input_bindings": _bindings(plugin)}
        )
        plugin.dispatch("fast_livo2", {"action": "load_map", "map_name": "office"})

        stopped = plugin.dispatch("fast_livo2", {"action": "stop"})

        self.assertEqual(stopped["stop_result"]["status"], "unloaded")
        self.assertEqual(
            [call[0] for call in backend.calls],
            [
                "configure_collection",
                "load_map",
                "unload_map",
                "configure_collection",
            ],
        )
        self.assertEqual(backend.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
