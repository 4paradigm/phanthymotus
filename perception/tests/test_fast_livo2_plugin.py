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
        if action == "start_mapping":
            return {"status": "mapping", "map_name": args["map_name"]}
        return {"status": "saved", "map_name": args["map_name"]}

    def stop(self) -> None:
        self.stop_calls += 1


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
        self.assertEqual([call[0] for call in backend.calls], ["start_mapping", "stop_mapping"])
        self.assertEqual(backend.stop_calls, 1)

    def test_missing_companion_and_invalid_config_fail_closed(self) -> None:
        unavailable = FastLivo2Plugin(
            {"discovery_timeout_sec": 0.5}, None, backend=ReadyBackend(0)
        )
        result = unavailable.dispatch(
            "fast_livo2",
            {"action": "start", "input_bindings": _bindings(unavailable)},
        )
        self.assertEqual(result["error_code"], "fast_livo2_companion_unavailable")

        invalid = FastLivo2Plugin({"map_max_points": 100}, None, backend=ReadyBackend())
        info = invalid.dispatch("fast_livo2", {"action": "info"})
        self.assertEqual(info["error_code"], "invalid_config")


if __name__ == "__main__":
    unittest.main()
