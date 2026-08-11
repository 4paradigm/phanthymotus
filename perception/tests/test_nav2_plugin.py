from __future__ import annotations

import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.nav2.plugin import Nav2Plugin  # noqa: E402


class ReadyBackend:
    def __init__(self, *, subscribers: int = 1, ready: bool = True) -> None:
        self.subscribers = subscribers
        self.ready = ready
        self.info_calls = 0
        self.stop_calls = 0
        self.execute_calls: list[tuple[str, dict, str | None]] = []

    def info(self) -> dict:
        self.info_calls += 1
        return {
            "state": "ready",
            "backend": "nav2_ros_topic",
            "bridge_subscribers": self.subscribers,
            "navigation_ready": self.ready,
            "readiness_blockers": [] if self.ready else ["registered_cloud_stale"],
        }

    def execute(self, action: str, args: dict, *, nav_id: str | None) -> dict:
        self.execute_calls.append((action, dict(args), nav_id))
        if action == "navigate_to_pose":
            return {"status": "navigating"}
        if action == "stop_nav":
            return {"status": "stopped", "terminal_confirmed": True}
        return {"status": "ok"}

    def stop(self) -> None:
        self.stop_calls += 1


class DelayedReadyBackend(ReadyBackend):
    def info(self) -> dict:
        self.info_calls += 1
        discovered = self.info_calls >= 2
        return {
            "state": "ready",
            "backend": "nav2_ros_topic",
            "bridge_subscribers": 1 if discovered else 0,
            "navigation_ready": False,
            "readiness_blockers": ["fast_livo2_not_started"],
        }


def _bindings(plugin: Nav2Plugin) -> list[dict]:
    tool = plugin.get_tools()[0]
    return [
        {"port": item["port"], "topic": item["topic"]}
        for item in tool["topic_in"]
        if item.get("required", True)
    ]


class Nav2PluginLifecycleTest(unittest.TestCase):
    def test_constructor_and_info_do_not_acquire_backend(self) -> None:
        backend = ReadyBackend()
        plugin = Nav2Plugin({"enabled": True}, None, backend=backend)

        info = plugin.dispatch("nav2", {"action": "info"})
        self.assertEqual(info["state"], "idle")
        self.assertEqual(info["backend"], "not_started")
        self.assertEqual(backend.info_calls, 0)
        self.assertEqual(backend.stop_calls, 0)

    def test_start_requires_exact_sensor_wiring_and_is_idempotent(self) -> None:
        backend = ReadyBackend()
        plugin = Nav2Plugin({}, None, backend=backend)

        missing = plugin.dispatch(
            "nav2",
            {
                "action": "start",
                "input_bindings": _bindings(plugin)[:1],
            },
        )
        self.assertEqual(missing["error_code"], "invalid_canvas_wiring")
        self.assertEqual(backend.info_calls, 0)

        unexpected = plugin.dispatch(
            "nav2",
            {
                "action": "start",
                "input_topics": [
                    *(binding["topic"] for binding in _bindings(plugin)),
                    "/ubuntu/unrelated",
                ],
            },
        )
        self.assertEqual(unexpected["error_code"], "invalid_canvas_wiring")
        self.assertEqual(backend.info_calls, 0)

        ready = plugin.dispatch(
            "nav2",
            {
                "action": "start",
                "instance_id": "canvas-1",
                "input_bindings": _bindings(plugin),
            },
        )
        self.assertEqual(ready["state"], "ready")
        self.assertTrue(ready["canvas_wired"])
        self.assertEqual(ready["instance_id"], "canvas-1")

        again = plugin.dispatch(
            "nav2", {"action": "start", "input_bindings": _bindings(plugin)}
        )
        self.assertTrue(again["already_started"])
        self.assertEqual(backend.stop_calls, 0)

        config = plugin.dispatch("nav2", {"action": "config", "request_timeout_sec": 20})
        self.assertEqual(config["error_code"], "config_while_running")

        stopped = plugin.dispatch("nav2", {"action": "stop"})
        self.assertEqual(stopped["state"], "idle")
        self.assertEqual(backend.stop_calls, 1)
        stopped_again = plugin.dispatch("nav2", {"action": "stop"})
        self.assertEqual(stopped_again["state"], "idle")
        self.assertEqual(backend.stop_calls, 1)

    def test_invalid_configuration_and_missing_companion_fail_closed(self) -> None:
        invalid = Nav2Plugin({"shadow_only": False}, None, backend=ReadyBackend())
        info = invalid.dispatch("nav2", {"action": "info"})
        self.assertEqual(info["error_code"], "invalid_config")
        start = invalid.dispatch(
            "nav2", {"action": "start", "input_bindings": _bindings(invalid)}
        )
        self.assertEqual(start["error_code"], "invalid_config")

        unavailable_backend = ReadyBackend(subscribers=0)
        unavailable = Nav2Plugin(
            {"discovery_timeout_sec": 0.5}, None, backend=unavailable_backend
        )
        result = unavailable.dispatch(
            "nav2",
            {"action": "start", "input_bindings": _bindings(unavailable)},
        )
        self.assertEqual(result["error_code"], "nav2_companion_unavailable")
        self.assertEqual(unavailable_backend.stop_calls, 1)

    def test_legacy_runtime_switch_timeout_is_ignored_but_unknown_fields_fail(self) -> None:
        plugin = Nav2Plugin(
            {"runtime_switch_timeout_sec": 45.0}, None, backend=ReadyBackend()
        )
        self.assertEqual(plugin.dispatch("nav2", {"action": "info"})["state"], "idle")

        migrated = plugin.dispatch(
            "nav2",
            {"action": "config", "runtime_switch_timeout_sec": 60.0},
        )
        self.assertEqual(migrated["state"], "configured")
        self.assertNotIn("runtime_switch_timeout_sec", migrated["config"])

        invalid = plugin.dispatch(
            "nav2", {"action": "config", "unexpected_field": True}
        )
        self.assertEqual(invalid["error_code"], "invalid_config")
        self.assertIn("unexpected_field", invalid["error"])

    def test_start_waits_for_dds_discovery_without_requiring_sensor_data(self) -> None:
        backend = DelayedReadyBackend()
        plugin = Nav2Plugin(
            {"discovery_timeout_sec": 0.5}, None, backend=backend
        )

        result = plugin.dispatch(
            "nav2", {"action": "start", "input_bindings": _bindings(plugin)}
        )

        self.assertEqual(result["state"], "ready")
        self.assertGreaterEqual(backend.info_calls, 2)
        self.assertFalse(result["navigation_ready"])
        self.assertEqual(result["readiness_blockers"], ["fast_livo2_not_started"])

    def test_config_validation_and_active_stop(self) -> None:
        backend = ReadyBackend()
        plugin = Nav2Plugin({}, None, backend=backend)
        bad = plugin.dispatch("nav2", {"action": "config", "proposal_ttl_ms": 100})
        self.assertEqual(bad["error_code"], "invalid_config")
        lateral = plugin.dispatch(
            "nav2", {"action": "config", "max_lateral_mps": 0.01}
        )
        self.assertEqual(lateral["error_code"], "invalid_config")
        configured = plugin.dispatch(
            "nav2", {"action": "config", "request_timeout_sec": 20}
        )
        self.assertEqual(configured["state"], "configured")

        plugin.dispatch(
            "nav2", {"action": "start", "input_bindings": _bindings(plugin)}
        )
        moving = plugin.dispatch(
            "nav2",
            {
                "action": "navigate_to_pose",
                "x": 0.5,
                "y": 0,
                "yaw": 0,
                "_control_nav_id": "lease-003",
            },
        )
        self.assertEqual(moving["status"], "navigating")
        plugin.stop()
        self.assertEqual(backend.execute_calls[-1][0], "stop_nav")
        self.assertEqual(backend.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
