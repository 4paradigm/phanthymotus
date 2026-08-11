from __future__ import annotations

import sys
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.nav2.core import Nav2Core  # noqa: E402


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.stopped = 0

    def info(self) -> dict:
        return {"state": "ready", "backend": "fake"}

    def execute(self, action: str, args: dict, *, nav_id: str | None) -> dict:
        self.calls.append((action, dict(args), nav_id))
        if action == "wait_navigation_done":
            return {"status": "arrived"}
        if action in {"navigate_to_pose", "navigate_to_tag"}:
            return {"status": "navigating"}
        if action == "stop_nav":
            return {"status": "stopped", "terminal_confirmed": True}
        return {"status": "ok"}

    def stop(self) -> None:
        self.stopped += 1


class Nav2CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeBackend()
        self.core = Nav2Core(self.backend)

    def test_navigation_uses_trusted_id_and_safe_default_speed(self) -> None:
        result = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 1,
                "y": -2,
                "yaw": 0,
                "_control_nav_id": "lease-001",
            }
        )

        self.assertEqual(result["status"], "navigating")
        self.assertEqual(result["nav_id"], "lease-001")
        action, args, nav_id = self.backend.calls[-1]
        self.assertEqual(action, "navigate_to_pose")
        self.assertEqual(nav_id, "lease-001")
        self.assertEqual(args["speed"], 0.15)
        self.assertEqual(args["mode"], 0)

        terminal = self.core.dispatch(
            {"action": "wait_navigation_done", "stall_timeout": 2}
        )
        self.assertEqual(terminal["status"], "arrived")
        self.assertIsNone(self.core.info()["active_nav_id"])

    def test_speed_mode_and_non_finite_values_are_rejected(self) -> None:
        cases = (
            ({"speed": 0.151}, "invalid_argument"),
            ({"speed": 0.049}, "invalid_argument"),
            ({"mode": "0"}, "invalid_argument"),
            ({"x": float("nan")}, "invalid_argument"),
        )
        for override, code in cases:
            with self.subTest(override=override):
                request = {
                    "action": "navigate_to_pose",
                    "x": 0,
                    "y": 0,
                    "yaw": 0,
                    **override,
                }
                result = self.core.dispatch(request)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_code"], code)
        self.assertEqual(self.backend.calls, [])

    def test_map_names_are_path_safe(self) -> None:
        result = self.core.dispatch(
            {"action": "start_mapping", "map_name": "../escape"}
        )
        self.assertEqual(result["error_code"], "invalid_argument")
        self.assertEqual(self.backend.calls, [])

    def test_runtime_mode_switch_is_explicit_and_localization_requires_map(self) -> None:
        mapping = self.core.dispatch(
            {"action": "switch_runtime_mode", "runtime_mode": "mapping"}
        )
        self.assertEqual(mapping["status"], "ok")
        self.assertEqual(
            self.backend.calls[-1],
            (
                "switch_runtime_mode",
                {"runtime_mode": "mapping", "map_name": ""},
                None,
            ),
        )

        missing_map = self.core.dispatch(
            {"action": "switch_runtime_mode", "runtime_mode": "localization"}
        )
        self.assertEqual(missing_map["error_code"], "missing_argument")

        localization = self.core.dispatch(
            {
                "action": "switch_runtime_mode",
                "runtime_mode": "localization",
                "map_name": "room-a",
            }
        )
        self.assertEqual(localization["status"], "ok")
        self.assertEqual(
            self.backend.calls[-1],
            (
                "switch_runtime_mode",
                {"runtime_mode": "localization", "map_name": "room-a"},
                None,
            ),
        )

        invalid = self.core.dispatch(
            {"action": "switch_runtime_mode", "runtime_mode": "invalid"}
        )
        self.assertEqual(invalid["error_code"], "invalid_argument")

    def test_map_mutation_is_blocked_during_navigation(self) -> None:
        started = self.core.dispatch(
            {
                "action": "navigate_to_tag",
                "tag_name": "door",
                "_control_nav_id": "lease-002",
            }
        )
        self.assertEqual(started["status"], "navigating")

        blocked = self.core.dispatch(
            {"action": "delete_map", "map_name": "room-a"}
        )
        self.assertEqual(blocked["error_code"], "navigation_active")
        blocked_switch = self.core.dispatch(
            {"action": "switch_runtime_mode", "runtime_mode": "mapping"}
        )
        self.assertEqual(blocked_switch["error_code"], "navigation_active")
        stopped = self.core.dispatch({"action": "stop_nav"})
        self.assertEqual(stopped["status"], "stopped")
        self.assertIsNone(self.core.info()["active_nav_id"])

    def test_stop_without_navigation_is_idempotent(self) -> None:
        result = self.core.dispatch({"action": "stop_nav"})
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["already_idle"])


if __name__ == "__main__":
    unittest.main()
