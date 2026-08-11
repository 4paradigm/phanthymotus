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
        if action == "navigate_to_pose":
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
        self.assertEqual(args["speed"], 0.50)
        self.assertEqual(args["mode"], 0)

        terminal = self.core.dispatch(
            {"action": "wait_navigation_done", "stall_timeout": 2}
        )
        self.assertEqual(terminal["status"], "arrived")
        self.assertIsNone(self.core.info()["active_nav_id"])

    def test_minimum_navigation_speed_is_accepted(self) -> None:
        result = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 0.5,
                "y": 0.0,
                "yaw": 0.0,
                "speed": 0.30,
                "_control_nav_id": "lease-min-speed",
            }
        )

        self.assertEqual(result["status"], "navigating")
        action, args, nav_id = self.backend.calls[-1]
        self.assertEqual(action, "navigate_to_pose")
        self.assertEqual(args["speed"], 0.30)
        self.assertEqual(nav_id, "lease-min-speed")

    def test_maximum_navigation_speed_is_accepted(self) -> None:
        result = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 2.0,
                "y": 0.0,
                "yaw": 0.0,
                "speed": 1.0,
                "_control_nav_id": "lease-max-speed",
            }
        )

        self.assertEqual(result["status"], "navigating")
        action, args, nav_id = self.backend.calls[-1]
        self.assertEqual(action, "navigate_to_pose")
        self.assertEqual(args["speed"], 1.0)
        self.assertEqual(nav_id, "lease-max-speed")

    def test_speed_mode_and_non_finite_values_are_rejected(self) -> None:
        cases = (
            ({"speed": 1.001}, "invalid_argument"),
            ({"speed": 0.299}, "invalid_argument"),
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

    def test_removed_mapping_actions_are_rejected(self) -> None:
        result = self.core.dispatch(
            {"action": "start_mapping", "map_name": "../escape"}
        )
        self.assertEqual(result["error_code"], "unsupported_action")
        self.assertEqual(self.backend.calls, [])

    def test_stop_without_navigation_is_idempotent(self) -> None:
        result = self.core.dispatch({"action": "stop_nav"})
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["already_idle"])


if __name__ == "__main__":
    unittest.main()
