from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.navigation.planning.core import (  # noqa: E402
    Nav2Core,
    NavigationBackendError,
)
from plugins.navigation.planning.backend import (  # noqa: E402
    RosTopicNavigationBackend,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.stopped = 0
        self.terminal_callback = None

    def set_terminal_callback(self, callback) -> None:
        self.terminal_callback = callback

    def emit_terminal(self, nav_id: str, status: str = "arrived") -> None:
        assert self.terminal_callback is not None
        self.terminal_callback({"nav_id": nav_id, "status": status})

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
        self.completions = []
        self.core = Nav2Core(
            self.backend, completion_callback=self.completions.append
        )

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
        self.assertEqual(result["action_id"], "lease-001")
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
        self.assertEqual(self.completions[-1]["status"], "arrived")

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

    def test_async_arrival_releases_task_and_preserves_wait_receipt(self) -> None:
        first = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 0.5,
                "y": 0.0,
                "yaw": 0.0,
                "_control_nav_id": "lease-first",
            }
        )
        self.assertEqual(first["status"], "navigating")

        self.backend.emit_terminal("lease-first")

        self.assertIsNone(self.core.info()["active_nav_id"])
        self.assertEqual(self.completions[-1]["action_id"], "lease-first")
        terminal = self.core.dispatch(
            {"action": "wait_navigation_done", "stall_timeout": 2}
        )
        self.assertEqual(terminal["status"], "arrived")
        self.assertEqual(terminal["nav_id"], "lease-first")
        self.assertTrue(terminal["terminal_replayed"])

        second = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
                "_control_nav_id": "lease-second",
            }
        )
        self.assertEqual(second["status"], "navigating")
        self.assertEqual(second["nav_id"], "lease-second")

    def test_trusted_goal_id_replays_active_and_terminal_result(self) -> None:
        request = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-idempotent",
        }

        first = self.core.dispatch(request)
        active_replay = self.core.dispatch(request)
        self.backend.emit_terminal("goal-idempotent")
        terminal_replay = self.core.dispatch(request)

        self.assertEqual(first["status"], "navigating")
        self.assertTrue(active_replay["idempotent_replay"])
        self.assertEqual(terminal_replay["status"], "arrived")
        self.assertEqual(terminal_replay["action_id"], "goal-idempotent")
        self.assertTrue(terminal_replay["idempotent_replay"])
        navigate_calls = [call for call in self.backend.calls if call[0] == "navigate_to_pose"]
        self.assertEqual(len(navigate_calls), 1)

    def test_outcome_unknown_keeps_trusted_goal_active_and_at_most_once(self) -> None:
        request = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-timeout",
        }
        def timeout_once(action, args, *, nav_id):
            self.backend.calls.append((action, dict(args), nav_id))
            raise TimeoutError("backend outcome unknown")

        with mock.patch.object(self.backend, "execute", side_effect=timeout_once):
            failed = self.core.dispatch(request)
            replay = self.core.dispatch(request)

        self.assertEqual(failed["error_code"], "backend_error")
        self.assertEqual(replay["status"], "navigating")
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_backend_close_after_dispatch_is_also_at_most_once(self) -> None:
        request = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-closed",
        }

        def closed(action, args, *, nav_id):
            self.backend.calls.append((action, dict(args), nav_id))
            raise NavigationBackendError("backend_closed", "closed while waiting")

        with mock.patch.object(self.backend, "execute", side_effect=closed):
            failed = self.core.dispatch(request)
            replay = self.core.dispatch(request)

        self.assertEqual(failed["error_code"], "backend_closed")
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_known_backend_rejection_releases_trusted_goal_for_retry(self) -> None:
        request = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-rejected",
        }

        def rejected(action, args, *, nav_id):
            self.backend.calls.append((action, dict(args), nav_id))
            raise NavigationBackendError(
                "goal_in_collision", "blocked", outcome_known=True
            )

        with mock.patch.object(self.backend, "execute", side_effect=rejected):
            first = self.core.dispatch(request)
            second = self.core.dispatch(request)

        self.assertEqual(first["error_code"], "goal_in_collision")
        self.assertEqual(second["error_code"], "goal_in_collision")
        self.assertEqual(len(self.backend.calls), 2)
        self.assertIsNone(self.core.info()["active_nav_id"])

    def test_known_rejection_releases_id_after_terminal_callback_race(self) -> None:
        request = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-raced-rejection",
        }

        def rejected(action, args, *, nav_id):
            self.backend.calls.append((action, dict(args), nav_id))
            self.backend.emit_terminal(nav_id, "rejected")
            raise NavigationBackendError(
                "goal_in_collision", "blocked", outcome_known=True
            )

        with mock.patch.object(self.backend, "execute", side_effect=rejected):
            first = self.core.dispatch(request)
            second = self.core.dispatch(request)

        self.assertEqual(first["error_code"], "goal_in_collision")
        self.assertEqual(second["error_code"], "goal_in_collision")
        self.assertEqual(len(self.backend.calls), 2)

    def test_trusted_goal_id_replays_an_older_terminal_receipt(self) -> None:
        first = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-first",
        }
        second = dict(first, x=1.0, _control_nav_id="goal-second")

        self.core.dispatch(first)
        self.backend.emit_terminal("goal-first")
        self.core.dispatch(second)
        self.backend.emit_terminal("goal-second")
        replay = self.core.dispatch(first)

        self.assertEqual(replay["status"], "arrived")
        self.assertTrue(replay["idempotent_replay"])
        navigate_calls = [call for call in self.backend.calls if call[0] == "navigate_to_pose"]
        self.assertEqual(len(navigate_calls), 2)

    def test_trusted_goal_id_cannot_be_reused_for_another_goal(self) -> None:
        first = {
            "action": "navigate_to_pose",
            "x": 0.5,
            "y": 0.0,
            "yaw": 0.0,
            "_control_nav_id": "goal-conflict",
        }
        self.core.dispatch(first)
        self.backend.emit_terminal("goal-conflict")

        conflict = self.core.dispatch(dict(first, x=2.0))

        self.assertEqual(conflict["error_code"], "control_nav_id_conflict")
        navigate_calls = [call for call in self.backend.calls if call[0] == "navigate_to_pose"]
        self.assertEqual(len(navigate_calls), 1)

    def test_consecutive_manual_goals_generate_distinct_task_ids(self) -> None:
        first = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 0.5,
                "y": 0.0,
                "yaw": 0.0,
            }
        )
        self.assertEqual(first["status"], "navigating")
        self.assertRegex(first["nav_id"], r"^[0-9a-f]{32}$")

        self.backend.emit_terminal(first["nav_id"])

        second = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
            }
        )
        self.assertEqual(second["status"], "navigating")
        self.assertRegex(second["nav_id"], r"^[0-9a-f]{32}$")
        self.assertNotEqual(first["nav_id"], second["nav_id"])
        navigate_calls = [
            call for call in self.backend.calls if call[0] == "navigate_to_pose"
        ]
        self.assertEqual(
            [call[2] for call in navigate_calls],
            [first["nav_id"], second["nav_id"]],
        )

    def test_terminal_for_another_navigation_does_not_release_task(self) -> None:
        self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 0.5,
                "y": 0.0,
                "yaw": 0.0,
                "_control_nav_id": "lease-active",
            }
        )

        self.backend.emit_terminal("lease-other")

        rejected = self.core.dispatch(
            {
                "action": "navigate_to_pose",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
            }
        )
        self.assertEqual(rejected["error_code"], "navigation_active")
        self.assertEqual(self.core.info()["active_nav_id"], "lease-active")

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


class RosTopicNavigationBackendTest(unittest.TestCase):
    def test_runtime_error_preserves_outcome_certainty(self) -> None:
        backend = object.__new__(RosTopicNavigationBackend)
        backend._request = mock.Mock(
            side_effect=(
                {
                    "status": "error",
                    "error_code": "goal_in_collision",
                    "error": "blocked",
                    "outcome_known": True,
                },
                {
                    "status": "error",
                    "error_code": "goal_response_timeout",
                    "error": "late response",
                    "outcome_known": False,
                },
            )
        )

        with self.assertRaises(NavigationBackendError) as known:
            backend.execute("navigate_to_pose", {}, nav_id="known")
        with self.assertRaises(NavigationBackendError) as unknown:
            backend.execute("navigate_to_pose", {}, nav_id="unknown")

        self.assertTrue(known.exception.outcome_known)
        self.assertFalse(unknown.exception.outcome_known)

    def test_terminal_status_notifies_registered_callback(self) -> None:
        backend = object.__new__(RosTopicNavigationBackend)
        backend._condition = threading.Condition()
        backend._last_status = {}
        backend._responses = {}
        backend._navigation = {}
        received = []
        backend._terminal_callback = received.append

        backend._on_status(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "event": "heartbeat",
                        "nav_id": "lease-001",
                        "status": "arrived",
                        "progress_seq": 5,
                    }
                )
            )
        )

        self.assertEqual(received[0]["nav_id"], "lease-001")
        self.assertEqual(received[0]["status"], "arrived")

    def test_pending_goal_response_does_not_emit_a_terminal_callback(self) -> None:
        backend = object.__new__(RosTopicNavigationBackend)
        backend._condition = threading.Condition()
        backend._last_status = {}
        backend._responses = {}
        backend._navigation = {}
        received = []
        backend._terminal_callback = received.append

        backend._on_status(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "event": "navigation_status",
                        "nav_id": "lease-timeout",
                        "status": "starting",
                        "error_code": "goal_response_timeout",
                        "goal_response_pending": True,
                    }
                )
            )
        )

        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
