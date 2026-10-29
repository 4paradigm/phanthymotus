"""Local Nav2Core reservation behavior after backend errors.

These fake-backend tests do not establish ROS delivery or physical stop safety.
"""

import sys
import threading
import unittest
from pathlib import Path

ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.controlled_semantic_spatial.planning.core import Nav2Core, NavigationBackendError  # noqa: E402


class RecordingBackend:
    def __init__(self, failure, *, terminal_first=False):
        self.failure = failure
        self.terminal_first = terminal_first
        self.callback = None
        self.calls = []

    def set_terminal_callback(self, callback):
        self.callback = callback

    def info(self):
        return {"state": "ready"}

    def execute(self, action, args, *, nav_id):
        self.calls.append((action, nav_id))
        if action == "navigate_to_pose":
            if self.terminal_first:
                self.callback({"nav_id": nav_id, "status": "arrived"})
            raise self.failure
        if action == "stop_nav":
            return {"status": "stopped", "terminal_confirmed": True}
        raise AssertionError(f"unexpected action: {action}")

    def stop(self):
        pass


class UnknownOutcomeSafetyTest(unittest.TestCase):
    @staticmethod
    def goal(trusted):
        result = {"action": "navigate_to_pose", "x": 1, "y": 2, "yaw": 0}
        if trusted:
            result["_control_nav_id"] = "trusted-goal"
        return result

    def test_unknown_and_generic_failures_retain_reservation_and_stop_can_see_it(self):
        for trusted in (False, True):
            for failure in (
                NavigationBackendError("response_timeout", "unknown", outcome_known=False),
                TimeoutError("unknown"),
            ):
                with self.subTest(trusted=trusted, failure=type(failure).__name__):
                    backend = RecordingBackend(failure)
                    core = Nav2Core(backend)
                    first = core.dispatch(self.goal(trusted))
                    self.assertEqual(first["status"], "error")
                    active = core.info()["active_nav_id"]
                    self.assertIsInstance(active, str)
                    self.assertTrue(active)
                    if trusted:
                        self.assertEqual(active, "trusted-goal")
                    other = self.goal(False)
                    other["x"] = 3
                    rejected = core.dispatch(other)
                    self.assertEqual(rejected["error_code"], "navigation_active")
                    self.assertEqual(backend.calls, [("navigate_to_pose", active)])
                    stopped = core.dispatch({"action": "stop_nav"})
                    self.assertEqual(stopped["status"], "stopped")
                    self.assertEqual(stopped["nav_id"], active)
                    self.assertEqual(backend.calls[-1], ("stop_nav", active))
                    self.assertIsNone(core.info()["active_nav_id"])

    def test_known_failure_releases_reservation_with_or_without_trusted_id(self):
        for trusted in (False, True):
            with self.subTest(trusted=trusted):
                backend = RecordingBackend(
                    NavigationBackendError("rejected", "known", outcome_known=True)
                )
                core = Nav2Core(backend)
                first = core.dispatch(self.goal(trusted))
                self.assertEqual(first["error_code"], "rejected")
                self.assertIsNone(core.info()["active_nav_id"])
                second = core.dispatch(self.goal(trusted))
                self.assertEqual(second["error_code"], "rejected")
                self.assertEqual(len(backend.calls), 2)

    def test_terminal_before_failure_releases_only_completed_goal(self):
        for trusted in (False, True):
            for failure in (
                NavigationBackendError("response_timeout", "unknown", outcome_known=False),
                NavigationBackendError("rejected", "known", outcome_known=True),
                TimeoutError("unknown"),
            ):
                with self.subTest(trusted=trusted, failure=type(failure).__name__,
                                  known=getattr(failure, "outcome_known", None)):
                    backend = RecordingBackend(failure, terminal_first=True)
                    core = Nav2Core(backend)
                    first = core.dispatch(self.goal(trusted))
                    self.assertEqual(first["status"], "error")
                    self.assertIsNone(core.info()["active_nav_id"])
                    second = core.dispatch(self.goal(False))
                    self.assertEqual(second["status"], "error")
                    self.assertEqual(len(backend.calls), 2)

    def test_terminal_racing_known_error_does_not_clear_other_active_goal(self):
        entered = threading.Event()
        release = threading.Event()

        class RacingBackend(RecordingBackend):
            def execute(self, action, args, *, nav_id):
                self.calls.append((action, nav_id))
                if nav_id == "trusted-goal":
                    self.callback({"nav_id": nav_id, "status": "arrived"})
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test did not release old request")
                    raise NavigationBackendError("rejected", "known", outcome_known=True)
                return {"status": "navigating"}

        backend = RacingBackend(None)
        core = Nav2Core(backend)
        results = {}
        thread = threading.Thread(target=lambda: results.update(first=core.dispatch(self.goal(True))))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            replay = core.dispatch(self.goal(True))
            self.assertEqual(replay["status"], "arrived")
            self.assertTrue(replay["idempotent_replay"])
            other = self.goal(True)
            other["_control_nav_id"] = "other-goal"
            self.assertEqual(core.dispatch(other)["status"], "navigating")
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results["first"]["error_code"], "rejected")
        self.assertEqual(core.info()["active_nav_id"], "other-goal")
        self.assertEqual(backend.calls, [("navigate_to_pose", "trusted-goal"),
                                         ("navigate_to_pose", "other-goal")])

    def test_concurrent_same_id_does_not_execute_backend_twice(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingBackend(RecordingBackend):
            def execute(self, action, args, *, nav_id):
                self.calls.append((action, nav_id))
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release first request")
                return {"status": "navigating"}

        backend = BlockingBackend(None)
        core = Nav2Core(backend)
        results = {}
        thread = threading.Thread(target=lambda: results.update(first=core.dispatch(self.goal(True))))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            replay = core.dispatch(self.goal(True))
            self.assertTrue(replay["idempotent_replay"])
            conflict = self.goal(True)
            conflict["x"] = 3
            self.assertEqual(core.dispatch(conflict)["error_code"], "control_nav_id_conflict")
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results["first"]["status"], "navigating")
        self.assertEqual(backend.calls, [("navigate_to_pose", "trusted-goal")])


if __name__ == "__main__":
    unittest.main()
