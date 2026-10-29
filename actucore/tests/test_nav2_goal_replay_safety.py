"""Trusted goal replay classification; fake backends do not prove ROS or physical safety."""

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.navigation.planning.core import Nav2Core, NavigationBackendError  # noqa: E402


def goal(nav_id="goal-1", x=1):
    return {"action": "navigate_to_pose", "x": x, "y": 2, "yaw": 0,
            "_control_nav_id": nav_id}


class Backend:
    def __init__(self, start):
        self.start = start
        self.calls = []
        self.callback = None

    def set_terminal_callback(self, callback):
        self.callback = callback

    def info(self):
        return {"state": "ready"}

    def execute(self, action, args, *, nav_id):
        self.calls.append((action, nav_id))
        if action == "navigate_to_pose":
            return self.start(nav_id)
        if action == "stop_nav":
            return {"status": "stopped", "terminal_confirmed": True}
        raise AssertionError(action)

    def stop(self):
        pass


class GoalReplaySafetyTest(unittest.TestCase):
    def assert_unknown(self, replay, code):
        self.assertEqual(replay["status"], "error")
        self.assertEqual(replay["error_code"], code)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["nav_id"], "goal-1")
        self.assertNotEqual(replay["status"], "navigating")

    def test_pending_same_id_then_success_and_stop(self):
        entered, release = threading.Event(), threading.Event()

        def start(_nav_id):
            entered.set()
            if not release.wait(5):
                raise AssertionError("start not released")
            return {"status": "navigating"}

        backend = Backend(start)
        core = Nav2Core(backend)
        first = []
        thread = threading.Thread(target=lambda: first.append(core.dispatch(goal())))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assert_unknown(core.dispatch(goal()), "navigation_start_pending")
            self.assertEqual(core.dispatch(goal(x=3))["error_code"], "control_nav_id_conflict")
            self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1")])
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(first[0]["status"], "navigating")
        replay = core.dispatch(goal())
        self.assertEqual(replay["status"], "navigating")
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")
        self.assertEqual(core.dispatch(goal())["status"], "stopped")
        self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1"), ("stop_nav", "goal-1")])

    def test_unknown_failure_replay_and_stop(self):
        failures = (TimeoutError("timeout"),
                    NavigationBackendError("timeout", "unknown", outcome_known=False))
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                def start(_nav_id):
                    raise failure
                backend = Backend(start)
                core = Nav2Core(backend)
                self.assertEqual(core.dispatch(goal())["status"], "error")
                self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
                self.assertEqual(core.dispatch(goal("other"))["error_code"], "navigation_active")
                self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")
                self.assertEqual(core.dispatch(goal())["status"], "stopped")
                self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1"), ("stop_nav", "goal-1")])

    def test_unexpected_nonterminal_result_never_implies_acceptance(self):
        for raw in ({}, {"status": "pending"}, {"status": "unknown"}, {"status": "ok"}):
            with self.subTest(raw=raw):
                backend = Backend(lambda _id: raw)
                core = Nav2Core(backend)
                core.dispatch(goal())
                self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
                self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1")])

    def test_inconsistent_navigating_response_cannot_confirm_acceptance(self):
        for raw in ({"status": "navigating", "nav_id": "wrong"},
                    {"status": "navigating", "action_id": "wrong"},
                    {"status": "navigating", "ok": False},
                    {"status": "navigating", "error": "uncertain"}):
            with self.subTest(raw=raw):
                backend = Backend(lambda _id: raw)
                core = Nav2Core(backend)
                self.assertEqual(core.dispatch(goal())["status"], "error")
                self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
                self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1")])

    def test_known_rejection_retains_retry_compatibility(self):
        def start(_id):
            raise NavigationBackendError("rejected", "refused", outcome_known=True)
        backend = Backend(start)
        core = Nav2Core(backend)
        self.assertEqual(core.dispatch(goal())["error_code"], "rejected")
        self.assertIsNone(core.info()["active_nav_id"])
        self.assertEqual(core.dispatch(goal())["error_code"], "rejected")
        self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1")] * 2)

    def test_terminal_first_then_late_return_does_not_resurrect_goal(self):
        for late in ("success", "timeout", "known_error"):
            with self.subTest(late=late):
                entered, release = threading.Event(), threading.Event()
                backend = None
                def start(nav_id):
                    if nav_id == "goal-2":
                        return {"status": "navigating"}
                    backend.callback({"nav_id": nav_id, "status": "arrived"})
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("start not released")
                    if late == "timeout":
                        raise TimeoutError("late")
                    if late == "known_error":
                        raise NavigationBackendError("rejected", "late", outcome_known=True)
                    return {"status": "navigating"}
                backend = Backend(start)
                core = Nav2Core(backend)
                first = []
                thread = threading.Thread(target=lambda: first.append(core.dispatch(goal())))
                thread.start()
                try:
                    self.assertTrue(entered.wait(3))
                    self.assertEqual(core.dispatch(goal())["status"], "arrived")
                    self.assertEqual(core.dispatch(goal("goal-2"))["status"], "navigating")
                    self.assertEqual(core.dispatch(goal())["status"], "arrived")
                finally:
                    release.set()
                    thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(core.info()["active_nav_id"], "goal-2")
                replay = core.dispatch(goal())
                self.assertEqual(replay["status"], "arrived")
                self.assertTrue(replay["idempotent_replay"])
                self.assertEqual(core.dispatch({"action": "stop_nav"})["nav_id"], "goal-2")
                self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1"),
                                                 ("navigate_to_pose", "goal-2"),
                                                 ("stop_nav", "goal-2")])

    def test_generation_token_defends_reused_id_after_cache_retirement(self):
        for late in ("success", "timeout", "known_error", "terminal"):
            with self.subTest(late=late):
                first_entered, first_release = threading.Event(), threading.Event()
                retry_entered, retry_release = threading.Event(), threading.Event()
                backend = None
                count = 0
                def start(nav_id):
                    nonlocal count
                    count += 1
                    if count == 1:
                        backend.callback({"nav_id": nav_id, "status": "rejected"})
                        first_entered.set()
                        if not first_release.wait(5):
                            raise AssertionError("first not released")
                        if late == "timeout":
                            raise TimeoutError("late")
                        if late == "known_error":
                            raise NavigationBackendError("rejected", "late", outcome_known=True)
                        return {"status": "arrived" if late == "terminal" else "navigating"}
                    retry_entered.set()
                    if not retry_release.wait(5):
                        raise AssertionError("retry not released")
                    return {"status": "navigating"}
                backend = Backend(start)
                core = Nav2Core(backend)
                old = threading.Thread(target=lambda: core.dispatch(goal()))
                new = None
                old.start()
                try:
                    self.assertTrue(first_entered.wait(3))
                    # Simulate retiring the completed receipt while the first
                    # backend call is still in flight; this is an adversarial
                    # internal-state test, not proof of safe ID reuse in ROS.
                    with core._lock:
                        core._trusted_requests.pop("goal-1")
                        core._trusted_tokens.pop("goal-1")
                        core._terminal_results.pop("goal-1")
                    new = threading.Thread(target=lambda: core.dispatch(goal()))
                    new.start()
                    self.assertTrue(retry_entered.wait(3))
                    first_release.set()
                    old.join(5)
                    self.assertFalse(old.is_alive())
                    self.assert_unknown(core.dispatch(goal()), "navigation_start_pending")
                    self.assertEqual(core.info()["active_nav_id"], "goal-1")
                finally:
                    first_release.set()
                    retry_release.set()
                    old.join(5)
                    if new:
                        new.join(5)
                self.assertFalse(new.is_alive())
                self.assertEqual(core.dispatch(goal())["status"], "navigating")
                self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1")] * 2)

    def test_late_old_stop_response_cannot_release_reused_id(self):
        stop_entered, stop_release = threading.Event(), threading.Event()
        second_entered, second_release = threading.Event(), threading.Event()
        backend = None
        count = 0
        def start(nav_id):
            nonlocal count
            count += 1
            if count == 1:
                return {"status": "navigating"}
            second_entered.set()
            if not second_release.wait(5):
                raise AssertionError("second start not released")
            return {"status": "navigating"}
        backend = Backend(start)
        original = backend.execute
        def execute(action, args, *, nav_id):
            if action == "stop_nav":
                backend.calls.append((action, nav_id))
                stop_entered.set()
                if not stop_release.wait(5):
                    raise AssertionError("stop not released")
                return {"status": "stopped", "terminal_confirmed": True}
            return original(action, args, nav_id=nav_id)
        backend.execute = execute
        core = Nav2Core(backend)
        self.assertEqual(core.dispatch(goal())["status"], "navigating")
        old = threading.Thread(target=lambda: core.dispatch({"action": "stop_nav"}))
        new = None
        old.start()
        try:
            self.assertTrue(stop_entered.wait(3))
            backend.callback({"nav_id": "goal-1", "status": "stopped"})
            with core._lock:
                core._trusted_requests.pop("goal-1")
                core._trusted_tokens.pop("goal-1")
                core._terminal_results.pop("goal-1")
            new = threading.Thread(target=lambda: core.dispatch(goal()))
            new.start()
            self.assertTrue(second_entered.wait(3))
            stop_release.set()
            old.join(5)
            self.assertFalse(old.is_alive())
            self.assert_unknown(core.dispatch(goal()), "navigation_start_pending")
        finally:
            stop_release.set()
            second_release.set()
            old.join(5)
            if new:
                new.join(5)
        self.assertFalse(new.is_alive())
        self.assertEqual(core.dispatch(goal())["status"], "navigating")

    def test_unconfirmed_stop_retries_and_preserves_active_until_confirmation(self):
        malformed = (
            None, [], {}, {"status": "stopped"},
            {"status": "stopped", "terminal_confirmed": 1},
            {"status": "stopped", "terminal_confirmed": True, "state": "navigating"},
            {"status": "stopped", "terminal_confirmed": True, "ok": False},
            {"status": "stopped", "terminal_confirmed": True, "ok": "true"},
            {"status": "stopped", "terminal_confirmed": True, "error": "failed"},
            {"status": "stopped", "terminal_confirmed": True, "error_code": "failed"},
            {"status": "stopped", "terminal_confirmed": True, "already_idle": True},
            {"status": "arrived", "already_terminal": "arrived", "terminal_confirmed": False},
            {"status": "stopped", "terminal_confirmed": True, "nav_id": "other"},
            {"status": "stopped", "terminal_confirmed": True, "action_id": "other"},
            {"status": "stopped", "terminal_confirmed": True, "already_terminal": "arrived"},
            {"status": "arrived"},
            {"status": "arrived", "already_terminal": "stopped"},
            {"status": "error", "terminal_confirmed": True},
        )
        for raw in malformed:
            with self.subTest(raw=raw):
                backend = Backend(lambda _id: {"status": "navigating"})
                original = backend.execute
                responses = iter((raw, raw, {"status": "stopped", "terminal_confirmed": True}))
                def execute(action, args, *, nav_id):
                    if action == "stop_nav":
                        backend.calls.append((action, nav_id))
                        return next(responses)
                    return original(action, args, nav_id=nav_id)
                backend.execute = execute
                completions = []
                core = Nav2Core(backend, completion_callback=completions.append)
                core.dispatch(goal())
                for _ in range(2):
                    self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "error")
                    self.assertEqual(core.info()["active_nav_id"], "goal-1")
                    self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
                    self.assertEqual(completions, [])
                self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")
                self.assertIsNone(core.info()["active_nav_id"])
                self.assertEqual(len(completions), 1)
                self.assertEqual(len(backend.calls), 4)

    def test_stop_exception_keeps_active_and_invalidates_accepted_replay(self):
        backend = Backend(lambda _id: {"status": "navigating"})
        original = backend.execute
        def execute(action, args, *, nav_id):
            if action == "stop_nav":
                raise TimeoutError("uncertain stop")
            return original(action, args, nav_id=nav_id)
        backend.execute = execute
        core = Nav2Core(backend)
        core.dispatch(goal())
        self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "error")
        self.assertEqual(core.info()["active_nav_id"], "goal-1")
        self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
        backend.execute = original
        self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")

    def test_canvas_stop_retries_unconfirmed_then_releases_on_confirmation(self):
        from tests.test_nav2_plugin import Nav2Plugin, ReadyBackend, _bindings
        class StopBackend(ReadyBackend):
            def __init__(self):
                super().__init__()
                self.responses = iter(({"status": "stopped"}, {"status": "stopped"},
                                       {"status": "stopped", "terminal_confirmed": True}))
            def execute(self, action, args, *, nav_id):
                if action == "stop_nav":
                    self.execute_calls.append((action, dict(args), nav_id))
                    return next(self.responses)
                return super().execute(action, args, nav_id=nav_id)
        backend = StopBackend()
        plugin = Nav2Plugin({}, None, backend=backend)
        self.assertEqual(plugin.dispatch("nav2", {"action": "start", "input_bindings": _bindings(plugin)})["state"], "ready")
        self.assertEqual(plugin.dispatch("nav2", {"action": "navigate_to_pose", "x": 0, "y": 0, "yaw": 0})["status"], "navigating")
        for _ in range(2):
            self.assertEqual(plugin.dispatch("nav2", {"action": "stop"})["error_code"], "navigation_stop_pending")
            self.assertTrue(plugin._canvas_started)
            self.assertIsNotNone(plugin._core.info()["active_nav_id"])
            self.assertEqual(backend.stop_calls, 0)
        self.assertEqual(plugin.dispatch("nav2", {"action": "stop"})["state"], "idle")
        self.assertEqual(backend.stop_calls, 1)
        self.assertEqual(sum(call[0] == "stop_nav" for call in backend.execute_calls), 3)

    def test_known_already_terminal_stop_response_releases_active(self):
        for status in ("arrived", "succeeded", "cancelled", "stopped", "timeout", "aborted", "rejected"):
            with self.subTest(status=status):
                backend = Backend(lambda _id: {"status": "navigating"})
                original = backend.execute
                backend.execute = lambda action, args, *, nav_id: (
                    {"status": status, "already_terminal": status}
                    if action == "stop_nav" else original(action, args, nav_id=nav_id)
                )
                core = Nav2Core(backend)
                core.dispatch(goal())
                self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], status)
                self.assertIsNone(core.info()["active_nav_id"])

    def test_late_unconfirmed_stop_cannot_resurrect_terminal_callback(self):
        entered, release = threading.Event(), threading.Event()
        backend = Backend(lambda _id: {"status": "navigating"})
        original = backend.execute
        def execute(action, args, *, nav_id):
            if action == "stop_nav":
                entered.set()
                if not release.wait(5):
                    raise AssertionError("stop not released")
                return {"status": "stopped"}
            return original(action, args, nav_id=nav_id)
        backend.execute = execute
        completions = []
        core = Nav2Core(backend, completion_callback=completions.append)
        core.dispatch(goal())
        response = []
        thread = threading.Thread(target=lambda: response.append(core.dispatch({"action": "stop_nav"})))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            backend.callback({"nav_id": "goal-1", "status": "arrived"})
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(response[0]["error_code"], "navigation_stop_pending")
        self.assertIsNone(core.info()["active_nav_id"])
        self.assertEqual(core.dispatch(goal())["status"], "arrived")
        self.assertEqual(len(completions), 1)

    def test_unconfirmed_direct_error_keeps_reservation(self):
        backend = Backend(lambda _id: {"status": "error", "error": "unknown"})
        core = Nav2Core(backend)
        self.assertEqual(core.dispatch(goal())["status"], "error")
        self.assertEqual(core.info()["active_nav_id"], "goal-1")
        self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
        self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")

    def test_unconfirmed_error_does_not_release_active_for_callback_or_control(self):
        for origin in ("callback", "stop_nav", "wait_navigation_done"):
            with self.subTest(origin=origin):
                backend = Backend(lambda _id: {"status": "navigating"})
                original = backend.execute
                def execute(action, args, *, nav_id):
                    if action in ("stop_nav", "wait_navigation_done"):
                        backend.calls.append((action, nav_id))
                        return {"status": "error", "error": "unknown outcome"}
                    return original(action, args, nav_id=nav_id)
                backend.execute = execute
                completions = []
                core = Nav2Core(backend, completion_callback=completions.append)
                self.assertEqual(core.dispatch(goal())["status"], "navigating")
                if origin == "callback":
                    backend.callback({"nav_id": "goal-1", "status": "error"})
                else:
                    self.assertEqual(core.dispatch({"action": origin})["status"], "error")
                self.assertEqual(core.info()["active_nav_id"], "goal-1")
                self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
                self.assertEqual(completions, [])
                backend.execute = original
                self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")

    def test_confirmed_error_is_terminal(self):
        for origin in ("direct", "callback", "wait_navigation_done"):
            with self.subTest(origin=origin):
                backend = Backend(lambda _id: {"status": "error", "terminal_confirmed": True}
                                  if origin == "direct" else {"status": "navigating"})
                original = backend.execute
                def execute(action, args, *, nav_id):
                    if action == "wait_navigation_done":
                        return {"status": "error", "terminal_confirmed": True}
                    return original(action, args, nav_id=nav_id)
                backend.execute = execute
                core = Nav2Core(backend)
                core.dispatch(goal())
                if origin == "callback":
                    backend.callback({"nav_id": "goal-1", "status": "error",
                                      "terminal_confirmed": True})
                if origin == "wait_navigation_done":
                    core.dispatch({"action": origin})
                self.assertIsNone(core.info()["active_nav_id"])
                self.assertEqual(core.dispatch(goal())["status"], "error")
                self.assertTrue(core.dispatch(goal())["idempotent_replay"])

    def test_unconfirmed_error_before_navigating_reply_stays_unknown(self):
        entered, release = threading.Event(), threading.Event()
        def start(_id):
            entered.set()
            if not release.wait(5):
                raise AssertionError("start not released")
            return {"status": "navigating"}
        backend = Backend(start)
        core = Nav2Core(backend)
        thread = threading.Thread(target=lambda: core.dispatch(goal()))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            backend.callback({"nav_id": "goal-1", "status": "error"})
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assert_unknown(core.dispatch(goal()), "navigation_start_unknown")
        self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")

    def test_stop_before_start_response_preserves_stopped_receipt(self):
        entered, release = threading.Event(), threading.Event()
        def start(_id):
            entered.set()
            if not release.wait(5):
                raise AssertionError("start not released")
            return {"status": "navigating"}
        backend = Backend(start)
        core = Nav2Core(backend)
        first = []
        thread = threading.Thread(target=lambda: first.append(core.dispatch(goal())))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(core.dispatch({"action": "stop_nav"})["status"], "stopped")
            self.assertEqual(core.dispatch(goal())["status"], "stopped")
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(core.dispatch(goal())["status"], "stopped")
        self.assertIsNone(core.info()["active_nav_id"])
        self.assertEqual(backend.calls, [("navigate_to_pose", "goal-1"), ("stop_nav", "goal-1")])


if __name__ == "__main__":
    unittest.main()
