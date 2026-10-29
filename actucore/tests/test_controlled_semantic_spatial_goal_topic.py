from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ACTUCORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTUCORE_ROOT))

from plugins.controlled_semantic_spatial.planning.goal_input import GoalInputCleanupError  # noqa: E402
from plugins.controlled_semantic_spatial.plugin import ControlledSemanticSpatialPlugin  # noqa: E402
from tests.test_controlled_semantic_spatial_plugin import FakeComponent, FakeRuntime  # noqa: E402


class FakeGoalInput:
    instances = []

    def __init__(self, topic, executor, dispatch):
        self.topic = topic
        self.executor = executor
        self.dispatch = dispatch
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


class ControlledSemanticSpatialGoalTopicTest(unittest.TestCase):
    def setUp(self):
        FakeGoalInput.instances = []
        self.factory = mock.patch(
            "plugins.controlled_semantic_spatial.plugin.GoalTopicInput", FakeGoalInput
        )
        self.factory.start()
        self.addCleanup(self.factory.stop)
        self.runtime = FakeRuntime()
        self.planning = FakeComponent("planning")
        self.plugin = ControlledSemanticSpatialPlugin(
            {}, "ubuntu", executor=object(), runtime=self.runtime,
            mapping_plugin=FakeComponent("mapping"), planning_plugin=self.planning,
            semantic_plugin=FakeComponent("semantic"),
        )
        self.request = {
            "action": "start",
            "input_bindings": [
                {"port": "lidar", "topic": "/source/lidar"},
                {"port": "imu", "topic": "/source/imu"},
                {"port": "goal_pose", "topic": "/source/goal"},
            ],
        }
        self.goal = {"action": "navigate_to_pose", "_control_nav_id": "goal-1",
                     "x": 1.0, "y": 2.0, "yaw": 0.5}

    def call(self, args):
        return self.plugin.dispatch(self.plugin.PREFIX, args)

    def test_binding_dispatch_and_close(self):
        self.assertEqual(self.call(self.request)["state"], "ready")
        sub = FakeGoalInput.instances[0]
        self.assertEqual(sub.topic, "/source/goal")
        self.assertIs(sub.executor, self.plugin._executor)
        planning_start = next(args for _, args in self.planning.calls
                              if args.get("action") == "start")
        self.assertNotIn("goal_pose", [item["port"] for item in planning_start["input_bindings"]])
        self.assertEqual(sub.dispatch(self.goal)["status"], "navigate_to_pose")
        self.assertEqual(self.planning.calls[-1], ("nav2", self.goal))
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")
        self.assertTrue(sub.closed)
        self.assertEqual(sub.dispatch(self.goal)["error_code"], "navigation_stopping")

    def test_unbound_input_does_not_subscribe(self):
        request = {**self.request, "input_bindings": self.request["input_bindings"][:2]}
        self.assertEqual(self.call(request)["state"], "ready")
        self.assertEqual(FakeGoalInput.instances, [])
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")

    def test_old_callback_cannot_enter_restarted_generation(self):
        self.assertEqual(self.call(self.request)["state"], "ready")
        old = FakeGoalInput.instances[-1]
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")
        self.assertEqual(self.call(self.request)["state"], "ready")
        new = FakeGoalInput.instances[-1]
        count = len(self.planning.calls)
        self.assertEqual(old.dispatch(self.goal)["error_code"], "navigation_stopping")
        self.assertEqual(len(self.planning.calls), count)
        self.assertEqual(new.dispatch(self.goal)["status"], "navigate_to_pose")

    def test_stop_between_runtime_check_and_admission_rejects_old_goal(self):
        self.assertEqual(self.call(self.request)["state"], "ready")
        old = FakeGoalInput.instances[-1]
        entered, release = threading.Event(), threading.Event()
        original = self.runtime.info
        worker_id = []

        def blocked_info():
            if worker_id and threading.get_ident() == worker_id[0]:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timeout")
            return original()

        self.runtime.info = blocked_info
        result = {}

        def dispatch_old():
            worker_id.append(threading.get_ident())
            result["goal"] = old.dispatch(self.goal)

        worker = threading.Thread(target=dispatch_old)
        worker.start()
        self.assertTrue(entered.wait(5))
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")
        self.assertEqual(self.call(self.request)["state"], "ready")
        count = len(self.planning.calls)
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result["goal"]["error_code"], "navigation_stopping")
        self.assertEqual(len(self.planning.calls), count)

    def test_dead_runtime_rejects_goal(self):
        self.assertEqual(self.call(self.request)["state"], "ready")
        self.runtime.started = False
        count = len(self.planning.calls)
        result = FakeGoalInput.instances[-1].dispatch(self.goal)
        self.assertEqual(result["error_code"], "navigation_runtime_unavailable")
        self.assertEqual(len(self.planning.calls), count)

    def test_constructor_failure_rolls_back(self):
        with mock.patch("plugins.controlled_semantic_spatial.plugin.GoalTopicInput", side_effect=RuntimeError("boom")):
            result = self.call(self.request)
        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertFalse(self.runtime.started)
        self.assertFalse(self.plugin._started)
        self.assertFalse(self.planning.started)

    def test_constructor_cleanup_handle_is_retained_until_stop_retry(self):
        class CleanupHandle:
            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.close_calls < 3:
                    raise RuntimeError("cleanup still pending")

        handle = CleanupHandle()
        failure = GoalInputCleanupError(handle, RuntimeError("construction failed"))
        with mock.patch("plugins.controlled_semantic_spatial.plugin.GoalTopicInput", side_effect=failure):
            result = self.call(self.request)
        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertFalse(result["cleanup_confirmed"])
        self.assertEqual(result["cleanup"]["goal_input"]["error_code"], "cleanup_failed")
        self.assertIs(self.plugin._goal_input, handle)
        self.assertTrue(self.plugin._stop_requested.is_set())
        self.assertFalse(self.runtime.started)
        self.assertFalse(self.planning.started)
        self.assertEqual(self.call(self.request)["error_code"], "navigation_start_cancelled")
        self.assertEqual(self.call({"action": "stop"})["error_code"], "navigation_stop_pending")
        self.assertIs(self.plugin._goal_input, handle)
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")
        self.assertIsNone(self.plugin._goal_input)
        self.assertEqual(handle.close_calls, 3)
        self.assertEqual(self.call(self.request)["state"], "ready")
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")

    def test_constructor_cleanup_handle_immediately_retries_successfully(self):
        handle = mock.Mock()
        failure = GoalInputCleanupError(handle, RuntimeError("construction failed"))
        with mock.patch("plugins.controlled_semantic_spatial.plugin.GoalTopicInput", side_effect=failure):
            result = self.call(self.request)
        self.assertEqual(result["error_code"], "navigation_start_failed")
        self.assertTrue(result["cleanup_confirmed"])
        handle.close.assert_called_once_with()
        self.assertIsNone(self.plugin._goal_input)
        self.assertFalse(self.plugin._stop_requested.is_set())
        self.assertFalse(self.runtime.started)
        self.assertFalse(self.planning.started)

    def test_stop_during_subscription_creation_closes_and_rolls_back(self):
        entered, release = threading.Event(), threading.Event()
        early = []

        class BlockingInput(FakeGoalInput):
            def __init__(self, topic, executor, dispatch):
                super().__init__(topic, executor, dispatch)
                early.append(dispatch(self_goal))
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timeout")

        self_goal = self.goal
        result = {}
        with mock.patch("plugins.controlled_semantic_spatial.plugin.GoalTopicInput", BlockingInput):
            starter = threading.Thread(target=lambda: result.setdefault("start", self.call(self.request)))
            starter.start()
            self.assertTrue(entered.wait(5))
            stopper = threading.Thread(target=lambda: result.setdefault("stop", self.call({"action": "stop"})))
            stopper.start()
            self.assertTrue(self.plugin._stop_requested.wait(5))
            release.set()
            starter.join(5)
            stopper.join(5)
        self.assertFalse(starter.is_alive())
        self.assertFalse(stopper.is_alive())
        self.assertEqual(early[0]["error_code"], "navigation_stopping")
        self.assertEqual(result["start"]["error_code"], "navigation_start_cancelled")
        self.assertEqual(result["stop"]["state"], "idle")
        self.assertTrue(FakeGoalInput.instances[-1].closed)
        self.assertFalse(self.runtime.started)

    def test_inflight_topic_goal_follows_stop_retry_fence(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingPlanning(FakeComponent):
            def dispatch(self, prefix, args):
                if args.get("action") == "navigate_to_pose":
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test timeout")
                return super().dispatch(prefix, args)

            def stop_active_only(self):
                return {"state": "idle", "status": "idle"}

        self.planning = BlockingPlanning("planning")
        self.plugin._planning = self.planning
        self.assertEqual(self.call(self.request)["state"], "ready")
        sub = FakeGoalInput.instances[-1]
        result = {}
        worker = threading.Thread(target=lambda: result.setdefault("goal", sub.dispatch(self.goal)))
        worker.start()
        self.assertTrue(entered.wait(5))
        stop = self.call({"action": "stop"})
        self.assertEqual(stop["error_code"], "navigation_stop_pending")
        self.assertTrue(sub.closed)
        self.assertTrue(self.runtime.started)
        self.assertEqual(sub.dispatch(self.goal)["error_code"], "navigation_stopping")
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")

    def test_failed_close_during_cancelled_start_still_rolls_back_components(self):
        entered, release = threading.Event(), threading.Event()

        class FailingCloseInput(FakeGoalInput):
            def __init__(self, topic, executor, dispatch):
                super().__init__(topic, executor, dispatch)
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timeout")

            def close(self):
                if not self.closed:
                    self.closed = True
                    raise RuntimeError("close failed")

        result = {}
        with mock.patch("plugins.controlled_semantic_spatial.plugin.GoalTopicInput", FailingCloseInput):
            starter = threading.Thread(target=lambda: result.setdefault("start", self.call(self.request)))
            starter.start()
            self.assertTrue(entered.wait(5))
            stopper = threading.Thread(target=lambda: result.setdefault("stop", self.call({"action": "stop"})))
            stopper.start()
            self.assertTrue(self.plugin._stop_requested.wait(5))
            release.set()
            starter.join(5)
            stopper.join(5)
        self.assertFalse(starter.is_alive())
        self.assertFalse(stopper.is_alive())
        self.assertFalse(result["start"]["cleanup_confirmed"])
        self.assertEqual(result["stop"]["state"], "idle")
        self.assertFalse(self.runtime.started)
        self.assertFalse(self.planning.started)

    def test_close_failure_keeps_stop_fenced_until_retry(self):
        class FailingCloseInput(FakeGoalInput):
            def close(self):
                if not self.closed:
                    self.closed = True
                    raise RuntimeError("close failed")

        with mock.patch("plugins.controlled_semantic_spatial.plugin.GoalTopicInput", FailingCloseInput):
            self.assertEqual(self.call(self.request)["state"], "ready")
        sub = FakeGoalInput.instances[-1]
        self.assertEqual(self.call({"action": "stop"})["error_code"], "navigation_stop_pending")
        self.assertTrue(self.runtime.started)
        self.assertEqual(sub.dispatch(self.goal)["error_code"], "navigation_stopping")
        self.assertEqual(self.call({"action": "stop"})["state"], "idle")


if __name__ == "__main__":
    unittest.main()
