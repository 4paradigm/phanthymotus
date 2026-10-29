"""ROS backend close/publish ordering without requiring a ROS installation."""

import json
import threading
import unittest
from unittest.mock import Mock

from plugins.navigation.planning.backend import RosTopicNavigationBackend
from plugins.navigation.planning.core import NavigationBackendError


class _String:
    data = ""


def _backend():
    # Construct only the transport state, so the test exercises the real request
    # and stop methods without a ROS node, executor, or publisher.
    backend = RosTopicNavigationBackend.__new__(RosTopicNavigationBackend)
    backend._condition = threading.Condition()
    backend._closed = False
    backend._responses = {}
    backend._pending_goal_commands = {}
    backend._velocity_limits = {"max_x_mps": 1.0}
    backend._request_timeout = 1.0
    backend._discovery_timeout = 1.0
    backend._String = _String
    backend._command_pub = Mock()
    backend._command_pub.get_subscription_count.return_value = 1
    backend._executor = Mock()
    backend._node = Mock()
    return backend


class BackendClosePublishTest(unittest.TestCase):
    def test_close_after_bridge_discovery_prevents_publish(self):
        backend = _backend()
        discovered = threading.Event()
        resume = threading.Event()
        actual_wait = backend._wait_for_bridge
        errors = []

        def wait_then_pause():
            actual_wait()
            discovered.set()
            if not resume.wait(timeout=2):
                raise AssertionError("request was not resumed")

        def request():
            try:
                backend._request("navigate_to_pose", {"x": 1}, nav_id="nav-1")
            except Exception as exc:
                errors.append(exc)

        backend._wait_for_bridge = wait_then_pause
        thread = threading.Thread(target=request, daemon=True)
        thread.start()
        try:
            self.assertTrue(discovered.wait(timeout=2), "bridge was not discovered")
            backend.stop()
        finally:
            resume.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "request did not finish")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], NavigationBackendError)
        self.assertEqual(errors[0].code, "backend_closed")
        self.assertTrue(errors[0].outcome_known)
        backend._command_pub.publish.assert_not_called()
        backend._executor.remove_node.assert_called_once_with(backend._node)
        backend._node.destroy_node.assert_called_once_with()

    def test_close_after_publish_while_waiting_has_unknown_outcome(self):
        backend = _backend()
        backend._request_timeout = 5.0
        waiting = threading.Event()
        errors = []
        actual_wait = backend._condition.wait

        def signal_wait(timeout=None):
            # Called while holding the condition, so stop cannot close the
            # backend until wait atomically releases that same lock.
            waiting.set()
            return actual_wait(timeout=timeout)

        def request():
            try:
                backend._request("navigate_to_pose", {"x": 1}, nav_id="nav-1")
            except Exception as exc:
                errors.append(exc)

        backend._condition.wait = signal_wait
        thread = threading.Thread(target=request, daemon=True)
        thread.start()
        try:
            self.assertTrue(waiting.wait(timeout=2), "request did not wait for a reply")
        finally:
            backend.stop()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "request did not finish")
        backend._command_pub.publish.assert_called_once()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], NavigationBackendError)
        self.assertEqual(errors[0].code, "backend_closed")
        self.assertFalse(errors[0].outcome_known)

    def test_open_backend_publishes_and_returns_matching_response(self):
        backend = _backend()
        published = []

        def publish(message):
            payload = json.loads(message.data)
            published.append(payload)
            backend._responses[payload["request_id"]] = {
                "event": "response",
                "request_id": payload["request_id"],
                "status": "accepted",
            }

        backend._command_pub.publish.side_effect = publish
        response = backend._request("navigate_to_pose", {"x": 1}, nav_id="nav-1")
        backend._command_pub.publish.assert_called_once()
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["action"], "navigate_to_pose")
        self.assertEqual(published[0]["args"], {"x": 1})
        self.assertEqual(published[0]["nav_id"], "nav-1")
        self.assertEqual(response["request_id"], published[0]["request_id"])
        self.assertEqual(response["status"], "accepted")
        self.assertEqual(backend._responses, {})


if __name__ == "__main__":
    unittest.main()
