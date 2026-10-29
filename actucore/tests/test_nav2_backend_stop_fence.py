"""Local in-flight transport ordering, using fakes only; not a DDS fence."""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from plugins.navigation.planning.backend import RosTopicNavigationBackend
from plugins.navigation.planning.core import NavigationBackendError


def backend_fixture():
    backend = RosTopicNavigationBackend.__new__(RosTopicNavigationBackend)
    backend._condition = threading.Condition()
    backend._closed = False
    backend._responses = {}
    backend._pending_goal_commands = {}
    backend._velocity_limits = {}
    backend._request_timeout = 0.1
    backend._discovery_timeout = 1.0
    backend._String = SimpleNamespace
    backend._command_pub = Mock()
    backend._command_pub.get_subscription_count.return_value = 1
    return backend


class BackendStopFenceTest(unittest.TestCase):
    def test_stop_fences_registered_goal_and_resume_before_discovery(self):
        for action in ("navigate_to_pose", "resume_nav"):
            with self.subTest(action=action):
                backend = backend_fixture()
                entered, release = threading.Event(), threading.Event()
                errors = []

                def discovery():
                    if threading.current_thread().name == "motion":
                        entered.set()
                        if not release.wait(2):
                            raise RuntimeError("discovery release missing")
                    else:
                        raise NavigationBackendError("no_subscriber", "stop unavailable", outcome_known=True)

                backend._wait_for_bridge = discovery

                def motion():
                    try:
                        backend._request(action, {}, nav_id="n")
                    except Exception as exc:
                        errors.append(exc)

                thread = threading.Thread(target=motion, name="motion")
                thread.start()
                try:
                    self.assertTrue(entered.wait(1))
                    with self.assertRaises(NavigationBackendError):
                        backend._request("stop_nav", {}, nav_id="n")
                finally:
                    release.set()
                    thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0].code, "navigation_command_stopped")
                self.assertIs(errors[0].outcome_known, True)
                backend._command_pub.publish.assert_not_called()
                self.assertEqual(backend._pending_goal_commands, {})

    def test_published_goal_is_not_withdrawn_and_stop_still_publishes(self):
        backend = backend_fixture()
        published = threading.Event()
        sent = []
        errors = []

        def publish(message):
            payload = json.loads(message.data)
            sent.append(payload)
            if payload["action"] == "navigate_to_pose":
                published.set()
            else:
                backend._responses[payload["request_id"]] = {"status": "stopped"}

        backend._command_pub.publish.side_effect = publish

        def motion():
            try:
                backend._request("navigate_to_pose", {}, nav_id="n")
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=motion)
        thread.start()
        try:
            self.assertTrue(published.wait(1))
            self.assertEqual(backend._request("stop_nav", {}, nav_id="n")["status"], "stopped")
        finally:
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([item["action"] for item in sent], ["navigate_to_pose", "stop_nav"])
        self.assertEqual(errors[0].code, "nav2_response_timeout")
        self.assertFalse(errors[0].outcome_known)
        self.assertEqual(backend._pending_goal_commands, {})

    def test_other_ids_and_later_same_id_not_tombstoned(self):
        backend = backend_fixture()
        backend._command_pub.publish.side_effect = lambda msg: backend._responses.update({
            json.loads(msg.data)["request_id"]: {"status": "ok"}
        })
        self.assertEqual(backend._request("stop_nav", {}, nav_id="n")["status"], "ok")
        for action, nav_id in (("navigate_to_pose", "n"), ("resume_nav", "other")):
            self.assertEqual(backend._request(action, {}, nav_id=nav_id)["status"], "ok")
        self.assertEqual(backend._pending_goal_commands, {})

    def test_stop_only_fences_matching_concurrent_nav_id(self):
        backend = backend_fixture()
        entered = {name: threading.Event() for name in ("same", "other")}
        release = threading.Event()
        results = {}

        def discovery():
            name = threading.current_thread().name
            if name in entered:
                entered[name].set()
                if not release.wait(2):
                    raise RuntimeError("release missing")

        backend._wait_for_bridge = discovery
        backend._command_pub.publish.side_effect = lambda msg: backend._responses.update({
            json.loads(msg.data)["request_id"]: {"status": "ok"}
        })

        def request(name):
            try:
                results[name] = backend._request("navigate_to_pose", {}, nav_id=name)
            except Exception as exc:
                results[name] = exc

        threads = [threading.Thread(name=name, target=request, args=(name,)) for name in entered]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(all(event.wait(1) for event in entered.values()))
            backend._request("stop_nav", {}, nav_id="same")
        finally:
            release.set()
            for thread in threads:
                thread.join(3)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(results["same"].code, "navigation_command_stopped")
        self.assertEqual(results["other"]["status"], "ok")
        self.assertEqual(backend._pending_goal_commands, {})

    def test_unknown_publish_exception_cleans_tracking(self):
        backend = backend_fixture()
        backend._command_pub.publish.side_effect = RuntimeError("publish uncertain")
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            backend._request("navigate_to_pose", {}, nav_id="n")
        self.assertEqual(backend._pending_goal_commands, {})

    def test_pause_does_not_fence_inflight_resume(self):
        backend = backend_fixture()
        entered, release = threading.Event(), threading.Event()
        results = []

        def discovery():
            if threading.current_thread().name == "resume":
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("release missing")

        backend._wait_for_bridge = discovery
        backend._command_pub.publish.side_effect = lambda msg: backend._responses.update({
            json.loads(msg.data)["request_id"]: {"status": "ok"}
        })
        thread = threading.Thread(name="resume", target=lambda: results.append(
            backend._request("resume_nav", {}, nav_id="n")
        ))
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            backend._request("pause_nav", {}, nav_id="n")
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(backend._pending_goal_commands, {})


if __name__ == "__main__":
    unittest.main()
