"""Serialize action request registration with response taking on older rclpy."""

import threading

from rclpy.action import ActionClient


class RaceSafeActionClient(ActionClient):
    """Keep responses queued until their pending Future has been registered.

    Older rclpy sends a request before adding its sequence number to the pending
    maps. A MultiThreadedExecutor can take and discard the response in that gap.
    Only transport taking and synchronous registration share this lock: execute,
    user callbacks and Future waits must remain outside it.
    """

    def __init__(self, *args, **kwargs):
        self._registration_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def send_goal_async(self, goal, feedback_callback=None, goal_uuid=None):
        with self._registration_lock:
            return super().send_goal_async(goal, feedback_callback, goal_uuid)

    def _cancel_goal_async(self, goal_handle):
        with self._registration_lock:
            return super()._cancel_goal_async(goal_handle)

    def _get_result_async(self, goal_handle):
        with self._registration_lock:
            return super()._get_result_async(goal_handle)

    def take_data(self):
        with self._registration_lock:
            return super().take_data()
