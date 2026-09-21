"""Real ROS action races; run in the deployed ROS image with --network none."""
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plugins/navigation/runtime/nav2'))
try:
    import rclpy
    from action_msgs.msg import GoalStatus
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
except ImportError:
    rclpy = None

if rclpy is not None:
    from nav2.action_client import RaceSafeActionClient


@unittest.skipIf(rclpy is None, 'requires ROS action runtime')
class ActionClientRaceTests(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = rclpy.create_node('action_registration_race_test')
        group = ReentrantCallbackGroup()
        self.reject = False
        self.wait_for_cancel = False
        self.goal_delay = 0
        self.server = ActionServer(
            self.node, NavigateToPose, '/isolated_navigation_test', self.execute,
            callback_group=group, goal_callback=self.goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )
        self.client = RaceSafeActionClient(
            self.node, NavigateToPose, '/isolated_navigation_test', callback_group=group)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin)
        self.thread.start()
        self.assertTrue(self.client.wait_for_server(timeout_sec=3))
        self.original = self.client._client_handle

    def tearDown(self):
        self.executor.shutdown(timeout_sec=5)
        self.thread.join(5)
        self.client._client_handle = self.original
        self.client.destroy()
        self.server.destroy()
        self.node.destroy_node()
        rclpy.shutdown()

    def goal(self, _):
        time.sleep(self.goal_delay)
        return GoalResponse.REJECT if self.reject else GoalResponse.ACCEPT

    def execute(self, handle):
        deadline = time.monotonic() + 4
        while self.wait_for_cancel and not handle.is_cancel_requested:
            if time.monotonic() >= deadline:
                handle.abort()
                return NavigateToPose.Result()
            time.sleep(.01)
        if handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.succeed()
        return NavigateToPose.Result()

    def delay_transport_return(self, method):
        original = self.original
        class DelayedHandle:
            def __getattr__(self, name):
                target = getattr(original, name)
                if name != method:
                    return target
                def delayed(*args, **kwargs):
                    sequence = target(*args, **kwargs)
                    time.sleep(.3)  # Response arrives before pending-map registration.
                    return sequence
                return delayed
        self.client._client_handle = DelayedHandle()

    def resolved(self, future):
        deadline = time.monotonic() + 4
        while not future.done() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(future.done(), 'response was lost or callbacks deadlocked')
        return future.result()

    def test_early_goal_response_and_callback_result(self):
        self.delay_transport_return('send_goal_request')
        future = self.client.send_goal_async(NavigateToPose.Goal())
        results = []
        done = threading.Event()
        def accepted(completed):
            # Real production flow starts another request inside a done callback.
            result = completed.result().get_result_async()
            def terminal(completed_result):
                results.append(completed_result.result().status)
                done.set()
            result.add_done_callback(terminal)
        future.add_done_callback(accepted)
        self.assertTrue(done.wait(4))
        self.assertEqual(results, [GoalStatus.STATUS_SUCCEEDED])

    def test_early_result_response(self):
        handle = self.resolved(self.client.send_goal_async(NavigateToPose.Goal()))
        self.delay_transport_return('send_result_request')
        result = self.resolved(handle.get_result_async())
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED)

    def test_early_cancel_response_and_cancelled_terminal(self):
        self.wait_for_cancel = True
        handle = self.resolved(self.client.send_goal_async(NavigateToPose.Goal()))
        self.delay_transport_return('send_cancel_request')
        cancelled = self.resolved(handle.cancel_goal_async())
        self.assertEqual(len(cancelled.goals_canceling), 1)
        self.assertEqual(self.resolved(handle.get_result_async()).status,
                         GoalStatus.STATUS_CANCELED)

    def test_early_rejected_goal_response(self):
        self.reject = True
        self.delay_transport_return('send_goal_request')
        handle = self.resolved(self.client.send_goal_async(NavigateToPose.Goal()))
        self.assertFalse(handle.accepted)

    def test_slow_response_remains_pending_until_received(self):
        self.goal_delay = .3
        future = self.client.send_goal_async(NavigateToPose.Goal())
        time.sleep(.05)
        self.assertFalse(future.done())
        handle = self.resolved(future)
        self.assertTrue(handle.accepted)
        self.assertEqual(self.resolved(handle.get_result_async()).status,
                         GoalStatus.STATUS_SUCCEEDED)


if __name__ == '__main__':
    unittest.main()
