"""ROS-free tests of sticky cancellation for late goal acceptance."""

from concurrent.futures import Future
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_nav2_planner_cancel_characterization import (
    make_node, planner_module, result_future, result_future_object,
)


def test_pending_stop_late_accept_cancels_without_claiming_terminal(planner_module):
    node = make_node(planner_module)
    node._active.update(status="cancel_unconfirmed", goal_handle=None,
                        goal_response_pending=True, cancel_intent=None)
    with pytest.raises(planner_module.CommandError) as exc:
        node._stop("nav-1")
    assert exc.value.outcome_known is False
    assert node._active["cancel_intent"] == "stop"

    handle = Mock(accepted=True)
    result = Future()
    handle.get_result_async.return_value = result
    handle.cancel_goal_async.return_value = result_future_object(
        SimpleNamespace(goals_canceling=[object()])
    )
    node._on_goal_response("nav-1", 1, result_future_object(handle), threading.Event(), {})
    handle.cancel_goal_async.assert_called_once_with()
    assert node._active["status"] == "cancel_unconfirmed"
    assert node._active["goal_handle"] is handle
    result.set_result(SimpleNamespace(status=5))
    assert node._active["status"] == "stopped"


def test_failed_late_cancel_retains_handle_for_retry(planner_module):
    node = make_node(planner_module)
    node.get_logger = Mock(return_value=Mock())
    node._active.update(status="cancel_unconfirmed", goal_handle=None,
                        goal_response_pending=True, cancel_intent=None)
    with pytest.raises(planner_module.CommandError):
        node._stop("nav-1")
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = Future()
    handle.cancel_goal_async.side_effect = RuntimeError("cancel failure")
    node._on_goal_response("nav-1", 1, result_future_object(handle), threading.Event(), {})
    assert node._active["status"] == "cancel_unconfirmed"
    assert node._active["goal_handle"] is handle
    def retry():
        node._on_result("nav-1", 1, result_future(5))
        return result_future_object(SimpleNamespace(goals_canceling=[object()]))
    handle.cancel_goal_async.side_effect = retry
    assert node._stop("nav-1")["terminal_confirmed"] is True
    assert handle.cancel_goal_async.call_count == 2


def test_late_rejection_does_not_send_cancel(planner_module):
    node = make_node(planner_module)
    node._active.update(status="cancel_unconfirmed", goal_handle=None,
                        goal_response_pending=True, cancel_intent=None)
    with pytest.raises(planner_module.CommandError):
        node._stop("nav-1")
    handle = Mock(accepted=False)
    node._on_goal_response("nav-1", 1, result_future_object(handle), threading.Event(), {})
    handle.cancel_goal_async.assert_not_called()
    assert node._active["status"] == "rejected"


def test_synchronous_terminal_callback_does_not_deadlock(planner_module):
    node = make_node(planner_module)
    node._active.update(status="cancel_unconfirmed", goal_handle=None,
                        goal_response_pending=True, cancel_intent="stop")
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = result_future(5)
    thread = threading.Thread(target=node._on_goal_response, args=(
        "nav-1", 1, result_future_object(handle), threading.Event(), {}
    ))
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert node._active["status"] == "stopped"
