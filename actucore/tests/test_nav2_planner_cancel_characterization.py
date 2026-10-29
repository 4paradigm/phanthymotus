"""Exercise planner cancellation and proposal ordering without ROS or hardware."""
from concurrent.futures import Future
import importlib
import json
import time
from pathlib import Path
import sys
import threading
import types
from unittest.mock import Mock

import pytest


@pytest.fixture
def planner_module(monkeypatch):
    # Reuse conftest's lightweight QoS/ROS message stubs. Add only the imports
    # needed to load the actual planner module, and remove them after each case.
    monkeypatch.syspath_prepend(str(
        Path(__file__).resolve().parents[1] / 'plugins/navigation/runtime/nav2'
    ))

    def provide(module_name, **members):
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            monkeypatch.setitem(sys.modules, module_name, module)
        for name, value in members.items():
            if not hasattr(module, name):
                monkeypatch.setattr(module, name, value, raising=False)
        return module

    class GoalStatus:
        STATUS_UNKNOWN = 0
        STATUS_SUCCEEDED = 4
        STATUS_CANCELED = 5
        STATUS_ABORTED = 6

    def stub(name):
        return type(name, (), {})

    provide('logsafe', install=lambda: None)
    provide('action_msgs.msg', GoalStatus=GoalStatus)
    provide('geometry_msgs.msg', Twist=stub('Twist'))
    provide('lifecycle_msgs.srv', GetState=stub('GetState'))
    provide('nav2_msgs.action', NavigateToPose=stub('NavigateToPose'))
    provide('nav2_msgs.msg', SpeedLimit=stub('SpeedLimit'))
    provide('nav_msgs.msg', OccupancyGrid=stub('OccupancyGrid'), Odometry=stub('Odometry'))
    provide('rclpy.action', ActionClient=stub('ActionClient'))
    provide('rclpy.callback_groups', ReentrantCallbackGroup=stub('ReentrantCallbackGroup'))
    provide('rclpy.duration', Duration=stub('Duration'))
    provide('rclpy.executors', MultiThreadedExecutor=stub('MultiThreadedExecutor'))
    provide('rclpy.node', Node=stub('Node'))
    provide('rclpy.qos', qos_profile_sensor_data=object())
    provide('rclpy.time', Time=stub('Time'))
    provide('sensor_msgs.msg', PointCloud2=stub('PointCloud2'))
    provide('tf2_ros', Buffer=stub('Buffer'), TransformListener=stub('TransformListener'))

    # The planner imports its real ROS-free helper modules and its real action
    # client class; no replacement of the method under test is involved.
    monkeypatch.delitem(sys.modules, 'nav2.planner_command_node', raising=False)
    monkeypatch.delitem(sys.modules, 'nav2.action_client', raising=False)
    module = importlib.import_module('nav2.planner_command_node')
    yield module
    # Do not leave a planner class tied to temporary ROS stubs in sys.modules.
    sys.modules.pop('nav2.planner_command_node', None)
    sys.modules.pop('nav2.action_client', None)


def make_node(module):
    node = module.PlannerCommandNode.__new__(module.PlannerCommandNode)
    node._lock = threading.RLock()
    node._state_changed = threading.Condition(node._lock)
    node._command_lock = threading.Lock()
    node._global_frame = 'map'
    node._supported_mode = 0
    node._proposal_ttl_ms = 250
    node._proposal_sequence = 0
    node._latest_proposal_candidate = None
    node._last_published_proposal = None
    node._proposal_pub = Mock()
    node._emit = Mock()
    node._readiness = Mock(return_value={})
    node._execution_status = Mock(return_value={})
    node._global_costmap_diagnostics = Mock(return_value={})
    node._require_navigation_ready = Mock()
    node._assert_shadow_isolated = Mock()
    node._send_active_goal = Mock()
    node._active = {
        'nav_id': 'nav-1', 'attempt': 1, 'status': 'navigating',
        'cancel_intent': 'stop', 'goal_handle': Mock(),
        'motion_limits': module.MotionLimits(),
    }
    return node


def result_future(status=None, *, exception=None):
    future = Future()
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(types.SimpleNamespace(status=status))
    return future


def last_proposal(node):
    return json.loads(node._proposal_pub.publish.call_args.args[0].data)


@pytest.mark.parametrize('intent', ['stop', 'pause', None])
@pytest.mark.parametrize('result_kind', ['unknown', 'exception', 'executing'])
def test_unconfirmed_result_preserves_ownership_and_emits_zero(planner_module, intent, result_kind):
    node = make_node(planner_module)
    node._active['cancel_intent'] = intent
    handle = node._active['goal_handle']
    node._latest_proposal_candidate = {'unsafe_old_candidate': True}
    status = {'unknown': 0, 'executing': 2}.get(result_kind)
    future = result_future(status, exception=RuntimeError('result failed') if result_kind == 'exception' else None)

    node._on_result('nav-1', 1, future)

    assert node._active['status'] == 'cancel_unconfirmed'
    assert node._active['goal_handle'] is handle
    assert node._active['cancel_intent'] == intent
    assert node._active['error_code'] == 'cancel_terminal_unconfirmed'
    assert node._latest_proposal_candidate is None
    assert node._emit.call_args.args[0]['status'] == 'cancel_unconfirmed'
    proposal = last_proposal(node)
    assert proposal['nav_status'] == 'planning'
    assert proposal['velocity'] == {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    planner_module.build_motion_sequence(
        nav_id='nav-1', sequence=1, ttl_ms=250, navigation_status='planning',
        velocity=planner_module.Velocity.zero(),
    )
    with pytest.raises(planner_module.CommandError) as resume:
        node._resume('nav-1')
    assert resume.value.code == 'invalid_navigation_state'
    with pytest.raises(planner_module.CommandError) as new_goal:
        node._navigate_to_pose('nav-2', {'x': 0, 'y': 0, 'yaw': 0, 'speed': 1, 'mode': 0}, planner_module.MotionLimits())
    assert new_goal.value.code == 'navigation_active'
    node._send_active_goal.assert_not_called()


@pytest.mark.parametrize('intent,code,expected', [
    ('stop', 5, 'stopped'), ('pause', 5, 'paused'), (None, 5, 'cancelled'),
    ('stop', 4, 'arrived'), ('pause', 4, 'arrived'),
    ('stop', 6, 'aborted'), ('pause', 6, 'aborted'),
])
def test_known_result_obeys_real_terminal_status(planner_module, intent, code, expected):
    node = make_node(planner_module)
    node._active['cancel_intent'] = intent
    node._on_result('nav-1', 1, result_future(code))
    assert node._active['status'] == expected
    assert node._active['goal_handle'] is None
    assert last_proposal(node)['nav_status'] == expected
    assert last_proposal(node)['velocity'] == {'x': 0.0, 'y': 0.0, 'yaw': 0.0}


def test_late_stale_result_does_not_mutate_or_publish(planner_module):
    node = make_node(planner_module)
    before = dict(node._active)
    node._on_result('nav-1', 2, result_future(5))
    node._on_result('other', 1, result_future(5))
    assert node._active == before
    node._emit.assert_not_called()
    node._proposal_pub.publish.assert_not_called()


@pytest.mark.parametrize('result_kind', ['unknown', 'exception', 'canceled', 'succeeded'])
def test_cancel_ack_interleaved_with_result(planner_module, result_kind):
    node = make_node(planner_module)
    handle = node._active['goal_handle']
    def cancel():
        code = {'unknown': 0, 'canceled': 5, 'succeeded': 4}.get(result_kind)
        result = result_future(code, exception=RuntimeError('lost') if result_kind == 'exception' else None)
        node._on_result('nav-1', 1, result)
        return result_future_object(types.SimpleNamespace(goals_canceling=[object()]))
    handle.cancel_goal_async.side_effect = cancel
    if result_kind == 'canceled':
        result = node._stop('nav-1')
        assert result['status'] == 'stopped' and result['terminal_confirmed'] is True
    else:
        with pytest.raises(planner_module.CommandError) as exc:
            node._stop('nav-1')
        assert exc.value.code == 'cancel_terminal_unconfirmed'
        assert exc.value.outcome_known is (result_kind == 'succeeded')
        assert node._active['status'] == ('arrived' if result_kind == 'succeeded' else 'cancel_unconfirmed')
        assert node._active['goal_handle'] is (None if result_kind == 'succeeded' else handle)


def result_future_object(value):
    future = Future()
    future.set_result(value)
    return future


def test_failed_cancel_can_retry_with_same_handle(planner_module):
    node = make_node(planner_module)
    handle = node._active['goal_handle']
    handle.cancel_goal_async.side_effect = RuntimeError('send failed')
    with pytest.raises(planner_module.CommandError) as exc:
        node._stop('nav-1')
    assert exc.value.outcome_known is False
    assert node._active['status'] == 'cancel_unconfirmed'
    assert node._active['goal_handle'] is handle
    def retry():
        node._on_result('nav-1', 1, result_future(5))
        return result_future_object(types.SimpleNamespace(goals_canceling=[object()]))
    handle.cancel_goal_async.side_effect = retry
    assert node._stop('nav-1')['terminal_confirmed'] is True
    assert handle.cancel_goal_async.call_count == 2


def test_goal_response_exception_retains_unknown_ownership_without_handle(planner_module):
    node = make_node(planner_module)
    node._active['goal_handle'] = None
    node._active['goal_response_pending'] = True
    accepted = threading.Event()
    outcome = {}
    node._on_goal_response('nav-1', 1, result_future(exception=RuntimeError('response lost')), accepted, outcome)
    assert accepted.is_set()
    assert outcome['outcome_known'] is False
    assert node._active['status'] == 'cancel_unconfirmed'
    assert node._active['goal_response_pending'] is True
    assert last_proposal(node)['nav_status'] == 'planning'
    for _ in range(2):
        with pytest.raises(planner_module.CommandError) as exc:
            node._stop('nav-1')
        assert exc.value.outcome_known is False
        assert node._active['status'] == 'cancel_unconfirmed'
        assert node._active['goal_handle'] is None
        assert node._active['attempt'] == 1
    with pytest.raises(planner_module.CommandError):
        node._resume('nav-1')
    with pytest.raises(planner_module.CommandError):
        node._navigate_to_pose('nav-2', {'x': 0, 'y': 0, 'yaw': 0, 'speed': 1, 'mode': 0}, planner_module.MotionLimits())


def test_late_goal_acceptance_after_timeout_preserves_fail_closed_state(planner_module):
    node = make_node(planner_module)
    node._active.update(status='cancel_unconfirmed', goal_handle=None, goal_response_pending=True)
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = Future()
    outcome = {}
    node._on_goal_response('nav-1', 1, result_future_object(handle), threading.Event(), outcome)
    assert node._active['goal_handle'] is handle
    assert node._active['status'] == 'cancel_unconfirmed'
    assert outcome['accepted'] is True
    assert last_proposal(node)['nav_status'] == 'planning'
    def cancel():
        node._on_result('nav-1', 1, result_future(5))
        return result_future_object(types.SimpleNamespace(goals_canceling=[object()]))
    handle.cancel_goal_async.side_effect = cancel
    assert node._stop('nav-1')['terminal_confirmed'] is True


@pytest.mark.parametrize('cancel_raises', [False, True])
def test_stale_accepted_goal_cancel_remains_unknown(planner_module, cancel_raises):
    node = make_node(planner_module)
    active_before = dict(node._active)
    handle = Mock(accepted=True)
    if cancel_raises:
        handle.cancel_goal_async.side_effect = RuntimeError('cancel transport lost')
    accepted = threading.Event()
    outcome = {}
    node._on_goal_response('old-nav', 1, result_future_object(handle), accepted, outcome)
    assert accepted.is_set()
    assert outcome['accepted'] is False
    assert outcome['outcome_known'] is False
    assert outcome['error_code'] == 'stale_goal_response'
    assert ('cancel request failed' if cancel_raises else 'terminal outcome is unconfirmed') in outcome['error']
    handle.cancel_goal_async.assert_called_once_with()
    assert node._active == active_before


def test_stale_rejected_goal_has_known_outcome(planner_module):
    node = make_node(planner_module)
    handle = Mock(accepted=False)
    accepted = threading.Event()
    outcome = {}
    node._on_goal_response('old-nav', 1, result_future_object(handle), accepted, outcome)
    assert accepted.is_set()
    assert outcome['accepted'] is False
    assert outcome['outcome_known'] is True
    assert outcome['error'] == 'late goal response was rejected'
    handle.cancel_goal_async.assert_not_called()


def test_result_registration_exception_retains_handle_and_blocks_output(planner_module):
    node = make_node(planner_module)
    node._active.update(status='starting', goal_handle=None, goal_response_pending=True)
    handle = Mock(accepted=True)
    handle.get_result_async.side_effect = RuntimeError('result registration failed')
    accepted = threading.Event()
    outcome = {}
    node._on_goal_response('nav-1', 1, result_future_object(handle), accepted, outcome)
    assert accepted.is_set()
    assert outcome['outcome_known'] is False
    assert node._active['status'] == 'cancel_unconfirmed'
    assert node._active['goal_handle'] is handle
    assert last_proposal(node)['nav_status'] == 'planning'


def test_unsafe_shadow_velocity_does_not_retire_goal(planner_module):
    node = make_node(planner_module)
    node.get_logger = Mock(return_value=Mock())
    node._control_odom_max_age_sec = 1
    node._control_odom_source_max_age_sec = 1
    node._active['effective_speed_limit'] = 1.0
    handle = node._active['goal_handle']
    message = types.SimpleNamespace(
        linear=types.SimpleNamespace(x=float('nan'), y=0),
        angular=types.SimpleNamespace(z=0),
    )
    original_blocker = planner_module.control_odom_motion_blocker
    original_apply = planner_module.apply_motion_limits
    planner_module.control_odom_motion_blocker = lambda *args, **kwargs: None
    def reject(*args, **kwargs):
        raise planner_module.ProtocolError('non_finite_velocity', 'velocity must be finite')
    planner_module.apply_motion_limits = reject
    try:
        node._on_shadow_velocity(message)
    finally:
        planner_module.control_odom_motion_blocker = original_blocker
        planner_module.apply_motion_limits = original_apply
    assert node._active['status'] == 'cancel_unconfirmed'
    assert node._active['error_code'] == 'unsafe_shadow_velocity'
    assert node._active['goal_handle'] is handle
    assert last_proposal(node)['nav_status'] == 'planning'
    with pytest.raises(planner_module.CommandError):
        node._resume('nav-1')
    def cancel():
        node._on_result('nav-1', 1, result_future(5))
        return result_future_object(types.SimpleNamespace(goals_canceling=[object()]))
    handle.cancel_goal_async.side_effect = cancel
    assert node._stop('nav-1')['terminal_confirmed'] is True


@pytest.mark.parametrize('status', ['arrived', 'aborted', 'cancel_unconfirmed'])
@pytest.mark.parametrize('action', ['start', 'resume'])
def test_early_result_never_returns_false_navigating(planner_module, status, action):
    node = make_node(planner_module)
    node._max_shadow_speed = 1.0
    node._controller_speed_limit_topic = '/speed_limit'
    if action == 'start':
        node._active = None
    else:
        node._active['status'] = 'paused'
    def send():
        node._active['status'] = status
        if status == 'cancel_unconfirmed':
            node._active['goal_handle'] = Mock()
        else:
            node._active['goal_handle'] = None
    node._send_active_goal.side_effect = send
    def invoke():
        if action == 'start':
            return node._navigate_to_pose('nav-2', {'x': 0, 'y': 0, 'yaw': 0, 'speed': 1, 'mode': 0}, planner_module.MotionLimits())
        return node._resume('nav-1')
    if status == 'cancel_unconfirmed':
        with pytest.raises(planner_module.CommandError) as exc:
            invoke()
        assert exc.value.outcome_known is False
    else:
        assert invoke()['status'] == status


def test_speed_limit_failure_before_send_has_no_pending_goal(planner_module):
    node = make_node(planner_module)
    node._active.update(target_pose={'x': 0, 'y': 0, 'yaw': 0}, effective_speed_limit=0.5)
    node._validate_goal_cell = Mock(return_value={})
    node._action_client = Mock()
    node._action_client.wait_for_server.return_value = True
    node._publish_controller_speed_limit = Mock(side_effect=planner_module.CommandError('speed_limit_unavailable', 'not subscribed'))
    with pytest.raises(planner_module.CommandError) as exc:
        planner_module.PlannerCommandNode._send_active_goal(node)
    assert exc.value.outcome_known is True
    assert node._active['goal_response_pending'] is False
    node._action_client.send_goal_async.assert_not_called()


def test_pending_periodic_publish_cannot_follow_cancel_zero(planner_module):
    node = make_node(planner_module)
    node._latest_proposal_candidate = {
        'nav_id': 'nav-1', 'attempt': 1, 'navigation_status': 'navigating',
        'velocity': planner_module.Velocity(x=0.5), 'reason': None,
        'received_monotonic_ns': time.monotonic_ns(),
    }
    node._control_odom_max_age_sec = 1
    node._control_odom_source_max_age_sec = 1
    checking = threading.Event()
    release = threading.Event()
    def blocker(*args, **kwargs):
        checking.set()
        assert release.wait(2)
        return None
    original = planner_module.control_odom_motion_blocker
    planner_module.control_odom_motion_blocker = blocker
    thread = threading.Thread(target=node._publish_latest_motion_sequence, daemon=True)
    try:
        thread.start()
        assert checking.wait(2)
        node._on_result('nav-1', 1, result_future(0))
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert node._proposal_pub.publish.call_count == 1
        assert last_proposal(node)['velocity'] == {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    finally:
        release.set()
        thread.join(2)
        planner_module.control_odom_motion_blocker = original


def test_inflight_nonzero_publish_finishes_before_cancel_zero(planner_module):
    node = make_node(planner_module)
    node._latest_proposal_candidate = {
        'nav_id': 'nav-1', 'attempt': 1, 'navigation_status': 'navigating',
        'velocity': planner_module.Velocity(x=0.5), 'reason': None,
        'received_monotonic_ns': time.monotonic_ns(),
    }
    node._control_odom_max_age_sec = 1
    node._control_odom_source_max_age_sec = 1
    entered = threading.Event()
    release = threading.Event()
    outputs = []
    def publish(message):
        payload = json.loads(message.data)
        if payload['velocity']['x'] != 0:
            entered.set()
            assert release.wait(2)
        outputs.append(payload)
    node._proposal_pub.publish.side_effect = publish
    original = planner_module.control_odom_motion_blocker
    planner_module.control_odom_motion_blocker = lambda *args, **kwargs: None
    publisher = threading.Thread(target=node._publish_latest_motion_sequence, daemon=True)
    result = threading.Thread(target=lambda: node._on_result('nav-1', 1, result_future(0)), daemon=True)
    try:
        publisher.start()
        assert entered.wait(2)
        result.start()
        release.set()
        publisher.join(2)
        result.join(2)
        assert not publisher.is_alive() and not result.is_alive()
        assert [payload['velocity']['x'] for payload in outputs] == [0.5, 0.0]
        assert [payload['sequence'] for payload in outputs] == [1, 2]
        assert outputs[-1]['nav_status'] == 'planning'
    finally:
        release.set()
        publisher.join(2)
        result.join(2)
        planner_module.control_odom_motion_blocker = original


def shadow_twist(x):
    return types.SimpleNamespace(
        linear=types.SimpleNamespace(x=x, y=0.0),
        angular=types.SimpleNamespace(z=0.0),
    )


def configure_shadow_node(node):
    node._control_odom_max_age_sec = 1
    node._control_odom_source_max_age_sec = 1
    node._active['effective_speed_limit'] = 1.0


def test_new_safety_zero_fences_inflight_periodic_candidate(planner_module, monkeypatch):
    node = make_node(planner_module)
    configure_shadow_node(node)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)
    node._on_shadow_velocity(shadow_twist(0.5))
    entered = threading.Event()
    release = threading.Event()

    def readiness():
        if threading.current_thread().name == 'periodic':
            entered.set()
            assert release.wait(2)
        return {}

    node._readiness = readiness
    periodic = threading.Thread(target=node._publish_latest_motion_sequence, name='periodic', daemon=True)
    try:
        periodic.start()
        assert entered.wait(2)
        node._on_shadow_velocity(shadow_twist(0.0))
        assert node._proposal_pub.publish.call_count == 1
        release.set()
        periodic.join(2)
        assert not periodic.is_alive()
        assert node._proposal_pub.publish.call_count == 1
        assert last_proposal(node)['velocity']['x'] == 0.0
    finally:
        release.set()
        periodic.join(2)


@pytest.mark.parametrize('new_stamp', [1_100_000_000, 1_000_000_000])
def test_older_callback_cannot_overwrite_newer_safety_zero(planner_module, monkeypatch, new_stamp):
    node = make_node(planner_module)
    configure_shadow_node(node)
    entered = threading.Event()
    release = threading.Event()
    stamps = {'old': 1_000_000_000, 'new': new_stamp}
    monkeypatch.setattr(planner_module.time, 'monotonic_ns', lambda: stamps[threading.current_thread().name])
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)

    def readiness():
        if threading.current_thread().name == 'old':
            entered.set()
            assert release.wait(2)
        return {}

    node._readiness = readiness
    old = threading.Thread(target=lambda: node._on_shadow_velocity(shadow_twist(0.5)), name='old', daemon=True)
    new = threading.Thread(target=lambda: node._on_shadow_velocity(shadow_twist(0.0)), name='new', daemon=True)
    try:
        old.start()
        assert entered.wait(2)
        new.start()
        new.join(2)
        assert not new.is_alive()
        release.set()
        old.join(2)
        assert not old.is_alive()
        assert node._latest_proposal_candidate['velocity'] == planner_module.Velocity.zero()
        assert node._latest_proposal_candidate['received_monotonic_ns'] == stamps['new']
        assert node._proposal_pub.publish.call_count == 1
        assert last_proposal(node)['velocity']['x'] == 0.0
    finally:
        release.set()
        old.join(2)
        new.join(2)


def test_equal_timestamp_safety_zero_supersedes_nonzero(planner_module, monkeypatch):
    node = make_node(planner_module)
    configure_shadow_node(node)
    monkeypatch.setattr(planner_module.time, 'monotonic_ns', lambda: 1_000_000_000)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)
    node._on_shadow_velocity(shadow_twist(0.5))
    node._on_shadow_velocity(shadow_twist(0.0))
    assert node._latest_proposal_candidate['velocity'] == planner_module.Velocity.zero()
    assert node._proposal_pub.publish.call_count == 1
    assert last_proposal(node)['velocity']['x'] == 0.0


@pytest.mark.parametrize('now_ns, expected_x', [
    (1_250_000_000, 0.5),
    (1_250_000_001, 0.0),
    (999_999_999, 0.0),
])
def test_final_locked_ttl_check_uses_original_entry_time(planner_module, monkeypatch, now_ns, expected_x):
    node = make_node(planner_module)
    configure_shadow_node(node)
    clock = {'now': 1_000_000_000}
    monkeypatch.setattr(planner_module.time, 'monotonic_ns', lambda: clock['now'])
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)
    node._on_shadow_velocity(shadow_twist(0.5))
    assert node._latest_proposal_candidate['received_monotonic_ns'] == 1_000_000_000

    # The sample can expire while readiness is being evaluated, after an
    # earlier TTL check would have passed.
    def readiness():
        clock['now'] = now_ns
        return {}

    node._readiness = readiness
    node._publish_latest_motion_sequence()
    assert last_proposal(node)['velocity']['x'] == expected_x
    if expected_x == 0.0:
        assert last_proposal(node)['reason'] == 'shadow_velocity_stale'


def test_stale_candidate_publishes_zero_without_readiness(planner_module, monkeypatch):
    node = make_node(planner_module)
    configure_shadow_node(node)
    clock = {'now': 1_000_000_000}
    monkeypatch.setattr(planner_module.time, 'monotonic_ns', lambda: clock['now'])
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)
    node._on_shadow_velocity(shadow_twist(0.5))
    clock['now'] = 1_250_000_001
    node._readiness = Mock(side_effect=RuntimeError('readiness unavailable'))

    node._publish_latest_motion_sequence()

    node._readiness.assert_not_called()
    assert last_proposal(node)['velocity']['x'] == 0.0
    assert last_proposal(node)['reason'] == 'shadow_velocity_stale'


def test_final_ttl_check_runs_after_waiting_for_lock(planner_module, monkeypatch):
    node = make_node(planner_module)
    configure_shadow_node(node)
    clock = {'now': 1_000_000_000}
    checked = threading.Event()
    proceed = threading.Event()
    monkeypatch.setattr(planner_module.time, 'monotonic_ns', lambda: clock['now'])

    def blocker(*args, **kwargs):
        checked.set()
        assert proceed.wait(2)
        return None

    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)
    node._on_shadow_velocity(shadow_twist(0.5))
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', blocker)
    periodic = threading.Thread(target=node._publish_latest_motion_sequence, daemon=True)
    try:
        periodic.start()
        assert checked.wait(2)
        with node._lock:
            proceed.set()
            clock['now'] = 1_300_000_000
        periodic.join(2)
        assert not periodic.is_alive()
        assert last_proposal(node)['velocity']['x'] == 0.0
        assert last_proposal(node)['reason'] == 'shadow_velocity_stale'
    finally:
        proceed.set()
        periodic.join(2)


def test_callback_delay_does_not_refresh_candidate_ttl(planner_module, monkeypatch):
    node = make_node(planner_module)
    configure_shadow_node(node)
    clock = {'now': 1_000_000_000}
    monkeypatch.setattr(planner_module.time, 'monotonic_ns', lambda: clock['now'])
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *args, **kwargs: None)

    def readiness():
        clock['now'] = 1_300_000_000
        return {}

    node._readiness = readiness
    node._on_shadow_velocity(shadow_twist(0.5))
    assert node._latest_proposal_candidate['received_monotonic_ns'] == 1_000_000_000
    node._publish_latest_motion_sequence()
    assert last_proposal(node)['velocity']['x'] == 0.0
    assert last_proposal(node)['reason'] == 'shadow_velocity_stale'
