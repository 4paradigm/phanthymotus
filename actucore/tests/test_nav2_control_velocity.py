"""Nav2 standalone control stream: wire, timestamps, safety and races (no ROS)."""
import json
import importlib.util
import sys
from pathlib import Path
import threading
import time
import types
from unittest.mock import Mock

import pytest

from tests.test_nav2_planner_cancel_characterization import (
    planner_module, make_node, configure_shadow_node, shadow_twist, result_future,
)
from plugins.control_stream import message as shared_message, negotiate


def test_launch_wires_json_as_string(monkeypatch):
    class Capture:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
    def module(name, **members):
        item = types.ModuleType(name)
        item.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, item)
    module('ament_index_python.packages', get_package_share_directory=lambda n: '/tmp/' + n)
    module('launch', LaunchDescription=Capture)
    module('launch.actions', DeclareLaunchArgument=Capture, GroupAction=Capture,
           IncludeLaunchDescription=Capture)
    module('launch.launch_description_sources', PythonLaunchDescriptionSource=Capture)
    module('launch.substitutions', LaunchConfiguration=Capture)
    module('launch_ros.actions', Node=Capture, SetRemap=Capture)
    module('launch_ros.parameter_descriptions', ParameterValue=Capture)
    path = Path(__file__).resolve().parents[1] / 'plugins/navigation/runtime/nav2/launch/nav2.launch.py'
    spec = importlib.util.spec_from_file_location('test_control_launch', path)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    items = launch.generate_launch_description().args[0]
    declarations = {item.args[0]: item.kwargs.get('default_value') for item in items if item.args and isinstance(item.args[0], str)}
    assert declarations['control_velocity_topic'] == '/ubuntu/navigation/control_velocity'
    assert declarations['control_interface_json'] == '{}'
    node = next(item for item in items if item.kwargs.get('executable') == 'planner_command_bridge')
    params = node.kwargs['parameters'][0]
    assert params['control_velocity_topic'].args == ('control_velocity_topic',)
    assert params['control_interface_json'].kwargs['value_type'] is str
    assert params['control_interface_json'].args[0].args == ('control_interface_json',)
    assert params['proposal_frequency_hz'] == 5.0 and params['proposal_ttl_ms'] == 250


def descriptor():
    return {
        'control_interface': 'motus.control/1', 'mode': 'twist', 'dof': 6,
        'joint_names': ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
        'limits': {'lower': [-1.0] * 6, 'upper': [1.0] * 6},
        'rate': {'expected_hz': 8, 'max_hz': 20, 'watchdog_ms': 300},
    }


def setup_control(node, module, desc=None):
    desc = desc or {}
    (node._control_descriptor, node._control_rate_hz, node._control_ttl_ms,
     node._control_max_obs_age_ms, node._control_bounds, node._control_floors) = (
        module.control_velocity.configure(json.dumps(desc)))
    node._control_sequence = 0
    node._control_pub = Mock()
    node._sensor_source_max_age_sec = 1.0
    node.get_clock = Mock(return_value=types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=time.time_ns())))
    node.get_parameter = Mock(return_value=types.SimpleNamespace(value=False))
    now = time.time_ns()
    node._last_odom_source_stamp_ns = now - 100_000_000
    node._last_obstacle_source_stamp_ns = now - 200_000_000
    node._last_odom_monotonic = node._last_obstacle_monotonic = time.monotonic()
    node._last_odom_frame_ready = node._last_obstacle_frame_ready = True
    node._sensor_max_age_sec = 0.8
    node.get_logger = Mock(return_value=Mock())
    configure_shadow_node(node)


def latest(node):
    return json.loads(node._control_pub.publish.call_args.args[0].data)


def test_wire_and_rate_match_navi_helpers(planner_module):
    helper = planner_module.control_velocity
    desc = descriptor()
    _, hz, ttl, _, bounds, floors = helper.configure(json.dumps(desc))
    assert hz == negotiate.effective_rate({'control_hz': 10}, desc, 10)
    assert ttl == negotiate.ttl_ms(hz, desc)
    velocity = helper.limit_velocity(planner_module.Velocity(x=.5, yaw=.7), bounds, floors)
    payload = helper.build(seq=1, stamp_ms=1000, obs_stamp_ms=800, ttl_ms=ttl,
                           velocity=velocity, descriptor=desc)
    expected = shared_message.build(seq=1, values=[.5, 0, 0, 0, 0, .7], mode='twist',
                                    dof=6, source='mcp__actucore__navigation',
                                    stamp_ms=1000, obs_stamp_ms=800, ttl_ms=ttl)
    assert {k: payload[k] for k in expected} == expected
    assert payload['control_interface'] == desc
    assert payload['obs_stamp_kind'] == 'earliest_readiness_input_header'
    assert 'control_interface' not in helper.build(seq=2, stamp_ms=1, obs_stamp_ms=1,
                                                    ttl_ms=200, velocity=velocity, descriptor=desc)


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(mode='joint_position'),
    lambda d: d.update(dof=4),
    lambda d: d.update(control_interface='other'),
    lambda d: d['rate'].update(max_hz=5),
])
def test_bad_descriptor_fails_closed(planner_module, mutate):
    desc = descriptor()
    mutate(desc)
    with pytest.raises(ValueError):
        planner_module.control_velocity.configure(json.dumps(desc))


@pytest.mark.parametrize('change', [
    lambda d: d.pop('limits'),
    lambda d: d.pop('dof'),
    lambda d: d['rate'].update(watchdog_ms=20),
    lambda d: d['rate'].update(watchdog_ms=.5),
    lambda d: d['rate'].update(expected_hz='invalid', watchdog_ms='invalid'),
    lambda d: d['rate'].update(expected_hz=0, watchdog_ms=0, max_hz=0),
    lambda d: d.update(limits={'lower': [0, 0]}),
    lambda d: d.update(limits={'lower': ['bad'] * 6, 'upper': [1] * 6}),
    lambda d: d.update(limits={'lower': [1] * 6, 'upper': [2] * 6}),
    lambda d: d.update(limits='invalid'),
])
def test_navi_accepted_descriptors_are_accepted(planner_module, change):
    desc = descriptor()
    change(desc)
    capabilities = {'control_mode': 'twist', 'action_dim': 6, 'control_hz': 10, 'chunk_size': 1}
    assert negotiate.check(capabilities, desc) == []
    _, hz, ttl, _, _, _ = planner_module.control_velocity.configure(json.dumps(desc))
    assert hz == negotiate.effective_rate(capabilities, desc, 10)
    assert ttl == negotiate.ttl_ms(hz, desc)


@pytest.mark.parametrize('field', ['expected_hz', 'watchdog_ms', 'max_obs_age_ms'])
@pytest.mark.parametrize('value', [None, [], {}, 'bad', -1, 0, float('nan'), float('inf'), 1e-320])
def test_malformed_optional_numbers_do_not_crash(planner_module, field, value):
    desc = descriptor()
    desc['rate'][field] = value
    _, hz, ttl, _, _, _ = planner_module.control_velocity.configure(json.dumps(desc))
    assert hz > 0 and isinstance(ttl, int)


def test_minimal_descriptor(planner_module):
    desc = {'control_interface': 'motus.control/1', 'mode': 'twist'}
    _, hz, ttl, _, bounds, floors = planner_module.control_velocity.configure(json.dumps(desc))
    assert (hz, ttl) == (10, 200)
    assert planner_module.control_velocity.limit_velocity(
        planner_module.Velocity(x=.4, yaw=-.3), bounds, floors) == planner_module.Velocity(x=.4, yaw=-.3)


def test_restart_sequence_is_dropped_by_actual_sink(planner_module, monkeypatch):
    driver_root = Path(__file__).resolve().parents[4] / 'phanthymotus-driver'
    if not (driver_root / 'common/control/sink.py').exists():
        pytest.skip('sibling phanthymotus-driver checkout unavailable')
    monkeypatch.syspath_prepend(str(driver_root))
    from common.control import ControlSink, Verdict
    desc = descriptor()
    desc['units'] = {'linear': 'm/s', 'angular': 'rad/s'}
    desc['force_torque'] = None
    sink = ControlSink(desc, lambda values, gripper: None,
                       clock=lambda: 1_000_000, wall_clock=lambda: 1_000_000)
    def payload(seq):
        return planner_module.control_velocity.build(
            seq=seq, stamp_ms=1_000_000, obs_stamp_ms=1_000_000, ttl_ms=200,
            velocity=planner_module.Velocity.zero(), descriptor=desc, stop_fallback=True)
    assert sink.submit(payload(2)).verdict == Verdict.APPLIED
    restarted = sink.submit(payload(1))
    assert restarted.verdict == Verdict.DROPPED
    assert 'not newer' in restarted.reason
    assert sink.submit(payload(3)).verdict == Verdict.APPLIED
    tiny_ttl = payload(4)
    tiny_ttl['ttl_ms'] = 0  # Navi integer TTL for 0 < watchdog_ms < 1.
    assert sink.submit(tiny_ttl).verdict == Verdict.REJECTED
    sink.reset()
    assert sink.submit(payload(1)).verdict == Verdict.APPLIED


def test_invalid_and_incompatible_limits_only_stop_affected_axis(planner_module):
    desc = descriptor()
    desc['limits']['lower'] = ['bad', 1, -1, -1, -1, -1]
    _, _, _, _, bounds, floors = planner_module.control_velocity.configure(json.dumps(desc))
    out = planner_module.control_velocity.limit_velocity(planner_module.Velocity(x=.5, y=.4, yaw=.3), bounds, floors)
    assert out == planner_module.Velocity(y=0, x=0, yaw=.3)


def test_limits_only_reduce_never_amplify(planner_module):
    helper = planner_module.control_velocity
    desc = descriptor()
    desc['limits']['upper'][0] = .3
    desc['limits']['lower'][1] = desc['limits']['upper'][1] = 0
    desc['limits']['min_magnitude'] = [.4, 0, 0, 0, 0, .5]
    _, _, _, _, bounds, floors = helper.configure(json.dumps(desc))
    value = helper.limit_velocity(planner_module.Velocity(x=.8, y=.5, yaw=.2), bounds, floors)
    assert value == planner_module.Velocity.zero()


def test_nonzero_uses_older_input_header(planner_module, monkeypatch):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    node._on_shadow_velocity(shadow_twist(.5))
    node._publish_latest_control_velocity()
    assert latest(node)['values'] == [.5, 0, 0, 0, 0, 0]
    assert latest(node)['obs_stamp_ms'] == node._last_obstacle_source_stamp_ns // 1_000_000
    assert latest(node)['obs_stamp_kind'] == 'earliest_readiness_input_header'
    assert node._proposal_pub.publish.call_count == 0


def test_new_sensor_inputs_do_not_refresh_old_candidate(planner_module, monkeypatch):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    node._on_shadow_velocity(shadow_twist(.5))
    original = node._last_obstacle_source_stamp_ns // 1_000_000
    node._last_odom_source_stamp_ns = node._last_obstacle_source_stamp_ns = time.time_ns()
    node._publish_latest_control_velocity()
    assert latest(node)['obs_stamp_ms'] == original


@pytest.mark.parametrize('condition', ['missing', 'future', 'stale', 'sim', 'unknown_clock', 'clock_error', 'wrong_clock'])
def test_nonzero_fails_closed_when_stamp_unusable(planner_module, monkeypatch, condition):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    if condition == 'missing': node._last_odom_source_stamp_ns = None
    if condition == 'future': node._last_odom_source_stamp_ns = time.time_ns() + 1_000_000_000
    if condition == 'stale': node._last_odom_source_stamp_ns = time.time_ns() - 2_000_000_000
    if condition == 'sim': node.get_parameter.return_value.value = True
    if condition == 'unknown_clock': node.get_parameter.return_value.value = None
    if condition == 'clock_error': node.get_clock.side_effect = RuntimeError('clock unavailable')
    if condition == 'wrong_clock':
        node.get_clock.return_value.now = lambda: types.SimpleNamespace(nanoseconds=1)
    node._on_shadow_velocity(shadow_twist(.5))
    node._publish_latest_control_velocity()
    assert latest(node)['values'] == [0] * 6
    assert latest(node)['obs_stamp_kind'] == 'bridge_stop_generated'


def test_safety_zero_fallback_is_stop_generation_time(planner_module, monkeypatch):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    node._last_odom_source_stamp_ns = None
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    node._on_shadow_velocity(shadow_twist(0))
    payload = latest(node)
    assert payload['values'] == [0] * 6
    assert payload['obs_stamp_ms'] == payload['stamp_ms']
    assert payload['obs_stamp_kind'] == 'bridge_stop_generated'
    assert node._proposal_pub.publish.call_count == 1


def test_expired_candidate_stops_instead_of_republishing(planner_module, monkeypatch):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    node._on_shadow_velocity(shadow_twist(.5))
    node._latest_proposal_candidate['received_monotonic_ns'] -= 1_000_000_000
    node._publish_latest_control_velocity()
    assert latest(node)['values'] == [0] * 6


def test_pending_control_nonzero_cannot_follow_cancel_zero(planner_module, monkeypatch):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    node._on_shadow_velocity(shadow_twist(.5))
    entered, release = threading.Event(), threading.Event()
    def readiness():
        entered.set()
        assert release.wait(2)
        return {}
    node._readiness = readiness
    thread = threading.Thread(target=node._publish_latest_control_velocity, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        node._readiness = Mock(return_value={})
        node._on_result('nav-1', 1, result_future(0))
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert [json.loads(c.args[0].data)['values'] for c in node._control_pub.publish.call_args_list] == [[0] * 6]
    finally:
        release.set()
        thread.join(2)


def inject_control_failure(node, module, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError('injected control failure')
    if failure == 'limit':
        monkeypatch.setattr(module.control_velocity, 'limit_velocity', fail)
    elif failure == 'build':
        monkeypatch.setattr(module.control_velocity, 'build', fail)
    elif failure == 'json':
        # Fail serialization of control messages only, not legacy proposals.
        real_dumps = json.dumps
        def dumps(payload, *args, **kwargs):
            if payload.get('schema') == 'motus.control/1':
                fail()
            return real_dumps(payload, *args, **kwargs)
        monkeypatch.setattr(module, 'json', types.SimpleNamespace(dumps=dumps))
    elif failure == 'clock':
        # Fail the bridge's time reference without replacing the shared time module.
        monkeypatch.setattr(module, 'time', types.SimpleNamespace(
            time_ns=fail, monotonic_ns=time.monotonic_ns,
            monotonic=time.monotonic, time=time.time))
    elif failure == 'publish':
        node._control_pub.publish.side_effect = RuntimeError('injected control failure')


@pytest.mark.parametrize('failure', ['limit', 'build', 'json', 'clock', 'publish'])
@pytest.mark.parametrize('path', ['shadow_zero', 'terminal'])
def test_control_failure_does_not_hide_legacy_zero_or_terminal(
        planner_module, monkeypatch, failure, path):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    inject_control_failure(node, planner_module, monkeypatch, failure)
    if path == 'shadow_zero':
        node._on_shadow_velocity(shadow_twist(0))
    else:
        node._on_result('nav-1', 1, result_future(5))
        assert node._emit.call_args.args[0]['status'] == 'stopped'
    proposal = json.loads(node._proposal_pub.publish.call_args.args[0].data)
    assert proposal['velocity'] == {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    node.get_logger.return_value.error.assert_called_once()
    assert 'control velocity output failed' in node.get_logger.return_value.error.call_args.args[0]


@pytest.mark.parametrize('failure', ['limit', 'build', 'json', 'clock', 'publish'])
def test_timer_control_failure_is_isolated_and_recoverable(planner_module, monkeypatch, failure):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    node._on_shadow_velocity(shadow_twist(.5))
    with monkeypatch.context() as injected:
        inject_control_failure(node, planner_module, injected, failure)
        node._publish_latest_control_velocity()
        node.get_logger.return_value.error.assert_called_once()
    if failure == 'publish':
        node._control_pub.publish.side_effect = None
    node._publish_latest_control_velocity()
    assert latest(node)['values'][0] == .5


def test_control_exception_boundary_does_not_swallow_legacy_publish_errors(planner_module, monkeypatch):
    node = make_node(planner_module)
    setup_control(node, planner_module)
    monkeypatch.setattr(planner_module, 'control_odom_motion_blocker', lambda *a, **kw: None)
    inject_control_failure(node, planner_module, monkeypatch, 'build')
    node._proposal_pub.publish.side_effect = RuntimeError('legacy publish failed')
    with pytest.raises(RuntimeError, match='legacy publish failed'):
        node._on_shadow_velocity(shadow_twist(0))
