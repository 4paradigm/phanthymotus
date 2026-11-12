"""Compact card/config and real session JSON publication without native ROS."""
import json
import sys
from types import SimpleNamespace as NS

import pytest

from plugins import rgbd_odometry as card_module
from plugins import rgbd_odometry_runtime as runtime
from plugins.rgbd_odometry_contract import make_sample, validate_start_inputs
from test_rgbd_odometry_contract import declarations
from vision_stubs import _FakeExecutor


@pytest.fixture
def sessions(monkeypatch):
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'String', lambda **kw: NS(**kw))
    monkeypatch.setattr(sys.modules['rclpy.qos'], 'qos_profile_sensor_data', object(), raising=False)
    monkeypatch.setitem(sys.modules, 'rtabmap_msgs.msg', NS(RGBDImage=object, OdomInfo=object))
    monkeypatch.setitem(sys.modules, 'nav_msgs.msg', NS(Odometry=object))
    monkeypatch.setattr(runtime, 'require_runtime', lambda: '/offline/estimator')
    monkeypatch.setattr(card_module, 'require_runtime', lambda: '/offline/estimator')
    processes = []
    def process(*args, **kwargs):
        p = NS(returncode=None, wait=lambda **kw: 0)
        p.poll = lambda: p.returncode
        processes.append(p)
        return p
    monkeypatch.setattr(runtime.subprocess, 'Popen', process)
    monkeypatch.setattr(runtime.threading, 'Thread', lambda **kw: NS(
        start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    now, ticks = [1_010_000_000], [0]
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: now[0])
    monkeypatch.setattr(runtime.time, 'monotonic_ns', lambda: ticks[0])
    created = []
    def create(topic='/rgbd_odometry/i_6f6e65/state/odom'):
        calibration = validate_start_inputs(declarations(), ['/rgb', '/depth'])
        session = runtime.RgbdSession(calibration, _FakeExecutor(), topic)
        created.append(session)
        return session
    yield create, now, ticks
    for p in processes: p.returncode = 0
    for session in created:
        if not session.node.destroyed: session.close()


def pose(position):
    return dict(position=position, quaternion=[0., 0., 0., 1.], frame_id='run_odom',
                child_frame_id='camera_optical', covariance=[.001 if i%7==0 else 0. for i in range(36)])


def tracking(session, position, stamp=1_000_000_000):
    session._publish(make_sample('tracking', stamp, session.segment, 'camera-clock', pose(position)))


def status(session):
    return json.loads(session.status_publisher.messages[-1])


def test_compact_schema_keeps_configuration_and_original_odom_link(sessions):
    create, _, _ = sessions
    card = card_module.RgbdOdometryPlugin({}, _FakeExecutor())
    tool = card.get_tools()[0]
    assert set(tool['inputSchema']['properties']) == {'action'}
    assert tool['inputSchema']['properties']['action']['enum'] == ['start', 'stop', 'info']
    settings = dict(imu_assisted=False, short_term_compensation=False, depth_geometry=False,
                    visual_frontend='f2f_optical_flow', max_features=200)
    assert set(tool['configSchema']['properties']) == set(settings)
    assert all(p['scope'] == 'instance' for p in tool['configSchema']['properties'].values())
    result = card.dispatch('rgbd_odometry', dict(action='config', instance_id='one', **settings))
    assert result['state'] == 'configured'
    result = card.dispatch('rgbd_odometry', dict(action='start', instance_id='one',
        input_topics=['/rgb', '/depth'], camera_info=declarations()))
    session = card.instances['one'][0]
    session.process.returncode = 0
    try:
        assert result['state'] == 'running' and result['config'] == settings
        assert session.calibration['visual_frontend'] == 'f2f_optical_flow'
        assert result['topic_out'] == [
            dict(topic='/rgbd_odometry/i_6f6e65/state/odom', format='state/odom'),
            dict(topic='/rgbd_odometry/i_6f6e65/status', format='data/json')]
        other = create('/rgbd_odometry/i_74776f/state/odom')
        assert session.status_publisher.topic != other.status_publisher.topic
        assert {p.topic for p in session.node.publishers} >= {
            '/rgbd_odometry/i_6f6e65/state/odom', '/rgbd_odometry/i_6f6e65/status'}
        assert card.dispatch('rgbd_odometry', dict(action='info', instance_id='one'))['topic_out'] == result['topic_out']
    finally:
        card.shutdown()


def test_bumi_shaped_summary_uses_camera_pose_and_segment_displacement(sessions):
    create, now, ticks = sessions
    session = create()
    tracking(session, [1., 2., 3.])
    assert status(session) == dict(status='tracking', turn_angle_deg=None, displacement_m=0.,
        position_m=[1., 2., 3.], pose_reference='camera_optical', age_ms=10., reason=None)
    ticks[0] = 200_000_000
    now[0] = 1_210_000_000
    tracking(session, [4., 6., 3.], 1_200_000_000)
    assert status(session)['displacement_m'] == 5.
    assert status(session)['position_m'] == [4., 6., 3.]
    full = json.loads(session.publisher.messages[-1])
    assert full['schema'] == 'motus.odom/1' and full['pose']['position'] == [4., 6., 3.]
    fresh = create('/rgbd_odometry/i_74776f/state/odom')
    tracking(fresh, [4., 6., 3.], 1_200_000_000)
    assert status(fresh)['displacement_m'] == 0.


@pytest.mark.parametrize('state', ['initializing', 'lost', 'coasting', 'recovering'])
def test_nontracking_summary_never_exposes_held_pose_or_inertial_prediction(sessions, state):
    create, _, ticks = sessions
    session = create()
    tracking(session, [1., 2., 3.])
    ticks[0] = 1  # a transition must bypass the 200ms cap
    sample = make_sample(state, 1_000_000_001, session.segment, 'camera-clock', reason='waiting')
    sample['vendor']['fusion'] = dict(prediction=pose([99., 99., 99.]))
    session._publish(sample)
    assert status(session) == dict(status=state, turn_angle_deg=None, displacement_m=None,
        position_m=None, pose_reference='camera_optical', age_ms=None, reason='waiting')


def test_status_throttle_does_not_throttle_odometry_or_hide_reason_changes(sessions):
    create, _, ticks = sessions
    session = create()
    tracking(session, [0., 0., 0.])
    ticks[0] = 199_999_999
    tracking(session, [1., 0., 0.])
    assert len(session.publisher.messages) == 2
    assert len(session.status_publisher.messages) == 1
    ticks[0] = 200_000_000
    tracking(session, [1., 0., 0.])
    assert len(session.status_publisher.messages) == 2
    for reason in ('stale', 'IMU lost'):
        sample = make_sample('lost', 1_000_000_000, session.segment, 'camera-clock', reason=reason)
        session._publish(sample)
        assert status(session)['reason'] == reason
    assert len(session.status_publisher.messages) == 4


def test_watchdog_and_stop_replace_tracking_status_with_no_position(sessions):
    create, now, _ = sessions
    session = create()
    tracking(session, [1., 2., 3.])
    session.gate.state = 'tracking'
    session.gate.last_input = session.gate.last_stamp = 1_000_000_000
    now[0] = 2_600_000_001
    session._watchdog()
    assert status(session)['status'] == 'lost'
    assert status(session)['position_m'] is None
    assert status(session)['reason']
    session.process.returncode = 0
    session.close()
    assert status(session)['status'] == 'idle'
    assert status(session)['position_m'] is None
    assert session.node.destroyed


def test_status_transport_failure_during_stop_still_releases_session_resources(sessions, caplog):
    create, _, _ = sessions
    session = create()
    session.process.returncode = 0
    def fail(message): raise RuntimeError('status transport closed')
    session.status_publisher.publish = fail
    session.close()
    assert 'status transport closed' in caplog.text
    assert session.node.destroyed
    assert session.log.closed
    from pathlib import Path
    assert not Path(session.directory.name).exists()


def test_status_transport_failure_cannot_retire_valid_odometry(sessions, caplog):
    create, _, _ = sessions
    session = create()
    session.gate.state = 'tracking'
    def fail(message): raise RuntimeError('status transport closed')
    session.status_publisher.publish = fail
    tracking(session, [1., 2., 3.])
    assert json.loads(session.publisher.messages[-1])['pose']['position'] == [1., 2., 3.]
    assert session.gate.state == 'tracking' and not session.stop_event.is_set()
    assert 'status transport closed' in caplog.text


def test_failed_final_status_does_not_leave_a_disposed_card_instance(sessions):
    card = card_module.RgbdOdometryPlugin({}, _FakeExecutor())
    args = dict(action='start', instance_id='one', input_topics=['/rgb', '/depth'],
                camera_info=declarations())
    assert card.dispatch('rgbd_odometry', args)['state'] == 'running'
    original = card.instances['one'][0]
    original.process.returncode = 0
    publish = original.status_publisher.publish
    def fail(message): raise RuntimeError('status transport closed')
    original.status_publisher.publish = fail
    try:
        stopped = card.dispatch('rgbd_odometry', dict(action='stop', instance_id='one'))
        assert stopped['state'] == 'idle' and 'one' not in card.instances
        assert original.node.destroyed
        restarted = card.dispatch('rgbd_odometry', args)
        current = card.instances['one'][0]
        current.process.returncode = 0
        assert restarted['state'] == 'running' and current is not original
        assert current.segment != original.segment and not current.stop_event.is_set()
    finally:
        original.status_publisher.publish = publish
        card.shutdown()
