"""Real visual-session recovery while the external motion stream stays alive."""
import json
import math
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_odometry as card_module
from plugins import rgbd_odometry_runtime as runtime
from test_external_odometry_fusion import packet
from test_rgbd_fusion import images as rgbd_images, source
from test_rgbd_native_external_evidence import landmark_messages
from test_rgbd_odometry_contract import declarations
from test_rgbd_odometry_recovery import assisted_runtime, images
from test_rgbd_odometry_status import sessions  # noqa: F401
from vision_stubs import _FakeExecutor


@pytest.fixture
def external_camera(sessions, monkeypatch, request):
    _, now, ticks = sessions
    assisted_runtime(monkeypatch)
    seconds = [10.]
    monkeypatch.setattr(time, 'monotonic', lambda: seconds[0])
    # Exercise the real session with its existing optional geometry and hint
    # paths disabled, so native visual evidence alone determines this boundary.
    def session(calibration, *args, **kwargs):
        return runtime.RgbdSession(dict(calibration, depth_geometry=False,
                                       inertial_association=False), *args, **kwargs)
    monkeypatch.setattr(card_module, 'RgbdSession', session)
    camera = declarations()
    camera['/rgb']['vendor']['body_from_camera'] = np.eye(4).tolist()
    card = card_module.RgbdOdometryPlugin({}, _FakeExecutor())
    bindings = {'/imu': 'sensor/imu'}
    if getattr(request, 'param', True):
        bindings['/external'] = 'state/odom'
    result = card.dispatch('rgbd_odometry', dict(action='start', camera_info=camera,
        input_topics=['/rgb', '/depth', '/imu', '/external'],
        input_bindings=bindings,
        external_motion_learning=False))
    assert result['state'] == 'running', result
    try:
        yield card, now, ticks, seconds, rgbd_images()
    finally:
        for current, *_ in card.instances.values():
            current.process.returncode = 0
        card.shutdown()


def imu_until(session, begin, end, now, ticks, *, rate=0.):
    for stamp in range(begin, end+1, 5_000_000):
        now[0], ticks[0] = stamp+1_000_000, stamp
        session._imu(NS(data=json.dumps(source(stamp,
            camera_frame_id=session.fusion.frame, clock_id=session.fusion.clock,
            camera_from_accel_translation=[0., 0., 0.],
            camera_from_gyro_translation=[0., 0., 0.],
            angular_velocity=[0., 0., rate]))))


def observe(session, stamp, rgbd, now, ticks, *, yaw=0., x=0., weak=False, rotation_variance=None):
    now[0], ticks[0] = stamp+10_000_000, stamp+10_000_000
    images(session, stamp)
    for sample in session.observations.add_frame(stamp, *rgbd, now[0]):
        session._publish(sample)
    pose, info = landmark_messages(stamp, session.fusion.frame, .05 if weak else .001)
    pose.header.frame_id = session.segment+'_odom'
    pose.pose.pose.position.x = x
    pose.pose.pose.orientation.z = math.sin(yaw/2)
    pose.pose.pose.orientation.w = math.cos(yaw/2)
    if weak:
        pose.pose.covariance[35] = .10
    if rotation_variance is not None:
        pose.pose.covariance[35] = rotation_variance
    now[0] = stamp+20_000_000
    session._result('pose', pose)
    session._result('info', info)


def motion(stream, stamp, *, yaw=0.):
    value = packet(stamp/1e9-1., 100.)
    value['pose']['quaternion'] = [0., 0., math.sin(yaw/2), math.cos(yaw/2)]
    stream._receive('/external', NS(data=json.dumps(value)))
    stream._apply_ready(stream._take_ready())


def warm(camera, *, external=True):
    card, now, ticks, _, rgbd = camera
    session = card.instances['default'][0]
    stream = card.external_streams['default']
    for index in range(8):
        stamp = 1_000_000_000+index*100_000_000
        imu_until(session, stamp if index == 0 else stamp-95_000_000, stamp, now, ticks)
        if external:
            motion(stream, stamp)
        observe(session, stamp, rgbd, now, ticks)
        if not stream.events.empty():
            stream._apply_ready(stream._take_ready())
    assert session.fusion.state == 'tracking'
    assert session.observations.external_gate.state == 'tracking'
    return session, stream, stamp


def turn(camera, session, stream, anchor, *, weak_yaw=10., count=8, external=True,
         rotation_variance=None):
    _, now, ticks, _, rgbd = camera
    for index in range(count):
        stamp = anchor+(index+1)*100_000_000
        rate = -math.pi/2 if index >= 3 else 0.
        imu_until(session, stamp-95_000_000, stamp, now, ticks, rate=rate)
        yaw = math.radians(-9.*max(0, index-2))
        if external:
            motion(stream, stamp, yaw=yaw)
        observe(session, stamp, rgbd, now, ticks,
                yaw=math.radians(weak_yaw) if index == 3 else yaw, weak=index == 3,
                rotation_variance=rotation_variance if index == 3 else None)
        if index == 3 and rotation_variance == .15:
            assert session.observations.external_gate.state == 'recovering'
            assert session.info()['status'] != 'lost'
        if not stream.events.empty():
            stream._apply_ready(stream._take_ready())
    return stamp


def test_native_terminal_loss_restarts_visual_session_and_keeps_external_origin(external_camera):
    card, now, ticks, seconds, rgbd = external_camera
    old, stream, anchor = warm(external_camera)
    public_segment = stream.fuser.segment
    stamp = turn(external_camera, old, stream, anchor)
    assert old.observations.external_gate.state == 'lost'
    assert old.info()['status'] == 'lost'
    assert old.gate.loss_kind == 'visual'
    public = card.dispatch('rgbd_odometry', {'action': 'info'})
    assert public['status'] == 'tracking' and public['visual_status'] == 'lost'
    assert public['segment_id'] == public_segment

    old.process.returncode = 0
    card._recover()
    seconds[0] += .79
    card._recover()
    assert card.instances['default'][0] is old
    seconds[0] += .02
    images(old, now[0])
    card._recover()
    replacement = card.instances['default'][0]
    assert replacement is not old and replacement.segment != old.segment
    assert replacement.external_stream is stream and stream.fuser.segment == public_segment
    assert old.node.destroyed

    # New native coordinates must rejoin through the existing fusion stream,
    # not revive the old rejected visual gauge or reset the body origin.
    yaw = math.radians(-45.)
    for index in range(1, 10):
        current = stamp+index*100_000_000
        imu_until(replacement, current-95_000_000, current, now, ticks)
        motion(stream, current, yaw=yaw)
        observe(replacement, current, rgbd, now, ticks, x=100.)
        if not stream.events.empty():
            stream._apply_ready(stream._take_ready())
    assert replacement.info()['status'] == 'tracking'
    assert stream.fuser.raw_vision[current][0] == replacement.segment
    output = json.loads(stream.publisher.messages[-1])
    assert output['vendor']['motus_slam']['segment_id'] == public_segment
    assert output['pose']['position'] == pytest.approx([0., 0., 0.], abs=1e-8)


@pytest.mark.parametrize('rotation_variance', [.10, .15], ids=['supported_weak', 'recovering'])
def test_weak_or_recovering_native_observation_does_not_restart(external_camera, rotation_variance):
    card, now, _, seconds, _ = external_camera
    session, stream, anchor = warm(external_camera)
    turn(external_camera, session, stream, anchor, weak_yaw=-9., rotation_variance=rotation_variance)
    assert session.observations.external_gate.state == 'tracking'
    assert session.info()['status'] == 'tracking'
    seconds[0] += 1.
    images(session, now[0])
    card._recover()
    assert card.instances['default'][0] is session
    assert card.recovery['default']['restarts'] == 0


@pytest.mark.parametrize('external_camera', [False], indirect=True)
def test_inactive_external_candidate_does_not_retire_healthy_camera(external_camera):
    card, now, _, seconds, _ = external_camera
    session, stream, anchor = warm(external_camera, external=False)
    turn(external_camera, session, stream, anchor, external=False)
    assert session.observations.external_gate.state == 'lost'
    assert stream.active is False and stream.selected is None
    assert session.info()['status'] == 'tracking'
    seconds[0] += 1.
    images(session, now[0])
    card._recover()
    assert card.instances['default'][0] is session
    assert card.recovery['default']['restarts'] == 0


@pytest.mark.parametrize('external_camera', [False], indirect=True)
def test_inactive_native_failure_does_not_split_later_healthy_backlog(external_camera):
    _, now, ticks, _, rgbd = external_camera
    session, stream, anchor = warm(external_camera, external=False)
    last = turn(external_camera, session, stream, anchor, external=False)
    assert session.observations.external_gate.state == 'lost'
    assert stream.active is False and session.info()['status'] == 'tracking'

    stamps = [last+index*50_000_000 for index in range(1, 5)]
    for stamp in stamps:
        observe(session, stamp, rgbd, now, ticks, yaw=math.radians(-45.))
    assert list(session.observations.frames) == stamps
    imu_until(session, last+5_000_000, stamps[-1], now, ticks)
    now[0] = stamps[-1]+20_000_000
    ready = session.observations.drain(now[0])
    assert [sample['vendor']['motus_slam']['stamp_ns'] for sample in ready] == stamps
    assert not session.observations.frames
    published = len(session.publisher.messages)
    for sample in ready:
        session._publish(sample)
    outputs = [json.loads(message) for message in session.publisher.messages[published:]]
    assert [sample['vendor']['motus_slam']['stamp_ns'] for sample in outputs] == stamps
    assert all(sample['pose'] is not None for sample in outputs)
    assert session.info()['status'] == 'tracking'


def test_native_failure_waits_for_earlier_queued_captures(external_camera):
    _, now, ticks, _, rgbd = external_camera
    session, stream, anchor = warm(external_camera)
    captures = session.observations
    stamps = [anchor+i*100_000_000 for i in range(1, 5)]
    for index, (stamp, yaw) in enumerate(zip(stamps, (0., 10., -18., -27.))):
        observe(session, stamp, rgbd, now, ticks, yaw=math.radians(yaw), weak=index == 1)
    assert captures.external_gate.state == 'lost'
    assert list(captures.frames) == stamps
    assert session.info()['status'] == 'tracking'

    imu_until(session, anchor+5_000_000, stamps[0], now, ticks)
    imu_until(session, stamps[0]+5_000_000, stamps[-1], now, ticks, rate=-math.pi/2)
    now[0] = stamps[-1]+20_000_000
    prefix = captures.drain(now[0])
    assert [sample['vendor']['motus_slam']['stamp_ns'] for sample in prefix] == stamps[:3]
    assert prefix[0]['pose'] is not None
    for sample in prefix:
        session._publish(sample)
    assert session.info()['status'] != 'lost'
    assert stream.visual_state != 'lost'

    final, = captures.drain(now[0])
    session._publish(final)
    assert session.info()['status'] == 'lost'
    assert stream.visual_state == 'lost'


def test_unanchored_native_initialization_can_retry_without_restarting_session(external_camera):
    card, now, ticks, _, rgbd = external_camera
    session = card.instances['default'][0]
    stream = card.external_streams['default']
    initial_gate = None
    for index in range(28):
        stamp = 1_000_000_000+index*100_000_000
        imu_until(session, stamp if index == 0 else stamp-95_000_000, stamp, now, ticks)
        motion(stream, stamp)
        observe(session, stamp, rgbd, now, ticks, rotation_variance=.15 if index < 20 else None)
        if initial_gate is None:
            initial_gate = session.observations.external_gate
        if index < 20:
            assert session.observations.external_gate.last_pose is None
        assert session.info()['status'] != 'lost'
        if not stream.events.empty():
            stream._apply_ready(stream._take_ready())
    assert session.observations.external_gate is not initial_gate
    assert session.observations.external_gate.state == 'tracking'
    assert session.info()['status'] == 'tracking'
    card._recover()
    assert card.instances['default'][0] is session


def test_native_terminal_before_imu_initialization_reaches_supervisor(external_camera):
    card, now, ticks, _, rgbd = external_camera
    session = card.instances['default'][0]
    stream = card.external_streams['default']
    segment = stream.fuser.segment
    for index, degrees in enumerate((0., 0., 0., 10., -18., -27.)):
        stamp = 1_000_000_000+index*100_000_000
        now[0], ticks[0] = stamp+10_000_000, stamp+10_000_000
        motion(stream, stamp)
        observe(session, stamp, rgbd, now, ticks, yaw=math.radians(degrees), weak=index == 3)
        if not stream.events.empty():
            stream._apply_ready(stream._take_ready())
    assert session.fusion.filter is None and not session.fusion.samples
    assert session.observations.external_gate.last_pose is not None
    assert session.observations.external_gate.state == 'lost'
    assert session.info()['status'] == 'lost'
    assert session.gate.loss_kind == 'visual'
    info = card.dispatch('rgbd_odometry', {'action': 'info'})
    assert info['status'] == 'tracking' and info['visual_status'] == 'lost'
    assert info['segment_id'] == segment
