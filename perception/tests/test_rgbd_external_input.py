import json
import threading
import time
from threading import Event, Thread
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_external_input
from plugins.external_motion_learning import ExternalMotionLearner
from plugins.rgbd_body_mount import CaptureMounts
from plugins.rgbd_odometry import RgbdOdometryPlugin
from test_external_odometry_fusion import packet, pose_at
from test_pointcloud_slam_stream import stream as slam_stream
from test_rgbd_odometry_contract import declarations
from test_rgbd_odometry_status import sessions  # noqa: F401
from test_rgbd_odometry_recovery import assisted_runtime, images
from vision_stubs import _FakeExecutor


def test_visual_loss_does_not_change_public_fusion_status_or_segment(sessions, monkeypatch):
    executor = _FakeExecutor()
    card = RgbdOdometryPlugin({}, executor)
    camera = declarations()
    camera['/rgb']['vendor']['body_from_camera'] = np.eye(4).tolist()
    card.dispatch('rgbd_odometry', dict(action='start', camera_info=camera,
                  input_topics=['/rgb', '/depth', '/external'],
                  input_bindings={'/external': 'state/odom'}))
    session = card.instances['default'][0]
    stream = card.external_streams['default']
    source = next(sub for sub in stream.node.subscriptions if sub.topic == '/external')
    now = [time.monotonic()]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    try:
        for stamp, position in ((0., 10.), (.2, 10.2)):
            source.callback(NS(data=json.dumps(packet(stamp, position))))
            stream._apply(stream.events.get_nowait())
        first = card.dispatch('rgbd_odometry', dict(action='info'))
        assert first['status'] == 'tracking'
        segment = first['segment_id']
        assert stream.node not in executor.nodes
        session.gate.fault('native visual tracking lost', 'visual')
        source.callback(NS(data=json.dumps(packet(.4, 10.4))))
        stream._apply(stream.events.get_nowait())
        fused = card.dispatch('rgbd_odometry', dict(action='info'))
        assert fused['status'] == 'tracking' and fused['visual_status'] == 'lost'
        assert fused['segment_id'] == segment and fused['visual_segment_id'] == session.segment
        assert fused['last_measurement_ns'] == 1_400_000_000

        session.process.returncode = 0
        images(session, time.time_ns())
        card._recover()
        now[0] += .9
        card._recover()
        replacement = card.instances['default'][0]
        assert replacement is not session and replacement.external_stream is stream
        replacement.process.returncode = 0
        source.callback(NS(data=json.dumps(packet(.6, 10.6))))
        stream._apply(stream.events.get_nowait())
        resumed = card.dispatch('rgbd_odometry', dict(action='info'))
        assert resumed['status'] == 'tracking' and resumed['segment_id'] == segment
        assert resumed['visual_segment_id'] == replacement.segment
        assert resumed['last_measurement_ns'] == 1_600_000_000
        now[0] += 2.
        expired = card.dispatch('rgbd_odometry', dict(action='info'))
        assert expired['status'] == 'lost' and expired['segment_id'] == segment
    finally:
        session.process.returncode = 0
        card.shutdown()


@pytest.mark.parametrize('with_imu', [False, True])
def test_registered_card_publishes_external_body_motion_at_new_stamps(sessions, monkeypatch, with_imu):
    if with_imu:
        assisted_runtime(monkeypatch)
    camera = declarations()
    camera['/rgb']['vendor']['body_from_camera'] = pose_at(1).tolist()
    card = RgbdOdometryPlugin({}, _FakeExecutor())
    imu = ['/sensor'] if with_imu else []
    result = card.dispatch('rgbd_odometry', dict(action='start', camera_info=camera,
                           input_topics=['/rgb', '/depth', '/external', *imu],
                           input_bindings={'/external': 'state/odom', '/sensor': 'sensor/imu'}))
    assert result['state'] == 'running', result
    session = card.instances['default'][0]
    stream = card.external_streams['default']
    try:
        assert result['config']['imu_assisted'] is with_imu
        assert result['topic_in'][-1]['format'] == 'state/odom'
        stream._apply(('visual', 1_000_000_000, 'old', dict(
            position=[11., 0., 0.], quaternion=[0., 0., 0., 1.])))
        source = next(sub for sub in stream.node.subscriptions if sub.topic == '/external')
        for at, position in ((0., 100.), (.25, 100.25), (.5, 100.5)):
            source.callback(NS(data=json.dumps(packet(at, position))))
            stream._apply(stream.events.get_nowait())
        body = json.loads(stream.publisher.messages[-1])
        assert body['pose']['position'][0] == pytest.approx(10.5)
        assert body['pose']['child_frame_id'] == 'body'
        assert body['vendor']['motus_slam']['extrinsic']['matrix'][0][3] == 1.
        segment = body['vendor']['motus_slam']['segment_id']
        stream._apply(('visual', 1_500_000_000, 'new', dict(position=[-30., 0., 0.],
                                                          quaternion=[0., 0., 0., 1.])))
        source.callback(NS(data=json.dumps(packet(.75, 100.75))))
        stream._apply(stream.events.get_nowait())
        continued = json.loads(stream.publisher.messages[-1])
        assert continued['vendor']['motus_slam']['segment_id'] == segment
        assert continued['pose']['position'][0] == pytest.approx(10.75)
    finally:
        session.process.returncode = 0
        card.shutdown()


def test_dynamic_camera_transform_is_interpolated_in_capture_time():
    mounts = CaptureMounts(dict(topic='/mount'), 'camera', 'ignored')
    mounts.observe(dict(stamp_ns=100, matrix=pose_at(1).tolist()), 100)
    mounts.observe(dict(stamp_ns=200, matrix=pose_at(3).tolist()), 200)
    np.testing.assert_allclose(mounts.matrix_at(150), pose_at(2))
    assert mounts.matrix_at(50) is None


def test_external_pose_forwards_measured_camera_mount_at_its_own_stamp(sessions):
    camera = declarations()
    camera['/rgb']['vendor']['body_mount'] = dict(topic='/mount')
    card = RgbdOdometryPlugin({}, _FakeExecutor())
    card.dispatch('rgbd_odometry', dict(action='start', camera_info=camera,
                  input_topics=['/rgb', '/depth', '/external'],
                  input_bindings={'/external': 'state/odom'}))
    session = card.instances['default'][0]
    stream = card.external_streams['default']
    source = next(sub for sub in stream.node.subscriptions if sub.topic == '/external')
    try:
        capture = 1_100_000_000
        stream._apply(('mount', dict(stamp_ns=capture, matrix=pose_at(1).tolist()), capture))
        for stamp, position in ((0., 10.), (.2, 10.2)):
            source.callback(NS(data=json.dumps(packet(stamp, position))))
            stream._apply(stream.events.get_nowait())
        first = json.loads(stream.publisher.messages[-1])
        output = first['vendor']
        assert output['motus_slam']['stamp_ns'] == 1_200_000_000
        evidence = output['camera_mount']['packet']
        assert evidence['stamp_ns'] == capture
        assert evidence['matrix'][0][3] == 1.
        assert evidence['parent_frame'] == 'body' and evidence['child_frame'] == 'camera_optical'
        slam = slam_stream(cloud_frame='camera_optical', pose_capture_mount=True)
        slam.odom(first, now=1.2)
        slam.cloud(1_250_000_000, 'camera_optical', [[1., 0., 0.]], now=1.25)
        assert not slam.accepted
        capture = 1_300_000_000
        stream._apply(('mount', dict(stamp_ns=capture, matrix=pose_at(2).tolist()), capture))
        source.callback(NS(data=json.dumps(packet(.4, 10.4))))
        stream._apply(stream.events.get_nowait())
        slam.odom(json.loads(stream.publisher.messages[-1]), now=1.4)
        frame, = slam.accepted
        assert frame.cloud_stamp_ns == 1_250_000_000
        assert frame.child_from_cloud[0, 3] == pytest.approx(1.75)
    finally:
        session.process.returncode = 0
        card.shutdown()


def test_external_pose_publishes_while_parameter_solve_is_pending(sessions, monkeypatch):
    solving, release, entered, published = (Event() for _ in range(4))
    original_solve = ExternalMotionLearner._solve

    def slow_solve(learner):
        solving.set()
        release.wait(2.)
        return original_solve(learner)

    monkeypatch.setattr(ExternalMotionLearner, '_solve', slow_solve)
    monkeypatch.setattr(rgbd_external_input, '_LEARNING_BUDGET_MS', 3000.)
    card = RgbdOdometryPlugin({}, _FakeExecutor())
    camera = declarations()
    camera['/rgb']['vendor']['body_from_camera'] = np.eye(4).tolist()
    result = card.dispatch('rgbd_odometry', dict(action='start', camera_info=camera,
                           input_topics=['/rgb', '/depth', '/external'],
                           input_bindings={'/external': 'state/odom'}))
    assert result['state'] == 'running', result
    stream = card.external_streams['default']
    source = next(sub for sub in stream.node.subscriptions if sub.topic == '/external')
    monkeypatch.setattr(threading, 'Thread', Thread)

    def push_motion(t, x):
        source.callback(NS(data=json.dumps(packet(t, x))))
        stream._apply(stream.events.get(timeout=.5))

    def train():
        for step in range(4):
            t = step / 4
            push_motion(t, 4*t)
            stream._apply(('visual', 1_000_000_000+round(t*1e9), 'old',
                           dict(position=[3*t, 0., 0.], quaternion=[0., 0., 0., 1.])))
        entered.set()
        push_motion(1., 4.)
        published.set()

    runner = Thread(target=train)
    runner.start()
    try:
        assert solving.wait(1.)
        assert entered.wait(.25) and published.wait(.25)
        sample = json.loads(stream.publisher.messages[-1])
        assert sample['vendor']['motus_slam']['stamp_ns'] == 2_000_000_000
        assert sample['vendor']['external_fusion']['learning']['effective_after_ns'] is None

        # Changing the raw interval invalidates the result already in flight.
        push_motion(.375, 1.6)
        previous = stream.learning_work[2]
        release.set()
        previous.result(timeout=2.)
        stream._complete_learning()
        assert stream.fuser.current.learner.effective_after_ns is None
        replacement = stream.learning_work[2]
        assert replacement is not previous
        replacement.result(timeout=2.)
        stream._complete_learning()
        push_motion(1.25, 5.)
        sample = json.loads(stream.publisher.messages[-1])
        assert sample['vendor']['external_fusion']['learning']['effective_after_ns'] == 1_750_000_000
    finally:
        release.set()
        runner.join(3.)
        card.instances['default'][0].process.returncode = 0
        card.shutdown()


def test_untyped_core_wire_classifies_external_odometry_from_payload(sessions, monkeypatch):
    assisted_runtime(monkeypatch)
    card = RgbdOdometryPlugin({}, _FakeExecutor())
    try:
        result = card.dispatch('rgbd_odometry', dict(action='start', camera_info=declarations(),
                               input_topics=['/rgb', '/depth', '/arbitrary-motion']))
        assert result['state'] == 'running', result
        stream = card.external_streams['default']
        source = stream.node.subscriptions[0]
        assert not stream.active
        source.callback(NS(data=json.dumps(dict(schema='motus.imu/1'))))
        assert stream.events.empty()
        for t, x in ((0., 20.), (.25, 20.25)):
            source.callback(NS(data=json.dumps(packet(t, x))))
            stream._apply(stream.events.get_nowait())
            if t == 0.:
                assert not stream.active and not stream.publisher.messages
        assert stream.active and stream.selected == '/arbitrary-motion'
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(.25)
    finally:
        if 'default' in card.instances:
            card.instances['default'][0].process.returncode = 0
        card.shutdown()
