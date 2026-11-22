import json
import threading
import time
from itertools import count
from threading import Event, Thread
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_external_input
from plugins.external_motion_learning import ExternalMotionLearner
from plugins.rgbd_body_mount import CaptureMounts
from plugins.rgbd_odometry import RgbdOdometryPlugin
from test_external_odometry_fusion import attitude_packet, packet, pose_at
from plugins.external_odometry_fusion import decode_motion
from test_pointcloud_slam_stream import stream as slam_stream
from test_rgbd_odometry_contract import declarations
from test_rgbd_odometry_status import sessions  # noqa: F401
from test_rgbd_odometry_recovery import assisted_runtime, images
from vision_stubs import _FakeExecutor


def fusion_input():
    return rgbd_external_input.ExternalOdometryInput('/fused/state/odom', dict(
        frame_id='body', external_topic='/external', external_candidates=['/external']), learning=False)


def test_future_motion_cannot_advance_replay_or_publication(sessions):
    _, now, _ = sessions
    stream = fusion_input()
    try:
        for capture in (9.9, 10., 3609.9, 3610., 10.1, 10.2):
            if capture < 100.:
                now[0] = round(capture*1e9)
            stream._receive('/external', NS(data=json.dumps(packet(capture-1., capture))))
            stream._apply_ready(stream._take_ready())
        samples = [json.loads(message) for message in stream.publisher.messages]
        assert [sample['vendor']['motus_slam']['stamp_ns'] for sample in samples] == [
            10_000_000_000, 10_100_000_000, 10_200_000_000]
        assert samples[-1]['pose']['position'][0] == pytest.approx(.3)
        assert max(stream.fuser.raw_motion) == 10_200_000_000
    finally:
        stream.close()


def test_established_source_replays_delayed_motion_within_history(sessions):
    _, now, _ = sessions
    stream = fusion_input()

    def send(capture, velocity):
        motion = packet(capture-1., twist=[velocity, 0., 0., 0., 0., 0.])
        stream._receive('/external', NS(data=json.dumps(motion)))
        stream._apply_ready(stream._take_ready())

    try:
        for t in (7., 8., 9., 10.):
            now[0] = round(t*1e9)
            send(t, 1.)
        send(7.5, 1.04)  # Within replay history, older than the publication timeout.
        now[0] = 10_100_000_000
        send(10.1, 1.)
        samples = [json.loads(msg) for msg in stream.publisher.messages]
        assert [s['vendor']['motus_slam']['stamp_ns'] for s in samples] == [
            8_000_000_000, 9_000_000_000, 10_000_000_000, 10_100_000_000]
        assert samples[-1]['pose']['position'][0] == pytest.approx(3.12)
    finally:
        stream.close()


def test_auto_source_needs_its_own_fresh_usable_interval(sessions):
    _, now, _ = sessions
    now[0] = 10_000_000_000
    stream = rgbd_external_input.ExternalOdometryInput('/fused/state/odom', dict(
        frame_id='body', external_candidates=['/old', '/live']), learning=False)
    try:
        for source, t, x, reset in (('/old', 0., None, True), ('/old', 8.8, None, True),
                                     ('/old', 8.85, 100., False), ('/live', 8.9, 20., False)):
            stream._apply(('motion', source, *decode_motion(packet(t, x, reset=reset), now[0])))
            assert stream.selected is None
        for t in (8.95, 9.):
            yaw_only = packet(t, twist=[None, None, None, None, None, .5])
            stream._apply(('motion', '/old', *decode_motion(yaw_only, now[0])))
            assert stream.selected is None  # A partial stream alone cannot establish a body track.
        stream._apply(('motion', '/live', *decode_motion(packet(9., 20.1), now[0])))
        assert stream.selected == '/live'
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(.1)
        assert sorted(stream.fuser.raw_motion) == [9_900_000_000, 10_000_000_000]
    finally:
        stream.close()


def test_fusion_rejects_stale_measurements_even_during_publication(sessions):
    _, now, _ = sessions
    stream = fusion_input()
    try:
        now[0] = 31_000_000_000
        for t in (0., .2):
            stream._apply(('motion', '/external', *decode_motion(packet(t, t), now[0])))
        assert not stream.publisher.messages
        assert stream.info()['status'] == 'lost'
        assert stream.info()['stamp_ns'] == 1_200_000_000
        assert stream.info()['reason'] == 'external fusion measurement is stale'

        for t in (30., 30.2):
            now[0] = 1_010_000_000 + round(t*1e9)
            stream._apply(('motion', '/external', *decode_motion(packet(t, t), now[0])))
        assert stream.info()['status'] == 'tracking'
        assert json.loads(stream.publisher.messages[-1])['vendor']['motus_slam']['stamp_ns'] == 31_200_000_000
        now[0] += 2_000_000_000
        assert stream.info()['status'] == 'lost'
    finally:
        stream.close()


def test_fusion_queue_keeps_recent_events_and_skips_expired_work(sessions, monkeypatch):
    _, now, _ = sessions
    now[0] = 1_900_000_000
    enqueued_at = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: enqueued_at[0])
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 4, raising=False)
    stream = fusion_input()
    source = stream.node.subscriptions[0]
    try:
        for i in range(8):
            t = i / 20
            source.callback(NS(data=json.dumps(packet(t, t))))
        assert stream.events.qsize() == 4
        assert stream.dropped == 4
        assert [event[2] for event in stream._take_ready()] == [
            1_200_000_000, 1_250_000_000, 1_300_000_000, 1_350_000_000]
        source.callback(NS(data=json.dumps(packet(.4, .4))))
        enqueued_at[0] += .5
        now[0] = 3_000_000_000
        source.callback(NS(data=json.dumps(packet(.9, .9))))
        assert [event[2] for event in stream._take_ready()] == [1_900_000_000]
        assert stream.dropped == 5
        for t in (1.9, 1.95):
            source.callback(NS(data=json.dumps(packet(t, t))))
        stream._apply_ready(stream._take_ready())
        emitted, = stream.publisher.messages
        assert json.loads(emitted)['pose']['position'][0] == pytest.approx(.05)
    finally:
        stream.close()


def test_fusion_batch_publishes_fresh_intermediate_measurements(sessions, monkeypatch):
    _, now, _ = sessions
    stream = fusion_input()
    try:
        now[0] = 1_250_000_000
        for at in (0., .08, .16, .24):
            stream._receive('/external', NS(data=json.dumps(packet(at, at))))

        rebuilds = []
        rebuild = stream.fuser._rebuild

        def count_rebuild(stamp):
            rebuilds.append(stamp)
            rebuild(stamp)

        monkeypatch.setattr(stream.fuser, '_rebuild', count_rebuild)
        stream._apply_ready(stream._take_ready())
        packets = [json.loads(msg) for msg in stream.publisher.messages]
        assert [p['vendor']['motus_slam']['stamp_ns'] for p in packets] == [
            1_080_000_000, 1_160_000_000, 1_240_000_000]
        assert [p['pose']['position'][0] for p in packets] == pytest.approx([.08, .16, .24])
        assert len(rebuilds) == 1
    finally:
        stream.close()


def test_partial_attitude_cannot_publish_a_pose_until_measured_translation_returns(sessions):
    _, now, _ = sessions
    now[0] = 1_500_000_000
    stream = fusion_input()
    try:
        for item in (attitude_packet(0., 10., 0., .6),
                     attitude_packet(.1, 10.1, 0., .6),
                     attitude_packet(.31, -20., .126, .6, frame='new_leg')):
            stream._apply(('motion', '/external', *decode_motion(item, now[0])))
        previous, = [json.loads(msg) for msg in stream.publisher.messages]
        assert previous['vendor']['motus_slam']['stamp_ns'] == 1_100_000_000
        sample = attitude_packet(.41, -19.9, .186, .6, frame='new_leg')
        stream._apply(('motion', '/external', *decode_motion(sample, now[0])))
        latest = json.loads(stream.publisher.messages[-1])
        assert len(stream.publisher.messages) == 2
        assert latest['vendor']['motus_slam']['stamp_ns'] == 1_410_000_000
        assert latest['pose']['quaternion'][2] == pytest.approx(np.sin(.186/2))
    finally:
        stream.close()


def test_fusion_backlog_keeps_measured_pose_brackets_without_publishing_stale_poses(sessions, monkeypatch):
    _, now, _ = sessions
    tick = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: tick[0])
    stream = fusion_input()
    try:
        for at in (0., .1, 1.6, 1.7, 1.8):
            now[0] = 1_010_000_000 + round(at*1e9)
            stream._receive('/external', NS(data=json.dumps(packet(at, at))))
        tick[0] = .35
        now[0] = 2_950_000_000
        stream._apply_ready(stream._take_ready())
        stamps = [json.loads(msg)['vendor']['motus_slam']['stamp_ns'] for msg in stream.publisher.messages]
        assert stamps == [2_700_000_000, 2_800_000_000]
        assert stream.dropped == 2
    finally:
        stream.close()


@pytest.mark.parametrize('overflow', [False, True])
def test_fusion_backlog_preserves_external_pose_reset(sessions, monkeypatch, overflow):
    _, now, _ = sessions
    now[0] = 1_510_000_000
    timing = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: timing[0])
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = fusion_input()
    source = stream.node.subscriptions[0]
    try:
        for t, x in ((0., 10.), (.1, 10.1)):
            source.callback(NS(data=json.dumps(packet(t, x))))
        stream._apply_ready(stream._take_ready())
        source.callback(NS(data=json.dumps(packet(.2, -90., reset=True))))
        timing[0] = 0.3
        if overflow:
            source.callback(NS(data=json.dumps(packet(.3, -89.9))))
        source.callback(NS(data=json.dumps(packet(.5, -89.7))))
        now[0] = 1_510_000_000
        stream._apply_ready(stream._take_ready())
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(.4)
    finally:
        stream.close()


@pytest.mark.parametrize('overflow', [False, True])
def test_fusion_backlog_does_not_integrate_a_dropped_twist_interval(sessions, monkeypatch, overflow):
    _, now, _ = sessions
    now[0] = 1_510_000_000
    timing = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: timing[0])
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = fusion_input()
    source = stream.node.subscriptions[0]

    def send(t, x):
        source.callback(NS(data=json.dumps(packet(t, twist=[x, 0., 0., 0., 0., 0.]))))

    try:
        send(0., 1.)
        send(.1, 1.)
        stream._apply_ready(stream._take_ready())
        send(.2, 0.)
        timing[0] = 0.3
        if overflow:
            send(.3, 0.)
        send(.5, 0.)
        now[0] = 1_510_000_000
        stream._apply_ready(stream._take_ready())
        expected = .1 if overflow else .2
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(expected)
    finally:
        stream.close()


def test_fusion_does_not_bridge_overwritten_reset_with_out_of_order_pose(sessions, monkeypatch):
    _, now, _ = sessions
    now[0] = 1_610_000_000
    monkeypatch.setattr(rgbd_external_input, '_MAX_PENDING_EVENTS', 2)
    stream = fusion_input()
    source = stream.node.subscriptions[0]

    def send(t, x, reset=False):
        source.callback(NS(data=json.dumps(packet(t, x, reset=reset))))

    try:
        send(0., 100.)
        send(.1, 100.1)
        stream._apply_ready(stream._take_ready())
        send(.2, -90., reset=True)
        send(.45, 7., reset=True)
        send(.3, -89.9)
        send(.5, 7.1)
        now[0] = 1_510_000_000
        stream._apply_ready(stream._take_ready())
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(.2)
        send(.35, -89.8)
        now[0] = 1_610_000_000
        send(.6, 7.2)
        stream._apply_ready(stream._take_ready())
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(.3)
    finally:
        stream.close()


def test_visual_loss_does_not_change_public_fusion_status_or_segment(sessions, monkeypatch):
    _, wall_clock, _ = sessions
    wall_clock[0] = 1_610_000_000
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
            stream._apply(stream._take_ready()[0])
        first = card.dispatch('rgbd_odometry', dict(action='info'))
        assert first['status'] == 'tracking'
        segment = first['segment_id']
        assert stream.node not in executor.nodes
        session.gate.fault('native visual tracking lost', 'visual')
        source.callback(NS(data=json.dumps(packet(.4, 10.4))))
        stream._apply(stream._take_ready()[0])
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
        stream._apply(stream._take_ready()[0])
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
    _, wall_clock, _ = sessions
    wall_clock[0] = 1_760_000_000
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
            stream._apply(stream._take_ready()[0])
        body = json.loads(stream.publisher.messages[-1])
        assert body['pose']['position'][0] == pytest.approx(10.5)
        assert body['pose']['child_frame_id'] == 'body'
        assert body['vendor']['motus_slam']['extrinsic']['matrix'][0][3] == 1.
        segment = body['vendor']['motus_slam']['segment_id']
        stream._apply(('visual', 1_500_000_000, 'new', dict(position=[-30., 0., 0.],
                                                          quaternion=[0., 0., 0., 1.])))
        source.callback(NS(data=json.dumps(packet(.75, 100.75))))
        stream._apply(stream._take_ready()[0])
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
    _, wall_clock, _ = sessions
    wall_clock[0] = 1_410_000_000
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
            stream._apply(stream._take_ready()[0])
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
        stream._apply(stream._take_ready()[0])
        slam.odom(json.loads(stream.publisher.messages[-1]), now=1.4)
        frame, = slam.accepted
        assert frame.cloud_stamp_ns == 1_250_000_000
        assert frame.child_from_cloud[0, 3] == pytest.approx(1.75)
    finally:
        session.process.returncode = 0
        card.shutdown()


def test_external_pose_publishes_while_parameter_solve_is_pending(sessions, monkeypatch):
    _, wall_clock, _ = sessions
    wall_clock[0] = 2_260_000_000
    solving, release, entered, published = (Event() for _ in range(4))
    original_solve = ExternalMotionLearner._solve

    def slow_solve(learner):
        solving.set()
        release.wait(2.)
        return original_solve(learner)

    monkeypatch.setattr(ExternalMotionLearner, '_solve', slow_solve)
    # Both completed solves exceed the historical 50 ms wall-time budget.
    from plugins import external_motion_learning
    elapsed = count(0, 100_000_000)
    monkeypatch.setattr(external_motion_learning, 'perf_counter_ns', lambda: next(elapsed))
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
        stream._apply(stream._take_ready()[0])

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
    _, wall_clock, _ = sessions
    wall_clock[0] = 1_260_000_000
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
            stream._apply(stream._take_ready()[0])
            if t == 0.:
                assert not stream.active and not stream.publisher.messages
        assert stream.active and stream.selected == '/arbitrary-motion'
        assert json.loads(stream.publisher.messages[-1])['pose']['position'][0] == pytest.approx(.25)
    finally:
        if 'default' in card.instances:
            card.instances['default'][0].process.returncode = 0
        card.shutdown()


@pytest.mark.parametrize('boundary', ['reset', 'full_twist', 'other_origin'])
@pytest.mark.parametrize('delay', ['before_selection', 'after_selection'])
def test_candidate_respects_known_boundary_across_delayed_pose(sessions, boundary, delay):
    _, now, _ = sessions
    now[0] = 2_000_000_000
    stream = rgbd_external_input.ExternalOdometryInput('/fused/state/odom', dict(
        frame_id='body', external_candidates=['/candidate']), learning=False)

    def send(item):
        stream._apply(('motion', '/candidate', *decode_motion(item, now[0])))

    barrier = {'reset': packet(.8, reset=True), 'full_twist': packet(.8, twist=[0.]*6),
               'other_origin': packet(.8, 400., frame='old_world')}[boundary]
    try:
        send(barrier)
        send(packet(.9, 51.))
        if delay == 'before_selection':
            send(packet(.7, 1.))
        assert stream.selected is None
        send(packet(1., 51.1))
        assert stream.selected == '/candidate'
        if delay == 'after_selection':
            send(packet(.7, 1.))
        assert stream.fuser.current.transform[0, 3] == pytest.approx(.1)
    finally:
        stream.close()
