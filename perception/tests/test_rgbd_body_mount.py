"""Exact capture-time body mount propagation; main camera reference is preserved."""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pytest

from plugins.rgbd_body_mount import CaptureMounts, validate_mount_declaration
from plugins.rgbd_odometry_contract import make_sample, validate_start_inputs
from test_rgbd_odometry_contract import declarations


def declaration(**changes):
    d = dict(schema='motus.camera_mount/1', topic='/camera/rgbd/body_mount',
             parent_frame='base_link', child_frame='camera_optical', source='tf',
             version='capture_tf_v1', epoch='boot1', mode='dynamic', length_unit='m',
             transform_direction='body_from_camera', status='unavailable',
             reason='waiting for TF', accuracy_validated=False)
    d.update(changes)
    return d


def packet(stamp=1_000_000_123, matrix=None, **changes):
    p = {k: v for k, v in declaration().items() if k != 'topic'}
    p.update(stamp_ns=stamp, clock_id='camera-clock', stamp_source='measurement',
             status='available', reason=None,
             matrix=np.asarray(np.eye(4) if matrix is None else matrix).tolist())
    p.update(changes)
    return p


def sample(stamp=1_000_000_123, status='tracking', position=(0., 0., 0.), q=(0., 0., 0., 1.)):
    pose = dict(position=list(position), quaternion=list(q), frame_id='run_odom',
                child_frame_id='camera_optical', covariance=(np.eye(6)*.001).ravel().tolist())
    return make_sample(status, stamp, 'run', 'camera-clock', pose)


def mounts():
    return CaptureMounts(declaration(), 'camera_optical', 'camera-clock')


def test_known_mount_produces_explicit_body_pose_without_relabeling_camera_pose():
    observer = mounts()
    transform = np.eye(4); transform[0, 3] = .2  # camera origin0.2m ahead of body
    observer.observe(packet(matrix=transform), 1_010_000_123)
    camera = sample(position=(1., 0., 0.)); original = deepcopy(camera)
    out = observer.attach(camera, 1_020_000_123)
    assert out['pose'] == original['pose']
    assert out['vendor']['pose_reference'] == 'camera_optical'
    body = out['vendor']['body_pose']
    assert body['position'] == pytest.approx([.8, 0, 0])
    assert body['child_frame_id'] == 'base_link'
    assert body['stamp_ns'] == 1_000_000_123
    assert body['covariance'] is None and body['accuracy_validated'] is False


def test_camera_turn_uses_rotated_installation_offset():
    observer = mounts()
    transform = np.eye(4); transform[0, 3] = .2
    observer.observe(packet(matrix=transform), 1_010_000_123)
    out = observer.attach(sample(position=(1, 2, 0), q=(0, 0, np.sqrt(.5), np.sqrt(.5))), 1_020_000_123)
    assert out['vendor']['body_pose']['position'] == pytest.approx([1, 1.8, 0])


def test_dynamic_camera_motion_yields_stationary_body_without_freezing_mount():
    observer = mounts()
    first = np.eye(4); first[:3, 3] = [.1, 0, .3]
    observer.observe(packet(matrix=first), 1_010_000_123)
    out = observer.attach(sample(position=(.1, 0, .3)), 1_020_000_123)
    assert out['vendor']['body_pose']['position'] == pytest.approx([0, 0, 0])
    second = np.array([[0, -1, 0, .2], [1, 0, 0, 0], [0, 0, 1, .3], [0, 0, 0, 1.]])
    observer.observe(packet(1_100_000_123, second), 1_110_000_123)
    out = observer.attach(sample(1_100_000_123, position=(.2, 0, .3), q=(0, 0, np.sqrt(.5), np.sqrt(.5))), 1_120_000_123)
    body = out['vendor']['body_pose']
    assert body['position'] == pytest.approx([0, 0, 0], abs=1e-10)
    assert body['quaternion'] == pytest.approx([0, 0, 0, 1], abs=1e-10)


def test_exact_nanosecond_match_does_not_use_nearest_or_same_millisecond_mount():
    observer = mounts(); observer.observe(packet(), 1_010_000_123)
    assert sample()['vendor']['motus_slam']['stamp_ns'] == 1_000_000_123
    out = observer.attach(sample(1_000_000_124), 1_020_000_123)
    assert out['pose'] is not None
    assert out['vendor']['body_pose'] is None
    assert out['vendor']['camera_mount']['status'] == 'unavailable'


def test_missing_unavailable_and_expired_tf_keep_camera_tracking():
    observer = mounts()
    assert observer.attach(sample(), 1_020_000_123)['vendor']['body_pose'] is None
    unavailable = packet(status='unavailable', reason='no TF path base_link <- camera_optical')
    del unavailable['matrix']
    observer.observe(unavailable, 1_010_000_123)
    out = observer.attach(sample(), 1_020_000_123)
    assert out['vendor']['camera_mount']['reason'] == unavailable['reason']
    assert out['pose'] is not None
    observer.observe(packet(1_100_000_123), 1_110_000_123)
    assert observer.attach(sample(1_100_000_123), 1_700_000_123)['vendor']['body_pose'] is None


@pytest.mark.parametrize('change', [dict(epoch='boot2'), dict(clock_id='other'),
                                  dict(parent_frame='other_base'), dict(child_frame='other_camera'),
                                  dict(version='v2'), dict(length_unit='mm'),
                                  dict(transform_direction='camera_from_body'),
                                  dict(matrix=[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])])
def test_invalid_mount_cannot_be_used_or_silently_recovered(change):
    observer = mounts()
    with pytest.raises(ValueError):
        observer.observe(packet(**change), 1_010_000_123)
    observer.observe(packet(1_100_000_123), 1_110_000_123)
    out = observer.attach(sample(1_100_000_123), 1_120_000_123)
    assert out['pose'] is not None and out['vendor']['body_pose'] is None
    assert observer.info(1_120_000_123)['status'] == 'lost'


@pytest.mark.parametrize('stamp,now', [(1_000_000_123, 1_600_000_123),
                                      (1_000_000_123, 900_000_123),
                                      (True, 1_000_000_000)])
def test_invalid_packet_time_fails_closed(stamp, now):
    observer = mounts()
    with pytest.raises(ValueError):
        observer.observe(packet(stamp), now)
    assert observer.attach(sample(), now)['vendor']['body_pose'] is None


def test_nontracking_outputs_never_gain_a_tracking_body_pose():
    observer = mounts(); observer.observe(packet(), 1_010_000_123)
    for state in ('initializing', 'coasting', 'recovering', 'lost'):
        out = observer.attach(sample(status=state), 1_020_000_123)
        assert out['pose'] is None and out['vendor']['body_pose'] is None


def test_history_is_bounded_and_retired_capture_not_reused():
    observer = mounts()
    for i in range(200):
        stamp = 1_000_000_123+i*5_000_000
        observer.observe(packet(stamp), stamp)
    assert observer.info(stamp)['buffered_samples'] <= 128
    assert observer.attach(sample(), stamp)['vendor']['body_pose'] is None


def test_source_declarations_are_forwarded_and_conflicting_mounts_are_rejected():
    values = declarations()
    plain = validate_start_inputs(values, ['/rgb', '/depth'])
    assert 'body_mount' not in plain
    for entry in values.values():
        entry['vendor']['body_mount'] = declaration()
    # The fixtures use a different frame name; bind it to the actual camera frame.
    frame = values['/rgb']['vendor']['frame_id']
    for entry in values.values():
        entry['vendor']['body_mount']['child_frame'] = frame
    result = validate_start_inputs(values, ['/rgb', '/depth'])
    assert result['body_mount']['topic'] == '/camera/rgbd/body_mount'
    values['/depth']['vendor']['body_mount']['parent_frame'] = 'other_base'
    with pytest.raises(ValueError):
        validate_start_inputs(values, ['/rgb', '/depth'])


def test_mount_declaration_cannot_alias_an_image_topic_or_camera_as_body():
    for change in (dict(topic='/rgb'), dict(parent_frame='camera_optical'), dict(source='guess')):
        with pytest.raises(ValueError):
            validate_mount_declaration(declaration(**change), 'camera_optical', ['/rgb', '/depth'])


def test_runtime_automatically_attaches_mount_without_altering_main_odom(monkeypatch):
    import json, threading, sys
    from types import SimpleNamespace as NS
    from plugins import rgbd_odometry_runtime as runtime
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'String', lambda **kw: NS(**kw))
    obj = object.__new__(runtime.RgbdSession)
    obj.mounts = mounts(); obj.imu = None
    obj.calibration = {}; obj.fusion = None
    obj.gate = runtime.EstimateGate('run', 'camera_optical', 'camera-clock')
    obj.registration_by_stamp = OrderedDict()
    obj.lock = threading.RLock(); obj.stop_event = threading.Event()
    published = []
    obj.publisher = NS(publish=lambda msg: published.append(json.loads(msg.data)))
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: 1_010_000_123)
    transform = np.eye(4); transform[0, 3] = .2
    obj._mount(NS(data=json.dumps(packet(matrix=transform))))
    obj._publish(sample(position=(1., 0., 0.)))
    assert published[0]['pose']['position'] == [1., 0., 0.]
    assert published[0]['vendor']['body_pose']['position'] == pytest.approx([.8, 0, 0])
    obj._mount(NS(data='invalid-json'))
    obj._publish(sample(position=(1., 0., 0.)))
    assert published[1]['pose']['position'] == [1., 0., 0.]
    assert published[1]['vendor']['body_pose'] is None


def test_card_repeated_start_ignores_only_mount_liveness_snapshot(monkeypatch):
    from plugins import rgbd_odometry as module
    from test_rgbd_odometry import Session
    monkeypatch.setattr(module, 'RgbdSession', Session)
    monkeypatch.setattr(module, 'require_runtime', lambda: None)
    Session.created = []
    values = declarations()
    for entry in values.values():
        entry['vendor']['body_mount'] = declaration(child_frame=entry['vendor']['frame_id'])
    card = module.RgbdOdometryPlugin({}, object())
    args = dict(action='start', camera_info=values, input_topics=['/rgb', '/depth'])
    assert card.dispatch('rgbd_odometry', args)['state'] == 'running'
    for entry in values.values():
        entry['vendor']['body_mount'].update(status='available', reason=None)
    assert card.dispatch('rgbd_odometry', args)['state'] == 'running'
    assert len(Session.created) == 1
    for entry in values.values():
        entry['vendor']['body_mount']['epoch'] = 'new-boot'
    assert card.dispatch('rgbd_odometry', args)['state'] == 'error'
    card.shutdown()


def test_session_discovers_mount_topic_from_camera_metadata(monkeypatch):
    import json, sys
    from types import SimpleNamespace as NS
    from plugins import rgbd_odometry_runtime as runtime
    from vision_stubs import _FakeExecutor
    monkeypatch.setattr(sys.modules['rclpy.qos'], 'qos_profile_sensor_data', object(), raising=False)
    monkeypatch.setitem(sys.modules, 'rtabmap_msgs.msg', NS(RGBDImage=object, OdomInfo=object))
    monkeypatch.setitem(sys.modules, 'nav_msgs.msg', NS(Odometry=object))
    monkeypatch.setattr(sys.modules['std_msgs.msg'], 'String', lambda **kw: NS(**kw))
    monkeypatch.setattr(runtime, 'require_runtime', lambda: '/offline/estimator')
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **kw: NS(poll=lambda: 0, wait=lambda **kw: 0))
    monkeypatch.setattr(runtime.threading, 'Thread', lambda **kw: NS(start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: 1_010_000_123)
    values = declarations()
    frame = values['/rgb']['vendor']['frame_id']
    for entry in values.values():
        entry['vendor']['body_mount'] = declaration(child_frame=frame)
    calibration = validate_start_inputs(values, ['/rgb', '/depth'])
    shared = _FakeExecutor()
    session = runtime.RgbdSession(calibration, shared, '/out')
    try:
        assert session.executor is shared and not session.owns_executor
        subscriptions = {sub.topic: sub for sub in session.node.subscriptions}
        mount = packet(child_frame=frame, clock_id=calibration['clock_id'])
        subscriptions['/camera/rgbd/body_mount'].callback(NS(data=json.dumps(mount)))
        output = sample()
        output['pose']['child_frame_id'] = frame
        output['vendor']['motus_slam']['clock_id'] = calibration['clock_id']
        session._publish(output)
        body = json.loads(session.publisher.messages[-1])['vendor']['body_pose']
        assert body['child_frame_id'] == 'base_link'
        assert body['stamp_ns'] == mount['stamp_ns']
    finally:
        session.close()
    assert session.node.destroyed
