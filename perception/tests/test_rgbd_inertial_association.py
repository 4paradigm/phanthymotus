"""Real capture-time rotation hints, independent of native ROS availability."""
from copy import deepcopy
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins.rgbd_imu import ImuAttitude, rotation_from_quaternion
from test_rgbd_imu import packet
from test_rgbd_odometry_status import sessions  # noqa: F401


def observed_pair(first=1_000_000_000, interval=100_000_000):
    attitude = ImuAttitude('optical', 'capture-clock')
    attitude.observe(packet(first), first)
    attitude.observe(packet(first + interval, angular_velocity=[0., 0., 1.]),
                     first + interval)
    return attitude


def test_hint_interpolates_actual_capture_brackets_without_advancing_attitude():
    attitude = observed_pair()
    before = deepcopy(attitude.__dict__)
    hint = attitude.orientation_at(1_040_000_000, 1_100_000_000)
    expected = [0., 0., np.sin(.02), np.cos(.02)]
    np.testing.assert_allclose(hint, expected, atol=1e-12)
    assert attitude.last_stamp == before['last_stamp']
    assert attitude.state == before['state']
    assert list(attitude.history) == list(before['history'])
    np.testing.assert_array_equal(attitude.quaternion, before['quaternion'])
    # A consumer cannot mutate the stored orientation through the snapshot.
    hint[0] = 99.
    assert attitude.orientation_at(1_040_000_000, 1_100_000_000)[0] == 0.


def test_hint_preserves_adjacent_nanoseconds_at_unix_epoch():
    first = 1_800_000_000_000_000_123
    attitude = observed_pair(first, 20_000_000)
    q1 = attitude.orientation_at(first + 10_000_000, first + 20_000_000)
    q2 = attitude.orientation_at(first + 10_000_001, first + 20_000_000)
    # Absolute double seconds cannot distinguish these capture times.
    assert float(first + 10_000_000) == float(first + 10_000_001)
    angle = lambda q: 2 * np.arctan2(q[2], q[3])
    assert angle(q2) - angle(q1) == pytest.approx(1e-9, abs=2e-14)


def test_hint_shortest_arc_is_invariant_to_quaternion_sign():
    attitude = observed_pair()
    expected = attitude.orientation_at(1_050_000_000, 1_100_000_000)
    stamp, q = attitude.orientation_history[-1]
    attitude.orientation_history[-1] = stamp, -np.asarray(q)
    hint = attitude.orientation_at(1_050_000_000, 1_100_000_000)
    np.testing.assert_allclose(rotation_from_quaternion(hint),
                               rotation_from_quaternion(expected), atol=1e-12)


def test_identical_rotation_and_exact_endpoints_are_valid_hints():
    attitude = ImuAttitude('optical', 'capture-clock')
    for stamp in (1_000_000_000, 1_100_000_000):
        attitude.observe(packet(stamp), stamp)
    for stamp in (1_000_000_000, 1_050_000_000, 1_100_000_000):
        np.testing.assert_array_equal(attitude.orientation_at(stamp, 1_100_000_000),
                                      [0., 0., 0., 1.])


@pytest.mark.parametrize('stamp,now', [
    (999_999_999, 1_100_000_000),       # no earlier bracket
    (1_100_000_001, 1_100_000_000),   # no extrapolation
    (0, 1_100_000_000), (True, 1_100_000_000),
    (1_050_000_000.0, 1_100_000_000),
])
def test_absent_or_invalid_coverage_returns_no_hint_without_mutating_state(stamp, now):
    attitude = observed_pair()
    before = attitude.state, attitude.last_stamp, attitude.reason
    assert attitude.orientation_at(stamp, now) is None
    assert (attitude.state, attitude.last_stamp, attitude.reason) == before


def test_missing_or_malformed_numeric_source_has_no_rotation_hint():
    attitude=ImuAttitude('optical','capture-clock')
    assert attitude.orientation_at(1_000_000_000,1_000_000_000) is None
    attitude.observe(packet(linear_acceleration=[0.,0.,0.]),1_000_000_000)
    assert attitude.orientation_at(1_000_000_000,1_000_000_000)==[0.,0.,0.,1.]
    with pytest.raises(ValueError):
        attitude.observe(packet(1_010_000_000,angular_velocity=[0.,0.]),1_010_000_000)
    assert attitude.orientation_at(1_000_000_000,1_010_000_000) is None


def test_bounded_orientation_history_evicts_old_captures():
    attitude = ImuAttitude('optical', 'capture-clock')
    for index in range(600):
        stamp = 1_000_000_000 + index * 1_000_000
        attitude.observe(packet(stamp), stamp)
    assert len(attitude.orientation_history) == len(attitude.history) == 512
    assert attitude.orientation_at(1_000_000_000, stamp) is None
    assert attitude.orientation_at(stamp, stamp) is not None


@pytest.mark.parametrize('now',[999_999_999,1_850_000_001])
def test_numeric_hint_bracket_has_no_wall_age_or_future_policy(now):
    attitude=observed_pair()
    q=attitude.orientation_at(1_050_000_000,now)
    np.testing.assert_allclose(q,[0.,0.,np.sin(.025),np.cos(.025)],atol=1e-12)


def association_config(**changes):
    calibration = dict(frame_id='optical', clock_id='capture-clock', imu_topic='/motion',
                       short_term_compensation=True, inertial_association=True)
    calibration.update(changes)
    return calibration


@pytest.fixture
def hint_node(monkeypatch):
    class QuaternionStamped:
        def __init__(self):
            self.header = NS(frame_id='', stamp=NS(sec=0, nanosec=0))
            self.quaternion = NS(x=0., y=0., z=0., w=0.)
    monkeypatch.setitem(sys.modules, 'geometry_msgs.msg', NS(QuaternionStamped=QuaternionStamped))
    messages, publishers = [], []
    def create(kind, topic, qos):
        publishers.append((kind, topic, qos))
        return NS(publish=messages.append)
    return NS(create_publisher=create), publishers, messages


@pytest.mark.parametrize('changes', [
    {'inertial_association': False}, {'short_term_compensation': False},
    {'imu_topic': None},
])
def test_optional_hint_is_off_without_calibrated_fusion_mode(hint_node, changes):
    from plugins.rgbd_association import RotationHints
    node, publishers, messages = hint_node
    hints = RotationHints(association_config(**changes), node, 'segment-a', object())
    assert not hints.offer(observed_pair(), 1_050_000_000, 1_100_000_000)
    assert not publishers and not messages
    assert hints.info()['enabled'] is False


def test_private_hint_publishes_identity_at_exact_capture_without_assimilating_imu(hint_node):
    from plugins.rgbd_association import RotationHints
    node, publishers, messages = hint_node
    qos = object()
    hints = RotationHints(association_config(), node, 'segment-a', qos)
    stamp = 1_800_000_000_000_000_123
    attitude = ImuAttitude('optical', 'capture-clock')
    attitude.observe(packet(stamp), stamp)
    def must_not_observe(*args):
        pytest.fail('offering a hint must not assimilate IMU again')
    attitude.observe = must_not_observe
    assert hints.offer(attitude, stamp, stamp)
    msg = messages[0]
    assert publishers[0][1:] == ('/segment-a/rotation_hint', qos)
    assert msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec == stamp
    assert msg.header.frame_id == 'optical'
    assert [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w] == [0., 0., 0., 1.]
    report = hints.info()
    assert report['offered'] == 1 and report['last_offered_stamp_ns'] == stamp
    assert report['native_applied'] is None  # DDS delivery/native selection is not observed here.


def test_receipt_hint_keeps_selected_mode_and_does_not_gate_on_names(hint_node):
    from plugins.rgbd_association import RotationHints
    node,_,messages=hint_node
    hints=RotationHints(association_config(time_basis='receipt_estimated'),node,'segment-a',object())
    attitude=observed_pair()
    attitude.frame,attitude.clock='provider-frame','provider-clock'
    assert hints.offer(attitude,1_050_000_000,1_100_000_000)
    assert len(messages)==1 and hints.info()['enabled']


def test_missing_lost_or_degraded_hint_is_skipped_without_waiting(hint_node):
    from plugins.rgbd_association import RotationHints
    node, _, messages = hint_node
    hints = RotationHints(association_config(), node, 'segment-a', object())
    attitude = observed_pair()
    assert not hints.offer(attitude, 1_150_000_000, 1_150_000_000)
    assert not hints.offer(attitude, 1_050_000_000, 1_100_000_000, available=False)
    attitude.fault('source unavailable')
    assert not hints.offer(attitude, 1_050_000_000, 1_100_000_000)
    assert not messages and hints.info()['skipped'] == 3
    assert hints.info()['offered'] == 0


def test_optional_hint_transport_failure_does_not_throw_or_change_attitude(hint_node):
    from plugins.rgbd_association import RotationHints
    node, _, _ = hint_node
    def unavailable(message):
        raise RuntimeError('private channel unavailable')
    node.create_publisher = lambda *args: NS(publish=unavailable)
    hints = RotationHints(association_config(), node, 'segment-a', object())
    attitude = observed_pair()
    assert not hints.offer(attitude, 1_050_000_000, 1_100_000_000)
    assert attitude.state == 'active' and attitude.last_stamp == 1_100_000_000
    assert hints.info()['last_skip_reason'] == 'hint transport unavailable'


def test_association_requires_explicit_patched_native_capability(tmp_path):
    from plugins.rgbd_association import require_rotation_hint_runtime
    binary = tmp_path / 'lib' / 'rtabmap_odom' / 'rgbd_odometry'
    with pytest.raises(RuntimeError, match='rebuilt RTAB-Map'):
        require_rotation_hint_runtime(binary)
    marker = tmp_path / 'share' / 'rtabmap_odom' / 'motus_rotation_hint_v1'
    marker.parent.mkdir(parents=True)
    marker.write_text('incompatible capability\n')
    with pytest.raises(RuntimeError, match='rebuilt RTAB-Map'):
        require_rotation_hint_runtime(binary)
    marker.write_text('motus_rotation_hint/1\n')
    require_rotation_hint_runtime(binary)


def test_runtime_parameter_mode_keeps_optional_hint_separate_from_legacy_imu():
    from plugins.rgbd_odometry_runtime import estimator_parameters
    active = estimator_parameters(association_config(), 'run')
    assert active['motus_rotation_hint'] is True
    assert active['wait_imu_to_init'] is False
    assert active['guess_frame_id'] == ''
    assert active['Odom/GuessMotion'] == 'true'
    assert active['Optimizer/GravitySigma'] == '0'
    plain = estimator_parameters(association_config(inertial_association=False), 'run')
    assert not plain.get('motus_rotation_hint', False)
    assert not plain.get('wait_imu_to_init', False)
    legacy = estimator_parameters(association_config(short_term_compensation=False,
                                                     inertial_association=False), 'run')
    assert legacy['wait_imu_to_init'] is True
    assert not legacy.get('motus_rotation_hint', False)


@pytest.fixture
def hint_session(sessions, hint_node, monkeypatch, tmp_path):
    from plugins import rgbd_odometry_runtime as runtime
    from plugins.rgbd_odometry_contract import validate_start_inputs
    from test_rgbd_odometry_contract import declarations
    from test_rgbd_odometry_recovery import assisted_runtime
    from vision_stubs import _FakeExecutor
    assisted_runtime(monkeypatch)
    binary = tmp_path / 'lib/rtabmap_odom/rgbd_odometry'
    marker = tmp_path / 'share/rtabmap_odom/motus_rotation_hint_v1'
    marker.parent.mkdir(parents=True)
    marker.write_text('motus_rotation_hint/1\n')
    monkeypatch.setattr(runtime, 'require_runtime', lambda: binary)
    commands, spawn = [], runtime.subprocess.Popen
    def launch(command, **kwargs):
        commands.append(command)
        return spawn(command, **kwargs)
    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    calibration = validate_start_inputs(declarations(), ['/rgb', '/depth', '/imu'], True)
    calibration.update(short_term_compensation=True, inertial_association=True)
    session = runtime.RgbdSession(calibration, _FakeExecutor(), '/association/state/odom')
    try:
        yield session, sessions[1], commands
    finally:
        session.process.returncode = 0
        session.close()


def test_real_session_wires_private_hint_and_conservative_fusion(hint_session):
    import json
    from pathlib import Path
    session, _, commands = hint_session
    params = json.loads((Path(session.directory.name) / 'parameters.json').read_text())['/**']['ros__parameters']
    assert params['motus_rotation_hint'] is True
    assert params['wait_imu_to_init'] is False
    assert 'rotation_hint:=/' + session.segment + '/rotation_hint' in commands[0]
    assert not any(arg.startswith('imu:=') for arg in commands[0])
    assert not hasattr(session, 'imu_pub')
    assert session.fusion.inertial_association is True
    assert session.info()['inertial_association']['enabled'] is True
    session._watchdog()
    sample = json.loads(session.publisher.messages[-1])
    assert sample['vendor']['inertial_association']['native_applied'] is None


def test_visual_restart_preserves_optional_rotation_without_restarting_fusion(hint_session):
    import json
    from plugins.rgbd_odometry_runtime import RgbdSession
    from vision_stubs import _FakeExecutor
    session, now, _ = hint_session
    session.fusion.fault('corrected position uncertainty exceeded limit')
    session._check_fusion(now[0])
    session.gate.fault('native visual tracking lost', 'visual')
    target = session.recovery_target(100., now[0])
    assert target['action'] == 'restart'
    restarted = RgbdSession(target['calibration'], _FakeExecutor(), session.topic,
                            requested_calibration=session.requested_calibration)
    try:
        assert restarted.info()['mode']['effective'] == 'visual'
        assert restarted.fusion.filter is None and not restarted._fusing()
        assert restarted.geometry_worker is None
        assert restarted._wait_for_imu(now[0])
        for stamp in (now[0]-10_000_000, now[0]+10_000_000):
            restarted._imu(NS(data=json.dumps(packet(stamp))))
        offered = []
        restarted.rotation_hints.publisher.publish = offered.append
        assert restarted.rotation_hints.offer(restarted.imu, now[0], now[0])
        assert len(offered) == 1
    finally:
        restarted.process.returncode = 0
        restarted.close()


@pytest.mark.parametrize('translations', [False, True])
@pytest.mark.parametrize('coverage', [False, True])
@pytest.mark.parametrize('color_offset_ns', [1, 800_000])
def test_actual_image_worker_offers_exact_hint_before_image_and_never_waits(
        hint_session, monkeypatch, coverage, color_offset_ns, translations):
    import json
    import zlib
    from plugins import rgbd_odometry_runtime as runtime
    session, now, _ = hint_session
    depth_stamp = 1_800_000_000_000_000_123
    color_stamp = depth_stamp + color_offset_ns
    now[0] = depth_stamp + 20_000_000
    if coverage:
        for stamp in (depth_stamp - 10_000_000, depth_stamp + 10_000_000):
            imu = packet(stamp, camera_frame_id='camera_optical', clock_id='camera-global',
                         translation_source='sdk_factory_extrinsics',
                         camera_from_gyro_translation=[0., 0., 0.],
                         camera_from_accel_translation=[0., 0., 0.])
            if not translations:
                imu.pop('camera_from_gyro_translation')
                imu.pop('camera_from_accel_translation')
            session._imu(NS(data=json.dumps(imu)))
        if not translations:
            session._check_fusion(now[0])
            assert session.fusion.state == 'lost' and session.fallback is not None
            assert list(session.imu.history) == [depth_stamp - 10_000_000, depth_stamp + 10_000_000]

    # Codec and DDS are boundaries; run the actual worker, pairing stamp choice,
    # optional-hint decision and publication order.
    monkeypatch.setitem(sys.modules, 'cv2', NS(IMREAD_COLOR=1, IMWRITE_PNG_COMPRESSION=16,
        imdecode=lambda data, mode: np.zeros((480, 640, 3), np.uint8),
        imencode=lambda *args: (True, np.frombuffer(b'encoded-depth', np.uint8))))
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'CompressedImage', NS)
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'CameraInfo', NS, raising=False)
    monkeypatch.setattr(sys.modules['rtabmap_msgs.msg'], 'RGBDImage', NS)
    def frame(stamp, data):
        return NS(header=NS(frame_id='camera_optical',
                           stamp=NS(sec=stamp//10**9, nanosec=stamp%10**9)),
                  data=data, format='compressed')
    color = frame(color_stamp, b'color-image')
    depth = frame(depth_stamp, zlib.compress(np.full((480, 640), 1500, '<u2').tobytes()))
    originals = deepcopy((color.header, depth.header))
    session.pending.put(((color_stamp, color), (depth_stamp, depth)))
    def must_not_wait(seconds):
        pytest.fail('an optional association hint cannot wait for IMU')
    monkeypatch.setattr(session.stop_event, 'wait', must_not_wait)
    events = []
    session.rotation_hints.publisher.publish = lambda msg: events.append(('hint', msg))
    def image(msg):
        events.append(('image', msg))
        session.stop_event.set()
    session.rgbd.publish = image
    session._work()
    assert session.gate.state != 'lost', session.gate.reason
    assert session.counts['pairs'] == 1 and session.counts['dropped_pairs'] == 0
    assert [kind for kind, _ in events] == (['hint', 'image'] if coverage else ['image'])
    if coverage:
        capture = runtime.native_capture_stamp(color, depth)
        assert runtime.stamp_ns(events[0][1]) == capture
        assert capture == (depth_stamp if color_offset_ns == 1 else color_stamp)
        assert session.imu.last_stamp == depth_stamp + 10_000_000
    if coverage and not translations:
        assert session.info()['mode']['effective'] == 'visual'
        assert session.rotation_hints.info(capture)['offered_for_frame'] is True
        assert session.rotation_hints.info(capture + 100_000_000)['offered_for_frame'] is False
        assert session.rotation_hints.info(capture)['native_applied'] is None
        for kind in ('visual', 'input'):
            session.gate.loss_kind = None
            session.gate.state = 'tracking'
            session.gate.fault('lost principal input', kind)
            target = session.recovery_target(100., now[0])
            assert target['calibration'].get('imu_topic') and target['calibration']['inertial_association']
    assert events[-1][1].rgb_compressed.header == originals[0]
    assert events[-1][1].depth_compressed.header == originals[1]
    assert color.header == originals[0] and depth.header == originals[1]


def test_malformed_imu_during_rotation_fallback_does_not_supply_an_old_hint(hint_session):
    import json
    session, now, _ = hint_session
    stamp = now[0]
    session._imu(NS(data=json.dumps(packet(stamp))))
    session._check_fusion(now[0])
    assert session.fallback is not None and session.imu.state == 'active'
    session._imu(NS(data='invalid json'))
    assert session.imu.orientation_at(stamp, now[0]) is None
    assert session.gate.state != 'lost'


def test_native_hint_statistics_come_from_the_last_native_report(tmp_path):
    from plugins.rgbd_association import native_hint_statistics

    log = tmp_path / 'estimator.log'
    assert native_hint_statistics(log) is None
    log.write_bytes(b'[INFO] [1.0] [seg.rgbd_odometry]: motus_rotation_hint api=1: optional\n')
    assert native_hint_statistics(log) is None
    log.write_bytes(log.read_bytes() +
                    b'[INFO] [2.0] [seg.rgbd_odometry]: rotation_hint: frames=60 selected=49 supplied=38 '
                    b'rejected=0 late=0 evicted=0\n' + b'x'*100 +
                    b'\n[INFO] [3.0] [seg.rgbd_odometry]: rotation_hint: frames=120 selected=100 supplied=80 '
                    b'rejected=1 late=2 evicted=3\n')
    assert native_hint_statistics(log) == dict(frames=120, selected=100, supplied=80, rejected=1,
                                               late=2, evicted=3, supplied_fraction=.667)


def test_session_status_reports_native_hint_counters_from_the_estimator_log(hint_session):
    session, _, _ = hint_session
    assert session.info()['inertial_association']['native_applied'] is None
    session.log.write(b'[INFO] [2.0] [x.rgbd_odometry]: rotation_hint: frames=60 selected=49 supplied=38 '
                      b'rejected=0 late=0 evicted=0\n')
    session._hint_statistics = None
    report = session.info()['inertial_association']
    assert report['native_applied']['supplied'] == 38 and report['native_applied']['frames'] == 60
