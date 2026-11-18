from vision_stubs import _FakeExecutor
from collections import OrderedDict
from types import SimpleNamespace as NS

import pytest

from plugins.rgbd_odometry import RgbdOdometryPlugin
from plugins.rgbd_odometry_runtime import CaptureObservations, estimator_parameters
from test_rgbd_odometry import messages, Session
from test_rgbd_odometry_contract import declarations


class Fusion:
    def __init__(self):
        self.segment, self.frame = 'run', 'camera_optical'
        self.frames, self.reason = [], None
        self.samples = [(10_000_000_000, None, None)]
    def fault(self, reason, kind='fusion'):
        self.reason = reason
    def observe_frame(self, stamp, color, depth, visual, now, reason=None, arrival_ns=None):
        self.frames.append((stamp, visual, reason))
        self.arrivals = getattr(self, 'arrivals', []) + [arrival_ns]
        return {'stamp': stamp}


def test_capture_results_are_processed_in_order_at_original_time():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, 'rgb1', 'depth1', 1_010_000_000)
    frames.add_frame(1_100_000_000, 'rgb2', 'depth2', 1_110_000_000)
    m, i = messages(1_100_000_000)
    assert frames.result('pose', m, 1_120_000_000) == []
    assert frames.result('info', i, 1_120_000_000) == []
    m, i = messages(1_000_000_000)
    frames.result('info', i, 1_130_000_000)
    out = frames.result('pose', m, 1_140_000_000)
    assert [s['stamp'] for s in out] == [1_000_000_000, 1_100_000_000]
    assert [f[0] for f in fusion.frames] == [1_000_000_000, 1_100_000_000]
    assert fusion.arrivals == [1_010_000_000, 1_110_000_000]  # enqueue time, for latency diagnostics


def test_missing_pose_timeout_and_late_results_do_not_rewrite_history():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    assert frames.drain(1_200_000_000) == []
    assert frames.drain(1_260_000_000) == [{'stamp': 1_000_000_000}]
    assert fusion.frames[0][1] is None
    m, i = messages(1_000_000_000)
    frames.result('pose', m, 1_270_000_000)
    frames.result('info', i, 1_270_000_000)
    assert len(fusion.frames) == 1


def test_transport_late_capture_does_not_feed_fusion_or_block_the_next_capture():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_900_000_000)
    assert frames.drain(2_010_000_000) == []
    assert not fusion.frames and not frames.frames
    assert frames.counts['stale_capture'] == 1
    stamp = 2_040_000_000
    frames.add_frame(stamp, None, None, stamp+10_000_000)
    msg, info = messages(stamp)
    frames.result('pose', msg, stamp+20_000_000)
    assert frames.result('info', info, stamp+25_000_000) == [{'stamp': stamp}]
    assert fusion.frames[0][1] is not None and fusion.reason is None


def test_lost_diagnostic_can_trigger_coast_without_matching_pose():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    _, info = messages(1_000_000_000); info.lost = True
    assert frames.result('info', info, 1_020_000_000) == [{'stamp': 1_000_000_000}]
    assert fusion.frames[0][1] is None


@pytest.mark.parametrize('kinds,lost,inliers,reason', [
    ((), False, 50, 'visual pose and diagnostic missing'),
    (('pose',), False, 50, 'visual diagnostic missing'),
    (('info',), False, 50, 'visual pose missing'),
    (('info',), True, 50, 'native visual tracking lost'),
    (('pose', 'info'), False, 9, 'visual inliers below 10'),
])
def test_join_distinguishes_transport_absence_from_native_quality(kinds, lost, inliers, reason):
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    msg, info = messages(1_000_000_000)
    info.lost, info.inliers = lost, inliers
    for kind in kinds:
        frames.result(kind, msg if kind == 'pose' else info, 1_020_000_000)
    frames.drain(1_260_000_000)
    assert fusion.frames == [(1_000_000_000, None, reason)]


def test_join_accounts_for_late_and_unqueued_results_without_rewriting_history():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    msg, info = messages(1_000_000_000)
    frames.result('pose', msg, 1_200_000_000)
    frames.result('info', info, 1_260_000_001)  # expired while still pending
    frames.result('info', info, 1_270_000_000)  # already consumed
    assert len(fusion.frames) == 1 and fusion.frames[0][1] is None
    assert frames.counts['queued'] == frames.counts['completed'] == 1
    assert frames.counts['deadline_expired'] == 1
    assert frames.counts['pose_received'] == frames.counts['pose_joined'] == 1
    assert frames.counts['info_received'] == 2
    assert frames.counts['info_joined'] == 0
    assert frames.counts['info_after_deadline'] == 1
    assert frames.counts['info_unqueued'] == 1


def test_join_reports_imu_coverage_timeout_separately_from_missing_visual():
    fusion = Fusion(); fusion.samples = []
    frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    msg, info = messages(1_000_000_000)
    frames.result('pose', msg, 1_020_000_000)
    frames.result('info', info, 1_020_000_000)
    assert not fusion.frames
    frames.drain(1_260_000_000)
    assert fusion.frames[0][1] is not None
    assert frames.counts['imu_uncovered'] == frames.counts['deadline_expired'] == 1


def test_capture_backlog_and_repeated_frame_fail_closed():
    for stamps in ([1_000_000_000]*2, [1_000_000_000+i*10_000_000 for i in range(9)]):
        fusion = Fusion(); frames = CaptureObservations(fusion)
        for stamp in stamps:
            frames.add_frame(stamp, None, None, stamp+1_000_000)
        assert fusion.reason


def test_native_visual_rejects_bad_covariance_inliers_and_changed_frames():
    for mutate in (lambda m, i: setattr(i, 'inliers', 9),
                   lambda m, i: setattr(m.header, 'frame_id', 'new_segment'),
                   lambda m, i: m.pose.covariance.__setitem__(1, .1)):
        fusion = Fusion(); frames = CaptureObservations(fusion)
        frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
        m, i = messages(1_000_000_000); mutate(m, i)
        frames.result('pose', m, 1_020_000_000)
        frames.result('info', i, 1_020_000_000)
        assert fusion.frames[0][1] is None
        if m.header.frame_id == 'new_segment':
            assert fusion.reason  # resets cannot masquerade as short visual loss


def test_wired_fusion_mode_owns_imu_and_enables_geometry(monkeypatch):
    from plugins import rgbd_odometry as module
    class RecordingSession(Session):
        def __init__(self, calibration, executor, topic):
            super().__init__(calibration, executor, topic); self.calibration = calibration
    monkeypatch.setattr(module, 'RgbdSession', RecordingSession)
    monkeypatch.setattr(module, 'require_runtime', lambda: None)
    card = RgbdOdometryPlugin({}, _FakeExecutor())
    result = card.dispatch('rgbd_odometry', dict(action='start',
        input_topics=['/rgb', '/depth', '/imu'], camera_info=declarations()))
    assert result['state'] == 'running'
    calibration = card.instances['default'][0].calibration
    assert calibration['short_term_compensation'] and calibration['depth_geometry']
    params = estimator_parameters(calibration, 'run')
    assert not params.get('wait_imu_to_init', False)
    card.shutdown()


def test_late_native_diagnostic_is_not_accepted_between_watchdog_ticks():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_000_000_000)
    msg, info = messages(1_000_000_000)
    assert frames.result('pose', msg, 1_200_000_000) == []
    assert frames.result('info', info, 1_300_000_000) == [{'stamp': 1_000_000_000}]
    assert fusion.frames[0][1] is None


def test_camera_captures_expire_missing_native_results_before_queue_overflow():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    outputs = []
    for i in range(20):
        stamp = 1_000_000_000+round(i*1e9/30)
        outputs.extend(frames.add_frame(stamp, None, None, stamp) or [])
        if i%6 == 0:
            outputs.extend(frames.drain(stamp))
    assert fusion.reason is None
    assert len(outputs) >= 10
    assert all(visual is None for _, visual, _ in fusion.frames)


def test_native_worker_keeps_latest_pair_queued_until_rgbd_reader_matches(monkeypatch):
    import queue
    import sys
    import threading
    from plugins import rgbd_odometry_runtime as runtime

    monkeypatch.setitem(sys.modules, 'rtabmap_msgs.msg', NS(RGBDImage=NS))
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'CameraInfo', NS, raising=False)
    worker = object.__new__(runtime.RgbdSession)
    worker.stop_event = threading.Event()
    readers = [0]
    worker.rgbd = NS(get_subscription_count=lambda: readers[0])
    def discover(_):
        readers[0] = 1
    def dequeue(timeout):
        assert readers[0] == 1
        worker.stop_event.set()
        raise queue.Empty
    monkeypatch.setattr(worker.stop_event, 'wait', discover)
    worker.pending = NS(get=dequeue)
    worker._work()


def test_weak_cold_native_seed_uses_initialization_budget_not_map_recovery():
    from plugins.rgbd_odometry_runtime import EstimateGate

    gate = EstimateGate('run', 'camera_optical', 'capture')
    start = 1_000_000_000
    gate.input(start)
    msg, info = messages(start)
    info.inliers = 0  # The first native seed can legitimately report zero.
    assert gate.accept(msg, info, start) is None
    assert gate.status(start+1_050_000_000) == 'initializing'
    for i, offset in enumerate((1150, 1270, 1390)):
        stamp = start + offset*1_000_000
        gate.input(stamp)
        result = gate.accept(*messages(stamp), stamp)
        assert (result is not None) == (i == 2)


@pytest.mark.parametrize('frontend,strategy,cor_type', [
    ('f2m_feature', '0', '0'),
    ('f2f_optical_flow', '1', '1'),
])
def test_pinned_rtabmap_frontend_strategy_is_real(frontend, strategy, cor_type):
    parameters = estimator_parameters({
        'frame_id': 'camera_optical', 'visual_frontend': frontend,
        'max_features': 1500}, 'run')
    assert parameters['Odom/Strategy'] == strategy
    assert parameters['Vis/CorType'] == cor_type
    assert parameters['Vis/MaxFeatures'] == '1500'
    if frontend == 'f2f_optical_flow':
        assert parameters['Vis/CorFlowWinSize'] == '16'
        assert parameters['Vis/CorFlowMaxLevel'] == '3'
    else:
        assert 'Vis/CorFlowWinSize' not in parameters


def test_default_rtabmap_frontend_preserves_feature_matching():
    parameters = estimator_parameters({'frame_id': 'camera_optical'}, 'run')
    assert parameters['Odom/Strategy'] == '0'
    assert parameters['Vis/CorType'] == '0'
    assert parameters['Vis/MaxFeatures'] == '1000'
    assert parameters['Odom/ResetCountdown'] == '0'


@pytest.mark.parametrize('option', [
    {'visual_frontend': 'f2m_optical_flow'},
    {'visual_frontend': 'other'},
    {'max_features': True},
    {'max_features': 10},
    {'max_features': 10000},
])
def test_runtime_also_rejects_invalid_frontend_settings(option):
    with pytest.raises(ValueError):
        estimator_parameters({'frame_id': 'camera_optical', **option}, 'run')


def test_registration_report_does_not_turn_geometry_prediction_into_tracking():
    from plugins.rgbd_odometry_runtime import registration_report
    sample = {'vendor': {
        'motus_slam': {'status': 'coasting'},
        'fusion': {'aid': 'depth_geometry', 'quality': {
            'visual_usable': False, 'depth_fraction': .8, 'contrast': 5.,
            'sharpness': 2., 'reason': 'low_contrast',
            'geometry': {'attempted': True, 'accepted': True,
                         'inliers': 123, 'rmse': .008, 'overlap': .45,
                         'condition': 1234., 'reason': None}}}}}
    report = registration_report(
        sample, 'f2m_feature', {'inliers': 4, 'lost': True, 'latency_ms': 33.},
        depth_enabled=True)
    assert report['frontend'] == 'f2m_feature'
    assert report['status'] == 'coasting'
    assert report['native_inliers'] == 4
    assert report['native_lost'] is True
    assert report['capture_to_diagnostic_ms'] == 33.
    assert report['image_quality']['depth_fraction'] == .8
    assert report['depth_icp']['accepted'] is True
    assert report['depth_icp']['inliers'] == 123
    # These optional diagnostics cannot manufacture an admitted SLAM pose.
    assert 'pose' not in sample
    empty = registration_report({'vendor': {
        'motus_slam': {'status': 'lost'}}}, 'f2m_feature')
    assert empty['native_inliers'] is None
    assert empty['image_quality'] is None
    assert empty['depth_icp']['attempted'] is False


def test_native_registration_diagnostics_are_capture_stamped_and_bounded(monkeypatch):
    from plugins.rgbd_odometry_runtime import RgbdSession, registration_report
    from plugins import rgbd_odometry_runtime as runtime
    import threading

    session = object.__new__(RgbdSession)
    session.registration_by_stamp = OrderedDict()
    session.native_reports = 0
    first_stamp = 1_000_000_000
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: first_stamp + 35_000_000)
    msg, info = messages(first_stamp)
    session._remember_native(info)
    assert session.native_reports == 1
    native = session.registration_by_stamp.pop(first_stamp)
    assert native['inliers'] == 50 and native['lost'] is False
    assert native['latency_ms'] == 35.
    missing = registration_report({'vendor': {
        'motus_slam': {'status': 'coasting'}}},
        'f2m_feature', session.registration_by_stamp.pop(first_stamp, None))
    assert missing['native_inliers'] is None

    for index in range(50):
        stamp = first_stamp + index * 1_000_000
        _, info = messages(stamp)
        session._remember_native(info)
    assert len(session.registration_by_stamp) <= 32


@pytest.mark.parametrize('size', [(640, 480), (3, 2)])
@pytest.mark.parametrize('mode', ['visual', 'assisted', 'fusion'])
@pytest.mark.parametrize('color_offset_ns', [-800_000, 0, 800_000])
def test_native_rgbd_worker_preserves_encoder_bytes_in_typed_depth_payload(monkeypatch, mode, color_offset_ns, size):
    from array import array
    from copy import deepcopy
    import sys, threading, queue, zlib
    import numpy as np
    from plugins import rgbd_odometry_runtime as runtime
    obj = object.__new__(runtime.RgbdSession)
    obj.lock = threading.RLock(); obj.stop_event = threading.Event()
    obj.imu = obj.fusion = obj.observations = None
    obj.gate = runtime.EstimateGate('run', 'optical', 'capture-clock')
    width, height = size
    obj.calibration = dict(frame_id='optical', clock_id='capture-clock', K=np.eye(3),
                           color_size=size, depth_size=size)
    obj.counts = dict(pairs=0, estimates=0, dropped_pairs=0)
    header = NS(frame_id='producer-depth', stamp=NS(sec=1, nanosec=123))
    jpeg, png = b'camera-jpeg', b'encoded-png-depth'
    encoded = []
    def encode(format, image, params):
        encoded.append((format, image.copy(), params))
        if mode == 'assisted':
            elapsed[0] += .04  # IMU may arrive while image preparation runs.
        return True, np.frombuffer(png, np.uint8)
    cv2 = NS(IMREAD_COLOR=1, IMWRITE_PNG_COMPRESSION=16,
             imdecode=lambda data, mode: np.zeros((height, width, 3), np.uint8), imencode=encode)
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    depth = np.full((height, width), 1500, np.uint16)
    color_msg = NS(header=deepcopy(header), format='jpeg', data=jpeg)
    color_msg.header.frame_id = 'producer-color'
    color_stamp = 1_000_000_123+color_offset_ns
    color_msg.header.stamp.sec, color_msg.header.stamp.nanosec = divmod(color_stamp, 10**9)
    depth_msg = NS(header=deepcopy(header), data=zlib.compress(depth.tobytes(), 1))
    pair = ((color_stamp, color_msg), (1_000_000_123, depth_msg))
    pending = [pair]
    def get(timeout):
        if pending: return pending.pop()
        obj.stop_event.set(); raise queue.Empty
    obj.pending = NS(get=get)
    class Image:
        def __init__(self): self.header = NS()
        @property
        def data(self): return self._data
        @data.setter
        def data(self, value):
            # Model Humble's efficient array path, avoiding its per-byte loops.
            assert isinstance(value, array) and value.typecode == 'B'
            self._data = value
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', NS(CompressedImage=Image, CameraInfo=NS))
    monkeypatch.setitem(sys.modules, 'rtabmap_msgs.msg', NS(RGBDImage=NS))
    monkeypatch.setattr(runtime.time, 'time_ns', lambda: 1_001_000_123)
    published = []
    def publish(msg): published.append(msg); obj.stop_event.set()
    obj.rgbd = NS(publish=publish, get_subscription_count=lambda: 1)
    if mode == 'assisted':
        from test_rgbd_imu import aid, packet
        obj.imu = aid()
        obj.imu.observe(packet(990_000_000), 991_000_000)
        elapsed = [0.]
        monkeypatch.setattr(runtime.time, 'monotonic', lambda: elapsed[0])
        def wait(seconds): elapsed[0] += seconds
        monkeypatch.setattr(obj.stop_event, 'wait', wait)
        def drain():
            if elapsed[0] >= .04 and obj.imu.last_stamp == 990_000_000:
                obj.imu.observe(packet(1_010_000_000), 1_011_000_000)
        obj._drain_imu = drain
    if mode == 'fusion':
        obj.fusion = Fusion()
        obj.fusion.frame = 'optical'
        obj.observations = CaptureObservations(obj.fusion)
        obj._check_fusion = lambda now: None
        obj._publish = lambda sample: None
    coverage_stamps = []
    wait_for_imu = obj._wait_for_imu
    def wait(stamp, **kwargs):
        coverage_stamps.append(stamp)
        return wait_for_imu(stamp, **kwargs)
    obj._wait_for_imu = wait
    obj._work()
    assert obj.gate.state != 'lost', obj.gate.reason
    assert len(published) == obj.counts['pairs'] == 1
    assert obj.counts['dropped_pairs'] == 0
    native_stamp = max(color_stamp, 1_000_000_123)
    assert coverage_stamps == [native_stamp]
    if mode == 'assisted':
        assert elapsed[0] == .04  # No extra serial wait after IMU arrived during encoding.
    if mode == 'fusion':
        msg, info = messages(native_stamp)
        msg.child_frame_id = 'optical'
        obj.observations.result('pose', msg, 1_002_000_123)
        obj.observations.result('info', info, 1_002_000_123)
        assert len(obj.fusion.frames) == 1
        assert obj.fusion.frames[0][0] == native_stamp
        assert obj.fusion.frames[0][1] is not None
        assert obj.observations.counts['pose_joined'] == 1
        assert obj.observations.counts['info_joined'] == 1
    out = published[0]
    assert out.header.frame_id == out.depth_compressed.header.frame_id == 'optical'
    assert out.header.stamp == out.depth_compressed.header.stamp == header.stamp
    assert depth_msg.header.frame_id == 'producer-depth'
    assert color_msg.header.frame_id == 'producer-color'
    for info in (out.rgb_camera_info, out.depth_camera_info):
        assert info.header.frame_id == 'optical'
        assert (info.width, info.height) == size
    assert out.rgb_compressed.data == color_msg.data
    assert out.rgb_compressed.header.frame_id == 'optical'
    assert out.rgb_compressed.header.stamp == color_msg.header.stamp
    assert out.depth_compressed.format == 'png'
    assert bytes(out.depth_compressed.data) == png
    assert encoded[0][0] == '.png' and encoded[0][2] == [cv2.IMWRITE_PNG_COMPRESSION, 1]
    np.testing.assert_array_equal(encoded[0][1], depth)


def test_slow_native_join_accepts_a_measured_pair_after_250ms_in_original_order():
    fusion = Fusion()
    fusion.input_period_ns = 350_000_000
    frames = CaptureObservations(fusion)
    stamp = 1_000_000_000
    frames.add_frame(stamp, None, None, stamp+120_000_000)
    msg, info = messages(stamp)
    assert frames.result('pose', msg, stamp+590_000_000) == []
    assert frames.result('info', info, stamp+610_000_000) == [{'stamp': stamp}]
    assert fusion.frames[0][1] is not None

def test_startup_can_fall_back_to_repeated_measured_visual_before_the_full_timeout():
    from plugins.rgbd_odometry_runtime import RgbdSession, EstimateGate
    import time
    now = time.time_ns()
    session = RgbdSession.__new__(RgbdSession)
    session.started, session.segment = 0., 'run'
    session.imu = NS(state='ready')
    session.fallback = None
    session.fusion = Fusion()
    session.fusion.filter = None
    session._check_fusion = lambda now: None
    session.gate = EstimateGate('run', 'camera_optical', 'clock')
    session.gate.input(now)
    session.observations = CaptureObservations(session.fusion, startup_gate=EstimateGate('run', 'camera_optical', 'clock'))
    for i in range(3):
        stamp = now-500_000_000+i*200_000_000
        session.fusion.samples = [(stamp, None, None)]
        session.observations.add_frame(stamp, None, None, stamp)
        pose, info = messages(stamp)
        session.observations.result('pose', pose, stamp)
        session.observations.result('info', info, stamp)
    session._check_initialization(1.)
    assert session.fusion.reason is None
    session._check_initialization(6.)
    assert 'measured visual' in session.fusion.reason
