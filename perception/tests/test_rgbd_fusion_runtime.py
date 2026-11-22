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
        self.filter = None
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
    deadline = frames.frames[1_000_000_000]['deadline']
    assert frames.drain(deadline) == [{'stamp': 1_000_000_000}]
    assert fusion.frames[0][1] is None
    m, i = messages(1_000_000_000)
    frames.result('pose', m, deadline+1)
    frames.result('info', i, deadline+1)
    assert len(fusion.frames) == 1


def test_transport_late_capture_does_not_feed_fusion_or_block_the_next_capture():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_900_000_000)
    assert frames.drain(2_400_000_000) == []
    assert not fusion.frames and not frames.frames
    assert frames.counts['stale_capture'] == 1
    stamp = 2_410_000_000
    frames.add_frame(stamp, None, None, stamp+10_000_000)
    msg, info = messages(stamp)
    frames.result('pose', msg, stamp+20_000_000)
    assert frames.result('info', info, stamp+25_000_000) == [{'stamp': stamp}]
    assert fusion.frames[0][1] is not None and fusion.reason is None


@pytest.mark.parametrize('complete,delay,accepted', [
    (True, 1_105_000_000, True),
    (False, 1_300_000_000, False),
    (True, 1_500_000_001, False),
])
def test_submitted_capture_can_use_a_late_exact_visual_pair(complete, delay, accepted):
    fusion = Fusion()
    frames = CaptureObservations(fusion)
    fusion.input_period_ns = 400_000_000
    stamp = 1_000_000_000
    frames.add_frame(stamp, None, None, stamp+500_000_000)
    pose, info = messages(stamp)
    frames.result('pose', pose, stamp+delay-1)
    if complete:
        frames.result('info', info, stamp+delay)
    else:
        frames.drain(stamp+delay)
    assert [frame[0] for frame in fusion.frames] == ([stamp] if accepted else [])
    if accepted:
        assert fusion.frames[0][1] is not None


def test_capture_enqueued_promptly_can_use_a_late_exact_visual_pair():
    fusion = Fusion()
    fusion.input_period_ns = 600_000_000
    frames = CaptureObservations(fusion)
    stamp = 1_000_000_000
    frames.add_frame(stamp, None, None, stamp+10_000_000)
    pose, info = messages(stamp)
    frames.result('pose', pose, stamp+1_104_000_000)
    frames.result('info', info, stamp+1_105_000_000)
    assert len(fusion.frames) == 1 and fusion.frames[0][1] is not None


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
    frames.drain(1_510_000_000)
    assert fusion.frames == [(1_000_000_000, None, reason)]
    assert frames.counts['native_tracking_lost'] == int(lost)
    assert frames.counts['visual_result_missing'] == int(not lost and len(kinds) != 2)


def test_join_accounts_for_late_and_unqueued_results_without_rewriting_history():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    msg, info = messages(1_000_000_000)
    frames.result('pose', msg, 1_200_000_000)
    frames.result('info', info, 1_510_000_001)  # expired while still pending
    frames.result('info', info, 1_520_000_000)  # already consumed
    frames.result('info', messages(1_010_000_000)[1], 1_521_000_000)  # never submitted
    assert len(fusion.frames) == 1 and fusion.frames[0][1] is None
    assert frames.counts['queued'] == frames.counts['completed'] == 1
    assert frames.counts['deadline_expired'] == 1
    assert frames.counts['pose_received'] == frames.counts['pose_joined'] == 1
    assert frames.counts['info_received'] == 3
    assert frames.counts['info_joined'] == 0
    assert frames.counts['info_after_deadline'] == 2
    assert frames.counts['info_unqueued'] == 1


def test_join_reports_imu_coverage_timeout_separately_from_missing_visual():
    fusion = Fusion(); fusion.samples = []
    frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    msg, info = messages(1_000_000_000)
    frames.result('pose', msg, 1_020_000_000)
    frames.result('info', info, 1_020_000_000)
    assert fusion.frames[0][1] is not None
    assert frames.counts['imu_uncovered'] == 1 and frames.counts['deadline_expired'] == 0


def test_external_learning_only_receives_accepted_native_visual_pose():
    from test_rgbd_fusion import images, source, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda *values: forwarded.append(values)
    color, depth = images()
    for offset in range(5_000_000, 100_000_001, 5_000_000):
        fusion.observe_imu(source(anchor+offset), anchor+offset+1_000_000)
    for offset, attitude in ((50_000_000, (0., 1.)), (100_000_000, (1., 0.))):
        stamp = anchor+offset
        pose, info = messages(stamp)
        pose.child_frame_id = fusion.frame
        pose.pose.pose.orientation.z, pose.pose.pose.orientation.w = attitude
        captures.add_frame(stamp, color, depth, stamp+10_000_000)
        captures.result('pose', pose, stamp+20_000_000)
        captures.result('info', info, stamp+20_000_000)
    assert len(forwarded) == 1
    stamp, segment, observation = forwarded[0]
    assert (stamp, segment) == (anchor+50_000_000, fusion.segment)
    assert observation['quaternion'] == [0., 0., 0., 1.]


def test_external_native_visual_is_delivered_once_without_imu_and_recovers_from_bad_images():
    from test_rgbd_fusion import images, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda *values: forwarded.append(values)
    color, depth = images()

    for step in range(1, 6):
        stamp = anchor + step*100_000_000
        pose, info = messages(stamp)
        pose.child_frame_id = fusion.frame
        pose.pose.pose.position.x = step*.01
        captures.add_frame(stamp, color if step != 2 else color*0, depth, stamp)
        captures.result('pose', pose, stamp+10_000_000)
        captures.result('info', info, stamp+20_000_000)
        captures.result('info', info, stamp+21_000_000)
        # A missing IMU bracket can keep this capture queued for its own deadline.
        if step == 1:
            assert [item[0] for item in forwarded] == [stamp]
    for step in range(1, 6):
        captures.drain(anchor+650_000_000+step*100_000_000)
    assert [(stamp, p['position'][0]) for stamp, _, p in forwarded] == [
        (anchor+100_000_000, .01), (anchor+500_000_000, .05)]
    assert fusion.filter.stamp == anchor


@pytest.mark.parametrize('invalid', ['native_lost', 'few_inliers', 'covariance', 'unmatched_stamp'])
def test_uncovered_native_visual_keeps_quality_and_exact_join_requirements(invalid):
    from test_rgbd_fusion import images, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda *values: forwarded.append(values)
    stamp = anchor + 50_000_000
    pose, info = messages(stamp)
    pose.child_frame_id = fusion.frame
    if invalid == 'native_lost':
        info.lost = True
    elif invalid == 'few_inliers':
        info.inliers = 9
    elif invalid == 'covariance':
        pose.pose.covariance[0] = -.01
    else:
        _, info = messages(stamp+1)
    captures.add_frame(stamp, *images(), stamp)
    captures.result('pose', pose, stamp+10_000_000)
    captures.result('info', info, stamp+20_000_000)
    captures.drain(stamp+800_000_000)
    assert not forwarded
    assert fusion.filter.stamp == anchor


def test_native_visual_admission_does_not_depend_on_the_eskf_innovation(monkeypatch):
    from test_rgbd_fusion import images, source, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda *values: forwarded.append(values)
    monkeypatch.setattr(fusion.filter, 'correct', lambda *args, **kwargs: False)
    stamp = anchor + 50_000_000
    for t in range(anchor+5_000_000, stamp+1, 5_000_000):
        fusion.observe_imu(source(t), t+1_000_000)
    pose, info = messages(stamp)
    pose.child_frame_id = fusion.frame
    pose.pose.pose.position.x = .01
    captures.add_frame(stamp, *images(), stamp)
    captures.result('pose', pose, stamp+10_000_000)
    output, = captures.result('info', info, stamp+20_000_000)
    assert output['pose'] is None
    assert [(t, p['position'][0]) for t, _, p in forwarded] == [(stamp, .01)]


def test_external_native_visual_initializes_without_imu():
    from test_rgbd_fusion import images, K
    from plugins.rgbd_fusion import ShortTermFusion

    fusion = ShortTermFusion('run', 'camera_optical', 'clock', K)
    captures = CaptureObservations(fusion)
    forwarded = []
    captures.visual_sink = lambda *values: forwarded.append(values)
    color, depth = images()
    for step in range(3):
        stamp = 1_000_000_000 + step*100_000_000
        pose, info = messages(stamp)
        captures.add_frame(stamp, color, depth, stamp)
        captures.result('info', info, stamp+10_000_000)
        captures.result('pose', pose, stamp+20_000_000)
    assert [item[0] for item in forwarded] == [stamp]
    assert fusion.filter is None


@pytest.mark.parametrize('imu_resumes', ['same_capture', 'after_grace', 'next_capture', 'never'])
def test_uncovered_native_result_does_not_end_running_fusion_before_watchdog(imu_resumes):
    from test_rgbd_fusion import images, source, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    captures.native_delays.extend([25_000_000] * 4)
    color, depth = images()
    gap = anchor + 50_000_000
    pose, info = messages(gap)
    pose.child_frame_id = fusion.frame
    captures.add_frame(gap, color, depth, gap + 10_000_000)
    assert captures.result('pose', pose, gap + 20_000_000) == []
    assert captures.result('info', info, gap + 20_000_000) == []
    deadline = captures.frames[gap]['deadline']
    assert captures.drain(deadline) == []
    assert gap in captures.frames  # a complete visual result can wait for late IMU
    if imu_resumes in ('same_capture', 'after_grace'):
        now = deadline + (60_000_000 if imu_resumes == 'same_capture' else 160_000_000)
        for stamp in range(anchor + 5_000_000, gap + 1, 5_000_000):
            fusion.observe_imu(source(stamp), now)
        outputs = captures.drain(now)
        if imu_resumes == 'after_grace':
            assert outputs == []
            assert fusion.filter.stamp == anchor
            assert not captures.frames
            return
        sample, = outputs
        assert sample['pose'] is not None
        assert sample['vendor']['fusion']['last_visual_ns'] == gap
        return

    deadline += 150_000_000
    assert captures.drain(deadline) == []
    assert captures.counts['imu_uncovered'] == 1
    assert captures.result('info', info, deadline + 1) == []
    assert captures.counts['info_after_deadline'] == 1
    assert fusion.filter.stamp == anchor
    assert fusion.status(deadline) == 'tracking'

    if imu_resumes == 'never':
        lost = fusion.watchdog_output(anchor + 1_010_000_000)
        assert fusion.state == 'lost'
        assert lost['pose'] is None
        assert lost['vendor']['fusion']['prediction'] is None
        return

    next_stamp = anchor + 200_000_000
    for stamp in range(anchor + 5_000_000, next_stamp + 1, 5_000_000):
        fusion.observe_imu(source(stamp), deadline + 5_000_000)
    pose, info = messages(next_stamp)
    pose.child_frame_id = fusion.frame
    captures.add_frame(next_stamp, color, depth, deadline + 10_000_000)
    assert captures.result('pose', pose, deadline + 20_000_000) == []
    sample, = captures.result('info', info, deadline + 20_000_000)
    assert sample['pose'] is not None
    assert sample['vendor']['motus_slam']['segment_id'] == fusion.segment
    assert sample['vendor']['fusion']['last_visual_ns'] == next_stamp


def test_late_imu_wait_cannot_extend_visual_capture_age_limit():
    fusion = Fusion()
    fusion.filter = object()
    fusion.samples = []
    fusion.input_period_ns = 400_000_000
    captures = CaptureObservations(fusion)
    stamp = 1_000_000_000
    captures.add_frame(stamp, None, None, stamp + 900_000_000)
    pose, info = messages(stamp)
    captures.result('pose', pose, stamp + 1_490_000_000)
    captures.result('info', info, stamp + 1_490_000_000)
    assert captures.drain(stamp + 1_500_000_000) == []
    assert not captures.frames and not fusion.frames


def test_repeated_frame_fails_closed():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_001_000_000)
    frames.add_frame(1_000_000_000, None, None, 1_002_000_000)
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
    import sys
    class RecordingSession(Session):
        def __init__(self, calibration, executor, topic):
            super().__init__(calibration, executor, topic); self.calibration = calibration
    monkeypatch.setattr(module, 'RgbdSession', RecordingSession)
    monkeypatch.setattr(module, 'require_runtime', lambda: None)
    monkeypatch.setattr(sys.modules['rclpy.qos'], 'qos_profile_sensor_data', object(), raising=False)
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
    assert frames.result('info', info, 1_500_000_001) == [{'stamp': 1_000_000_000}]
    assert fusion.frames[0][1] is None


def test_fast_camera_can_join_a_400ms_native_result_during_startup():
    fusion = Fusion()
    fusion.input_period_ns = 33_333_333
    frames = CaptureObservations(fusion)
    stamp = 1_000_000_000
    frames.add_frame(stamp, None, None, stamp)
    pose, info = messages(stamp)
    assert frames.result('pose', pose, stamp+390_000_000) == []
    assert frames.result('info', info, stamp+400_000_000) == [{'stamp': stamp}]
    assert fusion.frames[0][1] is not None


def test_full_native_join_does_not_change_session_or_evict_submitted_captures():
    fusion = Fusion()
    frames = CaptureObservations(fusion)
    stamps = [1_000_000_000+i*10_000_000 for i in range(8)]
    for stamp in stamps:
        frames.add_frame(stamp, None, None, stamp)
    frames.add_frame(stamps[-1]+10_000_000, None, None, stamps[-1]+10_000_000)
    assert list(frames.frames) == stamps
    assert fusion.reason is None


def test_native_wait_adapts_when_a_warmed_estimator_slows_down():
    fusion = Fusion()
    fusion.input_period_ns = 33_333_333
    frames = CaptureObservations(fusion)
    for stamp in range(1_000_000_000, 1_200_000_000, 50_000_000):
        frames.add_frame(stamp, None, None, stamp)
        pose, info = messages(stamp)
        frames.result('pose', pose, stamp+20_000_000)
        frames.result('info', info, stamp+30_000_000)
    stamp = 1_200_000_000
    frames.add_frame(stamp, None, None, stamp)
    assert frames.frames[stamp]['deadline'] == stamp+250_000_000
    frames.drain(stamp+250_000_000)
    stamp = 1_500_000_000
    frames.add_frame(stamp, None, None, stamp)
    pose, info = messages(stamp)
    assert not frames.result('pose', pose, stamp+390_000_000)
    assert frames.result('info', info, stamp+400_000_000) == [{'stamp': stamp}]
    assert fusion.frames[-1][1] is not None


def test_camera_captures_expire_missing_native_results_before_queue_overflow():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    outputs = []
    for i in range(20):
        stamp = 1_000_000_000+round(i*1e9/30)
        outputs.extend(frames.add_frame(stamp, None, None, stamp) or [])
        assert len(frames.frames) <= 8
    outputs.extend(frames.drain(2_700_000_000))
    assert fusion.reason is None
    assert not frames.frames
    assert len(outputs)+frames.counts['stale_capture'] == frames.counts['queued']
    assert outputs and frames.counts['stale_capture'] > 0
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
        assert parameters['Vis/CorFlowPyramidWithGuess'] == 'true'
    else:
        assert 'Vis/CorFlowWinSize' not in parameters
        assert 'Vis/CorFlowPyramidWithGuess' not in parameters


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
@pytest.mark.parametrize('mode', ['visual', 'assisted', 'fusion', 'recovering'])
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
             IMWRITE_PNG_STRATEGY=17, IMWRITE_PNG_STRATEGY_RLE=3,
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
    def publish(msg):
        published.append(msg)
        if mode != 'recovering':
            obj.stop_event.set()
    obj.rgbd = NS(publish=publish, get_subscription_count=lambda: 1)
    if mode == 'recovering':
        # Drop this probe's diagnostic. The worker still owes the old map
        # another real frame before the frozen verification deadline.
        obj.gate.result_period_ns = 120_000_000
        obj.gate.processing_p95_ns = 350_000_000
        obj.gate.relocalization_deadline_mono = 3_000_000_000
        monkeypatch.setattr(runtime.time, 'monotonic_ns', lambda: 1_000_000_000)
        waits = []
        obj.probe_ack = NS(clear=lambda: None, wait=waits.append)
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
    if mode == 'recovering':
        assert waits == [pytest.approx(.35)]
    assert obj.counts['dropped_pairs'] == 0
    native_stamp = max(color_stamp, 1_000_000_123)
    assert coverage_stamps == [native_stamp]
    assert obj.quality_by_stamp[native_stamp].depth_fraction == pytest.approx(1.)
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
    assert encoded[0][0] == '.png'
    assert encoded[0][2] == [cv2.IMWRITE_PNG_COMPRESSION, 1,
                             cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]
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

@pytest.mark.parametrize('result_age', [100_000_000, 1_105_000_000])
def test_startup_can_fall_back_to_repeated_measured_visual_before_the_full_timeout(result_age):
    from plugins.rgbd_odometry_runtime import RgbdSession, EstimateGate
    import time
    now = time.time_ns()
    session = RgbdSession.__new__(RgbdSession)
    session.started, session.segment = 0., 'run'
    session.imu = NS(state='ready')
    session.fallback = None
    session.fusion = Fusion()
    session.fusion.filter = None
    session.fusion.input_period_ns = 350_000_000
    session._check_fusion = lambda now: None
    session.gate = EstimateGate('run', 'camera_optical', 'clock')
    session.gate.input(now)
    session.observations = CaptureObservations(session.fusion, startup_gate=EstimateGate('run', 'camera_optical', 'clock'))
    for i in range(3):
        stamp = now-result_age-(2-i)*200_000_000
        session.fusion.samples = [(stamp, None, None)]
        session.observations.add_frame(stamp, None, None, stamp+min(result_age, 500_000_000))
        pose, info = messages(stamp)
        session.observations.result('pose', pose, stamp+result_age)
        session.observations.result('info', info, stamp+result_age)
    session._check_initialization(1.)
    assert session.fusion.reason is None
    session._check_initialization(6.)
    assert 'measured visual' in session.fusion.reason


@pytest.mark.parametrize('through_session', [False, True])
@pytest.mark.parametrize('with_sink', [False, True])
def test_native_frame_fault_waits_for_earlier_eskf_correction(monkeypatch, through_session, with_sink):
    from threading import Event, Lock
    from plugins.rgbd_odometry_runtime import EstimateGate, RgbdSession
    from test_rgbd_fusion import images, source, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    session = RgbdSession.__new__(RgbdSession)
    session.fusion, session.observations = fusion, captures
    session.gate = EstimateGate(fusion.segment, fusion.frame, fusion.clock)
    session.lock, session.stop_event = Lock(), Event()
    forwarded = []
    if with_sink:
        captures.visual_sink = lambda stamp, *_: forwarded.append(stamp)
    good, bad = anchor+50_000_000, anchor+100_000_000
    color, depth = images()
    for stamp in (good, bad):
        now = stamp+10_000_000
        monkeypatch.setattr('time.time_ns', lambda: now)
        captures.add_frame(stamp, color, depth, now)
        msg, info = messages(stamp)
        msg.child_frame_id = fusion.frame if stamp == good else 'wrong_camera'
        msg.pose.pose.position.x = .01
        for kind, value in [('pose', msg), ('info', info)]:
            if through_session:
                session._result(kind, value)
            else:
                assert captures.result(kind, value, now) == []
    assert list(captures.frames) == [good, bad]
    assert fusion.state == 'tracking'
    if with_sink:
        assert forwarded == [good]
        assert captures.external_gate.loss_kind == 'frame'

    for t in range(anchor+5_000_000, good+1, 5_000_000):
        fusion.observe_imu(source(t), now)
    output, = captures.drain(now)
    assert output['pose'] is not None
    assert fusion.last_visual == good
    assert list(captures.frames) == [bad]
    for t in range(good+5_000_000, bad+1, 5_000_000):
        fusion.observe_imu(source(t), now)
    output, = captures.drain(now)
    assert output['pose'] is None
    assert fusion.state == 'lost' and fusion.loss_kind == 'frame'
    assert fusion.last_visual == good


def test_earlier_correction_is_publishable_when_imu_covers_both_queued_captures():
    from test_rgbd_fusion import images, source, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    captures.visual_sink = lambda *_: None
    good, bad = anchor+50_000_000, anchor+100_000_000
    color, depth = images()
    for stamp in (good, bad):
        pose, info = messages(stamp)
        pose.child_frame_id = fusion.frame if stamp == good else 'wrong_camera'
        captures.add_frame(stamp, color, depth, stamp)
        captures.result('pose', pose, stamp)
        captures.result('info', info, stamp)
    for t in range(anchor+5_000_000, bad+1, 5_000_000):
        fusion.observe_imu(source(t), bad)
    # RgbdSession publishes each drain's samples under the current ESKF state.
    earlier = captures.drain(bad)
    assert fusion.state == 'tracking'
    assert len(earlier) == 1 and earlier[0]['pose'] is not None
    rejected, = captures.drain(bad)
    assert rejected['pose'] is None and fusion.loss_kind == 'frame'


@pytest.mark.parametrize('through_session', [False, True])
def test_wrong_frame_duplicate_stops_later_native_delivery(monkeypatch, through_session):
    from threading import Event, Lock
    from plugins.rgbd_odometry_runtime import EstimateGate, RgbdSession
    from test_rgbd_fusion import images, source, warmed

    fusion, anchor = warmed()
    captures = CaptureObservations(fusion)
    session = RgbdSession.__new__(RgbdSession)
    session.fusion, session.observations = fusion, captures
    session.gate = EstimateGate(fusion.segment, fusion.frame, fusion.clock)
    session.lock, session.stop_event = Lock(), Event()
    forwarded = []
    captures.visual_sink = lambda stamp, *_: forwarded.append(stamp)
    earlier, duplicate, later = (anchor+i*1_000_000 for i in (25, 50, 100))
    color, depth = images()
    monkeypatch.setattr('time.time_ns', lambda: now)

    def receive(kind, msg):
        if through_session:
            session._result(kind, msg)
        else:
            assert captures.result(kind, msg, now) == []

    for stamp in (earlier, duplicate, later):
        now = stamp+1_000_000
        captures.add_frame(stamp, color, depth, now)
        msg, info = messages(stamp)
        msg.child_frame_id = fusion.frame
        receive('pose', msg)
        receive('info', info)
        if stamp == duplicate:
            assert forwarded == [earlier, duplicate]
            wrong, _ = messages(stamp)
            wrong.child_frame_id = 'wrong_camera'
            receive('pose', wrong)
            assert captures.external_gate.loss_kind == 'frame'
    assert forwarded == [earlier, duplicate]
    assert captures.external_gate.loss_kind == 'frame'
    assert fusion.state == 'tracking' and fusion.filter.stamp == anchor

    for t in range(anchor+5_000_000, duplicate+1, 5_000_000):
        fusion.observe_imu(source(t), now)
    corrected, = captures.drain(now)
    assert corrected['pose'] is not None and fusion.last_visual == earlier
    rejected, = captures.drain(now)
    assert rejected['pose'] is None and fusion.loss_kind == 'frame'
    assert forwarded == [earlier, duplicate]
