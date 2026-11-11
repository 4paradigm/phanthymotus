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
    def fault(self, reason):
        self.reason = reason
    def observe_frame(self, stamp, color, depth, visual, now, reason=None):
        self.frames.append((stamp, visual, reason))
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


def test_lost_diagnostic_can_trigger_coast_without_matching_pose():
    fusion = Fusion(); frames = CaptureObservations(fusion)
    frames.add_frame(1_000_000_000, None, None, 1_010_000_000)
    _, info = messages(1_000_000_000); info.lost = True
    assert frames.result('info', info, 1_020_000_000) == [{'stamp': 1_000_000_000}]
    assert fusion.frames[0][1] is None


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


def test_fusion_mode_owns_imu_and_geometry_requires_compensation(monkeypatch):
    from plugins import rgbd_odometry as module
    class RecordingSession(Session):
        def __init__(self, calibration, executor, topic):
            super().__init__(calibration, executor, topic); self.calibration = calibration
    monkeypatch.setattr(module, 'RgbdSession', RecordingSession)
    monkeypatch.setattr(module, 'require_runtime', lambda: None)
    card = RgbdOdometryPlugin({}, object())
    props = card.get_tools()[0]['configSchema']['properties']
    assert props['short_term_compensation']['default'] is False
    assert props['depth_geometry']['default'] is False
    assert card.dispatch('rgbd_odometry', dict(action='config', depth_geometry=True))['state'] == 'error'
    assert card.dispatch('rgbd_odometry', dict(action='config', short_term_compensation=True))['state'] == 'error'
    result = card.dispatch('rgbd_odometry', dict(action='start', imu_assisted=True,
        short_term_compensation=True, depth_geometry=True,
        input_topics=['/rgb', '/depth', '/imu'], camera_info=declarations()))
    assert result['state'] == 'running'
    calibration = card.instances['default'][0].calibration
    assert calibration['short_term_compensation'] and calibration['depth_geometry']
    params = estimator_parameters(calibration, 'run')
    assert not params.get('wait_imu_to_init', False)
    assert card.dispatch('rgbd_odometry', dict(action='config', depth_geometry=False))['state'] == 'error'
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
