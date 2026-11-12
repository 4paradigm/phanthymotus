"""Transient visual degradation, persistent loss and ESKF visual-only fallback.

Real RgbdSession/EstimateGate with ROS and native process boundaries stubbed.
These are logic tests; they do not establish physical tracking behaviour.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from plugins import rgbd_odometry_runtime as runtime
from plugins.rgbd_odometry_contract import validate_start_inputs
from test_rgbd_odometry import messages
from test_rgbd_odometry_contract import declarations
from test_rgbd_odometry_recovery import assisted_runtime, images, recovery  # noqa: F401
from test_rgbd_odometry_status import sessions, status  # noqa: F401
from vision_stubs import _FakeExecutor

MS = 1_000_000


def native(session, stamp, x=0., lost=False, cov=.001, inliers=50, frame=None):
    msg, info = messages(stamp)
    msg.header.frame_id = frame or session.segment + '_odom'
    msg.pose.pose.position.x = x
    msg.pose.covariance = [cov if i % 7 == 0 else 0. for i in range(36)]
    info.lost, info.inliers = lost, inliers
    return msg, info


def feed(session, now, stamp, **kw):
    now[0] = stamp + 20*MS
    msg, info = native(session, stamp, **kw)
    session._result('pose', msg)
    session._result('info', info)


def track(session, now, start=1_000_000_000, x=0.):
    for i in range(3):
        feed(session, now, start + i*100*MS, x=x)
    assert session.gate.state == 'tracking'
    return start + 200*MS


def last(session):
    return json.loads(session.publisher.messages[-1])


# ── C: pure RGB-D ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('bad', [dict(lost=True), dict(cov=9999.), dict(inliers=3)])
def test_single_bad_estimate_pauses_output_keeps_feeding_and_recovers_same_segment(sessions, bad):
    create, now, ticks = sessions
    s = create()
    stamp = track(s, now)
    count = len(s.publisher.messages)
    feed(s, now, stamp + 100*MS, **bad)
    assert s.gate.state == 'recovering' and len(s.publisher.messages) == count
    ticks[0] += 1
    s._watchdog()
    assert status(s)['status'] == 'recovering' and status(s)['position_m'] is None
    images(s, now[0])  # the native estimator keeps receiving frames
    assert s.pending.qsize() == 1
    for k in (2, 3):
        feed(s, now, stamp + k*100*MS)
        assert s.gate.state == 'recovering'
    feed(s, now, stamp + 400*MS, x=.01)
    assert s.gate.state == 'tracking'
    sample = last(s)
    assert sample['vendor']['motus_slam']['segment_id'] == s.segment
    assert sample['pose']['position'] == [.01, 0., 0.]


def test_persistent_degradation_retires_segment_and_recovers_into_new_one(recovery):
    card, _, now, seconds = recovery
    old = card.instances['one'][0]
    stamp = track(old, now)
    for k in range(1, 18):
        feed(old, now, stamp + k*100*MS, lost=True)
    assert old.gate.state == 'lost' and old.gate.loss_kind == 'visual'
    assert 'restart to begin a new segment' in old.gate.reason
    old.process.returncode = 0  # stubbed native process exits on stop
    images(old, now[0]); card._recover()
    seconds[0] += 1.; images(old, now[0]); card._recover()
    new = card.instances['one'][0]
    new.process.returncode = 0
    assert new is not old and new.segment != old.segment


def test_discontinuous_resume_is_not_spliced_into_the_segment(sessions):
    create, now, _ = sessions
    s = create()
    stamp = track(s, now)
    feed(s, now, stamp + 100*MS, lost=True)
    count = len(s.publisher.messages)
    feed(s, now, stamp + 200*MS, x=5.)
    assert s.gate.state == 'lost' and 'discontinu' in s.gate.reason
    assert len(s.publisher.messages) == count


def test_changed_native_frame_is_a_hard_error(sessions):
    create, now, _ = sessions
    s = create()
    stamp = track(s, now)
    feed(s, now, stamp + 100*MS, frame='other_odom')
    assert s.gate.state == 'lost' and s.gate.loss_kind == 'frame'


def test_out_of_order_estimate_is_ignored_without_counting_as_good(sessions):
    create, now, _ = sessions
    s = create()
    stamp = track(s, now)
    good = s.gate.good
    feed(s, now, stamp - 50*MS)
    assert s.gate.state == 'tracking' and s.gate.good == good
    assert last(s)['vendor']['motus_slam']['stamp_ns'] == stamp


def test_repeated_or_stale_image_is_dropped_not_fatal(sessions):
    create, now, _ = sessions
    s = create()
    images(s, now[0])
    images(s, now[0])          # repeated capture stamps
    images(s, now[0] - 2_000_000_000)  # older than the pairing window
    assert s.gate.state != 'lost'
    assert s.counts['rejected_inputs'] == 4
    assert s.info()['input_rejections']['last_reason']


# ── B: ESKF/aid failure degrades to independent visual output ───────────────

@pytest.fixture
def fused(sessions, monkeypatch):
    create, now, ticks = sessions
    assisted_runtime(monkeypatch)
    made = []

    def make(depth_geometry=False):
        calibration = validate_start_inputs(declarations(), ['/rgb', '/depth', '/imu'], True)
        calibration.update(imu_assisted=True, short_term_compensation=True, depth_geometry=depth_geometry,
                           visual_frontend='f2m_feature', max_features=1000)
        session = runtime.RgbdSession(calibration, _FakeExecutor(), '/fused/state/odom')
        made.append(session)
        return session
    yield make, now, ticks
    for session in made:
        session.process.returncode = 0
        if not session.node.destroyed:
            session.close()


def imu(session, stamp):
    from test_rgbd_imu import packet
    session._imu(NS(data=json.dumps(packet(
        stamp, camera_frame_id='camera_optical', clock_id='camera-global',
        camera_from_gyro_translation=[.03, 0, 0], camera_from_accel_translation=[.03, 0, 0],
        translation_source='sdk_factory_extrinsics'))))


def test_stale_imu_degrades_to_visual_only_with_explicit_provenance(fused):
    make, now, ticks = fused
    s = make()
    now[0] = 1_000_000_000
    imu(s, 999_000_000)
    now[0] = 1_700_000_000  # IMU now >500 ms old: ESKF must stop
    s._watchdog()
    assert s.fusion.state == 'lost' and 'IMU' in s.fusion.reason
    assert s.gate.state != 'lost', s.gate.reason
    assert s.fallback['reason'].startswith('IMU')
    for k in range(3):
        feed(s, now, 1_800_000_000 + k*100*MS)
    sample = last(s)
    assert sample['vendor']['motus_slam']['status'] == 'tracking'
    assert sample['vendor']['source'] == 'rtabmap_rgbd_odometry'
    assert sample['vendor']['fusion']['status'] == 'lost'
    assert sample['vendor']['fallback'] == 'visual_only'
    assert 'visual-only' in sample['vendor']['reason']
    assert status(s)['status'] == 'tracking'
    info = s.info()
    assert info['status'] == 'tracking' and info['fallback']['mode'] == 'visual_only'


def track_fused(s, now, offset=0.):
    """Publish one fused tracking pose and record its same-capture visual pose."""
    fused = dict(position=[0., 0., 0.], quaternion=[0., 0., 0., 1.], covariance=[0.]*36)
    visual = dict(position=[offset, 0., 0.], quaternion=[0., 0., 0., 1.], covariance=[0.]*36)
    s.fusion.last_tracking = (1_500_000_000, fused, visual)
    s.fusion.last_visual = 1_500_000_000
    s.fusion.fault('modeled uncertainty test fault')
    now[0] = 1_520_000_000
    s._watchdog()


def test_fallback_without_proven_continuity_starts_a_new_segment(fused):
    make, now, _ = fused
    s = make()
    track_fused(s, now, offset=.05)
    assert s.gate.state == 'lost' and 'continuity' in s.gate.reason
    assert s.fallback is None


def test_fallback_with_proven_continuity_keeps_segment_and_checks_resume(fused):
    make, now, _ = fused
    s = make()
    track_fused(s, now, offset=.005)
    assert s.gate.state == 'recovering' and s.fallback is not None
    feed(s, now, 1_600_000_000, x=2.)  # cannot jump 2 m in 100 ms
    assert s.gate.state == 'lost' and 'discontinu' in s.gate.reason


def test_fallback_that_never_tracks_is_bounded(fused):
    make, now, _ = fused
    s = make()
    track_fused(s, now)
    for k in range(1, 20):
        feed(s, now, 1_500_000_000 + k*100*MS, lost=True)
    assert s.gate.state == 'lost'


def test_fallback_does_not_resume_fusion_or_publish_predictions(fused):
    make, now, _ = fused
    s = make()
    track_fused(s, now)
    for k in range(3):
        feed(s, now, 1_600_000_000 + k*100*MS)
    assert s.fusion.state == 'lost'
    assert all(json.loads(m)['vendor'].get('source') != 'rgbd_eskf' or json.loads(m)['pose'] is None
               for m in s.publisher.messages)
    assert not s.observations.frames
