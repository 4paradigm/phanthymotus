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
    assert s.gate.state == 'recovering'
    barriers = [json.loads(m) for m in s.publisher.messages[count:]]
    assert barriers and all(b['pose'] is None and b['vendor']['pause']['mapping_allowed'] is False
                            for b in barriers)
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
    assert all(json.loads(m)['pose'] is None for m in s.publisher.messages[count:])


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


# ── SLAM consumer contract: an in-segment resume must remain mappable ───────

def slam_stream():
    from plugins.pointcloud_slam_stream import SlamStream
    accepted = []
    stream = SlamStream(cloud_frame='camera_optical', child_frame='camera_optical', clock_id='camera-global',
                        child_from_cloud=np.eye(4), max_gap_ms=250, max_age_s=2, max_pending=4,
                        max_cloud_points=3, on_frame=lambda *a: accepted.append(a))
    return stream, accepted


def deliver(session, stream, start):
    for raw in session.publisher.messages[start:]:
        sample = json.loads(raw)
        stream.odom(sample, now=sample['stamp_ms']/1000 + .02)
    return len(session.publisher.messages)


def test_transient_rejection_pauses_slam_and_resumed_segment_is_admitted(sessions):
    create, now, _ = sessions
    s = create()
    stream, accepted = slam_stream()
    stamp = track(s, now)
    seen = deliver(s, stream, 0)
    stream.cloud(stamp, 'camera_optical', [[1, 0, 0]], now=stamp/1e9 + .02)
    assert len(accepted) == 1
    feed(s, now, stamp + 100*MS, lost=True)
    pause = last(s)
    assert pause['pose'] is None and pause['vendor']['pause'] == dict(
        source='visual_gate', status='recovering', mapping_allowed=False)
    seen = deliver(s, stream, seen)
    with pytest.raises(ValueError):
        stream.cloud(stamp + 100*MS, 'camera_optical', [[1, 0, 0]], now=(stamp + 100*MS)/1e9 + .02)
    for k in (2, 3, 4):
        feed(s, now, stamp + k*100*MS)
    deliver(s, stream, seen)
    resumed = stamp + 400*MS
    stream.cloud(resumed, 'camera_optical', [[1, 0, 0]], now=resumed/1e9 + .02)
    assert len(accepted) == 2
    assert stream.status(now=resumed/1e9 + .02)['integration'] == 'admitting'


def test_resume_beyond_consumer_pause_bound_starts_a_new_segment(sessions):
    create, now, _ = sessions
    s = create()
    stamp = track(s, now)
    feed(s, now, stamp + 100*MS, lost=True)
    for k in (7, 8, 9):  # valid again, but 0.7-0.9 s after the last published capture
        feed(s, now, stamp + k*100*MS)
    assert s.gate.state == 'lost' and 'pause' in s.gate.reason


def test_watchdog_pause_uses_capture_time_with_delayed_native_results(sessions):
    create, now, _ = sessions
    s = create()
    stream, accepted = slam_stream()
    anchor = track(s, now)

    def deliver_at(start, receipt):
        for raw in s.publisher.messages[start:]:
            stream.odom(json.loads(raw), now=receipt)
        return len(s.publisher.messages)

    seen = deliver_at(0, 1.5)
    stream.cloud(anchor, 'camera_optical', [[1, 0, 0]], now=1.5)
    for stamp, lost in [(1_300_000_000, True), (1_600_000_000, False), (1_800_000_000, False)]:
        now[0] = stamp+300*MS
        msg, info = native(s, stamp, lost=lost)
        s._result('pose', msg); s._result('info', info)
        seen = deliver_at(seen, now[0]/1e9)
    now[0] = 2_210_000_000
    s._watchdog()
    seen = deliver_at(seen, 2.21)
    stamp = anchor+800*MS
    now[0] = stamp+300*MS
    msg, info = native(s, stamp)
    s._result('pose', msg); s._result('info', info)
    deliver_at(seen, now[0]/1e9)
    stream.cloud(stamp, 'camera_optical', [[1, 0, 0]], now=now[0]/1e9)
    assert s.gate.state == 'tracking'
    assert stream.status(now=now[0]/1e9)['integration'] == 'admitting'
    assert len(accepted) == 2


@pytest.mark.parametrize('field', ['frame_id', 'child_frame_id'])
@pytest.mark.parametrize('prior_imu_fault', [False, True])
def test_fusion_native_frame_change_requires_a_new_segment(fused, field, prior_imu_fault):
    make, now, _ = fused
    s = make()
    value = dict(position=[0., 0., 0.], quaternion=[0., 0., 0., 1.], covariance=[0.]*36)
    s.fusion.last_tracking = (1_500_000_000, value, value)
    s.fusion.last_visual = 1_500_000_000
    stamp = 1_600_000_000
    s.fusion.samples.append((stamp, None, None))
    s.observations.add_frame(stamp, None, None, stamp)
    if prior_imu_fault:
        s.fusion.fault('IMU ingress overflow; restart required')
    now[0] = stamp+20*MS
    msg, info = native(s, stamp)
    if field == 'frame_id':
        msg.header.frame_id = 'different_odom'
    else:
        msg.child_frame_id = 'different_optical'
    s._result('pose', msg); s._result('info', info)
    for k in range(3):
        feed(s, now, stamp+(k+1)*100*MS)
    assert s.gate.state == 'lost' and s.gate.loss_kind == 'frame'
    assert s.fallback is None
    assert all(json.loads(raw)['pose'] is None for raw in s.publisher.messages)
