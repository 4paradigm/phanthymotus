"""Transient visual degradation, persistent loss and ESKF visual-only fallback.

Real RgbdSession/EstimateGate with ROS and native process boundaries stubbed.
These are logic tests; they do not establish physical tracking behaviour.
"""
import json
from pathlib import Path
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


def test_recorded_weak_start_does_not_restart_before_first_tracking_pose():
    trace = json.loads((Path(__file__).parent/'fixtures/rgbd_motion_weak_start.json').read_text())
    gate = runtime.EstimateGate('run', 'camera_optical', 'camera-clock')
    for row in trace['frames']:
        msg, info = messages(row['stamp_ns'])  # synthetic identity, not recorded trajectory
        msg.pose.covariance = np.diag(row['covariance_diagonal']).ravel().tolist()
        info.inliers, info.lost = row['inliers'], row['lost']
        gate.input(row['stamp_ns'], now_ns=row['now_ns'])
        assert gate.accept(msg, info, row['now_ns']) is None
    assert gate.state == 'initializing'
    assert gate.last_pose is None and gate.good == 0
    # A continuation is synthetic: it proves warmup, not actual motion accuracy.
    for k in range(1, 4):
        stamp = row['stamp_ns'] + k*200*MS
        gate.input(stamp)
        sample = gate.accept(*messages(stamp), stamp+20*MS)
        assert (sample is not None) == (k == 3)
    assert gate.state == 'tracking'


def test_weak_warmup_after_two_good_frames_uses_result_liveness(sessions):
    create, now, ticks = sessions
    s = create()
    for k in range(30):
        stamp = 1_000_000_000+k*100*MS
        now[0] = stamp
        images(s, stamp)
        feed(s, now, stamp, cov=.001 if k < 2 else .25)
        ticks[0] += .1
        s._watchdog()
        assert s.gate.state == 'initializing'
    assert all(json.loads(m)['pose'] is None and 'pause' not in json.loads(m)['vendor']
               for m in s.publisher.messages)
    assert s.gate.good == 0
    # Input stays fresh, but a silent native estimator must still expire.
    now[0] += 1_500_000_001
    s.gate.last_input_received = now[0]
    s._watchdog()
    assert s.gate.state == 'lost' and s.gate.loss_kind == 'output'


def received_images(session, stamp):
    for kind in ('color', 'depth'):
        session._image(kind, NS(data=b'frame', header=NS(frame_id=session.gate.frame,
            stamp=NS(sec=stamp//10**9, nanosec=stamp%10**9))))


def test_native_silence_restarts_same_frontend_with_fresh_images(recovery):
    card, _, now, seconds = recovery
    session = card.instances['one'][0]
    stamp = track(session, now)
    for _ in range(17):
        stamp += 100*MS
        now[0] = stamp
        seconds[0] += .1
        received_images(session, stamp)
        session._watchdog()
    assert session.gate.state == 'lost' and session.gate.loss_kind == 'output'
    assert last(session)['pose'] is None
    session.process.returncode = 0
    card._recover()
    seconds[0] += 1.
    received_images(session, now[0]); card._recover()
    new = card.instances['one'][0]
    assert new.segment != session.segment
    assert new.calibration['visual_frontend'] == session.calibration['visual_frontend']


@pytest.mark.parametrize('duplicate_after_one_good', [False, True])
def test_insufficient_fresh_native_results_during_startup_is_an_output_timeout(
        recovery, duplicate_after_one_good):
    card, _, now, seconds = recovery
    session = card.instances['one'][0]
    if duplicate_after_one_good:
        stamp = now[0]
        feed(session, now, stamp)
        feed(session, now, stamp)
        assert session.gate.good == 1
    seconds[0] += runtime.VISUAL_INITIALIZATION_S + .01
    received_images(session, now[0])
    session._watchdog()
    assert session.gate.state == 'lost' and session.gate.loss_kind == 'output'
    assert last(session)['pose'] is None


@pytest.mark.parametrize('bad', ['lost', 'inliers', 'nan', 'asymmetric', 'negative', 'indefinite', 'quaternion'])
def test_invalid_warmup_is_not_treated_as_weak_but_usable_native_output(bad):
    gate = runtime.EstimateGate('run', 'camera_optical', 'camera-clock')
    for k in range(14):
        stamp = 1_000_000_000+k*100*MS
        msg, info = messages(stamp)
        msg.pose.covariance = np.diag([.25]*3+[.001]*3).ravel().tolist()
        if bad == 'lost': info.lost = True
        if bad == 'inliers': info.inliers = 9
        if bad == 'nan': msg.pose.covariance[0] = float('nan')
        if bad == 'asymmetric': msg.pose.covariance[1] = .1
        if bad == 'negative': msg.pose.covariance[0] = -.1
        if bad == 'indefinite': msg.pose.covariance[1] = msg.pose.covariance[6] = .5
        if bad == 'quaternion': msg.pose.pose.orientation.w = 0.
        gate.input(stamp)
        assert gate.accept(msg, info, stamp+20*MS) is None
    assert gate.state == 'lost'


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


@pytest.mark.parametrize('entries,accepted', [
    ({(0, 1): .002}, False),
    ({(3, 3): 99.}, False),
    ({(3, 3): .08, (4, 4): .08, (3, 4): .05}, False),
    ({(0, 1): .001 + 5e-11, (3, 3): .12}, True),
], ids=['indefinite', 'large-rotation', 'correlated-rotation', 'numerical-tolerance'])
def test_nominal_translation_requires_valid_visual_covariance(entries, accepted):
    gate = runtime.EstimateGate('run', 'camera_optical', 'camera-clock')
    for k in range(3):
        stamp = (1_000+100*k)*MS
        gate.input(stamp)
        msg, info = messages(stamp)
        covariance = np.eye(6)*.001
        for (row, col), value in entries.items():
            covariance[row, col] = covariance[col, row] = value
        msg.pose.covariance = covariance.ravel().tolist()
        sample = gate.accept(msg, info, stamp+20*MS)
        assert (sample is not None) == (accepted and k == 2)
    assert gate.state == ('tracking' if accepted else 'recovering')


def test_repeated_or_old_source_images_are_admitted(sessions):
    create, now, _ = sessions
    s = create()
    images(s, now[0])
    images(s, now[0])          # repeated capture stamps
    images(s, now[0] - 2_000_000_000)  # older than the pairing window
    assert s.gate.state != 'lost'
    assert s.counts['rejected_inputs'] == 0
    assert s.pending.qsize() >= 1
    assert not s.info()['input_rejections']['last_reason']


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


def test_old_formatted_imu_does_not_force_visual_only(fused):
    make,now,ticks=fused
    s=make();now[0]=1_000_000_000
    imu(s,999_000_000)
    now[0]=1_800_000_000;s._watchdog()
    assert s.imu.state=='active' and s.fusion.state!='lost'
    assert s.fallback is None


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
    track_fused(s, now, offset=.051)
    assert s.gate.state == 'lost' and 'continuity' in s.gate.reason
    assert s.fallback is None


def test_failed_fusion_continuity_selects_available_imu_attitude_for_next_segment(fused):
    make, now, ticks = fused
    s = make()
    s.rotation_hints.enabled = True  # native hint runtime is available
    stamp = 1_500*MS
    now[0] = stamp
    imu(s, stamp)
    s.gate.last_input = stamp
    track_fused(s, now, offset=.051)
    assert s.gate.state == 'lost' and s.gate.last_pose is None
    target = s.recovery_target(ticks[0], now[0])
    config = target['calibration']
    assert target['action'] == 'degrade_assistance'
    assert config['imu_topic'] == s.calibration['imu_topic']
    assert runtime.assistance_mode(config) == 'imu_attitude'
    assert runtime.estimator_parameters(config, s.segment)['wait_imu_to_init']


def test_fallback_with_proven_continuity_keeps_segment_and_checks_resume(fused):
    make, now, _ = fused
    s = make()
    track_fused(s, now, offset=.041)
    assert s.gate.state == 'recovering' and s.fallback is not None
    feed(s, now, 1_600_000_000, x=2.)  # cannot jump 2 m in 100 ms
    assert s.gate.state == 'lost' and 'discontinu' in s.gate.reason


def test_fallback_keeps_the_frozen_fusion_recovery_window(fused):
    make, now, _ = fused
    s = make()
    s.gate.recovery_policy = runtime.RecoveryPolicy()
    s.gate.result_period_ns = 134*MS
    s.fusion.pause_context = s.gate.recovery_policy.begin(134*MS, 800*MS, runtime.time.monotonic_ns())
    track_fused(s, now)
    assert s.gate.pause_context is s.fusion.pause_context
    assert s.gate.result_period_ns == 134*MS
    for offset in (803, 937, 1071):
        feed(s, now, 1_500*MS + offset*MS)
        if offset < 1071:
            assert s.gate.state == 'recovering'
    assert s.gate.state == 'tracking'
    assert s.gate.pause_context is None
    assert last(s)['pose'] is not None
    assert last(s)['vendor']['motus_slam']['segment_id'] == s.segment


def timed_fusion(fused, period):
    from test_rgbd_fusion import images as fusion_images

    make, now, ticks = fused
    s = make()
    s.gate.recovery_policy = s.fusion.recovery_policy = runtime.RecoveryPolicy()
    color, depth = fusion_images()
    next_input = next_imu = 1_000*MS

    def advance(deadline):
        nonlocal next_input, next_imu
        while next_imu <= deadline:
            now[0] = next_imu
            imu(s, next_imu)
            next_imu += 5*MS
        while next_input <= deadline:
            ticks[0] = next_input
            s.gate.input(next_input)
            next_input += 33_333_333
        ticks[0] = now[0] = deadline
        s.fusion.input_period_ns = s.gate.input_period_ns

    def result(stamp, *, lost=False, landmarks=False, processing_ns=None):
        advance(stamp)
        if s._fusing():
            assert s.observations.add_frame(stamp, color, depth, now[0]) == []
        advance(stamp+400*MS)
        if processing_ns is not None:
            context = s.time_policy.select({}, stamp, ticks[0]-processing_ns)
            pair = NS(association_id=str(stamp), method='receipt_estimated', skew_ns=0,
                      color_context=context, depth_context=context)
            s.contexts_by_stamp[stamp] = (context, pair)
        msg, info = native(s, stamp, lost=lost)
        if landmarks:
            info.type = 0
            info.local_map_keys = info.word_inliers = list(range(24))
            info.local_map_values = [NS(x=float(i % 5), y=float(i // 5), z=1.) for i in range(24)]
        s._result('pose', msg)
        s._result('info', info)

    anchor = 1_000*MS + 6*period
    for stamp in range(1_000*MS, anchor+1, period):
        result(stamp, landmarks=True)
    assert last(s)['pose'] is not None
    assert s.fusion.last_tracking[0] == anchor
    assert s.gate.input_period_ns == 33_333_333
    return s, advance, result, anchor


@pytest.mark.parametrize('failure,period_ms,budget_ms', [
    ('visual_rejection', 500, 2_400),
    ('fusion_failure', 500, 2_400),
    ('fusion_failure', 700, 3_000),
    ('fusion_failure_during_pause', 500, 2_400),
    ('inertial_expiry', 500, 2_400),
])
def test_fast_source_slow_native_results_have_time_for_measured_recovery(
        fused, failure, period_ms, budget_ms):
    _, now, _ = fused
    period = period_ms*MS
    s, advance, result, anchor = timed_fusion(fused, period)

    if failure != 'fusion_failure':
        result(anchor+period, lost=True)
        pause = s.fusion.pause_context
        assert last(s)['pose'] is None
        assert pause.budget_ns == budget_ms*MS

    if failure == 'inertial_expiry':
        advance(anchor+1_405*MS)
        sample = s.fusion.watchdog_output(now[0])
        assert s.fusion.state == 'lost' and 'inertial bridge' in s.fusion.reason
        assert sample['vendor']['fusion']['prediction'] is None
        return

    if failure != 'visual_rejection':
        s.fusion.fault('measured IMU unavailable')
        s._check_fusion(now[0])
        if failure == 'fusion_failure_during_pause':
            assert s.gate.pause_context is pause
        assert s.gate.pause_context.budget_ns == budget_ms*MS
        assert s.gate.good == 0

    first_offset = period if failure == 'fusion_failure' else 2*period
    for count in range(3):
        result(anchor+first_offset+count*period)
        assert (last(s)['pose'] is not None) == (count == 2)
    assert last(s)['vendor']['motus_slam']['segment_id'] == s.segment


def test_fallback_retains_native_timing_for_old_landmark_verification(fused):
    _, now, _ = fused
    period = 550*MS
    s, advance, result, anchor = timed_fusion(fused, period)
    result(anchor+period, lost=True)
    pause = s.fusion.pause_context
    assert pause.budget_ns == 2_600*MS

    pending = anchor+2*period
    advance(pending)
    assert s.observations.add_frame(pending, None, None, now[0]) == []
    s.fusion.fault('measured IMU unavailable')
    s._check_fusion(now[0])
    assert s.gate.pause_context is pause
    assert s.observations.counts['cancelled_on_fallback'] == 1

    result(pending, landmarks=True)
    assert s.gate.good == 0
    for k in (3, 4):
        result(anchor+k*period, landmarks=True, processing_ns=100*MS)
        assert s.gate.good == k-2 and last(s)['pose'] is None
    assert s.processing_ages[-1] == 100*MS

    # The third fresh capture is outside the frozen in-segment pause bound.
    result(anchor+5*period, landmarks=True)
    assert last(s)['pose'] is None
    assert s.gate.relocalization_points and s.gate.good == 0
    advance(now[0]+33_333_333)
    assert 2_566*MS < s.gate.relocalization_deadline_mono-now[0] <= 2_600*MS

    # The three old-landmark candidates span 1.1 s before processing delay.
    for k in (6, 7, 8):
        result(anchor+k*period, landmarks=True)
        assert (last(s)['pose'] is not None) == (k == 8)
    assert last(s)['vendor']['motus_slam']['segment_id'] == s.segment
    assert last(s)['vendor']['relocalization'] == dict(method='native_f2m_landmarks',
                                                     anchor_stamp_ns=anchor, shared_landmarks=24)
    assert s.gate.state == 'tracking'

    # Fresh results replace inherited timing for later pauses.
    for k in range(1, 7):
        result(anchor+8*period+k*450*MS, processing_ns=100*MS)
    s.gate.degrade('subsequent visual rejection', now[0])
    assert s.gate.pause_context.budget_ns == 1_900*MS


def test_fallback_does_not_count_cancelled_fusion_captures(fused):
    make, now, _ = fused
    s = make()
    pending_stamp = 1_510*MS
    s.observations.add_frame(pending_stamp, None, None, pending_stamp)
    track_fused(s, now)
    assert s.gate.state == 'recovering'
    feed(s, now, pending_stamp)
    assert s.gate.good == 0
    for count, stamp in enumerate((1_600*MS, 1_700*MS, 1_800*MS)):
        feed(s, now, stamp)
        assert (last(s)['pose'] is not None) == (count == 2)


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
    stream.cloud(stamp + 100*MS, 'camera_optical', [[1, 0, 0]], now=(stamp + 100*MS)/1e9 + .02)
    assert len(accepted) == 1 and stream.pending_count == 1
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


def test_geometry_tracking_cannot_fake_visual_fallback_continuity(fused):
    make, now, _ = fused
    s = make()
    from test_rgbd_fusion import pose
    s.fusion.last_tracking = (now[0]-10_000_000, pose(), None)
    s.fusion.fault('visual anchor expired')
    s._check_fusion(now[0])
    assert s.fallback is None
    assert s.gate.state == 'lost'
    assert 'no capture-matched visual pose' in s.gate.reason
