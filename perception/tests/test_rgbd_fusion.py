"""Synthetic truth tests: signs, units, uncertainty and outage policy."""
import numpy as np
import pytest

from plugins.rgbd_fusion import InertialFilter, ShortTermFusion
from plugins.pointcloud_slam_odom import decode_slam_odom
from test_rgbd_imu import packet

G = 9.80665
K = np.array([[500., 0, 319.5], [0, 500., 239.5], [0, 0, 1]])


def pose(x=0., quaternion=(0., 0., 0., 1.)):
    return dict(position=[x, 0., 0.], quaternion=list(quaternion),
                covariance=(np.eye(6)*.0001).ravel().tolist(),
                frame_id='run_odom', child_frame_id='optical')


def test_inertial_filter_integrates_acceleration_and_initial_velocity():
    f = InertialFilter(1_000_000_000, pose(), [.4, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    for _ in range(100):
        f.propagate(.005, [0, 0, 0], [1., 0, G])
    # x = v0*t + a*t*t/2, v = v0 + a*t
    np.testing.assert_allclose(f.camera_pose()['position'], [.325, 0, 0], atol=1e-8)
    np.testing.assert_allclose(f.v, [.9, 0, 0], atol=1e-8)
    np.testing.assert_allclose(f.P, f.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(f.P).min() >= -1e-12


def test_turning_with_offset_accelerometer_returns_camera_origin():
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [.1, 0, 0])
    for _ in range(100):
        f.propagate(.005, [0, 0, 1.], [0, 0, G])
    np.testing.assert_allclose(f.camera_pose()['position'],
                               [.1*(1-np.cos(.5)), -.1*np.sin(.5), 0], atol=1e-8)


def test_pose_update_reduces_uncertainty_and_rejects_large_innovation():
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    for _ in range(100):
        f.propagate(.005, [0, 0, 0], [.1, 0, G])
    before = f.P.trace()
    assert f.correct(pose(), np.eye(6)*.0001)
    assert abs(f.camera_pose()['position'][0]) < .0125
    assert f.P.trace() < before
    snapshot = f.camera_pose()['position'][:]
    assert not f.correct(pose(10.), np.eye(6)*.0001)
    np.testing.assert_allclose(f.camera_pose()['position'], snapshot)


def images():
    yy, xx = np.indices((480, 640))
    color = np.repeat((((xx//8+yy//8)%2)*220).astype(np.uint8)[..., None], 3, axis=2)
    return color, np.full((480, 640), 1500, np.uint16)


def source(t, **changes):
    values = dict(camera_from_accel_translation=[.03, 0, 0],
                  camera_from_gyro_translation=[.03, 0, 0],
                  translation_source='sdk_factory_extrinsics')
    values.update(changes)
    return packet(t, **values)


def warmed(monkeypatch_without_zupt=None, *, imu=source):
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    if monkeypatch_without_zupt is not None:
        monkeypatch_without_zupt.setattr(c, '_stationary_update', lambda *_: False)
    color, depth = images()
    for i in range(141):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(imu(t), t+1_000_000)
        if i%20 == 0:
            sample = c.observe_frame(t, color, depth, pose(), t+1_000_000)
    assert c.status(t+1_000_000) == 'tracking'
    assert decode_slam_odom(sample).position == pytest.approx((0., 0., 0.), abs=1e-8)
    return c, t


def advance(c, start, seconds, visual=None):
    color, depth = images()
    for i in range(1, round(seconds/.005)+1):
        t = start+i*5_000_000
        c.observe_imu(source(t), t+1_000_000)
    return c.observe_frame(t, color, depth, visual, t+1_000_000, reason='visual missing'), t


def test_short_outage_is_prediction_only_and_requires_three_visual_recoveries():
    c, t = warmed()
    sample, t = advance(c, t, .3)
    assert sample['pose'] is None
    assert sample['vendor']['motus_slam']['status'] == 'coasting'
    assert sample['vendor']['fusion']['prediction']['position'] == pytest.approx([0, 0, 0], abs=1e-8)
    with pytest.raises(ValueError):
        decode_slam_odom(sample)
    for count in range(3):
        sample, t = advance(c, t, .05, pose())
        assert (sample['pose'] is not None) == (count == 2)


def test_imu_continues_during_camera_silence_without_advancing_filter_anchor():
    c, t = warmed()
    for i in range(1, 121):
        stamp = t+i*5_000_000
        c.observe_imu(source(stamp, linear_acceleration=[1., 0., G]), stamp+1_000_000)
    sample = c.watchdog_output(stamp+1_000_000)
    vendor = sample['vendor']
    assert sample['pose'] is None and vendor['motus_slam']['status'] == 'coasting'
    assert vendor['motus_slam']['segment_id'] == 'run'
    assert vendor['motus_slam']['stamp_ns'] == vendor['fusion']['prediction_stamp_ns'] == stamp
    assert vendor['fusion']['prediction']['position'][0] > .1
    assert c.filter.stamp == t
    stale_imu = c.watchdog_output(stamp+120_000_000)
    assert stale_imu['vendor']['fusion']['prediction'] is None
    expired = c.watchdog_output(t+1_100_000_000)
    assert expired['vendor']['fusion']['prediction'] is None


def test_zero_velocity_needs_independent_stationarity_and_quiet_imu():
    c, t = warmed()
    c.filter.v[0] = .025
    assert c._stationary_update(t, pose(), t-300_000_000, pose())
    assert abs(c.filter.v[0]) < .025
    before = c.filter.v.copy()
    assert not c._stationary_update(t, pose(.05), t-300_000_000, pose())
    np.testing.assert_array_equal(c.filter.v, before)
    uncertain = dict(pose(), covariance=(np.eye(6)*.01).ravel().tolist())
    assert not c._stationary_update(t, uncertain, t-300_000_000, pose())
    c.samples[-1] = (t, np.array([0., 0., .5]), np.array([0., 0., G]))
    assert not c._stationary_update(t, pose(), t-300_000_000, pose())


@pytest.mark.parametrize('seconds', [.1, .3, .5, 1., 2., 5.])
def test_outage_limit_never_promotes_prediction_or_restarts_segment(seconds):
    c, t = warmed()
    sample, t = advance(c, t, seconds)
    assert sample['pose'] is None
    status = sample['vendor']['motus_slam']['status']
    if seconds <= .5:
        assert status == 'coasting'
    elif seconds > 1.:
        assert status == 'lost'
    else:
        assert status in ('coasting', 'lost')  # uncertainty may terminate before the hard time limit
    if seconds > 1.:
        sample = c.observe_frame(t+1, *images(), pose(), t+1_000_000)
        assert sample['pose'] is None and c.status(t+1_000_000) == 'lost'


def test_missing_numeric_translation_is_rejected_and_new_numeric_translation_is_used():
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    with pytest.raises(ValueError, match='translation'):
        c.observe_imu(packet(), 1_001_000_000)
    assert c.status(1_001_000_000) == 'lost'
    c, t = warmed()
    p = source(t+5_000_000)
    p['camera_from_accel_translation'] = [0., 0., 0.]
    c.observe_imu(p,t+6_000_000)
    assert c.status(t+6_000_000)=='tracking'
    np.testing.assert_allclose(c.lever,[0.,0.,0.])
    np.testing.assert_allclose(c.filter.lever,[0.,0.,0.])


def test_motion_during_warmup_does_not_assume_zero_velocity():
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    for i in range(141):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(source(t), t+1_000_000)
        if i%20 == 0:
            out = c.observe_frame(t, *images(), pose(i*.005), t+1_000_000)
    assert out['pose'] is None and c.status(t+1_000_000) == 'initializing'


def test_rejected_visual_recovery_does_not_refresh_outage_clock():
    c, t = warmed()
    _, t = advance(c, t, .4)
    for _ in range(4):
        sample, t = advance(c, t, .2, pose(3.))
    assert sample['pose'] is None and c.status(t+1_000_000) == 'lost'


def test_accepted_recovery_correction_refreshes_timeout_before_tracking_promotion():
    c, t = warmed()
    # An actual missing correction starts recovery; a normal 300 ms capture
    # interval alone no longer does so.
    advance(c, t, .05)
    for i in range(11, 221):
        stamp = t+i*5_000_000
        c.observe_imu(source(stamp), stamp)
    frame, now = t+300_000_000, t+1_100_000_000
    sample = c.observe_frame(frame, *images(), pose(), now)
    assert sample['vendor']['fusion']['aid'] == 'visual'
    assert sample['pose'] is None  # one correction does not establish tracking
    assert c.status(now) == 'recovering'
    assert c.last_visual == frame


def test_visual_rotation_during_warmup_is_not_estimated_as_gyro_bias():
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    for i in range(141):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(source(t, angular_velocity=[0, 0, .08]), t+1_000_000)
        if i%20 == 0:
            angle = .08*i*.005
            out = c.observe_frame(t, *images(), pose(quaternion=(0, 0, np.sin(angle/2), np.cos(angle/2))), t+1_000_000)
    assert out['pose'] is None and c.status(t+1_000_000) == 'initializing'


def test_missing_camera_frames_stop_claiming_tracking_before_hard_loss():
    c, t = warmed()
    # Wall-clock silence is judged after the 250 ms join deadline plus one
    # >=4 Hz frame interval, so ordinary native latency is not a loss of frames.
    for i in range(1, 121):
        stamp = t+i*5_000_000
        c.observe_imu(source(stamp), stamp)
    assert c.status(stamp) == 'coasting'
    sample = c.output(stamp)
    assert sample['pose'] is None
    assert sample['vendor']['fusion']['prediction_stamp_ns'] == t


def test_verified_geometry_keeps_tracking_at_its_measured_capture():
    from test_rgbd_geometry import render_room
    calibration = np.array([[460., 0, 319.5], [0, 460., 239.5], [0, 0, 1.]])
    c = ShortTermFusion('run', 'optical', 'capture-clock', calibration, depth_geometry=True)
    color, _ = images()
    reference = render_room(np.eye(4), (480, 640), calibration)
    for i in range(141):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(source(t), t)
        if i%20 == 0:
            c.observe_frame(t, color, reference, pose(), t)
    for i in range(1, 41):
        stamp = t+i*5_000_000
        c.observe_imu(source(stamp), stamp)
    transform = np.eye(4); transform[0, 3] = .025
    current = render_room(transform, (480, 640), calibration)
    out = c.observe_frame(stamp, color, current, None, stamp)
    assert out['vendor']['fusion']['aid'] == 'depth_geometry'
    assert 0 < out['pose']['position'][0] < .03
    assert out['vendor']['fusion']['mapping_allowed'] is True
    assert decode_slam_odom(out).stamp_ms == stamp//1_000_000
    assert c.last_visual == t


def test_accelerometer_bias_is_estimated_while_vision_is_available():
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    for i in range(800):
        f.propagate(.005, [0, 0, 0], [.08, 0, G])
        if i%20 == 19:
            assert f.correct(pose(), np.eye(6)*.0001)
    assert abs(f.camera_pose()['position'][0]) < .005
    assert abs(f.v[0]) < .01
    # Position/gravity/bias observability is coupled; do not demand the
    # unobservable static bias/tilt split equals the injected bias exactly.
    assert f.ba[0] > .005


def test_independently_sampled_camera_uses_bracketing_imu_to_initialize():
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    for i in range(401):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(source(t), t)
        if i%20 == 1:
            frame_stamp = t-4_000_000  # camera is1ms after the previous gyro sample
            sample = c.observe_frame(frame_stamp, *images(), pose(), t)
    assert c.filter is not None and sample['pose'] is not None


@pytest.mark.parametrize('change', [{'translation_source': 'tf'},
                                  {'clock_mapping': {'method': 'fixed_arrival_offset', 'offset_ns': 1000000}}])
def test_source_and_mapping_descriptions_do_not_retire_segment(change):
    c,stamp=warmed()
    c.observe_imu(source(stamp+5_000_000,**change),stamp+5_000_000)
    assert c.state=='tracking' and c.imu.state=='active'


def test_correlated_geometry_updates_do_not_average_away_shared_reference_error():
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    f.P[:3, :3] = np.eye(3)*.01
    for _ in range(20):
        assert f.correct(pose(), np.eye(6)*.0002, correlated=True)
    covariance = np.array(f.camera_pose()['covariance']).reshape(6, 6)
    assert np.linalg.eigvalsh(covariance[:3, :3]).min() >= .0001


def test_geometry_covariance_includes_rotation_induced_position_uncertainty():
    from plugins.rgbd_fusion import geometry_observation
    reference = pose()
    reference['covariance'] = [0.]*36
    transform = np.eye(4); transform[0, 3] = .1
    result = dict(transform=transform, covariance=np.diag([.0001]*3+[.04]*3))
    measured, covariance = geometry_observation(reference, result)
    assert measured['position'] == pytest.approx([.1, 0, 0])
    # A left-tangent z-rotation moves the translated camera in +y.
    # Add1deg reference orientation floor and5mm reference position floor.
    assert covariance[1, 1] == pytest.approx(.0005280461741978671)
    assert covariance[1, 5] == pytest.approx(.004030461741978671)
    assert covariance[2, 4] == pytest.approx(-.004030461741978671)
    assert np.linalg.eigvalsh(covariance).min() > 0


@pytest.mark.parametrize('correlated', [False, True])
def test_correction_matches_full_matrix_solution_with_cross_covariance(correlated):
    """Vector RHS solves must preserve every gain column and covariance reset."""
    from plugins.rgbd_fusion import exp_rotation, log_rotation, pose_values, skew
    f = InertialFilter(1_000_000_000, pose(), [.01, 0, 0], [0, 0, G],
                       [0, 0, 0], [.03, .01, 0])
    f.propagate(.05, [.01, -.02, .03], [.1, 0, G])
    measurement = pose(.002)
    position, rotation = pose_values(measurement)
    residual = np.r_[position-(f.p-f.R@f.lever), log_rotation(f.R.T@rotation)]
    h = f.measurement_matrix()
    basis = np.eye(6); basis[3:, 3:] = f.R.T
    covariance = basis@np.diag([.0001, .0002, .0003, .0004, .0005, .0006])@basis.T
    prior = f.P/.9 if correlated else f.P.copy()
    if correlated:
        covariance /= .1
    # Full matrix RHS is the numerical reference, never the optimized path.
    gain = np.linalg.solve(h@prior@h.T+covariance, h@prior).T
    delta = gain@residual
    expected = [f.p+delta[:3], f.v+delta[3:6], f.R@exp_rotation(delta[6:9]),
                f.bg+delta[9:12], f.ba+delta[12:15]]
    identity = np.eye(15)-gain@h
    reset = np.eye(15); reset[6:9, 6:9] -= .5*skew(delta[6:9])
    expected_p = reset@(identity@prior@identity.T+gain@covariance@gain.T)@reset.T
    assert f.correct(measurement, np.diag([.0001, .0002, .0003, .0004, .0005, .0006]),
                     correlated=correlated)
    for actual, wanted in zip((f.p, f.v, f.R, f.bg, f.ba), expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(f.P, expected_p, rtol=1e-12, atol=1e-14)


def test_two_hz_measured_visuals_can_initialize_and_remain_tracking():
    fusion = ShortTermFusion('run', 'optical', 'capture-clock', K)
    fusion.input_period_ns = 530_000_000
    stamps = set(1_000_000_000 + i*530_000_000 for i in range(6))
    tracking = []
    for i in range(531):
        t = 1_000_000_000 + i*5_000_000
        fusion.observe_imu(source(t), t)
        if t in stamps:
            sample = fusion.observe_frame(t, *images(), pose(), t)
            if sample['pose'] is not None:
                tracking.append(t)
    assert len(tracking) >= 2


def test_slow_warmup_cannot_initialize_across_an_unmeasured_imu_gap():
    fusion = ShortTermFusion('run', 'optical', 'capture-clock', K)
    t0 = 1_000_000_000
    fusion.lever = np.zeros(3)
    for i in range(151):
        t = t0 + i*5_000_000
        if i < 50 or i > 90:
            fusion.samples.append((t, np.zeros(3), np.array([0., 0., G])))
        if i % 50 == 0:
            assert not fusion._initialize(t, pose())

@pytest.mark.parametrize('side', ['start', 'end'])
def test_imu_interpolation_needs_measured_support_at_both_visual_boundaries(side):
    fusion = ShortTermFusion('run', 'optical', 'capture-clock', K)
    fusion.lever = np.zeros(3)
    t = 1_000_000_000
    offsets = list(range(5_000_000, 750_000_000, 5_000_000))
    offsets.extend([-150_000_000 if side == 'start' else 0,
                    900_000_000 if side == 'end' else 750_000_000])
    for offset in sorted(offsets):
        fusion.samples.append((t+offset, np.zeros(3), np.array([0., 0., G])))
    for offset in (0, 375_000_000, 750_000_000):
        assert not fusion._initialize(t+offset, pose())


def test_frame_and_watchdog_predictions_require_imu_coverage_back_to_measured_anchor():
    c, anchor = warmed()
    # Largest IMU interval recorded in the latest Tianyi CDR.
    after_gap = anchor + 240_235_008
    c.observe_imu(source(after_gap), after_gap)
    at_frame = c.observe_frame(after_gap, *images(), None, after_gap)
    end = after_gap + 20_000_000
    c.observe_imu(source(end), end)
    for sample in (at_frame, c.watchdog_output(end)):
        assert sample['pose'] is None
        assert sample['vendor']['fusion']['prediction'] is None


@pytest.mark.parametrize('change', [{'accel_stamp_ns': 1_000_000}, {'epoch': 'restarted'}])
def test_inertial_prediction_cannot_cross_incomparable_or_discontinuous_samples(change):
    c, anchor = warmed()
    t = anchor + 10_000_000
    c.observe_imu(source(t, **change), t)
    t += 10_000_000
    c.observe_imu(source(t), t)
    out = c.observe_frame(t, *images(), None, t)
    assert out['pose'] is None and out['vendor']['fusion']['prediction'] is None


@pytest.mark.parametrize('missing', ['source_id', 'epoch'])
def test_prediction_coverage_uses_numeric_observations_without_identities(missing):
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    for stamp in (1_000_000_000, 1_020_000_000):
        packet = source(stamp)
        del packet[missing]
        c.observe_imu(packet, stamp)
    assert c._imu_covers(1_000_000_000, 1_020_000_000)


def test_numeric_imu_without_epoch_can_aid_fusion_and_export_inertial_prediction(monkeypatch):
    def no_epoch(stamp):
        sample = source(stamp)
        del sample['epoch']
        return sample

    c, anchor = warmed(imu=no_epoch)
    assert c.zero_velocity_updates > 0
    c.filter.v[0] = .03
    initial_poses = []

    class Geometry:
        def register(self, depth, reference, initial):
            initial_poses.append(initial.copy())
            return None

    monkeypatch.setattr(c, 'geometry', Geometry())
    for i in range(1, 41):
        stamp = anchor + i*5_000_000
        c.observe_imu(no_epoch(stamp), stamp)
    out = c.observe_frame(stamp, *images(), None, stamp)
    assert len(initial_poses) == 1
    assert initial_poses[0][0, 3] == pytest.approx(.006)
    assert out['pose'] is None
    assert out['vendor']['fusion']['prediction'] is not None


@pytest.mark.parametrize('change', [
    {'epoch': 'restarted'},
    {'source_id': 'other-sensor'},
    {'camera_from_accel_translation': [.02, 0., 0.]},
])
def test_internal_stationarity_rejects_imu_provenance_changes(change):
    c, anchor = warmed()
    for i in range(1, 41):
        stamp = anchor + i*5_000_000
        c.observe_imu(source(stamp, **(change if i == 20 else {})), stamp)
    assert not c._stationary_update(stamp, pose(), anchor, pose())


@pytest.mark.parametrize('change', [
    {'camera_from_accel_translation': [.02, 0., 0.]},
    {'camera_from_gyro': [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]},
])
def test_new_calibration_does_not_extend_an_old_prediction_anchor(change):
    c, anchor = warmed()
    end = anchor + 20_000_000
    c.observe_imu(source(end, **change), end)
    out = c.observe_frame(end, *images(), None, end)
    assert out['vendor']['fusion']['prediction'] is None
