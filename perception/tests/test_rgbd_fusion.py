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


def warmed():
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    color, depth = images()
    for i in range(141):
        t = 1_000_000_000+i*5_000_000
        c.observe_imu(source(t), t+1_000_000)
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


def test_missing_translation_and_changed_calibration_fail_closed():
    c = ShortTermFusion('run', 'optical', 'capture-clock', K)
    with pytest.raises(ValueError, match='translation'):
        c.observe_imu(packet(), 1_001_000_000)
    assert c.status(1_001_000_000) == 'lost'
    c, t = warmed()
    p = source(t+5_000_000)
    p['camera_from_accel_translation'] = [0., 0., 0.]
    with pytest.raises(ValueError):
        c.observe_imu(p, t+6_000_000)
    assert c.status(t+6_000_000) == 'lost'


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


def test_geometry_assists_loss_but_never_becomes_a_tracking_measurement():
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
    prediction = out['vendor']['fusion']['prediction']
    assert 0 < prediction['position'][0] < .03
    assert out['pose'] is None and out['vendor']['fusion']['mapping_allowed'] is False
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
def test_calibration_source_and_time_mapping_changes_retire_segment(change):
    c, stamp = warmed()
    with pytest.raises(ValueError):
        c.observe_imu(source(stamp+5_000_000, **change), stamp+5_000_000)
    assert c.state == 'lost'


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
