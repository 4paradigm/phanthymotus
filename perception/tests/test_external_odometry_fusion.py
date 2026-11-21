import numpy as np
import pytest

from plugins.external_odometry_fusion import ExternalOdometry, decode_motion
from plugins.pointcloud_slam_odom import decode_slam_odom


NS = 1_000_000_000


def pose_at(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def packet(t, x=None, *, twist=None, frame='external', reset=False):
    pose = None if x is None else dict(position=[x, 0, 0], quaternion=[0, 0, 0, 2],
                                      frame_id=frame, child_frame_id='body')
    return dict(schema='motus.odom/1', stamp_ns=NS + round(t*NS), pose=pose, twist=twist, reset=reset)


def feed(engine, t, x=None, **kwargs):
    stamp, motion = decode_motion(packet(t, x, **kwargs), 1)
    return engine.motion(stamp, motion)


def vision(engine, t, x, segment='vision'):
    return engine.vision(NS+round(t*NS), segment, pose_at(x))


def test_external_pose_continues_through_long_visual_loss_and_world_restart():
    engine = ExternalOdometry()
    vision(engine, 0, 10)
    feed(engine, 0, 100)
    for step in range(1, 121):
        feed(engine, step/4, 100 + step/4)
    assert engine.current.transform[0, 3] == pytest.approx(40.)
    assert engine.current.mask.all()
    vision(engine, 30, -800, 'replacement')
    assert engine.current.transform[0, 3] == pytest.approx(40.)
    feed(engine, 30.25, 130.25)
    vision(engine, 30.25, -799.75, 'replacement')
    assert engine.current.transform[0, 3] == pytest.approx(40.25)


def test_twist_masks_unknown_components_and_identity_after_first_packet():
    engine = ExternalOdometry()
    feed(engine, 0, twist=[1., None, None, None, None, None])
    assert engine.current.last_update is None
    feed(engine, .5, twist=[1., None, None, None, None, None])
    assert engine.current.transform[0, 3] == pytest.approx(.5)
    assert engine.current.mask.tolist() == [True, False, False, False, False, False]
    assert engine.current.covariance[1, 1] > engine.current.covariance[0, 0]
    output = engine.output()
    assert output['twist'] == [1., None, None, None, None, None]
    with pytest.raises(ValueError, match='planar position'):
        decode_slam_odom(output)


def test_pose_precedes_twist_and_known_reset_does_not_create_motion():
    engine = ExternalOdometry()
    feed(engine, 0, 100, twist=[99.] * 6)
    feed(engine, .5, 101, twist=[99.] * 6)
    assert engine.current.transform[0, 3] == pytest.approx(1.)
    feed(engine, 1, -100, reset=True)
    assert engine.current.transform[0, 3] == pytest.approx(1.)
    feed(engine, 1.5, -99.5)
    assert engine.current.transform[0, 3] == pytest.approx(1.5)


def test_late_vision_and_external_pose_rebuild_newer_corrections():
    def run(delayed):
        engine = ExternalOdometry(learning=False)
        feed(engine, 0, 10)
        vision(engine, 0, 0)
        if not delayed:
            feed(engine, .25, 10.35)
            vision(engine, .25, .25)
        feed(engine, .5, 10.7)
        vision(engine, .5, .5)
        feed(engine, .75, 11.05)
        if delayed:
            vision(engine, .25, .25)
            feed(engine, .25, 10.35)
        return engine
    ordered, late = run(False), run(True)
    np.testing.assert_allclose(ordered.current.transform, late.current.transform)
    np.testing.assert_allclose(ordered.current.covariance, late.current.covariance)
    assert 0 < late.current.transform[0, 3] < 1.05


def test_visual_progress_during_external_outage_is_not_counted_twice():
    engine = ExternalOdometry(learning=False)
    vision(engine, 0, 0)
    feed(engine, 0, 100)
    vision(engine, .2, .2)
    vision(engine, .4, .4)
    assert engine.current.transform[0, 3] == pytest.approx(.4)
    feed(engine, .6, 100.6)
    assert engine.current.transform[0, 3] == pytest.approx(.6)


def test_numeric_stamp_and_reference_point_without_source_declarations():
    item = packet(0, 2)
    item.pop('stamp_ns')
    item['body_from_sensor'] = pose_at(1).tolist()
    t, measurement = decode_motion(item, 52)
    assert t == 52
    assert measurement.pose[0, 3] == pytest.approx(1.)


def test_fused_output_correction_does_not_republish_an_old_time():
    engine = ExternalOdometry()
    vision(engine, 0, 0)
    feed(engine, 0, 100)
    feed(engine, .5, 100.5)
    first = engine.output()
    assert first['pose']['position'][0] == pytest.approx(.5)
    vision(engine, .25, .23)
    assert engine.output() is None
    feed(engine, .75, 100.75)
    assert engine.output()['vendor']['motus_slam']['stamp_ns'] == NS + 750_000_000


def test_learning_changes_only_new_intervals_and_improves_heldout_motion():
    def run(enabled):
        engine = ExternalOdometry(learning=enabled)
        t, actual, raw = 0., 0., 0.
        feed(engine, t, raw)
        vision(engine, t, actual)
        for i in range(40):
            speed = [.3, 1., -.45, 1.5][i % 4]
            t += .2
            actual += speed*.2
            raw += (1.20*speed+.035)*.2
            feed(engine, t, raw)
            vision(engine, t, actual)
        start = engine.current.transform[0, 3]
        for i in range(30):
            t += .2
            raw += (1.20*.8+.035)*.2
            feed(engine, t, raw)
        return abs(engine.current.transform[0, 3]-start-4.8), engine.current.learner.info()
    baseline, _ = run(False)
    candidate, info = run(True)
    assert info['rank'] >= 2
    assert candidate < baseline/4


def test_same_time_conflicting_motion_has_an_order_independent_result():
    def run(values):
        engine = ExternalOdometry()
        feed(engine, 0, 0.)
        for value in values:
            feed(engine, .5, value)
        feed(engine, 1., 3.)
        return engine.current.transform
    np.testing.assert_allclose(run([1., 2.]), run([2., 1.]))


def test_prior_visual_world_can_correct_drift_after_another_world():
    engine = ExternalOdometry(learning=False)
    feed(engine, 0, 0.)
    vision(engine, 0, 0.)
    feed(engine, .5, 1.)
    vision(engine, .5, 50., 'new')
    feed(engine, 1., 2.)
    vision(engine, 1., 1.)
    assert 1.8 < engine.current.transform[0, 3] < 2.


def test_pose_model_change_resets_training_without_resetting_body_position():
    engine = ExternalOdometry()
    feed(engine, 0., 100.)
    vision(engine, 0., 0.)
    for i in range(1, 21):
        at = i/4
        feed(engine, at, 100.+at*1.2)
        vision(engine, at, at)
    before = engine.current.transform.copy()
    assert engine.current.learner.info()['rank'] > 0
    planar = dict(packet(5.25, 200.), motion_model='planar')
    stamp, sample = decode_motion(planar, 1)
    engine.motion(stamp, sample)
    np.testing.assert_allclose(engine.current.transform, before)
    assert engine.current.learner.info()['rank'] == 0
    planar = dict(packet(5.5, 200.25), motion_model='planar')
    stamp, sample = decode_motion(planar, 1)
    engine.motion(stamp, sample)
    assert engine.current.transform[0, 3] == pytest.approx(before[0, 3]+.25)


def feed_covariance(engine, t, diagonal, *, x=0., frame='leg_odom',
                    source='bumi_leg_kinematics', quaternion=None, mount=None, length_scale=1.):
    item = packet(t, x, frame=frame)
    item['vendor'] = {'source': source}
    item['pose']['covariance'] = (np.diag(diagonal).ravel().tolist()
                                   if diagonal is not None else None)
    if quaternion is not None:
        item['pose']['quaternion'] = quaternion
    if mount is not None:
        item['body_from_sensor'] = mount.tolist()
    item['length_scale_m'] = length_scale
    stamp, motion = decode_motion(item, 1)
    return engine.motion(stamp, motion)


def test_bumi_pose_covariance_adds_only_growth_within_one_leg_origin():
    engine = ExternalOdometry(learning=False)
    seed = engine.current.covariance.copy()
    baseline = np.arange(1., 7.)
    step = np.array([.01, .02, .03, .04, .05, .06])
    assert feed_covariance(engine, 0., baseline)
    np.testing.assert_array_equal(engine.current.covariance, seed)
    assert feed_covariance(engine, .5, baseline+step)
    np.testing.assert_allclose(engine.current.covariance, seed+np.diag(step))
    assert not feed_covariance(engine, .5, baseline+step)
    assert feed_covariance(engine, 1., baseline, x=10., frame='new_leg_odom')
    np.testing.assert_allclose(engine.current.covariance, seed+np.diag(step))
    np.testing.assert_array_equal(engine.current.transform, np.eye(4))
    assert feed_covariance(engine, 1.5, baseline+step, x=10., frame='new_leg_odom')
    np.testing.assert_allclose(engine.current.covariance, seed+2*np.diag(step))


def test_bumi_step_noise_is_split_once_when_vision_is_inside_the_interval():
    def run(late):
        engine = ExternalOdometry(learning=False)
        feed_covariance(engine, 0., [1.]*6)
        if not late:
            vision(engine, .25, 0.)
        feed_covariance(engine, .5, [1.125]*6)
        if late:
            vision(engine, .25, 0.)
        return engine
    ordered, delayed = run(False), run(True)
    split = delayed.checkpoints[NS + 250_000_000]
    np.testing.assert_allclose(split.covariance, np.eye(6)*1.0625)
    np.testing.assert_allclose(delayed.current.covariance, np.eye(6)*1.125)
    np.testing.assert_allclose(delayed.current.covariance, ordered.current.covariance)
    np.testing.assert_allclose(delayed.current.transform, ordered.current.transform)


def test_bumi_covariance_uses_body_axes_and_sensor_lever_arm():
    engine = ExternalOdometry(learning=False)
    q = [0., 0., 2**-.5, 2**-.5]
    mount = pose_at(1.)
    start = np.ones(6)
    end = start + [1., 4., 9., 0., 0., .01]
    feed_covariance(engine, 0., start, quaternion=q, mount=mount, length_scale=.001)
    feed_covariance(engine, .5, end, quaternion=q, mount=mount, length_scale=.001)
    increment = engine.current.covariance - np.eye(6)
    np.testing.assert_allclose(np.diag(increment), [4e-6, .010001, 9e-6, 0., 0., .01], atol=1e-12)
    assert increment[1, 5] == pytest.approx(-.01)


@pytest.mark.parametrize('source,left,right', [
    ('slamtec', [1.]*6, [2.]*6),
    ('bumi_leg_kinematics', None, [2.]*6),
    ('bumi_leg_kinematics', [2.]*6, [1.]*6),
    ('bumi_leg_kinematics', [1.]*6, [2., 2., -2., 2., 2., 2.]),
    ('bumi_leg_kinematics', [1.]*6, [2., 2., float('nan'), 2., 2., 2.]),
])
def test_chassis_or_unusable_leg_covariance_uses_existing_noise(source, left, right):
    reference = ExternalOdometry(learning=False)
    feed(reference, 0., 0.)
    feed(reference, .5, .2)
    engine = ExternalOdometry(learning=False)
    assert feed_covariance(engine, 0., left, source=source)
    assert feed_covariance(engine, .5, right, x=.2, source=source)
    np.testing.assert_array_equal(engine.current.transform, reference.current.transform)
    np.testing.assert_array_equal(engine.current.covariance, reference.current.covariance)
