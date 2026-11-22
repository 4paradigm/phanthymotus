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


def leg_attitude_packet(t, position, yaw, angular, *, frame='leg', child='body', source='bumi_leg_kinematics', reset=False):
    item = packet(t, position, twist=[4., 5., 6., 0., 0., angular], frame=frame, reset=reset)
    item['pose'].update(quaternion=[0., 0., np.sin(yaw/2), np.cos(yaw/2)], child_frame_id=child)
    item['vendor'] = {'source': source}
    return item


def test_leg_origin_change_preserves_attitude_only_with_measured_body_rate():
    engine = ExternalOdometry(learning=False)
    for item in (leg_attitude_packet(0., 10., 0., .6),
                 leg_attitude_packet(.1, 10.1, 0., .6),
                 leg_attitude_packet(.31, -20., .126, .6, frame='new_leg')):
        engine.motion(*decode_motion(item, 1))
    np.testing.assert_allclose(engine.current.transform[:3, 3], [.1, 0., 0.])
    assert np.arctan2(engine.current.transform[1, 0], engine.current.transform[0, 0]) == pytest.approx(.126)
    assert engine.current.mask.tolist() == [False, False, False, True, True, True]
    assert engine.current.twist[5] == pytest.approx(.6)


def test_leg_attitude_uses_only_the_axes_observed_by_both_endpoints():
    engine = ExternalOdometry(learning=False)
    first = leg_attitude_packet(0., 10., 0., .6)
    second = leg_attitude_packet(.21, -20., .126, .6, frame='new_leg')
    for item in (first, second):
        item['motion_model'] = 'planar'
        engine.motion(*decode_motion(item, 1))
    assert engine.current.mask.tolist() == [False, False, False, False, False, True]


@pytest.mark.parametrize('override', [dict(source='chassis'), dict(child='another_body'),
                                      dict(reset=True), dict(t=.5), dict(yaw=1.3), dict(angular=None)])
def test_leg_attitude_does_not_cross_unverified_source_resets(override):
    engine = ExternalOdometry(learning=False)
    first = leg_attitude_packet(0., 10., 0., .6)
    second = dict(t=.21, position=-20., yaw=.126, angular=.6, frame='new_leg')
    second.update(override)
    engine.motion(*decode_motion(first, 1))
    engine.motion(*decode_motion(leg_attitude_packet(**second), 1))
    assert engine.current.last_update is None
    np.testing.assert_array_equal(engine.current.transform, np.eye(4))


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


def test_unchanged_published_endpoint_does_not_recompute_pose_correction(monkeypatch):
    from plugins import external_odometry_fusion as fusion

    engine = ExternalOdometry(learning=False)
    feed(engine, 0., 10.)
    feed(engine, .2, 10.2)
    engine.output()
    feed(engine, .4, 10.4)
    correction = fusion.se3_log
    calls = []

    def record(transform):
        calls.append(transform)
        return correction(transform)

    monkeypatch.setattr(fusion, 'se3_log', record)
    assert engine.output()['pose']['position'][0] == pytest.approx(.4)
    assert calls == []

    # A visual correction at the published capture can revise the checkpoint.
    # Publication must still smooth a real revision in the original output gauge.
    vision(engine, .2, 0.)
    vision(engine, .4, .1)
    feed(engine, .6, 10.6)
    calls.clear()
    out = engine.output()['pose']['position'][0]
    assert len(calls) == 2  # Smoothing, then the remaining pose offset for covariance.
    assert out > engine.current.transform[0, 3]

    # Once a correction fits within one update, reuse the measured checkpoint
    # instead of carrying a numerical difference into later publications.
    feed(engine, .8, 10.8)
    assert engine.output()['pose']['position'][0] == pytest.approx(engine.current.transform[0, 3])
    feed(engine, 1., 11.)
    calls.clear()
    assert engine.output()['pose']['position'][0] == pytest.approx(engine.current.transform[0, 3])
    assert calls == []


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


def test_late_visual_reuses_leg_noise_until_an_endpoint_changes(monkeypatch):
    from plugins import external_odometry_fusion as fusion

    calculate = fusion._leg_increment_covariance
    calls = []

    def record(*args):
        calls.append(args)
        return calculate(*args)

    monkeypatch.setattr(fusion, '_leg_increment_covariance', record)
    engine = ExternalOdometry(learning=False)
    for t, variance in [(0., 1.), (.25, 1.125), (.5, 1.25), (.75, 1.5)]:
        assert feed_covariance(engine, t, [variance]*6)
    assert len(calls) == 3
    assert vision(engine, .25, 0.)
    assert len(calls) == 3
    assert feed_covariance(engine, .5, [1.375]*6)
    assert len(calls) == 5

    expected = ExternalOdometry(learning=False)
    for t, variance in [(0., 1.), (.25, 1.125), (.5, 1.375), (.75, 1.5)]:
        feed_covariance(expected, t, [variance]*6)
    vision(expected, .25, 0.)
    np.testing.assert_array_equal(engine.current.covariance, expected.current.covariance)


def test_repeated_measured_steps_reuse_motion_even_when_rewinding(monkeypatch):
    from plugins.external_motion_learning import ExternalMotionLearner

    engine = ExternalOdometry(learning=False)
    for t, x in [(0., 0.), (.25, .4), (.5, .6), (.75, .7)]:
        feed(engine, t, x)

    correct = ExternalMotionLearner.corrected_velocity
    calls = []

    def record(learner, velocity, observed):
        calls.append(velocity.copy())
        return correct(learner, velocity, observed)

    monkeypatch.setattr(ExternalMotionLearner, 'corrected_velocity', record)
    assert vision(engine, .25, 0.)
    assert calls == []
    assert engine.current.transform[0, 3] == pytest.approx(.7)
    assert feed(engine, .75, .8)
    assert len(calls) == 1
    assert engine.current.transform[0, 3] == pytest.approx(.8)


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


def test_bumi_covariance_uses_measured_endpoint_and_only_interpolates_split(monkeypatch):
    from plugins import external_odometry_fusion as fusion

    engine = ExternalOdometry(learning=False)
    feed_covariance(engine, 0., [1.]*6)
    feed_covariance(engine, .5, [1.125]*6, quaternion=[0., 0., .5, np.sqrt(.75)])
    left, right = engine.raw_motion.values()
    velocity = np.array([0., 0., 0., 0., 0., np.pi*2/3])
    interpolate = fusion.se3_exp
    calls = []

    def record(increment):
        calls.append(increment)
        return interpolate(increment)

    monkeypatch.setattr(fusion, 'se3_exp', record)
    full = fusion._leg_increment_covariance(left, right, velocity, .5, 1., True)
    assert calls == []
    partial = fusion._leg_increment_covariance(left, right, velocity, .25, .5, False)
    assert len(calls) == 1
    np.testing.assert_allclose(full, np.eye(6)*.125, atol=1e-12)
    np.testing.assert_allclose(partial, np.eye(6)*.0625, atol=1e-12)


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


def test_clipped_visual_correction_retains_the_unapplied_update_uncertainty():
    engine = ExternalOdometry(learning=False)
    vision(engine, 0., 0.)
    feed(engine, 0., 0.)
    feed(engine, 1., .3)
    vision(engine, 1., 0.)
    # The nominal gain is 2/3: the intended 20 cm correction is limited to
    # 5 cm. The remaining 15 cm must not disappear from reported uncertainty.
    assert engine.current.transform[0, 3] == pytest.approx(.25)
    assert engine.current.covariance[0, 0] >= .15**2
    assert np.linalg.eigvalsh(engine.current.covariance).min() >= 0.


def test_very_uncertain_vision_does_not_force_a_minimum_correction():
    engine = ExternalOdometry(learning=False)
    vision(engine, 0., 0.)
    feed(engine, 0., 0.)
    feed(engine, 1., 0.)
    engine.vision(2*NS, 'vision', pose_at(.3), np.full(6, 10_000.))
    assert abs(engine.current.transform[0, 3]) < 1e-6
    assert engine.current.covariance[0, 0] < .002


def test_smoothed_publication_includes_its_offset_from_the_estimated_pose():
    engine = ExternalOdometry(learning=False)
    vision(engine, 0., 0.)
    feed(engine, 0., 0.)
    feed(engine, .5, .5)
    assert engine.output()['pose']['position'][0] == pytest.approx(.5)
    vision(engine, .5, 0.)
    feed(engine, .6, .6)
    output = engine.output()['pose']
    # Late correction moved the estimate back 5 cm; publication catches up
    # by only 3 cm, leaving a known 2 cm difference from that estimate.
    assert engine.current.transform[0, 3] == pytest.approx(.55)
    assert output['position'][0] == pytest.approx(.57)
    assert output['covariance'][0] >= engine.current.covariance[0, 0] + .02**2 - 1e-12


def test_clipped_turn_keeps_the_actual_position_offset_in_covariance():
    from plugins.external_motion_learning import se3_exp

    engine = ExternalOdometry(learning=False)
    variance = np.full(6, .0001)
    engine.vision(NS, 'vision', np.eye(4), variance)
    feed(engine, 0., twist=[0.]*6)
    feed(engine, 1., twist=[0.]*6)
    increment = np.array([1., 0., 0., 0., 0., 1.5])
    engine.vision(2*NS, 'vision', se3_exp(increment), variance)
    output = engine.output()['pose']
    nominal = se3_exp(.8 * increment)
    offset = nominal[:3, 3] - output['position']
    covariance = np.asarray(output['covariance']).reshape(6, 6)
    assert offset[1] > .4
    assert covariance[1, 1] >= offset[1]**2
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_smoothed_turn_keeps_world_position_and_rotation_offsets_in_covariance():
    from plugins.external_motion_learning import se3_exp

    engine = ExternalOdometry(learning=False)
    variance = np.array([.0001, .0004, .0009, .0001, .0004, .0009])
    engine.vision(2*NS, 'replacement', np.eye(4), variance)
    engine.output()
    engine.vision(NS, 'original', se3_exp(np.array([1., 0., 0., 0., 0., 1.5])), variance)
    engine.vision(3*NS, 'replacement', np.eye(4), variance)
    output = engine.output()['pose']
    offset = np.r_[engine.current.transform[:3, 3] - output['position'], 0., 0., 1.5-.03]
    covariance = np.asarray(output['covariance']).reshape(6, 6)
    assert offset[1] > .6
    assert covariance[1, 1] >= offset[1]**2
    # Removing the known mean offset must leave the estimator's uncertainty,
    # including translation/rotation cross terms from the shared offset.
    remainder = covariance - np.outer(offset, offset)
    rotation = np.array([[np.cos(1.5), -np.sin(1.5), 0.],
                         [np.sin(1.5), np.cos(1.5), 0.], [0., 0., 1.]])
    axes = np.zeros((6, 6))
    axes[:3, :3] = axes[3:, 3:] = rotation
    np.testing.assert_allclose(remainder, axes @ engine.current.covariance @ axes.T, atol=1e-12)


def test_learning_uses_uncertainty_from_both_visual_endpoints():
    engine = ExternalOdometry()
    feed(engine, 0., 0.)
    engine.vision(NS, 'vision', pose_at(0.), np.full(6, .01))
    feed(engine, .2, 0.)
    engine.vision(NS+200_000_000, 'vision', pose_at(0.), np.full(6, .04))
    # Without known cross-correlation, each relative-component sigma is
    # conservatively bounded by the sum of endpoint sigmas: 0.1 + 0.2.
    np.testing.assert_allclose(engine.current.learner.window[0][5], [.3]*6)


def test_learning_transports_start_uncertainty_to_the_end_body_axes():
    engine = ExternalOdometry()
    feed(engine, 0., 0.)
    engine.vision(NS, 'vision', pose_at(0.), np.array([.01, .04, .09, .01, .04, .09]))
    feed(engine, .2, 0.)
    turned = np.eye(4)
    turned[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    engine.vision(NS+200_000_000, 'vision', turned, np.array([.16, .25, .36, .16, .25, .36]))
    # A quarter turn swaps the start x/y sigmas before adding the end ones.
    np.testing.assert_allclose(engine.current.learner.window[0][5], [.6, .6, .9, .6, .6, .9])
