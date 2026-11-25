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


def attitude_packet(t, position, yaw, angular, *, frame='odometry', child='body', source='independent_motion', reset=False):
    item = packet(t, position, twist=[4., 5., 6., 0., 0., angular], frame=frame, reset=reset)
    item['pose'].update(quaternion=[0., 0., np.sin(yaw/2), np.cos(yaw/2)], child_frame_id=child)
    item['vendor'] = {'source': source}
    return item


def test_replay_reuses_attitude_check_and_propagation_for_unchanged_endpoints(monkeypatch):
    from plugins import external_odometry_fusion as fusion

    checks = []
    jacobians = []
    corroborate = fusion._origin_attitude
    parameter_jacobian = fusion.ExternalMotionLearner.velocity_parameter_jacobian

    def observe(left, right, interval_ns):
        checks.append(right)
        return corroborate(left, right, interval_ns)

    def observe_jacobian(learner, velocity, calibrated):
        jacobians.append(velocity)
        return parameter_jacobian(learner, velocity, calibrated)

    monkeypatch.setattr(fusion, '_origin_attitude', observe)
    monkeypatch.setattr(fusion.ExternalMotionLearner, 'velocity_parameter_jacobian', observe_jacobian)
    engine = ExternalOdometry()
    start, left = decode_motion(attitude_packet(0., 0., 0., .6), 1)
    end, first = decode_motion(attitude_packet(.1, .1, .06, .6), 1)
    _, second = decode_motion(attitude_packet(.1, .2, .8, .6), 1)
    first, second = sorted((first, second), key=lambda motion: motion.order_key())
    engine.motion(start, left)
    engine.motion(end, first)
    engine.vision(start, 'native', pose_at(0.))
    assert len(checks) == 1
    assert len(jacobians) == 1
    assert engine.motion(end, second)
    assert len(checks) == 2 and checks[-1] is second
    assert len(jacobians) == 2
    piece, = engine._intervals([start, end], start, end)
    assert piece[-1] == (corroborate(left, second, end-start) is not None)
    assert len(checks) == 2


def test_origin_change_preserves_attitude_only_with_measured_body_rate():
    engine = ExternalOdometry(learning=False)
    for item in (attitude_packet(0., 10., 0., .6),
                 attitude_packet(.1, 10.1, 0., .6),
                 attitude_packet(.31, -20., .126, .6, frame='new_origin')):
        engine.motion(*decode_motion(item, 1))
    np.testing.assert_allclose(engine.current.transform[:3, 3], [.1, 0., 0.])
    assert np.arctan2(engine.current.transform[1, 0], engine.current.transform[0, 0]) == pytest.approx(.126)
    assert engine.current.mask.tolist() == [False, False, False, True, True, True]
    assert engine.current.twist[5] == pytest.approx(.6)


def test_attitude_uses_only_the_axes_observed_by_both_endpoints():
    engine = ExternalOdometry(learning=False)
    first = attitude_packet(0., 10., 0., .6)
    second = attitude_packet(.21, -20., .126, .6, frame='new_origin')
    for item in (first, second):
        item['motion_model'] = 'planar'
        engine.motion(*decode_motion(item, 1))
    assert engine.current.mask.tolist() == [False, False, False, False, False, True]


@pytest.mark.parametrize('override', [dict(child='another_body'),
                                      dict(reset=True), dict(t=.5), dict(yaw=1.3), dict(angular=None)])
def test_attitude_does_not_cross_unverified_origin_resets(override):
    engine = ExternalOdometry(learning=False)
    first = attitude_packet(0., 10., 0., .6)
    second = dict(t=.21, position=-20., yaw=.126, angular=.6, frame='new_origin')
    second.update(override)
    engine.motion(*decode_motion(first, 1))
    engine.motion(*decode_motion(attitude_packet(**second), 1))
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


def test_new_publication_reflects_revised_published_checkpoints():
    engine = ExternalOdometry(learning=False)
    feed(engine, 0., 10.)
    feed(engine, .2, 10.2)
    engine.output()
    feed(engine, .4, 10.4)
    assert engine.output()['pose']['position'][0] == pytest.approx(.4)

    # A visual correction at the published capture can revise the checkpoint.
    # Its accepted correction belongs in the next new measurement publication.
    vision(engine, .2, 0.)
    vision(engine, .4, .1)
    assert engine.output() is None
    feed(engine, .6, 10.6)
    assert engine.current.transform[0, 3] == pytest.approx(.55)
    assert engine.output()['pose']['position'][0] == pytest.approx(.55)

    feed(engine, .8, 10.8)
    assert engine.output()['pose']['position'][0] == pytest.approx(.75)
    feed(engine, 1., 11.)
    assert engine.output()['pose']['position'][0] == pytest.approx(.95)


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


def feed_covariance(engine, t, diagonal, *, x=0., frame='odometry',
                    source='independent_motion', quaternion=None, mount=None, length_scale=1.):
    item = packet(t, x, frame=frame)
    item['vendor'] = {'source': source}
    item['pose']['process_covariance'] = (np.diag(diagonal).ravel().tolist()
                                   if diagonal is not None else None)
    if quaternion is not None:
        item['pose']['quaternion'] = quaternion
    if mount is not None:
        item['body_from_sensor'] = mount.tolist()
    item['length_scale_m'] = length_scale
    stamp, motion = decode_motion(item, 1)
    return engine.motion(stamp, motion)


def test_process_covariance_adds_only_growth_within_one_origin():
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


def test_late_visual_reuses_process_noise_until_an_endpoint_changes(monkeypatch):
    from plugins import external_odometry_fusion as fusion

    calculate = fusion._pose_increment_covariance
    calls = []

    def record(*args):
        calls.append(args)
        return calculate(*args)

    monkeypatch.setattr(fusion, '_pose_increment_covariance', record)
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


def test_process_noise_is_split_once_when_vision_is_inside_the_interval():
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


def test_process_covariance_uses_measured_endpoint_and_only_interpolates_split(monkeypatch):
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
    full = fusion._pose_increment_covariance(left, right, velocity, .5, 1., True)
    assert calls == []
    partial = fusion._pose_increment_covariance(left, right, velocity, .25, .5, False)
    assert len(calls) == 1
    np.testing.assert_allclose(full, np.eye(6)*.125, atol=1e-12)
    np.testing.assert_allclose(partial, np.eye(6)*.0625, atol=1e-12)


def test_process_covariance_uses_body_axes_and_sensor_lever_arm():
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
    ('bumi_leg_kinematics', None, [2.]*6),
    ('bumi_leg_kinematics', [2.]*6, [1.]*6),
    ('bumi_leg_kinematics', [1.]*6, [2., 2., -2., 2., 2., 2.]),
    ('bumi_leg_kinematics', [1.]*6, [2., 2., float('nan'), 2., 2., 2.]),
])
def test_unusable_process_covariance_uses_existing_noise(source, left, right):
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


def test_publication_uses_corrected_estimate_and_its_uncertainty():
    engine = ExternalOdometry(learning=False)
    vision(engine, 0., 0.)
    feed(engine, 0., 0.)
    feed(engine, .5, .5)
    assert engine.output()['pose']['position'][0] == pytest.approx(.5)
    vision(engine, .5, 0.)
    feed(engine, .6, .6)
    output = engine.output()['pose']
    # The filter limits this late correction to 5 cm and retains uncertainty
    # for its unapplied part. Publication must use that same corrected mean.
    assert engine.current.transform[0, 3] == pytest.approx(.55)
    assert output['position'][0] == pytest.approx(.55)
    np.testing.assert_allclose(np.asarray(output['covariance']).reshape(6, 6),
                               engine.current.covariance, atol=1e-12)


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


def test_replayed_publication_transports_full_covariance_to_world_axes():
    from plugins.rgbd_imu import rotation_from_quaternion

    engine = ExternalOdometry(learning=False)
    variance = np.array([.01, .04, .09, .16, .25, .36])
    engine.vision(2*NS, 'replacement', np.eye(4), variance)
    engine.output()
    original = np.array([[0., 0., 1., 1.], [1., 0., 0., 2.],
                         [0., 1., 0., 3.], [0., 0., 0., 1.]])
    relative = np.array([[1., 0., 0., 1.], [0., 0., -1., 0.],
                         [0., 1., 0., 0.], [0., 0., 0., 1.]])
    engine.vision(NS, 'original', original, variance)
    assert engine.output() is None
    engine.vision(3*NS, 'replacement', relative, variance)
    output = engine.output()['pose']
    np.testing.assert_allclose(output['position'], [1., 3., 3.], atol=1e-12)
    np.testing.assert_allclose(rotation_from_quaternion(output['quaternion']),
                               [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]], atol=1e-12)
    # The original rotation permutes x/y/z to y/z/x. The new one-metre
    # translation along world y carries the two alignment endpoint variances:
    # angular world x adds .72 to position z and their cross covariance;
    # angular world z adds .50 to position x and a negative cross covariance.
    # The noncommuting final turn also permutes the new visual variance.
    expected = np.array([[.72, 0., 0., 0., 0., -.50],
                         [0., .03, 0., 0., 0., 0.],
                         [0., 0., .89, .72, 0., 0.],
                         [0., 0., .72, .97, 0., 0.],
                         [0., 0., 0., 0., .48, 0.],
                         [-.50, 0., 0., 0., 0., .86]])
    np.testing.assert_allclose(np.asarray(output['covariance']).reshape(6, 6), expected, atol=1e-12)


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


@pytest.mark.parametrize('source', ['bumi_leg_kinematics', 'wheel_encoder', 'independent_motion'])
@pytest.mark.parametrize('mount', [None, np.eye(4)])
def test_numeric_pose_uncertainty_and_attitude_do_not_depend_on_vendor_name(source, mount):
    engine = ExternalOdometry(learning=False)
    for t, frame, yaw, noise in [(0., 'first', 0., 0.), (.1, 'first', 0., .1),
                                  (.3, 'second', .12, 0.)]:
        item = attitude_packet(t, 0., yaw, .6, frame=frame, source=source)
        item['pose']['process_covariance'] = (np.eye(6)*noise).ravel().tolist()
        if mount is not None:
            item['body_from_sensor'] = mount.tolist()
        engine.motion(*decode_motion(item, 1))
    np.testing.assert_allclose(engine.checkpoints[NS+100_000_000].covariance, np.eye(6)*1.1)
    assert np.arctan2(engine.current.transform[1, 0], engine.current.transform[0, 0]) == pytest.approx(.12)


@pytest.mark.parametrize('source', ['bumi_leg_kinematics', 'wheel_encoder'])
def test_total_pose_uncertainty_is_not_a_process_accumulator(source):
    engine, reference = ExternalOdometry(learning=False), ExternalOdometry(learning=False)
    for t, noise in [(0., .1), (.5, 100.)]:
        item = packet(t, 0.)
        item['vendor'] = {'source': source}
        item['pose']['covariance'] = (np.eye(6)*noise).ravel().tolist()
        engine.motion(*decode_motion(item, 1))
        feed(reference, t, 0.)
    np.testing.assert_allclose(engine.current.covariance, reference.current.covariance)


def twist_noise_engine(covariance, *, scale=1., mount=None, end_covariance=None):
    engine = ExternalOdometry(learning=False)
    for t, noise in [(0., covariance), (.5, covariance if end_covariance is None else end_covariance)]:
        item = packet(t, twist=[0.]*6)
        item['linear_scale_mps'] = scale
        item['twist_covariance'] = np.asarray(noise).ravel().tolist()
        if mount is not None:
            item['body_from_sensor'] = mount.tolist()
        engine.motion(*decode_motion(item, 1))
    return engine


def test_twist_noise_follows_units_rotation_and_reference_point():
    raw = np.diag([.04, .01, .02, .001, .002, .09])
    metres = twist_noise_engine(raw)
    units = np.diag([1000.]*3+[1.]*3)
    millimetres = twist_noise_engine(units @ raw @ units, scale=.001)
    np.testing.assert_allclose(millimetres.current.covariance, metres.current.covariance)
    quarter_turn = np.eye(4)
    quarter_turn[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    turned = twist_noise_engine(raw, mount=quarter_turn)
    np.testing.assert_allclose(np.diag(turned.current.covariance)-1.,
                               [.005, .02, .01, .001, .0005, .045])
    shifted = twist_noise_engine(raw, mount=pose_at(1.))
    # v_body = v_sensor + [1,0,0] x omega: y variance includes yaw,
    # z variance includes pitch, and the y/yaw cross term is negative.
    added = shifted.current.covariance-np.eye(6)
    np.testing.assert_allclose(np.diag(added), [.02, .05, .011, .0005, .001, .045])
    assert added[1, 5] == pytest.approx(-.045)
    assert added[2, 4] == pytest.approx(.001)
    assert np.linalg.eigvalsh(added).min() > 0.


def test_held_twist_uses_its_own_uncertainty_before_the_next_sample():
    engine = twist_noise_engine(np.eye(6)*.01, end_covariance=np.eye(6)*100.)
    np.testing.assert_allclose(engine.current.covariance, np.eye(6)*1.005)


@pytest.mark.parametrize('external', [False, True])
def test_repeated_visual_motion_keeps_latest_measurement_uncertainty(external):
    engine = ExternalOdometry(learning=False)
    variance = np.full(6, .01)
    if external:
        feed(engine, -.1, 0.)
        feed(engine, 0., 0.)
    engine.vision(NS, 'native', pose_at(0.), variance)
    for i in range(1, 101):
        x = .1*(-1)**i
        engine.vision(NS+i*100_000_000, 'native', pose_at(x), variance)
        assert engine.current.transform[0, 3] == pytest.approx(x)
        assert engine.current.covariance[0, 0] >= .01-1e-12
    # No accumulation per repeated observation when the same visual alignment
    # is simply carried forward; also no R/N claim for an unaveraged pose.
    first = engine.checkpoints[NS+200_000_000].covariance
    np.testing.assert_allclose(engine.current.covariance, first, atol=1e-12)


@pytest.mark.parametrize('delayed_partial', [False, True])
def test_partial_twist_between_valid_pose_endpoints_does_not_lose_displacement(delayed_partial):
    engine, reference = ExternalOdometry(), ExternalOdometry()
    feed_covariance(engine, 0., [0.]*6, x=0.)
    if not delayed_partial:
        feed(engine, .05, twist=[None, None, None, 0., 0., 0.])
    feed_covariance(engine, .1, [.01]*6, x=.1)
    if delayed_partial:
        feed(engine, .05, twist=[None, None, None, 0., 0., 0.])
    feed_covariance(reference, 0., [0.]*6, x=0.)
    feed_covariance(reference, .1, [.01]*6, x=.1)
    np.testing.assert_allclose(engine.current.transform, reference.current.transform)
    np.testing.assert_allclose(engine.current.noise, reference.current.noise, atol=1e-14)
    np.testing.assert_allclose(engine.current.sensitivity, reference.current.sensitivity, atol=1e-14)
    np.testing.assert_allclose(engine.current.covariance, reference.current.covariance, atol=1e-14)


def test_available_gyro_continues_while_pose_changes_origin():
    engine = ExternalOdometry(learning=False)
    engine.motion(*decode_motion(attitude_packet(0., 10., 0., 2.), 1))
    feed(engine, .05, twist=[None, None, None, 0., 0., 2.])
    feed(engine, .1, twist=[None, None, None, 0., 0., 2.])
    engine.motion(*decode_motion(attitude_packet(.15, -10., .3, 2., frame='new'), 1))
    assert np.arctan2(engine.current.transform[1, 0], engine.current.transform[0, 0]) == pytest.approx(.3)
    np.testing.assert_allclose(engine.current.transform[:3, 3], 0.)


@pytest.mark.parametrize('reset,new_origin', [(True, False), (False, True)])
def test_partial_twist_cannot_bridge_reset_or_unknown_pose_origin(reset, new_origin):
    engine = ExternalOdometry(learning=False)
    feed(engine, 0., 0.)
    feed(engine, .05, twist=[None, None, None, 0., 0., 0.], reset=reset)
    feed(engine, .1, .1, frame='different' if new_origin else 'external')
    np.testing.assert_allclose(engine.current.transform, np.eye(4))


def test_finite_step_parameter_covariance_is_invariant_to_subdivision():
    def run(pieces):
        engine = ExternalOdometry()
        for i in range(pieces+1):
            feed(engine, i/pieces, twist=[1., 0., 0., 0., 0., np.pi/2])
        state = engine.current
        return state.sensitivity @ state.learner.covariance @ state.sensitivity.T
    full, split = run(1), run(20)
    np.testing.assert_allclose(full, split, atol=1e-12)
    # Turning through 90 degrees couples uncertain yaw calibration into y.
    assert full[1, 1] == pytest.approx(.103204696, abs=1e-9)


@pytest.mark.parametrize('boundary', ['reset', 'full_twist'])
@pytest.mark.parametrize('replace_partial', [False, True])
def test_delayed_boundary_rebuilds_every_interpolated_checkpoint(boundary, replace_partial):
    gyro = [None, None, None, 0., 0., 0.]
    # The nonzero final rate makes this full twist win the same-stamp tie.
    stop = packet(.2, reset=True) if boundary == 'reset' else packet(.2, twist=[0., 0., 0., 0., 0., 1.])

    def run(delayed):
        engine = ExternalOdometry()
        feed_covariance(engine, 0., [0.]*6)
        vision(engine, 0., 0.)
        feed(engine, .1, twist=gyro)
        vision(engine, .12, 0.)
        if replace_partial:
            feed(engine, .2, twist=gyro)
        if not delayed:
            assert engine.motion(*decode_motion(stop, 1))
        feed_covariance(engine, .3, [.03]*6, x=.3)
        if delayed:
            assert engine.checkpoints[NS+100_000_000].transform[0, 3] == pytest.approx(.1)
            assert engine.motion(*decode_motion(stop, 1))
        return engine.checkpoints

    ordered, delayed = run(False), run(True)
    for stamp, expected in ordered.items():
        actual = delayed[stamp]
        np.testing.assert_allclose(actual.transform, expected.transform)
        np.testing.assert_allclose(actual.noise, expected.noise)
        np.testing.assert_allclose(actual.sensitivity, expected.sensitivity)
        assert len(actual.learner.window) == len(expected.learner.window)


@pytest.mark.parametrize('alternate', ['planar_pose', 'full_twist'])
@pytest.mark.parametrize('reverse', [False, True])
def test_alternate_full_models_cannot_continue_gyro(alternate, reverse):
    engine = ExternalOdometry()
    alternate_index = 1 if reverse else 0
    for i in range(2):
        item = attitude_packet(.1*i, 10.*i, 0., 2.*(1-i))
        if i == alternate_index:
            if alternate == 'planar_pose':
                item['motion_model'] = 'planar'
                item['pose'].update(frame_id='another_world', child_frame_id='another_body')
            else:
                item['pose'] = None
        engine.motion(*decode_motion(item, 1))
    np.testing.assert_allclose(engine.current.transform, np.eye(4))
    assert engine.current.last_update is None


def test_replay_time_limit_keeps_pose_anchor_for_delayed_boundary():
    def run(delayed):
        engine = ExternalOdometry()
        feed(engine, 0., 0.)
        feed(engine, .1, twist=[None, None, None, 0., 0., 0.])
        if not delayed:
            feed(engine, .2, reset=True)
        feed(engine, .3, .3)
        # The nominal replay floor now falls inside the earlier pose bracket.
        feed(engine, 10.15, .3)
        if delayed:
            assert feed(engine, .2, reset=True)
        return engine.current

    ordered, delayed = run(False), run(True)
    np.testing.assert_allclose(delayed.transform, ordered.transform)
    np.testing.assert_allclose(delayed.noise, ordered.noise)
    np.testing.assert_allclose(delayed.sensitivity, ordered.sensitivity)


def test_replay_event_limit_freezes_complete_pose_bracket(monkeypatch):
    monkeypatch.setattr('plugins.external_odometry_fusion._MAX_EVENTS', 4)
    engine = ExternalOdometry()
    feed(engine, 0., 0.)
    for t in (.05, .1):
        feed(engine, t, twist=[None, None, None, 0., 0., 0.])
    feed(engine, .15, twist=[None, None, None, 0., 0., 0.])
    for i in range(4, 12):
        feed(engine, i*.05, i*.05)
        assert len(engine.checkpoints) <= 4 and len(engine.raw_motion) <= 4
        if i == 4:
            assert engine.base.stamp == NS+200_000_000
            assert not feed(engine, .1, reset=True)
            assert engine.pending_output_states(NS)[-1].stamp == engine.current.stamp
        assert engine.current.transform[0, 3] == pytest.approx(i*.05)


@pytest.mark.parametrize('generic', [False, True])
def test_measured_planar_chassis_keeps_its_motion_through_visual_updates(generic):
    engine = ExternalOdometry()
    for step in range(5):
        sample = packet(step * .2, 10. + step * .2)
        sample['vendor'] = {'motus_slam': dict(status='tracking', planar=True,
            pose_reference='body', pose_source='chassis_odometry', segment_id='one',
            stamp_source='measurement', stamp_ns=sample['stamp_ns'])}
        if generic:
            sample['vendor']['motus_slam'].update(pose_kind='measured', pose_source='body_tracker')
        engine.motion(*decode_motion(sample, 1))
        visual = pose_at(step * .3)
        pitch = step * .04
        visual[:3, :3] = [[np.cos(pitch), 0., np.sin(pitch)], [0., 1., 0.],
                           [-np.sin(pitch), 0., np.cos(pitch)]]
        visual[2, 3] = step * .1
        engine.vision(sample['stamp_ns'], 'camera', visual)
    np.testing.assert_allclose(engine.current.transform, pose_at(.8), atol=1e-9)
    # Without a closing chassis measurement, visual motion still advances the map.
    engine.vision(sample['stamp_ns'] + 100_000_000, 'camera', visual @ pose_at(.1))
    np.testing.assert_allclose(engine.current.transform, pose_at(.9), atol=1e-9)


@pytest.mark.parametrize('origin', ['one', 'reset'])
def test_measured_chassis_does_not_integrate_a_pose_jump(origin):
    engine = ExternalOdometry()
    for step, (x, segment) in enumerate([(0., 'one'), (.1, 'one'), (20., origin)]):
        sample = packet(step * .1, x)
        sample['vendor'] = {'motus_slam': dict(status='tracking', planar=True, pose_reference='body',
            pose_source='chassis_odometry', segment_id=segment, stamp_source='measurement',
            stamp_ns=sample['stamp_ns'])}
        engine.motion(*decode_motion(sample, 1))
        if step == 1:
            previous = decode_slam_odom(engine.output())
    np.testing.assert_allclose(engine.current.transform, pose_at(.1))
    assert engine.current.last_update is None
    sample['stamp_ns'] += 100_000_000
    sample['vendor']['motus_slam']['stamp_ns'] = sample['stamp_ns']
    sample['pose']['position'][0] += .1
    engine.motion(*decode_motion(sample, 1))
    resumed = decode_slam_odom(engine.output())
    assert resumed.external_origin != previous.external_origin
    assert resumed.position == pytest.approx((.2, 0., 0.))

def test_measured_spatial_body_preserves_height_and_roll():
    engine = ExternalOdometry()
    for step in range(5):
        sample = packet(step*.1, step*.05)
        angle = step*.03
        sample['pose'].update(position=[step*.05, 0., step*.02],
                              quaternion=[np.sin(angle/2), 0., 0., np.cos(angle/2)])
        sample['vendor'] = {'motus_slam': dict(status='tracking', planar=False,
            pose_reference='body', pose_kind='measured', segment_id='one',
            stamp_source='measurement', stamp_ns=sample['stamp_ns'])}
        engine.motion(*decode_motion(sample, 1))
        engine.vision(sample['stamp_ns'], 'camera', pose_at(step*.3))
    np.testing.assert_allclose(engine.current.transform[:3, 3], [.2, 0., .08], atol=1e-9)
    assert engine.current.transform[2, 1] == pytest.approx(np.sin(.12))


def test_explicit_estimate_overrides_legacy_chassis_declaration():
    sample = packet(0., 0.)
    sample['vendor'] = {'motus_slam': dict(status='tracking', planar=True, pose_kind='estimated',
        pose_reference='body', pose_source='chassis_odometry', segment_id='one',
        stamp_source='measurement', stamp_ns=sample['stamp_ns'])}
    assert decode_motion(sample, 1)[1].model == ('pose',)


@pytest.mark.parametrize('gauge_tilt', [0., 1.2])
@pytest.mark.parametrize('delayed', [False, True])
def test_planar_pose_fuses_only_footprint_and_preserves_visual_height_and_tilt(gauge_tilt, delayed):
    from plugins.external_motion_learning import se3_exp
    from plugins.external_odometry_fusion import pose_matrix
    engine = ExternalOdometry()
    gauge = se3_exp(np.array([0., 0., 0., gauge_tilt, 0., 0.]))
    pitch = se3_exp(np.array([0., 0., 0., 0., np.deg2rad(-4.), 0.]))
    roll = se3_exp(np.array([0., 0., 0., np.deg2rad(5.), 0., 0.]))
    observation = pitch @ roll
    observation[2, 3] = .08

    def chassis(t, x, yaw):
        sample = dict(packet(t, x), motion_model='planar')
        sample['pose']['quaternion'] = [0., 0., np.sin(yaw/2), np.cos(yaw/2)]
        engine.motion(*decode_motion(sample, 1))

    chassis(0., 0., 0.)
    engine.vision(NS, 'camera', gauge)
    if delayed:
        engine.vision(NS+100_000_000, 'camera', gauge @ observation)
    chassis(.1, .10, .1)
    if not delayed:
        engine.vision(NS+100_000_000, 'camera', gauge @ observation)
    fused = np.linalg.solve(gauge, pose_matrix(engine.output()['pose']))
    yaw = np.arctan2(fused[1, 0], fused[0, 0])
    assert 0 < fused[0, 3] < .10 and 0 < yaw < .1
    assert fused[2, 3] == pytest.approx(.08)
    expected = se3_exp(np.array([0., 0., 0., 0., 0., yaw])) @ observation
    np.testing.assert_allclose(fused[:3, :3], expected[:3, :3], atol=1e-9)
    # A later footprint translation and turn, without vision, must keep the last tilt and height.
    chassis(.2, .20, .15)
    carried = np.linalg.solve(gauge, pose_matrix(engine.output()['pose']))
    turn = se3_exp(np.array([0., 0., 0., 0., 0., .05]))
    assert carried[2, 3] == pytest.approx(.08)
    np.testing.assert_allclose(carried[:3, :3], turn[:3, :3] @ fused[:3, :3], atol=1e-9)
    assert np.linalg.eigvalsh(engine.current.covariance).min() >= -1e-12


def test_planar_learning_uses_footprint_displacement_during_a_visual_bob():
    from plugins.external_motion_learning import se3_exp
    engine = ExternalOdometry()
    for i in range(3):
        stamp, motion = decode_motion(dict(packet(.1*i, .1*i), motion_model='planar'), 1)
        engine.motion(stamp, motion)
        observation = se3_exp(np.array([0., 0., 0., 0., -.2 if i else 0., 0.]))
        observation[:3, 3] = [.1*i, 0., .08*i]
        engine.vision(stamp, 'camera', observation)
    # Training must compare the same two planar poses as the chassis path.
    visual = engine.current.learner.window[-1][4]
    assert visual[0, 3] == pytest.approx(.1)
    assert visual[1, 3] == pytest.approx(0.)


def test_planar_pose_ignores_unobserved_chassis_channels_before_taking_difference():
    engine = ExternalOdometry(learning=False)
    for step in (0, 1):
        sample = dict(packet(.1*step, .1*step), motion_model='planar')
        pitch = -.25*step
        sample['pose'].update(position=[.1*step, 0., .1*step],
                              quaternion=[0., np.sin(pitch/2), 0., np.cos(pitch/2)])
        engine.motion(*decode_motion(sample, 1))
    np.testing.assert_allclose(engine.current.transform, pose_at(.1), atol=1e-9)


def test_planar_output_uses_only_capture_matched_body_gravity():
    from plugins.external_odometry_fusion import pose_matrix
    engine = ExternalOdometry(body_frame='body')
    for idx in range(5):
        t = idx / 10
        p = packet(t, x=t)
        p['motion_model'] = 'planar'
        s = p['stamp_ns']
        gravity = dict(frame_id='body', stamp_ns=s, up=[0., .6, .8])
        if idx == 2:
            gravity['variance_bound_rad2'] = .9
        if idx == 3:
            gravity['frame_id'] = 'camera'
        if idx == 4:
            gravity['stamp_ns'] -= NS
        p['vendor'] = dict(body_gravity=gravity)
        engine.motion(*decode_motion(p, 1))
        if idx == 0:
            continue
        yaw = .35 * idx
        visual = pose_at(t)
        visual[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        visual[2, 3] = .08 * idx
        engine.vision(s, 'native', visual)
        output = engine.output()
        M = pose_matrix(output['pose'])
        assert M[:3, 3] == pytest.approx(engine.current.transform[:3, 3])
        expected = engine.current.transform
        assert np.arctan2(M[1,0], M[0,0]) == pytest.approx(np.arctan2(expected[1,0], expected[0,0]))
        if idx == 2:
            assert abs(expected[2,3]) > .05
            assert abs(np.arctan2(expected[1,0], expected[0,0])) > .01
        assert M[2,:3] == pytest.approx([0., .6, .8] if idx < 3 else [0., 0., 1.])
        covariance = np.reshape(output['pose']['covariance'], (6, 6))
        assert np.linalg.eigvalsh(covariance).min() >= -1e-9
        if idx == 2:
            assert np.linalg.eigvalsh(covariance[3:,3:])[-1] >= .9


def test_gravity_variance_preserves_heading_uncertainty_without_copying_it_into_tilt():
    engine = ExternalOdometry(learning=False)
    initial = packet(0, x=0.)
    initial['motion_model'] = 'planar'
    engine.motion(*decode_motion(initial, 1))
    engine.vision(NS, 'vision', pose_at(0.), np.array([.1, .1, .1, .1, .1, 2.]))
    sample = dict(packet(.1, x=.1), motion_model='planar')
    pitch = np.pi/3
    bound = .01
    sample['vendor'] = dict(body_gravity=dict(frame_id='body', stamp_ns=sample['stamp_ns'],
        up=[-np.sin(pitch), 0., np.cos(pitch)], variance_bound_rad2=bound))
    engine.motion(*decode_motion(sample, 1))
    heading_var = engine.current.covariance[5, 5]
    assert heading_var > 1.
    forward = np.array([np.cos(pitch), 0., -np.sin(pitch)])
    expected = np.diag([0., bound, heading_var]) + np.outer(forward, forward) * bound / np.cos(pitch)**2
    covariance = np.reshape(engine.output()['pose']['covariance'], (6, 6))
    np.testing.assert_allclose(covariance[3:, 3:], expected, atol=1e-10)


@pytest.mark.parametrize('boundary', ['continuous', 'origin', 'reset', 'expired'])
def test_short_gravity_gap_transports_body_rotation_only_within_one_reference(boundary):
    from dataclasses import replace
    from plugins.external_odometry_fusion import pose_matrix

    def tilted(degrees):
        c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
        T = np.eye(4)
        T[1:3, 1:3] = [[c, -s], [s, c]]
        return T

    engine = ExternalOdometry(body_frame='body')
    for t, angle in ((.1, 12.), (.8 if boundary == 'expired' else .2, 15.)):
        item = dict(packet(t, x=t, frame='other' if t > .1 and boundary == 'origin' else 'external'),
                    motion_model='planar')
        if t == .1:
            item['vendor'] = dict(body_gravity=dict(frame_id='body', stamp_ns=item['stamp_ns'],
                                                    up=[0., 0., 1.]))
        stamp, motion = decode_motion(item, 1)
        engine.raw_motion[stamp] = replace(motion, reset=t > .1 and boundary == 'reset')
        state = replace(engine.current, stamp=stamp, transform=tilted(angle), last_update='external',
                        motion_model=motion.model, mask=motion.mask)
        out = engine.output(state)
        if t == .1:
            assert pose_matrix(out['pose'])[2, :3] == pytest.approx([0., 0., 1.])
    expected = tilted(3. if boundary == 'continuous' else 15.)
    assert pose_matrix(out['pose'])[2, :3] == pytest.approx(expected[2, :3])
    assert 'body_gravity_stamp_ns' not in out['vendor']['external_fusion']
    covariance = np.array(out['pose']['covariance']).reshape(6, 6)
    assert np.linalg.eigvalsh(covariance).min() >= -1.e-9
