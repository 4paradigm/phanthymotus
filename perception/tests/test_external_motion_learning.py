"""Independent holdout checks for external odometry scale and bias correction."""
from copy import deepcopy

import numpy as np
import pytest

from plugins.external_motion_learning import ExternalMotionLearner, se3_exp, se3_log, se3_right_jacobian


def _pose(intervals, scale=None, bias=None):
    result = np.eye(4)
    scale = np.ones(6) if scale is None else np.asarray(scale)
    bias = np.zeros(6) if bias is None else np.asarray(bias)
    for velocity, dt, observed in intervals:
        mask = np.ones(6) if observed is None else observed
        result = result @ se3_exp((scale * np.asarray(velocity) + bias) * mask * dt)
    return result


def _intervals(rng, lengths=(.11, .09, .15, .15)):
    return [(rng.uniform(-1., 1., 6), dt, None) for dt in lengths]


def _add(learner, index, intervals, expected=None):
    start = index * 500_000_000
    return learner.add(start, start + 500_000_000, intervals,
                       _pose(intervals) if expected is None else expected)


@pytest.mark.parametrize("motion", [
    [1., -2., 3., 0., 0., 0.],
    [1.1, .4, -.6, -.4, .9, 1.7],
    [.2, .3, -.4, 0., 0., np.pi - 1e-7],
])
def test_body_increment_log_exp_round_trip(motion):
    transform = se3_exp(np.asarray(motion))
    np.testing.assert_allclose(se3_log(transform), motion, atol=1e-8)
    if motion[-1] and motion[0]:
        assert not np.allclose(transform[:3, 3], motion[:3])


@pytest.mark.parametrize("motion", [
    [1.7, -2.2, .9, -1.2, .8, 2.4],
    [-.8, .3, 1.2, 1e-9, -2e-9, 3e-9],
    [0.] * 6,
])
def test_se3_right_jacobian_matches_body_local_exponential_derivative(motion):
    motion = np.array(motion)
    inverse = np.linalg.inv(se3_exp(motion))
    epsilon = 1e-6
    columns = []
    for delta in epsilon * np.eye(6):
        plus = se3_log(inverse @ se3_exp(motion + delta))
        minus = se3_log(inverse @ se3_exp(motion - delta))
        columns.append((plus - minus) / (2. * epsilon))
    np.testing.assert_allclose(se3_right_jacobian(motion), np.column_stack(columns), atol=2e-9)


def test_mixed_motion_learning_improves_independent_blind_interval():
    rng = np.random.default_rng(4)
    scale = np.array([.91, 1.12, .94, 1.13, .88, 1.09])
    bias = np.array([-.02, .043, .012, -.037, .04, -.022])
    learned, baseline = ExternalMotionLearner(), ExternalMotionLearner(enabled=False)
    for index in range(22):
        intervals = _intervals(rng)
        visual = _pose(intervals, scale, bias)
        _add(learned, index, intervals, visual)
        _add(baseline, index, intervals, visual)
    info = learned.info()
    assert info["rank"] == 12
    np.testing.assert_allclose(info["scale"], scale, atol=.004)
    np.testing.assert_allclose(info["bias"], bias, atol=.004)
    assert np.all(np.linalg.eigvalsh(learned.covariance) > 0)

    # Both trajectories start from the same corrected anchor. The held-out
    # motions are never given to either learner as training observations.
    actual = predicted = uncorrected = se3_exp(np.array([1., -.2, 0., 0., .2, 0.]))
    test_motion = [(rng.uniform(-.4, 1.2, 6), .1, None) for _ in range(80)]
    actual = actual @ _pose(test_motion, scale, bias)
    for velocity, dt, observed in test_motion:
        predicted = predicted @ learned.correct(velocity, dt, observed)
        uncorrected = uncorrected @ baseline.correct(velocity, dt, observed)
    learned_error = np.linalg.norm(se3_log(np.linalg.inv(predicted) @ actual))
    baseline_error = np.linalg.norm(se3_log(np.linalg.inv(uncorrected) @ actual))
    assert baseline_error > .1
    assert learned_error < .05 * baseline_error
    assert learned.info()["effective_after_ns"] == info["effective_after_ns"]


@pytest.mark.parametrize("lateral", [-.3, .3])
def test_translation_model_mismatch_cannot_train_stationary_rotation(lateral):
    learner = ExternalMotionLearner()
    for index in range(30):
        velocity = np.array([[1.5, .2][index % 2], 0., 0., 0., 0., 0.])
        visual = np.eye(4)
        visual[:3, 3] = velocity[0] * .25 * np.array([1., lateral, 0.])
        start = index * 250_000_000
        learner.add(start, start + 250_000_000, [(velocity, .25, None)], visual)

    # The native and raw orientations agree exactly throughout training. A
    # translational model error supplies no evidence for an angular correction.
    np.testing.assert_allclose(learner.info()["scale"][3:], 1., atol=1e-10)
    np.testing.assert_allclose(learner.info()["bias"][3:], 0., atol=1e-10)
    np.testing.assert_allclose(learner.correct(np.zeros(6), 5.)[:3, :3], np.eye(3), atol=1e-10)
    assert learner.rank > 0


def test_angular_fit_and_uncertainty_ignore_translation_outliers():
    clean, corrupted = ExternalMotionLearner(), ExternalMotionLearner()
    scale = np.array([1., 1., 1., 1.13, .88, 1.09])
    bias = np.array([0., 0., 0., -.037, .04, -.022])
    rng = np.random.default_rng(42)
    for index in range(22):
        intervals = _intervals(rng)
        visual = _pose(intervals, scale, bias)
        _add(clean, index, intervals, visual)
        visual[:3, 3] += rng.uniform(-3., 3., 3)
        _add(corrupted, index, intervals, visual)

    angular = [3, 4, 5, 9, 10, 11]
    np.testing.assert_allclose(corrupted.info()["scale"][3:], scale[3:], atol=.004)
    np.testing.assert_allclose(corrupted.info()["bias"][3:], bias[3:], atol=.004)
    np.testing.assert_allclose(corrupted.parameters[angular], clean.parameters[angular], atol=1e-10)
    np.testing.assert_allclose(corrupted.covariance[np.ix_(angular, angular)],
                               clean.covariance[np.ix_(angular, angular)], atol=1e-10)


def test_conditional_linear_fit_retains_angular_uncertainty_and_cross_covariance():
    learner = ExternalMotionLearner()
    for index in range(20):
        velocity = np.array([[1.5, .2][index % 2], 0., 0., 0., 0., 0.])
        visual = np.eye(4)
        visual[0, 3] = velocity[0] * .5
        _add(learner, index, [(velocity, .5, None)], visual)

    # Twenty independent half-second rotations estimate yaw bias with variance
    # 2 sigma_w² / (20 dt²) = .001. A yaw-bias perturbation produces y
    # displacement vx dt²/2, so linear bias responds by -mean(vx) dt/2=-.2125.
    # Its variance includes both its own measurement noise and yaw uncertainty.
    assert learner.covariance[11, 11] == pytest.approx(.001, rel=1e-6)
    assert learner.covariance[7, 11] == pytest.approx(-.0002125, rel=1e-6)
    assert learner.covariance[7, 7] == pytest.approx(.00029515625, rel=1e-6)
    assert learner.covariance[5, 5] == pytest.approx(.0625)
    np.testing.assert_allclose(learner.covariance, learner.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(learner.covariance)[0] > 0.


def test_only_one_measured_angular_axis_can_receive_bias():
    learner = ExternalMotionLearner()
    yaw_only = np.array([False, False, False, False, False, True])
    known_bias = -.13
    for index, omega in enumerate([-.5, .8, -1., 1.2, -.2, .5, -.7, .9]):
        velocity = np.array([0., 0., 0., 0., 0., omega])
        intervals = [(velocity, .5, yaw_only)]
        visual = se3_exp(np.array([1., 0., 0., 0., 0., (.9 * omega + known_bias) * .5]))
        _add(learner, index, intervals, visual)
    assert learner.info()["rank"] == 2
    assert learner.info()["bias"][-1] == pytest.approx(known_bias, abs=.005)
    result = learner.correct([None, None, None, None, None, .6], .5, yaw_only)
    np.testing.assert_allclose(result[:3, 3], 0.)
    np.testing.assert_allclose(se3_log(result)[-1], (.9 * .6 + known_bias) * .5, atol=.005)
    jacobian = learner.velocity_parameter_jacobian([None, None, None, None, None, .6], yaw_only)
    assert np.count_nonzero(jacobian) == 2


@pytest.mark.parametrize("axes", [[5], [0, 1, 5]])
def test_unmeasured_roll_then_pitch_cannot_train_yaw_bias(axes):
    learner = ExternalMotionLearner()
    mask = np.isin(np.arange(6), axes)
    motion = [(np.zeros(6), .5, mask)]
    visual = se3_exp(np.array([0., 0., 0., .2, 0., 0.])) @ se3_exp(
        np.array([0., 0., 0., 0., .2, 0.]))
    prior = learner.covariance.copy()
    for index in range(16):
        _add(learner, index, motion, visual)
    np.testing.assert_array_equal(learner.parameters, 0.)
    np.testing.assert_array_equal(learner.covariance, prior)


@pytest.mark.parametrize("tilt, sigma, eligible", [
    (.004, .01, True),
    (.02, .003, False),
    (.20, 1., False),
])
def test_missing_rotation_gate_uses_visual_precision(tilt, sigma, eligible):
    learner = ExternalMotionLearner()
    motion = [(np.zeros(6), .5, np.eye(6, dtype=bool)[5])]
    visual = se3_exp(np.array([.4, 0., 0., tilt, 0., -.05]))
    for index in range(8):
        start = index * 500_000_000
        learner.add(start, start + 500_000_000, motion, visual, noise=np.full(6, sigma))
    assert learner.info()["bias"][5] == pytest.approx(-.1 if eligible else 0., abs=1e-6)


def test_supported_planar_motion_still_calibrates_with_unobserved_height():
    learner = ExternalMotionLearner()
    mask = np.array([True, True, False, False, False, True])
    bias = np.array([.04, -.02, 0., 0., 0., -.07])
    for index in range(8):
        velocity = np.array([.5, -.3, 0., 0., 0., .8]) * (-1)**index
        visual = se3_exp((velocity + bias) * .5)
        visual[2, 3] = .4
        _add(learner, index, [(velocity, .5, mask)], visual)
    np.testing.assert_allclose(learner.info()["bias"], bias, atol=1e-5)


def test_constant_speed_reports_unobservable_scale_bias_split():
    learner = ExternalMotionLearner()
    x_only = np.array([True, False, False, False, False, False])
    interval = [(np.array([1., 0., 0., 0., 0., 0.]), .5, x_only)]
    visual = se3_exp(np.array([.45, 0., 0., 0., 0., 0.]))
    for index in range(10):
        _add(learner, index, interval, visual)
    assert learner.info()["rank"] == 1
    assert learner.correct(interval[0][0], .5, x_only)[0, 3] == pytest.approx(.45, abs=.005)
    assert np.linalg.eigvalsh(learner.covariance[np.ix_([0, 6], [0, 6])])[-1] > .02
    # A single measurement constrains the combined scale/bias effect. Reversing
    # the same speed exposes the uncertainty in their unmeasured split.
    direction, = learner.info()["observable_directions"]
    assert direction[0] * direction[6] > 0.
    forward = learner.velocity_parameter_jacobian(interval[0][0], x_only)[0]
    reverse = learner.velocity_parameter_jacobian(-interval[0][0], x_only)[0]
    assert reverse @ learner.covariance @ reverse > 10. * (forward @ learner.covariance @ forward)


def test_checkpoint_replay_deduplicates_intervals_and_preserves_covariance():
    rng = np.random.default_rng(8)
    learner = ExternalMotionLearner()
    intervals = [_intervals(rng) for _ in range(16)]
    visuals = [_pose(values, [1.1, .9, 1., 1., .9, 1.1]) for values in intervals]
    for index in range(7):
        _add(learner, index, intervals[index], visuals[index])
    checkpoint = deepcopy(learner)
    for index in range(7, 16):
        _add(learner, index, intervals[index], visuals[index])
    for index in range(7, 16):
        _add(checkpoint, index, intervals[index], visuals[index])
        assert not _add(checkpoint, index, intervals[index], visuals[index])
    np.testing.assert_allclose(checkpoint.info()["scale"], learner.info()["scale"], atol=1e-12)
    np.testing.assert_allclose(checkpoint.covariance, learner.covariance, atol=1e-12)
    assert checkpoint.info()["effective_after_ns"] == learner.info()["effective_after_ns"]
    assert checkpoint.info()["samples"] == learner.info()["samples"]


def test_one_large_visual_outlier_is_downweighted():
    rng = np.random.default_rng(12)
    learner = ExternalMotionLearner()
    for index in range(20):
        intervals = _intervals(rng)
        visual_noise = rng.normal(0., [.001] * 3 + [.002] * 3)
        visual = _pose(intervals) @ se3_exp(visual_noise)
        if index == 13:
            visual = visual @ se3_exp(np.array([1.5, -1.2, .4, .4, .3, -.8]))
        _add(learner, index, intervals, visual)
    np.testing.assert_allclose(learner.info()["scale"], 1., atol=.12)
    np.testing.assert_allclose(learner.info()["bias"], 0., atol=.045)


@pytest.mark.parametrize("observed", [np.ones(6, dtype=bool), np.eye(6, dtype=bool)[0]])
@pytest.mark.parametrize("jitter", [0., .001])
def test_steady_speed_calibration_generalizes_to_rest_and_reverse(observed, jitter):
    learner = ExternalMotionLearner()
    previous_bias_variance = learner.covariance[6, 6]
    forward = np.array([.3, 0., 0., 0., 0., 0.])
    for index in range(16):
        velocity = forward.copy()
        velocity[0] += (-1)**index * jitter
        visual = np.eye(4)
        visual[0, 3] = .8 * velocity[0] * .5
        _add(learner, index, [(velocity, .5, observed)], visual)

    np.testing.assert_allclose(learner.correct(np.zeros(6), 10., observed), np.eye(4), atol=1e-7)
    assert learner.corrected_velocity(-forward, observed)[0] == pytest.approx(-.24, abs=.001)
    assert learner.covariance[6, 6] == pytest.approx(previous_bias_variance)
    np.testing.assert_allclose(learner.covariance, learner.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(learner.covariance)[0] > 0.


def test_steady_window_preserves_previously_identified_bias():
    learner = ExternalMotionLearner()
    x_only = np.eye(6, dtype=bool)[0]
    for index in range(36):
        raw = [-.4, .3][index % 2] if index < 12 else .3
        velocity = np.array([raw, 0., 0., 0., 0., 0.])
        visual = np.eye(4)
        visual[0, 3] = (.8 * raw + .035) * .5
        _add(learner, index, [(velocity, .5, x_only)], visual)
    np.testing.assert_allclose(learner.info()["scale"][0], .8, atol=1e-5)
    np.testing.assert_allclose(learner.info()["bias"][0], .035, atol=1e-5)
    calibrated = learner.parameters.copy()

    # A later same-speed discrepancy supports adjusting one parameter only.
    visual[0, 3] += .005
    for index in range(36, 56):
        _add(learner, index, [(velocity, .5, x_only)], visual)
    assert learner.parameters[6] == pytest.approx(calibrated[6], abs=1e-7)
    assert learner.corrected_velocity(velocity, x_only)[0] == pytest.approx(.285, abs=.001)


def test_zero_raw_velocity_supports_bias_alone():
    learner = ExternalMotionLearner()
    velocity = np.zeros(6)
    x_only = np.eye(6, dtype=bool)[0]
    visual = np.eye(4)
    visual[0, 3] = .012
    prior_scale_variance = learner.covariance[0, 0]
    for index in range(16):
        _add(learner, index, [(velocity, .5, x_only)], visual)
    assert learner.corrected_velocity(velocity, x_only)[0] == pytest.approx(.024, abs=.0001)
    assert learner.info()["scale"][0] == pytest.approx(1.)
    assert learner.covariance[0, 0] == pytest.approx(prior_scale_variance)


def test_subnoise_jitter_about_zero_preserves_unobserved_scale():
    learner = ExternalMotionLearner()
    x_only = np.eye(6, dtype=bool)[0]
    prior_scale_variance = learner.covariance[0, 0]
    for index in range(16):
        jitter = (-1)**index * .002
        velocity = np.array([jitter, 0., 0., 0., 0., 0.])
        visual = np.eye(4)
        visual[0, 3] = (.024 + 1.3 * jitter) * .5
        _add(learner, index, [(velocity, .5, x_only)], visual)

    assert learner.info()["bias"][0] == pytest.approx(.024, abs=1e-5)
    assert learner.info()["scale"][0] == pytest.approx(1.)
    assert learner.covariance[0, 0] == pytest.approx(prior_scale_variance)


def test_corroborated_attitude_stays_in_translation_model_without_training_angular_bias():
    learner = ExternalMotionLearner()
    for index in range(16):
        sign = (-1)**index
        rate = np.array([sign*.7, .2, 0., 0., 0., sign*.8])
        measured = rate.copy()
        measured[:3] *= [1.1, .9, 1.]
        target = se3_exp(measured*.4)
        target[:3, :3] = target[:3, :3] @ se3_exp([0., 0., 0., .15, 0., 0.])[:3, :3]
        start = index*400_000_000
        learner.add(start, start+400_000_000, [(rate, .4, None, True)], target)
    assert learner.rank > 0
    np.testing.assert_allclose(learner.info()['scale'][3:], 1.)
    np.testing.assert_allclose(learner.info()['bias'][3:], 0.)
