import numpy as np
import pytest

from plugins.external_motion_learning import se3_exp, se3_log
from plugins.external_odometry_fusion import ExternalOdometry, _adjoint, decode_motion
from plugins.rgbd_fusion import quaternion


def stamp(t):
    return 1_000_000_000 + round(t * 1e9)


def rotation(w):
    return se3_exp(np.r_[np.zeros(3), w])


def sample(t, rate, pose=None, *, frame='local', mount=None, noise=None, **fields):
    # Rate and pose arguments refer to the body origin; the wire reports the sensor.
    mount = np.eye(4) if mount is None else mount
    packet = dict(schema='motus.odom/1', stamp_ns=stamp(t), body_from_sensor=mount.tolist(), **fields)
    if rate is not None:
        if None not in rate:
            rate = mount[:3, :3].T @ rate
        packet['twist'] = [None]*3 + list(rate)
    if pose is not None:
        sensor = pose @ mount
        packet['pose'] = dict(frame_id=frame, child_frame_id='sensor', position=sensor[:3, 3].tolist(),
                              quaternion=quaternion(sensor[:3, :3]))
        if noise is not None:
            packet['pose']['process_covariance'] = noise.ravel().tolist()
    return decode_motion(packet, 1)


@pytest.mark.parametrize('with_vision', [False, True])
def test_closing_pose_keeps_out_and_back_turn_and_agreeing_visual_midpoint(with_vision):
    engine = ExternalOdometry(learning=False)
    if with_vision:
        engine.vision(stamp(0), 'native', np.eye(4))
    engine.motion(*sample(0, [0, 0, 1.5], np.eye(4)))
    engine.motion(*sample(.2, [0, 0, -1.5]))
    midpoint = rotation([0, 0, .3])
    if with_vision:
        engine.vision(stamp(.2), 'native', midpoint)
    np.testing.assert_allclose(engine.current.transform, midpoint, atol=1e-12)
    engine.motion(*sample(.4, [0, 0, 0], np.eye(4)))
    np.testing.assert_allclose(engine.checkpoints[stamp(.2)].transform, midpoint, atol=1e-12)
    np.testing.assert_allclose(engine.current.transform, np.eye(4), atol=1e-12)


def test_noncommuting_turn_and_return_keeps_order():
    engine = ExternalOdometry(learning=False)
    rx, ry = rotation([.2, 0, 0]), rotation([0, .3, 0])
    for i, rate in enumerate([[2, 0, 0], [0, 3, 0], [0, -3, 0], [-2, 0, 0]]):
        engine.motion(*sample(i*.1, rate, np.eye(4) if i == 0 else None))
    engine.motion(*sample(.4, None, np.eye(4)))
    for t, expected in zip([.1, .2, .3, .4], [rx, rx @ ry, rx, np.eye(4)]):
        np.testing.assert_allclose(engine.checkpoints[stamp(t)].transform, expected, atol=1e-12)


def test_rate_after_reset_propagates_only_outgoing_motion():
    engine = ExternalOdometry(learning=False)
    engine.motion(*sample(0, [0, 0, 3], np.eye(4)))
    engine.motion(*sample(.05, [0, 0, 1], reset=True))
    np.testing.assert_array_equal(engine.current.transform, np.eye(4))
    assert engine.current.last_update is None
    engine.motion(*sample(.075, [0, 0, 1], np.eye(4)))
    np.testing.assert_allclose(engine.current.transform, rotation([0, 0, .025]), atol=1e-12)
    assert engine.output()['twist'] == [None, None, None, 0., 0., 1.]


def test_anchored_knots_and_covariance_at_arbitrary_subdivisions():
    engine = ExternalOdometry(learning=False)
    start = se3_exp([.7, -.4, .1, -.3, .5, .2])
    mount = se3_exp([.2, -.1, -.3, .2, -.3, .4])
    # Endpoint residual is a z rotation after the integrated x then y turn.
    rx, ry, residual = rotation([.24, 0, 0]), rotation([0, -.3, 0]), rotation([0, 0, .08])
    delta = rx @ ry @ residual
    delta[:3, 3] = [.6, -.3, .2]
    end = start @ delta
    q = np.diag([.02, .04, .08, .1, .3, .5]) + .01*np.ones((6, 6))
    engine.motion(*sample(0, [1.2, 0, 0], start, mount=mount, noise=np.eye(6)))
    engine.motion(*sample(.2, [0, -1.5, 0], mount=mount))
    engine.motion(*sample(.4, None, end, mount=mount, noise=np.eye(6)+q))
    # The measured body translation follows the original screw at the knot.
    mid = se3_exp(.5*se3_log(delta))
    mid[:3, :3] = (rx @ rotation([0, 0, .04]))[:3, :3]
    axes = np.eye(4)
    axes[:3, :3] = (end @ mount)[:3, :3].T
    terminal_transport = _adjoint(mount) @ _adjoint(axes)
    terminal = terminal_transport @ q @ terminal_transport.T
    state = engine.checkpoints[stamp(0)]
    for t in [.067, .2, .243, .39, .4]:
        state = engine._advance(state, stamp(t), sorted(engine.raw_motion))
        expected = (se3_exp((t/.2)*se3_log(mid)) if t <= .2 else
                    mid @ se3_exp(((t-.2)/.2)*se3_log(np.linalg.inv(mid) @ delta)))
        np.testing.assert_allclose(state.transform, expected, atol=1e-12)
        forward = _adjoint(np.linalg.inv(expected))
        back = _adjoint(np.linalg.inv(expected) @ delta)
        noise = forward @ forward.T + (t/.4) * back @ terminal @ back.T
        np.testing.assert_allclose(state.noise, noise, atol=1e-12)
    np.testing.assert_allclose(state.noise, engine.current.noise, atol=1e-12)


def test_calibration_once_and_training_retain_the_raw_angular_pieces():
    engine = ExternalOdometry()
    engine.base.learner.parameters[5] = np.log(2.)
    engine.base.learner.parameters[11] = .1
    engine.vision(stamp(0), 'native', np.eye(4))
    engine.motion(*sample(0, [0, 0, 1.5], np.eye(4)))
    engine.motion(*sample(.2, [0, 0, -1.5]))
    engine.motion(*sample(.4, None, np.eye(4)))
    np.testing.assert_allclose(engine.checkpoints[stamp(.2)].transform, rotation([0, 0, .62]), atol=1e-12)
    np.testing.assert_allclose(engine.current.transform, rotation([0, 0, .04]), atol=1e-12)
    engine.vision(stamp(.4), 'native', engine.current.transform)
    pieces = engine.current.learner.window[-1][2]
    np.testing.assert_allclose([u[5] for u, _, _ in pieces], [1.5, -1.5])
    np.testing.assert_allclose([dt for _, _, dt in pieces], [.2, .2])


@pytest.mark.parametrize('replace', [False, True])
def test_delayed_interior_motion_revises_earlier_knots(replace):
    engine = ExternalOdometry(learning=False)
    engine.motion(*sample(0, [0, 0, 1], np.eye(4)))
    engine.motion(*sample(.1, [0, 0, 1]))
    if replace:
        engine.motion(*sample(.2, [0, 0, 0]))
    engine.motion(*sample(.4, None, np.eye(4)))
    earlier = engine.checkpoints[stamp(.1)].transform
    assert engine.motion(*sample(.2, [0, 0, -1]))
    np.testing.assert_allclose(engine.checkpoints[stamp(.1)].transform, rotation([0, 0, .1]), atol=1e-12)
    assert not np.allclose(engine.checkpoints[stamp(.1)].transform, earlier)
    np.testing.assert_allclose(engine.current.transform, np.eye(4), atol=1e-12)


@pytest.mark.parametrize('opening,interior,model', [
    (None, [0, 0, -1], 'pose'), ([0, 0, 1], [None, 0, -1], 'pose'), ([0, 0, 1], [0, 0, -1], 'planar'),
])
def test_incomplete_gyro_or_planar_model_keeps_endpoint_screw(opening, interior, model):
    engine = ExternalOdometry(learning=False)
    engine.motion(*sample(0, opening, np.eye(4), motion_model=model))
    engine.motion(*sample(.2, interior))
    engine.motion(*sample(.4, None, rotation([0, 0, .1]), motion_model=model))
    np.testing.assert_allclose(engine.checkpoints[stamp(.2)].transform, rotation([0, 0, .05]), atol=1e-12)


@pytest.mark.parametrize('boundary', ['origin', 'model', 'long_gap', 'reset'])
def test_separated_endpoints_do_not_anchor_intermediate_rate(boundary):
    engine = ExternalOdometry(learning=False)
    engine.motion(*sample(0, [0, 0, 1], np.eye(4)))
    engine.motion(*sample(.1, [0, 0, 1]))
    t = 2. if boundary == 'long_gap' else .2
    engine.motion(*sample(t, [0, 0, 1], se3_exp([10, 0, 0, 0, 0, .9]),
                          frame='rebase' if boundary == 'origin' else 'local',
                          motion_model='planar' if boundary == 'model' else 'pose', reset=boundary == 'reset'))
    np.testing.assert_allclose(engine.checkpoints[stamp(.1)].transform, rotation([0, 0, .1]), atol=1e-12)
    np.testing.assert_array_equal(engine.current.transform[:3, 3], np.zeros(3))


@pytest.mark.parametrize('pose_on_left', [False, True])
@pytest.mark.parametrize('elapsed,propagates', [(.249999999, True), (.25, True), (.250000001, False)])
def test_reset_mixed_edge_requires_fresh_right_endpoint(pose_on_left, elapsed, propagates):
    engine = ExternalOdometry(learning=False)
    pose = se3_exp([8., 0., 0., 0., 0., .5])
    engine.motion(*sample(0, [0, 0, 1], pose if pose_on_left else None, reset=True))
    engine.motion(*sample(elapsed, [0, 0, -3], None if pose_on_left else pose))
    expected = rotation([0, 0, elapsed]) if propagates else np.eye(4)
    np.testing.assert_allclose(engine.current.transform, expected, atol=1e-12)
    if propagates:
        assert engine.output()['twist'] == [None, None, None, 0., 0., 1.]
    else:
        assert engine.output() is None


@pytest.mark.parametrize('pose_rate,partial_rate,expected_twist', [
    (None, [0, 0, 1], None),
    ([None, 0, 1], [0, 0, 1], None),
    ([0, 0, 1], [0, 0, None], [None, None, None, 0., 0., None]),
])
def test_reset_full_pose_requires_measured_supported_rate(pose_rate, partial_rate, expected_twist):
    engine = ExternalOdometry(learning=False)
    engine.motion(*sample(0, pose_rate, np.eye(4), reset=True))
    engine.motion(*sample(.025, partial_rate))
    np.testing.assert_array_equal(engine.current.transform, np.eye(4))
    output = engine.output()
    assert (output['twist'] if output else None) == expected_twist


def test_pruning_keeps_cached_knots_within_mutable_motion_history(monkeypatch):
    monkeypatch.setattr('plugins.external_odometry_fusion._MAX_EVENTS', 7)
    engine = ExternalOdometry(learning=False)
    for i in range(60):
        full = i % 3 == 0
        engine.motion(*sample(i*.025, [0, 0, 1 if full else -1], np.eye(4) if full else None))
        assert len(engine.raw_motion) <= 7
        assert sum(map(len, engine._pose_profiles.values())) <= len(engine.raw_motion)
        if full:
            np.testing.assert_allclose(engine.current.transform, np.eye(4), atol=1e-12)
    assert engine.base.stamp > stamp(0)
    assert not engine.motion(*sample(.025, [0, 0, 3]))


@pytest.mark.parametrize('scales,bias', [
    ([1., 1., 1., 1., 1., 1.2], np.zeros(6)),
    ([1.6, .8, 1., .9, 1.4, 1.2], np.array([.1, -.2, .05, .3, -.15, .1])),
])
def test_subdividing_calibrated_turn_preserves_additive_noise(scales, bias):
    engine = ExternalOdometry()
    engine.base.learner.parameters = np.r_[np.log(scales), bias]
    growth = np.diag([.02, .5, .08, .1, .3, .5])
    if bias.any():
        cross_terms = np.array([.08, -.12, .05, .23, -.15, .09])
        growth += np.outer(cross_terms, cross_terms)
    engine.motion(*sample(0, [0, 0, 1.5], np.eye(4), noise=np.zeros((6, 6))))
    engine.motion(*sample(.2, [0, 0, -1.5]))
    engine.motion(*sample(.4, None, np.eye(4), noise=growth))
    times = sorted(engine.raw_motion)
    state = engine.checkpoints[stamp(0)]
    for t in [.1, .2, .3, .4]:
        state = engine._advance(state, stamp(t), times)
    np.testing.assert_allclose(state.transform, engine.current.transform, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.noise, engine.current.noise, rtol=0, atol=1e-12)
    # Half the growth belongs to each physical endpoint. The first share
    # returns through the calibrated closing turn, after scaling at the knot.
    to_knot = _adjoint(rotation([0, 0, -.3]))
    calibrated_rate = np.array([0., 0., 0., 0., 0., 1.8])
    opening = se3_exp(.2 * (calibrated_rate + bias))
    closing = se3_exp(.2 * (-calibrated_rate + bias))
    from_start = _adjoint(np.linalg.inv(opening @ closing))
    from_knot = _adjoint(np.linalg.inv(closing))
    units = np.diag(scales)
    mapped = from_knot @ units @ to_knot
    expected = from_start @ from_start.T + .5 * (mapped @ growth @ mapped.T + units @ growth @ units.T)
    np.testing.assert_allclose(state.noise, expected, rtol=0, atol=1e-12)
