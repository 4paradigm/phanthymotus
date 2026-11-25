"""Partial depth constraints preserve the inertial gauge and cannot map alone."""
from copy import deepcopy

import numpy as np
import pytest

from plugins.rgbd_fusion import InertialFilter, exp_rotation, log_rotation, skew
from plugins.rgbd_geometry import DepthRegistration
from test_rgbd_fusion import G, pose
from test_rgbd_geometry import K, known_motion, render_room


def render_plane(transform, normal=(0., 0., 1.), distance=2., shape=(240, 320), calibration=K):
    """Independent axial ray/plane intersection; transform maps camera to world."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    y, x = np.indices(shape)
    rays = np.stack(((x-calibration[0, 2])/calibration[0, 0],
                     (y-calibration[1, 2])/calibration[1, 1], np.ones(x.shape)), -1)
    direction = rays @ transform[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (distance-normal@transform[:3, 3])/(direction@normal)
    valid = np.isfinite(depth) & (depth >= .2) & (depth <= 8.)
    return np.rint(np.where(valid, depth, 0.)*1000).astype(np.uint16)


def partial_result(normal=(0., 0., 1.)):
    truth = known_motion()
    prior = np.eye(4)
    angle = np.deg2rad(4.)
    prior[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    prior[:3, 3] = [.065, -.045, .024]
    reference = render_plane(np.eye(4), normal)
    current = render_plane(truth, normal)
    result = DepthRegistration(K).register(current, reference, prior)
    return result, truth, prior, current, reference


@pytest.mark.parametrize('normal', [(0., 0., 1.), (.3, -.4, .866)])
def test_partial_plane_corrects_observed_motion_and_keeps_original_prior_gauge(normal):
    result, truth, prior, _, _ = partial_result(normal)
    assert result is not None
    assert result['rank'] == 3
    normal = np.asarray(normal, dtype=float); normal /= np.linalg.norm(normal)
    correction = result['transform'][:3, 3]-prior[:3, 3]
    tangent = np.eye(3)-np.outer(normal, normal)
    np.testing.assert_allclose(tangent@correction, 0., atol=3e-5)
    rotation = log_rotation(result['transform'][:3, :3]@prior[:3, :3].T)
    assert abs(normal@rotation) < 3e-5
    assert abs(normal@(result['transform'][:3, 3]-truth[:3, 3])) < .001
    np.testing.assert_allclose(normal@result['transform'][:3, :3],
                               normal@truth[:3, :3], atol=.001)
    assert result['projected_covariance'].shape == (3, 3)
    assert result['projection'].shape == (3, 6)
    assert 'covariance' not in result
    np.testing.assert_allclose(result['projection']@result['observable_lift'], np.eye(3), atol=1e-10)
    assert np.linalg.eigvalsh(result['projected_covariance']).min() > 0
    assert np.isfinite(result['condition']) and result['inliers'] >= 100


def render_tunnel(transform):
    y, x = np.indices((240, 320))
    rays = np.stack(((x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1], np.ones(x.shape)), -1)
    direction = rays@transform[:3, :3].T
    intersections = []
    for axis, coordinate in ((0, -.9), (0, .9), (1, -.65), (1, .65)):
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (coordinate-transform[axis, 3])/direction[..., axis]
        intersections.append(np.where(depth > 0, depth, np.inf))
    depth = np.minimum.reduce(intersections)
    return np.rint(np.where(depth <= 8., depth, 0.)*1000).astype(np.uint16)


def test_open_tunnel_preserves_the_unobserved_forward_prior():
    truth = known_motion()
    prior = truth.copy(); prior[:3, 3] += [.012, -.009, .09]
    result = DepthRegistration(K).register(render_tunnel(truth), render_tunnel(np.eye(4)), prior)
    assert result is not None and result['rank'] == 5
    assert result['transform'][2, 3] == pytest.approx(prior[2, 3], abs=3e-5)
    np.testing.assert_allclose(result['transform'][:2, 3], truth[:2, 3], atol=.003)


def test_full_rank_result_retains_legacy_covariance_and_explicit_rank():
    truth = known_motion()
    result = DepthRegistration(K).register(render_room(truth), render_room(np.eye(4)), np.eye(4))
    assert result is not None and result['rank'] == 6
    np.testing.assert_allclose(result['observable_lift']@result['projected_covariance']@
                               result['observable_lift'].T, result['covariance'], atol=1e-10)


def test_tilted_plane_quantization_does_not_invent_constraints_at_full_resolution():
    calibration = np.array([[460., 0, 319.5], [0, 460., 239.5], [0, 0, 1.]])
    normal = np.array([.3, -.4, .866]); normal /= np.linalg.norm(normal)
    truth = known_motion(); prior = truth.copy(); prior[:3, 3] += [.02, -.01, .01]
    current = render_plane(truth, normal, shape=(480, 640), calibration=calibration)
    reference = render_plane(np.eye(4), normal, shape=(480, 640), calibration=calibration)
    result = DepthRegistration(calibration).register(current, reference, prior)
    assert result is not None and result['rank'] == 3
    np.testing.assert_allclose((np.eye(3)-np.outer(normal, normal))@
                               (result['transform'][:3, 3]-prior[:3, 3]), 0., atol=.0001)


@pytest.mark.parametrize('correlated', [False, True])
def test_projected_filter_updates_only_supported_uncoupled_coordinates(correlated):
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    f.P = np.eye(15)*.01
    projection = np.eye(6)[[2, 3, 4]]
    measurement = pose(); measurement['position'] = [.25, -.2, .02]
    before = f.P.copy()
    assert f.correct(measurement, np.eye(3)*.001, projection=projection,
                     observable_lift=projection.T, correlated=correlated)
    np.testing.assert_allclose(f.p[:2], 0., atol=1e-14)
    assert 0 < f.p[2] < .02
    assert f.P[2, 2] < before[2, 2]
    np.testing.assert_allclose(f.P[[0, 1, 8], [0, 1, 8]],
                               before[[0, 1, 8], [0, 1, 8]]/(.9 if correlated else 1.))
    assert np.isfinite(f.P).all() and np.linalg.eigvalsh(f.P).min() > 0


def test_projected_innovation_gate_uses_number_of_observed_directions():
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    f.P = np.eye(15)*.0001
    measurement = pose(np.sqrt(.0002*15.))  # NIS=15: reject rank1, pass rank6.
    snapshot = deepcopy(f.__dict__)
    projection = np.eye(6)[[0]]
    assert not f.correct(measurement, np.array([[.0001]]), projection=projection,
                         observable_lift=projection.T)
    assert f.rejection == 'innovation_gate'
    np.testing.assert_array_equal(f.p, snapshot['p'])
    np.testing.assert_array_equal(f.P, snapshot['P'])


def test_geometry_projection_transports_reference_uncertainty_and_right_error_lever():
    from plugins.rgbd_fusion import geometry_projection, observation_covariance, quaternion
    result = partial_result((.3, -.4, .866))[0]
    reference_rotation = exp_rotation(np.array([.2, -.1, .3]))
    reference = pose(quaternion=quaternion(reference_rotation))
    reference['position'] = [.3, -.2, .1]
    cross = np.eye(6)*.01; cross[1, 5] = .005; cross[4, 0] = -.003
    reference['covariance'] = (cross@cross.T).ravel().tolist()
    measured, covariance, projection, lift = geometry_projection(reference, result)
    f = InertialFilter(1_000_000_000, measured, [0, 0, 0], [0, 0, G], [0, 0, 0], [.03, -.02, .01])
    world = np.eye(6); world[3:, 3:] = f.R
    analytic = projection@world@f.measurement_matrix()
    numeric = np.zeros((3, 15)); epsilon = 1e-7
    camera = f.p-f.R@f.lever
    for axis in range(15):
        position, rotation = f.p.copy(), f.R.copy()
        if axis < 3:
            position[axis] += epsilon
        if 6 <= axis < 9:
            delta = np.zeros(3); delta[axis-6] = epsilon
            rotation = rotation@exp_rotation(delta)
        change = np.r_[position-rotation@f.lever-camera, log_rotation(rotation@f.R.T)]/epsilon
        numeric[:, axis] = projection@change
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=2e-6)
    reference_jacobian = np.zeros((6, 6))
    translation = result['transform'][:3, 3]
    nominal = np.asarray(reference['position'])+reference_rotation@translation
    for axis in range(6):
        position = np.asarray(reference['position']).copy(); rotation = reference_rotation.copy()
        if axis < 3:
            position[axis] += epsilon
        else:
            delta = np.zeros(3); delta[axis-3] = epsilon
            rotation = exp_rotation(delta)@rotation
        reference_jacobian[:, axis] = np.r_[position+rotation@translation-nominal,
                                             log_rotation(rotation@reference_rotation.T)]/epsilon
    reference_projection = projection@reference_jacobian
    expected = result['projected_covariance'] + reference_projection@observation_covariance(reference)@reference_projection.T
    np.testing.assert_allclose(covariance, expected, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(projection@lift, np.eye(3), atol=1e-10)


@pytest.mark.parametrize('correlated', [False, True])
def test_projected_update_matches_dense_solution_with_state_cross_covariance(correlated):
    from plugins.rgbd_fusion import pose_values, quaternion
    rotation = exp_rotation(np.array([.1, -.2, .3]))
    f = InertialFilter(1_000_000_000, pose(quaternion=quaternion(rotation)), [.01, 0, 0],
                       [0, 0, G], [0, 0, 0], [.03, .01, -.02])
    random = np.random.default_rng(719).normal(size=(15, 15))*.01
    f.P = random@random.T + np.eye(15)*.001
    projection = np.eye(6)[[1, 2, 4]]
    measurement = f.camera_pose()
    measurement['position'] = (np.asarray(measurement['position'])+[.002, -.003, .001]).tolist()
    measurement['quaternion'] = quaternion(rotation@exp_rotation(np.array([.001, -.002, .003])))
    covariance = np.diag([.0002, .0003, .0004])
    position, observed_rotation = pose_values(measurement)
    world = np.eye(6); world[3:, 3:] = rotation
    local_projection = projection@world
    residual = local_projection@np.r_[position-(f.p-f.R@f.lever), log_rotation(f.R.T@observed_rotation)]
    h = local_projection@f.measurement_matrix()
    prior = f.P/(.9 if correlated else 1.)
    noise = covariance/(.1 if correlated else 1.)
    gain = np.linalg.solve(h@prior@h.T+noise, h@prior).T
    delta = gain@residual
    expected = [f.p+delta[:3], f.v+delta[3:6], f.R@exp_rotation(delta[6:9]),
                f.bg+delta[9:12], f.ba+delta[12:15]]
    identity = np.eye(15)-gain@h
    reset = np.eye(15); reset[6:9, 6:9] -= .5*skew(delta[6:9])
    posterior = reset@(identity@prior@identity.T+gain@noise@gain.T)@reset.T
    assert f.correct(measurement, covariance, projection=projection,
                     observable_lift=projection.T, correlated=correlated)
    for actual, wanted in zip((f.p, f.v, f.R, f.bg, f.ba), expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(f.P, posterior, rtol=1e-12, atol=1e-14)
    assert np.linalg.eigvalsh(f.P).min() > 0


@pytest.mark.parametrize('damage', ['nan', 'wrong_shape', 'zero_rank', 'negative_noise', 'wrong_lift'])
def test_invalid_partial_measurements_leave_filter_unchanged(damage):
    f = InertialFilter(1_000_000_000, pose(), [0, 0, 0], [0, 0, G], [0, 0, 0], [0, 0, 0])
    projection = np.eye(6)[[2, 3, 4]]; lift = projection.T.copy(); covariance = np.eye(3)*.001
    if damage == 'nan':
        projection[0, 0] = np.nan
    elif damage == 'wrong_shape':
        projection = projection[:, :5]
    elif damage == 'zero_rank':
        projection, lift, covariance = np.zeros((0, 6)), np.zeros((6, 0)), np.zeros((0, 0))
    elif damage == 'negative_noise':
        covariance[0, 0] = -.001
    else:
        lift[0, 0] = 1.
        lift[2, 0] = 0.
    snapshot = deepcopy(f.__dict__)
    assert not f.correct(pose(), covariance, projection=projection, observable_lift=lift)
    for key in ('p', 'v', 'R', 'bg', 'ba', 'P'):
        np.testing.assert_array_equal(getattr(f, key), snapshot[key])


def test_support_rank_change_cannot_silently_move_the_prior_gauge():
    # Controlled point/normal observations represent a second surface entering
    # overlap after an iteration. The previously unmeasured gauge is not a pose.
    yy, xx = np.indices((12, 12))
    points = np.stack((xx.ravel()*.03, yy.ravel()*.03, np.ones(xx.size)*2.), -1)
    plane = np.tile([0., 0., 1.], (len(points), 1))
    room = np.eye(3)[np.arange(len(points)) % 3]
    class ChangingSupport(DepthRegistration):
        def __init__(self, reverse):
            super().__init__(K)
            self.normals = [room, plane] if reverse else [plane, room]
        def _correspondences(self, *args):
            normal = self.normals.pop(0) if len(self.normals) > 1 else self.normals[0]
            return np.concatenate((normal, np.cross(points, normal)), axis=1), np.full(len(points), .01)
    for reverse in (False, True):
        assert ChangingSupport(reverse)._fit(points, None, None, np.eye(4), 3) is None


def test_partial_repeated_wall_still_rejects_a_competing_observed_basin():
    y, x = np.indices((240, 320))
    ray_x = (x-K[0, 2])/K[0, 0]
    depth = np.ones(x.shape)
    for _ in range(150):
        depth = 1+.005*np.sin(2*np.pi*ray_x*depth/.05)
    depth = np.rint(depth*1000).astype(np.uint16)
    assert DepthRegistration(K).register(depth, depth, np.eye(4)) is None


def prepare_partial_fusion(asynchronous):
    from test_rgbd_fusion import warmed, K as calibration
    from test_rgbd_geometry_admission import Runner
    fusion, stamp = warmed()
    fusion.geometry = DepthRegistration(calibration)
    if asynchronous:
        depth = fusion.reference[1]
        result = fusion.geometry.register(depth, depth, np.eye(4))
        assert result is not None and result['rank'] == 3
        fusion.geometry_runner = Runner(result)
    return fusion, stamp


def partial_frame(fusion, start):
    from test_rgbd_fusion import images, source
    stamp = start+50_000_000
    for instant in range(start+5_000_000, stamp+1, 5_000_000):
        fusion.observe_imu(source(instant), instant)
    sample = fusion.observe_frame(stamp, *images(), None, stamp)
    if fusion.geometry_runner is not None:
        fusion.geometry_runner.ready = True
        fusion.poll_geometry(stamp+10_000_000)
        sample = fusion.output(stamp)
    return sample, stamp


@pytest.mark.parametrize('asynchronous', [False, True])
def test_partial_depth_never_promotes_tracking_or_refreshes_visual_reference(asynchronous):
    fusion, stamp = prepare_partial_fusion(asynchronous)
    anchor = deepcopy(fusion.reference); confident = fusion.last_confident
    for _ in range(4):
        sample, stamp = partial_frame(fusion, stamp)
        assert fusion.quality['geometry']['accepted'] is True
        assert fusion.quality['geometry']['rank'] == 3
        assert fusion.good == 0 and fusion.state == 'coasting'
        assert sample['pose'] is None and sample['vendor']['fusion']['mapping_allowed'] is False
        assert sample['vendor']['fusion']['prediction'] is not None
        assert np.isfinite(fusion.filter.P).all() and np.linalg.eigvalsh(fusion.filter.P).min() >= 0
    assert fusion.reference[0] == fusion.last_visual == anchor[0]
    assert fusion.last_confident == confident
    np.testing.assert_array_equal(fusion.reference[1], anchor[1])


@pytest.mark.parametrize('full_source', ['visual', 'depth'])
def test_partial_to_full_recovery_requires_three_full_pose_observations(full_source):
    from test_rgbd_fusion import advance, images, source, K as calibration
    fusion, stamp = prepare_partial_fusion(False)
    _, stamp = partial_frame(fusion, stamp)
    original_visual = fusion.last_visual
    reference = render_room(np.eye(4), (480, 640), calibration)
    if full_source == 'depth':
        fusion.reference = (fusion.reference[0], reference, fusion.reference[2])
    for index in range(3):
        if full_source == 'visual':
            sample, stamp = advance(fusion, stamp, .05, pose())
        else:
            start = stamp; stamp += 50_000_000
            for instant in range(start+5_000_000, stamp+1, 5_000_000):
                fusion.observe_imu(source(instant), instant)
            sample = fusion.observe_frame(stamp, images()[0], reference, None, stamp)
        assert (sample['pose'] is not None) is (index == 2)
        assert fusion.good == index+1
    if full_source == 'depth':
        assert fusion.last_visual == original_visual


@pytest.mark.parametrize('asynchronous', [False, True])
def test_partial_depth_cannot_hide_excessive_null_position_uncertainty(asynchronous):
    fusion, stamp = prepare_partial_fusion(asynchronous)
    fusion.filter.P[0, 0] = .251**2
    sample, _ = partial_frame(fusion, stamp)
    assert fusion.quality['geometry']['accepted'] is True
    assert fusion.state == 'lost' and sample['pose'] is None


def test_partial_result_for_superseded_capture_is_discarded():
    from test_rgbd_fusion import advance, images, source
    fusion, start = prepare_partial_fusion(True)
    stamp = start+50_000_000
    for instant in range(start+5_000_000, stamp+1, 5_000_000):
        fusion.observe_imu(source(instant), instant)
    fusion.observe_frame(stamp, *images(), None, stamp)
    assert fusion.geometry_job is not None
    advance(fusion, stamp, .05, pose())
    assert fusion.frames[-2]['geometry'] == 'stale_result_discarded'
    snapshot = fusion.filter.P.copy()
    fusion.geometry_runner.ready = True
    fusion.poll_geometry(stamp+60_000_000)
    np.testing.assert_array_equal(fusion.filter.P, snapshot)
    assert fusion.aid == 'visual'


@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_partial_results_obey_the_exact_existing_geometry_deadline(offset):
    from test_rgbd_fusion import images, source
    fusion, start = prepare_partial_fusion(True)
    stamp = start+50_000_000
    for instant in range(start+5_000_000, stamp+1, 5_000_000):
        fusion.observe_imu(source(instant), instant)
    fusion.observe_frame(stamp, *images(), None, stamp)
    deadline = fusion.geometry_job['deadline']
    fusion.geometry_runner.ready = True
    fusion.poll_geometry(deadline+offset)
    assert fusion.quality['geometry']['accepted'] is (offset <= 0)
    assert fusion.state == 'coasting' and fusion.good == 0
    assert fusion.output(stamp)['pose'] is None


@pytest.mark.parametrize('damage', ['rank', 'nan', 'zero_noise', 'unbounded_transform'])
def test_malformed_geometry_cannot_create_full_pose_tracking_or_mutate_filter(damage):
    from test_rgbd_fusion import images, source
    fusion, start = prepare_partial_fusion(True)
    result = fusion.geometry_runner.result
    if damage == 'rank':
        result['rank'] = 6
        result['covariance'] = np.eye(6)*.001
    elif damage == 'nan':
        result['projection'][0, 0] = np.nan
    elif damage == 'zero_noise':
        result['projected_covariance'][:] = 0.
    else:
        result['transform'][0, 3] = .31
    stamp = start+50_000_000
    for instant in range(start+5_000_000, stamp+1, 5_000_000):
        fusion.observe_imu(source(instant), instant)
    fusion.observe_frame(stamp, *images(), None, stamp)
    before = deepcopy(fusion.filter.__dict__)
    fusion.geometry_runner.ready = True
    fusion.poll_geometry(stamp+10_000_000)
    assert fusion.state == 'coasting' and fusion.quality['geometry']['accepted'] is False
    for key in ('p', 'v', 'R', 'bg', 'ba', 'P'):
        np.testing.assert_array_equal(getattr(fusion.filter, key), before[key])


def test_partial_observations_never_extend_the_one_second_visual_budget():
    from test_rgbd_fusion import source
    fusion, stamp = prepare_partial_fusion(True)
    original = stamp
    for _ in range(4):
        _, stamp = partial_frame(fusion, stamp)
    deadline = original+1_000_000_000
    for instant in range(stamp+5_000_000, deadline+5_000_001, 5_000_000):
        fusion.observe_imu(source(instant), instant)
    assert fusion.status(deadline+5_000_000) == 'lost'
    assert fusion.last_visual == original and fusion.reference[0] == original


@pytest.mark.parametrize('associated', [False, True])
def test_native_visual_updates_keep_segment_association_correlation(associated):
    from plugins.rgbd_fusion import ShortTermFusion
    from test_rgbd_fusion import K as calibration, images, source
    fusion = ShortTermFusion('run', 'optical', 'capture-clock', calibration, inertial_association=associated)
    color, depth = images()
    for index in range(141):
        stamp = 1_000_000_000+index*5_000_000
        fusion.observe_imu(source(stamp), stamp)
        if index % 20 == 0:
            fusion.observe_frame(stamp, color, depth, pose(), stamp)
    reference = deepcopy(fusion.filter)
    end = stamp+50_000_000
    for instant in range(stamp+5_000_000, end+1, 5_000_000):
        fusion.observe_imu(source(instant), instant)
        reference.propagate(.005, [0, 0, 0], [0, 0, G])
    from plugins.rgbd_fusion import observation_covariance
    expected_covariance = observation_covariance(pose())*fusion.quality['weight']
    assert reference.correct(pose(), expected_covariance, correlated=associated)
    fusion.observe_frame(end, color, depth, pose(), end)
    np.testing.assert_allclose(fusion.filter.P, reference.P, atol=1e-12)
