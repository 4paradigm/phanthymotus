"""Depth registration truth comes from independent ray/plane rendering."""

import numpy as np
import pytest

from plugins.rgbd_geometry import DepthRegistration, observation_quality


K = np.array([[230., 0., 159.5], [0., 230., 119.5], [0., 0., 1.]])


def render_room(transform, shape=(240, 320), calibration=K):
    """Ray intersections with a back wall, side walls, ceiling and floor."""
    y, x = np.indices(shape)
    rays = np.stack(((x-calibration[0, 2])/calibration[0, 0],
                     (y-calibration[1, 2])/calibration[1, 1], np.ones(shape)), -1)
    directions = rays @ transform[:3, :3].T
    origin = transform[:3, 3]
    intersections = []
    for axis, coordinate in ((2, 3.), (0, -.9), (0, .9), (1, -.65), (1, .65)):
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = (coordinate-origin[axis])/directions[..., axis]
        intersections.append(np.where(distance > 0, distance, np.inf))
    return np.rint(np.minimum.reduce(intersections)*1000).astype(np.uint16)


def known_motion():
    # Deliberately independent from production Lie algebra helpers.
    x, y, z = np.deg2rad([-.8, 2., 1.])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = [.025, -.015, .012]
    return transform


def test_recovers_reference_from_current_transform_and_conservative_covariance():
    truth = known_motion()
    result = DepthRegistration(K).register(render_room(truth), render_room(np.eye(4)), np.eye(4))
    assert result is not None
    np.testing.assert_allclose(result['transform'][:3, 3], truth[:3, 3], atol=.004)
    relative = result['transform'][:3, :3] @ truth[:3, :3].T
    angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.))
    assert angle < np.deg2rad(.3)
    assert result['inliers'] >= 100
    assert result['rmse'] < .02
    covariance = result['covariance']
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.all(np.diag(covariance)[:3] >= .01**2)
    assert np.all(np.diag(covariance)[3:] >= np.deg2rad(1.)**2)


def test_accepts_geometry_with_sparse_outliers_and_holes():
    truth = known_motion()
    current = render_room(truth)
    rng = np.random.default_rng(61)
    current[rng.random(current.shape) < .1] = 0
    current[rng.random(current.shape) < .1] = 6500
    result = DepthRegistration(K).register(current, render_room(np.eye(4)), np.eye(4))
    assert result is not None
    np.testing.assert_allclose(result['transform'][:3, 3], truth[:3, 3], atol=.005)


def test_full_resolution_registration_uses_calibration_and_a_nonidentity_prior():
    calibration = np.array([[460., 0., 319.5], [0., 460., 239.5], [0., 0., 1.]])
    truth = known_motion()
    truth[:3, 3] = [.12, -.04, .03]
    prior = truth.copy()
    prior[:3, 3] += [.015, -.012, .009]
    result = DepthRegistration(calibration).register(
        render_room(truth, (480, 640), calibration),
        render_room(np.eye(4), (480, 640), calibration), prior)
    assert result is not None
    np.testing.assert_allclose(result['transform'], truth, atol=.003)


def test_rejects_single_plane_despite_perfect_residual():
    plane = np.full((240, 320), 2000, dtype=np.uint16)
    assert DepthRegistration(K).register(plane, plane, np.eye(4)) is None


def render_repeated_surface(period, angle=0.):
    """Intersect rays with a sinusoidal height field, independently of ICP.

    Fixed-point ray intersection converges for this amplitude and field of
    view. The two wave directions may be rotated relative to pixel axes.
    """
    y, x = np.indices((240, 320))
    ray_x = (x-K[0, 2])/K[0, 0]
    ray_y = (y-K[1, 2])/K[1, 1]
    depth = np.ones(ray_x.shape)
    for _ in range(150):
        a = (np.cos(angle)*ray_x+np.sin(angle)*ray_y)*depth
        b = (-np.sin(angle)*ray_x+np.cos(angle)*ray_y)*depth
        depth = 1+period*.1*(np.sin(2*np.pi*a/period)+np.cos(2*np.pi*b/period))
    return np.rint(depth*1000).astype(np.uint16)


@pytest.mark.parametrize('period, angle', [(.03, 0.), (.05, 0.), (.085, .4), (.13, 0.)])
@pytest.mark.parametrize('alias_prior', [False, True])
def test_rejects_repetitive_full_rank_geometry_even_from_correct_prior(period, angle, alias_prior):
    depth = render_repeated_surface(period, angle)
    prior = np.eye(4)
    if alias_prior:
        prior[:2, 3] = period*.98*np.array([np.cos(angle), np.sin(angle)])
    assert DepthRegistration(K).register(depth, depth, prior) is None


@pytest.mark.parametrize('case', ['empty', 'almost_empty', 'outliers', 'no_overlap', 'float_depth'])
def test_rejects_unusable_depth(case):
    reference = render_room(np.eye(4))
    current = reference.copy()
    if case == 'empty':
        current[:] = 0
    elif case == 'almost_empty':
        current[:230] = 0
    elif case == 'outliers':
        current = np.random.default_rng(5).integers(300, 7500, current.shape, dtype=np.uint16)
    elif case == 'no_overlap':
        current[:] = 7000
    else:
        current = current.astype(float)
    assert DepthRegistration(K).register(current, reference, np.eye(4)) is None


@pytest.mark.parametrize('case', ['nan', 'large_translation', 'large_rotation', 'reflection', 'bottom_row'])
def test_rejects_invalid_or_unbounded_prior(case):
    prior = np.eye(4)
    if case == 'nan':
        prior[0, 0] = np.nan
    elif case == 'large_translation':
        prior[0, 3] = .31
    elif case == 'large_rotation':
        prior[:3, :3] = np.diag([-1., -1., 1.])
    elif case == 'reflection':
        prior[0, 0] = -1
    else:
        prior[3, 0] = 1
    depth = render_room(np.eye(4))
    assert DepthRegistration(K).register(depth, depth, prior) is None


def test_blank_and_smooth_images_fail_visual_quality_without_losing_depth_information():
    depth = render_room(np.eye(4))
    for color in (np.zeros((240, 320, 3), np.uint8),
                  np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))):
        result = observation_quality(color, depth)
        assert result['visual_usable'] is False
        assert result['weight'] >= 1
        assert result['depth_fraction'] == 1
        assert result['reason']


def test_texture_is_usable_but_weak_images_and_missing_depth_increase_weight():
    rng = np.random.default_rng(4)
    textured = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    weak = (128 + (textured.astype(float)-128)*.2).astype(np.uint8)
    depth = render_room(np.eye(4))
    strong = observation_quality(textured, depth)
    weaker = observation_quality(weak, depth)
    assert strong['visual_usable'] is True
    assert weaker['visual_usable'] is True
    assert weaker['weight'] > strong['weight'] >= 1
    missing = observation_quality(textured, np.zeros_like(depth))
    assert missing['visual_usable'] is False
    assert missing['depth_fraction'] == 0


@pytest.mark.parametrize('depth', [np.full((10,), 1000, np.uint16),
                                  np.full((40, 50), 1000, np.uint16),
                                  np.full((240, 320), 1000., float)])
def test_visual_quality_rejects_invalid_depth_shapes_and_units(depth):
    image = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    result = observation_quality(image, depth)
    assert result['visual_usable'] is False
    assert result['depth_fraction'] == 0


@pytest.mark.parametrize('calibration', [np.zeros((3, 3)), np.eye(4),
                                        [[np.nan, 0, 1], [0, 200, 1], [0, 0, 1]],
                                        [[200, 1, 1], [0, 200, 1], [0, 0, 1]]])
def test_rejects_nonphysical_or_unsupported_intrinsics(calibration):
    with pytest.raises(ValueError):
        DepthRegistration(calibration)
