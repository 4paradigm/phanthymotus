"""Time alignment must never bridge missing tracking or a reset."""
import math
from dataclasses import replace

import numpy as np
import pytest

from plugins.pointcloud_slam_alignment import align_pose
from plugins.pointcloud_slam_odom import PoseSample


def pose(stamp, segment='s', x=0., quaternion=(0., 0., 0., 1.), frame='odom'):
    return PoseSample(stamp, frame, 'camera', segment, (x, 0., 0.), quaternion, None)


def test_interpolates_translation_and_rotation_at_cloud_measurement_time():
    first = pose(1000)
    second = pose(1200, x=2., quaternion=(0., 0., 1., 0.))
    result = align_pose([first, second], 1_100_000_000, max_gap_ms=250)
    assert result.position == pytest.approx((1., 0., 0.))
    assert result.quaternion == pytest.approx((0., 0., math.sqrt(.5), math.sqrt(.5)))


def test_midpoint_preserves_shared_endpoint_covariance():
    covariance = (.04,) + (0.,) * 35
    first = replace(pose(1000), covariance=covariance)
    second = replace(pose(1200, x=2.), covariance=covariance)

    result = align_pose([first, second], 1_100_000_000, max_gap_ms=250)

    assert result.covariance == covariance


def test_interpolates_full_covariance_with_translation_rotation_cross_terms():
    first_covariance = np.diag([.04, .05, .06, .07, .08, .09])
    first_covariance[0, 4] = first_covariance[4, 0] = .02
    second_covariance = np.diag([.12, .17, .22, .27, .32, .37])
    second_covariance[0, 4] = second_covariance[4, 0] = -.02
    first = replace(pose(1000), covariance=tuple(first_covariance.flat))
    second = replace(pose(1200, quaternion=(0., 0., 1., 0.)),
                     covariance=tuple(second_covariance.flat))

    result = align_pose([first, second], 1_050_000_000, max_gap_ms=250)

    assert result.covariance is not None
    covariance = np.reshape(result.covariance, (6, 6))
    expected = np.diag([.06, .08, .10, .12, .14, .16])
    expected[0, 4] = expected[4, 0] = .01
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_array_equal(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance)[0] >= 0.


@pytest.mark.parametrize('known_endpoint', [0, 1, None])
def test_interpolation_preserves_the_available_covariance(known_endpoint):
    covariance = (.04,) + (0.,) * 35
    samples = [pose(1000), pose(1200)]
    if known_endpoint is not None:
        samples[known_endpoint] = replace(samples[known_endpoint], covariance=covariance)

    result = align_pose(samples, 1_050_000_000, max_gap_ms=250)

    assert result.covariance == (None if known_endpoint is None else covariance)


@pytest.mark.parametrize('endpoint', [0, 1])
def test_exact_timestamp_preserves_endpoint_sample_and_covariance(endpoint):
    samples = [replace(pose(1000), covariance=(.04,) + (0.,) * 35),
               replace(pose(1200), covariance=(.16,) + (0.,) * 35)]
    exact = samples[endpoint]

    assert align_pose(samples, exact.stamp_ns, max_gap_ms=250) is exact


def test_unmarked_segment_and_frame_labels_do_not_block_interpolation():
    before = pose(1000, x=4)
    after = pose(1200, segment='another-label', frame='another-frame')
    result = align_pose([before, after], 1_100_000_000, max_gap_ms=250)
    assert result.position == pytest.approx((2., 0., 0.))


def test_same_visual_origin_keeps_numeric_pose_when_the_run_token_changes():
    origin = ('s', 'odom')
    before = replace(pose(1000, x=4), visual_origin=origin, external_continuity=b'first')
    after = replace(pose(1200), visual_origin=origin, external_continuity=b'second')
    result = align_pose([before, after], 1_100_000_000, max_gap_ms=250)
    assert result.position == pytest.approx((2., 0., 0.))
    assert result.visual_origin == origin and result.external_continuity is None


def test_exact_sample_does_not_require_neighboring_pose():
    assert align_pose([pose(1000)], 1_000_000_000, max_gap_ms=1).stamp_ms == 1000


@pytest.mark.parametrize('items, ns, gap', [
    ([pose(1000), None, pose(1200)], 1_100_000_000, 250),
    ([pose(1000), pose(1200)], 900_000_000, 250),
])
def test_rejects_discontinuities_stale_extrapolation_and_conflicting_samples(items, ns, gap):
    with pytest.raises(ValueError):
        align_pose(items, ns, max_gap_ms=gap)


def test_duplicate_identical_samples_are_safe():
    p = pose(1000)
    assert align_pose([p, p, pose(1200, x=2)], 1_100_000_000, max_gap_ms=250).position[0] == pytest.approx(1.)


def test_equivalent_opposite_quaternions_do_not_spin_or_divide_by_zero():
    a = pose(1000, quaternion=(0., 0., 0., 1.))
    b = pose(1200, quaternion=(0., 0., 0., -1.))
    result = align_pose([a, b], 1_100_000_000, max_gap_ms=250)
    assert result.quaternion == pytest.approx((0., 0., 0., 1.))
