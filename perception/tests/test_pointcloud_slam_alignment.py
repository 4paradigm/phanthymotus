"""Time alignment must never bridge missing tracking or a reset."""
import math
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


def test_exact_sample_does_not_require_neighboring_pose():
    assert align_pose([pose(1000)], 1_000_000_000, max_gap_ms=1).stamp_ms == 1000


@pytest.mark.parametrize('items, ns, gap', [
    ([pose(1000), pose(1200, segment='new')], 1_100_000_000, 250),
    ([pose(1000), pose(1200, frame='new')], 1_100_000_000, 250),
    ([pose(1000), None, pose(1200)], 1_100_000_000, 250),
    ([pose(1000), pose(1200)], 1_100_000_000, 199),
    ([pose(1000), pose(1200)], 900_000_000, 250),
    ([pose(1200), pose(1000)], 1_100_000_000, 250),
    ([pose(1000), pose(1000, x=1.)], 1_000_000_000, 250),
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
