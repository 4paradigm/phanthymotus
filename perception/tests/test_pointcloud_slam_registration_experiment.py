"""Pairwise registration diagnostics are not, by themselves, accepted loops."""
import numpy as np
import pytest

pytest.importorskip('open3d')
from tools.pointcloud_slam_registration_experiment import registration_diagnostics


def test_reciprocal_icp_recovers_known_translation_without_accepting_loop():
    points = np.random.default_rng(4).uniform(-1, 1, size=(500, 3))
    target = points + [0.03, -0.02, 0.01]
    result = registration_diagnostics(points, target, np.eye(4), voxel_m=.04, max_distance_m=.15)
    assert result['forward']['fitness'] > .9
    assert result['reverse']['fitness'] > .9
    np.testing.assert_allclose(np.array(result['forward']['transform'])[:3, 3], [.03, -.02, .01], atol=.01)
    assert result['cycle_translation_m'] < .01
    assert result['loop_accepted'] is False


def test_no_correspondences_is_explicitly_invalid_not_zero_error_success():
    points = np.random.default_rng(7).uniform(-.5, .5, size=(100, 3))
    result = registration_diagnostics(points, points + [10., 0., 0.], np.eye(4), voxel_m=.05, max_distance_m=.1)
    assert result['forward']['correspondences'] == 0
    assert result['forward']['rmse_m'] is None
    assert result['loop_accepted'] is False
