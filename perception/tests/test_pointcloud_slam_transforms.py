import numpy as np
import pytest

from plugins.pointcloud_slam_odom import PoseSample


def test_pose_matrix_rotates_xy_and_preserves_metre_translation():
    from plugins import pointcloud_slam_transforms as transforms
    pose = PoseSample(1000, 'odom', 'base', 'a', (2., -3., 4.),
                      (0., 0., 2 ** -.5, 2 ** -.5), None, 'ros')
    matrix = transforms.pose_matrix(pose)
    np.testing.assert_allclose(matrix, [[0, -1, 0, 2], [1, 0, 0, -3],
                                       [0, 0, 1, 4], [0, 0, 0, 1]], atol=1e-12)


def test_rigid_returns_independent_transform_and_rejects_nonrigid_input():
    from plugins import pointcloud_slam_transforms as transforms
    original = np.eye(4)
    copied = transforms.rigid(original)
    copied[0, 3] = 4
    assert original[0, 3] == 0
    for invalid in (np.zeros((4, 4)), np.eye(3), np.diag([-1, 1, 1, 1])):
        with pytest.raises(ValueError):
            transforms.rigid(invalid)
