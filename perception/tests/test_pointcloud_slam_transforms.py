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


def test_transform_returns_independent_numeric_matrix_and_rejects_bad_format():
    from plugins import pointcloud_slam_transforms as transforms
    original = np.eye(4)
    copied = transforms.rigid(original)
    copied[0, 3] = 4
    assert original[0, 3] == 0
    for invalid in (np.eye(3), np.full((4,4), np.nan), [['x']*4]*4):
        with pytest.raises(ValueError):
            transforms.rigid(invalid)


def test_transform_does_not_certify_physical_rigidity():
    from plugins.pointcloud_slam_transforms import rigid
    for matrix in (np.zeros((4,4)), np.diag([-1.,2.,3.,4.])):
        np.testing.assert_array_equal(rigid(matrix), matrix)
