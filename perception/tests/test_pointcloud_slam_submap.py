"""Submaps must use each capture pose and a cloud-to-pose extrinsic."""
import numpy as np
import pytest

from plugins.pointcloud_slam_odom import PoseSample
from tools.pointcloud_slam_submap import compose_submap


def pose(x, segment='s'):
    return PoseSample(1000, 'odom', 'body', segment, (x, 0., 0.), (0., 0., 0., 1.), None)


def test_submap_uses_each_cloud_pose_in_anchor_cloud_frame():
    cloud0 = np.array([[1., 0., 0.]])
    cloud1 = np.array([[0., 0., 0.]])
    result = compose_submap([cloud0, cloud1], [pose(0), pose(1)], np.eye(4), anchor=0)
    np.testing.assert_allclose(result, [[1., 0., 0.], [1., 0., 0.]])


def test_refuses_to_join_different_segments_or_invalid_extrinsics():
    clouds = [np.array([[0., 0., 1.]])] * 2
    with pytest.raises(ValueError):
        compose_submap(clouds, [pose(0), pose(1, 'new')], np.eye(4), anchor=0)
    bad = np.eye(4)
    bad[0, 0] = 0
    with pytest.raises(ValueError):
        compose_submap(clouds, [pose(0), pose(1)], bad, anchor=0)
