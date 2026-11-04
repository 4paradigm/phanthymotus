"""Synthetic graph experiments must visibly correct history, not only the pose arrow."""
import numpy as np
import pytest

gtsam = pytest.importorskip('gtsam')
from tools.pointcloud_slam_graph_experiment import optimize_graph, rebuild_points


def transform(x, y=0.):
    result = np.eye(4)
    result[:3, 3] = [x, y, 0.]
    return result


def test_verified_loop_moves_historical_poses_and_rebuilds_local_clouds():
    initial = [transform(0), transform(1.2), transform(2.2), transform(.5, .2)]
    edges = [(i, i+1, np.linalg.inv(initial[i]) @ initial[i+1]) for i in range(3)]
    corrected = optimize_graph(initial, edges, [(0, 3, transform(.05), True)])
    assert np.linalg.norm(corrected[3][:3, 3]) < np.linalg.norm(initial[3][:3, 3])
    assert np.linalg.norm(corrected[1][:3, 3] - initial[1][:3, 3]) > .01
    local = [np.array([[0., 0., 0.]]) for _ in initial]
    assert np.linalg.norm(rebuild_points(local, corrected)[1] - rebuild_points(local, initial)[1]) > .01


def test_unverified_loop_cannot_enter_the_pose_graph():
    initial = [transform(0), transform(1)]
    with pytest.raises(ValueError, match='verified'):
        optimize_graph(initial, [(0, 1, transform(1))], [(0, 1, transform(0), False)])
