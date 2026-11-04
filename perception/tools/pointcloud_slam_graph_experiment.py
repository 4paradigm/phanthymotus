"""Synthetic pose-graph experiment, not a runtime SLAM backend.

The example weights below are for synthetic tests only. The caller must supply
independently verified loop edges. A boolean flag is not geometric validation.
"""
import numpy as np


def _pose(matrix):
    import gtsam
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError('pose must be a finite 4x4 transform')
    r = m[:3, :3]
    if (not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(r), 1., atol=1e-6)):
        raise ValueError('invalid rigid transform')
    return gtsam.Pose3(m)


def optimize_graph(initial_poses, odometry_edges, loop_edges):
    """Optimize one synthetic connected segment with explicitly accepted loops."""
    import gtsam
    if not initial_poses:
        raise ValueError('graph needs an anchor')
    values, graph = gtsam.Values(), gtsam.NonlinearFactorGraph()
    for i, matrix in enumerate(initial_poses):
        values.insert(i, _pose(matrix))
    # These fixed weights demonstrate graph behavior; they are not a covariance
    # model and must not be used to accept or weight real robot measurements.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.001] * 6))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.05] * 6))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.02] * 6))
    graph.add(gtsam.PriorFactorPose3(0, _pose(initial_poses[0]), prior_noise))
    for a, b, transform in odometry_edges:
        if not (0 <= a < len(initial_poses) and 0 <= b < len(initial_poses) and a != b):
            raise ValueError('invalid odometry edge')
        graph.add(gtsam.BetweenFactorPose3(a, b, _pose(transform), odom_noise))
    for a, b, transform, verified in loop_edges:
        if verified is not True:
            raise ValueError('loop must already be independently verified')
        if not (0 <= a < len(initial_poses) and 0 <= b < len(initial_poses) and a != b):
            raise ValueError('invalid loop edge')
        graph.add(gtsam.BetweenFactorPose3(a, b, _pose(transform), loop_noise))
    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    return [result.atPose3(i).matrix() for i in range(len(initial_poses))]


def rebuild_points(local_clouds, poses):
    """Rebuild a synthetic map from original local-frame points and poses."""
    if len(local_clouds) != len(poses):
        raise ValueError('each keyframe needs a pose')
    transformed = []
    for cloud, matrix in zip(local_clouds, poses):
        points = np.asarray(cloud, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('cloud must be finite Nx3')
        pose = _pose(matrix).matrix()
        transformed.append(points @ pose[:3, :3].T + pose[:3, 3])
    return np.concatenate(transformed) if transformed else np.empty((0, 3))
