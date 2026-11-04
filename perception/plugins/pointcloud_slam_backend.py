"""GTSAM odometry-only backend; no loop acceptance or map correction yet."""
import numpy as np

from .pointcloud_slam_stream import _rigid


def optimize_odometry_graph(graph, *, sigmas):
    """Optimize a single connected segment using explicitly supplied noise.

    Sigmas are in GTSAM's rotation-then-translation tangent order. They are
    experimental until established for the sensor. This function accepts no
    loop edges and must not be presented as a validated SLAM map correction.
    """
    try:
        sigmas = np.asarray(sigmas, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('six positive finite noise sigmas required') from error
    if sigmas.shape != (6,) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError('six positive finite noise sigmas required')
    poses, edges = graph.get('poses'), graph.get('edges')
    if not poses or not isinstance(edges, list) or len(edges) != len(poses) - 1:
        raise ValueError('one connected odometry chain required')
    import gtsam
    noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    factors, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for i, pose in enumerate(poses):
        values.insert(i, gtsam.Pose3(_rigid(pose)))
    factors.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(_rigid(poses[0])), noise))
    for i, edge in enumerate(edges, 1):
        if edge.get('from') != i - 1 or edge.get('to') != i:
            raise ValueError('only consecutive odometry edges are allowed')
        factors.add(gtsam.BetweenFactorPose3(i - 1, i,
                    gtsam.Pose3(_rigid(edge['transform'])), noise))
    result = gtsam.LevenbergMarquardtOptimizer(factors, values).optimize()
    return [result.atPose3(i).matrix() for i in range(len(poses))]
