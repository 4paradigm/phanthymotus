"""Uncalibrated geometry-gated graph; never accepts real loop edges."""
import math
import numpy as np

from .pointcloud_slam_geometry import classify_registration
from .pointcloud_slam_stream import _rigid


def _sigmas(values):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('six positive finite sigmas required') from error
    if values.shape != (6,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('six positive finite sigmas required')
    return values


def optimize_candidates(graph, candidates, *, thresholds, odom_sigmas, loop_sigmas, huber_k):
    """Run an offline single-segment experiment with uncalibrated candidate gates.

    The ICP forward transform maps source points into target coordinates.
    GTSAM's between factor (source,target) instead takes T_source_target, so
    the forward registration result is inverted before it enters the graph.
    This function does not establish place identity or accept real loop edges.
    """
    import gtsam
    odom_sigmas, loop_sigmas = _sigmas(odom_sigmas), _sigmas(loop_sigmas)
    if isinstance(huber_k, bool) or not isinstance(huber_k, (int, float)) or not math.isfinite(huber_k) or huber_k <= 0:
        raise ValueError('positive finite Huber parameter required')
    classify_registration({}, thresholds)
    poses, edges = graph.get('poses'), graph.get('edges')
    if not poses or not isinstance(edges, list) or len(edges) != len(poses) - 1:
        raise ValueError('single connected segment required')
    factors, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
        gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas))
    for i, pose in enumerate(poses):
        values.insert(i, gtsam.Pose3(_rigid(pose)))
    factors.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(_rigid(poses[0])), odom_noise))
    for i, edge in enumerate(edges, 1):
        if edge.get('from') != i - 1 or edge.get('to') != i:
            raise ValueError('invalid odometry edge')
        factors.add(gtsam.BetweenFactorPose3(i - 1, i,
                    gtsam.Pose3(_rigid(edge['transform'])), odom_noise))
    passed, rejected, seen = [], [], set()
    for candidate in candidates:
        source, target = candidate.get('source'), candidate.get('target')
        if (type(source) is not int or type(target) is not int or
                not 0 <= source < len(poses) or not 0 <= target < len(poses) or source == target):
            raise ValueError('invalid candidate endpoints')
        pair = tuple(sorted((source, target)))
        if pair in seen:
            raise ValueError('duplicate candidate pair')
        seen.add(pair)
        diagnostic = candidate.get('diagnostic')
        gate = classify_registration(diagnostic, thresholds)
        entry = {'source': source, 'target': target, 'failed': gate['failed']}
        if not gate['passes']:
            rejected.append(entry)
            continue
        forward = _rigid(diagnostic['forward']['transform'])
        factors.add(gtsam.BetweenFactorPose3(source, target,
                    gtsam.Pose3(np.linalg.inv(forward)), loop_noise))
        if 'confirmation_stamps_ns' in candidate:
            entry['confirmation_stamps_ns'] = list(candidate['confirmation_stamps_ns'])
            entry['target_from_source'] = forward.tolist()
        passed.append(entry)
    optimized = gtsam.LevenbergMarquardtOptimizer(factors, values).optimize()
    return {'poses': [optimized.atPose3(i).matrix() for i in range(len(poses))],
            'passed_candidates': passed, 'rejected_candidates': rejected,
            'loop_accepted': False}
