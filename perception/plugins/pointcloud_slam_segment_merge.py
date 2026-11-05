"""Synthetic cross-segment graph hypotheses; no real relocalization claim."""
import math
import numpy as np

from .pointcloud_slam_experimental_backend import _sigmas
from .pointcloud_slam_geometry import classify_registration
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_map import map_from_odom, rebuild_segment
from .pointcloud_slam_registration import registration_diagnostics
from .pointcloud_slam_stream import _rigid


def experimental_segment_merge(target, target_graph, source, source_graph, *,
        target_index, source_index, initial, thresholds, odom_sigmas, loop_sigmas,
        huber_k, voxel_m, max_distance_m, max_total_frames, max_total_points):
    """Optimize a bounded provisional two-segment connection after geometry gates.

    ``initial`` and ICP's forward transform map source child points into target
    child coordinates. No place identity, clock synchronization or real loop is
    established by this experiment. Inputs and original odometry remain intact.
    """
    import gtsam
    segments = (target, source); graphs = (target_graph, source_graph)
    if (any(not isinstance(s, Segment) or not s.frames for s in segments) or
            target.segment_id == source.segment_id or
            any(not isinstance(g, dict) or g.get('segment_id') != s.segment_id or
                len(g.get('poses', ())) != len(s.frames) or
                not isinstance(g.get('edges'), list) or len(g['edges']) != len(s.frames)-1
                for s, g in zip(segments, graphs)) or
            any(type(v) is not int or v <= 0 for v in (max_total_frames, max_total_points)) or
            sum(len(s.frames) for s in segments) > max_total_frames or
            sum(len(f.points) for s in segments for f in s.frames) > max_total_points or
            type(target_index) is not int or not 0 <= target_index < len(target.frames) or
            type(source_index) is not int or not 0 <= source_index < len(source.frames)):
        raise ValueError('distinct bounded segments and matching graphs required')
    if isinstance(huber_k, bool) or not isinstance(huber_k, (int, float)) or not math.isfinite(huber_k) or huber_k <= 0:
        raise ValueError('positive finite Huber parameter required')
    odom_noise_values, loop_noise_values = _sigmas(odom_sigmas), _sigmas(loop_sigmas)
    classify_registration({}, thresholds)
    for graph in graphs:
        for i, edge in enumerate(graph['edges'], 1):
            if not isinstance(edge, dict) or edge.get('from') != i-1 or edge.get('to') != i:
                raise ValueError('only consecutive odometry edges are allowed')
            _rigid(edge.get('transform'))
        for pose in graph['poses']:
            _rigid(pose)
    diagnostic = registration_diagnostics(source.frames[source_index].points,
        target.frames[target_index].points, initial, voxel_m=voxel_m, max_distance_m=max_distance_m)
    gate = classify_registration(diagnostic, thresholds)
    result = {'hypothesis_created': False, 'loop_accepted': False,
              'failed': gate['failed'], 'diagnostic': diagnostic}
    if not gate['passes']:
        return result
    forward = _rigid(diagnostic['forward']['transform'])
    target_poses, source_poses = target_graph['poses'], source_graph['poses']
    align = _rigid(target_poses[target_index]) @ forward @ np.linalg.inv(_rigid(source_poses[source_index]))
    noise = gtsam.noiseModel.Diagonal.Sigmas(odom_noise_values)
    loop_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
                gtsam.noiseModel.Diagonal.Sigmas(loop_noise_values))
    factors, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    offset = len(target_poses)
    for i, pose in enumerate(target_poses): values.insert(i, gtsam.Pose3(_rigid(pose)))
    for i, pose in enumerate(source_poses): values.insert(offset+i, gtsam.Pose3(align @ _rigid(pose)))
    factors.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(_rigid(target_poses[0])), noise))
    for base, graph in ((0, target_graph), (offset, source_graph)):
        for edge in graph['edges']:
            factors.add(gtsam.BetweenFactorPose3(base+edge['from'], base+edge['to'],
                        gtsam.Pose3(_rigid(edge['transform'])), noise))
    factors.add(gtsam.BetweenFactorPose3(target_index, offset+source_index,
                gtsam.Pose3(forward), loop_noise))
    optimized = gtsam.LevenbergMarquardtOptimizer(factors, values).optimize()
    a = [optimized.atPose3(i).matrix() for i in range(offset)]
    b = [optimized.atPose3(offset+i).matrix() for i in range(len(source_poses))]
    return {**result, 'hypothesis_created': True, 'target_poses': a, 'source_poses': b,
            'map_points': np.concatenate((rebuild_segment(target, a), rebuild_segment(source, b))),
            'map_from_source_odom': map_from_odom(b[-1], source_poses[-1])}
