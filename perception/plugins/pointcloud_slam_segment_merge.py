"""Synthetic cross-segment graph hypotheses; no real relocalization claim."""
import math
from copy import deepcopy
import numpy as np

from .pointcloud_slam_experimental_backend import _sigmas
from .pointcloud_slam_geometry import classify_registration
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_map import map_from_odom, rebuild_segment
from .pointcloud_slam_registration import registration_diagnostics
from .pointcloud_slam_stream import _rigid
from .pointcloud_slam_observation import legacy_scoring, observation_for_frame, resolve_scoring


def _merge_strategy(thresholds, *, observation=None, scoring=None, legacy_profile=None,
                    target_observation=None):
    """Validate the same explicit score before retrieval and before registration."""
    target_observation = target_observation if target_observation is not None else observation
    explicit = observation is not None or scoring is not None or target_observation is not None
    if legacy_profile is not None:
        if scoring is not None or legacy_profile not in ('visible_optical_depth', 'child_x_depth', 'inlier_rmse'):
            raise ValueError('select one legacy merge profile or a new score contract')
        if legacy_profile == 'visible_optical_depth' and any(
                not isinstance(o, dict) or o.get('schema') != 'motus.slam_observation/1' or o.get('kind') != 'pinhole'
                for o in (observation, target_observation)):
            raise ValueError('legacy visible merge requires source and target pinhole observations')
        spec = legacy_scoring(visible=legacy_profile == 'visible_optical_depth',
                              rmse=legacy_profile == 'inlier_rmse')
        expected = 'max_rmse_m' if legacy_profile == 'inlier_rmse' else 'max_forward_depth_score_m'
        if expected not in thresholds or thresholds.get('score_contract', spec) != spec:
            raise ValueError('legacy merge threshold contract differs from its score')
        compatibility = ['explicit legacy merge profile: ' + spec['metric_id'] + '; original semantics retained']
    elif explicit:
        if observation is None:
            raise ValueError('explicit source observation required for new merge scoring')
        spec = resolve_scoring(scoring, observation)
        resolve_scoring(scoring, target_observation)
        if ('max_range_stratified_rmse_m' not in thresholds or thresholds.get('score_contract') != spec):
            raise ValueError('new merge score requires its matching threshold contract')
        compatibility = []
    else:
        if 'max_range_stratified_rmse_m' in thresholds:
            raise ValueError('new merge threshold requires observation and scoring')
        spec = legacy_scoring(rmse='max_forward_depth_score_m' not in thresholds)
        if thresholds.get('score_contract', spec) != spec:
            raise ValueError('legacy merge score contract requires an explicit matching profile')
        compatibility = ['legacy merge score: ' + spec['metric_id'] + '; original semantics retained']
    classify_registration({}, thresholds)
    return deepcopy(dict(resolved_observation=observation, resolved_target_observation=target_observation,
                         resolved_scoring=spec, compatibility=compatibility))


def experimental_segment_merge(target, target_graph, source, source_graph, *,
        target_index, source_index, initial, thresholds, odom_sigmas, loop_sigmas,
        huber_k, voxel_m, max_distance_m, max_total_frames, max_total_points,
        observation=None, scoring=None, legacy_profile=None, target_observation=None):
    """Optimize a bounded provisional two-segment connection after geometry gates.

    ``initial`` and ICP's forward transform map source child points into target
    child coordinates. No place identity, clock synchronization or real loop is
    established by this experiment. Inputs and original odometry remain intact.
    """
    import gtsam
    strategy = _merge_strategy(thresholds, observation=observation, scoring=scoring,
                               legacy_profile=legacy_profile, target_observation=target_observation)
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
    for graph in graphs:
        for i, edge in enumerate(graph['edges'], 1):
            if not isinstance(edge, dict) or edge.get('from') != i-1 or edge.get('to') != i:
                raise ValueError('only consecutive odometry edges are allowed')
            _rigid(edge.get('transform'))
        for pose in graph['poses']:
            _rigid(pose)
    source_frame, target_frame = source.frames[source_index], target.frames[target_index]
    observations = {}
    if scoring is not None or legacy_profile == 'visible_optical_depth':
        observations = dict(source=observation_for_frame(observation, source_frame),
            target=observation_for_frame(strategy['resolved_target_observation'], target_frame))
    diagnostic = registration_diagnostics(source_frame.points, target_frame.points, initial,
        voxel_m=voxel_m, max_distance_m=max_distance_m,
        forward_depth_score='max_forward_depth_score_m' in thresholds and legacy_profile != 'visible_optical_depth',
        **(dict(observation=observations['source'], target_observation=observations['target'], scoring=scoring)
           if scoring is not None else {}))
    if legacy_profile == 'visible_optical_depth':
        from .pointcloud_slam_visible_overlap import VisibilityView
        views = []
        for frame, model in ((source_frame, observations['source']), (target_frame, observations['target'])):
            p = model['parameters']; k = p['K']
            views.append(VisibilityView(frame.points, (p['width'], p['height'], k[0], k[4], k[2], k[5]),
                                        np.asarray(model['sensor_from_child']), voxel_m))
        for direction, src, dst in (('forward', views[0], views[1]), ('reverse', views[1], views[0])):
            score = src.score(dst, np.asarray(diagnostic[direction]['transform']), max_distance_m)
            diagnostic[direction].update(forward_depth_score_m=score['error_m'],
                visible_coverage=score['coverage'], visible_score=score, score_metadata=score['score_metadata'])
        diagnostic.update(score_metadata=legacy_scoring(visible=True), score_scope='visible_optical_depth_bands')
    diagnostic.update(compatibility=deepcopy(strategy['compatibility']), observations=observations)
    gate = classify_registration(diagnostic, thresholds)
    result = {**strategy, 'hypothesis_created': False, 'loop_accepted': False,
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
