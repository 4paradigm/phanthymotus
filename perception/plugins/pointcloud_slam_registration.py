"""Reciprocal point-cloud registration diagnostics; not a loop decision."""
import math

import numpy as np

from .pointcloud_slam_stream import _rigid


def _forward_depth_score(source, target, pairs, transform, limit):
    """Equal-weight depth strata in an X-forward camera-origin child frame."""
    bands = np.digitize(source[:, 0], (2., 4.))
    counts = np.bincount(bands, minlength=3)
    if np.any(counts == 0):
        return None
    squared = np.full(len(source), limit ** 2)
    src, dst = pairs.reshape(-1, 2).T
    delta = source[src] @ transform[:3, :3].T + transform[:3, 3] - target[dst]
    squared[src] = np.sum(delta ** 2, axis=1)
    return float(np.sqrt(np.mean(np.bincount(bands, weights=squared) / counts)))


def _clouds(source, target, voxel_m):
    import open3d as o3d
    clouds = []
    for value in (source, target):
        points = np.asarray(value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('expected Nx3 cloud')
        points = points[np.isfinite(points).all(axis=1)]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(voxel_m)
        clouds.append(cloud)
    return clouds


def registration_diagnostics(source, target, initial, *, voxel_m, max_distance_m,
                             forward_depth_score=False, observation=None, scoring=None, target_observation=None):
    """Refine an externally supplied initial transform in both directions.

    Directional fitness, error, and round-trip agreement describe the result;
    they cannot establish place identity or rule out repeated-scene matches.
    """
    import open3d as o3d
    if not all(math.isfinite(v) and v > 0 for v in (voxel_m, max_distance_m)):
        raise ValueError('voxel size and correspondence distance must be positive')
    initial = _rigid(initial)
    views = None
    if observation is not None or scoring is not None:
        if forward_depth_score:
            raise ValueError('legacy depth score cannot be mixed with a new score contract')
        from .pointcloud_slam_observation import make_view
        views = [make_view(source, observation, voxel_m, scoring),
                 make_view(target, target_observation if target_observation is not None else observation,
                           voxel_m, scoring)]
    clouds = _clouds(source, target, voxel_m)
    if any(len(cloud.points) < 3 for cloud in clouds):
        raise ValueError('insufficient finite points')
    reg = o3d.pipelines.registration
    def refine(a, b, transform):
        from .pointcloud_slam_observation import legacy_scoring
        value = reg.registration_icp(a, b, max_distance_m, transform,
                                     reg.TransformationEstimationPointToPoint(),
                                     reg.ICPConvergenceCriteria(max_iteration=60))
        count = len(value.correspondence_set)
        result = {'fitness': value.fitness, 'rmse_m': value.inlier_rmse if count else None,
                'correspondences': count, 'transform': value.transformation.tolist(),
                'score_metadata': legacy_scoring(rmse=True)}
        if forward_depth_score:
            result['forward_depth_score_m'] = _forward_depth_score(
                np.asarray(a.points), np.asarray(b.points), np.asarray(value.correspondence_set),
                value.transformation, max_distance_m)
            result['score_metadata'] = legacy_scoring()
        return result
    forward = refine(clouds[0], clouds[1], initial)
    reverse = refine(clouds[1], clouds[0], np.linalg.inv(initial))
    if views is not None:
        for result, a, b in ((forward, views[0], views[1]), (reverse, views[1], views[0])):
            result.update(a.score(b, np.asarray(result['transform']), max_distance_m))
    cycle_translation, cycle_angle = None, None
    if forward['correspondences'] and reverse['correspondences']:
        cycle = np.asarray(reverse['transform']) @ np.asarray(forward['transform'])
        cycle_translation = float(np.linalg.norm(cycle[:3, 3]))
        cosine = float(np.clip((np.trace(cycle[:3, :3]) - 1.) / 2., -1., 1.))
        cycle_angle = math.degrees(math.acos(cosine))
    return {'forward': forward, 'reverse': reverse,
            'cycle_translation_m': cycle_translation, 'cycle_angle_deg': cycle_angle,
            'loop_accepted': False,
            'score_metadata': views[0].score_metadata if views is not None else forward.get('score_metadata'),
            'compatibility': [] if views is not None else ['legacy registration error metric; original semantics retained']}



def fixed_overlap(source, target, transform, geometry):
    """Score a predicted pose in both directions without an ICP update."""
    import open3d as o3d
    a, b = _clouds(source, target, geometry['voxel_m'])
    thresholds = geometry['thresholds']
    if min(len(a.points), len(b.points)) < 3:
        return [{'passes': False, 'reason': 'insufficient_points'}]
    limit = geometry['max_distance_m']
    scores = []
    if geometry.get('observation') is not None or geometry.get('scoring') is not None:
        from .pointcloud_slam_observation import make_view, resolve_scoring
        spec = resolve_scoring(geometry.get('scoring'), geometry.get('observation'))
        if (thresholds.get('score_contract') != spec or
                'max_range_stratified_rmse_m' not in thresholds or
                'max_forward_depth_score_m' in thresholds or 'max_rmse_m' in thresholds):
            raise ValueError('fixed overlap requires a matching score threshold contract')
        views = [make_view(source, geometry['observation'], geometry['voxel_m'], spec),
                 make_view(target, geometry.get('target_observation', geometry['observation']),
                           geometry['voxel_m'], spec)]
        for src, dst, pose in ((views[0], views[1], transform), (views[1], views[0], np.linalg.inv(transform))):
            result = src.score(dst, pose, limit)
            error = result['range_stratified_rmse_m']
            result['passes'] = bool(result['fitness'] >= thresholds['min_fitness'] and
                result['correspondences'] >= thresholds['min_correspondences'] and
                error is not None and error <= thresholds['max_range_stratified_rmse_m'])
            scores.append(result)
        return scores
    for src, dst, pose in ((a, b, transform), (b, a, np.linalg.inv(transform))):
        from .pointcloud_slam_observation import legacy_scoring
        result = o3d.pipelines.registration.evaluate_registration(src, dst, limit, pose)
        pairs = np.asarray(result.correspondence_set)
        if 'max_forward_depth_score_m' in thresholds:
            error = _forward_depth_score(np.asarray(src.points), np.asarray(dst.points),
                                         pairs, pose, limit)
            bound = thresholds['max_forward_depth_score_m']
        else:
            error = result.inlier_rmse
            bound = thresholds['max_rmse_m']
        contract = legacy_scoring(rmse='max_forward_depth_score_m' not in thresholds)
        if thresholds.get('score_contract', contract) != contract:
            raise ValueError('legacy fixed overlap threshold contract differs from its score')
        scores.append({'fitness': result.fitness, 'error_m': error, 'count': len(pairs),
                       'score_metadata': contract,
                       'compatibility': ['legacy fixed overlap; original metric semantics retained'],
                       'passes': bool(result.fitness >= thresholds['min_fitness'] and
                                      len(pairs) >= thresholds['min_correspondences'] and
                                      error is not None and error <= bound)})
    return scores
