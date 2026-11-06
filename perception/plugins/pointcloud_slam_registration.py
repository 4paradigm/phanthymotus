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


def registration_diagnostics(source, target, initial, *, voxel_m, max_distance_m, forward_depth_score=False):
    """Refine an externally supplied initial transform in both directions.

    Directional fitness, error, and round-trip agreement describe the result;
    they cannot establish place identity or rule out repeated-scene matches.
    """
    import open3d as o3d
    if not all(math.isfinite(v) and v > 0 for v in (voxel_m, max_distance_m)):
        raise ValueError('voxel size and correspondence distance must be positive')
    initial = _rigid(initial)
    clouds = _clouds(source, target, voxel_m)
    if any(len(cloud.points) < 3 for cloud in clouds):
        raise ValueError('insufficient finite points')
    reg = o3d.pipelines.registration
    def refine(a, b, transform):
        value = reg.registration_icp(a, b, max_distance_m, transform,
                                     reg.TransformationEstimationPointToPoint(),
                                     reg.ICPConvergenceCriteria(max_iteration=60))
        count = len(value.correspondence_set)
        result = {'fitness': value.fitness, 'rmse_m': value.inlier_rmse if count else None,
                'correspondences': count, 'transform': value.transformation.tolist()}
        if forward_depth_score:
            result['forward_depth_score_m'] = _forward_depth_score(
                np.asarray(a.points), np.asarray(b.points), np.asarray(value.correspondence_set),
                value.transformation, max_distance_m)
        return result
    forward = refine(clouds[0], clouds[1], initial)
    reverse = refine(clouds[1], clouds[0], np.linalg.inv(initial))
    cycle_translation, cycle_angle = None, None
    if forward['correspondences'] and reverse['correspondences']:
        cycle = np.asarray(reverse['transform']) @ np.asarray(forward['transform'])
        cycle_translation = float(np.linalg.norm(cycle[:3, 3]))
        cosine = float(np.clip((np.trace(cycle[:3, :3]) - 1.) / 2., -1., 1.))
        cycle_angle = math.degrees(math.acos(cosine))
    return {'forward': forward, 'reverse': reverse,
            'cycle_translation_m': cycle_translation, 'cycle_angle_deg': cycle_angle,
            'loop_accepted': False}



def fixed_overlap(source, target, transform, geometry):
    """Score a predicted pose in both directions without an ICP update."""
    import open3d as o3d
    a, b = _clouds(source, target, geometry['voxel_m'])
    thresholds = geometry['thresholds']
    if min(len(a.points), len(b.points)) < 3:
        return [{'passes': False, 'reason': 'insufficient_points'}]
    limit = geometry['max_distance_m']
    scores = []
    for src, dst, pose in ((a, b, transform), (b, a, np.linalg.inv(transform))):
        result = o3d.pipelines.registration.evaluate_registration(src, dst, limit, pose)
        pairs = np.asarray(result.correspondence_set)
        if 'max_forward_depth_score_m' in thresholds:
            error = _forward_depth_score(np.asarray(src.points), np.asarray(dst.points),
                                         pairs, pose, limit)
            bound = thresholds['max_forward_depth_score_m']
        else:
            error = result.inlier_rmse
            bound = thresholds['max_rmse_m']
        scores.append({'fitness': result.fitness, 'error_m': error, 'count': len(pairs),
                       'passes': bool(result.fitness >= thresholds['min_fitness'] and
                                      len(pairs) >= thresholds['min_correspondences'] and
                                      error is not None and error <= bound)})
    return scores
