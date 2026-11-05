"""Reciprocal point-cloud registration diagnostics; not a loop decision."""
import math

import numpy as np

from .pointcloud_slam_stream import _rigid


def registration_diagnostics(source, target, initial, *, voxel_m, max_distance_m):
    """Refine an externally supplied initial transform in both directions.

    Directional fitness, error, and round-trip agreement describe the result;
    they cannot establish place identity or rule out repeated-scene matches.
    """
    import open3d as o3d
    if not all(math.isfinite(v) and v > 0 for v in (voxel_m, max_distance_m)):
        raise ValueError('voxel size and correspondence distance must be positive')
    initial = _rigid(initial)
    clouds = []
    for value in (source, target):
        points = np.asarray(value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('expected Nx3 cloud')
        points = points[np.isfinite(points).all(axis=1)]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(voxel_m)
        if len(cloud.points) < 3:
            raise ValueError('insufficient finite points')
        clouds.append(cloud)
    reg = o3d.pipelines.registration
    def refine(a, b, transform):
        value = reg.registration_icp(a, b, max_distance_m, transform,
                                     reg.TransformationEstimationPointToPoint(),
                                     reg.ICPConvergenceCriteria(max_iteration=60))
        count = len(value.correspondence_set)
        return {'fitness': value.fitness, 'rmse_m': value.inlier_rmse if count else None,
                'correspondences': count, 'transform': value.transformation.tolist()}
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


