"""Experimental coarse geometric alignment, not proof of place identity."""
import math
import numpy as np

from .pointcloud_slam_stream import _rigid


def coarse_alignment(source, target, *, voxel_m, max_distance_m):
    """Return a source-to-target FPFH alignment, or None if unavailable."""
    import open3d as o3d
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in (voxel_m, max_distance_m)):
        raise ValueError('positive finite registration settings required')
    clouds, features = [], []
    for value in (source, target):
        points = np.asarray(value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('finite Nx3 cloud required')
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.copy()))
        cloud = cloud.voxel_down_sample(voxel_m)
        if len(cloud.points) < 10:
            return None
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_m, max_nn=30))
        feature = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_m, max_nn=100))
        clouds.append(cloud)
        features.append(feature)
    reg = o3d.pipelines.registration
    result = reg.registration_fgr_based_on_feature_matching(
        clouds[0], clouds[1], features[0], features[1],
        reg.FastGlobalRegistrationOption(maximum_correspondence_distance=max_distance_m))
    if not len(result.correspondence_set):
        return None
    return _rigid(result.transformation)


def relocalize_pair(target, target_graph, source, source_graph, *,
                    descriptor_voxel_m, max_candidates, **merge_options):
    """Try bounded descriptor-ranked frame pairs; never certify a real loop."""
    from .pointcloud_slam_keyframes import Segment
    from .pointcloud_slam_loop_search import _descriptor
    from .pointcloud_slam_segment_merge import experimental_segment_merge
    if (not isinstance(target, Segment) or not isinstance(source, Segment) or
            not target.frames or not source.frames or target.segment_id == source.segment_id or
            type(max_candidates) is not int or max_candidates <= 0):
        raise ValueError('distinct nonempty segments and positive candidate limit required')
    source_index = len(source.frames) - 1
    try:
        query = _descriptor(source.frames[source_index].points, descriptor_voxel_m)
    except ValueError:
        return {'hypothesis_created': False, 'loop_accepted': False, 'attempts': 0}
    candidates = []
    for i, frame in enumerate(target.frames):
        try:
            descriptor = _descriptor(frame.points, descriptor_voxel_m)
        except ValueError:
            continue
        candidates.append((float(np.linalg.norm(query - descriptor)), i))
    attempts = 0
    for _, index in sorted(candidates)[:max_candidates]:
        initial = coarse_alignment(source.frames[source_index].points, target.frames[index].points,
                                   voxel_m=descriptor_voxel_m,
                                   max_distance_m=merge_options['max_distance_m'])
        attempts += 1
        if initial is None:
            continue
        result = experimental_segment_merge(target, target_graph, source, source_graph,
            target_index=index, source_index=source_index, initial=initial, **merge_options)
        if result['hypothesis_created']:
            return {**result, 'attempts': attempts, 'target_index': index, 'source_index': source_index}
    return {'hypothesis_created': False, 'loop_accepted': False, 'attempts': attempts}
