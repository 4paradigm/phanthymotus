"""Unvalidated bounded FPFH candidate scheduling for an experimental graph.

No result from this module is a verified loop. Parameters require independent
positive and hard-negative evaluation before use for real mapping.
"""
import math
import numpy as np

from .pointcloud_slam_incremental_backend import ExperimentalLoopGraph
from .pointcloud_slam_keyframes import Segment


def _descriptor(points, voxel):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points, dtype=float, copy=True)))
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) < 10:
        raise ValueError('insufficient points for candidate descriptor')
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
    result = np.mean(np.asarray(feature.data), axis=1)
    norm = np.linalg.norm(result)
    if not np.isfinite(result).all() or not math.isfinite(norm) or norm <= 0:
        raise ValueError('invalid candidate descriptor')
    return result / norm


class ExperimentalLoopSearch:
    """Rank old point clouds and geometrically test a bounded number per frame."""

    def __init__(self, *, descriptor_voxel_m, min_separation_s,
                 candidate_separation_s, max_per_frame, **graph_options):
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v < 0 for v in
                (descriptor_voxel_m, min_separation_s, candidate_separation_s)) or
                descriptor_voxel_m == 0 or type(max_per_frame) is not int or max_per_frame <= 0):
            raise ValueError('invalid explicit candidate search settings')
        self._voxel = descriptor_voxel_m
        self._separation_ns = min_separation_s * 1e9
        self._candidate_separation_ns = candidate_separation_s * 1e9
        self._max_per_frame = max_per_frame
        self._max_candidates = graph_options['max_candidates']
        self._graph = ExperimentalLoopGraph(**graph_options)
        self._descriptors, self._stamps = [], []
        self._segment_id, self._attempts, self._skips, self._last = None, 0, 0, None

    def update(self, segment, graph):
        if (not isinstance(segment, Segment) or not segment.frames or
                graph.get('segment_id') != segment.segment_id or
                (self._segment_id is not None and segment.segment_id != self._segment_id) or
                len(segment.frames) < len(self._descriptors) or
                len(graph.get('poses', ())) != len(segment.frames)):
            raise ValueError('one append-only segment and matching graph required')
        if self._segment_id is None:
            self._segment_id = segment.segment_id
        first_new = len(self._descriptors)
        for index in range(first_new, len(segment.frames)):
            frame = segment.frames[index]
            try:
                desc = _descriptor(frame.points, self._voxel)
            except ValueError:
                desc = None
            partial = Segment(segment.segment_id, segment.frames[:index+1])
            subgraph = {'segment_id': segment.segment_id, 'poses': graph['poses'][:index+1],
                        'edges': graph['edges'][:index]}
            result = self._graph.update(partial, subgraph)
            selected = []
            eligible = [i for i in range(index) if desc is not None and self._descriptors[i] is not None]
            for old in sorted(eligible, key=lambda i: (float(np.linalg.norm(desc-self._descriptors[i])), i)):
                if frame.cloud_stamp_ns - self._stamps[old] < self._separation_ns:
                    continue
                if any(abs(self._stamps[old]-self._stamps[i]) < self._candidate_separation_ns for i in selected):
                    continue
                selected.append(old)
                if len(selected) == self._max_per_frame:
                    break
            for old in selected:
                if len(result['passed_candidates']) >= self._max_candidates:
                    break
                initial = np.linalg.inv(subgraph['poses'][old]) @ subgraph['poses'][index]
                count = len(result['passed_candidates'])
                result = self._graph.update(partial, subgraph, candidate=(index, old, initial))
                self._attempts += 1
                if len(result['passed_candidates']) > count:
                    break
            self._descriptors.append(desc)
            self._stamps.append(frame.cloud_stamp_ns)
            if desc is None:
                self._skips += 1
            self._last = {**result, 'candidate_attempts': self._attempts, 'descriptor_skips': self._skips}
        if first_new == len(segment.frames):
            # Revalidate the snapshot even when no new descriptor is required.
            result = self._graph.update(segment, graph)
            self._last = {**result, 'candidate_attempts': self._attempts, 'descriptor_skips': self._skips}
        return self._last
