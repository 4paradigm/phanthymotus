"""Unvalidated bounded FPFH candidate scheduling for an experimental graph.

No result from this module is a verified loop. Parameters require independent
positive and hard-negative evaluation before use for real mapping.
"""
import math
import time
import numpy as np

from .pointcloud_slam_incremental_backend import ExperimentalLoopGraph
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_map import rebuild_segment


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
        self.revisit = None
        self._voxel = descriptor_voxel_m
        self._separation_ns = min_separation_s * 1e9
        self._candidate_separation_ns = candidate_separation_s * 1e9
        self._max_per_frame = max_per_frame
        self._max_candidates = graph_options['max_candidates']
        self._graph = ExperimentalLoopGraph(**graph_options)
        self._descriptors, self._stamps = [], []
        self._segment_id, self._attempts, self._skips, self._last = None, 0, 0, None

    def update(self, segment, graph, *, online=False, max_points=None, build_map=True):
        started = time.perf_counter()
        descriptor_s = graph_s = 0.
        if (not isinstance(segment, Segment) or not segment.frames or
                graph.get('segment_id') != segment.segment_id or
                (self._segment_id is not None and segment.segment_id != self._segment_id) or
                len(segment.frames) < len(self._descriptors) or
                len(graph.get('poses', ())) != len(segment.frames)):
            raise ValueError('one append-only segment and matching graph required')
        if self._segment_id is None:
            self._segment_id = segment.segment_id
        first_new = len(self._descriptors)
        # Revisit confirmation needs consecutive moving frames even when several
        # keyframes arrive between preview ticks. Legacy previews sample the latest.
        selected_indices = ({0, len(segment.frames) - 1} if online and self.revisit is None else
                            set(range(first_new, len(segment.frames))))
        for index in range(first_new, len(segment.frames)):
            if index not in selected_indices:
                self._descriptors.append(None)
                self._stamps.append(segment.frames[index].cloud_stamp_ns)
                continue
            frame = segment.frames[index]
            tick = time.perf_counter()
            try:
                desc = _descriptor(frame.points, self._voxel)
            except ValueError:
                desc = None
            descriptor_s += time.perf_counter() - tick
            tick = time.perf_counter()
            partial = Segment(segment.segment_id, segment.frames[:index+1])
            subgraph = {'segment_id': segment.segment_id, 'poses': graph['poses'][:index+1],
                        'edges': graph['edges'][:index]}
            result = self._graph.update(partial, subgraph, build_maps=False)
            selected = []
            eligible = [i for i in range(index) if desc is not None and self._descriptors[i] is not None]
            ranked = sorted(eligible, key=lambda j: (float(np.linalg.norm(desc-self._descriptors[j])), j))
            historical = [j for j in ranked if frame.cloud_stamp_ns-self._stamps[j] >= self._separation_ns]
            if self.revisit is not None:
                if len(result['passed_candidates']) < self._max_candidates:
                    confirmed = self.revisit.update(partial.frames, subgraph['poses'], historical[:3])
                    self._attempts = self.revisit.attempts
                    if confirmed is not None:
                        result = self._graph.update(partial, subgraph, confirmed=confirmed, build_maps=False)
            else:
                for old in historical:
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
                result = self._graph.update(partial, subgraph, candidate=(index, old, initial), build_maps=False)
                self._attempts += 1
                if len(result['passed_candidates']) > count:
                    break
            graph_s += time.perf_counter() - tick
            self._descriptors.append(desc)
            self._stamps.append(frame.cloud_stamp_ns)
            if desc is None:
                self._skips += 1
            self._last = {**result, 'candidate_attempts': self._attempts, 'descriptor_skips': self._skips}
        if first_new == len(segment.frames):
            # Revalidate the snapshot even when no new descriptor is required.
            tick = time.perf_counter()
            result = self._graph.update(segment, graph, build_maps=False)
            graph_s += time.perf_counter() - tick
            self._last = {**result, 'candidate_attempts': self._attempts, 'descriptor_skips': self._skips}
        tick = time.perf_counter()
        self._last['before_map'] = (None if online else rebuild_segment(segment, graph['poses']))
        self._last['after_map'] = (rebuild_segment(segment, self._last['poses'], max_points=max_points)
                                   if build_map else None)
        self._last['descriptor_unindexed'] = sum(d is None for d in self._descriptors) - self._skips
        self._last['timings_ms'] = {'descriptor': descriptor_s * 1000,
            'registration_and_graph': graph_s * 1000,
            'map': (time.perf_counter() - tick) * 1000,
            'total': (time.perf_counter() - started) * 1000}
        return self._last
