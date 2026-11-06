"""Incremental geometry-gated graph experiments; never accept real loop edges.

The caller must serialize access and run registration outside ROS callbacks. This
single-segment prototype has no retrieval, calibrated thresholds or safety case.
"""
import math

from .pointcloud_slam_experimental_backend import optimize_candidates
from .pointcloud_slam_geometry import classify_registration
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_map import map_from_odom, rebuild_segment
from .pointcloud_slam_registration import registration_diagnostics


class ExperimentalLoopGraph:
    """Retain bounded geometry-passing candidates and rebuild map history."""

    def __init__(self, *, thresholds, odom_sigmas, loop_sigmas, huber_k,
                 voxel_m, max_distance_m, max_candidates):
        classify_registration({}, thresholds)
        if (type(max_candidates) is not int or max_candidates <= 0 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(x) or x <= 0 for x in (voxel_m, max_distance_m))):
            raise ValueError('positive finite registration settings and capacity required')
        self._thresholds = dict(thresholds)
        self._odom_sigmas, self._loop_sigmas, self._huber_k = tuple(odom_sigmas), tuple(loop_sigmas), huber_k
        self._voxel_m, self._max_distance_m = voxel_m, max_distance_m
        self._max_candidates = max_candidates
        self._segment_id, self._frames, self._poses, self._edges, self._candidates = None, (), (), (), ()

    def update(self, segment, graph, *, candidate=None, confirmed=None, build_maps=True):
        """Update one immutable segment snapshot and optionally test one candidate.

        Candidate is ``(source_index, target_index, T_target_source_initial)``.
        The returned map and map-to-odom are diagnostic only; neither is published.
        Failed validation or optimization leaves retained state unchanged.
        """
        if (not isinstance(segment, Segment) or not segment.frames or
                graph.get('segment_id') != segment.segment_id or
                (self._segment_id is not None and segment.segment_id != self._segment_id) or
                len(graph.get('poses', ())) != len(segment.frames) or
                len(segment.frames) < len(self._frames) or
                any(a is not b for a, b in zip(segment.frames, self._frames))):
            raise ValueError('one append-only segment and matching graph required')
        poses = graph['poses']
        import numpy as np
        if any(not np.array_equal(a, b) for a, b in zip(poses, self._poses)):
            raise ValueError('previous odometry graph poses changed')
        edges = graph.get('edges')
        if (not isinstance(edges, list) or len(edges) < len(self._edges) or
                any(not isinstance(edge, dict) or
                    (edge.get('from'), edge.get('to')) != (old[0], old[1]) or
                    not np.array_equal(edge.get('transform'), old[2])
                    for edge, old in zip(edges, self._edges))):
            raise ValueError('previous odometry graph edges changed')
        pending = list(self._candidates)
        if confirmed is not None:
            if len(pending) >= self._max_candidates:
                raise ValueError('candidate capacity reached')
            if candidate is not None:
                raise ValueError('only one candidate may be tested per update')
            pending.append(confirmed)
        if candidate is not None:
            if not isinstance(candidate, (tuple, list)) or len(candidate) != 3:
                raise ValueError('candidate requires source, target and initial transform')
            source, target, initial = candidate
            if (type(source) is not int or type(target) is not int or source == target or
                    not 0 <= source < len(segment.frames) or not 0 <= target < len(segment.frames)):
                raise ValueError('invalid candidate endpoints')
            key = tuple(sorted((source, target)))
            if any(tuple(sorted((c['source'], c['target']))) == key for c in pending):
                raise ValueError('duplicate candidate')
            if len(pending) >= self._max_candidates:
                raise ValueError('candidate capacity reached')
            diagnostic = registration_diagnostics(segment.frames[source].points,
                segment.frames[target].points, initial, voxel_m=self._voxel_m,
                max_distance_m=self._max_distance_m,
                forward_depth_score='max_forward_depth_score_m' in self._thresholds)
            pending.append({'source': source, 'target': target, 'diagnostic': diagnostic})
        result = optimize_candidates(graph, pending, thresholds=self._thresholds,
            odom_sigmas=self._odom_sigmas, loop_sigmas=self._loop_sigmas, huber_k=self._huber_k)
        before = rebuild_segment(segment, poses) if build_maps else None
        after = rebuild_segment(segment, result['poses']) if build_maps else None
        correction = map_from_odom(result['poses'][-1], poses[-1])
        passing = {(c['source'], c['target']) for c in result['passed_candidates']}
        self._candidates = tuple(c for c in pending if (c['source'], c['target']) in passing)
        self._segment_id, self._frames = segment.segment_id, tuple(segment.frames)
        self._poses = tuple(np.array(pose, copy=True) for pose in poses)
        self._edges = tuple((edge['from'], edge['to'], np.array(edge['transform'], copy=True))
                            for edge in edges)
        return {**result, 'segment_id': segment.segment_id, 'before_map': before,
                'after_map': after, 'map_from_odom': correction}
