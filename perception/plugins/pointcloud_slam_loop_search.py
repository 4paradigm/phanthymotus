"""Unvalidated bounded FPFH candidate scheduling for an experimental graph.

No result from this module is a verified loop. Parameters require independent
positive and hard-negative evaluation before use for real mapping.
"""
import math
import time
from copy import deepcopy
import numpy as np

from .pointcloud_slam_incremental_backend import ExperimentalLoopGraph
from .pointcloud_slam_keyframes import Segment
from .pointcloud_slam_map import rebuild_segment
from .pointcloud_slam_observation import resolve_scoring, resolve_revisit, legacy_scoring


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
                 candidate_separation_s, max_per_frame, observation=None,
                 scoring=None, revisit=None, legacy_profile=None, **graph_options):
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v < 0 for v in
                (descriptor_voxel_m, min_separation_s, candidate_separation_s)) or
                descriptor_voxel_m == 0 or type(max_per_frame) is not int or max_per_frame <= 0):
            raise ValueError('invalid explicit candidate search settings')
        self.observation = deepcopy(observation)
        new_score = legacy_profile is None and (observation is not None or scoring is not None)
        if legacy_profile is not None:
            if scoring is not None or legacy_profile not in ('visible_optical_depth', 'child_x_depth', 'inlier_rmse'):
                raise ValueError('select one explicit legacy profile or a new score contract')
            if (legacy_profile == 'visible_optical_depth' and
                    (observation is None or observation.get('kind') != 'pinhole')):
                raise ValueError('legacy visible profile requires an explicit pinhole observation')
            expected = 'max_rmse_m' if legacy_profile == 'inlier_rmse' else 'max_forward_depth_score_m'
            if expected not in graph_options['thresholds']:
                raise ValueError('legacy profile requires its original threshold metric')
            self.scoring = legacy_scoring(visible=legacy_profile == 'visible_optical_depth',
                                          rmse=legacy_profile == 'inlier_rmse')
            self.compatibility = ['explicit legacy scoring profile: ' + self.scoring['metric_id'] +
                                  '; original numeric thresholds and support semantics retained']
        elif new_score:
            if observation is None:
                raise ValueError('explicit observation required for new scoring')
            self.scoring = resolve_scoring(scoring, observation)
            gates = graph_options['thresholds']
            if ('max_range_stratified_rmse_m' not in gates or
                    gates.get('score_contract') != self.scoring or
                    'max_forward_depth_score_m' in gates or 'max_rmse_m' in gates):
                raise ValueError('new score requires matching threshold contract')
            self.compatibility = []
        else:
            if 'max_range_stratified_rmse_m' in graph_options['thresholds']:
                raise ValueError('new score threshold requires observation and scoring')
            self.scoring = legacy_scoring(rmse='max_forward_depth_score_m' not in graph_options['thresholds'])
            self.compatibility = ['legacy scoring profile: ' + self.scoring['metric_id'] +
                                  '; numeric thresholds retain their original support and range semantics']
        self.revisit = None
        self._voxel = descriptor_voxel_m
        self._separation_ns = min_separation_s * 1e9
        self._candidate_separation_ns = candidate_separation_s * 1e9
        self._max_per_frame = max_per_frame
        self._max_candidates = graph_options['max_candidates']
        self._graph = ExperimentalLoopGraph(**graph_options, **(
            dict(observation=self.observation, scoring=self.scoring) if new_score else {}))
        revisit_policy = None
        if new_score or legacy_profile == 'visible_optical_depth':
            revisit_policy = resolve_revisit(revisit, observation)
            if legacy_profile == 'visible_optical_depth' and not revisit_policy['enabled']:
                raise ValueError('legacy visible profile requires its visible revisit strategy')
            if revisit_policy['enabled']:
                from .pointcloud_slam_revisit import RevisitSearch
                self.revisit = RevisitSearch(None, None, dict(graph_options, min_separation_s=min_separation_s),
                                             observation=self.observation,
                                             scoring=self.scoring if new_score else None, policy=revisit_policy)
        elif revisit is not None:
            raise ValueError('revisit policy requires an explicit observation')
        self.algorithm_profile = dict(schema='motus.slam_algorithm/1', descriptor_voxel_m=descriptor_voxel_m,
            descriptor_normal_radius_factor=2., descriptor_feature_radius_factor=5.,
            icp_max_iterations=60, min_separation_s=min_separation_s,
            candidate_separation_s=candidate_separation_s,
            voxel_m=graph_options['voxel_m'], max_distance_m=graph_options['max_distance_m'],
            thresholds=deepcopy(graph_options['thresholds']),
            revisit=revisit_policy)
        self.resource_profile = dict(schema='motus.slam_resources/1', max_per_frame=max_per_frame,
            max_candidates=self._max_candidates, candidate_capacity_scope='retained_edges',
            confirmation_evaluations_per_frame=1 if self.revisit is not None else 0,
            online_descriptor_policy='consecutive_frames' if self.revisit is not None else 'anchor_and_latest')
        self._descriptors, self._stamps = [], []
        self._segment_id, self._attempts, self._skips, self._last = None, 0, 0, None

    def strategy_snapshot(self):
        return deepcopy(dict(resolved_observation=self.observation, resolved_scoring=self.scoring,
                             algorithm_profile=self.algorithm_profile, resource_profile=self.resource_profile,
                             compatibility=self.compatibility))

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
            for old in historical:
                if any(abs(self._stamps[old]-self._stamps[i]) < self._candidate_separation_ns for i in selected):
                    continue
                selected.append(old)
                if len(selected) == self._max_per_frame:
                    break
            if self.revisit is not None:
                if len(result['passed_candidates']) < self._max_candidates:
                    confirmed = self.revisit.update(partial.frames, subgraph['poses'], selected)
                    self._attempts = self.revisit.attempts
                    if confirmed is not None:
                        result = self._graph.update(partial, subgraph, confirmed=confirmed, build_maps=False)
                selected = []
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
        self._last['confirmation_evaluations'] = getattr(self.revisit, 'confirmation_evaluations', 0)
        self._last.update(self.strategy_snapshot())
        return self._last
