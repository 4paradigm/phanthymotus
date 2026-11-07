"""Single-segment revisit factors confirmed by explicit moving holdout policy."""
import numpy as np
from copy import deepcopy

from .pointcloud_slam_geometry import classify_registration
from .pointcloud_slam_registration import registration_diagnostics
from .pointcloud_slam_visible_overlap import ReferenceWindow, VisibilityView, split_decision
from .pointcloud_slam_observation import make_view, resolve_revisit, resolve_scoring, legacy_scoring, observation_for_frame


class RevisitSearch:
    def __init__(self, camera, optical_from_child, geometry, *, observation=None, scoring=None, policy=None):
        self.observation = deepcopy(observation)
        self.scoring = resolve_scoring(scoring, observation) if scoring is not None else None
        self.policy = resolve_revisit(policy, observation)
        if observation is not None:
            optical_from_child = (np.asarray(observation['sensor_from_child'])
                                  if observation['sensor_from_child'] is not None else None)
            if observation['kind'] == 'pinhole':
                p = observation['parameters']; k = p['K']
                camera = (p['width'], p['height'], k[0], k[4], k[2], k[5])
            elif scoring is None:
                raise ValueError('lidar revisit requires a radial scoring contract')
        self.camera = camera
        self.optical = optical_from_child
        self.forward_axis = (optical_from_child[2 if camera is not None else 0, :3]
                             if optical_from_child is not None else None)
        self.geometry = geometry
        self.pending = None
        self.support = []
        self.last_seed_ns = None
        self.attempts = 0
        self.confirmation_evaluations = 0

    def view(self, frame):
        observation = observation_for_frame(self.observation, frame) if self.observation is not None else None
        if self.scoring is not None:
            return make_view(np.asarray(frame.points), observation, self.geometry['voxel_m'], self.scoring)
        optical = np.asarray(observation['sensor_from_child']) if observation is not None else self.optical
        return VisibilityView(np.asarray(frame.points), self.camera, optical, self.geometry['voxel_m'])

    def _axis(self, frame):
        if self.observation is None:
            return self.forward_axis
        return np.asarray(observation_for_frame(self.observation, frame)['sensor_from_child'])[2, :3]

    def update(self, frames, poses, eligible):
        index = len(frames)-1
        geom = self.geometry
        limit = geom['max_distance_m']
        gates = geom['thresholds']
        policy = self.policy
        error_limit = gates['max_range_stratified_rmse_m' if self.scoring is not None else 'max_forward_depth_score_m']
        if self.pending is None:
            if not eligible:
                return None
            if (self.last_seed_ns is not None and frames[index].cloud_stamp_ns-self.last_seed_ns
                    < geom['min_separation_s']*1e9):
                return None
        q = None

        def score(target, forward, reverse, coverage):
            nonlocal q
            if q is None:
                q = self.view(frames[index])
            scores = (q.score(target, forward, limit), target.score(q, reverse, limit))
            ok = all(all(split_decision(s, coverage, error_limit,
                                        gates['min_correspondences']).values())
                     and s['fitness'] >= gates['min_fitness']
                     and s['correspondences'] >= gates['min_correspondences'] for s in scores)
            return ok, scores

        if self.pending is not None:
            self.confirmation_evaluations += 1
            source, target, diagnostic, reference = self.pending
            forward = np.asarray(diagnostic['forward']['transform'])
            predicted = poses[target] @ forward @ np.linalg.inv(poses[source]) @ poses[index]
            nearest = min(range(len(reference)), key=lambda k:
                          np.linalg.norm(poses[reference[k]][:3, 3]-predicted[:3, 3]))
            size = policy['reference_window_frames']
            start = max(0, min(nearest-size//2, len(reference)-size))
            window = reference[start:start+size]
            views = [self.view(frames[j]) for j in window]
            submap = ReferenceWindow([np.asarray(frames[j].points) for j in window],
                                     views, [poses[j] for j in window], geom['voxel_m'])
            tr = np.linalg.inv(poses[window[0]]) @ predicted
            passed, _ = score(submap, tr, np.linalg.inv(tr), policy['confirmation_min_coverage'])
            if not passed:
                self.pending = None
            elif all(np.linalg.norm(poses[index][:3, 3]-poses[j][:3, 3]) >= policy['min_support_translation_m']
                     for j in [source]+self.support):
                self.support.append(index)
                if len(self.support) == policy['confirmation_frames']:
                    self.pending = None
                    self.last_seed_ns = frames[source].cloud_stamp_ns
                    return dict(source=source, target=target, diagnostic=diagnostic,
                                confirmation_stamps_ns=[frames[j].cloud_stamp_ns for j in self.support])
            return None

        # Use historical odometry only to initialize ICP within this segment.
        # A seed must be separated by a departure and return, not a long stop.
        proposals = []
        for target in eligible:
            initial = np.linalg.inv(poses[target]) @ poses[index]
            if (np.linalg.norm(initial[:3, 3]) > policy['return_distance_m'] or
                    (policy['direction_policy'] == 'sensor_forward' and
                     self._axis(frames[target]) @ initial[:3, :3] @ self._axis(frames[index])
                     < policy['direction_min_dot'])):
                continue
            if not any(np.linalg.norm(p[:3, 3]-poses[target][:3, 3]) > policy['departure_distance_m']
                       for p in poses[target+1:index]):
                continue
            self.attempts += 1
            d = registration_diagnostics(frames[index].points, frames[target].points, initial,
                                        voxel_m=geom['voxel_m'], max_distance_m=limit)
            ok, scores = score(self.view(frames[target]),
                               np.asarray(d['forward']['transform']), np.asarray(d['reverse']['transform']),
                               policy['seed_min_coverage'])
            d['score_scope'] = 'versioned_range_strata' if self.scoring is not None else 'visible_optical_depth_bands'
            d['score_metadata'] = deepcopy(self.scoring) if self.scoring is not None else legacy_scoring(visible=True)
            d['compatibility'] = [] if self.scoring is not None else ['legacy visible score; original thresholds retained']
            for direction, s in zip(('forward', 'reverse'), scores):
                if self.scoring is not None:
                    d[direction].update(s)
                else:
                    d[direction]['forward_depth_score_m'] = s['error_m']
                    d[direction]['score_metadata'] = deepcopy(s['score_metadata'])
                d[direction]['visible_coverage'] = s['coverage']
            if not ok or not classify_registration(d, gates)['passes']:
                continue
            # Follow this contiguous former visit; exclude the current traversal.
            lo, hi = target, target+1
            def nearby(j):
                return (frames[index].cloud_stamp_ns-frames[j].cloud_stamp_ns >= geom['min_separation_s']*1e9
                        and (policy['direction_policy'] == 'none' or
                             np.dot(poses[j][:3,:3] @ self._axis(frames[j]),
                                    poses[target][:3,:3] @ self._axis(frames[target]))
                             >= policy['direction_min_dot'])
                        and np.linalg.norm(poses[j][:3,3]-poses[target][:3,3]) <= policy['reference_radius_m'])
            while lo and nearby(lo-1):
                lo -= 1
            while hi < index and nearby(hi):
                hi += 1
            if hi-lo >= policy['reference_window_frames']:
                proposals.append((index, target, d, range(lo, hi)))
        if proposals:
            _, first, fit, _ = proposals[0]
            prediction = poses[first] @ fit['forward']['transform']
            for _, other, fit, _ in proposals[1:]:
                delta = np.linalg.inv(prediction) @ poses[other] @ fit['forward']['transform']
                angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2, -1., 1.)))
                if (np.linalg.norm(delta[:3,3]) > gates['max_cycle_translation_m'] or
                        angle > gates['max_cycle_angle_deg']):
                    return None
            self.pending = proposals[0]
            self.support = []
        return None
