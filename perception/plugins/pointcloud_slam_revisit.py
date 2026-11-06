"""Single-segment revisit factors confirmed by two later moving keyframes."""
import numpy as np

from .pointcloud_slam_geometry import classify_registration
from .pointcloud_slam_registration import registration_diagnostics
from .pointcloud_slam_visible_overlap import ReferenceWindow, VisibilityView, split_decision


class RevisitSearch:
    def __init__(self, camera, optical_from_child, geometry):
        self.camera = camera
        self.optical = optical_from_child
        self.forward_axis = optical_from_child[2, :3]
        self.geometry = geometry
        self.pending = None
        self.support = []
        self.last_seed_ns = None
        self.attempts = 0

    def view(self, frame):
        return VisibilityView(np.asarray(frame.points), self.camera, self.optical, self.geometry['voxel_m'])

    def update(self, frames, poses, eligible):
        index = len(frames)-1
        geom = self.geometry
        limit = geom['max_distance_m']
        gates = geom['thresholds']
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
            ok = all(all(split_decision(s, coverage, gates['max_forward_depth_score_m'],
                                        gates['min_correspondences']).values())
                     and s['fitness'] >= gates['min_fitness']
                     and s['correspondences'] >= gates['min_correspondences'] for s in scores)
            return ok, scores

        if self.pending is not None:
            source, target, diagnostic, reference = self.pending
            forward = np.asarray(diagnostic['forward']['transform'])
            predicted = poses[target] @ forward @ np.linalg.inv(poses[source]) @ poses[index]
            nearest = min(range(len(reference)), key=lambda k:
                          np.linalg.norm(poses[reference[k]][:3, 3]-predicted[:3, 3]))
            start = max(0, min(nearest-1, len(reference)-3))
            window = reference[start:start+3]
            views = [self.view(frames[j]) for j in window]
            submap = ReferenceWindow([np.asarray(frames[j].points) for j in window],
                                     views, [poses[j] for j in window], geom['voxel_m'])
            tr = np.linalg.inv(poses[window[0]]) @ predicted
            passed, _ = score(submap, tr, np.linalg.inv(tr), .25)
            if not passed:
                self.pending = None
            elif all(np.linalg.norm(poses[index][:3, 3]-poses[j][:3, 3]) >= .1
                     for j in [source]+self.support):
                self.support.append(index)
                if len(self.support) == 2:
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
            if np.linalg.norm(initial[:3, 3]) > 1.5 or self.forward_axis @ initial[:3, :3] @ self.forward_axis < .7:
                continue
            if not any(np.linalg.norm(p[:3, 3]-poses[target][:3, 3]) > 1.5
                       for p in poses[target+1:index]):
                continue
            self.attempts += 1
            d = registration_diagnostics(frames[index].points, frames[target].points, initial,
                                        voxel_m=geom['voxel_m'], max_distance_m=limit)
            ok, scores = score(self.view(frames[target]),
                               np.asarray(d['forward']['transform']), np.asarray(d['reverse']['transform']), .30)
            d['score_scope'] = 'visible_optical_depth_bands'
            for direction, s in zip(('forward', 'reverse'), scores):
                d[direction]['forward_depth_score_m'] = s['error_m']
                d[direction]['visible_coverage'] = s['coverage']
            if not ok or not classify_registration(d, gates)['passes']:
                continue
            # Follow this contiguous former visit; exclude the current traversal.
            lo, hi = target, target+1
            def nearby(j):
                return (frames[index].cloud_stamp_ns-frames[j].cloud_stamp_ns >= geom['min_separation_s']*1e9
                        and np.dot(poses[j][:3,:3] @ self.forward_axis, poses[target][:3,:3] @ self.forward_axis) >= .7
                        and np.linalg.norm(poses[j][:3,3]-poses[target][:3,3]) <= 2.)
            while lo and nearby(lo-1):
                lo -= 1
            while hi < index and nearby(hi):
                hi += 1
            if hi-lo >= 3:
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
