"""Read-only experimental localization against a fixed archived segment.

Run from perception with PYTHONPATH=. and pass archive, reference ID and query ID.
Frame numbers in the JSONL trace are one-based within each segment.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from plugins.pointcloud_slam_geometry import classify_registration
from plugins.pointcloud_slam_loop_search import _descriptor
from plugins.pointcloud_slam_map import history_poses, map_from_odom
from plugins.pointcloud_slam_paged_archive import load_paged_archive, MAX_CHECKPOINT_BYTES
from plugins.pointcloud_slam_registration import fixed_overlap, registration_diagnostics
from plugins.pointcloud_slam_relocalization import coarse_alignment
from plugins.pointcloud_slam_stream import _matrix


class ArchiveLocalizer:
    def __init__(self, reference, poses, geometry):
        self.reference = reference
        self.poses = poses
        self.geometry = geometry
        self.voxel = geometry['descriptor_voxel_m']
        self.index = []
        for i, frame in enumerate(reference.frames):
            try:
                descriptor = _descriptor(frame.points, self.voxel)
            except ValueError:
                continue
            self.index.append((i, descriptor))
        self.segment = None
        self.lost()

    def lost(self):
        self.transform = None
        self.seed = None
        self.support = []
        self.support_positions = []
        self.last_stamp = None
        self.blocked_segment = self.segment
        return {'state': 'lost', 'map_pose': None}

    def update(self, number, frame):
        segment = frame.pose.segment_id
        if segment == self.reference.segment_id:
            raise ValueError('query and reference must be distinct')
        if segment != self.segment:
            self.lost()
            self.segment = segment
        trace = {'reference_segment': self.reference.segment_id, 'query_segment': segment,
                 'query_frame': number, 'capture_ns': frame.cloud_stamp_ns,
                 'state': 'searching', 'map_pose': None, 'attempts': []}
        if segment == self.blocked_segment:
            trace['state'] = 'lost'
            return trace
        if self.last_stamp is not None and frame.cloud_stamp_ns <= self.last_stamp:
            raise ValueError('query capture timestamps must increase')
        self.last_stamp = frame.cloud_stamp_ns
        points = np.asarray(frame.points)
        odom = _matrix(frame.pose)
        if self.transform is None:
            try:
                descriptor = _descriptor(points, self.voxel)
            except ValueError:
                return trace
            ranked = sorted(self.index, key=lambda item: np.linalg.norm(item[1] - descriptor))
            proposals = []
            for i, _ in ranked[:3]:
                target = np.asarray(self.reference.frames[i].points)
                initial = coarse_alignment(points, target, voxel_m=self.voxel,
                                           max_distance_m=self.geometry['max_distance_m'])
                row = {'reference_frame': i + 1, 'failed': ['coarse_alignment']}
                trace['attempts'].append(row)
                if initial is None:
                    continue
                diagnostic = registration_diagnostics(points, target, initial,
                    voxel_m=self.geometry['voxel_m'], max_distance_m=self.geometry['max_distance_m'],
                    forward_depth_score='max_forward_depth_score_m' in self.geometry['thresholds'])
                row['diagnostic'] = diagnostic
                row['failed'] = classify_registration(diagnostic, self.geometry['thresholds'])['failed']
                if row['failed']:
                    continue
                aligned = self.poses[i] @ np.asarray(diagnostic['forward']['transform'])
                proposals.append((i, map_from_odom(aligned, odom)))
            if proposals:
                index, best = proposals[0]
                thresholds = self.geometry['thresholds']
                for _, other in proposals[1:]:
                    delta = np.linalg.inv(best) @ other
                    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1)))
                    if (np.linalg.norm(delta[:3, 3]) > thresholds['max_cycle_translation_m'] or
                            angle > thresholds['max_cycle_angle_deg']):
                        trace['reason'] = 'conflicting_reference_poses'
                        return trace
                self.transform = best
                self.seed = [number, index + 1]
                self.support_positions = [odom[:3, 3]]
                trace.update(state='confirming', seed=self.seed)
            return trace

        predicted = self.transform @ odom
        nearest = sorted(range(len(self.poses)),
                         key=lambda i: np.linalg.norm(self.poses[i][:3, 3] - predicted[:3, 3]))
        trace.update(seed=self.seed, state='confirming')
        for i in nearest[:3]:
            relative = np.linalg.inv(self.poses[i]) @ predicted
            scores = fixed_overlap(points, np.asarray(self.reference.frames[i].points),
                                   relative, self.geometry)
            trace['attempts'].append({'reference_frame': i + 1, 'fixed_scores': scores})
            if all(s['passes'] for s in scores):
                # Require two distinct, translated holdouts after the seed.
                position = odom[:3, 3]
                if len(self.support) < 2 and all(
                        np.linalg.norm(position - previous) >= 0.10
                        for previous in self.support_positions):
                    self.support.append([number, i + 1])
                    self.support_positions.append(position)
                trace['support'] = self.support.copy()
                if len(self.support) == 2:
                    trace.update(state='experimental_tracking', map_pose=predicted.tolist(),
                                 first_reference_distance_m=float(np.linalg.norm(
                                     self.poses[0][:3, 3] - predicted[:3, 3])))
                return trace
        self.transform = self.seed = None
        self.support = []
        trace['state'] = 'searching'
        return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('reference')
    parser.add_argument('query')
    args = parser.parse_args()
    import open3d as o3d
    o3d.utility.random.seed(0)
    if args.archive.stat().st_size > MAX_CHECKPOINT_BYTES:
        parser.error('oversized archive')
    settings = json.loads(args.archive.read_bytes())['configuration']
    store, history = load_paged_archive(args.archive, settings, max_frames=100000,
                                        max_points=settings['persistent_history']['max_disk_bytes']//24,
                                        read_only=True)
    segments = {s.segment_id: s for s in store.segments}
    ref, query = segments[args.reference], segments[args.query]
    localizer = ArchiveLocalizer(ref, history_poses(ref, history.get(ref.segment_id)),
                                 settings['preview_geometry'])
    for i, frame in enumerate(query.frames, 1):
        start = time.monotonic()
        trace = localizer.update(i, frame)
        print(json.dumps(dict(trace, elapsed_ms=1000*(time.monotonic()-start)), allow_nan=False),
              flush=True)


if __name__ == '__main__':
    main()
