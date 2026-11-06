"""Rescore an archived all-search trace. Only the stratified error gate is replaced.

Uses the original retrieved pairs and reciprocal fits. Tests subsequent query frames
at a fixed transform. Camera is width,height,fx,fy,cx,cy of the optical cloud.
"""
from functools import lru_cache
import argparse
import json
from pathlib import Path

import numpy as np

from plugins.pointcloud_slam_map import history_poses
from plugins.pointcloud_slam_paged_archive import load_paged_archive
from plugins.pointcloud_slam_stream import _matrix
from tools.pointcloud_slam_visible_overlap import ReferenceWindow, VisibilityView, split_decision


def replay(archive, trace, camera, coverage, window_coverage=None):
    cfg = json.loads(archive.read_bytes())['configuration']
    store, history = load_paged_archive(archive, cfg, max_frames=100000,
        max_points=cfg['persistent_history']['max_disk_bytes']//24, read_only=True)
    ref, query = store.segments
    poses = history_poses(ref, history.get(ref.segment_id))
    geom = cfg['preview_geometry']
    limits = geom['thresholds']
    optical = np.linalg.inv(cfg['child_from_cloud'])
    def view(f):
        return VisibilityView(np.asarray(f.points), camera, optical, geom['voxel_m'])
    refs = [view(f) for f in ref.frames]
    @lru_cache(maxsize=1)
    def reference_window(start):
        frames = ref.frames[start:start+3]
        return ReferenceWindow([np.asarray(frame.points) for frame in frames],
                               refs[start:start+3], poses[start:start+3], geom['voxel_m'])

    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if len(rows) != len(query.frames):
        raise ValueError('incomplete baseline trace')
    transform = None
    for n, (f, row) in enumerate(zip(query.frames, rows), 1):
        if (row['query_frame'] != n or row['capture_ns'] != f.cloud_stamp_ns or
                row['reference_segment'] != ref.segment_id or row['query_segment'] != query.segment_id
                or row['state'] != 'searching'):
            raise ValueError('expected matching all-search baseline trace')
        q = view(f)
        odom = _matrix(f.pose)
        report = dict(query=n, capture_ns=f.cloud_stamp_ns, attempts=[], state='searching')
        def check(j, target, forward, reverse, failed, minimum):
            scores = [q.score(target, forward, geom['max_distance_m']),
                      target.score(q, reverse, geom['max_distance_m'])]
            gates = [split_decision(s, minimum, limits['max_forward_depth_score_m'],
                                    limits['min_correspondences']) for s in scores]
            for direction, score in zip(('forward', 'reverse'), scores):
                if score['fitness'] < limits['min_fitness'] and direction+'_fitness' not in failed:
                    failed.append(direction+'_fitness')
                if score['correspondences'] < limits['min_correspondences'] and direction+'_correspondences' not in failed:
                    failed.append(direction+'_correspondences')
            report['attempts'].append(dict(reference=j+1, scores=scores, gates=gates, other_failed=failed))
            return not failed and all(all(g.values()) for g in gates)
        if transform is None:
            proposals = []
            for a in row['attempts']:
                j = a['reference_frame']-1
                if 'diagnostic' not in a:
                    report['attempts'].append(a)
                    continue
                d = a['diagnostic']
                forward = np.asarray(d['forward']['transform'])
                failed = [k for k in a['failed'] if k not in ('forward_depth_score','reverse_depth_score')]
                if check(j, refs[j], forward, np.asarray(d['reverse']['transform']), failed, coverage):
                    proposals.append((j+1, poses[j] @ forward @ np.linalg.inv(odom)))
            if proposals:
                index, candidate = proposals[0]
                consistent = True
                for _, other in proposals[1:]:
                    delta = np.linalg.inv(candidate) @ other
                    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
                    consistent &= (np.linalg.norm(delta[:3,3]) <= limits['max_cycle_translation_m']
                                   and angle <= limits['max_cycle_angle_deg'])
                if consistent:
                    transform = candidate
                    seed = [n, index]
                    positions = [odom[:3,3]]
                    support = []
                    report.update(state='confirming', seed=seed)
                else:
                    report['reason'] = 'conflicting_reference_poses'
        else:
            pose = transform @ odom
            ranked = sorted(range(len(refs)), key=lambda j: np.linalg.norm(poses[j][:3,3]-pose[:3,3]))
            report.update(seed=seed)
            targets = [(j, refs[j], coverage) for j in ranked[:3]]
            if window_coverage is not None and len(refs) >= 3:
                start = max(0, min(ranked[0]-1, len(refs)-3))
                targets = [(start, reference_window(start), window_coverage)]
                report['reference_window'] = list(range(start+1, start+4))
            for j, target, minimum in targets:
                t = np.linalg.inv(poses[j]) @ pose
                if check(j, target, t, np.linalg.inv(t), [], minimum):
                    if len(support)<2 and all(np.linalg.norm(odom[:3,3]-p)>=.10 for p in positions):
                        positions.append(odom[:3,3])
                        support.append([n,j+1])
                    report.update(state='confirming', support=support.copy())
                    if len(support)==2:
                        report.update(state='experimental_tracking', map_pose=pose.tolist())
                    break
            else:
                transform = None
        yield report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--camera', required=True, nargs=6, type=float)
    parser.add_argument('--coverage', required=True, type=float)
    parser.add_argument('--window-coverage', type=float,
                        help='enable three-frame sliding reference for post-seed checks')
    args = parser.parse_args()
    if args.window_coverage is not None and not 0 < args.window_coverage <= 1:
        parser.error('invalid window coverage')
    w,h,*intrinsics=args.camera
    if not (0 < args.coverage <= 1 and w.is_integer() and h.is_integer() and
            min(w,h,*intrinsics[:2])>0 and np.all(np.isfinite(args.camera))):
        parser.error('invalid coverage or camera')
    for item in replay(args.archive,args.trace,(int(w),int(h),*intrinsics),args.coverage,args.window_coverage):
        print(json.dumps(item, allow_nan=False))
