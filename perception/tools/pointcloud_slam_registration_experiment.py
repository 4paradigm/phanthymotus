"""Compute reciprocal ICP diagnostics only; this tool never accepts a loop."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

# Allow direct execution without installing the perception package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pointcloud_slam_retrieval import _read_cloud
from tools.pointcloud_slam_submap import _rigid


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('source', type=int)
    parser.add_argument('target', type=int)
    parser.add_argument('--initial', type=Path, required=True,
                        help='JSON from previous registration with transform_source_to_target')
    parser.add_argument('--voxel-m', type=float, required=True)
    parser.add_argument('--max-distance-m', type=float, required=True)
    args = parser.parse_args()
    entries = [json.loads(line) for line in (args.recording / 'cloud.jsonl').read_text().splitlines()]
    if not (0 <= args.source < len(entries) and 0 <= args.target < len(entries) and args.source != args.target):
        parser.error('source and target must be different valid frame indices')
    initial_data = json.loads(args.initial.read_text())
    if (initial_data.get('source_index'), initial_data.get('target_index')) != (args.source, args.target):
        parser.error('initial transform direction or frame indices do not match')
    with (args.recording / 'cloud.xyz').open('rb') as file:
        source = _read_cloud(file, entries[args.source])
        target = _read_cloud(file, entries[args.target])
    out = registration_diagnostics(source, target, initial_data['transform_source_to_target'],
                                   voxel_m=args.voxel_m, max_distance_m=args.max_distance_m)
    out.update({'source_index': args.source, 'target_index': args.target,
                'initial_transform_source': str(args.initial), 'voxel_m': args.voxel_m,
                'max_distance_m': args.max_distance_m,
                'status': 'exploratory only; initial transform is not independent evidence'})
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
