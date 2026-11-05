"""Compute reciprocal ICP diagnostics only; this tool never accepts a loop."""
import argparse
import json
from pathlib import Path
import sys


# Allow direct execution without installing the perception package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pointcloud_slam_retrieval import _read_cloud


from plugins.pointcloud_slam_registration import registration_diagnostics


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
