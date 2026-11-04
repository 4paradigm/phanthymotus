"""Audit Bumi's historical recording; no clock synchronization is inferred."""
import argparse
import json
import math
from pathlib import Path
import sys

# Direct execution has tools/ on sys.path, not the perception root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.pointcloud_slam_alignment import align_pose
from plugins.pointcloud_slam_odom import decode_bumi_odom


def audit(odometry, clouds, max_gap_ms):
    if (isinstance(max_gap_ms, bool) or not isinstance(max_gap_ms, (int, float))
            or not math.isfinite(max_gap_ms) or max_gap_ms <= 0):
        raise ValueError('max_gap_ms must be positive and finite')
    samples = []
    for item in odometry:
        try:
            samples.append(decode_bumi_odom(item))
        except ValueError:
            samples.append(None)
    aligned = 0
    for cloud in clouds:
        try:
            sec, nsec = cloud['sec'], cloud['nanosec']
            if type(sec) is not int or type(nsec) is not int or sec < 0 or not 0 <= nsec < 1_000_000_000:
                raise ValueError('invalid cloud timestamp')
            align_pose(samples, sec * 1_000_000_000 + nsec, max_gap_ms=max_gap_ms)
            aligned += 1
        except (KeyError, TypeError, ValueError):
            pass
    return {'total_clouds': len(clouds), 'aligned': aligned, 'rejected': len(clouds) - aligned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--max-gap-ms', type=float, required=True,
                        help='experimental limit, not a validated runtime threshold')
    args = parser.parse_args()
    if not math.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0:
        parser.error('max-gap-ms must be positive and finite')
    root = args.recording
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != 'bumi-pointcloud-slam-record/1':
        parser.error('unsupported recording')
    odometry = [json.loads(line) for line in (root / 'odom.jsonl').read_text().splitlines()]
    clouds = [json.loads(line) for line in (root / 'cloud.jsonl').read_text().splitlines()]
    result = audit(odometry, clouds, args.max_gap_ms)
    print(json.dumps({'assumption': 'recorded pose and cloud use a common clock; not proven by this audit',
                      'max_gap_ms': args.max_gap_ms, **result}, indent=2))


if __name__ == '__main__':
    main()
