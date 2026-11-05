"""Evaluate externally labelled recorded pairs; never publish or accept loops."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pointcloud_slam_geometry_gate import classify_registration, summarize
from tools.pointcloud_slam_registration_experiment import registration_diagnostics
from tools.pointcloud_slam_retrieval import _read_cloud, _sha256_file


def benchmark(root, spec):
    root = Path(root)
    if not isinstance(spec, dict) or not isinstance(spec.get('pairs'), list) or not spec['pairs']:
        raise ValueError('explicit labelled pairs are required')
    thresholds = spec.get('thresholds')
    # Validate thresholds before doing any registration work.
    classify_registration({}, thresholds)
    entries = [json.loads(line) for line in (root / 'cloud.jsonl').read_text().splitlines()]
    seen, pairs = set(), []
    for item in spec['pairs']:
        if not isinstance(item, dict):
            raise ValueError('invalid pair')
        source, target, match = item.get('source'), item.get('target'), item.get('match')
        if (type(source) is not int or type(target) is not int or type(match) is not bool or
                not 0 <= source < len(entries) or not 0 <= target < len(entries) or source == target):
            raise ValueError('pairs need distinct valid indices and independent boolean labels')
        key = tuple(sorted((source, target)))
        if key in seen:
            raise ValueError('duplicate pair')
        seen.add(key)
        pairs.append((source, target, match, item.get('initial')))
    results = []
    with (root / 'cloud.xyz').open('rb') as file:
        for source, target, match, initial in pairs:
            diagnostic = registration_diagnostics(
                _read_cloud(file, entries[source]), _read_cloud(file, entries[target]), initial,
                voxel_m=spec.get('voxel_m'), max_distance_m=spec.get('max_distance_m'))
            gate = classify_registration(diagnostic, thresholds)
            results.append({'source': source, 'target': target, 'match': match,
                            'passes': gate['passes'], 'failed': gate['failed'],
                            'diagnostic': diagnostic, 'loop_accepted': False})
    return {'status': 'offline threshold experiment; labels and thresholds not validated',
            'summary': summarize(results), 'pairs': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('spec', type=Path, help='JSON labels, initial transforms and explicit experimental thresholds')
    args = parser.parse_args()
    raw = args.spec.read_bytes()
    spec = json.loads(raw)
    result = benchmark(args.recording, spec)
    import open3d
    tools_dir = Path(__file__).resolve().parent
    result['provenance'] = {'spec_sha256': hashlib.sha256(raw).hexdigest(),
                            'cloud_index_sha256': _sha256_file(args.recording / 'cloud.jsonl'),
                            'cloud_data_sha256': _sha256_file(args.recording / 'cloud.xyz'),
                            'script_sha256': _sha256_file(Path(__file__)),
                            'gate_sha256': _sha256_file(tools_dir / 'pointcloud_slam_geometry_gate.py'),
                            'gate_impl_sha256': _sha256_file(tools_dir.parent / 'plugins/pointcloud_slam_geometry.py'),
                            'registration_sha256': _sha256_file(tools_dir / 'pointcloud_slam_registration_experiment.py'),
                            'registration_impl_sha256': _sha256_file(tools_dir.parent / 'plugins/pointcloud_slam_registration.py'),
                            'reader_sha256': _sha256_file(tools_dir / 'pointcloud_slam_retrieval.py'),
                            'open3d_version': open3d.__version__}
    result['parameters'] = {key: spec[key] for key in ('voxel_m', 'max_distance_m', 'thresholds')}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
