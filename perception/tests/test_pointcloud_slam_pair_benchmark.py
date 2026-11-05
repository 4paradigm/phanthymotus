"""Run genuine ICP over synthetic labelled pairs; labels remain external inputs."""
import json
import numpy as np
import pytest

pytest.importorskip('open3d')
from tools.pointcloud_slam_pair_benchmark import benchmark


def recording(tmp_path):
    base = np.random.default_rng(4).uniform(-1, 1, size=(200, 3)).astype('<f4')
    clouds = [base, base + [.03, -.02, .01], base + [10, 0, 0]]
    entries, data = [], bytearray()
    for cloud in clouds:
        raw = np.asarray(cloud, dtype='<f4').tobytes()
        entries.append({'offset': len(data), 'bytes': len(raw)})
        data.extend(raw)
    (tmp_path / 'cloud.xyz').write_bytes(data)
    (tmp_path / 'cloud.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    return tmp_path


def spec():
    return {'voxel_m': .04, 'max_distance_m': .15,
            'thresholds': {'min_fitness': .7, 'max_rmse_m': .05,
                           'min_correspondences': 30, 'max_cycle_translation_m': .03,
                           'max_cycle_angle_deg': 2.},
            'pairs': [{'source': 0, 'target': 1, 'match': True, 'initial': np.eye(4).tolist()},
                      {'source': 0, 'target': 2, 'match': False, 'initial': np.eye(4).tolist()}]}


def test_benchmark_uses_real_registration_and_never_accepts_loop(tmp_path):
    out = benchmark(recording(tmp_path), spec())
    assert out['summary']['tp'] == 1 and out['summary']['tn'] == 1
    assert all(item['loop_accepted'] is False for item in out['pairs'])


def test_rejects_duplicate_or_self_pairs_instead_of_skewing_metrics(tmp_path):
    root = recording(tmp_path)
    data = spec()
    data['pairs'].append({'source': 1, 'target': 0, 'match': False, 'initial': np.eye(4).tolist()})
    with pytest.raises(ValueError):
        benchmark(root, data)
    data = spec()
    data['pairs'][0]['target'] = 0
    with pytest.raises(ValueError):
        benchmark(root, data)


def test_cli_saves_reproducible_spec_and_tool_provenance(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = recording(tmp_path)
    path = root / 'synthetic-spec.json'
    path.write_text(json.dumps(spec()))
    tool = Path(__file__).resolve().parents[1] / 'tools' / 'pointcloud_slam_pair_benchmark.py'
    output = subprocess.check_output([sys.executable, str(tool), str(root), str(path)],
                                     env={**os.environ, 'OMP_NUM_THREADS': '1'}, text=True)
    report = json.loads(output)
    assert report['summary']['tp'] == 1 and report['summary']['tn'] == 1
    assert len(report['provenance']['spec_sha256']) == 64
    assert report['provenance']['open3d_version']
