"""Old paged maps remain viewable without loading the retired SLAM runtime."""
import hashlib
import json
import struct

import numpy as np
import pytest

from plugins import pointcloud_slam_library as library


def legacy_map(tmp_path, version=2):
    directory = tmp_path / 'maps' / ('a' * 32)
    points_dir = directory / 'sessions' / 'old_session'
    points_dir.mkdir(parents=True)
    frames = []
    for index, (segment, x) in enumerate((('a', 0), ('a', 1), ('b', 20))):
        file = points_dir / ('%08d.npy' % index)
        np.save(file, np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64))
        frames.append({'pose': {'stamp_ms': 1000 + index * 1000, 'frame_id': segment + '_odom',
            'child_frame_id': 'base', 'segment_id': segment, 'position': [x, 0, 0],
            'quaternion': [0, 0, 0, 1], 'covariance': None, 'clock_id': 'old_clock'},
            'stamp_ns': (1000 + index * 1000) * 1000000, 'session': 'old_session',
            'file': file.name, 'count': 3, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    optimized = np.eye(4)
    optimized[0, 3] = 10
    data = {'schema': 'motus.experimental-slam-paged/' + str(version),
            'configuration': {'clock_id': 'old_clock', 'cloud_frame': 'camera', 'child_frame': 'base',
                'child_from_cloud': np.eye(4).tolist(),
                'persistent_history': {'max_disk_bytes': 1000000, 'reserve_bytes': 0}},
            'frames': frames, 'history': {'a': {'poses': [optimized.tolist()],
                                               'stamps': [1000000000]}}}
    if version == 2:
        data['resolved'] = {'schema': 'motus.slam_archive_metadata/1', 'map_plane': None,
            'evidence': {'source': 'resolved_inputs'},
            'segment_metadata': {'a': {'map_plane': 'xy', 'evidence': {'source': 'resolved_inputs'}},
                                 'b': {'map_plane': None, 'evidence': {'source': 'unknown'}}}}
    (directory / '1.json').write_text(json.dumps(data))
    (directory / 'manifest.json').write_text(json.dumps({'id': 'a' * 32, 'name': '旧地图',
        'revision': 1, 'saves': [{'revision': 1, 'request_id': 'old', 'source': 'old_session'}]}))
    return directory, data


def test_legacy_view_uses_saved_optimized_prefix_and_odometry_tail_without_writes(tmp_path):
    directory, data = legacy_map(tmp_path)
    before = {p.relative_to(directory): (p.stat().st_mtime_ns, p.read_bytes())
              for p in directory.rglob('*') if p.is_file()}
    snapshots = library.load_map_snapshots(tmp_path, 'a' * 32, max_points=4)
    assert [s['segment_id'] for s in snapshots] == ['a', 'b']
    assert snapshots[0]['frame_id'] == 'a_odom_archive'
    assert struct.unpack('<fffBI', snapshots[0]['packet'][:17]) == (11., 0., 0., 39, 4)
    xyz = np.frombuffer(snapshots[0]['packet'][17:], dtype='<f4').reshape(-1, 3)
    assert xyz.tolist() == [[10., 0., 0.], [11., 0., 0.], [11., 0., 0.], [13., 0., 0.]]
    assert snapshots[0]['metadata']['tracking'] is False
    assert snapshots[0]['metadata']['map_plane'] == 'xy'
    assert snapshots[1]['metadata']['map_plane'] is None
    after = {p.relative_to(directory): (p.stat().st_mtime_ns, p.read_bytes())
             for p in directory.rglob('*') if p.is_file()}
    assert after == before


def test_legacy_v1_retains_unknown_evidence_and_has_no_annotation_plane(tmp_path):
    legacy_map(tmp_path, version=1)
    snapshots = library.load_map_snapshots(tmp_path, 'a' * 32)
    assert snapshots[0]['metadata']['evidence']['source'] == 'legacy_archive'
    assert snapshots[0]['metadata']['map_plane'] is None


@pytest.mark.parametrize('invalid', ['checksum', 'path', 'symlink', 'stamps', 'pose', 'dtype'])
def test_legacy_view_rejects_broken_archive_evidence(tmp_path, invalid):
    directory, data = legacy_map(tmp_path)
    record = data['frames'][0]
    file = directory / 'sessions' / record['session'] / record['file']
    if invalid == 'checksum':
        record['sha256'] = '0' * 64
    elif invalid == 'path':
        record['session'] = '../outside'
    elif invalid == 'symlink':
        external = tmp_path / 'outside.npy'
        external.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(external)
    elif invalid == 'stamps':
        data['history']['a']['stamps'] = [999]
    elif invalid == 'pose':
        data['history']['a']['poses'][0][0][0] = 2
    elif invalid == 'dtype':
        np.save(file, np.zeros((3, 3), dtype=np.float32))
        record['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    (directory / '1.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        library.load_map_snapshots(tmp_path, 'a' * 32)


def test_legacy_points_are_sampled_before_materializing_full_file(tmp_path, monkeypatch):
    legacy_map(tmp_path)
    original = np.load
    def mapped(path, **kwargs):
        assert kwargs == {'mmap_mode': 'r', 'allow_pickle': False}
        return original(path, **kwargs)
    monkeypatch.setattr(np, 'load', mapped)
    snapshots = library.load_map_snapshots(tmp_path, 'a' * 32, max_points=1)
    assert all(len(s['packet']) == 29 for s in snapshots)


def test_legacy_disk_budget_applies_across_independent_segments(tmp_path):
    directory, data = legacy_map(tmp_path)
    files = sorted((directory / 'sessions' / 'old_session').glob('*.npy'))
    data['configuration']['persistent_history']['max_disk_bytes'] = 2 * files[0].stat().st_size
    (directory / '1.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='disk budget'):
        library.load_map_snapshots(tmp_path, 'a' * 32)


@pytest.mark.parametrize('invalid', ['extrinsic', 'reused_segment', 'schema'])
def test_legacy_archive_identity_and_configuration_are_still_validated(tmp_path, invalid):
    directory, data = legacy_map(tmp_path)
    if invalid == 'extrinsic':
        data['configuration']['child_from_cloud'][0][0] = 2
    elif invalid == 'reused_segment':
        data['frames'] = [data['frames'][0], data['frames'][2], data['frames'][1]]
    else:
        data['schema'] = 'motus.experimental-slam-paged/999'
    (directory / '1.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        library.load_map_snapshots(tmp_path, 'a' * 32)
