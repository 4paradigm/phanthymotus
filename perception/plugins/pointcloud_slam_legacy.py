"""Read-only, bounded reconstruction of retired paged JSON + NPY maps."""
from copy import deepcopy
import hashlib
import re

import numpy as np

from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_preview import encode_mapping
from .pointcloud_slam_transforms import rigid, pose_matrix


SCHEMAS = ('motus.experimental-slam-paged/1', 'motus.experimental-slam-paged/2')


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_metadata(data):
    if data.get('schema') not in SCHEMAS:
        raise ValueError('unsupported paged map archive')
    if data['schema'].endswith('/2'):
        metadata = data.get('resolved')
        if not isinstance(metadata, dict) or metadata.get('schema') != 'motus.slam_archive_metadata/1':
            raise ValueError('archive resolved metadata required')
        return deepcopy(metadata)
    return {'schema': 'motus.slam_archive_metadata/1', 'map_plane': None,
            'evidence': {'source': 'legacy_archive'}, 'segment_metadata': {}}


def _pose(record, configuration):
    pose = record['pose']
    sample = decode_slam_odom({'schema': 'motus.odom/1', 'units': {'length': 'm'},
        'stamp_ms': pose['stamp_ms'], 'pose': {k: pose[k] for k in (
            'frame_id', 'child_frame_id', 'position', 'quaternion', 'covariance')},
        'vendor': {'motus_slam': {'status': 'tracking', 'stamp_source': 'measurement',
            'segment_id': pose['segment_id'], 'clock_id': pose['clock_id']}}})
    if (sample.clock_id != configuration['clock_id'] or sample.child_frame_id !=
            configuration['child_frame'].replace('{segment_id}', sample.segment_id) or
            type(record['stamp_ns']) is not int or record['stamp_ns'] <= 0 or
            record['stamp_ns'] // 1000000 != sample.stamp_ms):
        raise ValueError('archived pose frame, clock or capture stamp mismatch')
    return sample


def _points_path(root, record):
    if (not isinstance(record['session'], str) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', record['session']) or
            not isinstance(record['file'], str) or not re.fullmatch(r'[0-9]{8}\.npy', record['file']) or
            type(record['count']) is not int or record['count'] <= 0):
        raise ValueError('invalid point file reference')
    path = root / record['session'] / record['file']
    if path.resolve().parent.parent != root.resolve() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError('invalid point file location')
    if path.stat().st_size > record['count'] * 24 + 4096:
        raise ValueError('point file exceeds declared size')
    if file_digest(path) != record['sha256']:
        raise ValueError('point file checksum mismatch')
    return path


def load_legacy_snapshots(path, data, *, max_points=40000):
    """Sample immutable child-frame points; never instantiate a solver or writer."""
    if type(max_points) is not int or not 0 < max_points <= 80000:
        raise ValueError('preview point limit must be between 1 and 80000')
    try:
        archived = legacy_metadata(data)
        configuration, frames, history = data['configuration'], data['frames'], data['history']
        if not isinstance(frames, list) or not isinstance(history, dict):
            raise ValueError('invalid paged checkpoint')
        if any(not isinstance(configuration.get(key), str) or not configuration[key].strip()
               for key in ('cloud_frame', 'child_frame', 'clock_id')):
            raise ValueError('archive configuration required')
        mode = configuration.get('extrinsic_mode', 'static')
        if mode == 'static':
            rigid(configuration['child_from_cloud'])
        elif mode != 'dynamic' or configuration.get('child_from_cloud') is not None:
            raise ValueError('invalid archived extrinsic mode')
        segments = {}
        current = None
        for record in frames:
            pose = _pose(record, configuration)
            if pose.segment_id != current and pose.segment_id in segments:
                raise ValueError('archived segment cannot be reused')
            current = pose.segment_id
            entries = segments.setdefault(pose.segment_id, [])
            if entries and (pose.stamp_ms <= entries[-1][1].stamp_ms or
                    (pose.frame_id, pose.child_frame_id) !=
                    (entries[-1][1].frame_id, entries[-1][1].child_frame_id)):
                raise ValueError('inconsistent archived segment')
            entries.append((record, pose))
        if not set(history) <= segments.keys():
            raise ValueError('unknown archived history segment')
        result = []
        disk_bytes = 0
        for segment_id, entries in segments.items():
            snapshot = history.get(segment_id)
            poses = [pose_matrix(pose) for _, pose in entries]
            optimized = 0
            if snapshot:
                saved, stamps = snapshot['poses'], snapshot['stamps']
                optimized = len(saved)
                if (not saved or optimized > len(entries) or
                        stamps != [record['stamp_ns'] for record, _ in entries[:optimized]]):
                    raise ValueError('history capture stamps mismatch')
                saved = [rigid(pose) for pose in saved]
                correction = saved[-1] @ np.linalg.inv(poses[optimized - 1])
                poses = saved + [correction @ pose for pose in poses[optimized:]]
            total = sum(record['count'] for record, _ in entries)
            indices = np.linspace(0, total - 1, min(max_points, total), dtype=np.int64)
            chunks, offset = [], 0
            budget = configuration['persistent_history']['max_disk_bytes']
            for (record, _), matrix in zip(entries, poses):
                file = _points_path(path.parent / 'sessions', record)
                disk_bytes += file.stat().st_size
                if disk_bytes > budget:
                    raise ValueError('point files exceed archived disk budget')
                points = np.load(file, mmap_mode='r', allow_pickle=False)
                try:
                    if points.shape != (record['count'], 3) or points.dtype != np.float64:
                        raise ValueError('point file shape/type mismatch')
                    for start in range(0, len(points), 65536):
                        if not np.isfinite(points[start:start + 65536]).all():
                            raise ValueError('invalid archived points')
                    selected = indices[(indices >= offset) & (indices < offset + len(points))] - offset
                    sampled = np.array(points[selected], copy=True)
                finally:
                    points._mmap.close()
                chunks.append(sampled @ matrix[:3, :3].T + matrix[:3, 3])
                offset += record['count']
            evidence = deepcopy((archived.get('segment_metadata') or {}).get(segment_id, archived))
            metadata = {**evidence, 'archive_metadata': archived, 'backend': 'legacy', 'tracking': False,
                'map_plane': evidence.get('map_plane') if evidence.get('map_plane') == 'xy' else None,
                'stamp_ns': str(entries[-1][0]['stamp_ns']), 'captured_frames': len(entries),
                'optimized_frames': optimized, 'input_validation': evidence.get('evidence')}
            result.append({'segment_id': segment_id, 'frame_id': entries[0][1].frame_id + '_archive',
                'packet': encode_mapping(np.concatenate(chunks), poses[-1], max_points=max_points),
                'metadata': metadata})
        return result
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('invalid paged map archive') from error
