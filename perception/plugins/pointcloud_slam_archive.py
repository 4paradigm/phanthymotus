"""Bounded experimental map archive; restored data never establishes tracking."""
import json
import os
import tempfile
from dataclasses import asdict
from copy import deepcopy

from .pointcloud_slam_keyframes import KeyframeStore
from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_stream import _rigid


def _configuration(settings):
    if not isinstance(settings, dict):
        raise ValueError('archive configuration required')
    result = {key: settings[key] for key in ('cloud_frame', 'child_frame', 'clock_id')}
    if any(not isinstance(v, str) or not v.strip() for v in result.values()):
        raise ValueError('archive configuration required')
    mode = settings.get('extrinsic_mode', 'static')
    if mode == 'dynamic':
        if settings.get('child_from_cloud') is not None:
            raise ValueError('dynamic archive cannot use a startup transform')
        result.update(extrinsic_mode='dynamic', child_from_cloud=None)
    elif mode == 'static':
        result['child_from_cloud'] = _rigid(settings['child_from_cloud']).tolist()
    else:
        raise ValueError('unsupported archive extrinsic mode')
    result['preview_geometry'] = settings.get('preview_geometry')
    if settings.get('input_policy') == 'experimental':
        result.update(input_policy='experimental',
                      clock_relation_verified=settings.get('clock_relation_verified') is True,
                      extrinsic_calibrated=settings.get('extrinsic_calibrated') is True)
    if result['preview_geometry'] is not None and not isinstance(result['preview_geometry'], dict):
        raise ValueError('archive geometry configuration must be an object')
    return result



_RESOLVED_FIELDS = ('resolved_inputs', 'resolved_observation', 'resolved_scoring',
                    'algorithm_profile', 'resource_profile', 'compatibility', 'map_plane')


def archive_metadata(settings, *, segment_ids=(), inherited=None):
    """Snapshot actual resolved facts; a legacy segment never acquires new proof."""
    result = {key: deepcopy(settings.get(key)) for key in _RESOLVED_FIELDS}
    if result['map_plane'] not in (None, 'xy'):
        raise ValueError('unsupported map annotation plane')
    result.update(schema='motus.slam_archive_metadata/1', evidence={
        'source': 'resolved_inputs' if result['resolved_inputs'] is not None else 'explicit_configuration',
        'policy': settings.get('input_policy', 'strict'),
        'clock_relation_verified': settings.get('clock_relation_verified'),
        'extrinsic_calibrated': settings.get('extrinsic_calibrated')})
    segments = deepcopy(settings.get('segment_metadata') or
                        (inherited or {}).get('segment_metadata') or {})
    for sid in segment_ids:
        segments.setdefault(sid, deepcopy(result))
    result['segment_metadata'] = segments
    return result


def archived_metadata(data):
    """Read evidence from the archive only, keeping old unknowns explicit."""
    if data.get('schema', '').endswith('/2'):
        result = data.get('resolved')
        if not isinstance(result, dict) or result.get('schema') != 'motus.slam_archive_metadata/1':
            raise ValueError('archive resolved metadata required')
        return deepcopy(result)
    result = archive_metadata(data['configuration'])
    result['evidence']['source'] = 'legacy_archive'
    result['segment_metadata'] = {
        entry['pose']['segment_id']: deepcopy(result) for entry in data['frames']}
    return result


def save_archive(path, store, links, loops, settings):
    """Atomically replace a map archive, leaving any prior file on failure."""
    config = _configuration(settings)
    frames = []
    for segment in store.segments:
        for frame in segment.frames:
            entry = {'pose': asdict(frame.pose), 'stamp_ns': frame.cloud_stamp_ns,
                     'points': frame.points.tolist()}
            if frame.capture_context is not None:
                entry['capture_context'] = deepcopy(frame.capture_context)
            frames.append(entry)
    data = {'schema': 'motus.experimental-slam-archive/2', 'configuration': config,
            'resolved': archive_metadata(settings, segment_ids=[s.segment_id for s in store.segments],
                                          inherited=getattr(store, 'archive_metadata', None)),
            'frames': frames, 'links': links, 'loops': loops}
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp = tempfile.mkstemp(prefix='.slam-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(data, out, allow_nan=False, separators=(',', ':'))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_archive(path, settings, *, max_frames, max_points, read_only=False):
    """Load into a new store; validate limits and require a new input segment."""
    store = KeyframeStore(max_frames=max_frames, max_points=max_points)
    limit = 65536 + max_frames * 8192 + max_points * 256
    try:
        with open(path, 'rb') as src:
            raw = src.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('archive too large')
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get('schema') not in ('motus.experimental-slam-archive/1', 'motus.experimental-slam-archive/2'):
            raise ValueError('unsupported archive schema')
        if read_only:
            settings = data['configuration']
        if data.get('configuration') != _configuration(settings):
            raise ValueError('archive configuration mismatch')
        frames, links, loops = data['frames'], data['links'], data['loops']
        if not isinstance(frames, list) or len(frames) > max_frames or not isinstance(links, list) or not isinstance(loops, dict):
            raise ValueError('invalid archive content')
        for entry in frames:
            p = entry['pose']
            sample = decode_slam_odom({'schema': 'motus.odom/1', 'units': {'length': 'm'},
                'stamp_ms': p['stamp_ms'], 'pose': {'frame_id': p['frame_id'],
                'child_frame_id': p['child_frame_id'], 'position': p['position'],
                'quaternion': p['quaternion'], 'covariance': p['covariance']},
                'vendor': {'motus_slam': {'status': 'tracking', 'stamp_source': 'measurement',
                    'segment_id': p['segment_id'], 'clock_id': p['clock_id']}}})
            if sample.clock_id != settings['clock_id']:
                raise ValueError('archive clock mismatch')
            if sample.child_frame_id != settings['child_frame'].replace('{segment_id}', sample.segment_id):
                raise ValueError('archive child frame mismatch')
            store.add(sample, entry['points'], cloud_stamp_ns=entry['stamp_ns'],
                      capture_context=entry.get('capture_context'))
        ids = {s.segment_id: len(s.frames) for s in store.segments}
        connected = {store.segments[0].segment_id} if store.segments else set()
        if len(links) > max_frames or len(loops) > max_frames:
            raise ValueError('archive constraint capacity exceeded')
        for link in links:
            a, b = link['target'], link['source']
            i, j = link['target_index'], link['source_index']
            if (a not in connected or b not in ids or b in connected or type(i) is not int or
                    type(j) is not int or not 0 <= i < ids[a] or not 0 <= j < ids[b]):
                raise ValueError('invalid archive link')
            _rigid(link['transform']); connected.add(b)
        count = 0
        for segment, entries in loops.items():
            if segment not in ids or not isinstance(entries, list):
                raise ValueError('invalid archive loops')
            count += len(entries)
            if count > max_frames * max_frames:
                raise ValueError('archive constraint capacity exceeded')
            for loop in entries:
                a, b = loop['source'], loop['target']
                if (type(a) is not int or type(b) is not int or a == b or
                        not 0 <= a < ids[segment] or not 0 <= b < ids[segment]):
                    raise ValueError('invalid archive loop')
                _rigid(loop['transform'])
        store.archive_metadata = archived_metadata(data)
        store.mark_lost()
        return store, links, loops
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid map archive') from error
