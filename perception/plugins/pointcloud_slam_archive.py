"""Bounded experimental map archive; restored data never establishes tracking."""
import json
import os
import tempfile
from dataclasses import asdict

from .pointcloud_slam_keyframes import KeyframeStore
from .pointcloud_slam_odom import decode_slam_odom
from .pointcloud_slam_stream import _rigid


def _configuration(settings):
    if not isinstance(settings, dict):
        raise ValueError('archive configuration required')
    result = {key: settings[key] for key in ('cloud_frame', 'child_frame', 'clock_id')}
    if any(not isinstance(v, str) or not v.strip() for v in result.values()):
        raise ValueError('archive configuration required')
    result['child_from_cloud'] = _rigid(settings['child_from_cloud']).tolist()
    result['preview_geometry'] = settings.get('preview_geometry')
    if result['preview_geometry'] is not None and not isinstance(result['preview_geometry'], dict):
        raise ValueError('archive geometry configuration must be an object')
    return result


def save_archive(path, store, links, loops, settings):
    """Atomically replace a map archive, leaving any prior file on failure."""
    config = _configuration(settings)
    frames = []
    for segment in store.segments:
        for frame in segment.frames:
            frames.append({'pose': asdict(frame.pose), 'stamp_ns': frame.cloud_stamp_ns,
                           'points': frame.points.tolist()})
    data = {'schema': 'motus.experimental-slam-archive/1', 'configuration': config,
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


def load_archive(path, settings, *, max_frames, max_points):
    """Load into a new store; validate limits and require a new input segment."""
    store = KeyframeStore(max_frames=max_frames, max_points=max_points)
    limit = 65536 + max_frames * 8192 + max_points * 256
    try:
        with open(path, 'rb') as src:
            raw = src.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('archive too large')
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get('schema') != 'motus.experimental-slam-archive/1':
            raise ValueError('unsupported archive schema')
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
            store.add(sample, entry['points'], cloud_stamp_ns=entry['stamp_ns'])
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
        store.mark_lost()
        return store, links, loops
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid map archive') from error
