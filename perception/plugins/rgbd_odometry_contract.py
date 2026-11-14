"""Robot-independent calibrated RGB-D input and SLAM pose contracts."""
from collections import deque
from copy import deepcopy

import numpy as np

from .pointcloud_slam_contract import validate_tracking_pose


def validate_inputs(declarations, topics):
    if not isinstance(declarations, dict) or len(topics or []) != 2 or len(set(topics)) != 2:
        raise ValueError('connect one calibrated color and one measured depth input')
    entries = [declarations.get(topic, {}) for topic in topics]
    if {d.get('format') for d in entries} != {'image/jpeg', 'image/depth-zlib'}:
        raise ValueError('RGB-D requires image/jpeg and image/depth-zlib declarations')
    color = next(d for d in entries if d['format'] == 'image/jpeg')
    depth = next(d for d in entries if d['format'] == 'image/depth-zlib')
    for d in entries:
        if (d.get('schema') != 'motus.camera/1' or not d.get('id') or
                d.get('width') != 640 or d.get('height') != 480 or
                d.get('distortion_model') != 'pinhole'):
            raise ValueError('RGB-D requires declared rectified 640x480 camera geometry')
        k = np.asarray(d.get('K'), dtype=float)
        distortion = np.asarray(d.get('D'), dtype=float)
        if (k.size != 9 or not np.isfinite(k).all() or k.flat[0] <= 0 or k.flat[4] <= 0 or
                not np.allclose(k.reshape(3, 3)[2], [0, 0, 1]) or
                distortion.size == 0 or not np.isfinite(distortion).all() or np.any(distortion != 0)):
            raise ValueError('RGB-D calibration must be a finite pinhole with zero distortion')
        vendor = d.get('vendor') or {}
        if (vendor.get('registration') != 'registered_rectified' or
                vendor.get('timestamp_semantics') != 'source_capture_mapped' or
                not vendor.get('frame_id') or not vendor.get('clock_id')):
            raise ValueError('RGB-D registration, optical frame and capture clock are required')
    if (color['id'] != depth['id'] or color['K'] != depth['K'] or
            any(color['vendor'].get(k) != depth['vendor'].get(k)
                for k in ('frame_id', 'clock_id'))):
        raise ValueError('RGB-D inputs must share registered geometry and capture clock')
    if (depth['vendor'].get('depth_source') != 'measured' or
            depth['vendor'].get('depth_unit') != 'mm'):
        raise ValueError('RGB-D requires measured millimetre depth')
    mount = color['vendor'].get('body_mount')
    if mount != depth['vendor'].get('body_mount'):
        raise ValueError('RGB-D body_mount declarations conflict')
    if mount is not None:
        from .rgbd_body_mount import validate_mount_declaration
        mount = validate_mount_declaration(mount, color['vendor']['frame_id'], topics)
    result = dict(color_topic=color['topic'], depth_topic=depth['topic'],
                K=np.asarray(color['K'], dtype=float).reshape(3, 3),
                frame_id=color['vendor']['frame_id'], clock_id=color['vendor']['clock_id'])
    if mount is not None:
        result['body_mount'] = mount
    return result


class PairBuffer:
    """Match source times with bounded memory; never substitute receipt times."""
    def __init__(self, max_pending=5, max_skew_ns=20_000_000):
        self.queues = {kind: deque(maxlen=max_pending) for kind in ('color', 'depth')}
        self.previous = {}
        self.max_skew_ns = max_skew_ns

    @property
    def pending_count(self):
        return sum(len(q) for q in self.queues.values())

    def push(self, kind, stamp_ns, payload, now_ns):
        if (type(stamp_ns) is not int or stamp_ns <= self.previous.get(kind, 0) or
                not -50_000_000 <= now_ns - stamp_ns <= 1_000_000_000):
            raise ValueError('stale, repeated, future or reset RGB-D capture time')
        self.previous[kind] = stamp_ns
        self.queues[kind].append((stamp_ns, payload))
        color, depth = self.queues['color'], self.queues['depth']
        while color and depth:
            delta = color[0][0] - depth[0][0]
            if abs(delta) <= self.max_skew_ns:
                return color.popleft(), depth.popleft()
            (color if delta < 0 else depth).popleft()
        return None


def make_sample(status, stamp_ns, segment, clock, pose=None, *, reason=None):
    valid = status == 'tracking'
    sample = {'schema': 'motus.odom/1', 'stamp_ms': stamp_ns // 1_000_000,
              'units': {'length': 'm', 'linear': 'm/s', 'angular': 'rad/s'},
              'frame': 'body', 'twist': [None] * 6,
              'pose': deepcopy(pose) if valid else None,
              'vendor': {'source': 'rtabmap_rgbd_odometry', 'pose_reference': 'camera_optical',
                         'accuracy_validated': False, 'reason': reason,
                         'motus_slam': {'status': status, 'stamp_source': 'measurement' if valid else 'unknown',
                                        'segment_id': segment, 'clock_id': clock, 'stamp_ns': stamp_ns}}}
    if valid:
        validate_tracking_pose(sample)
    return sample


def start_topics(args):
    """Core sends a lone resolved input only as singular `input_topic`."""
    topics = args.get('input_topics')
    if topics is None and isinstance(args.get('input_topic'), str) and args['input_topic']:
        return [args['input_topic']]
    return topics


def _short(topic):
    text = str(topic)
    return text if len(text) <= 64 else text[:20] + '...' + text[-41:]


def _describe_inputs(declarations, topics):
    """Bounded per-input evidence for start failures; never infers calibration."""
    rows = []
    for topic in topics[:4]:
        entry = declarations.get(topic)
        rows.append(f'{_short(topic)}: ' + (
            'no camera_info declaration' if not isinstance(entry, dict) else
            'declared ' + str(entry.get('format') or 'without format')[:32]))
    # Problems first: Core keeps only the first 200 characters of the message.
    rows.sort(key=lambda row: 'declared image/' in row)
    if len(topics) > 4:
        rows.append(f'+{len(topics)-4} more input(s)')
    unused = sorted(str(t) for t in declarations if t not in topics)
    if unused:
        rows.append('declared but not connected: ' + ', '.join(_short(t) for t in unused[:2]) +
                    (f' +{len(unused)-2} more' if len(unused) > 2 else ''))
    return '; '.join(rows)


def validate_start_inputs(declarations, topics, imu_assisted=False):
    """The optional third input is a self-describing JSON camera IMU stream.

    Failures name which input lacks a declaration or has the wrong format.
    """
    if type(imu_assisted) is not bool:
        raise ValueError('imu_assisted must be boolean')
    if not isinstance(topics, list) or any(not isinstance(t, str) or not t for t in topics):
        raise ValueError('input_topics must be a list of topic names; no RGB-D input topic received')
    declarations = declarations if isinstance(declarations, dict) else {}
    detail = _describe_inputs(declarations, topics)
    duplicates = sorted({t for t in topics if topics.count(t) > 1})
    if duplicates:
        raise ValueError('connect distinct RGB-D/IMU input topics: duplicate ' +
                         ', '.join(_short(t) for t in duplicates[:2]))
    expected = 3 if imu_assisted else 2
    if len(topics) != expected:
        raise ValueError('connect two RGB-D topics and exactly one IMU topic when assistance is enabled: '
                         f'imu_assisted={str(imu_assisted).lower()}, {len(topics)} input(s) connected; {detail}')
    rgbd = [t for t in topics if isinstance(declarations.get(t), dict) and
            declarations[t].get('format') in ('image/jpeg', 'image/depth-zlib')]
    formats = [declarations[t]['format'] for t in rgbd]
    missing = [f for f in ('image/jpeg', 'image/depth-zlib') if f not in formats]
    if len(rgbd) != 2 or missing:
        raise ValueError('connect one calibrated color and one measured depth input: ' +
                         (f'missing {" and ".join(missing)}; ' if missing else
                          'more than one declaration per role; ') + detail)
    calibration = validate_inputs(declarations, rgbd)
    if imu_assisted:
        calibration['imu_topic'] = next(t for t in topics if t not in rgbd)
    return calibration
