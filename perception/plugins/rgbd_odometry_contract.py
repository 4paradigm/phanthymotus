"""Robot-independent calibrated RGB-D input and SLAM pose contracts."""
from collections import deque
from copy import deepcopy

import numpy as np

from .spatial_contract import validate_tracking_pose


def validate_inputs(declarations, topics):
    """Require decodable image roles and computable numeric camera geometry.

    Registration, depth units and coordinate conventions are producer contracts;
    their diagnostic declarations do not control admission.
    """
    if not isinstance(declarations, dict) or len(topics or []) != 2 or len(set(topics)) != 2:
        raise ValueError('connect one color and one depth input')
    entries = [declarations.get(topic) for topic in topics]
    if (any(not isinstance(d, dict) for d in entries) or
            {d.get('format') for d in entries} != {'image/jpeg', 'image/depth-zlib'}):
        raise ValueError('RGB-D requires image/jpeg and image/depth-zlib declarations')
    color = next(d for d in entries if d['format'] == 'image/jpeg')
    depth = next(d for d in entries if d['format'] == 'image/depth-zlib')
    for d in entries:
        if any(type(d.get(k)) is not int or not 0 < d[k] <= 4096 for k in ('width', 'height')):
            raise ValueError('RGB-D requires positive integer image dimensions up to 4096')
        try:
            k = np.asarray(d.get('K'), dtype=float)
            if k.size != 9 or not np.isfinite(k).all() or k.flat[0] == 0 or k.flat[4] == 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('RGB-D requires nine finite intrinsics with nonzero focal lengths') from None
    cv = color.get('vendor') if isinstance(color.get('vendor'), dict) else {}
    dv = depth.get('vendor') if isinstance(depth.get('vendor'), dict) else {}
    roles = {entry['format']: topic for topic, entry in zip(topics, entries)}
    result = dict(color_topic=roles['image/jpeg'], depth_topic=roles['image/depth-zlib'],
                  K=np.asarray(depth['K'], dtype=float).reshape(3, 3),
                  color_size=(color['width'], color['height']), depth_size=(depth['width'], depth['height']),
                  frame_id=cv.get('frame_id') or dv.get('frame_id') or 'camera_optical',
                  clock_id=cv.get('clock_id') or dv.get('clock_id'),
                  time_basis='source_capture' if cv.get('clock_id') and dv.get('clock_id') else 'receipt_estimated',
                  depth_source=dv.get('depth_source', 'unknown'),
                  motion_limits=deepcopy(cv.get('motion_limits')))
    mount = cv.get('body_mount', dv.get('body_mount'))
    if mount is not None:
        from .rgbd_body_mount import validate_mount_declaration
        result['body_mount'] = validate_mount_declaration(mount, result['frame_id'], topics)
    return result


class PairBuffer:
    """Pair image arrivals in FIFO order while preserving their source times."""
    def __init__(self, max_pending=5, max_skew_ns=20_000_000):
        self.queues = {kind: deque(maxlen=max_pending) for kind in ('color', 'depth')}

    @property
    def pending_count(self):
        return sum(len(q) for q in self.queues.values())

    def push(self, kind, stamp_ns, payload, now_ns):
        if kind not in self.queues or type(stamp_ns) is not int:
            raise ValueError('RGB-D requires an image role and integer timestamp')
        self.queues[kind].append((stamp_ns, payload))
        color, depth = self.queues['color'], self.queues['depth']
        return (color.popleft(), depth.popleft()) if color and depth else None


def make_sample(status, stamp_ns, segment, clock, pose=None, *, reason=None, context=None):
    valid = status == 'tracking'
    sample = {'schema': 'motus.odom/1', 'stamp_ms': stamp_ns // 1_000_000,
              'units': {'length': 'm', 'linear': 'm/s', 'angular': 'rad/s'},
              'frame': 'body', 'twist': [None] * 6,
              'pose': deepcopy(pose) if valid else None,
              'vendor': {'source': 'rtabmap_rgbd_odometry', 'pose_reference': 'camera_optical',
                         'accuracy_validated': False, 'reason': reason,
                         'motus_slam': {'status': status, 'stamp_source': 'measurement' if valid else 'unknown',
                                        'segment_id': segment, 'clock_id': clock, 'stamp_ns': stamp_ns}}}
    if context is not None:
        quality = sample['vendor']['motus_slam']
        quality.update(profile='motus_slam/2', observation=context.to_dict())
        if valid and context.time_basis == 'receipt_estimated':
            quality['stamp_source'] = 'receipt_estimated'
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
