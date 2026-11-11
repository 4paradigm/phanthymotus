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
    return dict(color_topic=color['topic'], depth_topic=depth['topic'],
                K=np.asarray(color['K'], dtype=float).reshape(3, 3),
                frame_id=color['vendor']['frame_id'], clock_id=color['vendor']['clock_id'])


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
                                        'segment_id': segment, 'clock_id': clock}}}
    if valid:
        validate_tracking_pose(sample)
    return sample


def validate_start_inputs(declarations, topics, imu_assisted=False):
    """The optional third input is a self-describing JSON camera IMU stream."""
    if type(imu_assisted) is not bool or not isinstance(topics, list):
        raise ValueError('imu_assisted must be boolean and input_topics must be a list')
    expected = 3 if imu_assisted else 2
    if len(topics) != expected or any(not isinstance(t, str) or not t for t in topics) or len(set(topics)) != expected:
        raise ValueError('connect two RGB-D topics and exactly one IMU topic when assistance is enabled')
    declarations = declarations if isinstance(declarations, dict) else {}
    rgbd = [t for t in topics if isinstance(declarations.get(t), dict) and
            declarations[t].get('format') in ('image/jpeg', 'image/depth-zlib')]
    calibration = validate_inputs(declarations, rgbd)
    if imu_assisted:
        calibration['imu_topic'] = next(t for t in topics if t not in rgbd)
    return calibration
