"""Capture-timed camera IMU attitude prior; no inertial translation integration."""
from collections import deque
from copy import deepcopy

import numpy as np


MAX_CAPTURE_AGE_NS = 750_000_000
MAX_CAPTURE_GAP_NS = 150_000_000


def _rotation(value):
    r = np.asarray(value, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all() or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(r), 1., atol=1e-6)):
        raise ValueError('camera/IMU rotation must be a proper orthonormal 3x3 matrix')
    return r


def _vector(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('IMU vectors must contain three finite SI measurements')
    return v


def _multiply(a, b):
    av, bv = np.asarray(a[:3]), np.asarray(b[:3])
    return np.r_[a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - av @ bv]


def rotation_from_quaternion(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class ImuAttitude:
    """Relative-heading gyro integration with bounded gravity tilt correction.

    Factory/TF rotations map raw motion axes into the camera optical axes once.
    Acceleration is used only as a plausible gravity direction, never as a
    velocity/position observation. Every discontinuity retires this filter.
    """
    def __init__(self, camera_frame, clock):
        self.frame, self.clock = camera_frame, clock
        self.state, self.reason = 'initializing', 'waiting for capture-timed camera IMU gravity'
        self.last_stamp = self.last_accel = 0
        self.quaternion = None
        self.identity = self.rotations = None
        self.history = deque(maxlen=512)
        self.orientation_history = deque(maxlen=512)
        self.provenance = {}

    def fault(self, reason):
        self.state, self.reason = 'lost', self.reason if self.state == 'lost' else str(reason)

    def status(self, now_ns):
        if self.last_stamp and now_ns - self.last_stamp > MAX_CAPTURE_AGE_NS:
            self.fault('IMU capture stale; restart for a new segment')
        return self.state

    def observe(self, packet, now_ns):
        try:
            if self.status(now_ns) == 'lost':
                raise ValueError(self.reason)
            if (not isinstance(packet, dict) or packet.get('schema') != 'motus.imu/1' or
                    packet.get('stamp_source') != 'measurement' or
                    packet.get('clock_id') != self.clock or packet.get('camera_frame_id') != self.frame):
                raise ValueError('IMU schema, capture clock or camera frame mismatch')
            if packet.get('units') != {'angular_velocity': 'rad/s', 'linear_acceleration': 'm/s^2'}:
                raise ValueError('IMU requires rad/s and m/s^2 units')
            names = ('source_id', 'epoch', 'gyro_frame_id', 'accel_frame_id')
            if any(not isinstance(packet.get(k), str) or not packet[k].strip() or
                   packet[k] == 'unknown' for k in names):
                raise ValueError('IMU source, epoch and physical motion frames are required')
            if packet.get('rotation_source') not in ('sdk_factory_extrinsics', 'tf'):
                raise ValueError('IMU rotation requires device calibration or source TF')
            rotations = (_rotation(packet.get('camera_from_gyro')), _rotation(packet.get('camera_from_accel')))
            identity = tuple(packet[k] for k in names) + (packet['rotation_source'],)
            if self.identity is not None and (identity != self.identity or any(
                    not np.allclose(old, new, atol=1e-7, rtol=0) for old, new in zip(self.rotations, rotations))):
                raise ValueError('IMU source epoch or calibration changed; restart for a new segment')
            stamp, accel_stamp = packet.get('stamp_ns'), packet.get('accel_stamp_ns')
            if (type(stamp) is not int or type(accel_stamp) is not int or
                    stamp <= self.last_stamp or accel_stamp <= 0 or accel_stamp < self.last_accel or
                    abs(stamp - accel_stamp) > 20_000_000 or
                    any(not -50_000_000 <= now_ns - t <= MAX_CAPTURE_AGE_NS for t in (stamp, accel_stamp)) or
                    (self.last_stamp and stamp - self.last_stamp > MAX_CAPTURE_GAP_NS)):
                raise ValueError('IMU capture is stale, repeated, reset or unsynchronized')
            angular = rotations[0] @ _vector(packet.get('angular_velocity'))
            acceleration = rotations[1] @ _vector(packet.get('linear_acceleration'))
            dt = (stamp - self.last_stamp) / 1e9 if self.last_stamp else 0.
            self.identity, self.rotations = identity, rotations
            self.last_stamp, self.last_accel = stamp, accel_stamp
            self.provenance = {'source_id': packet['source_id'], 'epoch': packet['epoch'],
                               'rotation_source': packet['rotation_source'],
                               'clock_mapping': deepcopy(packet.get('clock_mapping'))}
            magnitude = np.linalg.norm(acceleration)
            gravity_valid = 8. <= magnitude <= 12.
            if self.quaternion is None:
                if not gravity_valid:
                    return None
                measured = acceleration / magnitude
                q = np.r_[np.cross(measured, [0., 0., 1.]), 1. + measured[2]]
                self.quaternion = np.array([1., 0., 0., 0.]) if np.linalg.norm(q) < 1e-8 else q / np.linalg.norm(q)
            else:
                corrected = angular.copy()
                if gravity_valid:
                    predicted = rotation_from_quaternion(self.quaternion).T @ [0., 0., 1.]
                    corrected += 2. * np.cross(acceleration / magnitude, predicted)
                speed = np.linalg.norm(corrected)
                if speed > 1e-12:
                    half = speed * dt / 2
                    delta = np.r_[corrected / speed * np.sin(half), np.cos(half)]
                    self.quaternion = _multiply(self.quaternion, delta)
                    self.quaternion /= np.linalg.norm(self.quaternion)
            self.state, self.reason = 'active', None
            self.history.append(stamp)
            self.orientation_history.append((stamp, self.quaternion.copy()))
            return {'stamp_ns': stamp, 'quaternion': self.quaternion.tolist(),
                    'angular_velocity': angular.tolist(), 'linear_acceleration': acceleration.tolist()}
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.fault(error)
            raise ValueError(str(error)) from error

    def covers(self, image_stamp_ns, now_ns):
        return (self.status(now_ns) == 'active' and bool(self.history) and
                self.history[0] <= image_stamp_ns <= self.history[-1])

    def orientation_at(self, image_stamp_ns, now_ns):
        """Snapshot an actual capture bracket; never propagate or revive a filter.

        The caller holds its session lock. Absolute nanoseconds stay integers;
        only the difference between the two measured captures becomes a ratio.
        Missing coverage is an absent optional hint, never a reason to wait for
        IMU or suppress an otherwise usable visual frame.
        """
        if (self.state != 'active' or not self.orientation_history or
                type(image_stamp_ns) is not int or type(now_ns) is not int or
                image_stamp_ns <= 0 or
                not -50_000_000 <= now_ns-image_stamp_ns <= MAX_CAPTURE_AGE_NS or
                not -50_000_000 <= now_ns-self.last_stamp <= MAX_CAPTURE_AGE_NS):
            return None
        previous = None
        for stamp, quaternion in self.orientation_history:
            if stamp == image_stamp_ns:
                return quaternion.tolist()
            if stamp > image_stamp_ns:
                if previous is None:
                    return None
                before, a = previous
                if stamp-before > MAX_CAPTURE_GAP_NS:
                    return None
                ratio = (image_stamp_ns-before) / (stamp-before)
                b = quaternion
                dot = float(a @ b)
                if dot < 0.:
                    b, dot = -b, -dot  # q and -q have the same rotation.
                dot = min(1., max(0., dot))
                if dot > .9995:
                    result = (1.-ratio)*a + ratio*b
                else:
                    angle = np.arccos(dot)
                    result = (np.sin((1.-ratio)*angle)*a + np.sin(ratio*angle)*b) / np.sin(angle)
                return (result / np.linalg.norm(result)).tolist()
            previous = stamp, quaternion
        return None  # No extrapolation beyond the last measured capture.

    def info(self, now_ns):
        return {'enabled': True, 'status': self.status(now_ns), 'reason': self.reason,
                'orientation_method': 'gravity_gyro_complementary', 'heading': 'relative',
                'translation_integrated': False, 'last_measurement_ns': self.last_stamp,
                'buffered_samples': len(self.history), **deepcopy(self.provenance)}
