"""Numeric IMU input; producer conventions are documented, not admission gates."""
from collections import deque
from copy import deepcopy
from functools import lru_cache
from itertools import islice

import numpy as np

MAX_IMU_GAP_NS = 100_000_000


def imu_timestamps(packet, received_ns):
    """Normalize supplied timestamps and apply an optional numeric offset once."""
    stamp, accel = packet.get('stamp_ns'), packet.get('accel_stamp_ns')
    if stamp is None:
        milliseconds = packet.get('stamp_ms')
        if milliseconds is not None and type(milliseconds) is not int:
            raise ValueError('IMU timestamp must be integer milliseconds')
        stamp = received_ns if milliseconds is None else milliseconds * 1_000_000
    offset = packet.get('time_offset_ns', 0)
    accel = stamp if accel is None else accel
    if any(type(value) is not int for value in (stamp, accel, offset)):
        raise ValueError('IMU timestamps and offset must be integer nanoseconds')
    return stamp + offset, accel + offset


def imu_episode(packet, previous):
    """Keep the last supplied identity until an explicit source or epoch change."""
    previous_source, previous_epoch = previous
    source = packet.get('source_id') or previous_source
    epoch = packet.get('epoch') or (previous_epoch if source == previous_source else None)
    return source, epoch


@lru_cache(maxsize=16)
def _proper_rotation(matrix_bytes):
    r = np.frombuffer(matrix_bytes, dtype=float).reshape(3, 3)
    if np.any(np.abs(r) > 1.0002) or np.linalg.det(r) <= 0.:
        return None
    if np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0.):
        return r
    # Four decimal places can introduce up to 1.5e-4 of Frobenius error.
    left, _, right = np.linalg.svd(r)
    projected = left @ right
    if np.linalg.norm(r-projected) <= 2e-4:
        return projected
    return None


def _rotation(value):
    r = np.asarray(value, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError('camera/IMU rotation must contain a finite 3x3 matrix')
    proper = _proper_rotation(r.tobytes())
    if proper is None:
        raise ValueError('camera/IMU rotation must be a proper orthonormal rotation')
    return proper


def _vector(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('IMU vectors must contain three finite SI measurements')
    return v


def _scale(packet, field):
    value = packet.get(field, 1.)
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0.:
        raise ValueError('IMU unit scale must be positive and finite')
    return value

def translation_extrinsics(packet):
    """Numeric format shared by fusion and its recovery probe."""
    try:
        return np.r_[_vector(packet.get('camera_from_gyro_translation')),
                     _vector(packet.get('camera_from_accel_translation'))]
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError('IMU translations must contain two finite three-vectors') from error


def _multiply(a, b):
    av, bv = np.asarray(a[:3]), np.asarray(b[:3])
    return np.r_[a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - av @ bv]


def rotation_from_quaternion(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolate_orientation(history, image_stamp_ns):
    """Interpolate the shortest arc within a prevalidated orientation history."""
    previous = None
    for stamp, quaternion in history:
        if stamp == image_stamp_ns:
            return quaternion.tolist()
        if stamp > image_stamp_ns:
            if previous is None:
                return None
            before, a = previous
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


class ImuAttitude:
    """Relative-heading gyro integration with bounded gravity tilt correction.

    Supplied matrices map raw axes into camera optical axes once.
    Acceleration is used only as a plausible gravity direction, never as a
    velocity/position observation. Source metadata is diagnostic only.
    """
    def __init__(self, camera_frame, clock):
        self.frame, self.clock = camera_frame, clock
        self.state, self.reason = 'initializing', 'waiting for an IMU packet'
        self.last_stamp = self.last_accel = 0
        self.quaternion = None
        self.history = deque(maxlen=512)
        self.orientation_history = deque(maxlen=512)
        self.provenance = {}
        self.capture_source = None
        self.rotation_source = None
        self.rotation_run_length = 0
        self.rotation_generation = 0

    def fault(self, reason):
        self.state, self.reason = 'lost', self.reason if self.state == 'lost' else str(reason)

    def status(self, now_ns):
        return self.state

    def observe(self, packet, now_ns):
        try:
            if not isinstance(packet, dict) or packet.get('schema') != 'motus.imu/1':
                raise ValueError('IMU packet must use the motus.imu/1 object format')
            rotations = (_rotation(packet.get('camera_from_gyro')), _rotation(packet.get('camera_from_accel')))
            stamp, accel_stamp = imu_timestamps(packet, now_ns)
            gyro_scale = _scale(packet, 'gyro_scale_rad_s')
            accel_scale = _scale(packet, 'accel_scale_m_s2')
            angular = rotations[0] @ _vector(packet.get('angular_velocity')) * gyro_scale
            acceleration = rotations[1] @ _vector(packet.get('linear_acceleration')) * accel_scale
            episode = imu_episode(packet, (None, None) if self.rotation_source is None else self.rotation_source[:2])
            source = (*episode, rotations[0].tobytes(), rotations[1].tobytes(), gyro_scale, accel_scale)
            matched_acceleration = abs(stamp-accel_stamp) <= MAX_IMU_GAP_NS
            self.capture_source = source if matched_acceleration else None
            changed = source != self.rotation_source
            delta_ns = stamp-self.last_stamp if self.history else 0
            duplicate = bool(self.history) and not changed and delta_ns == 0
            if not duplicate and (changed or not 0 < delta_ns <= MAX_IMU_GAP_NS):
                self.rotation_generation += 1
                self.rotation_run_length = 0
            if not duplicate:
                self.rotation_run_length = min(self.rotation_run_length+1, self.orientation_history.maxlen)
            if changed or delta_ns < 0:
                self.quaternion = None
                dt = 0.
            else:
                dt = delta_ns/1e9
            self.rotation_source = source
            self.last_stamp, self.last_accel = stamp, accel_stamp
            self.provenance = {key: deepcopy(packet.get(key)) for key in (
                'source_id', 'epoch', 'rotation_source', 'translation_source', 'calibration_id',
                'stamp_source', 'clock_mapping', 'clock_id', 'camera_frame_id',
                'gyro_frame_id', 'accel_frame_id', 'units')}
            magnitude = np.linalg.norm(acceleration)
            gravity_valid = matched_acceleration and magnitude > 1e-12
            if self.quaternion is None:
                if not gravity_valid:
                    self.quaternion = np.array([0., 0., 0., 1.])
                else:
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
            if not duplicate:
                self.history.append(stamp)
                self.orientation_history.append((stamp, self.quaternion.copy()))
            return {'stamp_ns': stamp, 'quaternion': self.quaternion.tolist(),
                    'angular_velocity': angular.tolist(), 'linear_acceleration': acceleration.tolist()}
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.fault(error)
            raise ValueError(str(error)) from error

    def covers(self, image_stamp_ns, now_ns):
        return self.orientation_at(image_stamp_ns, now_ns, calibrated=True) is not None

    def orientation_at(self, image_stamp_ns, now_ns, *, calibrated=False):
        """Snapshot an actual capture bracket; never propagate or revive a filter.

        The caller holds its session lock. Absolute nanoseconds stay integers;
        only the difference between the two measured captures becomes a ratio.
        Missing coverage is an absent optional hint. Producer metadata and wall
        age do not change whether a numeric bracket can be interpolated.
        """
        if self.state != 'active' or not self.orientation_history or type(image_stamp_ns) is not int:
            return None
        history = self.orientation_history
        if calibrated:
            history = islice(history, len(history)-self.rotation_run_length, None)
        return interpolate_orientation(history, image_stamp_ns)

    def info(self, now_ns):
        return {'enabled': True, 'status': self.status(now_ns), 'reason': self.reason,
                'orientation_method': 'gravity_gyro_complementary', 'heading': 'relative',
                'translation_integrated': False, 'last_measurement_ns': self.last_stamp,
                'buffered_samples': len(self.history), **deepcopy(self.provenance)}
