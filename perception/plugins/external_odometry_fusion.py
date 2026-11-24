"""Card-lifetime body odometry with ordered external motion and visual corrections."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import uuid

import numpy as np

from .external_motion_learning import ExternalMotionLearner, se3_exp, se3_log, se3_right_jacobian
from .pointcloud_slam_contract import declares_measured_body
from .pointcloud_slam_transforms import rigid, within_measured_motion_bounds
from .rgbd_fusion import quaternion
from .rgbd_imu import rotation_from_quaternion

_HISTORY_NS = 10_000_000_000
_MAX_EVENTS = 512
_MAX_INTERVAL_NS = 1_500_000_000
_ATTITUDE_INTERVAL_NS = 250_000_000
_MEASURED_PLANAR = ('measured_planar_chassis',)
_MEASURED_SPATIAL = ('measured_spatial_body',)
_MEASURED_MODELS = (_MEASURED_PLANAR, _MEASURED_SPATIAL)
_FULL = np.ones(6, dtype=bool)
_ANGULAR = np.array([False, False, False, True, True, True])
_DEFAULT_VARIANCE = np.square([.025, .025, .035, .03, .03, .03])


def _vector(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('odometry requires finite numeric vectors')
    return array


def _scale(packet, name):
    value = packet.get(name, 1.)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not np.isfinite(value) or value <= 0.):
        raise ValueError(name + ' must be finite and positive')
    return value


def pose_matrix(pose):
    xyz = _vector(pose.get('position'), (3,))
    q = _vector(pose.get('quaternion'), (4,))
    scale = np.max(np.abs(q))
    if scale == 0:
        raise ValueError('odometry quaternion is zero')
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_from_quaternion(q / scale / np.linalg.norm(q / scale))
    matrix[:3, 3] = xyz
    return matrix


def _adjoint(matrix):
    rotation, (x, y, z) = matrix[:3, :3], matrix[:3, 3]
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    result = np.zeros((6, 6))
    result[:3, :3] = result[3:, 3:] = rotation
    result[:3, 3:] = cross @ rotation
    return result


def _world_pose_offset(reference, target):
    """Pose-origin displacement and rotation-vector error in fixed world axes."""
    rotation = np.eye(4)
    rotation[:3, :3] = target[:3, :3] @ reference[:3, :3].T
    offset = se3_log(rotation)
    offset[:3] = target[:3, 3] - reference[:3, 3]
    return offset


def body_pose_observation(pose, body_from_camera):
    camera = pose_matrix(pose)
    body = camera @ np.linalg.inv(body_from_camera)
    covariance = pose.get('covariance')
    if covariance is None:
        return body, None
    covariance = _vector(covariance, (36,)).reshape(6, 6)
    camera_axes = np.eye(4)
    camera_axes[:3, :3] = camera[:3, :3].T
    transport = _adjoint(body_from_camera) @ _adjoint(camera_axes)
    variance = np.maximum(np.diag(transport @ covariance @ transport.T), _DEFAULT_VARIANCE)
    return body, variance


def _positive_covariance(matrix):
    if not np.isfinite(matrix).all():
        return None
    scale = np.max(np.abs(matrix))
    if scale == 0.:
        return matrix
    normalized = matrix / scale
    if not np.allclose(normalized, normalized.T, rtol=0., atol=1e-10):
        return None
    normalized = .5 * (normalized + normalized.T)
    values, axes = np.linalg.eigh(normalized)
    if values[0] < -1e-10:
        return None
    if values[0] < 0.:
        normalized = (axes * np.maximum(values, 0.)) @ axes.T
    return normalized * scale


def _optional_covariance(value, units):
    try:
        covariance = _vector(value, (36,)).reshape(6, 6)
        units = np.asarray(units)
        with np.errstate(over='ignore', invalid='ignore'):
            return _positive_covariance(covariance * units[:, None] * units)
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None  # An optional covariance must not interrupt valid motion.


@dataclass(frozen=True)
class Motion:
    pose: np.ndarray | None
    twist: np.ndarray | None
    mask: np.ndarray
    origin: tuple
    reset: bool
    covariance: np.ndarray
    model: tuple | None
    process_covariance: np.ndarray | None = None
    body_from_sensor: np.ndarray | None = None
    body_angular_velocity: np.ndarray | None = None

    def order_key(self):
        vector = self.pose if self.pose is not None else self.twist
        covariance = self.process_covariance
        return (self.reset, self.pose is not None, vector.tobytes() if vector is not None else b'',
                self.mask.tobytes(), repr(self.origin), self.covariance.tobytes(), repr(self.model),
                covariance is not None, covariance.tobytes() if covariance is not None else b'',
                self.body_from_sensor.tobytes() if self.body_from_sensor is not None else b'',
                self.body_angular_velocity.tobytes() if self.body_angular_velocity is not None else b'')


def decode_motion(packet, receipt_ns):
    """Normalize body-frame pose or twist, with numeric time and an optional transform."""
    if not isinstance(packet, dict) or packet.get('schema') != 'motus.odom/1':
        raise ValueError('external odometry requires motus.odom/1')
    vendor = packet.get('vendor')
    vendor = vendor if isinstance(vendor, dict) else {}
    quality = vendor.get('motus_slam')
    quality = quality if isinstance(quality, dict) else {}
    if quality.get('pose_source') == 'external_fusion' or vendor.get('source') in (
            'rgbd_eskf', 'rtabmap_rgbd_odometry', 'rgbd_external_fusion'):
        raise ValueError('a visual odometry output cannot feed its own external motion path')
    stamp = packet.get('stamp_ns', quality.get('stamp_ns'))
    if stamp is None:
        ms = packet.get('stamp_ms')
        stamp = receipt_ns if ms is None else ms * 1_000_000 if type(ms) is int else None
    offset = packet.get('time_offset_ns', 0)
    if type(stamp) is not int or type(offset) is not int or stamp + offset <= 0:
        raise ValueError('external odometry timestamp must be positive integer nanoseconds')
    stamp += offset
    sensor = packet.get('body_from_sensor')
    mount = rigid(sensor) if sensor is not None else None
    record = packet.get('pose')
    pose = pose_matrix(record) if isinstance(record, dict) else None
    process_covariance = None
    body_angular_velocity = None
    if pose is not None:
        length_scale = _scale(packet, 'length_scale_m')
        pose[:3, 3] *= length_scale
        # Explicit cumulative process noise in this pose origin's world axes.
        # Total pose covariance is not generally an additive process accumulator.
        process_covariance = _optional_covariance(record.get('process_covariance'),
                                                  [length_scale]*3 + [1.]*3)
        rate = packet.get('twist')
        if isinstance(rate, (list, tuple)) and len(rate) == 6:
            try:
                body_angular_velocity = _vector(rate[3:], (3,)) * _scale(packet, 'angular_scale_radps')
                if mount is not None:
                    body_angular_velocity = mount[:3, :3] @ body_angular_velocity
            except (TypeError, ValueError, OverflowError):
                pass  # Attitude continuity is optional evidence.
    if pose is not None and mount is not None:
        pose = pose @ np.linalg.inv(mount)
    measured_body = (pose is not None and declares_measured_body(quality) and
                     (quality.get('planar') is True or quality.get('pose_kind') == 'measured') and
                     quality.get('status') == 'tracking' and quality.get('stamp_source') == 'measurement' and
                     quality.get('stamp_ns') == stamp - offset and
                     all(isinstance(v, str) and v for v in (quality.get('segment_id'), record.get('frame_id'),
                                                             record.get('child_frame_id'))))
    planar = quality['planar'] if measured_body else packet.get('motion_model') == 'planar'
    pose_mask = np.array([True, True, False, False, False, True]) if planar else _FULL
    twist, mask = None, pose_mask if pose is not None else np.zeros(6, dtype=bool)
    covariance = np.diag(_DEFAULT_VARIANCE)
    if pose is None and isinstance(packet.get('twist'), (list, tuple)) and len(packet['twist']) == 6:
        mask = np.array([value is not None for value in packet['twist']])
        twist = _vector([0. if value is None else value for value in packet['twist']], (6,))
        units = [_scale(packet, 'linear_scale_mps')]*3 + [_scale(packet, 'angular_scale_radps')]*3
        twist *= units
        supplied = _optional_covariance(packet.get('twist_covariance'), units)
        if supplied is not None:
            # A zero marginal is conventionally unspecified in legacy inputs.
            covariance = supplied + np.diag(np.where(supplied.diagonal() > 0., 0., _DEFAULT_VARIANCE))
        if mount is not None:
            adjoint = _adjoint(mount)
            twist = adjoint @ twist
            covariance = adjoint @ covariance @ adjoint.T
            mask = ~((np.abs(adjoint)[:, ~mask] > 1e-10).any(axis=1))
        if not mask.any():
            twist = None
    reset = packet.get('reset') is True
    if pose is None and twist is None and not reset:
        raise ValueError('no usable external pose or twist')
    origin = (record.get('frame_id'), record.get('child_frame_id')) if pose is not None else ()
    model = ('planar' if planar else 'pose',) if pose is not None else ('twist',) if twist is not None else None
    if measured_body:
        model = _MEASURED_PLANAR if planar else _MEASURED_SPATIAL
        origin += (quality['segment_id'],)
    return stamp, Motion(pose, twist, mask, origin, reset, covariance, model,
                         process_covariance, np.eye(4) if mount is None else mount,
                         body_angular_velocity)


def _continuous_chassis(left, right, interval_ns):
    if right.reset or left.model != right.model or left.origin != right.origin or not 0 < interval_ns <= _MAX_INTERVAL_NS:
        return False
    distance = np.linalg.norm(right.pose[:3, 3] - left.pose[:3, 3])
    cosine = (np.sum(left.pose[:3, :3] * right.pose[:3, :3]) - 1.) / 2.
    return within_measured_motion_bounds(distance, np.arccos(np.clip(cosine, -1., 1.)), interval_ns / 1e9)


def _origin_attitude(left, right, duration_ns):
    """Carry only measured attitude corroborated by body rates across origins."""
    if (duration_ns > _ATTITUDE_INTERVAL_NS or left.reset or right.reset or
            left.body_angular_velocity is None or right.body_angular_velocity is None or
            not all(left.origin) or not all(right.origin) or left.origin[1] != right.origin[1]):
        return None
    # A producer can rebase position while retaining its attitude reference.
    # Require numeric agreement with both endpoint rates before crossing it.
    change = np.eye(4)
    change[:3, :3] = left.pose[:3, :3].T @ right.pose[:3, :3]
    measured = se3_log(change)
    seconds = duration_ns / 1e9
    predicted = .5 * (left.body_angular_velocity + change[:3, :3] @ right.body_angular_velocity) * seconds
    # Allow endpoint-rate sampling error during a short reference transition.
    tolerance = np.deg2rad(2.) + .3 * np.linalg.norm(predicted)
    error = np.linalg.norm(measured[3:] - predicted)
    if not np.isfinite(error) or error > tolerance:
        return None
    return measured / seconds


def _angular_pose_profile(stamps, samples):
    """Raw body screws between physical rate knots, anchored at both measured poses."""
    left, right = samples[0], samples[-1]
    if (not left.mask.all() or not right.mask.all() or left.body_angular_velocity is None or
            any(not np.array_equal(sample.mask, _ANGULAR) for sample in samples[1:-1])):
        return {}
    rates = np.array([left.body_angular_velocity, *(sample.twist[3:] for sample in samples[1:-1])])
    if not np.isfinite(rates).all():
        return {}
    elapsed = np.array([stamp - stamps[0] for stamp in stamps])
    durations = np.diff(elapsed) / 1e9
    increments = np.zeros((len(durations), 6))
    increments[:, 3:] = rates * durations[:, None]
    integrated = [np.eye(3)]
    for rotation in se3_exp(increments)[:, :3, :3]:
        integrated.append(integrated[-1] @ rotation)

    delta = np.linalg.inv(left.pose) @ right.pose
    residual = np.eye(4)
    residual[:3, :3] = integrated[-1].T @ delta[:3, :3]
    fractions = (elapsed / elapsed[-1])[:, None]
    # Preserve the original body screw's translation at each physical knot.
    poses = left.pose @ se3_exp(fractions * se3_log(delta))
    correction = se3_exp(fractions * se3_log(residual))
    poses[:, :3, :3] = left.pose[:3, :3] @ np.array(integrated) @ correction[:, :3, :3]
    poses[0], poses[-1] = left.pose, right.pose
    velocities = se3_log(np.linalg.inv(poses[:-1]) @ poses[1:]) / durations[:, None]
    return {stamp: (stamp, pose.copy(), velocity.copy()) for stamp, pose, velocity in
            zip(stamps[:-1], poses[:-1], velocities)}


def _pose_increment_covariance(left, right, velocity, elapsed_s, fraction, at_measured_end,
                               step_origin=None):
    """Map the growth in upstream odometry axes into the step's end body axes."""
    start, end = left.process_covariance, right.process_covariance
    mount = right.body_from_sensor
    if (start is None or end is None or not all(left.origin) or
            not np.array_equal(left.body_from_sensor, mount)):
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        growth = _positive_covariance(end - start)
        if growth is None:
            return None
        sensor = right.pose @ mount if mount is not None else right.pose
        axes = np.eye(4)
        axes[:3, :3] = sensor[:3, :3].T
        transport = _adjoint(axes)
        if mount is not None:
            transport = _adjoint(mount) @ transport
        if not at_measured_end:
            # The accumulator describes error at the measured endpoint. Split
            # its growth in that tangent, then carry it back along the remaining
            # motion so later propagation cannot add the same lever arm twice.
            origin = left.pose if step_origin is None else step_origin
            body = origin @ se3_exp(velocity * elapsed_s)
            transport = _adjoint(np.linalg.inv(body) @ right.pose) @ transport
        step = transport @ (growth * fraction) @ transport.T
    return .5 * step + .5 * step.T if np.isfinite(step).all() else None


@dataclass(frozen=True)
class _State:
    stamp: int
    transform: np.ndarray
    noise: np.ndarray
    sensitivity: np.ndarray
    learner: ExternalMotionLearner
    alignments: dict = field(default_factory=dict)
    visual: tuple | None = None
    training: tuple | None = None
    mask: np.ndarray | None = None
    twist: np.ndarray | None = None
    last_update: str | None = None
    motion_since_visual: bool = False
    motion_model: tuple | None = None
    visual_carry: tuple | None = None
    corroborated_attitude_since: int | None = None
    last_chassis: tuple | None = None
    chassis_break_ns: int | None = None

    @property
    def covariance(self):
        return self.noise + self.sensitivity @ self.learner.covariance @ self.sensitivity.T


class ExternalOdometry:
    """Replay raw samples in timestamp order and publish in one local body frame."""

    def __init__(self, *, learning=True, body_frame='body', solve_learning=None):
        self.segment = 'rgbd_body_' + uuid.uuid4().hex[:12]
        self.body_frame = body_frame
        self.solve_learning = solve_learning
        self.raw_motion = {}
        self.raw_vision = {}
        self._pose_profiles = {}
        self._pose_intervals = OrderedDict()
        self._learned_steps = OrderedDict()
        self.checkpoints = {}
        self.base = _State(0, np.eye(4), np.eye(6), np.zeros((6, 12)), ExternalMotionLearner(enabled=learning))
        self.current = self.base
        self._output_stamp = 0
        self._output_pose = None
        self._slewing_output = False
        self._batch_changes = None

    @contextmanager
    def batch(self):
        """Rebuild once after staging a batch of observations."""
        changes = []
        self._batch_changes = changes
        try:
            yield
        finally:
            self._batch_changes = None
            if changes:
                self._rebuild(min(changes))

    def _changed(self, stamp):
        if self._batch_changes is None:
            self._rebuild(stamp)
        else:
            self._batch_changes.append(stamp)

    def motion(self, stamp, motion):
        prior = self.raw_motion.get(stamp)
        if stamp <= self.base.stamp or (prior is not None and prior.order_key() >= motion.order_key()):
            return False
        times = sorted(self.raw_motion)
        index = bisect_left(times, stamp)
        previous = times[index-1] if index else stamp-1
        if motion.pose is not None or index < len(times):
            # A pose can complete an interval across partial reports. Inserting
            # or replacing *any* endpoint can also break an existing bracket,
            # invalidating every interpolated checkpoint after its left pose.
            first = self._pose_start(times, index-1)
            if first is not None:
                previous = times[first]
        if index < len(times):
            # Any interior amendment can change the residual at every knot.
            self._pose_profiles = {bounds: profile for bounds, profile in self._pose_profiles.items()
                                   if not bounds[0] <= stamp <= bounds[1]}
        self.raw_motion[stamp] = motion
        self._changed(previous+1)
        return True

    def vision(self, stamp, segment, body_pose, variance=None):
        prior = self.raw_vision.get(stamp)
        if stamp <= self.base.stamp or (prior is not None and
                (prior[0], prior[1].tobytes()) >= (segment, body_pose.tobytes())):
            return False
        self.raw_vision[stamp] = (segment, body_pose, variance)
        self._changed(stamp)
        return True

    def revise_vision(self, stamp, segment, body_pose, variance=None):
        """Replace or retract an owned projection after its dependencies change."""
        if stamp <= self.base.stamp:
            return False
        prior = self.raw_vision.get(stamp)
        if body_pose is None:
            if prior is None or prior[0] != segment:
                return False
            del self.raw_vision[stamp]
        else:
            if (prior is not None and prior[0] == segment and
                    np.array_equal(prior[1], body_pose) and np.array_equal(prior[2], variance)):
                return False
            self.raw_vision[stamp] = (segment, body_pose, variance)
        self._changed(stamp)
        return True

    def replay_learning(self, stamp):
        # The solve's former endpoint can have been withdrawn or frozen while
        # work was in flight. Resume at the first surviving mutable observation.
        begin = max(stamp, self.base.stamp+1)
        next_visual = min((value for value in self.raw_vision if value >= begin), default=None)
        if next_visual is not None:
            self._changed(next_visual)

    def _pose_interval(self, t0, t1, left, right, end, overlap, knot=None):
        # A visual correction can replay these exact measured intervals many
        # times. A replacement at the same timestamp invalidates either end.
        key = (t0, t1, end, overlap)
        cached = self._pose_intervals.get(key)
        if cached is not None and cached[0] is left and cached[1] is right and cached[2] is knot:
            self._pose_intervals.move_to_end(key)
            return cached[3], cached[4], cached[5]
        if knot is None:
            start, origin = t0, left.pose
            velocity = se3_log(np.linalg.inv(left.pose) @ right.pose) / ((t1-t0) / 1e9)
        else:
            start, origin, velocity = knot
        covariance = _pose_increment_covariance(left, right, velocity, (end-start)/1e9,
                                                overlap/(t1-t0), end == t1, origin)
        fixed_attitude = (knot is None and (left.mask & right.mask)[3:].all() and
                          _origin_attitude(left, right, t1-t0) is not None)
        self._pose_intervals[key] = (left, right, knot, velocity, covariance, fixed_attitude)
        self._pose_intervals.move_to_end(key)
        if len(self._pose_intervals) > _MAX_EVENTS:
            self._pose_intervals.popitem(last=False)
        return velocity, covariance, fixed_attitude

    def _pose_start(self, times, index):
        while index >= 0:
            candidate = self.raw_motion[times[index]]
            if candidate.pose is not None:
                return index
            if candidate.reset or candidate.twist is None or candidate.mask.all():
                break
            index -= 1
        return None

    def _pose_bracket(self, times, t0, t1):
        """Accepted pose endpoints around partial reports, without a reset."""
        first = self._pose_start(times, bisect_right(times, t0)-1)
        if first is None:
            return None
        last = bisect_left(times, t1)
        while last < len(times) and self.raw_motion[times[last]].pose is None:
            candidate = self.raw_motion[times[last]]
            if candidate.reset or candidate.twist is None or candidate.mask.all():
                return None
            last += 1
        if last >= len(times):
            return None
        begin, end = times[first], times[last]
        left, right = self.raw_motion[begin], self.raw_motion[end]
        if (begin >= end or end-begin > _MAX_INTERVAL_NS or right.reset or
                left.origin != right.origin or left.model != right.model or
                any(self.raw_motion[t].reset for t in times[first+1:last])):
            return None
        return begin, end, left, right

    def _intervals(self, times, begin, end):
        """Return ordered clipped raw intervals in the producer's body tangent."""
        first = max(0, bisect_right(times, begin) - 1)
        last = min(len(times)-1, bisect_left(times, end))
        for t0, t1 in zip(times[first:last], times[first+1:last+1]):
            left, right = self.raw_motion[t0], self.raw_motion[t1]
            if right.reset or t1-t0 > _MAX_INTERVAL_NS:
                continue
            clipped_end = min(t1, end)
            overlap = clipped_end - max(t0, begin)
            if overlap <= 0:
                continue
            increment_covariance = None
            seconds_to_knot = 0.
            fixed_attitude = False
            bracket = (self._pose_bracket(times, t0, t1)
                       if left.pose is None or right.pose is None else None)
            if bracket is not None or (left.pose is not None and right.pose is not None):
                first_stamp, last_stamp, a, b = bracket or (t0, t1, left, right)
                if a.model == b.model and a.model in _MEASURED_MODELS and not _continuous_chassis(a, b, last_stamp-first_stamp):
                    continue
            if bracket is not None:
                p0, p1, pose_left, pose_right = bracket
                bounds = (p0, p1)
                if bounds not in self._pose_profiles:
                    stamps = times[bisect_left(times, p0):bisect_right(times, p1)]
                    self._pose_profiles[bounds] = _angular_pose_profile(
                        stamps, [self.raw_motion[t] for t in stamps])
                knot = self._pose_profiles[bounds].get(t0)
                # Map each shaped piece's covariance into the physical end
                # tangent before applying any learned scales or clipping.
                noise_end = t1 if knot is not None else clipped_end
                seconds_to_knot = (noise_end - clipped_end) / 1e9
                velocity, increment_covariance, _ = self._pose_interval(
                    p0, p1, pose_left, pose_right, noise_end, overlap, knot)
                mask = pose_left.mask & pose_right.mask
                covariance = pose_left.covariance
            elif left.model != right.model:
                # Only a pose with a measured rate and an angular-only report
                # can carry gyro through loss or recovery of translation.
                pose, partial = (left, right) if left.pose is not None else (right, left)
                if (pose.body_angular_velocity is None or partial.twist is None or
                        partial.mask[:3].any()):
                    continue
                if left.reset and t1-t0 > _ATTITUDE_INTERVAL_NS:
                    continue
                rate = left.twist[3:] if left.twist is not None else left.body_angular_velocity
                velocity = np.r_[np.zeros(3), rate]
                mask = _ANGULAR & left.mask & right.mask
                covariance = left.covariance
            elif left.origin != right.origin:
                velocity = _origin_attitude(left, right, t1-t0)
                if velocity is None:
                    continue
                mask = _ANGULAR & left.mask & right.mask
                fixed_attitude = mask[3:].all()
                covariance = left.covariance
            elif left.pose is not None and right.pose is not None:
                velocity, increment_covariance, fixed_attitude = self._pose_interval(
                    t0, t1, left, right, clipped_end, overlap)
                mask = left.mask & right.mask
                covariance = left.covariance
            elif left.twist is not None and right.twist is not None:
                mask = left.mask & right.mask
                velocity = np.where(mask, left.twist, 0.)
                covariance = left.covariance
            else:
                continue
            if mask.any():
                yield (np.where(mask, velocity, 0.), overlap/1e9, mask, covariance,
                       increment_covariance, seconds_to_knot, fixed_attitude)

    def _learned_step(self, learner, velocity, dt, observed, fixed_attitude, measured_chassis):
        # Delayed visual measurements often replay the same measured motion.
        # Its increment, noise scaling and parameter sensitivity depend only on this step.
        key = (learner.parameters.tobytes(), velocity.tobytes(), dt, observed.tobytes(),
               fixed_attitude, measured_chassis, learner.enabled)
        step = self._learned_steps.get(key)
        if step is None:
            calibrated = observed
            if fixed_attitude:
                calibrated = observed.copy()
                calibrated[3:] = False
            if measured_chassis:
                calibrated = np.zeros(6, dtype=bool)
            twist = velocity.copy() if measured_chassis else learner.corrected_velocity(velocity, calibrated)
            if fixed_attitude:
                # Independently corroborated angular increments already measure
                # body motion. A visual fit cannot add angular drift to them.
                twist[3:] = velocity[3:]
            integrated = twist * dt
            delta = se3_exp(integrated)
            sensitivity_increment = None
            if learner.enabled:
                sensitivity_increment = (
                    se3_right_jacobian(integrated) @ learner.velocity_parameter_jacobian(velocity, calibrated) * dt)
            scales = np.exp(np.where(calibrated, learner.parameters[:6], 0.)) * observed
            step = (twist, delta, _adjoint(np.linalg.inv(delta)), sensitivity_increment, np.outer(scales, scales))
            self._learned_steps[key] = step
            if len(self._learned_steps) > _MAX_EVENTS:
                self._learned_steps.popitem(last=False)
        else:
            self._learned_steps.move_to_end(key)
        return step

    def _advance(self, state, stamp, motion_times):
        transform, noise, sensitivity = state.transform, state.noise, state.sensitivity
        motion = self.raw_motion.get(stamp)
        model = motion.model if motion is not None and motion.model is not None else state.motion_model
        if motion is not None and motion.pose is None:
            bracket = self._pose_bracket(motion_times, stamp, stamp)
            if bracket is not None:
                model = bracket[2].model
        learner = state.learner
        if state.motion_model is not None and model != state.motion_model:
            noise, sensitivity = state.covariance, np.zeros((6, 12))
            learner = ExternalMotionLearner(enabled=learner.enabled)
        mask = np.zeros(6, dtype=bool)
        twist = None
        corroborated_span = 0.
        pieces = self._intervals(motion_times, state.stamp, stamp)
        for velocity, dt, observed, covariance, increment, seconds_to_knot, fixed_attitude in pieces:
            twist, delta, adjoint, sensitivity_increment, noise_scale = self._learned_step(
                learner, velocity, dt, observed, fixed_attitude, model in _MEASURED_MODELS)
            transform = transform @ delta
            if fixed_attitude:
                corroborated_span += dt
            if increment is None:
                # Preserve the stream's variance-per-second propagation while
                # retaining its full spatial covariance and learned unit scale.
                process_noise = covariance * noise_scale * dt
            else:
                # The increment already covers dt. Apply the learned velocity
                # scales and retain the default uncertainty on unobserved axes.
                process_noise = increment * noise_scale
                if seconds_to_knot:
                    # Later propagation uses this calibrated screw. Transport
                    # the knot's share back to the current clip in its tangent.
                    transport = _adjoint(se3_exp(twist * seconds_to_knot))
                    process_noise = transport @ process_noise @ transport.T
            process_noise = process_noise + np.diag(np.where(observed, 0., .25 * dt))
            noise = adjoint @ noise @ adjoint.T + process_noise
            if learner.enabled:
                sensitivity = adjoint @ sensitivity + sensitivity_increment
            mask |= observed
        attitude_since = state.corroborated_attitude_since
        if stamp > state.stamp:
            if abs(corroborated_span - (stamp-state.stamp)/1e9) > 1e-9:
                attitude_since = None
            elif attitude_since is None:
                attitude_since = state.stamp
        last_chassis, chassis_break = state.last_chassis, state.chassis_break_ns
        if motion is not None and motion.model in _MEASURED_MODELS:
            if last_chassis is not None and not _continuous_chassis(last_chassis[1], motion, stamp-last_chassis[0]):
                chassis_break = stamp
            last_chassis = (stamp, motion)
        interval_update = mask.any()
        return replace(state, stamp=stamp, transform=transform, noise=noise, sensitivity=sensitivity,
                       learner=learner, motion_model=model, corroborated_attitude_since=attitude_since,
                       last_chassis=last_chassis, chassis_break_ns=chassis_break,
                       mask=mask if interval_update else state.mask, twist=twist,
                       last_update='external' if interval_update else None,
                       motion_since_visual=state.motion_since_visual or bool(interval_update))

    def _correct(self, state, segment, observed, variance, motion_times):
        variance = _DEFAULT_VARIANCE if variance is None else variance
        visual = (state.stamp, segment, observed, variance)
        same_segment = state.visual is not None and state.visual[1] == segment
        alignment = state.alignments.get(segment)
        # A measured chassis interval already supplies the body displacement.
        # Align this visual observation to it so vision can take over from
        # this pose if the next chassis measurement is missing.
        chassis_interval = (state.motion_model in _MEASURED_MODELS and
                            state.last_update == 'external' and state.mask[[0, 1, 5]].all())
        if alignment is None or chassis_interval:
            seeded = state.mask is None
            transform = observed if seeded else state.transform
            alignments = dict(state.alignments, **{segment: transform @ np.linalg.inv(observed)})
            if len(alignments) > 8:
                del alignments[next(iter(alignments))]
            return replace(state, transform=transform, alignments=alignments,
                           visual=visual, training=visual, mask=_FULL if seeded else state.mask,
                           noise=np.diag(variance) if seeded else state.noise,
                           last_update='vision' if seeded else state.last_update,
                           visual_carry=(np.zeros((6, 6)), np.zeros((6, 12))) if seeded else
                                        (state.noise + np.diag(variance), state.sensitivity),
                           motion_since_visual=False)
        if same_segment and not state.motion_since_visual:
            # The new native pose supplies the motion AND the measurement.
            # Updating a prior propagated with it as if independent would count
            # that evidence twice. Carry the fixed alignment uncertainty and
            # replace the previous native measurement's noise with the new one.
            delta = np.linalg.inv(state.visual[2]) @ observed
            adjoint = _adjoint(np.linalg.inv(delta))
            carry_noise, carry_sensitivity = state.visual_carry
            carry_noise = adjoint @ carry_noise @ adjoint.T
            carry_sensitivity = adjoint @ carry_sensitivity
            return replace(state, transform=state.transform @ delta,
                           noise=carry_noise + np.diag(variance), sensitivity=carry_sensitivity,
                           visual=visual, training=visual, mask=_FULL, last_update='vision',
                           visual_carry=(carry_noise, carry_sensitivity))
        target = alignment @ observed
        predicted = state.transform
        residual = se3_log(np.linalg.inv(predicted) @ target)
        diag = np.diag(state.covariance)
        gain = np.clip(diag / (diag + variance), 0., .8)
        if (same_segment and state.corroborated_attitude_since is not None and
                state.visual[0] >= state.corroborated_attitude_since):
            # The independent pose and gyro cover the whole visual interval.
            # Use the visual position, without interpreting its tilt as
            # angular motion or coupling that tilt into the translation residual.
            gain[3:] = 0.
            residual[:3] = predicted[:3, :3].T @ (target[:3, 3] - predicted[:3, 3])
        correction = gain * residual
        applied = np.clip(correction, -.05, .05)
        increment = se3_exp(applied)
        transform = predicted @ increment
        damping = np.diag(1.-gain)
        noise = damping @ state.noise @ damping.T + np.diag(gain**2*variance)
        axes = np.eye(4)
        axes[:3, :3] = increment[:3, :3].T
        reframe = _adjoint(axes)
        noise = reframe @ noise @ reframe.T
        sensitivity = reframe @ damping @ state.sensitivity
        # For a following visual-only interval, posterior-minus-current-visual
        # contains (I-K) of both prior and visual error, not K of visual error.
        carry_noise = reframe @ damping @ (state.noise + np.diag(variance)) @ damping @ reframe.T
        if np.any(correction != applied):
            # A finite SE(3) log's linear part is not a pose-origin offset.
            # Keep the actual unresolved mean error in the bounded body axes.
            nominal = predicted @ se3_exp(correction)
            axes[:3, :3] = transform[:3, :3].T
            unapplied = _adjoint(axes) @ _world_pose_offset(transform, nominal)
            noise = noise + np.outer(unapplied, unapplied)
            carry_noise = carry_noise + np.outer(unapplied, unapplied)
        learner = state.learner
        training = visual
        previous = state.training[0]
        if learner.enabled and same_segment and state.stamp-previous >= 100_000_000:
            intervals = [(velocity, dt, observed, fixed_attitude)
                         for velocity, dt, observed, _, _, _, fixed_attitude
                         in self._intervals(motion_times, previous, state.stamp)]
            if intervals and abs(sum(x[1] for x in intervals)-(state.stamp-previous)/1e9) < 1e-6:
                learner = learner.checkpoint()
                delta = np.linalg.inv(state.training[2]) @ observed
                transport = _adjoint(np.linalg.inv(delta))
                start_variance = np.square(transport) @ state.training[3]
                # Endpoint errors share visual history. With unknown cross-
                # correlation, the sum of sigmas bounds each relative error;
                # a weak endpoint must not become a precise training target.
                sigma = np.sqrt(start_variance) + np.sqrt(variance)
                learner.add(previous, state.stamp, intervals, delta, noise=sigma,
                            solve=self.solve_learning)
        elif learner.enabled and same_segment:
            training = state.training
        return replace(state, transform=transform, noise=noise, sensitivity=sensitivity,
                       learner=learner, visual=visual, training=training, mask=_FULL, last_update='vision',
                       visual_carry=(carry_noise, sensitivity),
                       motion_since_visual=False)

    def _rebuild(self, begin):
        times = sorted(self.raw_motion)
        before = max((key for key in self.checkpoints if key < begin), default=self.base.stamp)
        state = self.checkpoints.get(before, self.base)
        # Retractions can remove an event timestamp entirely. Its old checkpoint
        # must not survive the replay or remain eligible for later publication.
        self.checkpoints = {stamp: checkpoint for stamp, checkpoint in self.checkpoints.items()
                            if stamp <= before}
        events = sorted(stamp for stamp in self.raw_motion.keys() | self.raw_vision.keys() if stamp > before)
        for stamp in events:
            state = self._advance(state, stamp, times)
            if stamp in self.raw_vision:
                state = self._correct(state, *self.raw_vision[stamp], times)
            self.checkpoints[stamp] = state
        self.current = state
        all_times = sorted(self.checkpoints)
        time_index = bisect_right(all_times, state.stamp-_HISTORY_NS)-1
        cut = all_times[time_index] if time_index >= 0 else self.base.stamp
        if cut > self.base.stamp:
            # A partial checkpoint inside an open or completed pose interval
            # can still change when a later in-window endpoint is amended.
            first = self._pose_start(times, bisect_right(times, cut)-1)
            if first is not None and cut-times[first] <= _MAX_INTERVAL_NS:
                cut = times[first]
        if len(all_times) >= _MAX_EVENTS:
            capacity_cut = all_times[-_MAX_EVENTS]
            if capacity_cut > cut:
                # If the event cap forces a completed bracket out, freeze at
                # its closing measurement, never at a synthetic checkpoint.
                bracket = self._pose_bracket(times, capacity_cut, capacity_cut)
                cut = bracket[1] if bracket is not None else capacity_cut
        if cut > self.base.stamp:
            self._pose_profiles = {bounds: profile for bounds, profile in self._pose_profiles.items()
                                   if bounds[1] > cut}
            self.base = self.checkpoints[cut]
            predecessor = max((stamp for stamp in times if stamp <= cut), default=cut)
            self.raw_motion = {stamp: event for stamp, event in self.raw_motion.items() if stamp >= predecessor}
            self.raw_vision = {stamp: event for stamp, event in self.raw_vision.items() if stamp > cut}
            # The closing measurement may be the newest publishable state.
            self.checkpoints = {stamp: checkpoint for stamp, checkpoint in self.checkpoints.items() if stamp >= cut}

    def pending_output_states(self, earliest_stamp):
        """Find new measurement checkpoints that can still be published."""
        return [self.checkpoints[stamp] for stamp in sorted(self.checkpoints)
                if stamp > self._output_stamp and stamp >= earliest_stamp
                and self.checkpoints[stamp].last_update is not None]

    def _publication_pose(self, state):
        pose = state.transform
        previous = self.checkpoints.get(self._output_stamp)
        interval = state.stamp - self._output_stamp
        if (state.motion_model is None or previous is None or
                self._output_pose is previous.transform or interval > _MAX_INTERVAL_NS):
            return pose
        dt = interval / 1e9

        # A late observation is allowed to revise internal history. Preserve
        # the interval measured since that revised stamp and release only the
        # resulting correction to the already published pose gradually.
        change = _world_pose_offset(self._output_pose, pose)
        if not self._slewing_output and within_measured_motion_bounds(
                np.linalg.norm(change[:3]), np.linalg.norm(change[3:]), dt):
            return pose
        advanced = self._output_pose @ np.linalg.solve(previous.transform, pose)
        correction = se3_log(np.linalg.solve(advanced, pose))
        # Use measurement time so a burst of publications cannot accelerate
        # correction. Cap each step even after a sparse sampling interval.
        step = min(.05, .5*dt)
        size = max(np.linalg.norm(correction[:3]), np.linalg.norm(correction[3:]))
        return advanced @ se3_exp(correction * (step/size)) if size > step else pose

    def segment_id(self, state):
        return self.segment if state.chassis_break_ns is None else f'{self.segment}_{state.chassis_break_ns}'

    def output(self, state=None):
        if state is None:
            state = self.current
        if state.last_update is None or state.stamp <= self._output_stamp:
            return None
        pose = self._publication_pose(state)
        self._output_stamp, self._output_pose = state.stamp, pose
        self._slewing_output = pose is not state.transform
        axes = np.eye(4)
        axes[:3, :3] = state.transform[:3, :3]
        frame_rotation = _adjoint(axes)
        covariance = frame_rotation @ state.covariance @ frame_rotation.T
        if self._slewing_output:
            unapplied = _world_pose_offset(pose, state.transform)
            covariance += np.outer(unapplied, unapplied)
        segment = self.segment_id(state)
        fused = dict(position=pose[:3, 3].tolist(), quaternion=quaternion(pose[:3, :3]),
                     frame_id=segment+'_odom', child_frame_id=self.body_frame,
                     covariance=covariance.ravel().tolist())
        twist = ([float(value) if observed else None for value, observed in zip(state.twist, state.mask)]
                 if state.twist is not None and state.last_update == 'external' else [None]*6)
        return dict(schema='motus.odom/1', frame='body', stamp_ms=state.stamp//1_000_000, pose=fused,
                    twist=twist, units=dict(length='m', linear='m/s', angular='rad/s'),
                    vendor=dict(source='rgbd_external_fusion', pose_reference='body', reason=None,
                                motus_slam=dict(status='tracking', stamp_source='measurement', stamp_ns=state.stamp,
                                                segment_id=segment, pose_source='external_fusion',
                                                pose_reference='body'),
                                external_fusion=dict(update=state.last_update, observed=state.mask.tolist(),
                                                     learning=state.learner.info())))
