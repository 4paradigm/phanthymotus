"""Card-lifetime body odometry with ordered external motion and visual corrections."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import uuid

import numpy as np

from .external_motion_learning import ExternalMotionLearner, se3_exp, se3_log
from .pointcloud_slam_transforms import rigid
from .rgbd_fusion import quaternion
from .rgbd_imu import rotation_from_quaternion

_HISTORY_NS = 10_000_000_000
_MAX_EVENTS = 512
_MAX_INTERVAL_NS = 1_500_000_000
_LEG_ATTITUDE_INTERVAL_NS = 250_000_000
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


def _bumi_pose_covariance(record, length_scale):
    try:
        covariance = _vector(record.get('covariance'), (36,)).reshape(6, 6)
        if not covariance.any():
            return None
        units = np.array([length_scale] * 3 + [1.] * 3)
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
    variance: np.ndarray
    model: tuple | None
    cumulative_pose_covariance: np.ndarray | None = None
    body_from_sensor: np.ndarray | None = None
    body_angular_velocity: np.ndarray | None = None

    def order_key(self):
        vector = self.pose if self.pose is not None else self.twist
        covariance = self.cumulative_pose_covariance
        return (self.reset, self.pose is not None, vector.tobytes() if vector is not None else b'',
                self.mask.tobytes(), repr(self.origin), self.variance.tobytes(), repr(self.model),
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
    cumulative_pose_covariance = None
    body_angular_velocity = None
    if pose is not None:
        length_scale = _scale(packet, 'length_scale_m')
        pose[:3, 3] *= length_scale
        # PR #471's leg producer accumulates random-walk covariance in
        # its odometry frame. Generic pose producers make no such guarantee.
        if vendor.get('source') == 'bumi_leg_kinematics':
            cumulative_pose_covariance = _bumi_pose_covariance(record, length_scale)
            rate = packet.get('twist')
            if mount is None and isinstance(rate, (list, tuple)) and len(rate) == 6:
                try:
                    body_angular_velocity = _vector(rate[3:], (3,)) * _scale(packet, 'angular_scale_radps')
                except (TypeError, ValueError, OverflowError):
                    pass  # Attitude continuity is optional evidence.
    if pose is not None and mount is not None:
        pose = pose @ np.linalg.inv(mount)
    planar = packet.get('motion_model') == 'planar'
    pose_mask = np.array([True, True, False, False, False, True]) if planar else _FULL
    twist, mask = None, pose_mask if pose is not None else np.zeros(6, dtype=bool)
    if pose is None and isinstance(packet.get('twist'), (list, tuple)) and len(packet['twist']) == 6:
        mask = np.array([value is not None for value in packet['twist']])
        twist = _vector([0. if value is None else value for value in packet['twist']], (6,))
        twist *= [_scale(packet, 'linear_scale_mps')]*3 + [_scale(packet, 'angular_scale_radps')]*3
        if mount is not None:
            adjoint = _adjoint(mount)
            twist = adjoint @ twist
            mask = ~((np.abs(adjoint)[:, ~mask] > 1e-10).any(axis=1))
        if not mask.any():
            twist = None
    reset = packet.get('reset') is True
    if pose is None and twist is None and not reset:
        raise ValueError('no usable external pose or twist')
    origin = (record.get('frame_id'), record.get('child_frame_id')) if pose is not None else ()
    variance = _DEFAULT_VARIANCE
    uncertainty = packet.get('twist_covariance') if pose is None else None
    if uncertainty is not None:
        diagonal = _vector(uncertainty, (36,)).reshape(6, 6).diagonal()
        variance = np.where(diagonal > 0., diagonal, _DEFAULT_VARIANCE)
    model = ('planar' if planar else 'pose',) if pose is not None else ('twist',) if twist is not None else None
    return stamp, Motion(pose, twist, mask, origin, reset, variance, model,
                         cumulative_pose_covariance, mount if cumulative_pose_covariance is not None else None,
                         body_angular_velocity)


def _leg_origin_attitude(left, right, duration_ns):
    """Carry SDK body attitude through an automatic leg-position rebase."""
    if (duration_ns > _LEG_ATTITUDE_INTERVAL_NS or left.reset or right.reset or
            left.body_angular_velocity is None or right.body_angular_velocity is None or
            not all(left.origin) or not all(right.origin) or left.origin[1] != right.origin[1]):
        return None
    # The leg contact origin rebases translation, while SDK attitude remains
    # in its controller frame. Require both endpoints' measured angular rate
    # to support the relative quaternion before using it across that boundary.
    change = np.eye(4)
    change[:3, :3] = left.pose[:3, :3].T @ right.pose[:3, :3]
    measured = se3_log(change)
    seconds = duration_ns / 1e9
    predicted = .5 * (left.body_angular_velocity + change[:3, :3] @ right.body_angular_velocity) * seconds
    # Allow endpoint-rate sampling error during the short contact transition.
    tolerance = np.deg2rad(2.) + .3 * np.linalg.norm(predicted)
    error = np.linalg.norm(measured[3:] - predicted)
    if not np.isfinite(error) or error > tolerance:
        return None
    return measured / seconds


def _leg_increment_covariance(left, right, velocity, elapsed_s, fraction, at_measured_end):
    """Map the growth in upstream odometry axes into the step's end body axes."""
    start, end = left.cumulative_pose_covariance, right.cumulative_pose_covariance
    mount = right.body_from_sensor
    if (start is None or end is None or not all(left.origin) or
            not np.array_equal(left.body_from_sensor, mount)):
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        growth = _positive_covariance(end - start)
        if growth is None:
            return None
        body = right.pose if at_measured_end else left.pose @ se3_exp(velocity * elapsed_s)
        sensor = body @ mount if mount is not None else body
        axes = np.eye(4)
        axes[:3, :3] = sensor[:3, :3].T
        transport = _adjoint(axes)
        if mount is not None:
            transport = _adjoint(mount) @ transport
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
        self._pose_intervals = OrderedDict()
        self._learned_steps = OrderedDict()
        self.checkpoints = {}
        self.base = _State(0, np.eye(4), np.eye(6), np.zeros((6, 12)), ExternalMotionLearner(enabled=learning))
        self.current = self.base
        self._output_stamp = 0
        self._output_pose = None
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
        previous = max((key for key in self.raw_motion if key < stamp), default=stamp-1)
        self.raw_motion[stamp] = motion
        # The new right endpoint changes only motion after the preceding sample.
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

    def replay_learning(self, stamp):
        if stamp in self.raw_vision:
            self._rebuild(stamp)

    def _pose_interval(self, t0, t1, left, right, end, overlap):
        # A visual correction can replay these exact measured intervals many
        # times. A replacement at the same timestamp invalidates either end.
        key = (t0, t1, end, overlap)
        cached = self._pose_intervals.get(key)
        if cached is not None and cached[0] is left and cached[1] is right:
            self._pose_intervals.move_to_end(key)
            return cached[2], cached[3]
        velocity = se3_log(np.linalg.inv(left.pose) @ right.pose) / ((t1-t0) / 1e9)
        covariance = _leg_increment_covariance(left, right, velocity, (end-t0)/1e9,
                                               overlap/(t1-t0), end == t1)
        self._pose_intervals[key] = (left, right, velocity, covariance)
        self._pose_intervals.move_to_end(key)
        if len(self._pose_intervals) > _MAX_EVENTS:
            self._pose_intervals.popitem(last=False)
        return velocity, covariance

    def _intervals(self, times, begin, end):
        """Return ordered clipped raw intervals in the producer's body tangent."""
        first = max(0, bisect_right(times, begin) - 1)
        last = min(len(times)-1, bisect_left(times, end))
        for t0, t1 in zip(times[first:last], times[first+1:last+1]):
            left, right = self.raw_motion[t0], self.raw_motion[t1]
            if right.reset or t1-t0 > _MAX_INTERVAL_NS or left.model != right.model:
                continue
            clipped_end = min(t1, end)
            overlap = clipped_end - max(t0, begin)
            if overlap <= 0:
                continue
            increment_covariance = None
            if left.origin != right.origin:
                velocity = _leg_origin_attitude(left, right, t1-t0)
                if velocity is None:
                    continue
                mask = _ANGULAR & left.mask & right.mask
            elif left.pose is not None and right.pose is not None:
                velocity, increment_covariance = self._pose_interval(t0, t1, left, right, clipped_end, overlap)
                mask = left.mask & right.mask
            elif left.twist is not None and right.twist is not None:
                mask = left.mask & right.mask
                velocity = np.where(mask, left.twist, 0.)
            else:
                continue
            if mask.any():
                yield np.where(mask, velocity, 0.), overlap/1e9, mask, right.variance, increment_covariance

    def _learned_step(self, learner, velocity, dt, observed):
        # Delayed visual measurements often replay the same measured motion.
        # The prior can move without changing its body increment or adjoint.
        key = (learner.parameters.tobytes(), velocity.tobytes(), dt, observed.tobytes())
        step = self._learned_steps.get(key)
        if step is None:
            twist = learner.corrected_velocity(velocity, observed)
            delta = se3_exp(twist * dt)
            step = (twist, delta, _adjoint(np.linalg.inv(delta)))
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
        learner = state.learner
        if state.motion_model is not None and model != state.motion_model:
            noise, sensitivity = state.covariance, np.zeros((6, 12))
            learner = ExternalMotionLearner(enabled=learner.enabled)
        mask = np.zeros(6, dtype=bool)
        twist = None
        for velocity, dt, observed, variance, increment in self._intervals(motion_times, state.stamp, stamp):
            twist, delta, adjoint = self._learned_step(learner, velocity, dt, observed)
            transform = transform @ delta
            if increment is None:
                process_noise = np.diag(np.where(observed, variance, .25) * dt)
            else:
                # The increment already covers dt. Apply the learned velocity
                # scales and retain the default uncertainty on unobserved axes.
                scales = np.exp(learner.parameters[:6]) * observed
                process_noise = increment * np.outer(scales, scales) + np.diag(np.where(observed, 0., .25 * dt))
            noise = adjoint @ noise @ adjoint.T + process_noise
            if learner.enabled:
                sensitivity = adjoint @ sensitivity + learner.velocity_parameter_jacobian(velocity, observed)*dt
            mask |= observed
        interval_update = mask.any()
        return replace(state, stamp=stamp, transform=transform, noise=noise, sensitivity=sensitivity,
                       learner=learner, motion_model=model,
                       mask=mask if interval_update else state.mask, twist=twist,
                       last_update='external' if interval_update else None,
                       motion_since_visual=state.motion_since_visual or bool(interval_update))

    def _correct(self, state, segment, observed, variance, motion_times):
        variance = _DEFAULT_VARIANCE if variance is None else variance
        visual = (state.stamp, segment, observed, variance)
        same_segment = state.visual is not None and state.visual[1] == segment
        alignment = state.alignments.get(segment)
        if alignment is None:
            seeded = state.mask is None
            transform = observed if seeded else state.transform
            alignments = dict(state.alignments, **{segment: transform @ np.linalg.inv(observed)})
            if len(alignments) > 8:
                del alignments[next(iter(alignments))]
            return replace(state, transform=transform, alignments=alignments,
                           visual=visual, training=visual, mask=_FULL if seeded else state.mask,
                           noise=np.diag(variance) if seeded else state.noise,
                           last_update='vision' if seeded else state.last_update,
                           motion_since_visual=False)
        target = alignment @ observed
        predicted = state.transform
        if same_segment and not state.motion_since_visual:
            predicted = predicted @ np.linalg.inv(state.visual[2]) @ observed
        residual = se3_log(np.linalg.inv(predicted) @ target)
        diag = np.diag(state.covariance)
        gain = np.clip(diag / (diag + variance), 0., .8)
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
        if np.any(correction != applied):
            # A finite SE(3) log's linear part is not a pose-origin offset.
            # Keep the actual unresolved mean error in the bounded body axes.
            nominal = predicted @ se3_exp(correction)
            axes[:3, :3] = transform[:3, :3].T
            unapplied = _adjoint(axes) @ _world_pose_offset(transform, nominal)
            noise = noise + np.outer(unapplied, unapplied)
        learner = state.learner
        training = visual
        previous = state.training[0]
        if learner.enabled and same_segment and state.stamp-previous >= 100_000_000:
            intervals = [(u, dt, mask) for u, dt, mask, _, _ in
                         self._intervals(motion_times, previous, state.stamp)]
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
                       motion_since_visual=False)

    def _rebuild(self, begin):
        times = sorted(self.raw_motion)
        before = max((key for key in self.checkpoints if key < begin), default=self.base.stamp)
        state = self.checkpoints.get(before, self.base)
        events = sorted(stamp for stamp in self.raw_motion.keys() | self.raw_vision.keys() if stamp > before)
        for stamp in events:
            state = self._advance(state, stamp, times)
            if stamp in self.raw_vision:
                state = self._correct(state, *self.raw_vision[stamp], times)
            self.checkpoints[stamp] = state
        self.current = state
        cutoff = max(state.stamp-_HISTORY_NS, events[max(0, len(events)-_MAX_EVENTS)] if len(events) > _MAX_EVENTS else 0)
        # Retain an external endpoint so the first interval after a checkpoint
        # can still be recomputed from the original samples.
        all_times = sorted(self.checkpoints)
        if len(all_times) > _MAX_EVENTS:
            cutoff = max(cutoff, all_times[-_MAX_EVENTS])
        cut = max((stamp for stamp in all_times if stamp <= cutoff), default=0)
        if cut > self.base.stamp:
            self.base = self.checkpoints[cut]
            predecessor = max((stamp for stamp in times if stamp <= cut), default=cut)
            self.raw_motion = {stamp: event for stamp, event in self.raw_motion.items() if stamp >= predecessor}
            self.raw_vision = {stamp: event for stamp, event in self.raw_vision.items() if stamp > cut}
            self.checkpoints = {stamp: checkpoint for stamp, checkpoint in self.checkpoints.items() if stamp > cut}

    def pending_output_states(self, earliest_stamp):
        """Find new measurement checkpoints that can still be published."""
        return [self.checkpoints[stamp] for stamp in sorted(self.checkpoints)
                if stamp > self._output_stamp and stamp >= earliest_stamp
                and self.checkpoints[stamp].last_update is not None]

    def output(self, state=None):
        if state is None:
            state = self.current
        if state.last_update is None or state.stamp <= self._output_stamp:
            return None
        pose = state.transform
        publication_error = np.zeros(6)
        past = self.checkpoints.get(self._output_stamp)
        # Until a replay changes the published checkpoint, the correction is
        # the identity. Keep the measured transform without a log/exp round trip.
        if past is not None and self._output_pose is not None and self._output_pose is not past.transform:
            propagated = self._output_pose @ np.linalg.inv(past.transform) @ pose
            error = se3_log(np.linalg.inv(propagated) @ pose)
            if np.any(np.abs(error) > .03):
                applied = np.clip(error, -.03, .03)
                pose = propagated @ se3_exp(applied)
                publication_error = _world_pose_offset(pose, state.transform)
        self._output_pose, self._output_stamp = pose, state.stamp
        axes = np.eye(4)
        axes[:3, :3] = state.transform[:3, :3]
        frame_rotation = _adjoint(axes)
        covariance = (frame_rotation @ state.covariance @ frame_rotation.T +
                      np.outer(publication_error, publication_error))
        fused = dict(position=pose[:3, 3].tolist(), quaternion=quaternion(pose[:3, :3]),
                     frame_id=self.segment+'_odom', child_frame_id=self.body_frame,
                     covariance=covariance.ravel().tolist())
        twist = ([float(value) if observed else None for value, observed in zip(state.twist, state.mask)]
                 if state.twist is not None and state.last_update == 'external' else [None]*6)
        return dict(schema='motus.odom/1', frame='body', stamp_ms=state.stamp//1_000_000, pose=fused,
                    twist=twist, units=dict(length='m', linear='m/s', angular='rad/s'),
                    vendor=dict(source='rgbd_external_fusion', pose_reference='body', reason=None,
                                motus_slam=dict(status='tracking', stamp_source='measurement', stamp_ns=state.stamp,
                                                segment_id=self.segment, pose_source='external_fusion',
                                                pose_reference='body'),
                                external_fusion=dict(update=state.last_update, observed=state.mask.tolist(),
                                                     learning=state.learner.info())))
