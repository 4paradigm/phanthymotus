"""Optional rotation hints for the pinned native visual registration frontend."""
from collections import deque
from pathlib import Path
import re

import numpy as np

from .rgbd_imu import MAX_IMU_GAP_NS, _multiply, interpolate_orientation


def association_enabled(calibration):
    return bool(calibration.get('controller_rotation')) or (
            calibration.get('inertial_association') is True and
            calibration.get('short_term_compensation') is True and
            bool(calibration.get('imu_topic')))


def _unit_quaternion(value):
    try:
        q = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('controller rotation requires a unit quaternion in xyzw order') from None
    if q.shape != (4,) or not np.isfinite(q).all() or abs(q @ q - 1.) > .002:
        raise ValueError('controller rotation requires a unit quaternion in xyzw order')
    return q / np.linalg.norm(q)


def validate_controller_rotation(profile):
    """An explicit, capture-clock-specific alignment for a controller read-time prior."""
    if profile is None:
        return None
    if not isinstance(profile, dict) or any(not isinstance(profile.get(k), str) or not profile[k].strip()
                                            for k in ('topic', 'clock_id', 'calibration_id')):
        raise ValueError('controller rotation requires topic, clock_id and calibration_id')
    topic, offset = profile['topic'], profile.get('read_time_offset_ns')
    if not re.fullmatch(r'(/[A-Za-z_][A-Za-z_0-9]*)+', topic):
        raise ValueError('controller rotation requires an absolute ROS topic')
    if type(offset) is not int or abs(offset) > MAX_IMU_GAP_NS:
        raise ValueError('controller rotation read-time offset must be within 100 ms in integer nanoseconds')
    return dict(topic=topic, clock_id=profile['clock_id'], calibration_id=profile['calibration_id'],
                camera_from_body=_unit_quaternion(profile.get('camera_from_body')).tolist(),
                read_time_offset_ns=offset)


def require_rotation_hint_runtime(executable):
    """The image writes this capability only after building the patched native code."""
    marker = Path(executable).parents[2] / 'share/rtabmap_odom/motus_rotation_hint_v1'
    try:
        if marker.read_text().strip() == 'motus_rotation_hint/1':
            return
    except OSError:
        pass
    raise RuntimeError('inertial association requires rebuilt RTAB-Map rotation-hint runtime')


class RotationHints:
    """Offer capture-stamped search/solver guesses without owning an IMU filter.

    All methods run under the session lock. A hint is optional even when this
    mode is enabled: cross-topic delivery does not guarantee that native code
    selects it. The native helper chooses its own last-successful-image anchor.
    """
    def __init__(self, calibration, node, segment, qos):
        self.enabled = association_enabled(calibration)
        self.frame, self.clock = calibration['frame_id'], calibration['clock_id']
        self.offered = self.skipped = 0
        self.offered_stamps = deque(maxlen=64)
        self.last_offered_stamp_ns = self.last_skip_reason = self.last_offered_generation = None
        self.controller = calibration.get('controller_rotation')
        self.controller_source = None
        self.last_hint_source = None
        if self.controller is not None:
            self.body_from_camera = np.array(self.controller['camera_from_body']) * [-1., -1., -1., 1.]
        self.controller_history = deque(maxlen=64)
        self.controller_generation = 0
        self.publisher = None
        if self.enabled:
            from geometry_msgs.msg import QuaternionStamped
            self.message_type = QuaternionStamped
            self.publisher = node.create_publisher(QuaternionStamped,
                '/' + segment + '/rotation_hint', qos)

    def observe_controller(self, packet, now_ns, *, aligned=False):
        if not self.enabled or not isinstance(packet, dict):
            return
        stamp = packet.get('stamp_ns')
        if type(stamp) is not int or stamp <= 0 or not -50_000_000 <= now_ns - stamp <= 750_000_000:
            return
        try:
            if aligned:
                clock = packet.get('clock_id')
                if self.controller is not None or not isinstance(clock, str) or not clock:
                    return
                # The producer has already mapped stamp_ns to this camera clock.
                body_from_camera = _unit_quaternion(packet.get('camera_from_body')) * [-1., -1., -1., 1.]
                source = (clock, packet.get('source_id'), packet.get('epoch'), tuple(body_from_camera))
            else:
                if self.controller is None or packet.get('timestamp_source') != 'read_completion':
                    return
                source = (self.controller['clock_id'],)
                body_from_camera = self.body_from_camera
                stamp += self.controller['read_time_offset_ns']
            world_from_camera = _multiply(_unit_quaternion(packet.get('quaternion')), body_from_camera)
        except (ValueError, TypeError, OverflowError):
            return
        if self.controller_history:
            last, previous = self.controller_history[-1]
            if source == self.controller_source and stamp == last:
                return
            if (source != self.controller_source or not 0 < stamp-last <= MAX_IMU_GAP_NS or
                    abs(previous @ world_from_camera) < np.cos(np.pi/12)):
                # A gap or discrete orientation jump may be a controller reset.
                self.controller_history.clear()
                self.controller_generation += 1
        self.controller_source = source
        self.controller_history.append((stamp, world_from_camera))

    def offer(self, attitude, stamp_ns, now_ns, *, available=True, capture_clock=None):
        if not self.enabled:
            return False
        if not available:
            return self._skip('assisted mode is unavailable')
        if type(stamp_ns) is not int or stamp_ns <= 0:
            return self._skip('hint requires an integer capture timestamp')
        quaternion = None
        if self.controller_source is not None and capture_clock == self.controller_source[0]:
            quaternion = interpolate_orientation(self.controller_history, stamp_ns)
        if quaternion is not None:
            source, generation = 'controller_rotation', self.controller_generation
        elif attitude is not None:
            source, generation = 'gyro_capture', attitude.rotation_generation
            quaternion = attitude.orientation_at(stamp_ns, now_ns, calibrated=True)
        if quaternion is None:
            return self._skip('capture has no valid IMU orientation bracket')
        identity = source, generation
        # Native can retain a successful image across intervening failed registrations.
        # Its anchor expires after 750 ms, even if all skipped images failed.
        if (type(stamp_ns) is int and self.last_offered_stamp_ns is not None
                and self.last_offered_generation != identity
                and stamp_ns-self.last_offered_stamp_ns <= 750_000_000):
            return self._skip('previous native hint may precede current IMU coverage')
        message = self.message_type()
        message.header.frame_id = self.frame
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 10**9)
        (message.quaternion.x, message.quaternion.y,
         message.quaternion.z, message.quaternion.w) = quaternion
        try:
            self.publisher.publish(message)
        except (RuntimeError, ValueError, TypeError):
            return self._skip('hint transport unavailable')
        self.offered += 1
        self.offered_stamps.append(stamp_ns)
        self.last_offered_generation = identity
        self.last_hint_source = source
        self.last_offered_stamp_ns, self.last_skip_reason = stamp_ns, None
        return True

    def _skip(self, reason):
        self.skipped += 1
        self.last_skip_reason = reason
        return False

    def info(self, stamp_ns=None, *, native=None):
        return dict(enabled=self.enabled, offered=self.offered, skipped=self.skipped,
                    offered_for_frame=stamp_ns in self.offered_stamps if stamp_ns is not None else None,
                    last_offered_stamp_ns=self.last_offered_stamp_ns,
                    last_skip_reason=self.last_skip_reason, native_applied=native,
                    last_hint_source=self.last_hint_source,
                    role='rotation_initial_guess',
                    correlated_visual_updates=self.enabled)


_NATIVE_STATS = re.compile(rb'rotation_hint: frames=(\d+) selected=(\d+) supplied=(\d+) '
                           rb'rejected=(\d+) late=(\d+) evicted=(\d+)')


def native_hint_statistics(log_path, *, tail=16384):
    """Last native hint counters from the estimator log; None until native reports.

    The patched native node logs these every 60 frames. `supplied` counts
    registrations whose search/PnP guess used the measured rotation.
    """
    try:
        with open(log_path, 'rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell()-tail))
            chunk = handle.read()
    except (OSError, TypeError, ValueError):
        return None
    matches = _NATIVE_STATS.findall(chunk)
    if not matches:
        return None
    frames, selected, supplied, rejected, late, evicted = (int(v) for v in matches[-1])
    return dict(frames=frames, selected=selected, supplied=supplied, rejected=rejected,
                late=late, evicted=evicted, supplied_fraction=round(supplied/frames, 3) if frames else None)
