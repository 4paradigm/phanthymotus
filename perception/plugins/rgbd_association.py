"""Optional rotation hints for the pinned native visual registration frontend."""
from collections import deque
from pathlib import Path
import re


def association_enabled(calibration):
    return (calibration.get('inertial_association') is True and
            calibration.get('short_term_compensation') is True and
            bool(calibration.get('imu_topic')))


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
        self.publisher = None
        if self.enabled:
            from geometry_msgs.msg import QuaternionStamped
            self.message_type = QuaternionStamped
            self.publisher = node.create_publisher(QuaternionStamped,
                '/' + segment + '/rotation_hint', qos)

    def offer(self, attitude, stamp_ns, now_ns, *, available=True):
        if not self.enabled:
            return False
        if not available:
            return self._skip('assisted mode is unavailable')
        if attitude is None:
            return self._skip('IMU orientation is unavailable')
        # Native can retain a successful image across intervening failed registrations.
        # Its anchor expires after 750 ms, even if all skipped images failed.
        if (type(stamp_ns) is int and self.last_offered_stamp_ns is not None
                and self.last_offered_generation != attitude.rotation_generation
                and stamp_ns-self.last_offered_stamp_ns <= 750_000_000):
            return self._skip('previous native hint may precede current IMU coverage')
        quaternion = attitude.orientation_at(stamp_ns, now_ns, calibrated=True)
        if quaternion is None:
            return self._skip('capture has no valid IMU orientation bracket')
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
        self.last_offered_generation = attitude.rotation_generation
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
