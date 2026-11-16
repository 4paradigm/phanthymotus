"""Optional rotation hints for the pinned native visual registration frontend."""
from pathlib import Path


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
        self.last_offered_stamp_ns = self.last_skip_reason = None
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
        quaternion = attitude.orientation_at(stamp_ns, now_ns)
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
        self.last_offered_stamp_ns, self.last_skip_reason = stamp_ns, None
        return True

    def _skip(self, reason):
        self.skipped += 1
        self.last_skip_reason = reason
        return False

    def info(self):
        return dict(enabled=self.enabled, offered=self.offered, skipped=self.skipped,
                    last_offered_stamp_ns=self.last_offered_stamp_ns,
                    last_skip_reason=self.last_skip_reason, native_applied=None,
                    role='rotation_initial_guess',
                    correlated_visual_updates=self.enabled)
