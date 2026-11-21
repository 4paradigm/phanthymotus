"""Optional-source format evidence, independent of an odometry segment."""
from copy import deepcopy

from .rgbd_imu import ImuAttitude, translation_extrinsics


# Session scheduling budgets; IMU packet admission has no source-health dwell.
VISUAL_INITIALIZATION_S = 15.
ASSISTANCE_RETRY_S = 20.
TRACKING_RESET_S = 5.


def assistance_mode(calibration):
    if calibration.get('controller_rotation') and not calibration.get('short_term_compensation'):
        return 'controller_rotation'
    if not calibration.get('imu_topic'):
        return 'visual_only'
    if calibration.get('short_term_compensation'):
        return 'eskf_depth' if calibration.get('depth_geometry') else 'eskf'
    return 'imu_attitude'


def visual_calibration(calibration):
    """A replacement native process must actually disable its IMU wait path."""
    result = dict(calibration)
    result.pop('imu_topic', None)
    for flag in ('imu_assisted', 'short_term_compensation', 'depth_geometry', 'inertial_association'):
        if flag in result:
            result[flag] = False
    return result


class AuxiliaryHealthProbe:
    """Use the active path's numeric format rules before replacing a session."""
    def __init__(self, frame, clock, *, require_translation=False):
        self.frame, self.clock = frame, clock
        self.require_translation = require_translation
        self.translation_ready = False
        self.validator = None
        self.first_seen = self.last_seen = self.first_capture = None
        self.reason = 'waiting for usable auxiliary samples'
        self.resets = 0

    def reset(self, reason):
        self.validator = None
        self.first_seen = self.last_seen = self.first_capture = None
        self.reason = str(reason)
        self.resets += 1

    def observe(self, packet, now_ns, now_s):
        try:
            if not isinstance(packet, dict):
                raise ValueError('auxiliary packet must be an object')
            if self.validator is None:
                self.validator = ImuAttitude(self.frame, self.clock)
            sample = self.validator.observe(packet, now_ns)
            self.last_seen = now_s
            if sample is None:
                self.first_seen = self.first_capture = None
                self.reason = 'waiting for usable camera IMU gravity'
                return
            if self.first_seen is None:
                self.first_seen, self.first_capture = now_s, sample['stamp_ns']
            if self.require_translation:
                try:
                    translation_extrinsics(packet)
                except ValueError as error:
                    self.translation_ready = False
                    self.reason = str(error)
                    return
                self.translation_ready = True
            self.reason = 'IMU numeric format accepted'
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.reset(error)

    def info(self, now_ns, now_s):
        healthy = 0.
        validator = self.validator
        if validator is not None and self.first_seen is not None:
            healthy = max(0., min(now_s-self.first_seen,
                (validator.last_stamp-self.first_capture)/1e9))
        attitude_ready = validator is not None and validator.state == 'active'
        ready = attitude_ready and (not self.require_translation or self.translation_ready)
        return dict(enabled=True, attitude_ready=attitude_ready, status='ready' if ready else 'warming' if self.first_seen is not None else 'waiting',
                    reason='IMU numeric format ready for a new segment' if ready else self.reason,
                    healthy_for_s=round(healthy, 3), required_healthy_s=0.,
                    last_measurement_ns=validator.last_stamp if validator else None,
                    buffered_samples=len(validator.history) if validator else 0,
                    source=deepcopy(validator.provenance) if validator else None, resets=self.resets)
