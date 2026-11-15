"""Bounded optional-source health evidence, independent of an odometry segment."""
from copy import deepcopy

import numpy as np

from .rgbd_imu import ImuAttitude, MAX_CAPTURE_AGE_NS


# Engineering budgets. Recovery still needs fresh, validated capture samples;
# elapsed wall time alone is never evidence of recovered sensor health.
VISUAL_INITIALIZATION_S = 15.
AUXILIARY_HEALTH_S = 3.
ASSISTANCE_RETRY_S = 20.
TRACKING_RESET_S = 5.


def assistance_mode(calibration):
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
    """Validate recovery evidence without reviving or feeding a retired filter.

    A fresh validator may learn a changed epoch/calibration, but only a new
    session may use that source. Both actual capture duration and monotonic
    arrival duration must cover the dwell; a burst of old samples cannot pass.
    """
    def __init__(self, frame, clock, *, require_translation=False):
        self.frame, self.clock = frame, clock
        self.require_translation = require_translation
        self.validator = None
        self.first_seen = self.last_seen = self.first_capture = None
        self.translations = self.provenance = None
        self.reason = 'waiting for usable auxiliary samples'
        self.resets = 0

    def reset(self, reason):
        self.validator = None
        self.first_seen = self.last_seen = self.first_capture = None
        self.translations = self.provenance = None
        self.reason = str(reason)
        self.resets += 1

    def _expire(self, now_ns, now_s):
        if self.validator is not None and (
                self.validator.status(now_ns) == 'lost' or
                (self.last_seen is not None and (now_s < self.last_seen or
                 now_s-self.last_seen > MAX_CAPTURE_AGE_NS/1e9))):
            self.reset('auxiliary capture or arrival stale; health dwell restarted')

    def observe(self, packet, now_ns, now_s):
        self._expire(now_ns, now_s)
        try:
            if not isinstance(packet, dict):
                raise ValueError('auxiliary packet must be an object')
            translations = None
            if self.require_translation:
                if packet.get('translation_source') not in ('tf', 'sdk_factory_extrinsics'):
                    raise ValueError('calibrated IMU translation required for compensation recovery')
                values = [np.asarray(packet.get(key), dtype=float) for key in
                          ('camera_from_gyro_translation', 'camera_from_accel_translation')]
                if any(v.shape != (3,) or not np.isfinite(v).all() for v in values):
                    raise ValueError('finite calibrated IMU translations required for compensation recovery')
                translations = np.r_[values[0], values[1]]
            provenance = (packet.get('translation_source') if self.require_translation else None,
                          deepcopy(packet.get('clock_mapping')))
            if ((self.provenance is not None and provenance != self.provenance) or
                    (self.translations is not None and not np.allclose(
                        translations, self.translations, atol=1e-7, rtol=0))):
                raise ValueError('auxiliary calibration or clock mapping changed; health dwell restarted')
            if self.validator is None:
                self.validator = ImuAttitude(self.frame, self.clock)
            sample = self.validator.observe(packet, now_ns)
            self.provenance, self.translations = provenance, translations
            self.last_seen = now_s
            if sample is None:
                self.first_seen = self.first_capture = None
                self.reason = 'waiting for usable camera IMU gravity'
                return
            if self.first_seen is None:
                self.first_seen, self.first_capture = now_s, sample['stamp_ns']
            self.reason = 'collecting continuous auxiliary capture evidence'
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            # A changed source must build a new dwell starting with a later
            # packet; this rejected packet is not silently assimilated twice.
            self.reset(error)

    def info(self, now_ns, now_s):
        self._expire(now_ns, now_s)
        healthy = 0.
        validator = self.validator
        if validator is not None and self.first_seen is not None:
            healthy = max(0., min(now_s-self.first_seen,
                (validator.last_stamp-self.first_capture)/1e9))
        ready = healthy >= AUXILIARY_HEALTH_S
        return dict(enabled=True, status='ready' if ready else 'warming' if self.first_seen is not None else 'waiting',
                    reason='continuous auxiliary source ready for a new segment' if ready else self.reason,
                    healthy_for_s=round(healthy, 3), required_healthy_s=AUXILIARY_HEALTH_S,
                    last_measurement_ns=validator.last_stamp if validator else None,
                    buffered_samples=len(validator.history) if validator else 0,
                    source=deepcopy(validator.provenance) if validator else None, resets=self.resets)
