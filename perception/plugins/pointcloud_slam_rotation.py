"""Capture-bracketed gyro rotation for native SLAM initialization."""
from collections import deque
from itertools import islice
import math
import re
import struct

import numpy as np

from .rgbd_imu import (MAX_IMU_GAP_NS, _multiply, _rotation, _scale, _vector,
                       imu_episode, imu_timestamps, interpolate_orientation)


class SlamRotationHint:
    """The deployment profile supplies the calibrated gyro-to-base rotation."""
    def __init__(self, profile):
        if not isinstance(profile, dict):
            raise ValueError('SLAM rotation hint profile must be an object')
        self.topic = profile.get('topic')
        if not isinstance(self.topic, str) or not re.fullmatch(r'(/[A-Za-z_][A-Za-z_0-9]*)+', self.topic):
            raise ValueError('SLAM rotation hint requires an absolute gyro topic')
        self.child_frame, self.gyro_frame = profile.get('child_frame_id'), profile.get('gyro_frame_id')
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.child_frame, self.gyro_frame)):
            raise ValueError('SLAM rotation hint requires the base frame and gyro frame')
        self.child_from_gyro = _rotation(profile.get('child_from_gyro'))
        self.history = deque(maxlen=512)
        self.generation = 0
        self.source = (None, None)
        self.last_rate = None

    def observe(self, packet):
        if (not isinstance(packet, dict) or packet.get('schema') != 'motus.imu/1' or
                packet.get('gyro_frame_id') != self.gyro_frame or
                packet.get('time_basis', 'source_capture') != 'source_capture' or
                type(packet.get('stamp_ns')) is not int):
            self.history.clear()
            return
        try:
            stamp, _ = imu_timestamps(packet, None)
            scale = _scale(packet, 'gyro_scale_rad_s')
            rate = self.child_from_gyro @ _vector(packet.get('angular_velocity')) * scale
            if stamp <= 0 or not np.isfinite(rate).all():
                raise ValueError('invalid gyro sample')
        except (TypeError, ValueError, OverflowError):
            self.history.clear()
            return
        source = (*imu_episode(packet, self.source[:2]), packet.get('time_offset_ns', 0), scale)
        dt = stamp-self.history[-1][0] if self.history else 0
        if self.history and self.source == source and dt == 0:
            return
        if self.source != source or not 0 < dt <= MAX_IMU_GAP_NS:
            if self.source != source or dt < 0:
                self.history.clear()
            # Keep usable brackets for delayed clouds, with a new generation
            # on the far side of the gap so relative rotations cannot cross it.
            self.generation += 1
            orientation = np.array([0., 0., 0., 1.])
        else:
            turn = self.last_rate * (.5e-9 * dt) + rate * (.5e-9 * dt)
            angle = math.hypot(*turn)
            delta = np.r_[turn * (.5 * np.sinc(angle / (2 * np.pi))), np.cos(angle / 2)]
            orientation = _multiply(self.history[-1][1], delta)
            orientation /= np.linalg.norm(orientation)
        self.source, self.last_rate = source, rate
        self.history.append((stamp, orientation, self.generation))

    def payload(self, stamp):
        adjacent = zip(self.history, islice(self.history, 1, None))
        for (before, a, generation), (after, b, next_generation) in adjacent:
            if stamp < before:
                break
            if before <= stamp <= after and generation == next_generation:
                orientation = interpolate_orientation(((before, a), (after, b)), stamp)
                return struct.pack('<8sQ4d', b'MOTUSRH1', generation, *orientation)
        return b''
