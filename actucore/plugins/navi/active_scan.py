"""Bounded, odometry-confirmed observations for front-only local planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .local_path import observe, RESOLUTION, START, SHAPE, RIGHT, FORWARD

ANGLE = math.radians(15)
ANGLE_TOLERANCE = math.radians(2)
MAX_YAW_RATE = .25
RECEIPT_YAW_RATE = .18
RECEIPT_ANGLE = math.radians(12)
STILL_LINEAR = .05
STILL_YAW = .04
SETTLE_MS = 120
RECEIPT_SETTLE_MS = 3500
ODOM_GAP_MS = 250
MAP_AGE_MS = 2500
SCAN_TIMEOUT_MS = 8000


@dataclass(frozen=True)
class ScanCommand:
    yaw: float = 0.
    reason: str = "主动观察：停车等候"


@dataclass
class ActiveScan:
    """Reuse recent front views around one obstacle, with measured body motion."""

    phase: str = "idle"
    attempted: bool = False
    locked: bool = False
    directions: tuple = ()
    start_ms: int = 0
    origin: tuple = (0., 0., 0.)
    origin_set: bool = False
    still_since: int = 0
    settle_ms: int | None = None
    settle_frame: int = 0
    settle_frames: int = 0
    issued_yaw: bool = False
    pose: tuple = (0., 0., 0.)
    motion: tuple = (0., 0., 0.)
    poses: deque = field(default_factory=lambda: deque(maxlen=32))
    views: deque = field(default_factory=lambda: deque(maxlen=40))
    clear_frame: int = 0
    clear_frames: int = 0
    current_frame: int = 0
    pending_depth: dict | None = None
    side_frame: int = 0
    result_frame: int = 0
    receipt_only: bool = False
    source_frame: int | None = None
    odom_source: int | None = None

    @property
    def active(self):
        return self.phase != "idle"

    @property
    def ready(self):
        return self.phase == "evaluate"

    @property
    def has_result(self):
        return bool(self.current_frame and self.result_frame)

    @property
    def max_yaw_rate(self):
        return RECEIPT_YAW_RATE if self.receipt_only else MAX_YAW_RATE

    def update(self, depth, odom, geometry, *, receipt_only=False):
        """Update measured motion and the latest usable depth observation."""
        if self.receipt_only != receipt_only:
            self.invalidate()
            self.receipt_only = receipt_only
        stamp = (odom or {}).get("received_ms" if self.receipt_only else "stamp_ms")
        capture = (depth or {}).get("capture_ms")
        motion = tuple((odom or {}).get(axis) for axis in ("vx", "vy", "wz"))
        if (type(stamp) is not int or stamp <= 0 or geometry is None
                or any(v is None or not math.isfinite(v) for v in motion)):
            return self.invalidate()
        source = (odom or {}).get("source_sample_id") if self.receipt_only else None
        if source is not None and self.odom_source is not None and self.poses:
            if source < self.odom_source:
                return self.invalidate()
            if source == self.odom_source:
                previous = self.poses[-1][0]
                if not 0 <= stamp - previous <= ODOM_GAP_MS:
                    return self.invalidate()
                stamp = previous
        if self.poses:
            previous = self.poses[-1][0]
            if not 0 <= stamp - previous <= ODOM_GAP_MS:
                self.invalidate()
            elif stamp > previous:
                dt = (stamp - previous) / 1000
                vx, vy, wz = motion
                forward, right, heading = self.pose
                mid = heading - .5 * wz * dt
                self.pose = (forward + dt * (vx * math.cos(mid) + vy * math.sin(mid)),
                             right + dt * (vx * math.sin(mid) - vy * math.cos(mid)),
                             heading - dt * wz)
        if not self.poses or stamp > self.poses[-1][0]:
            self.poses.append((stamp, self.pose, motion[2]))
        self.odom_source = source
        self.motion = motion
        timeout = 2 * SCAN_TIMEOUT_MS if self.receipt_only else SCAN_TIMEOUT_MS
        if self.active and stamp - self.start_ms > timeout:
            return self.interrupt("主动观察：观测超时，停车")

        self.current_frame = 0
        if self.receipt_only:
            return self._update_latest(depth, geometry, stamp)
        if type(capture) is int and 0 < capture - stamp <= ODOM_GAP_MS:
            pending = self.pending_depth
            if pending is not None and pending["capture_ms"] <= stamp:
                self.pending_depth = depth
                depth, capture = pending, pending["capture_ms"]
            else:
                if pending is None:
                    self.pending_depth = depth
                return ScanCommand(reason="主动观察：等待里程计与深度对齐")
        else:
            self.pending_depth = None
        ground = (depth or {}).get("ground")
        if (type(capture) is not int or ground is None
                or depth.get("map") is None or not 0 <= stamp - capture <= 500):
            return self.interrupt()
        if self.views and capture < self.views[-1][0]:
            # Reject out-of-order observations without replacing the map.
            return self.interrupt()
        self.current_frame = capture
        if self.views and capture == self.views[-1][0]:
            return
        sample = self._pose_at(capture)
        if sample is None or not ground[::4, ::4].any():
            self.current_frame = 0
            return self.interrupt()
        pose, stable_heading = sample
        free, occupied = observe(depth["map"], geometry, RIGHT, FORWARD, RESOLUTION, ground)
        if not stable_heading:
            free = np.zeros_like(free)
        self.views.append((capture, pose, free, occupied))

    def _update_latest(self, depth, geometry, stamp):
        # Receipt times establish liveness and the stopped waiting interval.
        # The original header, when present, only identifies repeated frames.
        frame = (depth or {}).get("frame_id")
        source = (depth or {}).get("source_frame_id")
        if source is not None and self.source_frame is not None and self.views:
            if source < self.source_frame:
                return self.invalidate()
            if source == self.source_frame:
                frame = self.views[-1][0]
        ground = (depth or {}).get("ground")
        if (type(frame) is not int or abs(stamp - frame) > 500
                or ground is None or not ground[::4, ::4].any() or depth.get("map") is None):
            return self.interrupt()
        self.current_frame = frame
        if self.views and self.views[-1][0] == frame:
            return
        free, occupied = observe(depth["map"], geometry, RIGHT, FORWARD, RESOLUTION, ground)
        self.source_frame = source
        self.views.clear()
        self.views.append((frame, self.pose, free, occupied))

    def _pose_at(self, stamp):
        for index, (upper_time, upper, upper_yaw_rate) in enumerate(self.poses):
            if upper_time == stamp:
                return upper, abs(upper_yaw_rate) <= STILL_YAW
            if upper_time > stamp:
                if index == 0:
                    return None
                lower_time, lower, lower_yaw_rate = self.poses[index - 1]
                part = (stamp - lower_time) / (upper_time - lower_time)
                pose = tuple(lo + part * (hi - lo) for lo, hi in zip(lower, upper))
                return pose, max(abs(lower_yaw_rate), abs(upper_yaw_rate)) <= STILL_YAW
        return None

    def grid(self):
        """Reproject nearby views. Occupation wins, and free cells need support."""
        if self.receipt_only:
            _, _, free, occupied = self.views[-1]
            return free, occupied
        free = np.zeros(SHAPE, dtype=bool)
        occupied = np.zeros_like(free)
        stamp = self.poses[-1][0]
        forward, right, yaw = self.pose
        for capture, (before_f, before_r, before_yaw), view_free, view_occupied in self.views:
            if stamp - capture > MAP_AGE_MS:
                continue
            df, dr = forward - before_f, right - before_r
            offset_f = df * math.cos(before_yaw) + dr * math.sin(before_yaw)
            offset_r = -df * math.sin(before_yaw) + dr * math.cos(before_yaw)
            delta = yaw - before_yaw
            row = (FORWARD * math.cos(delta) - RIGHT * math.sin(delta) + offset_f) / RESOLUTION + START[0]
            col = (FORWARD * math.sin(delta) + RIGHT * math.cos(delta) + offset_r) / RESOLUTION + START[1]
            row, col = np.round(row, 7), np.round(col, 7)
            confirmed = np.ones_like(free)
            for y in (np.floor(row), np.ceil(row)):
                for x in (np.floor(col), np.ceil(col)):
                    valid = (y >= 0) & (y < SHAPE[0]) & (x >= 0) & (x < SHAPE[1])
                    indices = (y.clip(0, SHAPE[0] - 1).astype(int),
                               x.clip(0, SHAPE[1] - 1).astype(int))
                    confirmed &= valid & view_free[indices]
                    occupied |= valid & view_occupied[indices]
            free |= confirmed
        return free & ~occupied, occupied

    def begin(self, directions):
        if self.attempted or not self.current_frame:
            return False
        self.directions = directions
        self.attempted = True
        self.phase = "brake"
        self.origin_set = False
        self.start_ms = self.poses[-1][0]
        return True

    def try_other_side(self):
        self.directions = self.directions[1:]
        if self.directions:
            self.phase = "brake"
        else:
            self.interrupt()
        return self.active

    def finish(self):
        self.result_frame = self.side_frame
        self.phase = "idle"

    def command(self, geometry, half_width, yaw_rate, watchdog_ms, obs_age_ms=500):
        if not self.active or self.ready:
            return ScanCommand()
        stamp = self.poses[-1][0]
        speed = math.hypot(*self.motion[:2])
        if not self.current_frame:
            return self.interrupt("主动观察：观测超时，停车")
        if self.phase == "brake":
            if speed > STILL_LINEAR or abs(self.motion[2]) > STILL_YAW:
                return ScanCommand(reason="主动观察：等待底盘停稳")
            if not self.origin_set:
                self.origin, self.origin_set = self.pose, True
            self.still_since = stamp
            self.phase = "turn"
            return ScanCommand(reason="主动观察：确认底盘已停稳")
        if (speed > STILL_LINEAR
                or math.dist(self.pose[:2], self.origin[:2]) > .18):
            return self.interrupt("主动观察：检测到平移，停车")
        angle = RECEIPT_ANGLE if self.receipt_only else ANGLE
        if (abs(self.motion[2]) > yaw_rate + .10
                or abs(self.pose[2] - self.origin[2]) > angle + 2 * ANGLE_TOLERANCE):
            return self.interrupt("主动观察：转速或转角超出限制，停车")

        returning = self.phase == "return"
        heading = self.origin[2] if returning else self.origin[2] + self.directions[0] * angle
        error = heading - self.pose[2]
        if abs(error) <= ANGLE_TOLERANCE:
            if abs(self.motion[2]) > STILL_YAW:
                self.settle_ms = None
                return ScanCommand(reason="主动观察：等待转向停止")
            if self.settle_ms is None:
                self.settle_ms = stamp
                self.settle_frame = self.settle_frames = 0
            delay = RECEIPT_SETTLE_MS if self.receipt_only else SETTLE_MS
            if self.receipt_only and self.current_frame > max(self.settle_frame, self.settle_ms + delay):
                self.settle_frame = self.current_frame
                self.settle_frames += 1
            if (stamp - self.settle_ms < delay or self.current_frame <= self.settle_ms
                    or (self.receipt_only and self.settle_frames < 3)):
                return ScanCommand(reason="主动观察：等待转稳后的新深度")
            self.settle_ms = None
            if returning:
                self.issued_yaw = False
            else:
                self.side_frame = self.current_frame
            self.phase = "evaluate" if returning or self.receipt_only else "return"
            return ScanCommand(reason="主动观察：已记录新视角")
        self.settle_ms = None
        if stamp - self.still_since < SETTLE_MS // 2:
            return ScanCommand(reason="主动观察：确认底盘已停稳")

        # Budget tracking latency and the watchdog, including a sink poll.
        horizon = (watchdog_ms + obs_age_ms + 50) / 1000
        sweep = math.copysign(abs(error) + (yaw_rate + .10) * horizon, error)
        if not geometry.front > 0 or not geometry.rear > 0 or not yaw_rate > 0:
            return self.interrupt()
        if not self._clear_sweep(geometry, half_width, sweep, STILL_LINEAR * horizon):
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            return self.interrupt("主动观察：转身扫掠区有障碍或超出网格范围，停车")
        self.issued_yaw = True
        direction = "右" if error > 0 else "左"
        description = "回到观察前的朝向" if returning else f"向{direction}观察"
        return ScanCommand(-math.copysign(yaw_rate, error), f"主动观察：{description}")

    def _clear_sweep(self, geometry, half_width, sweep, translation):
        # Rotate the rectangle in small increments and include cell diagonal
        # plus the greatest displacement between successive angular samples.
        samples = max(2, math.ceil(abs(sweep) / .04) + 1)
        radius = math.hypot(max(geometry.front, geometry.rear), half_width)
        slack = translation + RESOLUTION / math.sqrt(2) + radius * abs(sweep) / (2 * (samples - 1))
        _, occupied = self.grid()
        occupied = occupied | self.views[-1][3]
        swept = np.zeros(SHAPE, dtype=bool)
        for angle in np.linspace(0., sweep, samples):
            c, s = math.cos(angle), math.sin(angle)
            extent_f = max(geometry.front * c, -geometry.rear * c) + half_width * abs(s) + slack
            extent_back = max(geometry.rear * c, -geometry.front * c) + half_width * abs(s) + slack
            extent_r = max(geometry.front, geometry.rear) * abs(s) + half_width * abs(c) + slack
            if (extent_f > FORWARD[-1, 0] or extent_r > RIGHT[0, -1]
                    or extent_back > -FORWARD[0, 0]):
                return False
            forward = FORWARD * c + RIGHT * s
            right = -FORWARD * s + RIGHT * c
            outside_f = np.maximum(0., np.maximum(-geometry.rear - forward, forward - geometry.front))
            outside_r = np.maximum(0., abs(right) - half_width)
            swept |= outside_f**2 + outside_r**2 <= slack**2
        return not (occupied & swept).any()

    def interrupt(self, reason="主动观察：深度或里程计不可用，停车"):
        if self.active:
            self.locked |= self.issued_yaw
        self.phase = "idle"
        self.clear_frames = 0
        return ScanCommand(reason=reason)

    def invalidate(self):
        self.current_frame = 0
        self.pending_depth = None
        self.source_frame = None
        self.odom_source = None
        self.poses.clear()
        self.views.clear()
        self.pose = (0., 0., 0.)
        return self.interrupt()

    def note_clear_path(self, frame):
        if self.locked or not self.attempted or not frame or frame <= self.clear_frame:
            return
        self.clear_frame = frame
        self.clear_frames += 1
        if self.clear_frames >= 3:
            self.attempted = False
            self.issued_yaw = False
            self.clear_frames = 0
            self.result_frame = 0
