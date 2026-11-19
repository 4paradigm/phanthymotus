"""Bounded, odometry-confirmed observations for front-only local planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .local_path import observe, RESOLUTION, START, SHAPE, RIGHT, FORWARD
from .odom import PlanarPose, planar_feedback

ANGLE = math.radians(20)
ANGLE_TOLERANCE = math.radians(2)
STILL_LINEAR = .05
STILL_YAW = .04
SETTLE_MS = 120
RECEIPT_SETTLE_MS = 3500
ODOM_GAP_MS = 250
MAP_AGE_MS = 2500
SCAN_TIMEOUT_MS = 8000
POSE_TURN_BURST_MS = 400
TURN_PROGRESS_WINDOW_MS = 1500
TURN_PROGRESS_MIN_RAD = math.radians(3)


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
    turn_start_heading: float = 0.
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
    _pending_projection: tuple | None = None
    side_frame: int = 0
    advance_m: float | None = None
    advance_origin: tuple | None = None
    advance_commanded_m: float = 0.
    result_frame: int = 0
    receipt_only: bool = False
    source_frame: int | None = None
    odom_source: int | None = None
    pose_origin: PlanarPose | None = None
    pose_feedback_since_ms: int = 0
    turn_stop_ms: int | None = None
    turn_until_ms: int = 0
    turn_yaw: float = 0.
    command_only: bool = False
    single_control_ms: int = 0
    single_depth_ms: int = 0
    single_capture_ms: int | None = None
    single_source_id: int | None = None
    single_source_received_ms: int = 0
    timed_phase_s: float = 0.
    timed_total_s: float = 0.
    timed_turn_s: float = 0.

    @property
    def active(self):
        return self.phase != "idle"

    @property
    def ready(self):
        return self.phase == "evaluate"

    @property
    def has_result(self):
        return bool(self.current_frame and self.result_frame)

    def update(self, depth, odom, geometry, *, receipt_only=False, max_depth_age_ms=500):
        """Update measured motion and the latest usable depth observation."""
        if self.receipt_only != receipt_only:
            self.invalidate()
            self.receipt_only = receipt_only
        stamp = (odom or {}).get("received_ms" if self.receipt_only else "stamp_ms")
        capture = (depth or {}).get("capture_ms")
        motion = planar_feedback(odom)
        if type(stamp) is not int or stamp <= 0 or geometry is None or motion is None:
            if self.receipt_only and self.active and self.issued_yaw and geometry is not None:
                return self.hold_for_replan("主动观察：运动反馈暂不可用，停车后重规划")
            return self.invalidate()
        body_pose = (odom or {}).get("motion_pose")
        if self.poses and ((body_pose is None) != (self.pose_origin is None)
                           or (body_pose is not None and body_pose.frame != self.pose_origin.frame)):
            self.invalidate()
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
                if self.receipt_only and self.active and self.issued_yaw and stamp > previous:
                    # A slow control tick can skip valid odometry samples.
                    # Brake instead of integrating motion over the gap.
                    self.poses.clear()
                    return self.hold_for_replan("主动观察：运动反馈间隔过长，停车后重规划")
                self.invalidate()
            elif stamp > previous and body_pose is None:
                dt = (stamp - previous) / 1000
                vx, vy, wz = motion
                forward, right, heading = self.pose
                mid = heading - .5 * wz * dt
                self.pose = (forward + dt * (vx * math.cos(mid) + vy * math.sin(mid)),
                             right + dt * (vx * math.sin(mid) - vy * math.cos(mid)),
                             heading - dt * wz)
        if body_pose is not None:
            self.pose_origin = self.pose_origin or body_pose
            forward, left, yaw = body_pose.relative_to(self.pose_origin)
            self.pose = (forward, -left, -yaw)
            self.pose_feedback_since_ms = odom["pose_feedback_since_ms"]
        if not self.poses or stamp > self.poses[-1][0]:
            self.poses.append((stamp, self.pose, motion[2]))
        self.odom_source = source
        self.motion = motion
        timeout = 2 * SCAN_TIMEOUT_MS if self.receipt_only else SCAN_TIMEOUT_MS
        if self.active and self.advance_origin is None and stamp - self.start_ms > timeout:
            return self.interrupt("主动观察：观测超时，停车")

        self.current_frame = 0
        if self.receipt_only:
            return self._update_latest(depth, geometry, stamp, max_depth_age_ms)
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
                or depth.get("map") is None or not 0 <= stamp - capture <= max_depth_age_ms):
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

    def _update_latest(self, depth, geometry, stamp, max_depth_age_ms):
        # Receipt times establish liveness and the stopped waiting interval.
        # The original header, when present, only identifies repeated frames.
        frame = (depth or {}).get("frame_id")
        source = (depth or {}).get("source_frame_id")
        if source is not None and self.source_frame is not None and self.views:
            if source < self.source_frame:
                return self.invalidate()
            if source == self.source_frame:
                if frame != self.views[-1][0] and stamp - self.views[-1][0] > max_depth_age_ms:
                    return self.interrupt("主动观察：收到过期的重复深度帧，停车")
                frame = self.views[-1][0]
        ground = (depth or {}).get("ground")
        if (type(frame) is not int or abs(stamp - frame) > max_depth_age_ms
                or ground is None or not ground[::4, ::4].any() or depth.get("map") is None):
            if self.active and self.issued_yaw and not self.command_only:
                return self.hold_for_replan("主动观察：深度暂不可用，停车后重规划")
            return self.interrupt()
        self.current_frame = frame
        if self.views and self.views[-1][0] == frame:
            return
        # Only the latest receipt-time view can contribute to the grid.
        # Project it when a turn or local plan needs it.
        self._pending_projection = (depth["map"], geometry, ground)
        self.source_frame = source
        self.views.clear()
        self.views.append((frame, self.pose, None, None))

    def observe_single(self, depth, geometry, max_depth_age_ms, now_ms):
        """Use the latest single view, preserving the receipt of repeated frames."""
        if not self.receipt_only:
            self.invalidate()
            self.receipt_only = True
        self.command_only = True
        self.single_control_ms = now_ms
        self.current_frame = 0
        self.single_depth_ms = 0
        self.single_capture_ms = None
        frame = (depth or {}).get("frame_id")
        source = (depth or {}).get("source_frame_id")
        if type(source) is int and self.single_source_id is not None:
            if source < self.single_source_id:
                return self.interrupt()
            if source == self.single_source_id:
                frame = self.single_source_received_ms
        if type(frame) is not int or not 0 <= now_ms - frame <= max_depth_age_ms:
            return self.interrupt()
        if type(source) is int:
            self.single_source_id, self.single_source_received_ms = source, frame
        self.single_depth_ms = frame
        self.single_capture_ms = depth.get("capture_ms")
        return self._update_latest(dict(depth, frame_id=frame), geometry, now_ms, max_depth_age_ms)

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
            frame, pose, free, occupied = self.views[-1]
            if self._pending_projection is not None:
                depth, geometry, ground = self._pending_projection
                free, occupied = observe(depth, geometry, RIGHT, FORWARD, RESOLUTION, ground)
                self.views[-1] = (frame, pose, free, occupied)
                self._pending_projection = None
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

    def begin(self, directions, *, advance_m=None):
        if self.attempted or not self.current_frame:
            return False
        self.directions = directions
        self.advance_m, self.advance_origin, self.advance_commanded_m = advance_m, None, 0.
        self.attempted = True
        self.phase = "brake"
        self.origin_set = False
        self.start_ms = self.single_control_ms if self.command_only else self.poses[-1][0]
        self.still_since = self.start_ms
        self.timed_phase_s = self.timed_total_s = self.timed_turn_s = 0.
        return True

    def try_other_side(self):
        self.directions = self.directions[1:]
        if self.directions:
            self.phase = "brake"
            self.still_since = self.single_control_ms if self.command_only else self.poses[-1][0]
            self.timed_phase_s = self.timed_turn_s = 0.
        else:
            self.interrupt()
        return self.active

    def finish(self):
        self.result_frame = self.side_frame
        self.phase = "idle"

    def command(self, geometry, half_width, yaw_rate, watchdog_ms, obs_age_ms=500, *,
                min_yaw_rate=0., now_ms=None, tick_s=.1):
        if not self.active or self.ready:
            return ScanCommand()
        if self.command_only:
            return self._timed_command(geometry, half_width, yaw_rate, watchdog_ms,
                                       obs_age_ms, min_yaw_rate, tick_s)
        stamp = self.poses[-1][0]
        speed = math.hypot(*self.motion[:2])
        if not self.current_frame:
            return self.interrupt("主动观察：观测超时，停车")
        if self.phase == "brake":
            if (speed > STILL_LINEAR or abs(self.motion[2]) > STILL_YAW
                    or (self.pose_origin is not None and self.pose_feedback_since_ms < self.still_since)):
                return ScanCommand(reason="主动观察：等待底盘停稳")
            if not self.origin_set:
                self.origin, self.origin_set = self.pose, True
            self.still_since = stamp
            self.turn_start_heading = self.pose[2]
            self.phase = "turn"
            return ScanCommand(reason="主动观察：确认底盘已停稳")
        if math.dist(self.pose[:2], self.origin[:2]) > .18:
            return self.interrupt("主动观察：检测到平移，停车")
        if self.receipt_only and (speed > STILL_LINEAR or self.phase == "brake_for_replan"):
            self.phase = "brake_for_replan"
            if speed > STILL_LINEAR or abs(self.motion[2]) > STILL_YAW:
                self.settle_ms = None
                return ScanCommand(reason="主动观察：检测到平移，停车后重规划")
            if self.settle_ms is None:
                self.settle_ms = stamp
            if stamp - self.settle_ms < SETTLE_MS or self.current_frame <= self.settle_ms:
                return ScanCommand(reason="主动观察：等待停稳后的观测")
            # The interrupted sweep did not produce a completed scan. Resume
            # local planning from this view without retaining a scan result.
            self.phase = "idle"
            self.issued_yaw = False
            return ScanCommand(reason="主动观察：已停稳，重新规划")
        if speed > STILL_LINEAR:
            return self.interrupt("主动观察：检测到平移，停车")
        if (abs(self.motion[2]) > yaw_rate + .10
                or abs(self.pose[2] - self.origin[2]) > ANGLE + 2 * ANGLE_TOLERANCE):
            return self.interrupt("主动观察：转速或转角超出限制，停车")
        if (self.receipt_only and self.phase == "turn" and self.issued_yaw
                and stamp - self.still_since >= TURN_PROGRESS_WINDOW_MS
                and abs(self.pose[2] - self.turn_start_heading) < TURN_PROGRESS_MIN_RAD):
            return self.hold_for_replan("主动观察：转向无实测进展，停车后重规划")

        # Pose feedback can lag several ticks. Run a bounded burst, then wait
        # for measurements that bracket its explicit stop.
        control_ms = now_ms or stamp
        if self.turn_until_ms and control_ms >= self.turn_until_ms:
            self.turn_stop_ms = control_ms
            self.turn_until_ms = 0
        if self.turn_stop_ms is not None:
            if self.pose_feedback_since_ms < self.turn_stop_ms or abs(self.motion[2]) > STILL_YAW:
                return ScanCommand(reason="主动观察：停转等待新位姿")
            self.turn_stop_ms = None

        returning = self.phase == "return"
        heading = self.origin[2] if returning else self.origin[2] + self.directions[0] * ANGLE
        error = heading - self.pose[2]
        if self.turn_until_ms and (abs(error) <= ANGLE_TOLERANCE or self.turn_yaw * error > 0):
            self.turn_stop_ms = control_ms
            self.turn_until_ms = 0
            return ScanCommand(reason="主动观察：停转等待新位姿")
        if abs(error) <= ANGLE_TOLERANCE:
            if abs(self.motion[2]) > STILL_YAW:
                self.settle_ms = None
                return ScanCommand(reason="主动观察：等待转向停止")
            if self.settle_ms is None:
                self.settle_ms = stamp
            if stamp - self.settle_ms < SETTLE_MS or self.current_frame <= self.settle_ms:
                return ScanCommand(reason="主动观察：等待转稳后的新深度")
            self.settle_ms = None
            if returning:
                self.issued_yaw = False
            else:
                self.side_frame = self.current_frame
            self.phase = "evaluate" if returning or self.receipt_only or self.advance_m is not None else "return"
            return ScanCommand(reason="主动观察：已记录新视角")
        self.settle_ms = None
        if stamp - self.still_since < SETTLE_MS // 2:
            return ScanCommand(reason="主动观察：确认底盘已停稳")

        # Budget tracking latency and the watchdog, including a sink poll.
        horizon = (watchdog_ms + obs_age_ms + 50) / 1000
        if not geometry.front > 0 or not geometry.rear > 0 or not yaw_rate > 0 or yaw_rate < min_yaw_rate:
            return self.interrupt()
        if self.motion[2] * math.copysign(1., error) > STILL_YAW:
            return ScanCommand(reason="主动观察：等待反向转动停止")

        if self.pose_origin is not None:
            if self.turn_until_ms:
                duration_ms = self.turn_until_ms - control_ms
                yaw_rate = min(yaw_rate, abs(self.turn_yaw))
            else:
                duration_ms = POSE_TURN_BURST_MS
                yaw_rate = min(yaw_rate, max(min_yaw_rate, abs(error) * 1000 / duration_ms))
                duration_ms = min(duration_ms, int((abs(error) + 2 * ANGLE_TOLERANCE) * 1000 / yaw_rate))
            if duration_ms < max(1, int(tick_s * 1000)):
                self.turn_until_ms = 0
                self.turn_stop_ms = control_ms
                return ScanCommand(reason="主动观察：停转等待新位姿")
        elif self.advance_m is not None and tick_s > 0.:
            yaw_rate = min(yaw_rate, max(min_yaw_rate, abs(error) / tick_s))
        _, occupied = self.grid()
        occupied = occupied | self.views[-1][3]

        def clear_sweep(rate):
            turn_rate = max(rate, abs(self.motion[2]))
            sweep = math.copysign(abs(error) + (turn_rate + .10) * horizon, error)
            return self.clear_sweep(geometry, half_width, sweep, STILL_LINEAR * horizon,
                                    occupied=occupied)

        if not clear_sweep(yaw_rate):
            limit = yaw_rate
            yaw_rate = 0.
            if clear_sweep(min_yaw_rate):
                yaw_rate = min_yaw_rate
                for _ in range(5):
                    candidate = (yaw_rate + limit) / 2
                    if clear_sweep(candidate):
                        yaw_rate = candidate
                    else:
                        limit = candidate
        if not yaw_rate:
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            return self.interrupt("主动观察：转身扫掠区有障碍，停车")
        self.issued_yaw = True
        if self.pose_origin is not None:
            if not self.turn_until_ms:
                self.turn_until_ms = control_ms + duration_ms
            self.turn_yaw = -math.copysign(yaw_rate, error)
        direction = "右" if error > 0 else "左"
        description = "回到观察前的朝向" if returning else f"向{direction}观察"
        return ScanCommand(-math.copysign(yaw_rate, error), f"主动观察：{description}")

    def _timed_command(self, geometry, half_width, yaw_rate, watchdog_ms, obs_age_ms,
                       min_yaw, tick_s):
        self.timed_total_s += tick_s
        self.timed_phase_s += tick_s
        if not self.current_frame or self.timed_total_s * 1000 > 2 * SCAN_TIMEOUT_MS:
            return self.interrupt("主动观察：无反馈观察超时或深度失效，停车")
        if self.phase == "brake":
            if self.timed_phase_s * 1000 >= watchdog_ms + 50:
                self.phase = "turn"
                self.timed_phase_s = 0.
            return ScanCommand(reason="主动观察：停车，等待制动窗口")
        if self.phase == "settle":
            after = self.settle_ms + RECEIPT_SETTLE_MS
            captured = self.single_capture_ms or self.current_frame
            if captured > after and self.current_frame > self.settle_frame:
                self.settle_frame = self.current_frame
                self.settle_frames += 1
            if self.timed_phase_s * 1000 >= RECEIPT_SETTLE_MS and self.settle_frames >= 3:
                self.side_frame = self.current_frame
                self.phase = "evaluate"
                return ScanCommand(reason="主动观察：收到新视角，准备重新规划")
            return ScanCommand(reason="主动观察：停车等待新的独立深度帧")

        if not self.timed_turn_s:
            if yaw_rate <= 0. or yaw_rate < min_yaw:
                return self.interrupt("主动观察：底盘无可用转速，停车")
            duration = max(POSE_TURN_BURST_MS / 1000, ANGLE / yaw_rate)
            if tick_s > 0.:
                duration = math.ceil(duration / tick_s) * tick_s
            if min_yaw:
                duration = min(duration, ANGLE / min_yaw)
            self.timed_turn_s = duration
            self.turn_yaw = min(yaw_rate, ANGLE / duration)
            self.timed_phase_s = 0.
        if self.timed_phase_s + tick_s > self.timed_turn_s + 1e-9:
            self.phase = "settle"
            self.timed_phase_s = 0.
            self.settle_ms = self.single_control_ms
            self.settle_frames = self.settle_frame = 0
            return ScanCommand(reason="主动观察：定时转向结束，停车")
        horizon = (watchdog_ms + obs_age_ms + 50) / 1000
        sweep = self.directions[0] * (ANGLE + (self.turn_yaw + .10) * horizon)
        if (self.turn_yaw <= 0. or self.turn_yaw < min_yaw or not self.clear_sweep(
                geometry, half_width, sweep, STILL_LINEAR * horizon)):
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            return self.interrupt("主动观察：转身扫掠区有障碍，停车")
        self.issued_yaw = True
        return ScanCommand(-self.directions[0] * self.turn_yaw, "主动观察：有界短转向")

    def clear_sweep(self, geometry, half_width, sweep, translation, *, occupied=None):
        # Rotate the rectangle in small increments and include cell diagonal
        # plus the greatest displacement between successive angular samples.
        samples = max(2, math.ceil(abs(sweep) / .04) + 1)
        radius = math.hypot(max(geometry.front, geometry.rear), half_width)
        slack = translation + RESOLUTION / math.sqrt(2) + radius * abs(sweep) / (2 * (samples - 1))
        if occupied is None:
            _, occupied = self.grid()
            occupied = occupied | self.views[-1][3]
        occupied_forward, occupied_right = FORWARD[occupied], RIGHT[occupied]
        # An area outside this view is unknown. Only observed hits can block a turn.
        for angle in np.linspace(0., sweep, samples):
            c, s = math.cos(angle), math.sin(angle)
            forward = occupied_forward * c + occupied_right * s
            right = -occupied_forward * s + occupied_right * c
            outside_f = np.maximum(0., np.maximum(-geometry.rear - forward, forward - geometry.front))
            outside_r = np.maximum(0., abs(right) - half_width)
            if (outside_f**2 + outside_r**2 <= slack**2).any():
                return False
        return True

    def hold_for_replan(self, reason):
        self.phase = "brake_for_replan"
        self.current_frame = 0
        self.settle_ms = None
        self.turn_until_ms = 0
        self.turn_stop_ms = None
        return ScanCommand(reason=reason)

    def interrupt(self, reason="主动观察：深度或里程计不可用，停车"):
        if self.active:
            self.locked |= self.issued_yaw
        self.phase = "idle"
        self.turn_until_ms = 0
        self.turn_stop_ms = None
        self.clear_frames = 0
        return ScanCommand(reason=reason)

    def invalidate(self):
        self.current_frame = 0
        self.pending_depth = None
        self._pending_projection = None
        self.source_frame = None
        self.odom_source = None
        self.poses.clear()
        self.views.clear()
        self.pose = (0., 0., 0.)
        self.pose_origin = None
        self.pose_feedback_since_ms = 0
        self.turn_stop_ms = None
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
