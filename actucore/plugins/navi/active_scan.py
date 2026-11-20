"""Bounded, odometry-confirmed observations for front-only local planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .local_path import Geometry, observe, RESOLUTION, START, SHAPE, RIGHT, FORWARD
from .odom import PlanarPose, planar_feedback

ANGLE = math.radians(20)
ANGLE_TOLERANCE = math.radians(2)
STILL_LINEAR = .05
STILL_YAW = .04
ODOM_GAP_MS = 250
MAP_AGE_MS = 2500
SCAN_TIMEOUT_MS = 8000
TURN_BURST_MS = 400
TURN_PROGRESS_WINDOW_MS = 1500
TURN_PROGRESS_MIN_RAD = math.radians(3)


@dataclass(frozen=True)
class ScanCommand:
    yaw: float = 0.
    reason: str = "主动观察：停车等候"
    replanned_turn: bool = False


@dataclass(frozen=True)
class CheckedTurnSweep:
    geometry: Geometry
    half_width: float
    horizon: float
    occupied: np.ndarray
    pose: tuple


@dataclass
class ObservedObstacles:
    """Occupied cells from an observed foreground obstacle, in one odometry frame."""
    frame: tuple | None = None
    cells: set = field(default_factory=set)
    last_source: int = 0

    def remember(self, pose, occupied):
        if not occupied.any():
            return
        self.frame = pose.frame
        forward, left = FORWARD[occupied], -RIGHT[occupied]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        x = pose.x + c * forward - s * left
        y = pose.y + s * forward + c * left
        self.cells.update(zip(np.rint(x / RESOLUTION).astype(int),
                              np.rint(y / RESOLUTION).astype(int)))

    def project(self, pose, observed_free=None):
        if not self.cells or pose is None or pose.frame != self.frame:
            return None
        cells = np.array(list(self.cells))
        x, y = (cells * RESOLUTION - (pose.x, pose.y)).T
        # Keep hazards near the robot even while they are behind the optical view.
        nearby = np.hypot(x, y) <= math.hypot(FORWARD.max(), RIGHT.max()) + RESOLUTION
        if not nearby.all():
            self.cells = set(map(tuple, cells[nearby]))
            cells, x, y = cells[nearby], x[nearby], y[nearby]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        row = (x * c + y * s) / RESOLUTION + START[0]
        col = (x * s - y * c) / RESOLUTION + START[1]
        indices = []
        cleared = np.full(len(x), observed_free is not None, dtype=bool)
        # Reproject conservatively across cell boundaries.
        for r in (np.floor(row), np.ceil(row)):
            for k in (np.floor(col), np.ceil(col)):
                inside = (r >= 0) & (r < SHAPE[0]) & (k >= 0) & (k < SHAPE[1])
                r, k = r.astype(int), k.astype(int)
                indices.append((r, k, inside))
                if observed_free is not None:
                    cleared &= inside
                    cleared[inside] &= observed_free[r[inside], k[inside]]
        if cleared.any():
            self.cells.difference_update(map(tuple, cells[cleared]))
        hits = np.zeros(SHAPE, dtype=bool)
        for r, k, inside in indices:
            keep = inside & ~cleared
            hits[r[keep], k[keep]] = True
        return hits


@dataclass
class ActiveScan:
    """Reuse recent front views around one obstacle, with measured body motion."""

    obstacles: ObservedObstacles = field(default_factory=ObservedObstacles)
    historical_hits: np.ndarray | None = None
    phase: str = "idle"
    attempted: bool = False
    exhausted: bool = False
    locked: bool = False
    interrupted_frame: int = 0
    interrupted_source_frame: int | None = None
    interrupted_odom_ms: int = 0
    interrupt_reason: str = ""
    directions: tuple = ()
    turn_angle: float = ANGLE
    start_ms: int = 0
    origin: tuple = (0., 0., 0.)
    origin_set: bool = False
    still_since: int = 0
    turn_start_heading: float = 0.
    retry_after_hold: bool = False
    replan_cause: str = ""
    turn_replan_used: bool = False
    turn_retry_cause: str = ""
    issued_yaw: bool = False
    pose: tuple = (0., 0., 0.)
    motion: tuple = (0., 0., 0.)
    poses: deque = field(default_factory=lambda: deque(maxlen=32))
    views: deque = field(default_factory=lambda: deque(maxlen=40))
    current_frame: int = 0
    pending_depth: dict | None = None
    _pending_projection: tuple | None = None
    side_frame: int = 0
    advance_m: float | None = None
    route_segment: bool = False
    rerouted: bool = False
    advance_origin: tuple | None = None
    advance_commanded_m: float = 0.
    result_frame: int = 0
    receipt_only: bool = False
    source_frame: int | None = None
    odom_source: int | None = None
    buffered_odom: bool = False
    pose_origin: PlanarPose | None = None
    pose_feedback_since_ms: int = 0
    control_ms: int = 0
    turn_stop_ms: int | None = None
    turn_until_ms: int = 0
    turn_feedback_ms: int = 0
    turn_sweep: CheckedTurnSweep | None = None
    turn_yaw: float = 0.
    command_only: bool = False
    single_control_ms: int = 0
    single_depth_ms: int = 0
    single_source_id: int | None = None
    single_source_received_ms: int = 0
    timed_phase_s: float = 0.
    timed_total_s: float = 0.
    timed_turn_s: float = 0.
    timed_heading: float = 0.

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
    def buffered_turn(self):
        return (self.receipt_only and self.buffered_odom and self.pose_origin is None
                and self.advance_m is not None and self.phase == "turn")

    def update(self, depth, odom, geometry, *, receipt_only=False, max_depth_age_ms=500):
        """Update measured motion and the latest usable depth observation."""
        if self.receipt_only != receipt_only:
            self.invalidate()
            self.receipt_only = receipt_only
        stamp = (odom or {}).get("received_ms" if self.receipt_only else "stamp_ms")
        self.control_ms = (depth or {}).get("control_ms") or 0
        capture = (depth or {}).get("capture_ms")
        motion = planar_feedback(odom)
        if type(stamp) is not int or stamp <= 0 or geometry is None or motion is None:
            if self.receipt_only and self.active and geometry is not None:
                return self.hold_for_replan("主动观察：运动反馈暂不可用，停车后重规划")
            return self.invalidate()
        body_pose = (odom or {}).get("motion_pose")
        history = odom.get("velocity_history")
        self.buffered_odom = bool(history)
        if self.poses and ((body_pose is None) != (self.pose_origin is None)
                           or (body_pose is not None and body_pose.frame != self.pose_origin.frame)):
            self.invalidate()
        source = (odom or {}).get("source_sample_id") if self.receipt_only else None
        if self.poses:
            previous = self.poses[-1][0]
            if self.receipt_only and self.active and body_pose is None and stamp > previous and history:
                hold = self._integrate_receipt_motion(history, stamp, source, motion)
                if hold is not None:
                    return hold
                stamp, source, motion = self.poses[-1][0], self.odom_source, self.motion
            else:
                if source is not None and self.odom_source is not None:
                    if source < self.odom_source:
                        return self.invalidate()
                    if source == self.odom_source:
                        if not 0 <= stamp - previous <= ODOM_GAP_MS:
                            return self.invalidate()
                        stamp = previous
                if not 0 <= stamp - previous <= ODOM_GAP_MS:
                    if self.receipt_only and self.active and stamp > previous:
                        return self._hold_for_feedback_gap()
                    self.invalidate()
                elif stamp > previous and body_pose is None:
                    self._integrate_motion(motion, (stamp - previous) / 1000)
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
        pose = self._pose_at(capture)
        if pose is None or not ground[::4, ::4].any():
            self.current_frame = 0
            return self.interrupt()
        free, occupied = observe(depth["map"], geometry, RIGHT, FORWARD, RESOLUTION, ground)
        self.views.append((capture, pose, free, occupied))

    def _integrate_motion(self, motion, dt):
        vx, vy, wz = motion
        forward, right, heading = self.pose
        mid = heading - .5 * wz * dt
        self.pose = (forward + dt * (vx * math.cos(mid) + vy * math.sin(mid)),
                     right + dt * (vx * math.sin(mid) - vy * math.cos(mid)),
                     heading - dt * wz)

    def _integrate_receipt_motion(self, history, stamp, source, motion):
        # The control timer may run late while odometry is still arriving.
        # Validate the entire sample sequence before integrating any of it.
        previous = self.poses[-1][0]
        samples = [sample for sample in history if previous < sample[0] <= stamp]
        if not samples or samples[-1][0] != stamp:
            samples.append((stamp, source, motion))
        last_stamp, last_source, last_motion = previous, self.odom_source, self.motion
        intervals = []
        for received, identity, velocity in samples:
            if velocity is None:
                self.poses.clear()
                return self.hold_for_replan("主动观察：运动反馈暂不可用，停车后重规划")
            if identity is not None and last_source is not None:
                if identity < last_source:
                    return self.invalidate()
                if identity == last_source:
                    if received - last_stamp > ODOM_GAP_MS:
                        return self.invalidate()
                    continue
                # Both identities are producer stamp_ms values. Their elapsed
                # time also detects outages hidden by delivery in a batch.
                if identity - last_source > ODOM_GAP_MS:
                    return self._hold_for_feedback_gap()
            if not 0 < received - last_stamp <= ODOM_GAP_MS:
                return self._hold_for_feedback_gap()
            intervals.append((velocity, (received - last_stamp) / 1000))
            last_stamp, last_source, last_motion = received, identity, velocity
        for velocity, dt in intervals:
            self._integrate_motion(velocity, dt)
        if intervals:
            self.poses.append((last_stamp, self.pose, last_motion[2]))
            self.odom_source, self.motion = last_source, last_motion
        if self.phase == "turn" and not self.issued_yaw and any(
                math.hypot(vx, vy) > STILL_LINEAR for (vx, vy, _), _ in intervals):
            return self.hold_for_replan("主动观察：检测到平移，停车后重规划", retry_scan=False)

    def _hold_for_feedback_gap(self):
        self.poses.clear()
        return self.hold_for_replan("主动观察：运动反馈间隔过长，停车后重规划")

    def check_turn_feedback(self, history, now_ms, max_yaw):
        """Check a copy of the approved turn from the velocity timer."""
        if not history or not 0 <= now_ms - history[-1][0] <= ODOM_GAP_MS:
            return "主动观察：运动反馈间隔过长，停车后重规划", True
        stamp, source, motion = history[-1]
        previous = self.poses[-1][0]
        if motion is None or stamp < previous:
            return "主动观察：运动反馈暂不可用，停车后重规划", True
        if stamp > previous:
            hold = self._integrate_receipt_motion(history, stamp, source, motion)
            if hold is not None:
                return hold.reason, True
        if now_ms - self.poses[-1][0] > ODOM_GAP_MS:
            return "主动观察：运动反馈间隔过长，停车后重规划", True
        error = self.origin[2] + self.directions[0] * self.turn_angle - self.pose[2]
        if abs(motion[2]) > max_yaw + .10:
            return "主动观察：转速超出限制，停车后重规划", True
        if abs(error) <= ANGLE_TOLERANCE or self.turn_yaw * error > 0:
            return "主动观察：达到转角，停车等待速度反馈", False
        if motion[2] * math.copysign(1., error) > STILL_YAW:
            return "主动观察：等待反向转动停止", False
        checked = self.turn_sweep
        rate = max(abs(self.turn_yaw), abs(motion[2]))
        sweep = math.copysign(abs(error) + (rate + .10) * checked.horizon, error)
        translation = self._turn_translation(checked.horizon, motion)
        translation += math.dist(self.pose[:2], checked.pose[:2])
        if not self.clear_sweep(checked.geometry, checked.half_width, sweep, translation,
                                occupied=checked.occupied, start_angle=self.pose[2] - checked.pose[2]):
            return "主动观察：转身扫掠区有障碍，停车后重规划", True
        return None

    def _update_latest(self, depth, geometry, stamp, max_depth_age_ms):
        # Receipt times establish liveness.
        # The original header, when present, only identifies repeated frames.
        frame = (depth or {}).get("frame_id")
        source = (depth or {}).get("source_frame_id")
        if type(source) is not int or source <= 0:
            source = None
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
            if self.active and not self.command_only:
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
        return self._update_latest(dict(depth, frame_id=frame), geometry, now_ms, max_depth_age_ms)

    def _pose_at(self, stamp):
        for index, (upper_time, upper, _) in enumerate(self.poses):
            if upper_time == stamp:
                return upper
            if upper_time > stamp:
                if index == 0:
                    return None
                lower_time, lower, _ = self.poses[index - 1]
                part = (stamp - lower_time) / (upper_time - lower_time)
                return tuple(lo + part * (hi - lo) for lo, hi in zip(lower, upper))
        return None

    def grid(self):
        free, hits = self.observed_grid()
        if self.historical_hits is None:
            return free, hits
        return free & ~self.historical_hits, hits | self.historical_hits

    def observed_grid(self):
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

    def matches_route_view(self, depth):
        """Match the route's depth image to the projected observation."""
        if not self.current_frame:
            return False
        if not self.receipt_only:
            return self.current_frame == depth.get("capture_ms")
        source = depth.get("source_frame_id")
        if self.source_frame is not None:
            return source == self.source_frame
        return self.current_frame == depth.get("frame_id")

    def begin(self, directions, *, advance_m=None, turn_angle=ANGLE, route_segment=False, reroute=False):
        if not self.current_frame or (self.attempted and not reroute) or (reroute and self.rerouted):
            return False
        if reroute:
            self.rerouted = True
            self.side_frame = 0
        self.directions = directions
        self.turn_angle = turn_angle
        self.route_segment = route_segment
        self.advance_m, self.advance_origin, self.advance_commanded_m = advance_m, None, 0.
        self.attempted = True
        if not reroute:
            self.exhausted = False
        self.phase = "brake"
        self.origin_set = False
        self.turn_sweep = None
        self.start_ms = self.single_control_ms if self.command_only else self.poses[-1][0]
        self.still_since = self.start_ms
        self.timed_phase_s = self.timed_total_s = self.timed_turn_s = 0.
        self.timed_heading = 0.
        return True

    def try_other_side(self):
        self.directions = self.directions[1:]
        if self.directions:
            self.turn_angle = ANGLE
            self.phase = "brake"
            self.still_since = self.single_control_ms if self.command_only else self.poses[-1][0]
            self.timed_phase_s = self.timed_turn_s = 0.
        else:
            self.exhaust()
        return self.active

    def exhaust(self):
        """Finish a normal search whose candidate corridors were blocked."""
        self.issued_yaw = False
        self.interrupt()
        self.attempted = self.exhausted = True
        self.result_frame = 0

    def finish(self):
        self.result_frame = self.current_frame if self.route_segment else self.side_frame
        if self.advance_origin is not None and not self.exhausted:
            self.turn_replan_used = False
        if self.route_segment:
            self.attempted = self.exhausted
            self.rerouted = False
        self.phase = "idle"

    def command(self, geometry, half_width, yaw_rate, watchdog_ms, obs_age_ms=500, *,
                min_yaw_rate=0., now_ms=None, tick_s=.1, previous_yaw=None, control_period_s=None,
                previous_command_s=None, yaw_limits=None):
        if not self.active or self.ready:
            return ScanCommand()
        if self.command_only:
            if previous_yaw is not None and self.timed_turn_s:
                covered = min(tick_s, watchdog_ms / 1000) if previous_command_s is None else previous_command_s
                self.timed_heading -= previous_yaw * covered
            return self._timed_command(geometry, half_width, yaw_rate, watchdog_ms,
                                       obs_age_ms, min_yaw_rate, tick_s, previous_yaw is not None, yaw_limits)
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
        # A released straight segment legitimately leaves the turn origin. A
        # replan hold after it must settle and replan, not lock on that travel.
        turning = self.issued_yaw and self.phase in ("turn", "return")
        if (not turning and self.origin_set and self.advance_origin is None
                and math.dist(self.pose[:2], self.origin[:2]) > .18):
            return self.interrupt("主动观察：检测到平移，停车")
        if (self.receipt_only and speed > STILL_LINEAR and self.phase != "brake_for_replan"
                and not turning):
            return self.hold_for_replan("主动观察：检测到平移，停车后重规划", retry_scan=False)
        if self.phase == "brake_for_replan":
            if speed > STILL_LINEAR or abs(self.motion[2]) > STILL_YAW:
                motion = "检测到平移" if speed > STILL_LINEAR else "仍在转动"
                return ScanCommand(reason=f"主动观察：{motion}，停车后重规划")
            if self.rerouted and self.advance_origin is None:
                self.locked = True
                return self.interrupt("主动观察：替代路线尚未前进，停车等待新导航指令")
            released_route = self.route_segment and self.advance_origin is not None
            interrupted_turn = (self.receipt_only and self.issued_yaw and self.advance_m is not None
                                and self.advance_origin is None)
            if interrupted_turn:
                if self.turn_replan_used:
                    return self.interrupt(
                        f"主动观察：转向重试后再次中断；保持停车"
                        f"（首次：{self.turn_retry_cause}；重试：{self.replan_cause}）")
                self.turn_replan_used = True
                self.turn_retry_cause = self.replan_cause
                self.route_segment = False
                self.advance_m = None
                self.side_frame = self.result_frame = 0
            if interrupted_turn or self.retry_after_hold or not self.issued_yaw or released_route:
                self.attempted = False
                self.retry_after_hold = False
            self.phase = "idle"
            self.issued_yaw = False
            return ScanCommand(reason="主动观察：已停稳，重新规划", replanned_turn=interrupted_turn)
        if speed > STILL_LINEAR and not turning:
            return self.interrupt("主动观察：检测到平移，停车")
        if (abs(self.motion[2]) > yaw_rate + .10
                or abs(self.pose[2] - self.origin[2]) > max(ANGLE, self.turn_angle) + 2 * ANGLE_TOLERANCE):
            return self.interrupt("主动观察：转速或转角超出限制，停车")
        # Short route turns also use bounded bursts when only delayed velocity
        # feedback is available. Stop on the control clock even on a repeated sample.
        bounded_turn = self.pose_origin is not None or self.advance_m is not None
        control_ms = now_ms or self.control_ms or stamp
        wait_reason = "主动观察：停转等待新位姿" if self.pose_origin is not None else "主动观察：停转等待新速度反馈"
        if self.turn_until_ms and control_ms >= self.turn_until_ms:
            self.turn_stop_ms = control_ms
            self.turn_until_ms = 0
            return ScanCommand(reason=wait_reason)
        if self.turn_stop_ms is not None:
            # A stopped velocity alone can precede the chassis response. Allow
            # the prior command to expire, then require a new stopped sample.
            fresh = (self.pose_feedback_since_ms >= self.turn_stop_ms if self.pose_origin is not None
                     else stamp > self.turn_stop_ms + watchdog_ms)
            if not fresh or abs(self.motion[2]) > STILL_YAW or speed > STILL_LINEAR:
                return ScanCommand(reason=wait_reason)
            if self.pose_origin is None:
                # The chassis was commanded to stop during this interval.
                self.still_since += stamp - self.turn_stop_ms
            self.turn_stop_ms = None

        returning = self.phase == "return"
        heading = self.origin[2] if returning else self.origin[2] + self.directions[0] * self.turn_angle
        error = heading - self.pose[2]
        if self.turn_until_ms and (abs(error) <= ANGLE_TOLERANCE or self.turn_yaw * error > 0):
            self.turn_stop_ms = control_ms
            self.turn_until_ms = 0
            return ScanCommand(reason=wait_reason)
        if abs(error) <= ANGLE_TOLERANCE:
            if abs(self.motion[2]) > STILL_YAW or speed > STILL_LINEAR:
                return ScanCommand(reason="主动观察：等待转向停止")
            if returning:
                self.issued_yaw = False
            else:
                self.side_frame = self.current_frame
            self.phase = "evaluate" if returning or self.receipt_only or self.advance_m is not None else "return"
            return ScanCommand(reason="主动观察：使用当前深度评估路线")
        if (self.receipt_only and self.phase == "turn" and self.issued_yaw
                and stamp - self.still_since >= TURN_PROGRESS_WINDOW_MS
                and abs(self.pose[2] - self.turn_start_heading) < TURN_PROGRESS_MIN_RAD):
            return self.hold_for_replan("主动观察：转向无实测进展，停车后重规划", retry_scan=False)

        # Budget tracking latency and the watchdog, including a sink poll.
        horizon = (watchdog_ms + obs_age_ms + 50) / 1000
        if yaw_limits is not None:
            yaw_rate = min(yaw_rate, yaw_limits[0 if error > 0. else 1])
        if not geometry.front > 0 or not geometry.rear > 0 or not yaw_rate > 0 or yaw_rate < min_yaw_rate:
            return self.interrupt()
        if self.motion[2] * math.copysign(1., error) > STILL_YAW:
            return ScanCommand(reason="主动观察：等待反向转动停止")

        renew = self.buffered_turn and stamp > self.turn_feedback_ms
        if bounded_turn:
            minimum_s = tick_s
            if self.turn_until_ms and not renew:
                duration_ms = self.turn_until_ms - control_ms
                yaw_rate = min(yaw_rate, abs(self.turn_yaw))
            else:
                # A late tick has not consumed any of this new window.
                if control_period_s is not None:
                    minimum_s = control_period_s
                duration_ms = TURN_BURST_MS
                yaw_rate = min(yaw_rate, max(min_yaw_rate, abs(error) * 1000 / duration_ms))
                duration_ms = min(duration_ms, int((abs(error) + 2 * ANGLE_TOLERANCE) * 1000 / yaw_rate))
            if duration_ms < max(1, int(minimum_s * 1000)):
                self.turn_until_ms = 0
                self.turn_stop_ms = control_ms
                return ScanCommand(reason=wait_reason)
        occupied = self.turn_obstacles()

        yaw_rate = self.checked_turn_rate(geometry, half_width, error, yaw_rate, min_yaw_rate,
                                          horizon, occupied)
        if not yaw_rate:
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            return self.interrupt("主动观察：转身扫掠区有障碍，停车")
        self.issued_yaw = True
        if self.buffered_turn:
            self.turn_sweep = CheckedTurnSweep(geometry, half_width, horizon, occupied, self.pose)
        if bounded_turn:
            if not self.turn_until_ms or renew:
                self.turn_until_ms = control_ms + duration_ms
                self.turn_feedback_ms = stamp
            self.turn_yaw = -math.copysign(yaw_rate, error)
        direction = "右" if error > 0 else "左"
        description = "回到观察前的朝向" if returning else f"向{direction}观察"
        return ScanCommand(-math.copysign(yaw_rate, error), f"主动观察：{description}")

    def _timed_command(self, geometry, half_width, yaw_rate, watchdog_ms, obs_age_ms,
                       min_yaw, tick_s, tracks_commands, yaw_limits):
        self.timed_total_s += tick_s
        self.timed_phase_s += tick_s
        if not self.current_frame or self.timed_total_s * 1000 > 2 * SCAN_TIMEOUT_MS:
            return self.interrupt("主动观察：无反馈观察超时或深度失效，停车")
        if self.phase == "brake":
            if self.timed_phase_s * 1000 >= watchdog_ms + 50:
                self.phase = "turn"
                self.timed_phase_s = 0.
            return ScanCommand(reason="主动观察：停车，等待制动窗口")
        error = self.directions[0] * self.turn_angle - self.timed_heading
        if yaw_limits is not None:
            turn_direction = self.turn_yaw if self.timed_turn_s else error
            yaw_rate = min(yaw_rate, yaw_limits[0 if turn_direction > 0. else 1])
        if self.timed_turn_s and abs(self.turn_yaw) > yaw_rate:
            return self.interrupt("主动观察：底盘转向限速已变化，停车")
        if not self.timed_turn_s:
            if abs(error) <= ANGLE_TOLERANCE:
                self.side_frame = self.current_frame
                self.phase = "evaluate"
                return ScanCommand(reason="主动观察：使用当前深度评估路线")
            if yaw_rate <= 0. or yaw_rate < min_yaw:
                return self.interrupt("主动观察：底盘无可用转速，停车")
            duration = max(TURN_BURST_MS / 1000, abs(error) / yaw_rate)
            if tick_s > 0.:
                duration = math.ceil(duration / tick_s) * tick_s
            if min_yaw:
                duration = min(duration, abs(error) / min_yaw)
            if duration <= 0.:
                return self.interrupt("主动观察：无可执行转向，停车")
            self.timed_turn_s = duration
            self.turn_yaw = math.copysign(min(yaw_rate, abs(error) / duration), error)
            self.timed_phase_s = 0.
        if self.timed_phase_s + tick_s > self.timed_turn_s + 1e-9:
            if tracks_commands and abs(error) > ANGLE_TOLERANCE:
                return self.interrupt("主动观察：转向指令未完成，停车")
            self.side_frame = self.current_frame
            self.phase = "evaluate"
            return ScanCommand(reason="主动观察：定时转向结束，评估路线")
        horizon = (watchdog_ms + obs_age_ms + 50) / 1000
        rate = abs(self.turn_yaw)
        sweep = math.copysign(abs(error) + (rate + .10) * horizon, self.turn_yaw)
        if (rate <= 0. or rate < min_yaw or not self.clear_sweep(
                geometry, half_width, sweep, STILL_LINEAR * horizon)):
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            return self.interrupt("主动观察：转身扫掠区有障碍，停车")
        self.issued_yaw = True
        return ScanCommand(-self.turn_yaw, "主动观察：有界短转向")

    def turn_obstacles(self):
        return self.grid()[1] | self.views[-1][3]

    def _turn_translation(self, horizon, motion):
        translation = max(STILL_LINEAR, math.hypot(*motion[:2])) * horizon
        if self.receipt_only and self.buffered_odom and self.pose_origin is None:
            translation = max(translation, RESOLUTION)
        return translation

    def checked_turn_rate(self, geometry, half_width, error, yaw_rate, min_yaw_rate, horizon, occupied):
        translation = self._turn_translation(horizon, self.motion)

        def clear_sweep(rate):
            turn_rate = max(rate, abs(self.motion[2]))
            sweep = math.copysign(abs(error) + (turn_rate + .10) * horizon, error)
            return self.clear_sweep(geometry, half_width, sweep, translation,
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
        return yaw_rate

    def clear_sweep(self, geometry, half_width, sweep, translation, *, occupied=None, start_angle=0.):
        # Rotate the rectangle in small increments and include cell diagonal
        # plus the greatest displacement between successive angular samples.
        samples = max(2, math.ceil(abs(sweep) / .04) + 1)
        radius = math.hypot(max(geometry.front, geometry.rear), half_width)
        slack = translation + RESOLUTION / math.sqrt(2) + radius * abs(sweep) / (2 * (samples - 1))
        if occupied is None:
            occupied = self.turn_obstacles()
        nearby = occupied & (FORWARD**2 + RIGHT**2 <= (radius + slack)**2)
        if not nearby.any():
            return True
        occupied_forward, occupied_right = FORWARD[nearby], RIGHT[nearby]
        for angle in np.linspace(start_angle, start_angle + sweep, samples):
            c, s = math.cos(angle), math.sin(angle)
            forward = occupied_forward * c + occupied_right * s
            right = -occupied_forward * s + occupied_right * c
            outside_f = np.maximum(0., np.maximum(-geometry.rear - forward, forward - geometry.front))
            outside_r = np.maximum(0., abs(right) - half_width)
            if (outside_f**2 + outside_r**2 <= slack**2).any():
                return False
        return True

    def hold_for_replan(self, reason, *, retry_scan=True):
        if self.phase != "brake_for_replan":
            self.retry_after_hold = retry_scan
            prefix, suffix = "主动观察：", "，停车后重规划"
            cause = reason[len(prefix):] if reason.startswith(prefix) else reason
            self.replan_cause = cause[:-len(suffix)] if cause.endswith(suffix) else cause
        self.phase = "brake_for_replan"
        self.current_frame = 0
        self.turn_until_ms = 0
        self.turn_stop_ms = None
        return ScanCommand(reason=reason)

    def interrupt(self, reason="主动观察：深度或里程计不可用，停车"):
        if self.active and (self.issued_yaw or self.locked):
            self.locked = True
            self.interrupted_frame = self.views[-1][0] if self.views else self.current_frame
            self.interrupted_source_frame = self.source_frame
            self.interrupted_odom_ms = self.poses[-1][0] if self.poses else 0
            self.interrupt_reason = reason
        self.phase = "idle"
        self.retry_after_hold = False
        self.turn_until_ms = 0
        self.turn_stop_ms = None
        return ScanCommand(reason=reason)

    def invalidate(self):
        stop = self.interrupt()
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
        return stop

    def has_feedback_after_interruption(self):
        return (self.current_frame > self.interrupted_frame and bool(self.poses)
                and self.poses[-1][0] > self.interrupted_odom_ms
                and (self.interrupted_source_frame is None or (
                    self.source_frame is not None and self.source_frame > self.interrupted_source_frame)))

    def note_clear_path(self, frame):
        if self.locked or not frame or not (self.attempted or self.result_frame):
            return
        self.rerouted = False
        self.attempted = False
        self.exhausted = False
        self.issued_yaw = False
        self.turn_replan_used = False
        self.result_frame = 0
