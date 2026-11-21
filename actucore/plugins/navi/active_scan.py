"""Bounded, odometry-confirmed observations for front-only local planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math

import numpy as np

from .depth import has_valid_returns
from .local_path import (FORWARD, FREE, OCCUPIED, RESOLUTION, RIGHT, SHAPE, START, UNKNOWN,
                         Geometry, motion_sweep_hits, observe, project_observation)
from .odom import PlanarPose, planar_feedback

ANGLE = math.radians(20)
ANGLE_TOLERANCE = math.radians(2)
STILL_LINEAR = .05
SCAN_ODOM_LINEAR_TOLERANCE_MPS = .08
STILL_YAW = .04
ODOM_GAP_MS = 250
MAP_AGE_MS = 2500
FREE_MEMORY_RADIUS_CELLS = round(6. / RESOLUTION)
FREE_MEMORY_SHIFT_CELLS = round(1. / RESOLUTION)
OCCUPANCY_MEMORY_SAMPLES = 3
FREE_TO_OCCUPIED_SAMPLES = 6
CAMERA_OCCUPANCY_MEMORY_RANGE_M = 1.25
SCAN_TIMEOUT_MS = 8000
TURN_BURST_MS = 400
TURN_PROGRESS_WINDOW_MS = 2000
TURN_PROGRESS_MIN_RAD = math.radians(3)
DEPTH_RECHECK_HOLD = "主动观察：深度暂不可用，停车等待新深度复检当前路线"
TURN_SPEED_REDUCTION = "主动观察：实测转速超限，减速后继续"


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
    minimum_translation: float = 0.


@dataclass
class ObservedObstacles:
    """Remember measured cells in one odometry frame."""
    frame: tuple | None = None
    cells: set = field(default_factory=set)
    last_source: int = 0
    free_cells: np.ndarray | None = field(default=None, repr=False, compare=False)
    free_center: tuple[int, int] = (0, 0)
    hit_votes: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def has_cells(self):
        return bool(self.cells) or self.free_cells is not None

    @property
    def has_occupied(self):
        return bool(self.cells) or (self.free_cells is not None and (self.free_cells == OCCUPIED).any())

    @staticmethod
    def _world_coordinates(pose, mask):
        forward, right = FORWARD[mask], RIGHT[mask]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return ((pose.x + c * forward + s * right) / RESOLUTION,
                (pose.y + s * forward - c * right) / RESOLUTION)

    @staticmethod
    def _world_cells(pose, mask):
        return np.rint(ObservedObstacles._world_coordinates(pose, mask)).astype(int).T

    def _free_window(self, pose):
        # Keep a fixed world-aligned window, recentering only after a metre
        # of movement so a control tick never scans the traversed route.
        center = tuple(round(value / RESOLUTION) for value in (pose.x, pose.y))
        shift = tuple(new - old for new, old in zip(center, self.free_center))
        radius = FREE_MEMORY_RADIUS_CELLS
        size = 2 * radius + 1
        if self.free_cells is None:
            self.free_cells = np.zeros((size, size), dtype=np.uint8)
            self.hit_votes = np.zeros_like(self.free_cells)
            self.free_center = center
        elif max(map(abs, shift)) >= FREE_MEMORY_SHIFT_CELLS:
            moved = np.zeros_like(self.free_cells)
            votes = np.zeros_like(self.hit_votes)
            if max(map(abs, shift)) < size:
                old = tuple(slice(max(0, d), size + min(0, d)) for d in shift)
                new = tuple(slice(max(0, -d), size + min(0, -d)) for d in shift)
                moved[new] = self.free_cells[old]
                votes[new] = self.hit_votes[old]
            self.free_cells, self.hit_votes, self.free_center = moved, votes, center
        return self.free_cells, np.array(self.free_center) - radius

    def remember(self, pose, occupied):
        if not occupied.any() or (self.has_cells and self.frame != pose.frame):
            return
        self.frame = pose.frame
        cells = self._world_cells(pose, occupied)
        self.cells.update(map(tuple, cells))
        if self.free_cells is not None:
            window, origin = self._free_window(pose)
            window[tuple((cells - origin).T)] = UNKNOWN

    def remember_view(self, pose, free, occupied, *, remember_occupied=None, rolling_range=None):
        if self.has_cells and self.frame != pose.frame:
            return
        confirmed = free & ~occupied
        if self.has_occupied and confirmed.any():
            self.project(pose, observed_free=confirmed, observed_occupied=occupied)
        if self.free_cells is None and not confirmed.any() and remember_occupied is None:
            return

        self.frame = pose.frame
        window, origin = self._free_window(pose)
        hits = self._world_cells(pose, occupied)
        free_indices = tuple((self._world_cells(pose, confirmed) - origin).T)
        window[free_indices] = np.where(window[free_indices] == OCCUPIED, OCCUPIED, FREE)
        candidates = hits[window[tuple((hits - origin).T)] == FREE]
        if candidates.size:
            row, col = self._local_coordinates(pose, candidates)
            observed = np.ones(len(candidates), dtype=bool)
            # A cell touched only at an edge blocks this view, but does
            # not establish an aligned observation for future poses.
            for r in (np.floor(row), np.ceil(row)):
                for c in (np.floor(col), np.ceil(col)):
                    inside = (r >= 0) & (r < SHAPE[0]) & (c >= 0) & (c < SHAPE[1])
                    observed &= inside
                    observed[inside] &= occupied[r[inside].astype(int), c[inside].astype(int)]
            candidates = candidates[observed]
        recent = tuple((candidates - origin).T)
        samples = FREE_TO_OCCUPIED_SAMPLES
        recent_votes = np.minimum(self.hit_votes[recent] + 1, samples)
        selected = (self._world_cells(pose, remember_occupied) if remember_occupied is not None
                    else np.empty((0, 2), dtype=int))
        long_term = tuple((selected - origin).T)
        long_term_votes = np.minimum(self.hit_votes[long_term] + 1, OCCUPANCY_MEMORY_SAMPLES)
        self.hit_votes.fill(0)
        self.hit_votes[recent] = recent_votes
        self.hit_votes[long_term] = long_term_votes
        confirmed_hits = candidates[recent_votes == samples]
        remembered = OCCUPIED
        if rolling_range is not None:
            # Distant returns can disprove old free space. Persisting them as
            # blind-spot obstacles would accumulate the room as the base moves.
            offsets = confirmed_hits * RESOLUTION - (pose.x, pose.y)
            nearby = np.linalg.norm(offsets, axis=1) <= rolling_range
            remembered = np.where(nearby, OCCUPIED, UNKNOWN)
        window[tuple((confirmed_hits - origin).T)] = remembered
        retained = selected[long_term_votes == OCCUPANCY_MEMORY_SAMPLES]
        self.cells.update(map(tuple, retained))
        window[tuple((retained - origin).T)] = UNKNOWN

    def project_free(self, pose):
        if self.free_cells is None or pose is None or pose.frame != self.frame:
            return None
        window, origin = self._free_window(pose)
        x, y = self._world_coordinates(pose, np.s_[:, :])
        cols, rows = np.rint(x).astype(int) - origin[0], np.rint(y).astype(int) - origin[1]
        return window[cols, rows] == FREE

    @staticmethod
    def _local_coordinates(pose, cells):
        x, y = (cells * RESOLUTION - (pose.x, pose.y)).T
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        row = (x * c + y * s) / RESOLUTION + START[0]
        col = (x * s - y * c) / RESOLUTION + START[1]
        return np.round(row, 7), np.round(col, 7)

    def project(self, pose, observed_free=None, *, observed_occupied=None):
        if not self.has_cells or pose is None or pose.frame != self.frame:
            return None
        cells = np.asarray(list(self.cells), dtype=int).reshape(-1, 2)
        window = None
        if self.free_cells is not None:
            window, origin = self._free_window(pose)
            local = np.argwhere(window == OCCUPIED)
            if local.size:
                cells = np.concatenate((cells, local + origin))
        if not cells.size:
            return None
        x, y = (cells * RESOLUTION - (pose.x, pose.y)).T
        # The local grid has finite bounds. Points outside it are kept in
        # the odometry map and can become relevant again when revisited.
        nearby = np.hypot(x, y) <= math.hypot(FORWARD.max(), RIGHT.max()) + RESOLUTION
        cells = cells[nearby]
        row, col = self._local_coordinates(pose, cells)
        indices = []
        cleared = np.zeros(len(cells), dtype=bool)
        conflicting_hit = np.zeros(len(cells), dtype=bool)
        # A free observation anywhere in the cell's projected support can
        # clear its memory when the same exposure shows no overlapping hit.
        for r in (np.floor(row), np.ceil(row)):
            for k in (np.floor(col), np.ceil(col)):
                inside = (r >= 0) & (r < SHAPE[0]) & (k >= 0) & (k < SHAPE[1])
                r, k = r.astype(int), k.astype(int)
                indices.append((r, k, inside))
                if observed_free is not None:
                    cleared[inside] |= observed_free[r[inside], k[inside]]
                if observed_occupied is not None:
                    conflicting_hit[inside] |= observed_occupied[r[inside], k[inside]]
        cleared &= ~conflicting_hit
        if cleared.any():
            cleared_cells = cells[cleared]
            self.cells.difference_update(map(tuple, cleared_cells))
            if window is not None:
                window[tuple((cleared_cells - origin).T)] = FREE
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
    historical_free: np.ndarray | None = None
    near_obstacles: ObservedObstacles = field(default_factory=ObservedObstacles)
    near_hits: np.ndarray | None = None
    near_free: np.ndarray | None = None
    near_revision: int = 0
    near_odom_reference: tuple | None = None
    phase: str = "idle"
    attempted: bool = False
    exhausted: bool = False
    directions: tuple = ()
    turn_angle: float = ANGLE
    start_ms: int = 0
    origin: tuple = (0., 0., 0.)
    origin_set: bool = False
    still_since: int = 0
    turn_start_heading: float = 0.
    retry_after_hold: bool = False
    replan_after_ms: int = 0
    issued_yaw: bool = False
    pose: tuple = (0., 0., 0.)
    motion_frame: object = field(default_factory=object)
    motion: tuple = (0., 0., 0.)
    odom_linear_tolerance_mps: float = SCAN_ODOM_LINEAR_TOLERANCE_MPS
    max_feedback_gap_ms: int = ODOM_GAP_MS
    poses: deque = field(default_factory=lambda: deque(maxlen=32))
    views: deque = field(default_factory=lambda: deque(maxlen=40))
    observed_vantages: deque = field(default_factory=lambda: deque(maxlen=32))
    observed_world: deque[PlanarPose] = field(default_factory=lambda: deque(maxlen=32))
    observation_pose: PlanarPose | None = None
    current_frame: int = 0
    pending_depth: dict | None = None
    _pending_projection: tuple | None = None
    _fused_view: tuple | None = None
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
    turn_rate_scale: float = 1.
    turn_depth_paused: bool = False
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

    def update(self, depth, odom, geometry, *, receipt_only=False, max_depth_age_ms=500, follow_leg=False):
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
            gap = self.max_feedback_gap_ms if self.receipt_only and body_pose is None else ODOM_GAP_MS
            if (self.receipt_only and (self.active or follow_leg or self.near_obstacles.has_cells)
                    and body_pose is None and stamp > previous and history):
                hold = self._integrate_receipt_motion(history, stamp, source, motion)
                if hold is not None:
                    return hold
                stamp, source, motion = self.poses[-1][0], self.odom_source, self.motion
            else:
                if source is not None and self.odom_source is not None:
                    if source < self.odom_source:
                        return self.invalidate()
                    if source == self.odom_source:
                        if not 0 <= stamp - previous <= gap:
                            return self.invalidate()
                        stamp = previous
                if not 0 <= stamp - previous <= gap:
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
        if self.phase in ("brake", "turn", "return") and stamp - self.start_ms > timeout:
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
        if (type(capture) is not int or not 0 <= stamp - capture <= max_depth_age_ms
                or (ground is None and (geometry.horizontal is None or geometry.vertical is None))
                or not has_valid_returns((depth or {}).get("map"))):
            return self.interrupt()
        if self.views and capture < self.views[-1][0]:
            # Reject out-of-order observations without replacing the map.
            return self.interrupt()
        self.current_frame = capture
        if self.views and capture == self.views[-1][0]:
            return
        pose = self._pose_at(capture)
        if pose is None:
            self.current_frame = 0
            return self.interrupt()
        free, occupied = self._project_view(depth, geometry)
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
                hold = self.hold_for_replan("主动观察：运动反馈暂不可用，停车后重规划")
                self.poses.clear()
                return hold
            if identity is not None and last_source is not None:
                if identity < last_source:
                    return self.invalidate()
                if identity == last_source:
                    if received - last_stamp > self.max_feedback_gap_ms:
                        return self.invalidate()
                    continue
                # Both identities are producer stamp_ms values. Their elapsed
                # time also detects outages hidden by delivery in a batch.
                if identity - last_source > self.max_feedback_gap_ms:
                    return self._hold_for_feedback_gap()
            if not 0 < received - last_stamp <= self.max_feedback_gap_ms:
                return self._hold_for_feedback_gap()
            intervals.append((received, velocity, (received - last_stamp) / 1000))
            last_stamp, last_source, last_motion = received, identity, velocity
        for received, velocity, dt in intervals:
            self._integrate_motion(velocity, dt)
            self.poses.append((received, self.pose, velocity[2]))
        if intervals:
            self.odom_source, self.motion = last_source, last_motion
        if self.phase == "turn" and not self.issued_yaw and any(
                math.hypot(vx, vy) > self.odom_linear_tolerance_mps for _, (vx, vy, _), _ in intervals):
            return self.hold_for_replan("主动观察：检测到平移，停车后重规划", retry_scan=False)

    def _hold_for_feedback_gap(self):
        hold = self.hold_for_replan("主动观察：运动反馈间隔过长，停车后重规划")
        self.motion_frame = object()
        self.poses.clear()
        return hold

    def sync_buffered_motion(self, history, now_ms):
        """Consume every feedback sample since the copied control pose."""
        if not history or not 0 <= now_ms - history[-1][0] <= self.max_feedback_gap_ms:
            return "主动观察：运动反馈间隔过长，停车后重规划", True
        stamp, source, motion = history[-1]
        previous = self.poses[-1][0]
        if motion is None or stamp < previous:
            return "主动观察：运动反馈暂不可用，停车后重规划", True
        if stamp > previous:
            hold = self._integrate_receipt_motion(history, stamp, source, motion)
            if hold is not None:
                return hold.reason, True
        if now_ms - self.poses[-1][0] > self.max_feedback_gap_ms:
            return "主动观察：运动反馈间隔过长，停车后重规划", True
        return None

    def check_turn_feedback(self, history, now_ms, max_yaw):
        """Check a copy of the approved turn from the velocity timer."""
        stop = self.sync_buffered_motion(history, now_ms)
        if stop is not None:
            return stop
        motion = history[-1][2]
        error = self.origin[2] + self.directions[0] * self.turn_angle - self.pose[2]
        if abs(motion[2]) > max_yaw + .10:
            return TURN_SPEED_REDUCTION, False
        if abs(error) <= ANGLE_TOLERANCE or self.turn_yaw * error > 0:
            return "主动观察：达到转角，停车等待速度反馈", False
        if motion[2] * math.copysign(1., error) > STILL_YAW:
            return "主动观察：等待反向转动停止", False
        checked = self.turn_sweep
        translation = max(checked.minimum_translation, self._turn_translation(checked.horizon, motion))
        translation += math.dist(self.pose[:2], checked.pose[:2])
        if not self.clear_sweep(checked.geometry, checked.half_width, translation, occupied=checked.occupied):
            return "主动观察：转身扫掠区有障碍，停车后重规划", True
        return None

    def reduce_turn_speed(self, limit, measured):
        self.turn_rate_scale = min(self.turn_rate_scale, .8, limit / max(limit, abs(measured), 1e-6))
        self.turn_until_ms = 0

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
                or (ground is None and (geometry.horizontal is None or geometry.vertical is None))
                or not has_valid_returns((depth or {}).get("map"))):
            if self.active and not self.command_only:
                return self.pause_for_depth(self.control_ms or stamp)
            return self.interrupt()
        if self.turn_depth_paused:
            if ((self.source_frame is not None and source is None)
                    or (self.views and frame <= self.views[-1][0])):
                return ScanCommand(reason=DEPTH_RECHECK_HOLD)
            self.turn_depth_paused = False
        self.current_frame = frame
        if self.views and self.views[-1][0] == frame:
            return
        # Only the latest receipt-time view can contribute to the grid.
        # Project it when a turn or local plan needs it.
        self._pending_projection = (depth, geometry)
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

    def _near_pose(self, pose):
        memory = self.near_obstacles
        if pose is not None and (not memory.has_cells or pose.frame == memory.frame):
            self.near_odom_reference = (self.motion_frame, self.pose, pose)
            return pose
        if not self.poses or self.command_only:
            return None
        relative_frame = ("navi_relative", self.motion_frame)
        if not memory.has_cells or memory.frame == relative_frame:
            return self._relative_near_pose(self.poses[-1][0], self.pose)
        reference = self.near_odom_reference
        if reference is None or reference[0] is not self.motion_frame:
            return None
        # Continue in the same local frame with measured relative motion when
        # the separately sampled body pose is late.
        _, start, anchor = reference
        df, dr = self.pose[0] - start[0], self.pose[1] - start[1]
        angle = anchor.yaw + start[2]
        c, s = math.cos(angle), math.sin(angle)
        return PlanarPose(self.poses[-1][0], anchor.frame, anchor.x + c * df + s * dr,
                          anchor.y + s * df - c * dr, anchor.yaw + start[2] - self.pose[2])

    def _relative_near_pose(self, stamp, coords):
        forward, right, yaw = coords
        return PlanarPose(stamp, ("navi_relative", self.motion_frame), forward, -right, -yaw)

    @staticmethod
    def _footprint_origin(pose, geometry):
        # Camera-relative cells start at the chassis front in fallback mode.
        if geometry.front != 0.:
            return pose
        offset = geometry.half_length
        return replace(pose, x=pose.x + offset * math.cos(pose.yaw),
                       y=pose.y + offset * math.sin(pose.yaw))

    def update_near_obstacles(self, depth, geometry, pose, half_width, lookahead, *, approaching):
        memory = self.near_obstacles
        pose = self._near_pose(pose)
        if pose is None:
            self.near_free = None
            # Keep checking the last occupied projection until odometry resumes.
            return
        capture = (depth or {}).get("obstacle_capture_pose")
        if (self.views and self.current_frame > memory.last_source
                and depth is not None and self.matches_route_view(depth)):
            if pose.frame == ("navi_relative", self.motion_frame):
                # With velocity-only odometry, use capture time when it can
                # be paired, or the same host receipt timeline as odometry.
                stamp = depth.get("capture_ms") or self.current_frame
                coords = self._pose_at(stamp)
                capture = self._relative_near_pose(stamp, coords) if coords is not None else None
            if capture is not None and capture.frame == pose.frame:
                free, occupied = self.observed_grid()
                capture = self._footprint_origin(capture, geometry)
                memory.last_source = self.current_frame
                selected = None
                if approaching:
                    front = occupied & (FORWARD > 0.) & (FORWARD <= lookahead) & (abs(RIGHT) <= half_width)
                    if front.any():
                        nearest = FORWARD[front].min()
                        selected = front & (FORWARD <= nearest + 2 * RESOLUTION)
                memory.remember_view(capture, free, occupied, remember_occupied=selected,
                                     rolling_range=CAMERA_OCCUPANCY_MEMORY_RANGE_M)
        origin = self._footprint_origin(pose, geometry)
        hits = memory.project(origin)
        self.near_free = memory.project_free(origin)
        self.near_hits = hits if hits is not None and hits.any() else None
        self.near_revision = pose.stamp_ms

    @property
    def remembered_hits(self):
        if self.near_hits is None:
            return self.historical_hits
        if self.historical_hits is None:
            return self.near_hits
        return self.historical_hits | self.near_hits

    def grid(self):
        free, hits = self.observed_grid()
        remembered = self.remembered_hits
        if self.near_free is not None:
            free = free | self.near_free
        if self.historical_free is not None:
            free = free | self.historical_free
        if remembered is not None:
            hits = hits | remembered
        return free & ~hits, hits

    @staticmethod
    def _project_view(depth, geometry):
        if depth.get("grid_projection") is not None:
            return project_observation(depth, geometry)
        return observe(depth["map"], geometry, RIGHT, FORWARD, RESOLUTION, depth.get("ground"))

    def observed_grid(self):
        """Reproject nearby views. Occupation wins, and free cells need support."""
        if self.receipt_only:
            frame, pose, free, occupied = self.views[-1]
            if self._pending_projection is not None:
                depth, geometry = self._pending_projection
                free, occupied = self._project_view(depth, geometry)
                self.views[-1] = (frame, pose, free, occupied)
                self._pending_projection = None
            return free, occupied
        stamp = self.poses[-1][0]
        latest = self.views[-1] if self.views else None
        if self._fused_view is not None:
            last_view, pose, fused_at, expires, grid = self._fused_view
            if latest is last_view and self.pose == pose and fused_at <= stamp < expires:
                return grid
        free = np.zeros(SHAPE, dtype=bool)
        occupied = np.zeros_like(free)
        expires = math.inf
        forward, right, yaw = self.pose
        for capture, (before_f, before_r, before_yaw), view_free, view_occupied in self.views:
            if stamp - capture > MAP_AGE_MS:
                continue
            expires = min(expires, capture + MAP_AGE_MS + 1)
            df, dr = forward - before_f, right - before_r
            offset_f = df * math.cos(before_yaw) + dr * math.sin(before_yaw)
            offset_r = -df * math.sin(before_yaw) + dr * math.cos(before_yaw)
            delta = yaw - before_yaw
            row = (FORWARD * math.cos(delta) - RIGHT * math.sin(delta) + offset_f) / RESOLUTION + START[0]
            col = (FORWARD * math.sin(delta) + RIGHT * math.cos(delta) + offset_r) / RESOLUTION + START[1]
            row, col = np.round(row, 7), np.round(col, 7)
            confirmed = np.ones_like(free)
            projected_hits = np.zeros_like(occupied)
            for y in (np.floor(row), np.ceil(row)):
                for x in (np.floor(col), np.ceil(col)):
                    valid = (y >= 0) & (y < SHAPE[0]) & (x >= 0) & (x < SHAPE[1])
                    indices = (y.clip(0, SHAPE[0] - 1).astype(int),
                               x.clip(0, SHAPE[1] - 1).astype(int))
                    confirmed &= valid & view_free[indices]
                    projected_hits |= valid & view_occupied[indices]
            occupied = (occupied & ~confirmed) | projected_hits
            free = (free | confirmed) & ~projected_hits
        grid = free & ~occupied, occupied
        self._fused_view = (latest, self.pose, stamp, expires, grid)
        return grid

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

    @staticmethod
    def _same_vantage(left, right):
        return (left.frame == right.frame and math.hypot(left.x - right.x, left.y - right.y) <= .18
                and abs(math.remainder(left.yaw - right.yaw, math.tau)) <= ANGLE / 2)

    def record_observation(self, pose):
        if not any(self._same_vantage(pose, view) for view in self.observed_world):
            self.observed_world.append(pose)

    def _already_observed(self, origin, angle):
        if self.observation_pose is not None:
            pose = self.observation_pose
            f, r = origin[0] - self.pose[0], origin[1] - self.pose[1]
            rotation = pose.yaw + self.pose[2]
            c, s = math.cos(rotation), math.sin(rotation)
            goal = replace(pose, x=pose.x + c*f + s*r, y=pose.y + s*f - c*r,
                           yaw=rotation - origin[2] - angle)
            if any(self._same_vantage(goal, view) for view in self.observed_world):
                return True
        return any(math.dist(origin[:2], view[:2]) <= .18
                   and abs(math.remainder(origin[2] + angle - view[2], math.tau)) <= ANGLE / 2
                   for view in self.observed_vantages)

    def begin(self, directions, *, advance_m=None, turn_angle=ANGLE, route_segment=False, reroute=False):
        if not self.current_frame or (self.attempted and not reroute) or (reroute and self.rerouted):
            return False
        if not route_segment and not self.command_only:
            directions = tuple(side for side in directions
                               if not self._already_observed(self.pose, side * turn_angle))
        if not directions:
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
        self.turn_rate_scale = 1.
        self.start_ms = self.single_control_ms if self.command_only else self.poses[-1][0]
        self.still_since = self.start_ms
        self.timed_phase_s = self.timed_total_s = self.timed_turn_s = 0.
        self.timed_heading = 0.
        return True

    def try_other_side(self):
        self.directions = self.directions[1:]
        if not self.route_segment and not self.command_only:
            origin = self.origin if self.origin_set else self.pose
            self.directions = tuple(side for side in self.directions
                                    if not self._already_observed(origin, side * ANGLE))
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
        if self.route_segment:
            self.attempted = self.exhausted
            self.rerouted = False
        self.phase = "idle"

    def command(self, geometry, half_width, yaw_rate, watchdog_ms, obs_age_ms=500, *,
                min_yaw_rate=0., now_ms=None, tick_s=.1, previous_yaw=None, control_period_s=None,
                previous_command_s=None, yaw_limits=None, previous_linear=None):
        if not self.active or self.ready:
            return ScanCommand()
        if self.command_only:
            if previous_yaw is not None and self.timed_turn_s:
                covered = min(tick_s, watchdog_ms / 1000) if previous_command_s is None else previous_command_s
                self.timed_heading -= previous_yaw * covered
            return self._timed_command(geometry, half_width, yaw_rate, watchdog_ms,
                                       obs_age_ms, min_yaw_rate, tick_s, previous_yaw is not None, yaw_limits)
        stamp = self.poses[-1][0]
        translating = math.hypot(*self.motion[:2]) > self.odom_linear_tolerance_mps
        if not self.current_frame:
            return self.interrupt("主动观察：观测超时，停车")
        rolling_start = (self.phase == "brake" and previous_linear is not None
                         and self.receipt_only and self.buffered_odom and self.pose_origin is None
                         and self.advance_m is not None and abs(self.motion[2]) <= STILL_YAW
                         and abs(previous_yaw or 0.) <= STILL_YAW
                         and max(previous_linear, math.hypot(*self.motion[:2])) > STILL_LINEAR)
        if self.phase == "brake":
            if ((translating and not rolling_start) or abs(self.motion[2]) > STILL_YAW
                    or (self.pose_origin is not None and self.pose_feedback_since_ms < self.still_since)):
                return ScanCommand(reason="主动观察：等待底盘停稳")
            if not self.origin_set:
                self.origin, self.origin_set = self.pose, True
            self.still_since = stamp
            self.turn_start_heading = self.pose[2]
            self.phase = "turn"
            if not rolling_start:
                return ScanCommand(reason="主动观察：确认底盘已停稳")
        # A released straight segment legitimately leaves the turn origin. A
        # replan hold after it must settle and replan, not lock on that travel.
        turning = (self.issued_yaw or rolling_start) and self.phase in ("turn", "return")
        if (self.phase != "brake_for_replan" and (not turning or self.turn_stop_ms is not None)
                and self.origin_set and self.advance_origin is None
                and math.dist(self.pose[:2], self.origin[:2]) > .18):
            return self.interrupt("主动观察：检测到平移，停车")
        if (self.receipt_only and translating and self.phase != "brake_for_replan"
                and not turning):
            return self.hold_for_replan("主动观察：检测到平移，停车后重规划", retry_scan=False)
        if self.phase == "brake_for_replan":
            if stamp <= self.replan_after_ms:
                return ScanCommand(reason="主动观察：停车确认运动反馈")
            if translating or abs(self.motion[2]) > STILL_YAW:
                motion = "检测到平移" if translating else "仍在转动"
                return ScanCommand(reason=f"主动观察：{motion}，停车后重规划")
            interrupted_turn = self.issued_yaw and self.advance_origin is None
            if self.retry_after_hold or not self.issued_yaw or self.advance_origin is not None:
                self.attempted = False
            elif interrupted_turn:
                # A stalled observation can hand off to a checked local route.
                self.exhausted = True
            self.retry_after_hold = self.rerouted = False
            self.route_segment = False
            self.advance_origin = self.advance_m = None
            self.side_frame = self.result_frame = 0
            self.phase = "idle"
            self.issued_yaw = False
            return ScanCommand(reason="主动观察：已停稳，重新规划", replanned_turn=interrupted_turn)
        if translating and not turning:
            return self.interrupt("主动观察：检测到平移，停车")
        if abs(self.motion[2]) > yaw_rate + .10:
            self.reduce_turn_speed(yaw_rate, self.motion[2])
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
            if not fresh or abs(self.motion[2]) > STILL_YAW or translating:
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
        turn_direction = -self.directions[0] if returning else self.directions[0]
        crossed_heading = self.issued_yaw and error * turn_direction <= 0.
        if abs(error) <= ANGLE_TOLERANCE or crossed_heading:
            if abs(self.motion[2]) > STILL_YAW or translating:
                return ScanCommand(reason="主动观察：等待转向停止")
            if returning:
                self.issued_yaw = False
            else:
                self.side_frame = self.current_frame
                if self.views and self.views[-1][0] == self.current_frame:
                    vantage = self.views[-1][1]
                    if not self._already_observed(vantage, 0.):
                        self.observed_vantages.append(vantage)
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
        if 0. < min_yaw_rate <= yaw_rate:
            yaw_rate = max(min_yaw_rate, yaw_rate * self.turn_rate_scale)
        else:
            yaw_rate *= self.turn_rate_scale
        if not (geometry.front >= 0 and geometry.rear > 0 and yaw_rate > 0) or yaw_rate < min_yaw_rate:
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

        # The old forward command can still be executing when the yaw request arrives.
        minimum_translation = previous_linear * horizon if rolling_start else 0.
        translation = max(minimum_translation, self._turn_translation(horizon, self.motion))
        if not self.clear_sweep(geometry, half_width, translation, occupied=occupied):
            if rolling_start:
                self.phase = "brake"
                self.origin_set = False
                return ScanCommand(reason="主动观察：当前速度下转身扫掠区有障碍，停车")
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            if self.turn_rate_scale < 1. and abs(self.motion[2]) > STILL_YAW:
                return ScanCommand(reason=TURN_SPEED_REDUCTION)
            return self.interrupt("主动观察：转身扫掠区有障碍，停车")
        self.issued_yaw = True
        if self.buffered_turn:
            self.turn_sweep = CheckedTurnSweep(geometry, half_width, horizon, occupied, self.pose,
                                               minimum_translation)
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
            if yaw_rate <= 0. or yaw_rate < min_yaw:
                return ScanCommand(reason=TURN_SPEED_REDUCTION)
            self.timed_turn_s = self.timed_phase_s = 0.
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
        if (rate <= 0. or rate < min_yaw or not self.clear_sweep(
                geometry, half_width, STILL_LINEAR * horizon)):
            if not self.issued_yaw and self.try_other_side():
                return ScanCommand(reason="主动观察：尝试另一侧的转身空间")
            return self.interrupt("主动观察：转身扫掠区有障碍，停车")
        self.issued_yaw = True
        return ScanCommand(-self.turn_yaw, "主动观察：有界短转向")

    def turn_obstacles(self):
        return self.grid()[1] | self.views[-1][3]

    def _turn_translation(self, horizon, motion):
        translation = math.hypot(*motion[:2]) * horizon
        if self.receipt_only and self.buffered_odom and self.pose_origin is None and translation:
            translation = max(translation, RESOLUTION)
        return translation

    def clear_sweep(self, geometry, half_width, translation, *, occupied=None):
        if occupied is None:
            occupied = self.turn_obstacles()
        return not motion_sweep_hits(occupied, geometry, half_width, 0., ((0., 0., 0.),),
                                     padding=translation).any()

    def pause_for_depth(self, stop_ms):
        if not self.buffered_turn or not self.issued_yaw:
            return self.hold_for_replan("主动观察：深度暂不可用，停车后重规划")
        # Keep the chosen heading and straight segment through a brief outage.
        # The next burst must use a new view and the normal stopped feedback.
        self.turn_depth_paused = True
        self.current_frame = self.turn_until_ms = 0
        if self.turn_stop_ms is None:
            self.turn_stop_ms = stop_ms
        return ScanCommand(reason=DEPTH_RECHECK_HOLD)

    def hold_for_replan(self, reason, *, retry_scan=True):
        if self.phase != "brake_for_replan":
            self.retry_after_hold = retry_scan
            self.replan_after_ms = self.poses[-1][0] if self.poses else 0
        self.phase = "brake_for_replan"
        self.current_frame = 0
        self.turn_until_ms = 0
        self.turn_stop_ms = None
        self.turn_depth_paused = False
        return ScanCommand(reason=reason)

    def interrupt(self, reason="主动观察：深度或里程计不可用，停车"):
        if self.active and self.issued_yaw and not self.command_only:
            return self.hold_for_replan(reason)
        self.phase = "idle"
        self.retry_after_hold = False
        self.turn_until_ms = 0
        self.turn_stop_ms = None
        return ScanCommand(reason=reason)

    def invalidate(self):
        stop = self.interrupt()
        self.motion_frame = object()
        self.current_frame = 0
        self.pending_depth = None
        self._pending_projection = self._fused_view = None
        self.historical_free = self.near_free = None
        self.source_frame = None
        self.odom_source = None
        self.poses.clear()
        self.views.clear()
        self.observed_vantages.clear()
        self.observation_pose = None
        self.pose = (0., 0., 0.)
        self.pose_origin = None
        self.pose_feedback_since_ms = 0
        self.turn_stop_ms = None
        return stop

    def note_clear_path(self, frame):
        if not frame or not (self.attempted or self.result_frame):
            return
        self.rerouted = False
        self.attempted = False
        self.exhausted = False
        self.issued_yaw = False
        self.result_frame = 0
