"""Stop, turn in place and cross a verified narrow passage on a wheeled base."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from .active_scan import ODOM_GAP_MS, SCAN_TIMEOUT_MS
from .local_path import FORWARD, RESOLUTION, RIGHT
from .odom import PlanarPose, planar_feedback

STOP_DWELL_MS = 200
STILL_LINEAR = .015
STILL_YAW = .015
YAW_TOL = math.radians(.5)
HEADING_OFFSETS = (0.,) + tuple(map(math.radians, (-2, 2, -4, 4, -6, 6)))


def passage_clear(grid, geometry, physical_half_width, half_width, yaw, travel,
                  *, turn_slack=0., drift=0., corridor_reserve=0.):
    """Require observed free cells for the new swept area, including grid edges."""
    free, occupied = grid
    cell = RESOLUTION / math.sqrt(2)
    inside_body = ((FORWARD >= -geometry.rear + cell)
                   & (FORWARD <= geometry.front - cell)
                   & (abs(RIGHT) <= physical_half_width - cell))
    unverified = occupied | (~free & ~inside_body)
    sweeps = np.zeros_like(free)
    # Account for rotation between checks and the maximum travel before a stop.
    radius = math.hypot(max(geometry.front, geometry.rear), half_width)
    start, end = min(0., yaw) - turn_slack, max(0., yaw) + turn_slack
    intervals = max(1, math.ceil((end - start) / .04))
    turning_pad = radius * (end - start) / (2 * intervals) + drift

    def include(angle, distance, pad, width):
        nonlocal sweeps
        c, s = math.cos(angle), math.sin(angle)
        forward, right = c * FORWARD - s * RIGHT, s * FORWARD + c * RIGHT
        for left in (-width - pad, width + pad):
            for end_f in (-geometry.rear - pad, geometry.front + distance + pad):
                f, r = c * end_f + s * left, -s * end_f + c * left
                if not (FORWARD[0, 0] <= f <= FORWARD[-1, 0]
                        and RIGHT[0, 0] <= r <= RIGHT[0, -1]):
                    return False
        sweeps |= ((forward >= -geometry.rear - pad - cell)
                   & (forward <= geometry.front + distance + pad + cell)
                   & (abs(right) <= width + pad + cell))
        return True

    if not all(include(yaw_sample, 0., turning_pad, half_width)
               for yaw_sample in np.linspace(start, end, intervals + 1)):
        return False
    if travel and not include(yaw, travel, drift, half_width + corridor_reserve):
        return False
    return not (sweeps & unverified).any()


@dataclass
class Doorway:
    obstacle_depth: float
    phase: str = "brake"
    stopped_since: int | None = None
    last_stamp: int = 0
    origin: PlanarPose | None = None
    heading: float = 0.
    travel: float = 0.
    crossing_start: PlanarPose | None = None
    proof_since: int = 0
    evidence_ms: int = 0

    def interrupt(self, reason):
        self.stopped_since = None
        if self.phase == "turn":
            self.phase = "rebrake"
        return 0., 0., reason

    def step(self, odom, scan, geometry, config, ahead, target_distance):
        self.evidence_ms = 0
        odom = odom or {}
        pose = odom.get("door_pose")
        motion = planar_feedback(odom)
        capture = scan.current_frame
        if (geometry is None or pose is None or motion is None or scan.receipt_only
                or scan.pose_origin is None or scan.pose_origin.frame != pose.frame
                or (self.origin is not None and pose.frame != self.origin.frame)
                or not 0 < capture <= pose.stamp_ms
                or pose.stamp_ms - capture >= config.depth_age_limit_ms):
            return self.interrupt("门口停车：缺少可信的车体几何、位姿或同步深度")
        if self.last_stamp and not 0 <= pose.stamp_ms - self.last_stamp <= ODOM_GAP_MS:
            self.last_stamp = pose.stamp_ms
            return self.interrupt("门口停车：里程计不连续")
        self.last_stamp = pose.stamp_ms
        if (self.phase in ("turn", "rebrake", "settle") and self.origin is not None
                and pose.stamp_ms - self.origin.stamp_ms >= SCAN_TIMEOUT_MS):
            return self.interrupt("门口停车，未能在观察时限内完成对中")

        vx, vy, wz = motion
        speed = math.hypot(vx, vy)
        horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
        width = config.half_width_m
        keep = width + max(.10, config.clearance_margin_m)
        room = target_distance - config.stop_distance_m

        projected = {}

        def clear(since, yaw, distance, **sweep):
            if capture < since:
                return False
            since = max(since, pose.stamp_ms - config.depth_age_limit_ms)
            if since not in projected:
                projected[since] = scan.grid(since_ms=since)
            if passage_clear(projected[since], geometry, width, keep, yaw, distance, **sweep):
                self.evidence_ms = next(stamp for stamp, *_ in scan.views if stamp >= since)
                return True
            return False

        if self.phase in ("brake", "rebrake", "settle"):
            if speed > STILL_LINEAR or abs(wz) > STILL_YAW:
                self.stopped_since = None
            elif self.stopped_since is None:
                self.stopped_since = pose.stamp_ms
            after_stop = (self.stopped_since or pose.stamp_ms) + STOP_DWELL_MS
            if capture <= after_stop:
                return 0., 0., "门口停车：等待停稳后的新深度"
            if self.phase == "brake":
                still_blocked = ahead is not None and ahead < config.slow_distance_m
                tf = geometry.base_from_optical
                original = self.obstacle_depth
                obstacle = max(original, ahead) if still_blocked else original
                beyond = obstacle * tf[0, 2] + tf[0, 3] + geometry.rear + .15
                headings = HEADING_OFFSETS if still_blocked else (0.,)
                # Prefer a centerline with room for grid re-registration.
                for reserve, yaw in product((RESOLUTION, 0.), headings):
                    distance = beyond / math.cos(yaw)
                    if (0. < distance < room
                            and clear(after_stop + 1, yaw, distance, turn_slack=YAW_TOL,
                                      drift=STILL_LINEAR * horizon, corridor_reserve=reserve)):
                        self.origin, self.heading, self.travel = pose, pose.yaw + yaw, distance
                        self.phase, self.proof_since = "turn", after_stop + 1
                        break
                else:
                    return 0., 0., "门口停车：转向扫掠区或前进走廊缺少可通行证据"
            else:
                error = math.remainder(self.heading - pose.yaw, math.tau)
                if abs(error) > YAW_TOL:
                    self.phase, self.proof_since = "turn", after_stop + 1
                else:
                    if room <= self.travel or not clear(after_stop + 1, 0., self.travel):
                        return 0., 0., "门口停车：等待重新确认正前方走廊"
                    self.crossing_start = pose
                    self.phase, self.proof_since = "cross", after_stop + 1

        error = math.remainder(self.heading - pose.yaw, math.tau)
        if self.phase == "turn":
            if abs(error) <= YAW_TOL:
                self.phase, self.stopped_since = "settle", None
                return 0., 0., "门口对中：转向结束，停车复核"
            origin_f, origin_l, _ = pose.relative_to(self.origin)
            # Keep the commanded turn inside ActiveScan's stable-exposure limit.
            yaw_rate = math.copysign(min(.03, config.wz_max, abs(error) / (2 * horizon)), error)
            if (speed > STILL_LINEAR or math.hypot(origin_f, origin_l) > .02
                    or wz * yaw_rate < -STILL_YAW * abs(yaw_rate)
                    or abs(yaw_rate) < config.floor_wz
                    or not clear(self.proof_since, error, self.travel,
                                 turn_slack=max(YAW_TOL, abs(wz) * horizon),
                                 drift=STILL_LINEAR * horizon)):
                return self.interrupt("门口停车：转向扫掠区或前进走廊无法继续确认")
            return 0., yaw_rate, "门口对中：仅转向"

        progress, lateral, _ = pose.relative_to(self.crossing_start)
        remaining = self.travel - progress
        if (remaining <= .02 and abs(lateral) <= .02 and abs(error) <= YAW_TOL
                and speed <= STILL_LINEAR and abs(wz) <= STILL_YAW):
            self.phase = "done"
            return 0., 0., "门口通过：停车后恢复跟踪"
        floor = config.floor_vx_moving if config.floor_vx_moving is not None else config.floor_vx
        speed_limit = min(.12, config.vx_max, max(0., min(remaining, room)) / horizon)
        stop_travel = max(speed_limit, vx) * horizon
        radius = math.hypot(max(geometry.front, geometry.rear), keep)
        drift = (abs(vy) + (radius + stop_travel) * abs(wz)) * horizon
        if (abs(error) > YAW_TOL or abs(lateral) > .02 or abs(wz) > STILL_YAW
                or abs(vy) > STILL_LINEAR or vx < -STILL_LINEAR or stop_travel > room
                or speed_limit < floor or not clear(self.proof_since, 0.,
                                                    max(remaining, stop_travel), drift=drift)):
            return self.interrupt("门口停车：前进走廊或底盘运动超出已验证范围")
        return speed_limit, 0., "门口对中完成：直行通过已验证走廊"
