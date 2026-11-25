"""The control law: one observation in, one body twist out.

Every decision this card makes is here, as a pure function over plain data. No
ROS, no publisher, no clock of its own — `plugin.py` is this plus a timer and a
topic. That split is what makes "what does it do when the target disappears
behind a person while an obstacle is on the left" a test rather than an
afternoon on a robot.

── the shape of the behaviour ───────────────────────────────────────────────

Reactive visual servoing, and nothing more. There is no map, no global path,
and no memory of where the target was a minute ago. It will not get around a
U-shaped obstacle, and it is not supposed to: `path_planner.py` in the go2
bundle is the other approach, and conflating the two gives something that is
bad at both.

  1. pick the target out of what vop reports
  2. travel towards it      vx and vy, decomposed along its bearing
  3. turn to face it        wz from the same bearing, at the same time
  4. slow and sidestep for obstacles the depth bands report
  5. stop within the configured distance band and face the target

── all three axes move together, and that is the whole point ────────────────

Steps 2 and 3 run on the same tick. The first version of this file ran them in
sequence — `vx = 0` until the target was roughly centred — and a base that can
translate sideways has no reason to do that: the approach velocity is simply
decomposed along the target's bearing (`half_fov_rad`), so the robot walks the
straight line to the thing while turning to face it. What that removes is the
stop-turn-go gait, which on r1_sz read as a lurch at every bearing correction.

── the robot cannot move slowly, and that shapes everything above ───────────

A legged base may have to assemble a whole gait cycle, so below its declared
minimum magnitude it does not move at all even though the SDK accepted the
command. A proportional law then cannot make a small correction *slowly*; it
can only make it *briefly*. Every axis is therefore a switch with hysteresis
(`Gate`) whose magnitude, once on, is lifted to the Driver-declared floor
(`_lift`).

Two consequences that are not obvious and were both bugs here:

* **A ceiling below the floor makes an axis bang-bang.** `wz_max` was 0.8 against
  a 1.0 rad/s floor, so every turn command in the robot's entire reachable range
  snapped to either 0 or 1.0 — a 10 Hz square wave, which is what "顿挫" was.
  `adopt_limits` now raises the ceilings off the floor at start.
* **Proportional control dies in the last stretch.** `k_fwd * (d - stop)` falls
  under the floor 0.67 m before arriving, so the robot stopped short of a target
  it could see and then failed on the idle timeout. The forward gate holds the
  floor speed until the stop distance is genuinely reached.

── the safety property that matters more than any of the above ──────────────

When the policy goes quiet after commanding motion, the card sends one zero
twist if the downstream watchdog exceeds 300 ms. The driver watchdog still
protects an interrupted command stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial

import numpy as np

from .camera import HorizontalIntrinsics
from .camera import half_fov as _resolve_half_fov
from .active_scan import (ANGLE, ANGLE_TOLERANCE, ActiveScan, CAMERA_OCCUPANCY_MEMORY_RANGE_M,
                          DEFAULT_FEEDBACK_GAP_MS, MAP_AGE_MS, ODOM_GAP_MS,
                          NearProjection, SCAN_ODOM_LINEAR_TOLERANCE_MPS, STILL_LINEAR, STILL_YAW,
                          TURN_SPEED_REDUCTION, confirmed_square_coverage, relative_pose_at)
from .depth import has_valid_returns, observation_ms
from .track import AssociationSnapshot, ObservationFrame, Track, Tracker, observation_source, observation_transform, target_objects
from . import local_path
from . import dwa
from . import motion as motion_model
from .local_path import FORWARD, RIGHT, Detour, Geometry, PlanCache
from .odom import PlanarPose, planar_axes, planar_feedback

# Status values. `arrived` and `stuck` are terminal for one `navigate_to`;
# the rest are things that happen on the way.
APPROACHING = "approaching"
ALIGNING = "aligning"
AVOIDING = "avoiding"
SEARCHING = "searching"
ARRIVED = "arrived"
STUCK = "stuck"
BLIND = "blind"          # no usable observation — distinct from "nothing seen"
FAILED = "failed"        # terminal, and the caller is owed this answer
IDLE = "idle"
DEPTH_CAPTURE_STALE = "等待新的有效深度"


class LocalHold(Enum):
    TRANSITION = "等待底盘速度过渡，保留当前绕行路线"
    PREFIX = "局部规划：找到候选路线，首段不满足执行条件；停下"

# A distance schedule, not online plant identification. Bounds are fixed so a
# stalled chassis cannot learn an ever-larger command or a new speed floor.
ADAPTIVE_FWD_MAX_SCALE = 3.0
ADAPTIVE_FWD_SCALE_RATE = 2.0  # maximum scale change per second
ADAPTIVE_FWD_MAX_DT = 0.2     # a long scheduling gap must not jump the gain

# The blocked-route buffer reserves braking room outside the obstacle stop.
BLOCKED_APPROACH_BUFFER_M = 0.30
LOCAL_BLOCKED_APPROACH_BUFFER_M = 0.05
LOCAL_BLOCKED_APPROACH_MAX_MPS = 0.15

# Longitudinal separation for identifying a foreground obstacle and resuming.
FORWARD_CLEARANCE_MARGIN_M = 0.15
SIDE_SWITCH_ADVANTAGE_M = 0.15
SIDE_SWITCH_FRAMES = 3


@dataclass
class Config:
    """Everything an operator can turn. Defaults are for a walking humanoid.

    The three distances are **not** independent, and the relation between them
    is the one thing to preserve when tuning:

        obstacle_stop_m  <  stop_distance_m  <  slow_distance_m

    The target is itself an obstacle — it appears in the same depth bands — so
    if `obstacle_stop_m` ever reaches `stop_distance_m` the robot is stopped by
    the very thing it is walking towards, and never arrives.

    They are also measured **from the camera**, not from the front of the robot.
    On a walking humanoid the torso and a swinging leg are both ahead of the
    lens, which is why `obstacle_stop_m` is not as small as the geometry alone
    would suggest.

    Still unvalidated on hardware — see docs/visual-navigation.md.
    """

    stop_distance_m: float = 0.8
    # Allowed radial error around the nominal camera-to-target distance.
    # Five centimetres is the generic camera-depth tolerance. Live
    # target/heading requirements still apply; this is not a tolerance for
    # stale or predicted observations. Callers may request zero per action.
    stop_tolerance_m: float = 0.05
    slow_distance_m: float = 1.0
    side_clearance_m: float = 1.0
    local_planning: bool = False
    # Navi enables DWA by default; legacy remains an internal fallback/reference.
    local_controller: str = "dwa"  # "legacy" or "dwa"
    dwa_horizon_s: float = 1.2
    dwa_samples: int = 64
    motion_acceleration: tuple = motion_model._DEFAULT_ACCELERATION
    motion_braking: tuple = motion_model._DEFAULT_BRAKING
    odom_expected: bool = False
    local_geometry: Geometry | None = None
    scan_geometry: Geometry | None = None
    obstacle_stop_m: float = 0.5
    # Lateral offset that still allows full speed. **Not a gate** — past it the
    # speed is scaled down, reaching zero at align_full_stop. See
    # `_alignment_scale`.
    align_tol: float = 0.15
    # Turn only, no travel, past this: the target is near the edge of frame,
    # and driving straight at it is driving somewhere else.
    align_full_stop: float = 0.45
    # Alignment tolerance for *arriving*, far looser than the one above — and
    # it has to be.
    #
    # `bearing` is a normalised lateral offset, not an angle: the same sideways
    # step barely moves the frame at 5 m and swings it several tolerances wide
    # at 0.7 m. Requiring `align_tol` to arrive therefore asks a person standing
    # in front of the robot to hold still. On r1_sz the distance had long since
    # reached 0.73 m while the card kept reporting "turning in place first" and
    # never arrived — and `idle_timeout_s` could not catch it either, because
    # turning counts as motion and an oscillating aligner never stalls.
    arrive_align_tol: float = 0.30
    # How long the corridor may stay blocked before the task gives up **naming
    # the obstacle**. Without it a blocked robot sidesteps and turns for ever:
    # it is commanding motion, so the idle timeout never fires, and the caller
    # is told nothing. Generous, because stepping out from in front of something
    # legitimately takes a few seconds.
    blocked_timeout_s: float = 12.0
    # Speed retained when the target is at `align_full_stop` or further off.
    # 0.0 restores the original behaviour — stop dead and turn in place — and is
    # the right setting for a base that cannot translate sideways. Anything
    # above 0 keeps the robot moving through the correction, which is what makes
    # the path an arc instead of a sequence of stops.
    align_min_scale: float = 0.45
    k_yaw: float = 0.6
    k_fwd: float = 0.6
    adaptive_k_fwd: bool = False
    # Optional effective P derived from the downstream Driver's declared yaw
    # response.  Navi never measures or persists a robot-specific plant model;
    # the component that owns the hardware owns that declaration.
    response_yaw_gain: float | None = None
    vx_max: float = 1.0
    vy_max: float = 1.0
    wz_max: float = 1.5
    # The one place a normalised bearing becomes an angle.
    #
    # `position[0]` from vop is a fraction of the image half-width, **not** a
    # heading — which is why `align_tol` and friends are all in those same
    # normalised units and must stay that way.
    #
    # **A fallback, not the answer.** The camera declares this now
    # (`motus.camera/1`, see `_adopt_camera`); this value is what gets used when
    # nothing upstream says. It is deliberately on the *small* side, because the
    # two failure directions are not symmetric — see `_adopt_camera`.
    #
    # Two consumers, and the second is the one that bites. Splitting the approach
    # velocity between vx and vy needs a real angle, and being 20% out there only
    # makes the arc slightly wide because the yaw loop closes it anyway. But the
    # avoidance corridor is metric, so every tick it converts a half-width back
    # into a column range using this — and being 1.6x out there made the corridor
    # wider than any door.
    half_fov_rad: float = 0.888
    depth_intrinsics: HorizontalIntrinsics | None = None
    # Sidestep while approaching, and to get out from in front of an obstacle.
    # Turn it off for a base with no lateral degree of freedom — the descriptor
    # pinning vy to zero does that on its own, but this says so in one place.
    use_lateral: bool = True
    # Scales the lateral component only. Below 1.0 the robot leans on turning
    # more than on strafing, which is the conservative direction: the depth map
    # is forward-facing, so sideways is the direction it knows least about.
    lateral_scale: float = 1.0
    # Past this offset, turn rather than strafe: the bearing estimate is worst at
    # the frame edge, and so is the depth band that would have to clear it.
    lateral_max_bearing: float = 0.7
    # Strafe only when the honest lateral speed is at least this share of the
    # robot's lateral floor — i.e. when `_lift` amplifies by no more than 1/this.
    # See `_worth_strafing`: without it, a target a few percent off centre
    # produced a full-speed sidestep.
    lateral_lift_ratio: float = 0.5
    # Hysteresis. An axis engages at its threshold and releases at this fraction
    # of it — without the gap, an error hovering at the threshold toggles the
    # axis at tick rate, and on a deadbanded robot that toggle is full speed to
    # nothing and back.
    release_frac: float = 0.4
    # ...and stays engaged at least this long, so one noisy frame cannot end a
    # turn that has only just started.
    min_dwell_s: float = 0.25
    # Driver limits fit this rate to the executable bidirectional yaw envelope.
    search_rate: float = 0.6
    # Backstop only. The sweep below is what normally ends a search, because it
    # is behavioural — "I have turned all the way round and it is not here" —
    # whereas a clock says nothing about what the robot did with the time.
    search_timeout_s: float = 30.0
    # **How long the card may go without commanding any motion before the task
    # is declared failed.** This replaces a per-phase wall-clock timeout, and
    # the difference matters: a robot walking steadily towards something far
    # away must never fail for taking a while, while a robot that has emitted
    # nothing for ten seconds has plainly stopped making progress whatever the
    # reason — blind, occluded, refused downstream, deciding nothing.
    #
    # Turning counts as motion. Publishing an all-zero twist does not.
    idle_timeout_s: float = 10.0
    search_sweep_rad: float = 6.4        # a bit over one full turn
    lost_frames: int = 10
    max_obs_age_ms: int = 500
    max_depth_age_ms: int | None = None
    scan_odom_linear_tolerance_mps: float = SCAN_ODOM_LINEAR_TOLERANCE_MPS
    gap_speed_bounds: tuple | None = None
    use_obstacle_history: bool = False
    max_feedback_gap_ms: int = DEFAULT_FEEDBACK_GAP_MS
    watchdog_ms: float = 300
    stuck_window_s: float = 3.0
    stuck_speed_ratio: float = 0.2
    # Confidence needed to **start** following something. Below this a
    # detection may still sustain a track that already exists — see
    # `sustain_confidence`.
    min_confidence: float = 0.35

    # ── the space the robot occupies (see depth.corridor) ────────────────
    # Half the robot's width, in metres, **as the chassis declares it** — the
    # `footprint` block of its descriptor, read by `adopt_limits`. The default
    # here is deliberately wider than any humanoid in this project: a robot that
    # does not say how wide it is must be assumed to be wide, because the error
    # that hurts is believing it is narrow.
    half_width_m: float = 0.35
    # Prefer a better observed side when ordering detour candidates.
    coverage_min: float = 0.25

    # ── tracking (see track.py) ──────────────────────────────────────────
    # A detection this dim may not create a track, but it may keep one alive if
    # it lands inside the gate. Partial occlusion is exactly what makes a
    # detector lose confidence, so throwing this band away throws away the
    # frames the occlusion produced. ByteTrack's second association.
    #
    # It is only useful to the extent vop publishes that band at all. vop's own
    # `confidence` defaults to 0.3, so the usable low band today is just
    # [0.30, 0.35) — narrow but not empty. Widening it means lowering vop's
    # threshold, which inflates the whole detection stream (vop has no
    # `max_objects` cap and the payload reaches LLM context whole), so that is
    # a measurement to take on hardware rather than a default to change here.
    sustain_confidence: float = 0.15
    # Three of the last five frames before a track may move the robot. The old
    # behaviour was one frame to start and ten to give up — an asymmetry
    # pointing the wrong way, since starting is the direction that moves a
    # robot. (ByteTrack's default is two; this is deliberately stricter,
    # because a wrong start here walks a humanoid at somebody.)
    confirm_hits: int = 3
    confirm_window: int = 5
    # How long a confirmed track may be driven from prediction alone. A person
    # walking across in front of the robot is under a second; past that the
    # extrapolation is a guess, and OC-SORT measured how fast that guess rots —
    # ten frames of coasting can accumulate an error the size of the object.
    max_coast_s: float = 1.2
    # Association gate, as a chi-square on 2 DOF. 9.21 is the 99% contour:
    # generous, because there is only one track and the cost of dropping it is
    # a spurious search.
    gate_chi2: float = 9.21
    # Added to the cost when a candidate's hue disagrees with the track's. A
    # tie-break, never a rejection — see `_gate_cost`.
    hue_mismatch_cost: float = 4.0
    # What the *target* might do that the constant-velocity model does not
    # cover. Our own motion is not in here: it is known, not guessed.
    target_accel_std: float = 1.5      # m/s^2
    range_std_m: float = 0.15
    bearing_std_rad: float = 0.05
    # Measurement noise multiplier when depth had no reading for the object and
    # the track's own range had to stand in. Stops a bearing-only update from
    # asserting a distance it never measured.
    bearingless_std_factor: float = 6.0
    initial_speed_std: float = 1.0     # m/s, before any velocity is observed
    # Physical bound on what the *target* can be doing. See `_clamp_speed` —
    # this is what stops a detection that jumped across the frame from being
    # read as an object moving at 8 m/s.
    max_target_speed: float = 2.5      # m/s
    # How much less the tracker's predict step is trusted when our own motion
    # is the command we sent rather than a measurement. See track._predict.
    commanded_ego_noise_factor: float = 4.0
    # Where to put a target the depth source has never measured, so that it can
    # be tracked in bearing at all. Never used as a distance by the policy —
    # `track.range_known` stays false and `distance` stays None, which is what
    # keeps an assumed number out of the arrival test.
    assumed_range_m: float = 3.0

    # Filled in at start from the downstream descriptor's `min_magnitude` — the
    # robot's own deadband, per axis, in its own units. Left at 0 there is no
    # deadband to work around and every axis is plain proportional control,
    # which is the correct behaviour for a wheeled base.
    #
    # Not operator-editable: it is a reading of the hardware, and a hand-typed
    # copy is a copy that goes stale. See `adopt_limits`.
    floor_vx: float = 0.0
    floor_vy: float = 0.0
    floor_vx_moving: float | None = None
    floor_vy_moving: float | None = None
    left_vy_limit: float | None = None
    right_vy_limit: float | None = None
    scan_wz_limit: float | None = None
    left_wz_limit: float | None = None
    right_wz_limit: float | None = None
    floor_wz: float = 0.0
    # **The yaw floor while the robot is already translating**, which on a
    # legged robot is a different number entirely: a gait cycle that is running
    # can be steered a little per step, one that has to be started cannot.
    # Measured on r1_sz — standing, nothing under 1.0 rad/s moves it at all;
    # walking, a commanded 0.05 rad/s is visible. Twenty times smaller.
    #
    # Lifting every yaw command to the standing floor mid-approach overshoots a
    # small correction twenty-fold, reverses, and overshoots again. That is what
    # "the robot weaves left and right on its way to a target it is already
    # facing" turned out to be.
    floor_wz_moving: float = 0.0

    @property
    def depth_age_limit_ms(self) -> int:
        return self.max_obs_age_ms if self.max_depth_age_ms is None else self.max_depth_age_ms

    @property
    def approach_check_distance_m(self) -> float:
        # Keep checking obstacles before they enter the speed reduction band.
        return max(1.5, self.slow_distance_m)

    @property
    def clearance_margin_m(self):
        return local_path.clearance_margin(self.half_width_m)

    def __post_init__(self):
        if self.local_controller not in ("legacy", "dwa"):
            raise ValueError("local_controller must be 'legacy' or 'dwa'")
        if type(self.dwa_samples) is not int or not 1 <= self.dwa_samples <= 256:
            raise ValueError("dwa_samples must be an integer in [1, 256]")
        if (type(self.dwa_horizon_s) not in (int, float)
                or not math.isfinite(self.dwa_horizon_s)
                or not .2 <= self.dwa_horizon_s <= 3.):
            raise ValueError("dwa_horizon_s must be in [0.2, 3.0]")
        if self.max_depth_age_ms is not None and (
                type(self.max_depth_age_ms) is not int or self.max_depth_age_ms <= 0):
            raise ValueError("max_depth_age_ms must be a positive integer")
        for key in ("response_yaw_gain",):
            value = getattr(self, key)
            if value is None:
                continue
            if isinstance(value, bool):
                raise ValueError(f"{key} must be a finite nonnegative number")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{key} must be a finite nonnegative number") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            setattr(self, key, value)
        for flag in ("adaptive_k_fwd",):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f"{flag} must be a boolean")
        for key, adaptive, maximum in (("k_fwd", self.adaptive_k_fwd, ADAPTIVE_FWD_MAX_SCALE),
                                       ("k_yaw", False, 1.0)):
            if isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be finite and nonnegative")
            try:
                base_gain = float(getattr(self, key))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{key} must be finite and nonnegative") from error
            if not math.isfinite(base_gain) or base_gain < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            if adaptive and not math.isfinite(base_gain * maximum):
                raise ValueError(f"adaptive {key} and its maximum must be finite and nonnegative")
            setattr(self, key, base_gain)
        for key in ("stop_distance_m", "stop_tolerance_m", "scan_odom_linear_tolerance_mps"):
            value = getattr(self, key)
            if isinstance(value, bool):
                raise ValueError(f"{key} must be a finite nonnegative number")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{key} must be a finite nonnegative number") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            setattr(self, key, value)
        if not math.isfinite(self.arrival_distance_m):
            raise ValueError("stop_distance_m + stop_tolerance_m must be finite")

    @property
    def arrival_min_distance_m(self):
        """Minimum camera-to-target range in the parking ring."""
        return max(0., self.stop_distance_m - self.stop_tolerance_m)

    @property
    def arrival_distance_m(self):
        """Maximum camera-to-target range that satisfies the distance test."""
        return self.stop_distance_m + self.stop_tolerance_m


@dataclass
class Gate:
    """Whether one axis is commanding motion right now.

    A switch rather than a gain, because a robot with a deadband has no slow
    regime to be proportional in. The error decides *whether* to move; `_lift`
    decides how fast, and the answer is "at least the floor".

    Hysteresis plus a minimum dwell, and both are load-bearing. Without the gap
    between `engage` and `release`, an error sitting on the threshold turns the
    axis on and off every tick; without the dwell, a single bad detection frame
    ends a turn 100 ms after it started. Either one produces the same symptom —
    a robot that judders instead of moving.
    """

    on: bool = False
    held_s: float = 0.0

    def update(self, error: float, *, engage: float, release: float,
               dt: float, min_dwell_s: float) -> bool:
        self.held_s += dt
        if self.on:
            if error <= release and self.held_s >= min_dwell_s:
                self.on = False
                self.held_s = 0.0
        elif error >= engage:
            self.on = True
            self.held_s = 0.0
        return self.on


def _measured_displacement(start_ms, end_ms, odom, fallback):
    """Integrate measured body twists over the requested host-time interval."""
    x = y = yaw = 0.
    previous = start_ms
    history = (odom or {}).get("velocity_history", ())
    for stamp, _, twist in (*history, (end_ms, None, fallback)):
        if stamp <= previous or twist is None:
            continue
        dt = (min(end_ms, stamp) - previous) / 1000
        vx, vy, wz = twist
        turn = wz * dt
        half = .5 * turn
        length = dt * (math.sin(half) / half if half else 1.)
        c, s = math.cos(yaw + half), math.sin(yaw + half)
        x += length * (c * vx - s * vy)
        y += length * (s * vx + c * vy)
        yaw += turn
        previous = stamp
        if stamp >= end_ms:
            break
    return x, y, yaw


def target_body_pose(stamp, odom, fallback, capture=None):
    """Use a body pose at the observation time when one is available."""
    odom = odom or {}
    poses = [p for p in (capture, odom.get("obstacle_pose"), odom.get("motion_pose"))
             if p is not None and p.stamp_ms <= stamp]
    if not poses:
        return None
    pose = max(poses, key=lambda p: p.stamp_ms)
    if pose.stamp_ms == stamp:
        return pose
    x, y, yaw = _measured_displacement(pose.stamp_ms, stamp, odom, fallback)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return PlanarPose(stamp, pose.frame, pose.x + c * x - s * y, pose.y + s * x + c * y, pose.yaw + yaw)


def target_body_delta(reference, start_ms, end_ms, odom, fallback, pose):
    if reference is not None and pose is not None and reference.frame == pose.frame:
        return pose.relative_to(reference)
    return _measured_displacement(start_ms, end_ms, odom, fallback)


@dataclass
class DetourGoal:
    """Last trusted target point, transported by ego motion during a detour."""
    x: float
    y: float
    pose: PlanarPose | None = None
    predicted_m: float = 0.
    predicted_rad: float = 0.
    track: Track | None = None
    verified_range_m: float | None = None
    best_verified_distance_m: float | None = None
    approach_start_range_m: float | None = None
    approach_travel_m: float = 0.
    target_missing: bool = False

    generation: int | None = None
    reference_ms: int | None = None
    frame: ObservationFrame | None = None

    def advance(self, odom, command, dt, *, approach_command=None, geometry=None, now_ms=None):
        odom = odom or {}
        pose = None if odom.get("velocity_history") else odom.get("motion_pose")
        axes = planar_axes(odom)
        pose_delta = pose is not None and self.pose is not None and pose.frame == self.pose.frame
        if geometry is not None:
            now_ms = now_ms if now_ms is not None else (self.reference_ms or 0) + round(1000 * dt)
            start = self.reference_ms if self.reference_ms is not None else now_ms - round(1000 * dt)
            pose = target_body_pose(now_ms, odom, command)
            pose_delta = pose is not None and self.pose is not None and pose.frame == self.pose.frame
            dx, dy, yaw = target_body_delta(self.pose, start, now_ms, odom, command, pose)
            self.reference_ms = now_ms
        elif pose_delta:
            dx, dy, yaw = pose.relative_to(self.pose)
        else:
            vx, vy, wz = (value if value is not None else fallback
                          for value, fallback in zip(axes, command))
            yaw = wz * dt
            c, s = math.cos(yaw / 2), math.sin(yaw / 2)
            dx, dy = dt * (c * vx - s * vy), dt * (s * vx + c * vy)
        if not pose_delta:
            self.predicted_m += dt * math.hypot(command[0] if axes[0] is None else 0.,
                                                command[1] if axes[1] is None else 0.)
            if axes[2] is None:
                self.predicted_rad += abs(yaw)
        if approach_command is not None and (pose_delta or all(value is not None for value in axes[:2])):
            if self.approach_start_range_m is None:
                self.approach_start_range_m = self.verified_range_m
            vx, vy = approach_command
            progress = (dx * vx + dy * vy) / math.hypot(vx, vy)
            if abs(progress) > STILL_LINEAR * dt:
                self.approach_travel_m = max(0., self.approach_travel_m + progress)
        self.pose = pose
        if geometry is not None:
            frame = ObservationFrame.from_geometry(geometry)
            linear, delta = observation_transform((dx, dy, yaw), frame, self.frame)
            self.frame = frame
            self.x, self.y = linear @ (np.array([self.x, self.y]) - delta)
            return
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = self.x - dx, self.y - dy
        self.x, self.y = c * x + s * y, -s * x + c * y


EXECUTION_YAW_TOLERANCE_RAD_S = .5
EXECUTION_HEADING_TOLERANCE = math.radians(5)


@dataclass(frozen=True)
class PathLeg:
    """One checked translation, with a waypoint fixed in the odometry frame."""

    geometry: local_path.Geometry
    waypoint: tuple[float, float]   # forward, right in the origin's frame
    origin: tuple[float, float, float]
    motion_frame: object | None = None
    goal: tuple[float, float] | None = None
    command: tuple[float, float] = (0., 0.)

    @staticmethod
    def world_point(pose, point):
        f, r, heading = pose
        x, y = point
        c, s = math.cos(heading), math.sin(heading)
        return f + c * x - s * y, r + s * x + c * y

    def remaining(self, pose):
        f, r = self.world_point(self.origin, self.waypoint)
        x, y, heading = pose
        c, s = math.cos(heading), math.sin(heading)
        return c * (f - x) + s * (r - y), -s * (f - x) + c * (r - y)

    def check(self, pose, motion, previous, command, grid, keep, horizon, brake_horizon, *, padding=0.):
        f, r = self.remaining(pose)
        vx, vy = command
        speed = math.hypot(vx, vy)
        yaw = abs(motion[2])
        heading_error = abs(math.remainder(pose[2] - self.origin[2], math.tau))
        if yaw > EXECUTION_YAW_TOLERANCE_RAD_S or heading_error > EXECUTION_HEADING_TOLERANCE:
            return "局部规划：朝向偏离执行段，停车后重规划"
        along_origin, across_origin = self.remaining((*pose[:2], self.origin[2]))
        if abs(along_origin * vy + across_origin * vx) > speed * local_path.RESOLUTION:
            return "局部规划：平移偏离执行段，停车后重规划"
        distance = (f * vx - r * vy) / speed
        measured = motion[:2]
        velocities = (measured, previous, command)
        stopping = max(speed * horizon, math.hypot(*measured) * brake_horizon,
                       math.hypot(*previous) * brake_horizon)
        if distance <= local_path.RESOLUTION / 2 or stopping > distance + 1e-9:
            return "局部规划：已到执行段制动点"
        radius = math.hypot(max(self.geometry.front, self.geometry.rear), keep)
        padding += (radius + speed * horizon) * yaw * horizon
        if grid is None or not local_path.transition_clear(
                grid, self.geometry, keep, horizon, velocities, padding=padding):
            return "局部规划：执行段扫掠区有障碍或观测不可用，停车后重规划"
        return None


@dataclass
class PassageCoverage:
    """Measured pivot locations in one continuous frame, bounded without evicting old visits."""
    frame: tuple | None = None
    level: int = 0
    tiles: dict[tuple[int, int], tuple[int, int, int, int]] = field(default_factory=dict)

    @staticmethod
    def _include(bounds, other):
        if bounds is None:
            return other
        return (min(bounds[0], other[0]), min(bounds[1], other[1]),
                max(bounds[2], other[2]), max(bounds[3], other[3]))

    def visit(self, pose):
        if pose.frame != self.frame:
            self.frame, self.level = pose.frame, 0
            self.tiles.clear()
        # A 5 cm measured step cannot stay inside one 2.5 cm square.
        unit = local_path.RESOLUTION / 2
        x, y = math.floor(pose.x / unit), math.floor(pose.y / unit)
        key = x >> self.level, y >> self.level
        bounds = self.tiles.get(key)
        if bounds is not None and bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
            return False
        self.tiles[key] = self._include(bounds, (x, y, x, y))
        # Retain the actually explored extents in each coarser tile.
        # Outward travel can still extend them by a single fine cell.
        while len(self.tiles) > 1024:
            merged = {}
            for (i, j), extent in self.tiles.items():
                parent = i >> 1, j >> 1
                merged[parent] = self._include(merged.get(parent), extent)
            self.tiles = merged
            self.level += 1
        return True


@dataclass
class DirectApproachFix:
    """An observed target in the measured pivot's continuous navigation frame."""
    frame: tuple
    target: tuple[float, float]
    distance: float
    travelled: bool = False


@dataclass
class DwaProgress:
    """Body-pivot measurements and finite view coverage for one navigate action."""
    # Once feedback is expected or used, losing it cannot turn commands into
    # evidence of progress. Only a new navigation action resets this latch.
    requires_feedback: bool = False
    pose: PlanarPose | None = None
    source: int | None = None
    interrupted: bool = False
    checkpoint: PlanarPose | None = None
    direct_anchor: PlanarPose | None = None
    scan_reference: tuple[PlanarPose, PlanarPose] | None = None
    along_m: float = 0.
    route: tuple = ()
    route_origin: PlanarPose | None = None
    view_frame: tuple | None = None
    viewed: list[float] = field(default_factory=list)
    last_view: dict = field(default_factory=dict)
    world_passage: PassageCoverage = field(default_factory=PassageCoverage)
    relative_passage: PassageCoverage = field(default_factory=PassageCoverage)
    approach_fix: DirectApproachFix | None = None
    goal_generation: int | None = None
    credited_range: float | None = None

    def rebaseline(self):
        self.approach_fix = None
        self.pose = self.source = self.route_origin = self.checkpoint = None
        self.route = ()
        self.interrupted = False
        self.along_m = 0.

    def advance(self, before, pose, route):
        if route:
            self.route_origin, self.route = before, route
        if self.route_origin is None:
            return False
        # A backwards slide during a hold cancels uncredited forward travel
        # on the same checked segment.
        x0, y0, _ = before.relative_to(self.route_origin)
        x1, y1, _ = pose.relative_to(self.route_origin)
        start = dwa._project_to_route((x0, -y0), self.route)[0]
        end = dwa._project_to_route((x1, -y1), self.route)[0]
        net = math.hypot(pose.x - self.checkpoint.x, pose.y - self.checkpoint.y)
        # Opposite replans can label both halves of a round trip "forward".
        # Returning to the checkpoint erases this credit but retains reverse debt.
        self.along_m = min(net, self.along_m + end - start)
        if self.along_m + 1e-9 < local_path.RESOLUTION:
            return False
        self.along_m, self.checkpoint = 0., pose
        return True


@dataclass(frozen=True)
class DwaGuidance:
    frame: object
    anchor: DetourGoal | None
    waypoints: tuple
    approach_origin: tuple


@dataclass
class State:
    """What the policy remembers between ticks. Owned by the card, passed in."""

    target: str = ""
    yaw_gate: Gate = field(default_factory=Gate)
    fwd_gate: Gate = field(default_factory=Gate)
    # The lateral axis needs one for the same reason the other two do — see
    # `_approach`. It did not have one, and that is why every run began with a
    # sidestep nobody asked for.
    vy_gate: Gate = field(default_factory=Gate)
    escape_side: float = 0.0
    side_stall_s: dict[float, float] = field(default_factory=dict)
    side_compare_frame: int | None = None
    side_better_frames: int = 0
    local_plan: PlanCache = field(default_factory=PlanCache)
    view_detour: Detour | None = None
    camera_arc_selected: bool = False
    checked_prefix: bool = False
    dwa_selected: bool = False
    dwa_proposal: DwaProposal | DwaTurn | None = None
    published_dwa_route: DwaAcceptance | None = None
    published_camera_passage: CameraPassage | None = None
    dwa_handoff_yaw_step: float = 0.
    dwa_motion_guard: DwaMotionGuard | None = None
    dwa_association: tuple[AssociationSnapshot, PlanarPose] | None = None
    dwa_admission_ms: int = 0
    dwa_odom: dict | None = None
    dwa_projection: local_path.DepthProjection | None = None
    dwa_guidance: DwaGuidance | None = None
    dwa_trace: dict = field(default_factory=dict)
    dwa_totals: dict = field(default_factory=dict)
    dwa_budget: dwa.CandidateBudget | None = None
    dwa_predictions: dwa.PredictionCache = field(default_factory=dwa.PredictionCache)
    decision_id: int = 0
    search_sweep_checked: bool = False
    detour_leg: PathLeg | None = None
    executing_leg: PathLeg | None = None
    goal_anchor: DetourGoal | None = None
    active_scan: ActiveScan = field(default_factory=ActiveScan)
    use_body_mount: bool | None = None
    recovery: bool = False
    # What the corridor read on the last tick. Kept so the debug overlay can
    # draw the number the decision was made from, rather than measuring the
    # depth map a second time and risking a picture that disagrees with the
    # behaviour it is supposed to explain.
    last_clearance: float = None
    last_coverage: float = 1.0
    tracker: Tracker = field(default_factory=Tracker)
    # What we last asked the chassis to do. Stands in for odometry when none is
    # wired: with a deadband the robot either does roughly the commanded speed
    # or nothing at all, so the command is a serviceable proxy — much better
    # than assuming we are stationary while turning at 1 rad/s.
    commanded: tuple = (0.0, 0.0, 0.0)
    command_dt_s: float = 0.
    motion_source: str = ""
    motion_pose_frame: tuple | None = None
    feedback_boot_s: float = 0.
    feedback_guard_s: float = 0.
    feedback_switch_ms: int | None = None
    uses_received_speed: bool = False
    command_tail: tuple = (0., 0., 0.)
    command_tail_s: float = 0.
    reacquire_generation: int | None = None
    reacquire_receipt_ms: int = 0
    reacquire_source: float | None = None
    reacquire_candidate: dict | None = None
    missing_frames: int = 0
    last_seen_side: float = 1.0          # +1 target was right, -1 it was left
    searching_for_s: float = 0.0
    searched_rad: float = 0.0
    commanded_vx: float = 0.0
    adaptive_fwd_scale: float = 1.0
    avoidance_yaw_override: bool = False
    moving_for_s: float = 0.0
    idle_for_s: float = 0.0
    dwa_progress: DwaProgress = field(default_factory=DwaProgress)
    arrived: bool = False
    blocked_for_s: float = 0.0
    # Set once, read forever after: failure is terminal for one navigate_to.
    failed_reason: str = ""
    partial_result: dict | None = None
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class DwaMotionInputs:
    """Registered scene and absolute timing shared by translation and observation."""

    checked_ms: int
    deadline_ms: int
    prediction_horizon_s: float
    footprint: motion_model.FootprintSpec
    goal: tuple | None
    goal_error: float
    measured: tuple
    previous: tuple
    pose: PlanarPose | None
    observed: tuple
    remembered: tuple
    memory_revision: tuple
    target_revision: tuple
    generation: int | None
    association: AssociationSnapshot | None
    association_pose: PlanarPose | None
    measured_target: tuple | None
    front_required: bool
    target_name: str


@dataclass(frozen=True)
class DwaMotionGuard(DwaMotionInputs):
    trace: motion_model.MotionTrace
    check: motion_model.MotionCheck
    visibility_required: bool = False
    probe_region: tuple | None = None
    corner_only: bool = False
    route_error: float = 0.


@dataclass(frozen=True)
class DwaProposal:
    choice: dwa.Choice
    reference: Detour
    branch: str
    guard: DwaMotionGuard

    @property
    def command(self):
        return self.choice.command


@dataclass(frozen=True)
class DwaTurn:
    """An exact angular request with its own observation or parking objective."""
    command: tuple
    guard: DwaMotionGuard


@dataclass(frozen=True)
class DwaAcceptance:
    """Identity and exact values of a locally checked command, before publishing."""

    decision_id: int
    command: tuple[float, float, float]
    guard_horizon_s: float = 0.
    motion_guard: DwaMotionGuard | None = None
    angle_controlled: bool = False
    route_points: tuple = ()
    followed_route: bool = False
    passage_exit: tuple | None = None
    corridor_corner: bool = False

    @property
    def target_observed(self):
        return self.motion_guard is not None and self.motion_guard.association is not None


@dataclass
class Decision:
    """One tick's answer. `values` is None when nothing should be published."""

    values: list | None
    status: str
    reason: str = ""
    distance_m: float | None = None
    bearing: float | None = None
    k_fwd_effective: float | None = None
    k_yaw_effective: float | None = None
    dwa_acceptance: DwaAcceptance | None = None

    @property
    def publishes(self) -> bool:
        return self.values is not None


def apply_deadband(values, min_magnitude) -> list:
    """Lift each axis out of the robot's deadband, or drop it to zero.

    Some robots do nothing at all below a threshold — a legged base has to
    assemble a whole gait cycle, so there may be no "creep slowly" regime.
    Below its threshold the SDK may accept the command while the robot stands
    still, so a policy emitting a smooth ramp towards zero can spend its entire
    life commanding motion and producing none.

    The threshold is the robot's, declared in its descriptor, so this function
    takes it as data rather than knowing any robot's numbers.

    Anything at or above half the threshold is snapped **up**: the policy asked
    for motion and this is the slowest motion available, so rounding down would
    silently discard the request. Below half it is snapped to zero, which is at
    least honest — and the caller can see it did, because the value is 0.

    The cost is coarse control near zero. That is inherent to the robot, not
    something a smoother policy can fix.
    """
    if not min_magnitude:
        return list(values)
    out = []
    for value, floor in zip(values, list(min_magnitude) + [0.0] * len(values)):
        if not floor or value == 0.0:
            out.append(value)
        elif abs(value) >= floor:
            out.append(value)
        elif abs(value) >= floor / 2.0:
            out.append(floor if value > 0 else -floor)
        else:
            out.append(0.0)
    return out


def _twist(vx: float, vy: float, wz: float) -> list:
    """A body twist in `motus.control/1` order: [vx, vy, vz, wx, wy, wz].

    All three are required. They used to be two, and a default on the middle one
    would let the old two-argument calls keep compiling with the yaw rate landing
    on the lateral axis — a robot that strafes when told to turn, from a diff
    that looks harmless.

    `vy` is left of forward, matching every other body frame in this project.
    It is used, not pinned: a base that cannot strafe says so by pinning the
    axis in its descriptor, and `adopt_limits` reads that and sets `vy_max` to
    zero. The policy does not need to know which kind of robot it is on.
    """
    return [vx, vy, 0.0, 0.0, 0.0, wz]


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _lift(value: float, floor: float, ceiling: float) -> float:
    """Quantise one axis onto what the robot can actually execute.

    Clamp to the ceiling, then round the magnitude **up** to the floor. Unlike
    `apply_deadband` there is no "too small, drop it to zero" case, because by
    the time a value reaches here a `Gate` has already decided this axis should
    be moving. Splitting the decision from the magnitude is what keeps the two
    from disagreeing — the old arrangement had the policy ask for 0.2 m/s and a
    filter downstream silently answer 0.
    """
    value = _clamp(value, ceiling)
    if not floor or value == 0.0:
        return value
    if abs(value) >= floor:
        return value
    return floor if value > 0 else -floor


# Ceilings are raised to this multiple of the floor when they sit below it.
# 1.5 rather than 1.0 because equality leaves a single commandable speed, which
# is a switch, not a controller — the point is to have somewhere to be
# proportional in.
_CEILING_HEADROOM = 1.5


def _adopt_footprint(config: Config, descriptor: dict) -> list[str]:
    """Take the robot's own width from its descriptor, or stay wide.

    Same move as `min_magnitude`: the chassis knows its dimensions and the
    policy must not hard-code them. A missing declaration is not a reason to
    guess narrow — the corridor simply stays at the conservative default and
    says so, because every failure mode of this number is one-sided. Believing
    the robot is wider than it is costs some unnecessary slowing; believing it
    is narrower puts a shoulder into a doorframe.
    """
    footprint = (descriptor or {}).get("footprint") or {}
    declared = footprint.get("half_width")
    if not isinstance(declared, (int, float)) or declared <= 0:
        if descriptor:
            return [f"下游底盘没有声明 footprint —— 避障走廊按保守的 "
                    f"±{config.half_width_m:g} m 算，可能比实际车宽保守很多"]
        return []
    config.half_width_m = float(declared)
    if footprint.get("source") == "estimate":
        return [f"底盘的 footprint 标注为 estimate（±{declared:g} m），"
                "不是量出来的，避障余量按此理解"]
    return []


def _adopt_camera(config: Config, declaration: dict) -> list[str]:
    """Take the horizontal field of view from the camera that produced the depth.

    Same move as `_adopt_footprint` and `min_magnitude`: the thing that knows
    declares it, the policy adopts it at start, and anything it had to change or
    assume goes into `info().degraded`. Camera parameters were the last quantity
    here that a human had to copy by hand, and copying it wrong is what made the
    robot refuse doorways.

    **Unlike the footprint, this number's failure modes are not symmetric, and
    neither is safe.** Understate the field of view and the corridor comes out
    too wide, so the robot refuses gaps it fits through — stuck, but nothing is
    hit. Overstate it and the corridor comes out too narrow, so an obstacle
    beside the path is filed as being outside it and a shoulder goes into a
    doorframe. Stuck is recoverable and a collision is not, so a missing
    declaration keeps the conservative fallback rather than guessing wide — and
    says so, because on r1_sz "the robot keeps turning away at the door" was
    diagnosed as a tracking problem first, twice.
    """
    notes: list[str] = []
    if not isinstance(declaration, dict) or not declaration:
        return [f"没拿到相机的 camera_info —— 避障走廊按 half_fov_rad="
                f"{config.half_fov_rad:g} 的保守值算。如果机器人在门口反复转向或"
                f"绕开明明走得过的缝，先检查相机卡片有没有声明 camera_info"]

    angle, source = _resolve_half_fov(declaration)
    if angle is None:
        return [f"相机的 camera_info 里读不到可用的视场角（{source}）—— 走廊按保守的"
                f" half_fov_rad={config.half_fov_rad:g} 算。用 tools/measure_fov.py"
                f" 量一次，填到**那颗镜头的驱动声明**里，不要填到这张卡片上"]

    previous = config.half_fov_rad
    config.half_fov_rad = angle
    # Report the adoption whenever it moved the number enough to change a
    # decision. Silence would repeat the original mistake from the other side:
    # an operator who set this by hand is owed the news that the camera
    # overruled them, rather than a card that quietly ignores the setting.
    if abs(angle - previous) > 0.02:
        notes.append(
            f"相机声明的水平半视场角是 {angle:.3f} rad（{source}），已采纳 —— "
            f"卡片配置里的 {previous:.3f} 不再生效。"
            f"两者相差 {abs(angle - previous) / max(previous, 1e-6) * 100:.0f}%")
    if source in ("unknown", "unspecified"):
        notes.append("相机没说这个视场角是量出来的还是猜的（source 缺失）")
    return notes


def adopt_limits(config: Config, descriptor: dict) -> list[str]:
    """Fit the policy's ceilings to the robot that will execute them.

    Returns human-readable notes about anything it had to change, for the card
    to surface in `info().degraded`. Silence would be wrong here: an operator
    who set `vx_max: 0.2` is owed the news that this robot cannot go that slowly,
    rather than a robot that ignores the setting.

    Two directions are handled:

    * **Up, off the deadband.** A ceiling below the Driver-declared floor makes
      the policy's entire output range unexecutable and every nonzero command
      snap to that floor.
    * **Down, into the descriptor's limits.** A ceiling above `limits.upper` is
      a command `ControlSink` rejects, at the full command rate, for the whole
      run.
    """
    notes: list[str] = []
    notes.extend(_adopt_footprint(config, descriptor))
    rate = (descriptor or {}).get("rate") or {}
    watchdog = rate.get("watchdog_ms")
    if (isinstance(watchdog, (int, float)) and not isinstance(watchdog, bool)
            and math.isfinite(watchdog) and watchdog > 0):
        config.watchdog_ms = watchdog
    max_age = rate.get("max_obs_age_ms")
    if (isinstance(max_age, (int, float)) and not isinstance(max_age, bool)
            and math.isfinite(max_age) and max_age > 0
            and max_age < config.max_obs_age_ms):
        config.max_obs_age_ms = max_age
        notes.append(f"观测有效期已按下游准入上限收紧到 {max_age:g}ms")
    limits = (descriptor or {}).get("limits") or {}
    floors = list(limits.get("min_magnitude") or [])
    moving_floors = list(limits.get("min_magnitude_moving") or [])
    lower = list(limits.get("lower") or [])
    upper = list(limits.get("upper") or [])

    # Hardware can disable lateral motion; it cannot override an explicit
    # operator choice to avoid strafing.
    try:
        config.use_lateral = config.use_lateral and not (
            float(lower[1]) == 0.0 and float(upper[1]) == 0.0)
    except (IndexError, TypeError, ValueError):
        pass

    def at(row, index):
        try:
            value = abs(float(row[index]))
            return value if math.isfinite(value) else None
        except (IndexError, TypeError, ValueError):
            return None

    def signed_at(row, index):
        try:
            value = float(row[index])
            return value if math.isfinite(value) else None
        except (IndexError, TypeError, ValueError):
            return None

    # The command sink limits velocity changes per accepted packet. Model
    # its nominal ramp at the producer cadence declared by the sink.
    deltas = limits.get("max_delta_per_step") or ()
    hz = rate.get("expected_hz")
    nominal_ramp = [None] * 3
    if isinstance(deltas, (list, tuple)) and type(hz) in (int, float) and 0 < hz < math.inf:
        for axis, index in enumerate((0, 1, 5)):
            if index >= len(deltas) or signed_at(lower, index) == signed_at(upper, index) == 0.:
                continue
            step = deltas[index]
            if type(step) in (int, float) and 0 < step < math.inf and math.isfinite(step * hz):
                nominal_ramp[axis] = step * hz
    config.motion_acceleration = tuple(rate or fallback for rate, fallback in
                                       zip(nominal_ramp, motion_model._DEFAULT_ACCELERATION))
    config.motion_braking = tuple(rate or fallback for rate, fallback in
                                  zip(nominal_ramp, motion_model._DEFAULT_BRAKING))

    bounds = [(signed_at(lower, i), signed_at(upper, i)) for i in (0, 1, 5)]
    config.gap_speed_bounds = (tuple(max(-lo, hi) for lo, hi in bounds)
                               if all(lo is not None and hi is not None and lo <= 0 <= hi for lo, hi in bounds)
                               else None)
    config.floor_vx_moving = at(moving_floors, 0)
    config.floor_vy_moving = at(moving_floors, 1)
    left_vy, right_vy = signed_at(upper, 1), signed_at(lower, 1)
    config.left_vy_limit = max(0., left_vy) if left_vy is not None else None
    config.right_vy_limit = max(0., -right_vy) if right_vy is not None else None
    wz_lower, wz_upper = signed_at(lower, 5), signed_at(upper, 5)
    config.left_wz_limit = max(0., wz_upper) if wz_upper is not None else None
    config.right_wz_limit = max(0., -wz_lower) if wz_lower is not None else None
    config.scan_wz_limit = (max(0., min(-wz_lower, wz_upper))
                            if wz_lower is not None and wz_upper is not None else None)

    # The three twist axes this policy drives, by their index in the six-wide
    # vector. vz/wx/wy are not here because nothing here produces them.
    for index, name, ceiling_attr, floor_attr in (
            (0, "vx", "vx_max", "floor_vx"),
            (1, "vy", "vy_max", "floor_vy"),
            (5, "wz", "wz_max", "floor_wz")):
        floor = at(floors, index) or 0.0
        setattr(config, floor_attr, floor)
        if index == 5:
            # The moving floor is optional: a chassis that does not declare one
            # keeps the standing floor everywhere, which is the old behaviour
            # and errs towards commanding too much rather than too little.
            moving = at(moving_floors, 5)
            config.floor_wz_moving = floor if moving is None else moving
            if moving is not None and moving < floor:
                notes.append(
                    f"行进中偏航下限 {moving:g} 远低于站立时的 {floor:g}，"
                    f"走动时按前者做比例控制")

        # An axis the robot does not have. Pinned lower == upper == 0 is how a
        # descriptor says so, and the policy has to stop asking for it — not
        # because the sink would reject it (it would, loudly) but because the
        # decomposition should put that speed on an axis that exists.
        bound = at(upper, index)
        if index == 1 and config.left_vy_limit is not None and config.right_vy_limit is not None:
            bound = max(config.left_vy_limit, config.right_vy_limit)
        if index == 5 and config.left_wz_limit is not None and config.right_wz_limit is not None:
            bound = max(config.left_wz_limit, config.right_wz_limit)
        if bound is not None and at(lower, index) == 0.0 and bound == 0.0:
            if getattr(config, ceiling_attr) > 0:
                notes.append(f"下游底盘没有 {name} 这个自由度，已停用该轴")
            setattr(config, ceiling_attr, 0.0)
            if index == 5 and config.search_rate != 0.0:
                config.search_rate = 0.0
                notes.append("下游底盘没有 wz 这个自由度，已停用搜索转身")
            continue

        ceiling = float(getattr(config, ceiling_attr))
        if floor and ceiling < floor * _CEILING_HEADROOM:
            raised = floor * _CEILING_HEADROOM
            if bound is not None:
                raised = min(raised, bound)
            if raised > ceiling:
                setattr(config, ceiling_attr, raised)
                notes.append(
                    f"{name}_max {ceiling:g} 低于机器人的最小可执行量 {floor:g}，"
                    f"已提到 {raised:g} —— 否则该轴只有「0 或 {floor:g}」两个值，"
                    f"表现为顿挫")
        if bound is not None and float(getattr(config, ceiling_attr)) > bound:
            notes.append(f"{name}_max 超过下游允许的 {bound:g}，已压回")
            setattr(config, ceiling_attr, bound)

        # Search sweeps in both directions at a fixed rate. Its usable
        # magnitude must fit both signed bounds and the effective yaw ceiling.
        if index == 5:
            search_cap = min(value for value in (config.wz_max, config.left_wz_limit,
                                                 config.right_wz_limit) if value is not None)
            # Negotiate the actual executable rate here, so sweep accounting
            # agrees with the command after deadband lifting. Zero stays off.
            candidate = config.search_rate
            if candidate > 0:
                candidate = min(max(candidate, floor), search_cap)
            if floor and search_cap + 1e-9 < floor:
                if config.search_rate != 0.0:
                    notes.append(
                        f"search_rate 的双向上限 {search_cap:g} 低于偏航最小"
                        f"可执行量 {floor:g}，已停用搜索转身")
                config.search_rate = 0.0
            elif candidate != config.search_rate:
                notes.append(f"search_rate 已按下游死区和双向上限调整为 {candidate:g}")
                config.search_rate = candidate

    return notes


def adopt_response(config: Config, descriptor: dict) -> list[str]:
    """Adopt the Driver-owned yaw response declaration, when present.

    ``response.wz`` describes the plant behind the velocity interface.  Navi
    uses it to choose an effective proportional gain, but does not calibrate,
    persist, or identify that plant itself. Missing data keeps the generic
    configured fixed P controller.
    """
    response = (descriptor or {}).get("response") or {}
    yaw = response.get("wz") if isinstance(response, dict) else None
    config.response_yaw_gain = None
    if yaw is None:
        return []
    if not isinstance(yaw, dict):
        return [f"下游 response.wz 声明无效，已使用配置 k_yaw={config.k_yaw:g}"]
    try:
        source = yaw["source"]
        numeric = (yaw["gain"], yaw["delay_s"], yaw["settling_s"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return [f"下游 response.wz 声明无效，已使用配置 k_yaw={config.k_yaw:g}"]
    if (not isinstance(source, str)
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float)) for value in numeric)):
        return [f"下游 response.wz 声明无效，已使用配置 k_yaw={config.k_yaw:g}"]
    source = source.strip().lower()
    gain, delay_s, settling_s = map(float, numeric)
    if (source not in {"measured", "vendor-spec", "estimate"}
            or not all(math.isfinite(value) for value in (gain, delay_s, settling_s))
            or gain <= 0 or delay_s < 0 or settling_s < 0):
        return [f"下游 response.wz 声明无效，已使用配置 k_yaw={config.k_yaw:g}"]

    effective_gain = max(0.5, min(2.5, gain))
    penalty = 1.0 + 0.2 * delay_s + 0.1 * settling_s
    config.response_yaw_gain = max(
        0.2, min(1.2, (config.k_yaw / effective_gain) / penalty))
    return [f"已采用下游 response.wz：转向P={config.response_yaw_gain:.3f}"
            f"（source={source}, gain={gain:g}, delay={delay_s:g}s, "
            f"settling={settling_s:g}s）"]


def select_target(objects, name: str, config: Config):
    """The detection to chase, or None.

    Exact name first, then substring, so `navigate_to("chair")` finds
    `"office chair"` without `navigate_to("air")` finding it too. Among equals,
    the most confident; among equally confident, the most central — which keeps
    the choice stable frame to frame when two of the same thing are in view.
    """
    wanted = (name or "").strip().lower()
    if not wanted:
        return None

    candidates = [o for o in (objects or [])
                  if float(o.get("confidence") or 0) >= config.min_confidence]
    # A key from `list_visible_objects` (`chair#azure`) beats a bare name. It is
    # the only thing in that list that separates two chairs of different
    # colours, so letting the caller pick one and then matching on the name
    # alone throws away the distinction that was just made for them.
    keyed = [o for o in candidates if object_key(o).lower() == wanted]
    if keyed:
        pool = keyed
    else:
        exact = [o for o in candidates if str(o.get("name", "")).lower() == wanted]
        pool = exact or [o for o in candidates
                         if wanted in str(o.get("name", "")).lower()]
    if not pool:
        return None
    return max(pool, key=lambda o: (round(float(o.get("confidence") or 0), 2),
                                    -abs(_bearing(o))))


def _bearing(obj) -> float:
    position = obj.get("position") or [0.0, 0.0]
    return float(position[0]) if position else 0.0


def target_distance(obj, depth, config: Config) -> float | None:
    """How far the chosen object is, by whatever the depth source allows.

    `bbox` is much better than a centre point — it samples the whole object
    rather than one patch that may land on a gap — so it is used when vop
    publishes it. Both paths return None rather than a guess.
    """
    if depth is None:
        return None
    from . import depth as depth_mod

    if depth.get("map") is not None:
        bbox = obj.get("bbox_norm")
        if bbox:
            return depth_mod.sample_box(depth["map"], bbox)
        return depth_mod.sample_point(depth["map"], obj.get("position") or [0, 0])
    # Summary-only gives the nearest obstacle in the target's band. It can
    # constrain motion but cannot confirm the distance to the target.
    bands = depth.get("bands") or {}
    return bands.get(_band_of(_bearing(obj)))


def _band_of(bearing: float) -> str:
    if bearing < -1 / 3:
        return "left"
    if bearing > 1 / 3:
        return "right"
    return "center"


def camera_odometry_inputs(config, state, depth, odom):
    if (config.local_controller == "dwa" and config.scan_geometry is None
            and (state.use_body_mount is False or not (depth or {}).get("body_mount_required"))):
        return state.active_scan.relative_inputs_after_pose_gap(depth, odom)
    return depth, odom


def step(*, detections, depth, odom, config: Config, state: State,
         dt: float, control_period_s: float | None = None, command_dt_s: float | None = None,
         command_interrupted: bool = False, dwa_admission_ms: int = 0, objects_ms: int = 0,
         dwa_time_left=None) -> Decision:
    depth, odom = camera_odometry_inputs(config, state, depth, odom)
    state.dwa_admission_ms = dwa_admission_ms
    state.dwa_odom = odom if config.local_controller == "dwa" else None
    if config.local_controller == "dwa":
        _latch_progress_feedback(state, config, odom)
    state.dwa_projection = (local_path.DepthProjection.for_observation(depth)
                            if config.local_controller == "dwa" and (depth or {}).get("map") is not None else None)
    if state.dwa_projection is not None:
        depth = dict(depth, grid_projection=state.dwa_projection)
    state.command_dt_s = min(dt, config.watchdog_ms / 1000) if command_dt_s is None else command_dt_s
    progress_route = _previous_progress_route(state) if config.local_controller == "dwa" else ()
    state.view_detour = None
    state.camera_arc_selected = False
    state.checked_prefix = False
    state.dwa_selected = False
    state.dwa_proposal = None
    state.dwa_motion_guard = None
    state.decision_id += 1
    state.dwa_budget = (dwa.CandidateBudget(min(config.dwa_samples, dwa.MAX_SEARCH_CANDIDATES),
                                          predictions=state.dwa_predictions, time_left=dwa_time_left,
                                          early_return_s=2 * control_period_s if control_period_s else math.inf)
                        if config.local_controller == "dwa" else None)
    state.dwa_trace = dict(decision_id=state.decision_id, controller_requested=config.local_controller,
                           controller_used="legacy", eligible=False, branch=None, geometry_mode=None,
                           select_calls=0, sampled_candidates=0, cached_candidates=0,
                           considered_candidates=0, admissible_candidates=0, reference_scores=0,
                           angular_checks=0, angular_cache_hits=0, rejected={},
                           proposed_twist=None, accepted_twist=None, fallback_reason=None, veto_reason=None)
    state.search_sweep_checked = False
    state.executing_leg = None
    if config.local_planning and config.local_geometry is not None and (depth or {}).get("body_mount_required"):
        capture_pose = depth.get("obstacle_capture_pose")
        pose = (odom or {}).get("obstacle_pose")
        paired = (depth.get("body_from_optical") is not None and capture_pose is not None
                  and pose is not None and capture_pose.frame == pose.frame)
        if state.use_body_mount is not False:
            if state.use_body_mount and not paired:
                scan = state.active_scan
                if scan.active or any(state.commanded):
                    state.feedback_switch_ms = None
                    state.feedback_guard_s = max(state.feedback_guard_s, (config.watchdog_ms + 50) / 1000)
                state.active_scan = ActiveScan(obstacles=scan.obstacles, near_obstacles=scan.near_obstacles,
                                               near_projection=scan.near_projection,
                                               observed_world=scan.observed_world)
                state.escape_side = 0.
            # Keep one frame of reference for the rest of this task after a
            # fallback. Intermittent transforms must not interrupt each tick.
            state.use_body_mount = paired
        if state.use_body_mount:
            calibrated = depth.get("mount_accuracy_validated") is True
            half_length = config.local_geometry.half_length
            geometry = replace(config.local_geometry, base_from_optical=depth["body_from_optical"],
                               front=half_length, rear=half_length, free_space_validated=calibrated,
                               min_obstacle_height=(local_path.MIN_OBSTACLE_HEIGHT_M if calibrated
                                                    else config.local_geometry.min_obstacle_height))
            config = replace(config, scan_geometry=geometry, local_geometry=geometry)
        else:
            depth = dict(depth, body_mount_required=False)
    scan_before, phase_before = state.active_scan, state.active_scan.phase
    decision = _decide(detections, depth, odom, config, state, dt, control_period_s=control_period_s, objects_ms=objects_ms)
    scan = state.active_scan
    if (decision.status == AVOIDING and decision.values is not None and not any(decision.values)
            and scan.phase == "brake" and (scan is not scan_before or phase_before != "brake")
            and scan.poses and not scan.command_only
            and state.goal_anchor is not None
            and max(math.hypot(*state.commanded[:2]), math.hypot(*scan.motion[:2])) > STILL_LINEAR):
        geometry = config.scan_geometry or config.local_geometry
        command = _observation_command(scan, geometry, config, state, odom, dt, control_period_s, depth)
        if command.yaw:
            decision.values = _twist(command.vx, 0., command.yaw)
        decision.reason = command.reason
    if config.local_planning and state.motion_source == "command" and decision.values is not None:
        last_x, last_y, last_yaw = state.command_tail
        vx, vy, yaw = decision.values[0], decision.values[1], decision.values[5]
        transitioning = (
            (math.hypot(last_x, last_y) > STILL_LINEAR and yaw and not (vx or vy))
            or (abs(last_yaw) > STILL_YAW and not (last_x or last_y) and (vx or vy))
            or (abs(last_x) > STILL_LINEAR and last_x * vx < 0.)
            or (abs(last_y) > STILL_LINEAR and last_y * vy < 0.)
            or (abs(last_yaw) > STILL_YAW and last_yaw * yaw < 0. and not (vx or vy))
        )
        if transitioning:
            decision = Decision(_twist(0., 0., 0.), AVOIDING, "运动方向切换，停车等待制动窗口")
    reduction = 1. if state.dwa_selected else _feedback_speed_reduction(config, odom)
    if reduction < 1. and decision.values is not None and any(decision.values):
        vx, vy, yaw = (decision.values[i] * reduction for i in (0, 1, 5))
        (floor_x, _), (floor_y, _) = _local_velocity_limits(config, 1. if vy < 0. else -1.)
        floor_yaw = config.floor_wz_moving if vx or vy else config.floor_wz
        if any(0. < abs(value) < floor for value, floor in ((vx, floor_x), (vy, floor_y), (yaw, floor_yaw))):
            vx = vy = yaw = 0.
        decision.values = _twist(vx, vy, yaw)
        decision.reason = f"{decision.reason}；实测速度超限，减速"
        state.commanded_vx = vx
    anchor, scan = state.goal_anchor, state.active_scan
    geometry = config.scan_geometry or config.local_geometry
    if (anchor is not None and decision.values is not None and any(decision.values)
            and not state.dwa_selected and not state.camera_arc_selected
            and not scan.active and not state.checked_prefix and state.executing_leg is None
            and geometry is not None and scan.current_frame and scan.views):
        twist = tuple(decision.values[i] for i in (0, 1, 5))
        motion = planar_feedback(odom) or state.command_tail
        horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
        grid = scan.grid(include_near=scan.near_squares is None)
        if not local_path.motion_sweep_clear(grid, geometry, config.half_width_m, horizon,
                                             (motion, state.commanded, twist)):
            decision = Decision(_twist(0., 0., 0.), AVOIDING, "前往记忆目标的扫掠区有障碍",
                                distance_m=decision.distance_m, bearing=decision.bearing)
    if (not state.dwa_selected and decision.values is not None and any(decision.values)
            and not scan.active and scan.near_hits is not None):
        twist = tuple(decision.values[i] for i in (0, 1, 5))
        searched = decision.status == SEARCHING and state.search_sweep_checked and not any(twist[:2])
        if not searched and _near_sweep_hits(config, scan, odom, state, twist).any():
            decision = Decision(_twist(0., 0., 0.), AVOIDING, "当前运动的扫掠区有盲区内已观测障碍")
    if (decision.reason == NO_LOCAL_ROUTE and scan.exhausted and scan.current_frame
            and anchor is not None and anchor.verified_range_m is not None
            and anchor.verified_range_m > config.arrival_distance_m
            and anchor.approach_travel_m >= local_path.RESOLUTION
            and anchor.approach_start_range_m is not None
            and anchor.approach_start_range_m - anchor.verified_range_m >= local_path.RESOLUTION
            and not any(state.commanded[:2])):
        feedback = planar_feedback(odom)
        if feedback is not None and math.hypot(*feedback[:2]) <= STILL_LINEAR and abs(feedback[2]) <= STILL_YAW:
            state.failed_reason = "当前已观测局部空间内没有进一步可执行的接近路线"
            state.partial_result = dict(outcome="partial", target_reached=False,
                                        reachability_scope="observed_local",
                                        remaining_distance_m=anchor.verified_range_m,
                                        distance_basis="measured_range_to_original_target",
                                        best_verified_distance_m=anchor.best_verified_distance_m)
            decision = Decision(_twist(0., 0., 0.), FAILED, state.failed_reason,
                                distance_m=anchor.verified_range_m, bearing=decision.bearing)
    if config.local_controller == "dwa":
        _latch_progress_feedback(state, config, odom)
    if config.local_controller == "dwa" and state.dwa_progress.requires_feedback:
        decision = _account_dwa_progress(decision, state, config, odom, depth, progress_route, dt)
    else:
        if (decision.values is None or decision.values[0] <= .05 or command_interrupted
                or not state.command_dt_s or reduction < 1.):
            state.moving_for_s = 0.
        else:
            stuck = _stuck(odom, state, state.commanded[0], state.command_dt_s, config)
            if stuck:
                state.failed_reason = stuck
                decision = Decision(None, FAILED, stuck, distance_m=decision.distance_m, bearing=decision.bearing)
        if reduction == 1. and decision.reason != TURN_SPEED_REDUCTION:
            decision = _account_idle(decision, state, config, dt)
        else:
            state.idle_for_s = 0.
    if (decision.values is None or decision.values[0] <= 0
            or decision.status in (ARRIVED, FAILED, IDLE, BLIND, SEARCHING, ALIGNING)):
        # Do not carry an increased approach gain through a hold, search or
        # blocked corridor and release it as a jump on the next approach.
        state.adaptive_fwd_scale = 1.0
    if state.dwa_selected and state.dwa_proposal is not None and decision.values is not None:
        final = tuple(decision.values[i] for i in (0, 1, 5))
        if any(decision.values) and (final != state.dwa_proposal.command or any(decision.values[2:5])):
            decision = Decision(_twist(0., 0., 0.), AVOIDING,
                                "DWA：后置处理改变已检查的速度，停车后重新采样",
                                distance_m=decision.distance_m, bearing=decision.bearing)
            state.dwa_trace["veto_reason"] = "postcheck_veto"
    # Remember what we asked for, so the tracker can compensate for our own
    # motion on the next tick even with no odometry wired.
    if state.executing_leg is None or not (decision.values and any(decision.values[:2]) and not decision.values[5]):
        state.detour_leg = state.executing_leg = None
    if decision.values is not None:
        state.commanded = (decision.values[0], decision.values[1],
                           decision.values[5])
    else:
        state.commanded = (0.0, 0.0, 0.0)
    if config.local_controller == "dwa":
        trace = state.dwa_trace
        final = tuple(decision.values[i] for i in (0, 1, 5)) if decision.values is not None else None
        proposal = state.dwa_proposal
        if state.dwa_selected and proposal is not None and final == proposal.command and any(final):
            decision.dwa_acceptance = DwaAcceptance(
                state.decision_id, proposal.command, trace["guard_horizon_s"],
                state.dwa_motion_guard, isinstance(proposal, DwaTurn),
                proposal.reference.points if isinstance(proposal, DwaProposal) and proposal.branch not in ("visual_probe", "direct") else (),
                followed_route=bool(trace.get("retained_route")),
                passage_exit=proposal.reference.passage_exit if isinstance(proposal, DwaProposal) else None,
                corridor_corner=(isinstance(proposal, DwaProposal) and proposal.reference.corridor_corner))
            trace.update(controller_used="dwa", accepted_twist=proposal.command, fallback_reason=None)
        else:
            if state.dwa_selected:
                state.checked_prefix = False
            state.dwa_selected = False
            if proposal is not None:
                trace["veto_reason"] = trace["veto_reason"] or "postcheck_veto"
            trace["controller_used"] = "legacy" if final is not None and any(final) else "stop"
        trace["final_twist"] = final
        if trace["fallback_reason"] is None and not state.dwa_selected:
            unavailable = _dwa_unavailable_reason(depth, config, state)
            if trace["veto_reason"]:
                trace["fallback_reason"] = "postcheck_veto"
            elif not trace["select_calls"]:
                direct = (decision.status in (APPROACHING, ALIGNING) and final and any(final)
                          and not scan.active and state.goal_anchor is None and state.view_detour is None)
                trace["fallback_reason"] = "not_needed" if direct else unavailable or "no_reference"
                if unavailable is None:
                    trace.update(eligible=True,
                                 geometry_mode="body" if config.scan_geometry is not None else "camera_approx")
            else:
                trace["fallback_reason"] = "no_admissible_candidates"
        counts = dict(requested_ticks=1, eligible_ticks=int(trace["eligible"]),
                      attempted_ticks=int(trace["select_calls"] > 0), select_calls=trace["select_calls"],
                      proposed_ticks=int(trace["proposed_twist"] is not None),
                      accepted_ticks=int(state.dwa_selected))
        for key, value in counts.items():
            state.dwa_totals[key] = state.dwa_totals.get(key, 0) + value
    return decision


def _forward_gain(config: Config, state: State, distance, live: bool,
                  within: bool, dt: float) -> float:
    """Bounded distance-based gain scheduling for a live forward approach.

    The nominal distance error and all later avoidance/speed limits are
    untouched. Closer targets use up to three times the configured P gain,
    approached at a bounded rate; no observation or motor bias is calibrated.
    """
    if (not config.adaptive_k_fwd or not live or distance is None or within
            or config.k_fwd == 0):
        state.adaptive_fwd_scale = 1.0
        return config.k_fwd
    window = max(0.1, config.approach_check_distance_m - config.stop_distance_m)
    error = max(0.0, distance - config.stop_distance_m)
    # This rational schedule keeps the settled, unsaturated P speed monotone
    # in distance error: approaching the target does not increase that speed.
    target = ADAPTIVE_FWD_MAX_SCALE / (
        1.0 + (ADAPTIVE_FWD_MAX_SCALE - 1.0) * min(1.0, error / window))
    previous = max(1.0, min(ADAPTIVE_FWD_MAX_SCALE, state.adaptive_fwd_scale))
    elapsed = min(ADAPTIVE_FWD_MAX_DT, max(0.0, dt)) if math.isfinite(dt) else 0.0
    change = ADAPTIVE_FWD_SCALE_RATE * elapsed
    state.adaptive_fwd_scale = max(previous - change, min(previous + change, target))
    return config.k_fwd * state.adaptive_fwd_scale


def _measured_yaw_rate(odom):
    """Actual angular feedback for ego-motion tracking.

    Freshness is enforced by the card before this observation reaches policy.
    JSON booleans, missing axes and non-finite readings are not measurements.
    """
    return planar_axes(odom)[2]


def _ego_twist(odom, state: State, *, command_fraction=1.):
    """Our own motion, for the tracker's predict step. Returns (twist, measured).

    Measured if odometry is wired, commanded otherwise. **Per axis**, because
    `motus.odom/1` reports an unmeasured axis as `None` rather than 0 — a robot
    that reports yaw rate but not lateral speed should have its yaw believed and
    only its `vy` guessed.

    `measured` is true only when **every** axis came from odometry. A partly
    guessed twist is a guessed twist as far as the filter's confidence goes, and
    erring towards "guessed" only makes it trust its own predictions less.
    """
    feedback = planar_axes(odom)
    return (tuple(value if value is not None else command * command_fraction
                  for value, command in zip(feedback, state.commanded)),
            all(value is not None for value in feedback))


MAX_PREDICTED_DETOUR_TRAVEL_M = 3.
NO_LOCAL_ROUTE = "局部规划未找到可执行的避让路线；停下"
INITIAL_MEMORY_OVERLAP = "近障记忆与车身起始轮廓重叠，停车等待重新观测或配准"


def current_target_observed(state):
    """Whether this decision used a current accepted target and depth range."""
    track, anchor = state.tracker.track, state.goal_anchor
    return bool(track is not None and track.observed and track.accepted_detection
                and track.measured_range_m is not None and track.navigation_range_known
                and (anchor is None or (anchor.generation == track.generation and not anchor.target_missing)))


def _fresh_detour_fix(anchor, track, config, depth):
    if (anchor is None or track is None or not track.observed or track.measured_range_m is None
            or (depth or {}).get("map") is None):
        return None
    last = getattr(track, "front_fix", None)
    if last is None:
        last = getattr(track, "last_obs", None)
    bearing = math.atan2(-last[1], last[0]) if last is not None else track.bearing_rad
    distance = track.measured_range_m
    x, y = distance * math.cos(bearing), -distance * math.sin(bearing)
    if config.local_controller == "dwa":
        same = anchor.generation is not None and anchor.generation == track.generation
    else:
        same = track is anchor.track or math.hypot(x - anchor.x, y - anchor.y) <= 3 * config.range_std_m
    if same:
        return distance, bearing, x, y
    return None


def _remember_detour_goal(state, config, track, odom):
    if not config.local_planning:
        return
    scan = state.active_scan
    if not (state.escape_side or state.view_detour or scan.active or scan.result_frame or scan.exhausted):
        if (config.local_controller != "dwa" and state.goal_anchor is not None
                and state.goal_anchor.verified_range_m is not None):
            state.goal_anchor = None
        return
    if state.goal_anchor is not None or track is None:
        return
    front = getattr(track, "front_fix", None)
    if front is not None:
        x, y = front
    elif track.observed and track.range_known:
        x = track.range_m * math.cos(track.bearing_rad)
        y = -track.range_m * math.sin(track.bearing_rad)
    elif track.range_known and track.last_obs is not None:
        # The last observation has been transported by ego motion only.
        x, y = track.last_obs
    else:
        return
    odom = odom or {}
    pose = None if odom.get("velocity_history") else odom.get("motion_pose")
    if config.local_controller == "dwa":
        now = state.active_scan.control_ms or odom.get("received_ms")
        pose = target_body_pose(now, odom, planar_feedback(odom) or state.commanded) if now is not None else None
        state.goal_anchor = DetourGoal(x, y, pose, generation=track.generation, reference_ms=now, frame=track.frame)
    else:
        state.goal_anchor = DetourGoal(x, y, pose, track=track)


def _reset_detour_route(state):
    scan = state.active_scan
    scan.finish()
    scan.note_clear_path(scan.current_frame)
    scan.advance_origin, scan.advance_m = None, None
    scan.turn_until_ms, scan.turn_stop_ms = 0, None
    scan.turn_sweep = None
    state.detour_leg = None
    state.escape_side, state.recovery = 0., False


def _parking_alignment(depth, config, state, odom, distance, bearing, measured, observed):
    """Finish at the measured radius with the chassis facing the target."""
    state.commanded_vx = 0.
    stop = _twist(0., 0., 0.)
    if depth.get("map") is None or (not observed and state.goal_anchor is None):
        return Decision(stop, AVOIDING, "已到预计停车范围，等待实际测距",
                        distance_m=distance, bearing=bearing)
    parking_range = measured if measured is not None else distance
    if parking_range < config.arrival_min_distance_m - 1e-6:
        return Decision(stop, AVOIDING, "已在目标停车圆环内侧，保持停车",
                        distance_m=distance, bearing=bearing)
    if abs(bearing) <= config.arrive_align_tol:
        if observed and measured is not None and measured <= config.arrival_distance_m + 1e-6:
            state.arrived = True
            return Decision(stop, ARRIVED,
                            f"target at {measured:.2f} m, stop distance reached and heading aligned "
                            f"({config.stop_distance_m:.2f} m ± {config.stop_tolerance_m:.2f} m tolerance)",
                            distance_m=measured, bearing=bearing)
        return Decision(stop, ALIGNING, "已朝向目标，等待停车距离复测",
                        distance_m=distance, bearing=bearing)

    gain = config.response_yaw_gain if config.response_yaw_gain is not None else config.k_yaw
    limit = _scan_yaw_rate(config, 1. if bearing > 0. else -1.)
    yaw = _lift(-gain * bearing, config.floor_wz, limit) if limit >= config.floor_wz else 0.
    reason = "已到停车范围，转向对准目标" if yaw else "当前转向能力无法对准目标"
    scan = state.active_scan
    geometry = config.scan_geometry or config.local_geometry
    if yaw and config.local_planning and geometry is not None:
        feedback = planar_feedback(odom) or state.command_tail
        horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
        if not scan.views or not scan.matches_route_view(depth):
            yaw, reason = 0., "停车对准目标前等待可用的扫掠观测"
        elif config.local_controller == "dwa" and not scan.command_only:
            if state.goal_anchor is not None:
                point = state.goal_anchor.x, -state.goal_anchor.y
            else:
                angle = bearing * config.half_fov_rad
                point = parking_range * math.cos(angle), parking_range * math.sin(angle)
            target = local_path.target_in_grid(geometry, *point)
            if not _check_dwa_turn(depth, config, state, geometry, target, "parking", (0., 0., yaw)):
                yaw, reason = 0., "停车对准目标的扫掠区有障碍"
        else:
            grid = scan.grid(include_near=scan.near_squares is None)
            if not local_path.motion_sweep_clear(
                    grid, geometry, config.half_width_m, horizon, (feedback, state.commanded, (0., 0., yaw))):
                yaw, reason = 0., "停车对准目标的扫掠区有障碍"
        state.avoidance_yaw_override = True
    return Decision(_twist(0., 0., yaw), ALIGNING, reason, distance_m=distance, bearing=bearing)


def _search_yaw_candidates(config, yaw, both, scales):
    for scale in scales:
        for requested in (yaw * scale, -yaw * scale) if both else (yaw * scale,):
            rate = _clamp(requested, _scan_yaw_rate(config, 1. if requested < 0. else -1.))
            if rate and abs(rate) >= config.floor_wz:
                yield rate


def _search_for_target(depth, config, state, dt, odom):
    reason = _search_turn_stop(depth, config, state)
    if reason is None:
        return _search(config, state, dt, odom)
    if "扫掠范围有障碍" in reason:
        yaw = -state.last_seen_side * config.search_rate
        for rate in _search_yaw_candidates(config, yaw, True, (.5, .25)):
            if _search_turn_stop(depth, config, state, requested_yaw=rate) is None:
                if state.missing_frames + 1 < config.lost_frames:
                    return _search(config, state, dt, odom)
                state.missing_frames += 1
                return _search(config, state, dt, odom, requested_yaw=rate)
    return Decision(_twist(0., 0., 0.), AVOIDING, reason)


def _observation_command(scan, geometry, config, state, odom, dt, control_period_s, depth=None):
    yaw_limits = (_scan_yaw_rate(config, 1.), _scan_yaw_rate(config, -1.))
    vx, vy, yaw = state.commanded
    previous_linear = vx if vx >= 0. and abs(vy) <= STILL_LINEAR else None
    anchor = state.goal_anchor
    horizon = (config.watchdog_ms + config.depth_age_limit_ms + 50) / 1000
    room = max(0., math.hypot(anchor.x, anchor.y) - config.stop_distance_m) if anchor is not None else 0.
    (floor, limit), _ = _local_velocity_limits(config, 1.)
    check_motion = None
    if config.local_controller == "dwa" and not scan.command_only:
        target = local_path.target_in_grid(geometry, anchor.x, -anchor.y) if anchor is not None else None
        check_motion = partial(_check_dwa_turn, depth, config, state, geometry, target, "observation")

    return scan.command(geometry, config.half_width_m, max(yaw_limits), config.watchdog_ms,
                        config.depth_age_limit_ms, min_yaw_rate=config.floor_wz,
                        now_ms=(odom or {}).get("control_ms"), tick_s=dt,
                        previous_yaw=yaw, control_period_s=control_period_s,
                        previous_command_s=state.command_dt_s, yaw_limits=yaw_limits,
                        previous_linear=previous_linear, forward_limit=min(limit, room / horizon),
                        min_forward_speed=floor, min_moving_yaw=config.floor_wz_moving,
                        check_motion=check_motion)


def _obstacle_reference(odom, depth):
    odom = odom or {}
    pose = odom.get("obstacle_pose")
    if pose is None and not (depth or {}).get("body_mount_required") and (
            not odom.get("velocity_history") or (depth or {}).get("obstacle_capture_pose") is not None):
        pose = odom.get("motion_pose")
    return pose


def _decide(detections, depth, odom, config: Config, state: State,
            dt: float, *, control_period_s: float | None, objects_ms=0) -> Decision:
    """One tick.

    `detections` — vop's latest payload, or None if stale/absent.
    `depth`      — `{"map": ndarray|None, "bands": dict}`, or None.
    `odom`       — `{"vx"|"vy"|"wz": float|None}` from motus.odom/1, or None if
                   unwired. **`None` and `0.0` are different** and the stuck
                   detector depends on it: a robot with no odometry must not
                   look like a robot that has stopped.
    `dt`         — seconds since the previous tick.
    """
    if state.arrived:
        return Decision(None, ARRIVED, "arrived; waiting for the next instruction")
    if state.failed_reason:
        # Terminal too. A failed task must keep reporting the same reason
        # rather than quietly re-entering the loop on the next frame.
        return Decision(None, FAILED, state.failed_reason)

    if not state.target:
        return Decision(None, IDLE, "no navigation target")

    stale_capture = bool((depth or {}).get("capture_invalid"))
    frame = (depth or {}).get("map")
    invalid_depth = frame is not None and not has_valid_returns(frame)
    if stale_capture or invalid_depth:
        depth = None
    # The tracker runs **before** the blind check, so a spell with no
    # observation is a run of misses rather than time that did not happen. It
    # is also the only way the coast clock advances while the card is blind —
    # otherwise a camera that stopped publishing would leave a track coasting
    # for ever, and the robot driving towards a prediction nobody is checking.
    ego, ego_measured = _ego_twist(odom, state, command_fraction=state.command_dt_s / dt if dt else 0.)
    recovering = state.reacquire_generation is not None
    detection_source = observation_source(detections) if recovering else None
    fresh_recovery = (recovering and detections is not None and objects_ms > state.reacquire_receipt_ms
                      and (state.reacquire_source is None or (detection_source is not None and detection_source > state.reacquire_source)))
    same_receipt = objects_ms == state.reacquire_receipt_ms and detection_source == state.reacquire_source
    if recovering and state.target == "person" and not (fresh_recovery or same_receipt):
        # Out-of-order receipts cannot vote for confirming a person.
        detections = None
    track = state.tracker.step(detections=detections, depth=depth,
                               target=state.target, config=config,
                               ego=ego, ego_measured=ego_measured, dt=dt,
                               receipt_ms=objects_ms if recovering and state.target == "person" else 0)
    if (config.local_controller == "dwa" and isinstance(track, Track) and track.observed
            and track.accepted_detection and track.range_known and track.source is not None):
        stamp = (depth or {}).get("control_ms")
        pose = target_body_pose(stamp, odom, ego, depth.get("obstacle_capture_pose")) if stamp is not None else None
        if pose is not None:
            state.dwa_association = track.snapshot(stamp), pose
    anchor = state.goal_anchor
    if (config.local_controller == "dwa" and anchor is not None and anchor.generation is not None
            and isinstance(track, Track) and track.observed
            and track.accepted_detection and track.measured_range_m is not None
            and track.navigation_range_known and track.generation != anchor.generation):
        # A newly confirmed range starts a route for this generation. Keeping
        # the detached goal makes the publisher treat it as a coasting route.
        _reset_detour_route(state)
        state.goal_anchor = anchor = None
    visual_only = bool(track is not None and track.observed and getattr(track, "front_required", False)
                       and track.front_fix is None)
    matched_person = (state.target == "person" and detections is not None
                      and state.tracker.track is not None and state.tracker.matched_detections is detections)
    if fresh_recovery:
        state.reacquire_receipt_ms, state.reacquire_source = objects_ms, detection_source
        state.reacquire_candidate = detections if matched_person else None
    # One gated detector receipt can span several control ticks. Keep the
    # camera still for confirmation, including when its depth arrives later.
    pending_person = (recovering and detections is not None and state.reacquire_candidate is detections
                      and objects_ms == state.reacquire_receipt_ms and state.tracker.track is not None)
    if track is not None and track.observed and (
            fresh_recovery and track.accepted_detection or pending_person and track.measured_range_m is not None):
        if matched_person and track.image_region and track.image_region[0]:
            state.last_seen_side = 1. if track.image_region[0][0] >= 0. else -1.
        # A confirmed replacement owns a new goal; it cannot inherit the old detour.
        if anchor is not None and (track.generation != anchor.generation or visual_only):
            _reset_detour_route(state)
            state.goal_anchor = anchor = None
        if track.measured_range_m is not None or visual_only:
            state.reacquire_generation = None
            state.reacquire_candidate = None
            state.missing_frames = 0
            state.searching_for_s = state.searched_rad = 0.
    needs_probe = visual_only and (anchor is None or config.local_controller == "dwa")
    if needs_probe:
        if anchor is not None and depth is not None:
            _reset_detour_route(state)
            state.goal_anchor = anchor = None
        scan = state.active_scan
        scan.interrupt()
        if not (scan.active or scan.exhausted or scan.has_result):
            scan.attempted = False
        state.commanded_vx = 0.
        if config.local_controller != "dwa" or not config.local_planning or config.local_geometry is None:
            history = (odom or {}).get("velocity_history")
            if scan.receipt_only and scan.near_obstacles.has_cells and scan.poses and history:
                scan.sync_buffered_motion(history, odom.get("received_ms", history[-1][0]))
            return Decision(_twist(0., 0., 0.), BLIND, "等待目标最前端的有效深度")
    elif track is not None and getattr(track, "front_required", False) and track.front_fix is None and anchor is None:
        track = None
    target_fix = None
    refresh_target = reobserved_target = False
    freeze_goal = config.local_controller == "dwa"
    if anchor is not None:
        approach_command = state.commanded[:2]
        geometry = (config.scan_geometry or config.local_geometry) if freeze_goal else None
        now_ms = (depth or {}).get("control_ms") or (odom or {}).get("received_ms")
        anchor.advance(odom, ego, dt, geometry=geometry, now_ms=now_ms, approach_command=(
            approach_command if state.active_scan.exhausted and any(approach_command) else None))
        target_fix = _fresh_detour_fix(anchor, track, config, depth)
        if track is None or not track.observed or (freeze_goal and track.generation != anchor.generation):
            anchor.target_missing = True
        if target_fix is not None:
            reobserved_target = anchor.target_missing
            anchor.target_missing = False
            if not freeze_goal:
                moved = math.hypot(target_fix[2] - anchor.x, target_fix[3] - anchor.y)
                refresh_target = moved > max(3 * config.range_std_m, 3 * local_path.RESOLUTION)
                anchor.x, anchor.y = target_fix[2:]
                anchor.track = track
                anchor.predicted_m = anchor.predicted_rad = 0.
        anchor.verified_range_m = target_fix[0] if target_fix is not None else None
        if anchor.verified_range_m is not None:
            best = anchor.best_verified_distance_m
            anchor.best_verified_distance_m = min(best, anchor.verified_range_m) if best is not None else anchor.verified_range_m
        if (target_fix is None and (anchor.predicted_m > MAX_PREDICTED_DETOUR_TRAVEL_M
                                    or anchor.predicted_rad > math.pi / 3 + 1e-9)):
            return Decision(_twist(0., 0., 0.), AVOIDING, "绕行目标的无反馈推算已达上限，停车")

        if math.hypot(anchor.x, anchor.y) <= config.arrival_distance_m + 1e-6:
            scan = state.active_scan
            if scan.active or scan.result_frame or scan.exhausted or state.detour_leg is not None:
                _reset_detour_route(state)
            refresh_target = reobserved_target = False

    feedback = planar_feedback(odom) if config.local_planning else None
    if config.local_planning and config.local_geometry is not None:
        if odom and all(odom.get(axis) is not None for axis in ("vx", "vy", "wz")):
            state.uses_received_speed = "received_speed_ms" in odom
        motion_pose = (odom or {}).get("motion_pose")
        velocity_history = (odom or {}).get("velocity_history")
        source = ("command" if feedback is None else
                  "pose" if motion_pose is not None and not velocity_history else "measured")
        # An absolute-pose reset is relevant only if this controller uses
        # that reference. Body velocities can continue through a SLAM reset.
        frame_changed = (source == "pose" and state.motion_pose_frame is not None
                         and motion_pose.frame != state.motion_pose_frame)
        if source == "command" and config.odom_expected and not state.motion_source:
            state.feedback_boot_s += dt
            if state.feedback_boot_s * 1000 < ODOM_GAP_MS:
                return Decision(_twist(0, 0, 0), AVOIDING, "等待首个有效运动反馈")
        if state.motion_source and (state.motion_source != source or frame_changed):
            needs_guard = any(state.commanded) or state.active_scan.active
            previous_scan = state.active_scan
            if source == "command" and previous_scan.active and not previous_scan.command_only:
                hold = (previous_scan._hold_for_feedback_gap if config.local_controller == "dwa"
                        else previous_scan.hold_for_replan)("主动观察：运动反馈暂不可用，停车后重规划")
                return Decision(_twist(0, 0, 0), AVOIDING, hold.reason)
            previous_scan.invalidate()
            state.active_scan = ActiveScan(obstacles=previous_scan.obstacles,
                                           near_obstacles=previous_scan.near_obstacles,
                                           near_projection=previous_scan.near_projection,
                                           observed_world=previous_scan.observed_world)
            if needs_guard:
                state.feedback_guard_s = (config.watchdog_ms + 50) / 1000
                state.feedback_switch_ms = (depth or {}).get("control_ms")
        state.motion_source = source
        if source == "pose":
            state.motion_pose_frame = motion_pose.frame
        frame = (depth or {}).get("frame_id") or 0
        if state.feedback_guard_s > 0. and state.feedback_switch_ms is not None and not any(state.commanded):
            # Confirm standstill with distinct speed updates after the stop decision.
            now_ms = (depth or {}).get("control_ms")
            recent = ((velocity_history or ())[-2:] if (odom or {}).get("stamp_ms") is not None
                      else (odom or {}).get("received_speed_pair", ()))
            if (now_ms is not None and len(recent) == 2
                    and all(row[1] is not None and row[2] is not None for row in recent)
                    and state.feedback_switch_ms < recent[0][1] < recent[1][1] <= now_ms
                    and now_ms - recent[0][1] <= config.max_feedback_gap_ms
                    and all(math.hypot(*v[:2]) <= STILL_LINEAR and abs(v[2]) <= STILL_YAW
                            for _, _, v in recent)):
                state.feedback_guard_s = 0.
        if state.feedback_guard_s > 0.:
            state.feedback_guard_s = max(0., state.feedback_guard_s - dt)
            return Decision(_twist(0, 0, 0), AVOIDING, "运动反馈来源已切换，等待底盘停稳")
        if config.local_controller == "dwa" and state.uses_received_speed:
            speed_ms = (odom or {}).get("received_speed_ms")
            now_ms = (depth or {}).get("control_ms") or (odom or {}).get("received_ms")
            if (feedback is None or speed_ms is None or now_ms is None
                    or not 0 <= now_ms - speed_ms <= config.max_feedback_gap_ms):
                return Decision(_twist(0., 0., 0.), AVOIDING, "运动反馈已到期，停车等待新速度")
        if any(state.commanded):
            state.command_tail = state.commanded
            state.command_tail_s = (config.watchdog_ms + 50) / 1000
        else:
            state.command_tail_s = max(0., state.command_tail_s - dt)
            if not state.command_tail_s:
                state.command_tail = (0., 0., 0.)

        scan = state.active_scan
        scan.keep_physical_views = config.local_controller == "dwa"
        if scan.use_obstacle_history and not config.use_obstacle_history:
            scan.discard_obstacle_history()
        scan.use_obstacle_history = config.use_obstacle_history
        scan.odom_linear_tolerance_mps = config.scan_odom_linear_tolerance_mps
        scan.max_feedback_gap_ms = config.max_feedback_gap_ms
        obstacle_pose = _obstacle_reference(odom, depth)
        if obstacle_pose is None and config.local_controller == "dwa" and scan.pose_registration is not None:
            received = (odom or {}).get("received_ms")
            if type(received) is int:
                registered = dwa_body_pose(scan, received, odom)
                if registered is not None and registered.stamp_ms == received:
                    obstacle_pose = registered
        scan.project_history(obstacle_pose)
        scan.historical_free = scan.obstacles.project_free(obstacle_pose)
        if scan.obstacles.has_occupied and scan.historical_hits is None:
            return Decision(_twist(0., 0., 0.), AVOIDING, "缺少同坐标系位姿，保留已观测障碍并停车")
        mount = (depth or {}).get("body_from_optical")
        if mount is not None and mount[0, 2] < .75:
            if scan.active and not scan.command_only:
                scan.hold_for_replan("主动观察：等候头部恢复前向观测")
            return Decision(_twist(0., 0., 0.), AVOIDING, "相机朝向未覆盖车体前方，停车")
        was_scanning = scan.active
        can_align = (config.scan_geometry is not None and not (depth or {}).get("receipt_timed")
                     and (depth or {}).get("capture_ms")
                     and (odom or {}).get("stamp_ms"))
        receipt_only = scan.receipt_only if was_scanning or scan.result_frame else not can_align
        geometry = (config.scan_geometry or config.local_geometry) if receipt_only else config.scan_geometry
        published = state.published_dwa_route
        camera_passage = state.published_camera_passage
        corner = (published is not None and published.corridor_corner
                  or camera_passage is not None and camera_passage.corridor_corner)
        scan_depth = (depth if (detections is not None or anchor is not None or corner)
                      and not (depth or {}).get("capture_invalid") else None)
        if source == "command":
            geometry = config.scan_geometry or config.local_geometry
            update = scan.observe_single(scan_depth, geometry, config.depth_age_limit_ms,
                                         (depth or {}).get("control_ms", frame))
            depth = (dict(depth, frame_id=scan.single_depth_ms) if scan.single_depth_ms else None)
        else:
            update = scan.update(scan_depth, odom, geometry,
                                 receipt_only=receipt_only, max_depth_age_ms=config.depth_age_limit_ms,
                                 follow_leg=state.detour_leg is not None,
                                 gap_speed_bounds=config.gap_speed_bounds
                                 if config.use_obstacle_history and config.scan_geometry is None else None)
        scan.observation_pose = obstacle_pose
        capture_pose = (depth or {}).get("obstacle_capture_pose")
        if (update is None and scan.phase == "turn" and not scan.route_segment
                and scan.current_frame and scan.matches_route_view(depth)):
            if obstacle_pose is not None and capture_pose is not None and obstacle_pose.frame == capture_pose.frame:
                scan.record_observation(capture_pose)
            elif scan.views and not scan.command_only:
                vantage = scan.views[-1][1]
                if not scan._already_observed(vantage, 0.):
                    scan.observed_vantages.append(vantage)
        if (config.use_obstacle_history and config.scan_geometry is None
                and geometry is not None and geometry.horizontal is not None):
            target_range = (track.range_m if track is not None and track.range_known else math.inf)
            if anchor is not None:
                target_range = min(target_range, math.hypot(anchor.x, anchor.y))
            reach = min(config.approach_check_distance_m + geometry.front,
                        target_range - FORWARD_CLEARANCE_MARGIN_M)
            approaching = (feedback is not None and (track is not None or anchor is not None)
                           and max(feedback[0], state.commanded[0]) > STILL_LINEAR)
            near_hold = scan.update_near_obstacles(scan_depth, geometry, obstacle_pose, _keepout_m(config), reach,
                                                   approaching=approaching)
            if near_hold is not None:
                return Decision(_twist(0., 0., 0.), AVOIDING, near_hold.reason)
        if scan.gap_recheck is not None:
            return Decision(_twist(0., 0., 0.), AVOIDING,
                            update.reason if update is not None else "主动观察：反馈恢复，停车等待旧视图到期")
        if state.reacquire_generation is not None:
            if scan.active or scan.result_frame or state.detour_leg is not None:
                _reset_detour_route(state)
            if depth is None:
                return Decision(_twist(0., 0., 0.), BLIND, "原地搜索：等待可用深度")
            if matched_person:
                return Decision(_twist(0., 0., 0.), ALIGNING, "已关联到人，停车等待目标确认或新深度")
            state.missing_frames = max(state.missing_frames, config.lost_frames)
            return _search_for_target(depth, config, state, dt, odom)
        if config.local_controller == "dwa" and depth is not None and geometry is not None:
            live_range = track is not None and track.observed and track.measured_range_m is not None
            arc = None
            if corner and not live_range:
                if scan.command_only:
                    arc = _camera_detour(depth, config, state, geometry, config.vx_max, None, None, corner_only=True)
                else:
                    arc = _follow_published_route(depth, config, state, geometry, None, None,
                                                  foreground=True, corner_only=True) if published is not None else None
                if arc is not None:
                    vx, vy, wz, status, _ = arc
                    return Decision(_twist(vx, vy, wz), status, "沿已验证转角续行，等待目标重新出现")
                return Decision(_twist(0., 0., 0.), BLIND, "转角续行证据不足，等待新深度或目标")
            elif live_range and ((published is not None and published.motion_guard.corner_only)
                  or (camera_passage is not None and camera_passage.corner_only)):
                # The next segment can use the newly seen person. The publisher
                # still finishes the sweep already admitted on the wire.
                state.goal_anchor = anchor = None
                target_fix = None
                refresh_target = reobserved_target = False
            elif live_range and update is None:
                distance, bearing = track.measured_range_m, track.bearing_rad
                if corner and scan.command_only and camera_passage is not None:
                    arc = _camera_detour(depth, config, state, geometry, config.vx_max, distance, bearing)
                elif corner and published is not None:
                    arc = _follow_published_route(depth, config, state, geometry, distance, bearing, foreground=True)
                if arc is None and not corner:
                    arc = _preview_corridor_corner(depth, config, state, geometry, distance, bearing)
                if arc is not None:
                    vx, vy, wz, status, reason = arc
                    return Decision(_twist(vx, vy, wz), status, reason, distance_m=distance)
        if needs_probe:
            return _visual_approach(depth, config, state, geometry, track, feedback, objects_ms, detections)
        reviewing_observation = (target_fix is not None and feedback is not None
                                 and geometry is not None and geometry.horizontal is not None
                                 and ((freeze_goal and not (scan.active and scan.route_segment)) or reobserved_target or (
                                     scan.phase in ("brake", "turn", "return") and not scan.route_segment)))
        keep_segment = scan.active and scan.route_segment
        measured_recovery = (config.local_controller == "dwa" and scan.active and not scan.command_only
                             and anchor is not None and scan.phase != "brake_for_replan")
        if ((refresh_target or reviewing_observation) and update is None
                and scan.current_frame and scan.matches_route_view(depth)):
            distance, angle = target_fix[:2]
            normalized = _clamp(angle / config.half_fov_rad, 1.)
            vx = min(config.vx_max, max(0., config.k_fwd * (distance - config.stop_distance_m)))
            vx *= _alignment_scale(normalized, config)
            yaw_gain = config.response_yaw_gain if config.response_yaw_gain is not None else config.k_yaw
            yaw = _clamp(-yaw_gain * normalized, config.wz_max)
            if yaw:
                yaw_floor = config.floor_wz_moving if vx else config.floor_wz
                yaw = math.copysign(min(max(abs(yaw), yaw_floor),
                                        _scan_yaw_rate(config, 1. if yaw < 0. else -1.)), yaw)
            direct = _target_corridor_clear(depth, config, scan, distance, angle,
                                            (feedback or state.command_tail, state.commanded, (vx, 0., yaw)))
            if direct or (refresh_target and not keep_segment):
                _reset_detour_route(state)
                was_scanning = False
                if direct:
                    state.goal_anchor = anchor = None
                refresh_target = False
        if (anchor is not None and measured_recovery and (not scan.ready or scan.advance_m is None) and update is None
                and _dwa_unavailable_reason(depth, config, state, recovery=True) is None and scan.new_dwa_view()):
            distance, bearing = math.hypot(anchor.x, anchor.y), math.atan2(-anchor.y, anchor.x)
            _, hits = scan.grid()
            ahead = projected_forward_clearance(hits, geometry, config)
            side = scan.directions[0] if scan.directions else 1.
            directions = side, -side
            result = _dwa_route_motion(depth, config, state, geometry, ahead, distance, bearing, directions)
            if result is not None:
                vx, vy, yaw, status, reason = result
                return Decision(_twist(vx, vy, yaw), status, reason, distance_m=distance,
                                bearing=_clamp(bearing / config.half_fov_rad, 1.))
        # The new target is already stored on the anchor. An executing segment
        # still has its own fresh sweep and stop-distance checks.
        if refresh_target and not keep_segment and not measured_recovery:
            _reset_detour_route(state)
            return Decision(_twist(0., 0., 0.), AVOIDING, "局部规划：已更新目标位置，等待可用的规划观测")
        if was_scanning and not scan.active:
            return Decision(_twist(0, 0, 0), AVOIDING,
                            update.reason if update else "主动观察中断；停车后重规划")
        if update is not None and (scan.active or scan.result_frame):
            return Decision(_twist(0, 0, 0), AVOIDING, update.reason)
        if scan.active and not scan.ready:
            command = _observation_command(scan, geometry, config, state, odom, dt, control_period_s, depth)
            if command.replanned_turn:
                state.view_detour = None
            return Decision(_twist(command.vx, 0, command.yaw), AVOIDING, command.reason,
                            distance_m=math.hypot(anchor.x, anchor.y) if state.dwa_selected and anchor else None)
        if (scan.ready or scan.result_frame) and anchor is None and (track is None or not track.observed):
            scan.finish()
            scan.note_clear_path(scan.current_frame)
            return _search_for_target(depth, config, state, dt, odom)

    if depth is None or (detections is None and anchor is None):
        if stale_capture:
            return Decision(_twist(0., 0., 0.), BLIND, DEPTH_CAPTURE_STALE)
        if invalid_depth:
            state.recovery = True
            state.escape_side = 0.0
            state.side_better_frames = 0
            return Decision(_twist(0., 0., 0.), AVOIDING, "深度图没有有效测距，看不清；保持停车")
        missing = "vop / depth"
        if detections is not None:
            missing = "depth"
        elif depth is not None:
            missing = "vop"
        return Decision(None, BLIND, f"observation stale or missing ({missing})")

    previous_vy = state.commanded[1]
    lateral_progress = (feedback is not None and previous_vy != 0.
                        and math.copysign(1., previous_vy) * feedback[1] >
                        max(STILL_LINEAR, abs(previous_vy) * config.stuck_speed_ratio))
    side = state.escape_side
    if feedback is not None and state.command_dt_s and side * previous_vy < 0.:
        if lateral_progress:
            state.side_stall_s.pop(side, None)
        else:
            # Count only time covered by a lateral request and measured
            # feedback. A small same-sign standing bias is not progress.
            state.side_stall_s[side] = state.side_stall_s.get(side, 0.) + state.command_dt_s

    bands = depth.get("bands") or {}

    scan = state.active_scan
    if (not freeze_goal and anchor is not None and target_fix is not None and target_fix[0] > config.arrival_distance_m + 1e-6
            and not (state.escape_side or scan.active or scan.result_frame or scan.exhausted)):
        anchor = state.goal_anchor = None

    person_approach = (config.local_planning and config.local_geometry is not None
                       and config.local_controller == "dwa" and state.target == "person")
    missing_person = person_approach and not (track is not None and track.observed)
    if missing_person and track is not None and anchor is None and not (
            state.published_dwa_route or state.published_camera_passage or scan.active):
        state.commanded_vx = 0.
        if matched_person:
            return Decision(_twist(0., 0., 0.), ALIGNING, "已关联到人，停车等待目标确认")
        return _search_for_target(depth, config, state, dt, odom)

    if track is None and anchor is None:
        if (state.searched_rad > 0. and state.tracker.track is not None
                and (matched_person if person_approach else any(target_objects(detections, state.target, config)))):
            return Decision(_twist(0., 0., 0.), ALIGNING, "已看到目标，等待新观测确认")
        return _search_for_target(depth, config, state, dt, odom)

    if not missing_person:
        state.missing_frames = 0
    if anchor is None and not missing_person:
        state.searching_for_s = 0.0
        state.searched_rad = 0.0
    # Back to the normalised offset the rest of this file is written in.
    # `track.py` works in metres and radians; every threshold here — align_tol,
    # arrive_align_tol, lateral_max_bearing — is a fraction of the image half
    # width, and converting them instead would change what every deployed
    # config means.
    route_fix = target_fix if anchor is None or not freeze_goal else None
    bearing_rad = (route_fix[1] if route_fix is not None else
                   math.atan2(-anchor.y, anchor.x) if anchor is not None else track.bearing_rad)
    bearing = _clamp(bearing_rad / max(1e-6, config.half_fov_rad), 1.0)
    if not missing_person:
        state.last_seen_side = 1.0 if bearing >= 0 else -1.0
    distance = (route_fix[0] if route_fix is not None else
                math.hypot(anchor.x, anchor.y) if anchor is not None else
                track.range_m if track.range_known or getattr(track, "front_fix", None) is not None else None)

    # Turn towards it, proportionally. Positive bearing means the target is
    # right of centre, and `wz` is positive counter-clockwise, so the sign is
    # inverted here.
    #
    # **Left unquantised for now.** Whether this has to be lifted to a deadband
    # depends on something not known yet at this point — whether the robot ends
    # up translating — so the decision is deferred to `_finalise_yaw` once vx is
    # settled. See `floor_wz_moving`.
    live = anchor is None and track.observed
    yaw_gain = (config.response_yaw_gain if config.response_yaw_gain is not None
                else config.k_yaw)
    # Pure proportional steering. odom.wz remains an observation for
    # ego-motion tracking, but is never fed back into the command as a hidden
    # D-like term.
    raw_wz = _clamp(-yaw_gain * bearing, config.wz_max)

    within = distance is not None and distance <= config.arrival_distance_m + 1e-6
    if freeze_goal and target_fix is not None and target_fix[0] <= config.arrival_distance_m + 1e-6:
        within = True
        distance = target_fix[0]
        bearing = _clamp(target_fix[1] / max(1e-6, config.half_fov_rad), 1.)
    if within:
        if missing_person:
            return _search_for_target(depth, config, state, dt, odom)
        observed = live or target_fix is not None or (
            anchor is not None and track is not None and (track.generation == anchor.generation if freeze_goal else track is anchor.track) and track.observed
            and not anchor.target_missing and (
                not getattr(track, "front_required", False) or track.front_fix is not None))
        measured = target_fix[0] if target_fix is not None else track.measured_range_m if live else None
        last = getattr(track, "last_obs", None)
        if live and measured is not None and last is not None and getattr(track, "front_fix", None) is None:
            bearing = _clamp(math.atan2(-last[1], last[0]) / max(1e-6, config.half_fov_rad), 1.)
        return _parking_alignment(depth, config, state, odom, distance, bearing, measured, observed)

    # Forward speed scales continuously with alignment rather than switching.
    #
    # A gate (vx = 0 whenever |bearing| > align_tol) deadlocks at close range:
    # `bearing` is a normalised lateral offset, so the same sideways step is far
    # larger at 0.7 m than at 5 m, and a person standing in front of the robot
    # only has to shift slightly for the card never to clear it. On r1_sz that
    # presented as "the person is right there and the only thing moving is wz".
    # Scaling keeps the intent — turn first when badly off — without the
    # threshold.
    align = _alignment_scale(bearing, config)
    if align <= 0.0:
        state.commanded_vx = 0.0
        state.avoidance_yaw_override = False
        reason = f"目标偏离画面中心 {bearing:+.2f}（归一化），先原地转正"
        vx, vy, turn, status, reason = _avoid(
            0.0, 0.0, raw_wz, depth, config, ALIGNING, reason, state,
            target_distance=distance, target_bearing=bearing_rad, measured_vx=(odom or {}).get("vx"),
            aligned=False, odom=odom, may_scan=live or anchor is not None, dt=dt)
        _remember_detour_goal(state, config, track, odom)
        wz = _finalise_yaw(turn, bool(vx or vy), bearing, config, state, dt)
        state.commanded_vx = vx
        return Decision(_twist(vx, vy, wz), status, reason,
                        distance_m=distance, bearing=bearing,
                        k_yaw_effective=None if state.avoidance_yaw_override else yaw_gain)

    speed = config.vx_max * align
    status, reason = APPROACHING, "approaching"
    forward_gain = _forward_gain(config, state, distance, live, within, dt)
    if distance is not None:
        speed = min(speed, max(0.0, forward_gain
                               * (distance - config.stop_distance_m) * align))
        reason = f"target at {distance:.2f} m"

    # Keep walking until the stop distance is genuinely reached.
    #
    # `speed` falls below the robot's floor well before then — at R1's numbers,
    # 0.67 m before — so without this gate the robot stopped short of a target
    # it could see perfectly well and then failed on the idle timeout. The gate
    # engages on metres remaining, and `_lift` supplies the only speed the robot
    # has for the final stretch.
    remaining = 99.0 if distance is None else distance - config.stop_distance_m
    driving = state.fwd_gate.update(
        remaining, engage=min(.05, config.stop_tolerance_m), release=0.0,
        dt=dt, min_dwell_s=config.min_dwell_s)

    vx, vy = _approach(speed if driving else 0.0, bearing, depth, config,
                       state, dt, odom=odom)
    tracking_wz = raw_wz
    state.avoidance_yaw_override = False
    vx, vy, raw_wz, status, reason = _avoid(
        vx, vy, raw_wz, depth, config, status, reason, state,
        target_distance=distance, target_bearing=bearing_rad, measured_vx=(odom or {}).get("vx"),
        aligned=abs(bearing) < config.align_full_stop, odom=odom,
        may_scan=live or anchor is not None, dt=dt, desired_forward=speed if driving else 0.)
    if missing_person and (vx or vy or raw_wz) and not (
            state.dwa_selected or state.camera_arc_selected or state.executing_leg):
        state.commanded_vx = 0.
        return _search_for_target(depth, config, state, dt, odom)
    if (person_approach and live and distance is not None and status == APPROACHING and vy
            and abs(bearing) > config.align_tol
            and not state.dwa_selected and not state.camera_arc_selected and not scan.command_only
            and scan.views and scan.matches_route_view(depth)):
        # A lateral prefix fixes the heading. Check a turn towards the visible
        # person before committing to another sideways step.
        target = local_path.target_in_grid(geometry, distance * math.cos(bearing_rad), distance * math.sin(bearing_rad))
        limit = _scan_yaw_rate(config, 1. if tracking_wz < 0. else -1.)
        yaw = _lift(tracking_wz, config.floor_wz, limit) if limit >= config.floor_wz else 0.
        if yaw and _check_dwa_turn(depth, config, state, geometry, target, "align", (0., 0., yaw)):
            state.commanded_vx = 0.
            return Decision(_twist(0., 0., yaw), ALIGNING, "原地转向当前可见目标", distance_m=distance, bearing=bearing)
    if not state.camera_arc_selected and not (isinstance(state.dwa_proposal, DwaProposal)
                                              and state.dwa_proposal.branch == "direct"):
        _remember_detour_goal(state, config, track, odom)
    # Legacy uses forward requests and lateral feedback for blockage. DWA
    # accounts measured route and viewpoint progress across all phases in step.
    if (config.local_controller != "dwa" and status == AVOIDING
            and not (vx or (lateral_progress and previous_vy * vy > 0.))):
        state.blocked_for_s += dt
        if state.blocked_for_s >= config.blocked_timeout_s:
            blocked = (f"前方被挡住了 {state.blocked_for_s:.0f} 秒，绕不过去："
                       f"{reason}")
            state.failed_reason = blocked
            return Decision(None, FAILED, blocked,
                            distance_m=distance, bearing=bearing)
    else:
        state.blocked_for_s = 0.0
    # Now vx is settled, so which deadband the yaw axis is subject to is known.
    if (config.local_planning and config.local_geometry is not None
            and not state.dwa_selected
            and (vy or abs(previous_vy) > STILL_LINEAR or (feedback and abs(feedback[1]) > STILL_LINEAR))):
        # The lateral prefix was checked at a fixed heading.
        raw_wz = 0.
    wz = _finalise_yaw(raw_wz, bool(vx or vy), bearing, config, state, dt)

    state.commanded_vx = vx
    tracking_yaw_used = raw_wz == tracking_wz and not state.avoidance_yaw_override
    return Decision(_twist(vx, vy, wz), status, reason,
                    distance_m=distance, bearing=bearing,
                    k_fwd_effective=forward_gain if distance is not None and not within else None,
                    k_yaw_effective=yaw_gain if tracking_yaw_used else None)


def _feedback_speed_reduction(config, odom):
    vx, vy, yaw = (value or 0. for value in planar_axes(odom))
    _, (_, vy_limit) = _local_velocity_limits(config, 1. if vy < 0. else -1.)
    yaw_limit = _scan_yaw_rate(config, 1. if yaw < 0. else -1.)
    ratios = [min(.8, limit / speed) for speed, limit, tolerance in
              ((abs(vx), config.vx_max, .02), (abs(vy), vy_limit, .02), (abs(yaw), yaw_limit, .10))
              if limit > 0. and speed > limit + tolerance]
    return min(ratios, default=1.)


def _previous_progress_route(state):
    """The checked reference in axes relative to the previous measured pivot."""
    detour = state.view_detour
    if state.checked_prefix and detour is not None and len(detour.preview) >= 2:
        guard = state.dwa_motion_guard
        if guard is not None and guard.trace.mode == "camera_approx":
            f, r = guard.trace.body_poses[0][:2]
            return tuple((x - f, y - r) for x, y in detour.preview)
        return detour.preview
    leg = state.executing_leg
    scan = state.active_scan
    if leg is not None and leg.motion_frame is scan.motion_frame:
        # The fixed leg was checked along its original axis, including when
        # the measured pose is already slightly beyond its end.
        dx, dy = PathLeg.world_point((0., 0., leg.origin[2] - scan.pose[2]), leg.waypoint)
    else:
        dx, dy = state.commanded[0], -state.commanded[1]
    norm = math.hypot(dx, dy)
    return ((0., 0.), (dx / norm, dy / norm)) if norm else ()


def _extend_progress_pose(progress, base, received, history, max_gap):
    if base is progress.pose and progress.source is not None:
        history = ((base.stamp_ms, progress.source, None),
                   *(sample for sample in history if sample[0] > base.stamp_ms))
    return _advance_dwa_pose(base, received, {"velocity_history": history}, max_gap)


def _direct_progress_pose(progress, odom, max_gap):
    """Register fresh velocities at the measured pivot, including between absolute pose fixes."""
    before, anchor = progress.pose, odom.get("obstacle_pose") or odom.get("motion_pose")
    last = progress.direct_anchor
    if anchor is not None:
        if last is not None and last.frame == anchor.frame and anchor.stamp_ms < last.stamp_ms:
            progress.interrupted = True
            return before, progress.source, False
        if last is not None and last.frame == anchor.frame and last.stamp_ms == anchor.stamp_ms:
            # A repeated original pose cannot correct the same interval twice.
            anchor = last
        progress.direct_anchor = anchor

    received, source = odom.get("received_ms"), odom.get("source_sample_id")
    motion = planar_feedback(odom)
    if type(received) is not int or motion is None:
        return anchor, None, False
    history = odom.get("velocity_history") or ((received, source, motion),)
    velocity_frame = ("measured_velocity", source is not None)
    frame = anchor.frame if anchor is not None else velocity_frame
    if anchor is None and last is not None and before is not None and last.frame == before.frame:
        frame = before.frame
    registered = before is not None and before.frame == frame
    bridged_to_anchor = False
    if registered and anchor is not None:
        bridged_to_anchor = anchor.stamp_ms - before.stamp_ms <= max_gap
        if not bridged_to_anchor and anchor.stamp_ms <= received:
            # The two absolute fixes may be many control periods apart. Prove
            # the interval with its velocity receipts; the new fix still owns
            # the position at its original stamp.
            covered = _extend_progress_pose(progress, before, received, history, max_gap)
            bridged_to_anchor = covered.stamp_ms == received
    if anchor is not None and anchor.stamp_ms >= received:
        return anchor, source if anchor.stamp_ms == received else None, bridged_to_anchor

    # A new absolute fix replaces the velocity estimate at its own stamp.
    # Extend it only by receipts after that fix. A duplicate uses the last
    # integrated pivot even if the old fix has left the history buffer.
    base = before if registered and (anchor is None or anchor is last) else anchor
    if base is not None:
        if received <= base.stamp_ms or (registered and source is not None and progress.source == source):
            return base, progress.source if base is before else None, False
        integrated = _extend_progress_pose(progress, base, received, history, max_gap)
        if integrated.stamp_ms == received:
            return integrated, source, base is before or bridged_to_anchor
        if anchor is not None:
            return anchor, None, False
        progress.interrupted = True
    # Velocity integration starts from a new arbitrary zero after an
    # uncovered interval. Keep any separately registered world coverage.
    progress.relative_passage = PassageCoverage()
    return PlanarPose(received, velocity_frame, 0., 0., 0.), source, False


def _progress_sample(state, odom, depth, max_gap):
    """Return measured pivot, producer identity and whether history bridges a slow policy tick."""
    progress, scan = state.dwa_progress, state.active_scan
    before = progress.pose
    now = (depth or {}).get("control_ms")
    odom = odom or {}
    bridged = False
    if scan.poses and not scan.command_only:
        stamp, (f, r, yaw), _ = scan.poses[-1]
        pose = PlanarPose(stamp, ("scan", scan.motion_frame), f, -r, -yaw)
        source = scan.odom_source
        # ActiveScan maintains one motion_frame per continuous integration.
        bridged = before is not None and scan.poses[0][0] <= before.stamp_ms
    else:
        pose, source, bridged = _direct_progress_pose(progress, odom, max_gap)
        if pose is None:
            return None, None, False
    if now is not None and not 0 <= now - pose.stamp_ms <= max_gap:
        return None, None, False
    return pose, source, bridged


def _progress_world_pose(progress, pose, scan, odom):
    """Relate scan and direct pivot samples across ordinary sensor handoffs."""
    if pose.frame != ("scan", scan.motion_frame):
        return pose
    if scan.pose_origin is not None:
        return scan._body_pose_at(pose.stamp_ms, (pose.x, -pose.y, -pose.yaw), scan.pose_origin)
    absolute = (odom or {}).get("obstacle_pose") or (odom or {}).get("motion_pose")
    if absolute is not None:
        local = scan._pose_at(absolute.stamp_ms)
        if local is not None:
            reference = PlanarPose(absolute.stamp_ms, pose.frame, local[0], -local[1], -local[2])
            progress.scan_reference = reference, absolute
    if progress.scan_reference is not None:
        reference, absolute = progress.scan_reference
        if reference.frame == pose.frame:
            f, left, yaw = pose.relative_to(reference)
            return scan._body_pose_at(pose.stamp_ms, (f, -left, -yaw), absolute)
    return pose


def _register_progress_views(progress, pose, basis, source, odom, max_gap, bridged):
    """Carry finite angular coverage through a measured coordinate handoff."""
    before = progress.pose
    if (not progress.viewed or progress.view_frame == basis.frame or before is None
            or progress.interrupted or pose.stamp_ms <= before.stamp_ms
            or (source is not None and progress.source is not None and source <= progress.source)):
        return
    reference = None
    if pose.frame == progress.view_frame and before.frame == pose.frame:
        if bridged or pose.stamp_ms - before.stamp_ms <= max_gap:
            # The same measured scan pivot is now registered in world axes.
            reference = pose
    elif before.frame == progress.view_frame:
        # Direct velocity integration may acquire its first absolute fix.
        # Relate both headings at the same stamp using only covered feedback.
        odom = odom or {}
        motion = planar_feedback(odom)
        if motion is not None and odom.get("received_ms") == pose.stamp_ms:
            history = odom.get("velocity_history") or ((pose.stamp_ms, source, motion),)
            reference = _extend_progress_pose(progress, before, pose.stamp_ms, history, max_gap)
            if reference.stamp_ms != pose.stamp_ms:
                reference = None
    if reference is not None:
        offset = reference.yaw - basis.yaw
        progress.viewed[:] = [math.remainder(seen + offset, math.tau) for seen in progress.viewed]
        progress.view_frame = basis.frame


def _new_recovery_view(progress, scan, depth, max_age, heading):
    if (not scan.views or scan.command_only or not depth or depth.get("capture_invalid")
            or not scan.matches_route_view(depth)):
        return False
    stamp, view, free, occupied, *_ = scan.views[-1]
    now = depth.get("control_ms") or scan.control_ms or stamp
    if (not stamp or stamp != scan.current_frame or free is None or occupied is None
            or not 0 <= now - stamp < max_age):
        return False
    key = (("source", scan.source_frame) if scan.source_frame is not None else
           ("receipt" if scan.receipt_only else "capture", stamp))
    evaluated = (scan.side_frame == stamp or scan.dwa_view == key
                 or (scan.phase == "turn" and (view in scan.observed_vantages or
                     (scan.observed_world and scan.observed_world[-1] == depth.get("obstacle_capture_pose")))))
    if not evaluated or key[1] <= progress.last_view.get(key[0], -1):
        return False
    progress.last_view[key[0]] = key[1]
    heading += view[2] - scan.pose[2]
    if any(abs(math.remainder(heading - seen, math.tau)) <= ANGLE / 2 for seen in progress.viewed):
        return False
    progress.viewed.append(heading)
    return True


def _direct_return_progress(progress, state, decision, pose, moved, obstacle_pose):
    track, anchor = state.tracker.track, state.goal_anchor
    direct = (decision.status == APPROACHING and not state.checked_prefix
              and state.view_detour is None and state.executing_leg is None)
    proposal = state.dwa_proposal
    checked_dwa = (decision.status == AVOIDING and state.dwa_selected and state.checked_prefix
                   and isinstance(proposal, DwaProposal) and proposal.guard is state.dwa_motion_guard
                   and proposal.guard.check.accepted and any(proposal.command[:2])
                   and decision.values is not None and not any(decision.values[2:5])
                   and tuple(decision.values[i] for i in (0, 1, 5)) == proposal.command)
    if (track is None or track.generation != progress.goal_generation
            or (anchor is not None and anchor.generation != track.generation)):
        progress.approach_fix = None
        return False

    # The capture and obstacle pose readers share the full SLAM continuity
    # frame. A pose-only scan may index this same measurement by host receipt.
    scan = state.active_scan
    if obstacle_pose is not None:
        stamp = obstacle_pose.stamp_ms
        if stamp == pose.stamp_ms or (scan.pose_origin is not None and stamp == scan.odom_source):
            pose = obstacle_pose
    measured = track.measured_point if current_target_observed(state) else None
    capture = measured.capture_pose if measured is not None else None
    origin = capture or pose
    if origin.frame != pose.frame:
        progress.approach_fix = None
        return False
    previous = progress.approach_fix
    if previous is not None:
        if previous.frame != pose.frame:
            previous = progress.approach_fix = None
        else:
            # A fresh range and a completed measured step need not arrive on
            # the same policy tick.
            previous.travelled |= moved
            if not previous.travelled:
                return False
    # A short observation hold cannot earn credit, but it does not erase a
    # same-target measured step. Keep that evidence until a qualified command
    # and a fresh target measurement can verify both kinds of closure.
    if (not (direct or checked_dwa) or not track.observed
            or (anchor is not None and anchor.target_missing)):
        return False
    if measured is None:
        return False
    point = measured.point
    if measured.frame is not None:
        point = np.asarray(measured.frame.axes) @ point + measured.frame.origin
    target = PathLeg.world_point((origin.x, origin.y, origin.yaw), point)
    pivot = pose.x, pose.y
    distance = math.dist(pivot, target)
    if previous is not None:
        # Compare both physical travel toward the previous fixed observation
        # and the current measured range. Image motion cannot supply the former.
        floor = previous.distance
        if progress.credited_range is not None:
            floor = min(floor, progress.credited_range)
        threshold = local_path.RESOLUTION - 1e-9
        if (previous.distance - math.dist(pivot, previous.target) < threshold
                or floor - distance < threshold):
            return False
        progress.credited_range = distance
    progress.approach_fix = DirectApproachFix(pose.frame, target, distance)
    return previous is not None


def _latch_progress_feedback(state, config, odom):
    """Keep depth-only fallback usable without weakening a measured action."""
    scan, sample = state.active_scan, odom or {}
    state.dwa_progress.requires_feedback |= bool(
        config.odom_expected or state.dwa_selected
        or state.motion_source in ("pose", "measured")
        or (scan.poses and not scan.command_only)
        or planar_feedback(odom) is not None
        or sample.get("obstacle_pose") is not None or sample.get("motion_pose") is not None)


def _account_dwa_progress(decision, state, config, odom, depth, route, dt):
    if decision.status in (ARRIVED, FAILED, IDLE):
        return decision
    progress = state.dwa_progress
    if progress.goal_generation is None and current_target_observed(state):
        progress.goal_generation = state.tracker.track.generation
    pose, source, bridged = _progress_sample(state, odom, depth, config.max_feedback_gap_ms)
    before = progress.pose
    if pose is not None:
        basis = _progress_world_pose(progress, pose, state.active_scan, odom)
        _register_progress_views(progress, pose, basis, source, odom,
                                 config.max_feedback_gap_ms, bridged)
        relative = basis.frame in (("scan", state.active_scan.motion_frame),
                                   ("measured_velocity", True), ("measured_velocity", False))
        passage = progress.relative_passage if relative else progress.world_passage
    advanced = False
    if pose is None:
        progress.rebaseline()
    elif (before is not None and pose.frame == before.frame and
          (pose.stamp_ms < before.stamp_ms or
           (source is not None and progress.source is not None and source < progress.source))):
        # Do not replay old distance. The next newer sample starts a baseline.
        progress.interrupted = True
        progress.approach_fix = None
    elif (before is None or pose.frame != before.frame or progress.interrupted or
          (not bridged and pose.stamp_ms - before.stamp_ms > config.max_feedback_gap_ms)):
        progress.rebaseline()
        progress.pose, progress.source, progress.checkpoint = pose, source, pose
        passage.visit(basis)
        if not progress.viewed:
            progress.view_frame = basis.frame
            progress.viewed.append(-basis.yaw)
    else:
        measured = pose.stamp_ms > before.stamp_ms and (
            pose is progress.direct_anchor or source is None or progress.source is None or source > progress.source)
        moved = False
        if measured:
            progress.pose, progress.source = pose, source
            moved = progress.advance(before, pose, route)
        if moved and passage.visit(basis):
            # New passage may legitimately take us farther from the original
            # goal. Only that new travel releases the previously credited range.
            progress.credited_range = progress.approach_fix = None
            advanced = True
        elif measured or (pose == before and source == progress.source):
            advanced = _direct_return_progress(progress, state, decision, basis, moved,
                                               (odom or {}).get("obstacle_pose"))
        if advanced:
            progress.view_frame = basis.frame
            progress.viewed[:] = [-basis.yaw]
        if (decision.status == SEARCHING and measured and decision.values is not None
                and not any(decision.values[:2]) and basis.frame == progress.view_frame):
            # Normal target search does not evaluate an ActiveScan route. A
            # fresh measured turn into a new sector is nevertheless useful
            # progress, bounded by the existing search sweep and deadline.
            turn = math.remainder(pose.yaw - before.yaw, math.tau)
            heading = -basis.yaw
            if (turn * decision.values[5] > 0. and not any(
                    abs(math.remainder(heading - seen, math.tau)) <= ANGLE / 2
                    for seen in progress.viewed)):
                progress.viewed.append(heading)
                advanced = True
        if pose.frame == ("scan", state.active_scan.motion_frame) and basis.frame == progress.view_frame:
            # After an unregistered scan reset the new zero heading cannot
            # identify an unseen sector in the old frame.
            if _new_recovery_view(progress, state.active_scan, depth, config.depth_age_limit_ms, -basis.yaw):
                advanced = True
    state.idle_for_s = 0. if advanced else state.idle_for_s + dt
    if state.idle_for_s < config.idle_timeout_s:
        return decision
    state.failed_reason = (f"no measured navigation progress for {config.idle_timeout_s:g}s "
                           f"(last state: {decision.status} — {decision.reason})")
    return Decision(_twist(0., 0., 0.), FAILED, state.failed_reason,
                    distance_m=decision.distance_m, bearing=decision.bearing)


def _account_idle(decision: Decision, state: State, config: Config,
                  dt: float) -> Decision:
    """Turn "has not moved for a while" into a failure, and nothing else into one.

    Deliberately not a per-phase wall clock. A robot walking steadily towards
    something thirty metres away is working, and a timeout on "how long has this
    navigate_to been running" would kill it for succeeding slowly. What actually
    distinguishes a stuck task is that **no motion is being commanded** — and
    that one test covers every way of getting there at once: blind, occluded,
    refused downstream, or simply deciding nothing, without this function having
    to enumerate them.

    Turning counts as motion; an all-zero twist does not. Arriving is exempt —
    it is a success that happens to command zero.
    """
    if decision.status in (ARRIVED, FAILED, IDLE):
        return decision

    moving = decision.values is not None and any(abs(v) > 1e-6
                                                 for v in decision.values)
    if moving:
        state.idle_for_s = 0.0
        return decision

    state.idle_for_s += dt
    if state.idle_for_s < config.idle_timeout_s:
        return decision

    state.failed_reason = (
        f"no motion command issued for {config.idle_timeout_s:.0f}s "
        f"(last state: {decision.status} — {decision.reason}); giving up")
    return Decision(None, FAILED, state.failed_reason,
                    distance_m=decision.distance_m, bearing=decision.bearing)


def _finalise_yaw(raw: float, translating: bool, bearing: float,
                  config: Config, state: State, dt: float) -> float:
    """Quantise the yaw command onto what the robot can execute *right now*.

    The floor is not a constant of the axis — it is a constant of the *gait*.
    Standing, R1 does nothing below 1.0 rad/s. Walking, 0.05 rad/s is visible.
    So the same 0.05 command is unexecutable in one state and fine in the other,
    and which state applies is only known after `_avoid` has had its say about
    vx.

    The gate and the lift apply to **one band only**: commands the robot cannot
    execute, i.e. below the floor. Above it, proportional control works and the
    machinery is not merely unnecessary but harmful — both exist *because* a
    deadbanded axis cannot correct slowly, and applying them where it can turns
    a 0.05 rad/s correction into a 1.0 rad/s one: a twenty-fold overshoot, a
    reversal, and another overshoot.

    Conditioning on the command rather than on the state is what makes this one
    rule instead of two: standing, almost every useful yaw lands under the 1.0
    floor and is gated; walking, almost nothing does.
    """
    if state.dwa_selected or state.camera_arc_selected:
        state.yaw_gate.on = False
        state.yaw_gate.held_s = 0.
        return raw
    floor = config.floor_wz_moving if translating else config.floor_wz
    ceiling = _scan_yaw_rate(config, 1. if raw < 0. else -1.)
    if ceiling < floor:
        return 0.
    raw = _clamp(raw, ceiling)
    if floor <= 1e-6 or abs(raw) >= floor:
        # Keep the gate's state coherent, so a later transition back to turning
        # in place does not inherit a stale "already engaged".
        state.yaw_gate.on = False
        state.yaw_gate.held_s = 0.0
        return raw

    turning = state.yaw_gate.update(
        abs(bearing), engage=config.align_tol,
        release=config.align_tol * config.release_frac,
        dt=dt, min_dwell_s=config.min_dwell_s)
    return _lift(raw, floor, ceiling) if turning else 0.0


def _alignment_scale(bearing: float, config: Config) -> float:
    """前进速度乘的那个系数：对得越正走得越快。

    1.0 直到 align_tol，之后线性降到 align_full_stop 处的 `align_min_scale`。

    这个下限以前是硬 0 —— 目标一偏就完全停下先转正。对能横移的底盘没有理由这么
    做：`_approach` 会把速度按方位角拆成 vx/vy，本来就是朝着目标走的，停下来只是
    多一次起步。留 `align_min_scale: 0.0` 可以退回原来的走法，不能横移的底盘应该
    这么配。
    """
    offset = abs(bearing)
    if offset <= config.align_tol:
        return 1.0
    floor = max(0.0, min(1.0, config.align_min_scale))
    if offset >= config.align_full_stop:
        return floor
    span = max(1e-6, config.align_full_stop - config.align_tol)
    return floor + (1.0 - floor) * (config.align_full_stop - offset) / span


def _keepout_m(config: Config) -> float:
    """The half-width to keep clear: the declared box plus a margin for the
    parts of the robot the box does not describe."""
    return config.half_width_m + config.clearance_margin_m


def _side_label(room, config: Config) -> str:
    """One side's verdict, for a reason string an operator has to act on."""
    if room is None:
        return "无读数"
    if room <= config.side_clearance_m:
        return f"{room:.2f} m 处有东西"
    return f"{room:.2f} m"


def _clearance(depth, config: Config, lateral_offset_m: float = 0.0):
    """`(nearest obstacle in the corridor, how much of it was measured)`.

    With a depth map this is the metric corridor — the space the robot will
    actually pass through. With only the summary it falls back to the angular
    thirds, and **says so by reporting full coverage it has not earned**: the
    summary has no way to express how much of a band was measured, so the
    fallback cannot detect a band full of holes. That is one of the degradations
    the card reports at start; it is not something this function can fix.
    """
    from . import depth as depth_mod

    bands = (depth or {}).get("bands") or {}
    if (depth or {}).get("map") is None:
        band = ("center" if abs(lateral_offset_m) < 1e-6
                else "right" if lateral_offset_m > 0 else "left")
        return bands.get(band), 1.0
    geometry = config.scan_geometry or config.local_geometry
    ignored = None
    if geometry is not None and depth.get("ground") is not None:
        grid = local_path.project_observation(depth, geometry, require_valid_depth=True)
        ignored = getattr(grid, "ignored_low_pixels", None)
    return depth_mod.corridor(depth["map"],
                              half_width_m=_keepout_m(config),
                              half_fov_rad=config.half_fov_rad,
                              reference_m=config.obstacle_stop_m,
                              lateral_offset_m=lateral_offset_m,
                              intrinsics=config.depth_intrinsics,
                              ground=depth.get("ground"),
                              ignored_low_pixels=ignored)


def forward_stop_line(config, speed):
    """Reserve forward travel during the existing depth freshness window."""
    return min(config.approach_check_distance_m,
               config.obstacle_stop_m + max(0., speed) * (config.depth_age_limit_ms / 1000))


def projected_forward_clearance(occupied, geometry, config):
    """Nearest chassis-corridor return expressed in the camera's range frame."""
    hits = occupied & (FORWARD >= 0.) & (abs(RIGHT) <= _keepout_m(config))
    transform = geometry.base_from_optical
    return (max(0., (float(FORWARD[hits].min()) - transform[0, 3]) / transform[0, 2])
            if hits.any() and transform[0, 2] > 0. else None)


def _may_strafe(towards_right: bool, depth, config: Config) -> bool:
    """Check measured obstacles in the destination corridor before strafing."""
    floor, limit = _local_velocity_limits(config, 1. if towards_right else -1.)[1]
    if limit <= 0 or floor > limit:
        return False
    offset = _keepout_m(config) * (1.0 if towards_right else -1.0)
    room, _ = _clearance(depth, config, lateral_offset_m=offset)
    return room is None or room > config.approach_check_distance_m


def _approach(speed: float, bearing: float, depth, config: Config,
              state: "State" = None, dt: float = 0.0, *, odom=None):
    """Split the approach speed between forward and sideways.

    The target sits at roughly `bearing * half_fov_rad` off the nose, so
    travelling towards it is that speed rotated by that angle — which is a
    straight line to the thing, walked while turning to face it, instead of a
    turn followed by a walk.

    The two floors quantise the result, so the realised direction is coarser
    than the computed one. That is the robot, not the arithmetic: it cannot put
    0.12 m/s on an axis. The yaw loop is closing the same error at the same
    time, so a coarse arc still converges.
    """
    if speed <= 0.0:
        return 0.0, 0.0

    angle = bearing * config.half_fov_rad
    (floor_x, limit_x), (floor_y, limit_y) = _local_velocity_limits(config, 1. if bearing > 0 else -1.)
    forward = _lift(speed * math.cos(angle), floor_x, limit_x)

    if abs(bearing) > config.lateral_max_bearing:
        # Too far off to trust either the bearing or the side band. Turn.
        return forward, 0.0
    lateral = -speed * math.sin(angle) * config.lateral_scale
    if lateral == 0.0 or not _may_strafe(lateral < 0, depth, config):
        return forward, 0.0
    if not _worth_strafing(lateral, config, state, dt):
        return forward, 0.0
    lateral = _lift(lateral, floor_y, limit_y)
    geometry = config.local_geometry
    if state is not None and config.local_planning and geometry is not None:
        if depth.get("map") is None or depth.get("frame_id") is None:
            return forward, 0.
        scan = state.active_scan
        same_geometry = (config.scan_geometry or config.local_geometry) is geometry
        grid = (scan.grid() if same_geometry and scan.views and scan.matches_route_view(depth)
                else state.local_plan.grid_for(depth, geometry))
        previous = state.commanded[:2]
        measured = planar_feedback(odom)
        yaw = measured[2] if measured is not None else state.command_tail[2]
        if max(abs(yaw), abs(state.commanded[2])) > STILL_YAW:
            return forward, 0.
        motion = measured[:2] if measured is not None else previous
        horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
        if not local_path.transition_clear(grid, geometry, _keepout_m(config), horizon,
                                           (motion, previous, (forward, lateral))):
            forward_clear = local_path.transition_clear(grid, geometry, _keepout_m(config), horizon,
                                                        (motion, previous, (forward, 0.)))
            return forward if forward_clear else 0., 0.
    return forward, lateral


def _worth_strafing(lateral: float, config: Config, state, dt: float) -> bool:
    """Whether this lateral component is big enough to be worth a real sidestep.

    **`_lift` is not free on a deadbanded axis.** It raises whatever it is given
    to the floor, so a target 0.05 off centre asks for 0.04 m/s and receives
    R1's minimum 0.4 — a **ninefold** amplification, and a visible sideways lurch
    at the start of every run before the robot settles into walking forward.
    That is what an operator sees as "it always does a sidestep it does not
    need, and then comes towards me".

    The yaw axis already had this covered (`_finalise_yaw` + its gate); the
    lateral one did not, which is the whole bug.

    The threshold is derived rather than tuned: strafe only when the honest
    lateral is at least half the floor, i.e. when `_lift` is amplifying by no
    more than 2x. Below that the yaw loop closes the same error, more slowly and
    without the lurch — and it is already turning anyway.

    Hysteresis and dwell for the same reason every other gate here has them: an
    error sitting on the threshold would otherwise switch the axis on and off at
    tick rate, which is the juddering this card was written to remove.
    """
    floor = config.floor_vy if config.floor_vy_moving is None else config.floor_vy_moving
    if floor <= 1e-6:
        return True                      # no deadband, no amplification, no gate
    engage = floor * config.lateral_lift_ratio
    if state is None:
        return abs(lateral) >= engage
    return state.vy_gate.update(
        abs(lateral), engage=engage, release=engage * config.release_frac,
        dt=dt, min_dwell_s=config.min_dwell_s)


def _search_turn_stop(depth, config: Config, state: State, *, requested_yaw=None) -> str | None:
    """Check the next target-search turn, including travel before a watchdog stop."""
    ahead, coverage = _clearance(depth, config)
    state.last_clearance, state.last_coverage = ahead, coverage
    scan = state.active_scan
    command_only = state.motion_source == "command"
    geometry = (config.scan_geometry or config.local_geometry
                if scan.receipt_only or command_only else config.scan_geometry)
    can_plan = config.local_planning and (config.search_rate > 0 or requested_yaw is not None)
    has_sweep = can_plan and geometry is not None and bool(scan.current_frame)
    if not has_sweep:
        if requested_yaw is None and ahead is not None and ahead <= config.obstacle_stop_m:
            state.recovery = True
            return f"原地搜索：前方 {ahead:.2f} m 触发近障停车"
        if requested_yaw is not None or (can_plan and config.local_geometry is not None):
            return "原地搜索：等待可用深度"
    if can_plan:
        direction = state.last_seen_side if requested_yaw is None else -math.copysign(1., requested_yaw)
        if (requested_yaw is None and not state.searching_for_s
                and ahead is not None and ahead < config.approach_check_distance_m
                and depth.get("map") is not None):
            keep = _keepout_m(config)
            alternate, seen = _clearance(depth, config, lateral_offset_m=-direction * keep)
            current, _ = _clearance(depth, config, lateral_offset_m=direction * keep)
            if (alternate is not None and alternate >= config.approach_check_distance_m
                    and seen >= config.coverage_min
                    and (current is None or alternate - current >= SIDE_SWITCH_ADVANTAGE_M)):
                direction = -direction
        motion = state.command_tail if command_only else scan.motion
        yaw = -direction * config.search_rate if requested_yaw is None else requested_yaw
        if any(math.hypot(vx, vy) > STILL_LINEAR or wz * direction > STILL_YAW
               for vx, vy, wz in (motion, state.commanded)):
            if (has_sweep and config.local_controller == "dwa" and not command_only and yaw
                    and state.missing_frames + 1 >= config.lost_frames
                    and _check_dwa_turn(depth, config, state, geometry, None, "search", (0., 0., yaw))):
                state.last_seen_side = direction
                state.search_sweep_checked = True
                return None
            return "原地搜索：等待平移或反向转动停稳"
        if has_sweep:
            horizon = (config.watchdog_ms + config.depth_age_limit_ms + 50) / 1000
            speed = max(math.hypot(*motion[:2]), math.hypot(*state.commanded[:2]))
            slack = speed * horizon
            age = max(0, scan.control_ms - scan.near_revision) / 1000
            blocked = not scan.clear_sweep(geometry, config.half_width_m, slack)
            if not blocked and scan.near_hits is not None and age:
                blocked = not scan.clear_sweep(geometry, config.half_width_m, slack + speed * age,
                                               occupied=scan.near_hits)
            if blocked and not (config.local_controller == "dwa" and not command_only and yaw
                                and state.missing_frames + 1 >= config.lost_frames
                                and _check_dwa_turn(depth, config, state, geometry, None, "search", (0., 0., yaw))):
                state.recovery = True
                return "原地搜索：转身扫掠范围有障碍，停车"
            state.search_sweep_checked = True
        state.last_seen_side = direction
    if state.recovery:
        if not config.local_planning and ahead is not None and ahead < config.approach_check_distance_m:
            return "近障停车，等待前方净空"
        state.recovery = False
    return None


def _local_side_order(rooms, config, state, frame):
    priority = {
        side: (coverage >= config.coverage_min,
               room if room is not None else
               (math.inf if coverage >= config.coverage_min else -math.inf),
               coverage)
        for side, (room, coverage) in rooms.items()
    }
    preferred = max(rooms, key=priority.__getitem__)
    if state is None or state.escape_side not in rooms:
        return preferred, -preferred

    incumbent = state.escape_side
    challenger = -incumbent
    measured, room, _ = priority[challenger]
    current_measured, current_room, _ = priority[incumbent]
    better = measured and (not current_measured or room > current_room + SIDE_SWITCH_ADVANTAGE_M)
    if not better or type(frame) is not int:
        state.side_better_frames = 0
    if type(frame) is int and frame != state.side_compare_frame:
        previous = state.side_compare_frame
        if previous is None or not 0 < frame - previous <= config.depth_age_limit_ms:
            state.side_better_frames = 0
        state.side_compare_frame = frame
        state.side_better_frames += int(better)
    if state.side_better_frames >= SIDE_SWITCH_FRAMES:
        return challenger, incumbent
    return incumbent, challenger


def _scan_yaw_rate(config, side=None):
    limit = (config.scan_wz_limit if side is None else
             config.right_wz_limit if side > 0. else config.left_wz_limit)
    return config.wz_max if limit is None else min(config.wz_max, limit)


def _local_velocity_limits(config, side):
    floor_x = config.floor_vx if config.floor_vx_moving is None else config.floor_vx_moving
    floor_y = config.floor_vy if config.floor_vy_moving is None else config.floor_vy_moving
    limit_y = config.vy_max if config.use_lateral else 0.
    directional_limit = config.right_vy_limit if side > 0 else config.left_vy_limit
    if directional_limit is not None:
        limit_y = min(limit_y, directional_limit)
    return (floor_x, config.vx_max), (floor_y, limit_y)


def _camera_frame_routes(config, scan):
    return (scan is not None and config.scan_geometry is None and scan.receipt_only
            and scan.buffered_odom and scan.pose_origin is None and not scan.command_only)



def _checked_motion_rejection(scan, geometry, config, state, motion, command, grid, horizon,
                              *, room=math.inf, points=(), shadow=None, bounded=False,
                              bounds_horizon=None, room_horizon=None, near_motion=None):
    """Check the legacy prefix envelope and its straight-segment stopping room."""
    velocities = motion, state.commanded, command
    rejection = local_path.motion_limits_rejection(
        geometry, config.half_width_m, horizon, velocities, room=room, bounded=bounded,
        room_horizon=room_horizon, bounds_horizon=bounds_horizon)
    if rejection is not None:
        return rejection
    if not local_path.motion_sweep_clear(grid, geometry, config.half_width_m, horizon, velocities):
        return "collision"
    if scan.near_hits is not None and scan.near_motion_hits(
            geometry, config.half_width_m, horizon,
            (motion if near_motion is None else near_motion, state.commanded, command)).any():
        return "remembered_obstacle"
    if shadow is not None and points and local_path.visible_prefix(points, shadow) < len(points):
        return "occluded_prefix"
    return None


def _dwa_unavailable_reason(depth, config, state, *, branch=None, recovery=False):
    scan = state.active_scan
    if state.motion_source not in ("pose", "measured") or scan.command_only:
        return "no_feedback"
    if (depth is None or depth.get("capture_invalid") or not scan.poses or not scan.views
            or not scan.current_frame or not scan.matches_route_view(depth)):
        return "stale_observation"
    if config.scan_geometry is None and config.local_geometry is None:
        return "geometry_mismatch"
    if scan.active and not recovery and not (branch == "advance" and scan.ready):
        return "scan_active"
    return None


@dataclass(frozen=True)
class DwaSceneSources:
    """A single reading of the physical sources shared by planning and publication."""

    views: tuple
    history: tuple
    near: NearProjection
    motion_frame: object
    pose_origin: PlanarPose | None
    pose_registration: tuple[PlanarPose, PlanarPose] | None
    poses: tuple
    max_feedback_gap_ms: int
    receipt_only: bool
    stamp: int
    current_frame: int
    revision: tuple


def _cell_revision(cells):
    if isinstance(cells, local_path.ObstacleSquares):
        return cells.yaw, cells.alignment_error, local_path.array_revision(cells.centers)
    return local_path.array_revision(cells)


def dwa_scene_sources(scan, now_ms, odom=None):
    if odom is not None and scan.pose_registration is None:
        # The command timer can receive the first absolute fix between policy
        # ticks. Retain its measured pairing before freezing the scene revision.
        scan.pose_registration = dwa_pose_reference(scan, odom)
    near = scan.near_projection
    frame, origin, pose = scan.motion_frame, scan.pose_origin, scan.pose
    poses = tuple(scan.poses)
    gap = scan.max_feedback_gap_ms
    stamp = poses[-1][0] if poses else scan.control_ms or now_ms
    current, receipt = scan.current_frame, scan.receipt_only
    views = tuple(v for v in (*scan.registered_history, *tuple(scan.views)) if v[3] is not None
                  and now_ms - (v[5][0] if len(v) > 5 else v[0]) <= MAP_AGE_MS)
    history = scan.historical_layers
    view_keys = tuple((v[0], v[1], local_path.array_revision(v[3]),
                       local_path.array_revision(v[4] if len(v) > 4 else v[2]), v[5:]) for v in views)
    reference = scan.pose_registration
    revision = (frame, origin, poses, gap, pose, stamp, current, receipt, near.stamp_ms, near.origin,
                _cell_revision(near.squares if near.squares is not None else near.hits),
                tuple((p, _cell_revision(cells)) for p, cells in history), reference, view_keys)
    return DwaSceneSources(views, history, near, frame, origin, reference, poses, gap, receipt, stamp, current, revision)


def dwa_memory_revision(scan, now_ms=None):
    return dwa_scene_sources(scan, now_ms if now_ms is not None else scan.control_ms).revision


def dwa_target_revision(state):
    anchor, track = state.goal_anchor, state.tracker.track
    return ((anchor.x, anchor.y, anchor.pose, anchor.target_missing, anchor.generation, anchor.frame) if anchor else None,
            (track.state, track.hits, track.front_required, track.front_frame_id,
             tuple(track.front_box or ()), local_path.array_revision(track.front_fix),
             local_path.array_revision(track.front_reference), local_path.array_revision(track.last_obs),
             track.measured_range_m, track.measured_point, track.generation,
             track.source, track.latest_source, track.image_region, track.frame) if track else None)


class DwaFrameMismatch(ValueError):
    """Measured poses belong to different odometry frames."""


def _advance_dwa_pose(pose, stamp, odom, max_gap):
    """Integrate the contiguous measured prefix after this pose, up to stamp."""
    last_source = None
    for received, source, twist in odom.get("velocity_history", ()):
        if received <= pose.stamp_ms:
            last_source = source
            continue
        if received > stamp or twist is None or received - pose.stamp_ms > max_gap:
            break
        if source is not None and last_source is not None:
            if source == last_source:
                continue
            if not 0 < source - last_source <= max_gap:
                break
        dt = (received - pose.stamp_ms) / 1000
        vx, vy, wz = twist
        mid = pose.yaw + .5 * wz * dt
        pose = PlanarPose(received, pose.frame,
                          pose.x + dt * (vx * math.cos(mid) - vy * math.sin(mid)),
                          pose.y + dt * (vx * math.sin(mid) + vy * math.cos(mid)), pose.yaw + wz * dt)
        last_source = source
    return pose


def dwa_pose_reference(scan, odom):
    """Pair an absolute fix with the same continuous measured relative pivot."""
    previous = scan.pose_registration
    if previous is not None:
        # This is one rigid frame relation. A later world-frame reset cannot
        # re-label the relative history, and a repeated fix cannot rewrite it.
        return previous
    odom = odom or {}
    anchor = odom.get("obstacle_pose") or odom.get("motion_pose")
    if not scan.receipt_only or scan.pose_origin is not None or not scan.poses or anchor is None:
        return previous
    coords = relative_pose_at(scan.poses, anchor.stamp_ms)
    frame = ("navi_relative", scan.motion_frame)
    if coords is not None:
        relative = PlanarPose(anchor.stamp_ms, frame, coords[0], -coords[1], -coords[2])
    elif anchor.stamp_ms > scan.poses[-1][0]:
        stamp, (f, r, yaw), _ = scan.poses[-1]
        relative = _advance_dwa_pose(PlanarPose(stamp, frame, f, -r, -yaw), anchor.stamp_ms,
                                     odom, scan.max_feedback_gap_ms)
        if relative.stamp_ms != anchor.stamp_ms:
            return previous
    else:
        return previous
    return relative, anchor


def _registered_dwa_pose(pose, reference):
    if pose is None or reference is None or pose.frame != reference[0].frame:
        return pose
    relative, absolute = reference
    f, left, yaw = pose.relative_to(relative)
    c, s = math.cos(absolute.yaw), math.sin(absolute.yaw)
    return PlanarPose(pose.stamp_ms, absolute.frame, absolute.x + c * f - s * left,
                      absolute.y + s * f + c * left, absolute.yaw + yaw)


def dwa_body_pose(scan, stamp, odom, *, capture=None):
    """Locate the measured body pivot without applying the camera-front offset."""
    odom = odom or {}
    pose = capture or odom.get("obstacle_pose") or odom.get("motion_pose")
    motion_pose = odom.get("motion_pose")
    if (capture is None and pose is not None and motion_pose is not None and motion_pose.frame == pose.frame
            and abs(stamp - motion_pose.stamp_ms) < abs(stamp - pose.stamp_ms)):
        pose = motion_pose
    if pose is None and scan.poses:
        covered = min(max(stamp, scan.poses[0][0]), scan.poses[-1][0])
        coords = relative_pose_at(scan.poses, covered)
        if scan.pose_origin is None:
            f, r, yaw = coords
            pose = PlanarPose(covered, ("navi_relative", scan.motion_frame), f, -r, -yaw)
        else:
            origin = scan.pose_origin
            f, r = PathLeg.world_point((origin.x, -origin.y, -origin.yaw), coords[:2])
            pose = PlanarPose(covered, origin.frame, f, -r, origin.yaw - coords[2])
    if pose is None:
        return None
    pose = _registered_dwa_pose(pose, dwa_pose_reference(scan, odom))
    if pose.stamp_ms < stamp:
        pose = _advance_dwa_pose(pose, stamp, odom, scan.max_feedback_gap_ms)
    return pose


def dwa_grid_pose(scan, geometry, stamp, odom, *, capture=None):
    """Locate the grid origin and retain the time actually covered by feedback."""
    pose = dwa_body_pose(scan, stamp, odom, capture=capture)
    return ActiveScan._footprint_origin(pose, geometry) if pose is not None else None


def dwa_registration(source, current, source_ms, now_ms, geometry, velocities, *, reference=None):
    source = _registered_dwa_pose(source, reference)
    current = _registered_dwa_pose(current, reference)
    f = left = yaw = 0.
    source_gap, current_gap = abs(now_ms - source_ms) / 1000, 0.
    if source is not None and current is not None:
        if source.frame != current.frame:
            raise DwaFrameMismatch
        f, left, yaw = source.relative_to(current)
        # Between distinct measurements, only the two endpoints remain uncovered.
        if source != current:
            source_gap = abs(source_ms - source.stamp_ms) / 1000
            current_gap = abs(now_ms - current.stamp_ms) / 1000
    elapsed = source_gap + current_gap
    yaw_rate = max(abs(v[2]) for v in velocities)
    source_turn, current_turn = source_gap * yaw_rate, current_gap * yaw_rate
    travel = elapsed * max(math.hypot(*v[:2]) for v in velocities)
    # Current-end rotation acts on the known displacement between grid origins
    # as well as on each cell's source-relative radius (handled by GridTransform).
    shift_error = travel + math.hypot(f, left) * min(2., current_turn)
    if geometry.front == 0.:
        shift_error += geometry.half_length * (min(2., source_turn) + min(2., current_turn))
    return local_path.GridTransform(f, -left, -yaw, shift_error, source_turn + current_turn)


def _rolling_origin(sources, view, geometry):
    stamp, (f, r, yaw) = view[:2]
    if len(view) > 5:
        return view[5]
    capture_ms, capture = stamp, None
    if capture is None and not sources.receipt_only:
        origin = sources.pose_origin
        if origin is None:
            capture = PlanarPose(stamp, ("navi_relative", sources.motion_frame), f, -r, -yaw)
        else:
            x, right = PathLeg.world_point((origin.x, -origin.y, -origin.yaw), (f, r))
            capture = PlanarPose(stamp, origin.frame, x, -right, origin.yaw - yaw)
    return capture_ms, ActiveScan._footprint_origin(capture, geometry) if capture is not None else None


def _retire_confirmed(layer, origin, clear, occupied, reference=None):
    stamp, before, squares = layer
    original = before
    before = _registered_dwa_pose(before, reference)
    origin = _registered_dwa_pose(origin, reference)
    if before is None or origin is None or before.frame != origin.frame:
        return layer
    forward, left, yaw = before.relative_to(origin)
    tf = local_path.GridTransform(forward, -left, -yaw)
    # A newer exposure must cover every grid cell intersecting the original
    # square in its registered orientation, including any source rotation.
    indices = np.round(tf.points(squares.centers) / local_path.RESOLUTION + local_path.START, 7)
    retire = confirmed_square_coverage(indices, squares.yaw + tf.yaw, clear, occupied)
    if not retire.any():
        return layer
    if retire.all():
        return None
    centers = squares.centers[~retire]
    centers.setflags(write=False)
    return stamp, original, local_path.ObstacleSquares(centers, squares.yaw, squares.alignment_error)


def _rolling_layers(scan, sources, geometry):
    keys = sources.revision[-1]
    axes = (sources.motion_frame, sources.pose_origin, sources.receipt_only, geometry.front, geometry.rear,
            sources.pose_registration)
    cache = scan._physical_views
    layers, start = [], 0
    if cache is not None and cache[0] == axes and keys:
        old_keys, old_layers = cache[1:]
        # Window eviction always removes a prefix. Changed pixels or clearing
        # evidence under a repeated stamp invalidate the affected history.
        for offset, key in enumerate(old_keys):
            count = len(old_keys) - offset
            if key == keys[0] and old_keys[offset:] == keys[:count]:
                oldest = sources.views[0]
                cutoff = oldest[5][0] if len(oldest) > 5 else oldest[0]
                layers = [layer for layer in old_layers if layer[0] >= cutoff]
                start = count
                break
    for view in sources.views[start:]:
        stamp, origin = _rolling_origin(sources, view, geometry)
        clear = view[4] if len(view) > 4 else view[2]
        if layers and clear.any():
            layers = [remaining for layer in layers
                      if (remaining := _retire_confirmed(layer, origin, clear, view[3],
                                                         sources.pose_registration)) is not None]
        cells = local_path.occupied_squares(view[3])
        if cells is not None and len(cells.centers):
            cells.centers.setflags(write=False)
            layers.append((stamp, origin, cells))
    frozen = tuple(layers)
    scan._physical_views = (axes, keys, frozen)
    return frozen


def dwa_remembered(sources, registration):
    historical = tuple(registration(pose.stamp_ms if pose else sources.stamp, grid_pose=pose).squares(cells)
                       for pose, cells in sources.history)
    near = sources.near
    cells = near.squares if near.squares is not None else near.hits
    if cells is None:
        return historical
    stamp = near.origin.stamp_ms if near.origin is not None else near.stamp_ms or sources.stamp
    return (*historical, registration(stamp, grid_pose=near.origin).squares(cells))


def dwa_obstacles(scan, sources, geometry, registration):
    observed = tuple(registration(stamp, grid_pose=pose).squares(cells)
                     for stamp, pose, cells in _rolling_layers(scan, sources, geometry))
    return observed, dwa_remembered(sources, registration)


def dwa_hold(config, admission_s):
    return max(config.dwa_horizon_s, (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000,
               admission_s + (config.watchdog_ms + 50) / 1000)


def dwa_measured_target(track, geometry, now_ms, depth_age_ms, registration, *, track_ms):
    """Register a physical constraint without refreshing the observation that supplied it."""
    fix = track.measured_point
    if fix is None:
        return None
    if fix.observation_ms is None:
        if track.measured_range_m is None:
            return None
    elif not 0 <= now_ms - fix.observation_ms < depth_age_ms:
        return None
    point, frame = fix.standoff_point or fix.point, fix.frame
    stamp, pose = fix.capture_ms or track_ms, fix.capture_pose
    if track.front_required and track.front_fix is not None and track.measured_range_m is None:
        # The tracker already transported this coasting front to its tick frame.
        # Its original capture still limits the lifetime above.
        surface = track.front_reference if fix.standoff_point is not None else track.front_fix
        point = min((surface, track.front_fix), key=lambda p: math.hypot(*p))
        frame, stamp, pose = track.frame, track_ms, None
    point = (frame.grid_point(point, geometry) if frame is not None else
             local_path.target_in_grid(geometry, point[0], -point[1]))
    view = registration(stamp, pose)
    return tuple(view.points(point)), view.error(math.hypot(*point))


def _dwa_motion_inputs(depth, config, state, geometry, target, motion, *, corner_only=False):
    scan = state.active_scan
    mode = "body" if config.scan_geometry is not None else "camera_approx"
    now_ms = depth.get("control_ms") or scan.control_ms or scan.current_frame
    admission_ms = max(0, min(state.dwa_admission_ms, scan.current_frame + config.depth_age_limit_ms - now_ms))
    admission = admission_ms / 1000
    horizon = dwa_hold(config, admission)
    if horizon > 3.:
        return None, "prediction_horizon"
    odom = state.dwa_odom
    previous = state.commanded
    velocities = (motion, previous, *(row[2] for row in (state.dwa_odom or {}).get("velocity_history", ())
                                      if row[2] is not None))
    # Materialize a pending full view before requesting its uncapped occupancy.
    grid = scan.observed_grid()
    raw = (state.dwa_projection.grid_for(geometry, require_valid_depth=True, check_free_space=False)
           if state.dwa_projection is not None else None)
    if state.dwa_projection is not None and raw is None:
        return None, "stale_observation"

    sources = dwa_scene_sources(scan, now_ms, odom)
    current = dwa_grid_pose(sources, geometry, now_ms, odom)
    reference = dwa_pose_reference(sources, odom)

    def registration(stamp, capture=None, *, grid_pose=None):
        pose = grid_pose or dwa_grid_pose(sources, geometry, stamp, odom, capture=capture)
        return dwa_registration(pose, current, stamp, now_ms, geometry, velocities, reference=reference)

    try:
        observed, remembered = (dwa_obstacles(scan, sources, geometry, registration)
                                if config.use_obstacle_history else ((), ()))
        if raw is not None or not config.use_obstacle_history:
            capture = depth.get("capture_ms") or sources.current_frame
            cells = (local_path.front_corner_sweep_grid(
                state.dwa_projection.frame, geometry, _keepout_m(config), raw[1],
                guard_blind_sides=not config.use_obstacle_history) if raw is not None else grid[1])
            observed += (registration(capture, depth.get("obstacle_capture_pose")).squares(cells),)
    except DwaFrameMismatch:
        return None, "geometry_mismatch"
    footprint = motion_model.FootprintSpec(geometry.front, geometry.rear, config.half_width_m,
                                           config.clearance_margin_m, mode)
    track = state.tracker.track
    association = track.snapshot(now_ms) if not corner_only and current_target_observed(state) else None
    association_pose = (target_body_pose(now_ms, state.dwa_odom, motion, depth.get("obstacle_capture_pose"))
                        if association else None)
    measured_target = None
    anchor = state.goal_anchor
    if not corner_only and track is not None and (anchor is None or anchor.generation == track.generation):
        try:
            measured_target = dwa_measured_target(track, geometry, now_ms, config.depth_age_limit_ms,
                                                  registration, track_ms=now_ms)
        except DwaFrameMismatch:
            return None, "geometry_mismatch"
    inputs = DwaMotionInputs(now_ms, now_ms + admission_ms, horizon, footprint, target, 0.,
                             motion, previous, current, observed, remembered,
                             sources.revision, dwa_target_revision(state),
                             track.generation if track is not None else None, association, association_pose, measured_target,
                             bool(track is not None and track.front_required), state.target)
    return inputs, grid


def _dwa_context(inputs, geometry, config, motion_frame):
    return dwa.MotionContext(
        inputs.measured, inputs.previous, inputs.prediction_horizon_s,
        (inputs.deadline_ms - inputs.checked_ms) / 1000, geometry, inputs.footprint,
        inputs.observed, inputs.remembered, local_path.SUPPORTED_REGION, inputs.goal,
        config.stop_distance_m, acceleration=config.motion_acceleration, braking=config.motion_braking,
        frame=(motion_frame, inputs.pose), measured_target=inputs.measured_target)


def _check_dwa_turn(depth, config, state, geometry, target, branch, command):
    if ((target is None and branch != "search")
            or _dwa_unavailable_reason(depth, config, state, recovery=True)):
        return False
    scan = state.active_scan
    inputs, _ = _dwa_motion_inputs(depth, config, state, geometry, target, scan.motion)
    if inputs is None:
        return False
    context = _dwa_context(inputs, geometry, config, scan.motion_frame)
    outcomes = state.dwa_budget.for_context(context) if state.dwa_budget is not None else {}
    if command in outcomes:
        prediction, check = outcomes[command]
        if check is None:
            check = context.evaluate(prediction)
            outcomes[command] = prediction, check
            state.dwa_trace["angular_checks"] += 1
        else:
            state.dwa_trace["angular_cache_hits"] += 1
    else:
        prediction, check = outcomes[command] = context.check(command)
        state.dwa_trace["angular_checks"] += 1
    if not check.accepted:
        state.dwa_trace["fallback_reason"] = check.reason
        return False
    guard = DwaMotionGuard(**vars(inputs), trace=prediction, check=check)
    state.dwa_proposal = DwaTurn(command, guard)
    state.dwa_motion_guard = guard
    state.dwa_selected = state.checked_prefix = state.avoidance_yaw_override = True
    state.dwa_trace.update(branch=branch, eligible=True, geometry_mode=inputs.footprint.mode,
                           proposed_twist=command, guard_horizon_s=prediction.duration)
    return True


def dwa_trace_visible(poses, shadow, view=None, *, pad_cell=False):
    if not shadow.any():
        return True
    if pad_cell:
        rows, cols = shadow.shape
        padded = np.pad(shadow, 1)
        shadow = np.logical_or.reduce([padded[r:r + rows, c:c + cols] for r in range(3) for c in range(3)])
    points = np.asarray(poses)[:, :2]
    if view is not None:
        points = view.source_points(points)
    return local_path.visible_prefix(points, shadow) == len(points)


def _blocking_memory_view(scan, context, budget):
    view = scan.memory_recheck()
    near = scan.near_projection.squares
    if view is None or near is None or not context.remembered:
        return None
    # Registration preserves the order of the cells in the near projection.
    point = (view["forward_m"], view["right_m"])
    match = np.flatnonzero(np.max(np.abs(near.centers - point), axis=1) < 1e-7)
    if not len(match):
        return None
    registered = context.remembered[-1]
    cell = replace(registered, centers=registered.centers[match[:1]])
    clear_scene = replace(context, remembered=())
    one_cell = replace(context, remembered=(cell,))
    for prediction, check in budget.for_context(context).values():
        if check is not None and check.reason == "remembered_obstacle":
            if clear_scene.evaluate(prediction).accepted and one_cell.evaluate(prediction).reason == check.reason:
                return view
            # Bound the extra recovery work to one checked trajectory per reference group.
            break
    return None


def _visual_approach(depth, config, state, geometry, track, feedback, objects_ms, detections):
    hold = Decision(_twist(0., 0., 0.), ALIGNING, "已看到人，等待可验证的接近空间")
    region = track.image_region
    if (not depth or depth.get("map") is None or not region or geometry is None
            or feedback is None or not objects_ms or _dwa_unavailable_reason(depth, config, state)):
        return hold
    if not track.accepted_detection and not (
            track.source is not None and track.source == observation_source(detections)
            and track.since_obs_s * 1000 < config.max_obs_age_ms
            and any((tuple(obj.get("position") or ()), tuple(obj.get("bbox_norm") or ())) == region
                    for obj in target_objects(detections, state.target, config))):
        return hold
    bearing = float(region[0][0]) * config.half_fov_rad if region[0] else 0.
    lookahead = config.approach_check_distance_m
    point = local_path.target_in_grid(geometry, lookahead * math.cos(bearing), lookahead * math.sin(bearing))
    route = Detour(None, ((0., 0.), point), geometry)
    side = math.copysign(1., bearing)
    heading = -_clamp(bearing, _scan_yaw_rate(config, side))
    scan = state.active_scan
    grid = scan.grid()
    ahead = _target_path_obstacle(grid[1], geometry, config, point, 0.)
    seeds = tuple(_prefix_commands(config, config.vx_max, heading, (side, -side) if ahead is not None else (), False))
    proposal = None
    if ahead is not None and ahead < lookahead:
        observation = dict(depth, projected_grid=grid, scan_revision=scan.poses[-1][0],
                           body_half_width=config.half_width_m, allow_unknown=True,
                           preview_occluded=True, finite_turn=True)
        routes = []
        budget = state.dwa_budget
        for direction in (side, -side):
            if budget is not None and budget.expired():
                break
            candidate = state.local_plan.route(observation, geometry, _keepout_m(config), ahead, direction,
                                               config.depth_age_limit_ms, config.watchdog_ms)
            if candidate is not None:
                routes.append(candidate)
            if routes and (direction == -side or (budget is not None and budget.short_window())):
                proposal = _dwa_detour(depth, config, state, routes[0], None, None, branch="visual_probe",
                                       motion=feedback, candidate_commands=seeds, alternates=routes[1:], guidance=True)
                if proposal is not None:
                    break
                routes.clear()
    if proposal is None:
        proposal = _dwa_detour(depth, config, state, route, None, None, branch="visual_probe",
                               motion=feedback, candidate_commands=seeds, guidance=True)
    if proposal is None:
        return hold
    turn = proposal.command[2]
    vx, vy, yaw, _, _ = _use_dwa(state, proposal, math.copysign(1., -turn) if turn else 0.)
    return Decision(_twist(vx, vy, yaw), APPROACHING, "沿目标图像方向接近，等待人体测距")


def _dwa_detour(depth, config, state, route, target_distance, target_bearing, *,
                branch="route", motion=None, speed_limit=None,
                candidate_commands=(), shadow=None, recovery=False, alternates=(), guidance=False,
                follow_published=False, route_error=0., corner_only=False):
    """Select a finite swept footprint on a local navigation reference."""
    if config.local_controller != "dwa":
        return None
    scan, trace = state.active_scan, state.dwa_trace
    trace.setdefault("select_calls", 0)
    trace.setdefault("rejected", {})
    trace.setdefault("attempts", [])

    def unavailable(reason):
        trace["fallback_reason"] = reason
        return None

    reason = _dwa_unavailable_reason(depth, config, state, branch=branch, recovery=recovery)
    if reason is not None:
        return unavailable(reason)
    geometry = config.scan_geometry or config.local_geometry
    routes = (route, *alternates)
    if geometry is None or any(ref is not None and ref.geometry is not geometry for ref in routes):
        return unavailable("geometry_mismatch")
    mode = "body" if config.scan_geometry is not None else "camera_approx"
    trace.update(eligible=True, geometry_mode=mode, observation_id=scan.current_frame)
    visual = branch == "visual_probe"
    if (route is None or len(route.points) < 2
            or (not (visual or corner_only) and (target_distance is None or target_bearing is None))):
        return unavailable("no_reference")
    limit = min(config.dwa_samples, dwa.MAX_SEARCH_CANDIDATES)
    if state.dwa_budget is None:
        state.dwa_budget = dwa.CandidateBudget(limit, predictions=state.dwa_predictions)
    if state.dwa_budget.expired():
        return unavailable("search_deadline")
    target = (None if visual or corner_only else local_path.target_in_grid(geometry, target_distance * math.cos(target_bearing),
                                                            target_distance * math.sin(target_bearing)))
    (floor_x, limit_x), _ = _local_velocity_limits(config, 1.)
    floor_y, left_vy = _local_velocity_limits(config, -1.)[1]
    right_vy = _local_velocity_limits(config, 1.)[1][1]
    if speed_limit is not None:
        limit_x = min(limit_x, speed_limit)
    motion = scan.motion if motion is None else motion
    inputs, grid = _dwa_motion_inputs(depth, config, state, geometry, target, motion, corner_only=corner_only)
    if inputs is None:
        return unavailable(grid)
    if corner_only:
        inputs = replace(inputs, front_required=False, generation=state.published_dwa_route.motion_guard.generation)
    if follow_published:
        reachable_vx = max(0., inputs.measured[0]) + config.motion_acceleration[0] * config.dwa_horizon_s
        limit_x = min(limit_x, reachable_vx)
    if shadow is None and scan.receipt_only and not config.use_lateral and not visual:
        shadow = scan.occlusion(geometry)

    # The shadow belongs to one exposure. Express a predicted path in its
    # camera/grid coordinates even when feedback has since moved the pivot.
    view = None
    if shadow is not None and shadow.any():
        stamp, origin = _rolling_origin(scan, scan.views[-1], geometry)
        origin = origin or dwa_grid_pose(scan, geometry, stamp, state.dwa_odom)
        try:
            view = dwa_registration(origin, inputs.pose, stamp, inputs.checked_ms,
                                    geometry, (inputs.measured, inputs.previous),
                                    reference=dwa_pose_reference(scan, state.dwa_odom))
        except DwaFrameMismatch:
            return unavailable("geometry_mismatch")

    def visible(command, trajectory, duration_s):
        clear = dwa_trace_visible(trajectory, shadow, view)
        return dwa.CandidateCheck(clear, None if clear else "occluded_prefix")

    diagnostics = {}
    trace["select_calls"] += 1
    proposed = dwa.select(
        route=route.points, grid=grid, geometry=geometry, footprint=inputs.footprint,
        occupied=inputs.observed, remembered=inputs.remembered, supported_region=local_path.SUPPORTED_REGION,
        latency_s=(inputs.deadline_ms - inputs.checked_ms) / 1000,
        measured=inputs.measured, previous=inputs.previous, x_limit=(0., max(0., limit_x)),
        y_limit=(0., 0.) if branch == "direct" else (-right_vy, left_vy),
        yaw_limit=(0., 0.) if branch == "direct" else (-_scan_yaw_rate(config, 1.), _scan_yaw_rate(config, -1.)),
        floors=(floor_x, floor_y, config.floor_wz, config.floor_wz_moving),
        target=target, stop_distance=config.stop_distance_m,
        acceleration=config.motion_acceleration, braking=config.motion_braking,
        horizon=inputs.prediction_horizon_s, max_candidates=limit,
        candidate_commands=candidate_commands, validate_candidate=visible if shadow is not None else None,
        diagnostics=diagnostics, alternate_routes=tuple(ref.points for ref in alternates),
        candidate_budget=state.dwa_budget, motion_frame=(scan.motion_frame, inputs.pose), guidance=guidance,
        measured_target=inputs.measured_target, follow_published=follow_published, route_error=route_error,
        handoff_yaw_step=state.dwa_handoff_yaw_step if not visual else 0.,
        retry_from_rest=(recovery and scan.dwa_failures > 0 and not any(inputs.previous)
                         and math.hypot(*inputs.measured[:2]) <= config.scan_odom_linear_tolerance_mps))
    for key in ("sampled_candidates", "cached_candidates", "considered_candidates", "physical_evaluations", "bound_skips", "response_reuse",
                "admissible_candidates", "reference_scores"):
        trace[key] = trace.get(key, 0) + diagnostics[key]
    for key, value in diagnostics.get("rejected", {}).items():
        trace["rejected"][key] = trace["rejected"].get(key, 0) + value
    trace["attempts"].append(dict(branch=branch, geometry_mode=mode, **diagnostics))
    if proposed is None:
        if (not follow_published and scan.dwa_failures >= 2 and not scan.attempted and "memory_recheck" not in trace
                and diagnostics["rejected"].get("remembered_obstacle")):
            context = _dwa_context(inputs, geometry, config, scan.motion_frame)
            view = _blocking_memory_view(scan, context, state.dwa_budget)
            if view is not None:
                trace["memory_recheck"] = view
        if (diagnostics["reason"] == "no_admissible_candidates"
                and diagnostics["rejected"].get("remembered_obstacle")):
            # Distinguish a registered footprint overlap at rest from a
            # collision that only a candidate's motion or braking would cause.
            footprint = inputs.footprint
            margin = footprint.clearance_margin_m
            start = ((0., 0., 0.),)
            if any(local_path.rectangle_clearance(
                    start, layer, footprint.front + margin, footprint.rear + margin,
                    footprint.half_width + margin)[1] == 0
                   for layer in inputs.remembered):
                return unavailable("initial_remembered_overlap")
        return unavailable(diagnostics["reason"])
    guard = DwaMotionGuard(**vars(inputs), trace=proposed.trace, check=proposed.check,
                           visibility_required=shadow is not None,
                           probe_region=state.tracker.track.image_region if visual else None,
                           corner_only=corner_only, route_error=route_error if route.corridor_corner else 0.)
    return DwaProposal(proposed, routes[proposed.reference_index], branch, guard)


def _use_dwa(state, proposal, side):
    """Keep the exact checked twist and retire any superseded fixed straight leg."""
    scan = state.active_scan
    if scan.active or scan.advance_m is not None or scan.attempted or scan.exhausted or scan.result_frame:
        scan.retire_motion()
    scan.dwa_failures = 0
    state.dwa_selected = True
    state.dwa_proposal = proposal
    state.dwa_motion_guard = proposal.guard
    choice, route = proposal.choice, proposal.reference
    state.dwa_trace.update(branch=proposal.branch, fallback_reason=None, proposed_twist=choice.command,
                           guard_horizon_s=choice.trace.duration, reference_index=choice.reference_index,
                           reference=route.points, score_components=vars(choice.components))
    if route.topology:
        state.dwa_trace.update(topology_reference=route.topology, topology_frame="current_grid",
                               reference_clipped=bool(route.preview))
    state.checked_prefix = True
    state.avoidance_yaw_override = True
    state.escape_side = side
    state.recovery = False
    state.detour_leg = state.executing_leg = None
    points = tuple((f, r) for f, r, _ in choice.trajectory)
    state.view_detour = Detour(None, points, route.geometry, route.points)
    if proposal.branch == "direct":
        return *proposal.command, APPROACHING, "沿目标方向安全直行"
    return (*proposal.command, AVOIDING, "局部规划：DWA沿已验证绕行轨迹运动")


def _follow_published_route(depth, config, state, geometry, distance, bearing, *,
                            foreground, corner_only=False, approach_speed=None):
    previous = state.published_dwa_route
    if _dwa_unavailable_reason(depth, config, state):
        return None
    scan, guard = state.active_scan, previous.motion_guard
    track = state.tracker.track
    now = depth.get("control_ms") or scan.control_ms or scan.current_frame
    current = dwa_grid_pose(scan, geometry, now, state.dwa_odom)
    generation_matches = corner_only or guard.generation == (track.generation if track is not None else None)
    if (guard.target_name != state.target or not generation_matches
            or current is None or not 0 <= now - current.stamp_ms < config.max_feedback_gap_ms
            or (guard.footprint.front, guard.footprint.rear) != (geometry.front, geometry.rear)):
        return None
    try:
        transform = dwa_registration(guard.pose, current, guard.checked_ms, now, geometry,
                                     (scan.motion, state.commanded), reference=dwa_pose_reference(scan, state.dwa_odom))
    except DwaFrameMismatch:
        return None
    exit_line = (tuple(map(tuple, transform.points(previous.passage_exit)))
                 if previous.passage_exit is not None else None)
    # In single-view mode the obstacle can leave the front camera before
    # the chassis clears it. Pursue the route using a fresh sweep.
    if config.use_obstacle_history and not foreground and not local_path.passage_ahead(exit_line):
        return None
    route = Detour(None, tuple(map(tuple, transform.points(previous.route_points))), geometry,
                   passage_exit=exit_line, corridor_corner=previous.corridor_corner)
    # Once the target enters the full-speed stopping envelope, carry the
    # normal approach speed into the route and its publisher-side pursuit.
    speed_limit = None
    if approach_speed and guard.measured_target is not None:
        point, error = guard.measured_target
        (floor, limit), _ = _local_velocity_limits(config, 1.)
        travel = limit * (guard.prediction_horizon_s + limit / (2 * config.motion_braking[0]))
        if math.hypot(*point) - error <= config.stop_distance_m + travel:
            speed_limit = max(floor, approach_speed)
    proposal = _dwa_detour(depth, config, state, route, distance, bearing, speed_limit=speed_limit,
                           follow_published=True, route_error=transform.error(math.hypot(*previous.route_points[-1])),
                           corner_only=corner_only)
    if proposal is None:
        return None
    state.dwa_trace["retained_route"] = True
    return _use_dwa(state, proposal, state.escape_side)


def _route_travel_limits(config):
    (_, forward), (_, right) = _local_velocity_limits(config, 1.)
    _, (_, left) = _local_velocity_limits(config, -1.)
    return (0., forward), (-right, left), (-_scan_yaw_rate(config, 1.), _scan_yaw_rate(config, -1.))


def _route_rank(plan, target, config):
    limits = _route_travel_limits(config)
    duration = local_path.route_time_s(plan.points, *limits)
    residual = math.dist(plan.points[-1], local_path.target_in_grid(plan.geometry, *target))
    speed = math.hypot(limits[0][1], max(-limits[1][0], limits[1][1]))
    remaining_s = max(0., residual - config.stop_distance_m) / speed if speed else math.inf
    return duration + remaining_s


def _dwa_route_reference(plan, visible, target, stop, tolerance):
    """Classify the whole topology and expose only the evidenced part to the selector."""
    points = plan.points
    topology = plan.topology or points
    residual = math.dist(topology[-1], target)
    complete = (not plan.reachable_prefix
                and stop - 1e-9 <= residual <= stop + max(tolerance, local_path.RESOLUTION) + 1e-9)
    ref = Detour(None, points[:visible], plan.geometry, points if visible < len(points) else (),
                 reachable_prefix=not complete, topology=topology, passage_exit=plan.passage_exit,
                 corridor_corner=plan.corridor_corner and visible == len(points))
    return ("route" if complete else "reachable_prefix"), ref


def _start_detour_segment(depth, config, state, geometry, ahead, directions, target_distance, target_bearing,
                          *, reroute=False, rooms=None, closest=False, rolling=None, dwa_only=False):
    """Choose a checked moving prefix or prepare a turn-and-drive segment."""
    floor_x, limit_x = _local_velocity_limits(config, 1.)[0]
    if limit_x <= 0. or limit_x < floor_x:
        return False
    scan = state.active_scan
    grid = scan.grid()
    free, occupied = grid
    observation = dict(depth, projected_grid=grid, body_half_width=config.half_width_m,
                       closest=closest, allow_unknown=True, finite_turn=dwa_only)
    shadow = scan.occlusion(geometry) if scan.receipt_only and not config.use_lateral else None
    if shadow is not None and shadow.any():
        observation["preview_occluded"] = True
    else:
        shadow = None
    if scan.near_hits is not None:
        observation["scan_revision"] = scan.near_revision
    if not scan.receipt_only:
        observation.update(scan_grid=grid, scan_revision=scan.poses[-1][0])
    target, optical_target = None, None
    if target_distance is not None and target_bearing is not None:
        forward = target_distance * math.cos(target_bearing)
        right = target_distance * math.sin(target_bearing)
        observation["stop_target"] = (
            forward, right, config.stop_distance_m, config.stop_tolerance_m)
        optical_target = (forward, right)
        target = local_path.target_in_grid(geometry, forward, right)
    keep = _keepout_m(config)
    # Prefer an A* passing lane with enough room for the turn to settle.
    # A tight passage may still have a feasible shorter leg at the normal width.
    plane = ahead * geometry.base_from_optical[0, 2] + geometry.base_from_optical[0, 3]
    turning_room = local_path.RESOLUTION + max(0., plane) * math.tan(ANGLE_TOLERANCE)
    if rooms and not state.escape_side and not reroute:
        left, right = rooms[-1.], rooms[1.]
        if (min(left[1], right[1]) >= config.coverage_min and left[0] is not None and right[0] is not None
                and min(left[0], right[0]) >= config.side_clearance_m
                and abs(left[0] - right[0]) < SIDE_SWITCH_ADVANTAGE_M):
            # A low off-centre post can miss both image-side depth bands.
            # Rank its measured body-grid position before insignificant range differences.
            side = directions[0]
            across = RIGHT[occupied & (abs(FORWARD - plane) <= .10) & (abs(RIGHT) <= keep)]
            if across.size and np.all(side * across > local_path.RESOLUTION / 2):
                directions = tuple(reversed(directions))

    opening = (local_path.opening_route(depth["map"], occupied, geometry, keep, plane, target)
               if not closest or dwa_only else None)
    travel_limits = _route_travel_limits(config)
    detour_opening = (local_path.opening_route(depth["map"], occupied, geometry, keep, plane, target, True, travel_limits)
                      if config.local_controller == "dwa" and (not closest or dwa_only) else None)
    if detour_opening is not None:
        joined = local_path.extend_passage(detour_opening, grid, keep, target, config.stop_distance_m)
        if joined is not detour_opening:
            visible = local_path.visible_prefix(joined.points, shadow) if shadow is not None else len(joined.points)
            if visible > len(detour_opening.points):
                branch, ref = _dwa_route_reference(joined, visible, target, config.stop_distance_m, config.stop_tolerance_m)
                proposal = _dwa_detour(depth, config, state, ref, target_distance, target_bearing,
                                       branch=branch, shadow=shadow, recovery=dwa_only, guidance=True,
                                       follow_published=True)
                if proposal is not None:
                    return _use_dwa(state, proposal, math.copysign(1., detour_opening.points[-1][1]))

    if dwa_only and target is not None and depth.get("ground") is not None:
        previous = ()
        origin = scan.pose[:2]
        reference_start = (0., 0.)
        saved = state.dwa_guidance
        if saved is not None and saved.frame is scan.motion_frame and saved.anchor is state.goal_anchor:
            f, r, yaw = scan.pose
            c, s = math.cos(yaw), math.sin(yaw)
            origin = saved.approach_origin
            reference_start = c*(origin[0]-f) + s*(origin[1]-r), -s*(origin[0]-f) + c*(origin[1]-r)
            previous = tuple((c*(x-f) + s*(y-r), -s*(x-f) + c*(y-r)) for x, y in saved.waypoints)
        guidance = local_path.clearance_route(grid, geometry, keep, target, config.stop_distance_m, reference_start, previous)
        if guidance is not None:
            points = guidance.points
            visible = local_path.visible_prefix(points, shadow) if shadow is not None else len(points)
            while visible > 1 and not local_path.observed_waypoint(free, *points[visible - 1]):
                visible -= 1
            if visible > 1:
                # Score only the visible prefix and check against the real
                # target even when the local topology ends before standoff.
                branch, ref = _dwa_route_reference(
                    guidance, visible, target, config.stop_distance_m, config.stop_tolerance_m)
                proposal = _dwa_detour(depth, config, state, ref, target_distance, target_bearing,
                                       branch=branch, shadow=shadow, recovery=True, guidance=True)
                if proposal is not None:
                    state.dwa_guidance = DwaGuidance(
                        scan.motion_frame, state.goal_anchor,
                        tuple(PathLeg.world_point(scan.pose, point) for point in points), origin)
                    return _use_dwa(state, proposal, math.copysign(1., ref.points[-1][1]))

    def routes():
        if opening is not None:
            yield opening
        for side in directions:
            plan = state.local_plan.route(observation, geometry, keep + turning_room, ahead, side,
                                          config.depth_age_limit_ms, config.watchdog_ms)
            # A wider probe can terminate before a legal slot. Check the
            # adopted width before treating its closest point as the reference.
            if plan is None or (dwa_only and plan.reachable_prefix):
                fitted = state.local_plan.route(observation, geometry, keep, ahead, side,
                                                config.depth_age_limit_ms, config.watchdog_ms)
                if fitted is not None:
                    plan = fitted
            if plan is not None:
                yield plan

    def candidates():
        plans = (sorted(routes(), key=lambda plan: _route_rank(plan, optical_target, config))
                 if closest and optical_target is not None else routes())
        for plan in plans:
            yield plan, True, "reachable_prefix" if plan.reachable_prefix else "route"
        if rolling is None and not config.use_lateral and not scan.command_only and scan.near_hits is not None:
            # A blocked long route can still have a safe forward prefix after
            # turning away from a nearby corner. Check the remembered grid too.
            # Compare both sides before accepting a shorter escape.
            for length in (.65, .35):
                for degrees in (15, 30, 45, 60):
                    for side in directions:
                        angle = math.radians(degrees) * side
                        points = ((0., 0.), (length * math.cos(angle), length * math.sin(angle)))
                        yield local_path.Detour(None, points, geometry), False, "prefix"

    latest_free = scan.views[-1][2]
    have_free_evidence = bool(latest_free.any())

    def evidence_eligible(point, require_observed, preview):
        return not (
            (require_observed and depth.get("ground") is not None and geometry.free_space_validated
             and not local_path.observed_waypoint(free, *point))
            or (preview and have_free_evidence and not local_path.observed_waypoint(latest_free, *point)))

    def visible_plan(plan):
        visible = local_path.visible_prefix(plan.points, shadow) if shadow is not None else len(plan.points)
        return visible, plan.points if visible < len(plan.points) else ()

    route_candidates = candidates()
    if config.local_controller == "dwa":
        route_candidates = tuple(route_candidates)
        dwa_candidates = route_candidates
        if detour_opening is not None and not any(
                plan.points == detour_opening.points for plan, _, _ in route_candidates):
            # Preserve existing reference seeds and the legacy fallback. An
            # offset opening is an extra hypothesis for complete DWA checks.
            dwa_candidates += ((detour_opening, True, "route"),)
        groups = {"route": [], "reachable_prefix": [], "prefix": []}
        for plan, require_observed, branch in dwa_candidates:
            visible, preview = visible_plan(plan)
            # Keep the observed-evidence rules on the bounded reference. The
            # scoring continuation cannot make an unobserved waypoint eligible.
            while visible > 1 and not evidence_eligible(plan.points[visible - 1], require_observed, preview):
                visible -= 1
            if visible > 1:
                if require_observed and target is not None:
                    branch, ref = _dwa_route_reference(
                        plan, visible, target, config.stop_distance_m, config.stop_tolerance_m)
                else:
                    ref = Detour(None, plan.points[:visible], geometry, plan.points if visible < len(plan.points) else ())
                groups[branch].append(ref)
        for branch, references in groups.items():
            if not references:
                continue
            proposal = _dwa_detour(
                depth, config, state, references[0], target_distance, target_bearing,
                branch=branch, motion=rolling[1] if rolling is not None else scan.motion,
                shadow=shadow,
                recovery=dwa_only, alternates=references[1:],
                guidance=branch != "prefix" and not (config.use_lateral and config.vy_max))
            if proposal is not None:
                side = math.copysign(1., proposal.reference.points[-1][1])
                return _use_dwa(state, proposal, side)
        if dwa_only:
            return False

    horizon = (config.watchdog_ms + config.depth_age_limit_ms + 50) / 1000
    blocked = local_path.route_obstacles(grid, geometry, keep, allow_unknown=True)
    turn_occupied = occupied | scan.views[-1][3]
    for plan, require_observed, _ in route_candidates:
        visible, preview = visible_plan(plan)
        for index in range(visible - 1, 0, -1):
            forward, right = plan.points[index]
            # A perpendicular waypoint can be reached by turning before driving.
            if forward < 0. or (forward == 0. and (config.use_lateral or not right)):
                continue
            if not evidence_eligible((forward, right), require_observed, preview):
                continue
            if shadow is not None and index > 1 and local_path.visible_prefix(
                    (plan.points[0], plan.points[index]), shadow) < 2:
                continue
            angle, length = math.atan2(right, forward), math.hypot(forward, right)
            if length < floor_x * horizon:
                continue
            turn_rate = _scan_yaw_rate(config, math.copysign(1., angle))
            if abs(angle) > ANGLE_TOLERANCE and (turn_rate <= 0. or turn_rate < config.floor_wz):
                continue
            if scan.command_only and abs(angle) >= math.pi / 3:
                continue
            if target is not None:
                f, r = forward / length, right / length
                along = min(length, max(0., target[0] * f + target[1] * r))
                if math.hypot(target[0] - along * f, target[1] - along * r) < config.stop_distance_m:
                    continue
            if not local_path.segment_clear(blocked, geometry, keep, angle, length,
                                            heading_tolerance=ANGLE_TOLERANCE if angle else 0.):
                continue
            if rolling is not None:
                if forward <= 0. or abs(angle) <= ANGLE_TOLERANCE:
                    continue
                odom, feedback, speed, room = rolling
                # Turn toward the verified leg while reserving travel at the
                # measured and commanded speeds over the full safety window.
                arc = _checked_forward_prefix(
                    scan, geometry, config, state, odom, feedback, min(speed, length / horizon),
                    horizon, min(room, length), -angle / horizon, steer_only=True, shadow=shadow,
                    depth=depth, target_distance=target_distance, target_bearing=target_bearing)
                if arc is not None:
                    state.escape_side = math.copysign(1., angle)
                    return arc
                continue
            if ((abs(angle) <= ANGLE_TOLERANCE or scan.clear_sweep(
                        geometry, config.half_width_m,
                        (STILL_LINEAR * horizon if scan.command_only
                         else scan._turn_translation(horizon, scan.motion)), occupied=turn_occupied))
                    and scan.begin((math.copysign(1., angle),), advance_m=length,
                                   turn_angle=abs(angle), route_segment=True, reroute=reroute,
                                   occluded_route=bool(preview))):
                points = ((plan.points[0], plan.points[index]) if preview
                          else (plan.points[0], *plan.points[index:visible]))
                state.view_detour = local_path.Detour(None, points, geometry, preview)
                return True
    return False


def _dwa_route_motion(depth, config, state, geometry, ahead, distance, bearing, directions):
    """Try reachable route geometry and then the short forward reference on this view."""
    if ahead is not None and ahead + FORWARD_CLEARANCE_MARGIN_M < distance:
        result = _start_detour_segment(depth, config, state, geometry, ahead, directions, distance,
                                       bearing, closest=True, dwa_only=True)
        if isinstance(result, tuple):
            return result
    forward = distance * math.cos(bearing)
    if forward <= 0.:
        return None
    (floor, cap), _ = _local_velocity_limits(config, 1.)
    if cap <= 0. or cap < floor:
        return None
    heading = -_clamp(bearing, config.wz_max)
    seeds = tuple(_prefix_commands(config, cap, heading, directions if ahead is not None else (), False))
    if not seeds:
        return None
    if ahead is None:
        points = ((0., 0.), local_path.target_in_grid(geometry, forward, distance * math.sin(bearing)))
        references = (Detour(None, points, geometry),)
    else:
        references = _prefix_references(geometry, seeds, dwa_hold(config, state.dwa_admission_ms / 1000))
    proposal = _dwa_detour(depth, config, state, references[0], distance, bearing, branch="prefix",
                           candidate_commands=seeds, recovery=True, alternates=references[1:])
    if proposal is not None:
        side = math.copysign(1., -proposal.command[2]) if proposal.command[2] else state.escape_side
        return _use_dwa(state, proposal, side)
    return None


def _dwa_recovery_route(depth, config, state, geometry, ahead, distance, bearing, directions=(1., -1.)):
    scan = state.active_scan
    fresh = scan.new_dwa_view()
    result = _dwa_route_motion(depth, config, state, geometry, ahead, distance, bearing, directions)
    if result is not None:
        return result
    reason = state.dwa_trace.get("fallback_reason")
    if fresh and reason not in ("search_deadline", "geometry_mismatch", "stale_observation",
                                "no_feedback", "prediction_horizon"):
        scan.dwa_failures += 1
    if scan.dwa_failures >= 3:
        if scan.ready:
            if scan.route_segment:
                scan.retire_motion()
            else:
                scan.try_other_side()
        if not (scan.active or scan.attempted):
            sides = tuple(side for side in directions
                          if _scan_yaw_rate(config, side) > 0. and _scan_yaw_rate(config, side) >= config.floor_wz)
            view = state.dwa_trace.get("memory_recheck")
            if view is not None:
                preferred = math.copysign(1., view["right_m"])
                sides = tuple(sorted(sides, key=lambda side: side != preferred))
            if scan.begin(sides):
                return 0., 0., 0., AVOIDING, "主动观察：连续新观测没有可执行路线，停车后观察"
    return 0., 0., 0., AVOIDING, INITIAL_MEMORY_OVERLAP if reason == "initial_remembered_overlap" else NO_LOCAL_ROUTE


def _observed_obstacle_ahead(scan, depth, config, geometry, pose, target_distance, may_scan):
    lookahead = FORWARD[-1, 0] - geometry.rear - .25
    corridor = (abs(RIGHT) <= _keepout_m(config)) & (FORWARD >= 0.) & (FORWARD <= lookahead)
    transform = geometry.base_from_optical
    memory = scan.obstacles
    source = depth.get("source_stamp_ns") or depth.get("source_frame_id") or depth.get("capture_ms")
    max_depth = target_distance - FORWARD_CLEARANCE_MARGIN_M if target_distance is not None else None
    if not config.use_obstacle_history or (depth.get("body_mount_required") and not geometry.free_space_validated):
        # An estimated moving mount suffices for this view, but must not
        # insert obstacles into an odometry map that cannot clear them again.
        if not scan.views or not scan.matches_route_view(depth) or (
                config.use_obstacle_history and geometry is not config.scan_geometry):
            return None
        hits = scan.observed_grid()[1] if scan.receipt_only else scan.views[-1][3]
        front = transform[0, 2]
        if front <= 0.:
            return None
        if max_depth is not None:
            corridor &= FORWARD <= max_depth * front + transform[0, 3]
        marked = corridor & hits
        if marked.any():
            return max(0., (float(FORWARD[marked].min()) - transform[0, 3]) / front)
        return None
    frame_ms = observation_ms(depth)
    verified = (not scan.receipt_only and geometry.free_space_validated
                and geometry is config.scan_geometry and scan.current_frame == frame_ms)
    capture_pose = depth.get("obstacle_capture_pose")
    if (capture_pose is None and not depth.get("body_mount_required")
            and pose.stamp_ms == frame_ms):
        capture_pose = pose
    current = None
    if (geometry is config.scan_geometry and scan.current_frame and type(source) is int
            and depth.get("map") is not None
            and source > memory.last_source and capture_pose is not None
            and capture_pose.frame == pose.frame and (may_scan or verified)):
        # Use the actual exposure, never a fused grid at a later robot pose.
        selected = None
        if may_scan:
            _, hits = local_path.observe(depth["map"], geometry, RIGHT, FORWARD,
                                         local_path.RESOLUTION, depth.get("ground"),
                                         max_hit_depth=max_depth, check_free_space=False)
            if (hits & corridor).any():
                selected = hits & (FORWARD >= 0.) & (FORWARD <= lookahead)
                current = hits
        memory.last_source = source
        if verified:
            grid = local_path.project_observation(depth, geometry)
            if config.local_controller == "dwa":
                nearby = grid[1] & (FORWARD**2 + RIGHT**2 <= CAMERA_OCCUPANCY_MEMORY_RANGE_M**2)
                selected = nearby if selected is None else selected | nearby
            memory.remember_view(capture_pose, *grid, remember_occupied=selected,
                                 clear=local_path.confirmed_free(grid), physical_support=config.local_controller == "dwa")
            scan.historical_free = memory.project_free(pose)
        elif selected is not None:
            memory.remember_view(capture_pose, np.zeros_like(hits), hits, remember_occupied=selected)
    scan.project_history(pose, current=current, capture_pose=capture_pose)
    if scan.historical_hits is not None:
        blocked = corridor & scan.historical_hits
        if blocked.any():
            return max(0., (float(FORWARD[blocked].min()) - transform[0, 3]) / transform[0, 2])
    return None


def _target_path_obstacle(occupied, geometry, config, target, stop_distance):
    forward, right = target
    length = max(0., math.hypot(forward, right) - stop_distance)
    hits = local_path.segment_obstacle_forwards(occupied, geometry, _keepout_m(config),
                                               math.atan2(right, forward), length)
    transform = geometry.base_from_optical
    if not hits.size:
        return None
    return max(0., (float(hits.min()) - transform[0, 3]) / transform[0, 2]) if transform[0, 2] > 0. else 0.


def _target_corridor_clear(depth, config, scan, distance, bearing, velocities):
    """Check the footprint up to the target's stand-off point in the current view."""
    geometry = (config.scan_geometry or config.local_geometry
                if scan.receipt_only else config.scan_geometry)
    if (geometry is None or distance is None or bearing is None or not scan.views
            or not scan.matches_route_view(depth)):
        return False
    free, occupied = scan.grid()
    if not (free.any() or occupied.any()):
        return False
    forward, right = local_path.target_in_grid(
        geometry, distance * math.cos(bearing), distance * math.sin(bearing))
    if forward <= 0.:
        return False
    horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
    if _target_path_obstacle(occupied, geometry, config, (forward, right), config.stop_distance_m) is not None:
        return False
    motion_grid = scan.grid(include_near=False) if scan.near_squares is not None else (free, occupied)
    if not local_path.motion_sweep_clear(motion_grid, geometry, config.half_width_m, horizon, velocities):
        # A straight forward command keeps the same rectangular cross-section through braking.
        if not all(vx >= 0. and vy == 0. and wz == 0. for vx, vy, wz in velocities):
            return False
        speed = max(vx for vx, _, _ in velocities)
        travel = speed * horizon + speed**2 / (2 * config.motion_braking[0])
        if not local_path.segment_clear(motion_grid[1], geometry, _keepout_m(config), 0., travel):
            return False
    return (scan.near_squares is None or not scan.near_motion_hits(
        geometry, config.half_width_m, horizon, velocities).any())


def _near_sweep_hits(config, scan, odom, state, twist):
    geometry = config.scan_geometry or config.local_geometry
    motion = planar_feedback(odom) or state.command_tail
    velocities = (motion, state.commanded, twist)
    horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
    return scan.near_motion_hits(geometry, config.half_width_m, horizon, velocities)


def camera_arc_clear(geometry, config, occupied, target, command, previous, residual, age_s, admission_s, *,
                     route_room=None):
    """Sweep one view through delivery uncertainty and a complete stop."""
    mode = "camera_approx" if geometry.front == 0. else "body"
    footprint = motion_model.FootprintSpec(geometry.front, geometry.rear, config.half_width_m,
                                           config.clearance_margin_m, mode)
    registration = dwa_registration(None, None, 0, age_s * 1000, geometry, (previous, residual))
    observed = registration.squares(local_path.occupied_squares(occupied))
    bounds = tuple(tuple(bound(0., p, r) for p, r in zip(previous, residual)) for bound in (min, max))
    trace = motion_model.predict_motion(
        previous, previous, command, hold_s=admission_s + (config.watchdog_ms + 50) / 1000,
        latency_s=admission_s, acceleration=config.motion_acceleration, braking=config.motion_braking,
        geometry=geometry, mode=mode, initial_bounds=bounds)
    check = motion_model.evaluate_motion(
        trace, footprint=footprint, occupied=observed, remembered=(),
        supported_region=local_path.SUPPORTED_REGION, original_goal=target,
        stop_distance_m=config.stop_distance_m + (registration.error(math.hypot(*target)) if target else 0.),
        terminal_boundary=motion_model.TravelBudget(route_room) if route_room is not None else None,
        cull_obstacles=True)
    return check.accepted


@dataclass(frozen=True)
class CameraPassage:
    route: Detour
    occupied: np.ndarray
    capture_ms: int
    generation: int | None
    corridor_corner: bool = False
    corner_only: bool = False
    keyframe: tuple | None = None

    def register(self, config, geometry, occupied, capture_ms, *, previous=None, residual=None):
        max_age_ms = min(config.max_feedback_gap_ms, config.depth_age_limit_ms)
        latest = (self.route, self.occupied, self.capture_ms)
        keyframe = self.keyframe if capture_ms > self.capture_ms else None
        if keyframe is not None and capture_ms - keyframe[2] > max_age_ms:
            keyframe = None
        source, earlier, stamp = keyframe or latest
        delta = capture_ms - stamp
        if (capture_ms < self.capture_ms or not 0 <= delta <= max_age_ms
                or (geometry.front, geometry.rear) != (source.geometry.front, source.geometry.rear)
                or not np.array_equal(geometry.base_from_optical, source.geometry.base_from_optical)):
            return None
        if delta == 0:
            return source
        travel = math.hypot(config.vx_max, config.vy_max if config.use_lateral else 0.) * delta / 1000
        turn = config.wz_max * delta / 1000
        if self.corridor_corner and previous is not None:
            yaw_rate = max(abs(previous[2]), abs(residual[2]) if residual is not None else 0.)
            turn = min(turn, yaw_rate * delta / 1000 + .025)
        pivot = -geometry.half_length if geometry.front == 0. else 0.
        exit_line = source.passage_exit
        points = local_path.register_passage_route(source.points + (exit_line or ()), earlier, occupied,
                                                   travel, turn, pivot=pivot)
        if points is None:
            if keyframe is not None:
                return replace(self, keyframe=None).register(config, geometry, occupied, capture_ms,
                                                              previous=previous, residual=residual)
            return None
        return Detour(None, points[:-2] if exit_line else points, geometry,
                      passage_exit=points[-2:] if exit_line else None, corridor_corner=source.corridor_corner)

    def advance(self, config, route, occupied, capture_ms, *, corner_only=False):
        keyframe = None
        if self.corridor_corner:
            keyframe = self.keyframe or (self.route, self.occupied, self.capture_ms)
            if capture_ms - keyframe[2] >= min(config.max_feedback_gap_ms, config.depth_age_limit_ms) / 2:
                keyframe = None
        return replace(self, route=route, occupied=occupied, capture_ms=capture_ms,
                       corner_only=corner_only, keyframe=keyframe)


def camera_route_command(route, occupied, config, target, nominal, previous, residual, age_s, admission,
                         *, require_turn):
    geometry = route.geometry
    (floor_x, max_x), _ = _local_velocity_limits(config, 1.)
    yaw_limits = (-_scan_yaw_rate(config, 1.), _scan_yaw_rate(config, -1.))
    nominal = min(nominal, max_x)
    lookahead = max(.45, nominal * config.dwa_horizon_s)
    for scale in (1., .75, .5, .25):
        tracked = dwa.track_route(route.points, nominal * scale, config.dwa_horizon_s,
                                  yaw_limits, body_origin=(0., 0.), lateral=config.use_lateral,
                                  lookahead_m=lookahead, yaw_floor=config.floor_wz_moving)
        if tracked is None:
            break
        command, room = tracked
        cx, cy, yaw = command
        (floor_y, max_y) = _local_velocity_limits(config, -1. if cy > 0. else 1.)[1]
        if (cx <= 0. or cx < floor_x or (cy and not floor_y <= abs(cy) <= max_y)
                or (yaw and not yaw_limits[0] <= yaw <= yaw_limits[1]) or (require_turn and not (cy or yaw))):
            continue
        if camera_arc_clear(geometry, config, occupied, target, command, previous, residual, age_s, admission,
                            route_room=room if target is None else None):
            return command
    return None


def _use_camera_route(state, route, command, *, retained=False, corner_only=False):
    scan = state.active_scan
    if scan.active or scan.result_frame:
        scan.retire_motion()
    state.camera_arc_selected = True
    state.view_detour = route
    state.avoidance_yaw_override = True
    state.dwa_trace.update(branch="camera_arc", reference=route.points, retained_route=retained, corner_only=corner_only,
                           proposed_twist=command, fallback_reason=None)
    return *command, AVOIDING, "局部规划：沿当前视野的制动校验弧线绕行"


def _preview_corridor_corner(depth, config, state, geometry, distance, bearing):
    scan = state.active_scan
    if (distance <= config.arrival_distance_m or not scan.current_frame or not scan.matches_route_view(depth)):
        return None
    grid = scan.observed_grid()
    target = local_path.target_in_grid(geometry, distance * math.cos(bearing), distance * math.sin(bearing))
    if _target_path_obstacle(grid[1], geometry, config, target, config.stop_distance_m) is None:
        return None
    route = local_path.corridor_corner_route(grid, geometry, _keepout_m(config), target)
    if route is None:
        return None
    if scan.command_only:
        if config.odom_expected or state.dwa_progress.requires_feedback:
            return None
        capture = depth.get("capture_ms") or scan.current_frame
        control_ms = depth.get("control_ms") or scan.current_frame
        age = (control_ms - capture) / 1000
        if not 0 <= control_ms - observation_ms(depth, scan.current_frame) < config.depth_age_limit_ms:
            return None
        command = camera_route_command(route, grid[1], config, target, config.vx_max, state.commanded,
                                       state.command_tail, age, state.dwa_admission_ms / 1000, require_turn=False)
        return _use_camera_route(state, route, command) if command is not None else None
    proposal = _dwa_detour(depth, config, state, route, distance, bearing, recovery=True, follow_published=True)
    return _use_dwa(state, proposal, math.copysign(1., route.points[-1][1])) if proposal is not None else None


def _camera_detour(depth, config, state, geometry, vx, distance, bearing, *,
                   ahead=None, corner_only=False, retain_only=False):
    scan = state.active_scan
    control_ms = depth.get("control_ms") or scan.current_frame
    capture_ms = depth.get("capture_ms") or scan.current_frame
    age_s = (control_ms - capture_ms) / 1000
    if (not 0 <= control_ms - observation_ms(depth, scan.current_frame) < config.depth_age_limit_ms
            or not scan.matches_route_view(depth)):
        return None
    target = (None if corner_only else
              local_path.target_in_grid(geometry, distance * math.cos(bearing), distance * math.sin(bearing)))
    if target is not None and target[0] <= 0.:
        return None
    grid = scan.observed_grid()
    admission = state.dwa_admission_ms / 1000
    keep = state.published_camera_passage
    track = state.tracker.track
    generation = track.generation if track else None
    route = (keep.register(config, geometry, grid[1], capture_ms,
                           previous=state.commanded, residual=state.command_tail)
             if keep and (keep.generation == generation or corner_only) else None)
    if corner_only and (keep is None or not keep.corridor_corner or route is None):
        return None
    if (not corner_only and ahead is None
            and (route is None or not (keep.corridor_corner or local_path.passage_ahead(route.passage_exit)))):
        return None
    command = (camera_route_command(route, grid[1], config, target, vx, state.commanded,
                                    state.command_tail, age_s, admission, require_turn=False) if route else None)
    retained = command is not None
    if not retained and (corner_only or ahead is None or retain_only):
        return None
    if command is None:
        width = _keepout_m(config)
        offset = geometry.base_from_optical
        travel_limits = _route_travel_limits(config)
        opening = (local_path.opening_route(depth["map"], grid[1], geometry, width,
                                            ahead * offset[0, 2] + offset[0, 3], target, True, travel_limits)
                   if ahead is not None else None)
        if opening is not None:
            route = local_path.extend_passage(opening, grid, width, target, config.stop_distance_m)
            command = camera_route_command(route, grid[1], config, target, vx, state.commanded,
                                           state.command_tail, age_s, admission, require_turn=False)
        if command is None:
            route = local_path.clearance_route(grid, geometry, width, target, config.stop_distance_m, (0., 0.))
            if route is not None:
                command = camera_route_command(route, grid[1], config, target, vx, state.commanded,
                                               state.command_tail, age_s, admission, require_turn=True)
        if command is None:
            return None
    return _use_camera_route(state, route, command, retained=retained, corner_only=corner_only)


def _avoid(vx, vy, wz, depth, config: Config, status, reason, state=None,
           target_distance=None, measured_vx=None, aligned=True, odom=None, may_scan=False,
           target_bearing=None, dt=0., desired_forward=0.):
    """Slow for foreground obstacles and look around them before local detours."""
    scan = state.active_scan if config.local_planning and state is not None else None
    geometry = config.scan_geometry or config.local_geometry
    ahead, coverage = _clearance(depth, config)
    if (scan is not None and geometry is not None and geometry.horizontal is not None
            and geometry.vertical is not None and scan.views and scan.matches_route_view(depth)
            and depth.get("ground") is None):
        _, occupied = scan.observed_grid()
        if scan.historical_hits is not None:
            occupied = occupied | scan.historical_hits
        ahead = projected_forward_clearance(occupied, geometry, config)
    if state is not None:
        state.last_clearance, state.last_coverage = ahead, coverage

    # The target itself also appears in the depth map. Only steer early
    # around a return distinctly in front of it.
    foreground = (ahead is not None
                  and (target_distance is None or depth.get("map") is None
                       or ahead + FORWARD_CLEARANCE_MARGIN_M < target_distance))
    pose = _obstacle_reference(odom, depth)
    plan_obstacle = False
    if scan is not None and geometry is not None and pose is not None:
        if scan.obstacles.has_occupied and scan.obstacles.frame != pose.frame:
            return 0., 0., 0., AVOIDING, "缺少同坐标系位姿，保留已观测障碍并停车"
        obstacle = _observed_obstacle_ahead(scan, depth, config, geometry, pose, target_distance, may_scan)
        if obstacle is not None:
            ahead = obstacle if ahead is None else min(ahead, obstacle)
            foreground = plan_obstacle = True
            state.last_clearance = ahead
    if scan is not None and scan.near_hits is not None:
        swept = _near_sweep_hits(config, scan, odom, state, (vx, vy, wz))
        if swept.any():
            # A hit in the proposed turn is reason to choose another motion.
            # Only a hit in the straight corridor shortens its stopping room.
            straight_edge = _keepout_m(config) + local_path.RESOLUTION / 2
            straight_hits = swept & (abs(RIGHT) < straight_edge - 1e-9)
            if ahead is None or straight_hits.any() or config.use_lateral:
                hits = straight_hits if straight_hits.any() and not config.use_lateral else swept
                obstacle = max(0., float(FORWARD[hits].min()))
                ahead = obstacle if ahead is None else min(ahead, obstacle)
            foreground = plan_obstacle = True
            state.last_clearance = ahead
    if (scan is not None and state.published_dwa_route is not None
            and (foreground or state.published_dwa_route.passage_exit is not None
                 or not config.use_obstacle_history) and geometry is not None
            and may_scan and target_distance is not None and target_bearing is not None):
        following = _follow_published_route(depth, config, state, geometry, target_distance, target_bearing,
                                            foreground=foreground, approach_speed=desired_forward)
        if following is not None:
            return following
    path_clear = None
    if (config.local_controller == "dwa" and may_scan and geometry is not None and scan is not None
            and not scan.has_result and not scan.attempted and target_distance is not None
            and target_bearing is not None and scan.views and scan.matches_route_view(depth)):
        target = local_path.target_in_grid(geometry, target_distance * math.cos(target_bearing),
                                           target_distance * math.sin(target_bearing))
        free, occupied = scan.grid(include_near=False)
        if target[0] > 0. and (occupied.any() or (current_target_observed(state) and free.any())):
            obstacle = _target_path_obstacle(occupied, geometry, config, target, config.stop_distance_m)
            if obstacle is None and ahead is not None and ahead <= config.obstacle_stop_m and state.dwa_projection:
                raw = state.dwa_projection.grid_for(geometry, require_valid_depth=True, check_free_space=False)
                if raw is None:
                    return 0., 0., 0., AVOIDING, "等待可用深度确认近障位置"
                obstacle = _target_path_obstacle(raw[1], geometry, config, target, config.stop_distance_m)
                if obstacle is None and not occupied.any():
                    return 0., 0., 0., AVOIDING, "近距回波尚未定位到目标路径，等待新观测"
            path_clear = obstacle is None
            if obstacle is not None:
                ahead = obstacle if ahead is None else min(ahead, obstacle)
                foreground = plan_obstacle = True
                state.last_clearance = ahead
    stop = config.obstacle_stop_m
    check_distance = config.approach_check_distance_m
    depth_age_s = config.depth_age_limit_ms / 1000
    if foreground:
        # Reserve the distance travelled in one observation freshness window.
        motion = planar_feedback(odom) if config.local_planning else None
        if motion is not None:
            measured_vx = motion[0]
        last_vx = state.commanded[0] if state is not None else 0.0
        speed = max(0.0, measured_vx or 0.0, last_vx)
        stop = forward_stop_line(config, speed)
    near = ahead is not None and ahead <= stop
    approaching = ahead is not None and ahead < check_distance
    side = 0.0
    local_command = None
    local_route = None
    dwa_lateral_routes = []
    camera_detour = (config.local_controller == "dwa" and not config.odom_expected and scan is not None
                     and scan.command_only and not state.dwa_progress.requires_feedback and scan.current_frame
                     and geometry is not None and may_scan and vx > 0. and target_distance is not None
                     and target_bearing is not None)
    if camera_detour and state.published_camera_passage is not None:
        arc = _camera_detour(depth, config, state, geometry, vx, target_distance, target_bearing,
                             ahead=ahead if foreground and approaching else None)
        if arc is not None:
            return arc
    if scan is not None and scan.ready and scan.advance_m is not None:
        return _advance_after_scan(depth, config, state, odom, ahead, stop, target_distance, dt,
                                   target_bearing=target_bearing if may_scan else None)
    if scan is not None and state.detour_leg is not None:
        leg = state.detour_leg
        current_geometry = (geometry if state.use_body_mount or leg.geometry is geometry
                            else config.local_geometry)
        goal = (None if target_distance is None or target_bearing is None or current_geometry is None
                else PathLeg.world_point(scan.pose, local_path.target_in_grid(
                    current_geometry, target_distance * math.cos(target_bearing),
                    target_distance * math.sin(target_bearing))))
        brake_horizon = (config.watchdog_ms + 50) / 1000
        horizon = brake_horizon + config.depth_age_limit_ms / 1000
        motion = planar_feedback(odom)
        vx, vy = leg.command
        state.avoidance_yaw_override = True
        if (not config.use_lateral or motion is None or not scan.poses or not scan.current_frame
                or scan.motion_frame is not leg.motion_frame
                or (current_geometry is not leg.geometry and not state.use_body_mount)
                or not may_scan or goal is None
                or math.dist(leg.goal, goal) > 3 * local_path.RESOLUTION
                or state.side_stall_s.get(state.escape_side, 0.) >= config.stuck_window_s
                or (near and foreground and vx)):
            state.detour_leg = None
            return 0., 0., 0., AVOIDING, "局部规划：执行段条件已变化，停车后重规划"
        f, r = leg.remaining(scan.pose)
        speed = math.hypot(vx, vy)
        along = (f * vx - r * vy) / speed
        room = min(along, target_distance - config.stop_distance_m)
        (floor_x, limit_x), (floor_y, limit_y) = _local_velocity_limits(config, state.escape_side)
        scale = max(0., min(1., room / (speed * horizon),
                            limit_x / abs(vx) if vx else 1., limit_y / abs(vy) if vy else 1.))
        command = vx * scale, vy * scale
        if not scale or (vx and abs(command[0]) < floor_x) or (vy and abs(command[1]) < floor_y):
            state.detour_leg = None
            return 0., 0., 0., AVOIDING, "局部规划：已到执行段制动点"
        # Calibrated scans carry occupied history. The camera fallback uses
        # its newest projection, with no path search in this control step.
        grid = (scan.grid() if scan.views and geometry is current_geometry and scan.matches_route_view(depth)
                else state.local_plan.grid_for(depth, current_geometry))
        padding = 0.
        if grid is not None and scan.receipt_only and scan.views and scan.matches_route_view(depth):
            view_pose = scan.views[-1][1]
            radius = math.hypot(max(leg.geometry.front, leg.geometry.rear), _keepout_m(config))
            padding = math.dist(view_pose[:2], scan.pose[:2]) + radius * abs(view_pose[2] - scan.pose[2])
        reason = leg.check(scan.pose, motion, state.commanded[:2], command, grid,
                           _keepout_m(config), horizon, brake_horizon, padding=padding)
        if reason is not None:
            state.detour_leg = None
            return 0., 0., 0., AVOIDING, reason
        state.executing_leg = leg
        return *command, 0., AVOIDING, "局部规划：沿避障路线行进"

    observed_obstacle = False
    if (scan is not None and geometry is not None and scan.current_frame and scan.views
            and may_scan and target_distance is not None and foreground and not approaching):
        transform = geometry.base_from_optical
        plane = ahead * transform[0, 2] + transform[0, 3]
        ahead_band = (FORWARD > 0.) & (abs(FORWARD - plane) <= .10)
        observed_obstacle = (scan.grid()[1] & ahead_band & (abs(RIGHT) <= _keepout_m(config))).any()
    needs_detour = near or plan_obstacle or observed_obstacle or (foreground and approaching)
    corridor_clear = False
    if (not near or path_clear) and may_scan and scan is not None and not scan.has_result:
        saved = state.dwa_guidance
        continuing_detour = (config.local_controller == "dwa"
                             and (state.published_dwa_route is not None
                                  or (saved is not None and saved.anchor is state.goal_anchor
                                      and saved.frame is scan.motion_frame)))
        stalled_approach = path_clear and not vx and desired_forward > 0.
        if needs_detour or continuing_detour or stalled_approach:
            motion = planar_feedback(odom) or state.command_tail
            floor = config.floor_wz_moving if vx or vy else config.floor_wz
            limit = _scan_yaw_rate(config, 1. if wz < 0. else -1.)
            yaw = math.copysign(min(max(abs(wz), floor), limit), wz) if wz else 0.
            corridor_clear = _target_corridor_clear(
                depth, config, scan, target_distance, target_bearing, (motion, state.commanded, (vx, vy, yaw)))
            needs_detour = not corridor_clear
            if path_clear and (needs_detour or stalled_approach) and not continuing_detour:
                # The requested twist can clip a jamb even when a slower motion
                # along the direct reference can brake safely before it.
                track = state.tracker.track
                if max(vx, desired_forward) > 0. and not scan.command_only and track is not None and track.observed:
                    route = Detour(None, ((0., 0.), target), geometry)
                    checked = _dwa_detour(depth, config, state, route, target_distance, target_bearing,
                                          branch="direct", speed_limit=vx or desired_forward, recovery=True)
                    if checked is not None:
                        return _use_dwa(state, checked, state.escape_side)
                return 0., 0., 0., AVOIDING, "目标直达路径畅通，等待当前运动扫掠范围清空"
    if (needs_detour and config.local_controller == "dwa" and state is not None and state.goal_anchor is not None
            and may_scan and geometry is not None and scan is not None and scan.views and scan.current_frame
            and target_distance is not None and target_bearing is not None and not scan.command_only
            and not _dwa_unavailable_reason(depth, config, state, recovery=True)):
        grid = scan.grid()
        if grid[1].any() and depth.get("ground") is not None:
            result = _start_detour_segment(depth, config, state, geometry,
                                           ahead if ahead is not None else target_distance,
                                           (1., -1.), target_distance, target_bearing, closest=True,
                                           dwa_only=True)
            if isinstance(result, tuple):
                return result
    if (camera_detour and needs_detour
            and ((foreground and approaching and state.published_camera_passage is None)
                 or (observed_obstacle and (state.published_camera_passage is not None
                     or (state.goal_anchor is None and not scan.attempted
                         and abs(target_bearing) >= config.align_tol * config.half_fov_rad))))):
        arc = _camera_detour(depth, config, state, geometry, vx, target_distance, target_bearing, ahead=ahead,
                             retain_only=state.published_camera_passage is not None)
        if arc is not None:
            return arc

    scale = 1.
    if corridor_clear:
        # The current route and velocity sweep have both passed the grid check.
        state.recovery = False
        state.blocked_for_s = 0.
    elif ahead is not None and ahead < config.slow_distance_m:
        scale = max(0., (ahead - stop) / max(1e-6, config.slow_distance_m - stop))
        if status != ALIGNING:
            status, reason = AVOIDING, f"{reason}；走廊内 {ahead:.2f} m 处有障碍，减速"
    if needs_detour:
        stationary_line = config.obstacle_stop_m + LOCAL_BLOCKED_APPROACH_BUFFER_M
        brake_at = min(check_distance, max(stop, stationary_line) if config.local_planning
                       else stop + BLOCKED_APPROACH_BUFFER_M)
        horizon_s = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
        approach = 0.
        if (foreground and aligned and depth.get("map") is not None and target_distance is not None
                and ahead > brake_at and (config.local_planning or ahead < check_distance)):
            limit = (ahead - brake_at) / horizon_s
            floor = config.floor_vx
            if config.local_planning:
                # Include the new command in the travel reserve used on the next tick.
                if config.floor_vx_moving is not None:
                    floor = config.floor_vx_moving
                limit = min(limit, (target_distance - config.stop_distance_m) / horizon_s,
                            (ahead - stationary_line) / (horizon_s + depth_age_s),
                            max(floor, LOCAL_BLOCKED_APPROACH_MAX_MPS))
            speed = min(vx * scale, config.vx_max, limit)
            if speed > 0 and speed >= floor:
                approach = speed

        camera_routes = _camera_frame_routes(config, scan) and pose is None
        keep = _keepout_m(config)
        rooms = {s: _clearance(depth, config, lateral_offset_m=s * keep)
                 for s in (-1.0, 1.0)}
        if config.local_planning:
            order = _local_side_order(rooms, config, state,
                                      depth.get("source_frame_id", depth.get("frame_id")))
            if (config.local_controller == "dwa" and may_scan and target_distance is not None
                    and target_bearing is not None and scan is not None and not scan.command_only):
                unavailable = _dwa_unavailable_reason(depth, config, state, recovery=True)
                if unavailable:
                    state.dwa_trace["fallback_reason"] = unavailable
                    return 0., 0., 0., AVOIDING, NO_LOCAL_ROUTE
                return _dwa_recovery_route(depth, config, state, geometry, ahead, target_distance,
                                           target_bearing, order)
            planning_depth = dict(depth, allow_forward=not near and aligned,
                                  body_half_width=config.half_width_m,
                                  closest=bool(scan is not None and scan.exhausted),
                                  allow_unknown=True)
            route_target = None
            if target_distance is not None and target_bearing is not None:
                planning_depth["stop_target"] = (
                    target_distance * math.cos(target_bearing), target_distance * math.sin(target_bearing),
                    config.stop_distance_m, config.stop_tolerance_m)
                route_target = planning_depth["stop_target"][:2]
            if (scan is not None and config.scan_geometry is not None and scan.current_frame
                    and (scan.historical_hits is not None
                         or (not scan.receipt_only and (scan.ready or scan.has_result)))):
                revision = pose.stamp_ms if pose is not None else (
                    scan.poses[-1][0] if scan.poses else scan.current_frame)
                planning_depth = dict(planning_depth, scan_grid=scan.grid(), scan_revision=revision)
            elif (scan is not None and scan.receipt_only and scan.views
                  and (config.scan_geometry or config.local_geometry) is config.local_geometry
                  and scan.matches_route_view(depth)):
                planning_depth.update(frame_id=scan.current_frame, project_scan_view=scan.grid)
                if scan.near_hits is not None:
                    planning_depth["scan_revision"] = scan.near_revision
            transitioning = False
            compare_dwa_routes = config.local_controller == "dwa" and config.use_lateral
            rank_fallback = planning_depth["closest"] and route_target is not None
            for candidate in order:
                if state is not None and state.side_stall_s.get(candidate, -math.inf) >= config.stuck_window_s:
                    continue
                command = _local_translation(planning_depth, config, state, odom, ahead, candidate)
                transitioning |= command is LocalHold.TRANSITION
                if command is LocalHold.TRANSITION and candidate == state.escape_side:
                    break
                if isinstance(command, tuple) and (planning_depth["allow_forward"] or command[0] <= 0):
                    floor, limit = _local_velocity_limits(config, candidate)[1]
                    if command[1] == 0. or floor <= -candidate * command[1] <= limit:
                        route = state.local_plan.plans.get(candidate)
                        if compare_dwa_routes and route is not None:
                            dwa_lateral_routes.append((candidate, route))
                        if local_command is not None and (not rank_fallback or local_route is None or
                                route is None or _route_rank(route, route_target, config)
                                >= _route_rank(local_route, route_target, config)):
                            continue
                        side, local_command, local_route = candidate, command, route
                        if not compare_dwa_routes and (not rank_fallback or route is None):
                            break
            if local_command is None and transitioning:
                return 0.0, 0.0, 0.0, AVOIDING, LocalHold.TRANSITION.value
            if scan is not None and local_command is None:
                feedback = planar_feedback(odom)
                yaw_rate = max(_scan_yaw_rate(config, 1.), _scan_yaw_rate(config, -1.))
                geometry = (config.scan_geometry or config.local_geometry
                            if scan.receipt_only else config.scan_geometry)
                can_plan = (may_scan and foreground and geometry is not None and scan.current_frame
                            and yaw_rate > 0. and config.floor_wz <= yaw_rate)
                retry = scan.exhausted and not scan.rerouted
                if can_plan and not config.use_lateral and scan.receipt_only and config.scan_geometry is None:
                    retry = scan.matches_route_view(depth) and scan.retry_route(config.max_obs_age_ms)
                plan_now = can_plan and (not scan.attempted or retry)
                if (plan_now and not config.use_lateral and approach and not near and feedback is not None
                        and (ahead > config.slow_distance_m or feedback[0] > STILL_LINEAR)
                        and scan.matches_route_view(depth)):
                    room = min(ahead - brake_at, target_distance - config.stop_distance_m)
                    travel = max(math.hypot(*feedback[:2]), math.hypot(*state.commanded[:2])) * horizon_s
                    if travel <= room:
                        arc = _start_detour_segment(
                            depth, config, state, geometry, ahead, order, target_distance, target_bearing,
                            rooms=rooms, reroute=scan.exhausted, closest=scan.exhausted,
                            rolling=(odom, feedback, approach, room))
                        if isinstance(arc, tuple):
                            return arc
                if (not config.use_lateral and may_scan and aligned and (not near or config.local_controller == "dwa") and foreground
                        and target_distance is not None and vx > 0. and feedback is not None
                        and geometry is not None and geometry.horizontal is not None
                        and geometry.vertical is not None and config.wz_max > 0.
                        and scan.views and scan.matches_route_view(depth)
                        and not scan.active and (ahead > config.slow_distance_m or scan.attempted
                                                 or config.local_controller == "dwa")):
                    goal_room = target_distance - config.stop_distance_m
                    limited = min(vx, config.vx_max, LOCAL_BLOCKED_APPROACH_MAX_MPS,
                                  (ahead - brake_at) / (horizon_s + depth_age_s), goal_room / horizon_s)
                    room = min(goal_room, ahead - brake_at)
                    prefix = _checked_forward_prefix(scan, geometry, config, state, odom, feedback,
                                                     limited, horizon_s, room, wz, order,
                                                     depth=depth, target_distance=target_distance,
                                                     target_bearing=target_bearing)
                    if prefix is not None:
                        return prefix
                if (camera_routes and approach and may_scan and geometry is not None and feedback is not None
                        and abs(feedback[1]) <= config.scan_odom_linear_tolerance_mps
                        and abs(state.commanded[1]) <= STILL_LINEAR
                        and not scan.active and not scan.result_frame and scan.views and scan.matches_route_view(depth)):
                    room = min(ahead - stationary_line, target_distance - config.stop_distance_m)
                    prefix = _checked_forward_prefix(scan, geometry, config, state, odom, feedback,
                                                     approach, horizon_s, room, wz, depth=depth,
                                                     target_distance=target_distance, target_bearing=target_bearing)
                    if prefix is not None:
                        state.escape_side = 0.
                        state.side_better_frames = 0
                        return prefix
                if plan_now:
                    detour = _start_detour_segment(
                        depth, config, state, geometry, ahead, order, target_distance, target_bearing,
                        rooms=rooms, reroute=scan.exhausted, closest=scan.exhausted)
                    if isinstance(detour, tuple):
                        return detour
                    if detour:
                        return 0., 0., 0., AVOIDING, "局部规划：停车对准路线首段"
            if scan is not None and scan.ready:
                if local_command is not None or (not near and len(scan.directions) <= 1):
                    scan.finish()
                elif not near and scan.try_other_side():
                    return 0.0, 0.0, 0.0, AVOIDING, "主动观察：尝试另一侧"
                else:
                    scan.exhaust()
            if local_command is None:
                feedback = planar_feedback(odom)
                approach_clear = False
                approach_yaw = 0.
                if approach and scan is not None and not scan.active and feedback is not None:
                    approach_geometry = (config.scan_geometry if "scan_grid" in planning_depth
                                         else config.local_geometry)
                    velocities = (feedback, state.commanded)
                    safe_speed = min(ahead - stationary_line,
                                     target_distance - config.stop_distance_m) / horizon_s
                    if (approach_geometry is not None and depth.get("map") is not None
                            and depth.get("frame_id") is not None
                            and all(abs(vy) <= STILL_LINEAR and abs(vx) <= safe_speed
                                    for vx, vy, _ in velocities)):
                        grid = state.local_plan.grid_for(planning_depth, approach_geometry)
                        yaw = _clamp(wz, _scan_yaw_rate(config, 1. if wz < 0. else -1.))
                        if (yaw and abs(yaw) >= config.floor_wz_moving
                                and (grid[0].any() or grid[1].any()) and local_path.motion_sweep_clear(
                                    grid, approach_geometry, config.half_width_m, horizon_s, (*velocities, (approach, 0., yaw)))):
                            approach_yaw, approach_clear = yaw, True
                        else:
                            approach_clear = local_path.motion_sweep_clear(
                                grid, approach_geometry, config.half_width_m, horizon_s, (*velocities, (approach, 0., 0.)))
                # A distant hit can block the full goal corridor while the
                # measured, camera-frame straight prefix is still executable.
                prefer_approach = approach_clear and not approaching and camera_routes
                if (not prefer_approach and scan is not None and not (scan.active or scan.attempted)
                        and may_scan and foreground
                        and geometry is not None and config.floor_wz <= yaw_rate and yaw_rate > 0
                        and scan.current_frame):
                    turn_angle = ANGLE
                    free, occupied = scan.grid()
                    unknown = ~(free | occupied)
                    transform = geometry.base_from_optical
                    obstacle = ahead * transform[0, 2] + transform[0, 3]
                    angles = np.arctan2(RIGHT, FORWARD)
                    relevant = unknown & (FORWARD > 0.) & (FORWARD <= obstacle + geometry.rear + .25)
                    order = tuple(sorted(order, key=lambda side: np.count_nonzero(
                        relevant & (RIGHT * side > keep)
                        & (abs(angles - side * turn_angle) <= config.half_fov_rad)), reverse=True))
                    order = tuple(side for side in order
                                  if _scan_yaw_rate(config, side) > 0.
                                  and _scan_yaw_rate(config, side) >= config.floor_wz)
                    if (not scan.command_only and
                            all(room is not None and room < check_distance for room, _ in rooms.values())):
                        # A nearby opening can require only small alignment.
                        # This preview ranks the heading; the drive still
                        # checks the corridor for measured obstacles.
                        alignment = math.radians(6)
                        for direction in order:
                            offset = direction * ahead * math.tan(alignment)
                            room, seen = _clearance(depth, config, lateral_offset_m=offset)
                            if room is not None and room >= check_distance and seen >= config.coverage_min:
                                turn_angle = alignment
                                order = ((direction, -direction, direction) if -direction in order else (direction,))
                                break
                    advance = (obstacle + geometry.rear + .25) / math.cos(ANGLE)
                    if order and advance > 0. and scan.begin(order, advance_m=advance, turn_angle=turn_angle):
                        return 0., 0., 0., AVOIDING, "主动观察：停车后观察较空旷的一侧"
                if (scan is not None and not (scan.active or scan.attempted)
                        and scan.current_frame and may_scan and foreground
                        and not any(_scan_yaw_rate(config, d) > 0.
                                    and _scan_yaw_rate(config, d) >= config.floor_wz for d in order)):
                    scan.exhaust()
                if state is not None:
                    state.escape_side = 0.0
                    state.side_better_frames = 0
                    state.recovery = True
                if state.motion_source != "command" and feedback is None:
                    return 0.0, 0.0, 0.0, AVOIDING, "缺少可用的速度或连续位姿，局部规划和主动观察暂停"
                if approach_clear:
                    state.recovery = False
                    state.view_detour = None
                    state.avoidance_yaw_override = True
                    detail = "低速接近并转向" if approach_yaw else "低速接近"
                    return (approach, 0., approach_yaw, AVOIDING,
                            f"两侧无可执行避让路线；{detail}（障碍停车线 {brake_at:.2f} m）")
                if state is not None and state.view_detour is not None and not (scan and scan.exhausted):
                    return 0.0, 0.0, 0.0, AVOIDING, LocalHold.PREFIX.value
                return 0.0, 0.0, 0.0, AVOIDING, NO_LOCAL_ROUTE
        else:
            usable = {s: (cov >= config.coverage_min, math.inf if room is None else room)
                      for s, (room, cov) in rooms.items()
                      if (target_distance is not None or near)
                      and (room is None or room > config.side_clearance_m)}
            if usable:
                side = (state.escape_side if state and state.escape_side in usable
                        else max(usable, key=usable.get))
            else:
                left, right = (_side_label(rooms[s][0], config) for s in (-1.0, 1.0))
                blocked = (f"走廊内 {ahead:.2f} m 处有障碍，两侧也都不可用"
                           f"（左 {left}，右 {right}）")
                if approach and (state is None or not state.recovery):
                    if state is not None:
                        state.escape_side = 0.0
                    return (approach, 0.0, 0.0, AVOIDING,
                            f"{blocked}；低速接近（停车线 {brake_at:.2f} m）")
                if state is not None:
                    state.escape_side = 0.0
                    state.recovery = True
                return 0.0, 0.0, 0.0, AVOIDING, f"{blocked}；停下"

    elif scan is not None:
        if scan.ready:
            scan.finish()
        clear = ahead is None or ahead >= check_distance
        if scan.has_result and not near and (clear or not foreground):
            _, occupied = scan.grid()
            geometry = config.scan_geometry or config.local_geometry
            front = geometry.front + config.obstacle_stop_m
            corridor = (abs(RIGHT) <= _keepout_m(config)) & (FORWARD >= 0) & (FORWARD <= front)
            clear = not occupied[corridor].any()
        if clear:
            scan.note_clear_path(scan.current_frame)
            if coverage >= config.coverage_min and not scan.result_frame:
                state.side_stall_s.clear()
        if scan.result_frame:
            return 0.0, 0.0, 0.0, AVOIDING, "主动观察：前方走廊尚不可通行"

    (floor_x, limit_x), (floor_y, limit_y) = _local_velocity_limits(config, 1. if vy < 0 else -1.)
    vx = _scaled(vx, scale, floor_x, limit_x)
    vy = _scaled(vy, scale, floor_y, limit_y)
    proposal = None
    if dwa_lateral_routes:
        sides, routes = zip(*dwa_lateral_routes)
        proposal = _dwa_detour(depth, config, state, routes[0], target_distance, target_bearing,
                               alternates=routes[1:])
        if proposal is not None:
            side = sides[proposal.choice.reference_index]
    if state is not None:
        if side != state.escape_side:
            state.side_better_frames = 0
            state.side_compare_frame = depth.get("source_frame_id", depth.get("frame_id"))
        state.escape_side = side
    if proposal is not None:
        return _use_dwa(state, proposal, side)
    if local_command is not None:
        vx, vy = local_command
        wz = 0.0
        state.view_detour = local_route
        if (may_scan and scan is not None and scan.poses and scan.current_frame and not scan.command_only
                and local_route is not None and len(local_route.points) > 1
                and target_distance is not None and target_bearing is not None):
            leg_geometry = local_route.geometry
            goal = PathLeg.world_point(scan.pose, local_path.target_in_grid(
                leg_geometry, target_distance * math.cos(target_bearing),
                target_distance * math.sin(target_bearing)))
            state.detour_leg = state.executing_leg = PathLeg(
                leg_geometry, local_route.points[1], scan.pose, scan.motion_frame, goal, local_command)
        state.avoidance_yaw_override = True
        status, reason = AVOIDING, "局部规划：沿避障路线行进"
    elif side:
        # Blend in the turn as clearance shrinks, keeping yaw towards the
        # chosen route even when the target is on the blocked side.
        weight = 1.0 if near or not aligned else 1.0 - (ahead - stop) / max(1e-6, check_distance - stop)
        yaw_gain = (config.response_yaw_gain if config.response_yaw_gain is not None
                    else config.k_yaw)
        escape_yaw = _clamp(-side * yaw_gain * math.atan2(keep, max(ahead, 0.1)),
                            config.wz_max)
        tracking_yaw = wz if wz * escape_yaw > 0 else 0.0
        wz = (1.0 - weight) * tracking_yaw + weight * escape_yaw
        vx = vy = 0.
        if state is not None:
            state.avoidance_yaw_override = True
        direction = "右" if side > 0 else "左"
        status, reason = AVOIDING, f"走廊内 {ahead:.2f} m 处有障碍，朝{direction}侧避让"
        if 0. < abs(wz) < config.floor_wz:
            rate = math.copysign(config.floor_wz, wz)
            geometry = config.local_geometry
            if (state is not None and geometry is not None and abs(rate) <= _scan_yaw_rate(config, side)
                    and depth.get("map") is not None
                    and depth.get("frame_id") is not None):
                _, hits = state.local_plan.grid_for(depth, geometry)
                if state.active_scan.clear_sweep(geometry, config.half_width_m, STILL_LINEAR * horizon_s,
                                                 occupied=hits):
                    wz = rate
                else:
                    wz = 0.
            else:
                wz = 0.
        if not vy and not wz:
            if state is not None:
                state.recovery = True
            return 0.0, 0.0, 0.0, AVOIDING, "当前控制限制下无法转向避让；停下"

    if (not needs_detour and scan is not None and scan.dwa_failures and (vx or vy or wz)
            and config.local_controller == "dwa" and scan.matches_route_view(depth) and scan.new_dwa_view()):
        scan.dwa_failures = 0
    if state is not None and state.recovery:
        if not side and ahead is not None and ahead <= stop + FORWARD_CLEARANCE_MARGIN_M:
            return 0.0, 0.0, 0.0, AVOIDING, "近障停车，等待通道有足够余量"
        if not near:
            state.blocked_for_s = 0.0
        state.recovery = False
    return vx, vy, wz, status, reason


def _prefix_commands(config, vx, yaw, sides, steer_only):
    """Preserve the legacy arc order; these also seed the DWA command budget."""
    (floor, _), _ = _local_velocity_limits(config, 1.)
    turns = (yaw, 0.) if yaw else (0.,)
    if steer_only:
        turns = yaw, .5 * yaw
    elif sides:
        steer = yaw or -sides[0] * .25
        turns = (steer, .5 * steer, -math.copysign(.25, steer), 0.)
    for fraction in (1., .5, .25):
        speed = vx * fraction
        if speed < floor:
            break
        for requested in turns:
            turn = _clamp(requested, _scan_yaw_rate(config, 1. if requested < 0. else -1.))
            if requested and (not turn or abs(turn) < config.floor_wz_moving):
                continue
            yield speed, 0., turn


def _prefix_points(command, horizon):
    speed, _, turn = command
    if not turn:
        return ((0., 0.), (speed * horizon, 0.))
    radius = speed / turn
    return tuple((radius * math.sin(turn * t), -radius * (1. - math.cos(turn * t)))
                 for t in np.linspace(0., horizon, 6))


def _prefix_references(geometry, commands, horizon):
    """Keep the leading prefix and the first arc in each eligible direction."""
    first_by_turn = {}
    for command in commands:
        turn = command[2]
        direction = math.copysign(1., turn) if turn else 0.
        if direction or not first_by_turn:
            first_by_turn.setdefault(direction, command)
    return tuple(Detour(None, _prefix_points(command, horizon), geometry) for command in first_by_turn.values())


def _checked_forward_prefix(scan, geometry, config, state, odom, motion, vx, horizon, room, yaw, sides=(), *,
                            steer_only=False, shadow=None, depth=None,
                            target_distance=None, target_bearing=None):
    """Select from checked short motions, retaining the legacy fallback order."""
    (floor, limit), _ = _local_velocity_limits(config, 1.)
    if config.local_controller == "dwa":
        commands = tuple(_prefix_commands(config, limit, yaw, sides, steer_only))
    else:
        commands = ()
    if commands:
        references = _prefix_references(geometry, commands, horizon)
        proposed = _dwa_detour(
            depth, config, state, references[0], target_distance, target_bearing,
            branch="prefix", motion=motion,
            candidate_commands=commands, shadow=shadow, alternates=references[1:])
        if proposed is not None:
            side = math.copysign(1., -proposed.command[2]) if proposed.command[2] else state.escape_side
            return _use_dwa(state, proposed, side)
    if vx <= 0. or vx < floor:
        return None
    grid = scan.grid(include_near=scan.near_squares is None)
    for command in _prefix_commands(config, vx, yaw, sides, steer_only):
        speed, _, turn = command
        points = _prefix_points(command, horizon)
        rejection = _checked_motion_rejection(
            scan, geometry, config, state, motion, command, grid, horizon,
            room=room, points=points, shadow=shadow,
            near_motion=planar_feedback(odom) or state.command_tail)
        if rejection is not None:
            continue
        state.view_detour = Detour(None, points, geometry)
        state.checked_prefix = True
        state.avoidance_yaw_override = True
        state.recovery = False
        note = "沿可通行弧段前进" if turn else "沿可通行直行段前进"
        return speed, 0., turn, AVOIDING, f"局部规划：{note}"
    return None


def _advance_after_scan(depth, config, state, odom, ahead, stop, target_distance, dt, *,
                        target_bearing=None):
    scan = state.active_scan
    geometry = (config.scan_geometry or config.local_geometry
                if scan.receipt_only else config.scan_geometry)
    state.avoidance_yaw_override = True
    if geometry is None or not scan.current_frame or target_distance is None:
        return 0., 0., 0., AVOIDING, "主动观察：等候可用的前进观测"

    motion = state.command_tail if scan.command_only else planar_feedback(odom)
    previous = state.commanded
    tolerance = STILL_LINEAR if scan.command_only else config.scan_odom_linear_tolerance_mps
    executing = scan.route_segment and not scan.command_only and scan.advance_origin is not None
    horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
    waypoint_horizon = horizon + dt
    floor = config.floor_vx_moving if config.floor_vx_moving is not None else config.floor_vx
    target_room = target_distance - config.stop_distance_m
    obstacle_room = ahead - stop if ahead is not None else math.inf
    room = min(target_room, obstacle_room)
    speed = min(config.vx_max, max(floor, LOCAL_BLOCKED_APPROACH_MAX_MPS), room / horizon)
    usable_speed = speed > 0. and speed >= floor
    remaining = scan.advance_m
    if config.local_controller == "dwa" and target_bearing is not None:
        endpoint = ((remaining, 0.) if scan.advance_origin is None else PathLeg(
            geometry, (scan.advance_m, 0.), scan.advance_origin, scan.motion_frame).remaining(scan.pose))
        length = math.hypot(*endpoint)
        if endpoint[0] > 0. and length > local_path.RESOLUTION / 2:
            reference = Detour(None, ((0., 0.), endpoint), geometry)
            proposed = _dwa_detour(
                depth, config, state, reference, target_distance, target_bearing,
                branch="advance", motion=motion)
            if proposed is not None:
                return _use_dwa(state, proposed, state.escape_side)
        if not scan.command_only:
            unavailable = _dwa_unavailable_reason(depth, config, state, branch="advance")
            if unavailable is not None:
                state.dwa_trace["fallback_reason"] = unavailable
                return 0., 0., 0., AVOIDING, NO_LOCAL_ROUTE
            return _dwa_recovery_route(depth, config, state, geometry, ahead, target_distance, target_bearing)
    if (motion is None or abs(motion[1]) > tolerance or abs(previous[1]) > STILL_LINEAR
            or (not executing and max(abs(motion[2]), abs(previous[2])) > STILL_YAW)):
        return 0., 0., 0., AVOIDING, "主动观察：等待转向停止"

    if scan.advance_origin is None:
        grid = scan.grid()
        keep = _keepout_m(config)
        corridor_clear = (local_path.segment_clear(grid[1], geometry, keep, 0., scan.advance_m)
                          and local_path.transition_clear(grid, geometry, keep, 1., ((scan.advance_m, 0.),)))
        too_close = not scan.route_segment and obstacle_room <= target_room and not usable_speed
        if not corridor_clear or too_close:
            matched_view = scan.matches_route_view(depth)
            if (not scan.rerouted and not scan.command_only and matched_view
                    and ahead is not None and target_bearing is not None
                    and math.hypot(*motion[:2]) <= tolerance
                    and math.hypot(*previous[:2]) <= STILL_LINEAR):
                rooms = {s: _clearance(depth, config, lateral_offset_m=s * keep) for s in (-1., 1.)}
                frame = depth.get("source_frame_id", depth.get("frame_id"))
                order = _local_side_order(rooms, config, state, frame)
                new_route = _start_detour_segment(
                    depth, config, state, geometry, ahead, order,
                    target_distance, target_bearing, reroute=True)
                if isinstance(new_route, tuple):
                    return new_route
                if new_route:
                    return 0., 0., 0., AVOIDING, "主动观察：原直行走廊被挡，停车对准新路线首段"
            if matched_view or scan.command_only:
                if not scan.route_segment and scan.try_other_side():
                    return 0., 0., 0., AVOIDING, "主动观察：尝试另一侧"
                scan.exhaust()
            else:
                scan.interrupt()
            if corridor_clear:
                return 0., 0., 0., AVOIDING, "主动观察：停车距离不足"
            reason = ("路线首段的直行走廊被挡住" if scan.route_segment
                      else "转向后的直行走廊仍然被挡住")
            return 0., 0., 0., AVOIDING, f"主动观察：{reason}"
    else:
        start_f, start_r, heading = scan.advance_origin
        if scan.command_only:
            scan.advance_commanded_m += max(0., previous[0]) * state.command_dt_s
            progress = scan.advance_commanded_m
        else:
            forward, right, _ = scan.pose
            progress = (forward - start_f) * math.cos(heading) + (right - start_r) * math.sin(heading)
        remaining -= progress
        if remaining <= 0. or (scan.route_segment
                              and remaining <= max(local_path.RESOLUTION / 2, floor * waypoint_horizon)):
            scan.finish()
            return 0., 0., 0., AVOIDING, "主动观察：直行段结束，重新评估前方"

    if (scan.occluded_route and local_path.visible_prefix(
            ((0., 0.), (remaining, 0.)), scan.occlusion(geometry)) < 2):
        scan.hold_for_replan("局部规划：前进段进入遮挡区，等待新视角复检", retry_scan=False)
        return 0., 0., 0., AVOIDING, "局部规划：前进段进入遮挡区，等待新视角复检"

    if not usable_speed or max(motion[0], previous[0]) * horizon > room:
        if (scan.route_segment and scan.advance_origin is not None and not scan.command_only
                and ahead is not None and ahead + FORWARD_CLEARANCE_MARGIN_M < target_distance):
            scan.hold_for_replan("主动观察：直行余量不足，停车后重规划", retry_scan=False)
            return 0., 0., 0., AVOIDING, "主动观察：直行余量不足，停车后重规划"
        return 0., 0., 0., AVOIDING, "主动观察：停车距离不足"
    if scan.route_segment:
        speed = min(speed, remaining / waypoint_horizon)
        if speed < floor:
            scan.exhaust()
            return 0., 0., 0., AVOIDING, "主动观察：路线点距离不足以执行底盘最低速度"
        if max(motion[0], previous[0]) * horizon > remaining + 1e-9:
            return 0., 0., 0., AVOIDING, "主动观察：在路线点前制动"
    grid = scan.grid()
    keep = _keepout_m(config)
    if executing:
        leg = PathLeg(geometry, (scan.advance_m, 0.), scan.advance_origin, scan.motion_frame)
        reason = leg.check(scan.pose, motion, previous[:2], (speed, 0.), grid, keep, horizon,
                           (config.watchdog_ms + 50) / 1000)
        if reason is not None:
            scan.hold_for_replan(reason, retry_scan=False)
            return 0., 0., 0., AVOIDING, reason
    elif not local_path.motion_sweep_clear(grid, geometry, config.half_width_m, horizon,
                                           (motion, previous, (speed, 0., 0.))):
        if scan.advance_origin is None and scan.try_other_side():
            return 0., 0., 0., AVOIDING, "主动观察：尝试另一侧"
        return 0., 0., 0., AVOIDING, "主动观察：直行扫掠区有障碍"
    if scan.advance_origin is None:
        scan.advance_origin = scan.pose
    if scan.route_segment and not scan.command_only:
        state.executing_leg = PathLeg(geometry, (scan.advance_m, 0.), scan.advance_origin, scan.motion_frame)
    return speed, 0., 0., AVOIDING, "主动观察：转稳后直行"


def _local_translation(depth, config, state, odom, ahead, side):
    if state is None or not config.use_lateral or depth.get("capture_invalid"):
        return None
    measured = planar_feedback(odom)
    if measured is None and state.motion_source != "command":
        return None
    predicted = measured is None
    measured = state.command_tail if predicted else measured
    measured_vx, measured_vy, measured_wz = measured
    allow_forward = depth.get("allow_forward", True)
    geometry = config.scan_geometry if "scan_grid" in depth else config.local_geometry
    limits = _local_velocity_limits(config, side)
    floor, limit = limits[1]
    if limit <= 0 or floor > limit:
        return None
    _, (return_floor, return_limit) = _local_velocity_limits(config, -side)
    if state.escape_side == side and any(state.commanded[:2]):
        heading = math.atan2(-state.commanded[1], state.commanded[0])
        depth = dict(depth, preferred_heading=round(heading, 2))
    command = state.local_plan.translation(depth, geometry,
                                           _keepout_m(config), ahead, side, config.depth_age_limit_ms,
                                           config.watchdog_ms,
                                           allow_forward=allow_forward,
                                           allow_return=return_limit > 0 and return_floor <= return_limit,
                                           velocity_limits=limits)
    if state.view_detour is None and state.local_plan.key is not None and (
            state.local_plan.key[0] == depth.get("frame_id")):
        state.view_detour = state.local_plan.plans.get(side)
    if command is None:
        return LocalHold.PREFIX if state.view_detour is not None else None
    if (abs(measured_wz) > .02 or (not allow_forward and abs(measured_vx) > .03)
            or measured_vy * side > .06 or (predicted and measured_vx < -.03)):
        return LocalHold.TRANSITION
    vx, vy = command
    (floor_x, limit_x), (floor_y, limit_y) = limits
    if not (floor_x <= abs(vx) <= limit_x or vx == 0):
        return None
    if not (vy == 0 or floor_y <= -side * vy <= limit_y):
        return None
    speed = math.hypot(vx, vy)
    measured_speed = math.hypot(measured_vx, measured_vy)
    if measured_speed > speed + .02:
        return LocalHold.TRANSITION
    previous = state.commanded[:2]
    drift = speed and abs(measured_vx * vy - measured_vy * vx) / speed > .04
    observed = depth.get("scan_grid") is not None or depth.get("map") is not None
    ramping = predicted or previous != command or math.hypot(measured_vx - vx, measured_vy - vy) > .02
    if depth.get("body_half_width") is not None or drift or (
            geometry is not None and observed and ramping
            and (vx or previous[0]) and (vy or previous[1])):
        grid = depth.get("scan_grid")
        if geometry is None:
            return LocalHold.TRANSITION
        if grid is None:
            if depth.get("map") is None or depth.get("frame_id") is None:
                return LocalHold.TRANSITION
            grid = state.local_plan.grid_for(depth, geometry)
        horizon_s = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
        keepout = _keepout_m(config)

        def clear(scale):
            return local_path.transition_clear(grid, geometry, keepout, horizon_s,
                                               (measured[:2], previous, (vx * scale, vy * scale)))

        if not clear(1.):
            low = max(floor_x / abs(vx) if vx else 0., floor_y / abs(vy) if vy else 0.,
                      (measured_speed - .02) / speed)
            if not clear(low):
                return LocalHold.TRANSITION
            high = 1.
            for _ in range(5):
                scale = (low + high) / 2
                if clear(scale):
                    low = scale
                else:
                    high = scale
            if low <= 0.:
                return LocalHold.TRANSITION
            return vx * low, vy * low
    return (vx, vy)


def _scaled(value: float, factor: float, floor: float, ceiling: float) -> float:
    """Slow an axis down, keeping it executable. Zero stays zero.

    The lift is what makes this honest on a robot with a deadband: scaling 0.4
    by 0.5 asks for a speed R1 does not have, and quietly produces no motion at
    all. Here it produces the slowest motion that exists, and the caller can see
    that it did.
    """
    if not value:
        return 0.0
    return _lift(value * factor, floor, ceiling)


def _search(config: Config, state: State, dt: float, odom=None, *, requested_yaw=None) -> Decision:
    """Target not in frame. Turn towards where it last was, then give up.

    Two ways to stop, and the first is the better one. With odometry the sweep
    is measured in **radians actually turned**, so a full circle means a full
    circle. Without it the only available measure is wall-clock, which gives up
    early on a robot that turns slowly and late on one that spins — a timeout
    is the fallback, not the design.

    That is what this docstring claimed while the code accumulated the
    **commanded** rate, which measures the turn the robot was asked for rather
    than the one it performed. The two diverge exactly where the sweep matters:
    a chassis that refuses the command, a gait that has not started yet, a robot
    held or blocked mid-turn. All of those "complete" a full circle on schedule
    and report `target lost` for something that was behind the robot the whole
    time.

    **Measured on r1_sz, 2026-09-24**, which settled both the direction of the
    error and its size. Neither was guessable beforehand, and they point
    opposite ways:

      - `wz` itself is sound — two full turns in place integrated to 362.9 deg
        and 349.1 deg against a true 360.
      - Under this card's own search command the chassis turns **slower** than
        commanded: about 0.93 rad/s against a commanded 1.0. So the old
        accumulator *over*counted and the sweep ended short — 6.4 rad of
        commanded rotation is about 5.95 rad of real rotation, giving up some
        **30 deg before closing the circle**. A target in that last sector was
        reported `target lost` without ever being looked at, and that sector is
        the far side from `last_seen_side`, which is where a target that went
        missing tends to be.
      - The run that produced those numbers swept 6.5 rad, and a human watching
        it described "a bit more than one turn". 6.5 rad is 372 deg. That
        agreement is the evidence that this counter now means what it says.

    The 30 deg is estimated from second-resolution logs; the honest range is
    10-55 deg. Tightening it needs `searched_rad` sampled at 100 ms through a
    sweep, which costs another spin of the robot and changes no decision here.

    Had the error gone the other way — a chassis turning *faster* than
    commanded — the same fix would instead stop the robot over-rotating, which
    on this robot is not merely slow: `_avoid` never restricts rotation, so
    every extra second of spin sweeps the shoulders through space no corridor
    has checked.
    """
    if requested_yaw is None:
        state.missing_frames += 1
    if requested_yaw is None and state.missing_frames < config.lost_frames:
        # Wait for confirmation before searching. End a translating person
        # approach immediately when the target drops out.
        brake = (config.local_planning and config.local_controller == "dwa" and state.target == "person"
                 and any(state.commanded[:2]))
        return Decision(_twist(0., 0., 0.) if brake else None, SEARCHING,
                        f"target briefly out of sight ({state.missing_frames}/{config.lost_frames} frames)")

    # ``adopt_limits`` has already established that no yaw command can clear
    # this chassis's limits/deadband.  Publishing a zero twist here would keep
    # its watchdog alive while claiming to search for up to the timeout, even
    # though the required motion is impossible.  Fail immediately so the
    # caller can select another strategy rather than waiting for a search that
    # cannot start.
    if requested_yaw is None and config.search_rate <= 0.0:
        state.failed_reason = (
            f"cannot search for {state.target!r}: downstream chassis has no "
            "executable yaw search rate")
        return Decision(None, FAILED, state.failed_reason)

    state.searching_for_s += dt
    # Negated, for the same reason the main loop negates its bearing:
    # `last_seen_side` is +1 when the target was to the *right*, and turning
    # right is *clockwise*, which is negative wz. Without the minus sign the
    # robot sweeps away from the last place it saw the thing — which still
    # finds it eventually, by going all the way round, and so reads as "search
    # is just slow" rather than as a wrong sign.
    wz = -config.search_rate * state.last_seen_side if requested_yaw is None else requested_yaw

    # **The component along the direction we are turning, signed — not `abs`.**
    #
    # Three things make the obvious `abs(measured) * dt` wrong, and all three
    # inflate the sweep, i.e. give up early on a target that is really there:
    #
    #   - yaw rate carries noise, and `abs` rectifies it, so a *stationary*
    #     robot accumulates sweep. R1's `yaw_speed` also has a small zero bias
    #     (~0.12 deg/s measured while standing still), which `abs` cannot
    #     cancel either. Signed accumulation cancels both exactly.
    #   - a rotation in the *opposite* direction would count as progress,
    #     including the robot being turned by hand or shoved.
    #   - body sway during a legged gait reverses sign within a step.
    #
    # The accumulator is floored at zero so an opposing turn cannot bank
    # negative credit that a later real sweep has to pay off.
    measured = _measured_yaw_rate(odom)
    sweep_measured = measured is not None
    direction = 1.0 if wz >= 0 else -1.0
    if sweep_measured:
        state.searched_rad = max(0.0, state.searched_rad
                                 + measured * direction * dt)
    else:
        # No odometry, or a robot that does not measure yaw rate — `None`, never
        # 0.0, which is the whole reason the format forbids reporting zero for an
        # unmeasured axis. Fall back to the commanded rate, which is the turn
        # actually performed only while `search_rate` clears the deadband: below
        # it the command is zeroed downstream and this would count a rotation
        # that never happened. Keeping search_rate above the threshold is what
        # makes the two agree.
        state.searched_rad = max(0., state.searched_rad + state.commanded[2] * direction * state.command_dt_s)

    # Exhausted searches used to go quiet, which left the caller with no answer
    # at all — the task neither succeeded nor failed, and only the ACP timeout
    # eventually noticed. "I turned all the way round and it is not here" is a
    # result, and the caller is owed it.
    if state.searched_rad >= config.search_sweep_rad:
        state.failed_reason = (
            f"swept {state.searched_rad:.1f} rad "
            f"({'measured' if sweep_measured else 'commanded, no odometry'}) "
            f"without a confirmed navigation fix for {state.target!r}: target lost")
        return Decision(None, FAILED, state.failed_reason)
    if state.searching_for_s >= config.search_timeout_s:
        state.failed_reason = (
            f"searched {state.searching_for_s:.0f}s without finding "
            f"{state.target!r}。用 list_visible_objects 看一眼现在认出了什么")
        return Decision(None, FAILED, state.failed_reason)

    return Decision(_twist(0.0, 0.0, wz), SEARCHING,
                    f"target lost; turning towards the side it was last seen")


def _stuck(odom, state: State, commanded_vx: float, dt: float,
           config: Config) -> str:
    """Commanded to move and measurably not moving — we have hit something.

    Returns a reason, or "" if not stuck (or if it cannot be judged).

    **Unwired odometry, and an axis the robot does not measure, both return ""
    rather than "stuck".** They arrive here as `None`, and the whole reason
    `motus.odom/1` forbids reporting `0.0` for an unmeasured axis is this
    function: a robot that cannot answer "am I moving" would otherwise be
    declared stuck on its first step, every time.

    This matters most exactly where the rest of the policy is weakest —
    monocular depth against a flat wall is unreliable, so "the wall is still
    1.5 m away" can persist while the robot leans into it.
    """
    if commanded_vx <= 0.05:
        state.moving_for_s = 0.0
        return ""
    measured = planar_axes(odom)[0]
    if measured is None:
        state.moving_for_s = 0.0
        return ""

    if measured >= config.stuck_speed_ratio * commanded_vx:
        state.moving_for_s = 0.0
        return ""

    state.moving_for_s += dt
    if state.moving_for_s >= config.stuck_window_s:
        return (f"commanded {commanded_vx:.2f} m/s for {state.moving_for_s:.1f}s but "
                f"measured {measured:.2f} m/s: blocked, stopping")
    return ""

# ── The stable object list ───────────────────────────────────────────────────
#
# Single-frame detections flicker. A chair that is really there may appear in
# seven of ten consecutive frames, and one frame will conjure a 0.4-confidence
# "backpack" out of nothing. Hand a single frame to an LLM to choose a target
# from and it will eventually choose something that does not exist in the next
# one — after which the card goes straight into its search behaviour, which
# looks like a robot turning towards a direction nothing was ever in.
#
# So what is exposed is the **intersection over the last N frames**: only what
# keeps appearing counts.

# Target identity. Colour belongs in the key as well as the name, or two chairs
# collapse into one entry and `navigate_to("chair")` reaches whichever of them
# happens to win — which is not a choice the caller made. The colour comes from
# vop's triple, a directly comparable string under `publish_color: name`.
def hue_of(obj) -> str:
    """色相，从 vop 的两种颜色形态里取，取不到返回 ""。

    `publish_color` 有两档会带颜色，形态不同，而两档都是长期支持的：

        name（默认）  "dim muted azure"      —— 三元组字符串，色相是最后一个词
        full          {"dominant_hue": …, …} —— 完整的 12 数 + 4 标签

    早先这里只认第一种，对字典直接做 `str(...).split(" ")[-1]`，于是从字典的
    repr 尾巴上抠出了 `'green'}` 当色相。真机上表现为 list_visible_objects 给出
    的 key 是 `person#'green'}`，照它填进 navigate_to 又对不上（字符会被 shell
    或人手改掉一个），机器人转满一圈报「目标不在视野内」—— 而目标一直在画面正
    中，10/10 帧都看得见。
    """
    colour = obj.get("color")
    if isinstance(colour, dict):
        return str(colour.get("dominant_hue") or "").strip()
    text = str(colour or "").strip()
    return text.split(" ")[-1] if text else ""


def colour_text(obj) -> str:
    """一句能读的颜色，同样两种形态都认。"""
    colour = obj.get("color")
    if isinstance(colour, dict):
        return str(colour.get("color_name") or colour.get("dominant_hue") or "")
    return str(colour or "")


def object_key(obj) -> str:
    name = str(obj.get("name") or "?")
    # Hue only: brightness swings frame to frame with the lighting, and putting
    # it in the identity makes one chair flicker between two entries, neither of
    # which reaches the stability threshold.
    hue = hue_of(obj)
    return f"{name}#{hue}" if hue and hue != "neutral" else name


def describe(obj, distance_m=None) -> str:
    """一句人（和 LLM）能读、也能原样回填给 navigate_to 的描述。"""
    parts = [str(obj.get("name") or "?")]
    colour = colour_text(obj)
    if colour:
        parts.append(colour)
    bearing = _bearing(obj)
    if abs(bearing) <= 0.15:
        where = "ahead"
    else:
        side = "left" if bearing < 0 else "right"
        where = f"far {side}" if abs(bearing) > 0.6 else side
    parts.append(where)
    if distance_m is not None:
        parts.append(f"{distance_m:.1f}m")
    return " · ".join(parts)


def stable_objects(frames, *, min_frames: int = 6, config: Config = None) -> list:
    """在最近这些帧里反复出现的物体。

    `frames` 是最近 N 帧的 vop 载荷，最新的在最后。`min_frames` 是至少要出现在
    几帧里 —— 不要求全部出现过，因为遮挡和边缘抖动会让一个确实在那里的东西漏掉
    一两帧，而把阈值定成「全部」等于把列表清空。

    返回按「出现帧数、置信度」排序，所以最可靠的排在最前 —— LLM 通常取第一个。
    """
    config = config or Config()
    seen: dict = {}
    for index, frame in enumerate(frames):
        for obj in (frame or {}).get("objects") or []:
            if float(obj.get("confidence") or 0) < config.min_confidence:
                continue
            key = object_key(obj)
            entry = seen.setdefault(key, {"frames": set(), "last": obj,
                                          "confidence": 0.0})
            entry["frames"].add(index)
            # Keep the newest frame as the source of bearing: both the object
            # and the robot may be moving, and an older bearing points at where
            # the target no longer is.
            entry["last"] = obj
            entry["confidence"] = max(entry["confidence"],
                                      float(obj.get("confidence") or 0))

    out = []
    for key, entry in seen.items():
        count = len(entry["frames"])
        if count < min_frames:
            continue
        out.append({
            "key": key,
            "name": entry["last"].get("name"),
            "color": entry["last"].get("color"),
            "position": entry["last"].get("position"),
            "bearing": _bearing(entry["last"]),
            "confidence": round(entry["confidence"], 2),
            "seen_in_frames": count,
            "of_frames": len(frames),
            "object": entry["last"],
        })
    out.sort(key=lambda o: (-o["seen_in_frames"], -o["confidence"]))
    return out
