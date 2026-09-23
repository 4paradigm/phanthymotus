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
  5. stop at `stop_distance_m`, and say so

── all three axes move together, and that is the whole point ────────────────

Steps 2 and 3 run on the same tick. The first version of this file ran them in
sequence — `vx = 0` until the target was roughly centred — and a base that can
translate sideways has no reason to do that: the approach velocity is simply
decomposed along the target's bearing (`half_fov_rad`), so the robot walks the
straight line to the thing while turning to face it. What that removes is the
stop-turn-go gait, which on r1_sz read as a lurch at every bearing correction.

── the robot cannot move slowly, and that shapes everything above ───────────

A legged base has to assemble a whole gait cycle, so below some speed it does
not move at all — R1 needs 0.4 m/s and 1.0 rad/s, and under that the SDK accepts
the command, returns 0, and nothing happens. So a proportional law cannot make a
small correction *slowly*; it can only make it *briefly*. Every axis is
therefore a switch with hysteresis (`Gate`) whose magnitude, once on, is lifted
to at least the floor (`_lift`).

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

**When in doubt, emit nothing.** Stale observation, unreadable depth, target
lost for too long, stuck, arrived — all of them return `values=None`, and the
card then publishes nothing at all. The driver's watchdog brings the chassis to
a stop within `watchdog_ms`.

That is a deliberate choice against the alternative, which is to emit an
explicit zero. Emitting zero would also stop the robot, but it would keep the
watchdog fed, so a policy that silently died would leave a robot that believes
it is being driven. Silence is the one signal a broken upstream cannot fake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .camera import half_fov as _resolve_half_fov
from .track import Tracker

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
    slow_distance_m: float = 1.5
    obstacle_stop_m: float = 0.6
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
    # How long to keep trying to align, once inside the stop distance, before
    # accepting arrival anyway. Spinning for ever in front of the target is a
    # far worse outcome than facing it a little off: the position is what
    # arriving means, the heading is a courtesy.
    arrive_patience_s: float = 3.0
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
    k_yaw: float = 1.2
    k_fwd: float = 0.6
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
    # Searching turns at this rate. It must clear the robot's deadband or the
    # search commands are snapped to zero and the robot simply stands there
    # while the card reports "searching" — which is what happened on r1_sz once
    # the deadband handling went in: R1 needs 1.0 rad/s and this was 0.4.
    #
    # Every speed in this file has to be read against `min_magnitude`; a value
    # that looks conservatively slow may be one the robot cannot execute at all.
    search_rate: float = 1.0
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
    # Added to the declared half-width before anything is checked. The footprint
    # a chassis declares is its **static envelope with the arms at rest**, and
    # what hits a doorframe is a swinging arm and a leg mid-stride. This is the
    # difference between "the box the robot is" and "the space to keep clear".
    clearance_margin_m: float = 0.15
    # How much of the corridor has to carry a real depth reading before its
    # clearance is believed. Below `coverage_min` the card stops going forward
    # and says so; between the two it slows in proportion.
    #
    # **Unknown is not free.** A depth map has holes, and the things that make
    # them — chair legs, table edges, glass — are exactly the things that catch
    # a shoulder. Without this a corridor full of holes reads as an empty one,
    # because every invalid pixel drops silently out of the minimum.
    coverage_min: float = 0.25
    coverage_full: float = 0.60

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
    tracker: Tracker = field(default_factory=Tracker)
    # What we last asked the chassis to do. Stands in for odometry when none is
    # wired: with a deadband the robot either does roughly the commanded speed
    # or nothing at all, so the command is a serviceable proxy — much better
    # than assuming we are stationary while turning at 1 rad/s.
    commanded: tuple = (0.0, 0.0, 0.0)
    missing_frames: int = 0
    last_seen_side: float = 1.0          # +1 target was right, -1 it was left
    searching_for_s: float = 0.0
    searched_rad: float = 0.0
    commanded_vx: float = 0.0
    moving_for_s: float = 0.0
    close_for_s: float = 0.0
    idle_for_s: float = 0.0
    arrived: bool = False
    blocked_for_s: float = 0.0
    # Set once, read forever after: failure is terminal for one navigate_to.
    failed_reason: str = ""
    notes: list = field(default_factory=list)


@dataclass
class Decision:
    """One tick's answer. `values` is None when nothing should be published."""

    values: list | None
    status: str
    reason: str = ""
    distance_m: float | None = None
    bearing: float | None = None

    @property
    def publishes(self) -> bool:
        return self.values is not None


def apply_deadband(values, min_magnitude) -> list:
    """Lift each axis out of the robot's deadband, or drop it to zero.

    Some robots do nothing at all below a threshold — a legged base has to
    assemble a whole gait cycle, so there is no "creep slowly" regime. R1 needs
    0.4 m/s and 1.0 rad/s. Below that the SDK accepts the command, returns 0,
    and the robot stands still, so a policy emitting a smooth ramp towards zero
    spends its entire life commanding motion and producing none — with every
    layer in between reporting success. That is exactly how it presented on
    r1_sz: 159 commands applied, no errors anywhere, a motionless robot.

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
            f"卡片配置里的 {previous:.3f} 不再生效。走廊宽度按这个数算，"
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

    Two directions, and both were real failures on r1_sz:

    * **Up, off the deadband.** `wz_max` defaulted to 0.8 against a 1.0 rad/s
      floor, so the policy's entire output range was unexecutable and every turn
      command snapped to 0 or ±1.0. Nothing reported anything — the SDK accepted
      all of it.
    * **Down, into the descriptor's limits.** A ceiling above `limits.upper` is
      a command `ControlSink` rejects, at the full command rate, for the whole
      run.
    """
    notes: list[str] = []
    notes.extend(_adopt_footprint(config, descriptor))
    limits = (descriptor or {}).get("limits") or {}
    floors = list(limits.get("min_magnitude") or [])
    lower = list(limits.get("lower") or [])
    upper = list(limits.get("upper") or [])

    def at(row, index):
        try:
            return abs(float(row[index]))
        except (IndexError, TypeError, ValueError):
            return None

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
            moving = at(list(limits.get("min_magnitude_moving") or []), 5)
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
        if bound is not None and at(lower, index) == 0.0 and bound == 0.0:
            if getattr(config, ceiling_attr) > 0:
                notes.append(f"下游底盘没有 {name} 这个自由度，已停用该轴")
            setattr(config, ceiling_attr, 0.0)
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

    return notes


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
    # Summary-only: the best available is the band the target sits in, which is
    # the nearest obstacle there and not the target itself. Usable for "am I
    # about to arrive", misleading for anything finer.
    bands = depth.get("bands") or {}
    return bands.get(_band_of(_bearing(obj)))


def _band_of(bearing: float) -> str:
    if bearing < -1 / 3:
        return "left"
    if bearing > 1 / 3:
        return "right"
    return "center"


def step(*, detections, depth, odom, config: Config, state: State,
         dt: float) -> Decision:
    decision = _account_idle(_decide(detections, depth, odom, config, state, dt),
                             state, config, dt)
    # Remember what we asked for, so the tracker can compensate for our own
    # motion on the next tick even with no odometry wired.
    if decision.values is not None:
        state.commanded = (decision.values[0], decision.values[1],
                           decision.values[5])
    else:
        state.commanded = (0.0, 0.0, 0.0)
    return decision


def _ego_twist(odom, state: State):
    """Our own motion, for the tracker's predict step. Returns (twist, measured).

    Measured if odometry is wired, commanded otherwise. **Per axis**, because
    `motus.odom/1` reports an unmeasured axis as `None` rather than 0 — a robot
    that reports yaw rate but not lateral speed should have its yaw believed and
    only its `vy` guessed.

    `measured` is true only when **every** axis came from odometry. A partly
    guessed twist is a guessed twist as far as the filter's confidence goes, and
    erring towards "guessed" only makes it trust its own predictions less.
    """
    commanded = state.commanded
    if not odom:
        return commanded, False
    out, measured = [], True
    for index, name in enumerate(("vx", "vy", "wz")):
        value = odom.get(name)
        if value is None:
            measured = False
            out.append(commanded[index])
        else:
            out.append(float(value))
    return tuple(out), measured


def _decide(detections, depth, odom, config: Config, state: State,
            dt: float) -> Decision:
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

    # The tracker runs **before** the blind check, so a spell with no
    # observation is a run of misses rather than time that did not happen. It
    # is also the only way the coast clock advances while the card is blind —
    # otherwise a camera that stopped publishing would leave a track coasting
    # for ever, and the robot driving towards a prediction nobody is checking.
    ego, ego_measured = _ego_twist(odom, state)
    track = state.tracker.step(detections=detections, depth=depth,
                               target=state.target, config=config,
                               ego=ego, ego_measured=ego_measured, dt=dt)

    # No usable observation is not the same as "nothing is there". Emitting
    # nothing lets the watchdog stop the robot, which is the right resting
    # state for "this policy cannot see".
    if detections is None or depth is None:
        return Decision(None, BLIND,
                        "observation stale or missing (vop / depth); publishing nothing, "
                        "leaving the stop to the downstream watchdog")

    bands = depth.get("bands") or {}

    if track is None:
        return _search(config, state, dt)

    state.missing_frames = 0
    state.searching_for_s = 0.0
    state.searched_rad = 0.0
    # Back to the normalised offset the rest of this file is written in.
    # `track.py` works in metres and radians; every threshold here — align_tol,
    # arrive_align_tol, lateral_max_bearing — is a fraction of the image half
    # width, and converting them instead would change what every deployed
    # config means.
    bearing = _clamp(track.bearing_rad / max(1e-6, config.half_fov_rad), 1.0)
    state.last_seen_side = 1.0 if bearing >= 0 else -1.0
    # None, not a number, when depth has never had a reading for this object.
    # The summary-only path legitimately produces that, and the whole approach
    # law below already handles an unknown distance — what it must not do is
    # act on the placeholder the tracker needed in order to exist.
    distance = track.range_m if track.range_known else None

    # Turn towards it, proportionally. Positive bearing means the target is
    # right of centre, and `wz` is positive counter-clockwise, so the sign is
    # inverted here.
    #
    # **Left unquantised for now.** Whether this has to be lifted to a deadband
    # depends on something not known yet at this point — whether the robot ends
    # up translating — so the decision is deferred to `_finalise_yaw` once vx is
    # settled. See `floor_wz_moving`.
    raw_wz = _clamp(-config.k_yaw * bearing, config.wz_max)

    within = distance is not None and distance <= config.stop_distance_m
    if within:
        state.close_for_s += dt
    else:
        state.close_for_s = 0.0

    # Arrived: in position and roughly facing it — or turning in place for long
    # enough. The second clause is not optional; see `arrive_patience_s`.
    # **Arriving is a claim, and a coasted position cannot support it.**
    #
    # "I have reached the target" and "something is in the way" are different
    # conclusions, and until now the first could impersonate the second: the
    # arrival test runs before `_avoid`, so a track sitting on an obstacle
    # reported success while the real target was metres away. On r1_sz that
    # happened twice in a row — the track had been resurrected onto a traffic
    # cone and the card declared it had arrived.
    #
    # A prediction is a good enough reason to keep walking and a bad one to
    # declare the journey over, so arrival needs a live fix. Without one the
    # code below falls through to the obstacle logic, which will stop for the
    # thing that is actually there and say so.
    live = track.observed
    if within and live and (abs(bearing) <= config.arrive_align_tol
                            or state.close_for_s >= config.arrive_patience_s):
        state.arrived = True
        # One explicit zero before going quiet, so the chassis stops on a
        # command rather than on a watchdog timeout — arriving is a success and
        # should not look like a dropped link in the driver's log.
        return Decision(_twist(0.0, 0.0, 0.0), ARRIVED,
                        f"target at {distance:.2f} m, stop distance reached"
                        + ("" if abs(bearing) <= config.arrive_align_tol else
                           f" (still {bearing:+.2f} off-centre after "
                           f"{state.close_for_s:.1f}s in range; not waiting to "
                           f"align — see arrive_patience_s)"),
                        distance_m=distance, bearing=bearing)

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
        # Turning in place: the standing floor applies, so this is the one path
        # that still needs the gate and the lift.
        wz = _finalise_yaw(raw_wz, False, bearing, config, state, dt)
        return Decision(_twist(0.0, 0.0, wz), ALIGNING,
                        f"目标偏离画面中心 {bearing:+.2f}（归一化），先原地转正",
                        distance_m=distance, bearing=bearing)

    speed = config.vx_max * align
    status, reason = APPROACHING, "approaching"
    if distance is not None:
        speed = min(speed, max(0.0, config.k_fwd
                               * (distance - config.stop_distance_m) * align))
        reason = f"target at {distance:.2f} m"

    if within and not live:
        # Inside the stop distance by the tracker's reckoning, but the position
        # is a prediction. Do not walk into whatever is actually there, and do
        # not call it an arrival either — hold still and let `_avoid` below
        # name the thing in the corridor, if there is one.
        speed = 0.0
        status = SEARCHING
        reason = ("目标看起来已经在 {:.2f} m 内，但当前并没有真正看到它"
                  "（轨迹处于预测状态），不按到达处理").format(distance)

    # Keep walking until the stop distance is genuinely reached.
    #
    # `speed` falls below the robot's floor well before then — at R1's numbers,
    # 0.67 m before — so without this gate the robot stopped short of a target
    # it could see perfectly well and then failed on the idle timeout. The gate
    # engages on metres remaining, and `_lift` supplies the only speed the robot
    # has for the final stretch.
    remaining = 99.0 if distance is None else distance - config.stop_distance_m
    driving = state.fwd_gate.update(
        remaining, engage=0.05, release=0.0,
        dt=dt, min_dwell_s=config.min_dwell_s)

    vx, vy = _approach(speed if driving else 0.0, bearing, depth, config,
                       state, dt)
    vx, vy, raw_wz, status, reason = _avoid(vx, vy, raw_wz, depth, config,
                                            status, reason)

    # **Blocked is an outcome, not a mood.** Trying to get round something is
    # the right first response, and it is what `_avoid` just did — but a robot
    # that has been shuffling sideways for twelve seconds is not making
    # progress, and the caller deserves the real reason rather than a timeout
    # or, worse, a report that it arrived.
    if status == AVOIDING and not vx:
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
    wz = _finalise_yaw(raw_wz, bool(vx or vy), bearing, config, state, dt)

    stuck = _stuck(odom, state, vx, dt, config)
    if stuck:
        state.failed_reason = stuck
        return Decision(None, FAILED, stuck, distance_m=distance, bearing=bearing)

    state.commanded_vx = vx
    return Decision(_twist(vx, vy, wz), status, reason,
                    distance_m=distance, bearing=bearing)


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
    floor = config.floor_wz_moving if translating else config.floor_wz
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
    return _lift(raw, floor, config.wz_max) if turning else 0.0


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


def _side_label(room, coverage, config: Config) -> str:
    """One side's verdict, for a reason string an operator has to act on."""
    if coverage < config.coverage_min:
        return f"只有 {coverage * 100:.0f}% 深度有效"
    if room is None:
        return "无读数"
    if room <= config.slow_distance_m:
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
    return depth_mod.corridor(depth["map"],
                              half_width_m=_keepout_m(config),
                              half_fov_rad=config.half_fov_rad,
                              reference_m=config.obstacle_stop_m,
                              lateral_offset_m=lateral_offset_m)


def _may_strafe(towards_right: bool, depth, config: Config) -> bool:
    """Whether it is honest to put speed on `vy` right now.

    The question is not "is the right third of the picture empty" — that
    describes what is ahead-and-to-the-right at a couple of metres, not what is
    beside the shoulder. It is whether the corridor the robot would **move
    into** is both measured and clear. An unmeasured one blocks: sideways is
    the direction a forward-facing camera knows least about, and this is the one
    place the policy could move into space it cannot see.

    A sidestep whose corridor falls outside the lens gets zero coverage and is
    refused on that alone, which is the correct answer rather than a special
    case.
    """
    if not config.use_lateral or config.vy_max <= 0:
        return False
    offset = _keepout_m(config) * (1.0 if towards_right else -1.0)
    room, coverage = _clearance(depth, config, lateral_offset_m=offset)
    if coverage < config.coverage_min:
        return False
    return room is None or room > config.slow_distance_m


def _approach(speed: float, bearing: float, depth, config: Config,
              state: "State" = None, dt: float = 0.0):
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
    forward = _lift(speed * math.cos(angle), config.floor_vx, config.vx_max)

    if abs(bearing) > config.lateral_max_bearing:
        # Too far off to trust either the bearing or the side band. Turn.
        return forward, 0.0
    lateral = -speed * math.sin(angle) * config.lateral_scale
    if lateral == 0.0 or not _may_strafe(lateral < 0, depth, config):
        return forward, 0.0
    if not _worth_strafing(lateral, config, state, dt):
        return forward, 0.0
    return forward, _lift(lateral, config.floor_vy, config.vy_max)


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
    if config.floor_vy <= 1e-6:
        return True                      # no deadband, no amplification, no gate
    engage = config.floor_vy * config.lateral_lift_ratio
    if state is None:
        return abs(lateral) >= engage
    return state.vy_gate.update(
        abs(lateral), engage=engage, release=engage * config.release_frac,
        dt=dt, min_dwell_s=config.min_dwell_s)


def _avoid(vx, vy, wz, depth, config: Config, status, reason):
    """Slow for what is ahead, and move out from in front of it if it is close.

    Backing away is not available — the depth map covers what the camera sees
    and nothing behind the robot, so reversing is moving blind.

    When something is close enough to stop for, the robot both turns towards
    the freer side **and** steps towards it. Turning alone only changes where it
    is pointing; on a base that can translate, the sidestep is what actually
    gets it out of the way, and doing both at once is one continuous motion
    rather than a pirouette followed by a walk.
    """
    bands = (depth or {}).get("bands") or {}
    ahead, coverage = _clearance(depth, config)

    # Too little of the corridor measured to say anything about it. Not the same
    # as "clear", and it used to be treated as such — an invalid pixel simply
    # dropped out of the minimum, so a corridor full of holes and an empty one
    # produced the same number. The objects that make holes are the ones that
    # catch a shoulder.
    if coverage < config.coverage_min:
        return (0.0, 0.0, wz, AVOIDING,
                f"正前方只有 {coverage * 100:.0f}% 的深度有效，看不清就不往前走"
                f"（需要 {config.coverage_min * 100:.0f}%）")

    # Between the two thresholds the clearance is believed, but less of it than
    # we would like was measured, so it is approached more slowly.
    trust = min(1.0, max(0.0, (coverage - config.coverage_min)
                         / max(1e-6, config.coverage_full - config.coverage_min)))

    if ahead is None:
        return _scaled(vx, trust, config.floor_vx, config.vx_max), \
                _scaled(vy, trust, config.floor_vy, config.vy_max), \
                wz, status, reason

    if ahead <= config.obstacle_stop_m:
        # **Which way out, decided on metric corridors — not on the angular
        # thirds.**
        #
        # This branch was the other half of the bug the metric corridor was
        # introduced to fix. "Is the way ahead blocked" became metric; "which way
        # do I go" stayed on `bands`, which describe what is ahead-and-to-the-side
        # at a couple of metres rather than what is beside the shoulder. A
        # **doorway** beside the path is the worst case for that: you can see
        # through it, so it reads as the *emptiest* band, and the robot steers
        # into the one thing in reach.
        #
        # That is what happened on r1_sz. The robot could have walked straight
        # through a 1.1 m door; instead it reached the doorway, deliberately
        # turned towards it, and put a shoulder into the frame — at
        # `wz_max` = 1.5 rad/s and `vy_max` = 1.0 m/s, because the escape was
        # full-scale and unconditional.
        #
        # So both sides are now judged by the corridor the robot would actually
        # move into, which is the same question `_may_strafe` asks, and the turn
        # is refused outright when neither side is measured and clear.
        keep = _keepout_m(config)
        left_room, left_cov = _clearance(depth, config, lateral_offset_m=-keep)
        right_room, right_cov = _clearance(depth, config, lateral_offset_m=keep)

        def _room(room, coverage):
            """How much room a side offers, or None if the answer is not usable.

            Unmeasured is not free. Sideways is the direction a forward-facing
            camera knows least about, and this is the one place the policy moves
            into space it cannot see.
            """
            if coverage < config.coverage_min:
                return None
            if room is None:
                return float("inf")
            return room if room > config.slow_distance_m else None

        options = {-1.0: _room(left_room, left_cov),
                   1.0: _room(right_room, right_cov)}
        usable = {sign: room for sign, room in options.items() if room is not None}

        if not usable:
            # Nowhere to go that we can see. Keep the target-derived yaw — it at
            # least points at the target — and do not spin towards a band: a
            # walking humanoid rotating at wz_max sweeps its shoulders through
            # space no corridor checked, which is exactly how one found a
            # doorframe.
            return (0.0, 0.0, wz, AVOIDING,
                    f"走廊内 {ahead:.2f} m 处有障碍，两侧也都不可用"
                    f"（左 {_side_label(left_room, left_cov, config)}，"
                    f"右 {_side_label(right_room, right_cov, config)}）；停下")

        sign = max(usable, key=lambda k: usable[k])
        # Turn towards the gap rather than at full scale. The gap sits at
        # `atan(keep / ahead)` off the nose, so this is the same proportional law
        # the main loop uses applied to the angle that actually has to be closed
        # — bounded, derived, and about a third of what wz_max was commanding.
        gap_angle = math.atan2(keep, max(ahead, 0.1))
        wz = _clamp(-sign * config.k_yaw * gap_angle, config.wz_max)
        # `sign` indexes the **offset corridor** (+1 = the one displaced to the
        # right), while the body frame has y pointing **left** — so the sidestep
        # takes the opposite sign. Getting this wrong steps into the obstacle it
        # just decided to avoid, at full speed, and every other signal in the
        # decision still looks right.
        vy = (_lift(-sign * config.vy_max * config.lateral_scale,
                    config.floor_vy, config.vy_max)
              if _may_strafe(sign > 0, depth, config) else 0.0)
        side = "右" if sign > 0 else "左"
        return (0.0, vy, wz, AVOIDING,
                f"走廊内 {ahead:.2f} m 处有障碍；停止前进，朝{side}侧"
                + ("让开并转向" if vy else "转向"))

    if ahead < config.slow_distance_m:
        span = max(1e-6, config.slow_distance_m - config.obstacle_stop_m)
        scale = max(0.0, (ahead - config.obstacle_stop_m) / span)
        # Both translation axes scale together, so slowing down does not also
        # change the direction of travel.
        scale *= trust
        return (_scaled(vx, scale, config.floor_vx, config.vx_max),
                _scaled(vy, scale, config.floor_vy, config.vy_max), wz, AVOIDING,
                f"{reason}；走廊内 {ahead:.2f} m 处有障碍，减速")

    return (_scaled(vx, trust, config.floor_vx, config.vx_max),
            _scaled(vy, trust, config.floor_vy, config.vy_max), wz,
            status, reason)


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


def _search(config: Config, state: State, dt: float) -> Decision:
    """Target not in frame. Turn towards where it last was, then give up.

    Two ways to stop, and the first is the better one. With odometry the sweep
    is measured in **radians actually turned**, so a full circle means a full
    circle. Without it the only available measure is wall-clock, which gives up
    early on a robot that turns slowly and late on one that spins — a timeout
    is the fallback, not the design.
    """
    state.missing_frames += 1
    if state.missing_frames < config.lost_frames:
        # Briefly occluded — a person walking past. Emitting nothing for a few
        # frames lets the watchdog hold rather than starting a search for
        # something that has not actually gone.
        return Decision(None, SEARCHING,
                        f"target briefly out of sight ({state.missing_frames}/{config.lost_frames} frames)")

    state.searching_for_s += dt
    # Negated, for the same reason the main loop negates its bearing:
    # `last_seen_side` is +1 when the target was to the *right*, and turning
    # right is *clockwise*, which is negative wz. Without the minus sign the
    # robot sweeps away from the last place it saw the thing — which still
    # finds it eventually, by going all the way round, and so reads as "search
    # is just slow" rather than as a wrong sign.
    wz = -config.search_rate * state.last_seen_side
    # Accumulated from the commanded rate, which is only the turn actually
    # performed while `search_rate` clears the deadband — below it the command
    # is zeroed downstream and this would count a rotation that never happened.
    # Keeping search_rate above the threshold is what makes the two agree.
    state.searched_rad += abs(wz) * dt

    # Exhausted searches used to go quiet, which left the caller with no answer
    # at all — the task neither succeeded nor failed, and only the ACP timeout
    # eventually noticed. "I turned all the way round and it is not here" is a
    # result, and the caller is owed it.
    if state.searched_rad >= config.search_sweep_rad:
        # The suggestion is part of the answer, not politeness. A detector's
        # class for one object is not stable — the same fire extinguisher on
        # r1_sz was reported 373 times as `fire extinguisher` in one recording
        # and as `bottle` twenty minutes later — so "not found" far more often
        # means the name does not match than that the thing is absent. A caller
        # told only "lost" retries the same wrong name or gives up.
        state.failed_reason = (
            f"swept {state.searched_rad:.1f} rad without seeing "
            f"{state.target!r}: target lost. "
            f"检测模型对同一物体的类别并不稳定，「没找到」多半是名字对不上 —— "
            f"用 list_visible_objects 看一眼现在认出了什么，再和用户确认")
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
    if odom is None:
        return ""
    measured = odom.get("vx")
    if measured is None:
        return ""

    if abs(measured) >= config.stuck_speed_ratio * commanded_vx:
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
