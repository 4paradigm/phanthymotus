"""The navigation control law, branch by branch.

`policy.step` is a pure function precisely so that "what does it do when the
target disappears behind a person while an obstacle is on the left" is a test
and not an afternoon on a robot. Everything here runs with no ROS, no camera and
no chassis.

The tests that matter most are the ones asserting **nothing is published**.
Publishing a command is the easy path to get right; the failure that hurts is a
policy that keeps talking when it can no longer see, because a fed watchdog is
a robot that believes it is being driven.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import depth as D  # noqa: E402
from plugins.navi import policy as P  # noqa: E402


def _cfg(**over):
    return P.Config(**over)


def _obj(name="chair", x=0.0, y=0.0, confidence=0.9, bbox=None):
    out = {"name": name, "position": [x, y], "confidence": confidence}
    if bbox:
        out["bbox_norm"] = bbox
    return out


# Detection payloads carry a stamp and the tracker counts one payload as one
# piece of evidence however many times it is read — so a test that means "the
# next frame" has to produce a new stamp. A constant here made every tick a
# re-read of the same frame, and nothing was ever confirmed.
_FRAME = [0.0]


def _detections(*objects):
    _FRAME[0] += 0.1
    return {"timestamp": round(_FRAME[0], 4), "count": len(objects),
            "objects": list(objects)}


def _next_frame(payload):
    """The same picture, taken again. A new stamp, same contents."""
    if payload is None:
        return None
    _FRAME[0] += 0.1
    return dict(payload, timestamp=round(_FRAME[0], 4))


def _depth(bands=None, map_=None):
    return {"map": map_, "bands": bands if bands is not None
            else {"left": 5.0, "center": 5.0, "right": 5.0}}


def _state(target="chair", **over):
    state = P.State(target=target)
    for key, value in over.items():
        setattr(state, key, value)
    return state


class Sim:
    """A world with one target in it, and a robot that does what it is told.

    Hand-written sequences of bearings describe a target that teleports, and
    since the tracker went in it **correctly refuses to believe them** — a jump
    of 1 m in 100 ms is 10 m/s, which is not a chair. Several tests here were
    quietly passing for that reason: the assertion held because the track had
    coasted, not because the control law did anything.

    So the target is placed in the world and the robot is integrated with the
    twist the policy actually emitted. Two things follow that matter more than
    the tidiness: rotation genuinely moves the target across the frame (which is
    the whole premise of the ego-motion compensation), and the approach
    converges or fails to for real reasons.

    The kinematics here are written independently of `track._predict` — world
    pose forward, rather than body-frame inverse — so a sign error in one does
    not cancel against the other.
    """

    def __init__(self, range_m=3.0, bearing=0.0, config=None, velocity=(0.0, 0.0),
                 confidence=0.9, name="chair"):
        self.config = config or _cfg()
        angle = bearing * self.config.half_fov_rad
        # World frame starts aligned with the robot: x forward, y left.
        self.target = np.array([range_m * np.cos(angle),
                                -range_m * np.sin(angle)])
        self.velocity = np.array(velocity, dtype=float)
        self.pose = np.array([0.0, 0.0, 0.0])      # x, y, theta
        self.confidence = confidence
        self.name = name
        self.visible = True

    # ── what the robot can see ───────────────────────────────────────────────

    def relative(self):
        dx, dy = self.target - self.pose[:2]
        theta = self.pose[2]
        cos, sin = np.cos(-theta), np.sin(-theta)
        bx = cos * dx - sin * dy
        by = sin * dx + cos * dy
        return float(np.hypot(bx, by)), float(np.arctan2(-by, bx))

    def observe(self):
        range_m, angle = self.relative()
        depth = _depth(map_=_solid_depth(range_m))
        if not self.visible:
            return _detections(), depth
        bearing = angle / self.config.half_fov_rad
        return (_detections(_obj(name=self.name, x=bearing,
                                 confidence=self.confidence)), depth)

    # ── what the robot does about it ─────────────────────────────────────────

    def apply(self, values, dt):
        if values is not None:
            theta = self.pose[2]
            cos, sin = np.cos(theta), np.sin(theta)
            vx, vy = values[0], values[1]
            self.pose[0] += (cos * vx - sin * vy) * dt
            self.pose[1] += (sin * vx + cos * vy) * dt
            self.pose[2] += values[5] * dt
        self.target += self.velocity * dt

    def run(self, state, ticks, config=None, dt=0.1, odom=True):
        """Closed loop. Returns the last decision."""
        config = config or self.config
        decision = None
        for _ in range(ticks):
            detections, depth = self.observe()
            measured = None
            if odom and decision is not None and decision.values is not None:
                measured = {"vx": decision.values[0], "vy": decision.values[1],
                            "wz": decision.values[5]}
            elif odom:
                measured = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            decision = P.step(detections=detections, depth=depth, odom=measured,
                              config=config, state=state, dt=dt)
            self.apply(decision.values, dt)
        return decision


def seed(state, detections, depth=None, config=None, frames=None, dt=0.1):
    """Give the tracker enough frames to confirm, without running the policy.

    Almost every test in this file is about the **control law**, and the control
    law only ever sees a confirmed track — a track needs `confirm_hits` frames
    before it may move a robot at all. Seeding the tracker directly rather than
    stepping the policy keeps every other piece of state (gates, arrival
    patience, idle clock) untouched, so these stay single-tick tests of one
    thing. Confirmation itself is tested in its own section below.
    """
    config = config or _cfg()
    for _ in range(config.confirm_hits if frames is None else frames):
        state.tracker.step(detections=_next_frame(detections), depth=depth,
                           target=state.target, config=config,
                           ego=(0.0, 0.0, 0.0), dt=dt)
    return state


def _step(detections=None, depth=None, odom=None, config=None, state=None,
          dt=0.1):
    config = config or _cfg()
    if state is None:
        # Single-tick call: seed so the assertion is about the control law.
        # A caller that supplies its own state is running a sequence and seeds
        # it itself — seeding on every tick of a loop would keep resetting the
        # very lifecycle the loop is exercising.
        state = seed(_state(), detections, depth, config)
    return P.step(detections=detections, depth=depth, odom=odom,
                  config=config, state=state, dt=dt)


# ── silence is the safe answer ───────────────────────────────────────────────

def test_a_stale_observation_publishes_nothing():
    """The single most important behaviour here. Not a zero command — nothing:
    a zero would keep feeding the downstream watchdog, so a policy that died
    would leave a robot that believes it is still being driven."""
    decision = _step(detections=None, depth=_depth())
    assert decision.publishes is False
    assert decision.status == P.BLIND


def test_missing_depth_publishes_nothing_even_with_a_visible_target():
    decision = _step(detections=_detections(_obj()), depth=None)
    assert decision.publishes is False
    assert decision.status == P.BLIND


def test_no_target_set_publishes_nothing():
    decision = _step(detections=_detections(_obj()), depth=_depth(),
                     state=_state(target=""))
    assert decision.publishes is False
    assert decision.status == P.IDLE


# ── turning and approaching ──────────────────────────────────────────────────

def test_a_target_to_the_right_turns_clockwise():
    """Positive bearing means right of centre; wz is positive counter-clockwise,
    so it must come out negative. Getting this sign backwards produces a robot
    that turns away from whatever it is looking for, and it is the one quantity
    here that cannot be settled by reading the driver's code."""
    decision = _step(detections=_detections(_obj(x=0.5)), depth=_depth())
    assert decision.values[5] < 0


def test_a_target_to_the_left_turns_counter_clockwise():
    assert _step(detections=_detections(_obj(x=-0.5)), depth=_depth()).values[5] > 0


def test_a_misaligned_target_is_approached_and_turned_to_at_once():
    """The point of the rewrite. Three axes on one tick.

    The first version stopped dead (`vx = 0`) until the target was centred, then
    walked — a stop-turn-go gait that on r1_sz read as a lurch at every bearing
    correction. A base that can translate has no reason for it: the approach
    velocity is decomposed along the bearing, so the robot walks the straight
    line to the target *while* turning to face it.
    """
    decision = _step(detections=_detections(_obj(x=0.5)), depth=_depth())
    assert decision.values[0] > 0, "still walking"
    assert decision.values[1] < 0, "and leaning right, towards the target"
    assert decision.values[5] < 0, "and turning to face it, on the same tick"


def test_a_base_that_cannot_strafe_falls_back_to_turning_first():
    """`align_min_scale: 0` restores the original behaviour, for a chassis with
    no lateral degree of freedom. The old gate is a configuration now, not a
    law."""
    decision = _step(detections=_detections(_obj(x=0.5)), depth=_depth(),
                     config=_cfg(align_min_scale=0.0, use_lateral=False))
    assert decision.values[0] == 0.0
    assert decision.values[1] == 0.0
    assert decision.status == P.ALIGNING


def test_an_aligned_target_is_approached():
    decision = _step(detections=_detections(_obj(x=0.01)), depth=_depth())
    assert decision.values[0] > 0
    assert decision.status == P.APPROACHING


def test_vy_points_at_the_target_and_is_zero_dead_ahead():
    assert _step(detections=_detections(_obj(x=0.0)), depth=_depth()).values[1] == 0.0
    assert _step(detections=_detections(_obj(x=0.5)), depth=_depth()).values[1] < 0
    assert _step(detections=_detections(_obj(x=-0.5)), depth=_depth()).values[1] > 0


def test_strafing_checks_observed_obstacles_with_an_unmeasured_side():
    blocked = _scene(5.0, [(0.55, 1.0, 0.5)])
    unmeasured = {"map": _obstacle_map(5.0), "bands": {"left": 5.0, "center": 5.0}}
    unmeasured["map"][:, 340:] = np.nan
    turn_away = _step(detections=_detections(_obj(x=0.5)), depth=blocked)
    assert turn_away.values[:2] == [0.0, 0.0]
    assert turn_away.values[5] > 0  # The verified route goes left.
    strafe = _step(detections=_detections(_obj(x=0.5)), depth=unmeasured)
    assert strafe.values[1] < 0.0
    assert strafe.values[5] < 0


def test_a_target_too_far_off_axis_is_turned_to_rather_than_strafed_at():
    """At the frame edge the bearing is least trustworthy and so is the band
    that would have to clear the sidestep."""
    decision = _step(detections=_detections(_obj(x=0.95)), depth=_depth(),
                     config=_cfg(lateral_max_bearing=0.7))
    assert decision.values[1] == 0.0


def test_the_twist_is_six_wide_in_control_order():
    values = _step(detections=_detections(_obj(x=0.3)), depth=_depth()).values
    assert len(values) == 6


# ── arriving ─────────────────────────────────────────────────────────────────

def test_arriving_emits_one_explicit_zero_then_goes_quiet():
    """Arriving is a success and should stop the chassis on a command, not on a
    watchdog timeout — the latter reads as a dropped link in the driver's log."""
    depth = _depth(map_=_solid_depth(0.8))
    detections = _detections(_obj(x=0.0, bbox=(0.4, 0.4, 0.6, 0.6)))
    state = seed(_state(), detections, depth)
    first = _step(detections=detections, depth=depth, state=state)
    assert first.status == P.ARRIVED
    assert first.values == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    second = _step(detections=_detections(_obj(x=0.0)), depth=depth, state=state)
    assert second.publishes is False
    assert second.status == P.ARRIVED


def test_it_does_not_arrive_while_still_misaligned():
    """Stopping at the right distance but pointing elsewhere is not arrival."""
    state = _state()
    decision = _step(detections=_detections(_obj(x=0.6, bbox=(0.7, 0.4, 0.9, 0.6))),
                     depth=_depth(map_=_solid_depth(0.8)), state=state)
    assert decision.status != P.ARRIVED
    assert state.arrived is False


@pytest.mark.parametrize("distance,arrived", [(1.25, True), (1.3, True), (1.301, False)])
def test_arrival_tolerance_accepts_only_distances_within_the_extended_threshold(distance, arrived):
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.1)
    decision = _step(detections=_detections(_obj(bbox=(0.4, 0.4, 0.6, 0.6))),
                     depth=_depth(map_=_solid_depth(distance)), config=config)
    assert (decision.status == P.ARRIVED) is arrived
    if arrived:
        assert decision.values == [0.0] * 6
        assert "threshold 1.30 m" in decision.reason
        assert "0.10 m tolerance" in decision.reason


def test_default_tolerance_is_five_centimetres():
    config = _cfg(stop_distance_m=1.2)
    decision = _step(detections=_detections(_obj()),
                     depth=_depth(map_=_solid_depth(1.25)), config=config)
    assert config.stop_tolerance_m == 0.05
    assert decision.status == P.ARRIVED


def test_repeated_detection_cannot_confirm_arrival_from_commanded_advance():
    state = seed(_state(), _detections(_obj()), _depth(map_=_solid_depth(.95)))
    detections = _detections(_obj())
    depth = _depth(map_=_solid_depth(.95))
    _step(detections=detections, depth=depth, state=state)
    for _ in range(4):
        state.commanded = (.4, 0., 0.)
        result = _step(detections=detections, depth=depth, state=state)
        assert result.status != P.ARRIVED
    assert state.tracker.track.range_m < P.Config().arrival_distance_m


def test_fresh_detection_with_invalid_target_depth_cannot_confirm_arrival():
    state = seed(_state(), _detections(_obj()), _depth(map_=_solid_depth(.80)))
    result = _step(detections=_detections(_obj()), state=state,
                   depth=_depth(map_=np.full((480, 640), np.nan)))
    assert result.status != P.ARRIVED


def test_reported_stale_capture_holds_even_with_fresh_detection():
    state = seed(_state(), _detections(_obj()), _depth(map_=_solid_depth(2.5)))
    depth = dict(_depth(map_=_solid_depth(2.5)), capture_invalid=True)
    result = _step(detections=_detections(_obj()), depth=depth, state=state)
    assert result.status == P.BLIND and result.values == [0.] * 6


def test_explicit_zero_tolerance_keeps_the_nominal_distance_threshold():
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.0)
    decision = _step(detections=_detections(_obj()),
                     depth=_depth(map_=_solid_depth(1.25)), config=config)
    assert decision.status != P.ARRIVED


def test_tolerance_arrival_emits_one_zero_and_stays_complete_if_distance_changes():
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.1)
    detections = _detections(_obj(bbox=(0.4, 0.4, 0.6, 0.6)))
    depth = _depth(map_=_solid_depth(1.25))
    state = seed(_state(), detections, depth, config)
    first = _step(detections=detections, depth=depth, state=state, config=config)
    assert first.status == P.ARRIVED and first.values == [0.0] * 6
    later = _step(detections=_next_frame(detections),
                  depth=_depth(map_=_solid_depth(1.35)), state=state, config=config)
    assert later.status == P.ARRIVED and later.values is None


def test_tolerance_band_stops_translation_while_existing_heading_patience_runs():
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.1,
                  arrive_align_tol=0.01, arrive_patience_s=0.5)
    detections = _detections(_obj(x=0.25, bbox=(0.5, 0.4, 0.7, 0.6)))
    depth = _depth(map_=_solid_depth(1.25))
    state = seed(_state(), detections, depth, config)
    first = _step(detections=detections, depth=depth, state=state, config=config)
    assert first.status != P.ARRIVED
    assert first.values[:2] == [0.0, 0.0]
    for _ in range(8):
        decision = _step(detections=_next_frame(detections), depth=depth,
                         state=state, config=config)
        if decision.status == P.ARRIVED:
            break
        assert decision.values[:2] == [0.0, 0.0]
    assert decision.status == P.ARRIVED and decision.values == [0.0] * 6


def test_tolerance_does_not_complete_a_coasted_target_or_missing_observation():
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.1)
    detections = _detections(_obj(bbox=(0.4, 0.4, 0.6, 0.6)))
    depth = _depth(map_=_solid_depth(1.25))
    state = seed(_state(), detections, depth, config)
    prediction = _step(detections=_detections(), depth=depth, state=state, config=config)
    assert state.tracker.track is not None and not state.tracker.track.observed
    assert prediction.status != P.ARRIVED and not state.arrived
    blind = _step(detections=None, depth=depth, state=state, config=config)
    assert blind.status == P.BLIND and blind.values is None and not state.arrived


@pytest.mark.parametrize("tolerance", [0.0, 0.1])
def test_tolerance_hold_also_stops_avoidance_translation_without_disabling_yaw(monkeypatch, tolerance):
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=tolerance,
                  arrive_align_tol=0.01, arrive_patience_s=5.0)

    def clearance(depth, config, lateral_offset_m=0.0):
        # A near pillar in the forward corridor, and clear space to its side.
        return (0.4 if lateral_offset_m == 0 else 5.0), 1.0

    monkeypatch.setattr(P, "_clearance", clearance)
    decision = _step(detections=_detections(_obj(x=0.25, bbox=(0.5, 0.4, 0.7, 0.6))),
                     depth=_depth(map_=_solid_depth(1.25)), config=config)
    assert decision.status != P.ARRIVED and decision.values[5] != 0
    assert decision.values[:2] == [0., 0.]


def test_approach_finishes_in_tolerance_band_without_creeping_to_nominal_distance():
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.1, k_fwd=0.1)
    sim = Sim(range_m=1.6, config=config)
    state = seed(_state(), *sim.observe(), config)
    for _ in range(500):
        decision = sim.run(state, ticks=1, dt=0.1, odom=False)
        if state.arrived:
            break
    assert state.arrived
    assert config.stop_distance_m + 0.08 < sim.relative()[0] <= config.arrival_distance_m + 0.01


@pytest.mark.parametrize("key", ["stop_distance_m", "stop_tolerance_m"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, None, "bad"])
def test_arrival_distances_reject_invalid_configuration(key, value):
    with pytest.raises(ValueError, match=key):
        _cfg(**{key: value})


def test_arrival_distance_sum_cannot_overflow_to_infinity():
    with pytest.raises(ValueError, match="must be finite"):
        _cfg(stop_distance_m=1e308, stop_tolerance_m=1e308)


# ── obstacles ────────────────────────────────────────────────────────────────

# These pass a depth *map* (which is where the target's own distance comes
# from) alongside explicitly supplied *bands* (which is where obstacles come
# from). On a robot the two are computed from the same frame and agree; feeding
# them separately here is what isolates the avoidance branch from the arrival
# branch, which would otherwise both fire on the same number.

def test_a_near_obstacle_with_an_open_side_stops_forward_before_turning():
    far = _step(detections=_detections(_obj(x=0.0)), depth=_scene(4.0))
    near = _step(detections=_detections(_obj(x=0.0)),
                 depth=_scene(4.0, [(0.25, 1.0, 0.35)]))
    assert far.values[0] > 0. and near.values[:2] == [0., 0.]
    assert near.values[5] > 0. and near.status == P.AVOIDING


def test_an_obstacle_inside_the_stop_distance_halts_forward_motion():
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_scene(4.0, [(0.0, 0.4, 0.4)]))
    assert decision.values[0] == 0.0
    assert decision.status == P.AVOIDING


def test_it_turns_towards_the_side_the_map_says_is_free():
    """The way out is chosen by the corridor the robot would move **into**.

    It used to be chosen by the angular thirds, which describe what is
    ahead-and-to-the-side at a couple of metres rather than what is beside the
    shoulder — the other half of the bug the metric corridor was introduced to
    fix, left behind when "is it blocked" was converted and "which way out" was
    not. See the doorway test below for what that cost.
    """
    # The obstacle sits off-axis, so exactly one offset corridor clears it.
    to_the_right = _step(detections=_detections(_obj(x=0.0)),
                         depth=_depth(map_=_obstacle_map(5.0, [(0.25, 0.6, 0.3)])))
    to_the_left = _step(detections=_detections(_obj(x=0.0)),
                        depth=_depth(map_=_obstacle_map(5.0, [(-0.25, 0.6, 0.3)])))

    assert to_the_right.values[5] > 0, "障碍在右，应当朝左转（逆时针为正）"
    assert to_the_left.values[5] < 0
    assert to_the_right.values[:2] == to_the_left.values[:2] == [0., 0.]


def test_a_doorway_beside_the_path_is_not_mistaken_for_the_way_out():
    """r1_sz, 2026-09-23: the robot could have walked straight through a 1.1 m
    door. Instead it reached the doorway, deliberately turned towards it, and put
    a shoulder into the frame.

    A doorway is the worst case for choosing by the angular bands, because you
    can **see through it** — so it reads as the emptiest band while being the one
    thing in reach. Here the bands say the right side is wide open and the map
    says the right side is a wall at 0.5 m; the map has to win.
    """
    decision = _step(
        detections=_detections(_obj(x=0.0)),
        depth=_depth({"left": 2.0, "center": 0.5, "right": 9.0},
                     map_=_obstacle_map(5.0, [(0.0, 0.5, 0.25), (0.6, 0.5, 0.5)])))

    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0
    assert decision.values[1] >= 0.0, "朝右让开就是撞门框那一下"
    assert decision.values[5] >= 0.0, "更不该朝右转"


def test_the_escape_turn_is_aimed_at_the_gap_rather_than_slammed_to_the_limit():
    """It used to command `wz_max` outright — 1.5 rad/s on r1_sz, unrelated to
    how blocked it was or where the target was. That is a lunge, not an evasion,
    and a walking humanoid rotating that fast sweeps its shoulders through space
    no corridor checked.

    The gap sits at `atan(keep / ahead)` off the nose, so the turn closes that
    angle with the same proportional law the main loop uses.
    """
    config = _cfg()
    decision = _step(detections=_detections(_obj(x=0.0)), config=config,
                     depth=_depth(map_=_obstacle_map(5.0, [(0.25, 0.6, 0.3)])))
    expected = config.k_yaw * math.atan2(P._keepout_m(config), 0.6)
    assert abs(decision.values[5]) == pytest.approx(min(expected, config.wz_max),
                                                    abs=1e-6)
    assert abs(decision.values[5]) < config.wz_max


def test_with_no_usable_side_it_stops_instead_of_spinning_towards_a_band():
    """Nowhere to go that it can see. Keeping the target-derived yaw is right;
    spinning towards whichever third looks emptier is how the shoulder found the
    doorframe."""
    # `stop_distance_m` pinned under the wall's range, or the target — whose
    # distance is read from that same wall — counts as reached and arrival
    # pre-empts the whole branch.
    config = _cfg(stop_distance_m=0.3)
    decision = _step(detections=_detections(_obj(x=0.0)), config=config,
                     depth=_depth({"left": 9.0, "center": 0.5, "right": 9.0},
                                  map_=_obstacle_map(0.5, [])))
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0 and decision.values[1] == 0.0
    assert "两侧也都不可用" in decision.reason


def test_the_summary_only_path_cannot_separate_target_from_obstacle():
    """A band reading is enough to stop but cannot confirm the target range."""
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth({"left": 5.0, "center": 0.4, "right": 5.0}))
    assert decision.status == P.AVOIDING
    assert decision.values == [0.] * 6


def test_unknown_bands_do_not_invent_an_obstacle():
    """A band with no valid depth is None, not zero. Treating it as zero would
    report an obstacle touching the robot and stop it permanently."""
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth({"left": None, "center": None, "right": None}))
    assert decision.values[0] > 0
    assert decision.status == P.APPROACHING


def test_it_never_reverses():
    """Depth covers what the camera sees and nothing behind the robot."""
    for center in (0.1, 0.4, 0.6, 1.0, 5.0):
        decision = _step(detections=_detections(_obj(x=0.0)),
                         depth=_depth({"left": 1.0, "center": center, "right": 1.0}))
        if decision.publishes:
            assert decision.values[0] >= 0.0


# ── losing the target ────────────────────────────────────────────────────────

def test_a_brief_occlusion_does_not_start_a_search():
    """Somebody walking past should not send the robot spinning."""
    state = _state()
    config = _cfg(lost_frames=10)
    for _ in range(9):
        decision = _step(detections=_detections(), depth=_depth(),
                         config=config, state=state)
        assert decision.publishes is False
        assert decision.status == P.SEARCHING


def test_a_lost_target_is_searched_for_towards_where_it_last_was():
    state = _state(last_seen_side=-1.0, missing_frames=20)
    decision = _step(detections=_detections(), depth=_depth(), state=state)
    assert decision.publishes is True
    assert decision.values[0] == 0.0          # never drive while searching
    assert decision.values[5] > 0             # towards the left, where it was


def test_the_search_gives_up_after_a_full_sweep():
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20)
    for _ in range(12):
        decision = _step(detections=_detections(), depth=_depth(),
                         config=config, state=state, dt=0.1)
    assert decision.publishes is False
    assert "rad" in decision.reason


def test_the_search_falls_back_to_a_timeout_without_odometry():
    """Radians are the better measure but need odometry; wall-clock is the
    fallback, not the design."""
    config = _cfg(search_timeout_s=0.5, search_sweep_rad=1e6)
    state = _state(missing_frames=20)
    for _ in range(10):
        decision = _step(detections=_detections(), depth=_depth(),
                         config=config, state=state, dt=0.1)
    assert decision.publishes is False
    assert "searched" in decision.reason


def test_the_sweep_counts_radians_actually_turned_when_odometry_says_so():
    """The point of the sweep: a full circle means a full circle.

    Here the robot is commanded to turn and reports turning at half the
    commanded rate — a gait still starting, a chassis clipping the command, a
    robot leaning on something. Counting the command would declare a full sweep
    on schedule and report `target lost` for a thing that was never passed.
    """
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20)
    for _ in range(12):
        decision = _step(detections=_detections(), depth=_depth(),
                         odom={"vx": 0.0, "vy": 0.0, "wz": -0.5},
                         config=config, state=state, dt=0.1)
    # Commanded would have reached 1.2 rad and failed; measured is at 0.6.
    assert decision.publishes is True
    assert state.searched_rad == pytest.approx(0.6, abs=0.01)


@pytest.mark.parametrize("measured", [
    None, True, False, "-0.5", "bad", float("nan"), float("inf"),
    -float("inf"), [], {}, pytest.param(10 ** 400, id="overflow_int"),
])
def test_invalid_search_yaw_falls_back_to_commanded_sweep(measured):
    config = _cfg(search_rate=1.0, search_sweep_rad=0.25, search_timeout_s=30)
    state = _state(missing_frames=20, searched_rad=0.1)
    for expected in (0.2, 0.3):
        decision = _step(detections=_detections(), depth=_depth(),
                         odom={"vx": 0.0, "vy": 0.0, "wz": measured},
                         config=config, state=state, dt=0.1)
        assert math.isfinite(state.searched_rad)
        assert state.searched_rad == pytest.approx(expected)
        if expected == 0.2:
            assert decision.status == P.SEARCHING and decision.publishes
            assert decision.values[5] == -1.0
    assert decision.status == P.FAILED and not decision.publishes
    assert "commanded, no odometry" in decision.reason


def test_a_robot_that_does_not_turn_never_completes_a_sweep():
    """The failure this is for: the sweep ending without the robot moving."""
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20)
    for _ in range(50):
        decision = _step(detections=_detections(), depth=_depth(),
                         odom={"vx": 0.0, "vy": 0.0, "wz": 0.0},
                         config=config, state=state, dt=0.1)
    assert state.searched_rad == 0.0
    assert decision.status != P.FAILED or "searched" in decision.reason


def test_yaw_noise_and_bias_do_not_accumulate_a_phantom_sweep():
    """Why the accumulation is signed rather than `abs`.

    `abs(measured)` rectifies noise, so a robot standing still banks sweep it
    never performed — and R1's `yaw_speed` carries a small zero bias on top
    (~0.12 deg/s measured while stationary), which `abs` cannot cancel either.
    Every such error inflates the sweep, i.e. gives up early on a target that is
    really there.
    """
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20)
    for tick in range(100):
        _step(detections=_detections(), depth=_depth(),
              odom={"vx": 0.0, "vy": 0.0, "wz": 0.3 * (-1) ** tick},
              config=config, state=state, dt=0.1)
    assert state.searched_rad < 0.05


def test_turning_the_wrong_way_does_not_count_as_progress():
    """Being shoved, or turned by hand, is not searching."""
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20, last_seen_side=1.0)
    for _ in range(10):
        _step(detections=_detections(), depth=_depth(),
              # Commanded wz is negative here (towards the right); this is the
              # opposite sign, so it must not bank credit — nor go negative.
              odom={"vx": 0.0, "vy": 0.0, "wz": +1.0},
              config=config, state=state, dt=0.1)
    assert state.searched_rad == 0.0


def test_the_failure_says_which_measure_ended_the_sweep():
    """"Swept a full circle" and "was told to" are different claims."""
    config = _cfg(search_sweep_rad=0.5, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20)
    for _ in range(12):
        decision = _step(detections=_detections(), depth=_depth(),
                         odom={"vx": 0.0, "vy": 0.0, "wz": -1.0},
                         config=config, state=state, dt=0.1)
    assert "measured" in decision.reason

    state = _state(missing_frames=20)
    for _ in range(12):
        decision = _step(detections=_detections(), depth=_depth(),
                         config=config, state=state, dt=0.1)
    assert "no odometry" in decision.reason


def test_an_axis_the_robot_does_not_measure_falls_back_to_the_command():
    """`None` is not `0.0`, here as everywhere.

    A robot reporting forward speed but not yaw rate must keep the commanded
    sweep, not freeze at zero radians for ever — which is what reading `wz` as
    0.0 would do, turning "cannot answer" into "never turned".
    """
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, search_timeout_s=1e6)
    state = _state(missing_frames=20)
    for _ in range(12):
        decision = _step(detections=_detections(), depth=_depth(),
                         odom={"vx": 0.2, "vy": None, "wz": None},
                         config=config, state=state, dt=0.1)
    assert decision.publishes is False
    assert "no odometry" in decision.reason


def test_seeing_the_target_again_resets_the_search():
    state = _state(missing_frames=20, searching_for_s=5.0, searched_rad=3.0)
    seed(state, _detections(_obj(x=0.0)), _depth())
    _step(detections=_detections(_obj(x=0.0)), depth=_depth(), state=state)
    assert state.missing_frames == 0
    assert state.searching_for_s == 0.0
    assert state.searched_rad == 0.0


# ── stuck detection, and the null-versus-zero rule it rests on ───────────────

def test_commanded_but_not_moving_is_stuck():
    config = _cfg(stuck_window_s=0.3)
    state = _state()
    for _ in range(5):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         odom={"vx": 0.0}, config=config, state=state, dt=0.1)
    # STUCK is a *reason*; the terminal status the caller sees is FAILED.
    assert decision.status == P.FAILED
    assert "blocked" in decision.reason
    assert decision.publishes is False


def test_an_unmeasured_axis_never_reports_stuck():
    """The whole reason motus.odom/1 forbids 0.0 for an unmeasured axis. A
    robot that cannot answer "am I moving" must not be declared stuck on its
    first step — which is exactly what would happen if None became 0.0."""
    config = _cfg(stuck_window_s=0.3)
    state = _state()
    for _ in range(20):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         odom={"vx": None}, config=config, state=state, dt=0.1)
        assert decision.status != P.STUCK


def test_unwired_odometry_never_reports_stuck():
    config = _cfg(stuck_window_s=0.3)
    state = _state()
    for _ in range(20):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         odom=None, config=config, state=state, dt=0.1)
        assert decision.status != P.STUCK


def test_a_robot_that_is_moving_is_not_stuck():
    config = _cfg(stuck_window_s=0.3)
    state = _state()
    for _ in range(20):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         odom={"vx": 0.35}, config=config, state=state, dt=0.1)
        assert decision.status != P.STUCK


def test_standing_still_on_purpose_is_not_stuck():
    """While turning in place, vx is commanded zero — measuring zero then is
    agreement, not a collision."""
    config = _cfg(stuck_window_s=0.3)
    state = _state()
    for _ in range(20):
        decision = _step(detections=_detections(_obj(x=0.9)), depth=_depth(),
                         odom={"vx": 0.0}, config=config, state=state, dt=0.1)
        assert decision.status != P.STUCK


# ── target selection ─────────────────────────────────────────────────────────

def test_an_exact_name_beats_a_substring():
    """`navigate_to("chair")` must not prefer "chairman" over "chair"."""
    chosen = P.select_target(
        [_obj(name="chairman", confidence=0.99), _obj(name="chair", confidence=0.5)],
        "chair", _cfg())
    assert chosen["name"] == "chair"


def test_a_substring_is_used_when_there_is_no_exact_match():
    chosen = P.select_target([_obj(name="office chair")], "chair", _cfg())
    assert chosen["name"] == "office chair"


def test_low_confidence_detections_are_ignored():
    assert P.select_target([_obj(confidence=0.1)], "chair",
                           _cfg(min_confidence=0.35)) is None


def test_among_equals_the_most_central_is_chosen():
    """Keeps the choice stable frame to frame when two of the same thing are in
    view — flipping between them would make the robot oscillate."""
    chosen = P.select_target(
        [_obj(x=0.8, confidence=0.9), _obj(x=0.1, confidence=0.9)], "chair", _cfg())
    assert chosen["position"][0] == 0.1


def test_an_empty_target_name_selects_nothing():
    assert P.select_target([_obj()], "", _cfg()) is None


# ── helpers ──────────────────────────────────────────────────────────────────

def _solid_depth(metres):
    from plugins.navi import depth as D
    return np.full((D.HEIGHT, D.WIDTH), metres, dtype=np.float32)


def _obstacle_map(background=5.0, patches=(), rows=(290, 400), config=None):
    """A depth map with rectangular patches placed at a metric position.

    `patches` are `(lateral_m, distance_m, width_m)`, lateral **right-positive**
    at that distance — which is how the corridor thinks, and what turns "0.30 m
    to the left, 0.7 m away" into a picture instead of a column index.

    Rows default to the **lower** part of `OBSTACLE_ROW_SPAN` — inside what the
    obstacle scan looks at, and clear of the centre window `sample_point` reads
    the target's own distance from. Otherwise a scene cannot hold a target and
    an obstacle at once: the thing in the corridor would simply be the target,
    and the card would report arrival instead of avoidance. Physically this is
    a low obstacle (a box on the floor) between the robot and a distant target.
    """
    from plugins.navi import depth as D

    config = config or _cfg()
    out = np.full((D.HEIGHT, D.WIDTH), float(background), dtype=np.float32)
    tangent = ((np.arange(D.WIDTH) - (D.WIDTH - 1) / 2.0)
               * (2.0 * np.tan(config.half_fov_rad) / D.WIDTH))
    for lateral, distance, width in patches:
        columns = np.abs(tangent * distance - lateral) <= width / 2.0
        out[rows[0]:rows[1], columns] = distance
    return out


def _scene(background=5.0, patches=(), config=None):
    """A depth map **and** the bands derived from it, the way the card sees them.

    `_depth(bands=..., map_=...)` lets the two disagree, which is useful for
    isolating one path; this is for tests that want the picture to be coherent.
    """
    from plugins.navi import depth as D

    depth_map = _obstacle_map(background, patches, config=config)
    return {"map": depth_map, "bands": D.nearest_by_band(depth_map)}


# ── stable object list ───────────────────────────────────────────────────────
#
# What `list_visible_objects` is built on. The point is the intersection: a
# single frame is a bad thing to choose a navigation target from, because a
# detection that appears once and vanishes sends the robot turning towards
# something that was never there.

def _frames(*per_frame):
    return [_detections(*objs) for objs in per_frame]


def test_an_object_in_every_frame_is_stable():
    frames = _frames(*[[_obj(name="chair")] for _ in range(10)])
    out = P.stable_objects(frames, min_frames=6)
    assert [o["name"] for o in out] == ["chair"]
    assert out[0]["seen_in_frames"] == 10


def test_a_one_frame_flicker_is_dropped():
    """The whole reason this function exists."""
    frames = _frames(*[[_obj(name="chair")] for _ in range(9)])
    frames.append(_detections(_obj(name="chair"), _obj(name="backpack")))
    out = P.stable_objects(frames, min_frames=6)
    assert [o["name"] for o in out] == ["chair"]


def test_an_object_missing_a_couple_of_frames_still_counts():
    """Occlusion and edge flicker drop a frame or two from something that is
    really there; requiring every frame would empty the list."""
    frames = _frames(*([[_obj(name="chair")]] * 7 + [[]] * 3))
    assert [o["name"] for o in P.stable_objects(frames, min_frames=6)] == ["chair"]


def test_results_are_ordered_most_reliable_first():
    frames = _frames(*[[_obj(name="chair"), _obj(name="tv")] for _ in range(6)])
    frames += _frames(*[[_obj(name="chair")] for _ in range(4)])
    assert [o["name"] for o in P.stable_objects(frames, min_frames=5)][0] == "chair"


def test_low_confidence_detections_never_become_stable():
    frames = _frames(*[[_obj(name="chair", confidence=0.1)] for _ in range(10)])
    assert P.stable_objects(frames, min_frames=6, config=_cfg()) == []


# ── identity: colour is part of it ───────────────────────────────────────────

def test_two_chairs_of_different_colours_are_two_entries():
    """Without this the list has one "chair" and navigate_to("chair") picks
    whichever one happens to win, which is not a choice the caller made."""
    frames = _frames(*[[dict(_obj(name="chair", x=-0.5), color="dim muted azure"),
                        dict(_obj(name="chair", x=0.5), color="bright vivid red")]
                       for _ in range(10)])
    keys = {o["key"] for o in P.stable_objects(frames, min_frames=6)}
    assert keys == {"chair#azure", "chair#red"}


def test_brightness_changes_do_not_split_one_object_in_two():
    """Only the hue is part of the identity: brightness swings frame to frame
    with the lighting, and including it would make one chair flicker between
    two entries and neither would reach the stability threshold."""
    frames = _frames(*[[dict(_obj(name="chair"),
                             color=("dim muted azure" if i % 2 else
                                    "bright muted azure"))]
                       for i in range(10)])
    out = P.stable_objects(frames, min_frames=6)
    assert len(out) == 1 and out[0]["key"] == "chair#azure"


def test_a_neutral_colour_does_not_enter_the_key():
    """"neutral" says the hue is meaningless, so putting it in the key would be
    inventing a distinction from the absence of one."""
    frames = _frames(*[[dict(_obj(name="wall"), color="dim gray neutral")]
                       for _ in range(10)])
    assert P.stable_objects(frames, min_frames=6)[0]["key"] == "wall"


def test_the_key_round_trips_through_select_target():
    """`list_visible_objects` hands out keys and `navigate_to` takes them back;
    if these two disagree the whole feature is decorative."""
    red = dict(_obj(name="chair", x=0.5), color="bright vivid red")
    blue = dict(_obj(name="chair", x=-0.5), color="dim muted azure")
    chosen = P.select_target([red, blue], "chair#red", _cfg())
    assert chosen is red


def test_a_plain_name_still_works():
    assert P.select_target([_obj(name="chair")], "chair", _cfg()) is not None


def test_the_latest_frames_bearing_is_used():
    """Both the robot and the object may be moving; an old bearing points at
    where the target no longer is."""
    frames = _frames(*[[_obj(name="chair", x=-0.5)] for _ in range(9)])
    frames.append(_detections(_obj(name="chair", x=0.4)))
    assert P.stable_objects(frames, min_frames=6)[0]["bearing"] == 0.4


def test_describe_mentions_name_colour_and_side():
    text = P.describe(dict(_obj(name="chair", x=0.5), color="dim muted azure"), 2.3)
    assert "chair" in text and "azure" in text and "right" in text and "2.3m" in text


def test_the_three_distances_keep_their_ordering():
    """obstacle_stop < stop_distance < slow_distance.

    The target is itself an obstacle — it shows up in the same depth bands — so
    if `obstacle_stop_m` ever reaches `stop_distance_m`, the robot is halted by
    the very thing it is walking towards and never arrives. Raising one of these
    without the others is the natural way to break it.
    """
    c = P.Config()
    assert c.obstacle_stop_m < c.stop_distance_m < c.slow_distance_m


def test_a_target_at_the_stop_distance_is_not_halted_by_itself():
    """The ordering above, exercised rather than asserted."""
    c = P.Config()
    depth = _depth({"left": 5.0, "center": c.stop_distance_m, "right": 5.0},
                   map_=_solid_depth(c.stop_distance_m))
    detections = _detections(_obj(x=0.0))
    state = seed(_state(), detections, depth, c)
    decision = _step(detections=detections, depth=depth, config=c, state=state)
    assert decision.status == P.ARRIVED


# ── terminal conditions ──────────────────────────────────────────────────────

def test_idle_for_too_long_fails_the_task():
    """The replacement for a per-phase wall clock: no motion commanded for
    `idle_timeout_s` means the task has stopped making progress, whatever the
    reason — and one test covers blind, occluded and refused all at once."""
    config = _cfg(idle_timeout_s=1.0)
    state = _state()
    for _ in range(12):
        decision = _step(detections=None, depth=_depth(),   # blind
                         config=config, state=state, dt=0.1)
    assert decision.status == P.FAILED
    assert "no motion command issued" in decision.reason


def test_a_slow_but_moving_approach_never_times_out():
    """A robot walking steadily towards something far away is working. A clock
    on 'how long has this navigate_to run' would kill it for succeeding slowly."""
    config = _cfg(idle_timeout_s=1.0)
    state = _state()
    for _ in range(200):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         config=config, state=state, dt=0.1)
    assert decision.status == P.APPROACHING
    assert state.idle_for_s == 0.0


def test_turning_counts_as_motion():
    """Turning in place is motion. A base configured not to strafe spends whole
    seconds doing only that, and must not be failed for it."""
    config = _cfg(idle_timeout_s=1.0, align_min_scale=0.0, use_lateral=False)
    state = _state()
    for _ in range(50):
        decision = _step(detections=_detections(_obj(x=0.9)), depth=_depth(),
                         config=config, state=state, dt=0.1)
    assert decision.status == P.ALIGNING
    assert decision.values[0] == 0.0 and decision.values[5] != 0.0
    assert state.idle_for_s == 0.0


def test_a_brief_blind_spell_does_not_fail():
    config = _cfg(idle_timeout_s=5.0)
    state = seed(_state(), _detections(_obj(x=0.0)), _depth(), config)
    # Shorter than max_coast_s: a brief blind spell is exactly what the coast
    # is for, and the track has to survive it for this to test idleness at all.
    for _ in range(10):
        _step(detections=None, depth=_depth(), config=config, state=state, dt=0.1)
    decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                     config=config, state=state, dt=0.1)
    assert decision.status == P.APPROACHING
    assert state.idle_for_s == 0.0


def test_an_exhausted_search_fails_rather_than_going_quiet():
    """It used to just stop emitting, leaving the caller with no answer at all
    — neither success nor failure, only the ACP timeout eventually noticing."""
    config = _cfg(search_sweep_rad=1.0, search_rate=1.0, idle_timeout_s=1e6)
    state = _state(missing_frames=20)
    for _ in range(15):
        decision = _step(detections=_detections(), depth=_depth(),
                         config=config, state=state, dt=0.1)
    assert decision.status == P.FAILED
    assert "chair" in decision.reason


def test_failure_is_terminal_and_keeps_its_reason():
    """A failed task must not quietly re-enter the loop on the next frame."""
    state = _state(failed_reason="测试原因")
    for _ in range(5):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         state=state)
        assert decision.status == P.FAILED and decision.reason == "测试原因"


def test_arriving_is_not_counted_as_idleness():
    """Arriving commands a zero on purpose; it is a success, not a stall."""
    config = _cfg(idle_timeout_s=0.2)
    depth = _depth(map_=_solid_depth(0.8))
    state = seed(_state(), _detections(_obj(x=0.0)), depth, config)
    for _ in range(10):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=depth,
                         config=config, state=state, dt=0.1)
    assert decision.status == P.ARRIVED


# ── colour comes in two shapes, and both are supported for good ──────────────

_FULL_COLOUR = {"rgb_mean": [72.2, 62.8, 60.3], "rgb_var": [1624.0, 1178.6, 1030.5],
                "hsv_mean": [54.1, 46.3, 72.9], "hsv_var": [4403.8, 1105.3, 1571.8],
                "dominant_hue": "green", "dominant_saturation": "muted",
                "dominant_brightness": "dim", "color_name": "green"}


def test_the_hue_is_read_from_the_full_colour_dict():
    """`publish_color: full` is a supported level, not a legacy accident.

    Treating the dict as a string and splitting on spaces lifts `'green'}` off
    the end of its repr — which on r1_sz produced the key `person#'green'}`,
    unusable in navigate_to, and a robot that searched a full circle for
    something sitting in the middle of its view.
    """
    assert P.hue_of({"color": _FULL_COLOUR}) == "green"
    assert P.object_key({"name": "person", "color": _FULL_COLOUR}) == "person#green"


def test_the_hue_is_read_from_the_triple_string():
    assert P.hue_of({"color": "dim muted azure"}) == "azure"


def test_a_missing_colour_yields_a_bare_name():
    assert P.object_key({"name": "person"}) == "person"
    assert P.hue_of({"color": None}) == ""


def test_describe_never_embeds_a_dict_repr():
    """The description is shown to a person and handed to an LLM; a pasted
    dict repr makes it unreadable and unusable as a target."""
    text = P.describe({"name": "person", "position": [0.0, 0.0],
                       "color": _FULL_COLOUR}, 6.9)
    assert "rgb_mean" not in text and "green" in text


def test_both_colour_shapes_produce_the_same_key():
    """So a card keeps working across a change of `publish_color`."""
    assert (P.object_key({"name": "person", "color": _FULL_COLOUR})
            == P.object_key({"name": "person", "color": "dim muted green"}))


# ── close range: the two deadlocks ───────────────────────────────────────────
#
# Both were reported from r1_sz with a person standing right in front of the
# robot: it never declared arrival, and the only thing moving was wz. Both come
# from the same property — `bearing` is a normalised lateral offset, not an
# angle, so the same sideways step subtends far more of it at 0.7 m than at
# 5 m. Any fixed threshold that works far away becomes unreachable up close.


def test_a_target_straight_ahead_and_close_is_driven_towards_not_just_turned():
    """"I am right in front of it and all it does is turn." A hard gate on
    `align_tol` made vx zero for any wobble a standing person produces."""
    decision = _step(detections=_detections(_obj(x=0.18)),
                     depth=_depth(map_=_solid_depth(4.0)))
    assert decision.status == P.APPROACHING
    assert decision.values[0] > 0


def test_forward_speed_scales_with_alignment_instead_of_switching():
    aligned = _step(detections=_detections(_obj(x=0.0)),
                    depth=_depth(map_=_solid_depth(4.0))).values[0]
    off = _step(detections=_detections(_obj(x=0.30)),
                depth=_depth(map_=_solid_depth(4.0))).values[0]
    assert 0 < off < aligned


def test_a_target_near_the_edge_is_approached_more_slowly():
    """The intent of the old gate survives as a speed reduction rather than a
    stop: the bearing is least reliable at the frame edge, so the robot closes
    on it carefully instead of refusing to move."""
    centred = _step(detections=_detections(_obj(x=0.0)),
                    depth=_depth(map_=_solid_depth(4.0))).values[0]
    edge = _step(detections=_detections(_obj(x=0.8)),
                 depth=_depth(map_=_solid_depth(4.0))).values[0]
    assert 0 < edge < centred


def test_arrival_does_not_require_the_precision_the_drive_gate_does():
    """Arrival used to need |bearing| <= align_tol, which at 0.7 m asks a person
    to hold still. The position is what arriving is about."""
    decision = _step(detections=_detections(_obj(x=0.25)),
                     depth=_depth(map_=_solid_depth(0.8)))
    assert decision.status == P.ARRIVED


def test_being_close_but_never_aligned_still_arrives_eventually():
    """The deadlock in full: inside the stop distance, turning for ever.
    `idle_timeout_s` cannot catch it either — turning counts as motion."""
    config = _cfg(arrive_patience_s=0.5, arrive_align_tol=0.01)
    state = _state()
    for _ in range(12):
        decision = _step(detections=_detections(_obj(x=0.30)),
                         depth=_depth(map_=_solid_depth(0.5)),
                         config=config, state=state, dt=0.1)
        if decision.status == P.ARRIVED:
            break          # the terminal reply is sticky; catch the moment
    assert decision.status == P.ARRIVED
    assert "not waiting to align" in decision.reason


def test_leaving_the_stop_distance_resets_the_patience():
    """Otherwise a moment spent close early on would count towards arriving
    much later, somewhere else entirely.

    The target walks away rather than teleporting: 1 m/s, which the tracker will
    believe. At 8 m/s it would not, and rightly.
    """
    # No turning, so the heading never improves and arrival can only be decided
    # by distance — which is what this test is about.
    config = _cfg(arrive_patience_s=5.0, arrive_align_tol=0.01, wz_max=0.0)
    sim = Sim(range_m=0.8, bearing=0.3, config=config, velocity=(1.0, 0.0))
    state = seed(_state(), *sim.observe(), config)

    sim.velocity = np.array([0.0, 0.0])
    sim.run(state, ticks=5, dt=0.1)
    assert state.close_for_s == pytest.approx(0.5)

    sim.velocity = np.array([1.0, 0.0])
    sim.run(state, ticks=15, dt=0.1)
    assert state.close_for_s == 0.0


# ── the robot's deadband ─────────────────────────────────────────────────────
#
# R1 does nothing below 0.4 m/s or 1.0 rad/s — measured on r1_sz one axis at a
# time. The SDK accepts smaller commands, returns 0, and the robot stands still,
# so a policy that does not know about this emits a smooth ramp and never moves
# while every layer reports success.

_R1 = [0.4, 0.4, 0.0, 0.0, 0.0, 1.0]


def test_a_command_inside_the_deadband_is_lifted_out_of_it():
    """The policy asked for motion; this is the slowest motion available."""
    assert P.apply_deadband([0.0, 0, 0, 0, 0, -0.6], _R1)[5] == -1.0
    assert P.apply_deadband([0.3, 0, 0, 0, 0, 0], _R1)[0] == 0.4


def test_a_command_well_below_the_deadband_is_zeroed_rather_than_amplified():
    """Snapping 0.02 up to 1.0 rad/s would turn a rounding error into a lurch.
    Zero is at least honest, and the caller can see that it is zero."""
    assert P.apply_deadband([0, 0, 0, 0, 0, -0.1], _R1)[5] == 0.0


def test_a_command_above_the_deadband_is_untouched():
    assert P.apply_deadband([0.7, 0, 0, 0, 0, -1.5], _R1) == [0.7, 0, 0, 0, 0, -1.5]


def test_an_exact_zero_stays_zero():
    """Stopping must never become the slowest possible motion."""
    assert P.apply_deadband([0.0] * 6, _R1) == [0.0] * 6


def test_axes_with_no_threshold_pass_through():
    assert P.apply_deadband([0, 0, 0.01, 0, 0, 0], _R1)[2] == 0.01


def test_no_declared_deadband_changes_nothing():
    """A robot that never declared one — most of them — is unaffected."""
    assert P.apply_deadband([0.01] * 6, None) == [0.01] * 6
    assert P.apply_deadband([0.01] * 6, []) == [0.01] * 6


def test_the_sign_survives():
    assert P.apply_deadband([-0.3, 0, 0, 0, 0, 0.6], _R1)[:1] == [-0.4]
    assert P.apply_deadband([-0.3, 0, 0, 0, 0, 0.6], _R1)[5] == 1.0


def test_the_default_search_rate_clears_a_typical_deadband():
    """The shared deadband lifts the default search request when necessary."""
    rate = P.Config().search_rate
    assert rate == 0.6
    assert P.apply_deadband([0, 0, 0, 0, 0, -rate], _R1)[5] != 0.0


# ── the robot cannot move slowly ─────────────────────────────────────────────
#
# Everything in this section exists because a legged base has a deadband: below
# some speed it does not move at all, and the SDK accepts the command, returns
# 0, and says nothing. Three separate bugs came out of that, and each one below
# is one of them.

# R1's, as `loco_servo.build_descriptor` declares them.
_R1_DESC = {
    "limits": {"lower": [-1.0, -1.0, 0.0, 0.0, 0.0, -2.0],
               "upper": [1.0, 1.0, 0.0, 0.0, 0.0, 2.0],
               "min_magnitude": [0.4, 0.4, 0.0, 0.0, 0.0, 1.0]},
}


def test_a_ceiling_below_the_floor_is_raised_off_it():
    """The cause of the stutter, and the reason `adopt_limits` exists.

    `wz_max` defaulted to 0.8 against a 1.0 rad/s floor, so the policy's entire
    output range was unexecutable: every turn command snapped to 0 or ±1.0 and
    the robot turned in a 10 Hz square wave. Nothing reported it — the SDK
    accepted all of it.
    """
    # Pinned, not inherited: the shipped `wz_max` has since been raised past
    # R1's floor, and this test is about what happens when it is not.
    config = _cfg(wz_max=0.8)
    assert config.wz_max < _R1_DESC["limits"]["min_magnitude"][5]
    notes = P.adopt_limits(config, _R1_DESC)

    assert config.wz_max > config.floor_wz, "there is somewhere to be proportional"
    assert config.vx_max > config.floor_vx
    assert any("wz_max" in note for note in notes), "and it says so out loud"


def test_ceilings_are_also_pulled_down_into_the_descriptor():
    """A ceiling above `limits.upper` is a command the sink rejects — at the
    full command rate, for the whole run."""
    config = _cfg(vx_max=9.0)
    notes = P.adopt_limits(config, _R1_DESC)
    assert config.vx_max == 1.0
    assert any("超过下游允许" in note for note in notes)


def test_search_rate_is_also_clamped_to_the_yaw_limit():
    """Searching is a twist too; it cannot bypass the descriptor's yaw cap."""
    config = _cfg(search_rate=1.0)
    bumi = {"limits": dict(_R1_DESC["limits"],
                             lower=[-0.55, -0.2, 0, 0, 0, -0.83],
                             upper=[0.55, 0.2, 0, 0, 0, 0.83],
                             min_magnitude=[0.25, 0.12, 0, 0, 0, 0.33])}
    notes = P.adopt_limits(config, bumi)
    assert config.search_rate == 0.83
    assert config.search_rate <= bumi["limits"]["upper"][5]
    assert config.search_rate >= bumi["limits"]["min_magnitude"][5]
    assert any("search_rate" in note for note in notes)


def test_search_rate_uses_the_tighter_signed_yaw_limit():
    """A search sweeps both signs, so the smaller side is the safe rate."""
    config = _cfg(search_rate=1.0)
    asymmetric = {"limits": dict(_R1_DESC["limits"],
                                  lower=[-1, -1, 0, 0, 0, -0.55],
                                  upper=[1, 1, 0, 0, 0, 0.83],
                                  min_magnitude=[0.4, 0.4, 0, 0, 0, 0.33])}
    P.adopt_limits(config, asymmetric)
    assert config.search_rate == 0.55


def test_search_is_disabled_when_yaw_ceiling_is_below_its_deadband():
    """A capped rate below the declared floor would be zeroed downstream."""
    config = _cfg(search_rate=1.0)
    impossible = {"limits": dict(_R1_DESC["limits"],
                                  lower=[-1, -1, 0, 0, 0, -0.2],
                                  upper=[1, 1, 0, 0, 0, 0.2],
                                  min_magnitude=[0, 0, 0, 0, 0, 0.33])}
    notes = P.adopt_limits(config, impossible)
    assert config.search_rate == 0.0
    assert any("最小可执行量" in note for note in notes)

def test_search_is_disabled_when_the_chassis_has_no_yaw_axis():
    config = _cfg(search_rate=1.0)
    no_yaw = {"limits": dict(_R1_DESC["limits"],
                               lower=[-1, -1, 0, 0, 0, 0],
                               upper=[1, 1, 0, 0, 0, 0])}
    notes = P.adopt_limits(config, no_yaw)
    assert config.wz_max == 0.0
    assert config.search_rate == 0.0
    assert any("搜索转身" in note for note in notes)


@pytest.mark.parametrize("descriptor", [
    {"limits": dict(_R1_DESC["limits"],
                    lower=[-1, -1, 0, 0, 0, -0.2],
                    upper=[1, 1, 0, 0, 0, 0.2],
                    min_magnitude=[0, 0, 0, 0, 0, 0.33])},
    {"limits": dict(_R1_DESC["limits"],
                    lower=[-1, -1, 0, 0, 0, 0],
                    upper=[1, 1, 0, 0, 0, 0])},
])
def test_a_disabled_search_fails_instead_of_publishing_zero_twists(descriptor):
    config = _cfg(search_rate=1.0, search_timeout_s=1e6)
    P.adopt_limits(config, descriptor)
    assert config.search_rate == 0.0
    state = _state(missing_frames=config.lost_frames)
    decision = _step(detections=_detections(), depth=_depth(),
                     config=config, state=state)
    assert decision.status == P.FAILED
    assert decision.publishes is False
    assert "cannot search" in decision.reason

def test_an_axis_the_chassis_does_not_have_is_switched_off():
    config = _cfg()
    pinned = {"limits": dict(_R1_DESC["limits"],
                             lower=[-1.0, 0.0, 0.0, 0.0, 0.0, -2.0],
                             upper=[1.0, 0.0, 0.0, 0.0, 0.0, 2.0])}
    notes = P.adopt_limits(config, pinned)
    assert config.vy_max == 0.0
    assert any("vy" in note for note in notes)

    decision = _step(detections=_detections(_obj(x=0.5)), depth=_depth(),
                     config=config)
    assert decision.values[1] == 0.0


def test_a_robot_with_no_deadband_keeps_plain_proportional_control():
    """The floors are read from the descriptor, so a wheeled base — which can
    creep — is unaffected by any of this."""
    config = _cfg(vx_max=0.4)
    P.adopt_limits(config, {"limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}})
    assert (config.floor_vx, config.floor_vy, config.floor_wz) == (0.0, 0.0, 0.0)
    # 0.1 m short of the stop distance, so the proportional term is small on
    # its own — no floor, no lift, nothing rounding it up.
    values = _step(detections=_detections(_obj(x=0.02)),
                   depth=_depth(map_=_solid_depth(config.stop_distance_m + 0.1)),
                   config=config).values
    assert 0 < values[0] < 0.1, "a small residual distance, commanded small"


def test_the_last_stretch_is_actually_walked():
    """`k_fwd * (d - stop)` falls under the floor 0.67 m before arriving, so the
    robot used to stop short of a target it could see perfectly well and then
    fail on the idle timeout. The forward gate holds the floor speed until the
    stop distance is genuinely reached."""
    config = _cfg()
    P.adopt_limits(config, _R1_DESC)
    sim = Sim(range_m=3.0, bearing=0.0, config=config)
    state = seed(_state(), *sim.observe(), config)

    stalls, last = [], None
    for _ in range(80):
        detections, depth = sim.observe()
        # Odometry reports what the robot did last tick — feeding a constant
        # zero here would trip the stuck detector on a robot that is walking.
        odom = ({"vx": last[0], "vy": last[1], "wz": last[5]} if last
                else {"vx": 0.0, "vy": 0.0, "wz": 0.0})
        decision = P.step(detections=detections, depth=depth, odom=odom,
                          config=config, state=state, dt=0.1)
        last = decision.values
        if state.arrived:
            break
        distance = sim.relative()[0]
        if distance > config.stop_distance_m and (
                decision.values is None or decision.values[0] < config.floor_vx):
            stalls.append(round(distance, 2))
        sim.apply(decision.values, 0.1)

    assert state.arrived, f"never arrived; stopped at {sim.relative()[0]:.2f} m"
    assert not stalls, (
        f"commanded less than the robot can execute at {stalls} m — "
        "the last stretch is exactly where k_fwd*(d-stop) falls under the floor")


def test_the_yaw_axis_does_not_chatter_around_its_threshold():
    """Hysteresis plus a dwell. Without them an error sitting on `align_tol`
    toggles the axis every tick, and on a deadbanded robot that toggle is full
    speed to nothing and back — the judder this branch is named for."""
    config = _cfg()
    P.adopt_limits(config, _R1_DESC)
    # Parked on the threshold: the robot cannot turn, so the bearing stays put
    # and the only thing that can move the gate is the gate itself.
    config.wz_max = 0.0
    config.floor_wz = 0.0
    sim = Sim(range_m=3.0, bearing=config.align_tol, config=config)
    state = seed(_state(), *sim.observe(), config)

    turning = []
    for _ in range(20):
        detections, depth = sim.observe()
        decision = P.step(detections=detections, depth=depth,
                          odom={"vx": 0.0, "vy": 0.0, "wz": 0.0},
                          config=config, state=state, dt=0.1)
        turning.append(state.yaw_gate.on)

    switches = sum(1 for a, b in zip(turning, turning[1:]) if a != b)
    assert switches <= 1, f"the yaw axis toggled {switches} times in 2 seconds"


def test_a_turn_ends_once_the_target_is_well_centred():
    """Hysteresis must not become a latch: the release threshold is real."""
    # The gate only exists for commands under the robot's floor, so a config
    # with no deadband never engages it — adopt R1's so there is one.
    config = _cfg(align_min_scale=0.0, use_lateral=False)
    P.adopt_limits(config, _R1_DESC)
    state = seed(_state(), _detections(_obj(x=0.5)), _depth(), config)
    P.step(detections=_detections(_obj(x=0.5)), depth=_depth(), odom=None,
           config=config, state=state, dt=0.1)
    assert state.yaw_gate.on

    for _ in range(10):
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                          odom=None, config=config, state=state, dt=0.1)
    assert decision.values[5] == 0.0
    assert state.yaw_gate.on is False


def test_an_obstacle_ahead_first_causes_a_turn_in_place():
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth(bands={"left": 5.0, "center": 0.6, "right": 1.0},
                                  map_=_obstacle_map(5.0, [(0.25, 0.6, 0.3)])))
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0, "no forward motion into it"
    assert decision.values[1] == 0.
    assert decision.values[5] > 0, "turning left, the side with room"


def test_an_obstacle_with_no_room_either_side_is_not_stepped_into():
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_scene(1.0, [(0.0, 0.5, 0.5)]))
    assert decision.values[1] == 0.0


# ── the space the robot occupies ─────────────────────────────────────────────

_R1_FOOTPRINT = {"shape": "box", "half_width": 0.179, "front": 0.095,
                 "rear": 0.095, "height": 1.23, "source": "vendor-spec"}


def test_the_chassis_declares_its_own_width():
    """Same move as `min_magnitude`: the robot knows its dimensions and the
    policy must not hard-code them."""
    config = _cfg()
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    assert config.half_width_m == pytest.approx(0.179)


def test_an_undeclared_footprint_stays_wide_and_says_so():
    """Every failure mode of this number is one-sided. Believing the robot is
    wider than it is costs some unnecessary slowing; believing it is narrower
    puts a shoulder into a doorframe."""
    config = _cfg()
    notes = P.adopt_limits(config, _R1_DESC)
    assert config.half_width_m >= 0.3
    assert any("footprint" in note for note in notes)


def test_a_footprint_that_is_only_an_estimate_is_reported_as_one():
    config = _cfg()
    notes = P.adopt_limits(config, dict(
        _R1_DESC, footprint=dict(_R1_FOOTPRINT, source="estimate")))
    assert any("estimate" in note for note in notes)


def test_the_keepout_is_wider_than_the_declared_box():
    """What a chassis declares is its static envelope with the arms at rest, and
    what hits a doorframe is a swinging arm and a leg mid-stride."""
    config = _cfg()
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    assert P._keepout_m(config) > config.half_width_m


def test_an_obstacle_the_bands_call_left_now_stops_the_robot():
    """End to end, the reason all of the above exists.

    A doorframe 0.25 m off the axis at 0.7 m is inside a humanoid's width and
    outside the centre third of the picture. The bands file it under "left",
    and the left band has never stopped forward motion."""
    # `obstacle_stop_m` pinned to what this scenario was measured at; the
    # shipped default has since been tuned down on r1_sz.
    config = _cfg(obstacle_stop_m=0.8)
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    depth = {"map": _obstacle_map(5.0, [(-0.25, 0.7, 0.1)], config=config),
             "bands": {"left": 0.7, "center": 5.0, "right": 5.0}}
    state = seed(_state(), _detections(_obj(x=0.0)), depth, config)
    decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                      odom=None, config=config, state=state, dt=0.1)
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0


def test_a_corridor_it_cannot_measure_is_not_driven_into():
    """Unknown is not free. This was the default: an invalid pixel dropped out
    of the minimum, so a corridor full of holes and an empty one gave the same
    answer — and the objects that make holes are the ones that catch a
    shoulder."""
    config = _cfg()
    depth = {"map": np.full((480, 640), np.nan, dtype=np.float32),
             "bands": {"left": 5.0, "center": 5.0, "right": 5.0}}
    state = seed(_state(), _detections(_obj(x=0.0)),
                 _depth(map_=_solid_depth(3.0)), config)
    decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                      odom=None, config=config, state=state, dt=0.1)
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0
    assert "看不清" in decision.reason


def test_a_small_valid_part_of_the_corridor_can_still_support_approach():
    config = _cfg()
    frame = np.full((480, 640), np.nan, dtype=np.float32)
    frame[:, 310:330] = 3.
    depth = _depth(map_=frame)
    _, coverage = P._clearance(depth, config)
    assert 0. < coverage < config.coverage_min
    decision = _step(detections=_detections(_obj(x=0.0)), depth=depth, config=config)
    assert decision.status == P.APPROACHING and decision.values[0] > 0.


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("has_extrinsic", [False, True])
def test_search_turn_checks_occupied_cells_with_unseen_sides(monkeypatch, blocked, has_extrinsic):
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .23, .23, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry,
               scan_geometry=geometry if has_extrinsic else None,
               half_width_m=.23, search_rate=.25)
    state = _state(motion_source="command")
    free = local_path.FORWARD > 0.
    occupied = np.zeros_like(free)
    row, col = local_path.START
    occupied[row + 4, col + 6] = blocked
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, occupied))
    monkeypatch.setattr(P, "_clearance", lambda *args: (None, .05))
    frame = np.ones((4, 4))
    depth = {"map": frame, "ground": frame.astype(bool), "frame_id": 1000}
    state.active_scan.observe_single(depth, geometry, 800, 1000)

    reason = P._search_turn_stop(depth, cfg, state)
    if blocked:
        assert "扫掠范围有障碍" in reason
    else:
        assert reason is None


@pytest.mark.parametrize(("distance", "age_ms", "status"), [
    (3.0, 0, P.SEARCHING),
    (0.4, 0, P.AVOIDING),
    (np.nan, 0, P.AVOIDING),
    (3.0, 801, P.BLIND),
])
def test_search_without_a_sweep_grid_uses_live_front_depth(distance, age_ms, status):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .23, .23, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, max_depth_age_ms=800)
    depth = dict(_depth(map_=_solid_depth(distance)), frame_id=1000,
                 control_ms=1000 + age_ms)
    state = _state(missing_frames=cfg.lost_frames - 1)
    decision = P.step(detections=_detections(), depth=depth, odom=None,
                      config=cfg, state=state, dt=.1)

    assert decision.status == status
    assert bool(decision.values and decision.values[5]) == (status == P.SEARCHING)


def test_search_can_turn_to_the_other_side_when_the_last_seen_side_is_blocked(monkeypatch):
    from plugins.navi.local_path import Geometry

    geometry = Geometry(None, None, np.eye(4), .23, .23, None)
    config = _cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="command", last_seen_side=1., searched_rad=1.)
    state.active_scan.current_frame = 1000
    monkeypatch.setattr(state.active_scan, "clear_sweep",
                        lambda _geometry, _keepout, sweep, _slack: sweep < 0.)

    assert P._search_turn_stop(_depth(), config, state) is None
    assert state.last_seen_side == -1.
    assert state.searched_rad == 0.
    assert P._search(config, state, .1).values is None  # short-occlusion dwell still applies


def test_command_only_active_observation_uses_camera_geometry_without_extrinsics(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .21, .21, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.21,
               max_depth_age_ms=800, floor_wz=.33, wz_max=.83)
    state = _state()
    target = SimpleNamespace(bearing_rad=0., range_m=2., range_known=True,
                             observed=True, measured_range_m=2.)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.9, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *args: None)
    free = local_path.FORWARD > 0.
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, np.zeros_like(free)))
    frame = np.ones((4, 4))

    def tick(stamp):
        depth = {"map": frame, "ground": frame.astype(bool), "frame_id": stamp}
        return P.step(detections={"objects": []}, depth=depth, odom=None,
                      config=cfg, state=state, dt=.1)

    assert tick(1000).values == [0.] * 6
    assert state.active_scan.active
    for stamp in range(1100, 1800, 100):
        decision = tick(stamp)
        if decision.values[5]:
            assert cfg.floor_wz <= abs(decision.values[5]) <= cfg.wz_max
            return
    pytest.fail("expected a bounded turn with camera geometry")

def test_the_summary_only_path_cannot_detect_a_corridor_full_of_holes():
    """Pinned as a known limitation rather than discovered on a robot.

    `visual_depth`'s summary is three numbers; it has no way to say how much of
    a band was measured, so the fallback reports a coverage it has not earned.
    That is one of the reasons the card calls the summary-only wiring degraded.
    """
    config = _cfg()
    depth = {"map": None, "bands": {"left": None, "center": None, "right": None}}
    assert P._clearance(depth, config) == (None, 1.0)


# ── the deadband is a property of the gait, not of the axis ──────────────────

_R1_COUPLED = {"limits": dict(_R1_DESC["limits"],
                              min_magnitude_moving=[0.4, 0.4, 0.0, 0.0, 0.0, 0.05])}


def test_yaw_is_proportional_while_the_robot_is_walking():
    """The bug this whole section is about.

    R1 needs 1.0 rad/s to start turning from a standstill and 0.05 once it is
    already walking — twenty times smaller. Lifting every yaw command to the
    standing floor mid-approach turns a 0.05 correction into a 1.0 one, then
    reverses, then overshoots again. On the robot that looked like weaving left
    and right on the way to a target it was already facing.
    """
    config = _cfg()
    P.adopt_limits(config, _R1_COUPLED)
    assert config.floor_wz_moving == 0.05

    depth = _depth(map_=_solid_depth(3.0))
    detections = _detections(_obj(x=0.12))          # slightly off centre
    decision = _step(detections=detections, depth=depth, config=config)

    assert decision.values[0] > 0, "it is walking"
    wz = decision.values[5]
    assert 0 < abs(wz) < 0.5, (
        f"wz={wz:+.2f} — a small bearing error while walking must produce a "
        "small turn, not the standing floor")


def test_turning_in_place_still_uses_the_standing_floor():
    """Which is the whole reason the gate and the lift exist: from a standstill
    the robot genuinely cannot turn slowly."""
    config = _cfg(align_min_scale=0.0, use_lateral=False)
    P.adopt_limits(config, _R1_COUPLED)
    decision = _step(detections=_detections(_obj(x=0.9)), depth=_depth(),
                     config=config)
    assert decision.values[0] == 0.0, "not translating"
    assert abs(decision.values[5]) >= config.floor_wz


def test_a_chassis_that_declares_no_moving_floor_keeps_the_standing_one():
    """Optional field: absent means the old behaviour, which errs towards
    commanding too much rather than too little."""
    config = _cfg()
    P.adopt_limits(config, _R1_DESC)
    assert config.floor_wz_moving == config.floor_wz


def test_the_card_picks_the_floor_from_the_command_it_is_about_to_send():
    """`apply_deadband` runs after the policy and would otherwise re-inflate
    exactly the commands the policy was careful not to quantise."""
    from plugins.navi import plugin as navi_plugin

    card = navi_plugin.NaviPlugin({}, executor=None)
    card._descriptor = _R1_COUPLED
    assert card._deadband_for([0.4, 0.0, 0, 0, 0, 0.05])[5] == 0.05, "walking"
    assert card._deadband_for([0.0, 0.0, 0, 0, 0, 0.05])[5] == 1.0, "standing"


# ── "arrived" and "blocked" must stop impersonating each other ───────────────

def test_a_coasted_position_cannot_be_called_an_arrival():
    """What ended two navigations on r1_sz in a row.

    The arrival test ran before the obstacle logic, so a track whose position
    was a prediction — and which had in fact been re-acquired onto a traffic
    cone — reported success while the person it was following was metres away.
    Walking on a prediction is fine; declaring the journey over on one is not.
    """
    config = _cfg()
    depth = _depth(map_=_solid_depth(0.5))
    state = seed(_state(), _detections(_obj(x=0.0)), depth, config)

    # The target goes out of view. The track survives on its prediction, which
    # still says "right in front of us".
    decision = P.step(detections={"objects": []}, depth=depth, odom=None,
                      config=config, state=state, dt=0.1)
    assert decision.status != P.ARRIVED
    assert not state.arrived
    # And the reason names what is actually there, rather than claiming
    # success — which is the whole point of the reordering.
    assert "障碍" in decision.reason

    # It comes back, and now the claim is supported.
    for _ in range(config.confirm_hits + 1):
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                          odom=None, config=config, state=state, dt=0.1)
    assert decision.status == P.ARRIVED


def test_a_corridor_that_stays_blocked_fails_saying_so():
    """Trying to get round it is the right first move, and `_avoid` makes it.
    But a robot that has been shuffling sideways for twelve seconds is not
    making progress, and the caller is owed the real reason — "blocked", not a
    generic idle timeout, and certainly not "arrived"."""
    config = _cfg(blocked_timeout_s=1.0, stop_distance_m=0.3)
    depth = _depth(map_=_solid_depth(0.4))       # inside obstacle_stop_m
    state = seed(_state(), _detections(_obj(x=0.0)), depth, config)

    for _ in range(20):
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                          odom=None, config=config, state=state, dt=0.1)
        if decision.status == P.FAILED:
            break
    assert decision.status == P.FAILED
    assert "挡住" in decision.reason and "障碍" in decision.reason
    assert decision.values is None, "a failure publishes nothing"


def test_a_corridor_that_clears_resets_the_blocked_clock():
    """A person stepping across the path is not a blockage."""
    config = _cfg(blocked_timeout_s=1.0, stop_distance_m=0.3)
    blocked = _depth(map_=_solid_depth(0.4))
    clear = _depth(map_=_solid_depth(5.0))
    state = seed(_state(), _detections(_obj(x=0.0)), blocked, config)

    for _ in range(30):
        depth = blocked if _ % 10 < 8 else clear
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                          odom=None, config=config, state=state, dt=0.1)
        assert decision.status != P.FAILED


@pytest.mark.parametrize(("feedback", "fails"), [(-.12, False), (-.039, True), (.12, True)])
def test_lateral_only_detour_uses_measured_progress_for_the_blockage_clock(monkeypatch, feedback, fails):
    from types import SimpleNamespace

    cfg = _cfg(local_planning=True, blocked_timeout_s=.2)
    state = _state(commanded=(0., -.30, 0.))
    target = SimpleNamespace(bearing_rad=0., range_m=2.5, range_known=True,
                             observed=True, measured_range_m=2.5)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_avoid", lambda *args, **kwargs:
                        (0., -.30, 0., P.AVOIDING, "局部规划"))
    odom = {"vx": 0., "vy": feedback, "wz": 0.}
    for _ in range(3):
        decision = P.step(detections=_detections(), depth=_scene(2.5), odom=odom,
                          config=cfg, state=state, dt=.1)
    assert (decision.status == P.FAILED) is fails


@pytest.mark.parametrize("moves_sideways", [False, True])
def test_local_avoidance_does_not_execute_lateral_candidates(monkeypatch,
                                                                            moves_sideways):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    cfg = _cfg(local_planning=True, local_geometry=geometry, stuck_window_s=.4)
    state = _state(motion_source="measured")
    state.active_scan.attempted = True
    target = SimpleNamespace(bearing_rad=0., range_m=2.4, range_known=True,
                             observed=True, measured_range_m=2.4)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0.: (
        1.45 if lateral_offset_m > 0. else 1.35 if lateral_offset_m < 0. else 1.1, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda depth, config, state, odom, ahead, side:
                        (0., -.03 * side))
    free = np.ones(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, ~free))
    scene = _scene(2.4)
    decisions = []
    for stamp in range(1000, 2400, 200):
        depth = dict(scene, frame_id=stamp, ground=np.ones_like(scene["map"], dtype=bool))
        odom = {"vx": state.commanded[0], "vy": -.08 if moves_sideways else -.039,
                "wz": 0., "received_ms": stamp}
        decisions.append(P.step(detections=_detections(), depth=depth, odom=odom,
                                config=cfg, state=state, dt=.2))
    assert all(decision.values[1:] == [0.] * 5 for decision in decisions)
    if moves_sideways:
        assert all(decision.values[0] == 0. for decision in decisions)
    else:
        assert decisions[-1].values[0] == pytest.approx(.15)
        assert "低速接近" in decisions[-1].reason


# ── half_fov_rad: 一个数填错，走廊就比门宽 ───────────────────────────────────

def _doorway(open_w_m, wall_m, *, half_fov_rad, far_m=6.0):
    """一面 `wall_m` 远的墙，中间开一个 `open_w_m` 宽的洞，洞后面是 `far_m`。

    深度值按**真实**视场角摆放，配置里的值才是被考问的那个 —— 这正是真机上出错
    的方式：图是相机给的，反算用的是配置。
    """
    m = np.full((480, 640), far_m, dtype=float)
    u = (np.arange(640) + 0.5) / 640 * 2 - 1
    m[:, np.abs(np.tan(u * half_fov_rad) * wall_m) > open_w_m / 2] = wall_m
    return m


def test_a_standard_doorway_reads_as_passable():
    """r1_sz 上过不了门的原因，量化。

    走廊是米制的，所以代码要把半宽反算成画面上的像素列，而那一步用的就是
    `half_fov_rad`。填小了，切片就偏宽：配 0.55 而实测 0.888 时，1 m 处「半宽
    0.5 m 的走廊」采样了画面半宽的 84%，真实对应 ±0.93 m —— 走廊有 1.86 m 宽，
    比任何一扇门都宽。于是门框永远算在正前方，clearance 读到门平面而不是门洞。
    """
    true_fov = 0.888
    config = _cfg(half_fov_rad=true_fov)
    # 必须用机器人自己声明的 footprint。Config 的兜底半宽 0.35 加上余量要求
    # **1.0 m 净宽**，而标准门只有 0.8~0.9 m —— 也就是说没有 footprint 声明的底盘
    # 本来就过不了门，而且过不去的理由跟视场角无关。R1 声明的是 0.179。
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    keep = config.half_width_m + config.clearance_margin_m
    assert 2 * keep < 0.90, "0.9 m 的门本来就该放得下这台机器人"

    for wall in (1.2, 1.0, 0.8, 0.7, 0.6):
        depth = _doorway(0.90, wall, half_fov_rad=true_fov)
        clearance, coverage = D.corridor(
            depth, half_width_m=keep, half_fov_rad=config.half_fov_rad,
            reference_m=config.obstacle_stop_m)
        assert clearance > config.obstacle_stop_m, (
            f"离门 {wall} m 时走廊被门框挡住了：clearance={clearance}")
        assert coverage >= config.coverage_min

    # 反过来钉住这条测试确实有效：把视场角填成出厂的 0.55，同一张图就会在 0.6 m
    # 处判定挡住 —— 也就是真机上看到的行为。
    depth = _doorway(0.90, 0.6, half_fov_rad=true_fov)
    wrong, _ = D.corridor(depth, half_width_m=keep, half_fov_rad=0.55,
                          reference_m=config.obstacle_stop_m)
    assert wrong <= config.obstacle_stop_m


def test_something_genuinely_too_narrow_still_stops_the_robot():
    """放宽视场角不是把避障关掉 —— 过不去的洞还是要挡住。"""
    true_fov = 0.888
    config = _cfg(half_fov_rad=true_fov)
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    keep = config.half_width_m + config.clearance_margin_m
    depth = _doorway(2 * keep - 0.2, 0.5, half_fov_rad=true_fov)
    clearance, _ = D.corridor(depth, half_width_m=keep,
                              half_fov_rad=config.half_fov_rad,
                              reference_m=config.obstacle_stop_m)
    assert clearance is not None and clearance <= config.obstacle_stop_m


# ── 相机几何由相机声明，这张卡片采纳（motus.camera/1）──────────────────────────

def _declaration(**over):
    out = {"schema": "motus.camera/1", "topic": "/ubuntu/camera/main/visual_depth",
           "id": "unitree/r1/camera_main", "width": 640, "height": 480,
           "distortion_model": "unknown", "D": None, "K": None,
           "half_fov_rad": 0.888, "half_fov_v_rad": None,
           "source": "inherited", "pipeline": ["unitree/r1/camera_main"]}
    out.update(over)
    return out


def test_the_camera_supplies_the_field_of_view():
    """The quantity this card used to make a human copy by hand, and the last
    one here that a human still did."""
    config = _cfg(half_fov_rad=0.55)
    notes = P._adopt_camera(config, _declaration(half_fov_rad=0.888,
                                                 source="measured"))
    assert config.half_fov_rad == pytest.approx(0.888)
    # The adoption is reported, because an operator who typed 0.55 is owed the
    # news that the camera overruled them — the same rule adopt_limits follows.
    assert any("0.888" in n and "0.550" in n for n in notes), notes


def test_an_adoption_that_changes_nothing_says_nothing():
    """`degraded` is read by people. A note per start that reports the status quo
    trains them to ignore the list."""
    config = _cfg(half_fov_rad=0.888)
    assert P._adopt_camera(config, _declaration()) == []


def test_no_declaration_keeps_the_conservative_value_and_names_the_symptom():
    """The failure directions are not symmetric and neither is safe: too small a
    field of view widens the corridor and the robot refuses gaps it fits through;
    too large narrows it and a shoulder goes into a doorframe. Stuck is
    recoverable, a collision is not — so the fallback stays conservative.

    And it names the symptom, because "the robot keeps turning away at the door"
    was diagnosed as a tracking problem first, twice.
    """
    config = _cfg(half_fov_rad=0.55)
    notes = P._adopt_camera(config, {})
    assert config.half_fov_rad == 0.55
    assert len(notes) == 1 and "门口" in notes[0] and "camera_info" in notes[0]


def test_a_camera_that_declares_no_angle_is_not_a_camera_that_declares_zero():
    """Rule 1 of the format, from the consumer's side. Substituting a plausible
    number here would recreate the bug the format was written to prevent."""
    config = _cfg(half_fov_rad=0.55)
    notes = P._adopt_camera(config, _declaration(half_fov_rad=None,
                                                 source="unknown"))
    assert config.half_fov_rad == 0.55
    assert "measure_fov" in notes[0], "得告诉人怎么把这个数量出来"


def test_a_full_angle_in_the_half_angle_field_is_refused_not_halved():
    """The likeliest way a declaration is wrong. Quietly halving it would be a
    guess about somebody else's bug; the fallback plus a note is not."""
    config = _cfg(half_fov_rad=0.55)
    notes = P._adopt_camera(config, _declaration(half_fov_rad=1.776,
                                                 source="measured"))
    assert config.half_fov_rad == 0.55
    assert "半" in notes[0] and "1.776" in notes[0]


def test_a_foreign_schema_is_not_silently_read_as_ours():
    config = _cfg(half_fov_rad=0.55)
    notes = P._adopt_camera(config, _declaration(schema="motus.camera/2"))
    assert config.half_fov_rad == 0.55
    assert "schema" in notes[0]


def test_a_solved_calibration_beats_the_declared_angle():
    """`K` comes out of many observations; the angle beside it is usually a tape
    measure. Preferring the coarser number would be backwards."""
    config = _cfg(half_fov_rad=0.55)
    fx = 400.0
    P._adopt_camera(config, _declaration(width=1280, K=[fx, 0, 640, 0, fx, 360, 0, 0, 1],
                                         half_fov_rad=0.2, source="manual"))
    assert config.half_fov_rad == pytest.approx(math.atan(640.0 / fx))


def test_the_doorway_only_reads_as_passable_because_the_camera_said_so():
    """Ties the corridor geometry to the declaration chain.

    The same scenario as `test_a_standard_doorway_reads_as_passable`, except the
    field of view arrives the way it does on a robot. So if the chain ever breaks
    — a camera card that stops declaring, a plumbing change in agent-core — this
    test goes red here instead of the robot finding out at a door.
    """
    true_fov = 0.888
    config = _cfg(half_fov_rad=0.55)          # the value that caused the bug
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    P._adopt_camera(config, _declaration(half_fov_rad=true_fov, source="measured"))

    keep = config.half_width_m + config.clearance_margin_m
    for wall in (1.2, 1.0, 0.8, 0.7, 0.6):
        depth = _doorway(0.90, wall, half_fov_rad=true_fov)
        clearance, _ = D.corridor(depth, half_width_m=keep,
                                  half_fov_rad=config.half_fov_rad,
                                  reference_m=config.obstacle_stop_m)
        assert clearance > config.obstacle_stop_m, (
            f"离门 {wall} m 时走廊被门框挡住了：clearance={clearance}")


# ── 起步时那个没必要的侧移 ───────────────────────────────────────────────────

def test_a_target_a_few_percent_off_centre_does_not_trigger_a_sidestep():
    """What an operator saw as "every run starts with a sidestep it does not
    need, then it comes towards me".

    `_lift` raises whatever it is given to the axis floor, so a bearing of 0.05
    asks for ~0.04 m/s of lateral and receives R1's minimum 0.4 — a **ninefold**
    amplification, and a visible sideways lurch. The yaw axis had a gate for
    exactly this; the lateral axis did not.
    """
    config = _cfg()
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    assert config.floor_vy > 0, "前提：这台机器人有横移死区"

    state = seed(_state(), _detections(_obj(x=0.05)), _depth(), config)
    decision = P.step(detections=_detections(_obj(x=0.05)), depth=_depth(),
                      odom=None, config=config, state=state, dt=0.1)

    assert decision.values[1] == 0.0, "微小的方位差不该换来一次全速横移"
    assert decision.values[0] > 0.0, "但还是要往前走"


def test_a_target_well_off_to_one_side_still_gets_a_real_sidestep():
    """The gate must not turn the feature off. Walking a straight line to
    something off to one side is the whole reason vy exists.

    **On R1 that band is narrow, and honestly so.** The lateral floor is
    0.4 m/s, so *any* sidestep this robot makes is a 0.4 m/s one — at an
    approach speed near 1 m/s that is a 22-degree crab, a large manoeuvre. The
    honest lateral demand only reaches half the floor when the target is well
    off axis, so on this chassis strafing is rare by construction rather than by
    tuning. A base that can creep sideways is unaffected (see below).
    """
    config = _cfg()
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    state = seed(_state(), _detections(_obj(x=0.65)), _depth(), config)
    decision = P.step(detections=_detections(_obj(x=0.65)), depth=_depth(),
                      odom=None, config=config, state=state, dt=0.1)
    assert abs(decision.values[1]) >= config.floor_vy


def test_where_the_sidestep_starts_on_r1_is_pinned_rather_than_incidental():
    """Which bearings strafe is a consequence of the robot's floor and the
    alignment ramp, not of a number somebody chose. Pinned so that a change to
    either shows up here instead of on a robot."""
    config = _cfg()
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    strafing = []
    for bearing in [b / 100 for b in range(0, 71, 5)]:
        state = seed(_state(), _detections(_obj(x=bearing)), _depth(), config)
        values = P.step(detections=_detections(_obj(x=bearing)), depth=_depth(),
                        odom=None, config=config, state=state, dt=0.1).values
        strafing.append(values is not None and abs(values[1]) > 0)
    first = next((i for i, on in enumerate(strafing) if on), None)
    assert first is not None, "完全不横移就是把功能关掉了"
    assert 0.40 <= first * 0.05 <= 0.70, f"起始方位 {first * 0.05}"


def test_a_base_with_no_lateral_deadband_is_not_gated_at_all():
    """A wheeled base can creep sideways, so `_lift` is not amplifying anything
    and there is nothing to protect against."""
    config = _cfg()
    P.adopt_limits(config, {"limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}})
    assert config.floor_vy == 0.0
    assert P._worth_strafing(0.01, config, None, 0.1) is True


# ── bounded forward-P distance schedule ────────────────────────────────────

def _adaptive_ready(config, distance=1.0, bearing=0.0):
    observations = _detections(_obj(x=bearing, bbox=(0.4, 0.4, 0.6, 0.6)))
    depth = _depth(map_=_solid_depth(distance))
    return seed(_state(), observations, depth, config)


def _adaptive_frame(config, state, distance=1.0, bearing=0.0, dt=0.1):
    return _step(detections=_detections(_obj(x=bearing, bbox=(0.4, 0.4, 0.6, 0.6))),
                 depth=_depth(map_=_solid_depth(distance)),
                 odom={"vx": 0.0, "vy": 0.0, "wz": 0.0},
                 config=config, state=state, dt=dt)


def test_adaptive_forward_defaults_off_and_keeps_the_fixed_p_law(monkeypatch):
    config = _cfg(k_fwd=0.2, vx_max=0.3, use_lateral=False)
    state = _adaptive_ready(config, distance=1.2)
    state.adaptive_fwd_scale = 3.0  # Disabling also clears a previous schedule.
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    decision = _adaptive_frame(config, state, distance=1.2)
    assert config.adaptive_k_fwd is False
    assert decision.values[0] == pytest.approx(0.2 * (1.2 - 0.8), abs=1e-7)
    assert decision.k_fwd_effective == 0.2
    assert state.adaptive_fwd_scale == 1.0


def test_forward_gain_is_bounded_and_increases_only_smoothly_near_the_target():
    config = _cfg(adaptive_k_fwd=True, k_fwd=0.4, stop_tolerance_m=0.05)
    far = _state()
    assert P._forward_gain(config, far, 3.0, True, False, 0.1) == config.k_fwd
    near = _state()
    gains = []
    for _ in range(20):
        previous = near.adaptive_fwd_scale
        gains.append(P._forward_gain(config, near, 0.86, True, False, 0.1))
        assert abs(near.adaptive_fwd_scale - previous) <= 0.2 + 1e-12
    assert gains[-1] > gains[0] > config.k_fwd
    assert all(config.k_fwd <= gain <= 3 * config.k_fwd for gain in gains)


def test_forward_schedule_does_not_jump_after_a_long_tick_gap():
    config = _cfg(adaptive_k_fwd=True)
    state = _state()
    gain = P._forward_gain(config, state, 0.81, True, False, 30.0)
    assert gain <= config.k_fwd * 1.4
    before = state.adaptive_fwd_scale
    P._forward_gain(config, state, 0.81, True, False, -1.0)
    assert state.adaptive_fwd_scale == before


def test_settled_adaptive_p_speed_decreases_with_distance_error():
    config = _cfg(adaptive_k_fwd=True, k_fwd=0.1, stop_tolerance_m=0.05)
    speeds = []
    for distance in (3.0, 1.6, 1.4, 1.2, 1.0, 0.86):
        state = _state()
        for _ in range(25):
            gain = P._forward_gain(config, state, distance, True, False, 0.1)
        speeds.append(gain * (distance - config.stop_distance_m))
    assert all(a > b for a, b in zip(speeds, speeds[1:]))


@pytest.mark.parametrize("kind", ["blind", "coasting", "aligning", "within"])
def test_forward_schedule_resets_when_the_live_forward_approach_stops(kind):
    config = _cfg(adaptive_k_fwd=True, align_min_scale=0.0, stop_tolerance_m=0.05)
    state = _adaptive_ready(config)
    for _ in range(8):
        _adaptive_frame(config, state)
    assert state.adaptive_fwd_scale > 1
    if kind == "blind":
        result = _step(detections=None, depth=_depth(), config=config, state=state)
        assert result.status == P.BLIND and result.values is None
    elif kind == "coasting":
        result = _step(detections=_detections(), depth=_depth(map_=_solid_depth(1.0)),
                       config=config, state=state)
        assert state.tracker.track is not None and not state.tracker.track.observed
        assert result.k_fwd_effective == config.k_fwd
    elif kind == "aligning":
        # A fresh target session cannot inherit the previous approach gain.
        state = _adaptive_ready(config, distance=1.0, bearing=0.9)
        state.adaptive_fwd_scale = 2.0
        result = _adaptive_frame(config, state, bearing=0.9)
        assert result.status == P.ALIGNING
    else:
        state = _adaptive_ready(config, distance=0.84)
        state.adaptive_fwd_scale = 2.0
        result = _adaptive_frame(config, state, distance=0.84)
        assert result.status == P.ARRIVED and result.values == [0.0] * 6
        assert result.k_fwd_effective is None
    assert state.adaptive_fwd_scale == 1.0


def test_unknown_range_and_zero_baseline_gain_do_not_acquire_an_artificial_floor():
    config = _cfg(adaptive_k_fwd=True)
    state = _state()
    state.adaptive_fwd_scale = 2.5
    assert P._forward_gain(config, state, None, True, False, 0.1) == config.k_fwd
    assert state.adaptive_fwd_scale == 1
    zero = _cfg(adaptive_k_fwd=True, k_fwd=0, floor_vx=0.4)
    state = _adaptive_ready(zero, distance=1.2)
    decision = _adaptive_frame(zero, state, distance=1.2)
    assert decision.values[0] == 0.0 and decision.k_fwd_effective == 0


def test_a_blocked_corridor_does_not_accumulate_forward_gain():
    config = _cfg(adaptive_k_fwd=True, stop_distance_m=0.4, stop_tolerance_m=0.05,
                  k_fwd=0.4, use_lateral=False)
    state = _adaptive_ready(config, distance=0.58)
    gains = []
    for _ in range(10):
        decision = _adaptive_frame(config, state, distance=0.58)
        assert decision.status == P.AVOIDING
        assert decision.values[:2] == [0.0, 0.0]
        assert state.adaptive_fwd_scale == 1.0
        gains.append(decision.k_fwd_effective)
    assert gains == pytest.approx([gains[0]] * len(gains))


def test_adaptive_forward_cannot_cross_a_conflicting_obstacle_stop_threshold():
    config = _cfg(adaptive_k_fwd=True, stop_distance_m=0.4, stop_tolerance_m=0.05,
                  vx_max=0.3, use_lateral=False, k_fwd=0.4)
    sim = Sim(range_m=1.2, config=config)
    state = seed(_state(), *sim.observe(), config)
    for _ in range(200):
        decision = sim.run(state, ticks=1, dt=0.1, odom=False)
        if decision.values is not None:
            assert 0 <= decision.values[0] <= config.vx_max
    assert not state.arrived and sim.relative()[0] >= config.obstacle_stop_m - 0.001


def test_adaptive_forward_finishes_a_gentle_ideal_approach_sooner_and_stops_in_tolerance():
    def run(adaptive):
        config = _cfg(adaptive_k_fwd=adaptive, k_fwd=0.1, vx_max=0.3,
                      stop_distance_m=1.2, stop_tolerance_m=0.1, use_lateral=False)
        sim = Sim(range_m=1.6, config=config)
        state = seed(_state(), *sim.observe(), config)
        for tick in range(600):
            decision = sim.run(state, ticks=1, dt=0.1, odom=False)
            if decision.values is not None:
                assert 0 <= decision.values[0] <= config.vx_max
            if state.arrived:
                assert decision.values == [0.0] * 6
                assert config.stop_distance_m <= sim.relative()[0] <= config.arrival_distance_m + 0.01
                return tick
        pytest.fail("ideal approach never completed")
    assert run(True) < run(False)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_adaptive_forward_flag_is_strictly_boolean(value):
    with pytest.raises(ValueError, match="adaptive_k_fwd"):
        _cfg(adaptive_k_fwd=value)


@pytest.mark.parametrize("gain", [-0.1, float("nan"), float("inf"), True, 1e308])
def test_adaptive_forward_rejects_invalid_or_unbounded_baseline_gain(gain):
    with pytest.raises(ValueError, match="k_fwd"):
        _cfg(adaptive_k_fwd=True, k_fwd=gain)


@pytest.mark.parametrize("gain", [-0.1, float("nan"), float("inf"), True])
def test_fixed_forward_rejects_invalid_baseline_gain(gain):
    with pytest.raises(ValueError, match="k_fwd"):
        _cfg(adaptive_k_fwd=False, k_fwd=gain)


# ── fixed proportional yaw ────────────────────────────────────────

def test_default_gains_keep_fixed_forward_and_yaw():
    config = _cfg()
    assert config.k_fwd == 0.6
    assert config.adaptive_k_fwd is False
    assert config.k_yaw == 0.6


def test_default_yaw_preserves_the_original_tracking_p_law(monkeypatch):
    config = _cfg(use_lateral=False, floor_wz=0.0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    state = _adaptive_ready(config, distance=2.0, bearing=0.1)
    decision = _adaptive_frame(config, state, distance=2.0, bearing=0.1)
    assert decision.values[5] == pytest.approx(-config.k_yaw * decision.bearing)
    assert decision.k_yaw_effective == config.k_yaw


@pytest.mark.parametrize("bearing", [-0.2, -0.02, 0.02, 0.2])
@pytest.mark.parametrize("response,expected_gain", [(None, 0.6), (0.3, 0.3)])
def test_tracking_yaw_gain_stays_fixed_across_repeated_frames(
        monkeypatch, bearing, response, expected_gain):
    config = _cfg(use_lateral=False, floor_wz=0.0, response_yaw_gain=response)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    state = _adaptive_ready(config, distance=2.0, bearing=bearing)
    for _ in range(12):
        decision = _adaptive_frame(config, state, distance=2.0, bearing=bearing)
        assert decision.k_yaw_effective == expected_gain
        assert decision.values[5] == pytest.approx(-expected_gain * decision.bearing)


def test_fixed_yaw_gain_is_retained_during_brief_target_occlusion(monkeypatch):
    config = _cfg(use_lateral=False, floor_wz=0.0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    state = _adaptive_ready(config, distance=2.0, bearing=0.1)
    for _ in range(12):
        _adaptive_frame(config, state, distance=2.0, bearing=0.1)
    decision = _step(detections=_detections(), depth=_depth(map_=_solid_depth(2.0)),
                     odom={"vx": 0.0, "vy": 0.0, "wz": 0.0}, config=config, state=state)
    assert not state.tracker.track.observed
    assert decision.k_yaw_effective == 0.6
    assert decision.values[5] == pytest.approx(-0.6 * decision.bearing)


@pytest.mark.parametrize("gain", [0.0, 0.6, 1.2])
def test_search_speed_does_not_use_the_target_bearing_gain(gain):
    config = _cfg(k_yaw=gain, search_rate=0.3)
    state = _state(missing_frames=config.lost_frames)
    decision = _step(detections=_detections(), depth=_depth(), config=config, state=state)
    assert decision.status == P.SEARCHING
    assert decision.k_yaw_effective is None
    assert decision.values[5] == pytest.approx(-0.3)


def test_obstacle_gap_turns_keep_their_nominal_gain_and_do_not_claim_tracking_gain(monkeypatch):
    config = _cfg(use_lateral=False, stop_distance_m=0.3, floor_wz=0.0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs:
                        (5.0, 1.0) if kwargs.get("lateral_offset_m") else (0.4, 1.0))
    state = _adaptive_ready(config, distance=1.2, bearing=0.02)
    decision = _adaptive_frame(config, state, distance=1.2, bearing=0.02)
    expected = min(config.wz_max, config.k_yaw * math.atan2(P._keepout_m(config), 0.4))
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0
    assert abs(decision.values[5]) == pytest.approx(expected)
    assert decision.k_yaw_effective is None


def test_zero_nominal_yaw_gain_does_not_acquire_an_artificial_target_turn(monkeypatch):
    config = _cfg(k_yaw=0.0, use_lateral=False, floor_wz=0.4)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    state = _adaptive_ready(config, distance=2.0, bearing=0.2)
    decision = _adaptive_frame(config, state, distance=2.0, bearing=0.2)
    assert decision.values[5] == 0.0
    assert decision.k_yaw_effective == 0.0


@pytest.mark.parametrize("gain", [-0.1, float("nan"), float("inf"), True])
def test_fixed_yaw_rejects_invalid_nominal_gain(gain):
    with pytest.raises(ValueError, match="k_yaw"):
        _cfg(k_yaw=gain)


# ── Driver-declared target yaw response ──────────────────────────────

def _profiled_frame(config, *, bearing=0.2, odom=None, distance=2.0, state=None):
    state = state or _adaptive_ready(config, distance=distance, bearing=bearing)
    return _step(detections=_detections(_obj(x=bearing, bbox=(0.4, 0.4, 0.6, 0.6))),
                 depth=_depth(map_=_solid_depth(distance)), odom=odom,
                 config=config, state=state, dt=0.1)


def test_default_has_no_driver_response_override():
    config = _cfg()
    assert config.response_yaw_gain is None


@pytest.mark.parametrize("bearing", [-0.2, 0.2])
@pytest.mark.parametrize("measured_wz", [-5.0, -0.2, 0.0, 0.2, 5.0, None])
def test_driver_response_tracking_is_pure_p_and_ignores_actual_yaw_rate(
        bearing, measured_wz, monkeypatch):
    config = _cfg(response_yaw_gain=0.8, use_lateral=False, floor_wz=0.0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    decision = _profiled_frame(config, bearing=bearing,
                               odom={"vx": 0.0, "vy": 0.0, "wz": measured_wz})
    assert decision.values[0] > 0.0 and decision.values[1] == 0.0
    assert decision.values[5] == pytest.approx(-0.8 * decision.bearing)
    assert decision.values[5] * decision.bearing < 0.0
    assert decision.k_yaw_effective == 0.8


def test_driver_response_p_command_stays_inside_the_angular_limit(monkeypatch):
    config = _cfg(response_yaw_gain=1.0, use_lateral=False,
                  floor_wz=0.0, wz_max=0.15)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    decision = _profiled_frame(config, odom={"wz": 5.0})
    assert abs(decision.values[5]) <= config.wz_max
    assert decision.values[5] == pytest.approx(
        max(-config.wz_max, min(config.wz_max, -decision.bearing)))


def test_driver_response_alignment_uses_the_same_p_only_law(monkeypatch):
    config = _cfg(response_yaw_gain=0.8, align_min_scale=0.0, floor_wz=0.0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (5.0, 1.0))
    decision = _profiled_frame(config, bearing=0.9, odom={"wz": -0.2})
    assert decision.status == P.ALIGNING
    assert decision.values[:2] == [0.0, 0.0]
    assert decision.values[5] == pytest.approx(-0.8 * decision.bearing)
    assert decision.k_yaw_effective == 0.8


def test_driver_response_profile_leaves_search_rate_unchanged():
    config = _cfg(response_yaw_gain=0.3, search_rate=0.35)
    state = _state(missing_frames=config.lost_frames)
    decision = _step(detections=_detections(), depth=_depth(),
                     odom={"wz": 0.2}, config=config, state=state)
    assert decision.status == P.SEARCHING
    assert decision.values[5] == -0.35
    assert decision.k_yaw_effective is None


def test_driver_response_profile_controls_obstacle_gap_turns(monkeypatch):
    config = _cfg(response_yaw_gain=0.3, use_lateral=False,
                  stop_distance_m=0.3, floor_wz=0.0)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs:
                        (5.0, 1.0) if kwargs.get("lateral_offset_m") else (0.4, 1.0))
    decision = _profiled_frame(config, distance=1.2, bearing=0.02, odom={"wz": 0.2})
    expected = min(config.wz_max, config.response_yaw_gain * math.atan2(P._keepout_m(config), 0.4))
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0
    assert abs(decision.values[5]) == pytest.approx(expected)
    assert decision.k_yaw_effective is None


def test_gap_turn_does_not_claim_tracking_gains_when_both_turns_equal_the_cap(monkeypatch):
    config = _cfg(response_yaw_gain=0.8, use_lateral=False,
                  stop_distance_m=0.3, floor_wz=0.0, wz_max=0.1)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs:
                        (5.0, 1.0) if kwargs.get("lateral_offset_m") else (0.4, 1.0))
    state = _adaptive_ready(config, distance=1.2, bearing=-0.9)
    decision = _profiled_frame(config, distance=1.2, bearing=-0.9,
                               odom={"wz": 0.1}, state=state)
    assert decision.status == P.AVOIDING
    assert -config.response_yaw_gain * decision.bearing > config.wz_max
    assert decision.values[5] == config.wz_max
    assert decision.k_yaw_effective is None


def test_driver_response_profile_does_not_prevent_arrival():
    config = _cfg(response_yaw_gain=0.8)
    decision = _profiled_frame(config, distance=0.7, bearing=0.0, odom={"wz": 0.1})
    assert decision.status == P.ARRIVED
    assert decision.values == [0.0] * 6
    assert decision.k_yaw_effective is None


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, {}, "bad"])
def test_driver_response_tracking_gain_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="response_yaw_gain"):
        _cfg(response_yaw_gain=value)


def test_driver_response_declaration_is_adopted_without_calibration():
    config = _cfg(k_yaw=0.6)
    notes = P.adopt_response(config, {
        "response": {"wz": {"source": "measured", "gain": 0.75,
                              "delay_s": 0.2, "settling_s": 0.4}},
    })
    expected = (0.6 / 0.75) / (1.0 + 0.2 * 0.2 + 0.1 * 0.4)
    assert config.response_yaw_gain == pytest.approx(expected)
    assert len(notes) == 1 and "response.wz" in notes[0]


def test_absent_driver_response_keeps_the_configured_gain():
    config = _cfg()
    assert P.adopt_response(config, {}) == []
    assert config.response_yaw_gain is None
    assert config.k_yaw == 0.6


@pytest.mark.parametrize("response", [
    {"wz": "bad"},
    {"wz": {}},
    {"wz": {"source": "measured", "gain": 1.0}},
    {"wz": {"source": "measured", "gain": 1.0, "delay_s": 0.1}},
    {"wz": {"source": "unknown", "gain": 1.0}},
    {"wz": {"source": "measured", "gain": 0}},
    {"wz": {"source": "measured", "gain": True,
            "delay_s": 0.1, "settling_s": 0.2}},
    {"wz": {"source": "measured", "gain": "1.0",
            "delay_s": 0.1, "settling_s": 0.2}},
    {"wz": {"source": "vendor-spec", "gain": float("nan")}},
    {"wz": {"source": "estimate", "gain": 1.0, "delay_s": -1}},
    {"wz": {"source": "measured", "gain": 1.0,
            "settling_s": float("inf")}},
])
def test_invalid_driver_response_falls_back_to_generic_yaw(response):
    config = _cfg()
    notes = P.adopt_response(config, {"response": response})
    assert config.response_yaw_gain is None
    assert len(notes) == 1 and "0.6" in notes[0]


def test_steer_early_only_for_an_obstacle_between_robot_and_target():
    obstacle = _scene(4.0, [(0.25, 1.0, 0.35)])
    turn = _step(detections=_detections(_obj()), depth=obstacle)
    assert turn.values[:2] == [0., 0.] and turn.values[5] > 0
    target = _step(detections=_detections(_obj()), depth=_scene(1.0),
                   config=_cfg(stop_distance_m=0.6))
    assert target.values[1] == target.values[5] == 0


def test_brake_before_box_and_resume_only_on_successive_new_clear_frames():
    cfg = _cfg(stop_distance_m=0.6)
    blocked = _scene(3.0, [(0.0, 0.66, 0.65)])
    clear = _scene(3.0)
    state = seed(_state(), _detections(_obj()), blocked, cfg)

    def tick(depth, speed=0.0):
        return _step(detections=_detections(_obj()), depth=depth,
                     state=state, config=cfg, odom={'vx': speed})

    assert tick(blocked, 0.34).values[:2] == [0.0, 0.0]
    first = dict(clear, frame_id=1)
    for _ in range(6):
        assert tick(first).values[:2] == [0.0, 0.0]
    assert tick(None).status == P.BLIND
    speeds = [tick(dict(clear, frame_id=i)).values[0] for i in range(2, 7)]
    assert speeds[:2] == [0.0, 0.0]
    assert 0 < speeds[2] < speeds[3] < speeds[4]


def test_box_blocking_both_routes_allows_only_a_slow_straight_approach():
    cfg = _cfg(stop_distance_m=0.6, floor_vx=0.04)
    box = _scene(2.8, [(0.0, 1.16, 0.65), (-0.58, 0.95, 0.2), (0.58, 0.95, 0.2)])
    detections = _detections(_obj(x=0.2))
    decision = _step(detections=detections, depth=box, config=cfg)
    horizon_s = (cfg.max_obs_age_ms + cfg.watchdog_ms + 50) / 1000
    assert decision.values[0] >= .04
    assert decision.values[0] * horizon_s <= 1.16 - .9 + 1e-12
    assert not any(decision.values[1:])
    assert "两侧也都不可用" in decision.reason

    # Reserve distance when the measured speed is still high.
    moving = _step(detections=detections, depth=box, config=cfg, odom={"vx": 0.5})
    assert moving.values == [0.0] * 6


def test_blocked_approach_stops_before_the_chassis_speed_floor_and_holds():
    cfg = _cfg(stop_distance_m=0.6, floor_vx=0.04)
    farther = _scene(2.8, [(0.0, 1.16, 0.65), (-0.58, 0.95, 0.2), (0.58, 0.95, 0.2)])
    closer = _scene(2.8, [(0.0, 0.98, 0.65)])
    state = seed(_state(), _detections(_obj()), farther, cfg)

    def tick(depth, frame):
        return _step(detections=_detections(_obj()), depth=dict(depth, frame_id=frame),
                     config=cfg, state=state)

    assert tick(farther, 100).values[0] > 0.04
    assert tick(closer, 200).values == [0.0] * 6
    for frame in (300, 400, 500):
        assert tick(farther, frame).values == [0.0] * 6


@pytest.mark.parametrize(("stop_distance", "target_range", "clearance", "expected_speed"), [
    (.60, 2.0, 1.28, .15),
    (1.45, 1.60, 1.28, .15 / 1.45),
    (.60, 2.0, .90, .25 / 2.25),
    (1.60, 1.60, 1.28, 0.),
    (.60, 2.0, .65, 0.),
    (.60, 2.0, .70, 0.),  # 5 cm left cannot supply the declared gait floor.
])
def test_local_no_route_approach_stops_at_the_stricter_boundary(
        monkeypatch, stop_distance, target_range, clearance, expected_speed):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    cfg = _cfg(local_planning=True, local_geometry=geometry, stop_distance_m=stop_distance,
               obstacle_stop_m=.60, max_depth_age_ms=800, watchdog_ms=600, floor_vx=.04)
    state = _state(motion_source="measured")
    state.active_scan.attempted = True
    odom = {"vx": 0., "vy": 0., "wz": 0.}
    tried = []
    local_translation = P._local_translation

    def plan(depth, config, state, odom, ahead, side):
        tried.append(side)
        return local_translation(depth, config, state, odom, ahead, side)

    monkeypatch.setattr(P, "_local_translation", plan)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (clearance, 1.))
    occupied = np.isclose(local_path.FORWARD, round(clearance / .05) * .05)
    monkeypatch.setattr(local_path, "observe", lambda *args: (~occupied, occupied))
    depth = dict(_scene(2.), ground=occupied, frame_id=1000)

    decision = P._avoid(.60, 0., 0., depth, cfg, P.APPROACHING, "", state,
                        target_distance=target_range, odom=odom)

    assert sorted(tried) == [-1., 1.]
    assert decision[0] == pytest.approx(expected_speed)
    assert decision[1:4] == (0., 0., P.AVOIDING)
    if expected_speed:
        assert "低速接近" in decision[4]
        if clearance == 1.28 and stop_distance == .60:
            # Braking remains safe while the previous speed still fits the gap.
            state.commanded = (decision[0], 0., 0.)
            odom["vx"] = decision[0]
            monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.88, 1.))
            following = P._avoid(.60, 0., 0., dict(depth, frame_id=1100), cfg,
                                 P.APPROACHING, "", state, target_distance=target_range, odom=odom)
            assert 0. < following[0] < decision[0]


@pytest.mark.parametrize("use_lateral", [True, False])
def test_shared_search_prefers_the_observed_open_side_if_forward_is_blocked(monkeypatch, use_lateral):
    config = _cfg(local_planning=True, use_lateral=use_lateral, lost_frames=1)
    state = _state(last_seen_side=1.)
    monkeypatch.setattr(P, "_clearance", lambda _d, _c, lateral_offset_m=0.: (
        1.80 if lateral_offset_m < 0. else 1.39 if lateral_offset_m > 0. else 1.37, .51))
    decision = P.step(detections=_detections(), depth=_depth(map_=_solid_depth(1.37)),
                      odom=None, config=config, state=state, dt=.1)
    assert decision.status == P.SEARCHING
    assert decision.values[:5] == [0.] * 5 and decision.values[5] > 0.


@pytest.mark.parametrize("use_lateral", [True, False])
def test_shared_avoidance_turns_and_settles_before_driving_straight(monkeypatch, use_lateral):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _cfg(local_planning=True, local_geometry=geometry, use_lateral=use_lateral,
                  vy_max=.4 if use_lateral else 0., half_width_m=.275, vx_max=.3, wz_max=1.)
    state = _state(motion_source="measured")
    target = SimpleNamespace(bearing_rad=0., range_m=4., range_known=True,
                             measured_range_m=4., observed=True)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    free = np.ones(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, ~free))

    def clearance(_, _config, lateral_offset_m=0.):
        if lateral_offset_m < 0.:
            return 2.50, .95
        if lateral_offset_m > 0.:
            return .80, .95
        return (2.50 if state.active_scan.pose[2] < -.30 else 1.30), .95

    monkeypatch.setattr(P, "_clearance", clearance)
    frame = np.ones((4, 4))
    previous = [0.] * 6
    last_turn = None
    for stamp in range(1000, 9000, 100):
        depth = dict(map=frame, frame_id=stamp, ground=frame.astype(bool))
        feedback = dict(vx=previous[0], vy=0., wz=previous[5], received_ms=stamp)
        decision = P.step(detections=_detections(), depth=depth, odom=feedback,
                          config=config, state=state, dt=.1)
        assert decision.values is not None
        previous = decision.values
        assert previous[1] == 0. and not (previous[0] and previous[5])
        if previous[5]:
            assert previous[5] > 0.
            last_turn = stamp
        if previous[0]:
            assert last_turn is not None and stamp - last_turn >= active_scan.SETTLE_MS
            break
    else:
        pytest.fail("The settled scan did not release a straight segment")


def test_local_blocked_approach_waits_for_active_observation_and_checks_the_forward_sweep(monkeypatch):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    cfg = _cfg(local_planning=True, local_geometry=geometry, stop_distance_m=.6)
    state = _state(motion_source="command")
    scan = state.active_scan
    scan.phase, scan.directions = "evaluate", (1., -1.)
    scan.receipt_only = scan.command_only = scan.attempted = scan.issued_yaw = True
    scan.side_frame = 1000
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(P, "_local_translation", lambda *args: None)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.28, 1.))
    monkeypatch.setattr(local_path, "observe", lambda *args: (~occupied, occupied))

    def tick(frame):
        depth = dict(_scene(2.), ground=occupied, frame_id=frame)
        return P._avoid(.60, 0., 0., depth, cfg, P.APPROACHING, "", state,
                        target_distance=2., may_scan=True)

    assert tick(1000)[:3] == (0., 0., 0.)
    assert scan.active and scan.directions == (-1.,)
    scan.phase, scan.side_frame = "evaluate", 1100
    assert tick(1100)[0] > 0.
    assert not scan.active and not scan.locked
    occupied[local_path.START[0] + 5, local_path.START[1]] = True
    assert tick(1200)[:3] == (0., 0., 0.)


def test_near_turn_holds_forward_when_the_obstacle_leaves_view():
    cfg = _cfg(stop_distance_m=0.6)
    near = _scene(4.0, [(0.25, 0.6, 0.3)])
    clear = _scene(4.0)
    state = seed(_state(), _detections(_obj()), near, cfg)

    def tick(depth, frame_id):
        return _step(detections=_detections(_obj()), depth=dict(depth, frame_id=frame_id),
                     state=state, config=cfg)

    for frame_id in (100, 200):
        turn = tick(near, frame_id)
        assert turn.values[:2] == [0., 0.]
        assert turn.values[5] > 0.
    assert tick(clear, 300).values == [0.0] * 6
    assert tick(clear, 400).values == [0.0] * 6
    assert tick(clear, 500).values[0] > 0


def test_clear_frames_after_a_stop_allow_searching_for_a_lost_target():
    cfg = _cfg(stop_distance_m=0.6, lost_frames=1)
    blocked = _scene(3.0, [(0.0, 0.85, 0.65)])
    state = seed(_state(), _detections(_obj()), blocked, cfg)
    stopped = _step(detections=_detections(_obj()), depth=blocked, state=state, config=cfg)
    assert stopped.values == [0.0] * 6
    state.tracker = P.Tracker()
    for frame_id in (100, 200, 300):
        decision = _step(detections=_detections(), state=state, config=cfg,
                         depth=dict(_scene(3.0), frame_id=frame_id))
        assert decision.values[:2] == [0.0, 0.0]
        if frame_id < 300:
            assert decision.values[5] == 0
        else:
            assert decision.status == P.SEARCHING and decision.values[5] != 0


def test_lost_target_does_not_start_a_turn_in_a_close_obstacle():
    cfg = _cfg(lost_frames=1)
    decision = _step(detections=_detections(), depth=_scene(0.4),
                     config=cfg, state=_state())
    assert decision.values == [0.0] * 6
    assert decision.status == P.AVOIDING


@pytest.mark.parametrize(("ahead", "side_obstacles", "expected_yaw"), [
    (.80, (), -.60),
    (.60, (), 0.),
    (2.0, (.40,), .60),
    (2.0, (.40, -.40), 0.),
])
def test_lost_target_checks_the_padded_turn_sweep(monkeypatch, ahead, side_obstacles, expected_yaw):
    from plugins.navi import active_scan as scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 600, 320), (480, 600, 240), .21)
    cfg = _cfg(lost_frames=1, local_planning=True, local_geometry=geometry,
               half_width_m=.21, obstacle_stop_m=.60, search_rate=.60)
    depth = dict(_scene(3.0, [(0., ahead, .5)]), frame_id=1000)
    depth["ground"] = np.zeros_like(depth["map"], dtype=bool)
    depth["ground"][0, 0] = True
    occupied = np.isclose(scan.FORWARD, ahead) & np.isclose(scan.RIGHT, 0.)
    for right in side_obstacles:
        # Beyond the stationary envelope and outside the front corridor, but
        # inside the right-turn sweep once the 10 cm margin is included.
        occupied |= np.isclose(scan.FORWARD, .15) & np.isclose(scan.RIGHT, right)
    monkeypatch.setattr(scan, "observe", lambda *args: (np.zeros_like(occupied), occupied))
    odom = {"vx": 0., "vy": 0., "wz": 0., "received_ms": 1000}

    decision = _step(detections=_detections(), depth=depth, odom=odom, config=cfg, state=_state())

    assert decision.values == [0., 0., 0., 0., 0., expected_yaw]
    assert decision.status == (P.SEARCHING if expected_yaw else P.AVOIDING)


def test_missing_target_range_does_not_disable_early_stop():
    mat = _obstacle_map(2.8, [(0.0, 1.16, 0.65)])
    mat[190:270, 260:380] = np.nan
    decision = _step(detections=_detections(_obj(bbox=(0.42, 0.43, 0.58, 0.54))),
                     depth=_depth(map_=mat), config=_cfg(stop_distance_m=0.6))
    assert decision.values == [0.0] * 6


def test_an_unexecutable_side_manoeuvre_does_not_keep_going_straight():
    obstacle = _scene(4.0, [(0.25, 1.0, 0.35)])
    decision = _step(detections=_detections(_obj()), depth=obstacle,
                     config=_cfg(use_lateral=False, k_yaw=0.0))
    assert decision.values == [0.0] * 6


def test_a_near_side_escape_has_a_blocked_deadline():
    cfg = _cfg(stop_distance_m=0.6, blocked_timeout_s=0.4)
    near = _scene(4.0, [(0.25, 0.6, 0.3)])
    state = seed(_state(), _detections(_obj()), near, cfg)
    for frame_id in range(100, 700, 100):
        decision = _step(detections=_detections(_obj()), depth=dict(near, frame_id=frame_id),
                         state=state, config=cfg, odom={"vx": 0.0, "vy": 0.0, "wz": 0.0})
        if decision.status == P.FAILED:
            break
    assert decision.status == P.FAILED


def test_right_route_shorter_than_forward_slow_distance_can_be_selected():
    cfg = _cfg(stop_distance_m=0.6)
    obstacle = _scene(1.2, [(-0.25, 0.85, 0.28)])
    decision = _step(detections=_detections(_obj()), depth=obstacle, config=cfg)
    assert decision.status == P.AVOIDING
    assert decision.values[:2] == [0., 0.] and decision.values[5] < 0
    assert "朝右侧避让" in decision.reason
    assert cfg.slow_distance_m == 1.5


def test_local_planning_holds_instead_of_using_an_unchecked_avoidance_turn():
    cfg = _cfg(stop_distance_m=0.6)
    cfg.local_planning = True
    obstacle = _scene(1.2, [(-0.25, 0.85, 0.28)])
    decision = _step(detections=_detections(_obj()), depth=obstacle, config=cfg,
                     odom={"vx": 0.0, "vy": 0.0, "wz": 0.0})
    assert decision.values == [0.0] * 6
    assert "局部规划" in decision.reason


def test_local_planning_reports_pose_only_odometry(monkeypatch):
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.85, .84))
    cfg, state = _cfg(local_planning=True, use_lateral=True), _state()
    depth = _scene(2.5)
    odom = {"vx": None, "vy": None, "wz": None, "received_ms": 500}

    decision = P._avoid(.25, 0., 0., depth, cfg, P.APPROACHING, "", state,
                        target_distance=2.5, odom=odom, may_scan=True)

    assert decision[:3] == (0., 0., 0.)
    assert not state.active_scan.active
    assert "缺少可用的速度或连续位姿" in decision[4]


def test_local_detour_starts_at_the_slowdown_boundary(monkeypatch):
    cfg, state = _cfg(local_planning=True, obstacle_stop_m=.60, slow_distance_m=1.50), _state()
    ahead = 1.50
    attempts = []
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0: (
        2.0 if lateral_offset_m else ahead, 1.0))

    def plan(depth, config, state, odom, ahead, side):
        attempts.append(ahead)
        return (.06, 0.)

    monkeypatch.setattr(P, "_local_translation", plan)

    def avoid():
        return P._avoid(.20, 0., 0., _scene(3.), cfg, P.APPROACHING, "", state,
                        target_distance=3., measured_vx=0.)

    assert avoid()[0] == .20
    assert attempts == []
    ahead = 1.15
    assert avoid()[:3] == (.06, 0., 0.)
    ahead = 1.35
    assert avoid()[:3] == (.06, 0., 0.)
    assert attempts == [1.15, 1.35]


def test_local_planner_does_not_execute_a_lateral_route_near_an_obstacle(monkeypatch):
    cfg = _cfg(local_planning=True)
    state = _state()
    attempts = []
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0: (
        .95 if lateral_offset_m > 0 else .85 if lateral_offset_m < 0 else .4, 1.0))

    def plan(depth, config, state, odom, ahead, side):
        attempts.append(side)
        return (.04, -.06) if side > 0 else (0.0, .06)

    monkeypatch.setattr(P, "_local_translation", plan)
    command = P._avoid(.1, 0, 0, _scene(2.8), cfg, P.AVOIDING, "", state,
                       target_distance=2.8)

    assert attempts == [1.0, -1.0]
    assert command[:3] == (0., 0., 0.)
    assert state.escape_side == 0.


def test_local_planner_holds_side_through_centimetre_noise_and_switches_if_blocked(monkeypatch):
    cfg, state = _cfg(local_planning=True), _state()
    rooms = {-1.: 2., 1.: 2.01}
    blocked = set()
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0: (
        rooms[np.sign(lateral_offset_m)] if lateral_offset_m else .9, 1.0))
    monkeypatch.setattr(P, "_local_translation", lambda depth, config, state, odom, ahead, side:
                        None if side in blocked else (.06, 0.))

    def step(frame):
        P._avoid(.1, 0, 0, dict(_scene(2.8), frame_id=frame), cfg,
                 P.AVOIDING, "", state, target_distance=2.8)
        return state.escape_side

    assert step(100) > 0
    for frame in range(200, 800, 100):
        rooms[-1.], rooms[1.] = rooms[1.], rooms[-1.]
        assert step(frame) > 0
    blocked.add(1.)
    assert step(800) < 0


def test_local_planner_checks_the_obstacle_at_the_start_of_the_slowdown_band(monkeypatch):
    cfg = _cfg(local_planning=True)
    state = _state(motion_source="command")
    monkeypatch.setattr(P, "_clearance", lambda *_args, **_kwargs: (1.48, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *_args: (.10, 0.))

    vx, vy, _, _, _ = P._avoid(.2, 0., 0., _scene(2.16), cfg,
                               P.APPROACHING, "", state, target_distance=2.16)

    assert (vx, vy) == (.10, 0.)


def test_local_planner_requires_distinct_frames_before_switching_for_more_space(monkeypatch):
    cfg, state = _cfg(local_planning=True), _state(escape_side=1.)
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0: (
        2.3 if lateral_offset_m < 0 else 2.0 if lateral_offset_m > 0 else .9, 1.0))
    monkeypatch.setattr(P, "_local_translation", lambda depth, config, state, odom, ahead, side:
                        (.06, 0.))

    def step(frame):
        P._avoid(.1, 0, 0, dict(_scene(2.8), frame_id=frame), cfg,
                 P.AVOIDING, "", state, target_distance=2.8)
        return state.escape_side

    assert all(step(100) > 0 for _ in range(5))
    assert step(200) > 0
    assert step(300) < 0


def test_local_recovery_uses_the_current_checked_speed_without_waiting_for_frames(monkeypatch):
    cfg = _cfg(local_planning=True, floor_vx=.04, floor_vy=.04)
    state = _state(escape_side=1., recovery=P.ClearanceRecovery("right", 2, 100))
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0: (
        2.1 if lateral_offset_m > 0 else 2.0 if lateral_offset_m < 0 else .9, 1.0))
    monkeypatch.setattr(P, "_local_translation", lambda *args: (.063, 0.))

    def step(frame):
        return P._avoid(.1, 0, 0, dict(_scene(2.8), frame_id=frame), cfg,
                        P.AVOIDING, "", state, target_distance=2.8)[:3]

    assert step(200) == (.063, 0., 0.)
    assert state.recovery is None


def test_local_detour_reuses_receipt_only_projection_for_the_same_geometry(monkeypatch):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .23, .23, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="command")
    empty = np.zeros_like(local_path.FORWARD, dtype=bool)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.views.append((1000, (0., 0., 0.), empty, empty))
    depth = dict(_scene(2.), ground=empty, frame_id=1000)
    calls = []
    monkeypatch.setattr(local_path, "observe", lambda *args: (calls.append(True) or empty, empty))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.3, 1.))

    P._avoid(.2, 0., 0., depth, cfg, P.APPROACHING, "", state, target_distance=2.)
    assert not calls
    P._avoid(.2, 0., 0., dict(depth, frame_id=1100), cfg, P.APPROACHING, "", state,
             target_distance=2.)
    assert len(calls) == 1


def test_local_command_respects_moving_floors_and_the_signed_driver_bounds():
    cfg = _cfg(local_planning=True)
    P.adopt_limits(cfg, {"limits": {"lower": [-.3, -.03, 0., 0., 0., -.08],
                                   "upper": [.3, .2, 0., 0., 0., .4],
                                   "min_magnitude": [.04, 0., 0., 0., 0., 0.],
                                   "min_magnitude_moving": [.06, .05, 0., 0., 0., 0.]}})
    assert P._local_velocity_limits(cfg, 1.) == ((.06, .3), (.05, .03))
    assert P._local_velocity_limits(cfg, -1.) == ((.06, .3), (.05, .2))
    assert P._scan_yaw_rate(cfg) == .08


def test_local_translation_continues_at_declared_speed_if_checked(monkeypatch):
    cfg, state = _cfg(local_planning=True), _state()
    monkeypatch.setattr(state.local_plan, "translation", lambda *args, **kwargs: (.25, 0.))
    odom = {"vx": .20, "vy": 0., "wz": 0.}
    assert P._local_translation({}, cfg, state, odom, .9, 1.) == (.25, 0.)
    odom["vx"] = .30
    assert P._local_translation({}, cfg, state, odom, .9, 1.) is P.LocalHold.TRANSITION


@pytest.mark.parametrize("odom", [{"vx": 0., "vy": -.05, "wz": 0.},
                                  {"vx": 0., "vy": 0., "wz": .035}])
def test_local_prefix_waits_for_motion_off_its_checked_axis(monkeypatch, odom):
    state = _state()
    monkeypatch.setattr(state.local_plan, "translation", lambda *args, **kwargs: (.08, 0.))
    assert (P._local_translation({}, _cfg(local_planning=True), state, odom, .9, 1.)
            is P.LocalHold.TRANSITION)


@pytest.mark.parametrize("occupied,known", [(False, True), (True, True), (False, False)])
def test_local_transition_checks_independent_axis_ramp(monkeypatch, occupied, known):
    import numpy as np
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg, state = _cfg(local_planning=True, scan_geometry=geometry), _state(escape_side=1.)
    grid_free = np.ones(local_path.SHAPE, dtype=bool)
    grid_occupied = np.zeros_like(grid_free)
    row, col = local_path.START
    grid_occupied[row + 7, col - 6] = occupied
    grid_free[row + 7, col - 6] = known
    depth = {"scan_grid": (grid_free, grid_occupied)}
    monkeypatch.setattr(state.local_plan, "translation", lambda *args, **kwargs: (.25, -.10))
    odom = {"vx": .16, "vy": 0., "wz": 0.}
    expected = P.LocalHold.TRANSITION if occupied else (.25, -.10)
    assert P._local_translation(depth, cfg, state, odom, .9, 1.) == expected


def test_blocked_transition_keeps_current_detour_and_recovery(monkeypatch):
    cfg = _cfg(local_planning=True)
    recovery = P.ClearanceRecovery("right", 4, 100)
    state = _state(escape_side=1., recovery=recovery)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.9, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *args: P.LocalHold.TRANSITION)
    decision = P._avoid(.2, 0., 0., {}, cfg, P.APPROACHING, "", state, target_distance=2.)
    assert decision[:3] == (0., 0., 0.)
    assert state.escape_side == 1. and state.recovery is recovery
    assert not state.active_scan.attempted


def test_command_only_checks_the_ramp_even_for_a_repeated_diagonal(monkeypatch):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _cfg(local_planning=True, scan_geometry=geometry)
    command = (.25, -.10)
    state = _state(motion_source="command", commanded=(*command, 0.), command_tail=(*command, 0.))
    monkeypatch.setattr(state.local_plan, "translation", lambda *args, **kwargs: command)
    free = np.ones(local_path.SHAPE, dtype=bool)
    row, col = local_path.START
    free[row + 7, col - 6] = False
    occupied = np.zeros_like(free)
    depth = {"scan_grid": (free, occupied)}
    assert P._local_translation(depth, cfg, state, None, .9, 1.) == command
    occupied[row + 7, col - 6] = True
    assert P._local_translation(depth, cfg, state, None, .9, 1.) is P.LocalHold.TRANSITION


@pytest.mark.parametrize("odom", [None, {"vx": 0., "vy": 0., "wz": 0.}])
@pytest.mark.parametrize("already_occluded", [False, True])
def test_local_detour_keeps_original_target_through_retarget_and_occlusion(monkeypatch, odom,
                                                                         already_occluded):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry)
    state = _state()
    track = SimpleNamespace(bearing_rad=0., range_m=2., range_known=True, observed=True,
                             measured_range_m=2.)
    observation = [SimpleNamespace(bearing_rad=0., range_m=2.2, range_known=True, observed=False,
                                   measured_range_m=None, last_obs=(2., 0.))
                   if already_occluded else track]
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: observation[0])
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.9, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *args: (.1, 0.))

    stamp = iter(range(1000, 3000, 100))

    def tick():
        return P.step(detections={"objects": []}, depth=dict(_scene(2.), frame_id=next(stamp)), odom=odom,
                      config=cfg, state=state, dt=.1)

    assert tick().values[0] == .1
    observation[0] = SimpleNamespace(bearing_rad=0., range_m=.5, range_known=True,
                                     observed=True, measured_range_m=.5)
    decision = tick()
    assert decision.status == P.AVOIDING
    assert decision.distance_m == pytest.approx(1.99 if odom is None else 2.)
    observation[0] = None
    decision = tick()
    assert decision.status == P.AVOIDING and decision.values[0] == .1
    assert decision.distance_m == pytest.approx(1.98 if odom is None else 2.)

    # Once the route opens, wait for a live range from the frozen goal.
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2., 1.))
    decision = tick()
    assert decision.values == [0.] * 6 and not state.arrived
    observation[0] = track
    tick()
    assert state.goal_anchor is None


def test_command_only_stops_before_switching_from_turn_to_translation(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="command", commanded=(0., 0., .4))
    target = SimpleNamespace(bearing_rad=0., range_m=2.5, range_known=True, observed=True,
                             measured_range_m=2.5)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.5, 1.))

    def tick(frame):
        return P.step(detections={"objects": []}, depth=dict(_scene(2.5), frame_id=frame),
                      odom=None, config=cfg, state=state, dt=.1)

    assert tick(1000).values == [0.] * 6
    assert tick(1100).values == [0.] * 6
    assert tick(1200).values == [0.] * 6
    assert tick(1300).values == [0.] * 6
    assert tick(1400).values[0] > 0.


def test_lost_motion_feedback_only_brakes_and_reuses_unexpired_depth(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="measured", commanded=(.2, 0., 0.))
    target = SimpleNamespace(bearing_rad=0., range_m=2.5, range_known=True, observed=True,
                             measured_range_m=2.5)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.5, 1.))

    def tick(frame, source):
        depth = dict(_scene(2.5), frame_id=1000, control_ms=frame, source_frame_id=source)
        return P.step(detections={"objects": []}, depth=depth,
                      odom=None, config=cfg, state=state, dt=.1)

    for frame in range(1000, 1400, 100):
        assert tick(frame, 42).values == [0.] * 6
    assert state.motion_source == "command"
    assert tick(1400, 42).values[0] > 0.
    assert tick(1501, 42).status == P.BLIND


def test_losing_feedback_during_a_turn_preserves_the_scan_lock(monkeypatch):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, scan_geometry=geometry)
    state = _state(motion_source="measured")
    state.active_scan.phase = "turn"
    state.active_scan.attempted = state.active_scan.issued_yaw = True

    def tick(frame):
        return P.step(detections={"objects": []}, depth=dict(_scene(2.5), frame_id=frame),
                      odom=None, config=cfg, state=state, dt=.1)

    for frame in range(1000, 1800, 100):
        assert tick(frame).values == [0.] * 6
    assert state.active_scan.locked and state.active_scan.attempted
    assert not state.active_scan.active


def test_command_only_cannot_reuse_a_frozen_depth_frame(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, max_depth_age_ms=800)
    state = _state()
    target = SimpleNamespace(bearing_rad=0., range_m=2.5, range_known=True, observed=True,
                             measured_range_m=2.5)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.5, 1.))

    def tick(received, source):
        depth = dict(_scene(2.5), frame_id=received, control_ms=received, source_frame_id=source)
        return P.step(detections={"objects": []}, depth=depth, odom=None,
                      config=cfg, state=state, dt=.1)

    assert tick(1000, 100).values[0] > 0.
    assert tick(1800, 100).values[0] > 0.
    assert tick(1801, 100).status == P.BLIND
    assert tick(1802, 99).status == P.BLIND
    assert tick(1803, 101).values[0] > 0.
