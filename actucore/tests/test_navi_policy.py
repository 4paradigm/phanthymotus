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

from plugins.navi.active_scan import NearProjection  # noqa: E402
from plugins.navi import depth as D  # noqa: E402
from plugins.navi import policy as P  # noqa: E402


def _cfg(**over):
    return P.Config(**over)


def _legacy_cfg(**over):
    """Exercise historical segment ordering and timers as fallback contracts.

    General policy tests use the production default through _cfg. Only tests
    named test_legacy_* select the old controller explicitly; its turn/settle
    phases, moving target anchors and blockage clock differ from DWA.
    """
    return P.Config(local_controller="legacy", **over)


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
        self.elapsed_s = 0.
        self.last_motion = (0., 0., 0.)
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
        self.last_motion = (tuple(values[i] for i in (0, 1, 5))
                            if values is not None else (0., 0., 0.))
        if values is not None:
            theta = self.pose[2]
            cos, sin = np.cos(theta), np.sin(theta)
            vx, vy = values[0], values[1]
            self.pose[0] += (cos * vx - sin * vy) * dt
            self.pose[1] += (sin * vx + cos * vy) * dt
            self.pose[2] += values[5] * dt
        self.target += self.velocity * dt
        self.elapsed_s += dt

    def run(self, state, ticks, config=None, dt=0.1, odom=True):
        """Closed loop. Returns the last decision."""
        config = config or self.config
        decision = None
        for _ in range(ticks):
            detections, depth = self.observe()
            measured = None
            if odom:
                # Preserve the actual preceding motion across run(ticks=1)
                # calls. DWA progress requires an ordered producer sample,
                # rather than an unclocked velocity or a new zero each call.
                stamp = 1000 + round(self.elapsed_s * 1000)
                measured = dict(zip(("vx", "vy", "wz"), self.last_motion),
                                received_ms=stamp, source_sample_id=stamp)
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
    count = config.confirm_hits if frames is None else frames
    for index in range(count):
        # Confirmation is history preceding the observation under test. DWA
        # rejects out-of-order producer frames, so seeding with later stamps
        # would make a fresh-looking single-tick test actually replay old data.
        history = (dict(detections, timestamp=detections["timestamp"] - (count - index) * dt)
                   if detections is not None and "timestamp" in detections else _next_frame(detections))
        state.tracker.step(detections=history, depth=depth,
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


@pytest.mark.parametrize("distance,arrived", [(1.099, False), (1.1, True), (1.2, True),
                                              (1.25, True), (1.3, True), (1.301, False)])
def test_arrival_tolerance_accepts_only_distances_within_the_parking_ring(distance, arrived):
    config = _cfg(stop_distance_m=1.2, stop_tolerance_m=0.1)
    decision = _step(detections=_detections(_obj(bbox=(0.4, 0.4, 0.6, 0.6))),
                     depth=_depth(map_=_solid_depth(distance)), config=config)
    assert (decision.status == P.ARRIVED) is arrived
    if arrived:
        assert decision.values == [0.0] * 6
        assert "1.20 m ± 0.10 m tolerance" in decision.reason


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


@pytest.mark.parametrize("start_heading", [0., math.pi / 2, math.pi, -math.pi / 2])
def test_parking_handles_any_approach_direction_and_aligns_in_place(start_heading):
    config = _cfg(stop_distance_m=.60, stop_tolerance_m=.05, arrive_align_tol=.05)
    sim = Sim(range_m=.62, bearing=.65, config=config)
    sim.pose[2] = start_heading
    c, t = math.cos(start_heading), math.sin(start_heading)
    sim.target = np.array(((c, -t), (t, c))) @ sim.target
    state = seed(_state(), *sim.observe(), config)
    for _ in range(100):
        decision = sim.run(state, ticks=1, odom=False)
        assert decision.values[:2] == [0., 0.]
        if state.arrived:
            break
    assert decision.status == P.ARRIVED and decision.values == [0.] * 6
    distance, angle = sim.relative()
    assert distance == pytest.approx(.62)
    assert abs(angle) <= config.arrive_align_tol * config.half_fov_rad


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
                  arrive_align_tol=0.01)

    def clearance(depth, config, lateral_offset_m=0.0):
        # A near pillar in the forward corridor, and clear space to its side.
        return (0.4 if lateral_offset_m == 0 else 5.0), 1.0

    monkeypatch.setattr(P, "_clearance", clearance)
    decision = _step(detections=_detections(_obj(x=0.25, bbox=(0.5, 0.4, 0.7, 0.6))),
                     depth=_depth(map_=_solid_depth(1.20)), config=config)
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


def test_small_tolerance_can_start_and_finish_a_short_approach():
    config = _cfg(stop_distance_m=.80, stop_tolerance_m=.01, floor_vx=.05)
    sim = Sim(range_m=.83, config=config)
    state = seed(_state(), *sim.observe(), config)

    command = sim.run(state, ticks=1)
    assert command.values[0] > 0.
    sim.run(state, ticks=50)

    assert state.arrived
    assert config.arrival_min_distance_m <= sim.relative()[0] <= config.arrival_distance_m + .001


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
    config = _cfg(obstacle_stop_m=.60)
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
    for expected in (0.1, 0.2, 0.3):
        decision = _step(detections=_detections(), depth=_depth(),
                         odom={"vx": 0.0, "vy": 0.0, "wz": measured},
                         config=config, state=state, dt=0.1)
        assert math.isfinite(state.searched_rad)
        assert state.searched_rad == pytest.approx(expected)
        if expected < 0.3:
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

def test_legacy_commanded_but_not_moving_is_stuck():
    config = _legacy_cfg(stuck_window_s=0.3)
    state = _state()
    for _ in range(6):
        decision = _step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                         odom={"vx": 0.0}, config=config, state=state, dt=0.1)
    # STUCK is a *reason*; the terminal status the caller sees is FAILED.
    assert decision.status == P.FAILED
    assert "blocked" in decision.reason
    assert decision.publishes is False


def test_legacy_blind_stops_end_the_current_pose_only_stuck_interval():
    config = _legacy_cfg(stuck_window_s=.3, confirm_hits=1)
    state = _state()
    odom = {"pose_feedback": (0., 0., 0.)}
    for _ in range(6):
        for _ in range(3):
            decision = _step(detections=_detections(_obj()), depth=_depth(),
                             odom=odom, config=config, state=state, dt=.1)
            assert decision.status != P.FAILED
        assert state.moving_for_s > 0.
        stopped = _step(detections=_detections(_obj()), depth=None,
                        odom=odom, config=config, state=state, dt=.1)
        assert stopped.status == P.BLIND
        assert state.moving_for_s == 0.


def test_ego_motion_prefers_pose_feedback_to_an_unmeasured_axis():
    state = _state(commanded=(.3, .2, .5))
    motion, measured = P._ego_twist({"vx": 0., "pose_feedback": (.03, None, .02)}, state)
    assert motion == (0., .2, .02)
    assert not measured


def test_anchor_predicts_motion_only_until_the_previous_command_expires():
    anchor = P.DetourGoal(3., 0.)
    state = _state(goal_anchor=anchor, commanded=(0., 0., .5))
    P.step(detections=None, depth=None, odom=None, config=_cfg(watchdog_ms=300),
           state=state, dt=3.1)
    assert anchor.predicted_rad == pytest.approx(.15)


@pytest.mark.parametrize("association", ["same_track", "reacquired", "different_position"])
def test_legacy_arrival_inside_an_active_drive_segment_requires_the_original_measurement(monkeypatch, association):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="measured")
    target = SimpleNamespace(observed=True, range_known=True, range_m=.84, measured_range_m=.84,
                             bearing_rad=1. if association != "reacquired" else 0.)
    original = target if association == "same_track" else SimpleNamespace()
    state.goal_anchor = P.DetourGoal(.84, 0., track=original)
    scan = state.active_scan
    scan.receipt_only, scan.phase, scan.advance_m = True, "evaluate", 1.5
    scan.advance_origin = scan.pose
    scan.result_frame = 1000
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    free = np.ones(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, ~free))
    frame = np.full((8, 8), 3.)
    depth = dict(map=frame, ground=frame.astype(bool), frame_id=1000)
    decision = P.step(detections=_detections(), depth=depth,
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000),
                      config=config, state=state, dt=.1)
    assert not scan.active and not scan.result_frame
    assert decision.values[:2] == [0., 0.]
    if association == "same_track":
        assert decision.status == P.ALIGNING and decision.values[5] < 0.
        target.bearing_rad = 0.
        decision = P.step(detections=_detections(), depth=dict(depth, frame_id=1100),
                          odom=dict(vx=0., vy=0., wz=0., received_ms=1100),
                          config=config, state=state, dt=.1)
    assert (decision.status == P.ARRIVED) is (association != "different_position")
    if association == "different_position":
        assert state.goal_anchor is not None and not any(decision.values)


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


@pytest.mark.parametrize("odom", [False, True], ids=["unwired", "measured"])
def test_default_controller_finishes_a_long_approach_with_available_feedback(odom):
    """Default DWA preserves direct motion with or without wired odometry."""
    config = P.Config(vx_max=.15)
    assert config.local_controller == "dwa"
    sim = Sim(range_m=3., config=config)
    state = seed(_state(), *sim.observe(), config)
    for tick in range(250):
        decision = sim.run(state, ticks=1, dt=.1, odom=odom)
        assert decision.status != P.FAILED, decision.reason
        assert not state.dwa_selected
        if state.arrived:
            break
    assert state.arrived
    assert (tick + 1) * .1 > config.idle_timeout_s
    assert config.arrival_min_distance_m <= sim.relative()[0] <= config.arrival_distance_m


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


def test_being_at_stop_distance_never_completes_while_heading_is_wrong():
    config = _cfg(arrive_align_tol=0.01)
    state = _state()
    for _ in range(42):
        decision = _step(detections=_detections(_obj(x=0.30)),
                         depth=_depth(map_=_solid_depth(.8)),
                         config=config, state=state, dt=0.1)
    assert decision.status == P.ALIGNING
    assert decision.values[:2] == [0., 0.] and decision.values[5] < 0.
    assert not state.arrived


def test_aligned_target_inside_stop_ring_does_not_count_as_arrived():
    config = _cfg(stop_distance_m=.6, stop_tolerance_m=.05)
    decision = _step(detections=_detections(_obj()), depth=_depth(map_=_solid_depth(.50)),
                     config=config)
    assert decision.status != P.ARRIVED and not any(decision.values)


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

    The near edge of a doorframe centred 0.22 m off-axis at 0.7 m falls inside the keepout and
    outside the centre third of the picture. The bands file it under "left",
    and the left band has never stopped forward motion."""
    # `obstacle_stop_m` pinned to what this scenario was measured at; the
    # shipped default has since been tuned down on r1_sz.
    config = _cfg(obstacle_stop_m=0.8)
    P.adopt_limits(config, dict(_R1_DESC, footprint=_R1_FOOTPRINT))
    depth = {"map": _obstacle_map(5.0, [(-0.22, 0.7, 0.1)], config=config),
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
    (3.0, 0, P.AVOIDING),
    (0.4, 0, P.AVOIDING),
    (np.nan, 0, P.AVOIDING),
    (3.0, 801, P.BLIND),
])
def test_local_search_waits_for_a_checkable_sweep_grid(distance, age_ms, status):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .23, .23, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, max_depth_age_ms=800)
    depth = dict(_depth(map_=_solid_depth(distance)), frame_id=1000,
                 control_ms=1000 + age_ms)
    state = _state(missing_frames=cfg.lost_frames - 1)
    decision = P.step(detections=_detections(), depth=depth, odom=None,
                      config=cfg, state=state, dt=.1)

    assert decision.status == status
    assert not any(decision.values or ())


def test_search_near_stop_keeps_clearance_reserve_after_target_reappears():
    from plugins.navi.local_path import Geometry

    config = _cfg(local_planning=True, local_geometry=Geometry(None, None, np.eye(4), .275, .275, None),
                  obstacle_stop_m=.6, stop_distance_m=.6, stop_tolerance_m=.05, lost_frames=1)
    state = _state()
    target = _obj(x=0., y=-.65, bbox=[.40, .03, .60, .25])
    for index, distance in enumerate((.40, .40, .40, .72, .72)):
        stamp = 1000 + index * 100
        frame = np.full((48, 64), distance, dtype=np.float32)
        frame[1:15, 24:40] = .72
        decision = P.step(detections=_detections(target) if index else _detections(),
                          depth=dict(map=frame, frame_id=stamp, control_ms=stamp),
                          odom=dict(vx=0., vy=0., wz=0., received_ms=stamp),
                          config=config, state=state, dt=.1)
    assert decision.distance_m == pytest.approx(.72)
    assert decision.values == [0.] * 6
    assert "足够余量" in decision.reason


def test_search_holds_when_the_circular_sweep_is_blocked(monkeypatch):
    from plugins.navi.local_path import Geometry

    geometry = Geometry(None, None, np.eye(4), .23, .23, None)
    config = _cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="command", last_seen_side=1., searched_rad=1.)
    state.active_scan.current_frame = 1000
    for direction in (-1., 1.):
        state.last_seen_side = direction
        monkeypatch.setattr(state.active_scan, "clear_sweep", lambda *_: False)
        assert "转身扫掠范围有障碍" in P._search_turn_stop(_depth(), config, state)
        assert state.last_seen_side == direction


def test_legacy_command_only_active_observation_uses_camera_geometry_without_extrinsics(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .21, .21, None)
    cfg = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.21,
               max_depth_age_ms=800, floor_wz=.33, wz_max=.83)
    state = _state()
    target = SimpleNamespace(bearing_rad=0., range_m=2., range_known=True,
                             observed=True, measured_range_m=2.)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.9, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *args: None)
    free = (abs(local_path.RIGHT) <= .60) & (abs(local_path.FORWARD) <= .60)
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
    depth = _depth(map_=_solid_depth(0.8))
    state = seed(_state(), _detections(_obj(x=0.0)), depth, config)

    # The target goes out of view. The track survives on its prediction, which
    # still says "right in front of us".
    decision = P.step(detections={"objects": []}, depth=depth, odom=None,
                      config=config, state=state, dt=0.1)
    assert decision.status != P.ARRIVED
    assert not state.arrived
    assert "等待实际测距" in decision.reason and not any(decision.values)

    # It comes back, and now the claim is supported.
    for _ in range(config.confirm_hits + 1):
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                          odom=None, config=config, state=state, dt=0.1)
    assert decision.status == P.ARRIVED


def test_legacy_a_corridor_that_stays_blocked_fails_saying_so():
    """Trying to get round it is the right first move, and `_avoid` makes it.
    But a robot that has been shuffling sideways for twelve seconds is not
    making progress, and the caller is owed the real reason — "blocked", not a
    generic idle timeout, and certainly not "arrived"."""
    config = _legacy_cfg(blocked_timeout_s=1.0, stop_distance_m=0.3)
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


def test_legacy_a_corridor_that_clears_resets_the_blocked_clock():
    """A person stepping across the path is not a blockage."""
    config = _legacy_cfg(blocked_timeout_s=1.0, stop_distance_m=0.3)
    blocked = _depth(map_=_solid_depth(0.4))
    clear = _depth(map_=_solid_depth(5.0))
    state = seed(_state(), _detections(_obj(x=0.0)), blocked, config)

    for _ in range(30):
        depth = blocked if _ % 10 < 8 else clear
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=depth,
                          odom=None, config=config, state=state, dt=0.1)
        assert decision.status != P.FAILED


@pytest.mark.parametrize(("feedback", "fails"), [(-.12, False), (-.039, True), (.12, True)])
def test_legacy_lateral_only_detour_uses_measured_progress_for_the_blockage_clock(monkeypatch, feedback, fails):
    from types import SimpleNamespace

    cfg = _legacy_cfg(local_planning=True, blocked_timeout_s=.2)
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
def test_legacy_local_avoidance_executes_lateral_candidates_with_progress(monkeypatch, moves_sideways):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    cfg = _legacy_cfg(local_planning=True, local_geometry=geometry, stuck_window_s=.4)
    state = _state(motion_source="measured")
    state.active_scan.attempted = True
    target = SimpleNamespace(bearing_rad=0., range_m=2.4, range_known=True,
                             observed=True, measured_range_m=2.4)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0.: (
        1.45 if lateral_offset_m > 0. else 1.35 if lateral_offset_m < 0. else 1.1, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda depth, config, state, odom, ahead, side:
                        (0., -.03 * side))
    occupied = np.isclose(local_path.FORWARD, 1.10) & (abs(local_path.RIGHT) <= .10)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~occupied, occupied))
    scene = _scene(2.4)
    decisions = []
    for stamp in range(1000, 2400, 200):
        depth = dict(scene, frame_id=stamp, ground=np.ones_like(scene["map"], dtype=bool))
        odom = {"vx": state.commanded[0], "vy": -.08 if moves_sideways else -.039,
                "wz": 0., "received_ms": stamp}
        decisions.append(P.step(detections=_detections(), depth=depth, odom=odom,
                                config=cfg, state=state, dt=.2))
    assert decisions[0].values[:2] == [0., -.03]
    if moves_sideways:
        assert all(decision.values[:2] == [0., -.03] for decision in decisions)
    else:
        assert any(decision.values[1] >= 0. for decision in decisions[1:])


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
    config = _cfg(half_fov_rad=true_fov, obstacle_stop_m=.60)
    # R1 声明物理半宽 0.179 m；走廊使用协商后的半宽和每侧 10% 余量。
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

    # With the smaller derived keepout the miscalibration shows up in a narrower door.
    depth = _doorway(0.60, 0.6, half_fov_rad=true_fov)
    correct, _ = D.corridor(depth, half_width_m=keep, half_fov_rad=true_fov,
                            reference_m=config.obstacle_stop_m)
    assert correct > config.obstacle_stop_m
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
                  k_fwd=0.4, use_lateral=False, obstacle_stop_m=.60)
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
@pytest.mark.parametrize("measured_wz", [-1.0, -0.2, 0.0, 0.2, 1.0, None])
def test_driver_response_tracking_is_pure_p_within_the_speed_limit(
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
    decision = _profiled_frame(config, odom={"wz": .1})
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
    decision = _profiled_frame(config, distance=0.8, bearing=0.0, odom={"wz": 0.1})
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


def test_brake_before_box_and_resume_on_the_current_clear_observation():
    cfg = _cfg(stop_distance_m=0.6)
    blocked = _scene(3.0, [(0.0, 0.66, 0.65)])
    clear = _scene(3.0)
    state = seed(_state(), _detections(_obj()), blocked, cfg)

    def tick(depth, speed=0.0):
        return _step(detections=_detections(_obj()), depth=depth,
                     state=state, config=cfg, odom={'vx': speed})

    assert tick(blocked, 0.34).values[:2] == [0.0, 0.0]
    first = dict(clear, frame_id=1)
    assert tick(first).values[0] > 0.
    assert tick(None).status == P.BLIND
    assert tick(first).values[0] > 0.


def test_box_blocking_both_routes_allows_only_a_slow_straight_approach():
    cfg = _cfg(stop_distance_m=0.6, obstacle_stop_m=.60, floor_vx=0.04)
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


@pytest.mark.parametrize("case", ["clear_prefix", "clear_prefix_inside_preview", "blocked_prefix", "detour_found"])
def test_legacy_local_planning_uses_checked_forward_prefix_before_observing(monkeypatch, case):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20, at_front=True)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, stop_distance_m=.60,
                  obstacle_stop_m=.60, max_depth_age_ms=800, watchdog_ms=600)
    state = _state(motion_source="measured")
    look = state.active_scan
    look.receipt_only = look.buffered_odom = True
    look.current_frame = 1000
    look.poses.append((1000, look.pose, 0.))
    occupied = np.isclose(local_path.FORWARD, 1.15) & np.isclose(local_path.RIGHT, .20)
    occupied |= np.isclose(local_path.FORWARD, 1.75) & (local_path.RIGHT == 0.)
    if case in ("blocked_prefix", "detour_found"):
        occupied |= np.isclose(local_path.FORWARD, .15) & (local_path.RIGHT == 0.)
    look.views.append((1000, look.pose, ~occupied, occupied))
    frame = _scene(2.)
    frame.update(ground=np.ones_like(frame["map"], dtype=bool), frame_id=1000)
    clearance = 1.15 if case == "clear_prefix_inside_preview" else 1.76
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (clearance, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *args: None)
    routes_tried = []

    def plan(*args, **kwargs):
        routes_tried.append(True)
        return case == "detour_found" and look.begin((-1.,), advance_m=1., route_segment=True)

    monkeypatch.setattr(P, "_start_detour_segment", plan)
    decision = P._avoid(.5, 0., -.12, frame, config, P.APPROACHING, "", state,
                        target_distance=1.93, target_bearing=.15, may_scan=True,
                        odom={"vx": 0., "vy": -.034, "wz": 0.})
    if case.startswith("clear_prefix"):
        assert not routes_tried
        assert decision[:3] == (.15, 0., -.12)
        assert not look.active
        assert state.checked_prefix
    else:
        assert routes_tried
        assert decision[:3] == (0., 0., 0.)
        assert look.active and look.route_segment == (case == "detour_found")


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
@pytest.mark.parametrize("front", [.59, 1.30])
def test_legacy_shared_avoidance_turns_and_settles_before_driving_straight(monkeypatch, use_lateral, front):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=use_lateral,
                  vy_max=.4 if use_lateral else 0., half_width_m=.275, vx_max=.3, wz_max=1.)
    state = _state(motion_source="measured")
    target = SimpleNamespace(bearing_rad=0., range_m=4., range_known=True,
                             measured_range_m=4., observed=True)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    free = np.ones(local_path.SHAPE, dtype=bool)
    obstacle = np.isclose(local_path.FORWARD, round(front / .05) * .05) & np.isclose(local_path.RIGHT, .25)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (
        (free, ~free) if state.active_scan.pose[2] < -.30 else (~free, obstacle)))

    def clearance(_, _config, lateral_offset_m=0.):
        if lateral_offset_m < 0.:
            return 2.50, .95
        if lateral_offset_m > 0.:
            return .80, .95
        return (2.50 if state.active_scan.pose[2] < -.30 else front), .95

    monkeypatch.setattr(P, "_clearance", clearance)
    frame = np.ones((4, 4))
    previous = [0.] * 6
    last_turn = None
    for stamp in range(1000, 9000, 100):
        depth = dict(map=frame, frame_id=stamp, source_frame_id=stamp, ground=frame.astype(bool))
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
            assert last_turn is not None and stamp > last_turn
            break
    else:
        pytest.fail("The settled scan did not release a straight segment")


@pytest.mark.parametrize("fault", [
    "none", "depth_gap", "odom_gap", "sweeps_blocked", "corridor_blocked",
    "recovery_left_clear", "recovery_sweeps_blocked", "recovery_corridors_blocked",
])
def test_legacy_shared_avoidance_closed_loop_survives_gaps_and_reports_blocked_sweeps(monkeypatch, fault):
    """Tianyi 2.0, 2026-11-20 06:06:52 attempt, closed loop: a 0.59 m obstacle
    centre-left, the right side open, the person 3.4 m ahead. The chassis is a
    simulation that executes the command and reports it back. The recorded run
    also had a 2.3 s gap in the recorded observations. Inject the gap after
    the simulated straight segment starts."""
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False, vy_max=0.,
                  half_width_m=.275, vx_max=.3, wz_max=1., obstacle_stop_m=.6,
                  stop_distance_m=.6)
    state = _state(motion_source="measured")
    robot = [0., 0., 0.]          # SIM pose: forward, left, yaw (ccw)
    world_target = (3.4, -0.57)   # metres, (forward, left) at the start

    def target():
        c, s = math.cos(-robot[2]), math.sin(-robot[2])
        dx, dy = world_target[0] - robot[0], world_target[1] - robot[1]
        x, y = c * dx - s * dy, s * dx + c * dy
        distance = math.hypot(x, y)
        return SimpleNamespace(bearing_rad=math.atan2(-y, x), range_m=distance, range_known=True,
                               measured_range_m=distance, observed=True, last_obs=None)

    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target())
    free = np.ones(local_path.SHAPE, dtype=bool)
    # Posts 0.30 m ahead at +-0.30 m enter the circular turn sweep.
    posts = np.isclose(local_path.FORWARD, .30) & np.isclose(np.abs(local_path.RIGHT), .30)
    recovery_changes_scene = fault.startswith("recovery_")
    after_gap = False

    def observe(*args):
        blocked = fault == "sweeps_blocked" or (fault == "recovery_sweeps_blocked" and after_gap)
        if blocked:
            return free & ~posts, posts
        if fault == "corridor_blocked" and abs(robot[2]) > .30:
            wall = np.isclose(local_path.FORWARD, .85)
            return free & ~wall, wall
        return free, ~free

    monkeypatch.setattr(active_scan, "observe", observe)

    def clearance(_, _config, lateral_offset_m=0.):
        yaw = robot[2]
        if lateral_offset_m < 0.:
            return .70, .95
        if lateral_offset_m > 0.:
            return 2.30, .95
        if recovery_changes_scene and after_gap:
            return (2.50 if fault == "recovery_left_clear" and yaw > -.10 else .55), .95
        if yaw < -.30:
            if fault == "corridor_blocked":
                return .90, .95
            return 2.30 - robot[0], .95
        if yaw > .30:
            return .70, .95
        return (.50 if fault == "sweeps_blocked" else .59), .95

    monkeypatch.setattr(P, "_clearance", clearance)
    frame = np.ones((4, 4))
    previous = [0.] * 6
    straight_started = None
    gap = None
    resumed = None
    seen = []
    recovery_turns = []
    for stamp in range(1000, 30000, 100):
        depth = dict(map=frame, frame_id=stamp, source_frame_id=stamp, ground=frame.astype(bool))
        feedback = dict(vx=previous[0], vy=0., wz=previous[5], received_ms=stamp)
        if (straight_started is not None and gap is None and robot[0] >= .35
                and (recovery_changes_scene or fault in ("depth_gap", "odom_gap"))):
            gap = (stamp, stamp + 2300)
        after_gap = gap is not None and stamp >= gap[1]
        if gap and gap[0] <= stamp < gap[1]:
            if fault != "odom_gap":
                depth = None
            if fault != "depth_gap":
                feedback = None
        decision = P.step(detections=_detections(), depth=depth, odom=feedback,
                          config=config, state=state, dt=.1)
        seen.append(decision.reason)
        values = decision.values or [0.] * 6
        if gap and gap[0] <= stamp < gap[1]:
            assert values == [0.] * 6
        assert values[1] == 0.
        if state.active_scan.active:
            assert not (values[0] and values[5])
        assert "主动观察中断" not in decision.reason, decision.reason
        assert decision.status != P.FAILED, decision.reason
        robot[2] += values[5] * .1
        robot[0] += values[0] * .1 * math.cos(robot[2])
        robot[1] += values[0] * .1 * math.sin(robot[2])
        previous = values
        if recovery_changes_scene and after_gap:
            if values[5]:
                assert "主动观察：已停稳，重新规划" in seen
                recovery_turns.append(values[5])
            if fault == "recovery_sweeps_blocked":
                assert values[:2] == [0., 0.] and values[5] == 0.
                if "转身扫掠区有障碍" in decision.reason:
                    assert "主动观察：尝试另一侧的转身空间" in seen
                    assert state.active_scan.attempted and not state.active_scan.active
                    return
            elif fault == "recovery_corridors_blocked":
                assert values[:2] == [0., 0.]
                if "停车距离不足" in decision.reason:
                    assert min(recovery_turns) < 0. < max(recovery_turns)
                    assert state.active_scan.attempted and not state.active_scan.active
                    return
            elif values[0]:
                assert min(recovery_turns) < 0. < max(recovery_turns)
                assert decision.reason == "主动观察：转稳后直行"
                return
        if fault == "sweeps_blocked":
            assert values[5] == 0. and values[0] == 0.
            if "转身扫掠区有障碍" in decision.reason:
                assert not state.active_scan.active
                return
            continue
        if fault == "corridor_blocked":
            if "转向后的直行走廊仍然被挡住" in decision.reason:
                assert values[0] == 0. and not state.active_scan.active
                assert any("尝试另一侧" in reason for reason in seen)
                return
            continue
        if values[0] and straight_started is None:
            assert robot[2] < -.30, "drove before the right turn settled"
            straight_started = stamp
        if after_gap and not recovery_changes_scene and values[0] and not state.active_scan.active:
            resumed = stamp
            break
        if fault == "none" and straight_started is not None and robot[0] > .6:
            resumed = stamp
            break
    assert resumed is not None, seen[-12:]
    if gap:
        assert any("停车后重规划" in reason for reason in seen)
        assert any("重新规划" in reason and "已停稳" in reason for reason in seen)


@pytest.mark.parametrize("blocked_sweep", [False, True])
def test_legacy_small_heading_can_align_a_narrow_opening_before_the_shared_straight_segment(
        monkeypatch, blocked_sweep):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False,
                  half_width_m=.275)
    state = _state(motion_source="measured")
    target = SimpleNamespace(bearing_rad=0., range_m=3.5, range_known=True,
                             observed=True, measured_range_m=3.5)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    free = np.ones(local_path.SHAPE, dtype=bool)
    obstacle = np.isclose(local_path.FORWARD, 1.10) & np.isclose(local_path.RIGHT, .30)
    if blocked_sweep:
        obstacle[local_path.START[0] + 5, local_path.START[1] + 7] = True

    def clearance(_, _config, lateral_offset_m=0.):
        if lateral_offset_m:
            return (2.15 if -.15 < lateral_offset_m < -.08 else 1.10), .96
        heading = -math.degrees(state.active_scan.pose[2])
        return (2.15 if 4. <= heading <= 8. else 1.10), .96

    monkeypatch.setattr(P, "_clearance", clearance)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (
        (free, ~free) if clearance(None, None)[0] == 2.15 else (~free, obstacle)))
    frame = np.ones((4, 4))
    previous = [0.] * 6
    last_turn = None
    for stamp in range(1000, 4000, 100):
        decision = P.step(detections=_detections(),
                          depth=dict(map=frame, frame_id=stamp, source_frame_id=stamp, ground=frame.astype(bool)),
                          odom=dict(vx=previous[0], vy=0., wz=previous[5], received_ms=stamp),
                          config=config, state=state, dt=.1)
        previous = decision.values
        assert previous is not None and previous[1] == 0.
        if blocked_sweep:
            assert previous == [0.] * 6
            if "转身扫掠区有障碍" in decision.reason:
                return
        if previous[5]:
            assert previous[0] == 0.
            last_turn = stamp
        if previous[0]:
            assert decision.reason == "主动观察：转稳后直行"
            assert last_turn is not None and stamp > last_turn
            assert 4. <= -math.degrees(state.active_scan.pose[2]) <= 8.
            return
    pytest.fail("No straight segment was released through the measured opening")


@pytest.mark.parametrize("blocked_in_new_view", [False, True])
@pytest.mark.parametrize("camera_fallback", [False, True])
def test_legacy_obstacle_route_supplies_a_turn_and_straight_waypoint_without_lateral_motion(
        monkeypatch, blocked_in_new_view, camera_fallback):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False,
                  half_width_m=.275, vy_max=0.)
    state = _state(motion_source="measured")
    target = SimpleNamespace(bearing_rad=0., range_m=4., range_known=True,
                             observed=True, measured_range_m=4.)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    occupied = np.isclose(local_path.FORWARD, .90) & (np.abs(local_path.RIGHT) <= .12)
    new_view = np.zeros_like(occupied)
    new_view[local_path.START[0] + 6, local_path.START[1]] = blocked_in_new_view

    reached_view = False

    def observe(*args):
        nonlocal reached_view
        reached_view |= state.active_scan.ready
        grid = new_view if reached_view else occupied
        free = ~grid
        if camera_fallback and not reached_view:
            free &= local_path.FORWARD >= .45
        return free, grid

    # Exercise the turn-and-straight fallback after the prefix candidate fails.
    monkeypatch.setattr(P, "_checked_forward_prefix", lambda *args, **kwargs: None)
    monkeypatch.setattr(active_scan, "observe", observe)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (
        .30 if blocked_in_new_view and (reached_view or state.active_scan.ready) else .90, .98))
    frame = np.ones((4, 4))

    def tick(stamp, last, frame_id=None):
        odom = dict(vx=last[0], vy=0., wz=last[5], received_ms=stamp)
        if camera_fallback:
            odom.update(source_sample_id=stamp, velocity_history=((stamp, stamp, (last[0], 0., last[5])),))
        return P.step(detections=_detections(),
                      depth=dict(map=frame, frame_id=frame_id or stamp, source_frame_id=frame_id or stamp,
                                 ground=frame.astype(bool)),
                      odom=odom,
                      config=config, state=state, dt=.1)

    decision = tick(1000, [0.] * 6)
    assert decision.values == [0.] * 6
    assert state.view_detour is not None
    target.observed = False
    target.measured_range_m = None
    scan = state.active_scan
    forward, right = state.view_detour.points[1]
    assert scan.directions[0] * scan.turn_angle == pytest.approx(math.atan2(right, forward))
    assert scan.advance_m == pytest.approx(math.hypot(forward, right))
    previous = decision.values
    last_turn = None
    straight = False
    for stamp in range(1100, 11500, 100):
        decision = tick(stamp, previous)
        previous = decision.values
        assert previous is not None and previous[1] == 0.
        if previous[5]:
            assert previous[0] == 0.
            last_turn = stamp
        if blocked_in_new_view:
            assert previous[0] == 0.
            if (scan.exhausted or "路线首段的直行走廊被挡住" in decision.reason
                    or "原直行走廊被挡" in decision.reason):
                assert last_turn is not None
                return
        elif previous[0]:
            assert decision.reason == "主动观察：转稳后直行"
            assert last_turn is not None and stamp > last_turn
            straight = True
        elif "直行段结束" in decision.reason:
            assert straight
            assert "等待直行结束后的新深度" not in tick(stamp + 100, previous, frame_id=stamp).reason
            return
    pytest.fail("The segment neither completed nor held for the newly observed obstruction")


def test_legacy_local_obstacle_beyond_slowdown_starts_a_detour_with_width_based_margin(monkeypatch):
    from plugins.navi import local_path

    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 336), (480, 420, 243), .20)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.20)
    assert P._keepout_m(config) == pytest.approx(.22)
    state = _state(motion_source="measured")
    view = dict(frame_id=1000, map=np.ones((4, 4)), ground=np.ones((4, 4), dtype=bool))
    occupied = (abs(local_path.RIGHT) <= .10) & np.isclose(local_path.FORWARD, 2.)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2., 1.))

    result = P._avoid(.2, 0., 0., view, config, P.APPROACHING, "", state,
                      target_distance=3., target_bearing=0., may_scan=True)
    assert result[:3] == (0., 0., 0.)
    assert scan.route_segment and scan.phase == "brake"


@pytest.mark.parametrize("reported_hit,projected_hit,distance,detour", [
    (1.10, 1.10, 1.31, False), (.72, .72, 1.31, True), (1.10, 2.80, 2.60, False),
])
def test_legacy_local_avoidance_checks_the_path_to_the_target_stop_point(
        monkeypatch, reported_hit, projected_hit, distance, detour):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 336),
                                        (480, 420, 243), .20, half_length=.23)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.20,
                  stop_distance_m=.80, use_lateral=False, watchdog_ms=600)
    state = _state(motion_source="measured")
    depth = dict(frame_id=1000, map=np.ones((8, 8)), ground=np.ones((8, 8), dtype=bool))
    odom = dict(vx=.14, vy=0., wz=0.)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    occupied = (abs(local_path.RIGHT) <= .15) & np.isclose(
        local_path.FORWARD, round(projected_hit / .05) * .05)
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, np.zeros_like(occupied), occupied))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (reported_hit, 1.))

    move = P._avoid(.18, 0., .04, depth, config, P.APPROACHING, "target", state,
                    odom=odom, target_distance=distance, target_bearing=-.062, may_scan=True)

    assert (move[0] == 0.) is detour
    assert scan.active is detour


def test_legacy_goal_corridor_also_checks_the_current_turning_motion(monkeypatch):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 336),
                                        (480, 420, 243), .20, half_length=.23)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.20,
                  stop_distance_m=.80, use_lateral=False, watchdog_ms=600, max_depth_age_ms=800)
    state = _state(motion_source="measured", commanded=(.306, 0., -.15))
    depth = dict(frame_id=1000, map=np.ones((8, 8)), ground=np.ones((8, 8), dtype=bool))
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    occupied = np.isclose(local_path.FORWARD, .70) & np.isclose(local_path.RIGHT, -.15)
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, np.zeros_like(occupied), occupied))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.10, 1.))

    move = P._avoid(.306, 0., -.15, depth, config, P.APPROACHING, "target", state,
                    odom=dict(vx=.306, vy=0., wz=-.15), target_distance=1.31,
                    target_bearing=.20, may_scan=True)

    assert move[:3] == (0., 0., 0.)


@pytest.mark.parametrize("band", ["free", "unknown", "occupied"])
def test_post_turn_drive_checks_obstacles_without_requiring_coverage(monkeypatch, band):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .20, .20, None)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    view = dict(frame_id=1200, source_frame_id=1200, map=np.ones((4, 4)),
                ground=np.ones((4, 4), dtype=bool))
    scan.receipt_only, scan.current_frame, scan.side_frame = True, 1200, 1200
    scan.source_frame = 1200
    scan.phase, scan.route_segment, scan.advance_m = "evaluate", True, .60
    free = np.ones(local_path.SHAPE, dtype=bool)
    strip = np.isclose(local_path.FORWARD, .35)
    if band != "free":
        free &= ~strip
    scan.views.append((1200, scan.pose, free, strip if band == "occupied" else np.zeros_like(free)))
    feedback = dict(vx=0., vy=0., wz=0.)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (None, .1 if band == "unknown" else 1.))
    result = P._avoid(.2, 0., 0., view, config, P.APPROACHING, "", state,
                      target_distance=3., odom=feedback, dt=.1)
    assert (result[0] > 0.) is (band != "occupied")


@pytest.mark.parametrize("side", [-1., 1.])
def test_detour_first_leg_passes_the_obstacle_at_the_permitted_heading_error(side):
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.375 / 1.10)
    state = _state()
    scan = state.active_scan
    scan.receipt_only = True
    occupied = np.isclose(local_path.FORWARD, 1.7) & (side * local_path.RIGHT >= -.25) & (side * local_path.RIGHT <= -.05)
    scan.current_frame = 1000
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    observation = dict(frame_id=1000, map=np.ones((4, 4)), ground=np.ones((4, 4), dtype=bool))
    assert P._start_detour_segment(observation, config, state, geometry, 1.7, (side,), 5., .16 * side)
    forward, right = state.view_detour.points[1]
    assert forward >= 1.7 and side * right > 0
    heading = math.atan2(right, forward)
    # The hardware may settle two degrees short. A newly projected occupied cell
    # can also move laterally by one grid cell when the grid axes rotate.
    assert local_path.segment_clear(occupied, geometry, .375 + local_path.RESOLUTION,
                                    heading - side * active_scan.ANGLE_TOLERANCE, scan.advance_m)


def test_legacy_detour_first_turn_can_execute_with_the_braking_reserve():
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 336), (480, 420, 243),
                                        .20, half_length=.23)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.20,
                  max_depth_age_ms=800, watchdog_ms=600)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only = True
    scan.current_frame = 1000
    scan.poses.append((1000, scan.pose, 0.))
    # A centre obstacle and a close left return outside the stationary circle.
    occupied = ((np.isclose(local_path.FORWARD, .75) & (np.abs(local_path.RIGHT) < .11)) |
                (np.isclose(local_path.FORWARD, .35) & np.isclose(local_path.RIGHT, -.30)))
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    frame = np.ones((8, 8))
    observation = dict(frame_id=1000, map=frame, ground=frame.astype(bool))

    assert P._start_detour_segment(observation, config, state, geometry, .75, (1., -1.), 3., 0.)
    assert scan.command(geometry, config.half_width_m, 1., 600, 800).yaw == 0.
    scan.poses.append((1100, scan.pose, 0.))
    assert scan.command(geometry, config.half_width_m, 1., 600, 800).yaw < 0.


def test_legacy_initial_route_points_at_measured_opening_between_offset_jambs():
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(320, 247, 159.5), (240, 247, 115), .275)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.275)
    state = _state(motion_source="measured")
    scan = state.active_scan
    occupied = (np.isclose(local_path.FORWARD, 1.) &
                ((local_path.RIGHT <= -.349) | (local_path.RIGHT >= .549)))
    scan.current_frame = 1000
    scan.receipt_only = True
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    rays = (np.arange(320) - 159.5) / 247
    frame = np.broadcast_to(np.where((rays < -.35) | (rays > .55), 1., 4.), (240, 320))
    observation = dict(frame_id=1000, map=frame, ground=np.ones_like(frame, dtype=bool))

    assert P._start_detour_segment(observation, config, state, geometry, 1., (1., -1.), 4., 0.)
    assert scan.directions[0] * scan.turn_angle == pytest.approx(math.atan2(.1, 1.))
    assert scan.advance_m > .3


@pytest.mark.parametrize("scene", ["too_narrow", "one_jamb", "missing_rays", "near_returns", "target_outside"])
def test_centering_requires_two_observed_jambs_and_returns_beyond_the_opening(scene):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(320, 247, 159.5), (240, 247, 115), .275)
    left = local_path.RIGHT <= -.349
    right = local_path.RIGHT >= (.299 if scene == "too_narrow" else .549)
    occupied = np.isclose(local_path.FORWARD, 1.) & (left if scene == "one_jamb" else left | right)
    frame = np.full((240, 320), 4.)
    if scene in ("missing_rays", "near_returns"):
        frame[:, 154:168] = np.nan if scene == "missing_rays" else .9
    target = (4., -3.) if scene == "target_outside" else (4., 0.)
    assert local_path.opening_route(frame, occupied, geometry, .375, 1., target) is None


@pytest.mark.parametrize("source_at_stop", [1000, 1100, None])
def test_legacy_shortened_obstacle_braking_room_replans_from_the_current_depth(source_at_stop):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 493, 318), (480, 493, 229), .275)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="measured", commanded=(.15, 0., 0.))
    scan = state.active_scan
    scan.phase, scan.receipt_only, scan.route_segment = "evaluate", True, True
    scan.attempted = scan.issued_yaw = True
    scan.advance_origin, scan.advance_m, scan.pose = (0., 0., 0.), 1.55, (.96, 0., 0.)
    frame = np.ones((4, 4))

    def update(stamp, source, vx):
        obs = dict(map=frame, ground=frame.astype(bool), frame_id=stamp, source_frame_id=source)
        odom = dict(vx=vx, vy=0., wz=0., received_ms=stamp)
        scan.update(obs, odom, geometry, receipt_only=True, max_depth_age_ms=800)
        return obs, odom

    view, motion = update(1000, 1000, .15)
    result = P._advance_after_scan(view, config, state, motion, .79, .72, 3.5, .1,
                                   target_bearing=.1)
    assert result[:3] == (0., 0., 0.)
    assert scan.phase == "brake_for_replan"
    update(1100, source_at_stop, 0.)
    assert "重新规划" in scan.command(geometry, .375, 1., 500).reason
    assert not scan.active and not scan.attempted


@pytest.mark.parametrize("post_side", [-1., 1.])
@pytest.mark.parametrize("range_offset,expected", [(.004, -1.), (.35, 1.)])
def test_legacy_similar_side_ranges_prefer_the_side_away_from_the_measured_post(monkeypatch, post_side, range_offset, expected):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.275)
    state = _state()
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    columns = post_side * local_path.RIGHT
    occupied = np.isclose(local_path.FORWARD, 1.4) & (columns >= .2) & (columns <= .4)
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    monkeypatch.setattr(P, "_clearance", lambda obs, cfg, lateral_offset_m=0.:
                        ((4.9 + math.copysign(range_offset, lateral_offset_m * post_side))
                         if lateral_offset_m else 1.4, .52))
    obs = dict(frame_id=1000, map=np.ones((4, 4)), ground=np.ones((4, 4), dtype=bool))
    P._avoid(.25, 0., 0., obs, config, P.APPROACHING, "", state, target_distance=4.9,
             target_bearing=.16 * post_side, may_scan=True)
    assert scan.route_segment and scan.directions == (expected * post_side,)


def test_synchronized_reroute_requires_the_evaluated_capture():
    from plugins.navi.active_scan import ActiveScan

    scan = ActiveScan(side_frame=1200, current_frame=1300)
    assert scan.matches_route_view({"capture_ms": 1300})
    assert not scan.matches_route_view({"capture_ms": 1200})
    assert not scan.matches_route_view({"frame_id": 1300})


@pytest.mark.parametrize(("identity", "second_corridor_clear"), [
    ("fresh", False), ("fresh", True), ("missing", False), ("repeated", False),
    ("missing_midway", False), ("missing_initial", False), ("mismatched", False), ("gap_after_reroute", False),
])
def test_legacy_blocked_route_replans_once_from_current_heading(monkeypatch, identity, second_corridor_clear):
    from plugins.navi import active_scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 493, 318), (480, 493, 229), .275)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.275,
                  use_lateral=False)
    state = _state(motion_source="measured")
    scan = state.active_scan
    occupied = np.isclose(local_path.FORWARD, 1.) & (local_path.RIGHT <= .10)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~occupied, occupied))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1., .99))
    frame = np.ones((4, 4))

    def update(stamp, yaw):
        obs = dict(map=frame, ground=frame.astype(bool), frame_id=stamp, control_ms=stamp)
        if (identity != "missing" and not (identity == "missing_midway" and stamp == 1100)
                and not (identity == "missing_initial" and stamp == 1000)):
            obs["source_frame_id"] = 900 if identity == "repeated" and stamp <= 1200 else stamp - 100
        odom = dict(vx=0., vy=0., wz=yaw, received_ms=stamp)
        scan.update(obs, odom, geometry, receipt_only=True)
        return obs, odom

    update(1000, 0.)
    scan.phase = "turn"
    scan.attempted = scan.issued_yaw = scan.route_segment = scan.origin_set = True
    scan.advance_m, scan.directions, scan.turn_angle = 1.90, (-1.,), .4
    scan.pose = (0., 0., -.4)  # Current orientation after the initial left turn.
    for stamp in (1000, 1100, 1200):
        obs, odom = update(stamp, 0.)
        assert scan.command(geometry, .375, 1., 500, 800, now_ms=stamp).yaw == 0.
    assert scan.ready

    def avoid(obs, odom):
        return P._avoid(.15, 0., 0., obs, config, P.APPROACHING, "", state,
                        target_distance=4.4, target_bearing=.5, odom=odom, may_scan=True, dt=.1)

    assert avoid(obs, None)[:3] == (0., 0., 0.)
    assert scan.ready
    if identity == "mismatched":
        obs = dict(obs, source_frame_id=99999)
    assert avoid(obs, odom)[:3] == (0., 0., 0.)
    if identity == "mismatched":
        assert scan.phase == "brake_for_replan"
        return
    assert scan.phase == "brake" and scan.directions == (1.,)
    assert scan.advance_origin is None
    if identity == "gap_after_reroute":
        for stamp in range(1300, 1800, 100):
            update(stamp, 0.)
            scan.command(geometry, .375, 1., 500, 800, now_ms=stamp)
        obs, odom = update(1800, 0.)
        assert scan.update(None, odom, geometry, receipt_only=True).yaw == 0.
        assert scan.phase == "brake_for_replan"
        update(1900, 0.)
        assert scan.command(geometry, .375, 1., 500, 800, now_ms=1900).replanned_turn
        assert not scan.active and not scan.attempted and not scan.rerouted
        return
    yaw = 0.
    last_turn = None
    for stamp in range(1300, 11000, 100):
        obs, odom = update(stamp, yaw)
        if scan.ready:
            break
        cmd = scan.command(geometry, .375, 1., 500, 800, now_ms=stamp)
        yaw = cmd.yaw
        if yaw:
            assert yaw < 0.
            last_turn = stamp
    assert scan.ready and scan.origin[2] == pytest.approx(-.4)
    assert last_turn is not None and scan.side_frame > last_turn
    if second_corridor_clear:
        occupied[:] = False
    obs, odom = update(stamp + 100, 0.)
    decision = avoid(obs, odom)
    if second_corridor_clear:
        assert decision[0] > 0. and decision[1:3] == (0., 0.)
    else:
        assert decision[:3] == (0., 0., 0.)
        assert not scan.active and scan.exhausted


def test_close_obstacle_in_the_new_view_requires_another_turn_before_advancing(monkeypatch):
    from plugins.navi.local_path import Geometry

    geometry = Geometry(None, None, np.eye(4), .275, .275, None)
    config = _cfg(local_planning=True, scan_geometry=geometry, obstacle_stop_m=.60)
    state = _state()
    scan = state.active_scan
    scan.phase, scan.current_frame = "evaluate", 1000
    scan.advance_m, scan.directions = 1., (-1., 1.)
    scan.poses.append((1000, scan.pose, 0.))
    depth = {"ground": np.ones((4, 4), dtype=bool), "capture_ms": 1000}
    odom = {"vx": 0., "vy": 0., "wz": 0.}

    first = P._advance_after_scan(depth, config, state, odom, .59, .60, 3., .1)
    assert first[:3] == (0., 0., 0.)
    assert scan.phase == "brake" and scan.directions == (1.,)
    assert scan.advance_origin is None

    scan.phase = "evaluate"
    second = P._advance_after_scan(depth, config, state, odom, .59, .60, 3., .1)
    assert second[:3] == (0., 0., 0.)
    assert not scan.active


@pytest.mark.parametrize("measured", [False, True])
def test_local_blocked_approach_requires_measured_feedback_and_checks_the_forward_sweep(monkeypatch, measured):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    cfg = _cfg(local_planning=True, local_geometry=geometry, stop_distance_m=.6)
    state = _state(motion_source="measured" if measured else "command")
    scan = state.active_scan
    scan.phase, scan.directions = "evaluate", (1., -1.)
    scan.receipt_only = scan.attempted = scan.issued_yaw = True
    scan.command_only = not measured
    scan.side_frame = 1000
    if measured:
        scan.poses.append((1000, scan.pose, 0.))
    odom = {"vx": 0., "vy": 0., "wz": 0.} if measured else None
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(P, "_local_translation", lambda *args: None)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.28, 1.))
    monkeypatch.setattr(local_path, "observe", lambda *args: (~occupied, occupied))

    def tick(frame):
        depth = dict(_scene(2.), ground=occupied, frame_id=frame)
        return P._avoid(.60, 0., 0., depth, cfg, P.APPROACHING, "", state,
                        target_distance=2., may_scan=True, odom=odom)

    assert tick(1000)[:3] == (0., 0., 0.)
    assert scan.active and scan.directions == (-1.,)
    scan.phase, scan.side_frame = "evaluate", 1100
    decision = tick(1100)
    if measured:
        assert decision[0] > 0.
    else:
        assert decision[:3] == (0., 0., 0.)
        assert decision[-1] == P.NO_LOCAL_ROUTE
    assert not scan.active
    occupied[local_path.START[0] + 5, local_path.START[1]] = True
    assert tick(1200)[:3] == (0., 0., 0.)


def test_near_turn_stops_forward_until_the_current_corridor_is_clear():
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
    assert tick(clear, 300).values[0] > 0.


def test_a_clear_observation_after_a_stop_allows_searching_for_a_lost_target():
    cfg = _cfg(stop_distance_m=0.6, obstacle_stop_m=.60, lost_frames=1)
    blocked = _scene(3.0, [(0.0, 0.85, 0.65)])
    state = seed(_state(), _detections(_obj()), blocked, cfg)
    stopped = _step(detections=_detections(_obj()), depth=blocked, state=state, config=cfg)
    assert stopped.values == [0.0] * 6
    state.tracker = P.Tracker()
    decision = _step(detections=_detections(), state=state, config=cfg,
                     depth=dict(_scene(3.0), frame_id=100))
    assert decision.values[:2] == [0.0, 0.0]
    assert decision.status == P.SEARCHING and decision.values[5] != 0


def test_lost_target_does_not_start_a_turn_in_a_close_obstacle():
    cfg = _cfg(lost_frames=1)
    decision = _step(detections=_detections(), depth=_scene(0.4),
                     config=cfg, state=_state())
    assert decision.values == [0.0] * 6
    assert decision.status == P.AVOIDING


@pytest.mark.parametrize(("ahead", "side_obstacles", "expected_yaw"), [
    (.80, (), -.60),
    (.60, (), -.60),
    (.60, (.40,), -.60),
    (.60, (.30,), 0.),
    (2.0, (-.40,), -.60),
    (2.0, (-.30,), 0.),
])
def test_legacy_lost_target_checks_the_padded_turn_sweep(monkeypatch, ahead, side_obstacles, expected_yaw):
    from plugins.navi import active_scan as scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 600, 320), (480, 600, 240), .21)
    cfg = _legacy_cfg(lost_frames=1, local_planning=True, local_geometry=geometry,
               half_width_m=.21, obstacle_stop_m=.60, search_rate=.60)
    depth = dict(_scene(3.0, [(0., ahead, .5)]), frame_id=1000)
    depth["ground"] = np.zeros_like(depth["map"], dtype=bool)
    depth["ground"][0, 0] = True
    occupied = np.isclose(scan.FORWARD, ahead) & np.isclose(scan.RIGHT, 0.)
    for right in side_obstacles:
        # Hit either inside or outside the same circle used for both turns.
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


def test_legacy_a_near_side_escape_has_a_blocked_deadline():
    cfg = _legacy_cfg(stop_distance_m=0.6, blocked_timeout_s=0.4)
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
    assert cfg.slow_distance_m == 1.0


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


def test_legacy_local_detour_starts_at_the_slowdown_boundary(monkeypatch):
    cfg, state = _legacy_cfg(local_planning=True, obstacle_stop_m=.60, slow_distance_m=1.50), _state()
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


@pytest.mark.parametrize("use_lateral", [False, True])
def test_local_planner_admits_a_checked_sideways_route_near_an_obstacle(monkeypatch, use_lateral):
    cfg = _cfg(local_planning=True, use_lateral=use_lateral)
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
    assert command[:3] == (0., .06 if use_lateral else 0., 0.)
    assert state.escape_side == (-1. if use_lateral else 0.)


@pytest.mark.parametrize("remember", [False, True])
def test_legacy_remembered_wall_allows_the_same_checked_approach(monkeypatch, remember):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.odom import PlanarPose

    geometry = local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .2)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, scan_geometry=geometry,
                         use_lateral=False, wz_max=0., half_width_m=.2, use_obstacle_history=True)
    state = _state(motion_source="measured")
    wall = np.isclose(local_path.FORWARD, 1.25)
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (~wall, wall))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.25, 1.))
    frame = np.full((8, 8), 3.)
    depth = dict(map=frame, ground=frame.astype(bool), frame_id=1000, capture_ms=1000)
    odom = dict(vx=0., vy=0., wz=0.)
    pose = PlanarPose(1000, ("odom", "body"), 0., 0., 0.)
    if remember:
        depth["obstacle_capture_pose"] = pose
        odom["obstacle_pose"] = pose
    scan = state.active_scan
    scan.current_frame = 1000
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, ~wall, wall))
    for stamp in (1000, 1010, 1020):
        scan.current_frame = stamp
        pose = PlanarPose(stamp, pose.frame, 0., 0., 0.)
        depth.update(frame_id=stamp, capture_ms=stamp)
        if remember:
            depth["obstacle_capture_pose"] = odom["obstacle_pose"] = pose
        command = P._avoid(.6, 0., 0., depth, config, P.APPROACHING, "", state,
                           target_distance=3., target_bearing=0., odom=odom, may_scan=True)
    assert bool(scan.obstacles.cells) is remember
    assert command[:3] == (.15, 0., 0.)


def test_legacy_exhausted_observation_can_hand_off_to_approach(monkeypatch):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .2)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False, wz_max=0., half_width_m=.2)
    state = _state(motion_source="measured")
    wall = np.isclose(local_path.FORWARD, 1.25)
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (~wall, wall))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.25, 1.))
    frame = np.ones((8, 8))
    depth = dict(map=frame, ground=frame.astype(bool), frame_id=1000)
    odom = dict(vx=0., vy=0., wz=0.)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.views.append((1000, scan.pose, ~wall, wall))
    scan.phase, scan.directions, scan.advance_m = "evaluate", (-1.,), 1.5
    scan.attempted = scan.issued_yaw = True
    assert P._advance_after_scan(depth, config, state, odom, 1.25, .60, 3., .1)[:3] == (0., 0., 0.)
    assert not scan.active and scan.attempted
    command = P._avoid(.6, 0., 0., depth, config, P.APPROACHING, "", state,
                       target_distance=3., target_bearing=0., odom=odom, may_scan=True)
    assert command[:3] == (.15, 0., 0.)
    assert not scan.active


def test_memory_can_use_the_command_only_scan_without_a_motion_pose_history(monkeypatch):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.odom import PlanarPose

    geometry = local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .2)
    config = _cfg(local_planning=True, local_geometry=geometry, scan_geometry=geometry,
                  use_lateral=False, wz_max=0., half_width_m=.2, use_obstacle_history=True)
    state = _state(motion_source="command")
    wall = np.isclose(local_path.FORWARD, 1.25)
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (~wall, wall))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.25, 1.))
    frame = np.ones((8, 8))
    pose = PlanarPose(1000, ("map", "body"), 0., 0., 0.)
    depth = dict(map=frame, ground=frame.astype(bool), frame_id=1000, capture_ms=1000,
                 obstacle_capture_pose=pose)
    for stamp in (1000, 1010, 1020):
        pose = PlanarPose(stamp, pose.frame, 0., 0., 0.)
        depth.update(frame_id=stamp, capture_ms=stamp, obstacle_capture_pose=pose)
        state.active_scan.observe_single(depth, geometry, 500, stamp)
        decision = P._avoid(.3, 0., 0., depth, config, P.APPROACHING, "", state,
                            target_distance=3., target_bearing=0., may_scan=True,
                            odom={"obstacle_pose": pose})
    assert not state.active_scan.poses and state.active_scan.obstacles.cells
    assert decision[:3] == (0., 0., 0.)
    assert decision[-1] == P.NO_LOCAL_ROUTE


def test_checked_route_speed_fits_its_remaining_waypoint_horizon():
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    config = _cfg(local_planning=True, local_geometry=geometry, floor_vx_moving=.02)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.phase, scan.advance_m, scan.current_frame = True, "evaluate", .6, 1000
    scan.route_segment, scan.advance_origin, scan.pose = True, (0., 0., 0.), (.54, 0., 0.)
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    depth = {"ground": np.ones((8, 8), dtype=bool)}
    feedback = dict(vx=0., vy=0., wz=0.)
    command = P._advance_after_scan(depth, config, state, feedback, None, .60, 3., .1)
    horizon = (config.depth_age_limit_ms + config.watchdog_ms + 50) / 1000
    assert 0. < command[0] * horizon <= .06 + 1e-9


@pytest.mark.parametrize("motion_direction", [-1., 1.])
@pytest.mark.parametrize("observation_exhausted", [False, True])
def test_legacy_partial_outcome_requires_measured_progress_toward_the_original_target(
        monkeypatch, motion_direction, observation_exhausted):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .2)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False, wz_max=0.,
                  half_width_m=.2, stop_distance_m=.6, obstacle_stop_m=.60)
    state = _state(motion_source="measured")
    target = SimpleNamespace(observed=True, range_known=True, range_m=2.1,
                             measured_range_m=2.1, bearing_rad=0.)
    state.goal_anchor = P.DetourGoal(2.1, 0., track=target)
    state.active_scan.attempted = state.active_scan.exhausted = observation_exhausted
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: target)
    wall = np.isclose(local_path.FORWARD, 1.25)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~wall, wall))
    ahead = 1.25
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (ahead, 1.))
    frame = np.ones((8, 8))

    def tick(stamp, vx):
        return P.step(detections=_detections(),
                      depth=dict(map=frame, ground=frame.astype(bool), frame_id=stamp),
                      odom=dict(vx=vx, vy=0., wz=0., received_ms=stamp),
                      config=config, state=state, dt=.1)

    vx = 0.
    for stamp in range(1000, 1800, 100):
        target.range_m -= motion_direction * vx * .1
        target.measured_range_m = target.range_m
        vx = tick(stamp, motion_direction * vx).values[0]
        assert vx > 0.
    ahead = .60
    assert tick(1800, motion_direction * vx).values[0] == 0.
    stopped = tick(1900, 0.)
    assert state.goal_anchor.track is target
    if motion_direction < 0.:
        assert stopped.status == P.AVOIDING and state.partial_result is None
        return
    assert stopped.status == P.FAILED
    assert state.partial_result["outcome"] == "partial"
    assert state.partial_result["best_verified_distance_m"] == pytest.approx(target.measured_range_m)


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


def test_local_planner_checks_obstacles_before_the_slowdown_band(monkeypatch):
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
    state = _state(escape_side=1., recovery=True)
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0: (
        2.1 if lateral_offset_m > 0 else 2.0 if lateral_offset_m < 0 else .9, 1.0))
    monkeypatch.setattr(P, "_local_translation", lambda *args: (.063, 0.))

    def step(frame):
        return P._avoid(.1, 0, 0, dict(_scene(2.8), frame_id=frame), cfg,
                        P.AVOIDING, "", state, target_distance=2.8)[:3]

    assert step(200) == (.063, 0., 0.)
    assert not state.recovery


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


def test_right_only_lateral_range_is_shared_by_approach_and_planner():
    cfg = _cfg(vy_max=.4)
    P.adopt_limits(cfg, {"limits": {"lower": [-.3, -.12, 0., 0., 0., -.5],
                                   "upper": [.3, 0., 0., 0., 0., .5],
                                   "min_magnitude_moving": [0., .09, 0., 0., 0., 0.]}})
    assert P._local_velocity_limits(cfg, 1.)[1] == (.09, .12)
    assert P._local_velocity_limits(cfg, -1.)[1] == (.09, 0.)
    assert P._approach(.4, .35, _depth(), cfg)[1] == -.12
    assert P._approach(.4, -.35, _depth(), cfg)[1] == 0.


def test_yaw_keeps_each_declared_direction_limit():
    cfg = _cfg(wz_max=1.)
    P.adopt_limits(cfg, {"limits": {"lower": [-.3, -.1, 0., 0., 0., -.6],
                                   "upper": [.3, .1, 0., 0., 0., .2]}})
    state = _state()
    assert P._finalise_yaw(-.8, False, .6, cfg, state, .1) == -.6
    assert P._finalise_yaw(.8, False, -.6, cfg, state, .1) == .2


def test_approach_keeps_the_moving_floor_after_checking_its_sweep(monkeypatch):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .23, .23, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20,
               floor_vx=.40, floor_vx_moving=.10, lateral_scale=1.)
    state = _state()
    hit = np.isclose(local_path.FORWARD, .55) & np.isclose(local_path.RIGHT, .30)
    monkeypatch.setattr(state.local_plan, "grid_for", lambda *_: (~hit, hit))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (None, 1.))
    frame = np.ones((8, 8))
    depth = dict(map=frame, ground=frame.astype(bool), frame_id=1000)
    vx, vy = P._approach(.2, .4, depth, cfg, state)
    assert .10 < vx < .40 and vy < 0
    output = P._avoid(vx, vy, 0., depth, cfg, P.APPROACHING, "", state, target_distance=3.)
    assert output[:2] == pytest.approx((vx, vy))
    horizon = (cfg.depth_age_limit_ms + cfg.watchdog_ms + 50) / 1000
    assert local_path.transition_clear((~hit, hit), geometry, .25, horizon, (output[:2],))


@pytest.mark.parametrize("yaw_feedback", [0., .15])
def test_ordinary_local_strafe_keeps_its_checked_heading(monkeypatch, yaw_feedback):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .23, .23, None)
    cfg = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20, lateral_scale=1., vx_max=.20)
    state = _state(motion_source="measured")
    track = SimpleNamespace(observed=True, range_known=True, range_m=2., measured_range_m=2.,
                            bearing_rad=.4 * cfg.half_fov_rad)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: track)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (None, 1.))
    hit = np.isclose(local_path.FORWARD, .35) & np.isclose(local_path.RIGHT, .35)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~hit, hit))
    frame = np.ones((8, 8))
    decision = P.step(detections=_detections(),
                      depth=dict(map=frame, ground=frame.astype(bool), frame_id=1000),
                      odom=dict(vx=0., vy=0., wz=yaw_feedback, received_ms=1000),
                      config=cfg, state=state, dt=.1)
    if yaw_feedback:
        assert decision.values[1] == 0.
    else:
        assert decision.values[1] < 0. and decision.values[5] == 0.


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
    depth = {"scan_grid": (grid_free, grid_occupied), "body_half_width": .20}
    monkeypatch.setattr(state.local_plan, "translation", lambda *args, **kwargs: (.25, -.10))
    odom = {"vx": .16, "vy": 0., "wz": 0.}
    expected = P.LocalHold.TRANSITION if occupied else (.25, -.10)
    assert P._local_translation(depth, cfg, state, odom, .9, 1.) == expected


def test_blocked_transition_keeps_current_detour_and_recovery(monkeypatch):
    cfg = _cfg(local_planning=True)
    state = _state(escape_side=1., recovery=True)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.9, 1.))
    monkeypatch.setattr(P, "_local_translation", lambda *args: P.LocalHold.TRANSITION)
    decision = P._avoid(.2, 0., 0., {}, cfg, P.APPROACHING, "", state, target_distance=2.)
    assert decision[:3] == (0., 0., 0.)
    assert state.escape_side == 1. and state.recovery
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


def test_legacy_checked_straight_approach_keeps_tracking_a_moving_target(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .30, .30, None)
    cfg = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False, vy_max=0.)
    state = _state()
    track = SimpleNamespace(bearing_rad=0., range_m=3., range_known=True, observed=True,
                            measured_range_m=3.)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: track)
    monkeypatch.setattr(local_path, "observe", lambda *args: (
        np.ones(local_path.SHAPE, dtype=bool), np.zeros(local_path.SHAPE, dtype=bool)))
    odom = dict(vx=0., vy=0., wz=0.)
    frame = _solid_depth(3.)

    def tick(stamp):
        depth = dict(_depth(map_=frame), ground=np.ones_like(frame, dtype=bool), frame_id=stamp)
        return P.step(detections=_detections(), depth=depth, odom=odom, config=cfg, state=state, dt=.1)

    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.31, .99))
    assert tick(1000).values[0] > 0.
    assert state.goal_anchor is None
    track.range_m = track.measured_range_m = 4.
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.2, .99))
    decision = tick(1100)
    assert decision.status == P.APPROACHING and decision.values[0] > 0.
    assert decision.distance_m == pytest.approx(4.)


@pytest.mark.parametrize("same_track, measured", [(True, True), (False, True), (True, False)])
def test_legacy_scan_replan_updates_goal_only_with_a_fresh_range_from_the_original_track(monkeypatch, same_track, measured):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False)
    state = _state(motion_source="measured")
    track = SimpleNamespace(bearing_rad=0., range_m=3.81, range_known=True,
                            observed=True, measured_range_m=3.81)
    observation = track
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: observation)
    clearance = 1.21
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (clearance, .98))

    def tick(stamp):
        frame = np.ones((4, 4))
        return P.step(detections=_detections(),
                      depth=dict(map=frame, frame_id=stamp, ground=frame.astype(bool)),
                      odom=dict(vx=0., vy=0., wz=0., received_ms=stamp),
                      config=config, state=state, dt=.1)

    assert tick(1000).values == [0.] * 6
    assert state.active_scan.active and state.goal_anchor is not None
    state.active_scan.hold_for_replan("input gap")
    assert "已停稳，重新规划" in tick(1100).reason
    clearance = 2.1
    if not same_track:
        observation = SimpleNamespace(**vars(track))
    observation.range_m = 4.6
    observation.measured_range_m = 4.6 if measured else None
    if not measured:
        observation.front_required, observation.front_fix = True, None
    decision = tick(1400)
    if same_track and measured:
        assert decision.values[0] > 0. and decision.distance_m == pytest.approx(4.6)
        assert state.goal_anchor is None
    else:
        assert decision.values[0] > 0. and decision.distance_m == pytest.approx(3.81)
        assert state.goal_anchor is not None


@pytest.mark.parametrize("odom", [None, {"vx": 0., "vy": 0., "wz": 0.}])
@pytest.mark.parametrize("already_occluded", [False, True])
def test_legacy_local_detour_keeps_original_target_through_retarget_and_occlusion(monkeypatch, odom,
                                                                         already_occluded):
    from types import SimpleNamespace
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    cfg = _legacy_cfg(local_planning=True, local_geometry=geometry)
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

    # Once the route opens, use odometry to approach the saved goal.
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2., 1.))
    decision = tick()
    assert decision.values[0] > 0. and not state.arrived
    assert decision.status != P.SEARCHING and state.goal_anchor is not None
    observation[0] = track
    tick()
    assert state.goal_anchor is None


@pytest.mark.parametrize("blocked, predicted", [(False, 1.6), (True, 1.6), (False, 3.1)])
def test_legacy_remembered_goal_motion_checks_sweep_and_prediction_budget(
        monkeypatch, blocked, predicted):
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .275, .275, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="command", goal_anchor=P.DetourGoal(2., 1., predicted_m=predicted))
    free = np.ones(local_path.SHAPE, dtype=bool)
    occupied = ~free
    occupied[local_path.START] = blocked
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free & ~occupied, occupied))
    frame = np.ones((4, 4))
    depth = dict(map=frame, ground=frame.astype(bool), frame_id=1000)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.5, .1))
    decision = P.step(detections=None, depth=depth, odom=None, config=config, state=state, dt=.1)

    if blocked or predicted >= 3.:
        assert not any(decision.values)
    else:
        assert decision.values[0] > 0.
    assert decision.status != P.SEARCHING and not state.arrived and state.goal_anchor is not None


@pytest.mark.parametrize("blocked", [False, True])
def test_legacy_remembered_side_goal_aligns_without_search_and_waits_for_a_live_fix(monkeypatch, blocked):
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .22, .22, None)
    cfg = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False,
               stop_distance_m=.6, stop_tolerance_m=.05, arrive_align_tol=.05)
    state = _state(goal_anchor=P.DetourGoal(0., -.62), motion_source="measured")
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    occupied[local_path.START[0], local_path.START[1] + 3] = blocked
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~occupied, occupied))
    frame = np.ones((8, 8))
    yaw = 0.
    for stamp in range(1000, 7000, 100):
        decision = P.step(detections=None, depth=dict(map=frame, ground=frame.astype(bool), frame_id=stamp),
                          odom=dict(vx=0., vy=0., wz=yaw, received_ms=stamp),
                          config=cfg, state=state, dt=.1)
        assert decision.values[:2] == [0., 0.] and not state.arrived
        assert decision.status != P.SEARCHING and not state.searching_for_s
        next_yaw = decision.values[5]
        if blocked:
            assert next_yaw == 0. and "扫掠区有障碍" in decision.reason
            break
        if not next_yaw:
            assert stamp > 1000
            assert abs(decision.bearing) <= cfg.arrive_align_tol
            assert "等待" in decision.reason
            break
        assert next_yaw < 0.
        yaw = next_yaw
    else:
        pytest.fail("The remembered bearing did not converge")
    assert state.goal_anchor is not None


def test_command_only_scan_keeps_fresh_depth_when_detections_lapse(monkeypatch):
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .275, .275, None)
    config = _cfg(local_planning=True, local_geometry=geometry)
    state = _state(motion_source="command", goal_anchor=P.DetourGoal(2., 0.))
    free = np.ones(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, ~free))
    frame = np.ones((4, 4))

    def depth(stamp):
        return dict(map=frame, ground=frame.astype(bool), frame_id=stamp)

    scan = state.active_scan
    scan.observe_single(depth(1000), geometry, 800, 1000)
    assert scan.begin((1., -1.))
    for stamp in range(1100, 2000, 100):
        decision = P.step(detections=_detections(), depth=depth(stamp), odom=None,
                          config=config, state=state, dt=.1)
        if decision.values[5]:
            break
    assert scan.issued_yaw
    decision = P.step(detections=None, depth=depth(stamp + 100), odom=None,
                      config=config, state=state, dt=.1)
    assert scan.current_frame == stamp + 100
    assert scan.active
    assert decision.values[:5] == [0.] * 5


def test_legacy_command_only_straight_segment_can_exceed_the_slowdown_distance(monkeypatch):
    from plugins.navi import active_scan, local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .2, .2, None)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, slow_distance_m=1.5)
    anchor = P.DetourGoal(4., 0.)
    state = _state(motion_source="command", goal_anchor=anchor)
    free = np.ones(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, ~free))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.5, 1.))
    scan = state.active_scan
    scan.receipt_only = scan.command_only = scan.attempted = True
    scan.phase = "evaluate"
    scan.advance_m = 1.969
    frame = np.ones((4, 4))

    for stamp in range(1000, 18000, 100):
        decision = P.step(detections=None,
                          depth=dict(map=frame, frame_id=stamp, ground=frame.astype(bool)),
                          odom=None, config=config, state=state, dt=.1)
        assert "推算已达上限" not in decision.reason
        assert state.goal_anchor is anchor
        if not scan.active:
            assert scan.advance_commanded_m >= scan.advance_m
            assert config.slow_distance_m < anchor.predicted_m < P.MAX_PREDICTED_DETOUR_TRAVEL_M
            return
    pytest.fail("The straight segment did not complete")


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


def test_losing_feedback_during_a_turn_waits_for_valid_feedback(monkeypatch):
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
    assert state.active_scan.phase == "brake_for_replan" and state.active_scan.attempted


def test_legacy_interrupted_observation_replans_while_the_target_corridor_is_blocked(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import active_scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .20)
    cfg = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False)
    state = _state(motion_source="measured")
    track = SimpleNamespace(bearing_rad=0., range_m=2., range_known=True,
                            observed=True, measured_range_m=2.)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: track)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.75, 1.))
    blocked = np.isclose(local_path.FORWARD, .75) & (abs(local_path.RIGHT) <= .20)
    monkeypatch.setattr(active_scan, "observe", lambda *args, **kwargs: (~blocked, blocked))

    def replan(depth, config, state, geometry, ahead, directions, *args, **kwargs):
        assert ahead == .75
        return state.active_scan.begin((1.,), advance_m=1., route_segment=True)

    monkeypatch.setattr(P, "_start_detour_segment", replan)
    frame = np.ones((8, 8))

    def tick(now, received, wz=0., *, source=None, odom_received=None):
        return P.step(detections=_detections(),
                      depth=dict(map=frame, ground=frame.astype(bool), frame_id=received,
                                 source_frame_id=received if source is None else source, control_ms=now),
                      odom=dict(vx=0., vy=0., wz=wz,
                                received_ms=now if odom_received is None else odom_received),
                      config=cfg, state=state, dt=.1)

    tick(1000, 1000)
    scan = state.active_scan
    scan.phase, scan.attempted, scan.issued_yaw = "turn", True, True
    scan.interrupt("转向无进展，停车")
    assert tick(1100, 1000, odom_received=1000).values == [0.] * 6
    assert scan.phase == "brake_for_replan"
    assert tick(1200, 1200, wz=.20).values == [0.] * 6
    assert scan.phase == "brake_for_replan"
    assert "重新规划" in tick(1300, 1300).reason
    assert not scan.active
    decision = tick(1400, 1300)
    assert decision.values == [0.] * 6 and "对准路线首段" in decision.reason
    assert scan.active and scan.route_segment


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


def test_legacy_visible_foreground_triggers_detour_before_slowdown_then_survives_blind_view(monkeypatch):
    from plugins.navi.odom import PlanarPose
    from plugins.navi import active_scan, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geom = local_path.camera_square(HorizontalIntrinsics(64, 50., 32.), (48, 50., 24.), .275)
    cfg = _legacy_cfg(scan_geometry=geom, local_geometry=geom, local_planning=True,
                      use_lateral=False, slow_distance_m=1.5, use_obstacle_history=True)
    state = _state()
    scan = state.active_scan
    visible = np.zeros(local_path.SHAPE, dtype=bool)
    hits = visible.copy()
    hits[local_path.START[0]+38, local_path.START[1]] = True
    monkeypatch.setattr(active_scan, "observe", lambda *args: (visible.copy(), hits.copy()))
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0.: (depth["ahead"], .95))
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (visible.copy(), hits.copy()))
    image = np.ones((48, 64))

    def tick(stamp, distance, x, frame="room"):
        pose = PlanarPose(stamp, (frame,), x, 0., 0.)
        odom = dict(vx=0., vy=0., wz=0., received_ms=stamp, obstacle_pose=pose)
        obs = dict(map=image, ground=image.astype(bool), frame_id=stamp, source_frame_id=stamp,
                   ahead=distance, obstacle_capture_pose=pose)
        scan.update(obs, odom, geom, receipt_only=True)
        return P._avoid(.3, 0., 0., obs, cfg, P.APPROACHING, "target", state,
                        target_distance=3., target_bearing=0., odom=odom, may_scan=True)

    for stamp in (1000, 1010, 1020):
        command = tick(stamp, 1.90, 0.)
    assert command[0] == .15 and command[1] == 0. and command[2]
    assert not scan.active and scan.historical_hits.any()
    # The saved hit survives when the original obstacle leaves the view.
    hits[:] = False
    assert tick(2000, 2.70, .35)[0] == .15
    assert scan.historical_hits.any()
    assert tick(3000, 2.70, .35, frame="new segment")[0] == 0


def test_cold_approach_can_use_current_depth_without_ground():
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics
    geom = local_path.camera_square(HorizontalIntrinsics(64, 50., 32.), (48, 50., 24.), .275)
    cfg = _cfg(local_planning=True, local_geometry=geom)
    decision = P._avoid(.3, 0., 0., {"map": np.full((48, 64), 3.), "frame_id": 1200,
                                         "body_mount_required": True}, cfg,
                        P.APPROACHING, "", _state(), target_distance=3., may_scan=True)
    assert decision[:3] == (.3, 0., 0.)


def test_dynamic_head_facing_sideways_does_not_authorize_forward_navigation():
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.odom import PlanarPose

    geo = local_path.camera_square(HorizontalIntrinsics(64, 50., 32.), (48, 50., 24.), .275)
    cfg = _cfg(local_planning=True, local_geometry=geo)
    pose = PlanarPose(1200, ("odom",), 0., 0., 0.)
    lateral = geo.base_from_optical.copy()
    lateral[:2] = np.array([[0., -1.], [1., 0.]]) @ lateral[:2]
    depth = dict(map=np.full((48,64), 3.), ground=np.ones((48,64), dtype=bool),
                 frame_id=1200, source_frame_id=1200, body_mount_required=True,
                 body_from_optical=lateral, obstacle_capture_pose=pose)
    feedback = dict(vx=0., vy=0., wz=0., received_ms=1200, obstacle_pose=pose)
    result = P.step(detections=_detections(_obj()), depth=depth, odom=feedback,
                    state=_state(), config=cfg, dt=.1)
    assert result.values == [0.] * 6 and "朝向" in result.reason


@pytest.mark.parametrize("aligned_before", [False, True])
def test_dynamic_mount_without_matching_pose_uses_camera_frame(monkeypatch, aligned_before):
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(64, 50., 32.), (48, 50., 24.), .275)
    cfg = _cfg(local_planning=True, local_geometry=geometry, confirm_hits=1)
    state = _state()
    depth = dict(map=np.full((48, 64), 3.), ground=np.ones((48, 64), dtype=bool),
                 frame_id=1200, capture_ms=1150, body_mount_required=True)
    motion = dict(vx=0., vy=0., wz=0., received_ms=1200)
    seen = []
    monkeypatch.setattr(P, "_avoid", lambda vx, vy, wz, depth, config, status, reason, *args, **kwargs:
                        (seen.append(config.local_geometry) or vx, vy, wz, status, reason))
    if aligned_before:
        state.use_body_mount = True
        state.active_scan.phase = "turn"
        state.commanded = (.2, 0., 0.)
        stop = P.step(detections=_detections(_obj()), depth=depth, odom=motion,
                      state=state, config=cfg, dt=.1)
        assert stop.values == [0.] * 6 and not state.active_scan.active
    for _ in range(8):
        result = P.step(detections=_detections(_obj()), depth=depth, odom=motion,
                        state=state, config=cfg, dt=.1)
    assert result.values is not None and result.values[0] > 0., result.reason
    assert seen[0] is geometry


def test_foreground_in_bottom_of_depth_image_is_remembered_before_it_disappears(monkeypatch):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.odom import PlanarPose

    geom = local_path.camera_square(HorizontalIntrinsics(64, 50., 32.), (48, 50., 24.), .275)
    cfg = _cfg(use_obstacle_history=True, local_planning=True, local_geometry=geom, scan_geometry=geom)
    state = _state()
    scan = state.active_scan
    pose = PlanarPose(1100, ("room",), 0., 0., 0.)
    img = np.full((48, 64), 4.8)
    img[-4:, 28:36] = 1.4
    ground = np.zeros_like(img, dtype=bool)
    ground[-5:, :24] = True
    depth = dict(map=img, ground=ground, frame_id=1100, source_frame_id=1100, obstacle_capture_pose=pose)
    feedback = dict(obstacle_pose=pose, received_ms=1100, vx=0., vy=0., wz=0.)
    scan.update(depth, feedback, geom, receipt_only=True)
    # Isolate the real control path after a confirmed full-frame projected hit.
    # The centre-band clearance excludes the bottom of the image.
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (4.8, 1.))
    hit = (abs(local_path.RIGHT) < .06) & (abs(local_path.FORWARD-1.4) < .03)
    # A second measured obstruction must survive the first one leaving the corridor.
    hit |= (abs(local_path.RIGHT) < .06) & (abs(local_path.FORWARD-1.9) < .03)
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (~hit, hit))
    decision = P._avoid(.3, 0., 0., depth, cfg, P.APPROACHING, "", state, target_distance=4.8,
                        target_bearing=0., may_scan=True, odom=feedback)
    assert decision[3] == P.AVOIDING and not scan.obstacles.cells
    for stamp in (1110, 1120):
        pose = PlanarPose(stamp, pose.frame, 0., 0., 0.)
        depth.update(frame_id=stamp, source_frame_id=stamp, obstacle_capture_pose=pose)
        feedback.update(obstacle_pose=pose, received_ms=stamp)
        scan.update(depth, feedback, geom, receipt_only=True)
        decision = P._avoid(.3, 0., 0., depth, cfg, P.APPROACHING, "", state,
                            target_distance=4.8, target_bearing=0., may_scan=True, odom=feedback)
    assert scan.obstacles.cells
    if decision[0]:
        assert decision[1] and decision[2] == 0.
        assert local_path.transition_clear((~hit, hit), geom, P._keepout_m(cfg),
                                           (cfg.depth_age_limit_ms + cfg.watchdog_ms + 50) / 1000,
                                           (decision[:2],))
    assert scan.obstacles.project(pose)[local_path.START[0]+38, local_path.START[1]]



@pytest.mark.parametrize("change", ["none", "obstacle", "waypoint", "goal", "stall"])
def test_legacy_measured_detour_keeps_its_first_leg_across_new_depth(monkeypatch, change):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    cfg = _legacy_cfg(local_planning=True, use_lateral=True, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.attempted = True
    scan.receipt_only = True
    scan.poses.append((1000, scan.pose, 0.))
    scan.current_frame = 1000
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    ground = np.ones((4, 4), dtype=bool)
    command = (0., -.10)
    calls = []

    def plan(depth, config, state, odom, ahead, side):
        calls.append((depth["frame_id"], depth["allow_unknown"]))
        state.local_plan.plans[side] = local_path.Detour(command, ((0., 0.), (0., .60)), geometry)
        return command

    monkeypatch.setattr(P, "_local_translation", plan)
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (~occupied, occupied))
    monkeypatch.setattr(P, "_clearance", lambda depth, config, lateral_offset_m=0.: (
        (.5 if lateral_offset_m == 0 else (2. if lateral_offset_m > 0 else 1.5)), 1.))

    def tick(stamp, measured, distance=3.):
        return P._avoid(.15, 0., 0., dict(map=ground, ground=ground, frame_id=stamp),
                        cfg, P.APPROACHING, "", state, target_distance=distance,
                        target_bearing=0., odom=dict(vx=0., vy=measured, wz=0.), may_scan=True)[:3]

    assert tick(1000, 0.) == (*command, 0.)
    state.commanded = (*command, 0.)
    scan.pose = (0., .60 if change == "waypoint" else .01, 0.)
    scan.poses.append((1100, scan.pose, 0.))
    scan.current_frame = 1100
    occupied[local_path.START[0], local_path.START[1] + 6] = change == "obstacle"
    if change == "stall":
        state.side_stall_s[state.escape_side] = cfg.stuck_window_s
    assert tick(1100, -.10, 3.6 if change == "goal" else 3.) == (
        (*command, 0.) if change == "none" else (0., 0., 0.))
    assert calls == [(1000, True)]


def test_legacy_checked_local_translation_takes_priority_over_another_turn(monkeypatch):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .20, .20, None)
    cfg = _legacy_cfg(local_planning=True, use_lateral=True, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only = True
    scan.current_frame = 1000
    scan.poses.append((1000, scan.pose, 0.))
    command = (0., -.10)
    calls = []

    def plan(depth, config, state, odom, ahead, side):
        state.local_plan.plans[side] = local_path.Detour(command, ((0., 0.), (0., .60)), geometry)
        return command

    def plan_turn(*args, **kwargs):
        calls.append(True)
        return True

    monkeypatch.setattr(P, "_local_translation", plan)
    monkeypatch.setattr(P, "_start_detour_segment", plan_turn)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.50, 1.))
    ground = np.ones((4, 4), dtype=bool)
    decision = P._avoid(.15, 0., 0., dict(map=ground, ground=ground, frame_id=1000),
                        cfg, P.APPROACHING, "", state, target_distance=3., target_bearing=0.,
                        odom=dict(vx=0., vy=0., wz=0.), may_scan=True)
    assert decision[:3] == (*command, 0.)
    assert not calls


@pytest.mark.parametrize("arc_blocked", [False, True])
def test_checked_approach_can_move_while_turning(monkeypatch, arc_blocked):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    cfg = _cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    state.active_scan.attempted = True
    blocked = np.isclose(local_path.FORWARD, .95)
    if arc_blocked:
        blocked[local_path.START[0] + 6, local_path.START[1] + 6] = True
    ground = np.ones((4, 4), dtype=bool)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.95, 1.))
    monkeypatch.setattr(local_path, "observe", lambda *args, **kwargs: (~blocked, blocked))
    decision = P._avoid(.20, 0., .20, dict(map=ground, ground=ground, frame_id=1000),
                        cfg, P.APPROACHING, "", state, target_distance=3., target_bearing=-.15,
                        odom=dict(vx=0., vy=0., wz=0.))
    assert decision[0] > 0. and decision[1] == 0.
    assert decision[2] == (0. if arc_blocked else .20)


def test_legacy_local_plan_rechecks_and_drives_without_a_ground_mask():
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(640, 420, 320),
                                           (480, 420, 240), .20)
    cfg = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    frame = np.full((480, 640), np.nan)
    frame[220:370, 302:338] = .95

    def tick(stamp, image):
        depth = dict(map=image, frame_id=stamp)
        odom = dict(vx=0., vy=0., wz=0., received_ms=stamp)
        assert scan.update(depth, odom, geometry, receipt_only=True) is None
        return P._avoid(.15, 0., 0., depth, cfg, P.APPROACHING, "", state,
                        target_distance=3., target_bearing=0., odom=odom, may_scan=True, dt=.1)

    decision = tick(1000, frame)
    assert decision[:3] == (0., 0., 0.) and scan.route_segment
    assert state.view_detour is not None
    scan.phase = "evaluate"
    assert tick(1100, frame)[0] == 0.  # A measured hit still blocks the corridor.
    # At the new heading the post has moved out of the swept corridor.
    frame = np.full_like(frame, np.nan)
    frame[220:370, 520:538] = .95
    scan.phase = "evaluate"
    decision = tick(1200, frame)
    assert decision[0] > 0. and decision[1:3] == (0., 0.)


def test_legacy_local_plan_dispatches_a_checked_turn_on_the_first_tick_while_approaching():
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(640, 420, 320), (480, 420, 240), .20)
    config = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=geometry,
                  half_width_m=.20, watchdog_ms=500, wz_max=1.)
    state = _state(target="person", motion_source="measured", commanded=(.2, 0., 0.),
                   goal_anchor=P.DetourGoal(3., 0.))
    frame = np.full((480, 640), np.nan)
    frame[220:370, 302:338] = .65
    depth = dict(map=frame, frame_id=1000, capture_ms=980, control_ms=1000)
    odom = dict(vx=.12, vy=0., wz=0., received_ms=1000, source_sample_id=1000,
                velocity_history=((1000, 1000, (.12, 0., 0.)),))

    decision = P.step(detections=None, depth=depth, odom=odom, config=config,
                      state=state, dt=.1, control_period_s=.1)

    assert state.active_scan.route_segment and state.active_scan.phase == "turn"
    assert state.view_detour is not None
    assert decision.values[:5] == [.20, 0., 0., 0., 0.] and 0. < abs(decision.values[5]) <= config.wz_max


@pytest.mark.parametrize("ahead, obstacle, moves", [
    (.83, False, True), (.48, False, False), (.83, True, False),
])
def test_post_turn_clear_corridor_uses_stopping_distance(ahead, obstacle, moves):
    from plugins.navi import local_path

    geometry = local_path.Geometry(None, None, np.eye(4), .20, .20, None)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20,
                  obstacle_stop_m=.50)
    state = _state()
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1200
    scan.phase, scan.advance_m = "evaluate", .45
    scan.poses.append((1200, scan.pose, 0.))
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    if obstacle:
        occupied = np.isclose(local_path.FORWARD, .35)
    scan.views.append((1200, scan.pose, ~occupied, occupied))
    feedback = dict(vx=0., vy=0., wz=0.)
    result = P._advance_after_scan({"frame_id": 1200}, config, state, feedback,
                                   ahead, config.obstacle_stop_m, 2., .1)
    assert (result[0] > 0.) is moves


def test_verified_target_corridor_releases_near_stop_recovery(monkeypatch):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .20)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20,
                  obstacle_stop_m=.50, use_lateral=False)
    state = _state(recovery=True, blocked_for_s=1.)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1200
    occupied = np.isclose(local_path.FORWARD, .58, atol=.025) & (abs(local_path.RIGHT) > .50)
    scan.views.append((1200, scan.pose, ~occupied, occupied))
    frame = np.ones((8, 8))
    depth = dict(frame_id=1200, map=frame, ground=frame.astype(bool))
    feedback = dict(vx=0., vy=0., wz=0.)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.58, 1.))
    result = P._avoid(.2, 0., 0., depth, config, P.APPROACHING, "", state,
                      target_distance=2., target_bearing=0., odom=feedback, may_scan=True)
    assert result[:3] == (.2, 0., 0.)
    assert not state.recovery and not state.blocked_for_s


@pytest.mark.parametrize("forward,right,range_hint,blocked", [
    (.4, .55, .4, False), (.25, .30, .25, False), (2., 0., 2., False),
    (.4, .10, .4, True), (1., .50, None, True),
])
def test_new_detour_requires_an_obstacle_along_the_target_path(monkeypatch, forward, right, range_hint, blocked):
    g = P.local_path
    geometry = g.camera_square(P.HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .20)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20, stop_distance_m=.60)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    occupied = np.isclose(g.FORWARD, forward) & np.isclose(g.RIGHT, right)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    frame = np.ones((8, 8))
    depth = dict(frame_id=1000, map=frame, ground=frame.astype(bool))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (range_hint, 1.))
    detour = (.10, 0., -.25, P.AVOIDING, "detour")
    calls = []
    monkeypatch.setattr(P, "_dwa_unavailable_reason", lambda *args, **kwargs: None)
    def plan(*args):
        calls.append(True)
        return detour
    monkeypatch.setattr(P, "_dwa_recovery_route", plan)
    result = P._avoid(.2, 0., 0., depth, config, P.APPROACHING, "tracking", state,
                      target_distance=2., target_bearing=.45 if range_hint is None else 0.,
                      odom=dict(vx=0., vy=0., wz=0.), may_scan=True)
    assert bool(calls) == blocked
    assert result == (detour if blocked else (.2, 0., 0., P.APPROACHING, "tracking"))


@pytest.mark.parametrize("range_m, speed", [(1.1, .2), (.75, .1)])
def test_default_speed_reduction_starts_at_one_metre(monkeypatch, range_m, speed):
    config = _cfg(stop_distance_m=.60)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (range_m, 1.))
    result = P._avoid(.2, 0., 0., _scene(range_m), config, P.APPROACHING, "",
                      target_distance=range_m)
    assert config.obstacle_stop_m == .50
    assert result[0] == pytest.approx(speed)


def test_unvalidated_dynamic_mount_uses_measured_hits_without_world_memory(monkeypatch):
    from dataclasses import replace

    from plugins.navi.local_path import camera_square
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.odom import PlanarPose

    geometry = replace(camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .2),
                       free_space_validated=False)
    cfg = _cfg(local_planning=True, local_geometry=geometry, scan_geometry=geometry,
               half_width_m=.2, use_lateral=False, wz_max=0.)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only = True
    wall = np.isclose(P.FORWARD, .60)
    frame = np.full((8, 8), 3.)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (3., 1.))
    monkeypatch.setattr(P.local_path, "observe", lambda *args, **kwargs: (~wall, wall))

    def command(stamp, hits):
        pose = PlanarPose(stamp, ("odom", "body"), 0., 0., 0.)
        scan.current_frame = stamp
        scan.poses.append((stamp, scan.pose, 0.))
        scan.views.clear()
        scan.views.append((stamp, scan.pose, ~hits, hits))
        observation = dict(frame_id=stamp, capture_ms=stamp, map=frame, ground=frame.astype(bool),
                           body_mount_required=True, obstacle_capture_pose=pose)
        feedback = dict(vx=0., vy=0., wz=0., obstacle_pose=pose)
        return P._avoid(.3, 0., 0., observation, cfg, P.APPROACHING, "", state,
                        target_distance=3., target_bearing=0., odom=feedback, may_scan=True)

    assert command(1000, wall)[0] < .3
    assert not scan.obstacles.cells
    assert command(1100, np.zeros_like(wall))[0] == .3


@pytest.mark.parametrize("right, can_move", [(0., False), (.65, True)])
def test_blind_spot_history_only_blocks_a_motion_that_sweeps_through_it(monkeypatch, right, can_move):
    geometry = P.local_path.Geometry(None, None, np.eye(4), .25, .25, None)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.25,
                  use_lateral=False, wz_max=0.)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    empty = np.zeros(P.local_path.SHAPE, dtype=bool)
    scan.views.append((1000, scan.pose, empty, empty))
    scan.near_projection = NearProjection(hits=np.isclose(P.FORWARD, .40) & np.isclose(P.RIGHT, right))
    frame = np.full((8, 8), 3.)
    depth = dict(frame_id=1000, map=frame, ground=frame.astype(bool))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (3., .9))
    result = P._avoid(.30, 0., 0., depth, config, P.APPROACHING, "", state,
                      target_distance=3., target_bearing=0.,
                      odom={"vx": 0., "vy": 0., "wz": 0.}, may_scan=True)
    assert (result[0] > 0.) is can_move


def test_legacy_reobserved_original_target_rechecks_direct_corridor_after_exhaustion(monkeypatch):
    from types import SimpleNamespace
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .25)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.25, use_lateral=False)
    track = SimpleNamespace(observed=True, range_known=True, range_m=2.5, measured_range_m=2.7,
                            bearing_rad=0., last_obs=(2.65, .52))
    state = _state(motion_source="measured", goal_anchor=P.DetourGoal(2.1, 0., track=track))
    state.active_scan.attempted = state.active_scan.exhausted = True
    empty = np.zeros(P.local_path.SHAPE, dtype=bool)
    table = np.isclose(P.FORWARD, 1.2) & np.isclose(P.RIGHT, .30)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: track)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~table, table))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1.2, 1.))
    frame = np.ones((8, 8))
    decision = P.step(detections=_detections(), depth=dict(map=frame, ground=frame.astype(bool),
                    frame_id=1000), odom=dict(vx=0., vy=0., wz=0., received_ms=1000),
                    config=config, state=state, dt=.1)
    assert decision.values[0] > 0.
    assert decision.distance_m == pytest.approx(2.7)
    assert decision.bearing < 0.
    assert state.goal_anchor is None and not state.active_scan.exhausted


@pytest.mark.parametrize("direct_blocked", [False, True])
def test_legacy_moved_original_target_updates_goal_during_active_detour(monkeypatch, direct_blocked):
    from types import SimpleNamespace
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .25)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, use_lateral=False)
    point = (2.6, -.8)
    angle = math.atan2(-point[1], point[0])
    distance = math.hypot(*point)
    track = SimpleNamespace(observed=True, range_known=True, front_fix=point,
                            measured_range_m=distance, range_m=distance, bearing_rad=angle)
    anchor = P.DetourGoal(3.3, 0., track=track, target_missing=True)
    state = _state(motion_source="measured", goal_anchor=anchor)
    scan = state.active_scan
    scan.receipt_only, scan.phase, scan.advance_m, scan.attempted = True, "evaluate", 1., True
    scan.advance_origin = scan.pose
    scan.route_segment = True
    free = np.ones(P.local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (free, ~free))
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: track)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (3.5, 1.))
    if direct_blocked:
        monkeypatch.setattr(P, "_target_corridor_clear", lambda *args: False)
    frame = np.full((8, 8), 3.5)
    decision = P.step(detections=_detections(), depth=dict(map=frame, frame_id=1000),
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000),
                      config=config, state=state, dt=.1)

    assert (anchor.x, anchor.y) == pytest.approx(point)
    if direct_blocked:
        assert state.goal_anchor is anchor and scan.advance_m == 1.
    else:
        assert state.goal_anchor is None and scan.advance_m is None
    assert decision.values[0] > 0. and decision.distance_m == pytest.approx(distance)
    assert decision.bearing == pytest.approx(angle / config.half_fov_rad)


@pytest.mark.parametrize("blocked", [False, True])
def test_legacy_lost_detour_target_checks_motion_towards_remembered_goal(monkeypatch, blocked):
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .25)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.25)
    state = _state(motion_source="measured", goal_anchor=P.DetourGoal(2., .8), last_clearance=.8)
    occupancy = np.zeros(P.local_path.SHAPE, dtype=bool)
    occupancy[P.local_path.START] = blocked
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~occupancy, occupancy))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.8, 1.))
    frame = np.ones((8, 8))
    decision = P.step(detections=None, depth=dict(map=frame, ground=frame.astype(bool), frame_id=1000),
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000),
                      config=config, state=state, dt=.1)
    if blocked:
        assert not any(decision.values)
    else:
        assert decision.values[0] > 0. and decision.values[5] > 0.
    assert decision.status != P.SEARCHING and state.goal_anchor is not None


def test_linear_feedback_above_speed_limit_reduces_command_without_failing():
    cfg = _cfg(vx_max=.30)
    detection, frame = _detections(_obj()), _scene(3.)
    state = seed(_state(), detection, frame, cfg)
    decision = _step(detections=_next_frame(detection), depth=frame, config=cfg, state=state,
                     odom=dict(vx=.55, vy=0., wz=0.))
    assert 0. < decision.values[0] < cfg.vx_max
    assert not state.failed_reason


@pytest.mark.parametrize("heading,yaw,allowed", [
    (0.92, .0386, True), (5., .50, True), (5.01, 0., False), (0., .501, False),
])
def test_execution_leg_heading_and_yaw_thresholds(heading, yaw, allowed):
    grid = np.zeros(P.local_path.SHAPE, dtype=bool)
    geometry = P.local_path.Geometry(None, None, np.eye(4), .20, .20, None)
    leg = P.PathLeg(geometry, (1., 0.), (0., 0., 0.))
    command = (.1, 0.)
    reason = leg.check((0., 0., math.radians(heading)), (.1, 0., yaw), command,
                       command, (grid, grid), .25, 1., .35)
    assert (reason is None) == allowed


def test_legacy_distant_blocked_corridor_releases_a_checked_arc_prefix(monkeypatch):
    from plugins.navi.camera import HorizontalIntrinsics

    grid = P.local_path
    geometry = grid.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    config = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    observed = dict(map=np.ones((4, 4)), ground=None, frame_id=1000, control_ms=1000)
    occupied = np.isclose(grid.FORWARD, 2.) & (abs(grid.RIGHT) <= .10)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, ~occupied, occupied))

    result = P._avoid(.20, 0., 0., observed, config, P.APPROACHING, "", state,
                      target_distance=3., target_bearing=0., may_scan=True, dt=.1,
                      odom=dict(vx=0., vy=0., wz=0.))
    assert result[0] > 0. and result[1] == 0. and result[2] != 0.
    assert not scan.active
    assert state.view_detour is not None and len(state.view_detour.points) > 2


def test_legacy_early_detour_follows_a_checked_passing_lane_instead_of_target_yaw(monkeypatch):
    from plugins.navi.camera import HorizontalIntrinsics

    grid = P.local_path
    geometry = grid.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    config = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.poses.append((1000, scan.pose, 0.))
    occupied = np.isclose(grid.FORWARD, 1.8) & (abs(grid.RIGHT) <= .10)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    depth = dict(map=np.ones((4, 4)), ground=None, frame_id=1000, control_ms=1000)
    waypoint = (1.2, -.40)
    route = grid.Detour(None, ((0., 0.), waypoint), geometry)
    monkeypatch.setattr(grid, "opening_route", lambda *args: route)

    result = P._avoid(.25, 0., -.15, depth, config, P.APPROACHING, "", state,
                      target_distance=3., target_bearing=0., may_scan=True, dt=.1,
                      odom=dict(vx=.15, vy=0., wz=0.))
    assert result[0] > 0. and result[2] > 0.
    assert state.checked_prefix and not scan.active
    assert state.view_detour.points[-1][1] < 0.


def test_side_waypoint_can_be_reached_by_turning_then_driving(monkeypatch):
    geometry = P.local_path.Geometry(None, None, np.eye(4), .20, .20, None)
    config = _cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.poses.append((1000, scan.pose, 0.))
    free = np.zeros(P.local_path.SHAPE, dtype=bool)
    scan.views.append((1000, scan.pose, free, free.copy()))
    side_route = P.local_path.Detour(None, ((0., 0.), (0., .55)), geometry)
    monkeypatch.setattr(P.local_path, "opening_route", lambda *args: None)
    monkeypatch.setattr(P.local_path.PlanCache, "route", lambda *args: side_route)

    assert P._start_detour_segment(dict(frame_id=1000, map=np.ones((4, 4))), config, state,
                                   geometry, 1., (1.,), None, None)
    assert scan.turn_angle == pytest.approx(math.pi / 2)
    assert scan.advance_m == pytest.approx(.55)


def test_started_route_accepts_residual_turn_but_still_checks_new_obstacles():
    geometry = P.local_path.Geometry(None, None, np.eye(4), .20, .20, None)
    config = _cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.route_segment, scan.advance_m = True, 1.
    scan.advance_origin = (0., 0., 0.)
    scan.pose = (.05, 0., math.radians(1.))
    occupied = np.zeros(P.local_path.SHAPE, dtype=bool)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    odom = dict(vx=.1, vy=0., wz=.05)
    result = P._advance_after_scan({}, config, state, odom, 1.5, .5, 2.5, .1)
    assert result[0] > 0. and result[1:3] == (0., 0.)

    occupied[P.local_path.START[0] + 2, P.local_path.START[1] + 5] = True
    result = P._advance_after_scan({}, config, state, odom, 1.5, .5, 2.5, .1)
    assert result[:3] == (0., 0., 0.)


@pytest.mark.parametrize("blocked", [False, True])
def test_legacy_new_view_rechecks_the_target_corridor_before_continuing_an_observation_turn(monkeypatch, blocked):
    from types import SimpleNamespace
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .25)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.25, use_lateral=False)
    track = SimpleNamespace(observed=True, range_known=True, range_m=2.5, measured_range_m=2.5,
                            bearing_rad=0., last_obs=(2.5, 0.))
    state = _state(motion_source="measured", goal_anchor=P.DetourGoal(2.5, 0., track=track))
    scan = state.active_scan
    scan.receipt_only, scan.phase = True, "turn"
    scan.attempted, scan.directions, scan.origin_set = True, (1.,), True
    scan.start_ms = scan.still_since = 900
    occupancy = np.zeros(P.local_path.SHAPE, dtype=bool)
    occupancy |= blocked & np.isclose(P.FORWARD, 1.) & (abs(P.RIGHT) < .15)
    monkeypatch.setattr(state.tracker, "step", lambda **kwargs: track)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~occupancy, occupancy))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (1. if blocked else 2.5, 1.))
    frame = np.ones((8, 8))
    decision = P.step(detections=_detections(), depth=dict(map=frame, frame_id=1000, control_ms=1000),
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000),
                      config=config, state=state, dt=.1)
    assert (decision.values[0] > 0.) == (not blocked)
    assert scan.active == blocked


@pytest.mark.parametrize("blocked,expect_search", [("side", True), ("inside", False)])
def test_target_search_uses_a_directional_sweep_for_retained_obstacles(monkeypatch, blocked, expect_search):
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .23)
    config = _cfg(use_obstacle_history=True, local_planning=True, local_geometry=geometry, half_width_m=.23,
                  search_rate=.6, lost_frames=1, max_depth_age_ms=800)
    state = _state(motion_source="measured")
    scan = state.active_scan
    hits = np.zeros(P.local_path.SHAPE, dtype=bool)
    row, col = P.local_path.START
    if blocked == "side":
        hits[row + 7, col + 8] = True
    else:
        hits[row, col + 2] = True
    scan.near_projection = NearProjection(hits=hits, stamp_ms=1000)
    monkeypatch.setattr(active_scan.ActiveScan, "update_near_obstacles", lambda *args, **kwargs: None)
    empty = np.zeros_like(scan.near_hits)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~empty, empty))
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (None, .9))
    frame = np.ones((8, 8))
    decision = P.step(detections=_detections(), depth=dict(map=frame, frame_id=1000, control_ms=1000),
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000), state=state, config=config, dt=.1)
    assert (decision.status == P.SEARCHING and abs(decision.values[5]) > 0) == expect_search
    assert decision.values[0] == 0.


@pytest.mark.parametrize("obstacle_right,can_turn", [(.40, True), (.15, False)])
def test_stationary_search_checks_the_requested_arc_inside_the_turn_circle(monkeypatch, obstacle_right, can_turn):
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    g = P.local_path
    geometry = g.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .20, half_length=.45)
    config = _cfg(local_planning=True, local_geometry=geometry, scan_geometry=geometry,
                  half_width_m=.20, search_rate=.20, lost_frames=1, use_obstacle_history=True)
    state = _state(motion_source="measured")
    hits = np.isclose(g.FORWARD, 0.) & np.isclose(g.RIGHT, obstacle_right)
    assert hits.any() and not state.active_scan.clear_sweep(geometry, .20, 0., occupied=hits)
    state.active_scan.near_projection = NearProjection(hits=hits, stamp_ms=1000)
    empty = np.zeros_like(hits)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~empty, empty))
    frame = np.ones((8, 8))
    decision = P.step(detections=_detections(), depth=dict(map=frame, frame_id=1000, control_ms=1000),
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000), state=state, config=config, dt=.1)
    assert bool(decision.values[5]) == can_turn
    if can_turn:
        assert state.dwa_motion_guard.check.accepted and decision.status == P.SEARCHING


@pytest.mark.parametrize("lateral", [.35, .40])
def test_off_axis_remembered_door_jamb_allows_a_checked_prefix(monkeypatch, lateral):
    from plugins.navi.camera import HorizontalIntrinsics

    g = P.local_path
    geometry = g.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .275)
    config = _cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.275,
                  max_depth_age_ms=800, obstacle_stop_m=.50, vx_max=.30)
    state = _state(motion_source="measured", commanded=(.2, 0., 0.))
    scan = state.active_scan
    scan.receipt_only, scan.current_frame, scan.control_ms = True, 1000, 1000
    scan.poses.append((1000, scan.pose, 0.))
    grid = np.zeros(g.SHAPE, dtype=bool)
    scan.views.append((1000, scan.pose, ~grid, grid))
    hits = np.isclose(g.FORWARD, .65) & np.isclose(g.RIGHT, lateral)
    scan.near_projection = NearProjection(hits=hits, stamp_ms=1000)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (2.5, 1.))
    view = dict(map=np.ones((4, 4)), ground=np.ones((4, 4), dtype=bool), frame_id=1000)
    decision = P._avoid(.2, 0., .3, view, config, P.APPROACHING, "", state,
                        target_distance=2.5, target_bearing=.10, may_scan=True,
                        odom=dict(vx=.20, vy=0., wz=0.))
    assert decision[:3] == (.2, 0., .3)


def test_legacy_view_received_during_interrupted_observation_is_remembered_without_paired_pose(monkeypatch):
    from plugins.navi import active_scan
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = P.local_path.camera_square(HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .23)
    config = _legacy_cfg(local_planning=True, local_geometry=geometry, half_width_m=.23)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.phase = True, "turn"
    scan.attempted, scan.directions, scan.origin_set = True, (1.,), True
    scan.start_ms, scan.still_since = 900, 900
    scan.pose = (0., 0., .18)
    empty = np.zeros(P.local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(active_scan, "observe", lambda *args: (~empty, empty))
    frame = np.ones((8, 8))
    P.step(detections=_detections(), depth=dict(map=frame, frame_id=1000, control_ms=1000),
           odom=dict(vx=0., vy=0., wz=0., received_ms=1000), state=state, config=config, dt=.1)
    assert scan.phase == "turn"
    scan.interrupt()
    scan.note_clear_path(1000)
    scan.pose = (0., 0., 0.)
    assert not scan.begin((1.,), turn_angle=.18)


def test_legacy_wheeled_detour_can_leave_box_corner_when_grid_search_has_no_complete_route(monkeypatch):
    from plugins.navi.camera import HorizontalIntrinsics

    g = P.local_path
    body = g.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .275)
    config = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=body, half_width_m=.275,
                  max_depth_age_ms=1200, vx_max=.30)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.poses.append((1000, scan.pose, 0.))
    empty = np.zeros(g.SHAPE, dtype=bool)
    scan.views.append((1000, scan.pose, empty, empty.copy()))
    hits = (g.FORWARD >= .50) & (g.FORWARD <= .85) & (g.RIGHT >= .29)
    scan.near_projection = NearProjection(hits=hits)
    monkeypatch.setattr(g.PlanCache, "route", lambda *args, **kwargs: None)
    monkeypatch.setattr(g, "opening_route", lambda *args, **kwargs: None)

    assert P._start_detour_segment(dict(frame_id=1000, map=np.ones((4, 4))), config, state,
                                   body, .5, (-1., 1.), 3.9, 0.)
    assert scan.directions == (-1.,)
    assert scan.advance_m >= .30
    assert state.view_detour.points[-1][1] < 0.
    assert g.segment_clear(scan.grid()[1], body, P._keepout_m(config),
                           -scan.turn_angle, scan.advance_m, heading_tolerance=P.ANGLE_TOLERANCE)


@pytest.mark.parametrize("person_range,arrived", [(2.6, False), (.599, True)])
def test_person_arrival_uses_box_measurement_at_configured_standoff(person_range, arrived):
    from plugins.navi.camera import HorizontalIntrinsics

    config = _cfg(confirm_hits=1, stop_distance_m=.60, stop_tolerance_m=.01,
                  depth_intrinsics=HorizontalIntrinsics(80, 65., 39.5))
    state = P.State(target="person")
    person = _obj("person", bbox=(.2, .1, .8, .9))
    frame = np.full((60, 80), person_range, dtype=np.float32)
    frame[22:31, 36:44] = .599

    decision = P.step(detections=_detections(person), depth=dict(map=frame, frame_id=100),
                      odom=None, config=config, state=state, dt=.1)
    assert decision.status != P.ARRIVED and not any(decision.values or ())
    decision = P.step(detections=_detections(person), depth=dict(map=frame, frame_id=200),
                      odom=None, config=config, state=state, dt=.1)
    assert (decision.status == P.ARRIVED) == arrived
    assert not any(decision.values)
    assert decision.distance_m == pytest.approx(person_range, abs=.002)


def test_legacy_wheeled_box_corner_prefers_longer_checked_escape_to_the_open_side(monkeypatch):
    from plugins.navi.camera import HorizontalIntrinsics

    grid = P.local_path
    geometry = grid.camera_square(HorizontalIntrinsics(640, 492., 318.),
                                  (480, 492., 229.), .275, at_front=True)
    config = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=geometry,
                  half_width_m=.275, max_depth_age_ms=1200, watchdog_ms=500, vx_max=.30)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    empty = np.zeros(grid.SHAPE, dtype=bool)
    scan.poses.append((1000, scan.pose, 0.))
    scan.views.append((1000, scan.pose, empty, empty))
    hits = (grid.FORWARD >= .4) & (grid.FORWARD <= .71) & (grid.RIGHT >= .25) & (grid.RIGHT <= .40)
    scan.near_projection = NearProjection(hits=hits)
    monkeypatch.setattr(grid.PlanCache, "route", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid, "opening_route", lambda *args, **kwargs: None)

    assert P._start_detour_segment(dict(frame_id=1000, map=np.ones((4, 4))),
                                   config, state, geometry, .45, (1., -1.), 2.8, -.06)
    assert scan.directions == (-1.,)
    assert scan.advance_m == pytest.approx(.65)


def test_checked_prefix_slows_before_stopping_for_an_obstacle_near_the_sweep():
    grid = P.local_path
    geometry = grid.camera_square(P.HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .275, at_front=True)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.275)
    state = _state()
    scan = state.active_scan
    scan.receipt_only = True
    occupied = np.isclose(grid.FORWARD, .20) & np.isclose(grid.RIGHT, .30)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    result = P._checked_forward_prefix(scan, geometry, config, state, None, (0., 0., 0.),
                                       .15, 1.75, 1., 0.)
    assert result is not None and 0. < result[0] < .15
    assert not grid.motion_sweep_clear((~occupied, occupied), geometry, .275, 1.75, ((.15, 0., 0.),))
    assert grid.motion_sweep_clear((~occupied, occupied), geometry, .275, 1.75, (result[:3],))


@pytest.mark.parametrize("corridor_clear", [False, True])
@pytest.mark.parametrize("route_state", ["initial", "published", "memory"])
@pytest.mark.parametrize("ahead", [1.2, 1.8])
def test_camera_only_tries_an_arc_for_a_visible_blocked_corridor(monkeypatch, corridor_clear, route_state, ahead):
    grid = P.local_path
    geometry = grid.camera_square(P.HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .275, at_front=True)
    config = _cfg(local_planning=True, local_geometry=geometry, use_lateral=False)
    state = _state(motion_source="command")
    if route_state == "published":
        state.published_camera_passage = object()
    elif route_state == "memory":
        state.goal_anchor = P.DetourGoal(3., 0.)
    scan = state.active_scan
    scan.command_only, scan.receipt_only, scan.current_frame = True, True, 1000
    occupied = np.isclose(grid.FORWARD, ahead) & (abs(grid.RIGHT) <= .20)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    depth = dict(_scene(3.), frame_id=1000)
    command = (.2, 0., -.12, P.AVOIDING, "checked camera arc")
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (ahead, 1.))
    monkeypatch.setattr(P, "_target_corridor_clear", lambda *args: corridor_clear)
    seen = []

    def camera_arc(*args, ahead, **kwargs):
        seen.append(round(ahead, 6) if ahead is not None else None)
        return command if ahead is not None else None

    monkeypatch.setattr(P, "_camera_detour", camera_arc)
    result = P._avoid(.2, 0., -.2, depth, config, P.APPROACHING, "", state,
                      target_distance=3., target_bearing=.2, may_scan=True)
    if route_state == "published" and ahead < config.approach_check_distance_m:
        assert seen == [ahead]
        assert result == command
        return
    tries_arc = not corridor_clear and (route_state != "memory" or ahead < config.approach_check_distance_m)
    assert seen == ([None] if route_state == "published" else []) + ([ahead] if tries_arc else [])
    if tries_arc:
        assert result == command


@pytest.mark.parametrize(("ahead", "corridor_clear", "has_route"),
                         [(None, True, False), (1.2, True, False), (1.2, False, False),
                          (None, True, True), (None, False, True)])
def test_remembered_goal_only_plans_a_detour_for_a_blocked_corridor(monkeypatch, ahead, corridor_clear, has_route):
    grid = P.local_path
    geometry = grid.camera_square(P.HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .275, at_front=True)
    config = _cfg(local_planning=True, local_geometry=geometry, use_lateral=False)
    state = _state(motion_source="measured", goal_anchor=P.DetourGoal(3., 0.))
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    if has_route:
        state.dwa_guidance = P.DwaGuidance(scan.motion_frame, state.goal_anchor, ((0., 0.), (1., .4)), (0., 0.))
    occupied = np.isclose(grid.FORWARD, 1.2) & np.isclose(grid.RIGHT, .7 if corridor_clear else .25)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    depth = dict(_scene(3.), frame_id=1000, ground=occupied)
    command = (.2, 0., -.12, P.AVOIDING, "checked detour")
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (ahead, 1.))
    monkeypatch.setattr(P, "_target_corridor_clear", lambda *args: corridor_clear)
    monkeypatch.setattr(P, "_dwa_unavailable_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(P, "_start_detour_segment", lambda *args, **kwargs: command)

    result = P._avoid(.2, 0., -.2, depth, config, P.APPROACHING, "tracking", state,
                      target_distance=3., target_bearing=.2, may_scan=True)
    assert result == ((.2, 0., -.2, P.APPROACHING, "tracking") if corridor_clear else command)


def test_exhausted_camera_route_retries_only_with_new_depth_and_search_budget():
    from plugins.navi.active_scan import ActiveScan

    scan = ActiveScan(receipt_only=True, attempted=True, exhausted=True, rerouted=True,
                      current_frame=1000, source_frame=900)
    assert scan.retry_route(500)
    assert not scan.rerouted
    assert not scan.retry_route(500)
    scan.current_frame = 1600  # Re-receiving the same depth is no new evidence.
    assert not scan.retry_route(500)
    scan.source_frame = 1399
    assert not scan.retry_route(500)
    scan.source_frame = 1400
    assert scan.retry_route(500)


def test_occluded_cells_stay_unknown_and_new_free_returns_take_precedence():
    grid = P.local_path
    geometry = grid.Geometry(None, None, np.eye(4), 0., .55, None)
    empty = np.zeros(grid.SHAPE, dtype=bool)
    occupied = np.isclose(grid.FORWARD, 1.) & (abs(grid.RIGHT) <= .15)
    shadow = grid.occluded_cells(empty, occupied, geometry)
    assert shadow[grid.START[0] + 30, grid.START[1]]
    assert not shadow[grid.START[0] + 15, grid.START[1]]
    assert not shadow[grid.START[0] + 30, grid.START[1] + 15]
    assert not (shadow & occupied).any()

    free = np.isclose(grid.FORWARD, 1.5) & np.isclose(grid.RIGHT, 0.)
    restored = grid.occluded_cells(free, occupied, geometry)
    assert not (restored & free).any()
    assert grid.visible_prefix(((0., 0.), (.5, .30), (1.5, .30), (1.5, 0.)), shadow) == 3
    assert grid.visible_prefix(((0., 0.), (.5, .30), (1.5, .30)), restored) == 3


def test_legacy_local_detour_keeps_occluded_tail_as_preview_and_executes_visible_waypoint():
    grid = P.local_path
    geometry = grid.camera_square(P.HorizontalIntrinsics(640, 420, 320), (480, 420, 240), .275, at_front=True)
    config = _legacy_cfg(local_planning=True, use_lateral=False, local_geometry=geometry, half_width_m=.275,
                  stop_distance_m=.60)
    state = _state(motion_source="measured")
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    scan.poses.append((1000, scan.pose, 0.))
    occupied = np.isclose(grid.FORWARD, .80) & (abs(grid.RIGHT) <= .23)
    shadow = grid.occluded_cells(np.zeros(grid.SHAPE, dtype=bool), occupied, geometry)
    free = (grid.FORWARD > .1) & (grid.FORWARD < 3.) & ~occupied & ~shadow
    scan.views.append((1000, scan.pose, free, occupied))
    depth = dict(frame_id=1000, map=np.full((480, 640), 4.), ground=np.ones((480, 640), dtype=bool))

    assert P._start_detour_segment(depth, config, state, geometry, .8, (-1., 1.), 2.3, 0.)
    assert scan.occluded_route
    route = state.view_detour
    assert len(route.points) == grid.visible_prefix(route.points, shadow) == 2
    assert grid.observed_waypoint(free, *route.points[-1])
    assert grid.visible_prefix(route.preview, shadow) < len(route.preview)
    assert scan.advance_m == pytest.approx(math.hypot(*route.points[-1]))


def test_checked_arc_uses_original_cell_without_dropping_its_collision_margin():
    from plugins.navi.odom import PlanarPose

    grid = P.local_path
    geometry = grid.Geometry(None, None, np.eye(4), 0., .55, None)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.275,
                  max_depth_age_ms=1200, watchdog_ms=500)
    state = _state(commanded=(.0375, 0., -.06249))
    scan = state.active_scan
    # The aged rotated square separates the low and high speed envelopes.
    pose = PlanarPose(1891, ("odom",), 9.38, -.71, -.28)
    scan.near_obstacles.frame = pose.frame
    scan.near_obstacles.cells.add((185, -23))
    hits, squares = scan.near_obstacles.project(pose, include_squares=True)
    scan.near_projection = NearProjection(hits=hits, origin=pose, stamp_ms=1891)
    scan.control_ms = 2000
    empty = np.zeros(grid.SHAPE, dtype=bool)
    scan.receipt_only = True
    scan.views.append((2000, scan.pose, ~empty, empty))
    motion = (-.000305, 0., 0.)
    odom = dict(vx=motion[0], vy=motion[1], wz=motion[2])

    def prefix():
        return P._checked_forward_prefix(scan, geometry, config, state, odom, motion,
                                         .15, 1.75, 1., -.06249, (-1., 1.))

    assert prefix()[0] == pytest.approx(.0375)
    scan.near_projection = NearProjection(hits, squares, pose, 1891)
    assert prefix()[0] == pytest.approx(.075)
    assert scan.grid()[1].sum() == 4


@pytest.mark.parametrize("accelerates", [True, False])
def test_acceleration_lag_is_not_a_stop_reason(accelerates):
    cfg = _cfg(local_controller="dwa", stuck_window_s=.2, idle_timeout_s=.85)
    scene = _depth(map_=_solid_depth(5.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    for i in range(15):
        stamp = 1000 + 100 * i
        vx = min(.2, max(0., (i - 3) * .05)) if accelerates else 0.
        odom = dict(vx=vx, vy=0., wz=0., received_ms=stamp, source_sample_id=stamp)
        decision = _step(_detections(_obj()), scene, odom, cfg, state)
        if decision.status == P.FAILED:
            break
        assert decision.values[0] > 0.
    if accelerates:
        assert decision.status == P.APPROACHING
    else:
        assert .85 <= (i + 1) * .1 < 1.
        assert decision.status == P.FAILED and "progress" in decision.reason
        assert decision.values == [0.] * 6


@pytest.mark.parametrize("feedback", ["outward", "stationary", "duplicate", "regressing", "oscillating", "camera_turn"])
def test_detour_progress_requires_net_measured_body_displacement(monkeypatch, feedback):
    from plugins.navi.local_path import Detour
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.6, blocked_timeout_s=.15)
    scene = _depth(map_=_solid_depth(3.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    # A left turn in the planned route moves away from a target on the right.
    route = ((0., 0.), (0., -.5), (.2, -.7))

    def avoid(*args, **kwargs):
        state.view_detour = Detour(None, ((0., 0.), (.1, 0.)), None, route)
        state.checked_prefix = True
        return 0., .20, 0., P.AVOIDING, "detour"

    monkeypatch.setattr(P, "_avoid", avoid)
    for i in range(14):
        stamp = 1000 + i * 100
        if feedback == "duplicate":
            stamp = 1000
        elif feedback == "regressing":
            stamp = 3000 - 100 * i
        left = .025 * i if feedback != "stationary" else 0.
        if feedback == "oscillating":
            left = .03 * (i % 2)
        # Direct odometry uses the body pivot even when the head rotates.
        pose = (PlanarPose(stamp, ("odom",), 0., 0., .1 * i) if feedback == "camera_turn" else
                PlanarPose(stamp, ("odom",), 0., left, 0.))
        decision = _step(_detections(_obj()), scene, dict(obstacle_pose=pose), cfg, state)
        if decision.status == P.FAILED:
            break
    if feedback == "outward":
        assert decision.status == P.AVOIDING and state.idle_for_s < .4
    else:
        assert decision.status == P.FAILED and "progress" in decision.reason
        assert i >= 5, "only the measured idle budget may terminate this DWA action"


def test_turning_in_place_cannot_reset_progress_forever(monkeypatch):
    from plugins.navi.active_scan import ANGLE

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.35)
    state = _state()
    scan = state.active_scan
    grid = np.ones((2, 2), dtype=bool)
    stamp = 0
    heading = 0.

    def turn(*args, **kwargs):
        scan.pose = (0., 0., heading)
        scan.poses.append((stamp, scan.pose, -.3))
        scan.source_frame = scan.side_frame = scan.current_frame = stamp
        scan.views.append((stamp, scan.pose, grid, ~grid))
        return P.Decision(P._twist(0., 0., -.3), P.AVOIDING, "evaluate view")

    monkeypatch.setattr(P, "_decide", turn)
    for i, heading in enumerate((0., 0., ANGLE, ANGLE, 2*ANGLE, ANGLE, 0., ANGLE, 2*ANGLE)):
        stamp = 1000 + 100*i
        depth = dict(capture_ms=stamp, control_ms=stamp)
        decision = P.step(detections=None, depth=depth, odom=None, state=state, config=cfg, dt=.1)
        if i in (2, 4):
            assert state.idle_for_s == 0., "a newly evaluated heading earns one credit"
        if i == 3:
            assert state.idle_for_s == pytest.approx(.1)
    assert decision.status == P.FAILED and "progress" in decision.reason


def _measured_target_search(*, feedback="turning", speed=.35, local_planning=True,
                            seen_after=None, search_timeout_s=30., absolute="none"):
    """Native target search with physical yaw samples and fresh empty images."""
    from plugins.navi import local_path
    from plugins.navi.odom import PlanarPose

    geometry = local_path.Geometry(None, None, np.array([
        [0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.],
    ]), .275, .275, None)
    cfg = _cfg(local_controller="dwa", local_planning=local_planning,
               local_geometry=geometry if local_planning else None,
               half_width_m=.275, use_lateral=False, vy_max=0.,
               search_rate=.6, idle_timeout_s=10., search_timeout_s=search_timeout_s)
    state = _state(motion_source="measured")
    measured_yaw = 0.
    for index in range(351):
        stamp = 1000 + index * 100
        rate = -speed if state.commanded[2] < 0. else 0.
        if feedback == "stationary":
            rate = 0.
        elif feedback in ("oscillating", "revisited"):
            # The wider twin visits new headings once, then repeats them.
            amplitude = .1 if feedback == "oscillating" else .25
            rate = (amplitude * math.sin(index * math.pi / 10)
                    - amplitude * math.sin((index - 1) * math.pi / 10)) / .1 if index else 0.
        measured_yaw += .1 * rate
        sample = 1000 if feedback == "duplicate" else stamp
        source = 36000 - index * 100 if feedback == "regressing" else sample
        odom = dict(vx=0., vy=0., wz=rate, received_ms=sample,
                    source_sample_id=source, control_ms=stamp)
        absolute_present = (absolute == "late" and index >= 10 or
                            absolute == "dropout" and index < 10 or
                            absolute == "intermittent" and index % 20 >= 10 or
                            absolute == "changing-frame" and index >= 10)
        if absolute_present:
            frame = ("odom", index // 10) if absolute == "changing-frame" else ("odom",)
            yaw = 0. if feedback == "duplicate" else measured_yaw
            # Absolute yaw has a different, perfectly valid zero from the
            # velocity integrator. Registration must retain visited sectors.
            odom["obstacle_pose"] = PlanarPose(sample, frame, 0., 0., 1.3 + yaw)
        if feedback == "missing":
            odom = None
        depth = dict(map=np.full((120, 160), 5., dtype=np.float32),
                     ground=np.ones((120, 160), dtype=bool), frame_id=stamp,
                     source_frame_id=stamp, capture_ms=stamp, control_ms=stamp,
                     bands=dict(left=5., center=5., right=5.))
        objects = [_obj()] if seen_after is not None and index * .1 >= seen_after else []
        detections = dict(timestamp=stamp / 1000., count=len(objects), objects=objects)
        result = P.step(detections=detections, depth=depth, odom=odom,
                        config=cfg, state=state, dt=.1)
        yield index * .1, result, state
        if result.status == P.FAILED:
            return


@pytest.mark.parametrize("local_planning", [False, True])
def test_dwa_target_search_credits_new_measured_sectors(local_planning):
    # Missing normal-search progress would abort at ten seconds after only
    # half a measured sweep, although the robot is still looking in new sectors.
    for elapsed, result, state in _measured_target_search(local_planning=local_planning):
        assert result.status != P.FAILED
        if elapsed >= 12.:
            break
    assert elapsed == pytest.approx(12.)
    assert result.status == P.SEARCHING and result.values[5] < 0.
    assert 3.5 < state.searched_rad < 4.5 and state.idle_for_s < 1.


@pytest.mark.parametrize("feedback", ["stationary", "oscillating", "duplicate", "missing"])
def test_dwa_target_search_without_new_measured_sectors_times_out(feedback):
    # Commanded rotation, stale feedback and repeated views cannot buy time.
    elapsed, result, state = list(_measured_target_search(feedback=feedback))[-1]
    assert result.status == P.FAILED and "progress" in result.reason
    assert elapsed <= 11. and state.idle_for_s >= 10.
    assert result.values == [0.] * 6


@pytest.mark.parametrize("speed,reason,minimum_elapsed", [
    (.35, "swept", 18.), (.1, "searched 30s", 30.),
])
def test_dwa_target_search_retains_original_sweep_and_time_limits(speed, reason, minimum_elapsed):
    elapsed, result, state = list(_measured_target_search(speed=speed))[-1]
    assert result.status == P.FAILED and reason in result.reason
    assert minimum_elapsed <= elapsed <= 31.
    assert state.idle_for_s < 3.


def test_dwa_target_search_can_reacquire_after_ten_seconds():
    for elapsed, result, state in _measured_target_search(seen_after=11.7):
        assert result.status != P.FAILED
        if elapsed >= 12.5:
            break
    assert elapsed == pytest.approx(12.5)
    assert state.tracker.track is not None and state.tracker.track.observed
    assert result.values[0] > 0.


@pytest.mark.parametrize("local_planning", [False, True])
@pytest.mark.parametrize("absolute", ["late", "dropout", "intermittent"])
def test_dwa_target_search_keeps_measured_sectors_across_absolute_pose_handoffs(local_planning, absolute):
    for elapsed, result, state in _measured_target_search(local_planning=local_planning, absolute=absolute):
        assert result.status != P.FAILED
        if elapsed >= 12.:
            break
    assert elapsed == pytest.approx(12.) and result.status == P.SEARCHING
    assert 3.5 < state.searched_rad < 4.5 and state.idle_for_s < 1.


@pytest.mark.parametrize("local_planning", [False, True])
@pytest.mark.parametrize("feedback", ["stationary", "duplicate", "regressing", "revisited"])
def test_dwa_search_pose_handoffs_do_not_renew_old_or_unmeasured_views(local_planning, feedback):
    baseline_elapsed, _, _ = list(_measured_target_search(
        local_planning=local_planning, feedback=feedback))[-1]
    elapsed, result, state = list(_measured_target_search(
        local_planning=local_planning, absolute="intermittent", feedback=feedback))[-1]
    assert result.status == P.FAILED and "progress" in result.reason
    assert result.values == [0.] * 6 and state.idle_for_s >= 10.
    # Brief occlusion delays search; the wider oscillation first covers its
    # two sectors at 1.3/1.4 s and 2.6 s, then has no useful rotation left.
    assert baseline_elapsed <= 13.
    assert elapsed <= baseline_elapsed + .1, "a coordinate handoff cannot extend the measured idle budget"


@pytest.mark.parametrize("local_planning", [False, True])
def test_dwa_search_frame_changes_cannot_relabel_repeated_sectors(local_planning):
    baseline_elapsed, _, _ = list(_measured_target_search(
        local_planning=local_planning, feedback="revisited"))[-1]
    elapsed, result, state = list(_measured_target_search(
        local_planning=local_planning, absolute="changing-frame", feedback="revisited"))[-1]
    assert result.status == P.FAILED and "progress" in result.reason
    assert result.values == [0.] * 6 and baseline_elapsed <= 13.
    assert elapsed <= baseline_elapsed + .1


@pytest.mark.parametrize("local_planning", [False, True])
@pytest.mark.parametrize("speed,reason,minimum_elapsed", [
    (.35, "swept", 18.), (.1, "searched 30s", 30.),
])
def test_dwa_search_absolute_pose_handoff_retains_original_limits(local_planning, speed, reason, minimum_elapsed):
    elapsed, result, state = list(_measured_target_search(
        local_planning=local_planning, absolute="intermittent", speed=speed))[-1]
    assert result.status == P.FAILED and reason in result.reason
    assert minimum_elapsed <= elapsed <= 31. and state.idle_for_s < 3.


@pytest.mark.parametrize("broken", [False, True])
def test_continuous_velocity_history_counts_between_slow_ticks(broken):
    cfg = _cfg(max_feedback_gap_ms=250, local_controller="dwa", idle_timeout_s=.38)
    scene = _depth(map_=_solid_depth(4.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    history = []
    for i in range(4):
        stamp = 1000 + 100*i
        if broken and i in (1, 2):
            continue
        history.append((stamp, stamp, (.25, 0., 0.)))
        if i in (0, 3):
            odom = dict(vx=.25, vy=0., wz=0., received_ms=stamp, source_sample_id=stamp,
                        velocity_history=tuple(history))
            decision = _step(_detections(_obj()), scene, odom, cfg, state, dt=.1 if i == 0 else .3)
    if broken:
        assert decision.status == P.FAILED
    else:
        assert decision.status == P.APPROACHING and state.idle_for_s == 0.


def test_recovery_view_can_be_projected_after_its_measured_heading(monkeypatch):
    from plugins.navi.active_scan import ANGLE

    state = _state()
    scan = state.active_scan
    cfg = _cfg(local_controller="dwa")
    grid = np.ones((2, 2), dtype=bool)
    stamp = 0

    def turn(*args, **kwargs):
        scan.pose = (0., 0., 0. if stamp == 1000 else ANGLE)
        pose_ms = min(stamp, 1100)
        if not scan.poses or scan.poses[-1][0] < pose_ms:
            scan.poses.append((pose_ms, scan.pose, 0.))
        scan.source_frame = scan.current_frame = stamp
        scan.views.append((stamp, scan.pose, grid, ~grid))
        scan.side_frame = stamp if stamp >= 1150 else 0
        return P.Decision(P._twist(0., 0., -.3), P.AVOIDING, "observe")

    monkeypatch.setattr(P, "_decide", turn)
    for stamp in (1000, 1100, 1150, 1200):
        P.step(detections=None, depth=dict(control_ms=stamp, capture_ms=stamp),
               odom=None, state=state, config=cfg, dt=.05)
        if stamp == 1150:
            assert state.idle_for_s == 0.
    assert state.idle_for_s == pytest.approx(.05)


def test_reverse_travel_during_a_hold_cancels_subcell_progress(monkeypatch):
    from plugins.navi.odom import PlanarPose

    state = _state()
    cfg = _cfg(local_controller="dwa", idle_timeout_s=.55)
    i = 0

    def pulse(*args, **kwargs):
        return P.Decision(P._twist(0. if i % 2 else .20, 0., 0.), P.AVOIDING, "pulsed motion")

    monkeypatch.setattr(P, "_decide", pulse)
    for i in range(6):
        pose = PlanarPose(1000 + i * 100, ("body",), .03 * (i % 2), 0., 0.)
        decision = P.step(detections=None, depth=None, odom={"motion_pose": pose},
                          state=state, config=cfg, dt=.1)
    assert decision.status == P.FAILED


def test_unregistered_scan_restart_cannot_relabel_the_same_view(monkeypatch):
    from plugins.navi.active_scan import ANGLE

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.35)
    state = _state()
    scan = state.active_scan
    grid = np.ones((2, 2), dtype=bool)
    stamp = 0
    heading = 0.

    def observe(*args, **kwargs):
        if stamp == 1300:
            scan.invalidate()
        scan.pose = (0., 0., heading)
        scan.poses.append((stamp, scan.pose, 0.))
        scan.side_frame = scan.current_frame = scan.source_frame = stamp
        scan.views.append((stamp, scan.pose, grid, ~grid))
        return P.Decision(P._twist(0., 0., -.3), P.AVOIDING, "observe")

    monkeypatch.setattr(P, "_decide", observe)
    for stamp, heading in zip(range(1000, 1600, 100), (0., ANGLE, ANGLE, 0., ANGLE, ANGLE)):
        depth = dict(capture_ms=stamp, source_frame_id=stamp, control_ms=stamp)
        decision = P.step(detections=None, depth=depth, odom=None, config=cfg, state=state, dt=.1)
    assert decision.status == P.FAILED


@pytest.mark.parametrize("turns_corner", [False, True], ids=["returns-to-start", "turns-corner"])
def test_replanned_directions_require_net_body_displacement(monkeypatch, turns_corner):
    from plugins.navi.local_path import Detour
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.55, blocked_timeout_s=.15)
    scene = _depth(map_=_solid_depth(3.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    x = left = 0.
    vx = vy = 0.

    def avoid(*args, **kwargs):
        if vx or vy:
            state.checked_prefix = True
            state.view_detour = Detour(None, (), None, ((0., 0.), (vx, -vy)))
        return vx, vy, 0., P.AVOIDING, "detour"

    monkeypatch.setattr(P, "_avoid", avoid)
    for i in range(17):
        x += .1 * vx
        left += .1 * vy
        if i % 4 == 0:
            vx, vy = 0., .3
        elif i % 4 == 2:
            vx, vy = (.3, 0.) if turns_corner else (0., -.3)
        else:
            vx = vy = 0.
        pose = PlanarPose(1000 + 100*i, ("odom",), x, left, 0.)
        decision = _step(_detections(_obj()), scene, dict(obstacle_pose=pose), cfg, state)
        if decision.status == P.FAILED:
            break
    if turns_corner:
        assert decision.status == P.AVOIDING and max(x, left) > .10
    else:
        assert decision.status == P.FAILED and state.idle_for_s == pytest.approx(.6)


@pytest.mark.parametrize("stream", ["velocity", "mixed", "broken"])
def test_slow_measured_pose_and_velocity_share_one_progress_interval(stream):
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(max_feedback_gap_ms=250, local_controller="dwa", idle_timeout_s=.85)
    scene = _depth(map_=_solid_depth(5.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    history = []
    credits = []
    for i in range(16):
        stamp = 1000 + i*100
        velocity = (.25, 0., 0.) if stream != "broken" or i % 3 == 0 else None
        history.append((stamp, stamp, velocity))
        odom = dict(received_ms=stamp, source_sample_id=stamp, velocity_history=tuple(history))
        if velocity is not None:
            odom.update(zip(("vx", "vy", "wz"), velocity))
        if stream != "velocity":
            pose_i = i - i % 3
            odom["motion_pose"] = PlanarPose(1000 + 100*pose_i, ("odom",), .025*pose_i, 0., 0.)
        decision = _step(_detections(_obj()), scene, odom, cfg, state)
        if state.idle_for_s == 0.:
            credits.append(i)
        if decision.status == P.FAILED:
            break
    if stream == "broken":
        assert decision.status == P.FAILED and state.idle_for_s == pytest.approx(.9)
    else:
        assert decision.status == P.APPROACHING
        assert credits == list(range(2, 16, 2)), "each measured 5 cm earns a single credit"


@pytest.mark.parametrize("broken", [False, True])
@pytest.mark.parametrize("delay", [0, 1], ids=["current-pose", "delayed-pose"])
def test_fresh_absolute_pose_at_end_of_a_slow_tick_uses_measured_interval(broken, delay):
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.65)
    scene = _depth(map_=_solid_depth(5.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    history = []
    for i in range(11):
        stamp = 1000 + 100*i
        if not broken or i % 5 == 0:
            history.append((stamp, stamp, (.25, 0., 0.)))
        if i % 5 == 0:
            pose_i = max(0, i - delay)
            pose = PlanarPose(1000 + 100*pose_i, ("odom",), .025*pose_i, 0., 0.)
            odom = dict(motion_pose=pose, vx=.25, vy=0., wz=0., received_ms=stamp,
                        source_sample_id=stamp, velocity_history=tuple(history))
            decision = _step(_detections(_obj()), scene, odom, cfg, state, dt=.5 if i else .1)
            if decision.status == P.FAILED:
                break
    assert (decision.status == P.FAILED) == broken
    if not broken:
        assert state.idle_for_s == 0.


def test_delayed_absolute_pose_reconciles_only_its_measured_suffix():
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.85)
    scene = _depth(map_=_solid_depth(5.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    history = []
    anchor = PlanarPose(1000, ("odom",), 0., 0., 0.)
    for i, expected_x in enumerate((0., .025, .05, .065, .09)):
        stamp = 1000 + i*100
        history.append((stamp, stamp, (.25, 0., 0.)))
        if i == 3:
            # At 1300 we learn the measured pivot was at .04 at 1200.
            anchor = PlanarPose(1200, ("odom",), .04, 0., 0.)
        odom = dict(obstacle_pose=anchor, received_ms=stamp, source_sample_id=stamp,
                    vx=.25, vy=0., wz=0., velocity_history=tuple(history))
        _step(_detections(_obj()), scene, odom, cfg, state)
        actual = state.dwa_progress.pose
        assert actual.stamp_ms == stamp and actual.x == pytest.approx(expected_x)


@pytest.mark.parametrize("speed,covered", [(0., False), (0., True), (.25, True)],
                         ids=["stationary-gap", "stationary-covered", "moving-covered"])
def test_absolute_pose_dropout_requires_registered_travel(speed, covered):
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=.8)
    scene = _depth(map_=_solid_depth(5.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    history, credits = [], []
    previous_tick = 900
    for stamp in range(1000, 3401, 100):
        phase = (stamp - 1000) % 400
        if covered or phase in (0, 300):
            history.append((stamp, stamp, (speed, 0., 0.)))
        if phase not in (0, 300):
            continue
        odom = dict(vx=speed, vy=0., wz=0., received_ms=stamp, source_sample_id=stamp,
                    velocity_history=tuple(history))
        if phase == 0:
            x = 20. + speed * (stamp - 1000) / 1000
            odom["motion_pose"] = PlanarPose(stamp, ("odom",), x, 0., 0.)
        decision = _step(_detections(_obj()), scene, odom, cfg, state, dt=(stamp - previous_tick) / 1000)
        previous_tick = stamp
        if state.idle_for_s == 0.:
            credits.append(stamp)
        if decision.status == P.FAILED:
            break
    if speed:
        assert decision.status == P.APPROACHING
        assert credits == list(range(1300, 3401, 400))
        assert state.dwa_progress.pose.x == pytest.approx(20.6)
    else:
        assert decision.status == P.FAILED and not credits


def test_returning_world_fix_after_local_velocity_progress_only_rebaselines():
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(max_feedback_gap_ms=250, local_controller="dwa", idle_timeout_s=.8)
    scene = _depth(map_=_solid_depth(5.))
    state = seed(_state(), _detections(_obj()), scene, cfg)
    history, credits = [], []
    previous_tick = 900
    # Travel in the gap is unknown. The measured .05 m after it returns the
    # body to x=20 before the first new absolute fix.
    for stamp, speed, x in ((1000, 0., 20.), (1300, 0., None), (1400, .25, None),
                            (1500, .25, None), (1600, 0., 20.), (1700, .25, 20.025),
                            (1800, .25, 20.05)):
        history.append((stamp, stamp, (speed, 0., 0.)))
        odom = dict(vx=speed, vy=0., wz=0., received_ms=stamp, source_sample_id=stamp,
                    velocity_history=tuple(history))
        if x is not None:
            odom["motion_pose"] = PlanarPose(stamp, ("odom",), x, 0., 0.)
        decision = _step(_detections(_obj()), scene, odom, cfg, state, dt=(stamp - previous_tick) / 1000)
        assert decision.status == P.APPROACHING
        previous_tick = stamp
        if state.idle_for_s == 0.:
            credits.append(stamp)
    assert credits == [1500, 1800]


@pytest.mark.parametrize("feedback", ["absolute", "velocity"])
@pytest.mark.parametrize("continues", [False, True], ids=["two-subgoals", "new-passage"])
def test_measured_round_trips_only_credit_new_passage(monkeypatch, continues, feedback):
    from plugins.navi.local_path import Detour
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=10.)
    scene = Sim(range_m=3., config=cfg)
    state = seed(_state(), *scene.observe(), cfg)
    initial = PlanarPose(1000, ("odom",), 0., 0., 0.)
    anchor = state.goal_anchor = P.DetourGoal(3., 0., pose=initial, generation=state.tracker.track.generation)

    def avoid(*args, **kwargs):
        state.checked_prefix = True
        state.view_detour = Detour(None, (), None, ((0., 0.), (forward, left - subgoal)))
        return vx, vy, 0., P.AVOIDING, "detour"

    monkeypatch.setattr(P, "_avoid", avoid)
    credits = []
    for index in range(421):
        t = index / 10
        phase = math.tau * t / 4.
        left = .06 * (1. - math.cos(phase))
        vy = .06 * math.tau / 4. * math.sin(phase)
        subgoal = .12 if math.sin(phase + math.tau * .05 / 4.) > 0. else 0.
        # The positive twin leaves the explored strip around its first corner.
        x = .04 * max(0., t - 4.) if continues else 0.
        vx = .04 if continues and t >= 4. else 0.
        forward = .12 if vx else 0.
        pose = PlanarPose(1000 + 100 * index, initial.frame, x, left, 0.)
        odom = dict(motion_pose=pose, obstacle_pose=pose, vx=vx, vy=vy, wz=0.)
        if feedback == "velocity":
            # The mean measured twist covers precisely this 100 ms interval.
            odom = dict(received_ms=pose.stamp_ms, source_sample_id=pose.stamp_ms,
                        vx=(x - scene.pose[0]) / .1, vy=(left - scene.pose[1]) / .1, wz=0.)
        scene.pose[:2] = x, left
        decision = _step(*scene.observe(), odom, cfg, state)
        assert state.goal_anchor is anchor
        assert (anchor.x + x, anchor.y + left) == pytest.approx((3., 0.))
        if state.idle_for_s == 0.:
            credits.append(t)
        if decision.status == P.FAILED:
            break
    assert any(0. < stamp <= 2. for stamp in credits), "the first lateral detour must earn progress"
    if continues:
        assert decision.status == P.AVOIDING and credits[-1] > 40.
    else:
        assert decision.status == P.FAILED and "progress" in decision.reason
        assert decision.values == [0.] * 6 and credits[-1] + 10. <= t + .1
        assert t <= 18., "established four-second cycles must exhaust the ten-second idle budget"


def test_long_measured_route_retains_prior_visits_and_can_exit_at_a_corner():
    from plugins.navi.odom import PlanarPose

    cfg, state = _cfg(local_controller="dwa", idle_timeout_s=10.), _state()
    decision = P.Decision(P._twist(.3, 0., 0.), P.AVOIDING, "detour")
    stamp = 1000

    def move(x, y, direction):
        nonlocal stamp
        stamp += 200
        pose = PlanarPose(stamp, ("odom",), x, y, 0.)
        route = ((0., 0.), direction)
        result = P._account_dwa_progress(decision, state, cfg, {"motion_pose": pose}, None, route, .2)
        assert result.status == P.AVOIDING
        return state.idle_for_s == 0.

    # More than a thousand distinct checkpoints along a continuous 75 m route.
    for index in range(1201):
        assert move(-index / 16, 0., (-1., 0.)) is (index > 0)
    # A confirmed absolute fix returns to the early part of the same frame
    # after an uncovered gap. No part of a known segment becomes a first visit.
    stamp += 1000
    for index in range(12):
        assert not move(-index / 16, 0., (-1., 0.))
    for index in range(1, 12):
        assert move(-11 / 16, index / 16, (0., -1.))


def test_new_velocity_frame_after_uncovered_gap_can_measure_its_own_travel():
    cfg, state = _cfg(local_controller="dwa", idle_timeout_s=.8), _state()
    decision = P.Decision(P._twist(.25, 0., 0.), P.APPROACHING, "approach")
    credits = []
    for stamp in (*range(1000, 4000, 100), *range(4300, 5500, 100)):
        odom = dict(received_ms=stamp, source_sample_id=stamp, vx=.25, vy=0., wz=0.)
        elapsed = .4 if stamp == 4300 else .1
        result = P._account_dwa_progress(decision, state, cfg, odom, None, ((0., 0.), (1., 0.)), elapsed)
        assert result.status == P.APPROACHING
        if not state.idle_for_s and stamp >= 4300:
            credits.append(stamp)
    assert credits == list(range(4500, 5500, 200))


def test_continuous_scan_keeps_registered_passage_when_absolute_pose_is_late():
    from plugins.navi.odom import PlanarPose

    cfg, state = _cfg(local_controller="dwa"), _state()
    decision = P.Decision(P._twist(.25, 0., 0.), P.AVOIDING, "detour")
    scan = state.active_scan
    credits = []
    for index, (f, r, route) in enumerate(((0., 0., (1., 0.)), (.0625, 0., (1., 0.)),
                                           (0., 0., (-1., 0.)), (0., .0625, (0., 1.)))):
        stamp = 1000 + index * 100
        scan.pose = (f, r, 0.)
        scan.poses.append((stamp, scan.pose, 0.))
        odom = {}
        if index < 2:
            odom["obstacle_pose"] = PlanarPose(stamp, ("odom",), 10. + r, 20. + f, math.pi / 2)
        P._account_dwa_progress(decision, state, cfg, odom, None, ((0., 0.), route), .1)
        if state.idle_for_s == 0.:
            credits.append(index)
    assert credits == [1, 3]


@pytest.mark.parametrize("return_motion", ["direct", "approach-before-detour", "stationary-changing-target",
                                           "coasting", "missing-feedback", "replacement", "interleaved-detector"])
def test_native_return_to_observed_target_on_a_previously_traversed_strip(monkeypatch, return_motion):
    from plugins.navi.local_path import Detour
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=10., vx_max=.15625,
               half_fov_rad=math.radians(89.9), max_coast_s=20.)
    scene = Sim(range_m=2., config=cfg)
    scene.target[1] = .01
    scene.pose[:] = (.00625, 0., math.pi / 2)
    if return_motion == "approach-before-detour":
        scene.pose[:] = (-.18125, 0., 0.)
    state = seed(_state(), *scene.observe(), cfg)
    original = expected_track = state.tracker.track
    stamp = 1000
    detector_frame = None

    def tick(vx, vy, wz, *, returning=False):
        nonlocal stamp, detector_frame
        stamp += 100
        pose = PlanarPose(stamp, ("odom",), *scene.pose)
        odom = dict(motion_pose=pose, obstacle_pose=pose, vx=vx, vy=vy, wz=wz,
                    received_ms=stamp, source_sample_id=stamp)
        if returning and return_motion == "missing-feedback":
            odom = {}
        objects, depth = scene.observe()
        fresh = True
        if returning and return_motion == "interleaved-detector":
            fresh = detector_frame is None or stamp % 400 == 100
            if fresh:
                detector_frame = objects
            objects = detector_frame
        result = _step(objects, depth, odom, cfg, state)
        assert state.tracker.track is expected_track and original.generation == 1
        if not returning or return_motion != "coasting":
            assert expected_track.observed and expected_track.accepted_detection == fresh
            if fresh:
                assert expected_track.measured_range_m == pytest.approx(scene.relative()[0], abs=.001)
        return result

    if return_motion == "approach-before-detour":
        previous = [0.] * 6
        for _ in range(12):
            result = tick(previous[0], previous[1], previous[5])
            assert result.status == P.APPROACHING
            scene.apply(result.values, .1)
            previous = result.values

    with monkeypatch.context() as seam:
        def depart(*args, **kwargs):
            state.checked_prefix = True
            state.view_detour = Detour(None, (), None, ((0., 0.), (0., -4.)))
            return 0., vy, wz, P.AVOIDING, "detour leg"

        seam.setattr(P, "_avoid", depart)
        if return_motion == "approach-before-detour":
            vy, wz = 0., math.pi / 8
            for index in range(1, 41):
                scene.pose[2] = math.pi / 2 * index / 40
                assert tick(0., vy, wz).status == P.AVOIDING
        vy, wz = .15625, 0.
        for index in range(193):
            scene.pose[:] = (.00625 - index / 64., 0., math.pi / 2)
            assert tick(0., vy, wz).status == P.AVOIDING
        vy, wz = 0., -math.pi / 8
        for index in range(1, 41):
            scene.pose[2] = math.pi / 2 * (1. - index / 40)
            assert tick(0., vy, wz).status == P.AVOIDING

    # The entire return uses the native policy. Only the feedback/observation
    # condition named by a negative twin changes.
    scene.visible = return_motion != "coasting"
    initial_distance = scene.relative()[0]
    previous = [0.] * 6
    for index in range(194):
        if return_motion == "replacement" and index == 16:
            # Lose the old track and genuinely confirm another chair. Its
            # measurements cannot extend the original target's return credit.
            state.tracker.step(detections=None, depth=None, target=state.target, config=cfg,
                               ego=(0., 0., 0.), dt=cfg.max_coast_s + .01)
            scene.target[0] -= .125
            for _ in range(cfg.confirm_hits):
                objects, depth = scene.observe()
                state.tracker.step(detections=objects, depth=depth, target=state.target, config=cfg,
                                   ego=(0., 0., 0.), dt=.1)
            expected_track = state.tracker.track
            assert expected_track is not original and expected_track.generation == 2
        result = tick(previous[0], previous[1], previous[5], returning=True)
        if result.status == P.FAILED:
            break
        assert result.status == P.APPROACHING and result.values[0] > 0.
        if return_motion == "stationary-changing-target":
            scene.target[0] -= .015625
            scene.target[1] = .01 + .005 * (-1) ** index
            previous = [0.] * 6
        else:
            scene.apply(result.values, .1)
            previous = result.values
    if return_motion in ("direct", "approach-before-detour", "interleaved-detector"):
        assert result.status == P.APPROACHING and index == 193
        assert initial_distance - scene.relative()[0] == pytest.approx(194 / 64.)
        assert state.idle_for_s < .5
    else:
        assert result.status == P.FAILED and "progress" in result.reason
        assert result.values == [0.] * 6 and state.idle_for_s >= cfg.idle_timeout_s
        if return_motion == "stationary-changing-target":
            assert initial_distance - scene.relative()[0] > .9
        if return_motion == "coasting":
            assert not original.observed and original.measured_range_m is None


def test_alternating_direct_and_detour_legs_cannot_recredit_the_same_closure(monkeypatch):
    from plugins.navi.local_path import Detour
    from plugins.navi.odom import PlanarPose

    cfg = _cfg(local_controller="dwa", idle_timeout_s=10., vx_max=.15625)
    scene = Sim(range_m=3., config=cfg)
    state = seed(_state(), *scene.observe(), cfg)
    original = state.tracker.track
    native = P._avoid

    def avoid(*args, **kwargs):
        if returning:
            return native(*args, **kwargs)
        state.checked_prefix = True
        state.view_detour = Detour(None, (), None, ((0., 0.), (-.125, 0.)))
        return -.125, 0., 0., P.AVOIDING, "local detour"

    monkeypatch.setattr(P, "_avoid", avoid)
    direct_ticks = 0
    for index in range(421):
        t = index / 10
        phase = t * math.tau / 4.
        scene.pose[0] = -.0625 * (1. - math.cos(phase))
        vx = -.0625 * math.tau / 4. * math.sin(phase)
        returning = math.sin(phase + .05 * math.tau / 4.) < 0.
        pose = PlanarPose(1000 + 100 * index, ("odom",), *scene.pose)
        result = _step(*scene.observe(), dict(motion_pose=pose, vx=vx, vy=0., wz=0.), cfg, state)
        assert state.tracker.track is original and original.observed and original.accepted_detection
        if result.status == P.FAILED:
            break
        if returning:
            assert result.status == P.APPROACHING and result.values[0] > 0.
            direct_ticks += 1
        else:
            assert result.status == P.AVOIDING
    assert direct_ticks > 20
    assert result.status == P.FAILED and "progress" in result.reason and t < 22.


@pytest.mark.parametrize("evidence", ["target-closes-alone", "body-follows-receding-target"])
def test_revisited_motion_needs_body_and_observed_target_closure(evidence):
    from plugins.navi.odom import PlanarPose

    cfg, state = _cfg(local_controller="dwa", idle_timeout_s=.8), _state()
    scene = Sim(range_m=4., config=cfg)
    scene.target[1] = 1. if evidence == "target-closes-alone" else -.5
    seed(state, *scene.observe(), cfg)
    original = state.tracker.track
    stamp = 1000

    def sample(left, vy, status):
        nonlocal stamp
        stamp += 100
        scene.pose[1] = left
        objects, depth = scene.observe()
        state.tracker.step(detections=objects, depth=depth, target=state.target, config=cfg,
                           ego=(0., vy, 0.), dt=.1)
        assert state.tracker.track is original and P.current_target_observed(state)
        pose = PlanarPose(stamp, ("odom",), 0., left, 0.)
        route = ((0., 0.), (0., -math.copysign(1., vy)))
        return P._account_dwa_progress(P.Decision(P._twist(0., vy, 0.), status, "rollout"),
                                       state, cfg, {"motion_pose": pose}, depth, route, .1)

    for index in range(17):
        assert sample(index / 16, .625, P.AVOIDING).status == P.AVOIDING
    # For both returns the pivot follows a checked route through old cells.
    # One has only a closer detector observation. In the other the pivot
    # approaches the earlier target point, but the real target retreats.
    start_range = original.measured_range_m
    for index in range(17):
        if evidence == "target-closes-alone":
            scene.target[0] = 4. - index / 16
        else:
            scene.target[1] = -.5 - index / 16
        result = sample(1. - index / 16, -.625, P.APPROACHING)
        assert state.idle_for_s > 0.
        if result.status == P.FAILED:
            break
    if evidence == "target-closes-alone":
        assert original.measured_range_m < start_range - .25
    else:
        assert original.measured_range_m == pytest.approx(start_range, abs=.001)
    assert result.status == P.FAILED and result.values == [0.] * 6


@pytest.mark.parametrize("admission", ["checked-dwa", "unselected-reference", "changed-command", "repeated-closure"])
def test_checked_dwa_return_can_progress_through_a_previously_visited_corridor(admission):
    from plugins.navi import dwa, local_path
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.odom import PlanarPose

    geometry = local_path.camera_square(HorizontalIntrinsics(160, 85., 80.),
                                        (120, 72., 60.), .20, at_front=True)
    cfg = _cfg(local_controller="dwa", local_planning=True, local_geometry=geometry,
               half_width_m=.20, vx_max=.15625, use_lateral=False, vy_max=0., idle_timeout_s=.8)
    scene = Sim(range_m=3., config=cfg)
    state = seed(_state(motion_source="measured"), *scene.observe(), cfg)
    original = state.tracker.track
    stamp = 1000

    def sample(x, vx, returning):
        nonlocal stamp
        stamp += 100
        scene.pose[0] = x
        objects, _ = scene.observe()
        pose = PlanarPose(stamp, ("odom",), x, 0., 0.)
        depth = dict(map=np.full((120, 160), scene.relative()[0], dtype=np.float32),
                     frame_id=stamp, source_frame_id=stamp, capture_ms=stamp,
                     control_ms=stamp, obstacle_capture_pose=pose)
        state.tracker.step(detections=objects, depth=depth, target=state.target,
                           config=cfg, ego=(vx, 0., 0.), dt=.1)
        assert state.tracker.track is original and P.current_target_observed(state)
        odom = dict(obstacle_pose=pose, vx=vx, vy=0., wz=0.,
                    received_ms=stamp, source_sample_id=stamp)
        state.dwa_odom = odom
        state.active_scan.update(depth, odom, geometry, receipt_only=True)
        route = ((0., 0.), (1. if returning else -1., 0.))
        if returning:
            # Run the real physical selector and admission, not a fabricated
            # AVOIDING label. Only the reference path is supplied by the fixture.
            state.dwa_trace = {}
            state.dwa_budget = dwa.CandidateBudget(64)
            state.dwa_projection = local_path.DepthProjection(depth["map"], None)
            reference = local_path.Detour(None, route, geometry)
            proposal = P._dwa_detour(depth, cfg, state, reference, scene.relative()[0], 0.,
                                    candidate_commands=((.15625, 0., 0.),))
            assert proposal is not None and proposal.guard.check.accepted
            values = P._use_dwa(state, proposal, 0.)
            decision = P.Decision(P._twist(*values[:3]), values[3], values[4])
            assert state.dwa_selected and state.checked_prefix and decision.values[0] > 0.
            if admission == "unselected-reference":
                state.dwa_selected = False
            elif admission == "changed-command":
                decision.values[0] *= .5
        else:
            decision = P.Decision(P._twist(vx, 0., 0.), P.AVOIDING, "measured outbound leg")
        result = P._account_dwa_progress(decision, state, cfg, odom, depth, route, .1)
        state.commanded = tuple(decision.values[i] for i in (0, 1, 5))
        return result

    for index in range(33):
        assert sample(-index / 64., -.15625, False).status == P.AVOIDING
    initial_range = original.measured_range_m
    credits = []
    previous = -.5
    for index in range(1, 97 if admission == "repeated-closure" else 33):
        offset = min(index % 16, 16 - index % 16) if admission == "repeated-closure" else index
        current = -.5 + offset / 64.
        result = sample(current, (current - previous) / .1, True)
        previous = current
        if state.idle_for_s == 0.:
            credits.append(index)
        if result.status == P.FAILED:
            break
    if admission == "checked-dwa":
        assert result.status == P.AVOIDING and index == 32
        assert initial_range - original.measured_range_m == pytest.approx(.5, abs=.001)
        assert state.idle_for_s < .5
    else:
        assert result.status == P.FAILED and "progress" in result.reason
        assert result.values == [0.] * 6
        if admission == "repeated-closure":
            assert credits and index > 8
            assert index <= 24, "the already credited approach cannot renew repeated local cycles"


@pytest.mark.parametrize("pairing", ["precise", "receipt-lag", "ordinary", "missing", "outdated",
                                     "unregistered-capture", "clock_epoch", "source_generation", "segment_id", "calibration_id"])
def test_pose_only_return_uses_capture_paired_body_measurements(monkeypatch, pairing):
    from plugins.navi import odom as O, local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315., 320.), (480, 420., 240.), .20)
    cfg = _cfg(local_controller="dwa", local_planning=True, local_geometry=geometry,
               idle_timeout_s=1., vx_max=.15625)
    scene = Sim(range_m=2., config=cfg)
    state = seed(_state(), *scene.observe(), cfg)
    track = state.tracker.track
    stamp, prior = 1000, None
    reset = pairing in ("clock_epoch", "source_generation", "segment_id", "calibration_id")

    def sample(return_index=None):
        nonlocal stamp, prior
        stamp += 100
        receipt = stamp + (17 if pairing == "receipt-lag" else 0)
        changed = reset and return_index is not None and return_index >= 4
        epoch = 2 if changed and pairing == "clock_epoch" else 1
        generation = 2 if changed and pairing == "source_generation" else 1
        slam = dict(segment_id=2 if changed and pairing == "segment_id" else 1)
        if pairing != "ordinary":
            slam["measurement"] = dict(clock_epoch=epoch, source_generation=generation,
                                       stamp_ns=stamp * 10**6, frame_id="odom")
        x, y, yaw = scene.pose
        payload = dict(schema=O.SCHEMA, stamp_ms=stamp,
                       pose=dict(stamp_ms=stamp, frame_id="odom", child_frame_id="base_link",
                                 position=[float(x), float(y), 0.],
                                 quaternion=[0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]),
                       vendor=dict(motus_slam=slam, calibration_id=2 if changed and pairing == "calibration_id" else 1))
        pose = O.planar_pose(payload, stamp)
        _, capture = O.measured_obstacle_pose(payload, {})
        previous = prior or O.PlanarPose(stamp - 100, pose.frame, *scene.pose)
        prior = pose
        odom = dict(received_ms=receipt, source_sample_id=stamp, motion_pose=pose,
                    pose_feedback=pose.feedback_since(previous, 150), pose_feedback_since_ms=previous.stamp_ms)
        if return_index is None or pairing not in ("missing", "outdated"):
            odom["obstacle_pose"] = capture
        elif pairing == "outdated":
            odom["obstacle_pose"] = last_outbound
        objects, depth = scene.observe()
        if pairing == "unregistered-capture" and return_index == 2:
            # The earlier depth capture belongs to a different SLAM epoch.
            earlier = stamp - 10
            payload["pose"]["stamp_ms"] = earlier
            slam["measurement"].update(clock_epoch=0, stamp_ns=earlier * 10**6)
            _, capture = O.measured_obstacle_pose(payload, {})
        depth.update(frame_id=stamp, capture_ms=capture.stamp_ms, control_ms=receipt, obstacle_capture_pose=capture)
        result = _step(objects, depth, odom, cfg, state)
        assert state.tracker.track is track and track.generation == 1 and P.current_target_observed(state)
        assert track.measured_range_m == pytest.approx(scene.relative()[0], abs=.001)
        return result, capture

    with monkeypatch.context() as seam:
        def depart(*args, **kwargs):
            state.checked_prefix = True
            state.view_detour = local_path.Detour(None, (), None, ((0., 0.), (-4., 0.)))
            return -.15625, 0., 0., P.AVOIDING, "checked detour"

        seam.setattr(P, "_avoid", depart)
        for i in range(66):
            scene.pose[0] = .00625 - i / 64.
            result, last_outbound = sample()
            assert result.status == P.AVOIDING

    start = scene.pose[0]
    for i in range(65):
        assert scene.pose[0] == pytest.approx(start + i / 64.)
        result, capture = sample(i)
        if pairing == "unregistered-capture":
            if i == 2:
                assert state.dwa_progress.approach_fix is None
            if i == 4:
                assert state.dwa_progress.credited_range is None
        if reset and i == 3:
            fix = state.dwa_progress.approach_fix
            assert fix is not None and fix.frame == capture.frame
            assert state.dwa_progress.credited_range is None
        if reset and i == 4:
            # This step completes the first 5 cm. A fix from the previous
            # clock, generation, segment or calibration cannot certify it.
            assert capture.frame != fix.frame and state.dwa_progress.credited_range is None
            assert state.dwa_progress.approach_fix is None or state.dwa_progress.approach_fix.frame == capture.frame
            assert state.idle_for_s > 0.
            break
        if result.status == P.FAILED:
            break
        assert result.status == P.APPROACHING and result.values[0] == pytest.approx(.15625)
        scene.apply(result.values, .1)
    if reset:
        assert i == 4
    elif pairing in ("missing", "outdated"):
        assert result.status == P.FAILED and "progress" in result.reason
        assert result.values == [0.] * 6 and state.dwa_progress.credited_range is None
    else:
        assert i == 64 and result.status == P.APPROACHING and state.idle_for_s < .5
        assert state.dwa_progress.approach_fix.frame == capture.frame


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("observed", [True, False], ids=["visible", "lost"])
def test_person_approach_faces_the_last_visible_side_on_a_clear_floor(monkeypatch, observed, side):
    from types import SimpleNamespace

    geometry = P.local_path.camera_square(P.HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .2, at_front=True)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.2, floor_wz=.2)
    state = _state("person", motion_source="measured", last_seen_side=side, commanded=(.2, 0., 0.))
    track = SimpleNamespace(observed=observed, range_known=True, range_m=2., measured_range_m=2. if observed else None,
                            bearing_rad=side * .28 * config.half_fov_rad)
    monkeypatch.setattr(state.tracker, "step", lambda **kw: track)
    depth = dict(_scene(3.), frame_id=1000, control_ms=1000)
    decision = P.step(detections=_detections(*([_obj("person", side * .28)] if observed else [])), depth=depth,
                      odom=dict(vx=0., vy=0., wz=0., received_ms=1000), config=config, state=state, dt=.1)
    if observed:
        assert side * decision.values[5] < 0. and decision.values[:2] == [0., 0.]
    else:
        assert decision.values[:2] == [0., 0.]


def test_incomplete_route_search_does_not_count_as_an_obstructed_view(monkeypatch):
    config = _cfg(local_planning=True)
    state = _state()
    scan = state.active_scan
    scan.current_frame = 1000
    monkeypatch.setattr(scan, "new_dwa_view", lambda: True)

    def out_of_time(*args):
        state.dwa_trace["fallback_reason"] = "search_deadline"

    monkeypatch.setattr(P, "_dwa_route_motion", out_of_time)
    P._dwa_recovery_route({}, config, state, None, .8, 2., 0.)
    assert scan.dwa_failures == 0


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("forward", [.3, 0.], ids=["moving", "approach-stopped"])
def test_visible_goal_with_clear_path_checks_an_alternative_to_blocked_sweep(monkeypatch, accepted, forward):
    from types import SimpleNamespace

    grid = P.local_path
    geometry = grid.camera_square(P.HorizontalIntrinsics(8, 6, 4), (8, 6, 4), .20, at_front=True)
    config = _cfg(local_planning=True, local_geometry=geometry, half_width_m=.20)
    state = _state(motion_source="measured")
    state.tracker.track = SimpleNamespace(observed=True, accepted_detection=False)
    scan = state.active_scan
    scan.receipt_only, scan.current_frame = True, 1000
    occupied = np.isclose(grid.FORWARD, 1.) & np.isclose(grid.RIGHT, .45)
    scan.views.append((1000, scan.pose, ~occupied, occupied))
    depth = dict(_scene(3.), frame_id=1000, ground=occupied)
    monkeypatch.setattr(P, "_clearance", lambda *args, **kwargs: (.7, 1.))
    monkeypatch.setattr(P, "_target_corridor_clear", lambda *args: False)
    checked = (.12, 0., 0., P.AVOIDING, "checked direct prefix")
    calls = []

    def direct(view, cfg, current, route, distance, bearing, *, branch, speed_limit, recovery):
        calls.append((branch, speed_limit, distance, bearing, recovery))
        return checked if accepted else None

    monkeypatch.setattr(P, "_dwa_detour", direct)
    monkeypatch.setattr(P, "_use_dwa", lambda current, proposal, side: proposal)
    result = P._avoid(forward, .03 if forward else 0., -.2, depth, config, P.APPROACHING, "tracking", state,
                      target_distance=2.4, target_bearing=.04, may_scan=True, desired_forward=.3)
    assert calls == [("direct", .3, 2.4, .04, True)]
    assert result[:3] == (checked[:3] if accepted else (0., 0., 0.))
