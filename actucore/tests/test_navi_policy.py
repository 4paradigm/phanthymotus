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

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import policy as P  # noqa: E402


def _cfg(**over):
    return P.Config(**over)


def _obj(name="chair", x=0.0, y=0.0, confidence=0.9, bbox=None):
    out = {"name": name, "position": [x, y], "confidence": confidence}
    if bbox:
        out["bbox_norm"] = bbox
    return out


def _detections(*objects):
    return {"timestamp": 1.0, "count": len(objects), "objects": list(objects)}


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
        state.tracker.step(detections=detections, depth=depth,
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


def test_strafing_needs_the_side_it_moves_into_to_be_known_and_clear():
    """Sideways is the direction a forward-facing depth map knows least about,
    so an unknown band **blocks** rather than defaulting to permission. This is
    the one place the policy could move into space it cannot see, and it does
    not take it."""
    blocked = _depth(bands={"left": 5.0, "center": 5.0, "right": 0.9})
    unknown = _depth(bands={"left": 5.0, "center": 5.0})
    for bands in (blocked, unknown):
        decision = _step(detections=_detections(_obj(x=0.5)), depth=bands)
        assert decision.values[1] == 0.0
        assert decision.values[5] < 0, "it still turns towards the target"


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


# ── obstacles ────────────────────────────────────────────────────────────────

# These pass a depth *map* (which is where the target's own distance comes
# from) alongside explicitly supplied *bands* (which is where obstacles come
# from). On a robot the two are computed from the same frame and agree; feeding
# them separately here is what isolates the avoidance branch from the arrival
# branch, which would otherwise both fire on the same number.

def test_a_near_obstacle_ahead_slows_the_approach():
    target_far = _solid_depth(4.0)
    far = _step(detections=_detections(_obj(x=0.0)),
                depth=_depth({"left": 5.0, "center": 5.0, "right": 5.0},
                             map_=target_far))
    near = _step(detections=_detections(_obj(x=0.0)),
                 depth=_depth({"left": 5.0, "center": 1.0, "right": 5.0},
                              map_=target_far))
    assert 0 < near.values[0] < far.values[0]
    assert near.status == P.AVOIDING


def test_an_obstacle_inside_the_stop_distance_halts_forward_motion():
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth({"left": 5.0, "center": 0.4, "right": 1.0},
                                  map_=_solid_depth(4.0)))
    assert decision.values[0] == 0.0
    assert decision.status == P.AVOIDING


def test_it_turns_towards_the_freer_side():
    target_far = _solid_depth(4.0)
    left_free = _step(detections=_detections(_obj(x=0.0)),
                      depth=_depth({"left": 5.0, "center": 0.4, "right": 1.0},
                                   map_=target_far))
    right_free = _step(detections=_detections(_obj(x=0.0)),
                       depth=_depth({"left": 1.0, "center": 0.4, "right": 5.0},
                                    map_=target_far))
    assert left_free.values[5] > 0      # counter-clockwise, towards the left
    assert right_free.values[5] < 0


def test_the_summary_only_path_cannot_separate_target_from_obstacle():
    """The documented cost of wiring only the depth summary.

    With no map, the target's distance is the nearest reading in its band —
    which is the nearest *obstacle* there, not the target. When something is
    inside the stop distance the card therefore reports arrival rather than
    avoidance. Pinned so the degradation is a known property rather than a
    surprise, and so `info()`'s warning about it keeps earning its place.
    """
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth({"left": 5.0, "center": 0.4, "right": 5.0}))
    assert decision.status == P.ARRIVED


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
                         depth=_depth(map_=_solid_depth(0.8)),
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
    """A search that commands less than the robot can execute leaves the card
    reporting "searching" beside a motionless robot — seen on r1_sz the moment
    deadband handling went in, with search_rate 0.4 against R1's 1.0 rad/s."""
    rate = P.Config().search_rate
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
    config = _cfg()
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
    config = _cfg()
    P.adopt_limits(config, {"limits": {"lower": [-1.0] * 6, "upper": [1.0] * 6}})
    assert (config.floor_vx, config.floor_vy, config.floor_wz) == (0.0, 0.0, 0.0)
    values = _step(detections=_detections(_obj(x=0.02)),
                   depth=_depth(map_=_solid_depth(1.3)), config=config).values
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
    config = _cfg()
    state = seed(_state(), _detections(_obj(x=0.5)), _depth(), config)
    P.step(detections=_detections(_obj(x=0.5)), depth=_depth(), odom=None,
           config=config, state=state, dt=0.1)
    assert state.yaw_gate.on

    for _ in range(10):
        decision = P.step(detections=_detections(_obj(x=0.0)), depth=_depth(),
                          odom=None, config=config, state=state, dt=0.1)
    assert decision.values[5] == 0.0
    assert state.yaw_gate.on is False


def test_an_obstacle_ahead_is_stepped_around_not_only_turned_from():
    """Turning alone changes where the robot points; on a base that can
    translate, the sidestep is what gets it out of the way — and doing both at
    once is one motion instead of a pirouette followed by a walk."""
    # The map puts the target 4 m off; the bands put something in the way. A
    # single source cannot express that — the target would *be* the obstacle.
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth(bands={"left": 5.0, "center": 0.5, "right": 1.0},
                                  map_=_solid_depth(4.0)))
    assert decision.status == P.AVOIDING
    assert decision.values[0] == 0.0, "no forward motion into it"
    assert decision.values[1] > 0, "stepping left, the side with room"
    assert decision.values[5] > 0, "and turning that way too"


def test_an_obstacle_with_no_room_either_side_is_not_stepped_into():
    decision = _step(detections=_detections(_obj(x=0.0)),
                     depth=_depth(bands={"left": 1.0, "center": 0.5, "right": 0.9},
                                  map_=_solid_depth(4.0)))
    assert decision.values[1] == 0.0
