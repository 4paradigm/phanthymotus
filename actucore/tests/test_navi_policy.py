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


def _step(detections=None, depth=None, odom=None, config=None, state=None,
          dt=0.1):
    return P.step(detections=detections, depth=depth, odom=odom,
                  config=config or _cfg(), state=state or _state(), dt=dt)


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


def test_it_turns_in_place_before_driving():
    """Driving while badly misaligned traces an arc into whatever is beside
    the target."""
    decision = _step(detections=_detections(_obj(x=0.5)), depth=_depth())
    assert decision.values[0] == 0.0
    assert decision.status == P.ALIGNING


def test_an_aligned_target_is_approached():
    decision = _step(detections=_detections(_obj(x=0.01)), depth=_depth())
    assert decision.values[0] > 0
    assert decision.status == P.APPROACHING


def test_vy_is_always_zero():
    """The depth map says nothing about what is beside the robot, so
    sidestepping is moving blind. The axis stays open in the descriptor for a
    future policy with a wider sensor."""
    for x in (-0.5, 0.0, 0.5):
        assert _step(detections=_detections(_obj(x=x)), depth=_depth()).values[1] == 0.0


def test_the_twist_is_six_wide_in_control_order():
    values = _step(detections=_detections(_obj(x=0.3)), depth=_depth()).values
    assert len(values) == 6


# ── arriving ─────────────────────────────────────────────────────────────────

def test_arriving_emits_one_explicit_zero_then_goes_quiet():
    """Arriving is a success and should stop the chassis on a command, not on a
    watchdog timeout — the latter reads as a dropped link in the driver's log."""
    state = _state()
    depth = _depth(map_=_solid_depth(0.8))
    first = _step(detections=_detections(_obj(x=0.0, bbox=(0.4, 0.4, 0.6, 0.6))),
                  depth=depth, state=state)
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
    assert "搜索" in decision.reason


def test_seeing_the_target_again_resets_the_search():
    state = _state(missing_frames=20, searching_for_s=5.0, searched_rad=3.0)
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
    assert decision.status == P.STUCK
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
