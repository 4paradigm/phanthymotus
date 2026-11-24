"""Exercise route tracking at the wire with measured, frame-labelled body poses."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from plugins.navi import dwa, policy
from plugins.navi.odom import PlanarPose
from test_navi_dwa_integration import card_for_case, observation_case
from test_navi_dwa_publish import advance_snapshot, observation_card


def route_card(monkeypatch, *, advance=True):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    pose = PlanarPose(1000, ("odom",), .2, .1, .1)
    _, depth, odom, _ = snapshot[0]
    depth.update(obstacle_capture_pose=pose)
    odom.update(motion_pose=pose, obstacle_pose=pose)
    message = card.next_command()
    assert card._stage_command(message, card._output_epoch)
    assert packets[0]["values"][0] > 0
    if not advance:
        return card, packets, clock, snapshot
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    pose = PlanarPose(1100, ("odom",), .21, .1, .1)
    depth.update(obstacle_capture_pose=pose)
    odom.update(motion_pose=pose, obstacle_pose=pose, velocity_history=(
        (1000, 1000, (.10, 0., 0.)), (1100, 1100, (.10, 0., 0.))))
    return card, packets, clock, snapshot


@pytest.mark.parametrize("action", ["pause", "cancel"])
def test_operator_interrupts_inflight_route_tracking(monkeypatch, action):
    card, packets, _, _ = route_card(monkeypatch)
    entered, resume = Event(), Event()
    calculate = dwa.track_route

    def slow_track(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return calculate(*args, **kwargs)

    monkeypatch.setattr(dwa, "track_route", slow_track)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tracking = pool.submit(card._dispatch_command)
        try:
            assert entered.wait(2)
            stop = pool.submit(card._halt, True) if action == "pause" else pool.submit(card._stop_navigation)
            stop.result(timeout=2)
            assert not any(packets[-1]["values"])
        finally:
            resume.set()
        tracking.result(timeout=2)
    assert len(packets) == 2


@pytest.mark.parametrize("change", ["frame_reset", "feedback_lost", "unbridged_feedback"])
def test_unregistered_or_interrupted_route_cannot_move_the_wire(monkeypatch, change):
    card, packets, clock, snapshot = route_card(monkeypatch)
    objects, depth, odom, receipt = snapshot[0]
    if change == "frame_reset":
        pose = PlanarPose(1100, ("new odom",), 4., 0., 0.)
        depth.update(obstacle_capture_pose=pose)
        odom.update(motion_pose=pose, obstacle_pose=pose)
    elif change == "unbridged_feedback":
        # A speed sample after the gap cannot carry the last registered pose
        # across the unobserved interval.
        advance_snapshot(snapshot, clock, 1400)
        objects, depth, odom, _ = snapshot[0]
        del depth["obstacle_capture_pose"], odom["obstacle_pose"], odom["motion_pose"]
        odom["velocity_history"] = ((1000, 1000, (.10, 0., 0.)), (1400, 1400, (.10, 0., 0.)))
        snapshot[0] = dict(objects, timestamp=1.4), depth, odom, 1400
    else:
        snapshot[0] = objects, depth, None, receipt
    card._dispatch_command()
    clock[0] = 1500
    card._dispatch_command()
    assert len(packets) == 1
    assert card._scheduled_command is None


def test_expired_new_plan_leaves_current_route_with_controller(monkeypatch):
    card, packets, clock, _ = route_card(monkeypatch)
    active = card._scheduled_command
    proposal = card.next_command()
    assert any(proposal["values"])
    proposal["ttl_ms"] = 0
    assert card._stage_command(proposal, card._output_epoch)
    assert card._scheduled_command is active
    assert "上一条路线" in card._last_decision.reason
    assert card._state.commanded == card._wire_twist
    assert card._state.commanded_vx == packets[-1]["values"][0]
    assert len(packets) == 1
    # An explicit stop from the planner still interrupts this same route.
    stop = card._command_message([0.] * 6, clock[0] / 1000, clock[0])
    assert card._stage_command(stop, card._output_epoch)
    assert not any(packets[-1]["values"])


@pytest.mark.parametrize("reason", ["publish_expired", "stale_observation", "no_feedback", "superseded"])
def test_rejected_replacement_does_not_zero_the_active_route(monkeypatch, reason):
    card, packets, _, _ = route_card(monkeypatch, advance=False)
    current = card._scheduled_command
    proposal = card.next_command()
    assert any(proposal["values"])
    monkeypatch.setattr(card, "_recheck_dwa_inputs", lambda *args: (reason, None))
    assert card._stage_command(proposal, card._output_epoch)
    assert card._scheduled_command is current
    assert len(packets) == 1


def test_route_expiry_does_not_emit_a_zero_or_renew_the_packet(monkeypatch):
    card, packets, clock, _ = route_card(monkeypatch, advance=False)
    clock[0] = card._scheduled_command.expires_ms
    card._dispatch_command()
    card._dispatch_command()
    assert len(packets) == 1 and packets[0]["values"][0] > 0
    assert card._scheduled_command is None


def test_no_planner_result_leaves_the_valid_route_with_the_controller(monkeypatch):
    card, packets, _, _ = route_card(monkeypatch, advance=False)
    current = card._scheduled_command
    monkeypatch.setattr(policy, "step", lambda **_: policy.Decision(None, policy.BLIND))
    message = card.next_command()
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and card._scheduled_command is current


def test_confirmed_measured_track_replaces_an_old_detour_goal(monkeypatch):
    card, packets, _, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    _, depth, odom, _ = snapshot[0]
    pose = PlanarPose(1000, ("odom",), .2, .1, .1)
    depth.update(obstacle_capture_pose=pose)
    odom.update(motion_pose=pose, obstacle_pose=pose)
    state = card._state
    track = state.tracker.track
    old = state.goal_anchor
    old.generation = track.generation - 1
    old.target_missing = True

    message = card.next_command()
    assert state.goal_anchor is not old
    assert state.tracker.track is track and track.measured_range_m is not None
    assert card._last_decision.dwa_acceptance.target_observed
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and packets[0]["values"][0] > 0
    assert card._scheduled_command.repeat_until is not None


def test_expiry_during_new_plan_wire_check_does_not_evict_live_route(monkeypatch):
    card, packets, clock, _ = route_card(monkeypatch)
    active = card._scheduled_command
    proposal = card.next_command()
    assert any(proposal["values"])
    check = card._recheck_dwa_inputs

    def slow_admission(*args, **kwargs):
        checked = check(*args, **kwargs)
        clock[0] = proposal["stamp_ms"] + proposal["ttl_ms"]
        return checked

    monkeypatch.setattr(card, "_recheck_dwa_inputs", slow_admission)
    assert card._stage_command(proposal, card._output_epoch)
    assert clock[0] < active.expires_ms
    assert card._scheduled_command is active
    assert "上一条路线" in card._last_decision.reason
    assert card._state.commanded == card._wire_twist
    assert card._state.commanded_vx == packets[-1]["values"][0]
    assert len(packets) == 1


def test_collision_found_while_proposal_expires_stops_previous_route(monkeypatch):
    card, packets, clock, _ = route_card(monkeypatch)
    proposal = card.next_command()
    assert any(proposal["values"])

    def late_collision(check, values, max_age, now_ms, snapshot, **kwargs):
        clock[0] += 5
        kwargs["diagnostics"]["reason"] = "footprint_collision"
        return now_ms - 10, now_ms + 3, False, check

    monkeypatch.setattr(card, "_recheck_dwa_motion", late_collision)
    assert card._stage_command(proposal, card._output_epoch)
    assert not any(packets[-1]["values"])
    assert card._state.dwa_totals["stop_reasons"]["footprint_collision"]


def test_background_plan_starts_from_current_controller_twist(monkeypatch):
    from plugins.navi import policy

    card, packets, _, snapshot = route_card(monkeypatch)
    _, depth, odom, _ = snapshot[0]
    pose = PlanarPose(1100, ("odom",), .21, .1, .2)
    depth.update(obstacle_capture_pose=pose)
    odom.update(motion_pose=pose, obstacle_pose=pose)
    card._dispatch_command()
    applied = tuple(packets[-1]["values"][i] for i in (0, 1, 5))
    assert applied[2] != packets[0]["values"][5]
    start = []
    step = policy.step

    def observe(**kwargs):
        start.append(kwargs["state"].commanded)
        return step(**kwargs)

    monkeypatch.setattr(policy, "step", observe)
    card.next_command()
    assert start == [applied]


@pytest.mark.parametrize("blocking", [False, True], ids=["side-wall", "obstructed-corridor"])
def test_far_anchored_target_keeps_direct_control_when_the_corridor_clears(monkeypatch, blocking):
    config, state, depth, odom = observation_case(body=False, attempted=False)
    rows = np.arange(480)[:, None] - 240.
    floor = .35 * 420. / np.maximum(rows, 1.)
    image = np.broadcast_to(np.minimum(floor, 5.), (480, 640)).copy()
    ground = np.broadcast_to((rows > 0.) & (floor < 5.), image.shape).copy()
    # Same measured wall patch. Only its location relative to the goal ray changes.
    left = 270 if blocking else 2
    image[210:360, left:left + 135] = 1.4
    ground[210:360, left:left + 135] = False
    depth = dict(depth, map=image, ground=ground)
    objects = {"timestamp": 1., "objects": [
        {"name": "person", "position": [.34, -.6], "confidence": .95}]}
    for stamp in (700, 800, 900):
        state.tracker.step(detections=dict(objects, timestamp=stamp / 1000),
                           depth=dict(depth, frame_id=stamp, capture_ms=stamp, source_frame_id=stamp),
                           target=state.target, config=config, ego=(0., 0., 0.), ego_measured=True, dt=.1)
    state.goal_anchor = None
    state.escape_side = 1.
    policy._remember_detour_goal(state, config, state.tracker.track, odom)
    assert state.goal_anchor is not None
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: (objects, depth, odom, 1000))
    assert card._stage_command(card.next_command(), card._output_epoch)
    packet, = packets
    decision = card._last_decision
    assert decision.distance_m == pytest.approx(5.)
    if blocking:
        assert state.dwa_trace["branch"] == "reachable_prefix" and state.dwa_selected
        assert packet["values"][0] > 0
        route = decision.dwa_acceptance.route_points
        assert packet["values"][5] * route[-1][1] < 0
        assert decision.dwa_acceptance.motion_guard.check.accepted
    else:
        assert decision.status == policy.APPROACHING and not state.dwa_selected
        # Direct approach at this fixed range and bearing on main 881dfe4a.
        assert packet["values"] == pytest.approx([.18665701297624213, 0., 0., 0., 0., -.204])


@pytest.mark.parametrize("now,delay,admitted", [(1320, 0, False), (1240, 0, True), (1240, 11, False)])
def test_route_uses_speed_measurement_age_through_write(monkeypatch, now, delay, admitted):
    card, packets, clock, snapshot = route_card(monkeypatch, advance=False)
    card._descriptor["rate"]["max_obs_age_ms"] = card._config.max_obs_age_ms = 1000
    objects, depth, odom, receipt = snapshot[0]
    # Pose and envelope are current, but the independent speed sample is old.
    # The original depth exposure remains the same object throughout.
    pose = PlanarPose(now, ("odom",), .21, .1, .1)
    speed_ms = 1001
    odom = dict(odom, motion_pose=pose, obstacle_pose=pose, received_ms=now, source_sample_id=now,
                received_speed_ms=speed_ms, component_speed_ms=speed_ms)
    snapshot[0] = objects, depth, odom, receipt
    clock[0] = now
    check = card._recheck_dwa_motion

    def delayed_check(*args, **kwargs):
        result = check(*args, **kwargs)
        clock[0] += delay
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", delayed_check)
    card._dispatch_command()
    moving = [p for p in packets[1:] if any(p["values"])]
    assert bool(moving) == admitted
    if admitted:
        packet, = moving
        assert packet["obs_stamp_ms"] <= speed_ms
        assert packet["stamp_ms"] + packet["ttl_ms"] <= speed_ms + card._config.max_feedback_gap_ms
    else:
        clock[0] = 1500
        card._dispatch_command()
        assert len(packets) == 1
        assert card._scheduled_command is None
