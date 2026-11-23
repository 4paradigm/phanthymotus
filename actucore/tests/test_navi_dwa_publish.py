"""DWA provenance at the actual JSON publisher, including queued renewals."""
import json
import logging
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.navi import dwa, local_path, motion, plugin, policy
from plugins.navi.active_scan import NearProjection
from test_navi_dwa_integration import advance_case, card_for_case, observation_case, observe_selector


def observation_card(monkeypatch, *, body=True, live_target=False, mount_forward=None,
                     confirm_stamps=(700, 800, 900), detector_timestamp=True):
    config, state, depth, odom = observation_case(body=body)
    if mount_forward is not None:
        mount = config.scan_geometry.base_from_optical.copy()
        mount[0, 3] = mount_forward
        config.scan_geometry = replace(config.scan_geometry, base_from_optical=mount)
    objects = None
    if live_target:
        objects = {"timestamp": 1., "count": 1, "objects": [
            {"name": "person", "position": [0., -.6], "confidence": .95}]}
        if not detector_timestamp:
            del objects["timestamp"]
        # The measured target lies on the 5 m background above the 2 m
        # obstacle. Confirm it through real earlier frames, retaining the
        # production three-hit requirement rather than injecting a Track.
        for stamp in confirm_stamps:
            state.tracker.step(
                detections=(dict(objects, timestamp=stamp / 1000) if detector_timestamp else objects),
                depth=dict(depth, frame_id=stamp, capture_ms=stamp,
                           control_ms=stamp, source_frame_id=stamp),
                target=state.target, config=config, ego=(0., 0., 0.),
                ego_measured=True, dt=.1)
        state.goal_anchor = None
        state.escape_side = 1.
        policy._remember_detour_goal(state, config, state.tracker.track, odom)
        state.escape_side = 0.
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    clock = [1000]
    snapshot = [(objects, depth, odom, 1000)]
    monkeypatch.setattr(plugin.time, "time", lambda: clock[0] / 1000)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0] / 1000)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    return card, packets, clock, snapshot


def selected_card(monkeypatch, *, body=True, live_target=False):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=live_target)
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    assert any(message["values"])
    return card, packets, clock, snapshot, message


def advance_snapshot(snapshot, clock, stamp, *, obstacle=False, vx=.1):
    objects, depth, odom, target_ms = snapshot[0]
    image = depth["map"].copy()
    if obstacle:
        image[220:370, 302:338] = .25
    clock[0] = stamp
    snapshot[0] = (objects, dict(depth, map=image, frame_id=stamp, capture_ms=stamp,
                                 control_ms=stamp, source_frame_id=stamp),
                   dict(odom, vx=vx, received_ms=stamp, source_sample_id=stamp), target_ms)


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_real_dwa_publication_counts_only_successful_wire_packets(monkeypatch, body):
    calls = observe_selector(monkeypatch)
    card, packets, _, _, message = selected_card(monkeypatch, body=body)
    if body:
        card._publish_command(message)
    else:
        assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1
    assert packets[0]["values"] == message["values"]
    assert not any("dwa" in key for key in packets[0]), "Provenance is internal, not a wire extension"
    stats = card._info()["dwa"]
    assert stats["decision"]["wire_values"] == stats["publication"]["wire_values"] == packets[0]["values"]
    assert stats["decision"]["wire_seq"] == stats["publication"]["wire_seq"] == packets[0]["seq"]
    assert stats["publication"]["decision_id"] == stats["decision"]["decision_id"]
    assert stats["totals"]["attempted_ticks"] == 1
    assert stats["totals"]["select_calls"] == len(calls)
    assert stats["totals"]["published_decisions"] == 1
    assert stats["totals"]["published_packets"] == 1


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("change", ["deadband", "cap", "twist", "expiry"])
def test_changed_or_expired_candidate_cannot_reach_the_wire(monkeypatch, body, change):
    card, packets, clock, _, message = selected_card(monkeypatch, body=body)
    if change == "deadband":
        card._descriptor["limits"] = {"min_magnitude": [.3, 0., 0., 0., 0., 0.]}
    elif change == "cap":
        card._config.vx_max = .01
    elif change == "twist":
        message["values"][5] += .1
    else:
        clock[0] += message["ttl_ms"]
    if body:
        card._publish_command(message)
    else:
        card._wire_moving = True  # A prior command must be stopped on veto.
        card._stage_command(message, card._output_epoch)
    assert packets and all(not any(packet["values"]) for packet in packets)
    assert card._state.dwa_trace["accepted_twist"] is None
    assert card._state.dwa_trace["proposed_twist"] is not None
    assert card._state.dwa_trace["select_calls"] > 0
    assert card._state.dwa_trace["veto_reason"] == (
        "publish_expired" if change == "expiry" else "postcheck_veto")
    assert card._state.dwa_totals.get("published_packets", 0) == 0


@pytest.mark.parametrize("change,stage,detail", [
    ("missing_check", "physical_recheck", "missing_forward_check"),
    ("missing_guard", "physical_recheck", "missing_motion_guard"),
    ("values", "publish_contract", "values_changed"),
    ("constraints", "publish_contract", "constraints_changed"),
    ("deadband", "publish_contract", "deadband_changed"),
    ("axis", "publish_contract", "policy_axis_bounds"),
    ("descriptor", "publish_contract", "descriptor_bounds"),
])
def test_postcheck_veto_reports_which_publication_contract_failed(monkeypatch, change, stage, detail):
    card, packets, _, _, message = selected_card(monkeypatch, body=True)
    origin = card._staged_dwa_command[1]
    if change == "missing_check":
        origin = replace(origin, forward_check=None)
    elif change == "missing_guard":
        origin = replace(origin, forward_check=replace(origin.forward_check, motion_guard=None))
    elif change == "values":
        message["values"][0] *= .5
    elif change in ("constraints", "axis"):
        card._config.vx_max = .01
    elif change == "deadband":
        card._descriptor["limits"] = {"min_magnitude": [.3, 0., 0., 0., 0., 0.]}
    else:
        card._descriptor["limits"] = {"lower": [-1.] * 6, "upper": [.01, 1., 1., 1., 1., 1.]}
    if change in ("deadband", "axis", "descriptor"):
        # Exercise the wire contract independently of the snapshot-identity
        # gate. The real sampled twist still violates this explicit limit.
        origin = replace(origin, constraints=card._dwa_constraints())
    card._publish_command(message, dwa_origin=origin)
    packet, = packets
    assert packet["values"] == [0.] * 6
    publication = card._info()["dwa"]["publication"]
    assert publication["reason"] == "postcheck_veto"
    assert publication["veto_stage"] == stage and publication["veto_detail"] == detail
    assert card._state.dwa_trace["veto_stage"] == stage
    assert card._state.dwa_trace["veto_detail"] == detail
    assert card._state.dwa_totals["stop_reasons"]["postcheck_veto"] == 1


def test_planning_expiry_clears_acceptance_but_retains_real_attempts(monkeypatch):
    config, state, depth, odom = observation_case()
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    clock = [1000]
    select = dwa.select

    def delayed_select(**kwargs):
        choice = select(**kwargs)
        clock[0] = 1900
        return choice

    monkeypatch.setattr(plugin.time, "time", lambda: clock[0] / 1000)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0] / 1000)
    monkeypatch.setattr(dwa, "select", delayed_select)
    message = card.next_command()
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert not state.dwa_selected
    assert card._last_decision.dwa_acceptance is None
    assert state.dwa_trace["accepted_twist"] is None
    assert state.dwa_trace["proposed_twist"] is not None
    assert state.dwa_trace["veto_reason"] == "publish_expired"
    assert state.dwa_trace["fallback_reason"] == "publication_veto"
    assert card._dwa_publication["wire_values"] == packets[-1]["values"]
    assert card._dwa_publication["decision_id"] == state.decision_id
    assert state.dwa_totals["attempted_ticks"] == 1
    assert state.dwa_totals["accepted_ticks"] == 0


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_publish_error_never_counts_a_dwa_success(monkeypatch, body):
    card, _, _, _, message = selected_card(monkeypatch, body=body)

    def fail(_):
        raise RuntimeError("transport failed")

    card._publisher = SimpleNamespace(publish=fail)
    if body:
        with pytest.raises(RuntimeError, match="transport failed"):
            card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
        assert card._scheduled_command is None
    assert card._published == 0
    assert not card._wire_moving
    assert card._state.dwa_totals.get("published_packets", 0) == 0
    assert card._state.dwa_totals.get("published_decisions", 0) == 0
    assert card._info()["dwa"]["publication"]["reason"] == "publish_error"
    assert "wire_values" not in card._dwa_publication
    assert "wire_seq" not in card._state.dwa_trace


def test_renewals_keep_the_original_decision_and_full_tail_check(monkeypatch):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    original = card._last_decision.dwa_acceptance
    assert card._stage_command(message, card._output_epoch)
    assert card._scheduled_command.forward_check is not None
    assert len(packets) == 1
    predictions = []
    predict = motion.predict_motion

    def record(*args, **kwargs):
        trace = predict(*args, **kwargs)
        predictions.append(trace)
        return trace

    monkeypatch.setattr(motion, "predict_motion", record)
    # A new planner tick can reset mutable diagnostic flags while the timer
    # still owns a valid earlier command. The queued origin must survive.
    card._state.dwa_selected = False
    card._state.dwa_trace = {"decision_id": original.decision_id + 1}
    card._staged_dwa_command = None
    objects, depth, odom, target_ms = snapshot[0]
    for stamp in (1100, 1200, 1300):
        clock[0] = stamp
        snapshot[0] = (objects, dict(depth, frame_id=stamp, capture_ms=stamp,
                                     control_ms=stamp, source_frame_id=stamp),
                       dict(odom, received_ms=stamp, source_sample_id=stamp), target_ms)
        card._dispatch_command()
    assert len(packets) == 4
    assert all(packet["values"] == message["values"] for packet in packets)
    assert len(calls) == card._state.dwa_totals["select_calls"]
    assert card._state.dwa_totals["attempted_ticks"] == 1
    assert card._state.dwa_totals["published_decisions"] == 1
    assert card._state.dwa_totals["published_packets"] == 4
    assert len(predictions) == 3
    assert all(t.stopped and t.duration >= original.motion_guard.prediction_horizon_s for t in predictions)
    publication = card._info()["dwa"]["publication"]
    assert publication["decision_id"] == original.decision_id
    assert publication["wire_seq"] == packets[-1]["seq"] and publication["wire_values"] == packets[-1]["values"]
    assert card._state.dwa_trace == {"decision_id": original.decision_id + 1}


@pytest.mark.parametrize("change", ["deadband", "cap"])
def test_card_postselection_limits_cannot_modify_a_real_candidate(monkeypatch, change):
    config, state, depth, odom = observation_case()
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    step = policy.step

    def change_after_selection(**kwargs):
        decision = step(**kwargs)
        assert decision.dwa_acceptance is not None
        if change == "deadband":
            card._descriptor["limits"] = {"min_magnitude": [.3, 0., 0., 0., 0., 0.]}
        else:
            config.vx_max = .01
        return decision

    monkeypatch.setattr(policy, "step", change_after_selection)
    message = card.next_command()
    assert message["values"] == [0.] * 6
    assert card._last_decision.dwa_acceptance is None
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert not state.dwa_selected
    assert state.dwa_trace["veto_reason"] == "postcheck_veto"
    assert state.dwa_trace["veto_detail"] == (
        "deadband_changed" if change == "deadband" else "constraints_changed")


@pytest.mark.parametrize("body,event", [(False, event) for event in
                                        ("pause", "target", "epoch", "new_obstacle")]
                         + [(True, event) for event in ("pause", "target", "epoch")])
def test_queued_dwa_stops_on_cancellation_or_new_obstacle(monkeypatch, body, event):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    if body:
        card._publish_command(message)
    else:
        assert card._stage_command(message, card._output_epoch)
    assert any(packets[0]["values"])
    if event == "pause":
        card._halt(True)
    elif event == "target":
        card._navigate_to({"target": "chair"})
    elif event == "epoch":
        card._cancel_scheduled_motion()
    else:
        objects, depth, odom, target_ms = snapshot[0]
        clock[0] = 1100
        image = depth["map"].copy()
        image[220:370, 302:338] = .25
        snapshot[0] = (objects, dict(depth, map=image, frame_id=1100, capture_ms=1100,
                                     control_ms=1100, source_frame_id=1100),
                       dict(odom, received_ms=1100, source_sample_id=1100), target_ms)
        card._dispatch_command()
    card._dispatch_command()
    assert card._scheduled_command is None
    assert packets[-1]["values"] == [0.] * 6
    assert len([packet for packet in packets if any(packet["values"])]) == 1
    if card._state.dwa_trace:
        assert card._state.dwa_trace["accepted_twist"] is None


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("live_target", [True, False], ids=["live", "remembered"])
@pytest.mark.parametrize("obstacle", [True, False], ids=["blocked", "clear"])
def test_real_tick_rechecks_new_input_before_first_send(monkeypatch, body, live_target, obstacle):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=live_target)
    publish = card._publish_command

    def input_arrives(message, **kwargs):
        if any(message["values"]):
            advance_snapshot(snapshot, clock, 1010, obstacle=obstacle)
        return publish(message, **kwargs)

    monkeypatch.setattr(card, "_publish_command", input_arrives)
    card._tick()
    assert calls and any(choice is not None for _, choice in calls)
    assert packets
    if obstacle:
        assert all(packet["values"] == [0.] * 6 for packet in packets)
        assert card._state.dwa_totals.get("published_packets", 0) == 0
        assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"
    else:
        accepted = [choice for _, choice in calls if choice is not None][-1]
        assert len(packets) == 1
        assert tuple(packets[0]["values"][i] for i in (0, 1, 5)) == accepted.command
        assert card._state.dwa_totals["published_packets"] == 1


def test_renewal_continues_valid_deceleration_from_fresh_measured_overspeed(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100, vx=.36)
    assert policy._feedback_speed_reduction(card._config, snapshot[0][2]) < 1.
    card._dispatch_command()
    assert packets[-1]["values"] == message["values"]
    assert card._state.dwa_totals["published_packets"] == 2
    assert card._scheduled_command.forward_check.motion_guard.measured[0] == .36


def test_pause_stops_direct_motion_before_an_in_progress_plan_finishes(monkeypatch):
    card, packets, _, _, message = selected_card(monkeypatch)
    card._publish_command(message)
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    step, cancel = policy.step, card._cancel_scheduled_motion

    def blocked_step(**kwargs):
        entered.set()
        assert release.wait(2.)
        return step(**kwargs)

    def observed_cancel(**kwargs):
        result = cancel(**kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(policy, "step", blocked_step)
    monkeypatch.setattr(card, "_cancel_scheduled_motion", observed_cancel)
    planner = threading.Thread(target=card._tick)
    pauser = threading.Thread(target=card._halt, args=(True,))
    planner.start()
    try:
        assert entered.wait(2.)
        assert card._staged_dwa_command is None
        pauser.start()
        assert cancelled.wait(2.)
        assert planner.is_alive()
        assert packets[-1]["values"] == [0.] * 6
        assert not card._wire_moving
    finally:
        release.set()
        planner.join(2.)
        if pauser.ident is not None:
            pauser.join(2.)
    assert not planner.is_alive() and not pauser.is_alive()


def test_real_tick_keeps_publish_error_through_cleanup(monkeypatch):
    card, _, _, _ = observation_card(monkeypatch)

    def fail(_):
        raise RuntimeError("transport failed at real tick")

    card._publisher = SimpleNamespace(publish=fail)
    card._tick()
    assert "transport failed at real tick" in card._last_error
    assert card._state.dwa_trace["veto_reason"] == "publish_error"
    assert card._info()["dwa"]["publication"]["reason"] == "publish_error"
    assert card._state.dwa_totals.get("published_packets", 0) == 0
    assert card._state.dwa_totals.get("published_decisions", 0) == 0


def test_expired_old_work_keeps_its_stop_reason_and_logs_one_veto(monkeypatch, caplog):
    card, packets, clock, _, message = selected_card(monkeypatch, body=False)
    original = card._last_decision.dwa_acceptance
    assert card._stage_command(message, card._output_epoch)
    next_trace = dict(decision_id=original.decision_id + 1, veto_reason="newer_decision_reason",
                      accepted_twist=(.1, 0., 0.))
    card._state.dwa_trace = dict(next_trace)
    clock[0] = message["stamp_ms"] + message["ttl_ms"]
    with caplog.at_level(logging.INFO, logger=plugin.__name__):
        card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    assert card._state.dwa_trace == next_trace
    assert card._state.dwa_totals["stop_reasons"] == {"publish_expired": 1}
    publication = card._info()["dwa"]["publication"]
    assert publication["decision_id"] == original.decision_id
    assert publication["reason"] == "publish_expired"
    events = [json.loads(record.getMessage().split("navi dwa publication: ", 1)[1])
              for record in caplog.records if record.getMessage().startswith("navi dwa publication: ")]
    assert len([event for event in events if event["event"] == "veto"]) == 1


def test_remembered_goal_guard_cannot_renew_the_original_command_window(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False)
    assert card._stage_command(message, card._output_epoch)
    check = card._scheduled_command.forward_check
    assert check is not None and not check.renewable
    assert check.target_ms == 0 and check.checked_ms == 1000
    for stamp in (1100, 1200):
        advance_snapshot(snapshot, clock, stamp)
        card._dispatch_command()
        assert len(packets) == 1
    clock[0] = message["stamp_ms"] + message["ttl_ms"]
    card._dispatch_command()
    assert len(packets) == 2 and packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("body", [True, False])
def test_duplicate_receipts_keep_the_original_first_send_deadline(monkeypatch, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body)
    objects, depth, odom, target_ms = snapshot[0]
    clock[0] = 1010
    snapshot[0] = (objects, dict(depth, frame_id=1010, control_ms=1010),
                   dict(odom, received_ms=1010), target_ms)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and packets[0]["values"] == message["values"]
    assert packets[0]["obs_stamp_ms"] == message["obs_stamp_ms"]
    assert packets[0]["stamp_ms"] + packets[0]["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]


@pytest.mark.parametrize("twist", [(0., .1, 0.), (0., 0., .25)])
def test_capture_guard_for_nonforward_dwa_commands(monkeypatch, twist):
    # This tests capture of an already accepted command, independently of
    # whether a particular reference makes the real selector prefer that axis.
    card, _, _, snapshot, _ = selected_card(monkeypatch, body=False)
    decision = card._last_decision
    decision.values = [twist[0], twist[1], 0., 0., 0., twist[2]]
    decision.dwa_acceptance = policy.DwaAcceptance(1, twist, 1.8)
    _, depth, odom, _ = snapshot[0]
    check = card._forward_check(decision, (0., 0., 0.), depth, odom, 0, renewable=False)
    assert check is not None and check.guard_horizon_s == 1.8
    assert not check.renewable and check.target_ms == 0


def short_advance_card(monkeypatch, *, body, live_target=False):
    config, state, depth, odom = advance_case(body=body)
    state.active_scan.advance_m = .16
    payload = None
    if live_target:
        payload = {"timestamp": 1.1, "count": 1,
                   "objects": [{"name": "chair", "position": [0., -.6], "confidence": .95}]}
        for stamp in (800, 900, 1000):
            state.tracker.step(detections=dict(payload, timestamp=stamp / 1000),
                               depth=dict(depth, frame_id=stamp, source_frame_id=stamp,
                                          capture_ms=stamp, control_ms=stamp),
                               target=state.target, config=config, ego=(0., 0., 0.),
                               ego_measured=True, dt=.1)
        state.goal_anchor = None
        policy._remember_detour_goal(state, config, state.tracker.track, odom)
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    clock = [1100]
    snapshot = [(payload, depth, odom, 1100)]
    monkeypatch.setattr(plugin.time, "time", lambda: clock[0] / 1000)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0] / 1000)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    assert message["values"][0] == pytest.approx(.15)
    return card, packets, clock, snapshot, message


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("constraint", ["room", "stop_line", "grid_bounds"])
def test_changed_inputs_preserve_all_applicable_motion_limits(monkeypatch, body, constraint):
    if constraint == "room":
        card, packets, clock, snapshot, message = short_advance_card(monkeypatch, body=body)
        advance_snapshot(snapshot, clock, 1110, vx=.29)
        assert policy._feedback_speed_reduction(card._config, snapshot[0][2]) == 1.
        assert .29 * card._last_decision.dwa_acceptance.guard_horizon_s > .16
    else:
        card, packets, clock, snapshot = observation_card(monkeypatch, body=body)
        if constraint == "stop_line":
            card._config.obstacle_stop_m, card._config.stop_distance_m, card._config.slow_distance_m = .85, 1., 1.4
        else:
            card._config.vx_max = 1.5
        message = card.next_command()
        assert card._last_decision.dwa_acceptance is not None
        advance_snapshot(snapshot, clock, 1010, vx=-1.2 if constraint == "grid_bounds" else .1)
        if constraint == "stop_line":
            snapshot[0][1]["map"][220:370, 302:338] = .75
            assert policy._clearance(snapshot[0][1], card._config)[0] < card._config.obstacle_stop_m
        else:
            assert policy._feedback_speed_reduction(card._config, snapshot[0][2]) == 1.
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    if constraint == "grid_bounds":
        assert packets and all(packet["values"] == [0.] * 6 for packet in packets)
        assert card._info()["dwa"]["publication"]["reason"] == "outside_grid"
    else:
        assert packets and all(packet["values"] == message["values"] for packet in packets)
        assert card._wire_dwa_origin.forward_check.motion_guard.check.accepted


def test_renewal_can_continue_beyond_a_local_reference_endpoint(monkeypatch):
    card, packets, clock, snapshot, message = short_advance_card(monkeypatch, body=False, live_target=True)
    assert card._state.tracker.track.observed and card._state.tracker.track.measured_range_m is not None
    assert card._stage_command(message, card._output_epoch)
    assert card._scheduled_command.forward_check.renewable
    for stamp in (1200, 1300):
        advance_snapshot(snapshot, clock, stamp, vx=.1)
        card._dispatch_command()
        assert any(packets[-1]["values"]), "A fresh safe continuation remains executable"
    advance_snapshot(snapshot, clock, 1400, vx=.1)
    card._dispatch_command()
    assert packets[-1]["values"] == message["values"]
    assert card._wire_dwa_origin.forward_check.motion_guard.check.accepted


def test_fresh_obstacle_uses_the_translated_body_corridor(monkeypatch):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=True)
    config = card._config
    transform = config.scan_geometry.base_from_optical.copy()
    transform[1, 3] = .3
    geometry = replace(config.scan_geometry, base_from_optical=transform)
    config.local_geometry = config.scan_geometry = geometry
    config.depth_intrinsics = geometry.horizontal
    config.obstacle_stop_m, config.stop_distance_m, config.slow_distance_m = .85, 1., 1.4
    depth = snapshot[0][1]
    depth["map"].fill(5.)
    depth["map"][220:370, 365:400] = 2.
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    advance_snapshot(snapshot, clock, 1010)
    fresh = snapshot[0][1]
    fresh["map"].fill(5.)
    fresh["map"][220:370, 607:637] = .37
    assert policy._clearance(fresh, config)[0] == 5.
    _, occupied = local_path.project_observation(fresh, geometry, require_valid_depth=True)
    corridor = occupied & (local_path.FORWARD >= 0.) & (abs(local_path.RIGHT) <= policy._keepout_m(config))
    assert local_path.FORWARD[corridor].min() < config.obstacle_stop_m
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"


@pytest.mark.parametrize("cap", [.30, 3.])
def test_feedback_change_rebuilds_complete_braking(monkeypatch, cap):
    config, state, depth, odom = advance_case(body=False)
    config.vx_max = cap
    state.active_scan.advance_m = 4.
    state.goal_anchor = policy.DetourGoal(5., 0.)
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    clock = [1100]
    snapshot = [(None, depth, odom, 1100)]
    monkeypatch.setattr(plugin.time, "time", lambda: clock[0] / 1000)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0] / 1000)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    message = card.next_command()
    accepted = card._last_decision.dwa_acceptance
    assert accepted is not None and accepted.command == (.15, 0., 0.)
    advance_snapshot(snapshot, clock, 1110, vx=2.)
    fresh = snapshot[0][1]
    fresh["ground"] = depth["ground"].copy()
    fresh["map"][30:95, 76:85] = 4.6
    fresh["ground"][30:95, 76:85] = False
    predicted = dwa._predict(accepted.command, (2., 0., 0.), horizon=1.2, dt=.04,
                             acceleration=(.5, .5, 1.), braking=(.45, .45, .8), stop=True)
    assert predicted.duration > accepted.guard_horizon_s
    assert 5. - predicted.poses[-1][0] < config.stop_distance_m
    card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == [0.] * 6
    assert card._dwa_publication["reason"] in ("target_standoff", "footprint_collision", "outside_grid")


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_failed_stop_is_attempted_once_per_dispatch_and_later_recovers(monkeypatch, body):
    card, packets, clock, _, message = selected_card(monkeypatch, body=body)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    original = card._wire_dwa_origin
    attempts = []
    transport = card._publisher.publish

    def fail_four_stops(payload):
        packet = json.loads(payload.data)
        assert packet["values"] == [0.] * 6
        attempts.append(packet)
        if len(attempts) <= 4:
            raise RuntimeError("stop transport unavailable")
        transport(payload)

    card._publisher = SimpleNamespace(publish=fail_four_stops)
    clock[0] = message["stamp_ms"] + message["ttl_ms"]
    if body:
        card._halt(True)
    else:
        card._dispatch_command()
    assert len(attempts) == 1, "A failed zero must not recursively send another zero"
    epoch = card._output_epoch
    assert card._wire_moving and card._wire_dwa_origin is original
    assert card._state.dwa_totals.get("stop_packets", 0) == 0
    card._paused = card._output_blocked = True
    retry = card._tick if body else card._dispatch_command
    for expected in (2, 3, 4):
        clock[0] += 1000
        retry()
        assert len(attempts) == expected
        assert card._wire_moving and card._wire_dwa_origin is original
        assert card._output_epoch == epoch
    clock[0] += 1000
    retry()
    assert len(attempts) == 5 and len(packets) == 2
    assert attempts[-1]["stamp_ms"] == attempts[-1]["obs_stamp_ms"] == clock[0]
    assert not card._wire_moving and card._wire_dwa_origin is None
    assert card._output_epoch == epoch
    assert card._scheduled_command is None
    assert card._state.dwa_totals["stop_reasons"] == {"superseded" if body else "publish_expired": 1}

@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_new_obstacle_rechecks_shared_motion(monkeypatch, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body)
    objects, depth, odom, target_ms = snapshot[0]
    # A source may reuse a sample ID or replace its array in place.
    depth["map"][220:370, 270:370] = .12
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"


def test_duplicate_dispatch_does_not_extend_deadline(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    card._stage_command(message, card._output_epoch)
    deadline = message["stamp_ms"] + message["ttl_ms"]
    for stamp in range(message["stamp_ms"] + 10, deadline, 10):
        clock[0] = stamp
        card._dispatch_command()
        assert card._scheduled_command.expires_ms == deadline
    clock[0] = deadline
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6

@pytest.mark.parametrize("change", ["anchor", "front", "near"])
def test_motion_proof_checks_target_and_memory_revisions(monkeypatch, change):
    import numpy as np
    card, packets, _, _, message = selected_card(monkeypatch, live_target=True)
    state = card._state
    if change == "anchor":
        state.goal_anchor.x = card._config.stop_distance_m + .01
    elif change == "front":
        state.tracker.track.front_required = True
    else:
        squares = local_path.ObstacleSquares(np.array([[.22, 0.]]), .3)
        state.active_scan.near_projection = NearProjection(squares=squares, stamp_ms=1000)
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == {
        "anchor": "target_standoff",
        "front": "target_missing", "near": "remembered_obstacle"}[change]


def test_unchanged_first_send_reuses_the_selected_motion_check(monkeypatch):
    card, packets, clock, _, message = selected_card(monkeypatch)
    selected = card._last_decision.dwa_acceptance.motion_guard
    def unexpected(*args, **kwargs):
        pytest.fail("A covered execution window should reuse the selected physical result")
    monkeypatch.setattr(motion, "predict_motion", unexpected)
    clock[0] += 10
    card._publish_command(message)
    assert packets[-1]["values"] == message["values"]
    assert card._wire_dwa_origin.forward_check.motion_guard.check is selected.check
    clock[0] = message["stamp_ms"] + message["ttl_ms"]
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("scheduled", [False, True], ids=["direct", "scheduled"])
@pytest.mark.parametrize("pose_ms,memory_ms", [(1110, 1000), (1109, 1000), (1100, 1000), (1109, 999)])
@pytest.mark.parametrize("blocked", [True, False], ids=["contact", "clear"])
def test_remembered_square_is_registered_at_current_feedback_pose(monkeypatch, scheduled, pose_ms, memory_ms, blocked):
    import numpy as np
    from plugins.navi.odom import PlanarPose
    card, packets, clock, snapshot = observation_card(monkeypatch, body=True)
    if scheduled:
        card._command_timer = object()
    objects, depth, odom, _ = snapshot[0]
    start = PlanarPose(1000, ("odom",), 0., 0., 0.)
    odom["obstacle_pose"] = start
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    # This cell is clear at the planning pose, but in front after the new
    # measured translation and quarter turn. The old origin sees no contact.
    squares = local_path.ObstacleSquares(np.array([[.50 if blocked else .95, -.22]]), .27)
    card._state.active_scan.near_projection = NearProjection(squares=squares, stamp_ms=memory_ms)
    advance_snapshot(snapshot, clock, 1110)
    snapshot[0][2]["obstacle_pose"] = PlanarPose(pose_ms, start.frame, .5, 0., 1.57079632679)
    if scheduled:
        card._stage_command(message, card._output_epoch)
    else:
        card._publish_command(message)
    assert packets[-1]["values"] == ([0.] * 6 if blocked else message["values"])
    if blocked:
        assert card._info()["dwa"]["publication"]["reason"] == "remembered_obstacle"


@pytest.mark.parametrize("arrival", ["after_scene_read", "after_selection"])
def test_original_proof_records_memory_at_selection(monkeypatch, arrival):
    from plugins.navi.odom import PlanarPose
    card, packets, _, snapshot = observation_card(monkeypatch)
    pose = PlanarPose(1000, ("odom",), 0., 0., 0.)
    snapshot[0][2]["obstacle_pose"] = pose
    scan = card._state.active_scan
    module, function = (policy, "dwa_obstacles") if arrival == "after_scene_read" else (dwa, "select")
    read = getattr(module, function)

    def add_contact_after_read(*args, **kwargs):
        result = read(*args, **kwargs)
        scan.near_obstacles.frame = pose.frame
        scan.near_obstacles.cells.add((4, 0))
        scan.update_near_obstacles(None, card._config.local_geometry, pose, .20, 2., approaching=False)
        return result

    monkeypatch.setattr(module, function, add_contact_after_read)
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "remembered_obstacle"


def test_timer_stops_held_remembered_motion_on_a_new_hit(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1020, obstacle=True)
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"


def test_obstacle_arriving_during_first_send_recheck_reaches_the_wire_as_zero(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch)
    advance_snapshot(snapshot, clock, 1010)
    evaluate = motion.evaluate_motion
    delivered = False

    def new_depth_after_check(*args, **kwargs):
        nonlocal delivered
        result = evaluate(*args, **kwargs)
        if not delivered:
            advance_snapshot(snapshot, clock, 1020, obstacle=True)
            delivered = True
        return result

    monkeypatch.setattr(motion, "evaluate_motion", new_depth_after_check)
    card._publish_command(message)
    assert delivered
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"


def test_negotiated_admission_is_inside_the_existing_hold(monkeypatch):
    card, _, _, _ = observation_card(monkeypatch)
    calls = observe_selector(monkeypatch)
    message = card.next_command()
    accepted = card._last_decision.dwa_acceptance
    kwargs = next(k for k, choice in calls if choice and choice.command == accepted.command)
    admission = message["ttl_ms"] / 1000
    assert admission == .25
    assert kwargs["latency_s"] == admission
    assert kwargs["horizon"] == max(card._config.dwa_horizon_s, .8 + .5 + .05, admission + .5 + .05)
    trace = accepted.motion_guard.trace
    assert trace.times[trace.moving_count - 1] == pytest.approx(kwargs["horizon"])

@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_dwa_cached_comparison_sees_new_obstacle(monkeypatch, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body)
    revision = plugin._motion_inputs
    injected = False

    def changed_while_comparing(*args):
        nonlocal injected
        result = revision(*args)
        if not injected:
            injected = True
            advance_snapshot(snapshot, clock, 1010, obstacle=True)
        return result

    monkeypatch.setattr(plugin, "_motion_inputs", changed_while_comparing)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert injected and packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_unregistered_pose_frame_invalidates_dwa_origin(monkeypatch, body):
    from plugins.navi.odom import PlanarPose
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body)
    snapshot[0][2]["obstacle_pose"] = PlanarPose(1000, ("odom-A",), 0., 0., 0.)
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    advance_snapshot(snapshot, clock, 1010)
    snapshot[0][2]["obstacle_pose"] = PlanarPose(1009, ("odom-B",), 0., 0., 0.)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "geometry_mismatch"


@pytest.mark.parametrize("scheduled", [False, True], ids=["direct", "scheduled"])
@pytest.mark.parametrize("memory_frame", ["odom", "unregistered"])
def test_empty_near_projection_preserves_dwa_only_in_its_registered_frame(monkeypatch, scheduled, memory_frame):
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False)
    card._command_timer = object() if scheduled else None
    advance_snapshot(snapshot, clock, 21000)
    current = PlanarPose(21000, ("odom",), 0., 0., 0.)
    snapshot[0][2].update(obstacle_pose=current, motion_pose=current,
                          velocity_history=((21000, 21000, (.1, 0., 0.)),))
    scan, geometry = card._state.active_scan, card._config.local_geometry
    old = PlanarPose(1000, (memory_frame,), -2., 0., 0.)
    cells = np.isclose(local_path.FORWARD, .10) & np.isclose(local_path.RIGHT, .10)
    scan.near_obstacles.remember(scan._footprint_origin(old, geometry), cells)
    retained = set(scan.near_obstacles.cells)

    message = card.next_command()
    accepted = card._last_decision.dwa_acceptance
    assert accepted is not None and accepted.motion_guard.check.accepted
    assert accepted.motion_guard.trace.stopped and message["values"][0] > 0.
    assert scan.near_obstacles.has_occupied and scan.near_obstacles.cells == retained
    assert scan.near_hits is None and scan.near_squares is None
    if scheduled:
        card._stage_command(message, card._output_epoch)
    else:
        card._publish_command(message)

    packet, = packets
    if memory_frame == "odom":
        # The retained cell is 1.9 m behind this grid origin. A valid empty
        # projection must not erase the real selector's complete-stop proof.
        assert packet["values"] == message["values"]
        assert card._state.dwa_totals["published_decisions"] == 1
    else:
        # An unavailable projection is different: the old obstacle has no
        # registration to this robot pose and must still prevent publication.
        assert packet["values"] == [0.] * 6
        assert card._state.dwa_totals.get("published_decisions", 0) == 0
    assert scan.near_obstacles.cells == retained


@pytest.mark.parametrize("scheduled", [False, True], ids=["direct", "scheduled"])
@pytest.mark.parametrize("event", ["continuous", "feedback_gap", "expired", "new_contact"])
def test_first_absolute_pose_keeps_only_a_continuous_fresh_dwa_proof(monkeypatch, scheduled, event):
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False)
    card._command_timer = object() if scheduled else None
    advance_snapshot(snapshot, clock, 21000)
    snapshot[0][2]["velocity_history"] = ((21000, 21000, (.1, 0., 0.)),)
    card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    scan, geometry = card._state.active_scan, card._config.local_geometry
    relative = ("navi_relative", scan.motion_frame)
    old = PlanarPose(1000, relative, -2., 0., 0.)
    cells = np.isclose(local_path.FORWARD, .10) & np.isclose(local_path.RIGHT, .10)
    scan.near_obstacles.remember(scan._footprint_origin(old, geometry), cells)
    retained = set(scan.near_obstacles.cells)

    stamp = 21400 if event == "feedback_gap" else 21100
    advance_snapshot(snapshot, clock, stamp)
    absolute = PlanarPose(stamp, ("odom",), 10., 20., .2)
    snapshot[0][2].update(obstacle_pose=absolute, motion_pose=absolute, velocity_history=(
        (21000, 21000, (.1, 0., 0.)), (stamp, stamp, (.1, 0., 0.))))
    if event == "new_contact":
        select = dwa.select

        def record_contact_after_selection(**kwargs):
            choice = select(**kwargs)
            if choice is not None:
                # A memory update can precede publication of its projection.
                # The new cell lies inside the body at the selected pose.
                pose = scan._relative_near_pose(stamp, scan.pose)
                contact = np.isclose(local_path.FORWARD, -.05) & np.isclose(local_path.RIGHT, 0.)
                scan.near_obstacles.remember(scan._footprint_origin(pose, geometry), contact)
                retained.update(scan.near_obstacles.cells)
            return choice

        monkeypatch.setattr(dwa, "select", record_contact_after_selection)
    message = card.next_command()
    accepted = card._last_decision.dwa_acceptance
    if event == "feedback_gap":
        assert accepted is None and scan.pose_registration is None
        assert scan.motion_frame is not relative[1]
    else:
        assert accepted is not None and accepted.motion_guard.check.accepted
        assert accepted.motion_guard.pose.frame == absolute.frame
        assert scan.pose_registration[0].frame == scan.near_obstacles.frame == relative
        assert scan.near_hits is None and scan.near_squares is None
        assert message["values"][0] > 0.
    if event == "expired":
        clock[0] = message["stamp_ms"] + message["ttl_ms"]
    if scheduled:
        card._stage_command(message, card._output_epoch)
    else:
        card._publish_command(message)

    # The first absolute fix registers the existing relative memory; it
    # cannot discard that memory, bridge a feedback gap or extend a deadline.
    if event == "continuous":
        packet, = packets
        assert packet["values"] == message["values"]
    else:
        assert all(packet["values"] == [0.] * 6 for packet in packets)
        if event == "expired":
            assert card._dwa_publication["reason"] == "publish_expired"
    assert bool(card._state.dwa_totals.get("published_decisions", 0)) is (event == "continuous")
    assert scan.near_obstacles.cells == retained


def _moving_memory_case(monkeypatch, layer, world_y, path):
    from plugins.navi.odom import PlanarPose
    card, packets, clock, snapshot = observation_card(monkeypatch, body=layer == "historical")
    card._command_timer = None if path == "direct" else object()
    _, _, odom, _ = snapshot[0]
    old = PlanarPose(990, ("odom",), 0., 0., 0.)
    odom.update(obstacle_pose=old, motion_pose=PlanarPose(1000, old.frame, 0., -.5, 0.))
    scan = card._state.active_scan
    memory = scan.obstacles if layer == "historical" else scan.near_obstacles
    memory.frame = old.frame
    memory.cells.add((10, round(world_y / local_path.RESOLUTION)))
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    if layer == "historical":
        assert scan.historical_hits.any()
    else:
        assert scan.near_revision == 990 and scan.near_squares.centers.size
    if path == "held":
        card._stage_command(message, card._output_epoch)
        assert packets[-1]["values"] == message["values"]
    advance_snapshot(snapshot, clock, 1110)
    fresh = PlanarPose(1109, old.frame, .50, 0., 1.57079632679)
    snapshot[0][2].update(obstacle_pose=fresh, motion_pose=fresh)
    return card, packets, message, scan, fresh


@pytest.mark.parametrize("layer", ["historical", "near-camera"])
@pytest.mark.parametrize("path", ["direct", "scheduled", "held"])
@pytest.mark.parametrize("world_y,blocked", [(.20, True), (.65, False)], ids=["contact", "clear"])
def test_memory_layer_keeps_its_actual_projection_pose(monkeypatch, layer, path, world_y, blocked):
    card, packets, message, _, _ = _moving_memory_case(monkeypatch, layer, world_y, path)
    if path == "direct":
        card._publish_command(message)
    elif path == "scheduled":
        card._stage_command(message, card._output_epoch)
    else:
        card._dispatch_command()
    assert packets[-1]["values"] == ([0.] * 6 if blocked else message["values"])
    if blocked:
        assert card._info()["dwa"]["publication"]["reason"] == "remembered_obstacle"


@pytest.mark.parametrize("timing", ["during", "after"])
@pytest.mark.parametrize("world_y,blocked", [(.20, True), (.65, False)], ids=["contact", "clear"])
def test_held_timer_during_near_reprojection(monkeypatch, timing, world_y, blocked):
    card, packets, message, scan, fresh = _moving_memory_case(monkeypatch, "near-camera", world_y, "held")
    project_free = scan.near_obstacles.project_free

    def dispatch_during_projection(origin):
        card._dispatch_command()
        return project_free(origin)

    if timing == "during":
        monkeypatch.setattr(scan.near_obstacles, "project_free", dispatch_during_projection)
    scan.update_near_obstacles(None, card._config.local_geometry, fresh, .20, 2., approaching=False)
    if timing == "after":
        card._dispatch_command()
    assert packets[-1]["values"] == ([0.] * 6 if blocked else message["values"])
    if blocked:
        assert card._info()["dwa"]["publication"]["reason"] == "remembered_obstacle"


def test_held_timer_checks_the_projection_used_by_its_revision(monkeypatch):
    card, packets, _, scan, fresh = _moving_memory_case(monkeypatch, "near-camera", .20, "held")
    scene = card._registered_dwa_scene
    origin = scan.near_projection_pose
    old_body_pose = replace(origin, x=0.)
    memory = scan.near_obstacles
    cell = (10, 4)

    def reproject_while_rebuilding(*args, **kwargs):
        # A clear observation retires the cell. The original occupied scene
        # is restored before the publisher's final revision comparison.
        memory.cells.remove(cell)
        scan.update_near_obstacles(None, card._config.local_geometry, fresh, .20, 2., approaching=False)
        result = scene(*args, **kwargs)
        memory.cells.add(cell)
        scan.update_near_obstacles(None, card._config.local_geometry, old_body_pose, .20, 2., approaching=False)
        return result

    monkeypatch.setattr(card, "_registered_dwa_scene", reproject_while_rebuilding)
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "remembered_obstacle"


@pytest.mark.parametrize("mode", ["direct", "scheduled", "copied_turn"])
@pytest.mark.parametrize("event", ["takeover", "cancel", "new_contact"])
def test_scan_returns_control_on_fresh_feasible_motion(monkeypatch, mode, event):
    scheduled = mode != "direct"
    config, state, depth, odom = advance_case(body=False)
    scan, anchor = state.active_scan, state.goal_anchor
    scan.phase, scan.route_segment = "turn", False
    scan.origin, scan.origin_set = scan.pose, True
    scan.advance_m = .90 if mode == "copied_turn" else None
    if mode == "copied_turn":
        config.local_controller = "legacy"
    scan.observed_vantages.append((0., 0., -.8))
    frame = scan.motion_frame
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    card._command_timer = object() if scheduled else None
    clock = [1.]
    monkeypatch.setattr(plugin.time, "time", lambda: clock[0])
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0])
    def tick(stamp, source, wz):
        clock[0] = stamp / 1000
        feedback = dict(odom, received_ms=stamp, source_sample_id=stamp, wz=wz,
                        velocity_history=((stamp, stamp, (0., 0., wz)),))
        card._velocity_history.append((stamp, stamp, (0., 0., wz)))
        card._depth_ms, card._depth_map = stamp, depth["map"]
        view = dict(depth, frame_id=stamp, control_ms=stamp, source_frame_id=source)
        monkeypatch.setattr(card, "_observation_snapshot", lambda **_: (None, view, feedback, stamp))
        card._tick()
    # The existing view owns a bounded observation turn. A fresh source is
    # required to take over, even on a repeated receipt of the same view.
    tick(1000, 1000, 0.)
    if mode == "copied_turn":
        assert card._scheduled_command.turn and card._scheduled_command.turn is not scan
        config.local_controller = "dwa"
    if scheduled:
        card._dispatch_command()
    assert packets and packets[-1]["values"][5] < 0. and packets[-1]["values"][0] == 0.
    old = packets[-1]["values"]
    epoch = card._output_epoch
    if event != "takeover":
        select = card.next_command
        def interleave():
            message = select()
            assert card._last_decision.dwa_acceptance and message["values"][0] > 0.
            if event == "cancel":
                card._cancel_scheduled_motion(block=True)
            else:
                import numpy as np
                squares = local_path.ObstacleSquares(np.array([[0., .10]]), .02)
                scan.near_projection = NearProjection(squares=squares, stamp_ms=1100)
            return message
        monkeypatch.setattr(card, "next_command", interleave)
    tick(1100, 1100, -.12)
    if event != "takeover":
        if scheduled:
            card._dispatch_command()
        assert packets[-1]["values"] == [0.] * 6
        assert all(not any(p["values"]) for p in packets[1:])
        return
    assert card._last_decision.dwa_acceptance is not None
    assert packets[-1]["values"][0] > 0.
    assert len(packets) == 2 and card._output_epoch == epoch + 1
    assert not scan.active and not scan.result_frame and not scan.directions
    assert state.goal_anchor is anchor and frame is scan.motion_frame
    assert (0., 0., -.8) in scan.observed_vantages and scan.views and scan.poses
    if scheduled:
        card._dispatch_command()
    assert all(p["values"] != old for p in packets[1:])


@pytest.mark.parametrize("scheduled", [False, True])
def test_first_blocked_frame_retries_failed_zero_before_new_motion(monkeypatch, scheduled):
    import numpy as np
    card, packets, clock, snapshot = observation_card(monkeypatch, body=True)
    card._command_timer = object() if scheduled else None
    card._tick()
    assert packets[-1]["values"][0] > 0.
    scan = card._state.active_scan
    scan.near_projection = NearProjection(squares=local_path.ObstacleSquares(np.array([[.05, .18]]), .02),
                                          stamp_ms=1000)
    advance_snapshot(snapshot, clock, 1050)
    publish = card._publisher.publish
    attempts = []
    attempted_seq = [packets[-1]["seq"]]

    def fail_first_stop(payload):
        packet = json.loads(payload.data)
        attempts.append(packet["values"])
        attempted_seq.append(packet["seq"])
        if len(attempts) == 1:
            raise RuntimeError("missed first zero")
        publish(payload)
    card._publisher.publish = fail_first_stop
    card._tick()
    assert attempts == [[0.] * 6]
    scan.near_projection = NearProjection()
    advance_snapshot(snapshot, clock, 1100)
    card._tick()
    assert attempts == [[0.] * 6, [0.] * 6]
    assert not card._wire_moving
    advance_snapshot(snapshot, clock, 1150)
    card._tick()
    assert packets[-1]["values"][0] > 0.
    # Only successful publishes are delivered. Mirror a sink rejecting
    # duplicates/reordering independently for each producer.
    last_seq = {}
    accepted = []
    for packet in packets:
        source, seq = packet["source"], packet["seq"]
        if seq > last_seq.get(source, -1):
            last_seq[source] = seq
            accepted.append(packet)
    assert [bool(any(p["values"])) for p in accepted] == [True, False, True], [
        (p["seq"], p["values"]) for p in packets]
    assert all(a["seq"] < b["seq"] for a, b in zip(packets, packets[1:]))
    assert all(a < b for a, b in zip(attempted_seq, attempted_seq[1:]))


def direct_tracking_card(monkeypatch, body):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True)
    card._state.goal_anchor = None
    card._state.active_scan.attempted = False
    objects, depth, odom, stamp = snapshot[0]
    rows = np.arange(480)[:, None] - 240.
    floor = .35 * 420. / np.maximum(rows, 1.)
    depth = dict(depth, map=np.broadcast_to(np.minimum(floor, 5.), (480, 640)).copy(),
                 ground=np.broadcast_to((rows > 0.) & (floor < 5.), (480, 640)).copy())
    snapshot[0] = objects, depth, odom, stamp
    card._wire_seq = 40
    return card, packets, clock


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_normal_direct_tracking_is_not_no_reference(monkeypatch, body):
    card, packets, _ = direct_tracking_card(monkeypatch, body)
    message = card.next_command()
    trace = card._state.dwa_trace
    assert card._last_decision.status == policy.APPROACHING and message["values"][0] > 0.
    assert trace["fallback_reason"] == "not_needed"
    assert trace["select_calls"] == 0 and card._last_decision.dwa_acceptance is None
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and packets[0]["seq"] != message["seq"]
    assert card._dwa_publication["decision_id"] == trace["decision_id"]
    assert trace["wire_values"] == card._dwa_publication["wire_values"] == packets[0]["values"]
    assert trace["wire_seq"] == card._dwa_publication["wire_seq"] == packets[0]["seq"]
    assert card._state.dwa_totals.get("published_decisions", 0) == 0


def test_planning_expiry_keeps_measured_dwa_progress(monkeypatch):
    card, _, clock, snapshot, first = selected_card(monkeypatch)
    card._publish_command(first)
    state = card._state
    state.idle_for_s = card._config.idle_timeout_s - .01
    advance_snapshot(snapshot, clock, 1200, vx=.30)
    select = dwa.select

    def slow_select(**kwargs):
        choice = select(**kwargs)
        clock[0] = 2050
        return choice

    monkeypatch.setattr(dwa, "select", slow_select)
    message = card.next_command()
    assert state.dwa_trace["proposed_twist"] and message["values"] == [0.] * 6
    assert state.dwa_trace["veto_reason"] == "publish_expired"
    assert state.idle_for_s == 0. and not state.failed_reason

@pytest.mark.parametrize("body", [True, False])
def test_direct_tracking_expired_during_planning_reports_zero_with_its_decision(monkeypatch, body):
    card, packets, clock = direct_tracking_card(monkeypatch, body)
    choose = policy.step

    def slow_choice(**kwargs):
        choice = choose(**kwargs)
        assert choice.status == policy.APPROACHING and choice.dwa_acceptance is None
        clock[0] = 1950
        return choice

    monkeypatch.setattr(policy, "step", slow_choice)
    message = card.next_command()
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == [0.] * 6
    publication, trace = card._dwa_publication, card._state.dwa_trace
    assert publication["decision_id"] == trace["decision_id"]
    assert publication["reason"] == "publish_expired"
    assert trace["fallback_reason"] == "publication_veto"
    assert trace["final_twist"][0] > 0. and trace["wire_values"] == packets[-1]["values"]


def test_direct_tracking_reads_original_feedback_clocks_without_local_scan(monkeypatch):
    card, _, clock, snapshot = observation_card(monkeypatch, live_target=True)
    card._config.local_planning = False
    card._config.idle_timeout_s = .55
    card._state.goal_anchor = None
    objects, depth, _, _ = snapshot[0]
    monkeypatch.setattr(card, "_observation_snapshot", plugin.NaviPlugin._observation_snapshot.__get__(card))
    for stamp in range(1000, 2000, 100):
        clock[0] = stamp
        card._objects = dict(objects, timestamp=stamp / 1000)
        card._objects_ms = card._depth_ms = card._odom_ms = card._odom_received_ms = stamp
        card._depth_map = np.full_like(depth["map"], 5.)
        card._odom = dict(vx=.2, vy=0., wz=0.)
        card._odom_source_ms = stamp
        message = card.next_command()
        assert card._last_decision.status == policy.APPROACHING and message["values"][0] > 0.
    assert card._state.idle_for_s < .3


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("newer_trace", [False, True], ids=["current", "new-decision"])
def test_superseded_ordinary_approach_records_its_unpublished_decision(monkeypatch, body, newer_trace):
    card, packets, _ = direct_tracking_card(monkeypatch, body)
    next_command, choose = card.next_command, policy.step
    epoch = card._output_epoch

    def invalidate_during_planning(**kwargs):
        decision = choose(**kwargs)
        assert card._state.dwa_trace["fallback_reason"] == "not_needed" and decision.values[0] > 0.
        assert decision.dwa_acceptance is None
        card._cancel_scheduled_motion(block=True)
        return decision

    def replace_trace_before_send():
        message = next_command()
        if newer_trace:
            card._state.dwa_trace = dict(decision_id=card._state.decision_id + 1)
        return message

    monkeypatch.setattr(policy, "step", invalidate_during_planning)
    monkeypatch.setattr(card, "next_command", replace_trace_before_send)
    card._tick()
    assert not packets
    event = card._dwa_publication
    assert event and (event["reason"], event["event"], event["epoch"]) == ("superseded", "veto", epoch)
    assert event["decision_id"] == card._state.decision_id and "wire_values" not in event and "wire_seq" not in event
    trace = card._state.dwa_trace
    if newer_trace:
        assert trace == dict(decision_id=card._state.decision_id + 1)
    else:
        assert (trace["fallback_reason"], trace["veto_reason"]) == ("publication_veto", "superseded")
        assert "wire_values" not in trace and "wire_seq" not in trace


@pytest.mark.parametrize("body", [True, False])
@pytest.mark.parametrize("scheduled", [True, False])
@pytest.mark.parametrize("right, accepted", [(.30, True), (.25, False)])
def test_first_send_keeps_subcell_world_square(monkeypatch, body, scheduled, right, accepted):
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=body)
    card._command_timer = object() if scheduled else None
    pose = PlanarPose(1000, ("odom",), 0., -.026, 0.)
    snapshot[0][2].update(obstacle_pose=pose, motion_pose=pose)
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    geometry = card._config.scan_geometry or card._config.local_geometry
    scan = card._state.active_scan
    old = replace(pose, stamp_ms=950, y=0.)
    # The world cell stays on the forward corner's path during this held
    # motion. Changing lateral separation by one original cell makes contact.
    cells = np.isclose(local_path.FORWARD, .10 if not body else .30) & np.isclose(local_path.RIGHT, right)
    scan.obstacles.remember(scan._footprint_origin(old, geometry), cells)
    scan.project_history(scan._footprint_origin(pose, geometry))
    if scheduled:
        card._stage_command(message, card._output_epoch)
    else:
        card._publish_command(message)
    assert len(packets) == 1
    assert packets[0]["values"] == (message["values"] if accepted else [0.] * 6)
    assert bool(any(packets[0]["values"])) is accepted


@pytest.mark.parametrize("scheduled", [False, True], ids=["direct", "scheduled"])
@pytest.mark.parametrize("change,blocked", [
    ("clear-depth", False), ("new-post", True), ("translated-view", False), ("rotated-view", False),
    ("feedback-clear", False), ("feedback-shadow", True),
])
def test_first_send_rechecks_visibility_of_the_complete_trace(monkeypatch, scheduled, change, blocked):
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    if not scheduled:
        card._command_timer = None
    config = card._config
    config.half_width_m = .06
    config.dwa_horizon_s = 3.
    config.local_geometry = replace(config.local_geometry, rear=.12)
    _, depth, odom, _ = snapshot[0]
    feedback_only = change.startswith("feedback-")
    initial_vy = .65 if feedback_only else .50
    odom.update(vx=0., vy=initial_vy, velocity_history=((1000, 1000, (0., initial_vy, 0.)),))
    if feedback_only:
        depth["map"][220:370, 310:330] = .15
    if change in ("translated-view", "rotated-view"):
        odom["obstacle_pose"] = depth["obstacle_capture_pose"] = PlanarPose(1000, ("odom",), 0., 0., 0.)
    message = card.next_command()
    assert message["values"][0] > 0.
    if feedback_only:
        # Identical detector and depth exposure. Only the new measured velocity changes the path.
        clock[0] = 1010
        odom = dict(odom, received_ms=1010, source_sample_id=1010)
        objects, _, _, target_ms = snapshot[0]
        snapshot[0] = objects, depth, odom, target_ms
    else:
        advance_snapshot(snapshot, clock, 1010, vx=0.)
        _, depth, odom, _ = snapshot[0]
        if change != "clear-depth":
            depth["map"][220:370, 310:330] = .15
        if change in ("translated-view", "rotated-view"):
            # The pose at capture is distinct from the current grid origin.
            depth["capture_ms"] = 1000
            left, yaw = (.05, 0.) if change == "translated-view" else (0., .2)
            odom["obstacle_pose"] = PlanarPose(1010, ("odom",), 0., left, yaw)
    final_vy = initial_vy if change == "feedback-clear" else .50
    odom.update(vy=final_vy, velocity_history=(*odom["velocity_history"], (1010, 1010, (0., final_vy, 0.))))
    if scheduled:
        card._stage_command(message, card._output_epoch)
    else:
        card._publish_command(message)
    packet, = packets
    assert packet["values"] == ([0.] * 6 if blocked else message["values"])
    if blocked:
        assert card._dwa_publication["reason"] == "occluded_prefix"
    else:
        assert packet["stamp_ms"] + packet["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]
