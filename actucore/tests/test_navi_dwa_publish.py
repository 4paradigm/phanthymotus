"""DWA provenance at the actual JSON publisher, including queued renewals."""
import json
import logging
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from plugins.navi import dwa, local_path, plugin, policy
from test_navi_dwa_integration import advance_case, card_for_case, observation_case, observe_selector


def observation_card(monkeypatch, *, body=True, live_target=False):
    config, state, depth, odom = observation_case(body=body)
    objects = None
    if live_target:
        objects = {"timestamp": 1., "count": 1, "objects": [
            {"name": "person", "position": [0., -.6], "confidence": .95}]}
        # The measured target lies on the 5 m background above the 2 m
        # obstacle. Confirm it through real earlier frames, retaining the
        # production three-hit requirement rather than injecting a Track.
        state.goal_anchor = policy.DetourGoal(5., 0.)
        for stamp in (800, 900):
            state.tracker.step(
                detections=dict(objects, timestamp=stamp / 1000),
                depth=dict(depth, frame_id=stamp, capture_ms=stamp,
                           control_ms=stamp, source_frame_id=stamp),
                target=state.target, config=config, ego=(0., 0., 0.),
                ego_measured=True, dt=.1)
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


def test_renewals_keep_the_original_decision_and_full_tail_check(monkeypatch):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    original = card._last_decision.dwa_acceptance
    assert card._stage_command(message, card._output_epoch)
    assert card._scheduled_command.forward_check is not None
    assert len(packets) == 1
    horizons = []
    sweep = local_path.motion_sweep_hits

    def checked_sweep(occupied, geometry, half_width, horizon, velocities, **kwargs):
        horizons.append(horizon)
        return sweep(occupied, geometry, half_width, horizon, velocities, **kwargs)

    monkeypatch.setattr(local_path, "motion_sweep_hits", checked_sweep)
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
    assert horizons and min(horizons) >= original.guard_horizon_s
    assert card._info()["dwa"]["publication"]["decision_id"] == original.decision_id


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
        assert card._info()["dwa"]["publication"]["reason"] == "insufficient_room"
    else:
        accepted = [choice for _, choice in calls if choice is not None][-1]
        assert len(packets) == 1
        assert tuple(packets[0]["values"][i] for i in (0, 1, 5)) == accepted.command
        assert card._state.dwa_totals["published_packets"] == 1


def test_renewal_stops_on_fresh_measured_overspeed(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100, vx=.36)
    assert policy._feedback_speed_reduction(card._config, snapshot[0][2]) < 1.
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    assert card._scheduled_command is None
    assert card._state.dwa_totals["published_packets"] == 1
    assert card._info()["dwa"]["publication"]["reason"] == "measured_overspeed"


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
        state.goal_anchor = policy.DetourGoal(5., 0.)
        for stamp in (900, 1000):
            state.tracker.step(detections=dict(payload, timestamp=stamp / 1000),
                               depth=dict(depth, frame_id=stamp, source_frame_id=stamp,
                                          capture_ms=stamp, control_ms=stamp),
                               target=state.target, config=config, ego=(0., 0., 0.),
                               ego_measured=True, dt=.1)
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    clock = [1100]
    snapshot = [(payload, depth, odom, 1100)]
    monkeypatch.setattr(plugin.time, "time", lambda: clock[0] / 1000)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: clock[0] / 1000)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    assert message["values"][0] == pytest.approx(.075)
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
            card._config.vx_max = .85
        message = card.next_command()
        assert card._last_decision.dwa_acceptance is not None
        advance_snapshot(snapshot, clock, 1010, vx=-.64 if constraint == "grid_bounds" else .1)
        if constraint == "stop_line":
            snapshot[0][1]["map"][220:370, 302:338] = .75
            assert policy._clearance(snapshot[0][1], card._config)[0] < card._config.obstacle_stop_m
        else:
            assert policy._feedback_speed_reduction(card._config, snapshot[0][2]) == 1.
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert packets and all(packet["values"] == [0.] * 6 for packet in packets)
    assert card._state.dwa_totals.get("published_packets", 0) == 0


def test_renewal_deducts_travel_from_the_original_finite_segment_room(monkeypatch):
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
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "insufficient_room"


def test_fresh_stop_line_uses_the_translated_body_corridor(monkeypatch):
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
    fresh["map"][220:370, 555:590] = .75
    assert policy._clearance(fresh, config)[0] == 5.
    _, occupied = local_path.project_observation(fresh, geometry, require_valid_depth=True)
    corridor = occupied & (local_path.FORWARD >= 0.) & (abs(local_path.RIGHT) <= policy._keepout_m(config))
    assert local_path.FORWARD[corridor].min() < config.obstacle_stop_m
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "insufficient_room"


def test_fresh_feedback_rechecks_its_complete_modeled_braking_tail(monkeypatch):
    config, state, depth, odom = advance_case(body=False)
    config.vx_max = 3.
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
    assert policy._feedback_speed_reduction(config, snapshot[0][2]) == 1.
    predicted = dwa._predict(accepted.command, (2., 0., 0.), horizon=1.2, dt=.04,
                             acceleration=(.5, .5, 1.), braking=(.45, .45, .8), stop=True)
    assert predicted.duration > accepted.guard_horizon_s
    assert predicted.poses[-1][0] > accepted.motion_guard.room_m
    assert 5. - predicted.poses[-1][0] < config.stop_distance_m
    card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == [0.] * 6


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
