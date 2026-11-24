"""A DWA renewal needs actual evidence for the requested target."""
from copy import deepcopy

import pytest

from test_navi_dwa_integration import observe_selector
from test_navi_dwa_publish import advance_snapshot, observation_card, selected_card


def absent_target_payload(kind, stamp=1000):
    objects = ([] if kind == "empty" else
               [{"name": "chair", "position": [0., -.6], "confidence": .95}])
    return {"timestamp": stamp / 1000, "count": len(objects), "objects": objects}


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("kind", ["empty", "unrelated"])
def test_payload_without_requested_target_cannot_make_memory_motion_renewable(monkeypatch, body, kind):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (absent_target_payload(kind), depth, odom, 1000)
    message = card.next_command()
    assert calls and any(choice is not None for _, choice in calls)
    assert card._last_decision.dwa_acceptance is not None
    assert card._state.tracker.track is None
    assert card._state.goal_anchor.target_missing
    guard = card._staged_dwa_command[1].forward_check
    assert guard is not None and not guard.renewable
    assert guard.target_ms == 0, "A detector receipt cannot become a target observation"
    if body:
        card._publish_command(message)
    else:
        assert card._stage_command(message, card._output_epoch)
        for stamp in (1100, 1200, 1300):
            advance_snapshot(snapshot, clock, stamp)
            _, fresh_depth, fresh_odom, _ = snapshot[0]
            snapshot[0] = (absent_target_payload(kind, stamp), fresh_depth, fresh_odom, stamp)
            card._dispatch_command()
        assert packets[-1]["values"] == [0.] * 6
    nonzero = [packet for packet in packets if any(packet["values"])]
    assert len(nonzero) == 1 and nonzero[0]["values"] == message["values"]
    assert nonzero[0]["stamp_ms"] + nonzero[0]["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]


@pytest.mark.parametrize("kind", ["empty", "unrelated"])
def test_targetless_payload_preserves_sent_motion_only_until_its_fixed_deadline(monkeypatch, kind):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    track = card._state.tracker.track
    assert track.observed and track.accepted_detection and track.measured_range_m == 5.
    assert card._stage_command(message, card._output_epoch)
    assert card._scheduled_command.forward_check.renewable
    deadline = message["stamp_ms"] + message["ttl_ms"]
    assert deadline == 1250
    for stamp in (1100, 1150, 1200, 1249):
        advance_snapshot(snapshot, clock, stamp)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = (absent_target_payload(kind, stamp), depth, odom, stamp)
        card._dispatch_command()
        assert all(any(packet["values"]) for packet in packets)
        assert card._scheduled_command.expires_ms == deadline
        assert card._scheduled_command.forward_check.motion_guard.deadline_ms == deadline
    advance_snapshot(snapshot, clock, deadline)
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    assert len([packet for packet in packets if any(packet["values"])]) == 1
    assert card._scheduled_command is None
    assert len(calls) == card._state.dwa_totals["select_calls"]


def test_targetless_timer_tick_does_not_insert_zero_before_legal_coasting_replan(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (absent_target_payload("empty", 1100), depth, odom, 1100)
    card._dispatch_command()
    clock[0] = 1150
    next_message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    assert next_message["values"][0] > 0.
    assert card._stage_command(next_message, card._output_epoch)
    assert len(packets) == 2
    assert all(packet["values"][0] > 0. for packet in packets)


@pytest.mark.parametrize("event,stop_reason", [
    ("clear", None),
    ("obstacle", "footprint_collision"),
    ("stranger", None),
    ("generation", "superseded"),
    ("action", "superseded"),
    ("epoch", "superseded"),
    ("expired", "publish_expired"),
    ("front_requirement", None),
    ("source_regression", "target_missing"),
])
def test_timer_during_coasting_selection_keeps_only_a_still_valid_sent_command(monkeypatch, event, stop_reason):
    from plugins.navi import policy

    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    objects = snapshot[0][0]
    assert card._stage_command(message, card._output_epoch)
    original = card._scheduled_command
    assert original.expires_ms == 1250
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = absent_target_payload("empty", 1100), depth, odom, 1100
    select = policy._dwa_detour
    during = []

    def dispatch_during_selection(*args, **kwargs):
        if not during:
            track = card._state.tracker.track
            assert not track.observed and track.generation == 1
            assert card._continuing_dwa_output(original.dwa_origin)
            if event == "obstacle":
                advance_snapshot(snapshot, clock, 1101, obstacle=True)
            elif event == "stranger":
                _, latest_depth, latest_odom, _ = snapshot[0]
                person = dict(objects["objects"][0], position=[.98, -.6])
                snapshot[0] = dict(objects, timestamp=1.1, objects=[person]), latest_depth, latest_odom, 1100
            elif event == "generation":
                track.generation += 1
            elif event == "action":
                card._acp_action_id = "replacement-action"
            elif event == "epoch":
                card._output_epoch += 1
            elif event == "expired":
                clock[0] = original.expires_ms
            elif event == "front_requirement":
                track.front_required = True
            elif event == "source_regression":
                payload, latest_depth, latest_odom, receipt = snapshot[0]
                snapshot[0] = dict(payload, timestamp=.9), latest_depth, latest_odom, receipt
            card._dispatch_command()
            during.append(dict(card._dwa_publication))
            if stop_reason is None:
                assert card._scheduled_command is original
                assert original.expires_ms == original.forward_check.motion_guard.deadline_ms == 1250
        return select(*args, **kwargs)

    monkeypatch.setattr(policy, "_dwa_detour", dispatch_during_selection)
    next_message = card.next_command()
    assert len(during) == 1, "The timer must run after tracking and before the real DWA selector"
    if event in ("obstacle", "source_regression"):
        assert during[0]["event"] == "published" and len(packets) == 1
        assert card._stage_command(next_message, card._output_epoch)
        assert len(packets) == 2
        if event == "obstacle":
            assert packets[-1]["values"] == [0.] * 6
        else:
            # The replacement was planned against the coasting goal.
            assert packets[-1]["values"] == next_message["values"]
    elif stop_reason is None:
        assert card._last_decision.dwa_acceptance is not None
        assert card._stage_command(next_message, card._output_epoch)
        assert len(packets) == 2 and all(packet["values"][0] > 0. for packet in packets)
    else:
        assert packets[-1]["values"] == [0.] * 6
        assert during[0]["reason"] == stop_reason


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_unpublished_observed_command_cannot_coast_after_tracker_consumes_a_miss(monkeypatch, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    payload = absent_target_payload("empty", 1100)
    snapshot[0] = payload, depth, odom, 1100
    track = card._state.tracker.step(detections=payload, depth=depth, target=card._state.target,
                                     config=card._config, ego=(.1, 0., 0.), dt=.1)
    assert track.generation == 1 and not track.observed
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert all(packet["values"] == [0.] * 6 for packet in packets)
    assert card._state.dwa_totals.get("published_packets", 0) == 0
    assert card._dwa_publication["reason"] == "superseded"


def test_original_target_reappearance_waits_for_a_new_plan(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    objects = snapshot[0][0]
    original_target_ms = card._staged_dwa_command[1].forward_check.target_ms
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (absent_target_payload("empty", 1100), depth, odom, 1100)
    card._dispatch_command()
    advance_snapshot(snapshot, clock, 1150)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (dict(objects, timestamp=1.15), depth, odom, 1150)
    card._dispatch_command()
    assert [p["values"] for p in packets] == [message["values"]]
    assert card._scheduled_command.expires_ms == 1250
    assert card._scheduled_command.forward_check.target_ms == original_target_ms


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_first_send_still_requires_the_observed_original_target(monkeypatch, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (absent_target_payload("empty", 1100), depth, odom, 1100)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert all(packet["values"] == [0.] * 6 for packet in packets)
    assert card._state.dwa_totals.get("published_packets", 0) == 0
    assert card._dwa_publication["reason"] == "target_missing"


@pytest.mark.parametrize("change", ["obstacle", "replacement", "front_requirement"])
def test_sent_target_coast_checks_obstacles_and_holds_its_fixed_window(monkeypatch, change):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    objects = snapshot[0][0]
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100, obstacle=change == "obstacle")
    _, depth, odom, _ = snapshot[0]
    payload = absent_target_payload("empty", 1100)
    if change == "replacement":
        person = dict(objects["objects"][0], position=[.98, -.6])
        payload = dict(objects, timestamp=1.1, objects=[person])
    elif change == "front_requirement":
        card._state.tracker.track.front_required = True
    snapshot[0] = payload, depth, odom, 1100
    card._dispatch_command()
    assert [packet["values"] for packet in packets] == [message["values"]]
    assert card._scheduled_command.expires_ms == message["stamp_ms"] + message["ttl_ms"]


def test_repeated_matching_source_keeps_its_original_target_deadline(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    original_target_ms = snapshot[0][3]
    original_payload = snapshot[0][0]
    target_deadline = original_target_ms + min(card._config.max_obs_age_ms,
                                              card._descriptor["rate"]["max_obs_age_ms"])
    assert card._stage_command(message, card._output_epoch)
    for stamp in (1100, 1200, 1300, 1400):
        advance_snapshot(snapshot, clock, stamp)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = (original_payload, depth, odom, stamp)
        card._dispatch_command()
    nonzero = [packet for packet in packets if any(packet["values"])]
    assert len(nonzero) == 1
    assert all(packet["values"] == message["values"] for packet in nonzero)
    assert all(packet["stamp_ms"] + packet["ttl_ms"] <= target_deadline for packet in nonzero)
    assert card._scheduled_command is None
    advance_snapshot(snapshot, clock, target_deadline)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (original_payload, depth, odom, target_deadline)
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("kind", ["duplicate", "reordered", "older-unseen", "missing"])
def test_next_decision_cannot_promote_old_track_to_fresh_target_evidence(monkeypatch, body, kind):
    card, _, clock, snapshot, _ = selected_card(monkeypatch, body=body, live_target=True)
    assert card._state.tracker.track.accepted_detection
    advance_snapshot(snapshot, clock, 1100)
    payload, depth, odom, _ = snapshot[0]
    if kind == "reordered":
        payload = dict(payload, timestamp=.9)
    elif kind == "older-unseen":
        payload = dict(payload, timestamp=.95)
    elif kind == "missing":
        payload = None
    snapshot[0] = (payload, depth, odom, 1100)
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None and any(message["values"])
    track = card._state.tracker.track
    assert not track.accepted_detection and track.measured_range_m is None
    guard = card._staged_dwa_command[1].forward_check
    assert guard is not None and not guard.renewable
    assert guard.target_ms == 0


@pytest.mark.parametrize("confidence", [.95, .87], ids=["unchanged", "confidence-jitter"])
def test_matching_target_preserves_the_sent_packet_without_reissue(monkeypatch, confidence):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    track = card._state.tracker.track
    assert track.observed and track.accepted_detection and track.measured_range_m == 5.
    assert track.navigation_range_known
    assert card._state.goal_anchor.generation == track.generation and not card._state.goal_anchor.target_missing
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100)
    payload, depth, odom, _ = snapshot[0]
    objects = [dict(payload["objects"][0], confidence=confidence)]
    snapshot[0] = (dict(payload, timestamp=1.1, objects=objects), depth, odom, 1100)
    card._dispatch_command()
    assert [p["values"] for p in packets] == [message["values"]]
    assert card._state.dwa_totals["published_decisions"] == 1
    assert card._state.dwa_totals["published_packets"] == 1
    assert len(calls) == card._state.dwa_totals["select_calls"]


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_a_different_person_cannot_replace_the_accepted_target_before_first_send(monkeypatch, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    advance_snapshot(snapshot, clock, 1010)
    payload, depth, odom, _ = snapshot[0]
    # Even mutation of the source dict cannot rewrite captured acceptance.
    changed = payload["objects"][0]
    changed["position"] = [.98, -.6]
    payload = dict(payload, timestamp=1.01, objects=[changed])
    snapshot[0] = (payload, depth, odom, 1010)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == [0.] * 6
    assert not any(any(packet["values"]) for packet in packets)


def test_a_different_person_cannot_renew_the_original_target_command(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    original_tracker = deepcopy(card._state.tracker)
    assert card._stage_command(message, card._output_epoch)
    for stamp in (1100, 1200, 1300):
        advance_snapshot(snapshot, clock, stamp)
        payload, depth, odom, _ = snapshot[0]
        changed = dict(payload["objects"][0], position=[.98, -.6])
        payload = dict(payload, timestamp=stamp / 1000, objects=[changed])
        snapshot[0] = (payload, depth, odom, stamp)
        # The actual association rule rejects this other member of the class.
        probe = deepcopy(original_tracker)
        probe.step(detections=payload, depth=depth, target=card._state.target,
                   config=card._config, ego=(.1, 0., 0.), ego_measured=True, dt=(stamp - 1000) / 1000)
        assert not probe.track.accepted_detection and probe.track.measured_range_m is None
        card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    nonzero = [packet for packet in packets if any(packet["values"])]
    assert len(nonzero) == 1
    assert nonzero[0]["stamp_ms"] + nonzero[0]["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]


@pytest.mark.parametrize("retain_original", [True, False], ids=["original-present", "original-lost"])
def test_mixed_objects_keep_the_observation_actually_associated_by_the_planner(monkeypatch, retain_original):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    payload, depth, odom, stamp = snapshot[0]
    original = payload["objects"][0]
    other = dict(original, position=[.98, -.6], confidence=.99)
    snapshot[0] = (dict(payload, count=2, objects=[other, original]), depth, odom, stamp)
    message = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    assert card._state.tracker.track.accepted_detection
    assert card._state.tracker.track.last_obs[1] == pytest.approx(0.)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    objects = [original, other] if retain_original else [other]
    snapshot[0] = (dict(payload, timestamp=1.1, count=len(objects), objects=objects), depth, odom, 1100)
    card._dispatch_command()
    assert all(packet["values"] == message["values"] for packet in packets)
    assert len(packets) == 1
    assert card._scheduled_command.expires_ms == message["stamp_ms"] + message["ttl_ms"]


def coasting_person_case(monkeypatch, body, *, detached_anchor=False):
    from plugins.navi import policy
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body)
    state, config = card._state, card._config
    _, depth, _, _ = snapshot[0]
    config.depth_intrinsics = config.local_geometry.horizontal
    person = {"name": "person", "position": [0., -.6], "confidence": .95,
              "bbox_norm": [.45, .10, .55, .30]}
    for stamp in (700, 800, 900):
        state.tracker.step(detections={"timestamp": stamp / 1000, "objects": [person]},
                           depth=dict(depth, frame_id=stamp, capture_ms=stamp, source_frame_id=stamp),
                           target=state.target, config=config, ego=(0., 0., 0.), ego_measured=True, dt=.1)
    track = state.tracker.track
    assert track.observed and track.front_required and track.front_fix is not None
    generation = track.generation - 1 if detached_anchor else track.generation
    state.goal_anchor = policy.DetourGoal(*track.front_fix, generation=generation)
    message = card.next_command()
    assert track.state == "coasting" and state.goal_anchor.target_missing
    check = card._staged_dwa_command[1].forward_check
    assert check.front_required and not check.renewable and check.planning_guard.association is None
    return card, packets, clock, snapshot, message


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("blocked", [False, True], ids=["clear", "obstacle"])
def test_coasting_person_uses_remembered_front_within_original_deadline(monkeypatch, body, blocked):
    card, packets, clock, snapshot, message = coasting_person_case(monkeypatch, body)
    deadline = message["stamp_ms"] + message["ttl_ms"]
    objects, depth, odom, target_ms = snapshot[0]
    clock[0] = 1010
    snapshot[0] = (objects, depth, dict(odom, source_sample_id=1010, received_ms=1010), target_ms)
    if blocked:
        advance_snapshot(snapshot, clock, 1010, obstacle=True)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    if blocked:
        assert packets[-1]["values"] == [0.] * 6
        assert card._info()["dwa"]["publication"]["reason"] == "footprint_collision"
        return
    assert packets[-1]["values"] == message["values"]
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] == deadline
    clock[0] = deadline
    if body:
        card._publish_command(message)
    else:
        card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6


def test_coasting_person_timer_leaves_new_feedback_and_hits_for_planner(monkeypatch):
    card, packets, clock, snapshot, message = coasting_person_case(monkeypatch, False)
    card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"] == message["values"]
    objects, depth, odom, target_ms = snapshot[0]
    clock[0] = 1010
    snapshot[0] = (objects, depth, dict(odom, received_ms=1010, source_sample_id=1010), target_ms)
    card._dispatch_command()
    assert len(packets) == 1 and card._wire_moving
    advance_snapshot(snapshot, clock, 1020, obstacle=True)
    card._dispatch_command()
    assert len(packets) == 1 and card._wire_moving


@pytest.mark.parametrize("event", ["clear", "tracker_first", "detached_anchor", "changed_anchor",
                                  "no_front", "near", "near_new_depth", "obstacle", "expiry", "cancel"])
def test_sent_memory_command_rechecks_physics_during_target_reappearance(monkeypatch, event):
    card, packets, clock, snapshot, message = coasting_person_case(
        monkeypatch, False, detached_anchor=event == "detached_anchor")
    assert card._stage_command(message, card._output_epoch)
    work = card._scheduled_command
    epoch, deadline = card._output_epoch, work.expires_ms
    before = card._state.goal_anchor
    goal = before.x, before.y, before.generation
    advance_snapshot(snapshot, clock, 1010, obstacle=event == "obstacle")
    _, depth, odom, _ = snapshot[0]
    person = {"name": "person", "position": [0., -.6], "confidence": .95,
              "bbox_norm": [.45, .10, .55, .30]}
    if event == "near":
        # The high image region isolates person spacing from the obstacle grid.
        depth["map"][48:144, 288:352] = card._config.stop_distance_m + .03
    elif event == "no_front":
        del person["bbox_norm"]
    objects = {"timestamp": 1.01, "objects": [person]}
    snapshot[0] = objects, depth, odom, 1010
    if event == "changed_anchor":
        before.generation += 1
    if event == "tracker_first":
        state = card._state
        for stamp in (1008, 1009, 1010):
            state.tracker.step(detections=dict(objects, timestamp=stamp / 1000), depth=depth,
                               target=state.target, config=card._config, ego=(0., 0., 0.), dt=.01)
        assert state.tracker.track.observed
    if event == "expiry":
        clock[0] = deadline
    if event == "cancel":
        card._stop_navigation()
    card._dispatch_command()
    if event == "near_new_depth":
        assert [p["values"] for p in packets] == [message["values"]]
        advance_snapshot(snapshot, clock, 1020)
        snapshot[0][1]["map"][48:144, 288:352] = card._config.stop_distance_m + .03
        card._dispatch_command()
    if event in ("clear", "tracker_first", "detached_anchor", "no_front"):
        assert [p["values"] for p in packets] == [message["values"]]
        assert card._output_epoch == epoch
        assert card._scheduled_command is work and work.expires_ms == deadline
        assert (before.x, before.y, before.generation) == goal
        clock[0] = 1020
        replacement = card.next_command()
        assert card._stage_command(replacement, epoch)
        assert len(packets) == 2 and all(any(packet["values"]) for packet in packets)
    elif event in ("cancel", "expiry"):
        assert packets[-1]["values"] == [0.] * 6
        if event == "expiry":
            assert card._dwa_publication["reason"] == "publish_expired"
    else:
        assert len(packets) == 1 and card._scheduled_command.expires_ms == deadline


@pytest.mark.parametrize("change", ["new_detection", "dim_detection", "missing_front", "near_detection", "closer_front"])
def test_memory_first_send_checks_reappearing_people(monkeypatch, change):
    import numpy as np
    card, packets, clock, snapshot, message = coasting_person_case(monkeypatch, True)
    _, depth, odom, _ = snapshot[0]
    clock[0] = 1010
    if change != "closer_front":
        # The old memory decision has no accepted image region for this live fix.
        objects = {"timestamp": 1.01, "objects": [{"name": "person",
                   "confidence": .25 if change == "dim_detection" else .95, "bbox_norm": [.45, .11, .55, .31], "position": [0., -.58]}]}
        if change == "missing_front":
            del objects["objects"][0]["bbox_norm"]
        if change == "near_detection":
            depth["map"] = depth["map"].copy()
            depth["map"][:150] = card._config.stop_distance_m
        snapshot[0] = (objects, depth, odom, 1010)
    else:
        card._state.tracker.track.front_fix = np.array([card._config.stop_distance_m + .01, 0.])
    card._publish_command(message)
    if change in ("new_detection", "dim_detection"):
        assert packets[-1]["values"] == message["values"]
    else:
        assert packets[-1]["values"] == [0.] * 6
        assert card._info()["dwa"]["publication"]["reason"] == (
            "target_missing" if change == "missing_front" else "target_standoff")


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("offset,confidence", [(1e-6, .95), (.001, .95), (.01, .95), (.01, .25)])
def test_accepted_target_jitter_can_publish(monkeypatch, body, offset, confidence):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    advance_snapshot(snapshot, clock, 1010)
    payload, depth, odom, _ = snapshot[0]
    person = dict(payload["objects"][0], position=[offset, -.6], confidence=confidence)
    fresh = dict(payload, timestamp=1.01, objects=[person])
    snapshot[0] = fresh, depth, odom, 1010
    probe = deepcopy(card._state.tracker)
    matched = probe.step(detections=fresh, depth=depth, target="person", config=card._config,
                         ego=(.1, 0., 0.), dt=.01)
    assert matched.accepted_detection and matched.measured_range_m == 5.
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert [packet["values"] for packet in packets] == [message["values"]]


@pytest.mark.parametrize("life", ["reconfirmed", "pending", "replaced"])
def test_publication_follows_confirmed_target_generation(monkeypatch, life):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=True, live_target=True)
    tracker = card._state.tracker
    original = tracker.track
    from plugins.navi import policy
    for index, stamp in enumerate((1010, 1020, 1030, 1040)):
        advance_snapshot(snapshot, clock, stamp)
        payload, depth, odom, _ = snapshot[0]
        fresh = dict(payload, timestamp=stamp / 1000)
        if index == 0 and life != "replaced":
            fresh = dict(fresh, objects=[])
        snapshot[0] = fresh, depth, odom, stamp
        dt = card._config.max_coast_s + .01 if index == 0 and life == "replaced" else .01
        policy.step(detections=fresh, depth=depth, odom=odom, dt=dt,
                    config=card._config, state=card._state)
        if index == 0:
            payload = dict(payload, timestamp=stamp / 1000)
            snapshot[0] = payload, depth, odom, stamp
        if life == "pending" and index == 1:
            break
    assert (tracker.track is original) == (life != "replaced")
    assert tracker.track.observed == (life != "pending")
    card._publish_command(message)
    assert packets[-1]["values"] == (message["values"] if life == "reconfirmed" else [0.] * 6)


@pytest.mark.parametrize("method", ["sample", "turn"])
@pytest.mark.parametrize("stop", [4.6, 5.05], ids=["clear", "standoff"])
def test_measured_target_limits_motion_independently_of_fixed_anchor(monkeypatch, method, stop):
    from plugins.navi import policy
    card, _, _, snapshot, _ = selected_card(monkeypatch, body=True, live_target=True)
    _, depth, _, _ = snapshot[0]
    state, config = card._state, card._config
    anchor = state.goal_anchor
    anchor.x = 5.6  # Original destination; the same person is now measured at 5 m.
    assert state.tracker.track.measured_range_m == 5.
    config.stop_distance_m = stop
    geometry = config.scan_geometry
    if method == "turn":
        accepted = policy._check_dwa_turn(depth, config, state, geometry, (anchor.x, 0.), "parking", (0., 0., .25))
    else:
        state.dwa_budget = None
        outcome = policy._dwa_detour(depth, config, state, state.view_detour, anchor.x, 0.)
        accepted = outcome is not None
    assert accepted == (stop < 5.)
    assert anchor.x == 5.6


@pytest.mark.parametrize("body", [True, False], ids=["body-pose", "camera-twist-history"])
def test_association_uses_the_motion_at_the_detector_observation(monkeypatch, body):
    import math
    import numpy as np
    from plugins.navi.odom import PlanarPose

    card, _, _, snapshot = observation_card(monkeypatch, body=body, live_target=True,
                                            mount_forward=.2 if body else None)
    objects, depth, odom, receipt = snapshot[0]
    initial = PlanarPose(1000, ("odom",), 0., 0., 0.)
    odom = dict(odom, vx=0., velocity_history=((1000, 1000, (0., 0., 0.)),))
    if body:
        odom["obstacle_pose"] = initial
    snapshot[0] = objects, depth, odom, receipt
    card.next_command()
    check = card._staged_dwa_command[1].forward_check
    assert check.planning_guard.association.mean[:2] == pytest.approx((5., 0.))
    card._config.gate_chi2 = .00001  # separate a properly transported fix from a double transform
    yaw = .04
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = 5.2 - .02, -.03
    camera_x, camera_y = c * x + s * y - .2, -s * x + c * y
    distance = math.hypot(camera_x, camera_y)
    bearing = math.atan2(-camera_y, camera_x) / card._config.half_fov_rad
    fresh = dict(objects, timestamp=1.1, objects=[dict(objects["objects"][0], position=[bearing, -.6])])
    measured = dict(depth, map=np.full_like(depth["map"], distance), capture_ms=1100, frame_id=1100)
    if body:
        odom = dict(odom, obstacle_pose=PlanarPose(1100, initial.frame, .02, .03, yaw))
    else:
        odom = dict(odom, velocity_history=((1050, 1050, (.4, .6, 0.)), (1100, 1100, (0., 0., .8))))
    result = card._dwa_target_match(check, fresh, measured, odom, 1100, check.geometry, False)
    assert result is not None and result.point == pytest.approx((camera_x, camera_y), abs=.0005)
    # Feedback advanced, but this old detector box is still in the first image.
    same = card._dwa_target_match(check, objects, depth, odom, 1100, check.geometry, False)
    assert same is not None and same.point == pytest.approx((5., 0.))
    assert card._dwa_target_match(check, dict(objects, timestamp=1.1), depth, odom, 1100,
                                  check.geometry, False) is None


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("distance,duplicate", [(4.60, False), (5.2, False), (4.40, True), (4.83, True), (5.2, True)],
                         ids=["arrived", "clear", "duplicate-inside", "duplicate-braking", "duplicate-clear"])
def test_memory_first_send_checks_new_same_generation_range(monkeypatch, body, distance, duplicate):
    from plugins.navi import policy

    # The previously confirmed target outlives its depth budget. There is
    # genuinely no physical measurement cached on the original memory guard.
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True,
                                                       confirm_stamps=(100, 150, 190))
    state, config = card._state, card._config
    state.goal_anchor.x = 8.
    config.stop_distance_m = 4.65
    payload, depth, odom, receipt = snapshot[0]
    snapshot[0] = dict(payload, timestamp=.19), depth, odom, receipt
    message = card.next_command()
    guard = card._staged_dwa_command[1].forward_check
    generation = state.tracker.track.generation
    assert guard.target_ms == 0 and not guard.renewable and message["values"][0] == pytest.approx(config.vx_max)
    assert guard.planning_guard.measured_target is None
    advance_snapshot(snapshot, clock, 1010)
    _, depth, odom, _ = snapshot[0]
    fresh = dict(payload, timestamp=1.01)
    depth["map"][:150] = distance
    snapshot[0] = fresh, depth, odom, 1010
    decision = policy.step(detections=fresh, depth=depth, odom=odom, state=state, config=config, dt=.01)
    assert state.tracker.track.generation == generation and state.tracker.track.measured_range_m == distance
    if distance == 4.60:
        assert decision.status == policy.ARRIVED and not any(decision.values)
    if duplicate:
        advance_snapshot(snapshot, clock, 1020)
        _, latest_depth, latest_odom, _ = snapshot[0]
        snapshot[0] = fresh, latest_depth, latest_odom, 1020
        duplicate_decision = policy.step(detections=fresh, depth=latest_depth, odom=latest_odom,
                                         state=state, config=config, dt=.01)
        assert state.tracker.track.generation == generation and not state.tracker.track.accepted_detection
        assert state.tracker.track.measured_range_m is None
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["values"] == (message["values"] if distance == 5.2 else [0.] * 6)
    if duplicate and distance == 4.40:
        assert duplicate_decision.dwa_acceptance is None
    if any(packet["values"]):
        assert packet["stamp_ms"] + packet["ttl_ms"] <= guard.motion_guard.deadline_ms
        assert not card._wire_dwa_origin.forward_check.renewable
        if duplicate:
            advance_snapshot(snapshot, clock, guard.motion_guard.deadline_ms)
            if body:
                card._publish_command(message)
            else:
                card._dispatch_command()
            assert packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("source", [1.01, 1.02, 1.03], ids=["delayed", "same", "newer"])
def test_first_send_respects_source_consumed_after_selection(monkeypatch, body, source):
    from plugins.navi import policy

    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    check = card._staged_dwa_command[1].forward_check
    assert check.target_source == 1.
    advance_snapshot(snapshot, clock, 1020)
    objects, depth, odom, _ = snapshot[0]
    fresh = dict(objects, timestamp=1.02)
    policy.step(detections=fresh, depth=depth, odom=odom, state=card._state, config=card._config, dt=.02)
    assert card._state.tracker.track.source == 1.02
    assert card._state.tracker.track.generation == check.planning_guard.generation
    advance_snapshot(snapshot, clock, 1030)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=source), depth, odom, 1030
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["values"] == ([0.] * 6 if source < 1.02 else message["values"])
    if any(packet["values"]):
        assert card._wire_dwa_origin.forward_check.target_ms == 1000


def test_unstamped_update_cannot_erase_a_generations_source_order(monkeypatch):
    from plugins.navi import policy

    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=True, live_target=True)
    for stamp, source in ((1020, 1.02), (1025, None)):
        advance_snapshot(snapshot, clock, stamp)
        objects, depth, odom, _ = snapshot[0]
        payload = dict(objects, timestamp=source)
        policy.step(detections=payload, depth=depth, odom=odom, state=card._state,
                    config=card._config, dt=.005)
        assert card._state.tracker.track.accepted_detection
    advance_snapshot(snapshot, clock, 1030)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.01), depth, odom, 1030
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("duplicate", [False, True], ids=["fresh", "duplicate"])
def test_memory_range_is_registered_from_its_depth_capture_pose(monkeypatch, duplicate):
    from plugins.navi import policy
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=True, live_target=True)
    state, config = card._state, card._config
    state.goal_anchor.x = 8.
    config.stop_distance_m = 4.65
    payload, depth, odom, receipt = snapshot[0]
    start = PlanarPose(1000, ("odom",), 0., 0., 0.)
    snapshot[0] = dict(payload, timestamp=.9), depth, dict(odom, obstacle_pose=start), receipt
    message = card.next_command()
    assert message["values"][0] == pytest.approx(.15)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    capture = PlanarPose(1010, start.frame, .01, 0., 0.)
    current = PlanarPose(1100, start.frame, .16, 0., 0.)
    depth = dict(depth, capture_ms=1010, obstacle_capture_pose=capture)
    depth["map"][:150] = 4.98
    odom = dict(odom, obstacle_pose=current)
    fresh = dict(payload, timestamp=1.01)
    snapshot[0] = fresh, depth, odom, 1100
    policy.step(detections=fresh, depth=depth, odom=odom, state=state, config=config, dt=.1)
    assert state.tracker.track.measured_range_m == pytest.approx(4.98)
    if duplicate:
        advance_snapshot(snapshot, clock, 1110)
        _, latest_depth, latest_odom, _ = snapshot[0]
        current = PlanarPose(1110, start.frame, .18, 0., 0.)
        latest_depth = dict(latest_depth, obstacle_capture_pose=current)
        latest_odom = dict(latest_odom, obstacle_pose=current)
        snapshot[0] = fresh, latest_depth, latest_odom, 1110
        policy.step(detections=fresh, depth=latest_depth, odom=latest_odom,
                    state=state, config=config, dt=.01)
        assert state.tracker.track.measured_range_m is None
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    assert card._info()["dwa"]["publication"]["reason"] == "target_standoff"


def test_duplicate_does_not_extend_captured_point_lifetime(monkeypatch):
    card, _, clock, snapshot = observation_card(monkeypatch, body=True, live_target=True)
    card._config.max_depth_age_ms = 200
    card.next_command()
    track = card._state.tracker.track
    assert track.measured_range_m == 5.
    for stamp in (1100, 1199, 1200):
        advance_snapshot(snapshot, clock, stamp)
        card.next_command()
        guard = card._staged_dwa_command[1].forward_check
        assert track.measured_range_m is None
        assert not guard.renewable and guard.target_ms == 0
        # The camera still receives depth frames; only the first one measured
        # this detection. A new planner decision may use that fix as a veto.
        assert (guard.motion_guard.measured_target is not None) == (stamp < 1200)


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("x,confidence,matched", [(.98, .95, False), (.01, .25, True)])
def test_first_send_matches_observed_target_without_age_declaration(monkeypatch, body, x, confidence, matched):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True)
    del card._descriptor["rate"]["max_obs_age_ms"]
    message = card.next_command()
    assert message["values"][0] > 0.
    initial = card._staged_dwa_command[1].forward_check
    assert not initial.renewable and initial.planning_guard.association is not None
    advance_snapshot(snapshot, clock, 1010)
    objects, depth, odom, _ = snapshot[0]
    person = dict(objects["objects"][0], position=[x, -.6], confidence=confidence)
    snapshot[0] = dict(objects, timestamp=1.01, objects=[person]), depth, odom, 1010
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["values"] == (message["values"] if matched else [0.] * 6)
    assert card._state.tracker.track.latest_source == 1.
    if matched:
        checked = card._wire_dwa_origin.forward_check
        assert checked.target_source == 1.01
        assert checked.target_ms == initial.target_ms
        assert checked.motion_guard.deadline_ms <= initial.motion_guard.deadline_ms
        assert packet["stamp_ms"] + packet["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]
    else:
        assert card._dwa_publication["reason"] == "target_missing"


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("update,matched", [("unchanged", True), ("jitter", True), ("unrelated", False)])
def test_fixed_window_first_send_with_receipt_only_detector(monkeypatch, body, update, matched):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True,
                                                     detector_timestamp=False)
    del card._descriptor["rate"]["max_obs_age_ms"]
    message = card.next_command()
    initial = card._staged_dwa_command[1].forward_check
    assert message["values"][0] > 0. and initial.planning_guard.association.source is None
    advance_snapshot(snapshot, clock, 1010)
    objects, depth, odom, receipt = snapshot[0]
    if update != "unchanged":
        x = .01 if update == "jitter" else .98
        person = dict(objects["objects"][0], confidence=.25, position=[x, -.6])
        snapshot[0] = dict(objects, objects=[person]), depth, odom, 1010
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["values"] == (message["values"] if matched else [0.] * 6)
    if matched:
        check = card._wire_dwa_origin.forward_check
        assert check.target_ms == receipt and not check.renewable
        assert check.motion_guard.deadline_ms <= initial.motion_guard.deadline_ms
        if not body:
            advance_snapshot(snapshot, clock, initial.motion_guard.deadline_ms)
            card._dispatch_command()
            assert packets[-1]["values"] == [0.] * 6
    else:
        assert card._dwa_publication["reason"] == "target_missing"


@pytest.mark.parametrize("prior_stamp,next_source", [(True, None), (False, None), (False, 1.005)],
                         ids=["producer-history", "new-producer-missing", "new-producer-regression"])
def test_fixed_window_keeps_detector_ordering_after_a_producer_stamp(monkeypatch, prior_stamp, next_source):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True,
                                                     detector_timestamp=prior_stamp)
    del card._descriptor["rate"]["max_obs_age_ms"]
    message = card.next_command()
    assert message["values"][0] > 0.
    advance_snapshot(snapshot, clock, 1010)
    objects, depth, odom, _ = snapshot[0]
    if prior_stamp:
        missing_stamp = dict(objects)
        del missing_stamp["timestamp"]
        snapshot[0] = missing_stamp, depth, odom, 1010
        card._stage_command(message, card._output_epoch)
        assert [p["values"] for p in packets] == [[0.] * 6]
        assert card._dwa_publication["reason"] == "target_missing"
    else:
        snapshot[0] = dict(objects, timestamp=1.01), depth, odom, 1010
        card._stage_command(message, card._output_epoch)
        assert [p["values"] for p in packets] == [message["values"]]
        assert card._scheduled_command.forward_check.target_source == 1.01
        advance_snapshot(snapshot, clock, 1020)
        _, depth, odom, _ = snapshot[0]
        next_objects = objects if next_source is None else dict(objects, timestamp=next_source)
        snapshot[0] = next_objects, depth, odom, 1020
        card._dispatch_command()
        assert [p["values"] for p in packets] == [message["values"]]


@pytest.mark.parametrize("bad_receipt", [1005, 1025], ids=["regressing", "future"])
def test_duplicate_unstamped_receipt_preserves_its_fixed_window_and_order(monkeypatch, bad_receipt):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True,
                                                     detector_timestamp=False)
    del card._descriptor["rate"]["max_obs_age_ms"]
    message = card.next_command()
    original = card._staged_dwa_command[1].forward_check
    objects, depth, odom, _ = snapshot[0]
    # No image or producer stamp changed. Host receipts still have an order.
    clock[0] = 1010
    snapshot[0] = objects, depth, odom, 1010
    card._stage_command(message, card._output_epoch)
    assert [p["values"] for p in packets] == [message["values"]]
    checked = card._scheduled_command.forward_check
    assert checked.target_receipt_ms == 1010 and checked.target_source is None
    assert checked.motion_guard.deadline_ms <= original.motion_guard.deadline_ms
    clock[0] = 1020
    snapshot[0] = objects, depth, odom, bad_receipt
    card._dispatch_command()
    assert [p["values"] for p in packets] == [message["values"]]


@pytest.mark.parametrize("recover", [True, False, "replacement"])
def test_missing_target_veto_hands_off_to_search(monkeypatch, recover):
    card, packets, clock, snapshot, motion = selected_card(monkeypatch, live_target=True)
    original = snapshot[0][0]

    def observe(stamp, objects):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = dict(objects, timestamp=stamp / 1000), depth, odom, stamp

    observe(1010, absent_target_payload("empty"))
    card._publish_command(motion)
    assert card._dwa_publication["reason"] == "target_missing"
    # Search can start on the very snapshot that caused the veto.
    message = card.next_command()
    assert message["values"][:5] == [0.] * 5 and message["values"][5] != 0
    if recover == "replacement":
        # A confirmed track can expire while the card is searching.
        old_generation = card._state.reacquire_generation
        card._state.tracker.track = None
    for stamp in ((1050, 1100, 1150) if recover else (1050,)):
        observe(stamp, original if recover else absent_target_payload("empty"))
        message = card.next_command()
        if message:
            card._publish_command(message)
    twist = packets[-1]["values"]
    if recover == "replacement":
        assert card._state.tracker.track.generation > old_generation
    if recover:
        assert card._state.reacquire_generation is None
        assert twist[0] > 0
    else:
        assert twist[:5] == [0.] * 5 and twist[5] != 0


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("vx,blocked", [(0., False), (.25, False), (.25, True)],
                         ids=["stationary", "still-translating", "blocked-transition"])
def test_tick_checks_search_when_target_lost_before_first_send(monkeypatch, body, vx, blocked):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True)
    plan = card.next_command

    def lose_target_after_plan():
        message = plan()
        if card._last_decision.dwa_acceptance is not None and card._state.reacquire_generation is None:
            advance_snapshot(snapshot, clock, 1010, vx=vx, obstacle=blocked)
            _, depth, odom, _ = snapshot[0]
            snapshot[0] = absent_target_payload("empty", 1010), depth, odom, 1010
        return message

    monkeypatch.setattr(card, "next_command", lose_target_after_plan)
    card._tick()
    assert not card._last_error
    packet, = packets
    assert packet["values"][:5] == [0.] * 5
    if blocked:
        assert packet["values"][5] == 0.
    else:
        assert packet["values"][5] != 0


def test_active_route_hands_off_to_checked_search_without_a_zero(monkeypatch):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    assert card._stage_command(card.next_command(), card._output_epoch)
    assert len(packets) == 1 and packets[0]["values"][0] > 0.
    advance_snapshot(snapshot, clock, 1050)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.05), depth, odom, 1050
    plan = card.next_command

    def lose_target_after_plan():
        message = plan()
        if card._state.reacquire_generation is None:
            assert card._last_decision.dwa_acceptance is not None
            advance_snapshot(snapshot, clock, 1060, vx=.25)
            _, depth, odom, _ = snapshot[0]
            snapshot[0] = absent_target_payload("empty", 1060), depth, odom, 1060
        return message

    monkeypatch.setattr(card, "next_command", lose_target_after_plan)
    card._tick()
    assert not card._last_error
    assert len(packets) == 2
    assert packets[1]["values"][:5] == [0.] * 5 and packets[1]["values"][5] != 0.


@pytest.mark.parametrize("update", ["confirmed", "duplicate", "unrelated", "obstacle",
                                    "stale_depth", "stale_feedback", "missing_feedback"])
def test_slow_plan_refreshes_only_with_new_original_target_and_clear_sweep(monkeypatch, update):
    card, _, clock, snapshot, message = selected_card(monkeypatch, live_target=True)
    check = card._staged_dwa_command[1].forward_check
    objects = snapshot[0][0]
    card._config.max_obs_age_ms = 250
    advance_snapshot(snapshot, clock, 1270, obstacle=update == "obstacle")
    _, depth, odom, _ = snapshot[0]
    payload = dict(objects, timestamp=1.27) if update != "duplicate" else objects
    if update == "unrelated":
        payload = absent_target_payload("unrelated", 1270)
    if update == "stale_depth":
        depth = dict(depth, capture_ms=1270-card._config.depth_age_limit_ms)
    if update == "stale_feedback":
        odom = dict(odom, stamp_ms=1270-card._config.max_feedback_gap_ms)
    if update == "missing_feedback":
        odom = None
    snapshot[0] = payload, depth, odom, 1270
    result = card._recheck_forward(check, message["values"], 800, 1270)
    if update == "confirmed":
        assert result and result[2] and result[1] > 1270
        assert result[3].target_ms == 1270
    else:
        assert result is None or not result[2]


@pytest.mark.parametrize("scene,can_advance,vx_max,update", [
    ("clear", True, .30, None), ("clear", True, .12, None), ("hole", True, .30, None),
    ("invalid", False, .30, None), ("blocked", False, .30, None),
    ("detour", True, .30, None), ("anchored_hole", True, .30, None),
    ("hole", True, .30, "same_person"), ("clear", True, .30, "new_hole"),
    ("hole", False, .30, "closer_surface"), ("hole", False, .30, "obstacle"),
    ("hole", False, .30, "missing_person"), ("hole", False, .30, "moved_person"),
])
def test_visible_person_without_front_fix(monkeypatch, scene, can_advance, vx_max, update):
    from plugins.navi import policy, track as tracking

    card, packets, clock, snapshot = observation_card(monkeypatch, body=True)
    state, config = card._state, card._config
    calls = observe_selector(monkeypatch)
    config.vx_max = vx_max
    _, depth, odom, _ = snapshot[0]
    if scene == "invalid":
        depth["map"][:] = 0.
    if scene != "clear":
        depth["map"][48:145, 287:354] = 0.
    if scene == "blocked":
        depth["map"][220:370, 302:338] = .25
    elif scene == "detour":
        depth["map"][220:370, 302:338] = .65
    state.goal_anchor = None
    config.depth_intrinsics = config.local_geometry.horizontal
    person = {"name": "person", "position": [0., -.6], "confidence": .95,
              "bbox_norm": [.45, .10, .55, .30]}
    if scene == "clear":
        monkeypatch.setattr(tracking, "_front_point", lambda *args: None)
    for stamp in (700, 800, 900):
        state.tracker.step(detections={"timestamp": stamp / 1000, "objects": [person]},
                           depth=dict(depth, frame_id=stamp, source_frame_id=stamp),
                           target=state.target, config=config, ego=(0., 0., 0.), ego_measured=True, dt=.1)
    if scene == "anchored_hole":
        state.goal_anchor = policy.DetourGoal(3., 1., generation=state.tracker.track.generation)
    snapshot[0] = {"timestamp": 1., "objects": [person]}, depth, odom, 1000
    message = card.next_command()
    assert state.tracker.track.front_fix is None and state.goal_anchor is None
    assert card._last_decision.distance_m is None
    if any(message["values"]):
        guard = card._last_decision.dwa_acceptance.motion_guard
        assert guard.goal is None and guard.check.accepted and guard.trace.stopped
    if update:
        assert message["values"][0] > 0.
        stamp = 1050 if update == "moved_person" else 1010
        advance_snapshot(snapshot, clock, stamp, obstacle=update == "obstacle")
        objects, depth, odom, _ = snapshot[0]
        if update == "closer_surface":
            depth["map"][48:145, 287:354] = .4
        elif update == "new_hole":
            depth["map"][48:145, 287:354] = 0.
        elif update == "moved_person":
            person = dict(person, position=[.20, -.6], bbox_norm=[.55, .10, .65, .30])
            depth["map"][48:145, 352:416] = 0.
        people = [] if update == "missing_person" else [person]
        snapshot[0] = dict(objects, timestamp=stamp / 1000, objects=people), depth, odom, stamp
    card._publish_command(message)
    if can_advance:
        assert calls[-1][0]["x_limit"][1] == pytest.approx(vx_max)
        assert 0. < packets[-1]["values"][0] <= vx_max
        assert packets[-1]["values"][1:5] == [0.] * 4
        assert bool(packets[-1]["values"][5]) == (scene == "detour")
    else:
        assert packets[-1]["values"] == [0.] * 6
    if update == "moved_person":
        next_message = card.next_command()
        assert next_message["values"][5] < 0.
        assert card._last_decision.distance_m is None


@pytest.mark.parametrize("update", ["match", "same", "unrelated", "blocked", "future_receipt"])
def test_slow_duplicate_plan_uses_independent_new_target_and_physical_check(monkeypatch, update):
    from plugins.navi import policy

    card, packets, clock, snapshot, _ = selected_card(monkeypatch, live_target=True)
    old_objects = snapshot[0][0]
    advance_snapshot(snapshot, clock, 1100)
    step = policy.step

    def slow_step(**kwargs):
        decision = step(**kwargs)
        assert decision.dwa_acceptance is not None and not decision.dwa_acceptance.target_observed
        assert card._state.tracker.track.observed and not card._state.tracker.track.accepted_detection
        advance_snapshot(snapshot, clock, 1480, obstacle=update == "blocked")
        _, depth, odom, _ = snapshot[0]
        objects = old_objects if update == "same" else dict(old_objects, timestamp=1.47)
        if update == "unrelated":
            objects = dict(objects, objects=[])
        snapshot[0] = objects, depth, odom, {"same": 1000, "future_receipt": 1580}.get(update, 1470)
        return decision

    monkeypatch.setattr(policy, "step", slow_step)
    message = card.next_command()
    card._publish_command(message)
    assert (packets[-1]["values"][0] > 0.) == (update == "match")
    if update == "match":
        assert packets[-1]["obs_stamp_ms"] == 1470
        assert card._wire_dwa_origin.forward_check.target_source == 1.47
    assert not card._state.tracker.track.accepted_detection
