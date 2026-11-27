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
        assert card._scheduled_command is None
    packet, = packets
    assert packet["values"] == message["values"]
    assert packet["stamp_ms"] + packet["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]


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
    assert [packet["values"] for packet in packets] == [message["values"]]
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


@pytest.mark.parametrize("event,rejection", [
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
def test_timer_during_coasting_selection_keeps_only_a_still_valid_sent_command(monkeypatch, event, rejection):
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
            if rejection is None:
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
    elif rejection is None:
        assert card._last_decision.dwa_acceptance is not None
        assert card._stage_command(next_message, card._output_epoch)
        assert len(packets) == 2 and all(packet["values"][0] > 0. for packet in packets)
    else:
        assert [packet["values"] for packet in packets] == [message["values"]]
        assert card._scheduled_command is None
        assert (during[0]["event"], during[0]["reason"]) == ("veto", rejection)


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
    assert not packets
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
    packet, = packets
    assert packet["values"] == message["values"]
    assert packet["stamp_ms"] + packet["ttl_ms"] <= target_deadline
    assert card._scheduled_command is None
    advance_snapshot(snapshot, clock, target_deadline)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (original_payload, depth, odom, target_deadline)
    card._dispatch_command()
    assert packets == [packet]


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
    packet, = packets
    assert packet["values"] == message["values"]
    assert packet["stamp_ms"] + packet["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]
    assert card._scheduled_command is None


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
        assert card._scheduled_command is None
    assert [packet["values"] for packet in packets] == [message["values"]]
    assert card._dwa_publication["reason"] == "publish_expired"


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
    elif event == "cancel":
        assert packets[-1]["values"] == [0.] * 6
    elif event == "expiry":
        assert [packet["values"] for packet in packets] == [message["values"]]
        assert card._scheduled_command is None
        assert card._dwa_publication["reason"] == "publish_expired"
    elif event in ("near", "near_new_depth"):
        assert [packet["values"] for packet in packets] == [message["values"], [0.] * 6]
        assert card._scheduled_command is None
        assert card._dwa_publication["reason"] == "target_standoff"
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
        track = card._state.tracker.track
        track.front_fix = np.array([card._config.stop_distance_m + .01, 0.])
        track.measured_range_m = None  # Coasting transports the prior fix without issuing a new measurement.
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


@pytest.mark.parametrize("life", ["matched", "first_return", "replaced"])
def test_publication_follows_associated_target_generation(monkeypatch, life):
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
        if life == "first_return" and index == 1:
            break
    assert (tracker.track is original) == (life != "replaced")
    assert tracker.track.observed
    card._publish_command(message)
    if life != "replaced":
        assert [packet["values"] for packet in packets] == [message["values"]]
    else:
        assert not packets
        assert card._dwa_publication["reason"] == "superseded"


@pytest.mark.parametrize("method,stop,at_rest,expected", [
    pytest.param("sample", 4.6, False, True, id="clear-sample"),
    pytest.param("turn", 4.6, False, True, id="clear-turn"),
    pytest.param("sample", 5.1, False, False, id="standoff-sample"),
    pytest.param("turn", 5.1, True, False, id="standoff-turn"),
    pytest.param("turn", 5.02, False, False, id="parking-ring-still-braking"),
    pytest.param("turn", 5.02, True, True, id="parking-ring-at-rest"),
])
def test_measured_target_limits_motion_independently_of_fixed_anchor(monkeypatch, method, stop, at_rest, expected):
    from plugins.navi import policy
    card, _, _, snapshot, _ = selected_card(monkeypatch, body=True, live_target=True)
    _, depth, _, _ = snapshot[0]
    state, config = card._state, card._config
    anchor = state.goal_anchor
    anchor.x = 5.6  # Original destination; the same person is now measured at 5 m.
    assert state.tracker.track.measured_range_m == 5.
    config.stop_distance_m = stop
    geometry = config.scan_geometry
    if at_rest:
        state.commanded = state.active_scan.motion = (0., 0., 0.)
    if method == "turn":
        accepted = policy._check_dwa_turn(depth, config, state, geometry, (anchor.x, 0.), "parking", (0., 0., .25))
    else:
        state.dwa_budget = None
        outcome = policy._dwa_detour(depth, config, state, state.view_detour, anchor.x, 0.)
        accepted = outcome is not None
    assert accepted == expected
    assert anchor.x == 5.6


@pytest.mark.parametrize("front,expected_vx", [(1., None), (1.025, .075), (1.06, .075), (1.1, .15)])
def test_detour_sweeps_the_nearer_person_surface_before_selecting_a_speed(monkeypatch, front, expected_vx):
    from dataclasses import replace
    from plugins.navi import dwa, policy

    card, _, _, snapshot, _ = selected_card(monkeypatch, body=False, live_target=True)
    state, config = card._state, card._config
    track = state.tracker.track
    track.measured_point = replace(track.measured_point, standoff_point=(front, 0.))
    state.dwa_budget = None
    _, depth, _, _ = snapshot[0]
    proposal = policy._dwa_detour(depth, config, state, state.view_detour, track.measured_range_m, 0.,
                                  follow_published=True)

    if expected_vx is None:
        assert proposal is None
        return
    assert proposal is not None and proposal.command[0] == pytest.approx(expected_vx)
    point, error = proposal.guard.measured_target
    assert dwa.minimum_target_distance(proposal.guard.trace, point) >= config.stop_distance_m + error


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
                assert card._scheduled_command is None
            assert packets == [packet]
            assert card._dwa_publication["reason"] == "publish_expired"


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
@pytest.mark.parametrize("travel", [0., .15], ids=["stationary", "advancing"])
def test_memory_range_is_registered_from_its_depth_capture_pose(monkeypatch, duplicate, travel):
    from plugins.navi import policy
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=True, live_target=True)
    state, config = card._state, card._config
    state.goal_anchor.x = 8.
    config.stop_distance_m = 4.65
    config.vx_max = .15
    payload, depth, odom, receipt = snapshot[0]
    start = PlanarPose(1000, ("odom",), 0., 0., 0.)
    snapshot[0] = dict(payload, timestamp=.9), depth, dict(odom, obstacle_pose=start), receipt
    message = card.next_command()
    assert message["values"][0] == pytest.approx(.15)
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    capture = PlanarPose(1010, start.frame, .01, 0., 0.)
    current = PlanarPose(1100, start.frame, capture.x + travel, 0., 0.)
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
        current = PlanarPose(1110, start.frame, capture.x + 1.13 * travel, 0., 0.)
        latest_depth = dict(latest_depth, obstacle_capture_pose=current)
        latest_odom = dict(latest_odom, obstacle_pose=current)
        snapshot[0] = fresh, latest_depth, latest_odom, 1110
        policy.step(detections=fresh, depth=latest_depth, odom=latest_odom,
                    state=state, config=config, dt=.01)
        assert state.tracker.track.measured_range_m is None
    card._publish_command(message)
    if travel:
        assert packets[-1]["values"] == [0.] * 6
        assert card._info()["dwa"]["publication"]["reason"] == "target_standoff"
    else:
        assert packets[-1]["values"] == message["values"]


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
            assert packets == [packet] and card._scheduled_command is None
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


def test_active_route_keeps_the_checked_translation_on_target_loss(monkeypatch):
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
    assert packets[1]["values"] == packets[0]["values"]
    assert card._state.committed_route is not None


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
    ("detour", True, .30, None), ("short_detour", True, .30, None),
    ("anchored_hole", True, .30, None),
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
    elif scene in ("detour", "short_detour"):
        depth["map"][220:370, 302:338] = .65
    if scene == "short_detour":
        original_route = state.local_plan.route
        attempts = []

        def timed_route(*args, **kwargs):
            attempts.append(args[4])
            clock[0] += 120
            return original_route(*args, **kwargs)

        monkeypatch.setattr(state.local_plan, "route", timed_route)
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
        if scene == "short_detour":
            assert len(attempts) == 1 and clock[0] == 1120
            assert state.dwa_trace["branch"] == "visual_probe"
        else:
            assert bool(packets[-1]["values"][5]) == (scene == "detour")
    elif update == "moved_person":
        assert not packets
        assert card._dwa_publication["reason"] == "stale_observation"
        next_message = card.next_command()
        assert next_message["values"][5] < 0.
        assert card._last_decision.distance_m is None
    else:
        assert packets[-1]["values"] == [0.] * 6


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
    if update == "match":
        packet, = packets
        assert packet["values"][0] > 0. and packet["obs_stamp_ms"] == 1470
        assert card._wire_dwa_origin.forward_check.target_source == 1.47
    elif update == "blocked":
        packet, = packets
        assert packet["values"] == [0.] * 6
        assert card._dwa_publication["reason"] == "footprint_collision"
    else:
        assert not packets
        assert card._dwa_publication["reason"] == "publish_expired"
    assert not card._state.tracker.track.accepted_detection


def test_expired_person_track_does_not_adopt_a_distant_detection_during_detour(monkeypatch):
    card, _, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    state = card._state
    assert state.committed_route.locks_detour
    original = state.tracker.track
    objects = snapshot[0][0]
    original_person = objects["objects"][0]

    stamp = 2300
    advance_snapshot(snapshot, clock, stamp, vx=0.)
    _, depth, odom, _ = snapshot[0]
    device = dict(original_person, position=[-.95, -.6])
    snapshot[0] = dict(objects, timestamp=stamp / 1000, objects=[device]), depth, odom, stamp
    card.next_command()

    assert state.tracker.track is original
    assert not original.drivable
    advance_snapshot(snapshot, clock, stamp + 100, vx=0.)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=(stamp + 100) / 1000), depth, odom, stamp + 100
    card.next_command()
    assert state.tracker.track is original and original.observed


def test_live_person_outside_stop_ring_overrides_expired_detour_position(monkeypatch):
    from dataclasses import replace

    import numpy as np

    card, _, _, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    state = card._state
    state.goal_anchor = replace(state.goal_anchor, x=.25, y=0.)
    objects, depth, odom, receipt = snapshot[0]
    snapshot[0] = objects, dict(depth, map=np.full_like(depth["map"], 5.)), odom, receipt

    packet = card.next_command()

    assert card._last_decision.distance_m == pytest.approx(5.)
    assert packet["values"][0] > 0. and not state.arrived


@pytest.mark.parametrize("moving", [False, True])
def test_held_detour_processes_fresh_brake_feedback_before_pursuit(monkeypatch, moving):
    card, _, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    state = card._state
    committed = state.committed_route
    assert committed.locks_detour
    scan = state.active_scan
    scan.hold_for_replan("主动观察：深度暂不可用，停车后重规划")

    advance_snapshot(snapshot, clock, 1100, vx=.24 if moving else 0.)
    packet = card.next_command()

    assert not any(packet["values"])
    assert state.committed_route is committed
    assert scan.phase == ("brake_for_replan" if moving else "idle")
    if not moving:
        advance_snapshot(snapshot, clock, 1200, vx=0.)
        resumed = card.next_command()
        assert resumed["values"][0] > 0.
        assert state.dwa_trace["retained_route"]


def test_committed_detour_replans_when_old_odometry_frame_cannot_register(monkeypatch):
    from dataclasses import replace

    card, _, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    card._cancel_scheduled_motion(stop=(1050, "停稳后复查路线", False),
                                  dwa_reason="footprint_collision")
    state = card._state
    committed = state.committed_route
    old_pose = replace(committed.motion_guard.pose, frame=("navi_relative", object()))
    stale = replace(committed, motion_guard=replace(committed.motion_guard, pose=old_pose))
    state.committed_route = stale
    advance_snapshot(snapshot, clock, 1100, vx=0.)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100

    candidate = card.next_command()

    assert state.committed_route is not stale
    assert candidate["values"][0] > 0.
    assert not state.dwa_trace.get("retained_route")
    guard = card._last_decision.dwa_acceptance.motion_guard
    assert guard.pose.frame != old_pose.frame and guard.check.accepted and guard.trace.stopped


@pytest.mark.parametrize("use_lateral, side_only", [
    (True, False), (True, True), (False, False),
], ids=["forward-blocker", "remembered-side-blocker", "differential-forward-blocker"])
def test_blocked_camera_detour_replans_after_distinct_stationary_views(monkeypatch, use_lateral, side_only):
    import numpy as np
    from plugins.navi import policy
    from plugins.navi.active_scan import NearProjection

    card, _, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = use_lateral
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    state = card._state
    route = state.committed_route
    assert route.locks_detour and route.passage_exit is None

    def blocked(_depth, _config, _state, reference, *args, **kwargs):
        if kwargs.get("branch") == "reachable_prefix":
            replans.append(reference.points)
            return None
        reason = "remembered_obstacle" if side_only else "footprint_collision"
        state.dwa_trace.update(fallback_reason="no_admissible_candidates", rejected={reason: 4})
        if side_only:
            hits = np.zeros(policy.FORWARD.shape, dtype=bool)
            hits[(policy.FORWARD == 0.) & np.isclose(policy.RIGHT, -.40)] = True
            state.active_scan.near_projection = NearProjection(hits=hits)
        return None

    replans = []
    monkeypatch.setattr(policy, "_detour_still_locked", lambda *args: True)
    monkeypatch.setattr(policy, "_dwa_detour", blocked)
    monkeypatch.setattr(policy, "projected_forward_clearance", lambda *args: None if side_only else 1.)
    monkeypatch.setattr(policy, "_start_detour_segment", lambda *args, **kwargs: replans.append(args))
    for stamp, depth_stamp in ((1100, 1100), (1200, 1100), (1300, 1300), (1400, 1400)):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        objects, depth, odom, _ = snapshot[0]
        depth.update(frame_id=depth_stamp, source_frame_id=depth_stamp, capture_ms=depth_stamp)
        snapshot[0] = dict(objects, timestamp=stamp / 1000), depth, odom, stamp
        card.next_command()
        assert len(replans) == int(stamp == 1400)
        assert state.committed_route is route
    if side_only:
        points, = replans
        assert points[0] == (0., 0.)
        assert points[1][0] > 0. and points[1][1] == 0.


@pytest.mark.parametrize("clear", [True, False], ids=["clear-route", "new-obstacle"])
def test_expired_person_track_keeps_the_committed_detour(monkeypatch, clear):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    state = card._state
    original = state.tracker.track
    assert state.committed_route.locks_detour
    card._config.max_coast_s = .12
    for stamp in (1100, 1200, 1300):
        advance_snapshot(snapshot, clock, stamp, obstacle=not clear, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = {"timestamp": stamp / 1000, "objects": []}, depth, odom, stamp
        packet = card.next_command()
    assert any(packet["values"][:2]) == clear
    assert state.committed_route is not None
    assert state.reacquire_generation == original.generation
    assert state.tracker.track is original
    if clear:
        assert card._last_decision.dwa_acceptance.followed_route
        assert card._stage_command(packet, card._output_epoch)
        assert any(packets[-1]["values"][:2])


@pytest.mark.parametrize("obstacle", [False, True])
def test_committed_detour_keeps_its_target_when_visible_box_loses_range(monkeypatch, obstacle):
    from plugins.navi import track

    card, _, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    state = card._state
    route, anchor = state.committed_route, state.goal_anchor
    assert route.locks_detour and anchor.generation == state.tracker.track.generation

    card._config.depth_intrinsics = card._config.local_geometry.horizontal
    advance_snapshot(snapshot, clock, 1100, obstacle=obstacle)
    objects, depth, odom, _ = snapshot[0]
    person = dict(objects["objects"][0], bbox_norm=[.45, .10, .55, .30])
    snapshot[0] = dict(objects, timestamp=1.1, objects=[person]), depth, odom, 1100
    monkeypatch.setattr(track, "_front_point", lambda *args: None)
    message = card.next_command()

    assert state.tracker.track.observed and state.tracker.track.front_required
    assert state.tracker.track.front_fix is None
    assert state.goal_anchor is anchor and state.committed_route is route
    assert state.dwa_trace.get("branch") != "visual_probe"
    if obstacle:
        assert not any(message["values"])
    else:
        assert any(message["values"][:2])
        assert card._last_decision.dwa_acceptance.followed_route


def test_search_waits_for_committed_person_detour_to_lose_freshness(monkeypatch):
    from plugins.navi import policy

    card, _, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    state = card._state
    route = state.committed_route
    assert route.locks_detour
    card._config.max_coast_s = .12
    attempts = []
    monkeypatch.setattr(policy, "_follow_published_route", lambda *args, **kwargs: attempts.append(args))

    expires = route.motion_guard.checked_ms + card._config.depth_age_limit_ms
    for stamp in range(1100, expires + 100, 100):
        attempts.clear()
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = {"timestamp": stamp / 1000, "objects": []}, depth, odom, stamp
        packet = card.next_command()
        if stamp < expires:
            assert not any(packet["values"])
            assert state.committed_route is route

    assert len(attempts) == 1
    assert packet["values"][:5] == [0.] * 5
    assert packet["values"][5] < 0.
    assert state.committed_route is route


@pytest.mark.parametrize("new_blocker", [False, True])
@pytest.mark.parametrize("use_lateral, side_exit", [
    (True, False), (True, True), (False, False),
], ids=["lateral", "lateral-side-exit", "differential-past-blocker"])
def test_straight_handoff_from_committed_detour_uses_forward_clearance(monkeypatch, new_blocker, use_lateral, side_exit):
    from dataclasses import replace

    from plugins.navi import policy

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = use_lateral
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    original = card._state.committed_route
    assert original is not None and original.locks_detour
    if side_exit:
        original = replace(original, passage_exit=((.9, -.9), (1., -.10)))
    elif not use_lateral:
        original = replace(original, blocker_exit=((-.30, -.40), (-.30, .40)))
    card._state.committed_route = original
    work = card._scheduled_command
    work.dwa_origin = replace(work.dwa_origin, acceptance=original)
    monkeypatch.setattr(policy, "_detour_still_locked", lambda *args: False)

    advance_snapshot(snapshot, clock, 1100, vx=0.)
    objects, depth, odom, _ = snapshot[0]
    depth["map"].fill(5.)
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = card.next_command()
    assert message["values"][0] > 0.
    assert message["values"][1:] == [0.] * 5
    assert card._last_decision.dwa_acceptance is None
    assert card._state.committed_route is original

    if new_blocker:
        advance_snapshot(snapshot, clock, 1110, vx=0., obstacle=True)
    count = len(packets)
    card._stage_command(message, card._output_epoch)
    if new_blocker:
        assert all(not any(packet["values"]) for packet in packets[count:])
        assert card._state.committed_route is original
    else:
        assert len(packets) == count + 1
        assert packets[-1]["values"] == message["values"]
        assert card._state.committed_route is None
        advance_snapshot(snapshot, clock, 1200, vx=message["values"][0])
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = dict(objects, timestamp=1.2), depth, odom, 1200
        renewed = card.next_command()
        assert renewed["values"][0] > 0.
        assert renewed["values"][1:] == [0.] * 5
        assert card._staged_forward_check[1].straight_handoff is original
        assert card._stage_command(renewed, card._output_epoch)
        clock[0] = 1210
        card._cancel_scheduled_motion(stop=(1210, "等待新深度", True))
        advance_snapshot(snapshot, clock, 1300, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = dict(objects, timestamp=1.3), depth, odom, 1300
        resumed = card.next_command()
        assert resumed["values"][0] > 0.
        assert resumed["values"][1:] == [0.] * 5
        assert card._staged_forward_check[1].straight_handoff is original


def test_detour_only_locks_an_inbound_first_half():
    from plugins.navi import dwa, local_path, policy
    from test_navi_dwa_integration import observation_case

    config, _, _, _ = observation_case(body=False)
    for points, expected in (
        (((0., 0.), (1., 0.), (2., .5)), True),
        (((0., 0.), (1., .8), (2., .5)), False),
        (((-2., 0.), (0., 0.), (1., 0.)), False),
    ):
        route = local_path.Detour(None, points, config.local_geometry)
        assert policy._detour_still_locked(route, dwa.remaining_route_length(points), 4., 0.) == expected


@pytest.mark.parametrize("blocked", [False, True], ids=["side-face", "blocked-ahead"])
def test_near_side_face_can_be_passed_without_turning_into_it(monkeypatch, blocked):
    import numpy as np

    from plugins.navi import local_path, policy
    from plugins.navi.active_scan import NearProjection

    card, _, _, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    config, state = card._config, card._state
    config.use_lateral = True
    config.use_obstacle_history = False
    card.next_command()
    scan = state.active_scan
    state.goal_anchor = None
    objects, depth, odom, receipt = snapshot[0]
    pose = policy.dwa_grid_pose(scan, config.local_geometry, receipt, odom)
    points = [(-.20, -.34)]
    if blocked:
        points.append((.08, 0.))
    cells = local_path.ObstacleSquares(np.array(points), 0.)
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    scan.near_projection = NearProjection(occupied, cells, pose, receipt)
    monkeypatch.setattr(policy, "_decide", lambda *args, **kwargs: policy.Decision(
        policy._twist(.2, 0., .5), policy.APPROACHING, "approaching", distance_m=2.))

    decision = policy.step(detections=objects, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, objects_ms=receipt, dwa_admission_ms=100)

    assert not any(decision.values[1:])
    assert bool(decision.values[0]) is not blocked
    if not blocked:
        assert decision.dwa_acceptance.motion_guard.check.accepted


def test_straight_handoff_publishes_slower_speed_when_clearance_shrinks(monkeypatch):
    from plugins.navi import policy

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    monkeypatch.setattr(policy, "_detour_still_locked", lambda *args: False)
    advance_snapshot(snapshot, clock, 1100, vx=0.)
    objects, depth, odom, _ = snapshot[0]
    depth["map"].fill(5.)
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = card.next_command()
    assert message["values"][0] > 0.

    advance_snapshot(snapshot, clock, 1110, vx=0.)
    depth = snapshot[0][1]
    depth["map"][220:370, 302:338] = card._config.obstacle_stop_m + .1
    cap = policy.straight_handoff_speed(depth, card._config, card._last_decision.distance_m,
                                        (0., 0., 0.), card._wire_twist)
    assert 0. < cap < message["values"][0]
    assert card._stage_command(message, card._output_epoch)
    assert 0. < packets[-1]["values"][0] <= cap
    assert packets[-1]["values"][1:] == [0.] * 5
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= message["stamp_ms"] + message["ttl_ms"]
    assert card._state.committed_route is None


@pytest.mark.parametrize("opening", ["too_narrow", "fits", "one_jamb"])
def test_straight_handoff_keeps_a_measured_narrow_opening_in_the_braking_distance(monkeypatch, opening):
    import numpy as np

    from plugins.navi import local_path, policy
    from plugins.navi.active_scan import NearProjection

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    assert card._stage_command(card.next_command(), card._output_epoch)
    monkeypatch.setattr(policy, "_detour_still_locked", lambda *args: False)
    advance_snapshot(snapshot, clock, 1100, vx=0.)
    objects, depth, odom, _ = snapshot[0]
    depth["map"].fill(5.)
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = card.next_command()
    assert message["values"][0] > 0.

    # The latest camera view is clear even though previously seen jambs
    # are already beside the camera and still ahead of the chassis.
    span = .30 if opening == "fits" else .20
    right = np.isclose(local_path.RIGHT, span)
    if opening != "one_jamb":
        right |= np.isclose(local_path.RIGHT, -span)
    hits = right & np.isclose(local_path.FORWARD, .20)
    card._state.active_scan.near_projection = NearProjection(hits=hits, stamp_ms=1100)

    assert card._stage_command(message, card._output_epoch)
    assert bool(packets[-1]["values"][0]) is (opening != "too_narrow")
    assert packets[-1]["values"][1:] == [0.] * 5


@pytest.mark.parametrize("blocked", [False, True])
def test_only_a_clear_forward_corridor_can_bypass_unregistered_near_memory(monkeypatch, blocked):
    card, packets, _, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    scan = card._state.active_scan
    scan.use_obstacle_history = False
    scan.near_obstacles.frame = ("navi_relative", object())
    scan.near_obstacles.cells.add((5, 0))
    snapshot[0][1]["map"].fill(5.)
    if blocked:
        snapshot[0][1]["map"][220:370, 302:338] = .25

    message = card.next_command()
    assert message is not None
    assert card._stage_command(message, card._output_epoch)
    assert (packets[-1]["values"][0] > 0.) is not blocked
    assert packets[-1]["values"][1:] == [0.] * 5
    assert scan.near_obstacles.cells


def test_straight_handoff_slowing_uses_depth_source_and_control_times(monkeypatch):
    from plugins.navi import policy

    config = policy.Config()
    monkeypatch.setattr(policy, "_clearance", lambda *_: (config.obstacle_stop_m + .3, 1.))
    depth = {"frame_id": 1000, "map": object(), "control_ms": 1000}
    stopped = (0., 0., 0.)
    fresh = policy.straight_handoff_speed(depth, config, 2., stopped, stopped)
    delayed = policy.straight_handoff_speed(
        dict(depth, obstacle_driver_ms=800), config, 2., stopped, stopped)
    older = policy.straight_handoff_speed(
        dict(depth, obstacle_driver_ms=800, control_ms=1100), config, 2., stopped, stopped)
    assert 0. < older < delayed < fresh
    slowing = policy.straight_handoff_speed(
        dict(depth, obstacle_driver_ms=800, control_ms=1100), config, 2., stopped, (delayed, 0., 0.))
    assert 0. < slowing < older
    assert policy.straight_handoff_speed(depth, config, 2., stopped, (1., 0., 0.)) == 0.


def test_target_loss_in_parking_range_cannot_start_a_search_turn(monkeypatch):
    from test_navi_dwa_publish import direct_tracking_card

    card, packets, clock = direct_tracking_card(monkeypatch, body=False)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    card._config.stop_distance_m = .60
    card._config.lost_frames = 1
    card._config.max_coast_s = .12
    card._state.tracker = type(card._state.tracker)()
    objects, depth, odom, _ = card._observation_snapshot(with_capture_timing=True)
    depth["map"][72:160, 385:455] = .62
    depth["ground"][72:160, 385:455] = False
    person = dict(objects["objects"][0], bbox_norm=[.60, .15, .71, .333], position=[.31, -.52])
    snapshot = [(dict(objects, objects=[person]), depth, dict(odom, vx=0.), 1000)]
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    card._tick()
    state = card._state
    generation = state.tracker.track.generation
    assert state.parking_target_generation == generation

    for stamp in (1100, 1200, 1300, 1400):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = absent_target_payload("empty", stamp), depth, odom, stamp
        count = len(packets)
        card._tick()
        assert all(not any(packet["values"]) for packet in packets[count:])
    assert state.tracker.track.generation == generation


@pytest.mark.parametrize("top_range", [.73, 1.05], ids=["coherent-near-person", "far-top-occluder"])
def test_committed_detour_checks_that_near_range_disagrees_with_box_top(monkeypatch, top_range):
    from plugins.navi import policy

    card, _, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    config, state = card._config, card._state
    config.use_obstacle_history = False
    config.stop_distance_m = .70
    assert card._stage_command(card.next_command(), card._output_epoch)
    route, track = state.committed_route, state.tracker.track
    assert route.locks_detour and state.goal_anchor.generation == track.generation
    state.goal_anchor.x, state.goal_anchor.y = 1.05, 0.

    advance_snapshot(snapshot, clock, 1100, vx=0.)
    _, depth, _, _ = snapshot[0]
    depth["map"].fill(.73)
    rows = depth["map"].shape[0]
    depth["map"][rows // 4:3 * rows // 8, :] = top_range
    track.front_box = (.25, .25, .75, .75)
    track.front_fix = (.73, 0.)
    track.front_required = True
    track.record_range(.73, track.front_fix, depth)
    monkeypatch.setattr(state.tracker, "step", lambda **_: track)

    message = card.next_command()

    assert message["values"] == [0.] * 6
    if top_range == .73:
        assert state.committed_route is None
        assert card._last_decision.status == policy.ARRIVED
    else:
        assert state.committed_route is route
        assert card._last_decision.target_pending
        assert card._last_decision.status == policy.AVOIDING


def test_retained_target_does_not_start_parking_turn_on_a_predicted_range(monkeypatch):
    from plugins.navi.track import point_of

    card, _, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_lateral = True
    card._config.use_obstacle_history = False
    state = card._state
    state.goal_anchor = None
    state.fwd_gate.on = True
    track = state.tracker.track
    advance_snapshot(snapshot, clock, 1100, vx=.02)
    _, depth, _, _ = snapshot[0]
    depth["map"].fill(5.)
    arrival = card._config.arrival_distance_m
    bearing = -.4 * card._config.half_fov_rad
    track.record_range(arrival + .003, point_of(arrival + .003, bearing), depth)
    track.front_required = True
    track.front_fix = point_of(arrival - .003, bearing)
    track.measured_range_m = None
    monkeypatch.setattr(state.tracker, "step", lambda **_: track)

    packet = card.next_command()

    assert packet["values"][0] > 0.
    assert packet["values"][5] == 0.
    assert card._last_decision.dwa_acceptance is not None
    assert card._last_decision.dwa_acceptance.motion_guard.check.accepted
