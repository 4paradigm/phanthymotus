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
def test_new_targetless_payload_stops_an_initially_valid_renewal(monkeypatch, kind):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    track = card._state.tracker.track
    assert track.observed and track.accepted_detection and track.measured_range_m == 5.
    assert card._stage_command(message, card._output_epoch)
    assert card._scheduled_command.forward_check.renewable
    advance_snapshot(snapshot, clock, 1100)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (absent_target_payload(kind, 1100), depth, odom, 1100)
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6
    assert len([packet for packet in packets if any(packet["values"])]) == 1
    assert card._scheduled_command is None
    assert len(calls) == card._state.dwa_totals["select_calls"]


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
    assert len(nonzero) > 1, "A real matching observation permits checked renewals"
    assert all(packet["values"] == message["values"] for packet in nonzero)
    assert all(packet["stamp_ms"] + packet["ttl_ms"] <= target_deadline for packet in nonzero)
    assert card._scheduled_command.forward_check.target_ms == original_target_ms
    advance_snapshot(snapshot, clock, target_deadline)
    _, depth, odom, _ = snapshot[0]
    snapshot[0] = (original_payload, depth, odom, target_deadline)
    card._dispatch_command()
    assert packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("kind", ["duplicate", "reordered", "missing"])
def test_next_decision_cannot_promote_old_track_to_fresh_target_evidence(monkeypatch, body, kind):
    card, _, clock, snapshot, _ = selected_card(monkeypatch, body=body, live_target=True)
    assert card._state.tracker.track.accepted_detection
    advance_snapshot(snapshot, clock, 1100)
    payload, depth, odom, _ = snapshot[0]
    if kind == "reordered":
        payload = dict(payload, timestamp=.9)
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
def test_genuine_fresh_matching_target_can_continue_the_original_command(monkeypatch, confidence):
    calls = observe_selector(monkeypatch)
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    track = card._state.tracker.track
    assert track.observed and track.accepted_detection and track.measured_range_m == 5.
    assert track.navigation_range_known
    assert card._state.goal_anchor.track is track and not card._state.goal_anchor.target_missing
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100)
    payload, depth, odom, _ = snapshot[0]
    objects = [dict(payload["objects"][0], confidence=confidence)]
    snapshot[0] = (dict(payload, timestamp=1.1, objects=objects), depth, odom, 1100)
    card._dispatch_command()
    assert len(packets) == 2 and all(packet["values"] == message["values"] for packet in packets)
    assert card._state.dwa_totals["published_decisions"] == 1
    assert card._state.dwa_totals["published_packets"] == 2
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
    assert packets[-1]["values"] == (message["values"] if retain_original else [0.] * 6)
    assert len(packets) == 2
