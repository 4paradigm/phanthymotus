"""Recover the first associated person while retaining detector ordering."""
import numpy as np
import pytest

from plugins.navi import policy
from test_navi_dwa_publish import advance_snapshot, observation_card


def test_fresh_depth_on_the_last_seen_person_can_start_a_checked_approach(monkeypatch):
    from plugins.navi.depth import ground_mask
    from test_navi_dwa_integration import card_for_case, observation_case

    config, state, depth, odom = observation_case(attempted=False)
    geometry = config.local_geometry
    config.depth_intrinsics = geometry.horizontal
    config.use_obstacle_history = False
    state.goal_anchor = None
    state.commanded = (0., 0., 0.)
    ray = (np.arange(480) - 240.) / 420.
    floor = np.minimum(.42 / np.maximum(ray, .001), 5.)
    frame = np.broadcast_to(floor[:, None], (480, 640)).copy()
    frame[120:260, 255:385] = 1.8
    depth.update(map=frame, ground=ground_mask(frame, geometry.horizontal, geometry.vertical))
    person = {"name": "person", "position": [0., -.2], "confidence": .95,
              "bbox_norm": [.4, .25, .6, .54]}
    track = state.tracker.step(detections={"timestamp": .9, "objects": [person]},
                               depth=dict(depth, frame_id=900, source_frame_id=900, control_ms=900),
                               target="person", config=config, ego=(0., 0., 0.), dt=.1)
    assert track.front_fix is not None
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    monkeypatch.setattr(card, "_observation_snapshot",
                        lambda **_: ({"timestamp": 1., "objects": []}, depth, odom, 1000))
    command = card.next_command()
    assert not track.observed and track.measured_range_m == pytest.approx(1.8)
    assert card._last_decision.dwa_acceptance is not None, card._last_decision.reason
    card._publish_command(command)
    assert packets[-1]["values"][0] > 0.


@pytest.mark.parametrize("stamped", [True, False], ids=["source-stamp", "receipt-only"])
def test_search_resumes_on_the_first_associated_person_with_depth(monkeypatch, stamped):
    card, packets, clock, snapshot = observation_card(monkeypatch, live_target=True, detector_timestamp=stamped)
    motion = card.next_command()
    assert card._last_decision.dwa_acceptance is not None
    original = snapshot[0][0]["objects"]

    def observe(stamp, objects):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        frame = {"objects": objects}
        if stamped:
            frame["timestamp"] = stamp / 1000
        snapshot[0] = frame, depth, odom, stamp

    observe(1010, [])
    card._publish_command(motion)
    generation = card._state.tracker.track.generation
    assert card._state.reacquire_generation == generation
    assert card.next_command()["values"][5] != 0.

    # Same label and almost the same bearing, but on the foreground obstacle at 2 m.
    observe(1050, [dict(original[0], position=[0., .1])])
    assert card.next_command()["values"][5] != 0.
    assert not card._state.tracker.track.observed

    observe(1100, original)
    command = card.next_command()
    assert card._state.reacquire_generation is None
    assert card._state.tracker.track.generation == generation
    assert card._last_decision.dwa_acceptance is not None
    card._publish_command(command)
    assert packets[-1]["values"][0] > 0


def test_new_depth_can_finish_recovery_between_detector_frames(monkeypatch):
    from test_navi_dwa_target_evidence import coasting_person_case

    card, packets, clock, snapshot, _ = coasting_person_case(monkeypatch, True)
    state = card._state
    track = state.tracker.track
    state.reacquire_generation = track.generation
    state.reacquire_receipt_ms = 1000
    state.reacquire_source = 1.
    last_depth = track.front_frame_id
    person = {"name": "person", "position": [-.08, -.6], "confidence": .95,
              "bbox_norm": [.41, .10, .51, .30]}

    state.last_seen_side = 1.
    for stamp in (1050, 1100, 1150):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        depth.update(frame_id=last_depth, source_frame_id=last_depth, capture_ms=last_depth)
        snapshot[0] = {"timestamp": stamp / 1000, "objects": [person]}, depth, odom, stamp
        assert card.next_command()["values"] == [0.] * 6

    assert state.last_seen_side == -1., "Search should face the last associated side"
    assert track.observed and track.measured_range_m is None
    assert state.reacquire_generation == track.generation
    advance_snapshot(snapshot, clock, 1170, vx=0.)
    command = card.next_command()
    assert not track.accepted_detection and track.measured_range_m is not None
    assert state.reacquire_generation is None
    assert card._last_decision.dwa_acceptance is not None
    card._publish_command(command)
    assert packets[-1]["values"][0] > 0.


def test_confirmed_person_cannot_be_replaced_during_parking_alignment(monkeypatch):
    card, _, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card.next_command()
    state = card._state
    original = state.tracker.track
    assert state.dwa_target_receipt[0] == original.generation
    # Parking alignment can outlive the route that brought the robot to the ring.
    state.committed_route = state.published_dwa_route = state.goal_anchor = None
    _, depth, odom, _ = snapshot[0]
    policy._parking_alignment(depth, card._config, state, odom, .78, .8, .78, True)
    other = {"name": "person", "position": [.85, -.6], "confidence": .95}

    for stamp in range(1100, 2601, 100):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = {"timestamp": stamp / 1000, "objects": [other]}, depth, odom, stamp
        card.next_command()

    assert state.tracker.track is original
    assert not original.observed


@pytest.mark.parametrize("stale_stamp,stale_receipt", [(1.04, 1060), (1.06, 1040)])
def test_out_of_order_detection_cannot_restore_tracking(monkeypatch, stale_stamp, stale_receipt):
    card, _, clock, snapshot = observation_card(monkeypatch, live_target=True)
    motion = card.next_command()
    original = snapshot[0][0]["objects"]
    for now, source, receipt, objects in (
            (1010, 1.01, 1010, []), (1050, 1.05, 1050, []),
            (1060, stale_stamp, stale_receipt, original), (1100, 1.1, 1100, original)):
        advance_snapshot(snapshot, clock, now, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = {"timestamp": source, "objects": objects}, depth, odom, receipt
        if now == 1010:
            card._publish_command(motion)
        command = card.next_command()
        if now == 1060:
            assert card._state.reacquire_generation is not None
            assert not card._state.tracker.track.observed
    assert card._state.reacquire_generation is None
    assert command["values"][0] > 0.
