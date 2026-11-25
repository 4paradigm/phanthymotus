"""A search stops turning while a real association is awaiting confirmation."""
import pytest

from test_navi_dwa_publish import advance_snapshot, observation_card


@pytest.mark.parametrize("stamped", [True, False], ids=["source-stamp", "receipt-only"])
def test_search_waits_for_associated_person_and_resumes_only_after_confirmation(monkeypatch, stamped):
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
    assert card._state.tracker.track.reacquire_hits == 0

    for stamp, hits in ((1100, 1), (1150, 2)):
        observe(stamp, original)
        for _ in range(2):
            command = card.next_command()
            assert command["values"] == [0.] * 6
            card._publish_command(command)
            assert packets[-1]["values"] == [0.] * 6
            assert card._state.tracker.track.reacquire_hits == hits
            assert card._state.reacquire_generation == generation

    observe(1200, original)
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

    assert state.last_seen_side == -1., "Once confirmed, search should face the last measured side"
    assert track.observed and track.measured_range_m is None
    assert state.reacquire_generation == track.generation
    advance_snapshot(snapshot, clock, 1170, vx=0.)
    command = card.next_command()
    assert not track.accepted_detection and track.measured_range_m is not None
    assert state.reacquire_generation is None
    assert card._last_decision.dwa_acceptance is not None
    card._publish_command(command)
    assert packets[-1]["values"][0] > 0.


@pytest.mark.parametrize("stale_stamp,stale_receipt", [(1.04, 1060), (1.06, 1040)])
def test_out_of_order_return_does_not_count_toward_confirmation(monkeypatch, stale_stamp, stale_receipt):
    card, _, clock, snapshot = observation_card(monkeypatch, live_target=True)
    motion = card.next_command()
    original = snapshot[0][0]["objects"]
    for now, source, receipt, objects in (
            (1010, 1.01, 1010, []), (1050, 1.05, 1050, original),
            (1060, stale_stamp, stale_receipt, original), (1100, 1.1, 1100, original)):
        advance_snapshot(snapshot, clock, now, vx=0.)
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = {"timestamp": source, "objects": objects}, depth, odom, receipt
        if not objects:
            card._publish_command(motion)
        command = card.next_command()
    assert card._state.reacquire_generation is not None
    assert command["values"] == [0.] * 6
