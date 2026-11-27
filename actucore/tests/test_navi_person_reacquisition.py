"""Recover the first associated person while retaining detector ordering."""
import numpy as np
import pytest

from plugins.navi import policy
from test_navi_dwa_publish import advance_snapshot, observation_card


def test_detour_remembers_the_last_associated_image_side(monkeypatch):
    card, _, _, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    state = card._state
    objects, depth, odom, receipt = snapshot[0]
    person = dict(objects["objects"][0], position=[-.05, -.6])
    snapshot[0] = dict(objects, objects=[person]), depth, odom, receipt

    card.next_command()

    assert state.tracker.track.accepted_detection
    assert state.dwa_trace["branch"] in ("route", "reachable_prefix")
    assert state.last_seen_side == -1.


def test_recovery_turn_does_not_switch_to_the_clearer_side(monkeypatch):
    config = policy.Config(local_planning=True, lost_frames=1)
    state = policy.State(target="person", last_seen_side=1., reacquire_generation=1)
    monkeypatch.setattr(policy, "_clearance", lambda _d, _c, lateral_offset_m=0.: (
        2. if lateral_offset_m < 0. else .8, 1.))

    decision = policy._search_for_target({"map": np.ones((4, 4))}, config, state, .1, None)

    assert decision.values[:5] == [0.] * 5
    assert decision.values[5] < 0.


def test_blocked_recovery_turn_only_retries_toward_original_target(monkeypatch):
    config = policy.Config(local_planning=True, lost_frames=1, floor_wz=0.)
    state = policy.State(target="person", last_seen_side=1., reacquire_generation=1)

    def check(_depth, _config, _state, *, requested_yaw=None):
        if requested_yaw is None or requested_yaw < 0.:
            return "原地搜索：转身扫掠范围有障碍，停车"
        return None

    monkeypatch.setattr(policy, "_search_turn_stop", check)
    decision = policy._search_for_target({}, config, state, .1, None)

    assert decision.values == [0.] * 6
    assert state.last_seen_side == 1.


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


@pytest.mark.parametrize("view", ["same_box", "ambiguous", "empty"])
def test_lost_person_can_pause_once_for_matching_depth(monkeypatch, view):
    from test_navi_dwa_target_evidence import coasting_person_case

    card, packets, clock, snapshot, _ = coasting_person_case(monkeypatch, False)
    state, cfg = card._state, card._config
    track = state.tracker.track
    track.hue = "blue"
    track.since_obs_s = cfg.max_coast_s + .1
    state.goal_anchor = None
    state.reacquire_generation = track.generation
    state.reacquire_receipt_ms, state.reacquire_source = 1000, 1.
    state.searching_for_s = .2
    person = {"name": "person", "color": "blue", "position": [0., -.6], "confidence": .95,
              "bbox_norm": [.45, .1, .55, .3]}
    people = [] if view == "empty" else [person]
    if view == "ambiguous":
        people += [dict(person, position=[.4, -.6], bbox_norm=[.65, .1, .75, .3])]

    def observe(stamp, distance):
        advance_snapshot(snapshot, clock, stamp, vx=0.)
        _, depth, odom, _ = snapshot[0]
        image = depth["map"].copy()
        image[:150] = distance
        snapshot[0] = {"timestamp": stamp / 1000, "objects": people}, dict(depth, map=image), odom, stamp
        return card.next_command()

    message = observe(1050, 12.)
    assert state.tracker.track is track and not track.observed
    assert track.measured_range_m is None
    if view != "same_box":
        assert state.reacquire_pause_until_ms is None
        return
    assert message["values"] == [0.] * 6
    card._publish_command(message)
    assert packets[-1]["values"] == [0.] * 6
    deadline = state.reacquire_pause_until_ms
    assert deadline == 1050 + cfg.depth_age_limit_ms
    assert observe(1100, 12.)["values"] == [0.] * 6
    assert state.reacquire_pause_until_ms == deadline
    observe(deadline + 1, 12.)
    assert state.reacquire_pause_until_ms == deadline
    assert state.tracker.track is track and not track.observed
    assert "候选目标重新入画" not in card._last_decision.reason


@pytest.mark.parametrize("mixed", [True, False], ids=["mixed-depth", "uniform-near"])
def test_near_depth_does_not_release_committed_person_route_with_far_upper_support(monkeypatch, mixed):
    from dataclasses import replace
    from plugins.navi.depth import _box_slices
    from test_navi_dwa_target_evidence import coasting_person_case

    card, packets, clock, snapshot, message = coasting_person_case(monkeypatch, False)
    state = card._state
    track = state.tracker.track
    assert card._stage_command(message, card._output_epoch)
    commitment = state.committed_route
    assert commitment is not None
    commitment = replace(commitment, locks_detour=True)
    state.committed_route = commitment
    card._scheduled_command = None

    # The person was associated three metres away. Delay-induced foreground
    # range has contaminated the live filter but never changed the frozen goal.
    state.goal_anchor = anchor = policy.DetourGoal(3., 0., generation=track.generation)
    track.mean[:2] = (.85, 0.)
    track.mean[2:] = 0.
    track.front_reference = np.array([.85, 0.])
    track.front_fix = np.array([.85, 0.])

    stamp = 1100
    advance_snapshot(snapshot, clock, stamp, vx=0.)
    _, depth, odom, _ = snapshot[0]
    image = depth["map"].copy()
    box = [.45, .10, .55, .30]
    crop = image[_box_slices(image.shape, box)]
    crop[:] = .744
    if mixed:
        # Foreground spans 40% of the crop, including the upper edge.
        # A background layer agrees with the original target farther away.
        crop[:, int(crop.shape[1] * .4):] = 3.
    person = {"name": "person", "position": [0., -.6], "bbox_norm": box, "confidence": .95}
    snapshot[0] = {"timestamp": stamp / 1000, "objects": [person]}, dict(depth, map=image), odom, stamp

    next_packet = card.next_command()

    assert state.tracker.track is track
    assert track.measured_range_m == pytest.approx(.744)
    assert next_packet["values"][:2] == [0., 0.]
    if mixed:
        assert next_packet["values"] == [0.] * 6
        card._publish_command(next_packet)
        assert packets[-1]["values"] == [0.] * 6
        assert state.committed_route is commitment
        assert state.goal_anchor is anchor
        assert state.parking_target_generation is None
    else:
        assert state.committed_route is None
        assert state.parking_target_generation == track.generation
