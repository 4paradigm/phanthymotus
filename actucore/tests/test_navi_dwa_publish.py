"""DWA provenance at the actual JSON publisher, including queued renewals."""
import json
import logging
import queue
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


@pytest.mark.parametrize("first,last,yaw,settled", [
    (1005, 1010, 0., True), (995, 1010, 0., False),
    (1005, 1005, 0., False), (1005, 1010, .4, False)])
def test_source_switch_shortens_hold_only_after_measured_stop(monkeypatch, first, last, yaw, settled):
    card, _, clock, snapshot = observation_card(monkeypatch)
    state = card._state
    state.motion_source = "command"
    state.commanded = (.15, 0., 0.)
    card.next_command()
    assert "等待底盘停稳" in card._last_decision.reason
    advance_snapshot(snapshot, clock, 1010, vx=0.)
    odom = snapshot[0][2]
    odom.update(stamp_ms=last, velocity_history=((first, first, (0., 0., 0.)), (last, last, (0., 0., yaw))))
    card.next_command()
    assert (state.feedback_guard_s == 0.) == settled


@pytest.mark.parametrize("timing", ["speed_field", "sample_receipt", "declared_receipt"])
@pytest.mark.parametrize("samples,settled", [
    ([(1010, 1005, 0.), (1050, 1040, 0.), (1060, 1040, 0.)], True),
    ([(1050, 1010, 0.), (1050, 1040, 0.)], False),  # Queue drained together.
    ([(1010, 995, 0.), (1050, 1040, 0.)], False),
    ([(1010, 1005, 0.), (1050, 1005, 0.)], False),
    ([(1010, 1005, 0.), (1050, None, 0.)], False),
    ([(1010, 1005, 0.), (1050, 1040, .4)], False),
])
def test_received_speed_recovery_uses_driver_speed_updates(monkeypatch, samples, settled, timing):
    card, _, clock, snapshot = observation_card(monkeypatch)
    state = card._state
    state.motion_source = "command"
    state.commanded = (.15, 0., 0.)
    card.next_command()
    assert "等待底盘停稳" in card._last_decision.reason
    if timing == "declared_receipt":
        card._odom_interface = {"timing": {"stamp_source": "received"}}
    for receipt, speed_stamp, yaw in samples:
        clock[0] = receipt
        vendor = {} if timing == "declared_receipt" else {"stamp_source": "received"}
        if timing == "speed_field":
            vendor["sample_timestamps_ms"] = {"speed": speed_stamp}
        payload = dict(schema="motus.odom/1", frame="body", vendor=vendor,
                       stamp_ms=receipt if timing == "speed_field" else speed_stamp,
                       twist=[0., 0., None, None, None, yaw])
        card._on_string("odom", SimpleNamespace(data=json.dumps(payload)))
    clock[0] = 1070
    objects, depth, _, receipt = snapshot[0]
    odom = plugin.NaviPlugin._observation_snapshot(card, with_capture_timing=True)[2]
    assert "stamp_ms" not in (odom or {})
    snapshot[0] = objects, dict(depth, control_ms=1070), odom, receipt
    card.next_command()
    assert (state.feedback_guard_s == 0.) == settled


@pytest.mark.parametrize("update", [{"valid": False}, {"twist": [None, None, None, None, None, 0.]}])
def test_invalid_or_partial_speed_cannot_confirm_standstill(monkeypatch, update):
    card, _, clock, snapshot = observation_card(monkeypatch)
    for receipt in (1010, 1050):
        clock[0] = receipt
        payload = dict(schema="motus.odom/1", frame="body", stamp_ms=receipt - 5,
                       twist=[0., 0., None, None, None, 0.], vendor={"stamp_source": "received"})
        payload.update(update)
        card._on_string("odom", SimpleNamespace(data=json.dumps(payload)))
    odom = plugin.NaviPlugin._observation_snapshot(card, with_capture_timing=True)[2]
    assert "received_speed_pair" not in odom
    objects, depth, _, receipt = snapshot[0]
    snapshot[0] = objects, dict(depth, control_ms=1050), odom, receipt
    card.next_command()
    assert not card._state.uses_received_speed


@pytest.mark.parametrize("late_repeat", [False, True])
def test_lost_received_speed_holds_after_source_guard_expires(monkeypatch, late_repeat):
    card, _, clock, snapshot = observation_card(monkeypatch)
    objects, depth, odom, receipt = snapshot[0]
    snapshot[0] = objects, depth, dict(odom, received_speed_ms=1000), receipt
    card.next_command()
    for stamp in range(1100, 2400, 100):
        clock[0] = stamp
        # An old speed payload may arrive after a transport stall.
        repeat = dict(odom, received_speed_ms=1000, received_ms=stamp) if late_repeat else None
        snapshot[0] = objects, dict(depth, frame_id=stamp, control_ms=stamp), repeat, receipt
        card.next_command()
    assert card._state.feedback_guard_s == 0.
    assert card._last_decision.reason == "运动反馈已到期，停车等待新速度"
    assert not any(card._last_decision.values)


@pytest.mark.parametrize("event", ["clear", "obstacle", "long_gap", "target"])
def test_search_turn_uses_its_checked_window_through_a_speed_miss(monkeypatch, event):
    card, packets, clock, snapshot = observation_card(monkeypatch)
    card._command_timer = object()
    config, state = card._config, card._state
    config.use_obstacle_history = False
    state.goal_anchor = None
    state.commanded = (0., 0., 0.)
    state.missing_frames = config.lost_frames - 1
    _, depth, odom, _ = snapshot[0]
    clear = np.full_like(depth["map"], 5.)
    objects = {"timestamp": 1.0, "objects": []}
    measured = dict(odom, vx=0., received_speed_ms=1000,
                    velocity_history=((1000, 1000, (0., 0., 0.)),))
    snapshot[0] = objects, dict(depth, map=clear), measured, 1000
    card._tick()
    first, = packets
    assert first["values"][5] and not any(first["values"][:5])
    window = card._scheduled_command
    assert state.search_feedback_ms == 1000

    stamp = 1300 if event == "long_gap" else 1100
    clock[0] = stamp
    frame = clear.copy()
    if event == "obstacle":
        frame[220:370, 302:338] = .25
    if event == "target":
        objects = {"timestamp": 1.1, "objects": [
            {"name": "person", "position": [0., 0.], "confidence": .95}]}
    invalid = dict(vx=None, vy=None, wz=None, received_ms=stamp, received_speed_ms=None,
                   velocity_history=(*measured["velocity_history"], (stamp, stamp, None)))
    snapshot[0] = objects, dict(depth, map=frame, frame_id=stamp, capture_ms=stamp,
                                 control_ms=stamp, source_frame_id=stamp), invalid, stamp
    card._tick()
    if event == "clear":
        assert packets == [first]
        assert card._scheduled_command is window
        assert state.searched_rad == pytest.approx(.1 * abs(first["values"][5]))
        clock[0] = 1200
        normal = dict(measured, received_ms=1200, received_speed_ms=1200, source_sample_id=1200,
                      velocity_history=(*invalid["velocity_history"], (1200, 1200, (0., 0., 0.))))
        snapshot[0] = objects, dict(depth, map=clear, control_ms=1200), normal, stamp
        card._tick()
        assert len(packets) == 2 and packets[-1]["values"] == first["values"]
    else:
        assert len(packets) == 2 and not any(packets[-1]["values"])


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_selected_route_first_write_reuses_unchanged_planning_proof(monkeypatch, body):
    card, packets, _, _, message = selected_card(monkeypatch, body=body, live_target=True)
    origin = card._staged_dwa_command[1]
    assert origin.acceptance.route_points
    monkeypatch.setattr(card, "_registered_dwa_scene",
                        lambda *a, **kw: pytest.fail("unchanged inputs rebuilt the stopping sweep"))
    if body:
        card._publish_command(message)
    else:
        assert card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["values"] == message["values"]
    assert packet["obs_stamp_ms"] == message["obs_stamp_ms"]
    assert packet["stamp_ms"] + packet["ttl_ms"] == message["stamp_ms"] + message["ttl_ms"]


@pytest.mark.parametrize("detector_missing", [False, True])
def test_committed_arc_brakes_for_a_new_obstacle(monkeypatch, detector_missing):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms
    assert card._stage_command(message, card._output_epoch)
    work = card._scheduled_command
    assert work.sent
    origin = work.dwa_origin
    work.dwa_origin = card._wire_dwa_origin = replace(
        origin, acceptance=replace(origin.acceptance, corridor_corner=True))
    deadline = work.forward_check.motion_guard.deadline_ms
    advance_snapshot(snapshot, clock, 1010, obstacle=True)
    if detector_missing:
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = ({"timestamp": 1.01, "objects": []}, depth, odom, 1010)
        work.repeat_until = None

    card._dispatch_command()

    motion, stop = packets
    assert any(motion["values"])
    assert stop["stamp_ms"] < deadline and not any(stop["values"])
    assert card._scheduled_command is None


def test_followup_prefers_the_already_published_route(monkeypatch):
    card, packets, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    assert len(packets) == 1
    original = card._wire_dwa_origin.acceptance
    advance_snapshot(snapshot, clock, 1100, vx=first["values"][0])
    next_message = card.next_command()
    assert card._state.dwa_trace.get("retained_route")
    assert card._state.dwa_trace["select_calls"] == 1
    assert card._state.dwa_trace["sampled_candidates"] <= 3
    assert next_message["values"][0] > 0.
    assert len(card._last_decision.dwa_acceptance.route_points) == len(original.route_points)


def test_latest_only_detour_can_keep_its_checked_route_after_the_obstacle_leaves_view(monkeypatch):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_obstacle_history = False
    first = card.next_command()
    assert card._stage_command(first, card._output_epoch)
    assert card._wire_dwa_origin.acceptance.passage_exit is None
    advance_snapshot(snapshot, clock, 1100, vx=first["values"][0])
    snapshot[0][1]["map"].fill(5.)

    following = card.next_command()

    assert card._state.dwa_trace.get("retained_route")
    assert card._last_decision.dwa_acceptance.motion_guard.check.accepted
    assert not card._last_decision.dwa_acceptance.motion_guard.remembered
    assert card._stage_command(following, card._output_epoch)
    assert packets[-1]["values"][0] > 0.


@pytest.mark.parametrize("next_depth", ["expired", "clear", "obstacle"])
def test_expired_depth_leaves_the_sent_packet_until_its_original_deadline(monkeypatch, next_depth):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"].update(watchdog_ms=card._config.watchdog_ms, max_obs_age_ms=100)
    first = card.next_command()
    assert card._stage_command(first, card._output_epoch)
    work = card._scheduled_command
    assert work.sent and work.repeat_until is not None
    original = packets[0].copy()
    original_expiry = work.expires_ms, work.expires_at, work.repeat_until

    advance_snapshot(snapshot, clock, 1100, vx=first["values"][0])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = (dict(objects, timestamp=1.1), dict(depth, frame_id=1000, capture_ms=1000, source_frame_id=1000),
                   odom, 1100)
    stale = card.next_command()
    assert not any(stale["values"]) and card._last_decision.status == policy.BLIND
    if next_depth == "obstacle":
        advance_snapshot(snapshot, clock, 1110, vx=first["values"][0], obstacle=True)
        snapshot[0][1].pop("capture_invalid", None)

    assert card._stage_command(stale, card._output_epoch)

    if next_depth == "obstacle":
        assert len(packets) == 2 and not any(packets[-1]["values"])
        return
    assert packets == [original]
    assert card._scheduled_command is work
    assert (work.expires_ms, work.expires_at, work.repeat_until) == original_expiry
    card._dispatch_command()
    assert packets == [original]

    if next_depth == "clear":
        advance_snapshot(snapshot, clock, 1110, vx=first["values"][0])
        objects, depth, odom, _ = snapshot[0]
        depth.pop("capture_invalid", None)
        snapshot[0] = dict(objects, timestamp=1.11), depth, odom, 1110
        resumed = card.next_command()
        assert any(resumed["values"])
        assert card._stage_command(resumed, card._output_epoch)
        assert len(packets) == 2 and any(packets[-1]["values"])
        return
    clock[0] = work.expires_ms
    card._dispatch_command()
    assert packets == [original] and card._scheduled_command is None


def test_view_prefers_the_wire_route_over_the_latest_preview(monkeypatch):
    card, packets, _, _, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1
    original = card._wire_dwa_origin.acceptance.route_points
    card._state.dwa_selected = False
    card._state.view_detour = local_path.Detour(None, ((0., 0.), (.3, -.2)), card._config.local_geometry)
    card._view_queue = queue.Queue(maxsize=1)
    card._view_hz = 10

    card._publish_view()
    snapshot = card._view_queue.get_nowait()
    assert snapshot["detour"].points == original
    assert not snapshot["project_detour"]


@pytest.mark.parametrize("target_missing", [False, True])
@pytest.mark.parametrize("changed", [False, True], ids=["same-scene", "new-scene"])
def test_camera_successor_uses_the_registered_path_with_no_odom(monkeypatch, target_missing, changed):
    card, packets, clock, snapshots, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    work = card._scheduled_command
    check = work.forward_check
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    row, col = local_path.START
    occupied[row+24:row+56, col-20] = True
    occupied[row+45, col-20:col+12] = True
    route = local_path.Detour(None, ((0., 0.), (.65, 0.), (1.1, .08), (1.4, .25)), check.geometry)
    check = replace(check, mount_frame=None, target_source=None, target_receipt_ms=1000, frame_id=1000,
                    direct_association=(SimpleNamespace(reference_ms=1000), None),
                    camera_passage=policy.CameraPassage(route, occupied, 1000, None, corridor_corner=target_missing))
    current = np.roll(occupied, -1, axis=0)
    monkeypatch.setattr(local_path, "project_observation", lambda *a, **kw: (~current, current))
    match = None if target_missing else SimpleNamespace(point=(2., 0.), range_m=2.)
    monkeypatch.setattr(plugin, "match_observation", lambda *a, **kw: match)
    card._wire_twist = (.18, 0., -.05)
    work.forward_check = check
    work.command_only = True
    work.dwa_origin = None
    work.camera_arc_end_ms = 1150
    clock[0] = 1100
    snapshots[0] = ({"timestamp": 1.1}, {"frame_id": 1100, "capture_ms": 1100}, None, 1100)

    assert card._prepare_camera_arc(work, clock[0])
    successor = work.successor
    renewed = successor.forward_check
    assert successor.expires_ms > 1150 and successor.message["values"][0] > 0.
    assert renewed.camera_passage.corner_only is target_missing
    assert renewed.camera_passage.capture_ms == 1100
    assert renewed.camera_passage.route.points[-1] == pytest.approx((1.35, .25), abs=.06)
    clock[0] = 1150
    if changed:
        objects, _, _, receipt = snapshots[0]
        snapshots[0] = objects, {"frame_id": 1150, "capture_ms": 1150}, None, receipt

    card._dispatch_command()

    assert len(packets) == (1 if changed else 2)


def planned_camera_card(monkeypatch, *, target_missing=False):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    state, config = card._state, card._config
    state.motion_source = "command"
    objects, depth, _, received = snapshot[0]
    if target_missing:
        objects = {"timestamp": .95, "objects": []}
        received = 950
    snapshot[0] = objects, depth, None, received
    route = local_path.Detour(None, ((0., 0.), (.7, 0.), (1.2, .65)), config.local_geometry)

    def select_camera_arc(**kwargs):
        state.active_scan.observe_single(kwargs["depth"], config.local_geometry,
                                         config.depth_age_limit_ms, clock[0])
        state.active_scan.observed_grid()
        _, _, _, status, reason = policy._use_camera_route(state, route, (.18, 0., -.08),
                                                         corner_only=target_missing)
        return policy.Decision([.18, 0., 0., 0., 0., -.08], status, reason,
                               distance_m=None if target_missing else 3.)

    monkeypatch.setattr(policy, "step", select_camera_arc)
    return card, packets, clock, snapshot


def test_camera_first_write_uses_configured_age_without_a_driver_declaration(monkeypatch):
    card, packets, _, _ = planned_camera_card(monkeypatch)
    card._descriptor["rate"].pop("max_obs_age_ms")
    card._config.max_obs_age_ms = 150
    message = card.next_command()
    assert any(message["values"])
    assert card._staged_forward_check[1].camera_passage is not None
    assert card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["stamp_ms"] + packet["ttl_ms"] == 1000 + card._config.max_obs_age_ms


def test_camera_corner_without_target_uses_depth_stamp_for_admission(monkeypatch):
    card, packets, _, snapshot = planned_camera_card(monkeypatch, target_missing=True)
    message = card.next_command()
    check = card._staged_forward_check[1]
    assert check.target_ms == 0 and check.target_receipt_ms == snapshot[0][3]
    assert card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["obs_stamp_ms"] == snapshot[0][1]["capture_ms"] and packet["values"][0] > 0.


def test_camera_first_write_cannot_cross_cutoff_during_evidence_comparison(monkeypatch):
    card, packets, clock, _ = planned_camera_card(monkeypatch)
    message = card.next_command()
    motion_inputs = plugin._motion_inputs

    def read_revision(*args):
        clock[0] = message["stamp_ms"] + message["ttl_ms"]
        return motion_inputs(*args)

    monkeypatch.setattr(plugin, "_motion_inputs", read_revision)
    assert card._stage_command(message, card._output_epoch)
    assert not packets


@pytest.mark.parametrize("queued", [False, True], ids=["first", "queued"])
@pytest.mark.parametrize("changed", [False, True], ids=["same-scene", "new-obstacle"])
def test_camera_write_checks_the_scene_used_to_sweep(monkeypatch, queued, changed):
    card, packets, clock, snapshot = planned_camera_card(monkeypatch)
    first = card.next_command()
    assert any(first["values"])
    cutoff = first["stamp_ms"] + first["ttl_ms"]
    if queued:
        assert card._stage_command(first, card._output_epoch)
        clock[0] = cutoff - 50
        objects, depth, _, _ = snapshot[0]
        snapshot[0] = (dict(objects, timestamp=clock[0] / 1000),
                       dict(depth, frame_id=clock[0], capture_ms=clock[0], source_frame_id=clock[0],
                            control_ms=clock[0]), None, clock[0])
        successor = card.next_command()
        assert any(successor["values"])
        old = card._scheduled_command
        assert card._stage_command(successor, card._output_epoch)
        assert old.successor is not None and len(packets) == 1
        clock[0] = cutoff
    if changed:
        objects, depth, _, received = snapshot[0]
        blocked = depth["map"].copy()
        blocked[220:370, 302:338] = .22
        snapshot[0] = objects, dict(depth, map=blocked, frame_id=clock[0] + 1), None, received
    if queued:
        card._dispatch_command()
    else:
        assert card._stage_command(first, card._output_epoch)

    assert len(packets) == queued + (not changed)
    assert all(any(p["values"]) for p in packets)


@pytest.mark.parametrize("has_route_evidence", [True, False])
def test_published_corner_continues_only_on_fresh_route_evidence_after_detector_miss(
        monkeypatch, has_route_evidence):
    card, packets, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    active = card._scheduled_command
    original = active.dwa_origin
    old = replace(original, acceptance=replace(original.acceptance, corridor_corner=True))
    active.dwa_origin = card._wire_dwa_origin = old
    advance_snapshot(snapshot, clock, 1100, vx=first["values"][0])
    _, depth, feedback, _ = snapshot[0]
    snapshot[0] = ({"timestamp": 1.1, "objects": []}, depth, feedback, 1100)
    if not has_route_evidence:
        monkeypatch.setattr(policy, "_follow_published_route", lambda *args, **kwargs: None)

    next_message = card.next_command()

    if not has_route_evidence:
        assert not any(next_message["values"])
        assert card._last_decision.status == policy.BLIND
        return
    assert next_message["values"][0] > 0.
    accepted = card._last_decision.dwa_acceptance
    assert accepted is not None and accepted.motion_guard.corner_only
    assert accepted.motion_guard.check.accepted
    origin = card._staged_dwa_command[1]
    rejection, checked = card._recheck_dwa_inputs(origin, next_message["values"], clock[0])
    assert rejection is None and checked is not None and checked[2]
    assert len(packets) == 1


@pytest.mark.parametrize("feedback_vx,admitted", [(.1, True), (.45, False)])
def test_corner_first_write_keeps_the_route_end_when_feedback_changes(monkeypatch, feedback_vx, admitted):
    card, _, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    active = card._scheduled_command
    active.dwa_origin = card._wire_dwa_origin = replace(
        active.dwa_origin, acceptance=replace(active.dwa_origin.acceptance, corridor_corner=True))
    advance_snapshot(snapshot, clock, 1050, vx=.1)
    _, depth, feedback, _ = snapshot[0]
    snapshot[0] = ({"timestamp": 1.05, "objects": []}, depth, feedback, 1050)

    message = card.next_command()
    origin = card._staged_dwa_command[1]
    guard = origin.acceptance.motion_guard
    assert guard.corner_only and guard.check.accepted
    # The route has more room than the original complete stop. Only the
    # stopping sweep rebuilt from the faster feedback exceeds its end.
    room = guard.trace.body_travel_m + guard.trace.delay_travel_error + .04
    origin = replace(origin, acceptance=replace(
        origin.acceptance, route_points=((0., 0.), (room, 0.))))
    advance_snapshot(snapshot, clock, 1060, vx=feedback_vx)

    rejection, checked = card._recheck_dwa_inputs(origin, message["values"], clock[0])

    if admitted:
        assert rejection is None and checked is not None and checked[2]
        assert checked[3].motion_guard.measured[0] == feedback_vx
    else:
        assert rejection == "insufficient_room" and checked is None


@pytest.mark.parametrize("feedback_vx,brakes", [(.10, False), (.45, True)])
def test_committed_corner_brakes_before_the_route_end_on_faster_feedback(monkeypatch, feedback_vx, brakes):
    card, packets, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    active = card._scheduled_command
    origin = active.dwa_origin
    guard = origin.forward_check.motion_guard
    active.dwa_origin = card._wire_dwa_origin = replace(
        origin, acceptance=replace(origin.acceptance, corridor_corner=True,
                                   route_points=((0., 0.), (.33, 0.))))
    advance_snapshot(snapshot, clock, 1010, vx=feedback_vx)

    card._dispatch_command()

    assert len(packets) == 1 + brakes and any(packets[0]["values"])
    if brakes:
        stop = packets[-1]
        assert not any(stop["values"]) and stop["stamp_ms"] < guard.deadline_ms


def test_reused_route_can_repeat_within_its_original_deadline(monkeypatch):
    card, packets, clock, _, message = selected_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms
    assert card._stage_command(message, card._output_epoch)
    active = card._scheduled_command
    assert active.repeat_until is not None
    origin = replace(active.dwa_origin, acceptance=replace(active.dwa_origin.acceptance, followed_route=True))
    active.dwa_origin = card._wire_dwa_origin = origin
    clock[0] = packets[0]["stamp_ms"] + packets[0]["ttl_ms"] - 50

    monkeypatch.setattr(card, "_registered_dwa_scene",
                        lambda *a, **kw: pytest.fail("unchanged inputs rebuilt the stopping sweep"))
    card._dispatch_command()
    first, repeat = packets
    assert repeat["values"] == first["values"]
    assert repeat["obs_stamp_ms"] == first["obs_stamp_ms"]
    assert repeat["stamp_ms"] + repeat["ttl_ms"] == first["stamp_ms"] + first["ttl_ms"]


def test_next_arc_is_ready_at_the_cutoff_without_an_intermediate_zero(monkeypatch):
    card, packets, clock, snapshot, first = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(first, card._output_epoch)
    old = card._scheduled_command
    origin = old.dwa_origin
    old.dwa_origin = card._wire_dwa_origin = replace(
        origin, acceptance=replace(origin.acceptance, corridor_corner=True))
    cutoff = old.forward_check.motion_guard.deadline_ms
    advance_snapshot(snapshot, clock, cutoff - 45, vx=first["values"][0])
    successor = card.next_command()
    assert card._staged_dwa_command is not None
    assert card._stage_command(successor, card._output_epoch)
    assert old.successor is not None
    assert len(packets) == 1

    clock[0] = cutoff
    card._dispatch_command()
    assert len(packets) == 2
    assert all(any(packet["values"]) for packet in packets)
    assert packets[-1]["stamp_ms"] == cutoff
    assert packets[-1]["seq"] > packets[0]["seq"]
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] == (
        successor["stamp_ms"] + successor["ttl_ms"])


def test_operator_can_cancel_an_admitted_arc(monkeypatch):
    card, packets, clock, _, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    assert card._arc_in_progress(card._scheduled_command, clock[0])
    card._cancel_scheduled_motion(block=True)
    assert packets[-1]["values"] == [0.] * 6
    assert card._scheduled_command is None


@pytest.mark.parametrize("missing", ["depth", "feedback"])
@pytest.mark.parametrize("recovers", [True, False])
def test_first_write_recovers_only_with_valid_replacement(monkeypatch, missing, recovers):
    card, packets, _, snapshot, message = selected_card(monkeypatch)
    objects, depth, odom, stamp = snapshot[0]
    partial = (objects, None if missing == "depth" else depth, None if missing == "feedback" else odom, stamp)
    reads = iter([partial])
    replacement = snapshot[0] if recovers else partial
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: next(reads, replacement))
    card._publish_command(message)
    if recovers:
        packet, = packets
        assert packet["values"] == message["values"]
    else:
        assert not packets
        assert card._dwa_publication["event"] == "veto"
        assert card._dwa_publication["reason"] == ("stale_observation" if missing == "depth" else "no_feedback")


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
        card._stage_command(message, card._output_epoch)
    assert not packets
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
        card._descriptor["limits"] = {"min_magnitude": [abs(message["values"][0]) + .01, 0., 0., 0., 0., 0.]}
    else:
        card._descriptor["limits"] = {"lower": [-1.] * 6, "upper": [.01, 1., 1., 1., 1., 1.]}
    if change in ("deadband", "axis", "descriptor"):
        # Exercise the wire contract independently of the snapshot-identity
        # gate. The real sampled twist still violates this explicit limit.
        origin = replace(origin, constraints=card._dwa_constraints())
    card._publish_command(message, dwa_origin=origin)
    assert not packets
    publication = card._info()["dwa"]["publication"]
    assert publication["event"] == "veto"
    assert publication["reason"] == "postcheck_veto"
    assert publication["veto_stage"] == stage and publication["veto_detail"] == detail
    assert card._state.dwa_trace["veto_stage"] == stage
    assert card._state.dwa_trace["veto_detail"] == detail
    assert card._state.dwa_totals.get("stop_packets", 0) == 0


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
    assert not packets
    assert not state.dwa_selected
    assert card._last_decision.dwa_acceptance is None
    assert state.dwa_trace["accepted_twist"] is None
    assert state.dwa_trace["proposed_twist"] is not None
    assert state.dwa_trace["veto_reason"] == "publish_expired"
    assert state.dwa_trace["fallback_reason"] == "publication_veto"
    assert card._dwa_publication["event"] == "veto"
    assert "wire_values" not in card._dwa_publication
    assert card._dwa_publication["decision_id"] == state.decision_id
    assert state.dwa_totals["attempted_ticks"] == 1
    assert state.dwa_totals["accepted_ticks"] == 0


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
def test_publish_error_never_counts_a_dwa_success(monkeypatch, body):
    card, _, _, _, message = selected_card(monkeypatch, body=body)
    attempts = []

    def fail(payload):
        attempts.append(json.loads(payload.data)["values"])
        raise RuntimeError("transport failed")

    card._publisher = SimpleNamespace(publish=fail)
    if body:
        with pytest.raises(RuntimeError, match="transport failed"):
            card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
        assert card._scheduled_command is None
    assert attempts == [message["values"]]
    assert card._published == 0
    assert not card._wire_moving
    assert card._state.dwa_totals.get("published_packets", 0) == 0
    assert card._state.dwa_totals.get("published_decisions", 0) == 0
    assert card._info()["dwa"]["publication"]["reason"] == "publish_error"
    assert "wire_values" not in card._dwa_publication
    assert "wire_seq" not in card._state.dwa_trace


def test_expiry_reports_the_sent_decision_while_the_planner_selects_another(monkeypatch):
    card, packets, clock, _, message = selected_card(monkeypatch, body=False, live_target=True)
    original = card._last_decision.dwa_acceptance
    assert card._stage_command(message, card._output_epoch)
    card._state.dwa_trace = {"decision_id": original.decision_id + 1}
    clock[0] = packets[0]["stamp_ms"] + packets[0]["ttl_ms"]
    card._dispatch_command()
    assert len(packets) == 1 and card._scheduled_command is None
    assert card._info()["dwa"]["publication"]["decision_id"] == original.decision_id
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
            card._descriptor["limits"] = {"min_magnitude": [abs(decision.values[0]) + .01, 0., 0., 0., 0., 0.]}
        else:
            config.vx_max = .01
        return decision

    monkeypatch.setattr(policy, "step", change_after_selection)
    message = card.next_command()
    assert message["values"] == [0.] * 6
    assert card._last_decision.dwa_acceptance is None
    card._publish_command(message)
    assert not packets
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
        assert len(packets) == 1
        card._tick()
        assert len(packets) == 2 and packets[-1]["values"] != message["values"]
        return
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


def test_timer_does_not_reissue_motion_on_fresh_measured_overspeed(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100, vx=.36)
    assert policy._feedback_speed_reduction(card._config, snapshot[0][2]) < 1.
    card._dispatch_command()
    assert packets[-1]["values"] == message["values"]
    assert card._state.dwa_totals["published_packets"] == 1


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


def test_expired_old_work_logs_one_veto_without_writing_a_stop(monkeypatch, caplog):
    card, packets, clock, _, message = selected_card(monkeypatch, body=False)
    original = card._last_decision.dwa_acceptance
    assert card._stage_command(message, card._output_epoch)
    next_trace = dict(decision_id=original.decision_id + 1, veto_reason="newer_decision_reason",
                      accepted_twist=(.1, 0., 0.))
    card._state.dwa_trace = dict(next_trace)
    clock[0] = message["stamp_ms"] + message["ttl_ms"]
    with caplog.at_level(logging.INFO, logger=plugin.__name__):
        card._dispatch_command()
    assert len(packets) == 1 and card._scheduled_command is None
    assert card._state.dwa_trace == next_trace
    assert card._state.dwa_totals.get("stop_packets", 0) == 0
    publication = card._info()["dwa"]["publication"]
    assert publication["decision_id"] == original.decision_id
    assert publication["reason"] == "publish_expired"
    events = [json.loads(record.getMessage().split("navi dwa publication: ", 1)[1])
              for record in caplog.records if record.getMessage().startswith("navi dwa publication: ")]
    assert len([event for event in events if event["event"] == "veto"]) == 1


def test_dwa_logs_sample_steady_output_and_report_transitions(monkeypatch, caplog):
    card, packets, _, _ = observation_card(monkeypatch)
    state = card._state
    state.active_scan.current_frame = 0
    state.decision_id = 0

    def step(**_):
        state.decision_id += 1
        stopping = state.decision_id == 102
        state.dwa_trace = dict(decision_id=state.decision_id, controller_used="legacy",
                               branch="direct", select_calls=0)
        return policy.Decision([0. if stopping else .2, 0., 0., 0., 0., 0.], policy.APPROACHING)

    monkeypatch.setattr(policy, "step", step)
    with caplog.at_level(logging.INFO, logger=plugin.__name__):
        for _ in range(104):
            card._publish_command(card.next_command())

    decisions, publications = [], []
    for record in caplog.records:
        if record.msg == "navi dwa decision: %s":
            decisions.append(json.loads(record.args[0])["decision_id"])
        elif record.msg == "navi dwa publication: %s":
            event = json.loads(record.args[0])
            publications.append((event["decision_id"], event["event"]))
    assert decisions == [1, 100, 102, 103]
    assert publications == [(1, "published"), (100, "published"), (102, "stopped"), (103, "published")]
    assert len(packets) == 104
    assert [pkt["seq"] for pkt in packets if not any(pkt["values"])] == [packets[101]["seq"]]
    assert card._info()["dwa"]["publication"]["wire_seq"] == packets[-1]["seq"]
    assert card._info()["dwa"]["totals"]["stop_packets"] == 1


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
    assert len(packets) == 1 and card._scheduled_command is None


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
    if live_target:
        config.vx_max = .15
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
    assert 0. < message["values"][0] <= config.vx_max
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
            card._config.vx_max = .15
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


def test_fresh_feedback_at_a_local_endpoint_does_not_extend_the_sent_window(monkeypatch):
    card, packets, clock, snapshot, message = short_advance_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    deadline = packets[0]["stamp_ms"] + packets[0]["ttl_ms"]
    advance_snapshot(snapshot, clock, deadline - 1)
    card._dispatch_command()
    assert len(packets) == 1
    advance_snapshot(snapshot, clock, deadline)
    card._dispatch_command()
    assert len(packets) == 1 and card._scheduled_command is None


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
    assert accepted is not None and 0. < accepted.command[0] <= cap
    assert accepted.command[1:] == (0., 0.)
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
    card._halt(True)
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
    assert card._state.dwa_totals["stop_reasons"] == {"superseded": 1}

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
    assert len(packets) == 1 and card._scheduled_command is None

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
    assert len(packets) == 1
    assert card._dwa_publication["reason"] == "publish_expired"


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


def test_new_hit_is_handled_by_next_plan(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False)
    assert card._stage_command(message, card._output_epoch)
    advance_snapshot(snapshot, clock, 1020, obstacle=True)
    card._dispatch_command()
    assert len(packets) == 1
    card._tick()
    assert len(packets) == 2 and packets[-1]["values"] != message["values"]


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
    assert not packets
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

    if memory_frame == "odom":
        # The retained cell is 1.9 m behind this grid origin. A valid empty
        # projection must not erase the real selector's complete-stop proof.
        packet, = packets
        assert packet["values"] == message["values"]
        assert card._state.dwa_totals["published_decisions"] == 1
    else:
        # An unavailable projection is different: the old obstacle has no
        # registration to this robot pose and must still prevent publication.
        assert not packets
        assert card._dwa_publication["reason"] == "postcheck_veto"
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
    elif event == "expired":
        assert not packets
        assert card._dwa_publication["reason"] == "publish_expired"
    else:
        assert all(packet["values"] == [0.] * 6 for packet in packets)
    assert bool(card._state.dwa_totals.get("published_decisions", 0)) is (event == "continuous")
    assert scan.near_obstacles.cells == retained


def _moving_memory_case(monkeypatch, layer, world_y, path):
    from plugins.navi.odom import PlanarPose
    card, packets, clock, snapshot = observation_card(monkeypatch, body=layer == "historical")
    card._config.vx_max = .15
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
        assert len(packets) == 1 and packets[0]["values"] == message["values"]
        return
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
    assert len(packets) == 1 and packets[0]["values"] == message["values"]


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
def test_direct_tracking_expired_during_planning_records_the_unpublished_decision(monkeypatch, body):
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
    assert not packets
    publication, trace = card._dwa_publication, card._state.dwa_trace
    assert publication["decision_id"] == trace["decision_id"]
    assert publication["reason"] == "publish_expired"
    assert trace["fallback_reason"] == "publication_veto"
    assert trace["final_twist"][0] > 0.
    assert "wire_values" not in publication and "wire_values" not in trace


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
    card._config.vx_max = .15
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
    config.vx_max = .15
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


@pytest.mark.parametrize("new_input", [False, True])
def test_sent_dwa_command_runs_to_its_wire_deadline_without_rechecking(monkeypatch, new_input):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=False, live_target=True)
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and any(packets[0]["values"])
    deadline = packets[0]["stamp_ms"] + packets[0]["ttl_ms"]
    if new_input:
        advance_snapshot(snapshot, clock, deadline - 1, obstacle=True)
    else:
        clock[0] = deadline - 1
    monkeypatch.setattr(card, "_recheck_forward", lambda *a, **kw: pytest.fail("sent decision rechecked"))
    card._dispatch_command()
    assert len(packets) == 1
    clock[0] = deadline
    card._dispatch_command()
    assert len(packets) == 1 and card._scheduled_command is None


def test_admitted_drive_outlives_its_packet_ttl_until_the_declared_watchdog(monkeypatch):
    card, packets, clock, _, message = selected_card(monkeypatch, body=False)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    card._ttl_ms = 500
    # Reselect with the negotiated watchdog included in the prediction.
    message = card.next_command()
    assert card._last_decision.dwa_acceptance
    message["ttl_ms"] = 112
    assert card._stage_command(message, card._output_epoch)
    packet, = packets
    assert packet["ttl_ms"] == 112 and packet["values"][0] > 0
    for stamp in (1112, 1285, 1499):
        clock[0] = stamp
        card._dispatch_command()
        assert len(packets) == 1
    clock[0] = 1500
    card._dispatch_command()
    assert len(packets) == 1 and card._scheduled_command is None


@pytest.mark.parametrize("body", [True, False], ids=["direct", "scheduled"])
@pytest.mark.parametrize("update", ["fresh", "same_target", "obstacle", "missing_target", "no_new_input"])
def test_first_send_rechecks_new_evidence_when_publication_outlasts_proof(monkeypatch, update, body):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    original_deadline = message["stamp_ms"] + message["ttl_ms"]
    finish = original_deadline + 10
    recheck = card._recheck_dwa_motion
    calls = []

    def during_publication(*args, **kwargs):
        checked = recheck(*args, **kwargs)
        if not calls:
            if update == "no_new_input":
                clock[0] = finish
            else:
                advance_snapshot(snapshot, clock, finish, obstacle=(update == "obstacle"))
                objects, depth, odom, target_ms = snapshot[0]
                if update != "same_target":
                    objects = dict(objects, timestamp=finish / 1000)
                    target_ms = finish
                if update == "missing_target":
                    objects["objects"] = []
                snapshot[0] = objects, depth, odom, target_ms
        calls.append(checked)
        return checked

    monkeypatch.setattr(card, "_recheck_dwa_motion", during_publication)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    if update == "no_new_input":
        assert not packets
        assert card._dwa_publication["reason"] == "publish_expired"
    elif update in ("fresh", "same_target"):
        packet, = packets
        assert packet["values"] == message["values"]
        assert packet["stamp_ms"] == finish
        assert packet["obs_stamp_ms"] == (finish if update == "fresh" else message["obs_stamp_ms"])
        assert 0 < packet["ttl_ms"] <= card._config.max_feedback_gap_ms
        assert card._wire_dwa_origin.forward_check.motion_guard.deadline_ms == finish + packet["ttl_ms"]
        if not body:
            assert card._scheduled_command.expires_ms == finish + packet["ttl_ms"]
            clock[0] += 1
            card._dispatch_command()
            assert len(packets) == 1
    else:
        packet, = packets
        assert packet["values"] == [0.] * 6


@pytest.mark.parametrize("change", ["payload", "receipt_only", "new_decision", "previously_sent"])
def test_first_send_cannot_renew_superseded_evidence(monkeypatch, change):
    card, _, clock, snapshot, message = selected_card(monkeypatch, live_target=True)
    origin = card._staged_dwa_command[1]
    finish = message["stamp_ms"] + message["ttl_ms"] + 10
    advance_snapshot(snapshot, clock, finish)
    check_motion = card._recheck_dwa_motion

    def change_during_check(*args, **kwargs):
        result = check_motion(*args, **kwargs)
        if change != "receipt_only":
            advance_snapshot(snapshot, clock, finish + 1)
        objects, depth, odom, receipt = snapshot[0]
        if change == "payload":
            objects = dict(objects, objects=[dict(objects["objects"][0], confidence=.94)])
            receipt = finish + 1
        elif change == "receipt_only":
            receipt = finish
        elif change == "new_decision":
            card._state.decision_id += 1
        else:
            card._dwa_last_published = origin.identity
        snapshot[0] = objects, depth, odom, receipt
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", change_during_check)
    reason, checked = card._recheck_dwa_inputs(origin, message["values"], finish)
    assert reason and checked is None


@pytest.mark.parametrize("depth_age_ms,search_ms", [(700, 225), (850, 125), (1000, None)])
def test_usable_input_window_bounds_dwa_search(monkeypatch, depth_age_ms, search_ms):
    card, _, clock, snapshot = observation_card(monkeypatch, live_target=True)
    card._descriptor["rate"]["max_obs_age_ms"] = 1000
    card._config.max_depth_age_ms = 1200
    advance_snapshot(snapshot, clock, 1500)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = (objects, dict(depth, capture_ms=1500 - depth_age_ms), odom, 1500)

    def step(**kwargs):
        time_left = kwargs["dwa_time_left"]
        if search_ms is not None:
            assert time_left() == pytest.approx(search_ms / 1000)
            clock[0] += search_ms
            assert time_left() == pytest.approx(0.)
        else:
            assert time_left is None
        return policy.Decision(None, policy.SEARCHING)

    monkeypatch.setattr(policy, "step", step)
    assert card.next_command() is None


@pytest.mark.parametrize("update", ["new_depth", "obstacle", "unchanged"])
def test_route_expiring_during_planning_rechecks_only_fresh_inputs(monkeypatch, update):
    card, packets, clock, snapshot = observation_card(monkeypatch, live_target=True)
    card._config.max_obs_age_ms = 500
    card._config.max_depth_age_ms = 1200
    card._ttl_ms = 140
    step = policy.step
    planned = []

    def slow_plan(**kwargs):
        decision = step(**kwargs)
        planned.append(decision.dwa_acceptance)
        if update == "unchanged":
            clock[0] = 1160
        else:
            advance_snapshot(snapshot, clock, 1160, obstacle=update == "obstacle")
        return decision

    monkeypatch.setattr(policy, "step", slow_plan)
    message = card.next_command()

    accepted, = planned
    assert accepted.route_points and accepted.motion_guard.association is not None
    assert accepted.motion_guard.deadline_ms == 1140
    if update == "new_depth":
        assert message["values"][0] > 0.
        card._publish_command(message)
        packet, = packets
        assert packet["values"] == message["values"]
        assert packet["stamp_ms"] == 1160 and packet["obs_stamp_ms"] == 1000
        assert 0 < packet["ttl_ms"] <= 140
        assert card._wire_dwa_origin.forward_check.depth_ms == 1160
    else:
        assert not any(message["values"])
        assert card._dwa_publication["reason"] == (
            "footprint_collision" if update == "obstacle" else "publish_expired")


@pytest.mark.parametrize("update", ["clear", "obstacle", "same_capture"])
def test_late_depth_can_be_replaced_before_a_fast_first_send(monkeypatch, update):
    card, packets, clock, snapshot = observation_card(monkeypatch, live_target=True)
    card._descriptor["rate"]["max_obs_age_ms"] = 1000
    card._config.max_depth_age_ms = 1200
    objects, depth, odom, receipt = snapshot[0]
    snapshot[0] = objects, dict(depth, capture_ms=50, source_frame_id=50), odom, receipt
    message = card.next_command()
    assert message["values"][0] > 0.
    original_end = message["stamp_ms"] + message["ttl_ms"]
    assert original_end == 1050
    advance_snapshot(snapshot, clock, 1040, obstacle=update == "obstacle")
    objects, depth, odom, receipt = snapshot[0]
    capture = 50 if update == "same_capture" else 340
    snapshot[0] = objects, dict(depth, capture_ms=capture, source_frame_id=capture), odom, receipt

    card._publish_command(message)

    if update == "obstacle":
        assert all(not any(p["values"]) for p in packets)
        assert card._dwa_publication["reason"] == "footprint_collision"
    else:
        packet, = packets
        assert packet["values"] == message["values"]
        end = packet["stamp_ms"] + packet["ttl_ms"]
        if update == "clear":
            assert original_end < end <= 1340
        else:
            assert end == original_end


@pytest.mark.parametrize("obstacle", [False, True])
@pytest.mark.parametrize("use_obstacle_history", [True, False])
def test_receipt_timed_depth_rechecks_first_send_and_repeat(monkeypatch, obstacle, use_obstacle_history):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_obstacle_history = use_obstacle_history
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    snapshot[0][1].update(receipt_timed=True, capture_ms=100, source_frame_id=100)
    message = card.next_command()
    assert message["values"][0] > 0 and message["obs_stamp_ms"] == 1000
    assert card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"][0] > 0
    advance_snapshot(snapshot, clock, 1100)
    snapshot[0][1].update(capture_ms=150, source_frame_id=150)
    snapshot[0][2].update(received_speed_ms=1100, component_speed_ms=1100)
    if obstacle:
        snapshot[0][1]["map"][220:370, 302:338] = .10
    card._dispatch_command()
    assert len(packets) == 2
    if obstacle:
        assert not any(packets[-1]["values"])
        assert card._dwa_publication["reason"] == "footprint_collision"
    else:
        assert packets[-1]["values"][0] > 0


def test_slow_first_send_uses_one_check_for_the_new_window(monkeypatch):
    card, packets, clock, snapshot, message = selected_card(monkeypatch, live_target=True)
    old_end = message["stamp_ms"] + message["ttl_ms"]
    advance_snapshot(snapshot, clock, old_end - 50)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=clock[0] / 1000), depth, odom, clock[0]
    step = card._recheck_dwa_inputs
    checks = []

    def slow_publish(*args, **kwargs):
        checked = step(*args, **kwargs)
        checks.append(checked)
        clock[0] += 100
        return checked

    monkeypatch.setattr(card, "_recheck_dwa_inputs", slow_publish)
    card._publish_command(message)
    assert len(checks) == 1
    assert packets[0]["values"] == message["values"]
    assert old_end < packets[0]["stamp_ms"] < packets[0]["stamp_ms"] + packets[0]["ttl_ms"]


@pytest.mark.parametrize("change", [
    "stationary", "depth", "target", "nonzero_sample", "pose", "frame", "gap",
    "disjoint_history", "source_regression", "pose_regression", "missing_pose",
])
def test_only_contiguous_exactly_stationary_feedback_preserves_a_check(change):
    stopped = (0., 0., 0.)
    pose = plugin.odom_mod.PlanarPose(1000, ("odom",), 1., 2., .5)
    prior = dict(vx=0., vy=0., wz=0., source_sample_id=1000, stamp_ms=1000,
                 motion_pose=pose, obstacle_pose=pose, velocity_history=((1000, 1000, stopped),))
    current_pose = replace(pose, stamp_ms=1100)
    history = prior["velocity_history"] + ((1050, 1050, stopped), (1100, 1100, stopped))
    current = dict(prior, source_sample_id=1100, stamp_ms=1100, motion_pose=current_pose,
                   obstacle_pose=current_pose, velocity_history=history)
    target = depth = None
    if change == "depth":
        depth = {"map": np.ones((1, 1))}
    elif change == "target":
        target = {"objects": []}
    elif change == "nonzero_sample":
        current["velocity_history"] = history[:1] + ((1050, 1050, (.000001, 0., 0.)), history[-1])
    elif change == "pose":
        current["obstacle_pose"] = replace(current_pose, x=1.0000001)
    elif change == "frame":
        current["motion_pose"] = replace(current_pose, frame=("other",))
    elif change == "gap":
        current["velocity_history"] = history[:-1] + ((1450, 1100, stopped),)
    elif change == "disjoint_history":
        current["velocity_history"] = history[1:]
    elif change == "source_regression":
        current["velocity_history"] = history[:1] + ((1050, 900, stopped), history[-1])
    elif change == "pose_regression":
        current["motion_pose"] = replace(current_pose, stamp_ms=999)
    elif change == "missing_pose":
        current["motion_pose"] = None
    old_revision = plugin._motion_inputs(None, None, prior)
    new_revision = plugin._motion_inputs(target, depth, current)
    assert plugin._continuous_feedback_update(old_revision, new_revision, 350, stationary=True) == (change == "stationary")


def test_old_certified_obstacles_cover_only_expired_prefix_of_identical_memory():
    def view(stamp):
        return stamp, None, (), (), ((stamp - 10, None),)

    early, late, newest = view(1000), view(1100), view(1200)
    before = ("frame", (early, late))
    now = 1000 + policy.MAP_AGE_MS
    assert plugin._only_expired_views(before, ("frame", (late,)), now)
    assert not plugin._only_expired_views(before, ("frame", (late,)), now - 20)
    assert not plugin._only_expired_views(before, ("other", (late,)), now)
    assert not plugin._only_expired_views(before, ("frame", (late, newest)), now)
    assert not plugin._only_expired_views(before, ("frame", (early,)), now)
    assert not plugin._only_expired_views(before, ("frame", (newest,)), now)
    assert not plugin._only_expired_views(before, ("frame", ()), now + 200)


@pytest.mark.parametrize("body", [True, False])
@pytest.mark.parametrize("jump", [-.65, .65])
def test_pose_reset_during_publication_cannot_shift_the_checked_scene(monkeypatch, body, jump):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True)
    objects, depth, odom, stamp = snapshot[0]
    pose = plugin.odom_mod.PlanarPose(1000, ("odom",), 0., 0., 0.)
    snapshot[0] = objects, depth, dict(odom, motion_pose=pose, obstacle_pose=pose), stamp
    message = card.next_command()
    assert message["values"][0] > 0
    advance_snapshot(snapshot, clock, 1020)
    objects, depth, odom, stamp = snapshot[0]
    changed = replace(pose, x=jump, stamp_ms=1020)
    history = odom["velocity_history"] + ((1020, 1020, (.1, 0., 0.)),)
    snapshot[0] = objects, depth, dict(odom, motion_pose=changed, obstacle_pose=changed, velocity_history=history), stamp
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert not packets
    assert card._dwa_publication["reason"] == "geometry_mismatch"


@pytest.mark.parametrize("body", [True, False])
@pytest.mark.parametrize("obstacle", [True, False])
def test_filtered_feedback_sweep_agrees_with_exact_registered_scene(monkeypatch, body, obstacle):
    card, _, clock, snapshot, message = selected_card(monkeypatch, body=body, live_target=True)
    check = card._staged_dwa_command[1].forward_check
    advance_snapshot(snapshot, clock, 1050, obstacle=obstacle, vx=.105)
    outcomes = []
    for filtered in (False, True):
        diagnostics = {}
        verified = card._recheck_dwa_motion(check, message["values"], 800, clock[0], snapshot[0], renew=False,
                                            diagnostics=diagnostics, bounds_only=filtered)
        assert verified is not None
        outcomes.append((verified[:3], diagnostics.get("reason")))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0][2] == (not obstacle)


@pytest.mark.parametrize("change,compact,stationary", [
    ("moving", True, False), ("stationary", True, True),
    ("changed_twist", False, False), ("intervening_twist", False, False),
    ("regressed_source", False, False), ("missing_overlap", False, False),
    ("coordinate_jump", False, False), ("retimed", False, False), ("unchanged_history", False, False),
])
def test_repeated_speed_sample_with_independently_updated_pose(change, compact, stationary):
    rate = (0., 0., 0.) if change == "stationary" else (.15, 0., 0.)
    pose = plugin.odom_mod.PlanarPose(995, ("odom",), .5, 0., 0.)
    history = ((1000, 995, rate),)
    before = dict(vx=rate[0], vy=0., wz=0., stamp_ms=995, source_sample_id=995,
                  motion_pose=pose, obstacle_pose=pose, velocity_history=history)
    latest = replace(pose, x=pose.x + (.003 if rate[0] else 0.), stamp_ms=1015)
    if change == "coordinate_jump":
        latest = replace(latest, x=latest.x + 1.)
    mid_rate = (.12, 0., 0.) if change == "intervening_twist" else rate
    end_rate = (.1, 0., 0.) if change == "changed_twist" else rate
    tail = ((1010, 994 if change == "regressed_source" else 995, mid_rate), (1020, 995, end_rate))
    after = dict(before, vx=end_rate[0], motion_pose=latest, obstacle_pose=latest,
                 velocity_history=tail if change == "missing_overlap" else history + tail)
    if change == "retimed":
        after["stamp_ms"] = 1020
    elif change == "unchanged_history":
        after["velocity_history"] = history
    old = plugin._motion_inputs(None, None, before)
    new = plugin._motion_inputs(None, None, after)
    assert plugin._continuous_feedback_update(old, new, 350, stationary=False) == compact
    assert plugin._continuous_feedback_update(old, new, 350, stationary=True) == stationary


@pytest.mark.parametrize("stamp, speed_stamps, published", [
    (1100, {}, True),
    (1100, {"sample_timestamps_ms": {"speed": 1100}}, True),
    (1000, {}, False),
    (995, {}, False),
    (1101, {}, False),
    (1100, {"sample_timestamps_ms": {"speed": 1000}}, False),
    (1100, {"sample_timestamps_ms": {"pose": 1100}}, False),
])
def test_route_tracks_received_odometry_with_existing_sample_stamp(monkeypatch, stamp, speed_stamps, published):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 600

    def receive_odometry(stamp_ms, vendor):
        payload = dict(schema="motus.odom/1", frame="body", stamp_ms=stamp_ms, pose=None,
                       twist=[.1, 0., None, None, None, 0.],
                       vendor={"stamp_source": "received", **vendor})
        card._on_string("odom", SimpleNamespace(data=json.dumps(payload)))
        odom = plugin.NaviPlugin._observation_snapshot(card, with_capture_timing=True)[2]
        objects, depth, _, receipt = snapshot[0]
        snapshot[0] = objects, depth, odom, receipt

    receive_odometry(1000, {})
    assert card._stage_command(card.next_command(), card._output_epoch)
    assert len(packets) == 1 and packets[0]["values"][0] > 0
    advance_snapshot(snapshot, clock, 1100)
    receive_odometry(stamp, speed_stamps)
    card._dispatch_command()
    assert any(p["values"][0] > 0 for p in packets[1:]) == published


@pytest.mark.parametrize("new_vx,elapsed_ms,written", [
    (.02, 37, True), (.025, 37, False), (.02, 120, False),
], ids=["unchanged", "speed-changed", "outside-swept-margin"])
def test_route_repeat_accepts_unchanged_speed_only_feedback_during_sweep(monkeypatch, new_vx, elapsed_ms, written):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500

    def receive_speed(stamp, vx=.02):
        clock[0] = stamp
        payload = dict(schema="motus.odom/1", frame="body", stamp_ms=stamp,
                       twist=[vx, 0., None, None, None, -.004],
                       vendor={"stamp_source": "received"})
        card._on_string("odom", SimpleNamespace(data=json.dumps(payload)))
        odom = plugin.NaviPlugin._observation_snapshot(card, with_capture_timing=True)[2]
        assert odom.get("motion_pose") is None
        objects, depth, _, receipt = snapshot[0]
        snapshot[0] = objects, depth, odom, receipt

    receive_speed(1000)
    assert card._stage_command(card.next_command(), card._output_epoch)
    cutoff = card._scheduled_command.expires_ms
    advance_snapshot(snapshot, clock, 1100)
    receive_speed(1100)
    recheck = card._recheck_dwa_motion

    def speed_arrives_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        receive_speed(clock[0] + elapsed_ms, new_vx)
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", speed_arrives_during_sweep)
    card._dispatch_command()
    assert len(packets) == (2 if written else 1)
    if written:
        assert packets[-1]["values"][0] > 0.
        assert packets[-1]["stamp_ms"] == 1100 + elapsed_ms
        assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= cutoff


@pytest.mark.parametrize("change", [
    "clear", "obstacle", "obstacle_new_odom", "missing_target", "same_speed", "partial_speed",
    "lost_speed", "cancel", "late", "obstacle_failed_stop", "obstacle_duplicate_speed",
    "new_depth_clear", "new_depth_obstacle", "second_depth_clear", "second_depth_obstacle",
])
def test_bounded_watchdog_repeat_requires_new_clear_sweep(monkeypatch, change):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    message = card.next_command()
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and packets[0]["values"][0] > 0
    advance_snapshot(snapshot, clock, 1100)
    if change.startswith("obstacle"):
        snapshot[0][1]["map"][220:370, 302:338] = .10
    snapshot[0][2].update(received_speed_ms=1100, component_speed_ms=1000 if change in ("same_speed", "obstacle_duplicate_speed") else 1100)
    if change == "partial_speed":
        snapshot[0][2]["received_speed_ms"] = None
    if change == "missing_target":
        _, depth, odom, _ = snapshot[0]
        snapshot[0] = {"objects": [], "timestamp": 1.1}, depth, odom, 1100
    recheck = card._recheck_dwa_motion
    checks = 0

    def during_repeat(*args, **kwargs):
        nonlocal checks
        result = recheck(*args, **kwargs)
        checks += 1
        if (change.startswith("new_depth") and checks == 1
                or change.startswith("second_depth") and checks <= 2):
            stamp = 1100 + checks
            obstacle = change.endswith("obstacle") and (change.startswith("new_depth") or checks == 2)
            advance_snapshot(snapshot, clock, stamp)
            if obstacle:
                snapshot[0][1]["map"][220:370, 302:338] = .10
            snapshot[0][2].update(received_speed_ms=stamp, component_speed_ms=stamp)
        if change == "lost_speed":
            objects, depth, odom, receipt = snapshot[0]
            snapshot[0] = objects, depth, dict(odom, received_speed_ms=None, component_speed_ms=None), receipt
        if change == "obstacle_new_odom":
            snapshot[0][2]["source_sample_id"] += 1
        if change == "cancel":
            card._cancel_scheduled_motion(block=True)
        if change == "late":
            clock[0] = 1500
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", during_repeat)
    if change == "obstacle_failed_stop":
        attempts = []
        publish = card._publisher.publish

        def fail_once(packet):
            attempts.append(packet)
            if len(attempts) == 1:
                raise RuntimeError("transport unavailable")
            publish(packet)

        monkeypatch.setattr(card._publisher, "publish", fail_once)
        card._dispatch_command()
        assert len(attempts) == len(packets) == 1 and card._wire_moving
        card._dispatch_command()
        assert len(attempts) == len(packets) == 2 and not any(packets[-1]["values"])
        return
    card._dispatch_command()
    if change in ("same_speed", "partial_speed", "lost_speed", "late"):
        assert len(packets) == 1 and card._wire_moving
        clock[0] = 1500
        card._dispatch_command()
        assert len(packets) == 1 and card._scheduled_command is None
    elif change in ("clear", "new_depth_clear", "second_depth_clear"):
        assert len(packets) == 2 and packets[-1]["values"] == message["values"]
        if change.startswith("second_depth"):
            assert checks == 3 and packets[-1]["stamp_ms"] == 1102
        card._dispatch_command()
        assert len(packets) == 2
        clock[0] += 500
        card._dispatch_command()
        assert len(packets) == 2 and card._scheduled_command is None
    else:
        assert len(packets) == 2 and not any(packets[-1]["values"])
        if change.startswith("second_depth"):
            assert checks == 3
        card._dispatch_command()
        assert len(packets) == 2


@pytest.mark.parametrize("body", [False, True], ids=["camera", "body"])
@pytest.mark.parametrize("yaw", [-.06, .06])
def test_route_controller_changes_wire_curvature_on_feedback_between_depth_frames(monkeypatch, body, yaw):
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=body, live_target=True)
    card._command_timer = object()
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    objects, depth, odom, received = snapshot[0]
    pose = PlanarPose(1000, ("odom",), 0., 0., 0.)
    odom.update(motion_pose=pose, obstacle_pose=pose)
    depth["obstacle_capture_pose"] = pose
    message = card.next_command()
    assert card._stage_command(message, card._output_epoch)
    assert packets[-1]["values"][0] > 0 and not packets[-1]["values"][5]
    clock[0] = 1100
    current = PlanarPose(1100, pose.frame, .01, 0., yaw)
    snapshot[0] = objects, depth, dict(odom, motion_pose=current, obstacle_pose=current,
        source_sample_id=1100, received_ms=1100,
        velocity_history=(*odom["velocity_history"], (1100, 1100, (.1, 0., 0.)))), received
    card._dispatch_command()
    assert len(packets) == 2 and packets[1]["values"][0] > 0
    assert packets[1]["values"][5] * yaw < 0
    assert card._wire_dwa_origin.acceptance.command == tuple(message["values"][i] for i in (0, 1, 5))
    for stamp, remaining_yaw in ((1200, yaw / 2), (1300, 0.)):
        clock[0] = stamp
        current = PlanarPose(stamp, pose.frame, (stamp - 1000) / 10000, 0., remaining_yaw)
        previous = snapshot[0][2]
        snapshot[0] = objects, depth, dict(previous, motion_pose=current, obstacle_pose=current,
            source_sample_id=stamp, received_ms=stamp,
            velocity_history=(*previous["velocity_history"], (stamp, stamp, (.1, 0., 0.)))), received
        card._dispatch_command()
    assert [packet["stamp_ms"] for packet in packets] == [1000, 1100, 1200, 1300]
    assert all(packet["values"][0] > 0. for packet in packets)
    assert abs(packets[1]["values"][5]) > abs(packets[2]["values"][5]) > abs(packets[3]["values"][5])

def test_route_timer_checks_only_the_packet_window_through_complete_stop(monkeypatch):
    card, packets, _, _, feed = tracked_route_card(monkeypatch)
    cutoff = card._scheduled_command.expires_ms
    card._config.max_obs_age_ms = 800
    feed(cutoff - 20, .10)
    predict = motion.predict_motion
    evaluate = motion.evaluate_motion
    holds, budgets = [], []

    def track_holds(*args, **kwargs):
        holds.append((kwargs["hold_s"], kwargs["latency_s"]))
        return predict(*args, **kwargs)

    def track_checks(trace, **kwargs):
        assert trace.stopped
        budgets.append(kwargs.get("terminal_boundary"))
        return evaluate(trace, **kwargs)

    monkeypatch.setattr(motion, "predict_motion", track_holds)
    monkeypatch.setattr(motion, "evaluate_motion", track_checks)
    card._dispatch_command()
    assert len(packets) == 2 and packets[1]["values"][0] > 0
    assert holds == [(pytest.approx(.57), pytest.approx(.02))]
    assert budgets and all(isinstance(room, motion.TravelBudget) for room in budgets)
    assert packets[1]["stamp_ms"] + packets[1]["ttl_ms"] <= cutoff


@pytest.mark.parametrize("obstacle,clear", [(.40, True), (.10, False)])
@pytest.mark.parametrize("fresh_feedback", [True, False], ids=["retimed-packet", "outstanding-packet"])
def test_route_timer_brakes_for_new_depth_inside_its_stopping_sweep(monkeypatch, obstacle, clear, fresh_feedback):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    cutoff = card._scheduled_command.expires_ms
    frame = snapshot[0][1]["map"].copy()
    frame[220:370, 302:338] = obstacle
    feed(cutoff - 20 if fresh_feedback else 1100, .10, frame=frame, sample=None if fresh_feedback else 1000)
    evaluate = motion.evaluate_motion
    traces = []

    def count_evaluations(trace, **kwargs):
        traces.append(trace)
        return evaluate(trace, **kwargs)

    monkeypatch.setattr(motion, "evaluate_motion", count_evaluations)
    card._dispatch_command()
    assert len(traces) == 1 and traces[0].stopped
    if clear:
        assert len(packets) == (2 if fresh_feedback else 1)
        assert packets[-1]["values"][0] > 0
        assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= cutoff
    else:
        assert len(packets) == 2
        assert not any(packets[-1]["values"])
        assert card._dwa_publication["reason"] == "footprint_collision"


def test_visual_probe_rechecks_new_depth_when_detection_has_only_a_bearing(monkeypatch):
    card, packets, clock, snapshot = observation_card(monkeypatch)
    _, depth, odom, _ = snapshot[0]
    depth["map"][48:145, 287:354] = 0.
    state, config = card._state, card._config
    state.goal_anchor = None
    config.depth_intrinsics = config.local_geometry.horizontal
    person = {"name": "person", "position": [0., -.6], "confidence": .95}
    for stamp in (700, 800, 900):
        objects = {"timestamp": stamp / 1000, "objects": [dict(person, bbox_norm=[.45, .10, .55, .30])]}
        state.tracker.step(detections=objects, depth=dict(depth, frame_id=stamp, source_frame_id=stamp),
                           target=state.target, config=config, ego=(0., 0., 0.), ego_measured=True, dt=.1)
    snapshot[0] = {"timestamp": 1., "objects": [person]}, depth, odom, 1000
    message = card.next_command()
    assert card._last_decision.dwa_acceptance.motion_guard.probe_region == ((0., -.6), ())
    advance_snapshot(snapshot, clock, 1010)
    card._publish_command(message)
    assert packets[-1]["values"] == message["values"] and message["values"][0] > 0.


@pytest.mark.parametrize("latest", [
    "target_update", "target_lost", "blocked", "depth_update", "unswept_obstacle", "unswept_obstacle_near_goal",
])
def test_failed_route_handoff_keeps_only_a_still_valid_route(monkeypatch, latest):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    assert card._stage_command(card.next_command(), card._output_epoch)
    previous = card._scheduled_command
    prior_window = previous.expires_ms, previous.expires_at, previous.repeat_until
    advance_snapshot(snapshot, clock, 1100)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = card.next_command()
    assert card._last_decision.dwa_acceptance.route_points
    assert card._last_decision.dwa_acceptance.motion_guard.association is not None
    candidate = card._last_decision.dwa_acceptance
    recheck = card._recheck_dwa_motion
    checks = []

    def during_handoff(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        stamp = 1100 + len(checks)
        clock[0] = stamp
        if latest == "blocked" and len(checks) == 1:
            advance_snapshot(snapshot, clock, stamp, obstacle=True)
        elif latest == "depth_update" or (latest.startswith("unswept_obstacle") and len(checks) == 2):
            # Each candidate sweep is overtaken by another camera frame.
            advance_snapshot(snapshot, clock, stamp)
            if latest.startswith("unswept_obstacle"):
                snapshot[0][1]["map"][220:370, 302:338] = .10
        objects, depth, odom, _ = snapshot[0]
        people = [] if latest == "target_lost" and len(checks) == 2 else objects["objects"]
        snapshot[0] = dict(objects, timestamp=stamp / 1000, objects=people), depth, odom, stamp
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", during_handoff)
    if latest == "unswept_obstacle_near_goal":
        monkeypatch.setattr(card, "_goal_beyond_local_grid", lambda origin: False)
    assert card._stage_command(message, card._output_epoch)
    assert len(checks) == (3 if latest == "blocked" else 2)
    if latest in ("target_update", "depth_update") or latest.startswith("unswept_obstacle"):
        # The candidate never earned the wire. The sent route keeps its own
        # window, and its controller makes the next physical check.
        assert all(result and result[2] for result in checks)
        assert len(packets) == 1 and card._scheduled_command is previous
        assert (previous.expires_ms, previous.expires_at, previous.repeat_until) == prior_window
        assert card._dwa_publication["reason"] == "stale_observation"
        monkeypatch.setattr(card, "_recheck_dwa_motion", recheck)
        if latest.startswith("unswept_obstacle"):
            clock[0] = 1150
            card._dispatch_command()
            assert len(packets) == 2 and not any(packets[-1]["values"])
            assert card._dwa_publication["reason"] == "footprint_collision"
            return
        advance_snapshot(snapshot, clock, 1200)
        card._dispatch_command()
        assert len(packets) == 2 and any(packets[-1]["values"])
        assert card._wire_dwa_origin.acceptance is candidate
    else:
        assert len(packets) == 2 and not any(packets[-1]["values"])
        if latest == "blocked":
            assert card._dwa_publication["reason"] == "footprint_collision"



@pytest.mark.parametrize("obstacle_in_outgoing_sweep", [False, True])
@pytest.mark.parametrize("rejection_stage", ["publish", "no_route", "planning_finish"])
def test_blocked_candidate_brakes_only_if_outgoing_motion_is_also_blocked(
        monkeypatch, obstacle_in_outgoing_sweep, rejection_stage):
    card, packets, clock, snapshot, feed = tracked_route_card(monkeypatch)
    previous = card._scheduled_command
    window = previous.expires_ms, previous.expires_at, previous.repeat_until
    feed(1100, .10, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = card.next_command()
    assert any(message["values"])
    image = depth["map"].copy()
    if obstacle_in_outgoing_sweep:
        image[220:370, 302:338] = .10
    feed(clock[0] + 1, .10, frame=image)
    if rejection_stage != "publish":
        reason = policy.NO_LOCAL_ROUTE if rejection_stage == "no_route" else "候选扫掠被障碍否决"
        card._last_decision = policy.Decision([0.] * 6, policy.AVOIDING, reason)
        message = dict(message, values=card._last_decision.values)
        if rejection_stage == "planning_finish":
            card._staged_dwa_command = None
            card._remember_decision(message, card._output_epoch, card._acp_action_id, "remembered_obstacle")
    else:
        monkeypatch.setattr(card, "_recheck_dwa_inputs", lambda *_: ("footprint_collision", None))

    assert card._stage_command(message, card._output_epoch)
    if obstacle_in_outgoing_sweep:
        assert len(packets) == 2 and not any(packets[-1]["values"])
    else:
        assert len(packets) == 1 and card._scheduled_command is previous
        assert (previous.expires_ms, previous.expires_at, previous.repeat_until) == window

def tracked_route_card(monkeypatch):
    """A sent route on measured poses, and a feeder for later odometry reports."""
    from plugins.navi.odom import PlanarPose

    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    _, depth, odom, _ = snapshot[0]
    pose = PlanarPose(1000, ("odom",), 0., 0., 0.)
    odom.update(motion_pose=pose, obstacle_pose=pose)
    depth["obstacle_capture_pose"] = pose
    assert card._stage_command(card.next_command(), card._output_epoch)
    assert len(packets) == 1 and packets[0]["values"][0] > .1

    def feed(stamp, rate, *, sample=None, frame=None, same_pose=False):
        # One report: a new pose, with either a new or a repeated speed sample.
        objects, image, previous, received = snapshot[0]
        current = PlanarPose(stamp, pose.frame, (stamp - 1000) / 10000, 0., 0.)
        if same_pose:
            current = replace(previous["motion_pose"], stamp_ms=stamp)
        if frame is not None:
            image = dict(image, map=frame, frame_id=stamp, capture_ms=stamp, control_ms=stamp,
                         source_frame_id=stamp, obstacle_capture_pose=current)
        sample = stamp if sample is None else sample
        snapshot[0] = objects, image, dict(
            previous, vx=rate, motion_pose=current, obstacle_pose=current, source_sample_id=sample,
            received_ms=stamp, velocity_history=(*previous["velocity_history"], (stamp, sample, (rate, 0., 0.)))), received
        clock[0] = stamp

    return card, packets, clock, snapshot, feed


def test_route_controller_resweeps_late_feedback_before_expiration(monkeypatch):
    card, packets, clock, _, feed = tracked_route_card(monkeypatch)
    cutoff = card._scheduled_command.expires_ms
    feed(cutoff - 65, .1, frame=card._observation_snapshot()[1]["map"])
    recheck = card._recheck_dwa_motion
    checks = []

    def new_speed_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        if len(checks) == 1:
            feed(cutoff - 35, .05)
        else:
            clock[0] = cutoff - 5
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", new_speed_during_sweep)
    card._dispatch_command()
    assert len(checks) == 2 and all(check and check[2] for check in checks)
    assert len(packets) == 2 and packets[-1]["values"][0] > 0
    assert packets[-1]["stamp_ms"] == cutoff - 5
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= cutoff
    assert card._wire_dwa_origin.forward_check.motion_guard.measured == (.05, 0., 0.)


def test_retained_route_retries_early_when_feedback_overtakes_its_sweep(monkeypatch):
    card, packets, clock, snapshot, feed = tracked_route_card(monkeypatch)
    previous = card._scheduled_command
    cutoff = previous.expires_ms
    feed(1100, .1, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = dict(card.next_command(), values=[0.] * 6)
    card._staged_dwa_command = None
    card._remember_decision(message, card._output_epoch, card._acp_action_id, "remembered_obstacle")
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and card._scheduled_command is previous and previous.expires_ms == cutoff
    feed(cutoff - 330, .1)
    recheck = card._recheck_dwa_motion
    checks = []

    def feedback_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        if len(checks) == 1:
            feed(cutoff - 300, .05, same_pose=True)
        else:
            clock[0] = cutoff - 270
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", feedback_during_sweep)
    card._dispatch_command()
    assert len(checks) == 2 and all(check and check[2] for check in checks)
    assert len(packets) == 2 and packets[-1]["values"][0] > 0
    assert packets[-1]["stamp_ms"] == cutoff - 270
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= cutoff


def test_route_controller_covers_slower_forward_feedback_in_its_first_sweep(monkeypatch):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    feed(1100, .010, frame=snapshot[0][1]["map"])
    recheck = card._recheck_dwa_motion
    checks = []

    def slower_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        feed(1110, .002, same_pose=True)
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", slower_during_sweep)
    card._dispatch_command()
    assert len(checks) == 1 and checks[0][2]
    proof = checks[0][3]
    assert proof.forward_feedback_room_m is not None, (proof.motion_guard.measured, proof.motion_guard.previous,
        proof.motion_guard.visibility_required, proof.motion_guard.trace.duration)
    assert plugin._slower_forward_within_sweep(proof, plugin._motion_inputs(*snapshot[0][:3]), 350)
    assert len(packets) == 2 and packets[-1]["values"][0] > 0


@pytest.mark.parametrize("obstacle", [False, True])
def test_route_controller_checks_the_outgoing_packet_when_front_range_is_missing(monkeypatch, obstacle):
    from plugins.navi.track import TargetMatch

    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    work = card._scheduled_command
    cutoff = work.expires_ms
    image = snapshot[0][1]["map"].copy()
    if obstacle:
        image[220:370, 302:338] = .10
    feed(1100, .1, frame=image)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    track = card._state.tracker.track
    monkeypatch.setattr(card, "_dwa_target_match", lambda *args, **kwargs: TargetMatch(
        track.generation, 1.1, None, None))
    card._dispatch_command()
    if obstacle:
        assert len(packets) == 2 and not any(packets[-1]["values"])
        assert card._dwa_publication["reason"] == "footprint_collision"
    else:
        assert len(packets) == 1 and card._scheduled_command is work
        assert work.expires_ms == cutoff


def test_route_controller_follows_an_associated_image_bearing_without_a_range(monkeypatch):
    from plugins.navi.track import TargetMatch

    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    cutoff = card._scheduled_command.expires_ms
    feed(1100, .1, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    track = card._state.tracker.track
    region = ((.25, -.6), (.55, .1, .7, .3))
    monkeypatch.setattr(card, "_dwa_target_match", lambda *args, **kwargs: TargetMatch(
        track.generation, 1.1, None, None, region))
    card._dispatch_command()
    assert len(packets) == 2
    packet = packets[-1]
    assert packet["values"][0] > 0 and packet["values"][5] < 0
    proof = card._wire_dwa_origin.forward_check
    assert proof.bearing_only and proof.target_ms == 1100 and proof.target_capture_ms == 1000
    assert packet["stamp_ms"] + packet["ttl_ms"] <= cutoff


def test_route_controller_can_sweep_new_target_with_recent_component_speed(monkeypatch):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    previous = card._scheduled_command
    cutoff = previous.expires_ms = 1395
    previous.expires_at = cutoff / 1000
    stamp = cutoff - 195
    feed(stamp, 0., sample=1000)
    objects, depth, odom, _ = snapshot[0]
    odom = dict(odom, received_speed_ms=1000, component_speed_ms=1000)
    snapshot[0] = dict(objects, timestamp=stamp / 1000), depth, odom, stamp
    card._dispatch_command()
    assert len(packets) == 2 and packets[-1]["values"][0] > 0, card._dwa_publication["reason"]
    assert card._wire_dwa_origin.forward_check.target_ms == stamp
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= cutoff


def test_late_replacement_lets_the_controller_finish_its_in_flight_sweep(monkeypatch):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    previous = card._scheduled_command
    cutoff = previous.expires_ms = 1170
    previous.expires_at = cutoff / 1000
    stamp = cutoff - 70
    feed(stamp, 0., sample=1000, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    odom = dict(odom, received_speed_ms=1000, component_speed_ms=1000)
    snapshot[0] = dict(objects, timestamp=stamp / 1000), depth, odom, stamp
    message = card.next_command()
    assert card._last_decision.dwa_acceptance.route_points
    check = card._check_route_control
    swept = []

    def stage_during_check(*args, **kwargs):
        result = check(*args, **kwargs)
        swept.append(result)
        if len(swept) == 1:
            assert card._stage_command(message, card._output_epoch)
            assert len(packets) == 1 and card._scheduled_command is previous
            assert card._pending_route is not None
        return result

    monkeypatch.setattr(card, "_check_route_control", stage_during_check)
    card._dispatch_command()
    assert len(swept) == 2 and all(checked and checked[2] for checked, _ in swept)
    assert all(any(packet["values"]) for packet in packets)
    assert card._scheduled_command is not previous and card._pending_route is None


@pytest.mark.parametrize("blocked", [False, True])
def test_route_controller_resweeps_new_memory_before_expiration(monkeypatch, blocked):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    cutoff = card._scheduled_command.expires_ms
    feed(cutoff - 70, .1, frame=snapshot[0][1]["map"])
    recheck = card._recheck_dwa_motion
    checks = []

    def change_memory_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        if len(checks) == 1:
            squares = local_path.ObstacleSquares(np.array([[.22, 0.] if blocked else [.22, 2.]]), 0.)
            card._state.active_scan.near_projection = NearProjection(squares=squares, stamp_ms=cutoff - 40)
            feed(cutoff - 40, .1)
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", change_memory_during_sweep)
    card._dispatch_command()
    assert len(checks) == 2 and checks[0][2] and checks[1][2] == (not blocked)
    assert len(packets) == 2 and any(packets[-1]["values"]) == (not blocked)
    if blocked:
        assert card._dwa_publication["reason"] == "remembered_obstacle"


@pytest.mark.parametrize("arrival", ["new_detection", "already_confirmed", "no_new_feedback"])
def test_route_controller_uses_recent_confirmed_association_after_the_plan_ages(monkeypatch, arrival):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    card._config.max_coast_s = .15
    already_confirmed = arrival == "already_confirmed"
    no_new_feedback = arrival == "no_new_feedback"
    previous = card._scheduled_command
    cutoff = previous.expires_ms
    for stamp in (1100, 1200, 1250) if already_confirmed else (1100, 1200):
        feed(stamp, .1, frame=snapshot[0][1]["map"], sample=1000 if no_new_feedback else None)
        objects, depth, odom, receipt = snapshot[0]
        if stamp != 1250:
            receipt = stamp - 10
            objects = dict(objects, timestamp=receipt / 1000)
        snapshot[0] = objects, depth, odom, receipt
        if stamp == 1100 or (stamp == 1200 and already_confirmed):
            # The planner has consumed this detection while the old route
            # still owns the wire and the replacement is still pending.
            card.next_command()
    card._dispatch_command()
    if no_new_feedback:
        assert len(packets) == 1 and card._scheduled_command is previous and previous.expires_ms == cutoff
        return
    assert len(packets) == 2 and packets[-1]["values"][0] > 0
    proof = card._wire_dwa_origin.forward_check
    assert proof.planning_guard.association.source == 1.
    assert proof.target_source == 1.19 and proof.target_receipt_ms == 1190


def test_route_controller_keeps_an_updated_target_range_on_later_depth(monkeypatch):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    feed(1100, .1, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    card._dispatch_command()
    proof = card._wire_dwa_origin.forward_check
    assert proof.target_capture_ms == proof.target_receipt_ms == 1100
    image = depth["map"].copy()
    image[:185] = 12.
    feed(1200, .1, frame=image)
    card._dispatch_command()
    assert len(packets) == 3 and packets[-1]["values"][0] > 0
    current = card._wire_dwa_origin.forward_check
    assert current.target_capture_ms == current.target_receipt_ms == 1100
    assert packets[-1]["obs_stamp_ms"] == 1100


@pytest.mark.parametrize("new_obstacle", [False, True])
def test_timer_hands_lost_target_to_search_inside_the_active_window(monkeypatch, new_obstacle):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    work = card._scheduled_command
    cutoff = min(work.expires_ms, work.repeat_until[0])
    frame = None
    if new_obstacle:
        frame = snapshot[0][1]["map"].copy()
        frame[220:370, 302:338] = .10
    feed(1100, .10, frame=frame)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1, objects=[]), depth, odom, 1100
    card._dispatch_command()
    if new_obstacle:
        assert len(packets) == 2 and not any(packets[-1]["values"])
    else:
        assert len(packets) == 1 and work.expires_ms == cutoff
        assert card._state.reacquire_generation is not None
        card._tick()
        assert len(packets) == 2 and packets[-1]["values"][5] != 0.


@pytest.mark.parametrize("change", ["clear", "obstacle", "cancelled", "expired"])
@pytest.mark.parametrize("near_goal", [False, True], ids=["distant-goal", "local-goal"])
def test_timer_checks_replacement_route_before_taking_the_wire(monkeypatch, change, near_goal):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    if near_goal:
        monkeypatch.setattr(card, "_goal_beyond_local_grid", lambda origin: False)
    previous = card._scheduled_command
    feed(1100, .10, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    recheck = card._recheck_dwa_motion
    frames = iter((1110, 1120))

    def frame_during_publish(*args, **kwargs):
        result = recheck(*args, **kwargs)
        feed(next(frames), .10, frame=depth["map"])
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", frame_during_publish)
    card._tick()
    candidate = card._last_decision.dwa_acceptance
    assert candidate is not None and candidate.route_points
    assert len(packets) == 1 and card._scheduled_command is previous
    monkeypatch.setattr(card, "_recheck_dwa_motion", recheck)

    image = depth["map"].copy()
    if change == "obstacle":
        image[220:370, 302:338] = .10
    feed(1500 if change == "expired" else 1150, .10, frame=image)
    if change == "cancelled":
        card._cancel_scheduled_motion(block=True)
    card._dispatch_command()
    if change == "expired":
        assert len(packets) == 1 and card._scheduled_command is None
    elif change == "clear":
        assert len(packets) == 2
        assert any(packets[-1]["values"])
        origin = card._wire_dwa_origin
        assert origin.acceptance is candidate
        assert origin.tracking_command is not None
        assert origin.forward_check.source_id == 1150
    else:
        assert len(packets) == 2 and not any(packets[-1]["values"])


@pytest.mark.parametrize("expire", [None, "during_check", "during_write"])
def test_route_handoff_waits_for_the_timer_if_only_moving_odometry_changed(monkeypatch, expire):
    card, packets, clock, snapshot, feed = tracked_route_card(monkeypatch)
    previous = card._scheduled_command
    previous.expires_ms, previous.expires_at = 1130, 1.130
    feed(1100, .10, frame=snapshot[0][1]["map"])
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=1.1), depth, odom, 1100
    message = card.next_command()
    candidate = card._last_decision.dwa_acceptance
    recheck = card._recheck_dwa_motion

    def new_pose_during_first_check(*args, **kwargs):
        result = recheck(*args, **kwargs)
        if snapshot[0][2]["received_ms"] == 1100:
            feed(1110, .10)
        elif expire == "during_check":
            clock[0] = previous.expires_ms
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", new_pose_during_first_check)
    assert card._stage_command(message, card._output_epoch)
    assert len(packets) == 1 and card._scheduled_command is previous
    if expire == "during_write":
        send = card._publish_command

        def expire_before_write(message, **kwargs):
            if kwargs.get("repeat_feedback") is not None:
                clock[0] = previous.expires_ms
            return send(message, **kwargs)

        monkeypatch.setattr(card, "_publish_command", expire_before_write)
    card._dispatch_command()
    if expire is None:
        assert len(packets) == 2
        packet = packets[-1]
        assert packet["values"][0] > 0.
        assert card._wire_dwa_origin.acceptance is candidate
        assert packet["stamp_ms"] < previous.expires_ms < packet["stamp_ms"] + packet["ttl_ms"]
    else:
        assert len(packets) == 1 and card._scheduled_command is None


@pytest.mark.parametrize("change,covered", [
    ("repeated_speed", True), ("inside_swept_speeds", True), ("at_wire_speed", True),
    ("overspeed", False), ("slower_than_swept", False), ("yaw_outside", False), ("lateral", False),
    ("new_depth", False), ("new_target", False), ("pose_jump", False), ("gap", False), ("frame", False),
    ("unchanged", False),
])
def test_newer_feedback_keeps_a_sweep_only_inside_its_swept_speeds(change, covered):
    measured, wire, command = (.10, 0., 0.), (.15, 0., 0.), (.15, 0., .02)
    pose = plugin.odom_mod.PlanarPose(995, ("odom",), .5, 0., 0.)
    history = ((1000, 995, measured),)
    before = dict(vx=measured[0], vy=0., wz=0., stamp_ms=995, source_sample_id=995,
                  motion_pose=pose, obstacle_pose=pose, velocity_history=history)
    rate = {"inside_swept_speeds": (.12, 0., .01), "at_wire_speed": wire, "overspeed": (.151, 0., 0.),
            "slower_than_swept": (.099, 0., 0.), "yaw_outside": (.12, 0., -.001),
            "lateral": (.12, .001, 0.)}.get(change, measured)
    stamp = 995 if rate == measured else 1030
    latest = replace(pose, x=pose.x + .003, stamp_ms=1030)
    if change == "pose_jump":
        latest = replace(latest, x=latest.x + 1.)
    elif change == "frame":
        latest = replace(latest, frame=("other",))
    after = dict(before, vx=rate[0], vy=rate[1], wz=rate[2], stamp_ms=stamp, source_sample_id=stamp,
                 motion_pose=latest, obstacle_pose=latest,
                 velocity_history=history + ((1400 if change == "gap" else 1035, stamp, rate),))
    if change == "unchanged":
        after = before
    depth = {"map": np.ones((1, 1))} if change == "new_depth" else None
    target = {"objects": []} if change == "new_target" else None
    proof = SimpleNamespace(input_revision=plugin._motion_inputs(None, None, before),
                            motion_guard=SimpleNamespace(measured=measured, previous=wire))
    values = [command[0], command[1], 0., 0., 0., command[2]]
    newer = plugin._motion_inputs(target, depth, after)
    assert plugin._feedback_within_sweep(proof, newer, values, 350) == covered


@pytest.mark.parametrize("arrival,near_goal,checks_run,written", [
    ("repeated_speed", False, 1, True), ("inside_swept_speeds", False, 1, True),
    ("new_depth", False, 2, True), ("new_obstacle", False, 2, False),
    ("overspeed", False, 1, False), ("rising_speed", False, 1, False),
    ("inside_swept_speeds", True, 1, False), ("same_pose", True, 1, True),
])
def test_route_controller_writes_a_checked_sweep_while_feedback_keeps_arriving(
        monkeypatch, arrival, near_goal, checks_run, written):
    card, packets, clock, snapshot, feed = tracked_route_card(monkeypatch)
    if near_goal:
        monkeypatch.setattr(card, "_goal_beyond_local_grid", lambda origin: False)
    feed(1100, .1)
    recheck = card._recheck_dwa_motion
    checks = []

    def sample_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        stamp = 1100 + 10 * len(checks)
        frame = None
        if arrival in ("new_depth", "new_obstacle") and len(checks) == 1:
            frame = snapshot[0][1]["map"].copy()
            if arrival == "new_obstacle":
                frame[220:370, 302:338] = .10
        rate = .1
        if arrival == "rising_speed" or (arrival == "overspeed" and len(checks) == 1):
            rate = packets[0]["values"][0] + .05 * len(checks)
        # The odometry stream is at least as fast as one complete sweep.
        feed(stamp, rate, sample=1100 if arrival == "repeated_speed" else None, frame=frame,
             same_pose=arrival == "same_pose")
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", sample_during_sweep)
    card._dispatch_command()
    assert len(checks) == checks_run
    if arrival == "new_obstacle":
        assert len(packets) == 2 and not any(packets[-1]["values"])
        assert card._dwa_publication["reason"] == "footprint_collision"
    elif written:
        assert all(result and result[2] for result in checks)
        assert len(packets) == 2 and packets[-1]["values"][0] > 0
        # The window still belongs to the feedback sample that was swept.
        proof = card._wire_dwa_origin.forward_check
        assert proof is checks[-1][3] and packets[-1]["stamp_ms"] == clock[0]
        assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= proof.motion_guard.deadline_ms
    else:
        assert len(packets) == 1 and card._wire_moving
        if arrival == "overspeed":
            monkeypatch.setattr(card, "_recheck_dwa_motion", recheck)
            card._dispatch_command()
            assert len(packets) == 2 and packets[-1]["values"][0] > 0.
            assert card._wire_dwa_origin.forward_check.motion_guard.measured[0] == pytest.approx(
                packets[0]["values"][0] + .05)


@pytest.mark.parametrize("start,frames,checks_run,written,near_goal", [
    (1100, 3, 3, False, False), (1320, 3, 4, True, False), (1320, 99, 18, False, False), (1320, 3, 4, True, True),
], ids=["next_visit_in_time", "window_closing", "never_settles", "near_goal"])
def test_route_controller_keeps_checking_only_when_the_next_visit_is_too_late(
        monkeypatch, start, frames, checks_run, written, near_goal):
    card, packets, clock, snapshot, feed = tracked_route_card(monkeypatch)
    if near_goal:
        monkeypatch.setattr(card, "_goal_beyond_local_grid", lambda origin: False)
    feed(start, .1)
    recheck = card._recheck_dwa_motion
    checks = []

    def frame_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        if len(checks) <= frames:
            # Every sweep takes 10 ms and is overtaken by another camera frame.
            feed(start + 10 * len(checks), .1, frame=snapshot[0][1]["map"].copy())
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", frame_during_sweep)
    card._dispatch_command()
    assert all(result and result[2] for result in checks)
    assert len(checks) == checks_run
    if written:
        # The fourth sweep used the newest frame, inside the window already sent.
        assert len(packets) == 2 and packets[-1]["values"][0] > 0 and packets[-1]["stamp_ms"] == start + 30
        assert card._wire_dwa_origin.forward_check is checks[-1][3]
    elif frames == 3:
        assert len(packets) == 1 and card._wire_moving
    else:
        # The old packet's watchdog ends at 1500. No unchecked frame extends it.
        assert clock[0] == 1500 and len(packets) == 1 and card._scheduled_command is None
        assert card._dwa_publication["reason"] == "publish_expired"


def test_route_controller_takes_the_clock_of_a_detection_matched_without_renewal(monkeypatch):
    card, packets, clock, snapshot, feed = tracked_route_card(monkeypatch)
    feed(1100, .1)
    recheck = card._recheck_dwa_motion
    checks = []

    def detection_during_sweep(*args, **kwargs):
        result = recheck(*args, **kwargs)
        checks.append(result)
        if len(checks) == 1:
            # No depth or odometry follows it, so the second sweep keeps the old window.
            objects, depth, odom, _ = snapshot[0]
            snapshot[0] = dict(objects, timestamp=1.105), depth, odom, 1105
            clock[0] = 1105
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", detection_during_sweep)
    card._dispatch_command()
    assert len(checks) == 2 and len(packets) == 2 and packets[-1]["values"][0] > 0
    proof = card._wire_dwa_origin.forward_check
    assert (proof.target_source, proof.target_receipt_ms, proof.target_ms) == (1.105, 1105, 1000)
    assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= 1000 + card._config.max_obs_age_ms

    # The next fresh sweep matches that same frame and may now run on its receipt time.
    feed(1200, .1)
    card._dispatch_command()
    assert len(checks) == 3 and len(packets) == 3 and packets[-1]["values"][0] > 0
    proof = card._wire_dwa_origin.forward_check
    assert (proof.target_source, proof.target_receipt_ms, proof.target_ms) == (1.105, 1105, 1105)

    # One detector frame supplies one clock. A later sweep cannot restart it.
    feed(1300, .1)
    card._dispatch_command()
    assert len(packets) == 4 and card._wire_dwa_origin.forward_check.target_ms == 1105
    deadlines = [packet["stamp_ms"] + packet["ttl_ms"] for packet in packets]
    assert max(deadlines) <= 1105 + card._config.max_obs_age_ms



def test_route_controller_keeps_range_from_the_same_exposure(monkeypatch):
    card, packets, _, _, feed = tracked_route_card(monkeypatch)
    proof = card._wire_dwa_origin.forward_check
    assert proof.motion_guard.measured_target is not None
    feed(1100, .1)
    monkeypatch.setattr(card, "_dwa_target_match", lambda *_: pytest.fail("rematched the same exposure"))
    card._dispatch_command()
    assert len(packets) == 2 and packets[-1]["values"][0] > 0.
    current = card._wire_dwa_origin.forward_check
    assert current.target_ms == proof.target_ms
    assert current.target_capture_ms == proof.target_capture_ms

def test_route_controller_slows_for_curvature_at_directional_yaw_limit():
    command, _ = dwa.track_route(((0., 0.), (.3, .4), (1., .4)), .4, 1., (-.1, .3),
                                 body_origin=(0., 0.), lateral=False)
    vx, vy, wz = command
    # 0.45 m along the first segment is (0.27, 0.36) in the body grid.
    assert 0. < vx < .4 and vy == 0. and wz == pytest.approx(-.1)
    assert wz / vx == pytest.approx(-2 * .36 / (.27 ** 2 + .36 ** 2))


@pytest.mark.parametrize("change", ["background", "new_detection", "changed_payload", "close_person", "expired_range", "expired_matching_range", "driver_expired"])
def test_same_detector_exposure_keeps_original_range_when_depth_advances(monkeypatch, change):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._descriptor["rate"]["watchdog_ms"] = card._config.watchdog_ms = 500
    if change.startswith("expired"):
        card._config.max_depth_age_ms = 80
    if change in ("background", "driver_expired"):
        snapshot[0][1]["capture_ms"] = 950
    if change == "driver_expired":
        card._descriptor["rate"]["max_obs_age_ms"] = 130
    message = card.next_command()
    assert card._stage_command(message, card._output_epoch)
    original = card._wire_dwa_origin.forward_check
    advance_snapshot(snapshot, clock, 1100)
    objects, depth, odom, receipt = snapshot[0]
    # The detector centre was on the 5 m surface; a later depth exposure at
    # the same pixels has an unrelated range, outside the association gate.
    if change != "expired_matching_range":
        depth["map"][:185, :] = .30 if change == "close_person" else 12.
    if change == "new_detection":
        objects = dict(objects, timestamp=1.1)
        receipt = 1100
    elif change == "changed_payload":
        objects = dict(objects, objects=[dict(objects["objects"][0], confidence=.92)])
    snapshot[0] = objects, depth, odom, receipt
    card._dispatch_command()
    if change in ("expired_range", "expired_matching_range", "driver_expired"):
        assert len(packets) == 1
        deadline = packets[0]["stamp_ms"] + card._config.watchdog_ms
        clock[0] = deadline
        card._dispatch_command()
        assert len(packets) == 1 and card._scheduled_command is None
    elif change == "background":
        assert len(packets) == 2
        assert packets[-1]["values"][0] > 0.
        result = card._wire_dwa_origin.forward_check
        assert result.target_ms == original.target_ms and result.target_capture_ms == original.target_capture_ms
        assert packets[-1]["obs_stamp_ms"] == original.target_capture_ms == 950
        assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= min(
            original.target_ms + card._config.max_obs_age_ms,
            original.target_capture_ms + card._config.depth_age_limit_ms)
    else:
        assert len(packets) == 2 and not any(packets[-1]["values"])
        assert card._dwa_publication["reason"] in {"target_missing", "target_standoff", "footprint_collision"}


@pytest.mark.parametrize("change", ["clear", "different_goal", "near_person"])
def test_first_detour_freezes_its_measured_goal_before_publish(monkeypatch, change):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=True)
    state, config = card._state, card._config
    state.goal_anchor = None
    config.depth_intrinsics = config.local_geometry.horizontal
    _, depth, odom, _ = snapshot[0]
    person = {"name": "person", "position": [0., -.6], "confidence": .95,
              "bbox_norm": [.45, .10, .55, .30]}
    for stamp in (700, 800, 900):
        objects = {"timestamp": stamp / 1000, "objects": [person]}
        state.tracker.step(detections=objects, depth=dict(depth, frame_id=stamp, source_frame_id=stamp,
                           capture_ms=stamp), target="person", config=config, ego=(0., 0., 0.),
                           ego_measured=True, dt=.1)
    snapshot[0] = objects, depth, odom, 900
    if change == "different_goal":
        remember = policy._remember_detour_goal
        def move_goal(*args):
            remember(*args)
            if state.goal_anchor is not None:
                state.goal_anchor.x += .2
        monkeypatch.setattr(policy, "_remember_detour_goal", move_goal)
    message = card.next_command()
    check = card._staged_dwa_command[1].forward_check
    assert check.planning_guard.association is None and not check.renewable
    assert check.planning_guard.measured_target is not None and state.goal_anchor is not None
    clock[0] = 1010
    if change == "near_person":
        depth = dict(depth, map=depth["map"].copy())
        depth["map"][:150] = config.stop_distance_m
    snapshot[0] = dict(objects, timestamp=1.01), depth, odom, 1010
    card._publish_command(message)
    if change == "clear":
        assert packets[-1]["values"] == message["values"] and message["values"][0] > 0
        assert packets[-1]["stamp_ms"] + packets[-1]["ttl_ms"] <= check.motion_guard.deadline_ms
    else:
        assert packets[-1]["values"] == [0.] * 6


@pytest.mark.parametrize("update", [
    "fresh", "early", "slower", "below_moving_floor",
    "feedback_during_check", "repeat_feedback_during_check", "obstacle_during_check",
    "duplicate", "obstacle", "no_feedback", "different_target",
])
def test_direct_controller_checks_new_target_before_the_old_packet_expires(monkeypatch, update):
    card, packets, clock = direct_tracking_card(monkeypatch, body=True)
    snapshot = [card._observation_snapshot(with_capture_timing=True)]
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    card._command_timer = object()
    message = card.next_command()
    assert card._stage_command(message, card._output_epoch)
    first, = packets
    assert first["values"][0] > 0 and card._wire_dwa_origin is None
    deadline = first["stamp_ms"] + first["ttl_ms"]

    # New range, depth and feedback must all be available before cutoff.
    stamp = deadline - (155 if update == "early" else 10)
    card._config.max_obs_age_ms = stamp - snapshot[0][3] - 1
    advance_snapshot(snapshot, clock, stamp, obstacle=update == "obstacle")
    objects, depth, odom, target_ms = snapshot[0]
    if update != "duplicate":
        objects = dict(objects, timestamp=stamp / 1000)
    target_ms = stamp
    if update in ("slower", "below_moving_floor"):
        card._config.align_tol = .01
        objects["objects"] = [dict(objects["objects"][0], position=[.2, -.6])]
        if update == "below_moving_floor":
            card._config.floor_vx_moving = first["values"][0]
    if update == "different_target":
        objects["objects"] = [dict(objects["objects"][0], name="chair")]
    pose = policy.PlanarPose(stamp, ("body",), 0., 0., 0.)
    first_feedback = (stamp, stamp, (.1, 0., 0.))
    odom = dict(odom, motion_pose=pose, velocity_history=(first_feedback,))
    snapshot[0] = objects, depth, odom if update != "no_feedback" else None, target_ms
    feedback_updates = "feedback_during_check", "repeat_feedback_during_check"
    if update in (*feedback_updates, "obstacle_during_check"):
        recheck = card._recheck_forward

        def late_input(*args, **kwargs):
            result = recheck(*args, **kwargs)
            if kwargs.get("refresh_direct") and kwargs.get("outgoing_deadline_ms") is None:
                assert result and result[2]
                clock[0] = stamp + 1
                if update == "obstacle_during_check":
                    advance_snapshot(snapshot, clock, stamp + 1, obstacle=True)
                else:
                    source = stamp if update == "repeat_feedback_during_check" else stamp + 1
                    feedback = (stamp + 1, source, (.1, 0., 0.))
                    latest = dict(odom, received_ms=stamp + 1, source_sample_id=source,
                                  motion_pose=replace(pose, stamp_ms=stamp + 1),
                                  velocity_history=(first_feedback, feedback))
                    snapshot[0] = objects, depth, latest, target_ms
            return result

        monkeypatch.setattr(card, "_recheck_forward", late_input)
    card._dispatch_command()
    if update in ("fresh", "early", "slower", *feedback_updates):
        assert len(packets) == 2 and packets[-1]["values"][0] > 0
        assert packets[-1]["stamp_ms"] == clock[0]
        assert deadline < clock[0] + packets[-1]["ttl_ms"] <= stamp + card._config.max_obs_age_ms
        if update == "slower":
            assert packets[-1]["values"][0] < first["values"][0]
    else:
        clock[0] = deadline
        card._dispatch_command()
        if update in ("duplicate", "no_feedback", "below_moving_floor"):
            assert len(packets) == 1 and card._scheduled_command is None
        else:
            assert len(packets) == 2 and not any(packets[-1]["values"])


def test_first_send_retry_uses_original_association_to_refresh_the_target_clock(monkeypatch):
    card, _, clock, snapshot, message = selected_card(monkeypatch, live_target=True)
    origin = card._staged_dwa_command[1]
    proof = origin.forward_check
    deadline = min(proof.motion_guard.deadline_ms, proof.target_ms + card._config.max_obs_age_ms)
    start = proof.motion_guard.checked_ms + 50
    advance_snapshot(snapshot, clock, start)
    objects, depth, odom, _ = snapshot[0]
    snapshot[0] = dict(objects, timestamp=start / 1000), depth, odom, start
    recheck = card._recheck_dwa_motion
    results = []

    def new_depth_during_check(*args, **kwargs):
        result = recheck(*args, **kwargs)
        results.append(result)
        if len(results) == 1:
            assert result and result[2] and not kwargs["renew"]
            assert result[3].target_source != proof.target_source and result[3].target_ms == proof.target_ms
            advance_snapshot(snapshot, clock, deadline - 1)
        return result

    monkeypatch.setattr(card, "_recheck_dwa_motion", new_depth_during_check)
    reason, checked = card._recheck_dwa_inputs(origin, message["values"], start)
    assert reason is None and checked[2]
    assert checked[3].target_ms == start
    assert checked[1] > deadline


@pytest.mark.parametrize("hazard,stop_reason", [
    ("standoff", None), ("sweep", None),
    ("inside_sweep", "footprint_collision"), ("inside_standoff", "target_standoff"),
])
def test_failed_direct_renewal_checks_only_the_original_wire_window(monkeypatch, hazard, stop_reason):
    card, packets, clock = direct_tracking_card(monkeypatch, body=True)
    snapshot = [card._observation_snapshot(with_capture_timing=True)]
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    card._command_timer = object()
    card._config.watchdog_ms = 500
    card._config.max_depth_age_ms = 1200
    card._config.stop_distance_m = .6
    card._config.motion_braking = (.3, .45, 1.)
    assert card._stage_command(card.next_command(), card._output_epoch)
    work = card._scheduled_command
    first, = packets
    assert first["values"][0] > 0. and work.dwa_origin is None
    deadline = first["stamp_ms"] + first["ttl_ms"]
    stamp = deadline - 75
    advance_snapshot(snapshot, clock, stamp, vx=.46)
    distance = {"standoff": 1.46, "inside_standoff": .90}.get(hazard, 3.)
    work.forward_check = replace(work.forward_check, direct_association=None, target_ms=stamp - 200,
                                 distance_m=distance)
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    if hazard in ("sweep", "inside_sweep"):
        obstacle = .30 if hazard == "inside_sweep" else 1.10
        occupied[np.argmin(abs(local_path.FORWARD[:, 0] - obstacle)), local_path.START[1]] = True
    monkeypatch.setattr(local_path, "project_observation", lambda *args, **kwargs: (~occupied, occupied))

    card._dispatch_command()

    if stop_reason:
        assert len(packets) == 2 and not any(packets[-1]["values"])
        assert card._dwa_publication["reason"] == stop_reason
    else:
        assert packets == [first]
        assert card._scheduled_command is work and work.expires_ms == deadline


@pytest.mark.parametrize("target_front,expected_speed", [(1.2, .105), (5., .3)])
def test_published_detour_uses_the_target_approach_speed(monkeypatch, target_front, expected_speed):
    card, packets, clock, snapshot = observation_card(monkeypatch, body=False, live_target=True)
    card._config.use_obstacle_history = False
    first = card.next_command()
    assert card._stage_command(first, card._output_epoch)
    advance_snapshot(snapshot, clock, 1100, vx=first["values"][0])
    snapshot[0][1]["map"].fill(5.)
    card._config.k_fwd = .025
    origin = card._wire_dwa_origin
    guard = replace(origin.acceptance.motion_guard, measured_target=((target_front, 0.), 0.))
    origin = replace(origin, acceptance=replace(origin.acceptance, motion_guard=guard))
    card._scheduled_command.dwa_origin = card._wire_dwa_origin = origin

    following = card.next_command()

    assert card._state.dwa_trace.get("retained_route")
    assert following["values"][0] == pytest.approx(expected_speed, abs=.001)


@pytest.mark.parametrize("blocked", [False, True])
def test_route_renewal_standoff_rechecks_the_actual_packet(monkeypatch, blocked):
    card, packets, _, snapshot, feed = tracked_route_card(monkeypatch)
    work = card._scheduled_command
    first, = packets
    window = work.expires_ms, work.expires_at, work.repeat_until
    image = snapshot[0][1]["map"].copy()
    if blocked:
        image[220:370, 302:338] = .10
    feed(1100, .1, frame=image)

    def reject_renewal(*args, **kwargs):
        diagnostics = args[6]
        diagnostics["reason"] = "target_standoff"
        return (1100, 1200, False, work.forward_check), args[0].acceptance.command

    monkeypatch.setattr(card, "_check_route_control", reject_renewal)

    card._dispatch_command()

    if blocked:
        assert len(packets) == 2 and not any(packets[-1]["values"])
    else:
        assert packets == [first]
        assert card._scheduled_command is work
        assert (work.expires_ms, work.expires_at, work.repeat_until) == window


def test_missing_depth_does_not_rewrite_the_sent_direct_command(monkeypatch):
    card, packets, clock = direct_tracking_card(monkeypatch, body=True)
    snapshot = [card._observation_snapshot(with_capture_timing=True)]
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: snapshot[0])
    card._command_timer = object()
    assert card._stage_command(card.next_command(), card._output_epoch)
    work = card._scheduled_command
    first, = packets
    assert work.sent and first["values"][0] > 0.
    original_window = work.expires_ms, work.expires_at
    clock[0] += 100
    objects, _, odom, target_ms = snapshot[0]
    snapshot[0] = objects, None, odom, target_ms

    assert card._stage_command(card.next_command(), card._output_epoch)
    card._dispatch_command()

    assert packets == [first]
    assert card._scheduled_command is work
    assert (work.expires_ms, work.expires_at) == original_window
