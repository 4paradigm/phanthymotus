"""Real local-controller decisions through the Navi command boundary.

Only sensor delivery, the clock and ROS publishing are test edges. The depth
projection, route planner, DWA sampler and policy all execute production code.
"""
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.navi import active_scan, dwa, local_path, plugin, policy
from plugins.navi.active_scan import NearProjection
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.track import Track


def observation_case(*, controller="dwa", body=True, attempted=True):
    camera = local_path.camera_square(HorizontalIntrinsics(640, 420., 320.),
                                     (480, 420., 240.), .20, at_front=not body)
    geometry = (local_path.body_square(camera, {"body_from_optical": camera.base_from_optical})
                if body else camera)
    config = policy.Config(local_controller=controller, local_planning=True,
                           use_lateral=False, local_geometry=geometry,
                           scan_geometry=geometry if body else None,
                           half_width_m=.20, vx_max=.30, max_depth_age_ms=800,
                           watchdog_ms=500, max_feedback_gap_ms=250, use_obstacle_history=True)
    state = policy.State(target="person", motion_source="measured",
                         commanded=(.10, 0., 0.), goal_anchor=policy.DetourGoal(3., 0.))
    state.active_scan.receipt_only = True
    state.active_scan.attempted = attempted
    image = np.full((480, 640), 5.)
    image[220:370, 302:338] = 2.
    depth = dict(map=image, frame_id=1000, capture_ms=1000, control_ms=1000,
                 source_frame_id=1000)
    odom = dict(vx=.10, vy=0., wz=0., received_ms=1000, source_sample_id=1000,
                velocity_history=((1000, 1000, (.10, 0., 0.)),))
    return config, state, depth, odom


def observe_selector(monkeypatch):
    calls = []
    select = dwa.select

    def record(**kwargs):
        choice = select(**kwargs)
        calls.append((kwargs, choice))
        return choice

    monkeypatch.setattr(dwa, "select", record)
    return calls


def test_lateral_camera_detour_checks_an_extra_footprint_margin(monkeypatch):
    config, state, depth, odom = observation_case(body=False, attempted=False)
    config.use_lateral = True
    calls = observe_selector(monkeypatch)
    policy.step(detections=None, depth=depth, odom=odom, config=config, state=state, dt=.1)
    assert calls
    choices = [kwargs["footprint"] for kwargs, _ in calls]
    assert any(footprint.clearance_margin_m == local_path.RESOLUTION for footprint in choices)
    assert config.clearance_margin_m == pytest.approx(.02)


def card_for_case(monkeypatch, config, state, depth, odom):
    packets = []
    card = plugin.NaviPlugin({}, executor=None)
    card._config, card._state = config, state
    card._running = True
    card._publisher = SimpleNamespace(publish=lambda payload: packets.append(json.loads(payload.data)))
    card._command_timer = object() if config.scan_geometry is None else None
    card._descriptor = {"rate": {"max_obs_age_ms": 800}}
    card._ttl_ms = 400
    monkeypatch.setattr(plugin.time, "time", lambda: 1.)
    monkeypatch.setattr(plugin.time, "monotonic", lambda: 1.)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: (None, depth, odom, 1000))
    return card, packets


@pytest.mark.parametrize("body", [True, False], ids=["body", "front-camera"])
def test_checked_prefix_reaches_real_dwa_and_publishes_its_exact_command(monkeypatch, body):
    config, state, depth, odom = observation_case(body=body)
    anchor = state.goal_anchor
    calls = observe_selector(monkeypatch)
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    message = card.next_command()
    assert calls, "Explicit DWA must not bypass selection at the checked-prefix return"
    accepted = [choice for _, choice in calls if choice is not None]
    assert accepted, "Positive fixture must execute a real admissible candidate"
    assert state.dwa_selected
    assert state.goal_anchor is anchor
    # The same world target is 1 cm nearer after .1 s at the measured .1 m/s.
    assert (anchor.x, anchor.y) == (2.99, 0.)
    twist = tuple(message["values"][i] for i in (0, 1, 5))
    assert twist == accepted[-1].command
    assert twist[0] > 0. and twist[1] == 0.
    assert twist[0] <= config.vx_max
    if card._command_timer is None:
        card._publish_command(message)
    else:
        assert card._stage_command(message, card._output_epoch)
        card._dispatch_command()
    assert len(packets) == 1
    assert tuple(packets[0]["values"][i] for i in (0, 1, 5)) == twist
    assert 0 < packets[0]["ttl_ms"] <= 250
    assert card._published == 1


@pytest.mark.parametrize("body", [True, False])
def test_legacy_checked_prefix_does_not_call_dwa(monkeypatch, body):
    config, state, depth, odom = observation_case(controller="legacy", body=body)
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert calls == []
    assert decision.values == [.15, 0., 0., 0., 0., .25]
    assert not state.dwa_selected


def test_existing_body_route_still_uses_real_dwa(monkeypatch):
    config, state, depth, odom = observation_case(attempted=False)
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    accepted = [choice for _, choice in calls if choice is not None]
    assert accepted and state.dwa_selected
    assert tuple(decision.values[i] for i in (0, 1, 5)) == accepted[-1].command


def test_dwa_policy_intersects_operator_and_directional_lateral_limits(monkeypatch):
    config, state, depth, odom = observation_case(controller="legacy")
    policy.step(detections=None, depth=depth, odom=odom, config=config,
                state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    config.local_controller = "dwa"
    config.use_lateral = True
    config.vy_max = .2
    config.left_vy_limit, config.right_vy_limit = .6, .4
    calls = observe_selector(monkeypatch)
    policy._dwa_detour(depth, config, state, state.view_detour, 3., 0.)
    assert calls
    assert calls[0][0]["y_limit"] == (-.2, .2)


def advance_case(*, body):
    width, height = 160, 120
    camera = local_path.camera_square(HorizontalIntrinsics(width, 85., 80.),
                                     (height, 72., 60.), .20, at_front=not body)
    geometry = (local_path.body_square(camera, {"body_from_optical": camera.base_from_optical})
                if body else camera)
    config = policy.Config(local_controller="dwa", local_planning=True, use_lateral=False,
                           local_geometry=geometry, scan_geometry=geometry if body else None,
                           half_width_m=.20, vx_max=.30, max_depth_age_ms=500, watchdog_ms=300,
                           use_obstacle_history=True)
    rows = np.arange(height)[:, None] - 60.
    floor_depth = .35 * 72. / np.maximum(rows, 1.)
    image = np.broadcast_to(np.minimum(floor_depth, 5.), (height, width)).copy()
    ground = np.broadcast_to((rows > 0.) & (floor_depth < 5.), image.shape).copy()
    depth = dict(map=image, ground=ground, frame_id=1000, control_ms=1000, source_frame_id=1000)
    odom = dict(vx=0., vy=0., wz=0., received_ms=1000, source_sample_id=1000)
    state = policy.State(target="chair", motion_source="measured", goal_anchor=policy.DetourGoal(3., 0.))
    scan = state.active_scan
    assert scan.update(depth, odom, geometry, receipt_only=True) is None
    assert scan.begin((1.,), advance_m=1.25, turn_angle=.25, route_segment=True)
    scan.origin, scan.origin_set = (0., 0., -.25), True
    scan.phase, scan.side_frame = "evaluate", scan.current_frame
    return config, state, dict(depth, frame_id=1100, control_ms=1100, source_frame_id=1100), dict(
        odom, received_ms=1100, source_sample_id=1100)


@pytest.mark.parametrize("body", [True, False])
def test_ready_advance_hands_motion_to_dwa_and_retires_the_fixed_leg(monkeypatch, body):
    config, state, depth, odom = advance_case(body=body)
    anchor = state.goal_anchor
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert calls and any(choice is not None for _, choice in calls)
    assert state.dwa_selected and state.checked_prefix
    assert decision.values[0] > 0. and decision.values[1] == 0.
    assert state.goal_anchor is anchor
    assert not state.active_scan.active
    assert state.active_scan.advance_m is None
    assert state.executing_leg is None


@pytest.mark.parametrize("blocked", [False, True])
def test_advance_can_decelerate_yaw_without_standing_first(monkeypatch, blocked):
    config, state, depth, odom = advance_case(body=True)
    odom["wz"] = .12
    state.commanded = (0., 0., .12)
    if blocked:
        # A remembered square overlaps the side of the actual body.
        squares = local_path.ObstacleSquares(np.array([[.05, .20]]), .01)
        state.active_scan.near_projection = NearProjection(squares=squares, stamp_ms=1100)
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, dwa_admission_ms=100)
    assert calls and calls[0][0]["measured"][2] == .12
    assert calls[0][0]["previous"] == (0., 0., .12)
    if blocked:
        assert decision.values == [0.] * 6
        assert state.dwa_trace["rejected"]["remembered_obstacle"] > 0
    else:
        assert decision.dwa_acceptance is not None and decision.values[0] > 0.


def test_ready_advance_still_obeys_freshness_conditions(monkeypatch):
    config, state, depth, odom = advance_case(body=False)
    depth["capture_invalid"] = True
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert decision.values == [0.] * 6
    assert not state.dwa_selected
    assert calls == []


def test_a_changed_nonzero_policy_result_cannot_keep_an_unchecked_dwa_motion(monkeypatch):
    config, state, depth, odom = observation_case()
    calls = observe_selector(monkeypatch)
    account_progress = policy._account_dwa_progress

    def alter_after_selection(decision, *args):
        decision = account_progress(decision, *args)
        if state.dwa_selected:
            decision.values[0] *= .5
        return decision

    monkeypatch.setattr(policy, "_account_dwa_progress", alter_after_selection)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert any(choice is not None for _, choice in calls)
    assert decision.values == [0.] * 6
    assert decision.dwa_acceptance is None
    assert not state.dwa_selected and not state.checked_prefix
    assert state.commanded == (0., 0., 0.)
    assert state.dwa_trace["veto_reason"] == "postcheck_veto"
    assert state.dwa_trace["proposed_twist"] is not None


@pytest.mark.parametrize("body", [True, False])
def test_accepted_identity_retains_full_tail_check_and_true_participation_counts(monkeypatch, body):
    config, state, depth, odom = observation_case(body=body)
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    accepted = decision.dwa_acceptance
    assert accepted is not None
    assert accepted.command == tuple(decision.values[i] for i in (0, 1, 5))
    assert accepted.guard_horizon_s > calls[-1][0]["horizon"]
    trace = state.dwa_trace
    assert accepted.decision_id == trace["decision_id"]
    assert trace["controller_used"] == "dwa"
    assert trace["select_calls"] == len(calls)
    assert 0 < trace["sampled_candidates"] <= config.dwa_samples
    assert 0 < trace["admissible_candidates"] <= trace["sampled_candidates"]
    assert trace["sampled_candidates"] == sum(kwargs["diagnostics"]["sampled_candidates"]
                                              for kwargs, _ in calls)
    assert state.dwa_totals == dict(requested_ticks=1, eligible_ticks=1, attempted_ticks=1,
                                   select_calls=len(calls), proposed_ticks=1, accepted_ticks=1)


@pytest.mark.parametrize("body", [True, False])
def test_route_selection_precedes_the_legacy_straight_waypoint_gate(monkeypatch, body):
    config, state, depth, odom = observation_case(body=body, attempted=False)
    # Put the bend within this command's held trajectory.
    depth["map"][220:370, 302:338] = 1.
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    kwargs, choice = calls[0]
    assert choice is not None
    endpoint = (kwargs["route"], *kwargs["alternate_routes"])[choice.reference_index][-1]
    keepout = kwargs["footprint"].half_width + kwargs["footprint"].clearance_margin_m
    blocked = local_path.route_obstacles(kwargs["grid"], kwargs["geometry"], keepout, allow_unknown=True)
    assert not local_path.segment_clear(
        blocked, kwargs["geometry"], keepout, math.atan2(endpoint[1], endpoint[0]),
        math.hypot(*endpoint), heading_tolerance=policy.ANGLE_TOLERANCE)
    assert choice is not None and decision.dwa_acceptance is not None
    assert choice.command == decision.dwa_acceptance.command
    if body:
        assert choice.command[0] > 0. and choice.command[2] != 0.


def prepared_reference_case(*, body=False):
    config, state, depth, odom = observation_case(controller="legacy", body=body)
    policy.step(detections=None, depth=depth, odom=odom, config=config,
                state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert state.view_detour is not None
    config.local_controller = "dwa"
    return config, state, depth, state.view_detour


@pytest.mark.parametrize("reason", ["no_feedback", "stale_observation", "scan_active",
                                    "geometry_mismatch", "no_reference"])
def test_unavailable_selector_context_has_a_specific_reason_without_a_call(monkeypatch, reason):
    config, state, depth, reference = prepared_reference_case()
    if reason == "no_feedback":
        state.motion_source = "command"
    elif reason == "stale_observation":
        depth["capture_invalid"] = True
    elif reason == "scan_active":
        state.active_scan.phase = "evaluate"
    elif reason == "geometry_mismatch":
        reference = replace(reference, geometry=replace(reference.geometry))
    else:
        reference = None
    calls = observe_selector(monkeypatch)
    assert policy._dwa_detour(depth, config, state, reference, 3., 0.) is None
    assert calls == []
    assert state.dwa_trace["fallback_reason"] == reason


@pytest.mark.parametrize("committed,unfinished", [(False, False), (True, False), (True, True)])
def test_missing_range_may_continue_a_route_that_reaches_the_exit(monkeypatch, committed, unfinished):
    config, state, depth, reference = prepared_reference_case()
    if unfinished:
        end = reference.points[-1][0] + .30
        reference = replace(reference, blocker_exit=((end, -.5), (end, .5)))
    calls = observe_selector(monkeypatch)
    outcome = policy._dwa_detour(depth, config, state, reference, None, None, follow_published=committed)
    if committed and not unfinished:
        assert len(calls) == 1 and calls[0][0]["target"] is None
        assert outcome is not None and outcome.command[0] > 0.
    else:
        assert not calls and outcome is None
        assert state.dwa_trace["fallback_reason"] == "no_reference"


def test_retained_route_checks_the_measured_person_surface_even_when_ranges_agree(monkeypatch):
    config, state, depth, _ = prepared_reference_case()
    measured = .92
    track = state.tracker.track = Track(np.array([measured, 0., 0., 0.]), np.eye(4), range_known=True)
    track.record_range(measured, (measured, 0.), depth, standoff_point=(.91, 0.))
    state.goal_anchor.generation = track.generation
    route = local_path.Detour(None, ((0., 0.), (1., 0.)), config.local_geometry)
    calls = observe_selector(monkeypatch)

    proposal = policy._dwa_detour(depth, config, state, route, measured, 0., follow_published=True)

    options, choice = calls[0]
    assert options["measured_standoff_m"] == config.stop_distance_m
    point, error = options["measured_target"]
    assert proposal is not None and choice.check.accepted
    assert dwa.minimum_target_distance(choice.trace, point) >= config.stop_distance_m + error - 1e-10


@pytest.mark.parametrize("reason", ["footprint_collision", "remembered_obstacle", "occluded_prefix",
                                    "target_standoff"])
def test_candidate_rejections_preserve_the_actual_constraint(monkeypatch, reason):
    config, state, depth, reference = prepared_reference_case()
    extra, distance = {}, 3.
    scan = state.active_scan
    if reason in ("footprint_collision", "remembered_obstacle"):
        hits = np.zeros(local_path.SHAPE, dtype=bool)
        hits[local_path.START] = True
        if reason == "footprint_collision":
            stamp, pose, free, _, *extra_view = scan.views[-1]
            scan.views[-1] = (stamp, pose, free, hits, *extra_view)
        else:
            squares = local_path.ObstacleSquares(np.array([[0., .10]]), .27)
            scan.near_projection = NearProjection(hits, squares, stamp_ms=scan.control_ms)
    elif reason == "occluded_prefix":
        extra["shadow"] = np.ones(local_path.SHAPE, dtype=bool)
    elif reason == "target_standoff":
        measured = config.stop_distance_m / 2
        state.tracker.track = Track(np.array([measured, 0., 0., 0.]), np.eye(4), range_known=True)
        state.tracker.track.record_range(measured, (measured, 0.), depth)
        state.goal_anchor.generation = state.tracker.track.generation
    calls = observe_selector(monkeypatch)
    assert policy._dwa_detour(depth, config, state, reference, distance, 0., **extra) is None
    assert len(calls) == 1
    trace = state.dwa_trace
    assert trace["sampled_candidates"] > 0 and trace["admissible_candidates"] == 0
    assert trace["rejected"][reason] > 0
    assert set(trace["rejected"]) <= {reason, "outside_grid", "off_route", "insufficient_progress"}
    assert sum(trace["rejected"].values()) == trace["sampled_candidates"]


@pytest.mark.parametrize("budget", [7, 128])
@pytest.mark.parametrize("first", [-1., 1.])
def test_same_twist_is_validated_once_across_references(monkeypatch, budget, first):
    config, state, depth, _ = prepared_reference_case(body=True)
    config.dwa_samples = budget
    geometry = config.local_geometry
    plans = {-1.: local_path.Detour(None, ((0., 0.), (.55, -.48)), geometry),
              1.: local_path.Detour(None, ((0., 0.), (.75, .11)), geometry)}
    monkeypatch.setattr(local_path, "opening_route", lambda *args: None)

    requested = []
    def route(_depth, _geometry, _keep, _ahead, side, *_):
        requested.append(side)
        return plans[side]

    monkeypatch.setattr(state.local_plan, "route", route)
    predictions, checks = [], []
    predict, evaluate = dwa.predict_motion, dwa.evaluate_motion

    def record_prediction(measured, previous, command, **kwargs):
        result = predict(measured, previous, command, **kwargs)
        predictions.append((command, result))
        return result

    def record_check(trace, **kwargs):
        result = evaluate(trace, **kwargs)
        checks.append((trace, result))
        return result

    monkeypatch.setattr(dwa, "predict_motion", record_prediction)
    monkeypatch.setattr(dwa, "evaluate_motion", record_check)
    result = policy._start_detour_segment(depth, config, state, geometry, 2., (first, -first), 3., 0.)
    assert requested == [first, -first]
    assert isinstance(result, tuple) and state.dwa_selected
    assert state.view_detour.preview == plans[1.].points
    assert 0 < len(checks) <= len(predictions) <= min(budget, 64)
    assert state.dwa_trace["physical_evaluations"] == len(checks)
    assert len({command for command, _ in predictions}) == len(predictions)
    trace = state.dwa_trace
    assert trace["sampled_candidates"] == len(predictions)
    assert trace["reference_scores"] == trace["attempts"][0]["reference_scores"]
    assert all(count > 0 for count in trace["attempts"][0]["scores_by_reference"])
    guard = state.dwa_motion_guard
    assert result[:3] == state.dwa_proposal.command == trace["proposed_twist"]
    assert any(command == result[:3] and prediction is guard.trace for command, prediction in predictions)
    assert any(prediction is guard.trace and check is guard.check for prediction, check in checks)


@pytest.mark.parametrize("closest", [False, True], ids=["ordinary", "closest"])
@pytest.mark.parametrize("first", [-1., 1.], ids=["left-first", "right-first"])
def test_lateral_handoff_scores_both_eligible_sides_without_scanning(monkeypatch, closest, first):
    config, state, depth, odom = observation_case()
    config.use_lateral = True
    config.vy_max = .30
    state.goal_anchor = None
    state.commanded = (0., 0., 0.)
    odom.update(vx=0., velocity_history=((1000, 1000, (0., 0., 0.)),))
    depth["map"][:] = 5.
    depth["map"][220:370, 292:328] = .85
    scan = state.active_scan
    assert scan.update(depth, odom, config.scan_geometry, receipt_only=True,
                       max_depth_age_ms=config.depth_age_limit_ms) is None
    scan.exhausted = closest
    monkeypatch.setattr(policy, "_local_side_order", lambda *args: (first, -first))

    result = policy._avoid(.20, 0., 0., depth, config, policy.APPROACHING, "", state,
                           target_distance=3., target_bearing=.12, odom=odom, may_scan=False)

    assert state.dwa_selected
    trace = state.dwa_trace
    assert trace["select_calls"] == 1
    assert 0 < trace["sampled_candidates"] <= 64
    counts = trace["attempts"][0]["scores_by_reference"]
    assert len(counts) == 2 and min(counts) > 0
    assert state.escape_side == 1.
    assert state.view_detour.preview == state.local_plan.plans[1.].points
    proposal = state.dwa_proposal
    assert result[:3] == proposal.command == trace["proposed_twist"]
    assert state.dwa_motion_guard is proposal.guard
    assert proposal.guard.trace is proposal.choice.trace
    assert proposal.guard.check is proposal.choice.check and proposal.guard.check.accepted


@pytest.mark.parametrize("budget", [1, 7])
def test_physical_reuse_checks_changed_memory_and_visibility(monkeypatch, budget):
    config, state, depth, reference = prepared_reference_case(body=True)
    config.dwa_samples = budget
    config.vx_max = config.floor_vx = .08
    config.wz_max = 0.
    motion_checks = []
    evaluate = dwa.evaluate_motion

    def record(trace, **kwargs):
        checked = evaluate(trace, **kwargs)
        motion_checks.append(checked)
        return checked

    monkeypatch.setattr(dwa, "evaluate_motion", record)
    def select(route=reference, **kwargs):
        return policy._dwa_detour(depth, config, state, route, 3., 0., **kwargs)

    first = select()
    assert first is not None and len(motion_checks) == 1
    alternate = replace(reference, points=((0., 0.), (.18, .03)))
    second = select(alternate)
    assert second is not None and second.reference is alternate
    assert second.guard.trace is first.guard.trace
    assert second.guard.check is first.guard.check
    assert select(shadow=np.ones(local_path.SHAPE, dtype=bool)) is None
    assert state.dwa_trace["fallback_reason"] == "no_admissible_candidates"
    assert state.dwa_trace["rejected"]["occluded_prefix"] > 0
    assert len(motion_checks) == 1
    scan = state.active_scan
    scan.near_projection = NearProjection(
        squares=local_path.ObstacleSquares(np.array([[.05, .20]]), .01),
        stamp_ms=scan.near_projection.stamp_ms)
    assert select() is None
    if budget == 1:
        assert len(motion_checks) == 1
        assert state.dwa_trace["fallback_reason"] == "sample_budget_exhausted"
    else:
        assert len(motion_checks) == 2 and motion_checks[-1].reason == "remembered_obstacle"


@pytest.mark.parametrize("mode", ["synchronized", "body-receipt", "front-camera"])
def test_dwa_scan_and_raw_occupancy_share_the_fresh_projection(monkeypatch, mode):
    config, state, depth, odom = observation_case(body=mode != "front-camera")
    if mode == "synchronized":
        state.active_scan.receipt_only = False
        odom["stamp_ms"] = 1000
    real_observe = local_path.observe
    requests = []

    def observe(frame, *args, **kwargs):
        if frame is depth["map"] and kwargs.get("max_hit_depth") is None:
            requests.append(kwargs.get("check_free_space", True))
        return real_observe(frame, *args, **kwargs)

    monkeypatch.setattr(local_path, "observe", observe)
    monkeypatch.setattr(active_scan, "observe", observe)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, dwa_admission_ms=0)
    assert decision.dwa_acceptance is not None
    assert state.active_scan.receipt_only == (mode != "synchronized")
    assert requests == [True]
    assert "grid_projection" not in depth


@pytest.mark.parametrize("changed", ["map", "ground"])
def test_duplicate_source_keeps_fresh_raw_occupancy(changed):
    config, state, depth, odom = observation_case()
    if changed == "ground":
        depth["map"][220:370, 302:338] = .20
        depth["ground"] = np.ones_like(depth["map"], dtype=bool)

    def tick():
        return policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, dwa_admission_ms=0)

    assert tick().dwa_acceptance
    retained_view = state.active_scan.views[-1]
    if changed == "map":
        depth["map"][:] = .18
    else:
        depth["ground"][220:370, 302:338] = False
    assert tick().values == [0.] * 6
    assert state.active_scan.views[-1] is retained_view
    assert state.dwa_trace["rejected"]["footprint_collision"] > 0


def test_advance_handoff_does_not_resume_the_old_leg_on_the_next_tick():
    config, state, depth, odom = advance_case(body=False)
    anchor = state.goal_anchor
    first = policy.step(detections=None, depth=depth, odom=odom, config=config,
                        state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert first.dwa_acceptance is not None
    depth = dict(depth, frame_id=1200, control_ms=1200, source_frame_id=1200)
    odom = dict(odom, vx=first.values[0], wz=first.values[5],
                received_ms=1200, source_sample_id=1200)
    second = policy.step(detections=None, depth=depth, odom=odom, config=config,
                         state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert state.goal_anchor is anchor
    assert state.active_scan.advance_m is None
    assert state.executing_leg is None
    assert state.dwa_trace["branch"] != "advance"
    assert second.status != policy.FAILED


@pytest.mark.parametrize("reason", ["no_feedback", "stale_observation"])
def test_public_stop_reports_why_dwa_did_not_run(monkeypatch, reason):
    config, state, depth, odom = observation_case()
    if reason == "no_feedback":
        odom = None
    else:
        depth["capture_invalid"] = True
    calls = observe_selector(monkeypatch)
    policy.step(detections=None, depth=depth, odom=odom, config=config,
                state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert calls == []
    assert state.dwa_trace["fallback_reason"] == reason
    assert state.dwa_totals["attempted_ticks"] == 0


@pytest.mark.parametrize("spacing,hit_forward,expected", [
    (.01, 0., 1), (.01, .30, 28), (.01, .65, 63),
    (.025, 0., 1), (.025, .30, 11), (.025, .65, 26),
    (.08, 0., 1), (.08, .30, 4), (.08, .65, 8),
])
def test_trajectory_visibility_stops_at_the_first_shadow(spacing, hit_forward, expected):
    points = tuple((float(f), 0.) for f in np.arange(0., 1., spacing))
    shadow = np.zeros(local_path.SHAPE, dtype=bool)
    hit_row = local_path.START[0] + round(hit_forward / local_path.RESOLUTION)
    shadow[hit_row, local_path.START[1]] = True
    # Keep the segment index contract, including an occupied starting cell
    # and long segments whose hidden interior is not a trajectory waypoint.
    assert local_path.visible_prefix(points, shadow) == expected


@pytest.mark.parametrize("body", [True, False])
@pytest.mark.parametrize("lateral", [True, False])
def test_real_obstacle_is_evaluated_inside_the_configured_stop_line(monkeypatch, body, lateral):
    config, state, depth, odom = observation_case(body=body, attempted=False)
    config.use_lateral = lateral
    config.obstacle_stop_m = .85
    depth["map"][220:370, 302:338] = .75
    state.commanded = (0., 0., 0.)
    odom.update(vx=0., velocity_history=((1000, 1000, (0., 0., 0.)),))
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert calls, "Use the real route selector at the previously bypassed stop line"
    assert all(kwargs["x_limit"][1] > 0. for kwargs, _ in calls)
    assert decision.dwa_acceptance is not None
    assert next(choice for _, choice in calls if choice is not None).check.accepted


@pytest.mark.parametrize("body", [True, False])
def test_reference_endpoint_is_not_a_physical_wall(monkeypatch, body):
    config, state, depth, odom = advance_case(body=body)
    state.active_scan.advance_m = .10
    config.dwa_horizon_s = 3.
    config.floor_vx_moving = .08
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1, dwa_admission_ms=0)
    assert calls
    assert decision.dwa_acceptance is not None
    choice = next(choice for _, choice in calls if choice is not None)
    assert choice.trace.body_travel_m > .10
    assert choice.check.accepted


def test_constructed_escape_prefix_is_not_reported_as_an_astar_route(monkeypatch):
    config, state, depth, _ = prepared_reference_case(body=True)
    state.active_scan.near_projection = NearProjection(hits=np.zeros(local_path.SHAPE, dtype=bool))
    monkeypatch.setattr(local_path, "opening_route", lambda *args: None)
    monkeypatch.setattr(state.local_plan, "route", lambda *args, **kwargs: None)
    calls = observe_selector(monkeypatch)
    result = policy._start_detour_segment(depth, config, state, config.local_geometry,
                                         2., (-1., 1.), 3., 0.)
    assert calls and any(choice is not None for _, choice in calls)
    assert isinstance(result, tuple) and state.dwa_selected
    assert state.dwa_trace["branch"] == "prefix"


def test_dwa_registration_uses_measurement_and_only_uncovered_residual():
    from plugins.navi.odom import PlanarPose
    config, state, _, _ = observation_case()
    geometry = config.local_geometry
    start = PlanarPose(1000, ("odom",), 0., 0., 0.)
    measured = PlanarPose(1100, start.frame, .50, 0., math.pi / 2)
    twist = (.20, 0., 0.)
    odom = {"obstacle_pose": measured, "velocity_history": ((1108, 1108, twist),)}
    current = policy.dwa_grid_pose(state.active_scan, geometry, 1110, odom)
    assert current is not None and current.stamp_ms == 1108
    assert (current.x, current.y, current.yaw) == pytest.approx((.5, .0016, math.pi / 2))
    registered = policy.dwa_registration(start, current, 1000, 1110, geometry, (twist,))
    assert registered.points((.50, -.22)) == pytest.approx((.22 - .0016, 0.))
    assert registered.shift_error == pytest.approx(.20 * .002)
    assert registered.turn_error == 0.

@pytest.mark.parametrize("side", [-1., 1.])
def test_front_camera_prefix_steers_into_approaching_bend(monkeypatch, side):
    config, state, depth, _ = prepared_reference_case()
    geometry = config.local_geometry
    points = ((0., 0.), (.30, 0.), (.40, side * .10), (.75, side * .10), (1.4, side * .60))
    route = local_path.Detour(None, points, geometry, reachable_prefix=True)
    monkeypatch.setattr(local_path, "opening_route", lambda *args: None)
    monkeypatch.setattr(state.local_plan, "route", lambda *args, **kwargs: route)
    calls = observe_selector(monkeypatch)
    policy._start_detour_segment(depth, config, state, geometry, 2., (side,), 3., 0.)
    choice = next(choice for _, choice in calls if choice is not None)
    vx, vy, wz = choice.command
    assert state.dwa_trace["branch"] == "reachable_prefix"
    assert choice.check.accepted and vx > 0. and vy == 0. and wz * side < 0.


def test_partial_reference_precedes_scan_exhaustion(monkeypatch):
    config, state, depth, odom = observation_case(attempted=False)
    wall = np.isclose(local_path.FORWARD, 1.15)
    depth["map"][:] = 1.15
    # This observed wall extends across the complete local map.
    monkeypatch.setattr(state.active_scan, "_project_view", lambda *args: (~wall, wall))
    anchor = state.goal_anchor
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, dwa_admission_ms=100)
    assert decision.dwa_acceptance and decision.values[0] > 0.
    assert state.dwa_trace["branch"] == "reachable_prefix"
    endpoint = state.view_detour.preview[-1]
    assert 0. < endpoint[0] < 1.15
    assert not state.active_scan.attempted and state.goal_anchor is anchor


def test_observation_hysteresis_never_reuses_invalid_motion():
    config, state, depth, odom = observation_case(attempted=False)
    def tick(stamp, source):
        observation = dict(depth, control_ms=stamp, frame_id=stamp, capture_ms=stamp, source_frame_id=source)
        feedback = dict(odom, vx=0., received_ms=stamp, source_sample_id=stamp,
                        velocity_history=((stamp, stamp, (0., 0., 0.)),))
        return policy.step(detections=None, depth=observation, odom=feedback, config=config,
                           state=state, dt=.05, dwa_admission_ms=100)

    assert tick(1000, 1000).dwa_acceptance
    # Occupancy under the current footprint invalidates every continuation.
    squares = local_path.ObstacleSquares(np.array([[.06, .18]]), .025)
    state.active_scan.near_projection = NearProjection(squares=squares, stamp_ms=1000)
    for stamp, source in ((1050, 1050), (1100, 1050), (1150, 1050), (1200, 1200)):
        assert tick(stamp, source).values == [0.] * 6
        assert not state.active_scan.active
    assert tick(1250, 1250).values == [0.] * 6
    assert state.active_scan.active
    state.active_scan.near_projection = NearProjection()
    assert tick(1300, 1300).dwa_acceptance
    assert not state.active_scan.active


@pytest.mark.parametrize("contact", [False, True])
def test_parking_yaw_checks_the_actual_body_motion(contact):
    config, state, depth, odom = advance_case(body=True)
    scan = state.active_scan
    scan.finish()
    radius, angle = config.stop_distance_m + .01, .35
    state.goal_anchor = policy.DetourGoal(radius * math.cos(angle), -radius * math.sin(angle))
    config.wz_max = .08
    state.commanded = (0., 0., 0.)
    squares = local_path.ObstacleSquares(np.array([[0., .20 if contact else .31]]), .01)
    scan.near_projection = NearProjection(squares=squares, stamp_ms=1100)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, dwa_admission_ms=100)
    if contact:
        assert decision.values == [0.] * 6
    else:
        assert decision.values[:2] == [0., 0.] and decision.values[5] < 0.
        assert decision.dwa_acceptance.motion_guard.check.accepted


def test_policy_keeps_complete_adopted_width_route_through_slot(monkeypatch):
    config, state, depth, odom = observation_case(attempted=False)
    camera = local_path.camera_square(config.local_geometry.horizontal, config.local_geometry.vertical, .23)
    geometry = local_path.body_square(camera, {"body_from_optical": camera.base_from_optical})
    config = replace(config, half_width_m=.23, scan_geometry=geometry, local_geometry=geometry,
                     stop_distance_m=.50)
    state.goal_anchor = policy.DetourGoal(2.55, -.55)
    state.commanded = (0., 0., 0.)
    odom.update(vx=0., velocity_history=((1000, 1000, (0., 0., 0.)),))
    post = np.isclose(local_path.FORWARD, .65) & (abs(local_path.RIGHT) <= .05)
    slot = (local_path.FORWARD >= 1.35) & (abs(local_path.RIGHT - .55) >= .30)
    occupied = post | slot
    monkeypatch.setattr(state.active_scan, "_project_view", lambda *args: (~occupied, occupied))
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, dwa_admission_ms=100)
    assert decision.dwa_acceptance and decision.values[0] > 0.
    assert state.view_detour.preview[-1] == pytest.approx((2., .55))
    assert state.dwa_trace["branch"] == "route"


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("vx,vy,yaw", [(0., 0., 1.), (.1, -.05, .3)])
def test_detour_goal_is_fixed_in_motion_reference(body, vx, vy, yaw):
    from plugins.navi.odom import PlanarPose

    config, state, depth, odom = observation_case(body=body)
    geometry = config.local_geometry
    if body:
        mount = geometry.base_from_optical.copy()
        mount[0, 3] = .2
        config.local_geometry = config.scan_geometry = replace(geometry, base_from_optical=mount)
    initial = PlanarPose(1000, ("odom",), 0., 0., 0.)
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    arc = math.sin(yaw / 2) / (yaw / 2)
    dx, dy = arc * (c * vx - s * vy), arc * (s * vx + c * vy)
    end = PlanarPose(2000, initial.frame, dx, dy, yaw)
    anchor = state.goal_anchor = policy.DetourGoal(2.8, 0., pose=initial)
    odom = dict(vx=vx, vy=vy, wz=yaw, received_ms=2000)
    if body:
        odom["motion_pose"] = end
    policy.step(detections=None, depth=dict(depth, control_ms=2000, frame_id=2000),
                odom=odom, state=state, config=config, dt=1.)
    c, s = math.cos(yaw), math.sin(yaw)
    world = (dx + c * (anchor.x + .2) - s * anchor.y, dy + s * (anchor.x + .2) + c * anchor.y)
    assert world == pytest.approx((3., 0.), abs=1e-9)


@pytest.mark.parametrize("body", [True, False])
def test_visible_same_track_updates_range_but_only_clear_route_releases_goal(body):
    config, state, depth, odom = observation_case(body=body)
    person = {"name": "person", "position": [0., -.6], "confidence": .9}
    for stamp in (700, 800, 900):
        state.tracker.step(detections={"timestamp": stamp, "objects": [person]},
                           depth=dict(depth, frame_id=stamp, source_frame_id=stamp),
                           ego=(0., 0., 0.), dt=.1, config=config, target="person")
    state.goal_anchor = None
    state.escape_side = 1.
    policy._remember_detour_goal(state, config, state.tracker.track, dict(odom, received_ms=900))
    anchor = state.goal_anchor
    assert anchor is not None and anchor.x == pytest.approx(5.)
    # Two measured arcs combine left-positive translation with a right turn.
    # The camera approximation places the optical origin at the chassis front.
    offset = 0. if body else .2
    wx = wy = heading = 0.
    yaw = -.015
    for i, stamp in enumerate((1000, 1100), 1):
        image = depth["map"].copy()
        image[:150] = 4.9
        payload = {"timestamp": stamp, "objects": [dict(person, position=[.02 * i, -.6])]}
        odom = dict(vx=.1, vy=.04, wz=-.15, received_ms=stamp)
        arc = .1 * math.sin(yaw / 2) / (yaw / 2)
        c, s = math.cos(heading + yaw / 2), math.sin(heading + yaw / 2)
        wx += arc * (.1 * c - .04 * s)
        wy += arc * (.1 * s + .04 * c)
        heading += yaw
        frame = dict(depth, map=image, control_ms=stamp, capture_ms=stamp, frame_id=stamp, source_frame_id=stamp)
        policy.step(detections=payload, depth=frame, odom=odom, config=config, state=state, dt=.1)
        assert state.tracker.track.accepted_detection
        assert anchor.verified_range_m == pytest.approx(4.9)
        assert state.goal_anchor is anchor
        c, s = math.cos(heading), math.sin(heading)
        assert (wx + c * (anchor.x + offset) - s * anchor.y,
                wy + s * (anchor.x + offset) + c * anchor.y) == pytest.approx((5. + offset, 0.))
    # Once the obstruction clears, policy can steer to the moving live fix.
    rows = np.arange(480)[:, None] - 240.
    floor = .35 * 420. / np.maximum(rows, 1.)
    clear = np.broadcast_to(np.minimum(floor, 4.9), image.shape).copy()
    ground = np.broadcast_to((rows > 0.) & (floor < 4.9), image.shape).copy()
    for stamp in (1200, 1300, 1400):
        frame = dict(depth, map=clear, ground=ground, control_ms=stamp, capture_ms=stamp,
                     frame_id=stamp, source_frame_id=stamp)
        result = policy.step(detections=dict(payload, timestamp=stamp), depth=frame,
                             odom=dict(odom, received_ms=stamp), config=config, state=state, dt=.1)
    assert state.goal_anchor is None
    assert result.bearing > .01 and result.distance_m < 4.95


def test_paired_mount_update_preserves_original_world_anchor():
    from plugins.navi.odom import PlanarPose

    config, state, depth, odom = observation_case(body=True)
    anchor = state.goal_anchor
    for stamp, offset, yaw in ((1000, .20, 0.), (1100, .25, 0.), (1200, .25, .12)):
        pose = PlanarPose(stamp, ("odom",), 0., 0., 0.)
        mount = config.local_geometry.base_from_optical.copy()
        c, s = math.cos(yaw), math.sin(yaw)
        mount[:2] = np.array([[c, -s], [s, c]]) @ mount[:2]
        mount[0, 3] = offset
        frame = dict(depth, frame_id=stamp, capture_ms=stamp, source_frame_id=stamp, control_ms=stamp,
                     obstacle_capture_pose=pose, body_from_optical=mount, body_mount_required=True)
        feedback = dict(vx=0., vy=0., wz=0., received_ms=stamp, obstacle_pose=pose, motion_pose=pose, stamp_ms=stamp)
        policy.step(detections=None, depth=frame, odom=feedback, config=config, state=state, dt=.1)
        assert state.use_body_mount and state.goal_anchor is anchor
        world = offset + c * anchor.x - s * anchor.y, s * anchor.x + c * anchor.y
        assert world == pytest.approx((3.20, 0.))


def test_paired_mount_update_matches_in_the_current_camera_frame():
    from plugins.navi.odom import PlanarPose
    from plugins.navi.track import match_observation

    config, state, depth, _ = observation_case(body=True)
    geometry = config.local_geometry
    person = {"name": "person", "position": [0., -.6], "confidence": .95}
    mount = geometry.base_from_optical.copy()
    mount[0, 3] = .20
    for stamp in (800, 900, 1000):
        pose = PlanarPose(stamp, ("odom",), 0., 0., 0.)
        frame = dict(depth, frame_id=stamp, capture_ms=stamp, source_frame_id=stamp, control_ms=stamp,
                     obstacle_capture_pose=pose, body_from_optical=mount, body_mount_required=True)
        feedback = dict(vx=0., vy=0., wz=0., received_ms=stamp, obstacle_pose=pose, motion_pose=pose, stamp_ms=stamp)
        objects = {"timestamp": stamp / 1000, "objects": [person]}
        policy.step(detections=objects, depth=frame, odom=feedback, config=config, state=state, dt=.1)
    track = state.tracker.track
    assert state.use_body_mount and track.observed and track.measured_range_m == 5.
    frozen = track.snapshot(1000)
    # Reuse the mount array as a producer might; the snapshot owns its original frame.
    mount[0, 3] = .25
    pose = PlanarPose(1100, ("odom",), 0., 0., 0.)
    fresh = dict(objects, timestamp=1.1)
    frame = dict(frame, map=np.full_like(depth["map"], 4.95), frame_id=1100, capture_ms=1100,
                 source_frame_id=1100, control_ms=1100, obstacle_capture_pose=pose, body_from_optical=mount)
    current = replace(geometry, base_from_optical=mount)
    config.gate_chi2 = .00001
    match = match_observation(frozen, fresh, frame, ego_motion=(0., 0., 0.), elapsed_s=.1, target="person",
                              config=replace(config, scan_geometry=current))
    assert match is not None and match.point == pytest.approx((4.95, 0.))
    stale_range = dict(frame, map=np.full_like(depth["map"], 5.))
    assert match_observation(frozen, fresh, stale_range, ego_motion=(0., 0., 0.), elapsed_s=.1,
                             target="person", config=replace(config, scan_geometry=current)) is None
    feedback = dict(feedback, received_ms=1100, obstacle_pose=pose, motion_pose=pose, stamp_ms=1100)
    policy.step(detections=fresh, depth=frame, odom=feedback, config=config, state=state, dt=.1)
    assert track.accepted_detection and track.measured_range_m == 4.95


@pytest.mark.parametrize("body", [True, False])
@pytest.mark.parametrize("speed", [.319, .320, .321])
def test_small_feedback_overshoot_selects_valid_deceleration(monkeypatch, body, speed):
    config, state, depth, odom = observation_case(body=body)
    odom.update(vx=speed, velocity_history=((1000, 1000, (speed, 0., 0.)),))
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    message = card.next_command()
    accepted = card._last_decision.dwa_acceptance
    assert accepted is not None and 0. < accepted.command[0] <= config.vx_max == .30
    assert accepted.motion_guard.trace.velocities[0] == (speed, 0., 0.)
    if body:
        card._publish_command(message)
    else:
        card._stage_command(message, card._output_epoch)
    assert tuple(packets[-1]["values"][i] for i in (0, 1, 5)) == accepted.command


@pytest.mark.parametrize("branch", ["route", "prefix", "advance"])
@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
def test_real_reference_reaches_serialized_wire(monkeypatch, branch, body):
    if branch in ("advance", "prefix"):
        config, state, depth, odom = advance_case(body=body)
        if branch == "prefix":
            # After a fresh view during an observation turn, target-directed
            # progress can resume without completing the scan.
            state.active_scan.retire_motion()
            assert state.active_scan.begin((1.,))
    else:
        config, state, depth, odom = observation_case(body=body, attempted=branch == "prefix")
    policy.adopt_limits(config, {"limits": {"upper": [.28, 0., 0., 0., 0., config.wz_max]}})
    calls = observe_selector(monkeypatch)
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)
    now = depth["control_ms"]
    monkeypatch.setattr(plugin.time, "time", lambda: now / 1000)
    monkeypatch.setattr(card, "_observation_snapshot", lambda **_: (None, depth, odom, now))
    card._tick()
    assert len(packets) == 1
    choice = state.dwa_proposal.choice
    assert state.dwa_trace["branch"] == branch
    assert calls[-1][0]["x_limit"] == (0., .28)
    assert choice.check.accepted and choice.trace.stopped
    assert choice.trace.velocities[-1] == pytest.approx((0., 0., 0.), abs=1e-12)
    vx, vy, wz = choice.command
    assert vx > 0.
    assert packets[0]["values"] == [vx, vy, 0., 0., 0., wz]
    assert packets[0]["seq"] == state.dwa_trace["wire_seq"]
    assert card._last_decision.dwa_acceptance.decision_id == state.dwa_trace["decision_id"]


def test_full_topology_and_visible_execution_have_independent_evidence():
    config, _, _, _ = observation_case()
    geometry, goal = config.local_geometry, (2., 0.)
    # The original side A* uses the grid origin; clearance A* uses the pivot.
    # Both reached the same standoff cell and both see only the first 0.4 m.
    for pivot in (0., -.20):
        points = ((0., 0.), (.4, .05), (1.2 - pivot, 0.))
        original = tuple((f + pivot, r) for f, r in points)
        plan = local_path.Detour(None, points, geometry, topology=original)
        tier, ref = policy._dwa_route_reference(plan, 2, goal, .80, .05)
        assert tier == "route" and ref.preview == points and ref.points == points[:2]
        assert not ref.reachable_prefix
        short = replace(plan, points=points[:2], topology=original[:2])
        tier, ref = policy._dwa_route_reference(short, 2, goal, .80, .05)
        assert tier == "reachable_prefix" and ref.reachable_prefix and not ref.preview
        explicitly_partial = replace(plan, reachable_prefix=True)
        tier, _ = policy._dwa_route_reference(explicitly_partial, 2, goal, .80, .05)
        assert tier == "reachable_prefix"


def test_clearance_route_keeps_every_original_turn_node_as_its_topology(monkeypatch):
    config, _, _, _ = observation_case()
    geometry = config.local_geometry
    occupied = np.isclose(local_path.FORWARD, .25) & np.isclose(local_path.RIGHT, .05)
    seen = []
    search = local_path.shortest_path

    def observe(allowed, *args, **kwargs):
        path = search(allowed, *args, **kwargs)
        if path:
            seen.append((allowed.copy(), path))
        return path

    monkeypatch.setattr(local_path, "shortest_path", observe)
    plan = local_path.clearance_route((~occupied, occupied), geometry, .22, (1.8, 0.), .50, (0., 0.))
    allowed, path = seen[-1]
    raw = tuple((local_path.FORWARD[node], local_path.RIGHT[node]) for node in path)
    assert len(plan.points) < len(raw)
    assert not allowed[local_path.START[0] + 4, local_path.START[1] - 2]
    assert plan.topology == raw
    tier, ref = policy._dwa_route_reference(plan, 2, (1.8, 0.), .50, .05)
    assert tier == "route" and ref.points == plan.points[:2]
    assert ref.topology == raw and ref.geometry is geometry


def test_default_latest_depth_only_executes_a_visible_detour(monkeypatch):
    config, state, depth, odom = observation_case(body=False, attempted=False)
    defaults = policy.Config()
    config = replace(config, use_obstacle_history=defaults.use_obstacle_history,
                     max_feedback_gap_ms=defaults.max_feedback_gap_ms)
    scan = state.active_scan
    scan.near_obstacles.frame = ("navi_relative", scan.motion_frame)
    scan.near_obstacles.cells.add((4, -3))
    calls = observe_selector(monkeypatch)
    card, packets = card_for_case(monkeypatch, config, state, depth, odom)

    message = card.next_command()
    card._stage_command(message, card._output_epoch)

    assert config.use_obstacle_history is False
    assert calls and any(choice and choice.check.accepted for _, choice in calls)
    assert all(not inputs["remembered"] for inputs, _ in calls)
    assert state.view_detour is not None
    assert packets[-1]["values"][0] > 0.


@pytest.mark.parametrize("restriction", [None, "obstacle", "exit", "footprint"])
def test_exhausted_detour_tail_hands_off_only_to_a_clear_target_corridor(restriction):
    config, state, depth, odom = advance_case(body=True)
    scan, geometry = state.active_scan, config.scan_geometry
    scan.retire_motion()
    scan.use_obstacle_history = config.use_obstacle_history = False
    odom["vx"] = .30
    state.commanded = (.30, 0., 0.)
    state.dwa_odom = odom
    state.dwa_admission_ms = 0 if restriction == "footprint" else 250
    state.tracker.track = Track(np.array([3., 0., 0., 0.]), np.eye(4), range_known=True)
    assert scan.update(depth, odom, geometry, receipt_only=True) is None

    # Establish a checked guard, then put the last twenty centimetres of the
    # published route in that frame. A curved tail can sweep outside the corridor.
    reference = local_path.Detour(None, ((0., 0.), (1., 0.)), geometry)
    selected = policy._dwa_detour(depth, config, state, reference, 3., 0.)
    assert selected is not None and selected.guard.check.accepted
    end = (.16, .14) if restriction == "footprint" else (.20, 0.)
    exit_line = ((.05, -.20), (.05, .20)) if restriction == "exit" else None
    route = policy.DwaAcceptance(1, selected.command, motion_guard=selected.guard,
                                 route_points=((0., 0.), end), passage_exit=exit_line, locks_detour=True)
    state.committed_route = state.published_dwa_route = route
    if restriction in ("obstacle", "footprint"):
        forward, right = (.65, 0.) if restriction == "obstacle" else (.20, .30)
        stamp, pose, free, _, *evidence = scan.views[-1]
        hits = np.isclose(local_path.FORWARD, forward) & np.isclose(local_path.RIGHT, right)
        scan.views[-1] = (stamp, pose, free & ~hits, hits, *evidence)
    state.dwa_budget = None
    state.dwa_trace.clear()

    assert policy._follow_published_route(depth, config, state, geometry, 3., 0.,
                                          foreground=restriction == "obstacle") is None

    expected = {"insufficient_room"}
    if restriction == "footprint":
        expected.add("footprint_collision")
    assert set(state.dwa_trace["rejected"]) == expected
    assert policy._target_corridor_clear(depth, config, scan, 3., 0.) == (restriction != "obstacle")
    retained = route if restriction is not None else None
    assert state.committed_route is state.published_dwa_route is retained


@pytest.mark.parametrize("change", ["frame", "missing_depth", "missing_target"])
def test_committed_body_route_is_only_released_when_the_footprint_changes(change):
    config, state, depth, odom = advance_case(body=True)
    scan, geometry = state.active_scan, config.scan_geometry
    scan.retire_motion()
    state.dwa_odom = odom
    state.goal_anchor = policy.DetourGoal(3., 0., generation=1)
    anchor = state.goal_anchor
    state.tracker.track = Track(np.array([3., 0., 0., 0.]), np.eye(4), range_known=True, generation=1)
    target = state.tracker.track
    assert scan.update(depth, odom, geometry, receipt_only=True) is None
    reference = local_path.Detour(None, ((0., 0.), (1., .12), (1.5, 0.)), geometry)
    selected = policy._dwa_detour(depth, config, state, reference, 3., 0.)
    assert selected is not None and selected.guard.check.accepted

    published = policy.DwaAcceptance(1, selected.command, motion_guard=selected.guard,
                                     route_points=reference.points, locks_detour=True)
    state.committed_route = state.published_dwa_route = published
    if change == "frame":
        geometry = local_path.camera_square(geometry.horizontal, geometry.vertical, config.half_width_m,
                                            half_length=geometry.half_length, at_front=True)
        config.local_geometry, config.scan_geometry = geometry, None
    elif change == "missing_depth":
        depth = None
    else:
        state.tracker.track = None
    state.dwa_budget = None

    result = policy._follow_published_route(depth, config, state, geometry, 3., 0., foreground=True)

    if change == "frame":
        assert result is None
        assert state.published_dwa_route is state.committed_route is None
        assert state.tracker.track is target
    else:
        assert state.published_dwa_route is state.committed_route is published
    assert state.goal_anchor is anchor


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("blocked", [False, True])
def test_differential_route_can_continue_away_from_a_remembered_flank(side, blocked):
    config, state, depth, odom = advance_case(body=False)
    camera = config.local_geometry
    geometry = local_path.camera_square(camera.horizontal, camera.vertical, .30, half_length=.30, at_front=True)
    config.local_geometry, config.half_width_m = geometry, .30
    config.vx_max, config.max_depth_age_ms, config.watchdog_ms = .50, 1200, 500
    config.motion_acceleration = (.30, .30, 1.)
    scan = state.active_scan
    scan.retire_motion()
    scan.use_obstacle_history = config.use_obstacle_history = False
    state.dwa_odom = odom
    state.dwa_admission_ms = 200
    assert scan.update(depth, odom, geometry, receipt_only=True) is None
    route = local_path.Detour(None, ((0., 0.), (.45, side * .10), (1.45, side * .48)), geometry)
    selected = policy._dwa_detour(depth, config, state, route, 2., side * .27)
    assert selected is not None
    old = policy.DwaAcceptance(1, selected.command, motion_guard=selected.guard,
                               route_points=route.points, locks_detour=True)
    state.committed_route = state.published_dwa_route = old
    hits = np.isclose(local_path.FORWARD, .15) & np.isclose(local_path.RIGHT, side * .35)
    cells = [[.15, side * .36]]
    if blocked:
        cells.append([.15, -side * .31])
    squares = local_path.ObstacleSquares(np.array(cells), 0.)
    scan.near_projection = NearProjection(hits, squares, selected.guard.pose, 1100)
    state.dwa_budget = None
    state.dwa_trace.clear()

    result = policy._follow_published_route(depth, config, state, geometry, 2., side * .27, foreground=True)

    assert state.dwa_trace["attempts"][0]["rejected"] == {"remembered_obstacle": 4}
    if blocked:
        assert result is None and state.dwa_proposal is None
    else:
        assert result is not None and result[0] > 0. and result[2] * side > 0.
        assert state.dwa_proposal.guard.check.accepted
    assert scan.near_squares is squares
    assert state.committed_route is old
