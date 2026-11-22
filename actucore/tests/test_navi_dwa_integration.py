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

from plugins.navi import dwa, local_path, plugin, policy
from plugins.navi.camera import HorizontalIntrinsics


def observation_case(*, controller="dwa", body=True, attempted=True):
    camera = local_path.camera_square(HorizontalIntrinsics(640, 420., 320.),
                                     (480, 420., 240.), .20, at_front=not body)
    geometry = (local_path.body_square(camera, {"body_from_optical": camera.base_from_optical})
                if body else camera)
    config = policy.Config(local_controller=controller, local_planning=True,
                           use_lateral=False, local_geometry=geometry,
                           scan_geometry=geometry if body else None,
                           half_width_m=.20, vx_max=.30, max_depth_age_ms=800,
                           watchdog_ms=500)
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
    assert twist[0] <= .15, "A prefix must keep its local approach speed cap"
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
                           state=state, dt=.1, control_period_s=.1)
    assert calls == []
    assert decision.values == [.15, 0., 0., 0., 0., .25]
    assert not state.dwa_selected


def test_existing_body_route_still_uses_real_dwa(monkeypatch):
    config, state, depth, odom = observation_case(attempted=False)
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1)
    accepted = [choice for _, choice in calls if choice is not None]
    assert accepted and state.dwa_selected
    assert tuple(decision.values[i] for i in (0, 1, 5)) == accepted[-1].command


def test_dwa_policy_intersects_operator_and_directional_lateral_limits(monkeypatch):
    config, state, depth, odom = observation_case(controller="legacy")
    policy.step(detections=None, depth=depth, odom=odom, config=config,
                state=state, dt=.1, control_period_s=.1)
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
                           half_width_m=.20, vx_max=.30, max_depth_age_ms=500, watchdog_ms=300)
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
                           state=state, dt=.1, control_period_s=.1)
    assert calls and any(choice is not None for _, choice in calls)
    assert state.dwa_selected and state.checked_prefix
    assert decision.values[0] > 0. and decision.values[1] == 0.
    assert state.goal_anchor is anchor
    assert not state.active_scan.active
    assert state.active_scan.advance_m is None
    assert state.executing_leg is None


@pytest.mark.parametrize("turning,stale", [(True, False), (False, True)])
def test_ready_advance_still_obeys_stop_and_freshness_conditions(monkeypatch, turning, stale):
    config, state, depth, odom = advance_case(body=False)
    if turning:
        odom["wz"] = .12
    if stale:
        depth["capture_invalid"] = True
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1)
    assert decision.values == [0.] * 6
    assert not state.dwa_selected
    assert calls == []


def test_a_changed_nonzero_policy_result_cannot_keep_an_unchecked_dwa_motion(monkeypatch):
    config, state, depth, odom = observation_case()
    calls = observe_selector(monkeypatch)
    account_idle = policy._account_idle

    def alter_after_selection(decision, *args):
        decision = account_idle(decision, *args)
        if state.dwa_selected:
            decision.values[0] *= .5
        return decision

    monkeypatch.setattr(policy, "_account_idle", alter_after_selection)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1)
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
                           state=state, dt=.1, control_period_s=.1)
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
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1)
    kwargs, choice = calls[0]
    endpoint = kwargs["route"][-1]
    blocked = local_path.route_obstacles(kwargs["grid"], kwargs["geometry"],
                                         kwargs["half_width"], allow_unknown=True)
    assert not local_path.segment_clear(
        blocked, kwargs["geometry"], kwargs["half_width"], math.atan2(endpoint[1], endpoint[0]),
        math.hypot(*endpoint), heading_tolerance=policy.ANGLE_TOLERANCE)
    assert choice is not None and decision.dwa_acceptance is not None
    assert choice.command == decision.dwa_acceptance.command
    if body:
        assert choice.command[0] > 0. and choice.command[2] != 0.


def prepared_reference_case(*, body=False):
    config, state, depth, odom = observation_case(controller="legacy", body=body)
    policy.step(detections=None, depth=depth, odom=odom, config=config,
                state=state, dt=.1, control_period_s=.1)
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


@pytest.mark.parametrize("reason", ["collision", "remembered_obstacle", "occluded_prefix",
                                    "target_standoff", "insufficient_room"])
def test_candidate_rejections_preserve_the_actual_constraint(monkeypatch, reason):
    config, state, depth, reference = prepared_reference_case()
    extra, distance = {}, 3.
    scan = state.active_scan
    if reason in ("collision", "remembered_obstacle"):
        hits = np.zeros(local_path.SHAPE, dtype=bool)
        hits[local_path.START] = True
        if reason == "collision":
            scan.historical_hits = hits
        else:
            scan.near_hits = hits
            scan.near_squares = local_path.ObstacleSquares(np.array([[0., .10]]), .27)
            scan.near_revision = scan.control_ms
    elif reason == "occluded_prefix":
        extra["shadow"] = np.ones(local_path.SHAPE, dtype=bool)
    elif reason == "target_standoff":
        distance = config.stop_distance_m / 2
    else:
        extra["room"] = 0.
    calls = observe_selector(monkeypatch)
    assert policy._dwa_detour(depth, config, state, reference, distance, 0., **extra) is None
    assert len(calls) == 1
    trace = state.dwa_trace
    assert trace["sampled_candidates"] > 0 and trace["admissible_candidates"] == 0
    assert trace["rejected"][reason] > 0
    assert set(trace["rejected"]) <= {reason, "outside_grid"}
    assert sum(trace["rejected"].values()) == trace["sampled_candidates"]


@pytest.mark.parametrize("budget", [7, 64])
def test_multiple_references_share_one_tick_sampling_budget(monkeypatch, budget):
    config, state, depth, reference = prepared_reference_case()
    config.dwa_samples = budget
    calls = observe_selector(monkeypatch)
    for _ in range(5):
        assert policy._dwa_detour(depth, config, state, reference, 3., 0.,
                                   room=0., budget=max(1, budget // 3)) is None
    assert calls
    assert sum(kwargs["diagnostics"]["sampled_candidates"] for kwargs, _ in calls) == budget
    assert state.dwa_trace["sampled_candidates"] == budget
    assert state.dwa_remaining == 0
    assert state.dwa_trace["fallback_reason"] == "sample_budget_exhausted"


def test_advance_handoff_does_not_resume_the_old_leg_on_the_next_tick():
    config, state, depth, odom = advance_case(body=False)
    anchor = state.goal_anchor
    first = policy.step(detections=None, depth=depth, odom=odom, config=config,
                        state=state, dt=.1, control_period_s=.1)
    assert first.dwa_acceptance is not None
    depth = dict(depth, frame_id=1200, control_ms=1200, source_frame_id=1200)
    odom = dict(odom, vx=first.values[0], wz=first.values[5],
                received_ms=1200, source_sample_id=1200)
    second = policy.step(detections=None, depth=depth, odom=odom, config=config,
                         state=state, dt=.1, control_period_s=.1)
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
                state=state, dt=.1, control_period_s=.1)
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
def test_route_cannot_restore_forward_motion_inside_the_configured_stop_line(monkeypatch, body, lateral):
    config, state, depth, odom = observation_case(body=body, attempted=False)
    config.use_lateral = lateral
    config.obstacle_stop_m = .85
    depth["map"][220:370, 302:338] = .75
    state.commanded = (0., 0., 0.)
    odom.update(vx=0., velocity_history=((1000, 1000, (0., 0., 0.)),))
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1)
    assert calls, "Use the real route selector at the previously bypassed stop line"
    assert all(kwargs["x_limit"] == (0., 0.) for kwargs, _ in calls)
    assert decision.values[0] == 0.
    if lateral:
        assert decision.values[1] or decision.values[5], "A checked escape remains available"


@pytest.mark.parametrize("body", [True, False])
def test_short_ready_advance_room_bounds_the_complete_dwa_prediction(monkeypatch, body):
    config, state, depth, odom = advance_case(body=body)
    state.active_scan.advance_m = .10
    config.dwa_horizon_s = 3.
    config.floor_vx_moving = .08
    calls = observe_selector(monkeypatch)
    decision = policy.step(detections=None, depth=depth, odom=odom, config=config,
                           state=state, dt=.1, control_period_s=.1)
    assert calls
    assert all(choice is None for _, choice in calls)
    assert decision.dwa_acceptance is None
    assert state.dwa_trace["rejected"]["insufficient_room"] > 0


def test_constructed_escape_prefix_is_not_reported_as_an_astar_route(monkeypatch):
    config, state, depth, _ = prepared_reference_case(body=True)
    state.active_scan.near_hits = np.zeros(local_path.SHAPE, dtype=bool)
    monkeypatch.setattr(local_path, "opening_route", lambda *args: None)
    monkeypatch.setattr(state.local_plan, "route", lambda *args, **kwargs: None)
    calls = observe_selector(monkeypatch)
    result = policy._start_detour_segment(depth, config, state, config.local_geometry,
                                         2., (-1., 1.), 3., 0.)
    assert calls and any(choice is not None for _, choice in calls)
    assert isinstance(result, tuple) and state.dwa_selected
    assert state.dwa_trace["branch"] == "prefix"
