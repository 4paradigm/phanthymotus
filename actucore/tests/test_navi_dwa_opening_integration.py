"""Native opening admission and isolation from the fixed-leg fallback."""
import math

import pytest

from plugins.navi import depth as depth_mod, dwa, local_path, policy
from test_navi_opening_relaxation import _scene


def _case(body, controller):
    frame, geometry, _, _ = _scene(body=body)
    config = policy.Config(local_controller=controller, local_planning=True, use_lateral=False,
                           local_geometry=geometry, scan_geometry=geometry if body else None,
                           half_width_m=.20, vx_max=.30, floor_vx=.08,
                           max_depth_age_ms=500, watchdog_ms=300)
    # The original target is beyond the wall and to the side of the opening.
    # It remains the navigation goal even when the next local leg centres up.
    target = (3., 1.2)
    anchor = policy.DetourGoal(target[0], -target[1])
    state = policy.State(target="person", motion_source="measured", goal_anchor=anchor)
    depth = dict(map=frame, ground=depth_mod.ground_mask(frame, geometry.horizontal, geometry.vertical),
                 frame_id=1000, source_frame_id=1000, capture_ms=1000, control_ms=1000)
    odom = dict(vx=0., vy=0., wz=0., received_ms=1000, stamp_ms=1000, source_sample_id=1000)
    state.dwa_odom = odom
    state.dwa_projection = local_path.DepthProjection(frame, depth["ground"])
    assert state.active_scan.update(depth, odom, geometry, receipt_only=not body) is None
    return config, state, depth, geometry, target


def _observe_selector(monkeypatch):
    calls = []
    select = dwa.select

    def record(**kwargs):
        choice = select(**kwargs)
        calls.append((kwargs, choice))
        return choice

    monkeypatch.setattr(dwa, "select", record)
    return calls


def _start(case):
    config, state, depth, geometry, target = case
    return policy._start_detour_segment(depth, config, state, geometry, 1., (1., -1.),
                                        math.hypot(*target), math.atan2(target[1], target[0]))


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
def test_offset_target_opening_reaches_native_dwa_without_retargeting(monkeypatch, body):
    case = _case(body, "dwa")
    config, state, depth, geometry, target = case
    anchor = state.goal_anchor
    before = (anchor.x, anchor.y)
    calls = _observe_selector(monkeypatch)

    result = _start(case)

    assert isinstance(result, tuple), "A measured opening should provide a checked moving reference"
    accepted = [(inputs, choice) for inputs, choice in calls if choice is not None]
    assert accepted and state.dwa_selected
    inputs, choice = accepted[-1]
    assert result[:3] == choice.command and result[0] > 0.
    proposal = state.dwa_proposal
    assert proposal.choice is choice
    assert proposal.guard.trace is choice.trace and proposal.guard.check is choice.check
    assert choice.check.accepted and choice.trace.stopped
    assert proposal.guard.trace.duration > proposal.guard.prediction_horizon_s
    assert proposal.guard.goal == pytest.approx(local_path.target_in_grid(geometry, *target))
    assert inputs["target"] == proposal.guard.goal
    assert proposal.guard.footprint.clearance_margin_m == pytest.approx(.02)
    assert state.goal_anchor is anchor and (anchor.x, anchor.y) == before
    # This selected reference goes through the small measured opening rather
    # than pointing directly at the target on the other side of the wall.
    end_f, end_r = proposal.reference.points[-1]
    assert abs(end_r / end_f) < abs(target[1] / target[0]) / 2
    assert not state.active_scan.route_segment


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("controller", ["legacy", "dwa"])
def test_relaxed_reference_cannot_enter_the_unchecked_fixed_leg_fallback(monkeypatch, body, controller):
    case = _case(body, controller)
    _, state, _, _, _ = case
    if controller == "dwa":
        # A previous route family can consume this decision's search budget.
        # Let the actual selector reject the next family normally.
        state.dwa_budget = dwa.CandidateBudget(0)
    calls = _observe_selector(monkeypatch)

    result = _start(case)

    assert result is False
    assert not state.dwa_selected and state.dwa_proposal is None
    assert not state.active_scan.active and not state.active_scan.route_segment
    assert state.view_detour is None
    if controller == "legacy":
        assert calls == []
    else:
        assert calls and all(choice is None for _, choice in calls)
        assert state.dwa_trace["fallback_reason"] == "sample_budget_exhausted"
