"""ROS-free DWA tests. These exercise the planner independently of sensors."""
import math

import numpy as np
import pytest

from plugins.navi import dwa, local_path, policy
from plugins.navi.local_path import Geometry


def _geometry():
    return Geometry(None, None, np.eye(4), .30, .30, None)


def _grid(*, obstacles=()):
    free = np.ones(local_path.SHAPE, dtype=bool)
    occupied = np.zeros_like(free)
    for forward, right in obstacles:
        row = local_path.START[0] + round(forward / local_path.RESOLUTION)
        col = local_path.START[1] + round(right / local_path.RESOLUTION)
        occupied[row, col] = True
    return free, occupied


def _run(route, *, grid=None, measured=(.1, 0., 0.), previous=(.1, 0., 0.), lateral=False,
         target=(3., 0.), yaw_limits=(-.7,.7), stop_distance=.7, **kwargs):
    options = dict(route=route, grid=grid if grid is not None else _grid(),
                   geometry=_geometry(), half_width=.26, measured=measured, previous=previous,
                   x_limit=(0., .5), y_limit=(-.4, .4) if lateral else (0., 0.),
                   yaw_limit=yaw_limits, floors=(.08, .08, .7, .04),
                   target=target, stop_distance=stop_distance, horizon=1., max_candidates=64)
    options.update(kwargs)
    return dwa.select(**options)


def test_non_lateral_never_issues_vy():
    choice = _run(((0.,0.),(.2,.2),(.6,.4)), lateral=False)
    assert choice is not None
    assert choice.command[1] == 0.
    assert choice.progress > 0.


def test_lateral_route_chooses_lateral_velocity_and_keeps_original_target():
    choice = _run(((0.,0.),(0.,-.35),(.45,-.35)), lateral=True)
    assert choice is not None
    assert choice.command[1] > 0.
    assert choice.command[0] >= 0.
    assert choice.progress > 0.


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("follow_published", [False, True])
def test_lateral_detour_keeps_steering_authority_toward_the_target(side, follow_published):
    target_bearing = side * .60
    yaw_limits = dwa.view_yaw_limits((-.7, .7), target_bearing, .80, 1.)
    choice = _run(((0., 0.), (1.3, -side * .32)), lateral=True, follow_published=follow_published,
                  target=(3., 3. * math.tan(target_bearing)), yaw_limits=yaw_limits,
                  floors=(.08, .08, .4, .20), view_bearing=target_bearing)
    assert choice is not None and choice.check.accepted
    assert choice.command[0] > 0. and side * choice.command[1] > 0.
    assert side * choice.command[2] <= 0.
    if follow_published:
        assert choice.command[2] == pytest.approx(-target_bearing / 2.)
    assert yaw_limits[0 if side > 0. else 1] == -.7 * side


def test_braking_sweep_rejects_an_obstacle_near_the_chassis():
    obstacles = [(round(x,2), round(y,2))
                 for x in np.arange(.4,.9,.05) for y in np.arange(-.45,.46,.05)]
    blocked = _run(((0.,0.),(.8,0.)), grid=_grid(obstacles=obstacles))
    assert blocked is None


def test_rotation_checks_body_corners_not_only_center():
    geo = _geometry()
    obstacles = np.array([[.39, 0.]])
    assert not dwa._collides(((0.,0.,0.),), obstacles, geo, .26, .05)[0]
    assert dwa._collides(((0.,0.,math.pi/4),), obstacles, geo, .26, .05)[0]


def test_command_deadband_not_lifted_after_collision_check():
    choice = _run(((0.,0.),(.45,.0)), lateral=False)
    assert choice is not None
    vx,vy,wz = choice.command
    assert vx >= .08
    assert vy == 0.
    assert wz == 0. or abs(wz) >= .04


@pytest.mark.parametrize("right,yaw", [(.02, 0.), (.06, -.08), (-.06, .08)])
def test_retained_route_sweeps_an_executable_yaw_when_pursuit_falls_below_the_floor(right, yaw):
    choice = _run(((0., 0.), (1.6, right)), follow_published=True,
                  floors=(.12, .08, .3, .08))

    assert choice is not None and choice.check.accepted
    assert choice.command == pytest.approx((.5, 0., yaw))
    assert choice.check.reason is None


def test_retained_short_route_creeps_through_a_bend():
    choice = _run(((0., 0.), (.23, .0805)), follow_published=True, lateral=False,
                  measured=(.1, 0., 0.), previous=(.1, 0., 0.), target=None,
                  floors=(.08, 0., .4, .06))

    assert choice is not None and choice.check.accepted
    assert 0. < choice.command[0] < .25 and choice.command[2] < 0.


def test_retained_route_prefers_a_slower_arc_with_more_braking_clearance():
    choice = _run(((0., 0.), (1.6, .25)), grid=_grid(obstacles=[(.4, .4)]),
                  follow_published=True, target=None, measured=(.35, 0., -.1), previous=(.35, 0., -.1))
    assert choice is not None and choice.check.accepted
    assert 0. < choice.command[0] < .5 and choice.command[2] < 0.
    assert choice.clearance > .05


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("arc_clear", [False, True])
def test_a_bent_route_sweeps_the_declared_turn_floor_before_sending(side, arc_clear):
    swept = []

    def visible(command, poses, duration):
        swept.append(command)
        return dwa.CandidateCheck(arc_clear, "occluded_prefix")

    choice = _run(((0., 0.), (1., side * .15)), follow_published=True,
                  target=None, x_limit=(0., .12), floors=(.12, 0., .4, .2),
                  validate_candidate=visible)
    expected = (.12, 0., -side * .2)
    assert swept == [expected]
    if arc_clear:
        assert choice.command == expected and choice.check.accepted
    else:
        assert choice is None


@pytest.mark.parametrize("blocked", [False, True])
def test_short_window_checks_the_preferred_motion_before_fanning_out(monkeypatch, blocked):
    previous = (.2, 0., -.12)
    remaining = [.07]
    budget = dwa.CandidateBudget(64, time_left=lambda: remaining[0])
    evaluate = dwa.MotionContext.evaluate

    def finish_one_sweep(context, prediction):
        checked = evaluate(context, prediction)
        remaining[0] = -.001
        return checked

    monkeypatch.setattr(dwa.MotionContext, "evaluate", finish_one_sweep)
    diagnostics = {}
    choice = _run(((0., 0.), (.5, .10), (1.4, .20)), previous=previous,
                  grid=_grid(obstacles=[(.35, 0.)] if blocked else []),
                  candidate_budget=budget, diagnostics=diagnostics)

    assert diagnostics["sampled_candidates"] == diagnostics["physical_evaluations"] == 1
    if blocked:
        assert choice is None
        assert diagnostics["rejected"]["footprint_collision"] == 1
    else:
        assert choice is not None and choice.check.accepted
        assert choice.command == previous


def test_missing_feedback_and_invalid_route_do_not_start_dwa():
    assert _run(((0.,0.),(.4,0.)), measured=None) is None
    assert _run(((0.,0.),)) is None
    assert _run(((0.,0.),(.4,0.)), horizon=0.01) is None


def test_no_motion_outside_obstacle_target_stop_band():
    choice = _run(((0.,0.),(.6,0.)), target=(.8,0.),stop_distance=.72)
    assert choice is None or all(math.dist((f,r),(.8,0.)) >= .71
                                      for f,r,_ in choice.trajectory)


def test_response_ramps_actual_velocity_instead_of_instantaneous_command():
    poses = dwa._rollout((.5,0.,0.), (0.,0.,0.), horizon=.2, dt=.05,
                         acceleration=(.5,.5,1.), braking=(.45,.45,.8))
    assert 0 < poses[-1][0] < .02


@pytest.mark.parametrize("command,measured,horizon", [
    ((.15, 0., 0.), (2., 0., 0.), 1.2),
    ((.15, -.2, .3), (-.3, .1, -.4), 1.2),
    ((.15, 0., -.3), (-.8, 0., .9), .4),
    ((.15, 0., 0.), (0., 0., 0.), 1.2),
])
def test_stopping_horizon_matches_the_complete_response_prediction(command, measured, horizon):
    acceleration, braking = (.5, .5, 1.), (.45, .45, .8)
    predicted = dwa._predict(command, measured, horizon=horizon, dt=.04,
                             acceleration=acceleration, braking=braking, stop=True)
    assert predicted.stopped
    assert dwa.stopping_horizon(command, measured, horizon=horizon) == pytest.approx(predicted.duration)


def test_stopping_horizon_does_not_authorize_an_incomplete_braking_tail():
    command, measured = (.15, 0., 0.), (4., 0., 0.)
    predicted = dwa._predict(command, measured, horizon=1.2, dt=.04,
                             acceleration=(.5, .5, 1.), braking=(.45, .45, .8), stop=True)
    assert not predicted.stopped
    assert dwa.stopping_horizon(command, measured, horizon=1.2) is None


def test_stopping_horizon_uses_the_selected_response_parameters():
    command, measured, horizon = (.15, -.2, .3), (-.3, .1, -.4), .4
    acceleration, braking = (.3, .6, .8), (.2, .4, .5)
    predicted = dwa._predict(command, measured, horizon=horizon, dt=.04,
                             acceleration=acceleration, braking=braking, stop=True)
    assert predicted.stopped
    assert dwa.stopping_horizon(command, measured, horizon=horizon,
                                acceleration=acceleration, braking=braking) == pytest.approx(predicted.duration)


def test_can_keep_previously_commanded_speed_if_still_valid():
    choice = _run(((0.,0.),(.5,0.)), previous=(.2,0.,0.), measured=(.18,0.,0.))
    assert choice is not None and choice.command[0] > 0.


def test_normalizes_right_positive_path_against_ros_yaw():
    left = _run(((0.,0.),(.45,-.25)), lateral=False)
    right = _run(((0.,0.),(.45,.25)), lateral=False)
    assert left is not None and right is not None
    assert left.command[2] >= 0.
    assert right.command[2] <= 0.


def test_dwa_is_default_and_has_bounded_configuration():
    assert policy.Config().local_controller == "dwa"
    assert policy.Config(local_controller="dwa").local_controller == "dwa"
    assert policy.Config(local_controller="legacy").local_controller == "legacy"
    with pytest.raises(ValueError, match="local_controller"):
        policy.Config(local_controller="unknown")
    with pytest.raises(ValueError, match="dwa_horizon"):
        policy.Config(dwa_horizon_s=5.)
    with pytest.raises(ValueError, match="dwa_samples"):
        policy.Config(dwa_samples=0)


STRAIGHT = ((0., 0.), (.8, 0.))


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("arc_clear", [False, True])
@pytest.mark.parametrize("initial_yaw", [0., .0042])
def test_new_detour_prefers_a_checked_first_arc_on_its_selected_side(side, arc_clear, initial_yaw):
    route = ((0., 0.), (.65, side * .12), (1.4, side * .28))
    initial = (.20, 0., -side * initial_yaw)
    arc, _ = dwa.track_route(route, .20, 1., (-.7, .7), body_origin=(0., 0.),
                             lateral=False, yaw_floor=.04)
    assert arc[2] * side < 0.
    swept = []

    def visible(command, poses, duration):
        swept.append(command)
        return dwa.CandidateCheck(command == initial or (arc_clear and command == arc), "occluded_prefix")

    choice = _run(route, previous=initial, measured=initial, x_limit=(0., .20), floors=(.08, .08, .7, 0.),
                  candidate_budget=dwa.CandidateBudget(64), guidance=True, validate_candidate=visible)
    assert arc in swept
    assert choice.check.accepted
    assert choice.command == (arc if arc_clear else initial)


@pytest.mark.parametrize("follow_published", [False, True])
@pytest.mark.parametrize("allow_transition", [False, True])
def test_moving_arc_handoff_sweeps_interpolated_yaw(follow_published, allow_transition):
    checked = []

    def visible(command, poses, duration):
        yaw = command[2]
        checked.append(yaw)
        return dwa.CandidateCheck(yaw == 0. or (allow_transition and yaw == pytest.approx(-.4)), "blocked")

    budget = dwa.CandidateBudget(32)
    choice = _run(STRAIGHT, previous=(.2, 0., -.5), measured=(.2, 0., -.02),
                  x_limit=(0., .22), horizon=.8, candidate_budget=budget,
                  follow_published=follow_published, handoff_yaw_step=.1,
                  validate_candidate=visible)
    assert choice.command[0] > 0.
    assert -.4 in checked
    assert choice.command[2] == pytest.approx(-.4 if allow_transition else 0.)
    assert budget.remaining >= 0


def test_guard_rejects_fast_candidate_but_accepts_slow_candidate():
    geometry = _geometry()
    grid = _grid(obstacles=[(.60, .40)])
    previous = measured = (.10, 0., 0.)

    def clear(command):
        return local_path.motion_sweep_clear(grid, geometry, .26, 1., (measured, previous, command))

    assert not clear((.50, 0., 0.))
    assert clear((.10, 0., 0.))
    options = dict(grid=grid, previous=previous, measured=measured, horizon=1.2)
    unchecked = _run(STRAIGHT, **options)
    assert unchecked is not None and not clear(unchecked.command)
    diagnostics = {}
    choice = _run(STRAIGHT, **options,
                  validate_candidate=lambda cmd, poses, duration: dwa.CandidateCheck(clear(cmd), "sweep"),
                  diagnostics=diagnostics)
    assert choice is not None and 0. < choice.command[0] < unchecked.command[0]
    assert clear(choice.command)
    assert diagnostics["rejected"]["sweep"] > 0
    assert diagnostics["admissible_candidates"] + sum(diagnostics["rejected"].values()) == diagnostics["sampled_candidates"]


def test_camera_reference_rotates_about_body_center():
    geometry = Geometry(None, None, np.eye(4), 0., .60, None)
    yaw = -math.pi / 4
    twist = (0., 0., yaw)
    route = ((0., 0.), (-.087868, .212132))
    options = dict(geometry=geometry, rotation_center=(-.3, 0.), footprint_mode="camera_approx",
                   previous=twist, measured=twist, x_limit=(0., 0.), yaw_limits=(-1., 1.),
                   max_candidates=1, target=None)
    diags = {}
    assert _run(route, diagnostics=diags, **options) is None
    assert diags["reason"] == "missing_candidate_validator"

    grid = _grid()
    def guard(command, poses, duration):
        return dwa.CandidateCheck(local_path.motion_sweep_clear(
            grid, geometry, .26, duration, (twist, twist, command)), "sweep")

    choice = _run(route, validate_candidate=guard, **options)
    assert choice is not None
    assert choice.trajectory[-1] == pytest.approx((-.087868, .212132, -yaw), abs=1e-6)


@pytest.mark.parametrize('limit', [(-.2, .2), (0., 0.)])
def test_unique_bounded_sampler_covers_allowed_motion_families(limit):
    def collect(budget):
        seen, diag = [], {}
        def guard(command, poses, duration):
            seen.append(command)
            return dwa.CandidateCheck(True)
        _run(STRAIGHT, y_limit=limit, previous=(1., .3, 3.), measured=(0., 0., 0.),
             candidate_commands=((.2, 0., 0.), (.2, 0., 0.), (.3, .21, 0.)),
             max_candidates=budget, validate_candidate=guard, diagnostics=diag)
        assert len(set(seen)) == len(seen) == diag["sampled_candidates"] <= budget
        assert all(limit[0] <= vy <= limit[1] and 0 <= vx <= .5 and abs(wz) <= .7 for vx, vy, wz in seen)
        return seen, diag["uncovered_families"]

    seen, missing = collect(64)
    assert not missing
    assert (.2, 0., 0.) in seen and (.08, 0., 0.) in seen
    assert all(any(0 < vx < .35 and vy == 0. and sign*wz > 0 for vx, vy, wz in seen)
               for sign in (-1., 1.))
    if limit[1]:
        assert all(any(sign*vy > 0 for _, vy, _ in seen) for sign in (-1., 1.))
    small, missing = collect(2)
    assert missing and len(small) == 2 and collect(2) == (small, missing)


def test_floor_command_starts_from_rest_and_no_subfloor_command_is_checked():
    seen = []
    def guard(command, poses, duration):
        seen.append(command)
        return dwa.CandidateCheck(True)
    choice = _run(STRAIGHT, x_limit=(0., .08), yaw_limits=(0., 0.), measured=(0., 0., 0.),
                  previous=(.08 - 5e-10, 0., 0.), validate_candidate=guard)
    assert choice is not None and choice.command == (.08, 0., 0.)
    assert seen == [(.08, 0., 0.)]
    assert 0. < choice.trajectory[1][0] < .08 * .04


def test_braking_distance_covers_analytic_stop():
    poses = dwa._rollout((.45, 0., 0.), (.45, 0., 0.), horizon=.4, dt=.04,
                         acceleration=(.5, .5, 1.), braking=(.45, .45, .8), stop=True)
    tail_distance = poses[-1][0] - .45 * .4
    assert .225 - 1e-12 <= tail_distance <= .225 + 1e-12


def test_response_reverses_with_braking_then_acceleration_at_zero_crossing():
    # .1 m/s -> 0 in .1s, then 0 -> -.025 m/s in .1s. Areas .005 and -.00125.
    poses = dwa._rollout((-.1, 0., 0.), (.1, 0., 0.), horizon=.2, dt=.2,
                         acceleration=(.25, .25, 1.), braking=(1., 1., 1.))
    assert poses[-1][0] == pytest.approx(.00375)


def test_unfinished_braking_budget_rejects_candidate():
    diag = {}
    assert _run(STRAIGHT, measured=(3., 0., 0.), previous=(.5, 0., 0.), horizon=.1,
                braking=(.45, .45, .8), max_candidates=1, diagnostics=diag) is None
    assert diag["rejected"] == {"braking_incomplete": 1}


def test_body_sweep_covers_obstacle_between_pose_samples():
    # At 3 m/s the first .04s spans .12m. The .05m square at .05m misses both endpoints.
    geom = Geometry(None, None, np.eye(4), .002, .002, None)
    diag = {}
    assert _run(STRAIGHT, grid=_grid(obstacles=[(.05, 0.)]), geometry=geom, half_width=.002,
                x_limit=(0., 3.), measured=(3., 0., 0.), previous=(3., 0., 0.), target=None,
                braking=(3., 3., 1.), max_candidates=1, diagnostics=diag) is None
    assert diag["rejected"] == {"footprint_collision": 1}


def test_camera_guard_uses_full_braking_duration():
    grid = _grid(obstacles=[(.6, 0.)])
    geom = Geometry(None, None, np.eye(4), 0., .6, None)
    cmd = (.45, 0., 0.)
    def clear(horizon):
        return local_path.motion_sweep_clear(grid, geom, .26, horizon, (cmd, cmd, cmd))
    assert clear(.4) and not clear(1.4)
    durations, diag = [], {}
    def guard(command, poses, duration):
        durations.append(duration)
        return dwa.CandidateCheck(clear(duration), "braking_sweep")
    assert _run(STRAIGHT, grid=grid, geometry=geom, footprint_mode="camera_approx", rotation_center=(-.3, 0.),
                validate_candidate=guard, previous=cmd, measured=cmd, horizon=.4,
                max_candidates=1, diagnostics=diag) is None
    assert durations == pytest.approx([1.4]) and diag["rejected"] == {"braking_sweep": 1}


@pytest.mark.parametrize("mode", ["body", "camera_approx"])
def test_rotating_camera_standoff_includes_braking_tail(mode):
    # At t=1 the camera is still over .23 m from the target. Braking carries
    # it through the .12 m stand-off. The chassis origin stays well outside.
    cmd = (.1, 0., -1.)
    mount = np.eye(4)
    mount[0, 3] = .4
    if mode == "body":
        geometry = Geometry(None, None, mount, .4, .4, None)
        target = (.1, .52)
    else:
        geometry = Geometry(None, None, np.eye(4), 0., .8, None)
        target = (-.3, .52)
    diag = {}
    assert _run(STRAIGHT, geometry=geometry, previous=cmd, measured=cmd, yaw_limits=(-1., 1.),
                target=target, stop_distance=.12, footprint_mode=mode, rotation_center=(-.4, 0.),
                validate_candidate=lambda *_: dwa.CandidateCheck(True),
                max_candidates=1, diagnostics=diag) is None
    assert diag["rejected"] == {"target_standoff": 1}


def test_original_target_standoff_has_only_roundoff_tolerance():
    diag = {}
    # Terminal x = .1*1 + .1**2/(2*.45). The camera ends 5mm inside the stop band.
    assert _run(STRAIGHT, target=(.8061111111, 0.), stop_distance=.7, max_candidates=1,
                diagnostics=diag) is None
    assert diag["rejected"] == {"target_standoff": 1}


def test_straight_stop_can_reach_the_exact_camera_standoff_boundary():
    # The entire 1 s command plus brake moves .111111...m monotonically forward.
    choice = _run(STRAIGHT, target=(.7 + .1 + .1**2 / .9, 0.), stop_distance=.7,
                  max_candidates=1)
    assert choice is not None and choice.command == (.1, 0., 0.)


def test_nonfinite_target_cannot_yield_a_motion_command():
    assert _run(STRAIGHT, target=(math.nan, 0.)) is None


def test_body_prediction_without_target_does_not_require_optical_offset():
    geom = Geometry(None, None, None, .3, .3, None)
    assert _run(STRAIGHT, geometry=geom, target=None) is not None


@pytest.mark.parametrize("standing,moving,cap", [(.7, .04, 1.), (.25, .8, .7)])
@pytest.mark.parametrize("side", [-1, 1])
def test_sampler_finds_each_legal_standing_turn_from_rest(standing, moving, cap, side):
    geometry = Geometry(None, None, np.eye(4), 0., .6, None)
    seen = set()

    def guard(command, poses, duration):
        seen.add(command)
        return dwa.CandidateCheck(abs(command[2]) <= standing + .01, "turn_limit")

    choice = _run(((0., 0.), (0., side*.6)), geometry=geometry, measured=(0., 0., 0.),
                  previous=(0., 0., 0.), x_limit=(0., 0.), yaw_limits=(-cap, cap),
                  floors=(.08, .08, standing, moving), footprint_mode="camera_approx",
                  rotation_center=(-.3, 0.), validate_candidate=guard)
    assert choice is not None and choice.command == (0., 0., -side*standing)
    assert choice.progress == 0.
    # Wider standing speeds stay in the sampler when moving yaw is unavailable.
    assert (0., 0., side*cap) in seen


def test_small_budget_reports_omitted_standing_turns_in_both_directions():
    diagnostics = {}
    choice = _run(STRAIGHT, max_candidates=1, diagnostics=diagnostics)
    assert choice is not None and choice.command == (.1, 0., 0.)
    assert diagnostics["sampled_candidates"] == 1
    assert {"left_standing_turn", "right_standing_turn"} <= set(diagnostics["uncovered_families"])


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -.01])
def test_invalid_stop_distance_cannot_disable_original_target_guard(invalid):
    diagnostics = {}
    assert _run(STRAIGHT, target=(.2, 0.), stop_distance=invalid, diagnostics=diagnostics) is None
    assert diagnostics["reason"] == "invalid_inputs"
    assert diagnostics["sampled_candidates"] == 0


def test_dense_grid_checks_final_poses_and_preserves_minimum_clearance():
    geometry = Geometry(None, None, None, .015, .01, None)
    poses = tuple((i / 100, 0., 0.) for i in range(97))
    # A sizeable background forces the sweep to use multiple pose batches.
    obstacles = np.tile([.45, .15], (1201, 1))
    obstacles[-1] = (1.04, .09)
    hit, clearance = dwa._collides(poses, obstacles, geometry, .01, .05, margin=.007)
    assert not hit
    # Distance from the obstacle square to the body's front-right corner at x=.96.
    assert clearance == pytest.approx(math.hypot(.04, .055) - .007, abs=1e-12)
    obstacles[-1] = (.997, 0.)
    assert dwa._collides(poses, obstacles, geometry, .01, .05, margin=.007) == (True, 0.)


def test_body_centered_standing_turn_can_follow_route_heading_from_rest():
    choice = _run(((0., 0.), (0., -.6)), measured=(0., 0., 0.), previous=(0., 0., .10),
                  x_limit=(0., 0.), yaw_limits=(-.10, .10), floors=(.08, .08, .10, .10),
                  max_candidates=1)
    assert choice is not None and choice.command == (0., 0., .10)
    assert choice.progress == 0. and choice.trajectory[-1][2] < 0.


def test_physical_footprint_checks_camera_mode_without_an_external_guard():
    from plugins.navi.motion import FootprintSpec
    geom = Geometry(None, None, np.eye(4), 0., .40, None)
    spec = FootprintSpec(0., .40, .20, .02, "camera_approx")
    options = dict(geometry=geom, footprint=spec, half_width=None, footprint_mode="camera_approx",
                   rotation_center=(-.20, 0.), measured=(.15, 0., 0.), previous=(.15, 0., 0.),
                   max_candidates=1)
    assert _run(STRAIGHT, grid=_grid(obstacles=[(.10, .30)]), **options) is not None
    assert _run(STRAIGHT, grid=_grid(obstacles=[(.10, .20)]), **options) is None


@pytest.mark.parametrize("side", [-1., 1.])
def test_offset_camera_standing_turn_earns_only_heading_progress(side):
    from plugins.navi.motion import FootprintSpec

    geometry = Geometry(None, None, np.eye(4), 0., .40, None)
    command = (0., 0., -side*.5)
    choice = _run(((0., 0.), (0., side*.6)), geometry=geometry,
                  footprint=FootprintSpec(0., .40, .20, .02, "camera_approx"),
                  measured=(0., 0., 0.), previous=command, horizon=1.2,
                  x_limit=(0., 0.), yaw_limits=(-.5, .5),
                  floors=(.08, .08, .10, .10), max_candidates=1)
    assert choice is not None and choice.command == command
    assert choice.progress == 0.
    assert side * choice.trajectory[-1][1] == pytest.approx(.20 * math.sin(.475))


@pytest.mark.parametrize("route,expected", [
    # Body pivot moves from x=-.20 to -.08, behind the first forward waypoint.
    (((0., 0.), (.80, 0.)), .12),
    # Body pivot crosses the bend at x=-.10, with .30 m of route behind it at t=0.
    (((-.50, 0.), (-.10, 0.), (-.10, .80)), .10),
])
def test_offset_camera_translation_uses_body_change_along_the_grid_route(route, expected):
    from plugins.navi.motion import FootprintSpec

    geometry = Geometry(None, None, np.eye(4), 0., .40, None)
    command = (.12, 0., 0.)
    choice = _run(route, geometry=geometry,
                  footprint=FootprintSpec(0., .40, .20, .02, "camera_approx"),
                  measured=command, previous=command, horizon=1.,
                  x_limit=(0., .12), yaw_limits=(0., 0.), max_candidates=1)
    assert choice is not None and choice.command == command
    assert choice.progress == pytest.approx(expected)


def test_camera_clearance_changes_candidate_cost():
    from plugins.navi.motion import FootprintSpec

    geometry = Geometry(None, None, np.eye(4), 0., .40, None)
    grid = _grid(obstacles=[(.42, .28)])
    choices = []
    for speed in (.12, .24):
        command = (speed, 0., 0.)
        choice = _run(
            STRAIGHT, grid=grid, geometry=geometry,
            footprint=FootprintSpec(0., .40, .20, .02, "camera_approx"),
            previous=command, measured=command, x_limit=(0., .24), max_candidates=1)
        assert choice is not None and choice.check.accepted
        choices.append(choice)
    slow, fast = choices
    assert 0. < fast.clearance < slow.clearance < .25
    assert fast.components.obstacle > slow.components.obstacle > 0.


def test_reference_length_does_not_reward_wrong_straight_motion():
    # The post is right of the chassis. Both candidate sweeps remain clear,
    # but the known left-hand bend should retain its scoring opportunity.
    grid = _grid(obstacles=[(.43, .33)])
    straight = (.22, 0., 0.)
    left = (.22, 0., .18)
    options = dict(grid=grid, previous=straight, measured=straight, x_limit=(0., .22),
                   max_candidates=1)
    short = _run(((0., 0.), (.08, 0.)), **options)
    long = _run(((0., 0.), (1.2, 0.)), **options)
    assert short is not None and long is not None
    assert short.progress == pytest.approx(long.progress)
    assert short.score == pytest.approx(long.score)
    bend = ((0., 0.), (.08, 0.), (.23, -.05), (.51, -.18))
    bending = _run(bend, **{**options, "max_candidates": 2, "candidate_commands": (left,)})
    assert bending is not None and bending.check.accepted
    assert bending.command == left


def test_camera_rotation_is_not_body_translation_progress():
    from plugins.navi.motion import FootprintSpec

    geometry = Geometry(None, None, np.eye(4), 0., .40, None)
    command = (0., 0., .5)
    options = dict(geometry=geometry, footprint=FootprintSpec(0., .40, .20, .02, "camera_approx"),
                   measured=(0., 0., 0.), previous=command, horizon=1.2, x_limit=(0., 0.),
                   yaw_limits=(-.5, .5), floors=(.08, .08, .10, .10), max_candidates=1)
    route = ((0., 0.), (0., -.6))
    toward = _run(route, target=(3., -.9), **options)
    away = _run(route, target=(3., .9), **options)
    assert toward is not None and away is not None
    assert toward.command == away.command == command
    assert toward.progress == away.progress == 0.
    assert toward.score == pytest.approx(away.score)
    assert toward.components.goal == away.components.goal == 0.


def test_bounded_physics_keeps_exhaustive_winner_and_shadow_validation():
    routes = (STRAIGHT, ((0., 0.), (.6, .25), (1., .30)))
    for points, alternate in (routes, routes[::-1]):
        for blocked in ((), ((.45, .20),), ((.2, 0.),)):
            grid = _grid(obstacles=blocked)
            for guard in (None, lambda cmd, *_: dwa.CandidateCheck(cmd[2] >= 0., "shadow")):
                options = dict(grid=grid, alternate_routes=(alternate,), validate_candidate=guard)
                expected = _run(points, **options)
                diag = {}
                actual = _run(points, **options, candidate_budget=dwa.CandidateBudget(64), diagnostics=diag)
                assert actual == expected
                assert 0 < diag["physical_evaluations"] <= diag["sampled_candidates"] <= 64


def test_guidance_switches_sides_when_a_new_obstacle_blocks_the_prior_path():
    geometry = _geometry()
    f, r = local_path.FORWARD, local_path.RIGHT
    box = (f >= .8) & (f <= 1.35) & (abs(r) <= .25)
    options = (geometry, .22, (2.7, 0.), .50, (0., 0.))
    first = local_path.clearance_route((~box, box), *options)
    side = np.sign(min(first.points, key=lambda p: abs(p[0] - 1.05))[1])
    assert abs(side) == 1
    barrier = box | ((f >= .90) & (f <= 1.20) & (side*r >= 0.))
    rerouted = local_path.clearance_route((~barrier, barrier), *options, first.points)
    crossing = min(rerouted.points, key=lambda p: abs(p[0] - 1.05))
    assert side*crossing[1] < -.4
    assert abs(rerouted.points[-1][1]) <= local_path.RESOLUTION


def test_soft_clearance_starts_a_wide_detour_earlier():
    f, r = local_path.FORWARD, local_path.RIGHT
    obstacle = (f >= 1.) & (f <= 1.5) & (abs(r) <= .25)
    path = local_path.clearance_route((~obstacle, obstacle), _geometry(), .3025, (2.85, 0.), .60, (0., 0.))
    assert abs(path.topology[3][1]) >= local_path.RESOLUTION
    passing = [abs(right) for forward, right in path.topology if 1. <= forward < 1.4]
    assert min(passing) >= .69


def test_distant_goal_gets_early_clearance_around_a_visible_obstacle():
    forward, right = local_path.FORWARD, local_path.RIGHT
    obstacle = (forward >= 1.55) & (forward <= 1.95) & (abs(right) <= .25)
    geometry = _geometry()
    path = local_path.clearance_route((~obstacle, obstacle), geometry, .30, (4.2, 0.), .60, (0., 0.))
    assert path is not None and path.reachable_prefix
    assert any(f < 1. for f, r in path.topology if abs(r) >= .10)
    assert min(abs(r) for f, r in path.topology if 1.55 <= f <= 1.95) >= .65
    side_obstacle = obstacle & (right != 0.)
    side_obstacle = np.roll(side_obstacle, 18, axis=1)
    assert local_path.clearance_route((~side_obstacle, side_obstacle), geometry, .30,
                                      (4.2, 0.), .60, (0., 0.)) is None


def test_dwa_prefers_clearance_beyond_the_hard_footprint():
    route = ((0., 0.), (.65, 0.), (1., 0.))
    close = _run(route, grid=_grid(obstacles=[(.4, .55)]))
    wider = _run(route, grid=_grid(obstacles=[(.4, .6)]))
    assert close.check.accepted and wider.check.accepted
    assert close.components.obstacle > wider.components.obstacle == 0.


def test_soft_clearance_preserves_a_feasible_narrow_opening():
    walls = np.isclose(abs(local_path.RIGHT), .40)
    grid = ~walls, walls
    path = local_path.clearance_route(grid, _geometry(), .3025, (2.85, 0.), .60, (0., 0.))
    assert path is not None and all(right == 0. for _, right in path.topology)
    choice = _run(((0., 0.), (.65, 0.), (1., 0.)), grid=grid)
    assert choice.command[0] > 0. and choice.components.obstacle > 0.


def test_a_star_applies_short_range_clearance_after_detour_shape(monkeypatch):
    f, r = local_path.FORWARD, local_path.RIGHT
    occupied = (abs(f - 1.5) <= .11) & (abs(r) <= .11)
    shape_cost = local_path._detour_shape_cost
    shortest_path = local_path.shortest_path
    scores = {}

    def capture_shape(*args, **kwargs):
        cost, goal, blocker_exit = shape_cost(*args, **kwargs)
        scores["shape"] = cost.copy()
        return cost, goal, blocker_exit

    def capture_route(*args, **kwargs):
        scores["route"] = kwargs["cell_cost"].copy()
        return shortest_path(*args, **kwargs)

    monkeypatch.setattr(local_path, "_detour_shape_cost", capture_shape)
    monkeypatch.setattr(local_path, "shortest_path", capture_route)
    assert local_path.clearance_route((~occupied, occupied), _geometry(), .27, (3., 0.), .60, (0., 0.))
    extra = scores["route"] - scores["shape"]
    assert extra[np.isclose(f, 1.5) & np.isclose(r, .45)].item() > 0.
    # A cell 28 cm beyond the footprint still incurs the new shortfall.
    assert extra[np.isclose(f, 1.5) & np.isclose(r, .65)].item() == pytest.approx(2. * (1. - .28 / .30) ** 2)
    assert extra[np.isclose(f, 1.5) & np.isclose(r, .70)].item() == 0.


def test_unused_prediction_can_be_checked_by_a_later_reference_after_budget_exhaustion():
    budget = dwa.CandidateBudget(8)
    options = dict(candidate_budget=budget, max_candidates=8, yaw_limits=(0., 0.))
    back = {}
    assert _run(((0., 0.), (-1., 0.)), diagnostics=back, **options) is None
    assert back["sampled_candidates"] > 0 and back["physical_evaluations"] == 0
    # All executable straight commands in this context were already predicted.
    budget.remaining = 0
    forward = {}
    winner = _run(STRAIGHT, diagnostics=forward, **options)
    assert winner is not None and winner.command[0] > 0.
    assert forward["sampled_candidates"] == 0
    assert 0 < forward["physical_evaluations"] <= forward["cached_candidates"]


def test_bounded_selection_checks_visibility_before_using_a_winning_score():
    unrestricted = _run(STRAIGHT)
    shadow = local_path.FORWARD >= unrestricted.trace.body_poses[-1][0] - .05

    def visible(_, poses, duration):
        points = tuple((f, r) for f, r, _ in poses)
        return dwa.CandidateCheck(local_path.visible_prefix(points, shadow) == len(points), "shadow")

    options = dict(validate_candidate=visible)
    expected = _run(STRAIGHT, **options)
    diagnostics = {}
    winner = _run(STRAIGHT, **options, candidate_budget=dwa.CandidateBudget(64), diagnostics=diagnostics)
    assert winner == expected and winner is not None
    assert winner.command != unrestricted.command and winner.command[0] > 0.
    assert diagnostics["rejected"]["shadow"] > 0

def test_reused_response_preserves_full_inputs_and_never_reuses_obstacle_proof():
    from dataclasses import replace

    geometry = _geometry()
    footprint = dwa.FootprintSpec(.3, .3, .2, .02)
    context = dwa.MotionContext((0., 0., 0.), (0., 0., 0.), .8, .10, geometry, footprint,
                                None, (), None, (2., 0.), .4)
    cache = dwa.PredictionCache()
    cmd = (.1, 0., -.1)
    trace, reused = cache.get(context, cmd)
    assert not reused and trace.stopped
    assert cache.get(context, cmd) == (trace, True)
    assert context.evaluate(trace).accepted
    changed_obstacles = replace(context, occupied=local_path.ObstacleSquares(np.array([[0., 0.]]), 0.))
    again, reused = cache.get(changed_obstacles, cmd)
    assert reused and again is trace and not changed_obstacles.evaluate(again).accepted

    mount = geometry.base_from_optical.copy()
    mount[1, 3] = .1
    for changed in (replace(context, measured=(-0., 0., 0.)), replace(context, previous=(.1, 0., 0.)),
                    replace(context, horizon=.81), replace(context, latency=.11),
                    replace(context, acceleration=(.1, .2, .3)), replace(context, braking=(.2, .3, .4)),
                    replace(context, geometry=replace(geometry, base_from_optical=mount)),
                    replace(context, footprint=replace(footprint, mode="camera_approx"), rotation_center=(-.1, 0.))):
        prediction, reused = cache.get(changed, cmd)
        assert not reused and prediction == changed.predict(cmd)
    signed, reused = cache.get(context, (cmd[0], -0., cmd[2]))
    assert not reused and signed == context.predict((cmd[0], -0., cmd[2]))
    with pytest.raises(ValueError):
        cache.get(context, (float("nan"), 0., 0.))
    geometry.base_from_optical[0, 3] = .12
    changed_mount, reused = cache.get(context, cmd)
    assert not reused and changed_mount.camera_offset != trace.camera_offset
    # Integer inputs retain the predictor's own precision even when two
    # distinct values round to the same float64 bit pattern.
    integer = replace(context, previous=(2**54 + 1, 0., 0.))
    assert cache.get(integer, cmd)[1] is False
    assert cache.get(integer, cmd)[1] is False
    for n in range(70):
        cache.get(context, (.005*n, .01, 0.))
    assert len(cache.traces) == dwa.MAX_SEARCH_CANDIDATES


def test_explicit_invalid_rotation_center_keeps_the_native_predictor_behavior():
    from dataclasses import replace

    geometry = _geometry()
    footprint = dwa.FootprintSpec(.3, .3, .2, .02, mode="camera_approx")
    camera = dwa.MotionContext((0., 0., 0.), (0., 0., 0.), .8, .1, geometry, footprint,
                               None, (), None, (2., 0.), .4)
    cmd = (.1, 0., -.1)
    cache = dwa.PredictionCache()
    trace, _ = cache.get(camera, cmd)
    assert cache.get(camera, cmd) == (trace, True)
    for center in ((), 0, np.array([.1])):
        invalid = replace(camera, rotation_center=center)
        with pytest.raises((TypeError, IndexError, ValueError)) as validation:
            invalid.predict(cmd)
        with pytest.raises(type(validation.value)):
            cache.get(invalid, cmd)
    supplied_array = replace(camera, rotation_center=np.array([-.12, .03]))
    assert cache.get(supplied_array, cmd) == (supplied_array.predict(cmd), False)
    # The body predictor ignores rotation_center. Cache validation must not
    # reject an input that reaches that original behavior.
    body = replace(camera, footprint=replace(footprint, mode="body"), rotation_center=17)
    assert cache.get(body, cmd) == (body.predict(cmd), False)


@pytest.mark.parametrize("veto", [False, True])
@pytest.mark.parametrize("prior_speed", [0., .15])
def test_capped_straight_checks_visibility_before_early_return(veto, prior_speed):
    diagnostics = {}
    checked = []

    def visible(command, poses, duration):
        checked.append(command)
        return dwa.CandidateCheck(not veto, "occluded_prefix" if veto else None)

    choice = _run(((0., 0.), (1., 0.)), previous=(prior_speed, 0., 0.), measured=(prior_speed, 0., 0.),
                  x_limit=(0., .15), diagnostics=diagnostics, candidate_budget=dwa.CandidateBudget(64),
                  validate_candidate=visible)
    assert checked.count((.15, 0., 0.)) == 1
    if veto:
        assert choice is None
        assert diagnostics["considered_candidates"] > 1
    else:
        assert choice.command == (.15, 0., 0.) and choice.trace.stopped
        assert diagnostics["considered_candidates"] == diagnostics["physical_evaluations"] == 1


def test_guidance_penalizes_driving_parallel_to_an_offset_lane():
    command = (.15, 0., 0.)
    trace = dwa._predict(command, command, horizon=1., dt=.05,
                         acceleration=(1., 1., 1.), braking=(1., 1., 1.), stop=True)
    def heading(offset):
        score, _, _ = dwa.score_candidate(
            command, trace, None, ((0., offset), (.7, offset)), command,
            half_width=.26, limits=((0., .15), (0., 0.), (-1., 1.)),
            progress_scale=.2, target=None, guidance=True)
        return score.heading

    assert heading(.10) > heading(0.)


def test_local_score_prioritizes_route_progress_and_turn_continuity():
    command = (.30, 0., .10)
    trace = dwa._predict(command, command, horizon=1., dt=.05,
                         acceleration=(1., 1., 1.), braking=(1., 1., 1.), stop=True)
    route = ((0., 0.), (1., 0.))

    def score(previous, target):
        cost, progress, reason = dwa.score_candidate(
            command, trace, None, route, previous, half_width=.26,
            limits=((0., .5), (0., 0.), (-.5, .5)), progress_scale=.5, target=target)
        assert reason is None and progress > .25
        return cost

    turn_change = score((.30, 0., 0.), (1., 0.))
    speed_change = score((.20, 0., .10), (1., 0.))
    away = score((.30, 0., 0.), (-1., 0.))
    assert turn_change.smooth > 1.2 * speed_change.smooth
    assert 0. < away.total - turn_change.total < .07


def test_slow_entry_arc_can_gain_heading_before_half_a_cell_of_progress():
    cmd = (.015, 0., .30)
    trace = dwa._predict(cmd, cmd, horizon=.8, dt=.05, acceleration=(1., 1., 1.),
                         braking=(1., 1., 1.), stop=True)
    def score(side):
        return dwa.score_candidate(cmd, trace, None, ((0., 0.), (.4, side * .4)), cmd,
                                   half_width=.26, limits=((0., .15), (0., 0.), (-1., 1.)),
                                   progress_scale=.2, target=None)
    good, progress, _ = score(-1)
    assert 0 < progress < local_path.RESOLUTION / 2 and good is not None
    assert score(1)[0] is None


@pytest.mark.parametrize("offset,obstacles,fast", [
    (.018, (), True), (.12, (), False), (.018, ((.35, .30),), False)])
def test_only_clear_subcell_reference_jitter_keeps_checked_straight_speed(offset, obstacles, fast):
    diag = {}
    choice = _run(((0., 0.), (.30, offset), (.6, offset)), grid=_grid(obstacles=obstacles),
                  measured=(.15, 0., 0.), previous=(.15, 0., 0.), x_limit=(0., .15),
                  diagnostics=diag, candidate_budget=dwa.CandidateBudget(64))
    if fast:
        assert choice.command == (.15, 0., 0.)
        assert diag["considered_candidates"] == 1
    else:
        assert diag["considered_candidates"] > 1


def test_bent_reference_seeds_its_standing_turn_and_slow_entry_arc():
    previous = (.15, 0., 0.)
    preferred = ((.13, .12, .31),)
    limits = ((0., .15), (-.4, .4), (-1., 1.))
    sampled, _ = dwa._commands(previous, previous, preferred, limits, (0., 0., .1, .04), (), 24)
    assert (0., 0., .31) in sampled
    assert (.15 / 8, 0., .31) in sampled


def test_scoring_compacts_straight_grid_steps_but_keeps_turns_and_reversals():
    route = ((0., 0.), (.05, 0.), (.10, 0.), (.15, 0.), (.15, .05), (.15, 0.))
    segments = dwa._route_segments(route)
    assert len(segments) == 3
    assert [s.tangent for s in segments] == [(1., 0.), (0., 1.), (0., -1.)]
    progress, error, tangent = dwa._project_to_route((.09, .01), route, segments)
    assert (progress, error) == pytest.approx((.09, .01))
    assert tangent == (1., 0.)


def test_previous_clear_straight_is_checked_before_building_the_command_lattice(monkeypatch):
    monkeypatch.setattr(dwa, "_commands", lambda *args: pytest.fail("built unused command lattice"))
    previous = (.15, 0., 0.)
    choice = _run(STRAIGHT, previous=previous, measured=previous, x_limit=(0., .15),
                  candidate_budget=dwa.CandidateBudget(64))
    assert choice.command == previous and choice.check.accepted and choice.trace.stopped


def test_clear_near_equivalent_arc_retains_checked_chassis_command():
    previous = (.5, 0., -.04)
    route = ((0., 0.), (.45, -.03), (.75, -.03))
    options = dict(previous=previous, measured=previous)
    choice = _run(route, candidate_budget=dwa.CandidateBudget(64), **options)
    assert choice.command == previous
    guarded = _run(route, candidate_budget=dwa.CandidateBudget(64), **options,
                   validate_candidate=lambda cmd, *_: dwa.CandidateCheck(cmd != previous))
    assert guarded is not None and guarded.command != previous


@pytest.mark.parametrize("case", ["open", "short", "bend", "obstacle", "shadow", "lower_limit"])
def test_straight_tracking_speed_handoff(case):
    route = ((0., 0.), (2., 0.))
    if case == "short":
        route = ((0., 0.), (.15, 0.))
    if case == "bend":
        route += ((2.3, .3),)
    guard = (lambda cmd, *_: dwa.CandidateCheck(cmd[0] <= .15, "shadow")) if case == "shadow" else None
    choice = _run(route, previous=(.30, 0., .01), x_limit=(0., .15),
                  straight_handoff_limit=.25 if case == "lower_limit" else .30,
                  candidate_budget=dwa.CandidateBudget(64), validate_candidate=guard,
                  grid=_grid(obstacles=((.6, .15),) if case == "obstacle" else ()))
    if case == "open":
        assert choice.command == (.30, 0., 0.)
    else:
        assert choice is None or choice.command[0] <= .15


@pytest.mark.parametrize("blocked", [False, True])
def test_timed_route_keeps_only_a_verified_prefix_when_scoring_runs_out(monkeypatch, blocked):
    remaining = [1.]
    predictions = []
    predict = dwa.MotionContext.predict

    def predict_with_deadline(context, command):
        predictions.append(command)
        if len(predictions) == 2:
            remaining[0] = 0.
        return predict(context, command)

    monkeypatch.setattr(dwa.MotionContext, "predict", predict_with_deadline)
    choice = _run(((0., 0.), (.5, .10), (1.4, .20)), previous=(.2, 0., -.12),
                  grid=_grid(obstacles=[(.35, 0.)] if blocked else []),
                  candidate_budget=dwa.CandidateBudget(64, time_left=lambda: remaining[0], early_return_s=.1))
    if blocked:
        assert choice is None
    else:
        assert choice is not None and choice.command == predictions[0] and choice.check.accepted


def test_route_velocity_cost_prefers_keeping_forward_speed_to_a_shorter_turn():
    limits = ((0., .5), (0., 0.), (-.7, .7))
    previous = (.5, 0., 0.)
    straight = ((0., 0.), (1.8, 0.))
    shorter = ((0., 0.), (.35, .25), (.9, 0.))
    cost = local_path.route_velocity_change
    assert cost(straight, *limits, previous) == pytest.approx(0.)
    assert cost(shorter, *limits, previous) > 0.
    assert cost(((0., 0.), (.4, 0.), (1.8, 0.)), *limits, previous) == cost(straight, *limits, previous)


def test_route_velocity_cost_preserves_declared_lateral_motion():
    limits = ((0., .5), (-.25, .4), (0., 0.))
    previous = (.4, -.2, 0.)
    steady = ((0., 0.), (.6, .3), (1.2, .6))
    reversal = ((0., 0.), (.6, -.3), (1.2, 0.))
    cost = local_path.route_velocity_change
    assert cost(steady, *limits, previous) < cost(reversal, *limits, previous)
    assert math.isinf(cost(steady, (0., .5), (0., .4), (0., 0.), previous))


def test_alternate_references_with_the_same_visible_prefix_rank_by_velocity_change():
    # The same visible prefix can lead to either a zigzag or a straight continuation.
    shortcut = ((0., 0.), (.7, 0.), (.85, .2), (1.05, 0.), (1.3, 0.))
    steady = ((0., 0.), (.7, 0.), (1.3, 0.))
    choice = _run(shortcut, alternate_routes=(steady,), yaw_limits=(-.15, .15), target=(3., 0.))
    assert choice is not None and choice.check.accepted
    assert choice.reference_index == 1


def test_route_ranking_does_not_override_collision_checks():
    straight = ((0., 0.), (1.8, 0.))
    turn = ((0., 0.), (.4, .32), (1.5, .42))
    guarded = _run(straight, alternate_routes=(turn,), previous=(.3, 0., 0.),
                   validate_candidate=lambda cmd, *_: dwa.CandidateCheck(cmd[2] < 0, "blocked_lane"))
    assert guarded is not None and guarded.check.accepted and guarded.command[2] < 0.


def test_clearance_cost_has_no_attraction_outside_one_body_length():
    geometry = _geometry()
    wall = local_path.ObstacleSquares(np.column_stack((np.linspace(.4, 2.4, 41), np.full(41, 1.))), 0.)
    near = ((.5, .6), (2., .6))
    preferred = ((.5, .1), (2., .1))
    far = ((.5, -.6), (2., -.6))
    gap_cost = local_path.route_clearance_cost
    assert gap_cost(preferred, wall, geometry, .275) == pytest.approx(0.)
    assert gap_cost(near, wall, geometry, .275) > gap_cost(preferred, wall, geometry, .275)
    assert gap_cost(far, wall, geometry, .275) == pytest.approx(0.)
    assert gap_cost(far, None, geometry, .275) == 0.


def test_equal_velocity_change_routes_prefer_a_body_length_from_the_wall():
    near = ((0., 0.), (.7, 0.), (1.3, .3), (1.8, .3))
    clear = ((0., 0.), (.7, 0.), (1.3, -.3), (1.8, -.3))
    wall = _grid(obstacles=[(f, .65) for f in np.linspace(1.05, 1.8, 16)])
    choice = _run(near, grid=wall, alternate_routes=(clear,), occupied=(), target=(3., 0.))
    assert choice is not None and choice.reference_index == 1


def test_isolated_blocker_detour_enters_on_expanded_circle_tangent():
    f, r = local_path.FORWARD, local_path.RIGHT
    box = (abs(f - 1.95) <= .201) & (abs(r) <= .251)
    route = local_path.clearance_route((~box, box), _geometry(), .30, (3.05, 0.), .60, (0., 0.),
                                       body_half_width=.25)
    radius = math.hypot(.225, .275) + 4.5 * .25
    angle = math.asin(radius / 1.95)
    forward, right = route.points[1]
    assert abs(math.atan2(right, forward)) == pytest.approx(angle, abs=.09)

def test_grid_route_rounds_a_corner_only_with_clearance():
    points = ((0., 0.), (.4, 0.), (.7, .3), (1.2, .3))
    allowed = np.ones(local_path.SHAPE, dtype=bool)
    rounded = local_path._round_grid_route(points, allowed, (0., 0.))
    assert rounded[:2] == points[:2] and rounded[-1] == points[-1]
    assert len(rounded) > len(points)
    steps = np.diff(rounded[1:], axis=0)
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    assert np.max(abs(np.diff(np.unwrap(headings)))) < math.pi / 4 - 1e-3
    assert local_path._round_grid_route(points, ~allowed, (0., 0.)) == points


@pytest.mark.parametrize("jambs_only", [False, True])
def test_two_observed_boundaries_center_the_passing_route(jambs_only):
    f, r = local_path.FORWARD, local_path.RIGHT
    if jambs_only:
        walls = (abs(f - 1.40) < .026) & ((r <= .001) | (r >= .749))
        midpoint = .375
    else:
        walls = (f >= .45) & (np.isclose(r, -.65) | np.isclose(r, 1.15))
        midpoint = .25
    route = local_path.clearance_route((~walls, walls), _geometry(), .27, (2.8, .81), .55,
                                       (0., 0.))
    assert route is not None
    passage = next(right for forward, right in route.topology if forward >= 1.39)
    assert passage == pytest.approx(midpoint, abs=local_path.RESOLUTION)


def test_wide_corridor_scores_its_midline_below_both_offsets():
    r = local_path.RIGHT
    walls = local_path.occupied_squares(np.isclose(r, -.85) | np.isclose(r, 1.35))
    center = ((.5, .25), (2., .25))
    score = local_path.route_clearance_cost
    assert score(center, walls, _geometry(), .27) == pytest.approx(0.)
    for offset in (-.15, .15):
        shifted = ((.5, .25 + offset), (2., .25 + offset))
        assert score(shifted, walls, _geometry(), .27) > 0.


def test_box_detour_holds_passing_lane_until_the_rear_clears():
    geometry = _geometry()
    f, r = local_path.FORWARD, local_path.RIGHT
    occupied = (abs(f - 1.60) <= .401) & (abs(r) <= .301)
    route = local_path.clearance_route((~occupied, occupied), geometry, .30, (3.25, 0.), .60,
                                       (0., 0.), body_half_width=.25)
    assert route is not None
    clear_at = float(f[occupied].max()) + local_path.RESOLUTION / 2 + geometry.rear
    start, end = route.blocker_exit
    assert start[0] == pytest.approx(end[0]) and start[0] >= clear_at
    side = [abs(right) for forward, right in route.topology if 1.60 <= forward < clear_at]
    assert len(side) > 2
    assert all(after >= before - 1e-9 for before, after in zip(side, side[1:]))


def test_compact_detour_exit_follows_the_tangent_toward_the_target():
    f, r = local_path.FORWARD, local_path.RIGHT
    box_center, goal = 1.25, 3.75
    occupied = (abs(f - box_center) <= .301) & (abs(r) <= .151)
    route = local_path.clearance_route((~occupied, occupied), _geometry(), .30, (goal, 0.), .60,
                                       (0., 0.), body_half_width=.25)
    assert route is not None
    # At this range the wider soft radius includes the start.
    radius = math.hypot(.325, .175) + 3 * .25
    slope = math.tan(math.asin(radius / (goal - box_center)))
    exit_leg = [(forward, right) for forward, right in route.topology if forward >= 2.5]
    assert len(exit_leg) >= 4
    assert all(abs(abs(right) - (goal-forward)*slope) <= .1 for forward, right in exit_leg)


@pytest.mark.parametrize("visible_outside", [False, True])
def test_compact_blocker_beside_an_opening_needs_visible_room_on_its_outer_side(visible_outside):
    forward, right = local_path.FORWARD, local_path.RIGHT
    face = np.isclose(forward, 1.5)
    occupied = face & (((right >= -.10) & (right <= .30)) | ((right <= -.80) & (right >= -1.10)))
    free = ~occupied
    if not visible_outside:
        free &= right < .30
    route = local_path.clearance_route((free, occupied), _geometry(), .22, (3.0, 0.), .60,
                                       (0., 0.), body_half_width=.20)
    assert route is not None
    assert (route.blocker_exit is not None) == visible_outside
    if visible_outside:
        passing = next(r for f, r in route.topology if f >= 1.5)
        assert passing > .30 + .20
