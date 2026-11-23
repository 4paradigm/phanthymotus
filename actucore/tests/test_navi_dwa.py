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
def test_capped_straight_continuation_checks_visibility_before_early_return(veto):
    diagnostics = {}
    checked = []

    def visible(command, poses, duration):
        checked.append(command)
        return dwa.CandidateCheck(not veto, "occluded_prefix" if veto else None)

    choice = _run(((0., 0.), (1., 0.)), previous=(.5, 0., 0.),
                  diagnostics=diagnostics, candidate_budget=dwa.CandidateBudget(64),
                  validate_candidate=visible)
    assert checked.count((.5, 0., 0.)) == 1
    if veto:
        assert choice is None
        assert diagnostics["considered_candidates"] > 1
    else:
        assert choice.command == (.5, 0., 0.) and choice.trace.stopped
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
