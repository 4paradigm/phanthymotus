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
    return dwa.select(route=route, grid=grid if grid is not None else _grid(),
                      geometry=_geometry(), half_width=.26,
                      measured=measured, previous=previous, x_limit=(0., .5),
                      y_limit=(-.4, .4) if lateral else (0.,0.), yaw_limit=yaw_limits,
                      floors=(.08, .08, .7, .04), target=target, stop_distance=stop_distance,
                      horizon=kwargs.pop("horizon", 1.0), max_candidates=64, **kwargs)


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


def test_can_keep_previously_commanded_speed_if_still_valid():
    choice = _run(((0.,0.),(.5,0.)), previous=(.2,0.,0.), measured=(.18,0.,0.))
    assert choice is not None and choice.command[0] > 0.


def test_normalizes_right_positive_path_against_ros_yaw():
    left = _run(((0.,0.),(.45,-.25)), lateral=False)
    right = _run(((0.,0.),(.45,.25)), lateral=False)
    assert left is not None and right is not None
    assert left.command[2] >= 0.
    assert right.command[2] <= 0.


def test_dwa_is_opt_in_and_has_bounded_configuration():
    assert policy.Config().local_controller == "legacy"
    assert policy.Config(local_controller="dwa").local_controller == "dwa"
    with pytest.raises(ValueError, match="local_controller"):
        policy.Config(local_controller="unknown")
    with pytest.raises(ValueError, match="dwa_horizon"):
        policy.Config(dwa_horizon_s=5.)
    with pytest.raises(ValueError, match="dwa_samples"):
        policy.Config(dwa_samples=0)
