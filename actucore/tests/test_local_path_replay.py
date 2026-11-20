import numpy as np
import pytest

from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.depth import ground_mask
from plugins.navi import local_path as replay

DECLARED_SPEEDS = ((0., .6), (0., .4))


def test_grid_can_make_a_complete_right_detour_with_confirmed_space():
    grid = np.full((34, 33), replay.FREE, dtype=np.int8)
    grid[10:16, 12:19] = replay.OCCUPIED
    start, right, goal = (5, 15), (13, 22), (22, 15)
    path = replay.plan_right(grid, start, right, goal, (2, 2))
    assert path is not None
    assert path[0] == start and path[-1] == goal and right in path
    for row, col in path:
        assert not (grid[row-2:row+3, col-2:col+3] == replay.OCCUPIED).any()


def test_unknown_near_the_current_footprint_blocks_a_detour():
    grid = np.full((34, 33), replay.UNKNOWN, dtype=np.int8)
    # Distant right passage is clear. The first step still enters a blind area.
    grid[9:, :] = replay.FREE
    assert replay.plan_right(grid, (5, 15), (13, 22), (22, 15), (2, 2)) is None


def test_a_diagonal_step_checks_the_intermediate_sweep():
    allowed = np.zeros((8, 8), dtype=bool)
    allowed[3, 3] = allowed[4, 4] = True
    assert replay.shortest_path(allowed, (3, 3), (4, 4)) is None


def test_velocity_ramp_checks_hits_up_to_the_edge_of_the_grid():
    geometry = replay.Geometry(None, None, np.eye(4), .20, .20, None)
    free = np.zeros(replay.SHAPE, dtype=bool)
    occupied = np.zeros_like(free)
    velocities = ((-.30, 0.),)
    assert replay.transition_clear((free, occupied), geometry, .25, 1.45, velocities)
    occupied[0, replay.START[1]] = True
    assert not replay.transition_clear((free, occupied), geometry, .25, 1.45, velocities)


@pytest.mark.parametrize("observed", [False, True])
def test_observed_detour_cannot_route_through_an_unknown_band(observed):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    occupied = (abs(replay.RIGHT) <= .10) & np.isclose(replay.FORWARD, .90)
    free = ~occupied
    if not observed:
        free &= replay.FORWARD >= .30
    route = replay._checked_translation(None, None, geometry, .25, .90, 1., 500, 600,
                                        velocity_limits=None, observed_grid=(free, occupied),
                                        body_half_width=.20)
    assert (route is not None) is observed


def test_observed_detour_can_stop_at_a_known_prefix_before_occlusion():
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    occupied = (abs(replay.RIGHT) <= .10) & np.isclose(replay.FORWARD, 1.40)
    free = (replay.FORWARD <= 1.) & ~occupied
    route = replay._checked_translation(None, None, geometry, .25, 1.40, 1., 500, 600,
                                        velocity_limits=None, observed_grid=(free, occupied),
                                        body_half_width=.20)
    assert route is not None
    assert .1 <= route.points[-1][0] <= 1. - geometry.front + 1e-9


def test_front_depth_can_execute_a_detour_with_an_unmeasured_prefix():
    horizontal = HorizontalIntrinsics(640, 315, 337)
    vertical = (480, 420, 243)
    geometry = replay.camera_square(horizontal, vertical, .23)
    tangent = (np.arange(480) - 243) / 420
    floor = np.minimum(.42 / np.maximum(tangent - .02, .001), 4.0)
    frame = np.broadcast_to(floor[:, None], (480, 640)).copy()
    frame[200:240, 318:358] = .90
    ground = ground_mask(frame, horizontal, vertical)
    assert ground is not None
    free, occupied = replay.observe(frame, geometry, replay.RIGHT, replay.FORWARD,
                                    replay.RESOLUTION, ground)
    assert free.any() and occupied.any()
    detour = replay._checked_translation(frame, ground, geometry, .33, .90, 1., 500, 600,
                                         velocity_limits=DECLARED_SPEEDS)
    assert detour is not None and detour.command is not None
    assert -detour.command[1] > 0.


def test_detour_limits_travel_to_the_checked_prefix_over_the_full_timeout(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        known_obstacle = (abs(right) <= .05) & np.isclose(forward, .9)
        free = (abs(right) <= .35 + 1e-9) & (forward > .20) & (forward <= .30 + 1e-9)
        return free, known_obstacle

    monkeypatch.setattr(replay, "observe", observed)
    command = replay._checked_translation(None, None, geometry, .33, .9, 1, 500, 600,
                                          velocity_limits=DECLARED_SPEEDS)
    assert command is not None
    vx, vy = command.command
    assert .095 / 1.15 < vx <= .6
    assert 0 < -vy <= .4
    assert np.hypot(vx, vy) * 1.15 <= np.hypot(*command.points[1]) + 1e-12


@pytest.mark.parametrize("side", [-1., 1.])
def test_live_detour_tries_a_wider_bypass_and_later_rejoin(monkeypatch, side):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        box = (abs(right) <= .05) & (forward >= .85) & (forward <= 1.26)
        side_obstacle = np.isclose(right, side * .55) & np.isclose(forward, .90)
        return np.ones_like(box), box | side_obstacle

    monkeypatch.setattr(replay, "observe", observed)
    command = replay._checked_translation(None, None, geometry, .33, .90, side, 500, 600,
                                          velocity_limits=DECLARED_SPEEDS)
    assert command is not None
    vx, vy = command.command
    assert vx > 0 and vy * side < 0
    assert abs(vx) != pytest.approx(abs(vy))
    assert np.hypot(vx, vy) > .10
    assert vx <= .6 and abs(vy) <= .4


def test_near_detour_can_choose_a_verified_lateral_only_command(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = (abs(replay.RIGHT) <= .05) & np.isclose(replay.FORWARD, .9)
    monkeypatch.setattr(replay, "observe", lambda *args: (~box, box))
    depth = {"frame_id": 500, "ground": box, "map": box}
    planner = replay.PlanCache()
    limits = ((.04, .20), (.03, .07))
    command = planner.translation(depth, geometry, .33, .90, 1., 500, 600,
                                  allow_forward=False, velocity_limits=limits)
    assert command is not None and command[0] == 0
    assert .03 <= -command[1] <= .07


def test_short_lateral_prefix_uses_a_longer_checked_move_when_clear():
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .20)
    occupied = (abs(replay.RIGHT) <= .1) & np.isclose(replay.FORWARD, .90)
    detour = replay._checked_translation(
        None, None, geometry, .30, .90, 1., 800, 600, velocity_limits=DECLARED_SPEEDS,
        allow_forward=False, observed_grid=(~occupied, occupied))
    assert detour is not None
    assert detour.command[0] == 0.
    assert detour.command[1] <= -.15 / 1.45
    assert replay.transition_clear((~occupied, occupied), geometry, .30, 1.45,
                                   (detour.command,))


@pytest.mark.parametrize('side', [-1., 1.])
def test_detour_falls_back_to_strafe_then_forward_with_bumi_speed_ranges(side):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .21)
    box = (abs(replay.RIGHT) <= .05) & np.isclose(replay.FORWARD, .9)
    free = (replay.FORWARD <= .25) & (abs(replay.RIGHT) <= .65)
    limits = ((.25, .55), (.12, .20))
    args = (None, None, geometry, .31, .90, side, 500, 300)
    detour = replay._checked_translation(*args, velocity_limits=limits, allow_forward=False,
                                         observed_grid=(free, box))
    assert detour is not None
    assert detour.command[0] == 0. and .12 <= -side * detour.command[1] <= .20
    assert detour.points[1][0] == 0.
    assert detour.points[-1][0] > .9 and detour.points[-1][1] == 0.
    limited_view = replay._checked_translation(*args, velocity_limits=limits, allow_forward=False,
                                               observed_grid=(free & (abs(replay.RIGHT) < .4), box))
    assert limited_view.command == detour.command


@pytest.mark.parametrize("start_occupied", [False, True])
def test_live_detour_stops_if_the_start_or_passing_lane_is_blocked(monkeypatch, start_occupied):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        if start_occupied:
            occupied = np.isclose(right, 0) & (np.isclose(forward, .2) | np.isclose(forward, .9))
        else:
            occupied = ((np.isclose(right, 0) & np.isclose(forward, .90))
                        | np.isclose(forward, .60))
        return np.ones_like(occupied), occupied

    monkeypatch.setattr(replay, "observe", observed)
    assert replay._checked_translation(None, None, geometry, .33, .90, 1., 500, 600,
                                       velocity_limits=DECLARED_SPEEDS) is None


@pytest.mark.parametrize("side", [-1., 1.])
def test_live_detour_can_continue_along_the_side_when_centerline_is_blocked(monkeypatch, side):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = np.isclose(replay.RIGHT, 0.) & np.isclose(replay.FORWARD, .90)
    center_wall = np.isclose(replay.RIGHT, 0.) & (replay.FORWARD >= 1.35)
    occupied = box | center_wall
    monkeypatch.setattr(replay, "observe", lambda *args: (~occupied, occupied))

    detour = replay._checked_translation(None, None, geometry, .33, .90, side, 500, 600,
                                         velocity_limits=DECLARED_SPEEDS)

    assert detour is not None and detour.command is not None
    assert -side * detour.command[1] > 0.
    forward, right = detour.points[-1]
    assert forward >= 1.35 and side * right >= .35


def test_live_detour_can_enter_passing_lane_without_a_complete_exit(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = (abs(replay.RIGHT) <= .15) & np.isclose(replay.FORWARD, .90)
    wall = np.isclose(replay.FORWARD, 1.35)
    occupied = box | wall
    monkeypatch.setattr(replay, "observe", lambda *args: (~occupied, occupied))

    detour = replay._checked_translation(None, None, geometry, .33, .90, 1.,
                                         800, 600, velocity_limits=DECLARED_SPEEDS)

    assert detour is not None and detour.command is not None
    assert detour.command[1] < 0.
    assert .90 <= detour.points[-1][0] < 1.35
    assert replay.transition_clear((~occupied, occupied), geometry, .33, 1.45,
                                   ((0., 0.), detour.command))


@pytest.mark.parametrize("stop", [.45, .60, .90])
def test_detour_can_finish_at_the_requested_standoff_before_the_rejoin_point(monkeypatch, stop):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    occupied = (abs(replay.RIGHT) <= .1) & np.isclose(replay.FORWARD, 1.45)
    monkeypatch.setattr(replay, "observe", lambda *args: (~occupied, occupied))
    target_forward, target_right, tolerance = 1.90, .40, .05

    detour = replay._checked_translation(
        None, None, geometry, .33, 1.45, 1., 800, 600, velocity_limits=DECLARED_SPEEDS,
        target=(target_forward, target_right, stop, tolerance))

    assert detour is not None and detour.command is not None
    forward, right = detour.points[-1]
    assert stop <= np.hypot(target_forward - forward, target_right - right) <= stop + tolerance
    assert forward < 1.45 + geometry.rear + .25


def test_detour_can_start_sideways_when_the_passing_lane_is_only_partly_reachable(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = (abs(replay.RIGHT) <= .10) & np.isclose(replay.FORWARD, .90)
    occupied = box | np.isclose(replay.FORWARD, 1.10)
    monkeypatch.setattr(replay, "observe", lambda *args: (~occupied, occupied))

    detour = replay._checked_translation(None, None, geometry, .33, .90, 1.,
                                         800, 600, velocity_limits=DECLARED_SPEEDS)

    assert detour is not None and detour.command[0] == 0. and detour.command[1] < 0.
    assert replay.transition_clear((~occupied, occupied), geometry, .33, 1.45,
                                   (detour.command,))


def test_left_and_right_plans_reuse_the_same_depth_projection(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    empty = np.zeros_like(replay.RIGHT, dtype=bool)
    projections = []

    def observed(*_args):
        projections.append(True)
        return empty, empty

    monkeypatch.setattr(replay, "observe", observed)
    cache = replay.PlanCache()
    depth = {"map": empty, "ground": empty, "frame_id": 1000}
    for side in (-1., 1.):
        cache.translation(depth, geometry, .33, 1.4, side, 800,
                          velocity_limits=DECLARED_SPEEDS)
    assert len(projections) == 1
    cache.translation(dict(depth, frame_id=1100), geometry, .33, 1.4, 1., 800,
                      velocity_limits=DECLARED_SPEEDS)
    assert len(projections) == 2


def test_scanned_free_space_supports_a_bend_before_returning_to_the_route(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = np.isclose(replay.RIGHT, 0) & np.isclose(replay.FORWARD, .90)
    wall = (replay.RIGHT >= .65) & np.isclose(replay.FORWARD, 1.50)
    occupied = box | wall
    unmeasured = np.zeros_like(occupied)
    monkeypatch.setattr(replay, "observe", lambda *args: (unmeasured, occupied))

    args = (None, None, geometry, .33, .90, 1., 500, 600)
    prefix = replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS)
    assert prefix is not None and prefix.points[-1][0] < 1.35
    assert replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                       observed_grid=(unmeasured, unmeasured)) is None
    command = replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                          observed_grid=(~occupied, occupied))
    assert command is not None and 0 < np.hypot(*command.command)
    assert abs(command.command[0]) <= .6 and abs(command.command[1]) <= .4


def test_unmeasured_scanned_cells_do_not_change_the_detour_command():
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = (abs(replay.RIGHT) <= .05) & np.isclose(replay.FORWARD, .9)
    unknown = np.zeros_like(box)
    nearby = (replay.FORWARD <= .45) & (abs(replay.RIGHT) <= .5)
    args = (None, None, geometry, .33, .90, 1., 500, 600)
    narrow = replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                         observed_grid=(nearby, box))
    blind = replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                        observed_grid=(unknown, box))
    assert 0 < np.hypot(*narrow.command)
    assert blind.command == narrow.command and blind.points == narrow.points


def test_pose_aligned_scan_uses_only_obstacles_at_the_current_pose(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = np.isclose(replay.RIGHT, 0) & np.isclose(replay.FORWARD, .90)
    original = box | (np.isclose(replay.RIGHT, 0) & np.isclose(replay.FORWARD, .20))
    monkeypatch.setattr(replay, "observe", lambda *args: (~original, original))
    args = (None, None, geometry, .33, .90, 1., 500, 600)
    assert replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS) is None
    detour = replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                         observed_grid=(np.zeros_like(box), box))
    assert detour is not None and detour.command is not None
