import numpy as np
import pytest

from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.depth import ground_mask
from plugins.navi import local_path as replay

DECLARED_SPEEDS = ((0., .6), (0., .4))


@pytest.mark.parametrize("surface,clear", [("floor", True), ("missing", False), ("foreground", False)])
def test_near_ground_returns_clear_only_probes_with_valid_floor_evidence(surface, clear):
    horizontal = HorizontalIntrinsics(640, 420, 319.5)
    vertical = (480, 420, 239.5)
    geometry = replay.camera_square(horizontal, vertical, .20)
    tangent = (np.arange(480) - 239.5) / 420
    floor = .42 / np.maximum(tangent, .001)
    frame = np.broadcast_to(floor[:, None], (480, 640)).copy()
    if surface != "floor":
        frame[371:378, 310:330] = np.nan if surface == "missing" else 1.055
    ground = ground_mask(frame, horizontal, vertical)
    assert ground is not None

    free, occupied = replay.observe(frame, geometry, replay.RIGHT, replay.FORWARD,
                                    replay.RESOLUTION, ground)
    center_col = replay.START[1]
    row = replay.START[0] + round(1. / replay.RESOLUTION)
    assert bool(free[row, center_col]) is clear
    assert not occupied[row, center_col]
    # The lowest probe at 0.55 m remains below the image.
    blind = replay.START[0] + round(.55 / replay.RESOLUTION)
    assert not free[blind, center_col] and not occupied[blind, center_col]


@pytest.mark.parametrize("height,calibrated,blocked", [
    (.022, False, False), (.043, False, False), (.08, False, True), (.043, True, True),
])
def test_camera_grid_distinguishes_near_floor_depth_noise_from_a_low_obstacle(height, calibrated, blocked):
    horizontal = HorizontalIntrinsics(640, 315, 337)
    vertical = (480, 420, 243)
    geometry = replay.camera_square(horizontal, vertical, .20)
    if calibrated:
        geometry = replay.body_square(geometry, {"body_from_optical": geometry.base_from_optical})
    tangent = (np.arange(480) - 243) / 420
    floor = np.minimum(.42 / np.maximum(tangent - .02, .001), 4.)
    frame = np.broadcast_to(floor[:, None], (480, 640)).copy()
    patch = np.s_[411:423, 320:345]
    frame[patch] *= (.42 - height) / .42
    ground = ground_mask(frame, horizontal, vertical)
    assert ground is not None and not ground[patch].any()

    free, occupied = replay.observe(frame, geometry, replay.RIGHT, replay.FORWARD,
                                    replay.RESOLUTION, ground)
    center = (abs(replay.RIGHT) <= .05) & (replay.FORWARD > .75) & (replay.FORWARD < 1.12)
    assert bool((occupied & center).any()) is blocked
    if not blocked:
        assert (~free & ~occupied & center).any()


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


def test_search_prefers_a_clearer_path_when_distance_costs_are_close():
    allowed = np.ones((5, 7), dtype=bool)
    allowed[2, 3] = False
    clearance_cost = np.zeros_like(allowed, dtype=float)
    clearance_cost[:2, 1:6] = .2
    path = replay.shortest_path(allowed, (2, 0), (2, 6), cell_cost=clearance_cost)
    assert (3, 3) in path
    assert not any(row < 2 for row, _ in path)


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


@pytest.mark.parametrize("case", ["open_side", "wall", "unobserved"])
@pytest.mark.parametrize("execute", [False, True])
def test_camera_route_can_sweep_unknown_cells_towards_an_observed_waypoint(case, execute):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .20)
    occupied = (abs(replay.RIGHT) <= .10) & np.isclose(replay.FORWARD, .90)
    if case == "wall":
        occupied |= np.isclose(replay.FORWARD, .30)
    free = ((replay.FORWARD >= .30) & (replay.FORWARD <= 1.20)
            & (replay.RIGHT >= .45) & (replay.RIGHT <= .80) & ~occupied)
    if case == "unobserved":
        free[:] = False
    route = replay._checked_translation(None, None, geometry, .25, .90, 1., 500, 600,
                                        velocity_limits=DECLARED_SPEEDS if execute else None,
                                        observed_grid=(free, occupied),
                                        body_half_width=.20, allow_unknown=True)
    if case != "open_side":
        assert route is None
        return
    assert route is not None
    if execute:
        assert route.command and replay.transition_clear((free, occupied), geometry, .25, 1.15, (route.command,))
    forward, right = route.points[-1]
    assert .30 <= forward <= 1.20 and .45 <= right <= .80
    allowed = replay.allowed_centers(replay._route_grid((free, occupied)), replay.START,
                                     (4, 5, 4), allow_unknown=True)
    assert all(allowed[
        replay.START[0] + round(f / .05), replay.START[1] + round(r / .05)] for f, r in route.points)


@pytest.mark.parametrize("initial_hit", [False, True])
def test_current_discrete_envelope_can_contain_unknown_but_not_occupied(initial_hit):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    occupied = (abs(replay.RIGHT) <= .10) & np.isclose(replay.FORWARD, .90)
    occupied[replay.START[0] - 5, replay.START[1]] = initial_hit
    free = (replay.FORWARD > -.249) & ~occupied
    route = replay._checked_translation(None, None, geometry, .28, .90, 1., 500, 600,
                                        velocity_limits=None, observed_grid=(free, occupied),
                                        body_half_width=.23)
    assert (route is not None) is not initial_hit


@pytest.mark.parametrize("side", [-1., 1.])
def test_partial_route_selects_the_reachable_point_toward_the_original_target(side):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    wall = np.isclose(replay.FORWARD, 1.15)
    route = replay._checked_translation(None, None, geometry, .28, 1.15, side, 500, 600,
                                        velocity_limits=None, observed_grid=(~wall, wall),
                                        body_half_width=.23, target=(2.4, side * .95, .60, .05),
                                        closest=True)
    assert route is not None
    forward, right = route.points[-1]
    assert forward == pytest.approx(.85)
    assert right == pytest.approx(side * .95)


def test_partial_endpoint_has_a_prefix_above_the_declared_velocity_floor():
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337), (480, 420, 243), .23)
    wall = np.isclose(replay.FORWARD, .35)
    route = replay._checked_translation(None, None, geometry, .28, .35, 1., 500, 600,
                                        velocity_limits=((0., .35), (.30, .40)), allow_forward=False,
                                        observed_grid=(~wall, wall), body_half_width=.23,
                                        target=(1.8, .35, .6, .05), closest=True)
    assert route is not None and route.command is not None
    assert route.command[0] == 0. and -.4 <= route.command[1] <= -.30
    assert route.points[1] == pytest.approx((0., .35))


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


def test_parking_ring_is_considered_even_beyond_the_obstacle_rejoin_depth():
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .15)
    # The only opening near the object is on its left. The first box is
    # much farther back than the parking ring.
    start_box = (abs(replay.RIGHT) <= .10) & np.isclose(replay.FORWARD, .80)
    front_of_target = (replay.RIGHT >= -.15) & (replay.RIGHT <= .30) & (
        (replay.FORWARD >= 1.92) & (replay.FORWARD <= 2.36))
    occupied = start_box | front_of_target
    target = (2.60, 0., .60, .05)
    detour = replay._checked_translation(
        None, None, geometry, .20, .80, -1., 800, velocity_limits=None,
        target=target, projected_grid=(~occupied, occupied), body_half_width=.15,
        allow_unknown=True)

    assert detour is not None
    forward, right = detour.points[-1]
    assert right < -.30
    assert .55 <= np.hypot(2.60 - forward, right) <= .65


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


@pytest.mark.parametrize("case,occupied_cells", [
    ("weak_speckle", 0),
    ("low_single", 0),
    ("low_pair", 0),
    ("low_strip", 5),
    ("thin_leg", 1),
    ("near_obstacle", 1),
])
def test_depth_grid_drops_small_noise_and_low_patches(case, occupied_cells):
    horizontal = HorizontalIntrinsics(640, 420, 320)
    vertical = (480, 420, 240)
    geometry = replay.camera_square(horizontal, vertical, .20)
    geometry = replay.body_square(geometry, {"body_from_optical": geometry.base_from_optical})
    tangent = (np.arange(480) - 240) / 420
    frame = np.broadcast_to(.42 / np.maximum(tangent[:, None], .09), (480, 640)).copy()
    if case == "weak_speckle":
        frame[380, 319:322] = .90
    elif case == "low_single":
        frame[418:422, 317:324] = .90
    elif case == "low_pair":
        frame[418:422, 320:342] = .90
    elif case == "low_strip":
        frame[418:422, 267:374] = .90
    elif case == "thin_leg":
        frame[315:395, 319:321] = .90
    else:
        frame[415, 319:322] = .30
    ground = ground_mask(frame, horizontal, vertical)
    free, occupied = replay.observe(frame, geometry, replay.RIGHT, replay.FORWARD,
                                    replay.RESOLUTION, ground)
    assert np.count_nonzero(occupied) == occupied_cells
    _, fast = replay.observe(frame, geometry, replay.RIGHT, replay.FORWARD,
                            replay.RESOLUTION, ground, check_free_space=False)
    assert np.array_equal(fast, occupied)
    if occupied_cells == 0:
        assert not free[replay.START[0] + 18, replay.START[1]]


def test_depth_without_floor_can_plan_around_a_visible_post():
    horizontal = HorizontalIntrinsics(640, 420, 320)
    geometry = replay.camera_square(horizontal, (480, 420, 240), .20)
    tangent = (np.arange(480) - 240) / 420
    depth = np.broadcast_to(.42 / np.maximum(tangent[:, None], .09), (480, 640)).copy()
    depth[240:375, 306:335] = 1.
    free, occupied = replay.observe(depth, geometry, replay.RIGHT, replay.FORWARD, replay.RESOLUTION, None)
    post = (abs(replay.RIGHT) <= .05) & np.isclose(replay.FORWARD, 1.)
    assert occupied[post].any()
    assert not occupied[replay.FORWARD < .90].any()
    assert not free.any()
    destination = dict(map=depth, frame_id=1000, allow_unknown=True, body_half_width=.20)
    plan = replay.PlanCache().route(destination, geometry, .25, 1., 1., 500, 600)
    assert plan is not None
    assert max(right for _, right in plan.points) >= .30


def test_fresh_frames_without_floor_are_usable_for_active_scan():
    from plugins.navi.active_scan import ActiveScan

    horizontal = HorizontalIntrinsics(640, 420, 320)
    geometry = replay.camera_square(horizontal, (480, 420, 240), .20)
    frame = np.full((480, 640), np.nan)
    frame[200:345, 303:337] = 1.10
    look = ActiveScan()
    odom = dict(vx=0., vy=0., wz=0., received_ms=1000)
    assert look.update(dict(map=frame, frame_id=1000), odom, geometry, receipt_only=True) is None
    assert look.current_frame == 1000
    assert look.grid()[1].any()
    assert look.begin((1.,))
    assert look.update(dict(map=np.full_like(frame, np.nan), frame_id=1100),
                       dict(odom, received_ms=1100), geometry, receipt_only=True) is not None
    assert not look.current_frame


@pytest.mark.parametrize("case", ["floor_patch", "unknown_context", "low_edge", "tilted_edge", "thin_rod"])
def test_camera_projection_drops_small_low_patches_but_keeps_the_taller_rod(case):
    horizontal = HorizontalIntrinsics(640, 480, 320)
    geometry = replay.camera_square(horizontal, (480, 480, 240), .275, at_front=True)
    tangent = (np.arange(480) - 240) / 480
    angle = .2 if case == "tilted_edge" else 0.
    sn, cs = np.sin(angle), np.cos(angle)
    coefficient = sn + cs * tangent
    floor = .42 / np.where(coefficient > .1, coefficient, np.nan)
    frame = np.broadcast_to(floor[:, None], (480, 640)).copy()
    ground = np.isfinite(frame)
    if case in ("floor_patch", "unknown_context"):
        rows, cols = slice(316, 318), slice(316, 325)
        if case == "unknown_context":
            frame[280:351, cols] = np.nan
            ground[280:351, cols] = False
        frame[rows, cols] = (.42 - .056) / tangent[rows, None]
    else:
        obstacle_height = .16 if case == "thin_rod" else .058
        depth = 2.30 / (cs - sn * tangent)
        height = .42 - coefficient * depth
        visible = np.flatnonzero((height > .004) & (height <= obstacle_height))
        rows, cols = slice(visible[0], visible[-1] + 1), slice(320, 321 if case == "thin_rod" else 322)
        frame[rows, cols] = depth[rows, None]
    ground[rows, cols] = False

    args = (frame, geometry, replay.RIGHT, replay.FORWARD, replay.RESOLUTION, ground)
    free, occupied = replay.observe(*args)
    _, fast = replay.observe(*args, check_free_space=False)
    assert np.array_equal(fast, occupied)
    assert bool(occupied.any()) is (case == "thin_rod")
    if case == "floor_patch":
        assert free[replay.START[0] + 46, replay.START[1]]


@pytest.mark.parametrize("heights,remaining", [
    ((.09,), 0), ((.09, .09, .09), 0), ((.09,) * 4, 4),
    ((.09, .10, .09), 3), (None, 3),
])
def test_low_components_use_diagonal_connectivity_and_strict_height_limit(heights, remaining):
    n = len(heights) if heights is not None else 3
    horizontal = HorizontalIntrinsics(640, 480, 320)
    geometry = replay.camera_square(horizontal, (480, 480, 240), .275, at_front=True)
    depth = np.repeat(1.2 + np.arange(n) * .05, 8)
    right = np.repeat(.10 + np.arange(n) * .05, 8)
    raised = np.ones(len(depth), dtype=bool)
    kwargs = {"heights": np.repeat(heights, 8) if heights is not None else None}

    free, occupied = replay._project_observed(
        geometry, replay.RIGHT, replay.FORWARD, replay.RESOLUTION,
        right, np.zeros_like(depth), depth, raised, check_free_space=True, **kwargs)
    _, fast = replay._project_observed(
        geometry, replay.RIGHT, replay.FORWARD, replay.RESOLUTION,
        right, np.zeros_like(depth), depth, raised, check_free_space=False, **kwargs)
    assert occupied.sum() == remaining
    assert np.array_equal(fast, occupied)
    assert free.sum() == n - remaining
