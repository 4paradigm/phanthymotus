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


def test_front_depth_can_find_a_geometric_detour_without_measuring_its_prefix():
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
    assert replay._checked_translation(frame, ground, geometry, .33, .90, 1., 500, 600,
                                       velocity_limits=DECLARED_SPEEDS) is None


def test_detour_limits_speed_to_the_measured_prefix_and_full_timeout(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        known_obstacle = (abs(right) <= .05) & np.isclose(forward, .9)
        # The next unmeasured front cell is centred at .35 m. The initial
        # front edge is .23 m, so it can advance less than .095 m in 1.15 s.
        free = (abs(right) <= .35 + 1e-9) & (forward > .20) & (forward <= .30 + 1e-9)
        return free, known_obstacle

    monkeypatch.setattr(replay, "observe", observed)
    command = replay._checked_translation(None, None, geometry, .33, .9, 1, 500, 600,
                                          velocity_limits=DECLARED_SPEEDS)
    assert command is not None
    vx, vy = command
    assert 0 < vx < .095 / 1.15
    assert vy == 0


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
    vx, vy = command
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


@pytest.mark.parametrize("start_occupied", [False, True])
def test_live_detour_stops_if_the_start_or_rejoin_is_blocked(monkeypatch, start_occupied):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        if start_occupied:
            occupied = np.isclose(right, 0) & (np.isclose(forward, .2) | np.isclose(forward, .9))
        else:
            occupied = (abs(right) <= .05) & (forward >= .85)
        return np.ones_like(occupied), occupied

    monkeypatch.setattr(replay, "observe", observed)
    assert replay._checked_translation(None, None, geometry, .33, .90, 1., 500, 600,
                                       velocity_limits=DECLARED_SPEEDS) is None


def test_scanned_free_space_supports_a_bend_before_returning_to_the_route(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = np.isclose(replay.RIGHT, 0) & np.isclose(replay.FORWARD, .90)
    wall = (replay.RIGHT >= .65) & np.isclose(replay.FORWARD, 1.50)
    occupied = box | wall
    unmeasured = np.zeros_like(occupied)
    monkeypatch.setattr(replay, "observe", lambda *args: (unmeasured, occupied))

    args = (None, None, geometry, .33, .90, 1., 500, 600)
    assert replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS) is None
    assert replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                       observed_grid=(unmeasured, unmeasured)) is None
    command = replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                          observed_grid=(~occupied, occupied))
    assert command is not None and 0 < np.hypot(*command)
    assert abs(command[0]) <= .6 and abs(command[1]) <= .4


def test_scanned_prefix_can_be_measured_before_the_passage_is_fully_seen(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = (abs(replay.RIGHT) <= .05) & np.isclose(replay.FORWARD, .9)
    unknown = np.zeros_like(box)
    monkeypatch.setattr(replay, "observe", lambda *args: (unknown, box))
    nearby = (replay.FORWARD <= .45) & (abs(replay.RIGHT) <= .5)
    command = replay._checked_translation(None, None, geometry, .33, .90, 1., 500, 600,
                                          velocity_limits=DECLARED_SPEEDS,
                                          observed_grid=(nearby, box))
    assert command is not None and 0 < np.hypot(*command)
    assert command[0] * 1.15 <= .475 - geometry.front
    assert abs(command[1]) * 1.15 <= .525 - .33


def test_pose_aligned_scan_does_not_borrow_free_cells_from_the_original_exposure(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = np.isclose(replay.RIGHT, 0) & np.isclose(replay.FORWARD, .90)
    monkeypatch.setattr(replay, "observe", lambda *args: (~box, box))
    args = (None, None, geometry, .33, .90, 1., 500, 600)
    assert replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS) is not None
    assert replay._checked_translation(*args, velocity_limits=DECLARED_SPEEDS,
                                       observed_grid=(np.zeros_like(box), box)) is None
