import numpy as np
import pytest

from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi import local_path as replay


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


def test_front_only_plan_uses_square_footprint_and_passes_unknown_cells(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    assert geometry.front == geometry.rear == .23

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        known_obstacle = (abs(right) <= .05) & (abs(forward - .9) < .06)
        return np.zeros_like(known_obstacle), known_obstacle

    monkeypatch.setattr(replay, "observe", observed)
    result = replay._checked_translation(None, None, geometry, .33, .9, 1, 500)
    assert result is not None
    vx, vy = result
    assert (vx > 0 or vy < 0) and np.hypot(vx, vy) <= .10


@pytest.mark.parametrize("side", [-1., 1.])
def test_live_detour_tries_a_wider_bypass_and_later_rejoin(monkeypatch, side):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)

    def observed(_frame, _geometry, right, forward, _resolution, _ground):
        box = (abs(right) <= .05) & (forward >= .85) & (forward <= 1.26)
        side_obstacle = np.isclose(right, side * .55) & np.isclose(forward, .90)
        return np.ones_like(box), box | side_obstacle

    monkeypatch.setattr(replay, "observe", observed)
    command = replay._checked_translation(None, None, geometry, .33, .90, side, 500, 600)
    assert command is not None
    vx, vy = command
    assert vx >= 0 and vy * side < 0
    assert np.hypot(vx, vy) <= .10


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
    assert replay._checked_translation(None, None, geometry, .33, .90, 1., 500, 600) is None


def test_scanned_free_space_supports_a_bend_before_returning_to_the_route(monkeypatch):
    geometry = replay.camera_square(HorizontalIntrinsics(640, 315, 337),
                                    (480, 420, 243), .23)
    box = np.isclose(replay.RIGHT, 0) & np.isclose(replay.FORWARD, .90)
    wall = (replay.RIGHT >= .65) & np.isclose(replay.FORWARD, 1.50)
    occupied = box | wall
    unmeasured = np.zeros_like(occupied)
    monkeypatch.setattr(replay, "observe", lambda *args: (unmeasured, occupied))

    args = (None, None, geometry, .33, .90, 1., 500, 600)
    assert replay._checked_translation(*args) is None
    assert replay._checked_translation(*args, observed_grid=(unmeasured, unmeasured)) is None
    command = replay._checked_translation(*args, observed_grid=(~occupied, unmeasured))
    assert command is not None and 0 < np.hypot(*command) <= .10
