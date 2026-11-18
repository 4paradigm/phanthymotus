import numpy as np

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
    from plugins.navi.camera import HorizontalIntrinsics

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
