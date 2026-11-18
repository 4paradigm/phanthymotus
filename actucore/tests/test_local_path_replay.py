import numpy as np

from tools import replay_local_path as replay


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
