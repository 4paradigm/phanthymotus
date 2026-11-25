import numpy as np
import pytest

from plugins.navi import local_path, policy
from plugins.navi.active_scan import NearProjection
from test_navi_dwa_opening_integration import _case


@pytest.mark.parametrize(("room", "side_post", "lateral", "expected"), [
    (.12, False, False, "straight"),
    (.05, False, False, None),
    (.13, True, True, "diagonal"),
    (.13, True, False, None),
])
def test_blocked_route_retries_a_cell_of_advance_with_full_braking(room, side_post, lateral, expected):
    config, state, depth, geometry, _ = _case(True, "dwa")
    config.vx_max, config.floor_vx, config.dwa_horizon_s = .6, 0., 2.25
    config.wz_max = 0.
    config.use_lateral, config.vy_max = lateral, .03
    state.active_scan.dwa_failures = 2
    centers = [(geometry.front + room, 0.)]
    if side_post:
        centers.append((geometry.front + .07, config.half_width_m + .03))
    posts = local_path.ObstacleSquares(np.array(centers), 0.)
    state.active_scan.near_projection = NearProjection(squares=posts, stamp_ms=1000)
    route = local_path.Detour(None, ((0., 0.), (1., 0.)), geometry)

    proposal = policy._dwa_detour(depth, config, state, route, 3., 0., recovery=True)

    if expected is None:
        assert proposal is None
    else:
        assert proposal is not None and proposal.guard.check.accepted
        assert .025 <= proposal.choice.progress <= local_path.RESOLUTION
        assert proposal.command[0] > 0.
        assert (proposal.command[1] > 0.) == (expected == "diagonal")
        assert proposal.guard.trace.stopped


@pytest.mark.parametrize("budget", [1, 8, 24, 64])
def test_passage_retry_preserves_priority_families_and_budget(budget):
    from plugins.navi import dwa

    args = ((0., 0., 0.), (0., 0., 0.), ((.3, .08, .25), (.3, -.08, -.25)),
            ((0., .6), (-.4, .4), (-.5, .5)), (0., 0., .2, .2), (), budget)
    regular, missing = dwa._commands(*args)
    retry, retry_missing = dwa._commands(*args, recovery_speed=.022)
    assert retry_missing == missing
    assert len(retry) == len(set(retry)) <= budget
    assert retry[:min(8, budget)] == regular[:min(8, budget)]
    assert len(set(regular) - set(retry)) <= 3


@pytest.mark.parametrize("side", [-1, 1])
def test_close_wall_edge_only_constrains_the_current_front_corner_sweep(side):
    from plugins.navi.camera import HorizontalIntrinsics

    g = local_path.camera_square(HorizontalIntrinsics(640, 430., 320.), (480, 420., 240.), .2, at_front=True)
    frame = np.full((480, 640), 3., dtype=np.float32)
    edge = np.arange(161)
    if side > 0:
        edge = 639 - edge
    cell = tuple(np.rint(np.array([.17, side * .20]) / local_path.RESOLUTION).astype(int) + local_path.START)

    def evidence():
        _, occupied = local_path.DepthProjection(frame, None).grid_for(
            g, require_valid_depth=True, check_free_space=False)
        guard = local_path.front_corner_sweep_grid(frame, g, .22, occupied)
        assert not occupied[cell]
        return guard[cell]

    frame[:, edge[::40]] = .17
    assert not evidence()
    frame[:, edge] = .17
    assert evidence()
    frame[240, edge[20:35]] = np.nan
    assert not evidence()


def test_front_corner_wall_outside_supported_grid_cannot_index_or_wrap():
    from plugins.navi.camera import HorizontalIntrinsics

    g = local_path.camera_square(HorizontalIntrinsics(32, 500., 16.), (24, 500., 12.), .2, at_front=True)
    frame = np.full((24, 32), 4.)
    occupied = np.zeros(local_path.SHAPE, dtype=bool)
    occupied[local_path.START] = True
    np.testing.assert_array_equal(local_path.front_corner_sweep_grid(frame, g, .22, occupied), occupied)


def test_direct_prefix_cannot_turn_or_move_laterally_towards_the_reference():
    config, state, depth, geometry, _ = _case(True, "dwa")
    config.use_lateral, config.vy_max = True, .3
    route = local_path.Detour(None, ((0., 0.), (2., .2)), geometry)
    proposal = policy._dwa_detour(depth, config, state, route, 2., .1, branch="direct", speed_limit=.12)
    assert proposal is not None and proposal.guard.check.accepted and proposal.guard.trace.stopped
    assert 0 < proposal.command[0] <= .12
    assert proposal.command[1:] == (0., 0.)


@pytest.mark.parametrize(("forward", "right", "reason"), [
    (-.021, .134, "initial_remembered_overlap"),
    (.20, 0., "no_admissible_candidates"),
])
def test_doorway_distinguishes_initial_memory_overlap_from_later_sweep(forward, right, reason):
    config, state, depth, geometry, _ = _case(False, "dwa")
    assert geometry.front == 0.
    config.vx_max = config.floor_vx = .15
    config.vy_max = config.wz_max = 0.
    route = local_path.Detour(None, ((0., 0.), (1., 0.)), geometry)
    baseline = policy._dwa_detour(depth, config, state, route, 3., 0., recovery=True)
    assert baseline is not None and baseline.guard.check.accepted

    state.dwa_budget = None
    remembered = local_path.ObstacleSquares(np.array([[forward, right]]), .69)
    scan = state.active_scan
    scan.near_projection = NearProjection(squares=remembered, stamp_ms=1000)
    assert policy._dwa_detour(depth, config, state, route, 3., 0., recovery=True) is None
    assert state.dwa_trace["rejected"]["remembered_obstacle"] > 0
    assert state.dwa_trace["fallback_reason"] == reason
    assert scan.near_projection.squares is remembered


def test_fresh_close_jamb_constrains_motion_into_the_opposite_blind_side():
    from plugins.navi import dwa
    from plugins.navi.camera import HorizontalIntrinsics
    from plugins.navi.motion import FootprintSpec

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315., 337.),
                                        (480, 420., 243.), .20, at_front=True, half_length=.23)
    frame = np.full((480, 640), 6., dtype=np.float32)
    # The near right jamb is visible. The left shoulder points out of view
    # as the robot turns, so apparent space there cannot justify a side step.
    frame[:, 425:] = .15
    grid = np.zeros(local_path.SHAPE, dtype=bool)
    cell = local_path.START[0] + 3, local_path.START[1] + 3
    grid[cell] = True
    guarded = local_path.front_corner_sweep_grid(frame, geometry, .22, grid, guard_blind_sides=True)
    context = dwa.MotionContext((0., 0., 0.), (0., 0., 0.), 1., 0., geometry,
                                FootprintSpec(0., .46, .20, .02, "camera_approx"),
                                (local_path.occupied_squares(guarded),), (), None, None, 0.)
    assert context.check((0., .2, -.2))[1].reason == "footprint_collision"
    assert context.check((0., 0., 0.))[1].accepted
    assert not grid[local_path.START[0], local_path.START[1] - 5]
    assert guarded[local_path.START[0], local_path.START[1] - 5]
