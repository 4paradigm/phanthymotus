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
    config.use_obstacle_history = True
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


def test_retained_side_step_keeps_heading_when_the_turn_would_sweep_into_a_post():
    from plugins.navi import dwa, motion
    from plugins.navi.camera import HorizontalIntrinsics

    geometry = local_path.camera_square(HorizontalIntrinsics(640, 315., 337.),
                                        (480, 420., 243.), .20, half_length=.23, at_front=True)
    empty = np.zeros(local_path.SHAPE, dtype=bool)
    post = local_path.ObstacleSquares(np.array([[-.10, -.35]]), 0.)
    diagnostics = {}
    choice = dwa.select(
        route=((-0.23, 0.), (.38, -.35), (1.6, -.35)), grid=(~empty, empty),
        geometry=geometry, footprint=motion.FootprintSpec(0., .46, .20, .02, "camera_approx"),
        occupied=(post,), x_limit=(0., .30), y_limit=(-.40, .40), yaw_limit=(-1., 1.),
        measured=(0., 0., 0.), previous=(0., 0., 0.), floors=(0., 0., .20, .20),
        horizon=2.45, latency_s=.13, follow_published=True, max_candidates=8,
        diagnostics=diagnostics)

    assert diagnostics["rejected"]["footprint_collision"] >= 4
    assert choice is not None and choice.check.accepted and choice.trace.stopped
    assert choice.command[0] > 0. and choice.command[1] > 0. and choice.command[2] == 0.


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("slope,range_at_axis", [(0., .17), (1.2, .31)])
def test_close_wall_edge_only_constrains_the_current_front_corner_sweep(side, slope, range_at_axis):
    from plugins.navi.camera import HorizontalIntrinsics

    g = local_path.camera_square(HorizontalIntrinsics(640, 430., 320.), (480, 420., 240.), .2, at_front=True)
    frame = np.full((480, 640), 3., dtype=np.float32)
    edge = np.arange(161)
    if side > 0:
        edge = 639 - edge
    range_at_corner = range_at_axis - slope * .20
    cell = tuple(np.rint(np.array([range_at_corner, side * .20])
                         / local_path.RESOLUTION).astype(int) + local_path.START)
    ranges = range_at_axis / (1. + slope * side * (edge - 320.) / 430.)

    def evidence():
        _, occupied = local_path.DepthProjection(frame, None).grid_for(
            g, require_valid_depth=True, check_free_space=False)
        guard = local_path.front_corner_sweep_grid(frame, g, .22, occupied)
        assert not occupied[cell]
        return guard[cell]

    frame[:, edge[::40]] = ranges[::40]
    assert not evidence()
    frame[:, edge] = ranges
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
    config.use_obstacle_history = True
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


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("face_range", [.24, .50])
def test_missing_jamb_depth_keeps_the_checked_sweep_off_that_side(monkeypatch, side, face_range):
    from types import SimpleNamespace

    from plugins.navi import dwa, plugin
    from plugins.navi.camera import HorizontalIntrinsics

    card = plugin.NaviPlugin({}, None)
    config = card._config
    config.local_controller = "dwa"
    config.use_obstacle_history = False
    config.half_width_m = .20
    config.max_depth_age_ms = 1200
    config.depth_intrinsics = HorizontalIntrinsics(640, 315., 320.)
    card._depth_vertical = (480, 420., 240.)
    geometry = config.local_geometry = local_path.camera_square(
        config.depth_intrinsics, card._depth_vertical, .20, at_front=True, half_length=.23)
    close = np.full((480, 640), 3., dtype=np.float32)
    edge = slice(None, 100) if side < 0 else slice(-100, None)
    close[:, edge] = face_range
    missing = np.full_like(close, 3.)
    missing[:, edge] = np.nan
    card._set_depth_map(close, received_ms=1000)

    def contexts(stamp):
        card._set_depth_map(missing, received_ms=stamp)
        monkeypatch.setattr(plugin.time, "time", lambda: (stamp + 1) / 1000)
        _, observation, _ = card.observation(with_capture_timing=True)
        state = policy.State()
        state.dwa_projection = observation["grid_projection"]
        state.active_scan.observe_single(observation, geometry, config.depth_age_limit_ms, stamp + 1)
        inputs, _ = policy._dwa_motion_inputs(
            observation, config, state, geometry, (2., 0.), (0., 0., 0.))
        assert inputs is not None and not inputs.remembered
        guard = policy.DwaMotionGuard(**vars(inputs), trace=None, check=None)
        check = SimpleNamespace(motion_guard=guard, planning_guard=guard)
        scan = state.active_scan
        scene = card._registered_dwa_scene(
            check, observation, None, stamp + 1, stamp, geometry, ((0., 0., 0.),),
            scan=scan, sources=policy.dwa_scene_sources(scan, stamp + 1), anchor_changed=False)
        return [dwa.MotionContext((0., 0., 0.), (0., 0., 0.), 1., 0., geometry, inputs.footprint,
                                  occupied, (), None, None, 0.) for occupied in (inputs.observed, scene[2])]

    toward_jamb = (.074, -side * .023, -side * .2) if face_range < .4 else (.225, -side * .14, -side * .2)
    for context in contexts(1300):
        assert context.check(toward_jamb)[1].reason == "footprint_collision"
        assert context.check((.074, 0., 0.))[1].accepted
    for context in contexts(1000 + config.depth_age_limit_ms + config.watchdog_ms):
        assert context.check(toward_jamb)[1].accepted


def test_delayed_driver_depth_is_registered_before_the_robot_covered_the_gap(monkeypatch):
    from types import SimpleNamespace

    from plugins.navi import dwa, plugin
    from plugins.navi.camera import HorizontalIntrinsics

    card = plugin.NaviPlugin({}, None)
    config = card._config
    config.half_width_m = .20
    config.max_depth_age_ms = 1200
    config.depth_intrinsics = HorizontalIntrinsics(640, 315., 320.)
    card._depth_vertical = (480, 420., 240.)
    geometry = config.local_geometry = local_path.camera_square(
        config.depth_intrinsics, card._depth_vertical, .20, at_front=True, half_length=.23)
    frame = np.full((480, 640), 3., dtype=np.float32)
    frame[:400, 255:385] = .44
    rates = (.245, 0., 0.)
    now = 1800
    card._set_depth_map(frame, capture_ms=1000, received_ms=now)
    monkeypatch.setattr(plugin.time, "time", lambda: now / 1000)
    state = policy.State(commanded=(.075, 0., 0.))

    for provenance, expected in (("publication_clock", True), ("driver_receive", False)):
        card._depth_vendor["timestamp_source"] = provenance
        _, observation, _ = card.observation(with_capture_timing=True)
        scan = state.active_scan
        scan.observe_single(observation, geometry, config.depth_age_limit_ms, now)
        scan.poses.clear()
        scan.poses.extend(((1000, (0., 0., 0.), rates), (now, (.196, 0., 0.), rates)))
        state.dwa_projection = observation["grid_projection"]
        inputs, _ = policy._dwa_motion_inputs(observation, config, state, geometry, (2., 0.), rates)
        assert inputs is not None
        guard = policy.DwaMotionGuard(**vars(inputs), trace=None, check=None)
        check = SimpleNamespace(motion_guard=guard, planning_guard=guard)
        scene = card._registered_dwa_scene(
            check, observation, None, now, now, geometry, (rates,), scan=scan,
            sources=policy.dwa_scene_sources(scan, now), anchor_changed=False)
        for occupied in (inputs.observed, scene[2]):
            context = dwa.MotionContext(rates, state.commanded, 2.5, 0., geometry, inputs.footprint,
                                        occupied, (), None, None, 0.)
            assert context.check((.075, 0., 0.))[1].accepted is expected


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("face_range", [.36, .50])
def test_retained_route_keeps_passing_speed_after_losing_a_near_face(monkeypatch, side, face_range):
    from plugins.navi import plugin
    from plugins.navi.camera import HorizontalIntrinsics

    card = plugin.NaviPlugin({}, None)
    config = card._config
    config.half_width_m = .20
    config.vx_max = .30
    config.depth_intrinsics = HorizontalIntrinsics(640, 315., 320.)
    card._depth_vertical = (480, 420., 240.)
    geometry = config.local_geometry = local_path.camera_square(
        config.depth_intrinsics, card._depth_vertical, .20, at_front=True, half_length=.23)
    face = np.full((480, 640), 3., dtype=np.float32)
    edge = slice(None, 100) if side < 0 else slice(-100, None)
    face[:, edge] = face_range
    clear = np.full_like(face, 3.)
    card._set_depth_map(face, received_ms=1000)
    route = local_path.Detour(None, ((-.23, 0.), (2., 0.)), geometry)

    def forward_at(stamp):
        card._set_depth_map(clear, received_ms=stamp)
        monkeypatch.setattr(plugin.time, "time", lambda: stamp / 1000)
        _, observation, _ = card.observation(with_capture_timing=True)
        odom = dict(vx=0., vy=0., wz=0., received_ms=stamp, stamp_ms=stamp, source_sample_id=stamp)
        state = policy.State(motion_source="measured", dwa_odom=odom, dwa_projection=observation["grid_projection"])
        state.active_scan.update(observation, odom, geometry, receipt_only=True)
        proposal = policy._dwa_detour(observation, config, state, route, 3., 0., follow_published=True)
        assert proposal is not None and proposal.guard.check.accepted
        return proposal.command[0]

    assert 0. < forward_at(1300) <= .075 + 1e-9
    assert forward_at(1850) == pytest.approx(.30)


@pytest.mark.parametrize("restriction", [None, "blocked", "unobserved", "stale", "other_opening", "blocker_exit"])
def test_short_opening_tail_can_continue_forward_through_the_same_opening(restriction):
    from plugins.navi.track import Track
    from test_navi_dwa_integration import advance_case

    config, state, depth, odom = advance_case(body=True)
    config.use_obstacle_history = False
    scan, geometry = state.active_scan, config.scan_geometry
    scan.retire_motion()
    scan.use_obstacle_history = False
    odom["vx"] = .30
    state.commanded = (.30, 0., 0.)
    state.dwa_odom = odom
    state.dwa_admission_ms = 250
    state.escape_side = -1.
    state.goal_anchor = policy.DetourGoal(3., 0., generation=0)
    state.tracker.track = Track(np.array([3., 0., 0., 0.]), np.eye(4), range_known=True)
    state.tracker.track.record_range(3., (3., 0.), depth)
    assert scan.update(depth, odom, geometry, receipt_only=True) is None

    reference = local_path.Detour(None, ((0., 0.), (1., 0.)), geometry)
    current = policy._dwa_detour(depth, config, state, reference, 3., 0., follow_published=True)
    assert current is not None and current.guard.check.accepted
    assert current.guard.measured_target is not None
    exit_line = ((.08, -.40), (.12, .35))
    if restriction == "other_opening":
        exit_line = ((.08, .30), (.12, 1.05))
    committed = policy.DwaAcceptance(
        1, current.command, motion_guard=current.guard, route_points=((0., 0.), (.15, 0.)),
        passage_exit=exit_line, blocker_exit=exit_line if restriction == "blocker_exit" else None,
        locks_detour=True)
    state.committed_route = state.published_dwa_route = committed
    if restriction == "blocked":
        stamp, pose, free, _, *evidence = scan.views[-1]
        hits = np.isclose(local_path.FORWARD, .55) & np.isclose(local_path.RIGHT, 0.)
        scan.views[-1] = (stamp, pose, free & ~hits, hits, *evidence)
    if restriction == "unobserved":
        stamp, pose, free, hits, *evidence = scan.views[-1]
        scan.views[-1] = (stamp, pose, free & ~np.isclose(local_path.RIGHT, 0.), hits, *evidence)
    if restriction == "stale":
        depth = dict(depth, capture_ms=depth["control_ms"] - config.max_feedback_gap_ms)
    state.dwa_budget = None
    state.dwa_trace.clear()

    result = policy._follow_published_route(depth, config, state, geometry, 3., 0., foreground=True)

    assert state.committed_route is state.published_dwa_route is committed
    assert set(state.dwa_trace["rejected"]) == {"insufficient_room"}
    if restriction:
        assert result is None
        return
    assert result is not None and result[0] > 0. and result[1:3] == (0., 0.)
    assert state.escape_side == -1.
    proposal = state.dwa_proposal
    assert proposal.reference.passage_exit == exit_line
    assert proposal.guard.check.accepted and proposal.guard.trace.stopped
    assert proposal.guard.measured_target == current.guard.measured_target


def test_pursuit_after_blocker_exit_scales_candidates_before_the_braking_check(monkeypatch):
    from test_navi_dwa_integration import advance_case, observe_selector

    config, state, depth, odom = advance_case(body=True)
    scan, geometry = state.active_scan, config.scan_geometry
    scan.retire_motion()
    state.dwa_odom = odom
    state.dwa_admission_ms = 250
    state.commanded = (.12, 0., 0.)
    odom["vx"] = .12
    assert scan.update(depth, odom, geometry, receipt_only=True) is None
    reference = local_path.Detour(None, ((0., 0.), (1., 0.)), geometry)
    original = policy._dwa_detour(depth, config, state, reference, 3., 0., follow_published=True)
    room = .27
    committed = policy.DwaAcceptance(
        1, original.command, motion_guard=original.guard, route_points=((0., 0.), (room, 0.)),
        blocker_exit=((-0.1, -.4), (-0.1, .4)), locks_detour=True)
    state.committed_route = state.published_dwa_route = committed
    state.dwa_budget = None
    calls = observe_selector(monkeypatch)

    result = policy._follow_published_route(depth, config, state, geometry, 3., 0., foreground=True)

    assert result is not None and result[0] > 0.
    assert calls[-1][0]["x_limit"][1] == pytest.approx(room / original.guard.prediction_horizon_s)
    guard = state.dwa_proposal.guard
    assert guard.check.accepted and guard.trace.stopped
    assert guard.trace.body_travel_m + guard.trace.delay_travel_error <= room
