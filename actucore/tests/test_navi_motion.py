"""Physical motion checks in local forward/right metres."""
import math

import numpy as np
import pytest

from plugins.navi import local_path, policy


ZERO = (0., 0., 0.)
GEOMETRY = local_path.Geometry(None, None, np.eye(4), .20, .20, None)


def predict(command, *, measured=None, previous=None, hold=1.2, latency=0.,
            geometry=GEOMETRY, mode="body"):
    from plugins.navi.motion import predict_motion
    return predict_motion(measured if measured is not None else command,
                          previous if previous is not None else command, command,
                          hold_s=hold, latency_s=latency, acceleration=(.5, .5, 1.),
                          braking=(.45, .45, .8), geometry=geometry, mode=mode)


def check(trace, *, occupied=None, remembered=None, geometry=GEOMETRY, mode="body", **options):
    from plugins.navi.motion import FootprintSpec, evaluate_motion
    footprint = FootprintSpec(geometry.front, geometry.rear, .20, .02, mode)
    return evaluate_motion(trace, footprint=footprint, occupied=occupied, remembered=remembered,
                           supported_region=options.pop("supported_region", None), original_goal=None,
                           stop_distance_m=0., **options)


def test_narrow_corridor_accepts_actual_footprint():
    # Centres at ±.35, inner cell faces at ±.325.
    occupied = np.isclose(abs(local_path.RIGHT), .35)
    trace = predict((.15, 0., 0.))
    clear = check(trace, occupied=occupied)
    assert clear.accepted and .09 < clear.min_clearance_m <= .105 + 1e-10
    assert trace.stopped and trace.times[trace.moving_count-1] == pytest.approx(1.2)
    assert trace.velocities[-1] == pytest.approx(ZERO)
    assert trace.body_travel_m == pytest.approx(.205)
    # Inner face of the cell at .40 touches the front only during motion.
    occupied |= np.isclose(local_path.FORWARD, .40) & np.isclose(local_path.RIGHT, 0.)
    hit = check(trace, occupied=occupied)
    assert not hit.accepted and hit.first_hit is not None and hit.min_clearance_m == 0.


def test_small_turn_does_not_require_full_circle():
    walls = np.isclose(abs(local_path.RIGHT), .30)
    trace = predict((0., 0., .10))
    assert trace.body_travel_m == 0. and trace.body_poses[-1][2] < -.12
    assert check(trace, occupied=walls).accepted
    # Front-right corner crosses .225 as the body turns clockwise.
    corner = np.isclose(local_path.RIGHT, .25) & np.isclose(local_path.FORWARD, .20)
    cw = predict((0., 0., -.10))
    hit = check(cw, occupied=corner)
    assert not hit.accepted and hit.first_hit > 0


@pytest.mark.parametrize("sign", [-1., 1.])
def test_camera_offset_and_remembered_square_agree(sign):
    # Same .4 m body in a base-anchored grid and a front-camera-anchored grid.
    mount = np.eye(4)
    mount[0, 3] = .20
    body = local_path.Geometry(None, None, mount, .20, .20, None)
    approx = local_path.Geometry(None, None, np.eye(4), 0., .40, None)
    twist = (0., 0., sign * .10)
    body_trace = predict(twist, geometry=body)
    cam_trace = predict(twist, geometry=approx, mode="camera_approx")
    for index in (0, -1):
        bx, by, yaw = body_trace.camera_poses[index]
        assert cam_trace.camera_poses[index] == pytest.approx((bx - .20, by, yaw))
        assert cam_trace.body_poses[index] == pytest.approx((-.20, 0., yaw))
    assert sign * cam_trace.camera_poses[-1][1] < 0
    for r, accepted in ((.20, False), (.35, True)):
        cell = np.isclose(local_path.FORWARD, .15) & np.isclose(local_path.RIGHT, r)
        assert check(body_trace, occupied=cell, geometry=body).accepted is accepted
        aligned = np.isclose(local_path.FORWARD, -.05) & np.isclose(local_path.RIGHT, r)
        assert check(cam_trace, occupied=aligned, geometry=approx, mode="camera_approx").accepted is accepted
        memory = local_path.ObstacleSquares(np.array([[-.05, r]]), .37)
        assert check(cam_trace, remembered=memory, geometry=approx, mode="camera_approx").accepted is accepted


def test_margin_tracks_adopted_width_and_legacy_circle():
    config = policy.Config()
    for width, expected in ((.35, .035), (.20, .02)):
        policy.adopt_limits(config, {"footprint": {"half_width": width}})
        assert config.clearance_margin_m == pytest.approx(expected)
        _, radius = local_path._sweep_circle(GEOMETRY, width)
        assert radius == pytest.approx(math.hypot(.20, width) + expected)


def test_budget_is_body_arclength_and_stop_plane_is_footprint_support():
    from plugins.navi.motion import StopPlane, TravelBudget
    trace = predict((.15, 0., 0.))
    assert check(trace, terminal_boundary=TravelBudget(.30, .09)).accepted
    assert not check(trace, terminal_boundary=TravelBudget(.30, .10)).accepted
    assert check(trace, terminal_boundary=StopPlane((1., 0.), .44)).accepted
    assert not check(trace, terminal_boundary=StopPlane((1., 0.), .42)).accepted
    assert not check(trace, supported_region=((-1., -.50), (.42, .50))).accepted


def test_delivery_delay_stays_inside_hold_window():
    # Independent 1 ms Euler reference with 21 different command-arrival times.
    measured, previous, command = (.22, -.11, .36), (.30, -.18, .65), (.08, .20, -.55)
    trace = predict(command, measured=measured, previous=previous, hold=.8, latency=.2)
    times = np.asarray(trace.times)
    limit = math.ceil(times[-1] * 1000)
    accel, brake = (.5, .5, 1.), (.45, .45, .8)
    for delay in range(0, 201, 10):
        velocity = list(measured)
        f = r = yaw = 0.
        path = [(f, r, yaw)]
        for ms in range(limit):
            target = previous if ms < delay else command if ms < 800 else ZERO
            for axis, (v, requested) in enumerate(zip(velocity, target)):
                rate = brake[axis] if v * requested < 0. or abs(v) > abs(requested) else accel[axis]
                velocity[axis] += max(-rate / 1000, min(rate / 1000, requested-v))
            vx, vy, wz = velocity
            f += (vx * math.cos(yaw) + vy * math.sin(yaw)) / 1000
            r += (vx * math.sin(yaw) - vy * math.cos(yaw)) / 1000
            yaw -= wz / 1000
            path.append((f, r, yaw))
        assert max(map(abs, velocity)) < .002
        sampled = np.array(path)[np.rint(times * 1000).astype(int)]
        diff = sampled - np.asarray(trace.body_poses)
        assert np.linalg.norm(diff[:, :2], axis=1).max() <= trace.delay_position_error + trace.position_error + .002
        assert abs(diff[:, 2]).max() <= trace.delay_yaw_error + .002
        span = min(math.pi/2, trace.heading_span + trace.delay_yaw_error)
        dx, dy = np.asarray(trace.delay_axes).T
        linear = np.asarray(trace.interval_errors)[:, 0]
        assert np.all(abs(diff[1:, 0]) <= linear - (1. - math.sin(span)) * dy + .002)
        assert np.all(abs(diff[1:, 1]) <= linear - (1. - math.sin(span)) * dx + .002)
    assert trace.times[trace.moving_count - 1] == pytest.approx(.8)
    assert trace.stopped


@pytest.mark.parametrize("sign", [-1., 1.])
def test_front_camera_footprint_starts_at_lens_and_turns_about_chassis(sign):
    geometry = local_path.Geometry(None, None, np.eye(4), 0., .40, None)

    def at(f, r):
        return np.isclose(local_path.FORWARD, f) & np.isclose(local_path.RIGHT, r)

    rest = predict(ZERO, geometry=geometry, mode="camera_approx")
    # The chassis spans [-.40, 0] at rest. Placing that box at the pivot
    # a second time would miss its front and falsely hit the cell at -.50.
    for x, clear in ((0., False), (-.40, False), (.10, True), (-.50, True)):
        assert check(rest, occupied=at(x, 0.), geometry=geometry, mode="camera_approx").accepted is clear
    front_side = at(0., -sign * .25)
    assert check(rest, occupied=front_side, geometry=geometry, mode="camera_approx").accepted
    turning = predict((0., 0., sign*.10), geometry=geometry, mode="camera_approx")
    hit = check(turning, occupied=front_side, geometry=geometry, mode="camera_approx")
    assert not hit.accepted and hit.first_hit > 0


def test_remembered_square_orientation_decides_corner_contact():
    trace = predict(ZERO)
    centers = np.array([[.24, .24]])
    assert not check(trace, remembered=local_path.ObstacleSquares(centers, 0.)).accepted
    # A 45-degree square's near diagonal face lies outside the (.22,.22) corner.
    result = check(trace, remembered=local_path.ObstacleSquares(centers, math.pi/4))
    assert result.accepted
    assert result.min_clearance_m == pytest.approx(.02 * math.sqrt(2.) - .025)


def test_body_travel_counts_both_legs_of_a_reversal():
    trace = predict((-.15, 0., 0.), measured=(.15, 0., 0.), hold=2/3)
    # +.025 m while braking to zero, then -.0275 m during hold and -.025 m braking.
    assert trace.body_travel_m == pytest.approx(.0775)
    assert trace.body_poses[-1][0] == pytest.approx(-.0275)

def test_braking_distance_uses_integrated_motion():
    from plugins.navi.motion import TravelBudget
    trace = predict((.15, 0., 0.), measured=(.30, 0., 0.))
    assert trace.body_travel_m == pytest.approx(.23)
    assert check(trace, terminal_boundary=TravelBudget(.30, 0.)).accepted
    obstacle = np.isclose(local_path.FORWARD, .45) & np.isclose(local_path.RIGHT, 0.)
    assert not check(trace, occupied=obstacle, terminal_boundary=TravelBudget(.30, 0.)).accepted


def test_current_yaw_residual_encloses_translated_square():
    from plugins.navi.odom import PlanarPose
    source = PlanarPose(1000, ("odom",), .41, .0016, 0.)
    measured = PlanarPose(1100, source.frame, 0., 0., 0.)
    registration = policy.dwa_registration(source, measured, 1000, 1110, GEOMETRY, ((0., 0., -1.),))
    cells = local_path.ObstacleSquares(np.array([[0., .25]]), 0.)
    registered = registration.squares(cells)
    half = local_path.RESOLUTION / 2
    corners = np.array([(-half, -half), (-half, half), (half, -half), (half, half)])
    # At zero measured yaw the source grid origin is (.41, -.0016) in
    # forward/right coordinates. Rotate the entire translated world cell.
    nominal_center = np.array([.41, .25 - .0016])
    nominal_corners = nominal_center + corners
    assert registered.centers[0] == pytest.approx(nominal_center)
    for theta in np.linspace(-.01, .01, 21):
        c, s = math.cos(theta), math.sin(theta)
        rotation = np.array([[c, s], [-s, c]])
        exact_corners = nominal_corners @ rotation
        assert np.max(np.linalg.norm(exact_corners - nominal_corners, axis=1)) <= registered.alignment_error
        centre_error = np.linalg.norm(nominal_center @ rotation - nominal_center)
        assert centre_error <= registration.error(.25)
    theta = -.01
    c, s = math.cos(theta), math.sin(theta)
    exact_center = nominal_center @ np.array([[c, s], [-s, c]])
    exact = local_path.ObstacleSquares(exact_center[None, :], theta)
    assert exact_center == pytest.approx([.4124634, .2442877], abs=1e-6)
    assert local_path.rectangle_clearance([ZERO], exact, .45, .10, .22)[1] == 0
    assert local_path.rectangle_clearance([ZERO], registered, .45, .10, .22)[1] == 0


@pytest.mark.parametrize("mode", ["body", "camera_approx"])
@pytest.mark.parametrize("source", ["rolling", "world"])
@pytest.mark.parametrize("right, expected", [(.30, True), (.25, False)])
def test_turn_checks_original_subcell_obstacle_at_its_exposure(mode, source, right, expected):
    from plugins.navi.odom import PlanarPose

    camera = mode == "camera_approx"
    geometry = local_path.Geometry(None, None, np.eye(4), 0. if camera else .2,
                                   .4 if camera else .2, None)
    config = policy.Config(half_width_m=.20, scan_geometry=None if camera else geometry,
                           local_geometry=geometry)
    state = policy.State()
    scan = state.active_scan
    origin = PlanarPose(1000, ("odom",), 0., 0., 0.)
    current = PlanarPose(3000, origin.frame, 0., -.026, 0.)
    scan.pose_origin, scan.pose = origin, (0., .026, 0.)
    scan.poses.append((3000, scan.pose, 0.))  # exposure has left the interpolation buffer
    empty = np.zeros(local_path.SHAPE, bool)
    cells = np.isclose(local_path.FORWARD, -.05 if camera else .15) & np.isclose(local_path.RIGHT, right)
    if source == "rolling":
        scan.views.append((1000, (0., 0., 0.), empty, cells, empty))
    else:
        scan.obstacles.remember(scan._footprint_origin(origin, geometry), cells)
        scan.project_history(scan._footprint_origin(current, geometry))
    scan.views.append((3000, scan.pose, empty, empty, empty))  # blind side is unobserved
    scan.current_frame = 3000
    state.dwa_odom = dict(motion_pose=current)
    inputs, _ = policy._dwa_motion_inputs(dict(control_ms=3000), config, state, geometry, (2., 0.), ZERO)
    trace = predict((0., 0., -.10), geometry=geometry, mode=mode)
    result = check(trace, occupied=inputs.observed, remembered=inputs.remembered, geometry=geometry, mode=mode)
    assert result.accepted is expected
    if not expected:
        assert result.reason == ("footprint_collision" if source == "rolling" else "remembered_obstacle")


def test_rolling_source_age_and_confirmed_free_boundaries():
    from plugins.navi.odom import PlanarPose

    config = policy.Config(half_width_m=.20, scan_geometry=GEOMETRY, local_geometry=GEOMETRY)
    state = policy.State()
    scan = state.active_scan
    empty = np.zeros(local_path.SHAPE, bool)
    cell = np.isclose(local_path.FORWARD, .15) & np.isclose(local_path.RIGHT, .20)
    origin = PlanarPose(1000, ("odom",), 0., 0., 0.)
    current = PlanarPose(3500, origin.frame, 0., -.026, 0.)
    scan.pose_origin, scan.pose = origin, (0., .026, 0.)
    scan.current_frame = 3500
    scan.poses.append((3500, scan.pose, 0.))
    scan.views.append((1000, (0., 0., 0.), empty, cell, empty))
    scan.views.append((3500, scan.pose, empty, empty, empty))
    state.dwa_odom = dict(motion_pose=current)
    trace = predict((0., 0., -.10))

    def accepted(now):
        inputs, _ = policy._dwa_motion_inputs(dict(control_ms=now), config, state, GEOMETRY, (2., 0.), ZERO)
        return check(trace, occupied=inputs.observed).accepted

    assert not accepted(3500)
    assert accepted(3501)
    # Of the two rows of lateral support, only one supplies clear evidence.
    scan.views[-1] = (3500, scan.pose, cell, empty, cell)
    assert not accepted(3500)
    clear = np.ones_like(empty)
    scan.views[-1] = (3500, scan.pose, clear, empty, clear)
    assert accepted(3500)
    scan.views[-1] = (3500, scan.pose, clear, cell, clear)
    assert not accepted(3500)


@pytest.mark.parametrize("mode", ["body", "camera_approx"])
def test_packed_oriented_squares_match_native_complete_stop(mode):
    rng = np.random.default_rng(178)
    geometry = GEOMETRY if mode == "body" else local_path.Geometry(None, None, np.eye(4), 0., .40, None)
    for i in range(16):
        command = (.08 if i % 2 else 0., 0., (i - 8) / 25)
        trace = predict(command, geometry=geometry, mode=mode, measured=(.08, 0., .1), latency=.1)
        observed = tuple(local_path.ObstacleSquares(rng.uniform((.15, -.7), (1.2, .7), (8, 2)),
                                                   yaw, .001*i) for yaw in (0., .43, -.21))
        remembered = (local_path.ObstacleSquares(np.array([[.25, .24]]), .25),)
        layers = local_path.PreparedObstacleLayers(observed, remembered)
        options = dict(occupied=observed, remembered=remembered, geometry=geometry, mode=mode)
        assert check(trace, **options, prepared_layers=layers) == check(trace, **options)


def test_registered_camera_view_keeps_its_exposure_offset():
    from plugins.navi.odom import PlanarPose

    old = local_path.Geometry(None, None, np.eye(4), 0., .40, None)
    current = local_path.Geometry(None, None, np.eye(4), 0., .60, None)
    config = policy.Config(scan_geometry=None, local_geometry=current, half_width_m=.20)
    state = policy.State()
    scan = state.active_scan
    scan.keep_physical_views = True
    scan.receipt_only = True
    pose = PlanarPose(1000, ("odom",), 0., 0., 0.)
    state.dwa_odom = dict(motion_pose=pose)
    empty = np.zeros(local_path.SHAPE, bool)
    cell = np.isclose(local_path.FORWARD, 0.) & np.isclose(local_path.RIGHT, .30)
    scan.views.append(scan._source_view(1000, scan.pose, (empty, cell), dict(obstacle_capture_pose=pose), old))
    scan.current_frame = 1000
    scan.poses.append((1000, scan.pose, 0.))
    inputs, _ = policy._dwa_motion_inputs(dict(control_ms=1000), config, state, current, (2., 0.), ZERO)
    assert tuple(inputs.observed[0].centers[0]) == pytest.approx((-.10, .30))


@pytest.mark.parametrize("packed", [False, True])
def test_delayed_lateral_reversal_checks_contact_at_its_own_time(packed):
    trace = predict((.08, .05, 0.), measured=(.002, -.038, -.001),
                    previous=(.075, .016, -.056), hold=1.85, latency=.193)
    cell = local_path.ObstacleSquares(np.array([[.18, .272]]), 0., .022)
    layers = local_path.PreparedObstacleLayers(None, cell) if packed else None
    result = check(trace, remembered=cell, prepared_layers=layers)
    assert result.reason == "remembered_obstacle"
    assert 0. < trace.times[result.first_hit] < .193


def test_zero_duration_still_checks_initial_footprint():
    cell = local_path.ObstacleSquares(np.array([[0., 0.]]), 0.)
    assert check(predict(ZERO, hold=0.), remembered=cell).first_hit == 0


@pytest.mark.parametrize("packed", [False, True])
def test_longitudinal_delivery_error_preserves_side_clearance(packed):
    trace = predict((.3, 0., 0.), measured=ZERO, previous=ZERO, latency=.25)
    for right, accepted in ((.26, True), (.24, False)):
        side = local_path.ObstacleSquares(np.array([[.28, right]]), 0.)
        layers = local_path.PreparedObstacleLayers(None, side) if packed else None
        assert check(trace, remembered=side, prepared_layers=layers).accepted is accepted
    front = local_path.ObstacleSquares(np.array([[.46, 0.]]), 0.)
    assert not check(trace, occupied=front).accepted
