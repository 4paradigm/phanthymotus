import math

import numpy as np
import pytest

from plugins.navi import active_scan as scan
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.depth import ground_mask
from plugins.navi.local_path import Geometry, body_square, camera_square
from plugins.navi.odom import PlanarPose


def _depth(stamp, occupied=None):
    free = np.ones(scan.SHAPE, dtype=bool)
    if occupied is None:
        occupied = np.zeros_like(free)
    return {"map": free, "ground": free, "capture_ms": stamp, "frame_id": stamp}


def _odom(stamp, wz=0):
    return {"vx": 0.0, "vy": 0.0, "wz": wz, "stamp_ms": stamp}


def _setup(monkeypatch):
    blocked = np.zeros(scan.SHAPE, dtype=bool)
    visible = np.ones_like(blocked)
    monkeypatch.setattr(scan, "observe",
                        lambda *args: (visible.copy(), blocked.copy()))
    lookout = scan.ActiveScan()
    geometry = Geometry(None, None, np.eye(4), .23, .23, None)
    return lookout, geometry, visible, blocked


def _tick(lookout, geometry, stamp, wz=0, capture=True):
    depth = _depth(stamp if capture is True else capture)
    lookout.update(depth, _odom(stamp, wz), geometry)
    return lookout.command(geometry, .33, .25, 600)


@pytest.mark.parametrize("command_only", [False, True])
def test_receipt_only_projects_only_the_latest_requested_frame(monkeypatch, command_only):
    geometry = camera_square(HorizontalIntrinsics(64, 50, 32), (48, 50, 24), .23)
    tangent = (np.arange(48) - 24) / 50
    image = np.broadcast_to(np.minimum(.42 / np.maximum(tangent, .001), 4.)[:, None],
                            (48, 64)).copy()
    ground = image < 4.
    newer = image.copy()
    newer[:25, 20:45] = 1.
    expected = scan.observe(newer, geometry, scan.RIGHT, scan.FORWARD, scan.RESOLUTION, ground)
    observe = scan.observe
    projected = []

    def project(frame, *args):
        projected.append(frame)
        return observe(frame, *args)

    monkeypatch.setattr(scan, "observe", project)
    look = scan.ActiveScan()
    for stamp, frame in ((1000, image), (1100, newer)):
        depth = {"map": frame, "ground": ground, "frame_id": stamp}
        if command_only:
            assert look.observe_single(depth, geometry, 500, stamp) is None
        else:
            odom = {**_odom(stamp), "received_ms": stamp}
            assert look.update(depth, odom, geometry, receipt_only=True) is None
    assert look.current_frame == 1100
    assert not projected
    for actual, wanted in zip(look.grid(), expected):
        np.testing.assert_array_equal(actual, wanted)
    look.grid()
    assert len(projected) == 1 and projected[0] is newer


def test_active_scan_turns_right_and_waits_for_post_turn_depth(monkeypatch):
    lookout, geometry, _, blocked = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    assert _tick(lookout, geometry, 1050).yaw == 0
    assert _tick(lookout, geometry, 1150).yaw < 0
    for stamp in range(1250, 2550, 100):
        result = _tick(lookout, geometry, stamp, -.25)
    assert result.yaw == 0
    assert math.degrees(lookout.pose[2]) == pytest.approx(20., abs=2.)
    assert _tick(lookout, geometry, 2500, capture=2450).yaw == 0
    assert _tick(lookout, geometry, 2650, capture=2450).yaw == 0
    assert not lookout.ready
    blocked[30, 38] = True
    assert _tick(lookout, geometry, 2700).yaw == 0
    assert _tick(lookout, geometry, 2750).yaw > 0
    blocked[30, 38] = False
    for stamp in range(2850, 4150, 100):
        result = _tick(lookout, geometry, stamp, .25)
    assert result.yaw == 0
    _tick(lookout, geometry, 4150)
    _tick(lookout, geometry, 4300)
    assert lookout.ready
    _, occupied = lookout.grid()
    assert occupied.any()
    assert not occupied[30, 38]  # the side observation was reprojected
    assert lookout.try_other_side()
    assert _tick(lookout, geometry, 4400).yaw == 0
    assert _tick(lookout, geometry, 4500).yaw > 0


def test_active_scan_holds_an_occupied_sweep(monkeypatch):
    lookout, geometry, _, blocked = _setup(monkeypatch)
    blocked[14, 28] = True
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    for stamp in (1050, 1200, 1250, 1350):
        assert _tick(lookout, geometry, stamp).yaw == 0
    assert not lookout.active
    assert not lookout.begin((1., -1.))


@pytest.mark.parametrize("start_delay_ms", [0, 250])
def test_pose_only_scan_converges_with_delayed_pose_and_chassis(monkeypatch, start_delay_ms):
    lookout, geometry, _, _ = _setup(monkeypatch)
    history = {stamp: 0. for stamp in range(99_600, 100_000, 100)}
    yaw, rate, turn_start = 0., 0., None
    for stamp in range(100_000, 115_000, 100):
        if turn_start is not None:
            moving_ms = min(100, max(0, stamp - turn_start - start_delay_ms))
            yaw += rate * moving_ms / 1000
        history[stamp] = yaw
        measured = stamp - 300
        pose = PlanarPose(measured, ("odom", "body"), 0., 0., history[measured])
        rate_at_measurement = (history[measured] - history[measured - 100]) / .1
        lookout.update(_depth(stamp), {
            "received_ms": stamp, "source_sample_id": measured, "motion_pose": pose,
            "pose_feedback": (0., 0., rate_at_measurement), "pose_feedback_since_ms": measured - 100,
        }, geometry, receipt_only=True)
        if stamp == 100_000:
            assert lookout.begin((-1., 1.))
        rate = lookout.command(geometry, .31, .83, 300, min_yaw_rate=.33, now_ms=stamp).yaw
        turn_start = (turn_start if turn_start is not None else stamp) if rate else None
        assert abs(yaw) <= math.radians(22.)
        if lookout.ready:
            break

    assert lookout.ready
    assert math.degrees(yaw) == pytest.approx(20., abs=2.)


def test_active_scan_stops_with_stale_odom_and_cannot_restart_on_same_obstacle(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    _tick(lookout, geometry, 1050)
    assert _tick(lookout, geometry, 1200).yaw < 0
    lookout.update(_depth(1250), None, geometry)
    assert not lookout.active
    assert not lookout.begin((1., -1.))


@pytest.mark.parametrize("lose_depth, yaw_limit", [(False, .83), (False, .60), (True, .83)])
def test_command_only_scan_waits_for_a_new_settled_view(monkeypatch, lose_depth, yaw_limit):
    look, geometry, _, _ = _setup(monkeypatch)
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((1., -1.))
    rates = []
    first_turn = None
    for stamp in range(1100, 7400, 100):
        depth = None if lose_depth and first_turn is not None else _depth(stamp)
        look.observe_single(depth, geometry, 800, stamp)
        command = look.command(geometry, .31, yaw_limit, 300, 800, min_yaw_rate=.33)
        if command.yaw and first_turn is None:
            first_turn = stamp
        rates.append(command.yaw)
        if lose_depth and first_turn is not None and depth is None:
            assert command.yaw == 0 and not look.active
            return
        if look.ready:
            break
    assert first_turn is not None
    nonzero = [rate for rate in rates if rate]
    assert all(-yaw_limit <= rate <= -.33 for rate in nonzero)
    assert math.degrees(-sum(nonzero) * .1) == pytest.approx(20., abs=2.)
    assert look.ready and look.current_frame > first_turn + scan.RECEIPT_SETTLE_MS
    assert len(look.views) == 1 and not look.poses


@pytest.mark.parametrize("blocked", [False, True])
def test_command_only_scan_checks_obstacles_with_unseen_rear(monkeypatch, blocked):
    look, geometry, free, occupied = _setup(monkeypatch)
    free[:] = scan.FORWARD > 0.
    row, col = scan.START
    occupied[row + 4, col + 6] = blocked
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((1.,))
    for stamp in range(1100, 1700, 100):
        look.observe_single(_depth(stamp), geometry, 800, stamp)
        result = look.command(geometry, .31, .83, 300, 800, min_yaw_rate=.33)
        if result.yaw:
            assert not blocked and result.yaw < 0.
            break
    assert look.issued_yaw is (not blocked)
    assert look.active is (not blocked)


def test_command_only_scan_uses_capture_time_for_post_stop_frames(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)
    look.observe_single(_depth(1000), geometry, 800, 1000)
    assert look.begin((1.,))
    look.phase, look.settle_ms = "settle", 1000
    look.timed_phase_s = scan.RECEIPT_SETTLE_MS / 1000
    for stamp in (4501, 4600, 4700):
        look.observe_single(dict(_depth(stamp), capture_ms=4400), geometry, 800, stamp)
        assert look.command(geometry, .31, .83, 300).yaw == 0.
    assert not look.ready and look.settle_frames == 0
    for stamp in (4800, 4900, 5000):
        look.observe_single(_depth(stamp), geometry, 800, stamp)
        assert look.command(geometry, .31, .83, 300).yaw == 0.
    assert look.ready


def test_scan_rejects_unexpected_yaw_rate(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((-1., 1.))
    _tick(lookout, geometry, 1050)
    assert _tick(lookout, geometry, 1200).yaw > 0
    assert _tick(lookout, geometry, 1250, wz=.55).yaw == 0
    assert lookout.locked


def test_scan_time_limit_applies_even_with_fresh_frames_and_no_angular_progress(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((-1., 1.))
    _tick(lookout, geometry, 1050)
    for stamp in range(1200, 9200, 200):
        assert _tick(lookout, geometry, stamp).yaw > 0
    assert _tick(lookout, geometry, 9200).yaw == 0
    assert lookout.locked


def test_receipt_only_scan_replans_after_measured_translation_has_settled(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp, vx=0., vy=0., yaw=0.):
        look.update(_depth(stamp), {"vx": vx, "vy": vy, "wz": yaw, "received_ms": stamp},
                    geometry, receipt_only=True, max_depth_age_ms=800)
        return look.command(geometry, .33, 1., 600, 800)

    tick(1000)
    assert look.begin((1., -1.))
    assert tick(1050).yaw == 0.
    assert tick(1150).yaw < 0.
    assert tick(1250, vx=.113, vy=-.014, yaw=-.204).yaw == 0.
    assert look.active and not look.locked
    assert tick(1400, vx=-.006, vy=-.035).yaw == 0.
    assert look.active
    assert tick(1550, vx=-.006, vy=-.035).yaw == 0.
    assert not look.active and not look.locked and look.attempted


def test_receipt_scan_with_no_yaw_progress_replans_before_standing_bias_locks_it(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def tick(stamp):
        look.update(_depth(stamp), {**_odom(stamp), "vy": -.039, "received_ms": stamp},
                    geometry, receipt_only=True, max_depth_age_ms=800)
        return look.command(geometry, .30, 1., 600, 800)

    tick(1000)
    assert look.begin((1., -1.))
    assert tick(1050).yaw == 0.
    assert tick(1150).yaw < 0.
    for stamp in range(1250, 2550, 100):
        tick(stamp)
    hold = tick(2550)
    assert hold.yaw == 0. and "转向无实测进展" in hold.reason
    # A subsequent missing frame must not make a stalled turn retryable.
    look.update(None, {**_odom(2600), "received_ms": 2600},
                geometry, receipt_only=True, max_depth_age_ms=800)
    tick(2650)
    assert tick(2800).yaw == 0.
    assert not look.active and not look.locked and look.attempted


@pytest.mark.parametrize("missing", ["depth", "odom", "poll_gap"])
def test_receipt_scan_stops_then_replans_after_temporary_input_loss(monkeypatch, missing):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, depth=True, odom=True, wz=0.):
        return look.update(_depth(stamp) if depth else None,
                           {**_odom(stamp, wz), "received_ms": stamp} if odom else None,
                           geometry, receipt_only=True, max_depth_age_ms=800)

    update(1000)
    assert look.begin((1., -1.))
    update(1050)
    assert look.command(geometry, .33, 1., 600, 800).yaw == 0.
    update(1150)
    assert look.command(geometry, .33, 1., 600, 800).yaw < 0.
    loss = 1500 if missing == "poll_gap" else 1250
    hold = update(loss, depth=missing != "depth", odom=missing != "odom")
    assert hold.yaw == 0. and look.active and not look.locked
    for t, yaw in ((loss + 100, -.12), (loss + 200, 0.), (loss + 350, 0.)):
        update(t, wz=yaw)
        assert look.command(geometry, .33, 1., 600, 800).yaw == 0.
    assert not look.attempted and not look.has_result and not look.active and not look.locked
    assert look.begin((-1., 1.))


@pytest.mark.parametrize("released", [True, False])
def test_receipt_scan_replan_after_travel_locks_only_before_a_released_straight_segment(
        monkeypatch, released):
    """Tianyi 2.0, 2026-11-20 06:06:59: depth paused for 2.3 s while the scan's
    straight segment had already covered 0.62 m. The replan hold must settle and
    replan from the new view. Drift of the same size with no released segment
    is still an unexplained translation and keeps locking."""
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, vx=0., depth=True):
        return look.update(_depth(stamp) if depth else None,
                           {"vx": vx, "vy": 0., "wz": 0., "received_ms": stamp},
                           geometry, receipt_only=True, max_depth_age_ms=800)

    update(1000)
    assert look.begin((1., -1.), advance_m=1.)
    update(1050)
    assert look.command(geometry, .33, 1., 600, 800).yaw == 0.
    update(1150)
    assert look.command(geometry, .33, 1., 600, 800).yaw < 0.
    look.phase, look.issued_yaw = "evaluate", True
    if released:
        look.advance_origin = look.pose
    # 0.6 m of forward travel, as the policy's released straight segment produces.
    for stamp in range(1250, 5300, 100):
        assert update(stamp, vx=.15) is None
    assert math.dist(look.pose[:2], look.origin[:2]) > .18
    hold = update(5350, depth=False)
    assert hold.yaw == 0. and "停车后重规划" in hold.reason and look.phase == "brake_for_replan"
    for stamp in range(5450, 7650, 100):
        assert update(stamp, depth=False).reason == hold.reason
    assert update(7700) is None
    first = look.command(geometry, .33, 1., 600, 800)
    assert first.yaw == 0.
    if not released:
        assert not look.active and look.locked
        return
    assert look.active and not look.locked
    assert look.attempted
    # Fresh motion feedback alone cannot release the scan. The last depth
    # frame was taken before the settled observation window.
    look.update(_depth(7700), {**_odom(7850), "received_ms": 7850},
                geometry, receipt_only=True, max_depth_age_ms=800)
    assert "等待停稳后的观测" in look.command(geometry, .33, 1., 600, 800).reason
    assert look.active and look.attempted
    update(7900)
    replan = look.command(geometry, .33, 1., 600, 800)
    assert replan.yaw == 0. and "重新规划" in replan.reason
    assert not look.active and not look.locked and not look.issued_yaw and not look.attempted


@pytest.mark.parametrize("missing", ["depth", "odom"])
def test_receipt_scan_retries_if_input_lapses_while_braking_before_first_turn(monkeypatch, missing):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, *, depth=True, odom=True, vx=0.):
        return look.update(_depth(stamp) if depth else None,
                           {**_odom(stamp), "received_ms": stamp, "vx": vx} if odom else None,
                           geometry, receipt_only=True)

    for stamp in range(1000, 1900, 100):
        update(stamp, vx=.3)
    assert look.pose[0] > .18
    assert look.begin((-1., 1.))
    assert not look.origin_set and not look.issued_yaw
    update(1900, depth=missing != "depth", odom=missing != "odom")
    assert look.phase == "brake_for_replan" and look.attempted

    update(2000)
    assert look.command(geometry, .33, 1., 600).yaw == 0.
    assert look.active and look.attempted
    update(2200)
    assert "已停稳，重新规划" in look.command(geometry, .33, 1., 600).reason
    assert not look.locked
    assert look.begin((-1., 1.))


def test_scan_waits_for_the_odometry_bracket_instead_of_discarding_the_depth(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((-1., 1.))
    _tick(lookout, geometry, 1050)
    waiting = lookout.update(_depth(1120), _odom(1100), geometry)
    assert waiting.yaw == 0 and lookout.active
    assert lookout.update(_depth(1220), _odom(1200), geometry) is None
    assert lookout.current_frame == 1120
    assert lookout.command(geometry, .33, .25, 600).yaw > 0


def test_scan_allows_an_unobserved_rear_sweep(monkeypatch):
    lookout, geometry, visible, _ = _setup(monkeypatch)
    visible[:] = scan.FORWARD > 0
    _tick(lookout, geometry, 1000)
    assert lookout.clear_sweep(geometry, .33, scan.ANGLE, 0.)
    assert lookout.clear_sweep(geometry, .33, -scan.ANGLE, 0.)


def test_s10_search_sweep_is_checked_against_hits_not_the_rear_grid_extent():
    geometry = camera_square(object(), (480, 420, 243), .23)
    occupied = np.zeros_like(scan.RIGHT, dtype=bool)
    horizon = (600 + 800 + 50) / 1000
    sweep = (.60 + .10) * horizon
    lookout = scan.ActiveScan()

    for direction in (-1, 1):
        assert lookout.clear_sweep(geometry, .33, direction * sweep,
                                   scan.STILL_LINEAR * horizon, occupied=occupied)
    occupied[scan.START[0] + 6, scan.START[1]] = True
    for direction in (-1, 1):
        assert not lookout.clear_sweep(geometry, .33, direction * sweep,
                                       scan.STILL_LINEAR * horizon, occupied=occupied)


def test_body_geometry_does_not_assume_the_camera_or_cloud_is_the_yaw_pivot():
    geometry = camera_square(object(), (480, 420, 243), .23)
    transform = geometry.base_from_optical.copy()
    transform[:3, 3] = [.27, .04, .06]
    inverse = np.linalg.inv(transform).tolist()
    assert body_square(geometry, {"camera_optical_from_cloud": inverse,
                                  "cloud_frame_id": "lidar_link"}) is None
    mounted = body_square(geometry, {"camera_optical_from_cloud": inverse,
                                     "cloud_frame_id": "base_link"})
    np.testing.assert_allclose(mounted.base_from_optical, transform)


@pytest.mark.parametrize("half_width", [.23, .35])
def test_receipt_only_scan_turns_with_only_front_depth(half_width):
    horizontal = HorizontalIntrinsics(640, 315, 337)
    vertical = (480, 420, 243)
    geometry = camera_square(horizontal, vertical, half_width)
    tangent = (np.arange(480) - 243) / 420
    floor = np.minimum(.42 / np.maximum(tangent - .02, .001), 4.0)
    image = np.broadcast_to(floor[:, None], (480, 640)).copy()
    look = scan.ActiveScan()
    ground = ground_mask(image, horizontal, vertical)
    assert ground is not None

    for stamp in (1000, 1050, 1150):
        look.update({"map": image, "ground": ground, "frame_id": stamp},
                    {**_odom(stamp), "received_ms": stamp}, geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1., -1.))
        result = look.command(geometry, half_width + .10, .18, 600)
    assert result.yaw == -.18


@pytest.mark.parametrize("initial_hit", [False, True])
def test_receipt_only_sweep_uses_measured_cells_and_blocks_an_initial_hit(monkeypatch, initial_hit):
    look, geometry, _, occupied = _setup(monkeypatch)
    occupied[10, 28] = initial_hit
    for stamp in (1000, 1050, 1150):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp},
                    geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1.,))
        result = look.command(geometry, .33, .18, 600)
    assert (result.yaw < 0) is (not initial_hit)


@pytest.mark.parametrize("floor", [0., .5])
def test_scan_uses_the_fastest_executable_rate_supported_by_the_sweep(monkeypatch, floor):
    look, geometry, _, occupied = _setup(monkeypatch)
    occupied[scan.START[0] + 3, scan.START[1] - 9] = True
    for stamp in (1000, 1050, 1150):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp},
                    geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1.,))
        result = look.command(geometry, .33, 1., 600, min_yaw_rate=floor)
    if floor:
        assert result.yaw == 0.
    else:
        assert -.5 < result.yaw < -.2


def test_sweep_stops_for_an_occupied_cell_that_straddles_the_footprint(monkeypatch):
    look, _, _, occupied = _setup(monkeypatch)
    geometry = Geometry(None, None, np.eye(4), .22, .23, None)
    # This cell is centred at forward .20 m, left .30 m. It extends .005 m
    # past the front edge and the right turn sweeps through the exposed part.
    occupied[14, 22] = True
    _tick(look, geometry, 1000)
    assert look.begin((1.,))
    _tick(look, geometry, 1050)
    assert _tick(look, geometry, 1150).yaw == 0


def test_receipt_only_scan_rejects_a_republished_depth_frame(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)
    view = {**_depth(1000), "source_frame_id": 80_000}
    look.update(view, {**_odom(1000), "received_ms": 1000}, geometry, receipt_only=True)
    assert look.begin((1., -1.))
    for stamp in (1050, 1150, 1250, 1350, 1450):
        look.update({**view, "frame_id": stamp},
                    {**_odom(stamp), "received_ms": stamp}, geometry, receipt_only=True)
        look.command(geometry, .33, .18, 600)
    assert look.current_frame == 1000
    look.update({**view, "frame_id": 1550},
                {**_odom(1550), "received_ms": 1550}, geometry, receipt_only=True)
    assert look.current_frame == 0
    assert not look.active and look.locked


def test_receipt_stamp_cannot_extend_the_life_of_repeated_odometry(monkeypatch):
    look, geometry, _, _ = _setup(monkeypatch)

    def update(stamp, source):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp,
                                    "source_sample_id": source}, geometry, receipt_only=True)

    update(1000, 1)
    assert look.begin((1., -1.))
    update(1050, 2)
    assert look.command(geometry, .33, .18, 600).yaw == 0
    update(1150, 3)
    assert look.command(geometry, .33, .18, 600).yaw < 0
    update(1250, 3)
    update(1350, 3)
    update(1450, 3)
    assert not look.active and look.locked
