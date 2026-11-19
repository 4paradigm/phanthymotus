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


def test_active_scan_turns_right_and_waits_for_post_turn_depth(monkeypatch):
    lookout, geometry, _, blocked = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    assert _tick(lookout, geometry, 1050).yaw == 0
    assert _tick(lookout, geometry, 1150).yaw < 0
    for stamp in range(1250, 2250, 100):
        result = _tick(lookout, geometry, stamp, -.25)
    assert result.yaw == 0
    assert _tick(lookout, geometry, 2300, capture=2150).yaw == 0
    assert _tick(lookout, geometry, 2450, capture=2150).yaw == 0
    assert not lookout.ready
    blocked[30, 38] = True
    assert _tick(lookout, geometry, 2500).yaw == 0
    assert _tick(lookout, geometry, 2550).yaw > 0
    blocked[30, 38] = False
    for stamp in range(2650, 3650, 100):
        result = _tick(lookout, geometry, stamp, .25)
    assert result.yaw == 0
    _tick(lookout, geometry, 3650)
    _tick(lookout, geometry, 3800)
    assert lookout.ready
    _, occupied = lookout.grid()
    assert occupied.any()
    assert not occupied[30, 38]  # the side observation was reprojected
    assert lookout.try_other_side()
    assert _tick(lookout, geometry, 3900).yaw == 0
    assert _tick(lookout, geometry, 4000).yaw > 0


def test_active_scan_holds_an_occupied_sweep(monkeypatch):
    lookout, geometry, _, blocked = _setup(monkeypatch)
    blocked[14, 28] = True
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    for stamp in (1050, 1200, 1250, 1350):
        assert _tick(lookout, geometry, stamp).yaw == 0
    assert not lookout.active
    assert not lookout.begin((1., -1.))


def test_pose_only_scan_converges_with_a_300_ms_pose_delay(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    history = {stamp: 0. for stamp in range(99_600, 100_000, 100)}
    yaw, rate = 0., 0.
    for stamp in range(100_000, 107_000, 100):
        yaw += rate * .1
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
        rate = lookout.command(geometry, .31, .83, 300, min_yaw_rate=.33).yaw
        assert abs(yaw) <= scan.RECEIPT_ANGLE + scan.ANGLE_TOLERANCE
        if lookout.ready:
            break

    assert lookout.ready
    assert yaw == pytest.approx(scan.RECEIPT_ANGLE, abs=scan.ANGLE_TOLERANCE)


def test_active_scan_stops_with_stale_odom_and_cannot_restart_on_same_obstacle(monkeypatch):
    lookout, geometry, _, _ = _setup(monkeypatch)
    _tick(lookout, geometry, 1000)
    assert lookout.begin((1., -1.))
    _tick(lookout, geometry, 1050)
    assert _tick(lookout, geometry, 1200).yaw < 0
    lookout.update(_depth(1250), None, geometry)
    assert not lookout.active
    assert not lookout.begin((1., -1.))


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
    assert lookout._clear_sweep(geometry, .33, scan.ANGLE, 0.)
    assert lookout._clear_sweep(geometry, .33, -scan.ANGLE, 0.)


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


@pytest.mark.parametrize(("half_width", "expected_yaw"), [(.23, -.18), (.35, 0.)])
def test_receipt_only_scan_turns_with_front_depth_inside_the_grid(half_width, expected_yaw):
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
    assert result.yaw == expected_yaw


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
    look, geometry, _, _ = _setup(monkeypatch)
    for stamp in (1000, 1050, 1150):
        look.update(_depth(stamp), {**_odom(stamp), "received_ms": stamp},
                    geometry, receipt_only=True)
        if stamp == 1000:
            assert look.begin((1.,))
        result = look.command(geometry, .33, 1., 600, min_yaw_rate=floor)
    if floor:
        assert result.yaw == 0.
    else:
        assert -.5 < result.yaw < -.25


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
