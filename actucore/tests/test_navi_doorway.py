from types import SimpleNamespace

import numpy as np
import pytest

from plugins.navi.doorway import Doorway, passage_clear
from plugins.navi.local_path import FORWARD, RIGHT
from plugins.navi.odom import PlanarPose
from plugins.navi.policy import Config


def test_sweep_requires_observations_outside_the_current_body():
    geometry = SimpleNamespace(front=.30, rear=.30)
    free = np.ones_like(FORWARD, dtype=bool)
    occupied = np.zeros_like(free)
    assert passage_clear((free, occupied), geometry, .275, .375, .10, 0.)

    rear_corner = np.isclose(FORWARD, -.35) & np.isclose(RIGHT, .35)
    free[rear_corner] = False
    assert passage_clear((free, occupied), geometry, .275, .375, 0., 0.)
    assert not passage_clear((free, occupied), geometry, .275, .375, -.10, 0.)
    free[rear_corner] = True
    occupied[rear_corner] = True
    assert not passage_clear((free, occupied), geometry, .275, .375, -.10, 0.)


def test_corridor_keeps_both_ten_centimetre_side_margins():
    geometry = SimpleNamespace(front=.30, rear=.30)
    jamb = np.isclose(FORWARD, 1.) & (np.isclose(abs(RIGHT), .35))
    assert passage_clear((~jamb, jamb), geometry, .275, .30, 0., 1.5)
    assert not passage_clear((~jamb, jamb), geometry, .275, .375, 0., 1.5)


def test_stops_before_turning_then_waits_for_fresh_post_turn_corridor():
    transform = np.eye(4)
    transform[0, 2] = 1.
    geometry = SimpleNamespace(front=.30, rear=.30, base_from_optical=transform)
    config = Config(half_width_m=.275, clearance_margin_m=.10)
    door = Doorway(1.30)
    free = np.ones_like(FORWARD, dtype=bool)
    occupied = np.isclose(FORWARD, 1.30) & np.isclose(RIGHT, -.35)
    scan = SimpleNamespace(receipt_only=False, current_frame=0, pose_origin=None)
    scan.grid = lambda *, since_ms: (free & ~occupied, occupied)

    def tick(stamp, yaw=0., vx=0., capture=None, x=0.):
        capture = stamp if capture is None else capture
        pose = PlanarPose(stamp, ("odom", "body"), x, 0., yaw)
        scan.current_frame = capture
        scan.views = [(capture,)]
        scan.pose_origin = pose
        odom = dict(stamp_ms=stamp, door_pose=pose, vx=vx, vy=0., wz=0.)
        return door.step(odom, scan, geometry, config, 1.30, 3.0)[:2]

    assert tick(1000, vx=.15) == (0., 0.)
    assert tick(1100) == (0., 0.)
    assert tick(1200) == (0., 0.)
    assert tick(1300, capture=1200) == (0., 0.)
    vx, yaw = tick(1350)
    assert vx == 0. and yaw < 0.
    assert tick(1400, yaw=door.heading) == (0., 0.)
    assert tick(1500, yaw=door.heading) == (0., 0.)
    assert tick(1600, yaw=door.heading, capture=1500) == (0., 0.)
    occupied[:] = True
    assert tick(1650, yaw=door.heading) == (0., 0.)
    occupied[:] = False
    vx, yaw = tick(1720, yaw=door.heading)
    assert vx > 0. and yaw == 0.
    scan.receipt_only = True
    assert tick(1750, yaw=door.heading) == (0., 0.)


def test_small_turn_uses_aligned_depth_with_staggered_feedback(monkeypatch):
    from plugins.navi import active_scan

    geometry = SimpleNamespace(front=.30, rear=.30, base_from_optical=np.eye(4))
    geometry.base_from_optical[0, 2] = 1.
    config = Config(half_width_m=.275)
    scan = active_scan.ActiveScan()
    door = Doorway(1.30)
    free = np.ones_like(FORWARD, dtype=bool)
    ground = np.ones((1, 1), dtype=bool)

    def project(frame, *_):
        # Same jamb re-expressed in the camera pose at capture.
        yaw = frame.item()
        f = np.cos(yaw) * 1.30 - np.sin(yaw) * -.35
        r = np.sin(yaw) * 1.30 + np.cos(yaw) * -.35
        occupied = np.isclose(FORWARD, round(f / .05) * .05)
        occupied &= np.isclose(RIGHT, round(r / .05) * .05)
        return free & ~occupied, occupied

    monkeypatch.setattr(active_scan, "observe", project)
    angle = rate = 0.
    turns = 0
    for stamp in range(1000, 9000, 100):
        angle += rate * .10
        pose = PlanarPose(stamp, ("odom", "body"), 0., 0., angle)
        odom = dict(vx=0., vy=0., wz=rate, stamp_ms=stamp, door_pose=pose,
                    motion_pose=pose, pose_feedback_since_ms=stamp)
        depth = dict(capture_ms=stamp + 50, map=np.array([[angle + rate * .05]]), ground=ground)
        scan.update(depth, odom, geometry)
        vx, rate, reason = door.step(odom, scan, geometry, config, 1.30, 3.0)
        if vx:
            assert turns > 5
            assert rate == 0.
            assert 0 < door.evidence_ms < stamp
            break
        if rate:
            turns += 1
            assert 0 < abs(rate) < active_scan.STILL_YAW
    else:
        raise AssertionError((door.phase, reason, turns))


@pytest.mark.parametrize("rear_visible", [True, False])
def test_wheeled_observation_can_turn_before_the_forward_corridor_is_visible(rear_visible):
    transform = np.eye(4)
    transform[0, 2] = 1.
    geometry = SimpleNamespace(front=.30, rear=.30, base_from_optical=transform)
    config = Config(half_width_m=.275)
    door = Doorway(1.30, observe_side=-1.)
    free = FORWARD < .65
    if not rear_visible:
        free[FORWARD < -.31] = False
    occupied = np.zeros_like(free)
    scan = SimpleNamespace(receipt_only=False)
    scan.grid = lambda *, since_ms: (free, occupied)

    def tick(stamp, yaw=0.):
        pose = PlanarPose(stamp, ("odom", "base_link"), 0., 0., yaw)
        scan.current_frame = stamp
        scan.views = [(stamp,)]
        scan.pose_origin = pose
        odom = dict(stamp_ms=stamp, door_pose=pose, vx=0., vy=0., wz=0.)
        return door.step(odom, scan, geometry, config, 1.30, 3.0)[:2]

    assert tick(1000) == (0., 0.)
    assert tick(1150) == (0., 0.)
    vx, yaw = tick(1250)
    assert vx == 0.
    if not rear_visible:
        assert yaw == 0.
        return
    assert yaw > 0.

    assert tick(1350, door.heading) == (0., 0.)
    assert tick(1500, door.heading) == (0., 0.)
    assert tick(1700, door.heading) == (0., 0.)
    assert tick(1740, door.heading) == (0., 0.)
    free[:] = True
    vx, yaw = tick(1780, door.heading)
    assert vx > 0. and yaw == 0.
