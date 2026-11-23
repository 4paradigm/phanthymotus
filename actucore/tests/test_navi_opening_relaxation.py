"""Measured opening proposals still use the native complete-stop selector."""
import numpy as np
import pytest

from plugins.navi import depth, dwa, local_path
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.motion import FootprintSpec


def _scene(*, body, opening=(-.25, .35), thin=False, one_jamb=False, barrier=None):
    """Optical Z from a floor, two upright jambs and a back wall."""
    width, height = 160, 120
    fx, fy, cx, cy = 85., 72., 80., 60.
    camera = local_path.camera_square(HorizontalIntrinsics(width, fx, cx),
                                      (height, fy, cy), .20, at_front=not body)
    geometry = camera
    if body:
        mount = camera.base_from_optical.copy()
        mount[:3, 3] = (.2, 0., .35)
        geometry = local_path.body_square(camera, {"body_from_optical": mount})
    plane = 1.2 if body else 1.
    lo, hi = opening
    right = (np.arange(width)-cx) / fx
    if thin:
        jambs = (abs(right-lo) <= .012) | ((abs(right-hi) <= .012) & (not one_jamb))
    else:
        jambs = (right <= lo) | ((right >= hi) & (not one_jamb))
    v = (np.arange(height)[:, None]-cy) / fy
    floor = np.divide(.35, v, out=np.full_like(v, np.inf), where=v>0.)
    frame = np.minimum(np.broadcast_to(np.where(jambs, 1., 4.), (height, width)), floor).astype(np.float32)
    if barrier is not None:
        frame = np.minimum(frame, barrier)
    ground = depth.ground_mask(frame, geometry.horizontal, geometry.vertical)
    grid = local_path.project_observation({"map":frame, "ground":ground}, geometry)
    return frame, geometry, plane, grid


def _opening(scene, target=(3., 0.)):
    frame, geometry, plane, grid = scene
    return local_path.opening_route(frame, grid[1], geometry, .22, plane, target, True)


def _select(route, scene, target=(3., 0.), *, body):
    _, geometry, _, grid = scene
    diagnostics = {}
    selected = dwa.select(
        route=route.points, grid=grid, geometry=geometry,
        footprint=FootprintSpec(geometry.front, geometry.rear, .20, .02,
                                "body" if body else "camera_approx"),
        measured=(0., 0., 0.), previous=(0., 0., 0.),
        x_limit=(0., .15), y_limit=(0., 0.), yaw_limit=(-.6, .6),
        floors=(.08, 0., .08, .04), target=target, stop_distance=.8,
        horizon=1.2, latency_s=.15, supported_region=local_path.SUPPORTED_REGION,
        diagnostics=diagnostics)
    return selected, diagnostics


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("side", [-1., 1.], ids=["left", "right"])
@pytest.mark.parametrize("case", ["off_axis", "target_offset", "thin_jambs"])
def test_measured_opening_can_propose_a_detour_with_native_checked_motion(body, side, case):
    opening = (.15, .85) if case == "off_axis" else (-.25, .35)
    opening = opening if side > 0. else (-opening[1], -opening[0])
    target = (3., side * 1.2) if case == "target_offset" else (3., 0.)
    scene = _scene(body=body, opening=opening, thin=case == "thin_jambs")
    route = _opening(scene, target)
    assert route is not None, "A measured opening need not lie on the direct target ray"
    # The line actually crosses the measured gap rather than its near jamb.
    _, _, plane, grid = scene
    point = min(route.points, key=lambda p:abs(p[0]-plane))
    assert point[1] == pytest.approx(sum(opening)/2, abs=local_path.RESOLUTION)
    assert grid[1].any()
    choice, diagnostics = _select(route, scene, target, body=body)
    assert choice is not None, diagnostics
    assert choice.check.accepted and choice.trace.stopped
    assert choice.command[0] > 0. and choice.progress > 0.


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
@pytest.mark.parametrize("case", ["narrow", "one_jamb", "no_far_depth", "near_depth"])
def test_opening_proposal_still_requires_a_measured_passable_gap(body, case):
    scene = _scene(body=body, opening=(-.15, .15) if case == "narrow" else (-.25, .35),
                   thin=case == "one_jamb", one_jamb=case == "one_jamb")
    frame, geometry, plane, grid = scene
    if case in ("no_far_depth", "near_depth"):
        # Only change the subsequent optical evidence; the measured jambs
        # alone cannot prove an opening through a newly occluded view.
        frame = frame.copy()
        frame[:, 59:104] = np.nan if case == "no_far_depth" else .9
        scene = frame, geometry, plane, grid
    assert _opening(scene) is None


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
def test_new_depth_barrier_rejects_an_earlier_opening_in_native_motion_check(body):
    scene = _scene(body=body, opening=(-.25, .25))
    route = _opening(scene)
    assert route is not None
    choice, _ = _select(route, scene, body=body)
    assert choice is not None and choice.command[0] > 0.

    blocked = _scene(body=body, opening=(-.25, .25), barrier=.12)
    assert blocked[3][1].any()
    choice, diagnostics = _select(route, blocked, body=body)
    assert choice is None
    assert diagnostics["rejected"].get("footprint_collision", 0) > 0


@pytest.mark.parametrize("body", [True, False], ids=["body", "camera"])
def test_legacy_opening_gate_keeps_its_original_target_line_rule(body):
    frame, geometry, plane, grid = _scene(body=body)
    assert local_path.opening_route(frame, grid[1], geometry, .22, plane, (3., 0.)) is not None
    assert local_path.opening_route(frame, grid[1], geometry, .22, plane, (3., 1.2)) is None
