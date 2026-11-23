"""Clear remembered cells only from current evidence covering their actual area."""

import math
from dataclasses import replace

import numpy as np
import pytest

from plugins.navi import active_scan, local_path, policy
from plugins.navi.camera import HorizontalIntrinsics
from plugins.navi.motion import FootprintSpec, evaluate_motion, predict_motion
from plugins.navi.odom import PlanarPose


CROSS = ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3))


@pytest.mark.parametrize("center,yaw,free_cells,occupied_cells,cleared", [
    ((3., 3.), math.pi / 4, CROSS, (), True),
    ((3.1, 3.15), math.pi / 4, CROSS, (), True),
    ((3.1, 3.2), math.pi / 4, CROSS, (), False),
    ((3.1, 3.2), math.pi / 4, (*CROSS, (4, 4)), (), True),
    ((3., 3.), math.radians(1), CROSS, (), True),
    ((3., 3.), 0., ((3, 3),), (), True),
    ((3., 3.), math.pi / 2, ((3, 3),), (), True),
    ((3., 3.), math.pi / 4, CROSS[1:], (), False),
    ((3., 3.), math.pi / 4, CROSS, ((2, 3),), False),
    ((3., 3.), math.pi / 4, CROSS, ((4, 4),), True),
    ((0., 3.), math.pi / 4, ((0, 2), (0, 3), (0, 4), (1, 3)), (), False),
    ((0., 3.), math.pi / 2, ((0, 3),), (), True),
    ((3., 4. - 1 / math.sqrt(2)), math.pi / 4, CROSS, (), True),
], ids=["rotated", "translated", "translated-corner-unknown", "translated-corner-clear",
        "small-angle", "aligned", "quarter-turn", "overlap-unknown", "overlap-occupied",
        "aabb-corner-occupied", "outside-view", "view-edge-touch", "corner-point-touch"])
def test_confirmed_coverage_requires_only_positive_area_intersections(
        center, yaw, free_cells, occupied_cells, cleared):
    # For a centred 45-degree square, the four diagonal neighbours lie
    # outside its area. Translating it by (.1, .2) enters the lower-right
    # neighbour; that cell then needs its own confirmed free observation.
    clear = np.zeros((7, 7), dtype=bool)
    occupied = np.zeros_like(clear)
    for cell in free_cells:
        clear[cell] = True
    for cell in occupied_cells:
        occupied[cell] = True
    actual = active_scan.confirmed_square_coverage(np.array([center]), yaw, clear, occupied)
    assert actual.tolist() == [cleared]


@pytest.mark.parametrize("yaw,source_yaw", [(math.pi / 6, 0.), (-math.pi / 6, 0.), (math.pi / 6, .20)],
                         ids=["positive", "negative", "rotated-source"])
@pytest.mark.parametrize("occupied", [False, True], ids=["unknown", "occupied"])
def test_retained_view_clearing_uses_the_registered_square_orientation(yaw, source_yaw, occupied):
    # At +30 degrees and centre (.495, .025), a source vertex is
    # (.52915, .01585): strictly inside the cell centred at (.55, 0).
    # Its mirror instead enters the neighbour at (.55, .05).
    # Reflecting the complete arrangement exercises -30 degrees too.
    sign = 1 if yaw > 0 else -1
    transform_yaw = yaw - source_yaw
    before = PlanarPose(1000, ("odom",), .495 - .50 * math.cos(transform_yaw),
                        .50 * math.sin(transform_yaw) - sign * .025, -transform_yaw)
    current = PlanarPose(1100, before.frame, 0., 0., 0.)
    squares = local_path.ObstacleSquares(np.array([[.50, 0.]]), source_yaw)
    clear = np.zeros(local_path.SHAPE, dtype=bool)
    for row in (29, 30):
        for col in (28, 28 + sign):
            clear[row, col] = True
    clear[31, 28 + sign] = True
    hits = np.zeros_like(clear)
    if occupied:
        clear[31, 28] = True
        hits[31, 28] = True
    retained = policy._retire_confirmed((1000, before, squares), current, clear, hits)
    assert retained is not None and retained[2] is squares


@pytest.mark.parametrize("missing_support,corner_hit", [(False, False), (True, False), (False, True)],
                         ids=["covered", "missing-support", "new-obstacle-outside-square"])
def test_measured_floor_releases_rotated_memory_only_when_its_area_is_covered(missing_support, corner_hit):
    geometry = local_path.camera_square(HorizontalIntrinsics(160, 100., 80.),
                                      (120, 100., 20.), .20)
    # A real depth projection records one upright face in the .50 m cell.
    obstacle = np.full((120, 160), np.nan)
    obstacle[25:90, 78:83] = .50
    measured = local_path.observe(obstacle, geometry, local_path.RIGHT, local_path.FORWARD,
                                  local_path.RESOLUTION, None)
    assert np.argwhere(measured[1]).tolist() == [[30, 28]]
    memory = active_scan.ObservedObstacles()
    before = PlanarPose(1000, ("odom",), 0., 0., 0.)
    for index in range(3):
        memory.remember_view(replace(before, stamp_ms=1000 + 100 * index), *measured,
                             remember_occupied=measured[1], clear=local_path.confirmed_free(measured),
                             physical_support=True)
    assert memory.cells == {(10, 0)}

    # Move the camera so the remembered world square is centred .50 m
    # ahead, now rotated 45 degrees relative to the current depth grid.
    yaw = math.pi / 4
    forward, right = (.5025, .0025) if corner_hit else (.50, 0.)
    current = PlanarPose(1300, before.frame, .50 - forward * math.cos(yaw) - right * math.sin(yaw),
                         -forward * math.sin(yaw) + right * math.cos(yaw), yaw)
    z = .42 * 100 / np.maximum(np.arange(120) - 20, 1)
    floor = np.broadcast_to(z[:, None], obstacle.shape).copy()
    right = (np.arange(160) - 80) * floor / 100
    row = np.rint(floor / .05).astype(int)
    col = np.rint(right / .05).astype(int)
    cross = ((row == 10) & (abs(col) <= 1)) | ((abs(row - 10) <= 1) & (col == 0))
    if missing_support:
        cross &= ~((row == 10) & (col == 1))
    floor[~cross] = np.nan
    ground = np.isfinite(floor)
    if corner_hit:
        # This current face occupies the old square's bounding-box corner,
        # outside its actual area even after the .05-cell translation.
        floor[25:70, 89:92] = .55
    newer = local_path.observe(floor, geometry, local_path.RIGHT, local_path.FORWARD,
                               local_path.RESOLUTION, ground)
    assert np.argwhere(newer[1]).tolist() == ([[31, 29]] if corner_hit else [])
    assert local_path.confirmed_free(newer).sum() == (4 if missing_support else 5)

    trace = predict_motion((.25, 0., 0.), (.25, 0., 0.), (.25, 0., 0.), hold_s=1.2,
                           latency_s=0., acceleration=(.5, .5, 1.), braking=(.45, .45, .8),
                           geometry=geometry, mode="body")

    def motion_check():
        _, squares = memory.project(current, include_squares=True)
        return evaluate_motion(trace, footprint=FootprintSpec(.20, .20, .20, .02, "body"),
                               occupied=newer[1], remembered=squares, supported_region=None,
                               original_goal=None, stop_distance_m=0.)

    assert motion_check().reason == "remembered_obstacle"
    memory.remember_view(current, *newer, clear=local_path.confirmed_free(newer), physical_support=True)
    result = motion_check()
    assert result.accepted is (not missing_support and not corner_hit)
    expected_reason = "remembered_obstacle" if missing_support else "footprint_collision" if corner_hit else None
    assert result.reason == expected_reason
    assert bool(memory.has_occupied) is missing_support
