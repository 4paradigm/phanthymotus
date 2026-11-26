"""Decoding `image/depth-zlib` and sampling distances out of it.

The encoder lives in another bundle and another image
(`perception/plugins/visual_depth.py::encode_depth`), so this file **mirrors its
contract rather than importing it** — a test that needs both installed is a test
nobody runs. The buffers here are hand-built to that contract, which is also what
makes them a check on it: if perception changes the layout, these stop matching
and the card is told before a robot is.

The rule under test throughout is the same one the format has: **0 means "no
reading", never "zero metres"**. An obstacle at 0.0 m is the most alarming thing
a navigation policy can be told, and it is what the encoder writes for every
out-of-range pixel.
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.navi import depth as D  # noqa: E402


def _encode(metres_array):
    """The encoder's contract, mirrored: 640x480 little-endian uint16 mm, 0 for
    invalid. Same as perception's `encode_depth`."""
    mm = np.rint(np.nan_to_num(metres_array, nan=0.0) * 1000.0)
    mm[(mm < 1) | (mm > 65535)] = 0
    return zlib.compress(mm.astype("<u2").tobytes(), 1)


def _solid(metres):
    return np.full((D.HEIGHT, D.WIDTH), metres, dtype=np.float32)


# ── decoding ─────────────────────────────────────────────────────────────────

def test_a_round_trip_preserves_metres():
    decoded = D.decode(_encode(_solid(2.5)))
    assert decoded.shape == (D.HEIGHT, D.WIDTH)
    assert np.allclose(decoded, 2.5, atol=1e-3)


def test_zero_decodes_to_nan_not_to_zero_metres():
    """The whole reason this decoder exists rather than a bare frombuffer."""
    frame = _solid(2.0)
    frame[100:200, 100:200] = 0.0
    decoded = D.decode(_encode(frame))
    assert np.isnan(decoded[150, 150])
    assert not np.isnan(decoded[10, 10])


def test_a_short_buffer_is_refused_rather_than_reshaped():
    """The dashboard renderer silently drops a short buffer and shows a blank
    panel with nothing in any log. A consumer that steers a robot must not
    inherit that — a wrongly-shaped map would be read as real geometry."""
    with pytest.raises(D.DepthError, match="expected"):
        D.decode(zlib.compress(b"\x00" * 1000))


def test_a_non_zlib_payload_is_refused():
    with pytest.raises(D.DepthError, match="zlib"):
        D.decode(b"definitely not zlib")


# ── sampling a box ───────────────────────────────────────────────────────────

def test_a_box_reads_the_distance_of_what_fills_it():
    frame = _solid(5.0)
    frame[200:280, 300:380] = 1.5            # an object in the middle
    decoded = D.decode(_encode(frame))
    near = D.sample_box(decoded, (300 / D.WIDTH, 200 / D.HEIGHT,
                                  380 / D.WIDTH, 280 / D.HEIGHT))
    assert near == pytest.approx(1.5, abs=0.05)


def test_a_box_of_no_valid_pixels_returns_none_not_zero():
    frame = _solid(2.0)
    frame[:, :] = 0.0
    assert D.sample_box(D.decode(_encode(frame)), (0.4, 0.4, 0.6, 0.6)) is None


def test_a_box_running_off_the_edge_is_clamped():
    """Detector boxes routinely extend past the frame."""
    decoded = D.decode(_encode(_solid(3.0)))
    assert D.sample_box(decoded, (-0.5, -0.5, 1.5, 1.5)) == pytest.approx(3.0, abs=0.05)


def test_a_degenerate_box_still_yields_something_or_none():
    decoded = D.decode(_encode(_solid(3.0)))
    assert D.sample_box(decoded, (0.5, 0.5, 0.5, 0.5)) is not None


def test_the_percentile_ignores_edge_outliers():
    """A box around an object contains some background. The 25th percentile
    reports the object; a minimum would report the nearest stray pixel and a
    mean would be dragged towards the wall behind it."""
    frame = _solid(6.0)
    frame[200:280, 300:380] = 2.0
    frame[200:205, 300:305] = 0.3            # a few near outliers on the edge
    decoded = D.decode(_encode(frame))
    reading = D.sample_box(decoded, (300 / D.WIDTH, 200 / D.HEIGHT,
                                     380 / D.WIDTH, 280 / D.HEIGHT))
    assert reading == pytest.approx(2.0, abs=0.2)


# ── sampling a point (the no-bbox fallback) ──────────────────────────────────

def test_a_centre_point_maps_to_the_middle_of_the_frame():
    frame = _solid(5.0)
    frame[220:260, 300:340] = 1.2
    decoded = D.decode(_encode(frame))
    assert D.sample_point(decoded, [0.0, 0.0]) == pytest.approx(1.2, abs=0.2)


def test_position_minus_one_maps_to_the_left_edge():
    """vop's position is centre-relative in -1..1; getting this mapping wrong
    samples the wrong part of the image and produces plausible numbers."""
    frame = _solid(5.0)
    frame[:, :60] = 1.0
    decoded = D.decode(_encode(frame))
    assert D.sample_point(decoded, [-1.0, 0.0]) == pytest.approx(1.0, abs=0.2)


# ── bands ────────────────────────────────────────────────────────────────────

def test_bands_report_the_nearest_thing_in_each_third():
    frame = _solid(5.0)
    frame[200:300, 0:200] = 1.0              # something close on the left
    decoded = D.decode(_encode(frame))
    bands = D.nearest_by_band(decoded)
    assert bands["left"] == pytest.approx(1.0, abs=0.2)
    assert bands["center"] == pytest.approx(5.0, abs=0.2)


def test_a_band_with_no_valid_pixels_is_none_not_zero():
    """None means 'cannot say'. Zero would mean 'an obstacle is touching me',
    and the policy would stop for ever."""
    frame = _solid(5.0)
    # The full left third — computed the way the code splits it, not rounded to
    # a nice number, or a sliver of valid pixels survives at the boundary and
    # the band reports a distance after all.
    frame[:, :round(D.WIDTH / len(D.BANDS))] = 0.0
    bands = D.nearest_by_band(D.decode(_encode(frame)))
    assert bands["left"] is None
    assert bands["center"] is not None


def test_the_floor_and_the_ceiling_are_excluded():
    """The floor directly under the robot is always the closest thing in the
    frame and is never what it should steer around."""
    frame = _solid(5.0)
    frame[:100, :] = 0.2                     # ceiling
    frame[440:, :] = 0.2                     # floor at its feet
    bands = D.nearest_by_band(D.decode(_encode(frame)))
    assert all(v == pytest.approx(5.0, abs=0.3) for v in bands.values())


def test_band_names_match_the_summary_keys():
    """The precise path and the degraded path have to be interchangeable."""
    assert set(D.BANDS) == {"left", "center", "right"}
    summary = {"nearest_by_region": {"left": 1.0, "center": 2.0, "right": None}}
    assert D.bands_from_summary(summary) == {"left": 1.0, "center": 2.0,
                                             "right": None}


def test_a_summary_missing_a_region_yields_none_for_it():
    assert D.bands_from_summary({"nearest_by_region": {}})["center"] is None
    assert D.bands_from_summary({})["left"] is None
    assert D.bands_from_summary(None)["left"] is None


# ── the corridor ─────────────────────────────────────────────────────────────
#
# A band is a slice of the *image*; a collision is metric. That mismatch is the
# reason this section exists, and the first test is the whole argument.

_HALF_FOV = 0.55          # ~63 degrees, navi's default


def _columns():
    return ((np.arange(D.WIDTH) - (D.WIDTH - 1) / 2.0)
            * (2.0 * np.tan(_HALF_FOV) / D.WIDTH))


def _with_patch(background, lateral_m, distance_m, width_m, rows=(290, 400)):
    """A depth map holding one patch at a metric position, right-positive."""
    out = np.full((D.HEIGHT, D.WIDTH), float(background), dtype=np.float32)
    columns = np.abs(_columns() * distance_m - lateral_m) <= width_m / 2.0
    out[rows[0]:rows[1], columns] = distance_m
    return out


def _corridor(depth_map, **over):
    args = {"half_width_m": 0.33, "half_fov_rad": _HALF_FOV, "reference_m": 0.8}
    args.update(over)
    return D.corridor(depth_map, **args)


def test_an_obstacle_the_bands_call_left_can_be_inside_the_robots_width():
    """The failure this whole mechanism exists for.

    At 0.7 m the centre third of a 63° lens spans ±0.14 m — **narrower than a
    humanoid's shoulders**. So a doorframe 0.25 m off the axis is filed under
    "left", the left band has never stopped forward motion, and the shoulder
    goes into it while the depth map reports the way ahead as clear.
    """
    depth_map = _with_patch(5.0, lateral_m=-0.25, distance_m=0.7, width_m=0.1)

    assert D.nearest_by_band(depth_map)["center"] == pytest.approx(5.0), \
        "the centre band sees nothing — which is the bug"
    assert D.nearest_by_band(depth_map)["left"] == pytest.approx(0.7)

    clearance, coverage = _corridor(depth_map)
    assert clearance == pytest.approx(0.7), "the corridor sees it"
    assert coverage > 0.9


def test_the_corridor_still_ignores_what_the_robot_will_miss():
    """It has to cut both ways, or it is just a wider band: a thing well outside
    the robot's width must not stop it."""
    depth_map = _with_patch(5.0, lateral_m=-1.2, distance_m=2.0, width_m=0.3)
    clearance, _ = _corridor(depth_map)
    assert clearance == pytest.approx(5.0)


@pytest.mark.parametrize("invalid", [np.nan, 0., -1.])
def test_a_corridor_full_of_holes_does_not_read_as_an_empty_one(invalid):
    """`clearance` alone cannot express this: every invalid pixel drops silently
    out of the minimum, so a corridor nobody could measure and a corridor with
    nothing in it produce the same number. The things that make holes — chair
    legs, glass, dark edges — are the things that catch a shoulder."""
    depth_map = np.full((D.HEIGHT, D.WIDTH), invalid, dtype=np.float32)
    clearance, coverage = _corridor(depth_map)
    assert clearance is None
    assert coverage == 0.0


@pytest.mark.parametrize("invalid", [np.nan, 0.])
def test_coverage_is_graduated_rather_than_a_flag(invalid):
    solid = _corridor(np.full((D.HEIGHT, D.WIDTH), 5.0, dtype=np.float32))[1]
    patchy = np.full((D.HEIGHT, D.WIDTH), 5.0, dtype=np.float32)
    patchy[::2, :] = invalid
    assert solid == pytest.approx(1.0, abs=0.02)
    assert 0.3 < _corridor(patchy)[1] < 0.7


def test_a_corridor_wider_than_the_lens_cannot_be_fully_covered():
    """Close in, the robot is wider than the field of view — at half a metre a
    ±0.33 m corridor spans about 67° and a 63° lens does not reach its edges.
    Counting only the visible columns would report a fully-measured corridor
    while a third of the robot's width was never in frame."""
    solid = np.full((D.HEIGHT, D.WIDTH), 5.0, dtype=np.float32)
    assert _corridor(solid, reference_m=0.5)[1] < 0.95
    assert _corridor(solid, reference_m=1.5)[1] == pytest.approx(1.0, abs=0.02)


def test_a_corridor_entirely_outside_the_lens_reports_no_coverage():
    """Not an error and not a division by zero. Zero is the truth: asked about
    a place the camera cannot see, the honest answer is "I cannot see there"."""
    solid = np.full((D.HEIGHT, D.WIDTH), 5.0, dtype=np.float32)
    clearance, coverage = _corridor(solid, lateral_offset_m=6.0, reference_m=0.8)
    assert coverage == 0.0


def test_the_offset_corridor_is_the_space_a_sidestep_moves_into():
    """Which is a different question from "is the right of the picture empty",
    and the one that decides whether a shoulder clears."""
    depth_map = _with_patch(5.0, lateral_m=0.35, distance_m=1.0, width_m=0.2)
    assert _corridor(depth_map, lateral_offset_m=0.33)[0] == pytest.approx(1.0)
    assert _corridor(depth_map, lateral_offset_m=-0.33)[0] == pytest.approx(5.0)




def _floor_scene():
    # Camera looks slightly down at a floor 42 cm below it.
    row_tangent = (np.arange(D.HEIGHT) - 243.2) / 420.0
    floor_range = 0.42 / np.maximum(row_tangent - 0.02, 0.001)
    return np.broadcast_to(np.minimum(floor_range, 4.0)[:, None],
                           (D.HEIGHT, D.WIDTH)).astype(np.float32).copy()


def test_ground_is_excluded_from_obstacle_distance_but_retains_coverage():
    from plugins.navi.camera import HorizontalIntrinsics

    camera = HorizontalIntrinsics(640, 315, 337)
    frame = _floor_scene()
    unfiltered, coverage = _corridor(frame, intrinsics=camera)
    ground = D.ground_mask(frame, camera, (480, 420, 243.2))
    filtered, filtered_coverage = _corridor(frame, intrinsics=camera, ground=ground)
    assert unfiltered < 1.5
    assert ground is not None and ground[404, 320]
    assert filtered > 1.5
    assert filtered_coverage == coverage


@pytest.mark.parametrize("surface", ["slope", "step", "occluded_step", "tall_step", "gap", "leg"])
def test_projection_applies_obstacle_height_to_far_surfaces(surface):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    camera = HorizontalIntrinsics(640, 492, 319.5)
    vertical = (480, 492, 229)
    ray = (np.arange(D.HEIGHT) - 229) / 492 + .20
    floor = 1.25 / np.maximum(ray, .001)
    slope = .13
    if surface in ("slope", "gap"):
        far = (1.25 + 1.9 * slope) / np.maximum(ray + slope, .001)
        surface_depth = np.minimum(floor, far)
        if surface == "gap":
            surface_depth[(floor >= 1.8) & (floor <= 2.25)] = np.nan
    elif surface in ("step", "occluded_step", "tall_step"):
        height = {"step": .18, "occluded_step": .24, "tall_step": .24}[surface]
        top = (1.25 - height) / np.maximum(ray, .001)
        riser = np.nan if surface == "occluded_step" else 3.05
        surface_depth = np.where(top >= 3.05, top, np.where(floor >= 3.05, riser, floor))
    else:
        surface_depth = np.where(floor >= 3.20, 3.20, floor)
    frame = np.broadcast_to(floor[:, None], (D.HEIGHT, D.WIDTH)).astype(np.float32).copy()
    columns = slice(445, 480 if surface == "leg" else 550)
    frame[:, columns] = surface_depth[:, None]
    frame[frame > 4.] = np.nan

    ground = D.ground_mask(frame, camera, vertical)
    assert ground is not None
    geometry = local_path.camera_square(camera, vertical, .30, at_front=True)
    grid = local_path.project_observation(dict(map=frame, ground=ground), geometry)
    roi = (local_path.FORWARD >= 3.15) & (local_path.FORWARD <= 3.45) & (
        local_path.RIGHT >= .80) & (local_path.RIGHT <= 1.6)
    assert bool(grid[1][roi].any()) == (surface in ("occluded_step", "tall_step", "leg"))


def test_ground_filter_preserves_raised_obstacles_and_rejects_a_wall():
    from plugins.navi.camera import HorizontalIntrinsics

    camera = HorizontalIntrinsics(640, 315, 337)
    vertical = (480, 420, 243.2)
    frame = _floor_scene()
    # The top is 8 cm above the floor, in the bottom of the obstacle region.
    frame[381:408, 170:470] = 1.02
    ground = D.ground_mask(frame, camera, vertical)
    assert ground is not None and not ground[392, 320]
    assert _corridor(frame, intrinsics=camera, ground=ground)[0] < 1.5
    assert D.ground_mask(np.full_like(frame, 0.55), camera, vertical) is None
    frame[407:, :320] = np.nan
    assert D.ground_mask(frame, camera, vertical) is None


def test_ground_fit_recovers_floor_beside_a_box_that_reaches_the_bottom():
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[270:, 280:360] = .6
    camera = HorizontalIntrinsics(640, 315, 337)
    ground = D.ground_mask(frame, camera, (480, 420, 243.2))

    assert ground is not None
    assert ground[420, 120] and ground[420, 520]
    assert not ground[270:, 280:360].any()
    assert _corridor(frame, intrinsics=camera, ground=ground)[0] == pytest.approx(.6)


@pytest.mark.parametrize("occlusion", [slice(None, 220), slice(420, None)])
def test_ground_fit_uses_the_center_and_visible_side_beside_a_box(occlusion):
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[270:, occlusion] = .71
    camera = HorizontalIntrinsics(640, 315, 337)
    ground = D.ground_mask(frame, camera, (480, 420, 243.2))

    assert ground is not None and ground[420, 320]
    assert not ground[270:, occlusion].any()
    assert _corridor(frame, intrinsics=camera, ground=ground)[0] == pytest.approx(.71)


def test_ground_fit_accepts_floor_around_a_wider_center_box():
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[270:, 225:415] = .6
    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))

    assert ground is not None
    assert ground[420, 120] and ground[420, 520]
    assert not ground[270:, 225:415].any()


@pytest.mark.parametrize("mirror", [False, True])
def test_ground_fit_reseeds_from_the_visible_edge(mirror):
    from plugins.navi.camera import HorizontalIntrinsics

    ray_y = (np.arange(D.HEIGHT) - 229) / 492
    floor = np.minimum(1.45 / np.maximum(ray_y + .157, .001), 8.)
    frame = np.broadcast_to(floor[:, None], (D.HEIGHT, D.WIDTH)).astype(np.float32).copy()
    # Foreground covers one side and part of the centre. A nearer band
    # interrupts the floor in the far strips but leaves support above and below it.
    frame[270:, :210] = .85
    frame[360:, :252] = 1.6
    frame[350:390, 230:] = .85
    camera = HorizontalIntrinsics(640, 492, 319.5)
    depth = frame[:, ::-1] if mirror else frame

    ground = D.ground_mask(depth, camera, (480, 492, 229))

    assert ground is not None
    if mirror:
        ground = ground[:, ::-1]
    assert ground[440, 330] and ground[440, 515]
    assert not ground[360:, :252].any()
    assert not ground[350:390, 230:].any()


@pytest.mark.parametrize("mirror", [False, True])
def test_ground_fit_uses_a_continuous_side_window_around_a_wide_box(mirror):
    from plugins.navi.camera import HorizontalIntrinsics

    rays = (np.arange(D.HEIGHT) - 229) / 492 + .28
    floor = np.minimum(1.45 / np.maximum(rays, .001), 8.)
    frame = np.broadcast_to(floor[:, None], (D.HEIGHT, D.WIDTH)).astype(np.float32).copy()
    frame[280:, 272:] = 1.37
    camera = HorizontalIntrinsics(640, 492, 319.5)
    ground = D.ground_mask(frame[:, ::-1] if mirror else frame, camera, (480, 492, 229))

    assert ground is not None
    if mirror:
        ground = ground[:, ::-1]
    assert ground[440, 180] and ground[351, 192]
    assert not ground[280:, 272:].any()


def test_ground_fit_rejects_a_raised_side_window_with_lower_floor_farther_away():
    from plugins.navi.camera import HorizontalIntrinsics

    rays = (np.arange(D.HEIGHT) - 229) / 492 + .28
    floor = np.minimum(1.45 / np.maximum(rays, .001), 8.)
    frame = np.broadcast_to(floor[:, None], (D.HEIGHT, D.WIDTH)).astype(np.float32).copy()
    frame[280:, 272:] = 1.37
    frame[387:, :272] = (1.25 / rays[387:])[:, None]

    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 492, 319.5), (480, 492, 229))

    assert ground is None or not ground[387:, :272].any()


def test_visible_edge_seed_rejects_a_raised_platform():
    from plugins.navi.camera import HorizontalIntrinsics

    ray_y = np.maximum((np.arange(D.HEIGHT) - 229) / 492 + .157, .001)
    floor = np.minimum(1.60 / ray_y, 8.)
    frame = np.broadcast_to(floor[:, None], (D.HEIGHT, D.WIDTH)).astype(np.float32).copy()
    frame[350:, 210:] = (1.45 / ray_y[350:])[:, None]
    frame[270:, :210] = .85
    frame[360:, :252] = 1.6
    frame[358:390, 230:] = .85

    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 492, 319.5), (480, 492, 229))

    assert ground is None


def test_ground_fit_uses_single_visible_far_side_with_broad_near_support():
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    occluded = np.s_[287:401, :415]
    frame[occluded] = .73
    camera = HorizontalIntrinsics(640, 315, 337)
    ground = D.ground_mask(frame, camera, (480, 420, 243.2))

    assert ground is not None and ground[428, 200] and ground[342, 530]
    assert not ground[occluded].any()
    assert _corridor(frame, intrinsics=camera, ground=ground)[0] == pytest.approx(.73)


def test_ground_fit_recovers_from_an_unobstructed_patch_beside_a_near_box():
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    box = np.s_[270:, 139:312]
    frame[box] = .62
    camera = HorizontalIntrinsics(640, 315, 337)
    ground = D.ground_mask(frame, camera, (480, 420, 243.2))

    assert ground is not None and ground[425, 470]
    assert not ground[box].any()
    assert _corridor(frame, intrinsics=camera, ground=ground)[0] == pytest.approx(.62)


def test_ground_fit_keeps_sparse_missing_pixels_unclassified():
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    missing = np.s_[408:, 40:600:16]
    frame[missing] = np.nan
    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))

    assert ground is not None and ground[420, 320]
    assert not ground[missing].any()


@pytest.mark.parametrize(("height_m", "is_ground"), [(.018, True), (.021, False)])
def test_ground_mask_checks_the_two_centimetre_upper_tolerance(height_m, is_ground):
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[380:408, 160:480] *= (0.42 - height_m) / 0.42
    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))
    assert ground is not None
    assert np.all(ground[380:408, 160:480] == is_ground)


@pytest.mark.parametrize("raised_center", [False, True])
def test_floor_visible_between_two_near_walls(raised_center):
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[:, :205] = .7
    frame[:, 435:] = .9
    if raised_center:
        frame[385:, 205:435] *= .34 / .42
    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))

    if raised_center:
        assert ground is None or not ground[408:, 220:420].any()
    else:
        assert ground is not None
        assert ground[408:, 220:420].mean() > .95
        assert not ground[:, :205].any()
        assert not ground[:, 435:].any()


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("raised_strip", [False, True])
def test_floor_in_a_narrow_offset_strip_passes_only_with_far_support(mirror, raised_strip):
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[:, :196] = .7
    frame[:, 292:] = .9
    if raised_strip:
        frame[385:, 196:292] *= .34 / .42
    if mirror:
        frame = frame[:, ::-1]

    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))
    if raised_strip:
        assert ground is None
    else:
        assert ground is not None
        if mirror:
            ground = ground[:, ::-1]
        assert ground[408:, 196:292].mean() > .50
        assert not ground[:, :196].any()
        assert not ground[:, 292:].any()


@pytest.mark.parametrize("mirror", [False, True])
def test_floor_strip_at_the_edge_of_the_depth_image(mirror):
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[:, 108:] = .85
    if mirror:
        frame = frame[:, ::-1]
    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))
    assert ground is not None
    if mirror:
        ground = ground[:, ::-1]
    assert ground[408:, 40:108].all()
    assert not ground[:, 108:].any()


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("raised", [False, True])
def test_partial_far_floor_support_in_a_doorway_edge_strip(mirror, raised):
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    frame[:, :452] = .7
    # The near floor spans the outer quarter. Its outermost rays cannot see
    # the far floor, while the inner part still sees it beyond the jamb.
    frame[320:390, 564:] = .7
    if raised:
        frame[392:, 452:] *= .34 / .42
    if mirror:
        frame = frame[:, ::-1]

    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))

    if ground is not None and mirror:
        ground = ground[:, ::-1]
    if raised:
        assert ground is None or not ground[408:, 452:].any()
    else:
        assert ground is not None
        assert ground[420, 500] and ground[420, 580]
        assert not ground[:, :452].any()
        assert not ground[320:390, 564:].any()


@pytest.mark.parametrize("scene", ["wall", "raised_platform", "missing_background"])
@pytest.mark.parametrize("wall_near", [3.14, 2.80])
def test_broad_near_floor_ending_at_an_oblique_back_wall(scene, wall_near):
    from plugins.navi.camera import HorizontalIntrinsics

    # At this high mount, the back wall occludes most of the older far band.
    ray = (np.arange(480)[:, None] - 229) / 492 + .16
    floor = 1.45 / np.maximum(ray, .001)
    wall = np.linspace(wall_near, wall_near + .4, 640)[None, :]
    frame = np.minimum(floor, wall)
    if scene == "missing_background":
        frame[330:385] = np.nan
    elif scene == "raised_platform":
        frame[355:] = np.minimum(1.27 / ray[355:], wall)
        # Lower floor remains visible beyond the near platform.
        frame[331:355, :320] = floor[331:355]

    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 492, 319.5), (480, 492, 229))
    if scene == "wall":
        assert ground is not None and ground[408:].all()
        assert not ground[300:325].any()
    else:
        assert ground is None or not ground[408:].any()


def test_a_broad_raised_platform_does_not_replace_the_floor():
    from plugins.navi.camera import HorizontalIntrinsics

    frame = _floor_scene()
    # A raised horizontal surface spans the whole fitting band, but the
    # visible floor in the distance lies on a different plane.
    frame[355:] *= 0.34 / 0.42
    camera = HorizontalIntrinsics(640, 315, 337)
    assert D.ground_mask(frame, camera, (480, 420, 243.2)) is None


def test_outside_seed_does_not_use_an_off_center_raised_platform_as_floor():
    from plugins.navi.camera import HorizontalIntrinsics

    rays = (np.arange(D.HEIGHT)[:, None] - 243.2) / 420.0
    lateral = (np.arange(D.WIDTH)[None, :] - 337) / 315.0
    top = .34 / np.maximum(rays - .02, .001)
    raised = (lateral * top <= .40) & (top > .52) & (top < 1.9)
    frame = np.where(raised, top, _floor_scene())
    ground = D.ground_mask(frame, HorizontalIntrinsics(640, 315, 337), (480, 420, 243.2))

    assert ground is None or not ground[raised].any()


@pytest.mark.parametrize("has_floor", [False, True])
def test_grid_classifies_measured_cells_without_requiring_ground_or_vertical_probes(monkeypatch, has_floor):
    from plugins.navi import local_path
    from plugins.navi.camera import HorizontalIntrinsics

    geom = local_path.camera_square(HorizontalIntrinsics(640, 420, 320),
                                    (480, 420, 243), .20, at_front=True)
    image = np.full((480, 640), np.nan, dtype=np.float32)
    image[:, 100:200] = np.linspace(.70, 2.30, 480, dtype=np.float32)[:, None]
    image[:, 300:340] = 1.4
    ground = None
    if has_floor:
        ground = np.zeros_like(image, dtype=bool)
        ground[:, 100:200] = True
        monkeypatch.setattr(local_path, "optical_ground_plane",
                            lambda *_: (np.array([0., 1., 0.]), .42))

    grid_args = geom, local_path.RIGHT, local_path.FORWARD, local_path.RESOLUTION
    free, occupied = local_path.observe(image, *grid_args, ground)
    row, col = local_path.START
    assert free.any()
    assert occupied[row + 28, col]
    assert not (free & occupied).any()
    rows, cols = np.nonzero(np.isfinite(image))
    measured_depth = image[rows, cols].astype(float)
    measured_right = measured_depth * (cols - 320) / 420
    cell_rows = row + np.rint(measured_depth / local_path.RESOLUTION).astype(int)
    cell_cols = col + np.rint(measured_right / local_path.RESOLUTION).astype(int)
    visible = np.zeros_like(free)
    visible[cell_rows, cell_cols] = True
    assert np.array_equal(free | occupied, visible)
    assert not local_path.observe(image, *grid_args, ground, check_free_space=False)[0].any()


def test_floor_only_corridor_reports_its_observed_free_extent():
    from plugins.navi.camera import HorizontalIntrinsics

    row_tangent = (np.arange(D.HEIGHT) - 243.2) / 420.0
    ground_depth = 0.55 / np.maximum(row_tangent + 0.23, 0.001)
    frame = np.broadcast_to(ground_depth[:, None], (D.HEIGHT, D.WIDTH)).astype(np.float32)
    frame[ground_depth > 12] = np.nan
    camera = HorizontalIntrinsics(640, 315, 337)
    ground = D.ground_mask(frame, camera, (480, 420, 243.2))
    assert ground is not None
    clearance, coverage = _corridor(frame, intrinsics=camera, ground=ground)
    assert clearance is not None and clearance > 0.95
    assert coverage > 0.9


def test_only_received_immutable_pixels_can_share_a_projection():
    from plugins.navi.local_path import DepthProjection

    frame = D.decode(_encode(_solid(3.)))
    ground = np.zeros(frame.shape, dtype=bool)
    projection = DepthProjection.received(frame, ground)
    snapshot = dict(map=frame, ground=projection.ground, grid_projection=projection)
    assert DepthProjection.for_observation(snapshot) is projection
    with pytest.raises(ValueError):
        frame.setflags(write=True)
    ground[:] = True
    assert not projection.ground.any()
    mutable = frame.copy()
    mutable_projection = DepthProjection.received(mutable, ground)
    assert DepthProjection.for_observation(dict(map=mutable, ground=ground,
                                                grid_projection=mutable_projection)) is not mutable_projection


def test_repeated_depth_revisions_preserve_pixel_order_and_mutable_edits():
    from plugins.navi.local_path import array_revision

    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    frozen = np.frombuffer(pixels.tobytes(), dtype=pixels.dtype).reshape(pixels.shape)
    for view in (frozen, frozen[1:], frozen.T, frozen[:, ::-1], pixels):
        assert array_revision(view) == (view.shape, view.dtype.str, hash(view.tobytes()))
    revision = array_revision(pixels)
    pixels[0, 0] += 1
    assert array_revision(pixels) != revision


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("height", [.19, .21])
def test_metric_clearance_uses_the_same_height_threshold_as_the_planning_grid(wide, height):
    from plugins.navi import local_path, policy
    from plugins.navi.camera import HorizontalIntrinsics

    camera = HorizontalIntrinsics(640, 315, 337)
    vertical = (480, 420, 243.2)
    geometry = local_path.camera_square(camera, vertical, .275)
    frame = _floor_scene()
    columns = slice(180, 480) if wide else slice(318, 350)
    rows = slice(388, 402)
    frame[rows, columns] *= (.42 - height) / .42
    ground = D.ground_mask(frame, camera, vertical)
    assert ground is not None
    config = policy.Config(local_geometry=geometry, depth_intrinsics=camera)
    depth = {"map": frame, "ground": ground}
    raw, coverage = _corridor(frame, ground=ground, intrinsics=camera,
                              half_width_m=policy._keepout_m(config),
                              reference_m=config.obstacle_stop_m)
    assert raw < 1.
    clearance, measured = policy._clearance(depth, config)
    grid = local_path.project_observation(depth, geometry)
    assert bool(grid[1].any()) is (height >= .20)
    if height >= .20:
        obstacle = frame[rows, columns]
        assert obstacle.min() <= clearance <= obstacle.max()
    else:
        assert clearance > 1.5
    assert measured == coverage
