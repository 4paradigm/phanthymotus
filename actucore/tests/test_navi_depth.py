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


def test_a_corridor_full_of_holes_does_not_read_as_an_empty_one():
    """`clearance` alone cannot express this: every invalid pixel drops silently
    out of the minimum, so a corridor nobody could measure and a corridor with
    nothing in it produce the same number. The things that make holes — chair
    legs, glass, dark edges — are the things that catch a shoulder."""
    depth_map = np.full((D.HEIGHT, D.WIDTH), np.nan, dtype=np.float32)
    clearance, coverage = _corridor(depth_map)
    assert clearance is None
    assert coverage == 0.0


def test_coverage_is_graduated_rather_than_a_flag():
    solid = _corridor(np.full((D.HEIGHT, D.WIDTH), 5.0, dtype=np.float32))[1]
    patchy = np.full((D.HEIGHT, D.WIDTH), 5.0, dtype=np.float32)
    patchy[::2, :] = np.nan
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
