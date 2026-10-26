"""Geometry and depth-contract tests for the generic measurement card.

No ROS: these are the pure functions the card runs at 5 Hz. The depth buffer is
built by hand from the `image/depth-zlib` contract, so a producer changing the
layout fails here before it reaches a robot.
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.measure import camera as C  # noqa: E402
from plugins.measure import geometry as G  # noqa: E402


def _encode(metres):
    mm = np.rint(np.nan_to_num(metres, nan=0.0) * 1000.0)
    mm[(mm < 1) | (mm > 65535)] = 0
    return zlib.compress(mm.astype("<u2").tobytes(), 1)


def _solid(value=2.0):
    return np.full((G.HEIGHT, G.WIDTH), float(value), dtype=np.float32)


def _camera(fx=500.0, fy=500.0):
    return C.from_declaration({
        "schema": C.SCHEMA,
        "id": "test/camera",
        "width": G.WIDTH,
        "height": G.HEIGHT,
        "K": [fx, 0, G.WIDTH / 2.0, 0, fy, G.HEIGHT / 2.0, 0, 0, 1],
    })


# ── depth contract ───────────────────────────────────────────────────────────

def test_depth_round_trip_is_metres():
    depth = G.decode(_encode(_solid(2.5)))
    assert depth.shape == (G.HEIGHT, G.WIDTH)
    assert np.allclose(depth, 2.5, atol=1e-3)


def test_zero_is_invalid_not_zero_metres():
    frame = _solid(2.0)
    frame[100:200, 100:200] = 0.0
    depth = G.decode(_encode(frame))
    assert np.isnan(depth[150, 150])
    assert not np.isnan(depth[10, 10])


def test_short_buffer_is_refused():
    with pytest.raises(G.DepthError, match="expected"):
        G.decode(zlib.compress(b"\x00" * 1000))


def test_a_box_with_no_valid_depth_returns_none():
    sample = G.sample_box(G.decode(_encode(np.zeros_like(_solid()))),
                          (0.4, 0.4, 0.6, 0.6))
    assert sample.distance_m is None
    assert sample.valid_ratio == 0.0


# ── camera geometry ──────────────────────────────────────────────────────────

def test_K_is_used_for_width_and_height():
    camera = _camera()
    assert camera.has_width and camera.has_height
    assert camera.fx == pytest.approx(500.0)
    assert camera.fy == pytest.approx(500.0)
    assert camera.source == "derived-from-K"
    assert camera.half_fov_h == pytest.approx(np.arctan(320 / 500), rel=1e-6)


def test_declared_angles_derive_intrinsics():
    camera = C.from_declaration({
        "schema": C.SCHEMA, "width": 640, "height": 480,
        "half_fov_rad": 0.5, "half_fov_v_rad": 0.4,
    })
    assert camera.fx == pytest.approx(320 / np.tan(0.5), rel=1e-6)
    assert camera.fy == pytest.approx(240 / np.tan(0.4), rel=1e-6)
    assert camera.source == "declared-angle"


def test_missing_camera_geometry_is_reported_not_guessed():
    camera = C.from_declaration({}, fallback_width=640, fallback_height=480)
    assert not camera.has_width
    assert not camera.has_height
    assert camera.notes


def test_config_fallback_is_allowed_but_named():
    camera = C.from_declaration({}, fallback_width=640, fallback_height=480,
                                fallback_half_fov=0.5,
                                fallback_half_fov_v=0.4)
    assert camera.source == "config-angle"
    assert camera.has_width and camera.has_height


# ── bbox normalisation ───────────────────────────────────────────────────────

def test_pixel_bbox_is_normalised_using_the_detector_frame():
    box = G.normalise_bbox({"bbox": [256, 192, 384, 288]}, 640, 480)
    assert box.norm == pytest.approx((0.4, 0.4, 0.6, 0.6))
    assert box.clipped is False


def test_a_bbox_touching_the_edge_is_marked_as_a_lower_bound():
    box = G.normalise_bbox({"bbox": [-10, 100, 300, 400]}, 640, 480)
    assert box.clipped is True
    assert box.norm[0] == 0.0


def test_no_bbox_is_not_guessed_from_the_depth_shape():
    box = G.normalise_bbox({"name": "box"}, 640, 480)
    assert box.norm is None
    assert "bbox" in box.reason


# ── measurements ─────────────────────────────────────────────────────────────

def test_visible_extents_come_from_depth_and_intrinsics():
    depth = G.decode(_encode(_solid(2.0)))
    record = G.measure_object(
        {"name": "box", "confidence": 0.9,
         "bbox_norm": [0.4, 0.4, 0.6, 0.6]},
        depth, _camera(), (640, 480))
    assert record["distance_m"] == pytest.approx(2.0)
    # 0.2 of 640 px = 128 px; 2 m * 128 / 500 = 0.512 m
    assert record["visible_width_m"] == pytest.approx(0.512)
    # 0.2 of 480 px = 96 px; 2 m * 96 / 500 = 0.384 m
    assert record["visible_height_m"] == pytest.approx(0.384)
    assert record["extent_3d_m"] is None
    assert record["method"] == "bbox+depth"


def test_pixel_bbox_produces_the_same_measurement_as_normalised_bbox():
    depth = G.decode(_encode(_solid(2.0)))
    camera = _camera()
    by_pixels = G.measure_object(
        {"name": "box", "bbox": [256, 192, 384, 288]}, depth, camera, (640, 480))
    by_norm = G.measure_object(
        {"name": "box", "bbox_norm": [0.4, 0.4, 0.6, 0.6]}, depth, camera, (640, 480))
    assert by_pixels["visible_width_m"] == pytest.approx(by_norm["visible_width_m"])
    assert by_pixels["visible_height_m"] == pytest.approx(by_norm["visible_height_m"])


def test_no_camera_geometry_still_reports_distance_and_null_dimensions():
    depth = G.decode(_encode(_solid(3.0)))
    camera = C.from_declaration({}, fallback_width=640, fallback_height=480)
    record = G.measure_object(
        {"name": "box", "bbox_norm": [0.4, 0.4, 0.6, 0.6]},
        depth, camera, (640, 480))
    assert record["distance_m"] == pytest.approx(3.0)
    assert record["visible_width_m"] is None
    assert record["visible_height_m"] is None
    assert "camera geometry" in record["quality"]["note"]


def test_invalid_depth_leaves_distance_and_dimensions_null():
    depth = G.decode(_encode(np.zeros_like(_solid())))
    record = G.measure_object(
        {"name": "box", "bbox_norm": [0.4, 0.4, 0.6, 0.6]},
        depth, _camera(), (640, 480))
    assert record["distance_m"] is None
    assert record["visible_width_m"] is None
    assert "no valid depth" in record["quality"]["note"]


def test_a_clipped_box_is_a_lower_bound_not_a_precise_width():
    depth = G.decode(_encode(_solid(2.0)))
    record = G.measure_object(
        {"name": "box", "bbox_norm": [-0.1, 0.4, 0.2, 0.6]},
        depth, _camera(), (640, 480))
    assert record["clipped"] is True
    assert "lower bounds" in record["quality"]["note"]


def test_beyond_max_depth_is_not_reported_as_a_size():
    depth = G.decode(_encode(_solid(30.0)))
    record = G.measure_object(
        {"name": "box", "bbox_norm": [0.4, 0.4, 0.6, 0.6]},
        depth, _camera(), (640, 480), max_depth_m=20.0)
    assert record["distance_m"] is None
    assert "beyond max_depth_m" in record["quality"]["note"]


# ── stabilisation ────────────────────────────────────────────────────────────

def _record(width, center=0.5, name="box"):
    return {"name": name, "color": "red", "confidence": 0.9,
            "bbox_norm": [center - 0.1, 0.4, center + 0.1, 0.6],
            "visible_width_m": width, "visible_height_m": 0.4,
            "distance_m": 2.0, "depth_span_m": 0.1,
            "clipped": False, "approximate": False,
            "method": "bbox+depth", "quality": {}}


def test_stabilizer_quotes_the_median_not_the_last_frame():
    stabilizer = G.Stabilizer(window=3, min_samples=3, timeout_s=2.0)
    for index, width in enumerate((0.50, 0.60, 0.70)):
        out = stabilizer.update([_record(width)], index * 0.1)
    assert len(out) == 1
    assert out[0]["visible_width_m"] == pytest.approx(0.60)
    assert out[0]["samples"] == 3
    assert out[0]["stable"] is True


def test_different_names_do_not_merge():
    stabilizer = G.Stabilizer(window=3, min_samples=1, timeout_s=2.0)
    out = stabilizer.update([_record(0.5, name="a"), _record(0.7, name="b")], 0.0)
    assert {item["name"] for item in out} == {"a", "b"}


def test_a_lost_track_is_dropped_after_the_timeout():
    stabilizer = G.Stabilizer(window=3, min_samples=1, timeout_s=0.5)
    stabilizer.update([_record(0.5)], 0.0)
    assert stabilizer.update([], 0.2)
    assert stabilizer.update([], 0.6) == []


# ── session summary / bounded history ────────────────────────────────────────

def _session_record(height=1.7, width=0.5, key="person#", stable=True,
                    clipped=False):
    return {"key": key, "name": "person", "color": "", "stable": stable,
            "visible_height_m": height, "visible_width_m": width,
            "distance_m": 2.0, "clipped": clipped}


def test_session_summary_reports_min_max_median_latest():
    session = G.Session(history_size=10, record_interval_ms=0,
                        record_min_delta_m=0.0, only_stable=True)
    session.reset(1000)
    session.record([_session_record(1.60)], 1000)
    session.record([_session_record(1.80)], 1100)
    session.record([_session_record(1.70)], 1200)
    summary = session.summary(1300)
    item = summary["objects"][0]
    assert item["n"] == 3
    assert item["visible_h_cm"] == {
        "min": 160.0, "max": 180.0, "median": 170.0, "latest": 170.0}
    assert summary["duration_s"] == 0.3


def test_session_ignores_unstable_measurements_by_default():
    session = G.Session(only_stable=True)
    session.reset(0)
    session.record([_session_record(1.7, stable=False)], 1)
    assert session.summary(2)["objects"] == []


def test_session_can_record_unstable_when_configured():
    session = G.Session(only_stable=False)
    session.reset(0)
    session.record([_session_record(1.7, stable=False)], 1)
    assert session.summary(2)["objects"][0]["n"] == 1


def test_history_is_rate_limited_but_a_real_change_is_kept():
    session = G.Session(history_size=10, record_interval_ms=5000,
                        record_min_delta_m=0.05)
    session.reset(0)
    session.record([_session_record(1.70)], 1000)
    session.record([_session_record(1.70)], 2000)
    session.record([_session_record(1.76)], 2500)
    rows = session.history()
    assert len(rows) == 2
    assert rows[0]["h_cm"] == 170.0
    assert rows[1]["h_cm"] == 176.0


def test_session_reset_clears_summary_and_history():
    session = G.Session()
    session.reset(0)
    session.record([_session_record()], 1)
    assert session.history()
    session.reset(100)
    assert session.history() == []
    assert session.summary(101)["objects"] == []


def test_session_does_not_count_a_retained_track_as_a_new_observation():
    """Stabilizer keeps a vanished object for track_timeout_ms; session must not
    turn that retained record into another sample."""
    session = G.Session(only_stable=True)
    session.reset(0)
    observed = _session_record(1.7)
    observed["age_ms"] = 0
    observed["observed"] = True
    retained = _session_record(1.7)
    retained["age_ms"] = 400
    retained["observed"] = False
    session.record([observed], 1000)
    session.record([retained], 1400)
    item = session.summary(1400)["objects"][0]
    assert item["n"] == 1
    assert item["last_ms"] == 1000


def test_session_retains_at_most_max_objects():
    session = G.Session(max_objects=2, record_interval_ms=0)
    session.reset(0)
    for index, name in enumerate(("a", "b", "c")):
        session.record([{**_session_record(1.7, key=f"{name}#"),
                         "name": name}], index)
    summary = session.summary(10)
    assert len(summary["objects"]) == 2
    assert session._objects.keys() == {"b#", "c#"}
    assert session._last_history.keys() == {"b#", "c#"}


def test_a_bbox_touching_the_top_edge_is_clipped():
    box = G.normalise_bbox({"bbox_norm": [0.1, 0.0, 0.3, 0.4]}, 640, 480)
    assert box.clipped is True


def test_stabilizer_only_marks_the_track_it_updated_this_tick():
    stabilizer = G.Stabilizer(window=3, min_samples=1, timeout_s=2.0)
    stabilizer.update([_record(0.5, center=0.2), _record(0.6, center=0.8)], 0.0)
    out = stabilizer.update([_record(0.5, center=0.2)], 0.1)
    assert [item["observed"] for item in out] == [True, False]
