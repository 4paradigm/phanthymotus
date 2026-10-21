"""tests/test_height_math.py — pure geometry for the height card (no ROS).

These tests are the reviewable part: every formula the card depends on is
exercised against a synthetic depth map whose answer is known analytically.
The plugin tests then only have to prove the card wires these functions to a
topic correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plugins import height_math as geometry

W, H = geometry.DEPTH_WIDTH, geometry.DEPTH_HEIGHT
K = {"fx": 603.85986328125, "fy": 603.5826416015625,
     "cx": 324.97772216796875, "cy": 250.24855041503906}


def _floor_depth(camera_height_m: float, max_range_m: float = 6.0) -> np.ndarray:
    """A level camera looking forward at a floor `camera_height_m` below it.

    Pixel v>cy sees the floor at z = h*fy/(v-cy); rows above the horizon have
    no floor and stay NaN, which is exactly what a real frame of a room with a
    wall behind the floor looks like to the fit.
    """
    depth = np.full((H, W), np.nan, dtype=np.float32)
    for v in range(int(K["cy"]) + 1, H):
        z = camera_height_m * K["fy"] / (v - K["cy"])
        if 0.3 <= z <= max_range_m:
            depth[v, :] = z
    return depth


def _person_frame(camera_height_m: float = 1.0,
                  person_height_m: float = 1.70,
                  distance_m: float = 3.0) -> tuple:
    """Floor frame plus an opaque person rectangle standing on that floor."""
    depth = _floor_depth(camera_height_m)
    v_top = int(round(K["cy"] + K["fy"] * (camera_height_m - person_height_m) / distance_m))
    v_bottom = int(round(K["cy"] + K["fy"] * camera_height_m / distance_m))
    bbox = [200, v_top, 440, min(H - 1, v_bottom)]
    depth[bbox[1]:bbox[3], bbox[0]:bbox[2]] = distance_m
    return depth, bbox, v_top


# ── depth codec ──────────────────────────────────────────────────────────────

def test_depth_roundtrip_is_millimetres_with_invalid_as_nan():
    depth = np.zeros((H, W), dtype=np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = 1.234
    depth[1, 0] = np.nan
    depth[1, 1] = 3.5
    payload = geometry.encode_depth(depth)
    decoded = geometry.decode_depth(payload)
    assert decoded.shape == (H, W)
    assert decoded[0, 0] != decoded[0, 0]          # NaN: 0 means no reading
    assert decoded[0, 1] == pytest.approx(1.234, abs=1e-3)
    assert decoded[1, 0] != decoded[1, 0]
    assert decoded[1, 1] == pytest.approx(3.5, abs=1e-3)


def test_wrong_depth_size_is_refused_not_reshaped():
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.decode_depth(b"\x00" * 100)
    assert caught.value.reason == "bad_depth"


def test_garbage_depth_payload_is_refused():
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.decode_depth(b"not zlib")
    assert caught.value.reason == "bad_depth"


# ── intrinsics ───────────────────────────────────────────────────────────────

def _declaration(**overrides):
    declaration = {"schema": "motus.camera/1", "topic": "/cam/depth",
                   "format": "image/depth-zlib", "width": W, "height": H,
                   "K": [K["fx"], 0.0, K["cx"], 0.0, K["fy"], K["cy"],
                         0.0, 0.0, 1.0]}
    declaration.update(overrides)
    return declaration


def test_runtime_k_wins_and_notes_a_disagreeing_manual_config():
    intrinsics = geometry.resolve_intrinsics(
        _declaration(), {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0})
    assert intrinsics["source"] == "runtime-K"
    assert intrinsics["fx"] == pytest.approx(K["fx"])
    assert intrinsics["notes"]                      # disagreement is surfaced


def test_manual_intrinsics_are_the_fallback_when_the_camera_declares_none():
    intrinsics = geometry.resolve_intrinsics(
        _declaration(K=None), {"fx": 600.0, "fy": 601.0, "cx": 320.0, "cy": 240.0})
    assert intrinsics["source"] == "config-manual"
    assert (intrinsics["fx"], intrinsics["fy"]) == (600.0, 601.0)


def test_fov_derived_intrinsics_are_opt_in_only():
    declaration = _declaration(K=None, half_fov_rad=math.radians(27.4),
                               half_fov_v_rad=math.radians(21.25))
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.resolve_intrinsics(declaration, {})
    assert caught.value.reason == "no_camera_info"

    intrinsics = geometry.resolve_intrinsics(
        declaration, {"allow_fov_intrinsics": True})
    assert intrinsics["source"] == "derived-from-fov"
    assert intrinsics["fx"] == pytest.approx((W / 2) / math.tan(math.radians(27.4)))
    assert intrinsics["fy"] == pytest.approx((H / 2) / math.tan(math.radians(21.25)))


def test_declaration_for_another_picture_size_is_refused():
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.resolve_intrinsics(_declaration(width=1280, height=720), {})
    assert caught.value.reason == "intrinsics_size_mismatch"


# ── target selection ─────────────────────────────────────────────────────────

def _pose_payload(*boxes):
    return {"timestamp": 1.0, "count": len(boxes),
            "persons": [{"id": index + 1, "bbox": list(box)}
                        for index, box in enumerate(boxes)]}


def _vop_payload(*items):
    return {"timestamp": 1.0, "count": len(items), "objects": list(items)}


def test_pose_boxes_are_picked_by_area_and_rescaled_to_the_depth_picture():
    payload = _pose_payload([100, 120, 300, 470], [400, 100, 520, 300])
    target = geometry.pick_detection(payload, "pose", "person",
                                     (1280, 960), (W, H))
    assert target["kind"] == "person"
    assert target["candidate_count"] == 2
    assert target["bbox_px"][0] == pytest.approx(50.0)
    assert target["bbox_px"][2] == pytest.approx(150.0)


def test_two_similar_targets_are_ambiguous_rather_than_guessed():
    payload = _pose_payload([100, 120, 300, 470], [330, 120, 530, 470])
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.pick_detection(payload, "pose", "person", (W, H), (W, H))
    assert caught.value.reason == "ambiguous_target"


def test_a_clipped_top_is_refused_because_it_reads_short():
    payload = _pose_payload([100, 0, 300, 470])
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.pick_detection(payload, "pose", "person", (W, H), (W, H))
    assert caught.value.reason == "target_clipped"


def test_a_far_target_is_too_far_not_a_height():
    payload = _pose_payload([100, 200, 200, 240])
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.pick_detection(payload, "pose", "person", (W, H), (W, H))
    assert caught.value.reason == "too_far"


def test_vop_target_can_be_filtered_by_class_name():
    payload = _vop_payload(
        {"name": "chair", "bbox": [100, 200, 300, 470]},
        {"name": "box", "bbox": [350, 300, 450, 470]})
    target = geometry.pick_detection(payload, "vop", "box", (W, H), (W, H))
    assert target["label"] == "box"
    assert target["bbox_px"][0] == pytest.approx(350)


def test_no_detection_for_the_requested_class_is_reported():
    payload = _vop_payload({"name": "chair", "bbox": [100, 200, 300, 470]})
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.pick_detection(payload, "vop", "box", (W, H), (W, H))
    assert caught.value.reason == "target_not_found"


# ── top point and height ────────────────────────────────────────────────────

def test_top_point_undoes_the_pixels_skipped_for_a_valid_depth():
    depth = np.full((H, W), 3.0, dtype=np.float32)
    bbox = [200, 100, 440, 470]
    top = geometry.top_point(depth, bbox, K)
    expected = (100 - K["cy"]) * 3.0 / K["fy"]
    assert top["y_top"] == pytest.approx(expected, abs=1e-3)
    assert top["z"] == pytest.approx(3.0)
    assert top["valid_ratio"] == pytest.approx(1.0)


def test_top_point_refuses_a_target_with_no_valid_depth():
    depth = np.full((H, W), np.nan, dtype=np.float32)
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.top_point(depth, [200, 100, 440, 470], K)
    assert caught.value.reason == "no_depth_at_target"


def test_mount_mode_recovers_a_known_height():
    point = (0.0, -0.7, 3.0)   # 0.7 m above the camera, 3 m ahead
    assert geometry.height_from_mount(point, 1.0, 0.0, 0.0) == pytest.approx(1.7)


def test_mount_mode_compensates_the_mounting_pitch():
    pitch = math.radians(5.0)
    # The camera-frame coordinates of a point that is 0.7 m above the camera
    # and 3 m in front of it, when the camera is pitched 5 degrees down.
    y_g, z_g = -0.7, 3.0
    y_cam = y_g * math.cos(pitch) - z_g * math.sin(pitch)
    z_cam = y_g * math.sin(pitch) + z_g * math.cos(pitch)
    assert geometry.height_from_mount((0.0, y_cam, z_cam), 1.0,
                                      pitch, 0.0) == pytest.approx(1.7, abs=1e-9)


def test_floor_plane_from_a_synthetic_floor_is_where_it_was_put():
    plane = geometry.floor_plane_from_depth(_floor_depth(1.0), K,
                                            max_range_m=5.0)
    assert plane["height_m"] == pytest.approx(1.0, abs=0.02)
    assert plane["pitch_rad"] == pytest.approx(0.0, abs=0.01)
    assert plane["roll_rad"] == pytest.approx(0.0, abs=0.01)
    assert plane["residual_m"] < 0.01


def test_height_above_the_fitted_floor_is_the_person_height():
    depth, bbox, _ = _person_frame(1.0, 1.70, 3.0)
    plane = geometry.floor_plane_from_depth(depth, K, max_range_m=5.0,
                                            exclude_bbox=bbox)
    top = geometry.top_point(depth, bbox, K)
    height = geometry.height_above_plane(top["point"], plane)
    assert height == pytest.approx(1.70, abs=0.02)


def test_no_floor_in_view_is_a_reason_not_a_number():
    depth = np.full((H, W), 3.0, dtype=np.float32)
    with pytest.raises(geometry.GeometryError) as caught:
        geometry.floor_plane_from_depth(depth, K)
    assert caught.value.reason == "no_ground_reference"


# ── plumbing ────────────────────────────────────────────────────────────────

def test_result_topic_keeps_the_robot_namespace():
    assert geometry.output_topic_for("/bumi/camera/depth") == "/bumi/height/last"
    assert geometry.output_topic_for("/cam/visual_depth") == "/cam/height/last"
    assert geometry.output_topic_for("") == "/perception/height/last"


def test_median_and_spread_rejects_an_empty_sample():
    with pytest.raises(geometry.GeometryError):
        geometry.median_and_spread([])
    assert geometry.median_and_spread([1.7, 1.72, 1.68])[0] == pytest.approx(1.7)
    assert geometry.median_and_spread([1.7, 1.72, 1.68])[1] == pytest.approx(0.04)
