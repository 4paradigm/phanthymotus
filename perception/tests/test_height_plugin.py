"""tests/test_height_plugin.py — the `height` card's wiring and refusals.

The geometry itself is covered in test_height_math.py; this file feeds fake ROS
messages through the real plugin and asserts what a caller sees: the start
gate, the two-input binding, the measure/calibrate replies, and the cases where
the card must say "no" instead of returning a number.
"""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _FakeString,
)

import plugins.height as height_plugin  # noqa: E402
from plugins import height_math as geometry  # noqa: E402

W, H = geometry.DEPTH_WIDTH, geometry.DEPTH_HEIGHT
K = {"fx": 603.85986328125, "fy": 603.5826416015625,
     "cx": 324.97772216796875, "cy": 250.24855041503906}
DEPTH_TOPIC = "/bumi/camera/depth"
POSE_TOPIC = "/bumi/camera/color/poses"
DEPTH_DECL = {
    "schema": "motus.camera/1", "topic": DEPTH_TOPIC,
    "format": "image/depth-zlib", "width": W, "height": H,
    "K": [K["fx"], 0.0, K["cx"], 0.0, K["fy"], K["cy"], 0.0, 0.0, 1.0],
}
POSE_DECL = {
    "schema": "motus.camera/1", "topic": POSE_TOPIC, "format": "data/json",
    "width": W, "height": H, "K": None,
}


def _plugin(**cfg):
    return height_plugin.HeightPlugin(cfg, _FakeExecutor())


def _start(plugin, *, detection=None, camera_info=None, instance="card-1"):
    topics = [DEPTH_TOPIC] + ([detection] if detection else [])
    return plugin.dispatch("height", {
        "action": "start", "instance_id": instance,
        "input_topics": topics, "camera_info": camera_info or {},
    })


def _runtime_camera_info(*, detection_too=False):
    info = {DEPTH_TOPIC: DEPTH_DECL}
    if detection_too:
        info[POSE_TOPIC] = POSE_DECL
    return info


def _floor_depth(camera_height_m: float = 1.0, max_range_m: float = 6.0):
    depth = np.full((H, W), np.nan, dtype=np.float32)
    for v in range(int(K["cy"]) + 1, H):
        z = camera_height_m * K["fy"] / (v - K["cy"])
        if 0.3 <= z <= max_range_m:
            depth[v, :] = z
    return depth


def _person_frame(person_height_m=1.70, distance_m=3.0,
                  camera_height_m=1.0):
    depth = _floor_depth(camera_height_m)
    v_top = int(round(K["cy"] + K["fy"] *
                      (camera_height_m - person_height_m) / distance_m))
    v_bottom = int(round(K["cy"] + K["fy"] * camera_height_m / distance_m))
    bbox = [200, v_top, 440, min(H - 1, v_bottom)]
    depth[bbox[1]:bbox[3], bbox[0]:bbox[2]] = distance_m
    return depth, bbox


def _feed_depth(node, depths, interval=0.01):
    payloads = [geometry.encode_depth(depth) for depth in depths]

    def run():
        for payload in payloads:
            node._on_depth(_FakeCompressedImage(
                payload, "16UC1; compressedDepth zlib"))
            time.sleep(interval)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _set_pose(node, bbox, track_id=1):
    payload = {"timestamp": time.time(), "count": 1,
               "persons": [{"id": track_id, "bbox": [float(v) for v in bbox]}]}
    message = _FakeString()
    message.data = json.dumps(payload)
    node._on_detection(message)


def _node(plugin, instance="card-1"):
    return plugin._instances[instance]["node"]


# ── tool contract ───────────────────────────────────────────────────────────

def test_tool_is_a_measurement_card_with_user_actions():
    tool = height_plugin.TOOLS[0]
    assert height_plugin.HeightPlugin.PREFIX == "height"
    assert tool["name"] == "height"
    assert tool["type"] == "processor"
    assert tool["multiInstance"] is False
    enum = tool["inputSchema"]["properties"]["action"]["enum"]
    assert "info" in enum
    assert "measure" in enum and "calibrate" in enum
    assert set(tool["inputSchema"]["x-action-params"]) == {"measure", "calibrate"}
    formats = [port["format"] for port in tool["topic_in"]]
    assert formats == ["image/depth-zlib", "data/json"]
    assert tool["topic_out"][0]["format"] == "data/json"


def test_schema_defaults_match_the_documented_config_defaults():
    properties = height_plugin.TOOLS[0]["configSchema"]["properties"]
    for key, value in height_plugin._CONFIG_DEFAULTS.items():
        if key in properties:
            assert properties[key]["default"] == value, key


# ── start gates ─────────────────────────────────────────────────────────────

def test_start_without_a_depth_topic_is_refused():
    result = _plugin().dispatch("height", {
        "action": "start", "instance_id": "c",
        "input_topics": [POSE_TOPIC],
    })
    assert result["state"] == "error"
    assert result["reason"] == "no_depth_input"


def test_start_without_intrinsics_is_refused_with_an_actionable_reason():
    result = _start(_plugin())
    assert result["state"] == "error"
    assert result["reason"] == "no_camera_info"
    assert "restart" in result["message"] or "intrinsics" in result["message"]


def test_start_binds_depth_and_detection_and_reports_the_declaration():
    plugin = _plugin()
    result = _start(plugin, detection=POSE_TOPIC,
                    camera_info=_runtime_camera_info(detection_too=True))
    assert result["state"] == "running"
    assert result["intrinsics"]["source"] == "runtime-K"
    assert result["inputs"] == {"depth": DEPTH_TOPIC, "detection": POSE_TOPIC}
    info = plugin.dispatch("height", {"action": "info", "instance_id": "card-1"})
    assert info["state"] == "running"
    assert [port["topic"] for port in info["topic_in"]] == [DEPTH_TOPIC, POSE_TOPIC]
    assert info["topic_out"][0]["topic"] == "/bumi/height/last"


def test_manual_config_intrinsics_unblock_a_camera_that_declares_none():
    plugin = _plugin(fx=600.0, fy=601.0, cx=320.0, cy=240.0)
    result = _start(plugin)
    assert result["state"] == "running"
    assert result["intrinsics"]["source"] == "config-manual"


def test_config_action_validates_and_applies():
    plugin = _plugin()
    assert plugin.dispatch("height", {
        "action": "config", "camera_height_m": 1.05,
        "camera_pitch_deg": -2.5})["status"] == "configured"
    assert plugin._cfg["camera_height_m"] == pytest.approx(1.05)
    rejected = plugin.dispatch("height", {
        "action": "config", "camera_height_m": 9.0})
    assert rejected["status"] == "error"


# ── measure ─────────────────────────────────────────────────────────────────

def test_measure_a_person_from_pose_boxes_and_the_floor_plane():
    plugin = _plugin(frames=2)
    result = _start(plugin, detection=POSE_TOPIC,
                    camera_info=_runtime_camera_info(detection_too=True))
    assert result["state"] == "running"
    depth, bbox = _person_frame(1.70)
    _set_pose(_node(plugin), bbox)
    _feed_depth(_node(plugin), [depth, depth, depth, depth])
    measured = plugin.dispatch("height", {"action": "measure",
                                          "instance_id": "card-1"})
    assert measured["state"] == "ok"
    assert measured["height_m"] == pytest.approx(1.70, abs=0.03)
    assert measured["mode"] == "floor"
    assert measured["frames_used"] >= 2
    assert measured["target"]["kind"] == "person"
    assert measured["height_definition"] == "floor_to_top_of_head"


def test_measure_with_a_manual_bbox_and_a_calibrated_mount():
    plugin = _plugin(frames=2, camera_height_m=1.0)
    assert _start(plugin, camera_info=_runtime_camera_info())["state"] == "running"
    depth, bbox = _person_frame(1.70)
    normalized = [bbox[0] / W, bbox[1] / H, bbox[2] / W, bbox[3] / H]
    # No floor: a wall/blank depth frame forces mount mode.
    blank = np.full((H, W), 3.0, dtype=np.float32)
    _feed_depth(_node(plugin), [blank, blank, blank, blank])
    measured = plugin.dispatch("height", {
        "action": "measure", "instance_id": "card-1",
        "bbox": normalized})
    assert measured["state"] == "ok", measured
    assert measured["mode"] == "mount"
    assert measured["height_m"] == pytest.approx(1.70, abs=0.03)


def test_measure_without_detection_or_bbox_is_a_reason():
    plugin = _plugin(frames=2)
    assert _start(plugin, camera_info=_runtime_camera_info())["state"] == "running"
    _feed_depth(_node(plugin), [_floor_depth(), _floor_depth()])
    measured = plugin.dispatch("height", {"action": "measure",
                                          "instance_id": "card-1"})
    assert measured["state"] == "error"
    assert measured["reason"] == "no_detection"


def test_a_clipped_detection_is_refused_before_any_number():
    plugin = _plugin(frames=2)
    _start(plugin, detection=POSE_TOPIC,
           camera_info=_runtime_camera_info(detection_too=True))
    _set_pose(_node(plugin), [100, 0, 300, 470])
    _feed_depth(_node(plugin), [_floor_depth()] * 4)
    measured = plugin.dispatch("height", {"action": "measure",
                                          "instance_id": "card-1"})
    assert measured["state"] == "error"
    assert measured["reason"] == "target_clipped"


def test_a_stale_detection_is_refused():
    plugin = _plugin(frames=2)
    _start(plugin, detection=POSE_TOPIC,
           camera_info=_runtime_camera_info(detection_too=True))
    _set_pose(_node(plugin), [180, 100, 440, 470])
    _node(plugin)._detection_arrival -= 10.0
    _feed_depth(_node(plugin), [_floor_depth()] * 4)
    measured = plugin.dispatch("height", {"action": "measure",
                                          "instance_id": "card-1"})
    assert measured["state"] == "error"
    assert measured["reason"] == "stale_detection"


def test_frames_that_disagree_are_unstable_not_averaged_into_a_wrong_number():
    plugin = _plugin(frames=2)
    _start(plugin, detection=POSE_TOPIC,
           camera_info=_runtime_camera_info(detection_too=True))
    depth, bbox = _person_frame(1.70)
    _set_pose(_node(plugin), bbox)
    shifted = depth.copy()
    shifted[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 2.8
    _feed_depth(_node(plugin), [depth, shifted, depth, shifted])
    measured = plugin.dispatch("height", {"action": "measure",
                                          "instance_id": "card-1"})
    assert measured["state"] == "error"
    assert measured["reason"] == "unstable"


def test_measure_when_idle_is_not_running():
    result = _plugin().dispatch("height", {"action": "measure",
                                           "instance_id": "card-1"})
    assert result["state"] == "error"
    assert result["reason"] == "not_running"


# ── calibrate / stop / info ─────────────────────────────────────────────────

def test_calibrate_reads_the_floor_height_and_pitch():
    plugin = _plugin(calibration_frames=3)
    assert _start(plugin, camera_info=_runtime_camera_info())["state"] == "running"
    _feed_depth(_node(plugin), [_floor_depth(1.0)] * 6)
    result = plugin.dispatch("height", {"action": "calibrate",
                                        "instance_id": "card-1"})
    assert result["state"] == "ok", result
    assert result["camera_height_m"] == pytest.approx(1.0, abs=0.02)
    assert result["camera_pitch_deg"] == pytest.approx(0.0, abs=1.0)
    assert result["suggested_config"]["camera_height_m"] == \
        pytest.approx(1.0, abs=0.02)


def test_calibrate_without_a_floor_says_so():
    plugin = _plugin(calibration_frames=3)
    _start(plugin, camera_info=_runtime_camera_info())
    _feed_depth(_node(plugin), [np.full((H, W), 3.0, dtype=np.float32)] * 5)
    result = plugin.dispatch("height", {"action": "calibrate",
                                        "instance_id": "card-1"})
    assert result["state"] == "error"
    assert result["reason"] == "no_ground_reference"


def test_stop_retires_the_node_and_info_returns_idle():
    plugin = _plugin()
    assert _start(plugin, camera_info=_runtime_camera_info())["state"] == "running"
    node = _node(plugin)
    assert plugin.dispatch("height", {"action": "stop"})["state"] == "idle"
    assert node.destroyed is True
    info = plugin.dispatch("height", {"action": "info"})
    assert info["state"] == "idle"
    assert "topic_in" not in info


def test_info_is_answerable_before_start():
    info = _plugin().dispatch("height", {"action": "info"})
    assert info["state"] == "idle"
    assert info["config"]["frames"] == 5
