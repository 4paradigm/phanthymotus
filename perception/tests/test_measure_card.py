"""Contract tests for the generic measurement card.

No ROS: the card is started only far enough to exercise binding and refusal.
The geometry is covered in test_measure_geometry.py; this file checks the
contract an operator sees on the canvas and the degradation messages.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.measure import MeasurePlugin  # noqa: E402
from plugins.measure import geometry  # noqa: E402
from plugins.measure import plugin as measure_plugin  # noqa: E402


def _card(**cfg):
    return MeasurePlugin(cfg, executor=None)


class _FakeNode:
    def __init__(self):
        self.subscriptions = []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))
        return object()

    def get_topic_names_and_types(self):
        return []


def _bind(**over):
    card = _card()
    node = _FakeNode()
    topics = ["/cam/objects", "/cam/visual_depth"]
    args = {"topics": topics, "bindings": None}
    args.update(over)
    bound, problem = card._bind_inputs(
        node, args["topics"], None, args["bindings"])
    return card, node, bound, problem


# ── tool contract ────────────────────────────────────────────────────────────

def test_the_card_declares_one_processor_with_both_inputs():
    card = _card()
    tool = card.get_tools()[0]
    assert tool["name"] == "measure"
    assert tool["type"] == "processor"
    assert tool["multiInstance"] is False
    assert [port["topic"] for port in tool["topic_out"]] == [
        measure_plugin.DEFAULT_TOPIC, "/perception/measure/view"]
    assert [port["format"] for port in tool["topic_out"]] == [
        "data/json", "image/jpeg"]
    formats = {port["format"] for port in tool["topic_in"]}
    assert formats == {"data/json", "image/depth-zlib"}


def test_start_and_stop_are_lifecycle_not_model_functions():
    tool = _card().get_tools()[0]
    params = tool["inputSchema"]["x-action-params"]
    assert "start" not in params and "stop" not in params
    assert "info" in params


def test_info_returns_topic_out_for_the_dashboard():
    out = _card().dispatch("measure", {"action": "info"})
    assert [port["topic"] for port in out["topic_out"]] == [
        measure_plugin.DEFAULT_TOPIC, "/perception/measure/view"]


def test_config_is_bounded_and_reports_the_effective_values():
    card = _card()
    out = card.dispatch("measure", {"action": "config", "rate_hz": 2})
    assert out["status"] == "configured"
    assert out["config"]["rate_hz"] == 2.0
    assert out["changed"] == {"rate_hz": 2.0}


def test_an_invalid_config_does_not_leave_the_card_changed():
    card = _card()
    card.dispatch("measure", {"action": "config", "rate_hz": 2})
    out = card.dispatch("measure", {"action": "config", "rate_hz": 0})
    assert out["status"] == "error"
    assert card._config.rate_hz == 2.0


# ── start refusals ───────────────────────────────────────────────────────────

def test_no_ros_context_is_refused():
    out = _card().dispatch("measure", {
        "action": "start", "input_topics": ["/o", "/d"]})
    assert out["state"] == "error"
    assert "ROS" in out["message"]


def test_missing_inputs_are_refused():
    out = _card().dispatch("measure", {"action": "start", "input_topics": ["/o"]})
    assert out["state"] == "error"
    assert "ROS" in out["message"]


# ── input binding ────────────────────────────────────────────────────────────

def test_both_roles_are_bound_by_suffix():
    _, node, bound, problem = _bind()
    assert problem == ""
    assert bound["objects"] == "/cam/objects"
    assert bound["depth"] == "/cam/visual_depth"
    assert {topic for _, topic, _, _ in node.subscriptions} == {
        "/cam/objects", "/cam/visual_depth", "/cam"}


def test_a_driver_depth_topic_is_accepted_without_renaming():
    _, _, bound, problem = _bind(topics=["/cam/objects", "/camera/depth"])
    assert problem == ""
    assert bound["depth"] == "/camera/depth"


def test_missing_depth_is_refused():
    _, _, _, problem = _bind(topics=["/cam/objects"])
    assert "深度" in problem


def test_a_retired_depth_format_is_refused_not_decoded():
    _, _, _, problem = _bind(bindings={
        "/cam/visual_depth": {"format": "image/depth-z16"}})
    assert "image/depth-zlib" in problem


# ── camera adoption ──────────────────────────────────────────────────────────

def _declaration(topic, width=640, height=480, camera_id="cam"):
    return {
        "schema": "motus.camera/1", "topic": topic, "id": camera_id,
        "width": width, "height": height,
        "K": [500.0, 0, width / 2, 0, 500.0, height / 2, 0, 0, 1],
    }


def test_camera_geometry_is_adopted_from_the_depth_topic():
    card = _card()
    out = card._adopt_camera(
        {"/cam/visual_depth": _declaration("/cam/visual_depth")},
        {"objects": "/cam/objects", "depth": "/cam/visual_depth"})
    assert out == ""
    assert card._camera.fx == 500.0
    assert card._camera.has_width and card._camera.has_height


def test_two_cameras_are_refused():
    card = _card()
    out = card._adopt_camera({
        "/cam/visual_depth": _declaration("/cam/visual_depth", camera_id="a"),
        "/cam/objects": _declaration("/cam/objects", camera_id="b"),
    }, {"objects": "/cam/objects", "depth": "/cam/visual_depth"})
    assert "不同的镜头" in out


def test_missing_camera_is_visible_in_degraded():
    card = _card()
    card._running = True
    out = card._adopt_camera({}, {"objects": "/o", "depth": "/d"})
    assert "motus.camera/1" in " ".join(card._degradations())


# ── payload ──────────────────────────────────────────────────────────────────

def _depth():
    frame = np.full((geometry.HEIGHT, geometry.WIDTH), 2.0, dtype=np.float32)
    return frame


def _objects():
    return {"objects": [{
        "name": "box", "confidence": 0.9,
        "bbox_norm": [0.4, 0.4, 0.6, 0.6],
        "color": "red",
    }]}


def _payload_card():
    card = _card(stable_frames=1, stable_min=1)
    card._running = True
    card._objects = _objects()
    card._objects_ms = int(time.time() * 1000)
    card._depth = _depth()
    card._depth_ms = int(time.time() * 1000)
    card._objects_frame = (640, 480)
    card._camera = measure_plugin.camera_mod.from_declaration(
        _declaration("/d"))
    return card


def test_payload_contains_distance_visible_extents_and_no_fake_depth():
    payload = _payload_card()._payload()
    assert payload["schema"] == measure_plugin.SCHEMA
    assert payload["fresh"] is True
    assert payload["fresh"] is True
    item = payload["objects"][0]
    assert item["distance_cm"] == 200.0
    assert item["visible_w_cm"] == 51.2
    assert item["visible_h_cm"] == 38.4
    assert item["label"] == "box 51.2x38.4cm @2.00m"
    assert "session" not in payload
    assert "camera" not in payload
    assert "degraded" not in payload


def test_stale_inputs_are_published_as_unavailable_not_as_old_numbers():
    card = _payload_card()
    card._objects_ms = 0
    payload = card._payload()
    assert payload["fresh"] is False
    assert payload["fresh"] is False
    assert payload["objects"] == []


def test_it_starts_without_opening_a_real_ros_node():
    card = _card()
    node = _FakeNode()
    card._executor = object()
    card._open_publisher = lambda: setattr(card, "_node", node)

    out = card.dispatch("measure", {
        "action": "start",
        "input_topics": ["/cam/objects", "/cam/visual_depth"],
        "camera_info": {
            "/cam/visual_depth": _declaration("/cam/visual_depth"),
            "/cam/objects": _declaration("/cam/objects"),
        },
    })
    assert out["state"] == "running"
    assert out["inputs"]["objects"] == "/cam/objects"
    assert out["inputs"]["depth"] == "/cam/visual_depth"
    assert card._camera.has_width
    assert {topic for _, topic, _, _ in node.subscriptions} == {
        "/cam/objects", "/cam/visual_depth", "/cam"}


def test_payload_is_json_serialisable():
    payload = _payload_card()._payload()
    json.dumps(payload)


# ── session summary / history ────────────────────────────────────────────────

def test_info_carries_a_compact_session_summary_after_an_object_is_measured():
    card = _payload_card()
    card._payload()
    payload = card.dispatch("measure", {"action": "info"})
    assert payload["session"]["objects"]
    item = payload["session"]["objects"][0]
    assert item["key"].startswith("box")
    assert item["n"] == 1
    assert item["visible_h_cm"]["latest"] == 38.4
    assert "last" not in payload


def test_history_action_returns_the_bounded_records():
    card = _payload_card()
    card._payload()
    out = card.dispatch("measure", {"action": "history"})
    assert out["state"] == "running"
    assert len(out["records"]) == 1
    assert out["records"][0]["h_cm"] == 38.4
    assert out["session"]["objects"]


def test_reset_session_clears_summary_without_stopping_the_card():
    card = _payload_card()
    card._payload()
    out = card.dispatch("measure", {"action": "reset_session"})
    assert out["state"] == "reset"
    assert out["session"]["objects"] == []
    assert card._running is True


def test_info_includes_the_session_summary_and_history_count():
    card = _payload_card()
    card._payload()
    out = card.dispatch("measure", {"action": "info"})
    assert out["session"]["objects"]
    assert out["history_count"] == 1


def test_a_vanished_object_is_not_counted_again_by_the_session_summary():
    card = _payload_card()
    card._payload()
    # The stabilizer retains the track for track_timeout_ms; the session must
    # not treat that retained record as a new observation.
    card._objects = {"objects": []}
    payload = card._payload()
    item = card.dispatch("measure", {"action": "info"})["session"]["objects"][0]
    assert item["n"] == 1


def test_depth_wire_size_stays_640_even_if_camera_declares_other_resolution():
    card = _card()
    declaration = _declaration("/cam/visual_depth", width=1280, height=720)
    out = card._adopt_camera(
        {"/cam/visual_depth": declaration},
        {"objects": "/cam/objects", "depth": "/cam/visual_depth"})
    assert out == ""
    assert card._depth_frame == (640, 480)
    assert card._camera.width == 1280
    assert any("640x480" in note for note in card._camera_notes)


def test_compact_payload_does_not_include_retained_tracks():
    card = _payload_card()
    first = card._payload()
    assert first["count"] == 1
    card._objects = {"objects": []}
    second = card._payload()
    assert second["count"] == 0
    assert second["objects"] == []


def test_camera_topics_are_derived_from_the_depth_sibling():
    assert _card()._camera_candidates("/ns/camera/depth") == [
        "/ns/camera/color", "/ns/camera/rgb"]
    assert _card()._camera_candidates("/g1/camera/depth") == [
        "/g1/camera/color", "/g1/camera/rgb"]
    assert _card()._camera_candidates("/cam/visual_depth") == [
        "/cam", "/cam/color", "/cam/rgb"]
    assert _card()._camera_candidates("/camera/head/depth") == [
        "/camera/head/color", "/camera/head/rgb", "/camera/head"]
    assert _card()._camera_candidates("/not-a-depth") == []


def test_the_card_declares_the_overlay_port():
    tool = _card().get_tools()[0]
    assert any(port["topic"] == "/perception/measure/view"
               and port["format"] == "image/jpeg" for port in tool["topic_out"])


def test_read_returns_only_the_current_frame_and_filters_target():
    card = _payload_card()
    card._payload()
    out = card.dispatch("measure", {"action": "read"})
    assert out["state"] == "running"
    assert out["fresh"] is True
    assert [item["name"] for item in out["objects"]] == ["box"]
    filtered = card.dispatch("measure", {"action": "read", "target": "cup"})
    assert filtered["objects"] == []
    assert card.dispatch("measure", {"action": "read", "target": "box"})["objects"]
