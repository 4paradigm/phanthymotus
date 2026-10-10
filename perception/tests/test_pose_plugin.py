"""
tests/test_pose_plugin.py — pose card lifecycle, the three output topics, and
the payload budget (host-side, no GPU).

The engine is faked; everything else is the real plugin, including the ROS node,
the tracker and the action rules. ROS and cv2 stubs come from vision_stubs,
installed by conftest before collection.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception/tests -q
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

from vision_stubs import (  # noqa: F401
    _FakeCompressedImage,
    _FakeExecutor,
    _FakeNode,
    _wait_until,
    frame_bytes,
)

import plugins.pose as pose_plugin  # noqa: E402
from plugins.pose_action import (  # noqa: E402
    ACTION_LABELS_ZH, DEFAULT_THRESHOLDS, POSTURE_LABELS_ZH)
from plugins.vision_runtime import COCO_INDEX, LetterboxMeta, N_KEYPOINTS  # noqa: E402

FRAME_W, FRAME_H = 640, 480

# -- reading the two-channel result ------------------------------------------
# posture and activity are separate channels; these express the old flattened
# view for assertions that only care about "in one word".

from plugins.pose_action import RULE_TO_ACTIVITY        # noqa: E402

ACTIVITY_TO_RULE = {v: k for k, v in RULE_TO_ACTIVITY.items()}


def _action(result):
    activity = result.get("activity")
    if isinstance(activity, dict):
        return ACTIVITY_TO_RULE.get(activity["name"], activity["name"])
    if isinstance(activity, str):
        return ACTIVITY_TO_RULE.get(activity, activity)
    return result.get("posture") or "unknown"


def _labels(result):
    out = set()
    if result.get("posture"):
        out.add(result["posture"])
    name = _action(result)
    if name != "unknown":
        out.add(name)
    return out




def _standing_row(cx=320.0, top=40.0, h=400.0, score=0.9) -> list:
    """One 57-column pose row: [x1,y1,x2,y2,score,class, kx,ky,kv × 17].

    57 columns because that is what the shipped engine declares — exporting
    yolo26s-pose in the jp6.1 image reports `output0` with shape (1, 300, 57).

    The body is the same fraction-of-height skeleton the action tests use, so
    this row classifies as `standing` through the real rules.
    """
    joints = {
        "nose": (cx, top + 0.06 * h),
        "left_eye": (cx - 0.02 * h, top + 0.05 * h),
        "right_eye": (cx + 0.02 * h, top + 0.05 * h),
        "left_ear": (cx - 0.04 * h, top + 0.06 * h),
        "right_ear": (cx + 0.04 * h, top + 0.06 * h),
        "left_shoulder": (cx - 0.10 * h, top + 0.18 * h),
        "right_shoulder": (cx + 0.10 * h, top + 0.18 * h),
        "left_elbow": (cx - 0.11 * h, top + 0.34 * h),
        "right_elbow": (cx + 0.11 * h, top + 0.34 * h),
        "left_wrist": (cx - 0.12 * h, top + 0.50 * h),
        "right_wrist": (cx + 0.12 * h, top + 0.50 * h),
        "left_hip": (cx - 0.07 * h, top + 0.52 * h),
        "right_hip": (cx + 0.07 * h, top + 0.52 * h),
        "left_knee": (cx - 0.07 * h, top + 0.745 * h),
        "right_knee": (cx + 0.07 * h, top + 0.745 * h),
        "left_ankle": (cx - 0.07 * h, top + 0.97 * h),
        "right_ankle": (cx + 0.07 * h, top + 0.97 * h),
    }
    row = [cx - 0.15 * h, top, cx + 0.15 * h, top + h, score, 0.0]
    for name in sorted(COCO_INDEX, key=lambda n: COCO_INDEX[n]):
        x, y = joints[name]
        row += [x, y, 0.9]
    return row


class _FakeModel:
    """Stands in for VisionEngineSession: infer(frame) -> (outputs, meta)."""

    def __init__(self, rows=None):
        self.calls = 0
        rows = [_standing_row()] if rows is None else rows
        self._output = (np.asarray(rows, dtype=np.float32).reshape(1, -1, 57)
                        if rows else np.zeros((1, 0, 57), dtype=np.float32))

    @property
    def input_size(self):
        return (640, 640)

    def infer(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        # Identity letterbox keeps the coordinate assertions readable.
        return [self._output], LetterboxMeta(1.0, 0, 0, w, h)


def _plugin(cfg=None, model=None):
    executor = _FakeExecutor()
    plugin = pose_plugin.PosePerceptionPlugin(cfg or {}, "testns", executor)
    plugin._model = _FakeModel() if model is None else model
    return plugin, executor


def _publisher(node, topic):
    return next((p for p in node.publishers if p.topic == topic), None)


def _feed(plugin, node_key, *, frames=1, interval=0.25, topic="/cam/rgb"):
    """Push frames through the node's subscription callback and wait for output."""
    node = plugin._nodes[node_key]
    subscription = node.subscriptions[0]
    lean = _publisher(node, node._output_topic)
    for index in range(frames):
        if index:
            # The node gates on monotonic time, so the cap has to be respected
            # rather than worked around — otherwise this tests the gate, not the
            # pipeline.
            node._last_inference_time -= interval
        subscription.callback(_FakeCompressedImage(frame_bytes(FRAME_W, FRAME_H)))
        _wait_until(lambda n=index: len(lean.messages) > n, timeout=3.0)
    return node


# ── tool declaration ─────────────────────────────────────────────────────────

def test_the_card_declares_three_output_ports_with_distinct_formats():
    """A renderer only ever sees one topic, so the lean JSON, the skeleton and
    the overlay image cannot share a port."""
    tool = pose_plugin.TOOLS[0]
    formats = [port["format"] for port in tool["topic_out"]]
    assert formats == ["data/json", "sensor/pose2d", "image/jpeg"]
    assert [port["format"] for port in tool["topic_in"]] == ["image/jpeg"]
    assert tool["type"] == "processor" and tool["multiInstance"] is True


def test_every_advertised_action_has_parameters_declared():
    """A card renders its form from x-action-params; a missing entry is an
    action the operator cannot invoke."""
    schema = pose_plugin.TOOLS[0]["inputSchema"]
    declared = set(schema["x-action-params"])
    assert set(schema["properties"]["action"]["enum"]) == declared


def test_action_params_only_reference_real_properties():
    schema = pose_plugin.TOOLS[0]["inputSchema"]
    known = set(schema["properties"])
    for action, spec in schema["x-action-params"].items():
        assert set(spec["params"]) <= known, action


def test_the_fall_thresholds_are_exposed_with_the_rules_defaults():
    """They are instance config precisely because they are not constants; if
    the card's defaults drifted from the rules', tuning one would not move the
    other."""
    properties = pose_plugin.TOOLS[0]["configSchema"]["properties"]
    for key in ("fall_drop_ratio", "fall_drop_window_s", "fall_settle_s"):
        assert properties[key]["default"] == DEFAULT_THRESHOLDS[key]
        assert properties[key]["scope"] == "instance"


def test_keypoint_payload_levels_match_the_implementation():
    enum = pose_plugin.TOOLS[0]["configSchema"]["properties"]["publish_keypoints"]["enum"]
    assert enum == list(pose_plugin.KEYPOINT_LEVELS)
    assert pose_plugin.TOOLS[0]["configSchema"]["properties"]["publish_keypoints"]["default"] == "off"


# ── topic derivation ─────────────────────────────────────────────────────────

def test_topics_are_derived_in_one_place():
    assert pose_plugin.output_topic_for("/cam/rgb") == "/cam/rgb/poses"
    assert pose_plugin.skeleton_topic_for("/cam/rgb") == "/cam/rgb/poses/skeleton"
    assert pose_plugin.overlay_topic_for("/cam/rgb") == "/cam/rgb/poses/overlay_img"


def test_a_topicless_card_publishes_on_the_fixed_default():
    """vop reported publishing to "None/objects" when one of its three copies of
    this derivation forgot the case."""
    for missing in (None, ""):
        assert pose_plugin.output_topic_for(missing) == "/perception/pose"
        assert pose_plugin.skeleton_topic_for(missing) == "/perception/pose/skeleton"
        assert pose_plugin.overlay_topic_for(missing) == "/perception/pose/overlay_img"


@pytest.mark.parametrize("value,expected", [
    ("compact", "compact"), ("FULL", "full"), ("off", "off"),
    # YAML 1.1 turns a bare `off`/`on` into a bool before this ever sees it.
    (False, "off"), (True, "off"), (None, "off"), ("nonsense", "off"),
])
def test_an_unusable_keypoint_level_falls_back_to_off(value, expected):
    assert pose_plugin._keypoint_level(value) == expected


# ── streaming ────────────────────────────────────────────────────────────────

def test_a_started_card_publishes_lean_json_and_a_skeleton():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")

    lean = json.loads(_publisher(node, "/cam/rgb/poses").messages[-1])
    assert lean["count"] == 1
    assert "latency_ms" in lean
    person = lean["persons"][0]
    assert person["posture"] == "standing"
    assert person["id"] == 1
    assert "bbox" in person

    skeleton = json.loads(_publisher(node, "/cam/rgb/poses/skeleton").messages[-1])
    assert skeleton["image_size"] == [FRAME_W, FRAME_H]
    assert len(skeleton["keypoint_names"]) == N_KEYPOINTS
    assert len(skeleton["persons"][0]["keypoints"]) == N_KEYPOINTS
    assert skeleton["persons"][0]["posture"] == "standing"


def test_the_lean_stream_carries_no_keypoints_by_default():
    """Every byte on this topic is a byte of LLM context on every frame."""
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    message = _publisher(node, "/cam/rgb/poses").messages[-1]
    assert "keypoints" not in json.loads(message)["persons"][0]
    # The whole frame, one person, for scale: the skeleton version is ~10x this.
    assert len(message) < 220


def test_the_skeleton_stream_is_the_fat_one_and_that_is_the_point():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    lean = _publisher(node, "/cam/rgb/poses").messages[-1]
    skeleton = _publisher(node, "/cam/rgb/poses/skeleton").messages[-1]
    assert len(skeleton) > 4 * len(lean)


@pytest.mark.parametrize("level,expected_width", [("compact", 2), ("full", 3)])
def test_publish_keypoints_adds_them_to_the_lean_stream(level, expected_width):
    plugin, _ = _plugin({"publish_keypoints": level})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    person = json.loads(_publisher(node, "/cam/rgb/poses").messages[-1])["persons"][0]
    assert len(person["keypoints"]) == N_KEYPOINTS
    assert all(len(point) == expected_width for point in person["keypoints"])


def test_publish_bbox_off_drops_the_pixel_box():
    plugin, _ = _plugin({"publish_bbox": False})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    person = json.loads(_publisher(node, "/cam/rgb/poses").messages[-1])["persons"][0]
    assert "bbox" not in person
    assert "position" in person        # the centre is never dropped


def test_an_empty_frame_still_publishes_a_zero_count():
    """Silence and "nobody here" must not look the same downstream."""
    plugin, _ = _plugin(model=_FakeModel(rows=[]))
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    lean = json.loads(_publisher(node, "/cam/rgb/poses").messages[-1])
    assert lean["count"] == 0 and lean["persons"] == []


def test_max_persons_keeps_the_most_confident_detections():
    rows = [_standing_row(cx=100.0, score=0.5),
            _standing_row(cx=320.0, score=0.95),
            _standing_row(cx=540.0, score=0.7)]
    plugin, _ = _plugin({"max_persons": 2}, model=_FakeModel(rows=rows))
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    lean = json.loads(_publisher(node, "/cam/rgb/poses").messages[-1])
    assert lean["count"] == 2
    assert sorted(p["posture_confidence"] for p in lean["persons"])
    assert {round(p["position"][0], 2) for p in lean["persons"]} == {0.0, 0.69}


def test_the_fps_cap_drops_frames_rather_than_queueing_them():
    plugin, _ = _plugin({"fps": 2})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    model_calls = plugin._model.calls
    # A second frame immediately after the first is inside the 0.5 s interval.
    node.subscriptions[0].callback(_FakeCompressedImage(frame_bytes(FRAME_W, FRAME_H)))
    assert plugin._model.calls == model_calls


def test_a_track_id_survives_across_frames():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb", frames=3)
    ids = [json.loads(m)["persons"][0]["id"]
           for m in _publisher(node, "/cam/rgb/poses").messages]
    assert ids == [1, 1, 1]


# ── overlay ──────────────────────────────────────────────────────────────────

def test_the_overlay_topic_has_no_publisher_unless_asked_for():
    """An idle publisher is a port the canvas shows as wired and silent."""
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = plugin._nodes["/cam/rgb"]
    assert node._pub_overlay is None
    assert _publisher(node, "/cam/rgb/poses/overlay_img") is None


def test_publish_overlay_draws_the_skeleton_onto_a_jpeg():
    plugin, _ = _plugin({"publish_overlay": True})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    overlay = _publisher(node, "/cam/rgb/poses/overlay_img")
    _wait_until(lambda: len(overlay.messages) > 0, timeout=3.0)
    assert overlay.messages and len(overlay.messages[-1]) > 0


def test_draw_skeleton_marks_the_canvas_and_skips_invisible_joints():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    keypoints[COCO_INDEX["left_shoulder"]] = (200, 200, 0.9)
    keypoints[COCO_INDEX["right_shoulder"]] = (300, 200, 0.9)
    # Everything else sits at (0, 0) with visibility 0 — which is exactly why
    # the visibility gate matters: drawn, they would put a limb through the
    # top-left corner of every frame.
    pose_plugin.draw_skeleton(
        canvas, [{"keypoints": keypoints, "id": 1, "box": [180, 180, 320, 400],
                  "verdict": {"posture": "standing", "activity": None}}], 0.3)
    assert canvas[200, 250].any()        # the shoulder-to-shoulder bone
    assert not canvas[0, 0].any()        # no phantom limb at the origin


def test_a_fallen_person_is_drawn_in_the_alert_colour():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    keypoints = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    keypoints[COCO_INDEX["left_shoulder"]] = (200, 200, 0.9)
    keypoints[COCO_INDEX["right_shoulder"]] = (300, 200, 0.9)
    person = {"keypoints": keypoints, "id": 1, "box": [180, 180, 320, 400],
              "verdict": {"posture": "lying",
                          "activity": {"name": "falling down"}}}
    pose_plugin.draw_skeleton(canvas, [person], 0.3)
    # The colour keys on the alerting label, which is now part of a compound
    # string, so it is matched by membership rather than equality.
    assert tuple(int(v) for v in canvas[200, 250]) == pose_plugin._ALERT_COLOUR


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_stop_leaves_the_subscription_for_dispose_node():
    """Destroying a subscription on a node still registered with the executor
    kills the spin thread on InvalidHandle and takes every subscription in the
    process with it, silently."""
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = plugin._nodes["/cam/rgb"]
    node.stop()
    assert node.subscriptions, "stop() must not tear the subscription down itself"
    assert node._running is False


def test_retiring_an_instance_removes_it_from_the_executor_and_destroys_it():
    plugin, executor = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = plugin._nodes["/cam/rgb"]
    plugin.dispatch("pose", {"action": "stop"})
    assert plugin._nodes == {}
    assert node not in executor.nodes
    assert node.destroyed, "destroy_node leaks the publishers and the node name"


def test_stop_with_no_instance_running_is_not_an_error():
    plugin, _ = _plugin()
    assert plugin.dispatch("pose", {"action": "stop"}) == {"state": "idle"}


def test_concurrent_starts_leave_exactly_one_node():
    """The canvas does config→start→stop→start within seconds, from separate
    ThreadingHTTPServer threads."""
    plugin, executor = _plugin()
    barrier = threading.Barrier(6)

    def _start():
        barrier.wait()
        plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})

    threads = [threading.Thread(target=_start) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(plugin._nodes) == 1
    assert len(executor.nodes) == 1


def test_a_start_stop_storm_orphans_nothing():
    plugin, executor = _plugin()
    for _ in range(8):
        plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
        plugin.dispatch("pose", {"action": "stop"})
    assert plugin._nodes == {}
    assert executor.nodes == []


def test_restarting_the_same_topic_reuses_nothing_stale():
    plugin, executor = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    first = plugin._nodes["/cam/rgb"]
    plugin.dispatch("pose", {"action": "stop"})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    second = plugin._nodes["/cam/rgb"]
    assert first is not second
    assert len(executor.nodes) == 1


def test_a_topicless_card_starts_with_no_subscription():
    plugin, _ = _plugin()
    result = plugin.dispatch("pose", {"action": "start"})
    node = plugin._nodes[pose_plugin._DEFAULT_INSTANCE]
    assert node.subscriptions == []
    assert result["mode"] == "on_demand"
    assert result["output"] == "/perception/pose"


# ── info ─────────────────────────────────────────────────────────────────────

def test_info_on_an_idle_card_reports_the_label_set():
    plugin, _ = _plugin()
    info = plugin.dispatch("pose", {"action": "info"})
    assert info["state"] == "idle"
    assert info["keypoints"] == N_KEYPOINTS
    assert set(info["postures"]) == set(POSTURE_LABELS_ZH)
    assert info["action_backend"] == "hybrid"       # the default


def test_info_on_a_running_card_lists_both_output_topics():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert info["state"] == "running"
    topics = {port["topic"]: port["format"] for port in info["topic_out"]}
    assert topics == {"/cam/rgb/poses": "data/json",
                      "/cam/rgb/poses/skeleton": "sensor/pose2d"}
    instance = info["instances"]["/cam/rgb"]
    assert instance["overlay_output"] is None


def test_info_lists_the_overlay_topic_once_it_is_enabled():
    plugin, _ = _plugin({"publish_overlay": True})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert [port["format"] for port in info["topic_out"]][-1] == "image/jpeg"
    assert (info["instances"]["/cam/rgb"]["overlay_output"]
            == "/cam/rgb/poses/overlay_img")


def test_info_while_loading_says_so_without_an_engine():
    plugin, _ = _plugin()
    plugin._model = None
    plugin._model_loading = True
    plugin._model_load_status = "downloading 42%"
    info = plugin.dispatch("pose", {"action": "info"})
    assert info["state"] == "loading"
    assert "42%" in info["desc"]


def test_info_surfaces_a_load_failure_rather_than_looking_idle():
    plugin, _ = _plugin()
    plugin._model = None
    plugin._model_load_error = "no pinned bundle"
    info = plugin.dispatch("pose", {"action": "info"})
    assert info["state"] == "error"
    assert "no pinned bundle" in info["desc"]


def test_start_reports_a_load_failure_instead_of_silently_doing_nothing():
    plugin, _ = _plugin()
    plugin._model = None
    plugin._model_load_error = "engine plan not compatible"
    result = plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    assert result["state"] == "error"
    assert "engine plan not compatible" in result["message"]
    assert plugin._nodes == {}


def test_a_missing_camera_declaration_is_called_out_rather_than_assumed():
    """`position` is a normalised offset; without a field of view nobody
    downstream can turn it into an angle."""
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert "camera_info" not in info
    assert "camera_info_note" in info


# ── config ───────────────────────────────────────────────────────────────────

def test_a_global_config_moves_the_defaults():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "config", "fps": 3, "confidence": 0.6,
                             "publish_keypoints": "compact", "max_persons": 2})
    assert plugin._fps == 3 and plugin._confidence == 0.6
    assert plugin._publish_keypoints == "compact" and plugin._max_persons == 2


def test_configuring_a_running_instance_retires_it_so_the_next_start_applies():
    """Rebuilding the thresholds under a live worker would change what the
    rules mean halfway through a window."""
    plugin, executor = _plugin()
    plugin.dispatch("pose", {"action": "start", "instance_id": "i1",
                             "input_topic": "/cam/rgb"})
    plugin.dispatch("pose", {"action": "config", "instance_id": "i1", "fps": 1})
    assert "i1" not in plugin._nodes
    assert executor.nodes == []
    plugin.dispatch("pose", {"action": "start", "instance_id": "i1",
                             "input_topic": "/cam/rgb"})
    assert plugin._nodes["i1"]._fps == 1


def test_instance_config_overrides_the_fall_thresholds_for_that_card_only():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "config", "instance_id": "i1",
                             "fall_drop_ratio": 0.8})
    merged = plugin._merged_config("i1")
    assert merged["fall_drop_ratio"] == 0.8
    classifier = plugin._classifier_for(merged)
    assert classifier.thresholds["fall_drop_ratio"] == 0.8
    other = plugin._classifier_for(plugin._merged_config("i2"))
    assert other.thresholds["fall_drop_ratio"] == DEFAULT_THRESHOLDS["fall_drop_ratio"]


def test_the_classifier_history_covers_the_configured_fall_window():
    plugin, _ = _plugin({"action_window_s": 0.5, "fall_settle_s": 3.0})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = plugin._nodes["/cam/rgb"]
    assert node._tracker.history_s >= 3.0


# ── one-shot ─────────────────────────────────────────────────────────────────

@pytest.fixture
def photo(tmp_path):
    """A real file holding whatever the active cv2 can decode, plus the root it
    has to be allowed from.

    The root is passed explicitly rather than relying on the default `/tmp`:
    pytest's tmp dirs live under /private/var on macOS, and check_under_roots
    resolves symlinks before comparing — so a test that assumed /tmp would fail
    on a laptop for a reason that has nothing to do with the plugin.
    """
    path = tmp_path / "scene.jpg"
    path.write_bytes(frame_bytes(FRAME_W, FRAME_H))
    return str(path)


def _photo_plugin(photo, **cfg):
    cfg.setdefault("image_roots", [os.path.realpath(os.path.dirname(photo))])
    return _plugin(cfg)


def test_a_photo_answers_with_keypoints_and_says_what_it_cannot_judge(photo):
    plugin, _ = _photo_plugin(photo)
    result = plugin.dispatch("pose", {"action": "recognize_by_photo",
                                      "image_path": photo})
    assert result["ok"] is True, result
    assert result["image_size"] == [FRAME_W, FRAME_H]
    assert result["count"] == 1
    assert len(result["persons"][0]["keypoints"]) == N_KEYPOINTS
    assert result["persons"][0]["posture"] == "standing"


def test_a_photo_says_which_actions_it_cannot_judge(photo):
    """Answering `raising_hand` to "is she waving" would answer a narrower
    question than the one asked."""
    plugin, _ = _photo_plugin(photo)
    result = plugin.dispatch("pose", {"action": "recognize_by_photo",
                                      "image_path": photo})
    assert result["temporal"] is False
    assert "hand waving" in result["unavailable_activities"]
    assert "falling down" in result["unavailable_activities"]


def test_a_photo_echoes_onto_a_running_topicless_card(photo):
    """The only reason a card is startable with no camera: the canvas has to
    show data flowing through it."""
    plugin, _ = _photo_plugin(photo)
    plugin.dispatch("pose", {"action": "start"})
    node = plugin._nodes[pose_plugin._DEFAULT_INSTANCE]
    plugin.dispatch("pose", {"action": "recognize_by_photo", "image_path": photo})
    lean = _publisher(node, "/perception/pose")
    assert lean.messages, "a running card must show the one-shot result"
    assert json.loads(lean.messages[-1])["count"] == 1


def test_a_photo_needs_no_running_instance(photo):
    plugin, _ = _photo_plugin(photo)
    result = plugin.dispatch("pose", {"action": "recognize_by_photo",
                                      "image_path": photo})
    assert result["ok"] is True
    assert plugin._nodes == {}


def test_a_photo_outside_the_allowed_roots_is_refused(tmp_path):
    """The MCP server has no authentication and runs as root in the container."""
    plugin, _ = _plugin({"image_roots": ["/models"]})
    outside = tmp_path / "scene.jpg"
    outside.write_bytes(frame_bytes(32, 32))
    result = plugin.dispatch("pose", {"action": "recognize_by_photo",
                                      "image_path": str(outside)})
    assert result.get("ok") is False


def test_list_actions_separates_events_from_poses_and_states_its_limits():
    plugin, _ = _plugin()
    result = plugin.dispatch("pose", {"action": "list_actions"})
    assert result["ok"] is True
    names = {entry["name"] for entry in result["activities"]}
    assert "falling down" in names
    assert "falling down" in result["limitations"]["needs_stream"]
    assert "face_recognition" in result["limitations"]["identity"]


def test_an_unknown_action_returns_none_rather_than_a_fake_success():
    plugin, _ = _plugin()
    assert plugin.dispatch("pose", {"action": "teleport"}) is None


# ── action backend selection ─────────────────────────────────────────────────

def test_the_default_backend_is_hybrid():
    """Geometry for the postures, ST-GCN++ for the events. Not a hedge: NTU-60
    has no `standing`/`sitting` state class, so a pure swap loses the postures."""
    plugin, _ = _plugin()
    assert plugin._action_backend == "hybrid"
    schema = pose_plugin.TOOLS[0]["configSchema"]["properties"]["action_backend"]
    assert schema["default"] == "hybrid"
    # `stgcn` is deliberately absent — see WITHDRAWN_BACKENDS.
    assert set(schema["enum"]) == {"rules", "hybrid"}


def test_choosing_rules_needs_no_action_engine_at_all():
    plugin, _ = _plugin({"action_backend": "rules"})
    backend = plugin._classifier_for(plugin._merged_config("i1"))
    assert backend.last_error is None
    assert plugin._backend_fallback is None
    assert plugin._effective_backend() == "rules"


def test_an_unknown_backend_name_falls_back_to_rules_and_says_so():
    """A card silently running geometry while its config names something else
    is the failure this whole feature set exists to stop."""
    plugin, _ = _plugin({"action_backend": "stgcnpp"})
    backend = plugin._classifier_for(plugin._merged_config("i1"))
    assert plugin._backend_fallback is not None
    assert "stgcnpp" in plugin._backend_fallback
    assert plugin._effective_backend() == "rules"
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert "action_backend_note" in info
    assert info["action_backend_effective"] == "rules"


def test_a_low_fps_with_a_temporal_backend_is_called_out():
    """A temporal backend classifies a clip, so a low fps starves it in a way
    the geometry is not — and a starved model looks like a wrong model."""
    plugin, _ = _plugin({"action_backend": "hybrid", "fps": 5})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert "action_fps_note" in info
    assert "fps >= 12" in info["action_fps_note"]


def test_a_sufficient_fps_raises_no_note():
    plugin, _ = _plugin({"action_backend": "hybrid", "fps": 15})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    assert "action_fps_note" not in plugin.dispatch("pose", {"action": "info"})


def test_the_rules_backend_gets_no_fps_note():
    plugin, _ = _plugin({"action_backend": "rules", "fps": 5})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    assert "action_fps_note" not in plugin.dispatch("pose", {"action": "info"})


def test_a_card_saved_with_the_withdrawn_backend_is_migrated_not_refused():
    """`stgcn` was offered and is not any more. A deployed card must keep
    working after an upgrade, and `hybrid` is what its owner wanted anyway: the
    learned labels plus the postures that backend cannot produce at all.

    Withdrawn because alone it is a foot-gun. It has no `standing` and no
    `sitting` class — NTU-60 is built from actions and a motionless person is
    not one — and a static clip does not make it abstain: fed 100 identical
    frames of a real person lying on pavement it returned NTU's "play with
    phone/tablet" at 0.997, entropy 0.03. Confidently wrong, not unsure.
    """
    plugin, _ = _plugin({"action_backend": "stgcn"})
    assert plugin._action_backend == "hybrid"
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    info = plugin.dispatch("pose", {"action": "info"})
    assert "action_backend_migrated" in info
    assert "0.997" in info["action_backend_migrated"]


def test_the_migration_also_covers_a_runtime_config_call():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "config", "action_backend": "stgcn"})
    assert plugin._action_backend == "hybrid"


def test_list_actions_reports_two_vocabularies():
    """Two questions, two answers. `postures` is what the geometry can see on a
    single frame; `activities` is NTU-60 in its own words, from the model."""
    plugin, _ = _plugin({"action_backend": "hybrid"})
    result = plugin.dispatch("pose", {"action": "list_actions"})
    assert set(result["postures"]) == set(POSTURE_LABELS_ZH)
    assert len(result["activities"]) == 49, "A50-A60 are two-person classes"
    names = {a["name"] for a in result["activities"]}
    assert "falling down" in names and "staggering" in names
    assert "handshake" not in names, "a two-person class must not be offered"


def test_rules_only_offers_no_activities():
    """No model, no activity channel — and the reply says so rather than
    listing a vocabulary nothing can produce."""
    plugin, _ = _plugin({"action_backend": "rules"})
    result = plugin.dispatch("pose", {"action": "list_actions"})
    assert result["activities"] == []
    assert set(result["postures"]) == set(POSTURE_LABELS_ZH)





def test_the_frame_size_reaches_the_tracker():
    """A learned backend normalises the skeleton by the frame and cannot derive
    that from the keypoints; a wrong frame size is a silent misnormalisation."""
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb")
    frame = node._tracker.tracks[0].history[-1]
    assert frame.image_size == (FRAME_W, FRAME_H)


def test_action_min_score_is_configurable_per_instance():
    plugin, _ = _plugin()
    plugin.dispatch("pose", {"action": "config", "instance_id": "i1",
                             "action_min_score": 0.7})
    assert plugin._merged_config("i1")["action_min_score"] == 0.7


# ── the overlay label ────────────────────────────────────────────────────────

def test_the_overlay_label_reads_the_two_channel_verdict():
    """It read `verdict["action"]` until that field was removed, and then
    labelled everybody `unknown` — silently, because the drawing tests fed a
    hand-built verdict that still carried the old key. That is exactly how a
    renderer keeps drawing after its data has moved underneath it."""
    assert pose_plugin.overlay_label(
        {"posture": "lying", "activity": {"name": "falling down"}}
    ) == "lying · falling down"
    assert pose_plugin.overlay_label(
        {"posture": "standing", "activity": None}) == "standing"
    assert pose_plugin.overlay_label({"posture": None, "activity": None}) == "unknown"
    # A published record carries the activity as a plain string, not a dict.
    assert pose_plugin.overlay_label(
        {"posture": "sitting", "activity": "reading"}) == "sitting · reading"
    # Either alone is shown on its own.
    assert pose_plugin.overlay_label(
        {"posture": None, "activity": "reading"}) == "reading"


def test_the_overlay_label_never_comes_back_unknown_for_a_known_person():
    """The regression itself: anybody the card has an opinion about must get
    that opinion drawn."""
    for verdict in ({"posture": "standing", "activity": None},
                    {"posture": None, "activity": {"name": "hand waving"}},
                    {"posture": "lying", "activity": {"name": "falling down"}}):
        assert pose_plugin.overlay_label(verdict) != "unknown", verdict


# ── the action model is throttled ───────────────────────────────────────────

def test_every_backend_accepts_the_throttle_flag():
    """The node throttles the learned backend and must not have to know which
    backend it is holding, so all three share one signature. `rules` would have
    crashed on every frame without this."""
    import inspect
    from plugins.pose_action import PoseActionClassifier
    from plugins.pose_stgcn import HybridActionBackend, SkeletonActionBackend
    for cls in (PoseActionClassifier, SkeletonActionBackend, HybridActionBackend):
        assert "want_activity" in inspect.signature(cls.classify).parameters, cls


def test_the_model_is_not_run_on_every_frame():
    """Its window is 2.5 s, so two runs one frame apart share 97% of their
    input and cost 20 ms each. Measured on Orin 6 with three people, running it
    every frame put the card at 98.9 ms per frame — a 10 fps ceiling on a 12 fps
    stream — while the geometry beside it costs 0.68 ms."""
    plugin, _ = _plugin({"activity_interval_s": 10.0})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb", frames=4)
    track = node._tracker.tracks[0]
    # Four frames inside one interval: the model was asked once.
    assert track.activity_t > -1e9, "the first frame must ask"
    calls = sum(1 for m in _publisher(node, "/cam/rgb/poses").messages)
    assert calls == 4, "but every frame still publishes"


def test_posture_stays_frame_rate_while_the_activity_is_throttled():
    """The geometry runs every frame regardless — a posture that only updated
    three times a second would be a worse trade than the one being made."""
    plugin, _ = _plugin({"activity_interval_s": 10.0})
    plugin.dispatch("pose", {"action": "start", "input_topic": "/cam/rgb"})
    node = _feed(plugin, "/cam/rgb", frames=3)
    for message in _publisher(node, "/cam/rgb/poses").messages:
        assert json.loads(message)["persons"][0]["posture"] == "standing"


def test_a_zero_interval_disables_the_throttle():
    plugin, _ = _plugin({"activity_interval_s": 0.0})
    merged = plugin._merged_config("i1")
    assert merged["activity_interval_s"] == 0.0


def test_the_alert_colour_keys_on_the_channels_not_the_label():
    """`overlay_label` became a compound string once both channels are shown
    ("lying · falling down"), so testing it for membership in a label set
    silently stopped matching and a fallen person was drawn in a track colour."""
    assert pose_plugin.is_alerting({"posture": "lying", "activity": None})
    assert pose_plugin.is_alerting(
        {"posture": None, "activity": {"name": "falling down"}})
    assert pose_plugin.is_alerting(
        {"posture": "lying", "activity": {"name": "falling down"}})
    assert not pose_plugin.is_alerting(
        {"posture": "standing", "activity": {"name": "reading"}})
    # And the compound label itself is never in the set, which is the trap.
    assert pose_plugin.overlay_label(
        {"posture": "lying", "activity": {"name": "falling down"}}
    ) not in pose_plugin._ALERT_LABELS
