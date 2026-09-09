"""Numerical, lifecycle and wire-contract checks; real-model replay is separate."""
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision_stubs import _FakeExecutor, _FakeNode, _wait_until
from plugins import vop
from plugins.vop_depth import mask_min, render_preview, unavailable, ensure_weight


def test_mask_min_excludes_outside_and_invalid_and_keeps_exact_minimum():
    depth = np.array([[0.01, 4, 3], [np.nan, 2, np.inf], [-1, 0, 1.25]])
    mask = np.ones((3, 3), bool); mask[0, 0] = False
    assert mask_min(depth, mask) == {"value": 1.25, "unit": "relative", "method": "mask_min",
                                     "status": "ok", "pixel": [2, 2]}
    assert mask_min(depth, np.zeros((3, 3), bool))["status"] == "empty_mask"
    assert mask_min(np.zeros((3, 3)), mask)["status"] == "invalid_depth"
    with pytest.raises(ValueError):
        mask_min(depth, np.ones((2, 2)))


def test_cached_weight_must_match_before_loading(tmp_path):
    (tmp_path / "sam2.1_t.pt").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="checksum"):
        ensure_weight(tmp_path, "sam2.1_t.pt")


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setattr(_FakeNode, "create_timer", lambda *a: object(), raising=False)
    monkeypatch.setattr(_FakeNode, "destroy_timer", lambda *a: None, raising=False)
    monkeypatch.setattr(_FakeNode, "destroy_subscription", lambda *a: None, raising=False)
    class Image:
        def __init__(self):
            self.header = SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=500000000), frame_id="camera")
            self.format = "jpeg"
            self.data = cv2.imencode(".jpg", np.full((96, 160, 3), 90, np.uint8))[1].tobytes()
    monkeypatch.setattr(vop, "CompressedImage", Image)
    p = vop.VideoObjectPerceptionPlugin({}, "test", _FakeExecutor())
    yield p
    p.dispatch("vop", {"action": "stop"})


def test_preview_and_json_same_frame_and_stale_clears_objects(plugin, monkeypatch):
    obj = {"id": 1, "name": "cup", "position": [0, 0], "confidence": 0.9,
           "bbox_xyxy": [10, 40, 120, 80], "obstacle_depth":
           {**unavailable("ok"), "value": 1.2, "pixel": [30, 60]}}
    mask = np.zeros((1, 96, 160), bool); mask[:, 40:80, 10:120] = True
    monkeypatch.setattr(plugin, "_infer", lambda *a: ([obj], mask, None))
    captures = []
    original = vop.render_preview
    def capture(frame, objects, masks, **kwargs):
        captures.append((objects, kwargs))
        return original(frame, objects, masks, **kwargs)
    monkeypatch.setattr(vop, "render_preview", capture)
    plugin.dispatch("vop", {"action": "config", "instance_id": "one", "depth_enabled": True})
    plugin.dispatch("vop", {"action": "start", "instance_id": "one", "input_topic": "/camera"})
    node = plugin._nodes["one"]
    node._image_cb(vop.CompressedImage())
    assert _wait_until(lambda: node._detect_count == 1)
    data = json.loads(node._pub.messages[-1])
    assert data["source_timestamp"] == 123.5 and data["frame_id"] == "camera"
    assert data["objects"] == captures[-1][0]
    assert data["sequence"] == captures[-1][1]["sequence"]
    image = cv2.imdecode(np.frombuffer(node._preview_pub.messages[-1], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (130, 500, 3) and image.std() > 10
    node._last_received = time.monotonic() - 4
    node._watchdog()
    stale = json.loads(node._pub.messages[-1])
    assert stale["status"].startswith("stale") and stale["objects"] == []
    assert captures[-1][0] == []


def test_stop_during_model_work_prevents_late_publish_and_duplicate_start(plugin, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def infer(*args):
        entered.set(); release.wait(5)
        return [], None, None
    monkeypatch.setattr(plugin, "_infer", infer)
    args = {"action": "start", "input_topic": "/camera", "instance_id": "one"}
    threads = [threading.Thread(target=plugin.dispatch, args=("vop", args)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(plugin._executor.nodes) == 1
    node = plugin._nodes["one"]; node._image_cb(vop.CompressedImage())
    assert entered.wait(2)
    stopper = threading.Thread(target=plugin.dispatch, args=("vop", {"action": "stop", "instance_id": "one"}))
    stopper.start()
    assert node._stop_event.wait(2)
    count = len(node._pub.messages)
    release.set(); stopper.join(2)
    assert not stopper.is_alive() and len(node._pub.messages) == count
    assert node.destroyed and not plugin._nodes and not plugin._executor.nodes
    assert json.loads(node._pub.messages[-1])["status"] == "stopped"


def test_config_validation_and_topics(plugin):
    for cfg in ({"fps": 0}, {"fps": True}, {"confidence": float("nan")},
                {"depth_enabled": "false"}, {"classes": [""]}):
        with pytest.raises(ValueError):
            plugin.dispatch("vop", {"action": "config", **cfg})
    plugin.dispatch("vop", {"action": "config", "instance_id": "one", "fps": 2})
    plugin.dispatch("vop", {"action": "config", "instance_id": "one", "depth_enabled": True})
    assert plugin._config_for("one")["fps"] == 2
    info = plugin.dispatch("vop", {"action": "info", "input_topic": "/camera"})
    assert info["topic_out"][-1]["topic"] == "/camera/objects/preview"
    assert info["topic_out"][-1]["format"] == "image/jpeg"
    assert plugin._config_for("other")["depth_enabled"] is False


def test_depth_failure_preserves_detection_and_disabled_does_not_load(plugin, monkeypatch):
    monkeypatch.setattr(vop, "extract_objects", lambda *a: [{"bbox_xyxy": [0, 0, 2, 2]}])
    plugin._model = lambda *a, **kw: [object()]
    class BrokenDepth:
        def estimate(self, *a): raise RuntimeError("model unavailable")
    plugin._depth = BrokenDepth()
    frame = np.zeros((3, 3, 3), np.uint8)
    objects, masks, error = plugin._infer(frame, 0.3, True, threading.Event())
    assert error == "model unavailable" and len(objects) == 1 and masks is None
    assert objects[0]["obstacle_depth"] == unavailable("error")
    objects, _, error = plugin._infer(frame, 0.3, False, threading.Event())
    assert error is None and "obstacle_depth" not in objects[0]


def test_class_update_waits_for_shared_inference_lock(plugin, monkeypatch):
    moved = threading.Event()
    model = SimpleNamespace(cpu=lambda: None, to=lambda device: None)
    plugin._model = SimpleNamespace(model=model, set_classes=lambda classes: moved.set())
    plugin._model_lock.acquire()
    updater = threading.Thread(target=plugin._sync_model_classes)
    updater.start()
    try:
        assert not moved.wait(0.05)
    finally:
        plugin._model_lock.release()
    updater.join(2)
    assert moved.is_set() and not updater.is_alive()


def test_stop_cannot_restart_a_cancelled_node_or_share_output(plugin):
    plugin.dispatch("vop", {"action": "start", "instance_id": "one", "input_topic": "/camera"})
    node = plugin._nodes["one"]
    with pytest.raises(ValueError, match="already has"):
        plugin.dispatch("vop", {"action": "start", "instance_id": "two", "input_topic": "/camera"})
    plugin.dispatch("vop", {"action": "stop", "instance_id": "one"})
    assert node.start()["state"] == "idle"
    assert node.destroyed and node._sub is None


def test_actions_have_matching_parameter_metadata(plugin):
    schema = plugin.get_tools()[0]["inputSchema"]
    assert set(schema["properties"]["action"]["enum"]) == set(schema["x-action-params"])


def test_long_inference_is_eventually_disposed_without_second_stop(plugin, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def infer(*args):
        entered.set(); release.wait(8)
        return [], None, None
    monkeypatch.setattr(plugin, "_infer", infer)
    plugin.dispatch("vop", {"action": "start", "instance_id": "one", "input_topic": "/camera"})
    node = plugin._nodes["one"]; node._image_cb(vop.CompressedImage())
    assert entered.wait(2)
    try:
        assert plugin.dispatch("vop", {"action": "stop", "instance_id": "one"})["state"] == "stopping"
        assert not node.destroyed
    finally:
        release.set()
    assert _wait_until(lambda: node.destroyed and not plugin._nodes)
    assert not plugin._executor.nodes


def test_core_does_not_treat_waiting_or_stale_as_ready(plugin):
    started = plugin.dispatch("vop", {"action": "start", "instance_id": "one", "input_topic": "/camera"})
    assert started["state"] == "loading" and started["phase"] == "waiting"
    node = plugin._nodes["one"]
    node._state = "processing"
    assert node.info()["state"] == "loading"
    node._last_result = time.monotonic()
    assert node.info()["state"] == "running"
    node._last_received = time.monotonic() - 4
    node._watchdog()
    info = plugin.dispatch("vop", {"action": "info", "instance_id": "one"})
    assert info["state"] == "error" and info["phase"] == "stale" and info["error"]
    resumed = plugin.dispatch("vop", {"action": "start", "instance_id": "one", "input_topic": "/camera"})
    assert resumed["state"] == "loading" and plugin._nodes["one"] is node
