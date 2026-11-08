"""Shared RGB geometry, metadata propagation and lifecycle without ROS/GPU."""
import copy
import json
import zlib

import cv2
import numpy as np
import pytest

from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until, frame_bytes
from plugins.undistort import UndistortPlugin
from plugins.visual_depth import VideoDepthPerceptionPlugin
from plugins.vop import VideoObjectPerceptionPlugin


def calibration(ident="test/front"):
    return {"id": ident, "width": 64, "height": 48,
            "K": [35, 0, 32, 0, 35, 24, 0, 0, 1],
            "D": [0.3, 0.02, 0, 0], "distortion_model": "plumb_bob"}


def plugin():
    executor = _FakeExecutor()
    return UndistortPlugin({}, executor), executor


def start(p, ident="front", topic="/front/rgb", cal=None):
    cal = cal or calibration()
    assert p.dispatch("undistort", {"action": "config", "instance_id": ident,
                                   "calibration": cal})["status"] == "configured"
    return p.dispatch("undistort", {
        "action": "start", "instance_id": ident, "input_topic": topic,
        "camera_info": [{"schema": "motus.camera/1", "topic": topic, **cal}]})


def test_one_corrected_jpeg_feeds_depth_vop_and_rgb_with_matching_geometry():
    p, executor = plugin()
    state = start(p)
    camera = state["camera_info"][0]
    rgb_topic = state["topic_out"][0]["topic"]
    node = executor.nodes[0]
    message = _FakeCompressedImage(frame_bytes(64, 48, fill=100))
    message.header.stamp = 1234
    message.header.frame_id = "front_optical"

    class DepthEngine:
        def infer(self, frame):
            self.frame = frame.copy()
            return [np.full(frame.shape[:2], 2.0, np.float32)], None

    class DetectionEngine:
        def infer(self, frame):
            self.frame = frame.copy()
            return [np.array([[[10, 10, 20, 20, .9, 0]]], np.float32)], None

    depth = VideoDepthPerceptionPlugin({}, "test", executor)
    vop = VideoObjectPerceptionPlugin({}, "test", executor)
    depth._model = DepthEngine()
    vop._model = DetectionEngine()
    vop._vocabulary = ["box"]
    try:
        node._image_cb(message)
        assert _wait_until(lambda: bool(node.publisher.messages))
        corrected = node.publisher.raw_messages[0]
        assert corrected.header.stamp == 1234
        assert corrected.header.frame_id == "front_optical"
        rgb = cv2.imdecode(np.frombuffer(corrected.data, np.uint8), cv2.IMREAD_COLOR)
        assert rgb.shape[:2] == (camera["height"], camera["width"])
        assert rgb.min() >= 98  # no artificial black border
        for processor, name in ((depth, "visual_depth"), (vop, "vop")):
            args = {"action": "start", "instance_id": "consumer", "input_topic": rgb_topic,
                    "camera_info": [camera]}
            assert processor.dispatch(name, args)["state"] == "running"
            processor._nodes["consumer"]._image_cb(corrected)
        assert _wait_until(lambda: hasattr(depth._model, "frame") and hasattr(vop._model, "frame"))
        np.testing.assert_array_equal(depth._model.frame, rgb)
        np.testing.assert_array_equal(vop._model.frame, rgb)
        d_info = depth.dispatch("visual_depth", {"action": "info", "instance_id": "consumer"})
        v_info = vop.dispatch("vop", {"action": "info", "instance_id": "consumer"})
        d = d_info["camera_info"][0]
        v = v_info["camera_info"][0]
        assert d["id"] == v["id"] == camera["id"]
        assert d["D"] == v["D"] == [0] * 5
        assert d["half_fov_rad"] == v["half_fov_rad"] == camera["half_fov_rad"]
        np.testing.assert_allclose(np.array(d["K"][:3]) / 640,
                                   np.array(v["K"][:3]) / v["width"])
        assert "perception/undistort" in d["pipeline"]
        assert "perception/undistort" in v["pipeline"]
        assert _wait_until(lambda: len(depth._nodes["consumer"]._raw_depth_history) > 0)
        fit = depth.dispatch("visual_depth", {"action": "calibrate", "instance_id": "consumer",
                                               "distance_m": 1, "region": "center"})
        assert fit["ok"] is True
        assert fit["sample"]["predicted_m"] == 2
    finally:
        depth.dispatch("visual_depth", {"action": "stop"})
        vop.dispatch("vop", {"action": "stop"})
        p.shutdown()
    assert not executor.nodes


def test_upstream_calibration_can_be_used_without_manual_override():
    p, executor = plugin()
    state = p.dispatch("undistort", {"action": "start", "input_topic": "/front/rgb",
                                   "camera_info": [{"topic": "/front/rgb", **calibration()}]})
    try:
        assert state["state"] == "running"
        assert state["camera_info"][0]["id"] == "test/front"
    finally:
        p.shutdown()


@pytest.mark.parametrize("upstream", [
    {}, {"id": "test/front", "width": 64, "height": 48, "K": None, "D": None},
    {**calibration(), "pipeline": ["perception/undistort"]},
])
def test_missing_calibration_or_double_rectification_is_rejected(upstream):
    p, executor = plugin()
    result = p.dispatch("undistort", {"action": "start", "input_topic": "/front/rgb",
                                    "camera_info": [{"topic": "/front/rgb", **upstream}]})
    assert result["state"] == "error"
    assert not executor.nodes


def test_configuration_requires_stop_and_keeps_old_instance_on_error():
    p, executor = plugin()
    start(p)
    node = executor.nodes[0]
    try:
        bad = p.dispatch("undistort", {"action": "config", "instance_id": "front", "fps": 5})
        assert bad["status"] == "error"
        assert p._nodes["front"] is node
        assert p._configs["front"]["fps"] == 15
        assert p.dispatch("undistort", {"action": "start", "instance_id": "front",
                                       "input_topic": "/rear/rgb"})["state"] == "error"
        assert p.dispatch("undistort", {"action": "stop", "instance_id": "front"})["state"] == "idle"
        assert node.destroyed
        assert not node._worker.is_alive()
        assert not executor.nodes
        assert p.dispatch("undistort", {"action": "config", "instance_id": "front", "fps": 5})["status"] == "configured"
        assert p._configs["front"]["calibration"] == calibration()
    finally:
        p.shutdown()


def test_size_mismatch_produces_error_instead_of_misregistered_rgb():
    p, executor = plugin()
    start(p)
    node = executor.nodes[0]
    try:
        node._image_cb(_FakeCompressedImage(frame_bytes(128, 48)))
        assert _wait_until(lambda: bool(node.error))
        assert "input size" in node.error
        assert not node.publisher.messages
    finally:
        p.shutdown()


def test_instances_publish_distinct_topics_and_stop_independently():
    p, executor = plugin()
    front = start(p)
    rear = start(p, "rear", "/rear/rgb", calibration("test/rear"))
    try:
        assert front["topic_out"] != rear["topic_out"]
        p.dispatch("undistort", {"action": "stop", "instance_id": "front"})
        assert list(p._nodes) == ["rear"]
        assert len(executor.nodes) == 1
        assert p.dispatch("undistort", {"action": "info", "instance_id": "rear"})["camera_info"][0]["id"] == "test/rear"
    finally:
        p.shutdown()


def test_bad_calibration_is_rejected_before_modifying_saved_configuration():
    p, _ = plugin()
    cal = calibration()
    p.dispatch("undistort", {"action": "config", "instance_id": "front", "calibration": cal})
    before = copy.deepcopy(p._configs)
    cal["D"] = [10]
    assert p.dispatch("undistort", {"action": "config", "instance_id": "front",
                                   "calibration": cal})["status"] == "error"
    assert p._configs == before


# ── automatic calibration: one card type for every camera ─────────────────────

def driver_declaration(topic, ident):
    """What a camera that knows its id but not its optics declares (S10 today)."""
    return {"schema": "motus.camera/1", "topic": topic, "id": ident,
            "width": 64, "height": 48, "distortion_model": "unknown",
            "K": None, "D": None, "source": "unknown"}


def test_four_cameras_each_pick_their_own_row_with_nothing_typed_in(monkeypatch):
    import plugins.undistortion_runtime as runtime
    cameras = {}
    for i, name in enumerate(("front", "rear", "left", "right")):
        ident = f"test/{name}"
        row = dict(calibration(ident), K=[30 + i, 0, 32, 0, 30 + i, 24, 0, 0, 1])
        monkeypatch.setitem(runtime.CALIBRATIONS, ident, row)
        cameras[name] = (f"/{name}/rgb", ident, 30 + i)
    p, executor = plugin()
    try:
        for name, (topic, ident, _) in cameras.items():
            state = p.dispatch("undistort", {
                "action": "start", "instance_id": name, "input_topic": topic,
                "camera_info": [driver_declaration(topic, ident)]})
            assert state["state"] == "running", state
        assert len(executor.nodes) == 4
        for name, (topic, ident, fx) in cameras.items():
            info = p.dispatch("undistort", {"action": "info", "instance_id": name})
            assert info["calibration_source"] == f"table:{ident}"
            declared = info["camera_info"][0]
            assert declared["id"] == ident
            assert declared["vendor"]["undistortion"]["input_calibration"]["K"][0] == fx
            assert "perception/undistort" in declared["pipeline"]
        assert len({tuple(info["topic_out"][0].values())
                    for info in (p.dispatch("undistort", {"action": "info", "instance_id": n})
                                 for n in cameras)}) == 4
    finally:
        p.shutdown()


def test_typed_in_calibration_overrides_the_table(monkeypatch):
    import plugins.undistortion_runtime as runtime
    monkeypatch.setitem(runtime.CALIBRATIONS, "test/front",
                        dict(calibration(), K=[30, 0, 32, 0, 30, 24, 0, 0, 1]))
    p, _ = plugin()
    try:
        state = start(p)        # types calibration() with fx = 35
        assert state["state"] == "running"
        assert state["calibration_source"] == "manual"
        assert state["camera_info"][0]["vendor"]["undistortion"]["input_calibration"]["K"][0] == 35
    finally:
        p.shutdown()


def test_an_upstream_declaring_its_own_k_d_overrides_the_table(monkeypatch):
    import plugins.undistortion_runtime as runtime
    monkeypatch.setitem(runtime.CALIBRATIONS, "test/front",
                        dict(calibration(), K=[30, 0, 32, 0, 30, 24, 0, 0, 1]))
    p, _ = plugin()
    try:
        state = p.dispatch("undistort", {
            "action": "start", "instance_id": "front", "input_topic": "/front/rgb",
            "camera_info": [{"topic": "/front/rgb", **calibration()}]})
        assert state["calibration_source"] == "upstream"
        assert state["camera_info"][0]["vendor"]["undistortion"]["input_calibration"]["K"][0] == 35
    finally:
        p.shutdown()


def test_an_uncalibrated_camera_names_itself_and_the_fix_on_the_card():
    p, executor = plugin()
    result = p.dispatch("undistort", {
        "action": "start", "instance_id": "rear", "input_topic": "/rear/rgb",
        "camera_info": [driver_declaration("/rear/rgb", "acme/cam/rear")]})
    assert result["state"] == "error"
    assert "acme/cam/rear" in result["message"]
    assert "calibrate_chessboard.py" in result["message"]
    assert not executor.nodes


def test_the_real_s10_front_camera_starts_from_the_built_in_table():
    """The declaration the S10 driver publishes today: id and size, no K/D."""
    p, _ = plugin()
    topic = "/s10_pro/camera_front"
    upstream = {"schema": "motus.camera/1", "topic": topic,
                "id": "deeprobotics/s10_pro/camera_front", "width": 640, "height": 360,
                "distortion_model": "unknown", "K": None, "D": None,
                "half_fov_rad": 0.7926, "source": "measured"}
    try:
        state = p.dispatch("undistort", {"action": "start", "instance_id": "front",
                                       "input_topic": topic, "camera_info": [upstream]})
        assert state["state"] == "running", state
        assert state["calibration_source"] == "table:deeprobotics/s10_pro/camera_front"
        declared = state["camera_info"][0]
        assert (declared["width"], declared["height"]) == (640, 360)
        assert declared["half_fov_rad"] == pytest.approx(0.7926, abs=1e-3)
    finally:
        p.shutdown()
