"""Real OpenCV geometry and JPEGs, with ROS message/executor stubs."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plugins import aruco_runtime
from plugins.aruco import ArucoPlugin
from plugins.aruco_runtime import MarkerDetector, intrinsics
from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until


TOPIC = "/camera/color"
WIDTH, HEIGHT = 640, 480
K = [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]
CAMERA = {"schema": "motus.camera/1", "topic": TOPIC, "id": "calibrated-camera",
          "width": WIDTH, "height": HEIGHT, "K": K, "distortion_model": "plumb_bob", "D": [0] * 5}
# What R1 declares: a measured field of view and no K. atan(320 / 800) matches K above.
FOV_ONLY = {**CAMERA, "K": None, "D": None, "half_fov_rad": float(np.arctan(320 / 800))}
SQUARE = np.array([[-0.05, 0.05, 0], [0.05, 0.05, 0],
                   [0.05, -0.05, 0], [-0.05, -0.05, 0]], dtype=np.float64)
TILTED = np.array([2.7, -0.19, 0.28]), np.array([0.035, -0.022, 0.4])


@pytest.fixture(autouse=True)
def aruco_backend():
    if aruco_runtime.unavailable_reason():
        if Path("/etc/jetpack.env").is_file():
            # A skip here would let an image without ArUco pass CI.
            pytest.fail(aruco_runtime.unavailable_reason(), pytrace=False)
        pytest.skip(aruco_runtime.unavailable_reason())


def render_marker(rotation, translation, marker_id=23, dictionary_name="DICT_4X4_50"):
    """Perspective-project an actual dictionary marker at a known camera pose."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, marker_id, 240)
    else:
        marker = aruco.drawMarker(dictionary, marker_id, 240)
    source = np.array([[0, 0], [239, 0], [239, 239], [0, 239]], dtype=np.float32)
    points, _ = cv2.projectPoints(SQUARE, rotation, translation, np.array(K).reshape(3, 3), None)
    transform = cv2.getPerspectiveTransform(source, points.reshape(4, 2).astype(np.float32))
    gray = cv2.warpPerspective(marker, transform, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST, borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _frame(image, sec=100):
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    frame = _FakeCompressedImage(encoded.tobytes())
    frame.header.stamp = SimpleNamespace(sec=sec, nanosec=42)
    frame.header.frame_id = "camera_optical_frame"
    return frame


def _assert_pose(marker, rotation, translation):
    np.testing.assert_allclose(marker["translation_m"], translation, atol=0.006)
    assert marker["distance_m"] == pytest.approx(np.linalg.norm(translation), abs=0.006)
    actual, _ = cv2.Rodrigues(np.asarray(marker["rotation_rvec_rad"]))
    expected, _ = cv2.Rodrigues(rotation)
    assert np.linalg.norm(actual - expected) < 0.23


@pytest.mark.parametrize(("rotation", "translation"), [([np.pi, 0, 0], [0, 0, 0.4]), TILTED])
def test_metric_pose_for_frontal_and_tilted_markers(rotation, translation):
    rotation, translation = np.asarray(rotation, float), np.asarray(translation, float)
    image = render_marker(rotation, translation)
    matrix, distortion, status = intrinsics(CAMERA, WIDTH, HEIGHT)
    [marker] = MarkerDetector("DICT_4X4_50").detect(image, 0.1, matrix, distortion, draw=True)
    assert status == "calibrated" and marker["id"] == 23
    _assert_pose(marker, rotation, translation)
    # OpenCV draws the X and Y axes red and green.
    assert ((image[:, :, 2] > 240) & (image[:, :, 1] < 30)).any()


def test_intrinsics_use_k_then_field_of_view_then_nothing():
    calibrated, _, status = intrinsics(CAMERA, WIDTH, HEIGHT)
    assert status == "calibrated"
    approximate, distortion, status = intrinsics(FOV_ONLY, WIDTH, HEIGHT)
    assert status == "approximate" and distortion is None
    assert approximate[0, 0] == pytest.approx(800)
    assert intrinsics(None, WIDTH, HEIGHT) == (None, None, "uncalibrated")
    assert intrinsics({"width": WIDTH, "height": HEIGHT}, WIDTH, HEIGHT)[2] == "uncalibrated"
    # A K declared at another resolution is rescaled to the frame.
    half, _, _ = intrinsics(CAMERA, WIDTH // 2, HEIGHT // 2)
    np.testing.assert_allclose(half, calibrated / [[2], [2], [1]])


@pytest.mark.parametrize(("camera_info", "status"), [
    ({TOPIC: CAMERA}, "calibrated"), ([FOV_ONLY], "approximate"), (None, "uncalibrated")])
def test_card_runs_with_whatever_the_camera_declares(camera_info, status):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({"rate_hz": 30}, executor)
    try:
        started = plugin.dispatch("aruco", {"action": "start", "instance_id": "a", "input_topic": TOPIC,
                                            "camera_info": camera_info})
        assert started["state"] == "running"
        [node] = executor.nodes
        poses, overlay = node.publishers
        assert [poses.topic, overlay.topic] == [f"{TOPIC}/aruco", f"{TOPIC}/aruco/overlay"]
        node.subscriptions[0].callback(_frame(render_marker(*TILTED)))
        assert _wait_until(lambda: bool(overlay.raw_messages))
        result = json.loads(poses.messages[0])
        assert result["pose_status"] == status and result["stamp"] == {"sec": 100, "nanosec": 42}
        [marker] = result["markers"]
        assert marker["id"] == 23 and len(marker["center_px"]) == 2
        if status == "uncalibrated":
            assert "translation_m" not in marker
        else:
            _assert_pose(marker, *TILTED)
        assert plugin.dispatch("aruco", {"action": "info", "instance_id": "a"})["pose_status"] == status
    finally:
        plugin.shutdown()
    assert not executor.nodes and node.destroyed and not node.worker.is_alive()


def test_publishes_once_when_markers_leave_then_stays_quiet():
    executor = _FakeExecutor()
    plugin = ArucoPlugin({"rate_hz": 30, "publish_overlay": False}, executor)
    try:
        plugin.dispatch("aruco", {"action": "start", "input_topic": TOPIC, "camera_info": [CAMERA]})
        [node] = executor.nodes
        [poses] = node.publishers
        deliver = node.subscriptions[0].callback
        blank = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        for image, expected in ((render_marker(*TILTED), 1), (blank, 2)):
            deliver(_frame(image))
            assert _wait_until(lambda: len(poses.messages) == expected)
        assert json.loads(poses.messages[1])["markers"] == []
        deliver(_frame(blank))
        assert _wait_until(lambda: node.frames.pop(timeout=0) is None)
        assert len(poses.messages) == 2

        deliver(_FakeCompressedImage(b"not a JPEG"))
        assert _wait_until(lambda: "decode" in plugin.dispatch("aruco", {"action": "info",
                                                                           "input_topic": TOPIC})["last_error"])
    finally:
        plugin.shutdown()


@pytest.mark.parametrize(("dictionary", "second_id"), [
    ("DICT_4X4_50", 23), ("DICT_5X5_50", 23), ("DICT_6X6_250", 123), ("DICT_7X7_50", 23)])
def test_two_markers_in_each_dictionary(dictionary, second_id):
    rotations = (np.array([2.8, -0.13, 0.1]), np.array([2.7, -0.19, 0.28]))
    translations = (np.array([-0.08, -0.012, 0.4]), np.array([0.10, 0.02, 0.60]))
    image = np.minimum(render_marker(rotations[0], translations[0], 7, dictionary),
                       render_marker(rotations[1], translations[1], second_id, dictionary))
    matrix, distortion, _ = intrinsics(CAMERA, WIDTH, HEIGHT)
    markers = MarkerDetector(dictionary).detect(image, 0.1, matrix, distortion)
    assert [m["id"] for m in markers] == [7, second_id]
    for marker, rotation, translation in zip(markers, rotations, translations):
        _assert_pose(marker, rotation, translation)


def test_ports_before_start_and_overlay_off():
    plugin = ArucoPlugin({}, _FakeExecutor())
    ports = plugin.get_tools()[0]["topic_out"]
    assert plugin.dispatch("aruco", {"action": "info"})["topic_out"] == [{**p, "topic": ""} for p in ports]
    plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "publish_overlay": False})
    preview = plugin.dispatch("aruco", {"action": "info", "instance_id": "a", "input_topic": TOPIC})
    assert preview["state"] == "idle"
    assert [p["topic"] for p in preview["topic_out"]] == [f"{TOPIC}/aruco", ""]


def test_restart_applies_new_settings_and_camera_but_not_a_repeat():
    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    start = {"action": "start", "instance_id": "a", "input_topic": TOPIC, "camera_info": [CAMERA]}
    try:
        plugin.dispatch("aruco", start)
        [first] = executor.nodes
        plugin.dispatch("aruco", start)
        assert executor.nodes == [first]

        plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "marker_size_m": 0.05})
        plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "rate_hz": 2})
        plugin.dispatch("aruco", start)
        [second] = executor.nodes
        assert first.destroyed and second.options["marker_size_m"] == 0.05 and second.options["rate_hz"] == 2

        plugin.dispatch("aruco", {**start, "camera_info": [FOV_ONLY]})
        [third] = executor.nodes
        assert second.destroyed and third.declaration == FOV_ONLY
    finally:
        plugin.shutdown()


def test_config_rejects_bad_values():
    plugin = ArucoPlugin({}, _FakeExecutor())
    for bad in ({"dictionary": "DICT_ARUCO_ORIGINAL"}, {"marker_size_m": -1}, {"rate_hz": 100}):
        with pytest.raises(ValueError):
            plugin.dispatch("aruco", {"action": "config", **bad})


def test_stop_during_start_discards_the_new_node(monkeypatch):
    import plugins.aruco as module

    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    real_node = module._ArucoNode

    def node_then_stop(*args):
        node = real_node(*args)
        plugin.dispatch("aruco", {"action": "stop"})
        return node

    monkeypatch.setattr(module, "_ArucoNode", node_then_stop)
    result = plugin.dispatch("aruco", {"action": "start", "instance_id": "a", "input_topic": TOPIC})
    assert result == {"state": "idle"} and not executor.nodes


def test_bundle_loads_the_card_and_reports_a_missing_backend(monkeypatch):
    from main import PerceptionBundle

    monkeypatch.setattr(aruco_runtime, "unavailable_reason", lambda: "no ArUco here")
    monkeypatch.setattr("plugins.aruco.unavailable_reason", lambda: "no ArUco here")
    bundle = PerceptionBundle({"plugins": {"aruco": {"enabled": True},
                                          "echo_cancellation": {"enabled": False},
                                          "audio_mixer": {"enabled": False}}}, _FakeExecutor())
    try:
        [tool] = bundle.get_all_tools()
        assert tool["name"] == "aruco" and tool["type"] == "processor"
        info = bundle.dispatch("aruco", {"action": "info"})
        assert info["state"] == "error" and info["error"] == "no ArUco here"
        assert bundle.dispatch("aruco", {"action": "start", "input_topic": TOPIC})["state"] == "error"
    finally:
        bundle.shutdown()
