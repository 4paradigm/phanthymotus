"""Real OpenCV geometry and JPEGs, with ROS message/executor stubs."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plugins.aruco import ArucoPlugin
from plugins.aruco_runtime import Calibration, MarkerDetector
from vision_stubs import _FakeCompressedImage, _FakeExecutor, _wait_until


TOPIC = "/camera/color"
JETPACK_IMAGE = Path("/etc/jetpack.env")
WIDTH, HEIGHT = 640, 480
K = [800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0]
CAMERA = {"schema": "motus.camera/1", "topic": TOPIC, "id": "calibrated-camera",
          "width": WIDTH, "height": HEIGHT, "K": K, "distortion_model": "pinhole", "D": [0] * 5}
SQUARE = np.array([[-0.05, 0.05, 0], [0.05, 0.05, 0],
                   [0.05, -0.05, 0], [-0.05, -0.05, 0]], dtype=np.float64)


@pytest.fixture
def detector():
    try:
        return MarkerDetector("DICT_4X4_50")
    except RuntimeError as exc:
        if JETPACK_IMAGE.is_file():
            # The cloud runner uses the built image's OpenCV. A skipped
            # detection here would let an unusable JetPack image pass CI.
            version = getattr(cv2, "__version__", "unavailable")
            pytest.fail(f"JetPack OpenCV {version} cannot run ArUco: {exc} ({exc.__cause__})",
                        pytrace=False)
        pytest.skip(str(exc))


def render_marker(rotation, translation):
    """Perspective-project an actual dictionary marker at a known camera pose."""
    aruco = cv2.aruco
    get_dictionary = getattr(aruco, "getPredefinedDictionary", None) or aruco.Dictionary_get
    dictionary = get_dictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, 23, 240)
    else:
        marker = aruco.drawMarker(dictionary, 23, 240)
    source = np.array([[0, 0], [239, 0], [239, 239], [0, 239]], dtype=np.float32)
    points, _ = cv2.projectPoints(SQUARE, rotation, translation, np.array(K).reshape(3, 3), None)
    transform = cv2.getPerspectiveTransform(source, points.reshape(4, 2).astype(np.float32))
    gray = cv2.warpPerspective(marker, transform, (WIDTH, HEIGHT),
                               flags=cv2.INTER_NEAREST, borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize(("rotation", "translation"), [
    ([np.pi, 0, 0], [0, 0, 0.4]),
    ([2.7, -0.19, 0.28], [0.035, -0.022, 0.4]),
])
def test_metric_pose_for_frontal_and_tilted_jpegs(detector, rotation, translation):
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    frame = render_marker(rotation, translation)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    calibration = Calibration.from_declaration(CAMERA)
    [pose] = detector.estimate(jpeg, calibration, 0.1)
    assert pose["id"] == 23
    assert np.linalg.norm(np.asarray(pose["translation_m"]) - translation) < 0.006
    actual, _ = cv2.Rodrigues(np.asarray(pose["rotation_rvec_rad"]))
    expected, _ = cv2.Rodrigues(rotation)
    assert np.linalg.norm(actual - expected) < 0.23
    assert pose["reprojection_rms_px"] < 1.5
    # OpenCV draws the X and Y axes with red and green in BGR images.
    assert ((jpeg[:, :, 2] > 240) & (jpeg[:, :, 1] < 30)).any()
    assert ((jpeg[:, :, 1] > 240) & (jpeg[:, :, 2] < 30)).any()


@pytest.mark.parametrize("change", [
    {"K": [float("nan")] * 9},
    {"K": [-800.0, *K[1:]]},
    {"distortion_model": "unknown", "D": [0.1, 0, 0, 0, 0]},
    {"distortion_model": "plumb_bob", "D": [0.1]},
    {"distortion_model": "pinhole", "D": [0.1, 0, 0, 0, 0]},
])
def test_rejects_unusable_calibration(change):
    with pytest.raises(ValueError):
        Calibration.from_declaration({**CAMERA, **change})


def test_bundle_advertises_tool_and_explains_missing_backend(monkeypatch):
    from main import PerceptionBundle

    monkeypatch.delattr(cv2, "aruco", raising=False)
    executor = _FakeExecutor()
    bundle = PerceptionBundle({"plugins": {"aruco": {"enabled": True},
                                          "echo_cancellation": {"enabled": False},
                                          "audio_mixer": {"enabled": False}}}, executor)
    try:
        [tool] = bundle.get_all_tools()
        assert tool["name"] == "aruco" and tool["type"] == "processor"
        assert [port["format"] for port in tool["topic_out"]] == ["data/json", "image/jpeg"]
        unavailable = bundle.dispatch("aruco", {"action": "info"})
        assert unavailable["state"] == "error"
        assert "OpenCV ArUco" in unavailable["error"]
        assert not executor.nodes
    finally:
        bundle.shutdown()


def _frame(image, sec):
    if isinstance(image, np.ndarray):
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        image = encoded.tobytes()
    frame = _FakeCompressedImage(image)
    frame.header.stamp = SimpleNamespace(sec=sec, nanosec=42)
    frame.header.frame_id = "camera_optical_frame"
    return frame


def test_processor_publishes_matching_json_and_jpeg_then_clears_detection(detector):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({"rate_hz": 30}, executor)
    args = {"action": "start", "instance_id": "aruco-a", "input_topic": TOPIC,
            "camera_info": {TOPIC: CAMERA}}
    try:
        assert plugin.dispatch("aruco", {**args, "camera_info": {"/other": CAMERA}})["state"] == "error"
        assert not executor.nodes
        assert plugin.dispatch("aruco", args)["state"] == "running"
        assert plugin.dispatch("aruco", args)["state"] == "running"
        [node] = executor.nodes
        [pose_pub, overlay_pub] = node.publishers
        result = plugin.dispatch("aruco", {"action": "info", "instance_id": "aruco-a"})
        assert [port["topic"] for port in result["topic_out"]] == [pose_pub.topic, overlay_pub.topic]
        deliver = node.subscriptions[0].callback
        frame = _frame(render_marker(np.array([2.7, -0.19, 0.28]), np.array([0.035, -0.022, 0.4])), 100)
        deliver(frame)
        assert _wait_until(lambda: len(overlay_pub.raw_messages) == 1)
        first = json.loads(pose_pub.messages[0])
        assert first["markers"][0]["id"] == 23
        assert first["camera_id"] == CAMERA["id"]
        assert first["frame_id"] == frame.header.frame_id
        assert first["stamp"] == {"sec": 100, "nanosec": 42}
        overlay = overlay_pub.raw_messages[0]
        assert overlay.format == "jpeg" and overlay.header == frame.header
        view = cv2.imdecode(np.frombuffer(overlay.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert view.shape == (HEIGHT, WIDTH, 3)
        assert ((view[:, :, 1] > 140) & (view[:, :, 2] < 130)).any()

        deliver(_frame(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8), 101))
        assert _wait_until(lambda: len(overlay_pub.messages) == 2)
        blank = json.loads(pose_pub.messages[1])
        assert blank["markers"] == [] and "error" not in blank
        plain = cv2.imdecode(np.frombuffer(overlay_pub.messages[1], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert plain.min() == 255

        deliver(_frame(b"not a JPEG", 102))
        assert _wait_until(lambda: len(pose_pub.messages) == 3)
        assert "decode" in json.loads(pose_pub.messages[2])["error"]
        deliver(_frame(np.full((240, 320, 3), 255, dtype=np.uint8), 103))
        assert _wait_until(lambda: len(overlay_pub.messages) == 3)
        assert "dimensions" in json.loads(pose_pub.messages[3])["error"]

        assert plugin.dispatch("aruco", {"action": "stop", "instance_id": "aruco-a"})["state"] == "idle"
        deliver(frame)
        assert not executor.nodes and node.destroyed and not node.worker.is_alive()
        assert len(pose_pub.messages) == 4
    finally:
        plugin.shutdown()


def test_instances_on_same_camera_have_distinct_topics_and_override(detector):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    try:
        for instance in ("a", "b"):
            assert plugin.dispatch("aruco", {"action": "config", "instance_id": instance,
                                             "calibration": CAMERA})["state"] == "configured"
            assert plugin.dispatch("aruco", {"action": "start", "instance_id": instance,
                                             "input_topic": TOPIC})["state"] == "running"
        a, b = executor.nodes
        assert {out["topic"] for out in a.outputs}.isdisjoint(out["topic"] for out in b.outputs)
        with pytest.raises(ValueError, match="marker_size_m"):
            plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "marker_size_m": -1})
        plugin.dispatch("aruco", {"action": "stop", "instance_id": "a"})
        assert executor.nodes == [b]
        plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "calibration": {}})
        assert plugin.dispatch("aruco", {"action": "start", "instance_id": "a",
                                         "input_topic": TOPIC})["state"] == "error"
        assert executor.nodes == [b]
    finally:
        plugin.shutdown()
    assert not executor.nodes


@pytest.mark.parametrize(("camera_info", "expected_id"), [
    ({TOPIC: CAMERA}, CAMERA["id"]),
    ([CAMERA], CAMERA["id"]),
    ({TOPIC: {**CAMERA, "schema": "other.camera/1"}}, ""),
    ({TOPIC: {**CAMERA, "topic": "/another/color"}}, ""),
    ({TOPIC: {**CAMERA, "format": "image/depth-zlib"}}, ""),
])
def test_calibration_override_uses_only_matching_camera_identity(detector, camera_info, expected_id):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    args = {"action": "start", "instance_id": "a", "input_topic": TOPIC,
            "camera_info": camera_info}
    try:
        if not expected_id:
            assert plugin.dispatch("aruco", args)["state"] == "error"
            assert not executor.nodes
        plugin.dispatch("aruco", {"action": "config", "instance_id": "a",
                                  "calibration": {**CAMERA, "id": "calibration-file-id"}})
        assert plugin.dispatch("aruco", args)["state"] == "running"
        [node] = executor.nodes
        node.subscriptions[0].callback(_frame(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8), 1))
        assert _wait_until(lambda: bool(node.publishers[0].messages))
        assert json.loads(node.publishers[0].messages[0])["camera_id"] == expected_id
    finally:
        plugin.shutdown()


def test_worker_reports_processing_error_recovers_and_restarts(detector, caplog, monkeypatch):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({"rate_hz": 30}, executor)
    args = {"action": "start", "instance_id": "a", "input_topic": TOPIC,
            "camera_info": {TOPIC: CAMERA}}
    info_args = {"action": "info", "instance_id": "a"}
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    try:
        assert plugin.dispatch("aruco", args)["state"] == "running"
        [node] = executor.nodes
        deliver = node.subscriptions[0].callback
        pose_pub = node.publishers[0]

        def broken_detector(*_args):
            raise RuntimeError("detector unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(node.detector, "estimate", broken_detector)
            for index in (1, 2):
                deliver(_frame(image, index))
                assert _wait_until(lambda: len(pose_pub.messages) == index)
                failed = json.loads(pose_pub.messages[-1])
                assert failed["markers"] == []
                assert "detector unavailable" in failed["error"]
                assert failed["stamp"]["sec"] == index
            info = plugin.dispatch("aruco", info_args)
            assert info["state"] == "error" and "detector unavailable" in info["error"]
            assert sum("detector unavailable" in record.message for record in caplog.records) == 1

        deliver(_frame(image, 3))
        assert _wait_until(lambda: plugin.dispatch("aruco", info_args)["state"] == "running")
        assert "error" not in json.loads(pose_pub.messages[-1])

        node.close()
        assert "worker stopped" in plugin.dispatch("aruco", info_args)["error"]
        assert plugin.dispatch("aruco", args)["state"] == "running"
        [replacement] = executor.nodes
        assert node.destroyed and replacement is not node and replacement.worker.is_alive()
    finally:
        plugin.shutdown()
