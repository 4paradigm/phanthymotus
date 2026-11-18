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


def render_marker(rotation, translation, marker_id=23, dictionary_name="DICT_4X4_50"):
    """Perspective-project an actual dictionary marker at a known camera pose."""
    aruco = cv2.aruco
    get_dictionary = getattr(aruco, "getPredefinedDictionary", None) or aruco.Dictionary_get
    dictionary = get_dictionary(getattr(aruco, dictionary_name))
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, marker_id, 240)
    else:
        marker = aruco.drawMarker(dictionary, marker_id, 240)
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
        fields = tool["configSchema"]["properties"]
        assert "calibration" not in fields
        assert fields["publish_overlay"]["type"] == "boolean" and fields["publish_overlay"]["default"] is True
        assert fields["marker_size_m"]["default"] == 0.10 and "10 cm" in tool["description"]
        for dictionary in fields["dictionary"]["enum"]:
            assert dictionary.removeprefix("DICT_") in fields["dictionary"]["description"]
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
        assert first["pose_status"] == "calibrated"
        assert first["coordinate_frame"] == "camera_optical"
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


@pytest.mark.parametrize("camera_info", [
    None,
    {"/other": CAMERA},
    {TOPIC: {**CAMERA, "schema": "other.camera/1"}},
    {TOPIC: {**CAMERA, "topic": "/another/color"}},
    {TOPIC: {**CAMERA, "format": "image/depth-zlib"}},
    {TOPIC: {**CAMERA, "K": None}},
    {TOPIC: {**CAMERA, "K": []}},
    {TOPIC: {**CAMERA, "distortion_model": "unknown"}},
])
def test_start_requires_intrinsics_from_the_wired_camera(camera_info):
    executor = _FakeExecutor()
    # A previously saved manual calibration must not mask a broken camera declaration.
    plugin = ArucoPlugin({"calibration": CAMERA}, executor)
    try:
        plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "calibration": CAMERA})
        with pytest.raises(ValueError) as error:
            plugin.dispatch("aruco", {"action": "start", "instance_id": "a",
                                      "input_topic": TOPIC, "camera_info": camera_info})
        assert TOPIC in str(error.value) and "相机卡片开发者" in str(error.value)
        assert not executor.nodes
    finally:
        plugin.shutdown()


@pytest.mark.parametrize("camera_info", [{TOPIC: CAMERA}, [CAMERA]])
def test_saved_override_cannot_replace_camera_geometry(detector, camera_info):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    try:
        plugin.dispatch("aruco", {"action": "config", "instance_id": "a",
                                  "calibration": {**CAMERA, "id": "old-file", "K": [400, 0, 320, 0, 400, 240, 0, 0, 1]}})
        info = plugin.dispatch("aruco", {"action": "start", "instance_id": "a",
                                         "input_topic": TOPIC, "camera_info": camera_info})
        assert info["pose_status"] == "calibrated"
        [node] = executor.nodes
        assert node.camera_id == CAMERA["id"]
        np.testing.assert_array_equal(node.calibration.matrix, np.array(K).reshape(3, 3))
    finally:
        plugin.shutdown()


def test_overlay_is_per_instance_and_invalid_restart_stops_only_that_instance(detector):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    start = {"action": "start", "input_topic": TOPIC, "camera_info": {TOPIC: CAMERA}}
    try:
        with pytest.raises(ValueError, match="publish_overlay"):
            plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "publish_overlay": "false"})
        with pytest.raises(ValueError, match="marker_size_m"):
            plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "marker_size_m": -1})
        plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "publish_overlay": False})
        a = plugin.dispatch("aruco", {**start, "instance_id": "a"})
        b = plugin.dispatch("aruco", {**start, "instance_id": "b"})
        assert a["topic_out"][1]["topic"] == "" and b["topic_out"][1]["topic"].endswith("/overlay")
        assert a["topic_out"][0]["topic"] != b["topic_out"][0]["topic"]
        first, second = executor.nodes
        with pytest.raises(ValueError, match="相机卡片开发者"):
            plugin.dispatch("aruco", {**start, "instance_id": "a", "camera_info": {}})
        assert first.destroyed and not first.worker.is_alive()
        assert executor.nodes == [second] and second.worker.is_alive()
    finally:
        plugin.shutdown()


@pytest.mark.parametrize(("dictionary", "second_id"), [
    ("DICT_4X4_50", 23), ("DICT_5X5_50", 49), ("DICT_6X6_250", 203), ("DICT_7X7_50", 49),
])
def test_two_marker_poses_are_identical_with_or_without_overlay(detector, monkeypatch, dictionary, second_id):
    rotations = (np.array([2.8, -0.13, 0.1]), np.array([2.7, -0.19, 0.28]))
    translations = (np.array([-0.08, -0.012, 0.4]), np.array([0.10, 0.02, 0.60]))
    left = render_marker(rotations[0], translations[0], 7, dictionary)
    right = render_marker(rotations[1], translations[1], second_id, dictionary)
    frame = _frame(np.minimum(left, right), 123)
    executor = _FakeExecutor()
    plugin = ArucoPlugin({}, executor)
    config = {"action": "config", "instance_id": "test", "dictionary": dictionary}
    start = {"action": "start", "instance_id": "test", "input_topic": TOPIC, "camera_info": [CAMERA]}

    def forbid_overlay(*_args, **_kwargs):
        raise AssertionError("Overlay work ran in JSON-only mode")

    try:
        plugin.dispatch("aruco", config)
        enabled = plugin.dispatch("aruco", start)
        assert enabled["publish_overlay"] is True
        [node] = executor.nodes
        poses, overlay = node.publishers
        node.subscriptions[0].callback(frame)
        assert _wait_until(lambda: bool(overlay.raw_messages))
        reference = json.loads(poses.messages[0])
        assert reference["dictionary"] == dictionary
        assert [marker["id"] for marker in reference["markers"]] == [7, second_id]
        for marker, rotation, translation in zip(reference["markers"], rotations, translations):
            np.testing.assert_allclose(marker["translation_m"], translation, atol=0.006)
            actual, _ = cv2.Rodrigues(np.asarray(marker["rotation_rvec_rad"]))
            expected, _ = cv2.Rodrigues(rotation)
            assert np.linalg.norm(actual - expected) < 0.23
            assert marker["reprojection_rms_px"] < 1.5

        plugin.dispatch("aruco", {**config, "publish_overlay": False})
        disabled = plugin.dispatch("aruco", start)
        assert disabled["publish_overlay"] is False
        assert disabled["topic_out"][0] == enabled["topic_out"][0]
        assert disabled["topic_out"][1]["topic"] == ""
        assert node.destroyed and not node.worker.is_alive()
        [json_node] = executor.nodes
        [json_only] = json_node.publishers
        with monkeypatch.context() as patch:
            for op in ("drawFrameAxes", "polylines", "putText", "imencode"):
                patch.setattr(cv2, op, forbid_overlay)
            json_node.subscriptions[0].callback(frame)
            assert _wait_until(lambda: bool(json_only.messages))
            assert json.loads(json_only.messages[0]) == reference

        plugin.dispatch("aruco", {**config, "publish_overlay": True})
        restored = plugin.dispatch("aruco", start)
        assert restored["topic_out"] == enabled["topic_out"]
        assert len(executor.nodes[0].publishers) == 2
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


def test_same_topic_restart_applies_marker_size_and_calibration_changes(detector):
    executor = _FakeExecutor()
    plugin = ArucoPlugin({"rate_hz": 30}, executor)
    args = {"action": "start", "instance_id": "a", "input_topic": TOPIC,
            "camera_info": {TOPIC: CAMERA}}
    image = render_marker(np.array([np.pi, 0, 0]), np.array([0.0, 0.0, 0.4]))

    def depth(node, sec):
        node.subscriptions[0].callback(_frame(image, sec))
        assert _wait_until(lambda: bool(node.publishers[0].messages))
        [marker] = json.loads(node.publishers[0].messages[0])["markers"]
        return marker["translation_m"][2]

    try:
        without_calibration = {**args, "camera_info": {TOPIC: {**CAMERA, "K": []}}}
        with pytest.raises(ValueError, match="相机卡片开发者"):
            plugin.dispatch("aruco", without_calibration)
        assert plugin.dispatch("aruco", args)["pose_status"] == "calibrated"
        [first] = executor.nodes
        assert depth(first, 1) == pytest.approx(0.4, abs=0.006)

        plugin.dispatch("aruco", {"action": "config", "instance_id": "a", "marker_size_m": 0.2})
        assert plugin.dispatch("aruco", args)["state"] == "running"
        [second] = executor.nodes
        assert first.destroyed and not first.worker.is_alive()
        assert second.outputs == first.outputs
        assert depth(second, 2) == pytest.approx(0.8, abs=0.012)

        changed = {**CAMERA, "K": [400.0, 0, 320.0, 0, 400.0, 240.0, 0, 0, 1]}
        restart = {**args, "camera_info": {TOPIC: changed}}
        assert plugin.dispatch("aruco", restart)["state"] == "running"
        [third] = executor.nodes
        assert second.destroyed and not second.worker.is_alive()
        assert depth(third, 3) == pytest.approx(0.4, abs=0.006)
        assert plugin.dispatch("aruco", restart)["state"] == "running"
        assert executor.nodes == [third]
        with pytest.raises(ValueError, match="相机卡片开发者"):
            plugin.dispatch("aruco", without_calibration)
        assert third.destroyed and not third.worker.is_alive() and not executor.nodes
        assert plugin.dispatch("aruco", restart)["state"] == "running"
        [recovered] = executor.nodes
        assert recovered.outputs == third.outputs
        assert depth(recovered, 4) == pytest.approx(0.4, abs=0.006)
    finally:
        plugin.shutdown()
