"""Local ROS2 DDS and Perception HTTP MCP integration with synthetic camera input."""
import json
import shutil
import os
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import main
from plugins.yunet_lk_mosaic.plugin import PREFIX
from utils.qos import CAMERA_QOS

MODEL = os.getenv("TEST_YUNET_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL or os.getenv("TEST_YUNET_LOCAL_DDS") != "1",
    reason="set TEST_YUNET_MODEL and use an isolated DDS session with TEST_YUNET_LOCAL_DDS=1",
)


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    pytest.fail("timed out waiting for camera, status or MCP state")


def jpeg(image, quality=90):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


@pytest.fixture
def ros_session():
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(3)
    rclpy.init()
    try:
        yield
    finally:
        rclpy.try_shutdown()
        cv2.setNumThreads(old_threads)


@pytest.mark.parametrize("delay", [0, 300])
def test_mcp_ros_lifecycle_and_image_contract(monkeypatch, delay, ros_session):
    skimage = pytest.importorskip("skimage.data")
    executor = MultiThreadedExecutor(num_threads=3)
    camera = Node("yunet_lk_test_camera")
    executor.add_node(camera)
    source = "/yunet_lk_test/camera"
    pub = camera.create_publisher(CompressedImage, source, CAMERA_QOS)
    config = {
        "echo_cancellation": {"enabled": False},
        "audio_mixer": {"enabled": False},
        PREFIX: {"enabled": True, "model_path": MODEL},
    }
    bundle = main.PerceptionBundle({"plugins": config}, executor)
    monkeypatch.setattr(main, "_bundle", bundle)
    server = main.ThreadingHTTPServer(("127.0.0.1", 0), main.make_handler())
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    running = threading.Thread(target=executor.spin, daemon=True)
    serving.start()
    running.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/mcp"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rpc(method, **params):
        request = urllib.request.Request(endpoint, data=json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=5) as response:
            return json.load(response)["result"]

    def tool(action, **args):
        content = rpc("tools/call", name=PREFIX, arguments={
            "action": action, "instance_id": "test_instance", **args,
        })["content"]
        return json.loads(content[0]["text"])

    images, status = {}, []

    def received(msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        images[stamp] = (msg.header.frame_id, msg.format, bytes(msg.data), time.monotonic())

    def send(payload, offset_ns=0):
        sent_at = time.monotonic()
        stamp = camera.get_clock().now().nanoseconds + offset_ns
        msg = CompressedImage()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        msg.header.frame_id, msg.format, msg.data = "test_camera", "bgr8; jpeg compressed bgr8", payload
        pub.publish(msg)
        return stamp, sent_at

    try:
        tools = rpc("tools/list")["tools"]
        assert [(t["name"], t["type"]) for t in tools] == [(PREFIX, "processor")]
        assert tool("info", input_topics=[])["state"] == "idle"
        ports = tool("info", input_topics=[source])["topic_out"]
        camera.create_subscription(CompressedImage, ports[0]["topic"], received, CAMERA_QOS)
        camera.create_subscription(String, ports[1]["topic"],
                                   lambda msg: status.append(json.loads(msg.data)), CAMERA_QOS)
        assert tool("start", input_topic=source)["error"] == "consent_acknowledgment_required"
        assert tool("config", target_fps=29)["state"] == "error"
        assert tool("config", config={"consent_acknowledged": True, "publish_delay_ms": delay,
                                      "detect_every_frames": 1})["state"] == "configured"
        assert tool("start", input_topics=[source])["state"] in ("loading", "running")
        wait_for(lambda: tool("info")["state"] == "running")
        assert tool("info")["metrics"]["backend"] == "opencv"
        wait_for(lambda: pub.get_subscription_count() == 1)

        rng = np.random.default_rng(11)
        bg = rng.integers(8, 247, (360, 640, 3), np.uint8)
        blank = jpeg(bg)
        decoded = cv2.imdecode(np.frombuffer(blank, np.uint8), cv2.IMREAD_COLOR)
        old_stamp, _ = send(blank, -1_000_000_000)
        wait_for(lambda: tool("info")["metrics"]["stale_input_at_receipt"] == 1)
        assert old_stamp not in images
        blank_stamp, sent_at = send(blank)
        result = wait_for(lambda: images.get(blank_stamp))
        assert result[0:3] == ("test_camera", "jpeg", jpeg(decoded, quality=72))
        assert result[3] - sent_at >= delay / 1000 - .01
        assert tool("info")["reason"] == "no_face"

        broken_stamp, _ = send(b"\xff\xd8invalid\xff\xd9")
        wait_for(lambda: tool("info")["reason"].startswith("processing_failed"))
        assert broken_stamp not in images

        portrait = cv2.cvtColor(skimage.astronaut(), cv2.COLOR_RGB2BGR)
        face_stamp, _ = send(jpeg(portrait))
        out = wait_for(lambda: images.get(face_stamp))
        img = cv2.imdecode(np.frombuffer(out[2], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (640, 640, 3)
        assert tool("info")["reason"] == "face_roi_mosaic"
        assert tool("info")["metrics"]["processing_errors"] == 1
        wait_for(lambda: status)
        assert all(s["privacy_guaranteed"] is False for s in status)
        assert list(images) == sorted(images)

        other = "second_instance"
        other_frames = []
        other_port = tool("info", instance_id=other, input_topic=source)["topic_out"][0]["topic"]
        assert other_port != ports[0]["topic"]
        camera.create_subscription(CompressedImage, other_port, other_frames.append, CAMERA_QOS)
        assert tool("config", instance_id=other, consent_acknowledged=True)["state"] == "configured"
        assert tool("start", instance_id=other, input_topic=source)["state"] in ("loading", "running")
        wait_for(lambda: tool("info", instance_id=other)["state"] == "running")
        wait_for(lambda: pub.get_subscription_count() == 2)

        if delay:
            pending_stamp, _ = send(jpeg(portrait))
            wait_for(lambda: tool("info")["metrics"]["buffer_frames"] == 1)
        assert tool("stop")["state"] == "idle"
        assert tool("info")["state"] == "idle"
        last, _ = send(blank)
        wait_for(lambda: any(
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec == last
            for msg in other_frames
        ))
        time.sleep(max(.15, delay / 1000 + .05))
        assert list(images) == [blank_stamp, face_stamp]
        if delay:
            assert pending_stamp not in images
        assert tool("stop", instance_id=other)["state"] == "idle"
        wait_for(lambda: pub.get_subscription_count() == 0)
    finally:
        bundle.shutdown()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
        executor.remove_node(camera)
        camera.destroy_node()
        executor.shutdown(timeout_sec=3)
        running.join(timeout=2)



def test_shared_downloader_cold_start_then_cache(tmp_path, monkeypatch):
    from plugins.yunet_lk_mosaic import model

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    shutil.copyfile(MODEL, mirror / model.MODEL_FILE)
    cache = tmp_path / "models"
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            return super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=mirror))
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        monkeypatch.setenv("YUNET_LK_MOSAIC_MODEL_BASE_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setattr(model, "require_models_subpath", lambda _: str(cache))
        progress = []
        path = model.resolve_model(None, lambda *fields: progress.append(fields))
        assert path == str(cache / model.MODEL_FILE)
        assert model.resolve_model(path) == path
        assert model.resolve_model(None) == path
        assert requests == ["/" + model.MODEL_FILE]
        assert progress[-1][0] == 100
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
