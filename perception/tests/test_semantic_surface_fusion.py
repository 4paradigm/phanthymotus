import json
import time
import zlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plugins.semantic_surface_fusion import (
    NAME, NSEC, Frames, Runtime, SemanticSurfaceFusionPlugin, Settings, calibration,
)
from plugins.surface_geometry import MAX_PIXELS, decode_depth, decode_mask, estimate


TOPICS = {"mask": "/camera/color/segment/person/i_xyz/mask",
          "classes": "/camera/color/segment/person/i_xyz/classes", "depth": "/camera/depth"}
K = [10, 0, 2, 0, 20, 3, 0, 0, 1]


def declarations():
    common = dict(schema="motus.camera/1", id="camera", width=8, height=8,
                  distortion_model="pinhole", K=K)
    return {TOPICS["mask"]: dict(common), TOPICS["depth"]: dict(
        common, vendor={"depth_aligned_to": "color"})}


@pytest.mark.parametrize("k", [K, [40, 0, 4, 0, 15, 1, 0, 0, 1]])
def test_projection(k):
    mask = np.ones((8, 8), np.uint8)
    depth = decode_depth(zlib.compress(np.full((8, 8), 4000, "<u2").tobytes()), 8, 8, .0005)
    result, points = estimate(mask, depth, 1, k, 16)
    assert result["state"] == "valid"
    assert result["visible_center_m"] == pytest.approx([(3.5-k[2])*2/k[0], (3.5-k[5])*2/k[4], 2])
    assert points.shape == (16, 3)
    assert points[0] == pytest.approx([-k[2]*2/k[0], -k[5]*2/k[4], 2])


@pytest.mark.parametrize("count, expected", [(0, "insufficient_evidence"), (5, "insufficient_evidence"),
                                             (20, "degraded"), (64, "valid")])
def test_depth_support(count, expected):
    depth = np.zeros(64, np.float32)
    depth[:count] = 2
    depth[-1] = np.inf
    result, _ = estimate(np.ones((8, 8)), depth.reshape(8, 8), 1, K, 0)
    assert result["state"] == expected


def test_outlier_and_empty_and_wrong_size():
    mask = np.ones((8, 8), np.uint8)
    depth = np.full(mask.shape, 2.0)
    depth[0, 0] = 8
    result, _ = estimate(mask, depth, 1, K, 0)
    assert result["visible_bounds_m"]["max"][2] == 2
    assert result["point_count"] == 63
    assert estimate(mask, depth, 2, K, 0)[0]["reason"] == "empty_mask"
    with pytest.raises(ValueError, match="image_size_mismatch"):
        estimate(mask, depth[:4], 1, K, 0)
    assert decode_mask(cv2.imencode(".png", mask)[1].tobytes(), 8, 8).shape == (8, 8)
    with pytest.raises(ValueError, match="invalid_mask_png"):
        decode_mask(cv2.imencode(".png", mask)[1].tobytes(), 4, 4)


def test_limits():
    mask = np.ones((1080, 1920), np.uint8)
    result, cloud = estimate(mask, np.full(mask.shape, 2, np.float32), 1, K, 100)
    assert result["point_count"] == MAX_PIXELS
    assert len(cloud) == 100
    with pytest.raises(ValueError, match="depth_size"):
        decode_depth(zlib.compress(bytes(8192)), 8, 8, .001)
    with pytest.raises(ValueError, match="size_mismatch"):
        estimate(np.ones((1081, 1920)), np.ones((1081, 1920)), 1, K, 0)


def test_time_matching_replay_and_reconnect():
    frames = Frames(Settings())
    t = 3 * NSEC
    frames.add("mask", t, "cam", "mask")
    frames.add("classes", t, "cam", "table")
    frames.add("depth", t, "other", "wrong camera")
    frames.add("depth", t + 50_000_000, "cam", "late")
    assert frames.take(t + 60_000_000) is None
    frames.add("depth", t + 4_000_000, "cam", "depth")
    assert frames.take(t + 60_000_000)[-1] == "depth"
    frames.add("mask", t, "cam", "replay")
    assert frames.take(t + NSEC // 2) is None
    for role in TOPICS:
        frames.add(role, t + 2 * NSEC, "cam", role)
    assert frames.take(t + 2 * NSEC)[-1] == "depth"
    assert frames.take(t + 4 * NSEC) is None


@pytest.mark.parametrize("key,value", [("id", "other"), ("width", 4), ("vendor", {}),
                                       ("K", None), ("distortion_model", "unknown")])
def test_invalid_calibration(key, value):
    data = declarations()
    data[TOPICS["depth"]][key] = value
    with pytest.raises(ValueError):
        calibration(data, TOPICS)


@pytest.mark.parametrize("budget", [1, 10])
def test_multiple_classes(budget):
    mask = np.ones((8, 8), np.uint8)
    mask[:, 4:] = 2
    runtime = SimpleNamespace(calibration=(8, 8, K, "camera"), settings=Settings(max_points=budget))
    report = dict(width=8, height=8, classes=[
        dict(id=0, name="background", pixels=0), dict(id=1, name="car", pixels=32),
        dict(id=2, name="bus", pixels=32), dict(id=3, name="chair", pixels=0)])
    payload, cloud = Runtime.fuse(runtime, (NSEC, "optical", SimpleNamespace(
        format="png", data=cv2.imencode(".png", mask)[1].tobytes()), report,
        SimpleNamespace(format="16UC1; compressedDepth zlib",
                        data=zlib.compress(np.full(mask.shape, 1000, "<u2").tobytes()))))
    assert [o["class_name"] for o in payload["observations"]] == ["car", "bus", "chair"]
    assert len(cloud) == budget


def test_mcp_contract():
    tool, = SemanticSurfaceFusionPlugin({}, None).get_tools()
    assert tool["name"] == NAME
    assert tool["type"] == "processor"
    assert [p["format"] for p in tool["topic_in"]] == ["image/png", "data/json", "image/depth-zlib"]
    assert tool["inputSchema"]["properties"]["action"]["enum"] == ["start", "stop", "info", "config"]
    with pytest.raises(ValueError):
        Settings(max_points=True)


def test_two_chairs_have_no_object_center_or_instance_id():
    mask = np.zeros((16, 24), dtype=np.uint8)
    mask[3:13, 2:8] = 1
    mask[3:13, 16:22] = 1
    depth = np.full(mask.shape, 2, dtype=np.float32)
    result, cloud = estimate(mask, depth, 1, [20, 0, 12, 0, 20, 8, 0, 0, 1], 20)

    assert result["state"] == "degraded"
    assert result["reason"] == "multiple_class_regions"
    assert result["geometry_scope"] == "class_visible_surface"
    assert result["instance_attribution"] == "unknown"
    assert result["multiple_regions_detected"] is True
    assert result["visible_center_m"] is None
    assert result["visible_bounds_m"]["min"][0] < 0 < result["visible_bounds_m"]["max"][0]
    assert result["point_count"] == 120
    assert len(cloud) == 20
    assert "instance_id" not in result


def test_projection_matches_reference_quantiles_and_samples():
    rng = np.random.default_rng(340)
    depth = rng.uniform(1.5, 2.5, (64, 80)).astype(np.float32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[5:59, 3:76] = 1
    mask[10:12, 20:22] = 0
    rows, cols = np.nonzero(mask)
    z = depth[rows, cols]
    expected = np.column_stack(((cols - 40) * z / 91, (rows - 32) * z / 155, z))
    lo, center, hi = np.percentile(expected, [2, 50, 98], axis=0)
    result, cloud = estimate(mask, depth, 1, [91, 0, 40, 0, 155, 32, 0, 0, 1], 53)

    np.testing.assert_allclose(result["visible_center_m"], center, atol=1e-7)
    np.testing.assert_allclose(result["visible_bounds_m"]["min"], lo, atol=1e-7)
    np.testing.assert_allclose(result["visible_bounds_m"]["max"], hi, atol=1e-7)
    np.testing.assert_allclose(cloud, expected[np.linspace(0, len(z) - 1, 53, dtype=int)], atol=1e-7)
    assert result["multiple_regions_detected"] is False


def real_ros():
    rclpy = pytest.importorskip("rclpy")
    if not getattr(rclpy, "__file__", None):
        pytest.skip("Run with --noconftest and source the ROS 2 environment")
    return rclpy


def test_bundle_registration_from_shipped_config(monkeypatch):
    real_ros()
    from http.client import HTTPConnection
    from http.server import HTTPServer
    from pathlib import Path
    from threading import Thread

    import yaml
    import main

    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8"))
    config["plugins"] = {
        NAME: config["plugins"][NAME],
        "echo_cancellation": {"enabled": False},
        "audio_mixer": {"enabled": False},
    }
    bundle = main.PerceptionBundle(config, None)
    monkeypatch.setattr(main, "_bundle", bundle)
    with HTTPServer(("127.0.0.1", 0), main.make_handler()) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=4)

        def rpc(method, params=None):
            request = dict(jsonrpc="2.0", id=1, method=method, params=params)
            conn.request("POST", "/mcp", json.dumps(request), {"Content-Type": "application/json"})
            response = conn.getresponse()
            assert response.status == 200
            return json.loads(response.read())

        try:
            tools = rpc("tools/list")["result"]["tools"]
            assert NAME in [tool["name"] for tool in tools]
            answer = rpc("tools/call", dict(name=NAME, arguments={"action": "info"}))
            assert json.loads(answer["result"]["content"][0]["text"])["state"] == "idle"
            answer = rpc("tools/call", dict(name=NAME, arguments={"action": "config", "max_points": -1}))
            assert "max_points" in answer["error"]["message"]
        finally:
            conn.close()
            server.shutdown()
            worker.join()
            bundle.shutdown()


def test_real_ros_transport_lifecycle_and_watchdog():
    rclpy = real_ros()
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, PointCloud2
    from std_msgs.msg import String

    rclpy.init()
    executor = SingleThreadedExecutor()
    client = Node("fusion_integration_client")
    executor.add_node(client)
    plugin = SemanticSurfaceFusionPlugin({}, executor)
    received, clouds = [], []
    args = dict(action="start", input_topics=list(TOPICS.values()), camera_info=declarations(),
                max_points=16, stale_ms=350, timeout_ms=200)
    real_fuse = Runtime.fuse

    def wait_until(predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert predicate(), received[-5:]

    try:
        info = plugin.dispatch(NAME, args)
        old_node = next(iter(plugin.instances.values()))[1].node
        assert plugin.dispatch(NAME, args) == info
        assert next(iter(plugin.instances.values()))[1].node is old_node
        client.create_subscription(String, info["topic_out"][0]["topic"],
                                   lambda msg: received.append(json.loads(msg.data)), 10)
        client.create_subscription(PointCloud2, info["topic_out"][1]["topic"], clouds.append, 10)
        pubs = {role: client.create_publisher(String if role == "classes" else CompressedImage, topic, 10)
                for role, topic in TOPICS.items()}
        wait_until(lambda: all(p.get_subscription_count() for p in pubs.values()) and received)

        def send(roles=tuple(TOPICS)):
            capture = client.get_clock().now().to_msg()
            for role in roles:
                if role == "classes":
                    msg = String(data=json.dumps(dict(width=8, height=8, frame_id="",
                        stamp=dict(sec=capture.sec, nanosec=capture.nanosec), classes=[
                            dict(id=0, name="background", pixels=0), dict(id=1, name="car", pixels=64)])))
                else:
                    msg = CompressedImage()
                    msg.header.stamp = capture
                    msg.header.frame_id = ""
                    if role == "mask":
                        msg.format = "png"
                        msg.data = cv2.imencode(".png", np.ones((8, 8), np.uint8))[1].tobytes()
                    else:
                        msg.format = "16UC1; compressedDepth zlib"
                        msg.data = zlib.compress(np.full((8, 8), 1000, "<u2").tobytes())
                pubs[role].publish(msg)

        send()
        wait_until(lambda: clouds and any(p["state"] == "valid" for p in received))
        assert clouds[-1].width == 16
        assert clouds[-1].header.frame_id == "camera_color_optical_frame"
        time.sleep(.4)
        send(("mask", "classes"))
        wait_until(lambda: received[-1]["reason"] == "stream_stale:depth")
        assert received[-1]["observations"] == []
        assert received[-1]["visible_bounds_m"] is None
        send(("depth",))
        wait_until(lambda: received[-1]["reason"] == "stream_stale:depth,classes"
                   or received[-1]["reason"] == "stream_stale:mask,classes")
        send()
        wait_until(lambda: received[-1]["state"] == "valid")

        def slow_fuse(self, pair):
            time.sleep(.25)
            return real_fuse(self, pair)

        Runtime.fuse = slow_fuse
        send()
        wait_until(lambda: received[-1]["reason"] == "processing_timeout")
        runtime = next(iter(plugin.instances.values()))[1]
        assert plugin.dispatch(NAME, {"action": "stop"})["state"] == "idle"
        assert plugin.dispatch(NAME, {"action": "stop"})["state"] == "idle"
        assert runtime.node not in executor.get_nodes() and not runtime.worker.is_alive()
        wait_until(lambda: all(p.get_subscription_count() == 0 for p in pubs.values()))
        assert plugin.dispatch(NAME, args)["state"] == "running"
    finally:
        Runtime.fuse = real_fuse
        plugin.shutdown()
        executor.remove_node(client)
        client.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
