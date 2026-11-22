"""Run with PYTHONPATH=perception in a ROS 2 environment."""
import json
import threading
import time
import zlib
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import main as perception
from plugins.cross_view_object_consistency import NAME
from test_cross_view_object_consistency import capture_rig, capture_source


def main():
    rclpy.init()
    executor = SingleThreadedExecutor()
    fixture = Node("cross_view_virtual_rig")
    executor.add_node(fixture)
    perception._bundle = perception.PerceptionBundle(
        {"plugins": {NAME: {"enabled": True}}}, executor)
    server = perception.ThreadingHTTPServer(("127.0.0.1", 0), perception.make_handler())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def rpc(method, params=None):
        request = Request(f"http://127.0.0.1:{server.server_port}/mcp",
                          json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                      "params": params or {}}).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.load(response)["result"]

    def call(arguments):
        result = rpc("tools/call", {"name": NAME, "arguments": arguments})
        return json.loads(result["content"][0]["text"])
    config = capture_rig()
    config["topics"] = {side: {role: f"/cross_view_test/{side}/{role}"
                               for role in ("rgb", "objects", "depth")} for side in "ab"}
    publishers = {(side, role): fixture.create_publisher(
        String if role == "objects" else CompressedImage, topic, qos_profile_sensor_data)
        for side, roles in config["topics"].items() for role, topic in roles.items()}
    outputs = []
    sub = fixture.create_subscription(
        String, f"/perception/{NAME}/state", lambda m: outputs.append(json.loads(m.data)), 10)
    args = {"action": "start", "input_topics": [p.topic_name for p in publishers.values()]}
    raw = np.full((480, 640), 1000, dtype="<u2").tobytes()
    compressed = zlib.compress(raw)
    ok, image = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    assert ok
    jpeg = image.tobytes()

    sequence = 0

    def send_pair(sides="ab", generation=1):
        nonlocal sequence
        sequence += 1
        stamp = time.time_ns()
        for side in sides:
            u = 320 if side == "a" else 218
            source = capture_source("ab".index(side), stamp, sequence, generation)
            for role in ("rgb", "objects", "depth"):
                if role == "objects":
                    msg = String()
                    msg.data = json.dumps({
                        "source": source,
                        "source_size": [640, 480],
                        "objects": [{"name": "block", "confidence": .95,
                                     "bbox": [u-5, 235, u+5, 245]}],
                    })
                else:
                    msg = CompressedImage()
                    msg.header.frame_id = source["frame_id"]
                    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
                    msg.format = "jpeg" if role == "rgb" else "16UC1; compressedDepth zlib"
                    msg.data = jpeg if role == "rgb" else compressed
                publishers[side, role].publish(msg)

    try:
        rpc("initialize")
        tools = rpc("tools/list")["tools"]
        tool = next(tool for tool in tools if tool["name"] == NAME)
        assert len(tool["topic_in"]) == 6 and tool["type"] == "processor"
        assert call({"action": "config", "rig_json": json.dumps(config)})["calibrated"]
        assert call({**args, "input_topics": args["input_topics"][:-1]})["state"] == "error"
        for cycle in range(3):
            started = call(args)
            assert started["state"] == "running"
            assert {port["topic"] for port in started["topic_in"]} == set(args["input_topics"])
            outputs.clear()
            deadline = time.monotonic() + 8
            while not any(r["state"] == "consistent" for r in outputs):
                assert time.monotonic() < deadline, outputs[-5:]
                send_pair()
                until = time.monotonic() + .12
                while time.monotonic() < until:
                    executor.spin_once(timeout_sec=.02)
            if cycle == 0:
                outputs.clear()
                deadline = time.monotonic() + 4
                while not any(r["reason"] == "camera_b_rgb_stream_stale" for r in outputs):
                    assert time.monotonic() < deadline, outputs[-5:]
                    send_pair("a")
                    until = time.monotonic() + .12
                    while time.monotonic() < until:
                        executor.spin_once(timeout_sec=.02)
                outputs.clear()
                deadline = time.monotonic() + 4
                while not any(r["state"] == "consistent" for r in outputs):
                    assert time.monotonic() < deadline, outputs[-5:]
                    send_pair(generation=2)
                    until = time.monotonic() + .12
                    while time.monotonic() < until:
                        executor.spin_once(timeout_sec=.02)
            assert call({"action": "stop"})["state"] == "idle"
            assert executor.get_nodes() == [fixture]
        print("MCP discovery/config and 3 ROS 2 start/publish/stop cycles and dual-session recovery passed")
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
        perception._bundle.shutdown()
        fixture.destroy_subscription(sub)
        executor.remove_node(fixture)
        fixture.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
