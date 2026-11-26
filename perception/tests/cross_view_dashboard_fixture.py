"""Test-only MCP source cards backed by two virtual ROS cameras."""
import json
import time
import zlib

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

import main as perception
from plugins.cross_view_object_consistency import NAME
from test_cross_view_object_consistency import capture_source


class Sources(Node):
    PREFIX = "cross_view_fixture"
    FORMATS = {"rgb": "image/jpeg", "objects": "data/json", "depth": "image/depth-zlib"}

    def __init__(self):
        super().__init__(self.PREFIX)
        self.enabled = set()
        self.sequence = 0
        self.publishers_by_name = {}
        self.routes = {}
        for side in "ab":
            for role, fmt in self.FORMATS.items():
                name = f"{side}_{role}"
                msg_type = String if role == "objects" else CompressedImage
                topic = f"/cross_view_fixture/{side}/{role}"
                self.routes[name] = {"topic": topic, "format": fmt}
                self.publishers_by_name[name] = self.create_publisher(
                    msg_type, topic, qos_profile_sensor_data)
        _, image = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
        self.jpeg = image.tobytes()
        grid = np.full((480, 640), 1000, dtype="<u2")
        self.depth = zlib.compress(grid.tobytes())
        self.create_timer(.125, self.publish_frames)

    def get_tools(self):
        return [dict(name=name, type="sensor", title=f"Fixture {name}", description="Virtual test source",
                     inputSchema={"type": "object"}, topic_out=[port])
                for name, port in self.routes.items()]

    def dispatch(self, name, args):
        action = args.get("action")
        if action == "start":
            self.enabled.add(name)
        elif action == "stop":
            self.enabled.discard(name)
        return dict(state="running" if name in self.enabled else "idle", topic_out=[self.routes[name]])

    def start(self):
        pass

    def stop(self):
        self.enabled.clear()

    def publish_frames(self):
        self.sequence += 1
        stamp = time.time_ns()
        for name in tuple(self.enabled):
            side, role = name.split("_")
            source = capture_source("ab".index(side), stamp, self.sequence)
            if role == "objects":
                u = 320 if side == "a" else 218
                message = String()
                message.data = json.dumps({
                    "source": source, "source_size": [640, 480],
                    "objects": [{"name": "block", "confidence": .95,
                                 "bbox": [u - 5, 235, u + 5, 245]}],
                })
            else:
                message = CompressedImage()
                message.header.frame_id = source["frame_id"]
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 10**9)
                message.format = "jpeg" if role == "rgb" else "16UC1; compressedDepth zlib"
                message.data = self.jpeg if role == "rgb" else self.depth
            self.publishers_by_name[name].publish(message)


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    source = Sources()
    executor.add_node(source)
    perception._bundle = perception.PerceptionBundle({
        "plugins": {NAME: {"enabled": True}, "echo_cancellation": {"enabled": False},
                    "audio_mixer": {"enabled": False}}
    }, executor)
    perception._bundle._plugins.append(source)
    import threading

    server = perception.ThreadingHTTPServer(("0.0.0.0", 15720), perception.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        executor.spin()
    finally:
        server.shutdown()
        server.server_close()
        perception._bundle.shutdown()
        executor.remove_node(source)
        source.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
