"""Optional ROS2 adapter. Input conversion is testable without ROS imports."""
import uuid
import threading
import numpy as np
import cv2
from .push import PushSource


def decode_raw(msg):
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}
    if msg.encoding not in channels:
        raise ValueError("unsupported ROS Image encoding")
    ch = channels[msg.encoding]
    if not 2 <= msg.width <= 8192 or not 2 <= msg.height <= 8192 or msg.width*msg.height > 16777216:
        raise ValueError("unsupported ROS Image dimensions")
    if msg.step < msg.width*ch or len(msg.data) != msg.step*msg.height:
        raise ValueError("invalid ROS Image stride/data length")
    pixels = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    pixels = pixels[:, :msg.width*ch].reshape(msg.height, msg.width, ch)
    if msg.encoding == "bgr8":
        return pixels.copy()
    code = {"rgb8": cv2.COLOR_RGB2BGR, "bgra8": cv2.COLOR_BGRA2BGR,
            "rgba8": cv2.COLOR_RGBA2BGR, "mono8": cv2.COLOR_GRAY2BGR}[msg.encoding]
    return cv2.cvtColor(pixels, code)


def decode_compressed(msg):
    if "compressedDepth" in msg.format or not any(x in msg.format.lower() for x in ("jpeg", "jpg", "png")):
        raise ValueError("unsupported CompressedImage format")
    if len(msg.data) > 20_000_000:
        raise ValueError("compressed image too large")
    image = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0]*image.shape[1] > 16777216:
        raise ValueError("invalid compressed image")
    return image


class ROSSource(PushSource):
    def __init__(self, topic, kind, executor):
        super().__init__()
        if not topic or executor is None:
            raise ValueError("ROS input requires explicit topic and a running executor")
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, CompressedImage
        self.executor = executor
        self.node = Node("face_effects_input_"+uuid.uuid4().hex[:12])
        self.close_lock = threading.Lock()
        self.decoder = decode_raw if kind == "ros_image" else decode_compressed
        message = Image if kind == "ros_image" else CompressedImage
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        try:
            self.subscription = self.node.create_subscription(message, topic, self._callback, qos)
            executor.add_node(self.node)
        except Exception:
            self.node.destroy_node()
            raise

    def _callback(self, msg):
        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
            self.submit(self.decoder(msg), timestamp, "ros2", {"ros_frame_id": msg.header.frame_id})
            self.error = ""
        except (ValueError, TypeError, cv2.error):
            self.error = "invalid ROS image"

    def close(self):
        with self.close_lock:
            if not self.closed:
                super().close()
                self.executor.remove_node(self.node)
                self.node.destroy_node()
