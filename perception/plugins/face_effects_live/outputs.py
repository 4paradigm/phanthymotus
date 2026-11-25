"""ROS output contract consumed by the existing Canvas image renderer."""
import json
import uuid
from .telemetry import tag_jpeg


class ROSOutput:
    def __init__(self, topic, executor):
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        if executor is None:
            raise ValueError("ROS output requires executor")
        self.executor, self.topic = executor, topic
        self.image_type, self.json_type = CompressedImage, String
        self.node = Node("face_effects_output_"+uuid.uuid4().hex[:12])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        try:
            self.image_pub = self.node.create_publisher(CompressedImage, topic, qos)
            self.meta_pub = self.node.create_publisher(String, topic+"/metadata", qos)
            executor.add_node(self.node)
        except Exception:
            self.node.destroy_node()
            raise
        self.closed = False

    def __call__(self, jpeg, metadata):
        image = self.image_type()
        # Header represents publish time. Original source time/clock is in JSON.
        image.header.stamp = self.node.get_clock().now().to_msg()
        image.header.frame_id = "face_effects_"+str(metadata["frame_id"])
        image.format, image.data = "jpeg", tag_jpeg(jpeg, metadata)
        message = self.json_type()
        message.data = json.dumps(metadata, allow_nan=False)
        self.image_pub.publish(image)
        self.meta_pub.publish(message)

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.remove_node(self.node)
            self.node.destroy_node()
