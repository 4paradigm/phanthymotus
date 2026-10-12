"""Hardware-free DDS smoke check for two generic QR perception instances."""
import json
import time

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from plugins.qr_scan import QrScanPlugin
from utils.qos import CAMERA_QOS


def main():
    rclpy.init(domain_id=142)
    executor = SingleThreadedExecutor()
    peer = Node('qr_test_camera')
    executor.add_node(peer)
    plugin = QrScanPlugin({'scan_hz': 2, 'stale_after_s': 3}, executor)
    outputs = {'u1': [], 'g1': []}
    publishers = {}
    messages = {}
    try:
        for key, topic in [('u1', '/vision/camera/left'), ('g1', '/g1/camera/rgb')]:
            args = {'instance_id': key, 'input_topic': topic}
            info = plugin.dispatch('qr_scan', {**args, 'action': 'info'})
            peer.create_subscription(String, info['topic_out'][0]['topic'],
                lambda msg, key=key: outputs[key].append(json.loads(msg.data)), CAMERA_QOS)
            started = plugin.dispatch('qr_scan', {**args, 'action': 'start'})
            assert started['state'] == 'running', started
            publishers[key] = peer.create_publisher(CompressedImage, topic, CAMERA_QOS)
            image = cv2.resize(cv2.QRCodeEncoder_create().encode(key.upper()+'-DDS'), (290, 290),
                               interpolation=cv2.INTER_NEAREST)
            ok, data = cv2.imencode('.jpg', image)
            assert ok
            message = CompressedImage()
            message.format = 'jpeg'
            message.data = data.tobytes()
            messages[key] = message
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            for key, publisher in publishers.items():
                publisher.publish(messages[key])
            executor.spin_once(timeout_sec=0.03)
            if all(any(r['status'] == 'detected' for r in values) for values in outputs.values()):
                break
        for key, values in outputs.items():
            detected = next((r for r in values if r['status'] == 'detected'), None)
            assert detected and detected['codes'][0]['text'] == key.upper()+'-DDS', values[-3:]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.03)
            if all(any(r['status'] == 'stale' for r in values) for values in outputs.values()):
                break
        for values in outputs.values():
            stale = next((r for r in values if r['status'] == 'stale'), None)
            assert stale and stale['codes'] == []
        print(json.dumps({'dds': 'passed', 'independent_inputs': 2,
                          'decoded': ['U1-DDS', 'G1-DDS'], 'stale': 'passed'}))
    finally:
        plugin.stop()
        executor.remove_node(peer)
        peer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
