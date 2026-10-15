"""Hardware-free DDS smoke check for two generic QR perception instances."""
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from plugins.qr_scan import QrScanPlugin
from utils.qos import CAMERA_QOS


def main():
    profile = os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', '')
    if not profile or not Path(profile).is_file():
        raise RuntimeError('Set FASTRTPS_DEFAULT_PROFILES_FILE to the deployed DDS loopback profile')
    if os.environ.get('RMW_IMPLEMENTATION') != 'rmw_fastrtps_cpp':
        raise RuntimeError('Run with the deployed RMW_IMPLEMENTATION=rmw_fastrtps_cpp')
    rclpy.init(domain_id=42)
    executor = SingleThreadedExecutor()
    run_id = 'qr_check_' + uuid4().hex[:12]
    peer = Node(run_id)
    executor.add_node(peer)
    plugin = QrScanPlugin({'scan_hz': 2, 'stale_after_s': 3, 'auto_announce': True}, executor)
    outputs = {'source_a': [], 'source_b': []}
    announcements = {'source_a': [], 'source_b': []}
    publishers = {}
    messages = {}
    try:
        for key in outputs:
            topic = f'/{run_id}/{key}/image'
            args = {'instance_id': f'{run_id}_{key}', 'input_topic': topic}
            info = plugin.dispatch('qr_scan', {**args, 'action': 'info'})
            peer.create_subscription(String, info['topic_out'][0]['topic'],
                lambda msg, key=key: outputs[key].append(json.loads(msg.data)), CAMERA_QOS)
            peer.create_subscription(String, info['topic_out'][1]['topic'],
                lambda msg, key=key: announcements[key].append(json.loads(msg.data)), CAMERA_QOS)
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
        ready = time.monotonic() + 2
        while time.monotonic() < ready:
            executor.spin_once(timeout_sec=0.03)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            for key, publisher in publishers.items():
                publisher.publish(messages[key])
            executor.spin_once(timeout_sec=0.03)
            if all(any(r['status'] == 'detected' for r in values) for values in outputs.values()) and all(announcements.values()):
                break
        for key, values in outputs.items():
            detected = next((r for r in values if r['status'] == 'detected'), None)
            assert detected and detected['codes'][0]['text'] == key.upper()+'-DDS', values[-3:]
        for key, values in announcements.items():
            assert len(values) == 1 and values[0]['text'] == '扫码成功，内容是：' + key.upper()+'-DDS', values
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.03)
            if all(any(r['status'] == 'stale' for r in values) for values in outputs.values()):
                break
        for values in outputs.values():
            stale = next((r for r in values if r['status'] == 'stale'), None)
            assert stale and stale['codes'] == []
        print(json.dumps({'dds': 'passed', 'domain_id': 42, 'independent_inputs': 2,
                          'decoded': ['SOURCE_A-DDS', 'SOURCE_B-DDS'], 'stale': 'passed', 'announcement_per_input': 1}))
    finally:
        plugin.stop()
        executor.remove_node(peer)
        peer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
