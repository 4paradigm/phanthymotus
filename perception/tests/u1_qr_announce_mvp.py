"""One-shot U1 smoke check; synthetic JPEG and optional real native TTS.

This test bridge is not a deployed card or an agent command source.
"""
import argparse
import json
import time
import urllib.request

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from plugins.qr_scan import QrScanPlugin
from rclpy.qos import QoSProfile, ReliabilityPolicy


def tts(action, **args):
    request = urllib.request.Request('http://127.0.0.1:15740/mcp',
        data=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                         'params': {'name': 'tts', 'arguments': {'action': action, **args}}}).encode(),
        headers={'Content-Type': 'application/json'})
    reply = json.load(urllib.request.urlopen(request, timeout=10))
    assert 'result' in reply, reply
    result = json.loads(reply['result']['content'][0]['text'])
    assert result.get('state') != 'error', result
    return result


def main(speak=False):
    if speak:
        assert not tts('info').get('active'), 'Native TTS is busy; leave existing playback untouched'
    rclpy.init()
    executor = SingleThreadedExecutor()
    peer = Node('qr_announce_mvp_camera')
    executor.add_node(peer)
    plugin = QrScanPlugin({'auto_announce': True, 'scan_hz': 2}, executor)
    key = 'u1_announce_mvp'
    args = {'instance_id': key, 'input_topic': '/qr_announce_mvp/image'}
    messages = []
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    info = plugin.dispatch('qr_scan', {**args, 'action': 'info'})
    peer.create_subscription(String, info['topic_out'][1]['topic'],
                             lambda msg: messages.append(json.loads(msg.data)), qos)
    publisher = peer.create_publisher(CompressedImage, args['input_topic'], qos)
    pixels = cv2.resize(cv2.QRCodeEncoder_create().encode('U1-MVP-001'), (290,290),
                        interpolation=cv2.INTER_NEAREST)
    ok, data = cv2.imencode('.jpg', pixels)
    assert ok
    frame = CompressedImage()
    frame.format = 'jpeg'
    frame.data = data.tobytes()
    try:
        started = plugin.dispatch('qr_scan', {**args, 'action': 'start'})
        assert started['state'] == 'running', started
        # Establish DDS matches before the first new-event announcement.
        until = time.monotonic()+3
        while time.monotonic()<until:
            executor.spin_once(timeout_sec=0.05)
        until = time.monotonic()+20
        while time.monotonic()<until and not messages:
            publisher.publish(frame)
            executor.spin_once(timeout_sec=0.05)
        assert messages and messages[0]['text'] == '扫码成功，内容是：U1-MVP-001', messages
        first = messages[0]
        until = time.monotonic()+2
        while time.monotonic()<until:
            publisher.publish(frame)
            executor.spin_once(timeout_sec=0.05)
        assert len(messages) == 1, messages
        print(json.dumps({'qr': 'passed', 'announcement': first['text'], 'announcements': len(messages)}, ensure_ascii=False), flush=True)
        if speak:
            action_id = 'qr-announce-mvp-' + str(int(time.time()))
            queued = tts('speak', text=first['text'], action_id=action_id)
            print(json.dumps({'native_tts': queued.get('state'), 'action_id': action_id}), flush=True)
            until = time.monotonic()+30
            while time.monotonic()<until:
                active = tts('info').get('active')
                if not active or active.get('action_id') != action_id:
                    print('native_tts_action_cleared=true (audible playback must be confirmed by user)', flush=True)
                    break
                time.sleep(0.5)
            else:
                print('native_tts_completion_not_observed=true', flush=True)
    finally:
        plugin.stop()
        executor.remove_node(peer)
        peer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speak', action='store_true', help='Actually speak once on U1')
    main(parser.parse_args().speak)
