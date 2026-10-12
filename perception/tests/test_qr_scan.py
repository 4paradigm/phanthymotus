"""Real JPEG decoding plus multi-instance perception stream contracts."""
import json
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from vision_stubs import _FakeExecutor
from plugins.qr_scan import QrScanPlugin
from utils.qr_decoder import QrDecoder, QrTracker, self_check


def jpeg(text=None):
    image = np.full((290, 290), 255, dtype=np.uint8) if text is None else cv2.resize(
        cv2.QRCodeEncoder_create().encode(text), (290, 290), interpolation=cv2.INTER_NEAREST)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    return data.tobytes()


@pytest.mark.parametrize('text', ['U1-001', '展品编号：001', 'https://example.com/001'])
def test_real_decode(text):
    result = QrDecoder().decode(jpeg(text))
    assert result[0]['text'] == text
    assert len(result[0]['corners_px']) == 4


def test_decoder_selfcheck_blank_and_invalid():
    self_check()
    assert QrDecoder().decode(jpeg()) == []
    for data in (b'', b'broken', b'x'*5_000_001):
        with pytest.raises(ValueError):
            QrDecoder().decode(data)


def test_dedup():
    tracker = QrTracker(3)
    assert tracker.update([{'text': 'A'}], 0)
    assert tracker.update([{'text': 'A'}], 1) == []
    assert tracker.update([{'text': 'A'}], 5)


def wait(plugin, key, status, sequence=None):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        result = plugin.dispatch('qr_scan', {'action': 'read', 'instance_id': key})
        if result.get('status') == status and (sequence is None or result.get('frame_sequence') == sequence):
            return result
        time.sleep(0.01)
    pytest.fail(str(result))


def send(stream, text):
    stream.node.subscriptions[0].callback(SimpleNamespace(format='jpeg', data=jpeg(text)))


def test_two_robot_streams_independent_and_stop_restart():
    executor = _FakeExecutor()
    plugin = QrScanPlugin({'scan_hz': 10, 'stale_after_s': 0.5}, executor)
    try:
        for key, topic in [('u1', '/vision/camera/left'), ('g1', '/g1/camera/rgb')]:
            predicted = plugin.dispatch('qr_scan', {'action': 'info', 'instance_id': key, 'input_topic': topic})
            started = plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': key, 'input_topic': topic})
            assert started['state'] == 'running'
            assert predicted['topic_out'] == started['topic_out']
            assert all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) for part in started['topic_out'][0]['topic'].split('/')[1:])
            assert plugin._streams[key].node.subscriptions[0].topic == topic
        assert len(executor.nodes) == 2
        assert plugin._streams['u1'].output_topic != plugin._streams['g1'].output_topic
        send(plugin._streams['u1'], 'U1-001')
        send(plugin._streams['g1'], 'G1-001')
        a = wait(plugin, 'u1', 'detected', 1)
        b = wait(plugin, 'g1', 'detected', 1)
        assert a['codes'][0]['text'] == 'U1-001'
        assert b['codes'][0]['text'] == 'G1-001'
        assert a['new_events'][0]['sequence'] == b['new_events'][0]['sequence'] == 1
        # Check actual JSON published by the fake endpoint, not just read().
        messages = plugin._streams['u1'].node.publishers[0].messages
        assert any(json.loads(m)['status'] == 'detected' for m in messages)
        send(plugin._streams['u1'], 'U1-001')
        assert wait(plugin, 'u1', 'detected', 2)['new_events'] == []
        assert wait(plugin, 'u1', 'stale')['codes'] == []
        old_node = plugin._streams['u1'].node
        plugin.dispatch('qr_scan', {'action': 'stop', 'instance_id': 'u1'})
        assert old_node.destroyed and len(executor.nodes) == 1
        assert plugin.dispatch('qr_scan', {'action': 'read', 'instance_id': 'u1'})['state'] == 'idle'
        plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': 'u1', 'input_topic': '/vision/camera/left'})
        assert wait(plugin, 'u1', 'waiting_for_frame')['session_id'] != a['session_id']
    finally:
        plugin.stop()
    assert executor.nodes == []


def test_rebind_idempotent_and_bad_input():
    executor = _FakeExecutor()
    plugin = QrScanPlugin({'scan_hz': 10}, executor)
    try:
        args = {'action': 'start', 'instance_id': 'one', 'input_topic': '/a/image'}
        plugin.dispatch('qr_scan', args)
        plugin.dispatch('qr_scan', args)
        assert len(executor.nodes) == 1
        previous = executor.nodes[0]
        predicted = plugin.dispatch('qr_scan', {'action': 'info', 'instance_id': 'one', 'input_topic': '/b/image'})
        rebound = plugin.dispatch('qr_scan', {**args, 'input_topic': '/b/image'})
        assert predicted['topic_out'] == rebound['topic_out']
        assert previous.destroyed and len(executor.nodes) == 1
        stream = plugin._streams['one']
        stream.node.subscriptions[0].callback(SimpleNamespace(format='png', data=b'bad'))
        assert wait(plugin, 'one', 'decode_error')['codes'] == []
        assert plugin.dispatch('qr_scan', {'action': 'config', 'instance_id': 'one', 'config': {'scan_hz': 0}})['state'] == 'error'
        assert len(executor.nodes) == 1
        plugin.dispatch('qr_scan', {'action': 'config', 'instance_id': 'one', 'config': {'scan_hz': 3}})
        assert executor.nodes == []
    finally:
        plugin.stop()


def test_missing_input_and_add_node_failure():
    executor = _FakeExecutor()
    plugin = QrScanPlugin({}, executor)
    assert plugin.dispatch('qr_scan', {'action': 'start'})['state'] == 'error'
    def fail(node):
        raise RuntimeError('failed registration')
    executor.add_node = fail
    result = plugin.dispatch('qr_scan', {'action': 'start', 'input_topic': '/camera'})
    assert result['state'] == 'error' and not plugin._streams


def test_bundle_registers_qr_with_enabled_config():
    import main
    plugin = main.PerceptionBundle({'plugins': {'qr_scan': {'enabled': True}}}, _FakeExecutor())
    try:
        tool = next(t for t in plugin.get_all_tools() if t['name'] == 'qr_scan')
        assert tool['type'] == 'processor' and tool['multiInstance'] is True
        assert plugin.dispatch('qr_scan', {'action': 'info', 'input_topic': '/vision/camera/left'})['state'] == 'idle'
    finally:
        for p in plugin._plugins:
            p.stop()
