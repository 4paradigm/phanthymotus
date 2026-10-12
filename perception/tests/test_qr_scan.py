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
        for key, topic in [('u1', '/vision/camera/left'), ('tianyi', '/tianyi/camera/head')]:
            predicted = plugin.dispatch('qr_scan', {'action': 'info', 'instance_id': key, 'input_topic': topic})
            started = plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': key, 'input_topic': topic})
            assert started['state'] == 'running'
            assert predicted['topic_out'] == started['topic_out']
            assert all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) for part in started['topic_out'][0]['topic'].split('/')[1:])
            assert plugin._streams[key].node.subscriptions[0].topic == topic
        assert len(executor.nodes) == 2
        assert plugin._streams['u1'].output_topic != plugin._streams['tianyi'].output_topic
        send(plugin._streams['u1'], 'U1-001')
        send(plugin._streams['tianyi'], 'TIANYI-001')
        a = wait(plugin, 'u1', 'detected', 1)
        b = wait(plugin, 'tianyi', 'detected', 1)
        assert a['codes'][0]['text'] == 'U1-001'
        assert b['codes'][0]['text'] == 'TIANYI-001'
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


def test_integrated_announcement_once_and_cooldown():
    from plugins.qr_scan import _QrStream
    stream = _QrStream(None, '/camera', '/test/result', {'auto_announce': True})
    result = {'status': 'detected', 'new_events': [{'sequence': 1, 'text': 'U1-001'}]}
    assert stream._announcement(result, 0)['text'] == '扫码成功，内容是：U1-001'
    assert stream._announcement(result, 1) is None
    assert stream._announcement({'status': 'detected', 'new_events': []}, 11) is None
    result['new_events'] = [{'sequence': 2, 'text': 'https://example.com/private'}]
    assert stream._announcement(result, 11)['text'] == '扫码成功，识别到一个链接。'
    result['new_events'] = [{'sequence': 3, 'text': '字'*1000}]
    message = stream._announcement(result, 22)['text']
    assert len(message) < 120 and '已省略' in message
    disabled = _QrStream(None, '/camera', '/test/result', {})
    assert disabled._announcement(result, 0) is None


def test_integrated_announcement_published_from_real_qr():
    executor = _FakeExecutor()
    plugin = QrScanPlugin({'auto_announce': True, 'scan_hz': 10}, executor)
    try:
        started = plugin.dispatch('qr_scan', {'action': 'start', 'input_topic': '/camera'})
        stream = plugin._streams['/camera']
        send(stream, 'U1-001')
        wait(plugin, '/camera', 'detected', 1)
        until = time.monotonic()+2
        publisher = stream.node.publishers[1]
        while not publisher.messages and time.monotonic()<until:
            time.sleep(0.01)
        assert json.loads(publisher.messages[0])['text'] == '扫码成功，内容是：U1-001'
        send(stream, 'U1-001')
        wait(plugin, '/camera', 'detected', 2)
        assert len(publisher.messages) == 1
        assert publisher.topic == started['topic_out'][1]['topic']
    finally:
        plugin.stop()


@pytest.mark.parametrize('content', ['wxp://example', 'wx://example', 'weixin://example', 'alipays://example', 'WXPay://example', 'mailto:user@example.com', 'www.example.com/001', 'wxp.cn/001'])
def test_application_links_are_not_read_as_raw_text(content):
    from plugins.qr_scan import _QrStream
    stream = _QrStream(None, '/camera', '/test/result', {'auto_announce': True})
    result = {'status': 'detected', 'new_events': [{'sequence': 1, 'text': content}]}
    assert stream._announcement(result, 0)['text'] == '扫码成功，识别到一个链接。'


def test_announcement_disabled_instance_does_not_affect_other_robot():
    from plugins.qr_scan import _QrStream
    u1 = _QrStream(None, '/vision/camera/left', '/u1/result', {'auto_announce': True})
    tianyi = _QrStream(None, '/tianyi/camera/head', '/tianyi/result', {'auto_announce': False})
    event = {'status': 'detected', 'new_events': [{'sequence': 1, 'text': 'ITEM-001'}]}
    assert u1._announcement(event, 0) is not None
    assert tianyi._announcement(event, 0) is None


def test_native_tts_for_u1_and_tianyi_isolated(monkeypatch):
    import plugins.qr_scan as module
    clients = []
    class Client:
        def __init__(self, url):
            self.url, self.calls = url, []
            clients.append(self)
        def speak(self, text, action_id):
            self.calls.append((text, action_id))
            return {'state': 'submitted'}
    monkeypatch.setattr(module, 'LocalTTS', Client)
    executor = _FakeExecutor()
    plugin = QrScanPlugin({'auto_announce': True, 'scan_hz': 10}, executor)
    try:
        for key, topic, port in [('u1', '/vision/camera/left', 15740), ('tianyi', '/tianyi/camera/head', 15707)]:
            configured = plugin.dispatch('qr_scan', {'action': 'config', 'instance_id': key,
                'config': {'announce_backend': 'native_mcp', 'tts_mcp_url': f'http://127.0.0.1:{port}/mcp'}})
            assert configured['state'] == 'idle'
            plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': key, 'input_topic': topic})
            send(plugin._streams[key], key.upper()+'-001')
        wait(plugin, 'u1', 'detected')
        wait(plugin, 'tianyi', 'detected')
        deadline = time.monotonic()+3
        # config validation creates disposable clients too; select active ones.
        active = [s._native_tts for s in plugin._streams.values()]
        while not all(c.calls for c in active) and time.monotonic()<deadline:
            time.sleep(0.01)
        assert active[0].url.endswith(':15740/mcp') and active[1].url.endswith(':15707/mcp')
        assert active[0].calls[0][0] == '扫码成功，内容是：U1-001'
        assert active[1].calls[0][0] == '扫码成功，内容是：TIANYI-001'
        assert not plugin._streams['u1'].node.publishers[1].messages
    finally:
        plugin.stop()
    assert not executor.nodes


def test_native_tts_failure_does_not_stop_scan(monkeypatch):
    import plugins.qr_scan as module
    class Broken:
        def __init__(self, url):
            pass
        def speak(self, *args):
            raise RuntimeError('offline')
    monkeypatch.setattr(module, 'LocalTTS', Broken)
    plugin = QrScanPlugin({'auto_announce': True, 'announce_backend': 'native_mcp',
                         'tts_mcp_url': 'http://127.0.0.1:15740/mcp', 'scan_hz': 10}, _FakeExecutor())
    try:
        plugin.dispatch('qr_scan', {'action': 'start', 'input_topic': '/camera'})
        stream = plugin._streams['/camera']
        send(stream, 'U1-001')
        deadline = time.monotonic()+3
        while stream.info()['speech']['state'] != 'error' and time.monotonic()<deadline:
            time.sleep(0.01)
        assert stream.info()['speech']['state'] == 'error'
        send(stream, 'SECOND')
        assert wait(plugin, '/camera', 'detected', 2)['codes'][0]['text'] == 'SECOND'
    finally:
        plugin.stop()
