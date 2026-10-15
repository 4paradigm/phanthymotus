"""Real JPEG decoding plus multi-instance perception stream contracts."""
import json
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from vision_stubs import _FakeExecutor, _wait_until
from plugins.qr_scan import QrScanPlugin
from utils.qr_decoder import QrDecoder, QrTracker, self_check


def jpeg(text=None):
    image = np.full((290, 290), 255, dtype=np.uint8) if text is None else cv2.resize(
        cv2.QRCodeEncoder_create().encode(text), (290, 290), interpolation=cv2.INTER_NEAREST)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    return data.tobytes()


@pytest.mark.parametrize('text', ['ITEM-A', '展品编号：001', 'https://example.com/001'])
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


def test_crop_retry_keeps_original_coordinates_and_recovers_partial_multi():
    image = np.full((700, 1000, 3), 255, np.uint8)
    qr = cv2.imdecode(np.frombuffer(jpeg('CROP-RETRY'), np.uint8), cv2.IMREAD_COLOR)
    image[180:470, 300:590] = qr
    real = cv2.QRCodeDetector()
    found, corners = real.detect(image)
    assert found
    target = corners.reshape(4, 2)

    class PartialDetector:
        def detectAndDecodeMulti(self, frame):
            # Preserve an already decoded neighbouring code and recover only
            # this missing payload through real OpenCV on the cropped image.
            neighbour = target + [500, 0]
            return True, ['ALREADY-DECODED', ''], np.array([neighbour, target]), ()

        def detectAndDecode(self, frame):
            return real.detectAndDecode(frame)

    # Exercise OpenCV crop recovery independently of the earlier ZXing path.
    decoder = QrDecoder(use_zxing=False)
    decoder.detector = PartialDetector()
    decoder._retry_rectified = lambda *args: ('', 0)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    result = decoder.decode(data.tobytes())
    assert [c['text'] for c in result] == ['ALREADY-DECODED', 'CROP-RETRY']
    assert np.allclose(result[1]['corners_px'], target)
    assert result[1]['image_width'] == 1000 and result[1]['image_height'] == 700
    assert decoder.undecoded == []


def test_unreadable_candidates_bound_retries_and_reject_neighbour_payload():
    target = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)

    class UnreadableDetector:
        calls = 0

        def detectAndDecodeMulti(self, image):
            return False, [''] * 10, np.array([target] * 10), ()

        def detectAndDecode(self, image):
            self.calls += 1
            assert max(image.shape[:2]) <= 1600
            # A different code in the crop must not be attributed to target.
            return 'NEIGHBOUR', np.array([[[0, 0], [8, 0], [8, 8], [0, 8]]]), None

    decoder = QrDecoder()
    detector = decoder.detector = UnreadableDetector()
    assert decoder.decode(jpeg()) == []
    assert detector.calls <= 8
    assert len(decoder.undecoded) == 10
    # Diagnostic state must not leak into a subsequent blank frame.
    decoder.detector = cv2.QRCodeDetector()
    assert decoder.decode(jpeg()) == [] and decoder.undecoded == []


def test_stream_distinguishes_unreadable_from_blank_without_new_event():
    from plugins.qr_scan import _QrStream
    unresolved = [{'corners_px': [[1, 1], [9, 1], [9, 9], [1, 9]],
                   'image_width': 10, 'image_height': 10}]
    decoder = SimpleNamespace(decode=lambda data: [], undecoded=unresolved)
    stream = _QrStream(None, '/camera', '/result', {'auto_announce': True}, decoder)
    frame = {'data': b'test', 'received_monotonic': time.monotonic(),
             'received_at': 'test', 'frame_sequence': 1}
    result = stream.process_frame(frame)
    assert result['status'] == 'detected_unreadable'
    assert result['undecoded_codes'] == unresolved
    assert result['codes'] == result['new_events'] == []
    assert stream._announcement(result, time.monotonic()) is None
    decoder.undecoded = []
    assert stream.process_frame(frame)['status'] == 'no_qr'
    frame['received_monotonic'] -= stream.stale_s + 1
    stale = stream.process_frame(frame)
    assert stale['status'] == 'stale' and stale['undecoded_codes'] == []


def test_single_full_frame_fallback_does_not_require_crop_retry():
    points = np.array([[[40, 40], [200, 40], [200, 200], [40, 200]]], np.float32)

    class SingleDetector:
        calls = 0

        def detectAndDecodeMulti(self, image):
            return False, [''], points, ()

        def detectAndDecode(self, image):
            self.calls += 1
            return 'SINGLE-PATH', points, None

    decoder = QrDecoder()
    detector = decoder.detector = SingleDetector()
    assert decoder.decode(jpeg())[0]['text'] == 'SINGLE-PATH'
    assert detector.calls == 1 and decoder.undecoded == []


class _NoQrDetector:
    def detectAndDecodeMulti(self, image):
        return False, (), None, ()

    def detectAndDecode(self, image):
        return '', None, None


def test_zxing_real_fallback_when_opencv_misses_qr():
    pytest.importorskip('zxingcpp')
    decoder = QrDecoder()
    decoder.detector = _NoQrDetector()
    codes = decoder.decode(jpeg('ZXING-FALLBACK'))
    assert len(codes) == 1 and codes[0]['text'] == 'ZXING-FALLBACK'
    assert codes[0]['image_width'] == codes[0]['image_height'] == 290
    assert len(codes[0]['corners_px']) == 4
    assert decoder.undecoded == []


def test_zxing_recovers_unreadable_before_crop_retries():
    pytest.importorskip('zxingcpp')
    data = jpeg('RECOVER-BEFORE-CROP')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    found, points = cv2.QRCodeDetector().detect(image)
    assert found

    class UnreadableDetector(_NoQrDetector):
        def detectAndDecodeMulti(self, image):
            return True, [''], points, ()

    def unexpected(*args):
        pytest.fail('recovered candidate must not need OpenCV crop retries')

    decoder = QrDecoder()
    decoder.detector = UnreadableDetector()
    decoder._retry_crop = unexpected
    codes = decoder.decode(data)
    assert len(codes) == 1 and codes[0]['text'] == 'RECOVER-BEFORE-CROP'
    assert decoder.undecoded == []


def _zxing_result(text, corners, valid=True):
    return SimpleNamespace(valid=valid, text=text, position=SimpleNamespace(**{
        name: SimpleNamespace(x=x, y=y) for name, (x, y) in zip(
            ('top_left', 'top_right', 'bottom_right', 'bottom_left'), corners)}))


def test_zxing_partial_recovery_merges_positions_not_payloads():
    a = [[10, 10], [60, 10], [60, 60], [10, 60]]
    b = [[150, 150], [230, 150], [230, 230], [150, 230]]
    unreadable = [[80, 80], [120, 80], [120, 120], [80, 120]]
    calls = []

    class PartialDetector:
        def detectAndDecodeMulti(self, image):
            return True, ['SAME', '', ''], np.array([a, b, unreadable]), ()

    def read(image, **kwargs):
        calls.append(kwargs)
        return [_zxing_result('SAME', a), _zxing_result('SAME', b),
                _zxing_result('BAD-CHECKSUM', unreadable, valid=False)]

    decoder = QrDecoder()
    decoder.detector = PartialDetector()
    decoder._retry_rectified = lambda *args: ('', 0)
    decoder._retry_crop = lambda *args: ('', 0)
    decoder._zxing = SimpleNamespace(read_barcodes=read,
                                    BarcodeFormat=SimpleNamespace(QRCode='QR-only'))
    codes = decoder.decode(jpeg())
    assert [c['text'] for c in codes] == ['SAME', 'SAME']
    assert codes[0]['corners_px'] == a and codes[1]['corners_px'] == b
    assert len(decoder.undecoded) == 1
    assert decoder.undecoded[0]['corners_px'] == unreadable
    assert calls == [{'formats': 'QR-only', 'try_rotate': True, 'try_downscale': True}]
    tracker = QrTracker(3)
    assert len(tracker.update(codes, 0)) == 1
    assert tracker.update(codes, 1) == []


def test_zxing_failure_retains_opencv_success_and_unreadable_state(caplog):
    points = np.array([[[20, 20], [70, 20], [70, 70], [20, 70]],
                       [[150, 150], [230, 150], [230, 230], [150, 230]]])

    class PartialDetector:
        def detectAndDecodeMulti(self, image):
            return True, ['OPENCV-OK', ''], points, ()

    def fail(*args, **kwargs):
        raise RuntimeError('do not leak backend error or payload')

    decoder = QrDecoder()
    decoder.detector = PartialDetector()
    decoder._retry_crop = lambda *args: ('', 0)
    decoder._zxing = SimpleNamespace(read_barcodes=fail,
                                    BarcodeFormat=SimpleNamespace(QRCode='QR-only'))
    for _ in range(2):
        assert [c['text'] for c in decoder.decode(jpeg())] == ['OPENCV-OK']
        assert len(decoder.undecoded) == 1
    assert caplog.text.count('QR ZXing fallback failed') == 1
    assert 'do not leak' not in caplog.text


def test_zxing_missing_dependency_falls_back_to_opencv(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'zxingcpp', None)
    decoder = QrDecoder()
    decoder.detector = _NoQrDetector()
    assert decoder.decode(jpeg()) == []
    assert decoder._zxing_unavailable
    # Missing ZXing does not affect the normal OpenCV path.
    decoder.detector = cv2.QRCodeDetector()
    assert decoder.decode(jpeg('OPENCV-ONLY'))[0]['text'] == 'OPENCV-ONLY'


def test_zxing_not_called_when_opencv_succeeds():
    def unexpected(*args, **kwargs):
        pytest.fail('ZXing must only run on the fallback path')
    decoder = QrDecoder()
    decoder._zxing = SimpleNamespace(read_barcodes=unexpected)
    assert decoder.decode(jpeg('DIRECT'))[0]['text'] == 'DIRECT'


@pytest.mark.parametrize('use_zxing', [True, False])
def test_rectified_real_decode_keeps_original_frame_coordinates(use_zxing):
    if use_zxing:
        pytest.importorskip('zxingcpp')
    qr = cv2.imdecode(np.frombuffer(jpeg('RECTIFIED-001'), np.uint8), cv2.IMREAD_COLOR)
    _, qr_corners = cv2.QRCodeDetector().detect(qr)
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [289, 0], [289, 289], [0, 289]]),
        np.float32([[250, 110], [650, 180], [590, 610], [180, 480]]))
    image = cv2.warpPerspective(qr, transform, (1000, 720), borderValue=(255, 255, 255))
    corners = cv2.perspectiveTransform(qr_corners, transform)

    class UnreadableDetector(_NoQrDetector):
        def detectAndDecodeMulti(self, image):
            return True, [''], corners, ()

        def detectAndDecode(self, frame):
            if frame.shape[:2] == (720, 1000):
                return '', None, None
            return cv2.QRCodeDetector().detectAndDecode(frame)

    decoder = QrDecoder(use_zxing=use_zxing)
    decoder.detector = UnreadableDetector()
    real_zxing = decoder._decode_zxing
    # Force the initial full-frame attempt to fail, then use the real reader
    # on the perspective-corrected image instead of a fabricated payload.
    decoder._decode_zxing = lambda frame: ([] if frame.shape[:2] == (720, 1000)
                                           else real_zxing(frame))
    decoder._retry_crop = lambda *args: pytest.fail('rectified QR must decode first')
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    result = decoder.decode(data.tobytes())
    assert len(result) == 1 and result[0]['text'] == 'RECTIFIED-001'
    assert np.allclose(result[0]['corners_px'], corners.reshape(4, 2))
    assert result[0]['image_width'] == 1000 and result[0]['image_height'] == 720
    assert decoder.undecoded == []


def test_rectified_and_crop_retries_share_frame_budget():
    target = np.float32([[50, 50], [230, 50], [230, 230], [50, 230]])
    calls = []

    class UnreadableDetector(_NoQrDetector):
        def detectAndDecodeMulti(self, image):
            return False, [''] * 10, np.array([target] * 10), ()

        def detectAndDecode(self, image):
            calls.append(('crop', image.shape[:2]))
            return '', None, None

    decoder = QrDecoder()
    decoder.detector = UnreadableDetector()

    def no_decode(frame):
        calls.append(('zxing', frame.shape[:2]))
        return []

    decoder._decode_zxing = no_decode
    assert decoder.decode(jpeg()) == []
    assert calls[0] == ('zxing', (290, 290))  # Full-frame fallback is separate.
    assert len(calls[1:]) == 8
    assert sum(kind == 'zxing' for kind, _ in calls[1:]) == 4
    assert all(shape == (794, 794) for kind, shape in calls[1:] if kind == 'zxing')
    assert len(decoder.undecoded) == 10


def test_rectified_retry_rejects_neighbour_outside_candidate():
    decoder = QrDecoder()
    calls = []

    def neighbour(frame):
        calls.append(frame.shape)
        # A detection entirely in the padding maps outside the target QR.
        return [{'text': 'NEIGHBOUR', 'corners_px': [[0, 0], [10, 0], [10, 10], [0, 10]]}]

    decoder._decode_zxing = neighbour
    frame = np.full((300, 300, 3), 255, np.uint8)
    corners = np.float32([[40, 40], [250, 40], [250, 250], [40, 250]])
    assert decoder._retry_rectified(frame, corners, 2) == ('', 2)
    assert len(calls) == 2


@pytest.mark.parametrize('corners', [
    [[1, 1]] * 4,
    [[0, 0], [30, 30], [0, 30], [30, 0]],
    [[float('nan'), 0], [30, 0], [30, 30], [0, 30]],
])
def test_rectified_retry_skips_invalid_quadrangles(corners):
    frame = np.full((40, 40, 3), 255, np.uint8)
    decoder = QrDecoder()
    assert decoder._retry_rectified(frame, np.asarray(corners), 8) == ('', 0)


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
    plugin = QrScanPlugin({'scan_hz': 10, 'stale_after_s': 30}, executor)
    try:
        for key, topic in [('source_a', '/camera_a/image'), ('source_b', '/camera_b/image')]:
            predicted = plugin.dispatch('qr_scan', {'action': 'info', 'instance_id': key, 'input_topic': topic})
            started = plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': key, 'input_topic': topic})
            assert started['state'] == 'running'
            assert predicted['topic_out'] == started['topic_out']
            assert all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) for part in started['topic_out'][0]['topic'].split('/')[1:])
            assert plugin._streams[key].node.subscriptions[0].topic == topic
        assert len(executor.nodes) == 2
        assert plugin._streams['source_a'].output_topic != plugin._streams['source_b'].output_topic
        send(plugin._streams['source_a'], 'ITEM-A')
        send(plugin._streams['source_b'], 'ITEM-B')
        a = wait(plugin, 'source_a', 'detected', 1)
        b = wait(plugin, 'source_b', 'detected', 1)
        assert a['codes'][0]['text'] == 'ITEM-A'
        assert b['codes'][0]['text'] == 'ITEM-B'
        assert a['new_events'][0]['sequence'] == b['new_events'][0]['sequence'] == 1
        # Check actual JSON published by the fake endpoint, not just read().
        messages = plugin._streams['source_a'].node.publishers[0].messages
        assert any(json.loads(m)['status'] == 'detected' for m in messages)
        send(plugin._streams['source_a'], 'ITEM-A')
        assert wait(plugin, 'source_a', 'detected', 2)['new_events'] == []
        # Expire the cached frame explicitly: decoding speed must not determine
        # whether a multi-instance test sees detected before stale (ARM/QEMU).
        stream = plugin._streams['source_a']
        with stream._lock:
            stream._frame['received_monotonic'] -= stream.stale_s + 1
        assert wait(plugin, 'source_a', 'stale')['codes'] == []
        old_node = plugin._streams['source_a'].node
        plugin.dispatch('qr_scan', {'action': 'stop', 'instance_id': 'source_a'})
        assert old_node.destroyed and len(executor.nodes) == 1
        assert plugin.dispatch('qr_scan', {'action': 'read', 'instance_id': 'source_a'})['state'] == 'idle'
        plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': 'source_a', 'input_topic': '/camera_a/image'})
        assert wait(plugin, 'source_a', 'waiting_for_frame')['session_id'] != a['session_id']
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


@pytest.mark.parametrize('nested', [False, True])
def test_canvas_config_enables_and_disables_announcement(nested):
    plugin = QrScanPlugin({}, _FakeExecutor())
    target = {'instance_id': 'configured', 'input_topic': '/camera/image'}
    try:
        for enabled in (True, False):
            cfg = {'auto_announce': enabled, 'announce_cooldown_s': 0,
                   'scan_hz': 10, 'stale_after_s': 30}
            args = {'config': cfg} if nested else cfg
            result = plugin.dispatch('qr_scan', {**target, 'action': 'config', **args})
            assert result['config']['auto_announce'] is enabled
            assert 'action' not in result['config']
            started = plugin.dispatch('qr_scan', {**target, 'action': 'start'})
            assert started['auto_announce'] is enabled
            stream = plugin._streams['configured']
            send(stream, 'CONFIG-QR')
            wait(plugin, 'configured', 'detected', 1)
            announcements = stream.node.publishers[1].messages
            if enabled:
                assert _wait_until(lambda: bool(announcements))
            assert bool(announcements) is enabled
            if enabled:
                assert json.loads(announcements[0])['text'] == '扫码成功，内容是：CONFIG-QR'
        # Instance options must not become defaults for other cameras.
        other = plugin.dispatch('qr_scan', {'action': 'start', 'instance_id': 'other',
                                          'input_topic': '/other/image'})
        assert other['auto_announce'] is False
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
        assert plugin.dispatch('qr_scan', {'action': 'info', 'input_topic': '/camera_a/image'})['state'] == 'idle'
    finally:
        for p in plugin._plugins:
            stop = getattr(p, 'stop', None)
            if callable(stop):
                stop()


def test_integrated_announcement_once_and_cooldown():
    from plugins.qr_scan import _QrStream
    stream = _QrStream(None, '/camera', '/test/result', {'auto_announce': True})
    result = {'status': 'detected', 'new_events': [{'sequence': 1, 'text': 'ITEM-A'}]}
    assert stream._announcement(result, 0)['text'] == '扫码成功，内容是：ITEM-A'
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
    # Exercise event publication independently of decoding speed on ARM/QEMU.
    plugin = QrScanPlugin({'auto_announce': True, 'scan_hz': 10, 'stale_after_s': 30}, executor)
    try:
        started = plugin.dispatch('qr_scan', {'action': 'start', 'input_topic': '/camera'})
        stream = plugin._streams['/camera']
        send(stream, 'ITEM-A')
        wait(plugin, '/camera', 'detected', 1)
        until = time.monotonic()+2
        publisher = stream.node.publishers[1]
        while not publisher.messages and time.monotonic()<until:
            time.sleep(0.01)
        assert json.loads(publisher.messages[0])['text'] == '扫码成功，内容是：ITEM-A'
        send(stream, 'ITEM-A')
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


def test_announcement_disabled_instance_does_not_affect_other_input():
    from plugins.qr_scan import _QrStream
    source_a = _QrStream(None, '/camera_a/image', '/a/result', {'auto_announce': True})
    source_b = _QrStream(None, '/camera_b/image', '/b/result', {'auto_announce': False})
    event = {'status': 'detected', 'new_events': [{'sequence': 1, 'text': 'ITEM-001'}]}
    assert source_a._announcement(event, 0) is not None
    assert source_b._announcement(event, 0) is None
