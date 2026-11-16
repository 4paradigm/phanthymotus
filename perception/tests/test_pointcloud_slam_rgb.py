"""Optional RGB failures must leave the scan path immediately usable."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from plugins.pointcloud_slam_rgb import RgbLoopInput, project_depth, validate_rgb_camera
from spatial_test_helpers import declaration as mount_declaration, packet

STAMP = 1_000_000_123


def camera():
    return dict(schema='motus.camera/1', topic='/rgb', format='image/jpeg', width=4, height=3,
        distortion_model='pinhole', K=[2., 0., 1., 0., 2., 1., 0., 0., 1.], D=[0.]*5,
        vendor=dict(frame_id='camera_optical', clock_id='clock',
                    timestamp_semantics='source_capture_mapped', registration='registered_rectified'))


def settings():
    return dict(cloud_frame='camera_optical', clock_id='clock', extrinsic_mode='static')


def message(stamp=STAMP, **changes):
    msg = SimpleNamespace(header=SimpleNamespace(frame_id='camera_optical',
        stamp=SimpleNamespace(sec=stamp//10**9, nanosec=stamp%10**9)), format='jpeg', data=b'jpeg')
    for key, value in changes.items(): setattr(msg, key, value)
    return msg


def helper(**kwargs):
    result = RgbLoopInput(validate_rgb_camera(camera(), topic='/rgb'), settings(), **kwargs)
    result._decode_jpeg = lambda data: np.full((3, 4, 3), data[-1], dtype=np.uint8)
    result.start()
    return result


def wait_for(check):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        if check(): return
        time.sleep(.001)
    assert check()


def test_camera_validation_is_owned_and_exact():
    source = camera(); result = validate_rgb_camera(source, topic='/rgb')
    source['K'][0] = 500
    assert result['camera_info']['K'][0] == 2
    assert result['topic'] == '/rgb'


@pytest.mark.parametrize('change', [dict(topic='/other'), dict(width=0), dict(height=4097),
    dict(width=True), dict(format='image/png'), dict(K=[1]*8), dict(D=[.01]),
    dict(K=[0.,0,1,0,2,1,0,0,1]), dict(K=[float('nan')]*9), dict(distortion_model='fisheye'),
    dict(vendor={}), dict(schema='other')])
def test_invalid_camera_is_optional_fallback(change):
    bad = camera(); bad.update(change)
    with pytest.raises(ValueError): validate_rgb_camera(bad, topic='/rgb')
    result = RgbLoopInput(dict(topic='/rgb', camera_info=bad), settings())
    result.start(); result.image(message(), now_ns=STAMP)
    assert result.take(STAMP, STAMP) is None
    assert result.info()['state'] == 'fallback'
    result.close()


def test_disconnected_and_incompatible_clocks_or_frames_never_raise():
    result = RgbLoopInput(None, settings()); result.start(); result.close()
    assert result.info()['state'] == 'disconnected'
    for change in (dict(clock_id='other'), dict(cloud_frame='other')):
        config = settings(); config.update(change)
        result = RgbLoopInput(validate_rgb_camera(camera(), topic='/rgb'), config)
        assert result.camera is None and result.info()['state'] == 'fallback'


def test_decode_runs_off_ingress_and_cloud_take_never_waits():
    result = helper(); entered = threading.Event(); release = threading.Event()
    def decode(data):
        entered.set(); release.wait(2)
        return np.zeros((3, 4, 3), dtype=np.uint8)
    result._decode_jpeg = decode
    before = time.monotonic(); result.image(message(), now_ns=STAMP)
    assert time.monotonic()-before < .1 and entered.wait(1)
    assert result.take(STAMP, STAMP) is None
    before = time.monotonic(); result.close()
    assert time.monotonic()-before < .2
    release.set()


def test_nearest_fresh_capture_is_consumed_once_and_not_retimed():
    result = helper()
    try:
        result.image(message(), now_ns=STAMP)
        wait_for(lambda: result.info()['decoded'] == 1)
        stamp, image, generation = result.take(STAMP+30_000_000, STAMP+30_000_000)
        assert stamp == STAMP and image.shape == (3, 4, 3)
        assert result.take(STAMP+31_000_000, STAMP+31_000_000) is None
        result.image(message(STAMP+100_000_000), now_ns=STAMP+100_000_000)
        wait_for(lambda: result.info()['decoded'] == 2)
        assert result.take(STAMP+200_000_001, STAMP+200_000_001) is None
    finally: result.close()


@pytest.mark.parametrize('kind', ['stale', 'future', 'repeated', 'frame', 'receipt', 'clock', 'stamp', 'oversize', 'format'])
def test_invalid_ingress_clears_ready_images_and_fresh_input_recovers(kind):
    result = helper()
    try:
        result.image(message(), now_ns=STAMP); wait_for(lambda: result.info()['decoded'] == 1)
        msg, now, context = message(STAMP+1), STAMP+1, None
        if kind == 'stale': now += 500_000_001
        if kind == 'future': now -= 50_000_001
        if kind == 'repeated': msg = message()
        if kind == 'frame': msg.header.frame_id = 'wrong'
        if kind in ('receipt', 'clock', 'stamp'):
            context = SimpleNamespace(time_basis='receipt_estimated' if kind == 'receipt' else 'source_capture',
                normalized_clock_id='wrong' if kind == 'clock' else 'clock',
                observation_stamp_ns=STAMP+2 if kind == 'stamp' else STAMP+1)
        if kind == 'oversize': msg.data = b'x'*(4*1024*1024+1)
        if kind == 'format': msg.format = 'png'
        result.image(msg, context, now_ns=now)
        assert result.take(STAMP, STAMP) is None and result.info()['state'] == 'fallback'
        result.image(message(STAMP+2), now_ns=STAMP+2)
        wait_for(lambda: result.info()['decoded'] == 2)
        assert result.take(STAMP+2, STAMP+2)[0] == STAMP+2
    finally: result.close()


def test_decode_error_and_wrong_dimensions_recover_without_reusing_old_frame():
    result = helper()
    try:
        result._decode_jpeg = lambda data: None
        result.image(message(), now_ns=STAMP)
        wait_for(lambda: result.info()['decode_failures'] == 1)
        assert result.take(STAMP, STAMP) is None
        result._decode_jpeg = lambda data: np.zeros((4, 4, 3), dtype=np.uint8)
        result.image(message(STAMP+1), now_ns=STAMP+1)
        wait_for(lambda: result.info()['decode_failures'] == 2)
        result._decode_jpeg = lambda data: np.zeros((3, 4, 3), dtype=np.uint8)
        result.image(message(STAMP+2), now_ns=STAMP+2)
        wait_for(lambda: result.info()['decoded'] == 1)
        assert result.take(STAMP+2, STAMP+2) is not None
    finally: result.close()


def test_decoded_cache_is_bounded_and_expired_at_take():
    result = helper(max_pending=2)
    try:
        for i in range(4):
            result.image(message(STAMP+i), now_ns=STAMP+i)
            wait_for(lambda: result.info()['decoded'] == i+1)
        assert result.info()['buffered_frames'] == 2
        assert result.take(STAMP+600_000_000, STAMP+600_000_000) is None
        assert result.info()['buffered_frames'] == 0
    finally: result.close()


def test_dynamic_mount_requires_exact_fresh_evidence_and_common_reference():
    c, config = camera(), settings()
    reference = dict(clock_id='clock', domain_id='host', timestamp_basis='host_realtime')
    c['vendor'].update(clock_id='camera-clock', body_mount=mount_declaration(), clock_reference=reference)
    config.update(extrinsic_mode='dynamic', capture_mount=dict(declaration=mount_declaration(),
        source_clock='camera-clock', clock_reference=deepcopy(reference)))
    result = RgbLoopInput(validate_rgb_camera(c, topic='/rgb'), config)
    assert result.camera is not None and result.mounts is not None
    result.mount(packet(STAMP), STAMP)
    assert np.array_equal(result.capture_mount(STAMP, None, STAMP), np.eye(4))
    assert result.capture_mount(STAMP+1, np.eye(4), STAMP+1) is None
    assert result.capture_mount(STAMP, None, STAMP+600_000_000) is None
    result.mount(packet(STAMP+1, status='unavailable', matrix=None), STAMP+1)
    assert result.capture_mount(STAMP+1, np.eye(4), STAMP+1) is None
    bad = deepcopy(config); bad['capture_mount']['clock_reference']['domain_id'] = 'other'
    assert RgbLoopInput(validate_rgb_camera(c, topic='/rgb'), bad).camera is None


def test_static_mount_uses_only_valid_supplied_registered_transform():
    result = helper()
    try:
        matrix = np.eye(4); matrix[0,3] = .2
        assert np.array_equal(result.capture_mount(STAMP, matrix, STAMP), matrix)
        assert result.capture_mount(STAMP, None, STAMP) is None
    finally: result.close()


def test_projection_keeps_nearest_measured_z_and_zero_holes():
    points = np.array([[0,0,2], [0,0,1], [1,0,1], [0,0,-1], [np.nan,0,1], [99,0,1]])
    depth = project_depth(points, np.eye(4), np.eye(3), 3, 2)
    assert depth.dtype == np.float32
    assert np.array_equal(depth, [[1,1,0], [0,0,0]])
    moved = np.eye(4); moved[0,3] = 1
    assert project_depth([[0,0,1]], moved, np.eye(3), 3, 2)[0,1] == 1


def test_raw_queue_retains_only_latest_and_close_discards_inflight_decode():
    result = helper(max_pending=2); entered = threading.Event(); release = threading.Event()
    def decode(data):
        if data == b'first': entered.set(); release.wait(2)
        return np.full((3,4,3), data[-1], dtype=np.uint8)
    result._decode_jpeg = decode
    try:
        result.image(message(data=b'first'), now_ns=STAMP); assert entered.wait(1)
        for i in range(1, 5): result.image(message(STAMP+i, data=bytes([i])), now_ns=STAMP+i)
        assert result.info()['pending_raw'] == 1 and result.info()['dropped'] == 3
        release.set(); wait_for(lambda: result.info()['decoded'] == 2)
        stamp, image, generation = result.take(STAMP+4, STAMP+4)
        assert stamp == STAMP+4 and image[0,0,0] == 4
    finally: release.set(); result.close()


def test_bad_ingress_invalidates_inflight_decode():
    result = helper(); entered = threading.Event(); release = threading.Event()
    def decode(data):
        entered.set(); release.wait(2)
        return np.zeros((3,4,3), dtype=np.uint8)
    result._decode_jpeg = decode
    try:
        result.image(message(), now_ns=STAMP); assert entered.wait(1)
        result.image(message(STAMP+1, format='png'), now_ns=STAMP+1)
        release.set()
        # New input runs after the old decoder returns; old result must not survive.
        result.image(message(STAMP+2), now_ns=STAMP+2)
        wait_for(lambda: result.info()['decoded'] == 1)
        assert result.take(STAMP+2, STAMP+2)[0] == STAMP+2
        assert result.info()['buffered_frames'] == 0
    finally: release.set(); result.close()


def test_real_jpeg_decode_and_declared_size_guard():
    import cv2
    if getattr(cv2, '__motus_fake__', False): pytest.skip('real JPEG decoder unavailable')
    ok, encoded = cv2.imencode('.jpg', np.full((3,4,3), 32, dtype=np.uint8))
    assert ok
    result = RgbLoopInput(validate_rgb_camera(camera(), topic='/rgb'), settings()); result.start()
    try:
        result.image(message(data=encoded.tobytes()), now_ns=STAMP)
        wait_for(lambda: result.info()['decoded'] == 1)
        assert result.take(STAMP, STAMP)[1].shape == (3,4,3)
        ok, wrong = cv2.imencode('.jpg', np.zeros((30,40,3), np.uint8)); assert ok
        result.image(message(STAMP+1, data=wrong.tobytes()), now_ns=STAMP+1)
        wait_for(lambda: result.info()['decode_failures'] == 1)
        assert result.take(STAMP+1, STAMP+1) is None
    finally: result.close()


def test_predecode_rejects_oversized_jpeg_before_allocating(monkeypatch):
    import cv2
    result = RgbLoopInput(validate_rgb_camera(camera(), topic='/rgb'), settings())
    monkeypatch.setattr(cv2, 'imdecode', lambda *args: pytest.fail('oversized JPEG reached decoder'))
    # SOF0 claims 65535x65535, despite the tiny compressed payload.
    bomb = b'\xff\xd8\xff\xc0\x00\x11\x08\xff\xff\xff\xff'+b'\x00'*10
    with pytest.raises(ValueError): result._decode_jpeg(bomb)
