"""Optional RGB must never hold up an admitted scan or restart its map."""
import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_pointcloud_slam_ros import NativeProcess, cloud, settings, published, send
from vision_stubs import _wait_until


@pytest.fixture
def rgb_node(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    NativeProcess.instances = []
    n = ros.RtabmapNode('/o', '/c', settings(rgb_input={'topic': '/rgb', 'camera_info': {}}),
                        session_dir=tmp_path / 'session')
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    yield n
    n.close()
    n.destroy_node()


def test_bad_rgb_metadata_still_maps_every_scan_in_one_native_process(rgb_node):
    n = rgb_node
    for stamp in (1000, 1100, 1200):
        n._measurement_clock_ns = lambda s=stamp: s * 1_000_000
        send(n, stamp)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == (stamp - 1000)//100 + 1)
    envelopes = published(n, '/sensor_data')
    assert len(envelopes) == 1
    assert len(envelopes[0].raw_messages) == 3
    assert len(NativeProcess.instances) == 1
    assert n.status()['segments'] == 1
    assert n.status()['loop_closure_mode'] == 'pointcloud'
    assert n.status()['rgb_input']['state'] == 'fallback'
    for msg in envelopes[0].raw_messages:
        assert msg.laser_scan.data
        assert not msg.left.data and not msg.right.data
        assert msg.left_camera_info == [] and msg.local_transform == []
        assert msg.laser_scan_format == 5
        assert [f.name for f in msg.laser_scan.fields] == ['x', 'y', 'z']
        assert msg.header.stamp == msg.laser_scan.header.stamp


def test_fallback_does_not_bypass_required_odometry_loss(rgb_node):
    n = rgb_node
    send(n, status='lost')
    assert not _wait_until(lambda: n.status()['forwarded_clouds'], timeout=.1)
    assert not published(n, '/sensor_data')


def rgb_camera():
    return {'schema': 'motus.camera/1', 'topic': '/rgb', 'format': 'image/jpeg',
            'width': 4, 'height': 4, 'K': [1, 0, 1, 0, 1, 1, 0, 0, 1],
            'D': [0.]*5, 'distortion_model': 'pinhole',
            'vendor': {'frame_id': 'camera', 'clock_id': 'clock',
                       'registration': 'registered_rectified',
                       'timestamp_semantics': 'source_capture_mapped'}}


@pytest.fixture
def calibrated_node(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    NativeProcess.instances = []
    n = ros.RtabmapNode('/o', '/c', settings(rgb_input={'topic': '/rgb', 'camera_info': rgb_camera()}),
                        session_dir=tmp_path / 'session')
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    # Isolate the external JPEG codec; the actual decoder queue, joins, native
    # envelope construction and fallback/recovery all run normally.
    n._rgb._decode_jpeg = lambda data: np.ones((4, 4, 3), np.uint8) if data == b'good' else None
    n.start()
    yield n
    n.close()
    n.destroy_node()


def image(n, ms, data=b'good'):
    from sensor_msgs.msg import CompressedImage
    msg = CompressedImage()
    msg.data, msg.format = data, 'jpeg'
    msg.header = NS(stamp=NS(sec=ms//1000, nanosec=ms%1000*1_000_000), frame_id='camera')
    n._rgb_image(msg)
    return msg


def test_rgb_error_and_recovery_keep_forwarding_scan_in_same_map(calibrated_node):
    n = calibrated_node
    original = image(n, 1000)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    first = published(n, '/sensor_data')[0].raw_messages[-1]
    assert first.left.data and first.right.data
    assert n.status()['loop_closure_mode'] == 'image'
    assert first.left.header.stamp == original.header.stamp
    assert first.left_camera_info[0].k == rgb_camera()['K']
    assert first.left.width == first.right.width == 4
    assert first.right.encoding == '32FC1'
    assert np.frombuffer(first.right.data, '<f4').max() == 3

    n._measurement_clock_ns = lambda: 1_100_000_000
    image(n, 1100, b'bad')
    assert _wait_until(lambda: n._rgb.info()['decode_failures'] == 1)
    send(n, 1100)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
    second = published(n, '/sensor_data')[0].raw_messages[-1]
    assert second.laser_scan.data and not second.left.data
    assert n.status()['loop_closure_mode'] == 'pointcloud'

    n._measurement_clock_ns = lambda: 1_200_000_000
    image(n, 1200)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 2)
    send(n, 1200)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 3)
    assert published(n, '/sensor_data')[0].raw_messages[-1].left.data
    assert n.status()['loop_closure_mode'] == 'image'
    assert len(NativeProcess.instances) == 1
    assert n.status()['segments'] == 1


def test_stalled_rgb_decoder_does_not_block_scan_mapping(calibrated_node):
    n = calibrated_node
    entered, release = threading.Event(), threading.Event()

    def blocked(data):
        entered.set()
        assert release.wait(2)
        return np.ones((4, 4, 3), np.uint8)

    n._rgb._decode_jpeg = blocked
    try:
        image(n, 1000)
        assert entered.wait(1)
        send(n)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        assert n.status()['loop_closure_mode'] == 'pointcloud'
        assert not published(n, '/sensor_data')[0].raw_messages[-1].left.data
    finally:
        release.set()


def test_rgb_original_time_and_motion_are_compensated_without_retiming(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    n._measurement_clock_ns = lambda: 1_020_000_000
    source = image(n, 1020)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    n._odom(NS(data=json.dumps(odom(1000, x=0))))
    n._cloud(cloud(1_000_000_000))
    n._odom(NS(data=json.dumps(odom(1040, x=.4))))
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    msg = published(n, '/sensor_data')[0].raw_messages[-1]
    assert msg.left.data
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 0
    assert msg.left.header.stamp == source.header.stamp
    assert msg.left.header.stamp.nanosec == 20_000_000
    assert msg.local_transform[0].translation.x == pytest.approx(.2)
    assert source.header.frame_id == 'camera'


def test_rgb_subscription_failure_does_not_block_required_mapping(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)

    class Registry:
        clock_id = 'local-ingress'
        def acquire(self, topic, kind, declaration, callback):
            if topic == '/rgb':
                raise RuntimeError('RGB source unavailable')
            return NS(close=lambda: None)

    n = ros.RtabmapNode('/o', '/c', settings(rgb_input={'topic': '/rgb', 'camera_info': rgb_camera()}),
                        session_dir=tmp_path, input_registry=Registry())
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        assert n.status()['loop_closure_mode'] == 'pointcloud'
        assert 'unavailable' in n.status()['rgb_input']['reason']
    finally:
        n.close()
        n.destroy_node()


def test_stalled_optional_projection_does_not_block_scan_mapping(calibrated_node, monkeypatch):
    import plugins.pointcloud_slam_rgb as rgb
    n = calibrated_node
    entered, release = threading.Event(), threading.Event()
    original = rgb.project_depth

    def blocked(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)

    monkeypatch.setattr(rgb, 'project_depth', blocked)
    try:
        image(n, 1000)
        assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
        send(n)
        assert entered.wait(1)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1, timeout=.2)
        n._measurement_clock_ns = lambda: 1_100_000_000
        send(n, 1100)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2, timeout=.2)
        assert n.status()['loop_closure_mode'] == 'pointcloud'
        assert len(NativeProcess.instances) == 1
    finally:
        release.set()


def test_rgb_fault_invalidates_an_inflight_image_attachment(calibrated_node, monkeypatch):
    import plugins.pointcloud_slam_rgb as rgb
    from sensor_msgs.msg import CompressedImage
    n = calibrated_node
    entered, release = threading.Event(), threading.Event()
    project = rgb.project_depth

    def paused(*args):
        entered.set()
        assert release.wait(2)
        return project(*args)

    monkeypatch.setattr(rgb, 'project_depth', paused)
    try:
        image(n, 1000)
        assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
        send(n)
        assert entered.wait(1)
        invalid = CompressedImage()
        invalid.header = NS(stamp=NS(sec=1, nanosec=1_000_000), frame_id='wrong-frame')
        invalid.format, invalid.data = 'jpeg', b'bad'
        n._rgb_image(invalid)
        release.set()
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        assert not published(n, '/sensor_data')[0].raw_messages[-1].left.data
        assert n.status()['loop_closure_mode'] == 'pointcloud'
    finally:
        release.set()
