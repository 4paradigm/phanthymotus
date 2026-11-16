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
        rotation = msg.imu_local_transform.rotation
        assert (rotation.x, rotation.y, rotation.z, rotation.w) == (0., 0., 0., 0.)
        assert [f.name for f in msg.laser_scan.fields] == ['x', 'y', 'z']
        assert msg.header.stamp == msg.laser_scan.header.stamp


def test_fallback_does_not_bypass_required_odometry_loss(rgb_node):
    n = rgb_node
    from test_pointcloud_slam_stream import odom
    missing = odom(1000, status='lost'); missing['pose'] = None
    n._odom(NS(data=json.dumps(missing)))
    n._cloud(cloud())
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
        assert n.status()['rgb_preparation']['timeouts'] >= 1
        assert n.status()['rgb_preparation']['published_images'] == 0
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
        # A decode fault is asynchronous; wait for the actual invalidation.
        assert _wait_until(lambda: n._rgb.info()['decode_failures'] == 1)
        release.set()
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        assert not published(n, '/sensor_data')[0].raw_messages[-1].left.data
        assert n.status()['loop_closure_mode'] == 'pointcloud'
    finally:
        release.set()


def test_null_external_pose_reaches_existing_native_map_without_a_frozen_pose(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    image(n, 1000)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    lost = odom(1100, status='lost'); lost['pose'] = None
    n._odom(NS(data=json.dumps(lost)))
    image(n, 1100)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 2)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
    msg = published(n, '/sensor_data')[0].raw_messages[-1]
    assert msg.left.data and msg.right.data
    orientation = published(n, '/odom')[0].raw_messages[-1].pose.pose.orientation
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == (0., 0., 0., 0.)
    assert len(NativeProcess.instances) == 1
    assert n.stream.pending_count == 0


def test_no_pose_and_distinct_capture_times_does_not_invent_motion(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    lost = odom(1100); lost['pose'] = None
    n._odom(NS(data=json.dumps(lost)))
    image(n, 1110)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
    msg = published(n, '/sensor_data')[0].raw_messages[-1]
    assert not msg.left.data
    assert 'motion' in n.status()['rgb_input']['reason']


def test_pose_less_observations_are_bounded_and_stop_with_capture(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    lost = odom(1100); lost['pose'] = None
    n._odom(NS(data=json.dumps(lost)))
    for i in range(100):
        n._cloud(cloud(1_100_000_000+i))
    assert n.stream.pending_count == 0
    assert n.status()['runtime_queue'] <= n.stream.max_pending
    n.freeze_capture()
    count = n.status()['forwarded_clouds']
    n._cloud(cloud(2_000_000_000))
    assert n.status()['forwarded_clouds'] == count


def test_native_visual_tracking_status_requires_native_recovery_evidence(calibrated_node):
    n = calibrated_node
    send(n)
    assert _wait_until(lambda: n._active is not None)
    n._native_info(NS(ref_id=1, loop_closure_id=0, proximity_detection_id=1,
        stats_keys=['Motus/RecoveryState/', 'Motus/Relocalizations/', 'Motus/VisualInliers/'],
        stats_values=[3., 1., 47.]), n._active)
    assert n.status()['native_recovery'] == dict(state='visual_tracking', relocalizations=1, visual_inliers=47)
    assert n.status()['native_loop_closures'] == 0
    assert n.status()['native_proximity_detections'] == 1


def test_saved_map_can_receive_pose_less_observations_before_any_new_external_pose(tmp_path, monkeypatch):
    import sqlite3
    import plugins.pointcloud_slam_ros as ros
    class RestoredProcess(NativeProcess):
        def __init__(self, *args, restored=False, **kwargs):
            super().__init__(*args, **kwargs)
            assert restored
        def start(self):
            assert self.database_path.is_file()
            self.started = True
    monkeypatch.setattr(ros, 'RtabmapProcess', RestoredProcess)
    database = tmp_path/'restored.db'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE Node(id INTEGER PRIMARY KEY)'); db.execute('INSERT INTO Node VALUES(42)')
    n = ros.RtabmapNode('/o','/c',settings(rgb_input={'topic':'/rgb','camera_info':rgb_camera()}),
        session_dir=tmp_path,restored_map=dict(database=str(database),segment_id='saved',frame_id='saved_map'))
    n._rgb._decode_jpeg = lambda data: np.ones((4,4,3),np.uint8)
    n._receipt_clock = lambda:1.; n._measurement_clock_ns=lambda:1_000_000_000
    n.start()
    try:
        image(n,1000)
        assert _wait_until(lambda:n._rgb.info()['decoded']==1)
        n._cloud(cloud())
        assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
        assert n.status()['segment_id']=='saved'
        assert published(n,'/sensor_data')[0].raw_messages[-1].left.data
        assert not n.stream._poses
    finally:
        n.close(); n.destroy_node()


def test_odom_silence_keeps_each_pending_clouds_matching_rgb(calibrated_node):
    n=calibrated_node
    image(n,1000)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    for i,ms in enumerate(range(1100,1700,100),2):
        image(n,ms)
        assert _wait_until(lambda:n._rgb.info()['decoded']==i)
        n._cloud(cloud(ms*1_000_000))
    assert _wait_until(lambda:n.status()['forwarded_clouds']==5)
    messages=published(n,'/sensor_data')[0].raw_messages[1:]
    assert all(msg.left.data and msg.right.data for msg in messages)
    assert [msg.left.header.stamp.nanosec for msg in messages]==[100_000_000,200_000_000,300_000_000,400_000_000]
    assert len(n._rgb_reservations)<=n.stream.max_pending*3
    n.freeze_capture()
    assert n.stream.pending_count==0
    assert not n._rgb_reservations and not n._cloud_ingress


def test_late_jpeg_decode_is_reserved_for_the_waiting_cloud(calibrated_node):
    n=calibrated_node
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    for i,ms in enumerate(range(1100,1700,100),1):
        n._cloud(cloud(ms*1_000_000))
        image(n,ms)
        assert _wait_until(lambda:n._rgb.info()['decoded']==i)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==5)
    messages=published(n,'/sensor_data')[0].raw_messages[1:]
    assert all(msg.left.data and msg.right.data for msg in messages)


def test_missing_native_capability_does_not_register_an_unstarted_segment(calibrated_node, monkeypatch):
    n=calibrated_node
    def unavailable(self): raise RuntimeError('visual recovery runtime unavailable')
    monkeypatch.setattr(NativeProcess,'validate_runtime',unavailable,raising=False)
    send(n)
    assert _wait_until(lambda:n.status()['state']=='error')
    assert n._segments==[]
    with pytest.raises(RuntimeError,match='visual recovery runtime unavailable'):
        n.close()
    n._runtime_error=''  # Fixture can dispose the already stopped gateway.


def test_unlocated_native_result_clears_the_previous_tracking_diagnostic(calibrated_node):
    n=calibrated_node
    send(n)
    assert _wait_until(lambda:n._active is not None)
    keys=['Motus/RecoveryState/','Motus/Relocalizations/','Motus/VisualInliers/']
    n._native_info(NS(ref_id=1,loop_closure_id=0,proximity_detection_id=0,
        stats_keys=keys,stats_values=[3.,1.,47.]),n._active)
    n._native_info(NS(ref_id=0,stats_keys=keys,stats_values=[1.,1.,0.]),n._active)
    assert n.status()['native_recovery']['state']=='searching'
    assert n.status()['native_updates']==1


def test_pose_less_inflight_cloud_waits_for_its_pending_decode(calibrated_node, monkeypatch):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    started,release,preparing=threading.Event(),threading.Event(),threading.Event()
    def decode(data):
        started.set()
        assert release.wait(1.)
        return np.ones((4,4,3),np.uint8)
    original=n._rgb_attachment
    def attachment(*args):
        preparing.set()
        return original(*args)
    monkeypatch.setattr(n._rgb,'_decode_jpeg',decode)
    monkeypatch.setattr(n,'_rgb_attachment',attachment)
    lost=odom(1100);lost['pose']=None
    n._odom(NS(data=json.dumps(lost)))
    image(n,1100)
    assert started.wait(1.)
    n._cloud(cloud(1_100_000_000))
    try:
        assert preparing.wait(1.)
        assert not _wait_until(lambda:n.status()['forwarded_clouds']==2,timeout=.003)
    finally:
        release.set()
    assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
    message=published(n,'/sensor_data')[0].raw_messages[-1]
    assert message.left.data and message.right.data
    assert len(NativeProcess.instances)==1
