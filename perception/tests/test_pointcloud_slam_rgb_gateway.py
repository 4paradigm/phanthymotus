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
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1, timeout=.6)
        assert n.status()['rgb_preparation']['first_map_budget_ms'] == 250
        assert 'exceeded 250 ms' in n.status()['rgb_input']['reason']
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
    assert not _wait_until(lambda: n.status()['forwarded_clouds'] == 2, timeout=.05)
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


def test_native_relocalization_match_is_separate_from_local_visual_inliers(calibrated_node):
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n._active is not None)
    n._native_info(NS(ref_id=8,loop_closure_id=0,proximity_detection_id=0,
        stats_keys=['Motus/RecoveryState/','Motus/Relocalizations/','Motus/VisualInliers/',
                    'Motus/RecoveryNode/','Motus/RecoveryAnchor/','Motus/RecoveryInliers/'],
        stats_values=[2.,1.,141.,8.,2.,43.]),n._active)
    status=n.status()['native_recovery']
    assert status['relocalization_node']==8 and status['relocalization_anchor']==2
    assert status['relocalization_inliers']==43 and status['visual_inliers']==141
    assert n.status()['native_loop_closures']==0


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


def _local_motion(node):
    from plugins.pointcloud_slam_motion import MotionTimeline
    motion=NS(timeline=MotionTimeline(12),submit=lambda *args:None,close=lambda:None,
              status=lambda:{'state':'estimating'})
    node._motion=motion
    return motion.timeline


def _motion_result(timeline,stamp,x):
    from test_pointcloud_slam_motion import result
    pose,info=result(stamp,x)
    timeline.observe('pose',pose);timeline.observe('info',info)


def test_pose_less_async_projection_uses_measured_local_motion_without_global_pose(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    timeline=_local_motion(n)
    _motion_result(timeline,1_000_000_000,100.)
    _motion_result(timeline,1_200_000_000,100.2)
    lost=odom(1100);lost['pose']=None
    n._odom(NS(data=json.dumps(lost)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
    msg=published(n,'/sensor_data')[0].raw_messages[-1]
    assert msg.left.data and msg.right.data
    assert msg.left.header.stamp.nanosec==110_000_000
    assert msg.laser_scan.header.stamp.nanosec==100_000_000
    assert msg.local_transform[0].translation.x==pytest.approx(.01)
    assert published(n,'/odom')[0].raw_messages[-1].pose.pose.orientation.w==0.
    assert len(NativeProcess.instances)==1


def test_async_lost_observation_waits_boundedly_for_actual_motion_result(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    timeline=_local_motion(n)
    lost=odom(1100);lost['pose']=None
    n._odom(NS(data=json.dumps(lost)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    n._cloud(cloud(1_100_000_000))
    assert not _wait_until(lambda:n.status()['forwarded_clouds']>1,timeout=.05)
    assert len(n._motion_waiting)==1
    _motion_result(timeline,1_000_000_000,0.)
    _motion_result(timeline,1_200_000_000,.2)
    n._wake.set()
    assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
    assert published(n,'/sensor_data')[0].raw_messages[-1].left.data
    n.freeze_capture()
    assert not n._motion_waiting and not n._rgb_reservations


def test_waiting_async_observation_retains_its_numeric_rgb_mount(calibrated_node, monkeypatch):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    timeline=_local_motion(n)
    mount=np.eye(4);mount[0,3]=.02
    monkeypatch.setattr(n._rgb,'capture_mount',lambda *args:mount.copy())
    lost=odom(1100);lost['pose']=None;n._odom(NS(data=json.dumps(lost)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda:len(n._motion_waiting)==1)
    mount[0,3]=.4
    _motion_result(timeline,1_000_000_000,0.)
    _motion_result(timeline,1_200_000_000,.2)
    n._wake.set()
    assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
    msg=published(n,'/sensor_data')[0].raw_messages[-1]
    assert msg.local_transform[0].translation.x==pytest.approx(.03)


def test_motion_waits_and_mount_snapshots_are_bounded_and_cleared_on_stop(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    _local_motion(n)
    lost=odom(1100);lost['pose']=None;n._odom(NS(data=json.dumps(lost)))
    for index,ms in enumerate(range(1100,2100,100),1):
        image(n,ms+10)
        assert _wait_until(lambda:n._rgb.info()['decoded']==index)
        n._cloud(cloud(ms*1_000_000))
    assert _wait_until(lambda:len(n._motion_waiting)>0)
    assert len(n._motion_waiting)<=n.stream.max_pending
    assert len(n._motion_mounts)<=n.stream.max_pending*3
    n.freeze_capture()
    assert not n._motion_waiting and not n._motion_mounts and not n._rgb_reservations


def test_cloud_reaches_local_motion_before_an_external_pose_exists(calibrated_node):
    n=calibrated_node
    observed=[]
    _local_motion(n)
    n._motion.submit=lambda stamp,points,matrix:observed.append((stamp,points,matrix))
    n._cloud(cloud(1_100_000_003))
    assert len(observed)==1 and observed[0][0]==1_100_000_003
    np.testing.assert_array_equal(observed[0][1],[[1.,2.,3.]])
    np.testing.assert_array_equal(observed[0][2],np.eye(4))
    assert not published(n,'/sensor_data')


def test_valid_external_pose_keeps_scan_mapping_when_async_motion_is_unavailable(calibrated_node):
    n=calibrated_node
    _local_motion(n)
    image(n,1010)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    msg=published(n,'/sensor_data')[0].raw_messages[-1]
    assert msg.laser_scan.data and not msg.left.data
    assert not n._motion_waiting
    assert published(n,'/odom')[0].raw_messages[-1].pose.pose.orientation.w==1.


def test_external_recovery_discards_waiting_unlocated_frames(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    timeline=_local_motion(n)
    missing=odom(1100);missing['pose']=None
    n._odom(NS(data=json.dumps(missing)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda:len(n._motion_waiting)==1)
    send(n,1200)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
    _motion_result(timeline,1_000_000_000,0.)
    _motion_result(timeline,1_200_000_000,.2)
    n._wake.set()
    assert not _wait_until(lambda:n.status()['forwarded_clouds']>2,timeout=.1)
    assert not n._motion_waiting and not n._motion_mounts


def test_lost_pose_retains_a_slow_projection_without_extending_the_wait_budget(calibrated_node,monkeypatch):
    from test_pointcloud_slam_stream import odom
    import plugins.pointcloud_slam_rgb as rgb
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    missing=odom(1100);missing['pose']=None
    n._odom(NS(data=json.dumps(missing)))
    image(n,1100)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    entered,release=threading.Event(),threading.Event()
    original=rgb.project_depth
    def delayed(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(rgb,'project_depth',delayed)
    try:
        n._cloud(cloud(1_100_000_000))
        assert entered.wait(1)
        assert _wait_until(lambda:n.status()['rgb_preparation']['timeouts']>0)
        assert n.status()['forwarded_clouds']==1
        release.set();n._wake.set()
        assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
        msg=published(n,'/sensor_data')[0].raw_messages[-1]
        assert msg.left.data and msg.right.data
        assert n.status()['rgb_preparation']['budget_ms']==20
    finally:release.set()


def test_map_has_an_internal_archive_id_without_a_producer_segment_label(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node
    sample=odom(1000)
    sample['vendor']['motus_slam'].pop('segment_id')
    n._odom(NS(data=json.dumps(sample)))
    n._cloud(cloud(1_000_000_000))
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    assert n._active['segment_id']=='segment_0'
    n.close()
    assert n.artifacts()['segments'][0]['segment_id']=='segment_0'


@pytest.mark.parametrize('completion_during_publish',[False,True])
def test_ready_projection_is_not_discarded_by_an_older_unbracketed_observation(calibrated_node,monkeypatch,completion_during_publish):
    from test_pointcloud_slam_stream import odom
    import plugins.pointcloud_slam_rgb as rgb
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    timeline=_local_motion(n)
    _motion_result(timeline,1_200_000_000,0.)
    _motion_result(timeline,1_400_000_000,.2)
    missing=odom(1100);missing['pose']=None
    n._odom(NS(data=json.dumps(missing)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda:len(n._motion_waiting)==1)
    entered,release=threading.Event(),threading.Event()
    original=rgb.project_depth
    def delayed(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(rgb,'project_depth',delayed)
    race_enabled,raced=threading.Event(),threading.Event()
    publish=n._publish_pair
    def finish_before_preparation(segment,item,motion_generation=None):
        if completion_during_publish and race_enabled.is_set() and item[0]==1_100_000_000 and not raced.is_set():
            raced.set();release.set()
            assert _wait_until(lambda:n._rgb_task is not None and n._rgb_task[1].done())
        return publish(segment,item,motion_generation)
    monkeypatch.setattr(n,'_publish_pair',finish_before_preparation)
    try:
        image(n,1310)
        assert _wait_until(lambda:n._rgb.info()['decoded']==2)
        n._cloud(cloud(1_300_000_000))
        assert entered.wait(1)
        assert _wait_until(lambda:len(n._motion_waiting)==2)
        if completion_during_publish:
            race_enabled.set();n._wake.set()
            assert _wait_until(raced.is_set)
        else:release.set()
        assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
        message=published(n,'/sensor_data')[0].raw_messages[-1]
        assert message.header.stamp.nanosec==300_000_000
        assert message.left.data and message.right.data
        assert message.left.header.stamp.nanosec==310_000_000
        assert n.status()['rgb_preparation']['budget_ms']==20
        n.close()
        assert n._rgb_task is None
    finally:release.set()


def test_external_recovery_prevents_a_cancelled_projection_restoring_old_rgb(calibrated_node,monkeypatch):
    from test_pointcloud_slam_stream import odom
    from plugins.pointcloud_slam_motion import MotionPending
    n=calibrated_node;send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    timeline=_local_motion(n)
    entered,release=threading.Event(),threading.Event()
    def missing_motion(*args):
        entered.set()
        assert release.wait(2)
        raise MotionPending('no measured bracket')
    monkeypatch.setattr(timeline,'relative',missing_motion)
    missing=odom(1100);missing['pose']=None
    n._odom(NS(data=json.dumps(missing)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    try:
        n._cloud(cloud(1_100_000_000))
        assert entered.wait(1)
        future=n._rgb_task[1]
        assert _wait_until(lambda:len(n._motion_waiting)==1)
        send(n,1200)
        assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
        assert not n._motion_waiting and 1_100_000_000 not in n._rgb_reservations
        release.set()
        assert _wait_until(future.done)
        assert 1_100_000_000 not in n._rgb_reservations
        assert 1_100_000_000 not in n._motion_mounts
        assert len(NativeProcess.instances)==1
    finally:release.set()


def test_newer_null_observation_retires_an_overtaken_waiter_but_accepts_new_smaller_stamp(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n=calibrated_node;timeline=_local_motion(n)
    send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    missing=odom(1100);missing['pose']=None;n._odom(NS(data=json.dumps(missing)))
    image(n,1110)
    assert _wait_until(lambda:n._rgb.info()['decoded']==1)
    n._cloud(cloud(1_100_000_000))
    assert _wait_until(lambda:len(n._motion_waiting)==1)
    image(n,1200)
    assert _wait_until(lambda:n._rgb.info()['decoded']==2)
    n._cloud(cloud(1_200_000_000))
    assert _wait_until(lambda:n.status()['forwarded_clouds']==2)
    _motion_result(timeline,1_000_000_000,0.)
    _motion_result(timeline,1_200_000_000,.2)
    n._wake.set()
    assert not _wait_until(lambda:n.status()['forwarded_clouds']>2,timeout=.15)
    assert _wait_until(lambda:not n._motion_waiting)
    assert 1_100_000_000 not in n._rgb_reservations
    assert 1_100_000_000 not in n._motion_mounts
    assert n.status()['dropped_runtime_queue']>=1
    image(n,1050)
    assert _wait_until(lambda:n._rgb.info()['decoded']==3)
    n._cloud(cloud(1_050_000_000))
    assert _wait_until(lambda:n.status()['forwarded_clouds']==3)
    messages=published(n,'/sensor_data')[0].raw_messages
    assert [m.header.stamp.sec*10**9+m.header.stamp.nanosec for m in messages]==[
        1_000_000_000,1_200_000_000,1_050_000_000]
    assert messages[-1].left.data and messages[-1].right.data


def test_publish_lock_retires_an_overtaken_internal_admission(calibrated_node):
    n=calibrated_node;_local_motion(n);send(n)
    assert _wait_until(lambda:n.status()['forwarded_clouds']==1)
    with n._lock:
        n._published_seq=3
        n._rgb_reservations[1_100_000_000]=object()
        n._motion_mounts[1_100_000_000]=np.eye(4)
    n._publish_pair(n._active,(1_100_000_000,cloud(1_100_000_000),None,np.eye(4),1.,2))
    assert n.status()['forwarded_clouds']==1
    assert n.status()['dropped_runtime_queue']==1
    assert 1_100_000_000 not in n._rgb_reservations
    assert 1_100_000_000 not in n._motion_mounts


@pytest.fixture
def held_projection(monkeypatch):
    import plugins.pointcloud_slam_rgb as rgb
    entered, release = threading.Event(), threading.Event()
    project = rgb.project_depth

    def delayed(*args):
        entered.set()
        assert release.wait(2)
        return project(*args)

    monkeypatch.setattr(rgb, 'project_depth', delayed)
    yield entered, release
    release.set()


@pytest.mark.parametrize("action", ["deliver", "lose_pose_again", "stop"])
def test_searching_handles_slow_rgb_for_returning_external_pose(calibrated_node, held_projection, action):
    """A new external origin alone cannot locate an observation on the old map."""
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    missing = odom(1050, status='lost')
    missing['pose'] = None
    n._odom(NS(data=json.dumps(missing)))
    n._native_info(NS(ref_id=0, stats_keys=['Motus/RecoveryState/', 'Motus/Relocalizations/',
        'Motus/VisualInliers/'], stats_values=[1., 0., 0.]), n._active)

    entered, release = held_projection
    image(n, 1100)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    n._odom(NS(data=json.dumps(odom(1100, x=20, segment='new-origin'))))
    n._cloud(cloud(1_100_000_000))
    assert entered.wait(1)
    assert _wait_until(lambda: n.status()['rgb_preparation']['timeouts'] >= 1)
    future = n._rgb_task[1]
    if action == 'lose_pose_again':
        n._odom(NS(data=json.dumps(missing)))
        n._odom(NS(data=json.dumps(odom(1200, x=-40, segment='third-origin'))))
    elif action == 'stop':
        n.freeze_capture()
    release.set()
    assert _wait_until(future.done)
    if action != 'deliver':
        assert not _wait_until(lambda: n.status()['forwarded_clouds'] > 1, timeout=.1)
        assert not n._motion_waiting
        return
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
    replayed = published(n, '/sensor_data')[0].raw_messages[-1]
    assert replayed.left.data and replayed.right.data and replayed.laser_scan.data
    assert replayed.left.header.stamp.nanosec == 100_000_000
    assert published(n, '/odom')[0].raw_messages[-1].pose.pose.position.x == 20.
    assert len(NativeProcess.instances) == 1
    assert n.status()['native_recovery']['state'] == 'searching'
    assert not n._motion_waiting


def test_null_pose_heartbeats_preserve_admitted_localization_queries(calibrated_node):
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    absent = odom(1100, status='lost')
    absent['pose'] = None
    heartbeat = NS(data=json.dumps(absent))
    with n._lock:
        n._odom(heartbeat)
        n._cloud(cloud(1_100_000_000))
        assert len(n._queue) == 1
        n._odom(heartbeat)
        assert len(n._queue) == 1
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
    assert published(n, '/odom')[0].raw_messages[-1].pose.pose.orientation.w == 0.


def test_overtaken_external_observation_cannot_publish_after_later_query(calibrated_node):
    from plugins.pointcloud_slam_odom import decode_slam_odom
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    send(n)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    with n._lock:
        n._published_seq = 3
    pose = decode_slam_odom(odom(1100))
    n._publish_pair(n._active, (1_100_000_000, cloud(1_100_000_000), pose, np.eye(4), 1., 2))
    assert n.status()['forwarded_clouds'] == 1
    assert n.status()['dropped_runtime_queue'] >= 1


def test_new_map_retains_first_scan_with_its_slow_visual_anchor(calibrated_node, held_projection):
    n = calibrated_node
    entered, release = held_projection
    image(n, 1000)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    send(n)
    assert entered.wait(1)
    # The first view can become the only connected retrieval anchor.
    assert not _wait_until(lambda: n.status()['forwarded_clouds'], timeout=.06)
    release.set()
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    root = published(n, '/sensor_data')[0].raw_messages[0]
    assert root.left.data and root.right.data and root.laser_scan.data
    assert root.left.header.stamp == root.header.stamp


@pytest.mark.parametrize('first_visual_unconfirmed', [False, True])
def test_new_map_prefers_complete_observation_until_native_root(calibrated_node, first_visual_unconfirmed):
    from test_pointcloud_slam_stream import odom
    n = calibrated_node
    expected = []
    if first_visual_unconfirmed:
        image(n, 900)
        assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
        send(n, 900)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        expected.append(900)
    image(n, 1000)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == len(expected) + 1)
    n._odom(NS(data=json.dumps(odom(1000))))
    n._cloud(cloud(1_000_000_000, points=[(.1, .1, -1.)]))
    image(n, 1100)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == len(expected) + 2)
    send(n, 1100)
    expected.append(1100)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] >= len(expected))
    envelopes = published(n, '/sensor_data')[0].raw_messages
    assert [(m.header.stamp.sec * 1000 + m.header.stamp.nanosec // 1_000_000,
             bool(m.left.data and m.right.data)) for m in envelopes] == [(ms, True) for ms in expected]
    n._native_info(NS(ref_id=1, loop_closure_id=0, proximity_detection_id=0,
                      stats_keys=['Motus/RecoveryState/', 'Motus/Relocalizations/', 'Motus/VisualInliers/'],
                      stats_values=[0., 0., 0.]), n._active)
    send(n, 1200)
    assert _wait_until(lambda: n.status()['forwarded_clouds'] == len(expected) + 1, timeout=.2)
    assert not envelopes[-1].left.data


@pytest.mark.parametrize("phase", ["projection", "native_ack"])
def test_freeze_interrupts_pending_root_work(calibrated_node, held_projection, phase):
    n = calibrated_node
    entered, release = held_projection
    image(n, 1000)
    assert _wait_until(lambda: n._rgb.info()['decoded'] == 1)
    send(n)
    assert entered.wait(1)
    send(n, 1100)
    if phase == 'native_ack':
        release.set()
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    n.freeze_capture()
    assert _wait_until(lambda: not n._worker.is_alive(), timeout=.1)
    assert n.status()['forwarded_clouds'] == int(phase == 'native_ack')
