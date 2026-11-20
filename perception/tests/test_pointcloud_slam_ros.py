"""ROS adapter contract tests; these do not claim native engine acceptance."""
from copy import deepcopy
import json
import logging
import sqlite3
import struct
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

import rtabmap_stubs  # Message definitions only; no native runtime is simulated as acceptance.
from sensor_msgs.msg import PointCloud2, PointField
from vision_stubs import _wait_until
from test_pointcloud_slam_stream import odom


def cloud(stamp_ns=1_000_000_000, frame='camera', *, points=None):
    msg = PointCloud2()
    msg.header = NS(stamp=NS(sec=stamp_ns//1_000_000_000, nanosec=stamp_ns%1_000_000_000), frame_id=frame)
    points = [(1., 2., 3.)] if points is None else points
    msg.width, msg.height, msg.point_step = len(points), 1, 24
    msg.row_step, msg.is_bigendian, msg.is_dense = len(points)*24, False, True
    msg.fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                  for name, offset in [('x', 0), ('y', 4), ('z', 8), ('intensity', 12), ('time', 16)]]
    msg.data = b''.join(struct.pack('<ffffff', *point, .5, .001, 42.) for point in points)
    return msg


def settings(**extra):
    result = dict(cloud_frame='camera', child_frame='body', clock_id='clock',
        child_from_cloud=np.eye(4).tolist(), max_gap_ms=250, max_age_s=2,
        max_pending=2, max_cloud_points=100, max_measurement_age_ms=1500,
        max_future_skew_ms=100, voxel_size_m=.08, max_working_nodes=300,
        max_update_ms=100, max_disk_bytes=10_000_000, reserve_bytes=0,
        preview_interval_s=.1, preview_max_points=10)
    result.update(extra)
    return result


class NativeProcess:
    instances = []
    def __init__(self, database_path, namespace, settings):
        from pathlib import Path
        self.database_path, self.namespace = Path(database_path), namespace
        self.started, self.closed, self.code = False, False, None
        self.instances.append(self)
    def validate_runtime(self):
        pass
    def start(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as db:
            db.execute('CREATE TABLE Node(id INTEGER PRIMARY KEY)')
            db.execute('INSERT INTO Node VALUES(1)')
        self.started = True
    def poll(self):
        return self.code
    def close(self):
        if self.code not in (None, 0):
            raise RuntimeError('native exit ' + str(self.code))
        self.code, self.closed = 0, True
        return self.database_path


@pytest.fixture
def node(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    NativeProcess.instances = []
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    result = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path / 'session')
    result._receipt_clock = lambda: 1.
    result._measurement_clock_ns = lambda: 1_000_000_000
    result.start()
    yield result
    result.close()
    result.destroy_node()


def test_explicit_input_contract_can_start_without_an_optional_sidecar_topic(tmp_path,monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    from test_pointcloud_slam_input import declarations
    from plugins.pointcloud_slam_input import resolve_inputs
    from vision_stubs import _FakeNode
    original=_FakeNode.create_subscription
    def subscribe(self,kind,topic,callback,qos):
        if not isinstance(topic,str):raise TypeError('topic must be str')
        return original(self,kind,topic,callback,qos)
    monkeypatch.setattr(_FakeNode,'create_subscription',subscribe)
    metadata=declarations();metadata['/c'].pop('frame_metadata_topic')
    contract=resolve_inputs(metadata,{'/o':'state/odom','/c':'sensor/pointcloud2'})
    n=ros.RtabmapNode('/o','/c',settings(),session_dir=tmp_path,input_contract=contract)
    try:assert all(isinstance(sub.topic,str) for sub in n.subscriptions)
    finally:n.close();n.destroy_node()


def send(node, ms=1000, segment='a', *, stamp_ns=None, status='tracking'):
    node._odom(NS(data=json.dumps(odom(ms, segment=segment, status=status))))
    node._cloud(cloud(ms*1_000_000 if stamp_ns is None else stamp_ns))


def published(node, suffix):
    return [p for p in node.publishers if p.topic.endswith(suffix)]


def test_native_pair_preserves_all_point_fields_bytes_and_exact_nanosecond_stamp(node):
    node._odom(NS(data=json.dumps(odom(1000))))
    node._measurement_clock_ns = lambda: 1_200_000_000
    raw = cloud(1_100_000_001)
    original = deepcopy(raw)
    node._cloud(raw)
    node._odom(NS(data=json.dumps(odom(1200, x=2))))
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    native = published(node, '/sensor_data')[0].raw_messages[-1]
    assert native.laser_scan_format == 6  # Pinned RTAB-Map XYZ plus float intensity.
    forwarded = native.laser_scan
    pose = published(node, '/odom')[0].raw_messages[-1]
    assert forwarded.data == original.data
    assert [vars(f) for f in forwarded.fields] == [vars(f) for f in original.fields]
    assert (forwarded.point_step, forwarded.row_step, forwarded.width, forwarded.height) == (24, 24, 1, 1)
    assert forwarded.header.stamp.nanosec == pose.header.stamp.nanosec == 100_000_001
    assert pose.pose.pose.position.x == pytest.approx(1.00000001)
    assert raw.header.frame_id == 'camera'  # Header remapping doesn't mutate the producer message.



@pytest.mark.parametrize(('malformed', 'expected_error'), [
    ('big_endian', 'little-endian'),
    ('row_padding', 'row padding'),
])
def test_native_pointcloud_normalizes_big_endian_and_padded_rows_before_publish(
        node, malformed, expected_error):
    # decode_cloud can inspect more layouts than the native RTAB-Map converter
    # supports. Invalid memory layouts must never reach the native publisher.
    node._odom(NS(data=json.dumps(odom(1000))))
    bad = cloud()
    if malformed == 'big_endian':
        bad.is_bigendian = True
        bad.data = struct.pack('>ffffff', 1., 2., 3., .5, .001, 42.)
    else:
        bad.row_step += 8
        bad.data += bytes(8)

    node._cloud(bad)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    assert len(NativeProcess.instances) == 1
    forwarded = published(node, '/sensor_data')[0].raw_messages[-1].laser_scan
    assert forwarded.is_bigendian is False
    assert forwarded.row_step == forwarded.width * forwarded.point_step


def test_absent_numeric_vo_waits_and_stale_quality_label_does_not_gate(node):
    node._odom(NS(data='not JSON')); node._cloud(cloud())
    assert not NativeProcess.instances
    node._measurement_clock_ns = lambda: 9_000_000_000
    send(node,status='lost')
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 2)



def test_segment_label_changes_keep_one_native_process_and_database(node):
    for index,segment in enumerate(['a','b','a']):
        send(node,1000+100*index,segment)
        assert _wait_until(lambda: node.status()['forwarded_clouds'] == index+1)
    node.close()
    assert len(node.artifacts()['segments']) == 1
    assert len(NativeProcess.instances) == 1



def test_scan_only_rgbd_odom_labels_the_origin_on_each_native_envelope(tmp_path, monkeypatch):
    import zlib
    import plugins.pointcloud_slam_ros as ros

    NativeProcess.instances = []
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    n = ros.RtabmapNode('/rgbd_odometry/i_6f6e65/state/odom', '/c', settings(),
                        session_dir=tmp_path / 'session')
    n._receipt_clock = lambda: 1.
    n.start()
    try:
        for count, segment in enumerate(('first', 'first', 'reset'), 1):
            ms = 1000 + count*100
            n._measurement_clock_ns = lambda s=ms: s*1_000_000
            send(n, ms, segment)
            assert _wait_until(lambda: n.status()['forwarded_clouds'] == count)

        assert len(NativeProcess.instances) == 1
        native = published(n, '/sensor_data')[0].raw_messages
        assert all(msg.laser_scan.data and not msg.left.data for msg in native)
        markers = [zlib.decompress(bytes(msg.user_data)[:-12]) for msg in native]
        assert all(len(marker)==24 and marker.startswith(b'MOTUSOR1') for marker in markers)
        assert markers[0]==markers[1] and markers[0]!=markers[2]
    finally:
        n.close()
        n.destroy_node()


@pytest.mark.parametrize('legacy_first', [False, True])
def test_measured_chassis_without_rgb_marks_origin_and_dense_run(node, legacy_first):
    import zlib
    from test_pointcloud_slam_stream import chassis
    if legacy_first:
        send(node, 900, 'legacy')
        assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    for i, segment in enumerate(('old', 'old', 'reset')):
        ms = 1000 + i * 100
        node._measurement_clock_ns = lambda t=ms: t * 1_000_000
        node._odom(NS(data=json.dumps(chassis(ms, segment=segment))))
        node._cloud(cloud(ms * 1_000_000))
        assert _wait_until(lambda: node.status()['forwarded_clouds'] == i + 1 + legacy_first)

    native = published(node, '/sensor_data')
    assert len(native) == 1
    markers = [zlib.decompress(bytes(m.user_data)[:-12]) for m in native[0].raw_messages]
    if legacy_first:
        assert markers.pop(0) == b'MOTUSRAW'
    assert all(len(marker) == 40 and marker.startswith(b'MOTUSCH1') for marker in markers)
    assert markers[0] == markers[1]
    assert markers[1][8:24] != markers[2][8:24]
    assert markers[1][24:] != markers[2][24:]


def test_mapping_snapshot_uses_actual_native_frame_and_optimized_pose(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    graph_sub = next(s for s in node.subscriptions if s.topic.endswith('/mapGraph'))
    cloud_sub = next(s for s in node.subscriptions if s.topic.endswith('/cloud_map'))
    assert cloud_sub.qos.reliability == 'reliable' and cloud_sub.qos.durability == 'tl'
    graph_sub.callback(NS(header=NS(frame_id='native_map', stamp=NS(sec=1, nanosec=0)),
        poses_id=[1], poses=[NS(position=NS(x=7., y=8., z=0.),
            orientation=NS(x=0., y=0., z=0., w=1.))],
        map_to_odom=NS(translation=NS(x=7., y=8., z=0.), rotation=NS(x=0., y=0., z=0., w=1.))))
    cloud_sub.callback(cloud(frame='native_map'))
    packet = published(node, '/pointcloud_slam/map')[0].messages[-1]
    assert struct.unpack('<fffBI', bytes(packet[:17])) == (7., 8., 0., 39, 1)
    node.close()
    saved = node.artifacts()['segments'][0]
    assert saved['frame_id'] == 'native_map' and saved['snapshot'] == bytes(packet)
    assert [(output['topic'], output['format']) for output in node.outputs()] == [
        ('/pointcloud_slam/map', 'sensor/mapping')]
    assert [publisher.topic for publisher in node.publishers
            if publisher.topic == '/pointcloud_slam/map'] == ['/pointcloud_slam/map']
    assert not any(p.topic.endswith('slam_odom_preview_status') for p in node.publishers)


def test_native_info_counts_real_accepted_closures_without_duplicate_inflation(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    callback = next(s.callback for s in node.subscriptions if s.topic.endswith('/info'))
    value = NS(ref_id=12, loop_closure_id=3, proximity_detection_id=0)
    callback(value); callback(value)
    callback(NS(ref_id=13, loop_closure_id=0, proximity_detection_id=5))
    status = node.status()
    assert status['accepted_loops'] == 2
    assert status['native_loop_closures'] == 1 and status['native_proximity_detections'] == 1


def test_native_activity_does_not_count_unconnected_or_repeated_graph_nodes(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    info = next(s.callback for s in node.subscriptions if s.topic.endswith('/info'))
    graph = next(s.callback for s in node.subscriptions if s.topic.endswith('/mapGraph'))

    def header(seconds):
        return NS(frame_id='native_map', stamp=NS(sec=seconds, nanosec=0))

    def observe_graph(seconds, ids, links):
        pose = NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.))
        graph(NS(header=header(seconds), poses_id=ids, poses=[pose]*len(ids),
                 links=[NS(from_id=a, to_id=b, type=kind) for a,b,kind in links]))
        return node.status()['backend_progress']

    info(NS(header=header(2), ref_id=12, loop_closure_id=0, proximity_detection_id=0))
    progress = node.status()['backend_progress']
    assert progress['state'] == 'responsive_without_new_nodes'
    assert progress['optimized_nodes'] is None
    assert observe_graph(2, [1, 2], [(1, 2, 0)])['new_connected_nodes'] == 0

    # A node may leave the local graph while a new connected keyframe arrives.
    progress = observe_graph(3, [2, 3], [(2, 3, 0)])
    assert progress['optimized_nodes'] == 2 and progress['new_connected_nodes'] == 1
    assert progress['last_growth_stamp_ns'] == 3_000_000_000
    assert progress['state'] == 'responsive'

    # A virtual link and a detached cluster do not connect new poses to the original map.
    progress = observe_graph(4, [2, 3, 4, 5], [(2, 3, 0), (3, 4, 5), (4, 5, 0)])
    assert progress['optimized_nodes'] == 4 and progress['connected_nodes'] == 2
    assert progress['new_connected_nodes'] == 1
    assert progress['state'] == 'responsive_without_new_nodes'
    assert observe_graph(3, [2, 6], [(2, 6, 0)])['last_growth_stamp_ns'] == 3_000_000_000

    info(NS(header=header(5), ref_id=0, stats_keys=['Motus/RecoveryState/',
        'Motus/Relocalizations/', 'Motus/VisualInliers/'], stats_values=[1., 0., 0.]))
    progress = node.status()['backend_progress']
    assert progress['activity_stamp_ns'] == 5_000_000_000
    assert progress['last_growth_stamp_ns'] == 3_000_000_000


def test_nonfinite_native_output_never_publishes_a_mapping_packet(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    sub = next(s for s in node.subscriptions if s.topic.endswith('/cloud_map'))
    sub.callback(cloud(frame='native_map', points=[(float('inf'), 0., 0.)]))
    assert not published(node, '/pointcloud_slam/map')[0].messages



def test_native_rejection_warnings_are_rate_limited_per_input_and_preview_recovers(node, caplog):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    graph = next(s.callback for s in node.subscriptions if s.topic.endswith('/mapGraph'))
    output = next(s.callback for s in node.subscriptions if s.topic.endswith('/cloud_map'))
    invalid_graph = NS(poses_id=[1], poses=[NS(position=None)])
    invalid_cloud = cloud(frame='native_map')
    invalid_cloud.width = -1

    with caplog.at_level(logging.WARNING, logger='plugins.pointcloud_slam_ros'):
        for _ in range(250):
            graph(invalid_graph)
            output(invalid_cloud)

        def warnings(kind):
            return [record.getMessage() for record in caplog.records
                    if record.name == 'plugins.pointcloud_slam_ros'
                    and record.getMessage().startswith('Invalid native RTAB-Map ' + kind)]

        assert len(warnings('graph')) == len(warnings('cloud')) == 3
        assert node.status()['native_graph_rejections'] == 250
        assert node.status()['native_cloud_rejections'] == 250
        assert node.status()['native_graph_rejection_streak'] == 250
        assert node.status()['native_cloud_rejection_streak'] == 250

        graph(NS(header=NS(frame_id='native_map', stamp=NS(sec=1, nanosec=0)),
            poses_id=[1], poses=[NS(position=NS(x=1., y=2., z=0.),
                orientation=NS(x=0., y=0., z=0., w=1.))]))
        output(cloud(frame='native_map'))
        assert published(node, '/pointcloud_slam/map')[0].messages
        assert node.status()['native_graph_rejection_streak'] == 0
        assert node.status()['native_cloud_rejection_streak'] == 0
        assert node.status()['native_graph_rejections'] == 250
        assert node.status()['native_cloud_rejections'] == 250

        graph(invalid_graph)
        output(invalid_cloud)
        assert len(warnings('graph')) == len(warnings('cloud')) == 4
        assert node.status()['native_graph_rejections'] == 251
        assert node.status()['native_cloud_rejections'] == 251


def test_native_rejection_warning_sanitizes_and_bounds_detail(node, caplog, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    output = next(s.callback for s in node.subscriptions if s.topic.endswith('/cloud_map'))

    def invalid_cloud(*args, **kwargs):
        raise ValueError('bad\n' + 'x' * 300 + '\x1b[31m')

    monkeypatch.setattr(ros, 'decode_cloud', invalid_cloud)
    with caplog.at_level(logging.WARNING, logger='plugins.pointcloud_slam_ros'):
        output(cloud(frame='native_map'))
    warning = next(record.getMessage() for record in caplog.records
                   if record.getMessage().startswith('Invalid native RTAB-Map cloud'))
    detail = warning.split(': ', 1)[1]
    assert len(detail) == 200
    assert '\n' not in warning and '\x1b' not in warning


def test_freeze_during_slow_native_start_is_immediate_and_cannot_resume(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    entered, release = threading.Event(), threading.Event()
    class Slow(NativeProcess):
        def start(self):
            entered.set(); assert release.wait(3)
            super().start()
    monkeypatch.setattr(ros, 'RtabmapProcess', Slow)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path)
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n)
        assert entered.wait(1)
        before = time.monotonic(); n.freeze_capture()
        assert time.monotonic() - before < .1
        send(n, 1100, 'b')
    finally:
        release.set(); n.close()
    assert n.status()['forwarded_clouds'] == 0
    assert all(not p.raw_messages for p in published(n, '/sensor_data'))
    assert n.status()['capture_stopped'] is True


def test_native_failure_retains_database_and_close_reports_failure(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    process = NativeProcess.instances[-1]
    process.code = 7
    with pytest.raises(RuntimeError, match='exit'):
        node.close()
    assert process.database_path.exists()
    process.code = 0  # Allow fixture cleanup after the failure assertion.


def test_freeze_after_segment_label_change_keeps_one_engine(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    NativeProcess.instances = []
    monkeypatch.setattr(ros,'RtabmapProcess',NativeProcess)
    n = ros.RtabmapNode('/o','/c',settings(),session_dir=tmp_path)
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n); assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        send(n,1100,'b'); assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
        n.freeze_capture(); send(n,1200,'c')
        assert n.status()['forwarded_clouds'] == 2
    finally: n.close()
    assert len(NativeProcess.instances) == 1



def test_sidecar_digest_is_optional_for_native_forwarding(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    from test_pointcloud_slam_input import measured_odom, packet, resolved
    NativeProcess.instances = []
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path, input_contract=resolved())
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        n._odom(NS(data=json.dumps(measured_odom())))
        n._cloud(cloud())
        n._cloud_metadata(NS(data=json.dumps(packet())))
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    finally:
        n.close()


@pytest.mark.parametrize('with_sidecars', [False, True])
def test_admitted_clouds_forward_while_newer_clouds_wait_for_metadata(
        tmp_path, monkeypatch, with_sidecars):
    import plugins.pointcloud_slam_ros as ros
    from test_pointcloud_slam_input import measured_odom, packet, resolved
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    n = ros.RtabmapNode('/o', '/c', settings(max_pending=10),
        session_dir=tmp_path, input_contract=resolved())
    now = [1.]
    n._receipt_clock = lambda: now[0]
    n._measurement_clock_ns = lambda: int(now[0]*1_000_000_000)
    try:
        # Run the worker synchronously so a newer pending cloud is always
        # present when an already admitted frame reaches the native publisher.
        for i in range(30):
            ms = 1000 + i*33
            now[0] = ms/1000
            n._odom(NS(data=json.dumps(measured_odom(ms))))
            n._cloud(cloud(ms*1_000_000))
            if with_sidecars and i:
                n._cloud_metadata(NS(data=json.dumps(packet(ms-33))))
            while n._queue:
                item = n._queue.popleft()
                segment = n._active or n._activate(item[2])
                n._publish_pair(segment, item)
        status = n.status()
        assert status['unmatched_clouds'] == 0
        assert status['accepted_clouds'] > 0
        assert status['forwarded_clouds'] == status['accepted_clouds']
        forwarded = published(n, '/sensor_data')[0].raw_messages
        assert len(forwarded) == status['forwarded_clouds']
        assert forwarded[0].header.stamp.sec == 1
        assert forwarded[0].header.stamp.nanosec == 0
    finally:
        n.close()
        n.destroy_node()


def test_decode_padded_organized_big_endian_cloud_preserves_input_bytes():
    from plugins.pointcloud_slam_ros import decode_cloud
    msg = cloud(points=[(1., 2., 3.), (4., 5., 6.)])
    msg.width, msg.height, msg.point_step, msg.row_step = 1, 2, 24, 28
    msg.is_bigendian = True
    msg.data = b''.join(struct.pack('>ffffff', x, x+1, x+2, .5, .001, 42.) + b'PAD!'
                        for x in [1., 4.])
    original = bytes(msg.data)
    stamp, frame, points = decode_cloud(msg, max_points=2)
    assert points.tolist() == [[1., 2., 3.], [4., 5., 6.]]
    assert msg.data == original and stamp == 1_000_000_000


def test_native_snapshot_requires_matching_frame_and_capture_stamp(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    graph = next(s for s in node.subscriptions if s.topic.endswith('/mapGraph'))
    output = next(s for s in node.subscriptions if s.topic.endswith('/cloud_map'))
    graph.callback(NS(header=NS(frame_id='map_a', stamp=NS(sec=1, nanosec=0)),
        poses_id=[1], poses=[NS(position=NS(x=1., y=0., z=0.),
            orientation=NS(x=0., y=0., z=0., w=1.))],
        map_to_odom=NS(translation=NS(x=0., y=0., z=0.), rotation=NS(x=0., y=0., z=0., w=1.))))
    output.callback(cloud(frame='map_b'))
    output.callback(cloud(1_000_000_001, frame='map_a'))
    assert not published(node, '/pointcloud_slam/map')[0].messages


def test_loss_while_engine_starts_blocks_already_queued_pair(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    entered, release = threading.Event(), threading.Event()
    class SlowStart(NativeProcess):
        def start(self):
            entered.set(); assert release.wait(3)
            super().start()
    monkeypatch.setattr(ros, 'RtabmapProcess', SlowStart)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path)
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n); assert entered.wait(1)
        missing = odom(1100, status='lost'); missing['pose'] = None
        n._odom(NS(data=json.dumps(missing)))
        release.set()
        time.sleep(.1)
        assert n.status()['forwarded_clouds'] == 0
    finally:
        release.set(); n.close()


def test_disk_limit_stops_capture_but_valid_native_database_remains_exportable(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    node._settings['max_disk_bytes'] = 1
    assert _wait_until(lambda: node.status()['capture_stopped'])
    assert 'disk budget' in node.status()['stop_reason']
    node.close()
    artifacts = node.artifacts()
    assert len(artifacts['segments']) == 1
    assert 'disk budget' in artifacts['statistics']['stop_reason']
    assert artifacts['statistics']['state'] == 'storage_limit'


def test_disk_budget_tolerates_sqlite_journal_removed_during_scan(node, monkeypatch):
    from pathlib import Path
    node._status_stop.set()
    node._status_worker.join(timeout=1)
    database = node._session_dir / 'rtabmap.db'
    database.write_bytes(b'database')
    journal = node._session_dir / 'rtabmap.db-journal'
    journal.write_bytes(b'transient journal')
    is_file = Path.is_file

    def remove_after_discovery(path):
        result = is_file(path)
        if path == journal and result:
            path.unlink()
        return result

    monkeypatch.setattr(Path, 'is_file', remove_after_discovery)
    node._check_budget()
    assert node._disk_bytes == 8
    assert node.status()['capture_stopped'] is False
    assert database.read_bytes() == b'database'


def test_disk_budget_counts_existing_sqlite_journal(node):
    from plugins.pointcloud_slam_ros import _StorageLimit
    node._status_stop.set()
    node._status_worker.join(timeout=1)
    (node._session_dir / 'rtabmap.db').write_bytes(b'database')
    (node._session_dir / 'rtabmap.db-journal').write_bytes(b'journal')
    node._settings['max_disk_bytes'] = 10
    with pytest.raises(_StorageLimit, match='disk budget'):
        node._check_budget()
    assert node._disk_bytes == 15


def test_disk_budget_does_not_hide_other_io_failures(node, monkeypatch):
    from pathlib import Path
    node._status_stop.set()
    node._status_worker.join(timeout=1)
    journal = node._session_dir / 'rtabmap.db-journal'
    journal.write_bytes(b'journal')
    stat = Path.stat

    def unreadable(path, *args, **kwargs):
        if path == journal:
            raise PermissionError('cannot inspect journal')
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', unreadable)
    with pytest.raises(PermissionError, match='cannot inspect journal'):
        node._check_budget()


def test_segment_label_changes_do_not_close_engine_or_trigger_watchdog(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    NativeProcess.instances = []
    monkeypatch.setattr(ros,'RtabmapProcess',NativeProcess)
    n = ros.RtabmapNode('/o','/c',settings(),session_dir=tmp_path)
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n); assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        send(n,1100,'b'); assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
        time.sleep(.2)
        assert n.status()['capture_stopped'] is False
        assert not NativeProcess.instances[0].closed
    finally: n.close()



def test_freeze_is_not_blocked_by_session_directory_creation(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    from pathlib import Path
    entered, release, frozen = threading.Event(), threading.Event(), threading.Event()
    directory = tmp_path / 'slow-directory'
    original = Path.mkdir
    def slow(path, *args, **kwargs):
        if path == directory:
            entered.set(); assert release.wait(3)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', slow)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=directory)
    start = threading.Thread(target=n.start)
    freeze = threading.Thread(target=lambda: (n.freeze_capture(), frozen.set()))
    start.start()
    try:
        assert entered.wait(1)
        freeze.start()
        assert frozen.wait(.2), 'capture freeze must not wait for filesystem operations'
    finally:
        release.set(); start.join(3); freeze.join(3)
        n.close()
    assert n.status()['capture_stopped'] is True and not n.artifacts()['segments']


@pytest.mark.parametrize('layout,reason', [('trailing_bytes', 'invalid cloud dimensions')])
def test_native_unsupported_layout_is_rejected_before_process_start(node, layout, reason):
    """The pinned native converter cannot safely consume these valid ROS layouts."""
    msg = cloud()
    if layout == 'row_padding':
        msg.width, msg.height, msg.point_step, msg.row_step = 1, 2, 12, 20
        msg.fields = msg.fields[:3]
        msg.data = b''.join(struct.pack('<fff', *point) + b'padding!'
                            for point in [(1., 2., 3.), (4., 5., 6.)])
    elif layout == 'big_endian':
        msg.is_bigendian = True
        msg.data = struct.pack('>ffffff', 1., 2., 3., .5, .001, 42.)
    else:
        msg.data += b'trailing'
    node._odom(NS(data=json.dumps(odom(1000))))
    node._cloud(msg)
    assert _wait_until(lambda: reason in node.status()['last_error'])
    assert node.status()['accepted_clouds'] == 0
    assert node.status()['forwarded_clouds'] == 0
    assert not NativeProcess.instances


def test_camera_pose_reference_reaches_native_graph_preview(node):
    sample = odom(1000)
    sample['vendor']['pose_reference'] = 'camera_optical'
    node._odom(NS(data=json.dumps(sample)))
    node._cloud(cloud())
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    graph_sub = next(s for s in node.subscriptions if s.topic.endswith('/mapGraph'))
    cloud_sub = next(s for s in node.subscriptions if s.topic.endswith('/cloud_map'))
    # A +90 degree rotation around X points optical +Z toward map -Y.
    graph_sub.callback(NS(header=NS(frame_id='native_map', stamp=NS(sec=1, nanosec=0)),
        poses_id=[1], poses=[NS(position=NS(x=7., y=8., z=0.),
            orientation=NS(x=2 ** -.5, y=0., z=0., w=2 ** -.5))],
        map_to_odom=NS(translation=NS(x=0., y=0., z=0.),
            rotation=NS(x=0., y=0., z=0., w=1.))))
    cloud_sub.callback(cloud(frame='native_map'))
    packet = published(node, '/pointcloud_slam/map')[0].messages[-1]
    x, y, yaw, flags, count = struct.unpack('<fffBI', bytes(packet[:17]))
    assert (x, y, flags, count) == (7., 8., 15, 1)
    assert yaw == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("with_context", [False, True])
def test_gateway_waits_for_each_exact_dynamic_capture_mount(tmp_path, monkeypatch, with_context):
    import plugins.pointcloud_slam_ros as ros
    from plugins.pointcloud_slam_capture_mount import resolve_capture_mount
    from test_pointcloud_slam_capture_mount import camera, pose
    from test_rgbd_body_mount import packet
    monkeypatch.setattr(ros, 'RtabmapProcess', NativeProcess)
    cfg, _ = resolve_capture_mount(pose(), camera(), 'camera_optical')
    n = ros.RtabmapNode('/o', '/c', settings(**cfg), session_dir=tmp_path / 'session')
    n._receipt_clock = lambda: 1.2
    n._measurement_clock_ns = lambda: 1_200_000_000
    try:
        n.start()
        n._odom(NS(data=json.dumps(pose())))
        later = pose(); later['stamp_ms'] = 1200
        n._odom(NS(data=json.dumps(later)))
        raw = cloud(1_100_000_123, 'camera_optical')
        context = None
        if with_context:
            from plugins.spatial_time import TimePolicy
            context = TimePolicy('ingress').select(dict(stamp_ns=1_100_000_123,
                clock_id='camera-clock', timestamp_semantics='source_capture_mapped'),
                1_200_000_000, 1_200_000_000)
        n._cloud(raw, context)
        assert n.status()['forwarded_clouds'] == 0
        sub = next(s for s in n.subscriptions if s.topic == '/camera/rgbd/body_mount')
        sub.callback(NS(data=json.dumps(packet(1_100_000_123))))
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        transforms = published(n, '/tf')[0].raw_messages[-1].transforms
        assert len(transforms) == 2
        assert not published(n, '/tf_static')
        native_pose = published(n, '/odom')[0].raw_messages[-1]
        assert transforms[0].header.frame_id == native_pose.header.frame_id
        assert transforms[0].child_frame_id == native_pose.child_frame_id
        assert vars(transforms[0].header.stamp) == vars(native_pose.header.stamp)
        assert transforms[1].header.frame_id == native_pose.child_frame_id
    finally:
        n.close(); n.destroy_node()


def test_shared_input_leases_are_released_by_gateway_close(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    from plugins.spatial_input import InputAdapterRegistry
    subscriptions, released = [], []
    def subscribe(topic, kind, callback):
        subscriptions.append(topic)
        return lambda: released.append(topic)
    registry = InputAdapterRegistry(subscribe=subscribe)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path,
                        input_registry=registry)
    try:
        assert subscriptions == ['/o', '/c']
        n.close()
        assert sorted(released) == ['/c', '/o']
        n.close()
        assert len(released) == 2
    finally:
        registry.close()
        n.destroy_node()


def test_waiting_for_pose_does_not_refresh_original_ingress_age(node):
    # Stop only this test's worker so we can inspect the actual native queue.
    node._worker_stop.set();node._wake.set();node._worker.join(timeout=1)
    tick=[1.]
    node._receipt_clock=lambda:tick[0]
    node._odom(NS(data=json.dumps(odom(1000))))
    node._cloud(cloud(1_100_000_000))
    tick[0]=1.4
    node._measurement_clock_ns=lambda:1_200_000_000
    node._odom(NS(data=json.dumps(odom(1200))))
    assert len(node._queue)==1
    assert node._queue[0][4]==1.


def test_unknown_covariance_reports_backend_default(node):
    send(node)
    assert _wait_until(lambda:node.status()['forwarded_clouds']==1)
    assert node.status()['covariance_source']=='backend_default'
    assert node.status()['upstream_covariance']=='unknown'


def test_discovery_accepts_only_resolved_camera_clock_reference(monkeypatch):
    from importlib import import_module
    monkeypatch.setattr(import_module("rclpy.qos"), "qos_profile_sensor_data", object(), raising=False)
    from plugins.pointcloud_slam_config import discover_runtime_geometry
    from plugins.spatial_time import TimePolicy
    from test_pointcloud_slam_capture_mount import camera, pose
    context = TimePolicy('ingress').select(dict(stamp_ns=1_100_000_123,
        clock_id='camera-clock', timestamp_semantics='source_capture_mapped'),
        1_200_000_000, 1_200_000_000)
    class Registry:
        def acquire(self, topic, kind, declaration, callback):
            if topic == '/o': callback(NS(data=json.dumps(pose())), None)
            else: callback(cloud(1_100_000_123, 'camera_optical'), context)
            return NS(close=lambda: None)
    executor = NS(add_node=lambda n: None, remove_node=lambda n: None)
    cfg, provenance = discover_runtime_geometry(executor, '/o', '/c',
        input_registry=Registry(), camera_info=camera(), timeout_s=.01)
    assert cfg['clock_id'] == 'clock'
    assert cfg['capture_mount']['source_clock'] == 'camera-clock'
    assert provenance['clock_accuracy_validated'] is False
    bad = camera(); bad['vendor']['clock_reference']['domain_id'] = 'other-host'
    assert discover_runtime_geometry(executor, '/o', '/c', input_registry=Registry(),
        camera_info=bad, timeout_s=.01)[0]['extrinsic_mode'] == 'dynamic'

    from dataclasses import replace
    context = replace(context, observation_stamp_ns=1_100_000_124)
    assert discover_runtime_geometry(executor, '/o', '/c', input_registry=Registry(),
        camera_info=camera(), timeout_s=.01)[0]['extrinsic_mode'] == 'dynamic'

@pytest.mark.parametrize('ready_at', [18., None])
def test_native_discovery_allows_slow_start_but_remains_bounded(node, monkeypatch, ready_at):
    import plugins.pointcloud_slam_ros as ros
    tick = [0.]
    create = node.create_publisher
    def publisher(*args, **kwargs):
        p = create(*args, **kwargs)
        p.get_subscription_count = lambda: int(ready_at is not None and tick[0] >= ready_at)
        return p
    with monkeypatch.context() as patch:
        patch.setattr(ros, 'time', NS(monotonic=lambda: tick[0]))
        patch.setattr(node, 'create_publisher', publisher)
        patch.setattr(node, '_capture_stop', NS(is_set=lambda: False,
            wait=lambda delay: tick.__setitem__(0, tick[0] + 1.)))
        pose = NS(segment_id='slow', frame_id='odom')
        if ready_at is None:
            with pytest.raises(RuntimeError, match='discovery timed out'):
                node._activate(pose)
            assert tick[0] == 30.
        else:
            assert node._activate(pose)['process'].started
            assert tick[0] == ready_at


def test_native_odom_tf_matches_admitted_pose_and_capture(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    pose = published(node, '/odom')[0].raw_messages[-1]
    dynamic = published(node, '/tf')[0].raw_messages[-1].transforms
    transform = next(t for t in dynamic if t.child_frame_id == pose.child_frame_id)
    assert vars(transform.header.stamp) == vars(pose.header.stamp)
    assert transform.header.frame_id == pose.header.frame_id
    assert vars(transform.transform.translation) == vars(pose.pose.pose.position)
    assert vars(transform.transform.rotation) == vars(pose.pose.pose.orientation)
    static = published(node, '/tf_static')[0].raw_messages[-1].transforms
    assert all(t.header.frame_id != pose.header.frame_id for t in static)
