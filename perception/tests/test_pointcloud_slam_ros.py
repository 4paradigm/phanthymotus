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
    forwarded = published(node, '/scan_cloud')[0].raw_messages[-1]
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
def test_native_pointcloud_rejects_big_endian_and_padded_rows_before_publish(
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
    assert expected_error in node.stream.last_error
    assert node.status()['forwarded_clouds'] == 0
    assert not NativeProcess.instances

    # A malformed scan must not poison the next conforming measurement.
    node._cloud(cloud())
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    assert len(NativeProcess.instances) == 1
    forwarded = published(node, '/scan_cloud')[0].raw_messages[-1]
    assert forwarded.is_bigendian is False
    assert forwarded.row_step == forwarded.width * forwarded.point_step


def test_invalid_lost_and_stale_vo_never_reach_native_process(node):
    send(node, status='lost')
    node._odom(NS(data='not JSON'))
    node._cloud(cloud())
    node._measurement_clock_ns = lambda: 9_000_000_000
    send(node)
    time.sleep(.05)
    assert not NativeProcess.instances
    assert node.status()['forwarded_clouds'] == 0


def test_external_segments_get_separate_processes_databases_and_retired_input_is_blocked(node):
    send(node)
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 1)
    send(node, 1100, 'b')
    assert _wait_until(lambda: node.status()['forwarded_clouds'] == 2)
    send(node, 1200, 'a')
    time.sleep(.05)
    node.close()
    segments = node.artifacts()['segments']
    assert [s['segment_id'] for s in segments] == ['a', 'b']
    assert len({s['database'] for s in segments}) == 2
    assert all(p.closed for p in NativeProcess.instances)
    assert node.status()['forwarded_clouds'] == 2


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
    assert struct.unpack('<fffBI', bytes(packet[:17])) == (7., 8., 0., 3, 1)
    node.close()
    saved = node.artifacts()['segments'][0]
    assert saved['frame_id'] == 'native_map' and saved['snapshot'] == packet
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
    assert all(not p.raw_messages for p in published(n, '/scan_cloud'))
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


def test_stop_during_previous_segment_close_does_not_start_another_engine(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    entered, release = threading.Event(), threading.Event()
    class SlowClose(NativeProcess):
        def close(self):
            if not self.closed:
                entered.set(); assert release.wait(3)
            return super().close()
    NativeProcess.instances = []
    monkeypatch.setattr(ros, 'RtabmapProcess', SlowClose)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path)
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n)
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        send(n, 1100, 'b')
        assert entered.wait(1)
        n.freeze_capture()
    finally:
        release.set(); n.close()
    assert len(NativeProcess.instances) == 1
    assert [s['segment_id'] for s in n.artifacts()['segments']] == ['a']


def test_sidecar_digest_must_match_before_native_forwarding(tmp_path, monkeypatch):
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
        assert not NativeProcess.instances
        n._cloud_metadata(NS(data=json.dumps(packet())))
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
    finally:
        n.close()


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
        n._odom(NS(data=json.dumps(odom(1100, status='lost'))))
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


def test_watchdog_does_not_treat_normal_exit_during_segment_close_as_crash(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_ros as ros
    entered, release = threading.Event(), threading.Event()
    class ValidatingClose(NativeProcess):
        def close(self):
            if not self.closed:
                self.code = 0
                entered.set(); assert release.wait(3)
            return super().close()
    monkeypatch.setattr(ros, 'RtabmapProcess', ValidatingClose)
    n = ros.RtabmapNode('/o', '/c', settings(), session_dir=tmp_path)
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n.start()
    try:
        send(n); assert _wait_until(lambda: n.status()['forwarded_clouds'] == 1)
        send(n, 1100, 'b'); assert entered.wait(1)
        time.sleep(.2)
        assert n.status()['capture_stopped'] is False
        release.set()
        assert _wait_until(lambda: n.status()['forwarded_clouds'] == 2)
    finally:
        release.set(); n.close()


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


@pytest.mark.parametrize('layout,reason', [('row_padding', 'row padding'),
                                         ('big_endian', 'little-endian'),
                                         ('trailing_bytes', 'exactly match')])
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
