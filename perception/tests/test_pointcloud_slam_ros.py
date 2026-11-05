"""Real ROS message shape is decoded without inferring timestamps or axes."""
from types import SimpleNamespace as NS
import numpy as np
import pytest

from plugins.pointcloud_slam_ros import decode_cloud, SlamInputNode
from test_pointcloud_slam_stream import odom, stream


def cloud():
    points = np.array([[1, 2, 3]], dtype='<f4')
    return NS(header=NS(stamp=NS(sec=1, nanosec=0), frame_id='camera'),
              width=1, height=1, point_step=12, row_step=12, is_bigendian=False,
              fields=[NS(name=n, offset=i*4, datatype=7, count=1)
                      for i, n in enumerate('xyz')], data=points.tobytes())


def test_decodes_metric_xyz_and_preserves_header():
    stamp, frame, points = decode_cloud(cloud(), max_points=5)
    assert stamp == 1_000_000_000 and frame == 'camera'
    assert points.tolist() == [[1, 2, 3]]


def test_cloud_rejects_truncation_and_wrong_field_type():
    msg = cloud()
    msg.data = b'\0'
    with pytest.raises(ValueError):
        decode_cloud(msg, max_points=5)
    msg = cloud()
    msg.fields[1].offset = 1
    with pytest.raises(ValueError):
        decode_cloud(msg, max_points=5)
    msg = cloud()
    msg.fields[0].datatype = 2
    with pytest.raises(ValueError):
        decode_cloud(msg, max_points=5)


def test_two_subscriptions_feed_keyframe_store():
    import json
    from vision_stubs import _FakeExecutor
    s = stream()
    node = SlamInputNode('/dynamic/odom', '/dynamic/cloud', s, clock=lambda: 1.)
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/dynamic/odom'](NS(data=json.dumps(odom(1000))))
    callbacks['/dynamic/cloud'](cloud())
    assert len(s.store.segments[0].frames) == 1
    assert set(callbacks) == {'/dynamic/odom', '/dynamic/cloud'}


def test_node_uses_measurement_clock_for_age_checks():
    import json
    s = stream(max_measurement_age_ms=200, max_future_skew_ms=20)
    node = SlamInputNode('/o', '/c', s, clock=lambda: 5.,
                         measurement_clock_ns=lambda: 5_000_000_000)
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    assert not s.store.segments


def test_diagnostic_preview_publishes_one_segment_and_truthful_status():
    import json
    import struct
    s = stream()
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': 1., 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    node.publish_preview_once()
    pubs = {pub.topic: pub for pub in node.publishers}
    packet = pubs['/preview'].messages[-1]
    from test_pointcloud_slam_preview import unpack
    metadata, body = unpack(packet)
    assert struct.unpack('<fffBI', bytes(body[:17]))[-1] == 1
    assert metadata['segment_id'] == 'a' and metadata['frame_id'] == 'odom'
    status = json.loads(pubs['/preview_status'].messages[-1])
    assert status['kind'] == 'unvalidated_odometry_preview'
    assert status['segment_id'] == 'a' and status['accepted_loops'] == 0
    assert status['stamp_ns'] == '1000000000'
    callbacks['/o'](NS(data=json.dumps(odom(1100, status='lost'))))
    count = len(pubs['/preview'].messages)
    node.publish_preview_once()
    assert len(pubs['/preview'].messages) == count
    assert json.loads(pubs['/preview_status'].messages[-1])['state'] == 'lost'


def test_preview_does_not_publish_if_odometry_becomes_stale_during_rebuild():
    import json
    s = stream()
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': 1., 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    times = iter([1., 3.])
    node._clock = lambda: next(times)
    node.publish_preview_once()
    pubs = {pub.topic: pub for pub in node.publishers}
    assert not pubs['/preview'].messages
    assert json.loads(pubs['/preview_status'].messages[-1])['state'] == 'stale'


def test_preview_worker_reports_stale_without_new_input_and_stops():
    import json
    import threading
    from vision_stubs import _wait_until
    now = [1.]
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: now[0], preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': .1, 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    pubs = {pub.topic: pub for pub in node.publishers}
    node.start_preview()
    try:
        assert _wait_until(lambda: len(pubs['/preview'].messages) > 0)
        node._preview_wake.clear()
        threading.Event().wait(.15)
        now[0] = 3.
        assert _wait_until(lambda: pubs['/preview_status'].messages and
                           json.loads(pubs['/preview_status'].messages[-1])['state'] == 'stale')
    finally:
        node.stop_preview()
    assert not node._preview_worker.is_alive()


def test_preview_does_not_rebuild_large_map_without_a_subscriber(monkeypatch):
    import json
    import plugins.pointcloud_slam_map as map_module
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': 1., 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    pubs = {pub.topic: pub for pub in node.publishers}
    pubs['/preview'].subscription_count = 0
    def fail(*args):
        raise AssertionError('map should not be rebuilt')
    monkeypatch.setattr(map_module, 'rebuild_segment', fail)
    monkeypatch.setattr(node.stream, 'graph', fail)
    node.publish_preview_once()
    assert not pubs['/preview'].messages
    assert json.loads(pubs['/preview_status'].messages[-1])['state'] == 'tracking'


def test_preview_replaces_map_after_segment_change_without_merging():
    import json
    import struct
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': 1., 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    node.publish_preview_once()
    callbacks['/o'](NS(data=json.dumps(odom(1200, x=10, segment='b'))))
    msg = cloud()
    msg.header.stamp.nanosec = 200_000_000
    callbacks['/c'](msg)
    node.publish_preview_once()
    pubs = {pub.topic: pub for pub in node.publishers}
    packet = pubs['/preview'].messages[-1]
    from test_pointcloud_slam_preview import unpack
    metadata, body = unpack(packet)
    assert metadata['segment_id'] == 'b'
    assert struct.unpack('<fffBI', bytes(body[:17])) == (10., 0., 0., 3, 1)
    assert np.frombuffer(bytes(body[17:]), dtype='<f4').tolist() == [11., 2., 3.]
    assert json.loads(pubs['/preview_status'].messages[-1])['segment_id'] == 'b'


def test_gtsam_preview_runs_real_odom_backend():
    pytest.importorskip('gtsam')
    import json
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status', 'max_points': 10,
        'interval_s': 1., 'backend': 'gtsam', 'sigmas': [.1] * 6})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    node.publish_preview_once()
    pubs = {pub.topic: pub for pub in node.publishers}
    assert pubs['/preview'].messages
    assert json.loads(pubs['/preview_status'].messages[-1])['backend'] == 'gtsam'


def test_invalid_preview_configuration_does_not_leak_ros_node():
    from vision_stubs import _FakeNode
    with pytest.raises(ValueError):
        SlamInputNode('/o', '/c', stream(), preview={'topic': '/p', 'status_topic': '/s',
                      'max_points': -1, 'interval_s': 1., 'backend': 'raw'})
    assert not _FakeNode.instances or _FakeNode.instances[-1].destroyed


def test_preview_worker_reports_backend_error_in_status(monkeypatch):
    import json
    import plugins.pointcloud_slam_map as map_module
    from vision_stubs import _wait_until
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': .1, 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    monkeypatch.setattr(map_module, 'rebuild_segment', lambda *a: (_ for _ in ()).throw(ValueError('broken graph')))
    pubs = {pub.topic: pub for pub in node.publishers}
    node.start_preview()
    try:
        assert _wait_until(lambda: pubs['/preview_status'].messages and
                           json.loads(pubs['/preview_status'].messages[-1])['state'] == 'error')
    finally:
        node.stop_preview()
    assert 'broken graph' in json.loads(pubs['/preview_status'].messages[-1])['last_error']


def test_preview_stops_when_cloud_stale_even_if_odometry_is_fresh():
    import json
    now = [1.]
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: now[0], preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': 1., 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    node.publish_preview_once()
    pubs = {pub.topic: pub for pub in node.publishers}
    count = len(pubs['/preview'].messages)
    now[0] = 1.9
    callbacks['/o'](NS(data=json.dumps(odom(1900))))
    now[0] = 2.1
    callbacks['/o'](NS(data=json.dumps(odom(2100))))
    node.publish_preview_once()
    assert len(pubs['/preview'].messages) == count
    status = json.loads(pubs['/preview_status'].messages[-1])
    assert status['cloud_state'] == 'stale' and status['state'] == 'stale'


def test_inflight_preview_is_dropped_when_tracking_is_lost(monkeypatch):
    import json
    import plugins.pointcloud_slam_map as map_module
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/preview_status',
        'max_points': 10, 'interval_s': 1., 'backend': 'raw'})
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    original = map_module.rebuild_segment
    def lost_during_rebuild(*args):
        callbacks['/o'](NS(data=json.dumps(odom(1100, status='lost'))))
        return original(*args)
    monkeypatch.setattr(map_module, 'rebuild_segment', lost_during_rebuild)
    node.publish_preview_once()
    pubs = {pub.topic: pub for pub in node.publishers}
    assert not pubs['/preview'].messages
    assert json.loads(pubs['/preview_status'].messages[-1])['state'] == 'lost'


def test_preview_rejects_unbounded_update_rate():
    with pytest.raises(ValueError):
        SlamInputNode('/o', '/c', stream(), preview={'topic': '/p', 'status_topic': '/s',
                      'max_points': 10, 'interval_s': .001, 'backend': 'raw'})
