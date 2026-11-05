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
        'max_points': 10, 'interval_s': 1., 'backend': 'raw',
        'input_validation': {'policy': 'experimental', 'clock_relation_verified': False,
                             'extrinsic_calibrated': False}})
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
    assert metadata['input_validation']['policy'] == 'experimental'
    status = json.loads(pubs['/preview_status'].messages[-1])
    assert status['kind'] == 'unvalidated_odometry_preview'
    assert status['input_validation']['clock_relation_verified'] is False
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
    node._receipt_clock = lambda: next(times)
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


def test_geometry_worker_updates_historical_preview_without_touching_odom():
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import make_stream, add_frame
    from test_pointcloud_slam_preview import unpack
    s = make_stream(); points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        add_frame(s, t, x, points)
    original = s.graph()[0]['poses'][-1].copy()
    options = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.3, preview={
        'topic': '/preview', 'status_topic': '/preview_status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': options})
    node.publish_preview_once()
    pubs = {p.topic: p for p in node.publishers}
    meta, _ = unpack(pubs['/preview'].messages[-1])
    status = json.loads(pubs['/preview_status'].messages[-1])
    assert meta['kind'] == 'unvalidated_geometry_preview' and meta['candidate_count'] == 1
    assert status['accepted_loops'] == 0 and status['candidate_count'] == 1
    np.testing.assert_allclose(np.asarray(status['map_from_odom']) @ original,
                               np.asarray(status['map_pose']), atol=1e-6)
    np.testing.assert_allclose(s.graph()[0]['poses'][-1], original)
    pose_out = json.loads(pubs['/preview/pose'].messages[-1])
    correction = json.loads(pubs['/preview/map_to_odom'].messages[-1])
    assert pose_out['kind'] == 'unvalidated_geometry_pose'
    assert correction['kind'] == 'unvalidated_map_to_odom'
    np.testing.assert_allclose(np.asarray(correction['transform']) @ original,
                               np.asarray(pose_out['pose']), atol=1e-6)


def test_geometry_worker_processes_without_a_map_viewer():
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import make_stream, add_frame
    s = make_stream(); points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]: add_frame(s, t, x, points)
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.3, preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts})
    pubs = {p.topic: p for p in node.publishers}
    pubs['/preview'].subscription_count = 0
    node.publish_preview_once()
    assert not pubs['/preview'].messages
    assert json.loads(pubs['/status'].messages[-1])['candidate_count'] == 1


def test_geometry_worker_starts_new_unmerged_graph_after_segment_reset():
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import make_stream, add_frame
    s = make_stream(); points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]: add_frame(s, t, x, points)
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.5, preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts})
    node.publish_preview_once()
    s.odom(odom(1400, status='lost'), now=1.4)
    add_frame(s, 1500, 5, points, segment='b')
    node.publish_preview_once()
    pubs = {p.topic: p for p in node.publishers}
    status = json.loads(pubs['/status'].messages[-1])
    assert status['segment_id'] == 'b' and status['candidate_count'] == 0
    assert len(s.store.segments) == 2


def test_geometry_worker_discards_result_if_tracking_is_lost_during_optimization(monkeypatch):
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import make_stream, add_frame
    from plugins.pointcloud_slam_loop_search import ExperimentalLoopSearch
    s = make_stream(); points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    add_frame(s, 1000, 0, points)
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts})
    original = ExperimentalLoopSearch.update
    def lose_during_update(self, *args):
        result = original(self, *args)
        node._odom(NS(data=json.dumps(odom(1100, status='lost'))))
        return result
    monkeypatch.setattr(ExperimentalLoopSearch, 'update', lose_during_update)
    node.publish_preview_once()
    pubs = {p.topic: p for p in node.publishers}
    assert not pubs['/preview'].messages
    status = json.loads(pubs['/status'].messages[-1])
    assert status['state'] == 'lost' and 'map_pose' not in status
    assert node._loop_search is None
    assert json.loads(pubs['/preview/pose'].messages[-1])['pose'] is None
    assert json.loads(pubs['/preview/map_to_odom'].messages[-1])['transform'] is None


def test_geometry_worker_can_publish_fresh_snapshot_while_new_frame_arrives(monkeypatch):
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import add_frame
    from plugins.pointcloud_slam_loop_search import ExperimentalLoopSearch
    s = stream(max_cloud_points=400, max_frames=8, max_points=4000,
               max_measurement_age_ms=500, max_future_skew_ms=20)
    points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        s.odom(odom(t, x), now=t/1000, measurement_now_ns=t*1_000_000)
        s.cloud(t*1_000_000, 'camera', points, now=t/1000, measurement_now_ns=t*1_000_000)
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.4, measurement_clock_ns=lambda: 1_400_000_000,
        preview={'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
                 'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts})
    original = ExperimentalLoopSearch.update
    def append_during_update(self, *args):
        result = original(self, *args)
        s.odom(odom(1400, .6), now=1.4, measurement_now_ns=1_400_000_000)
        s.cloud(1_400_000_000, 'camera', points, now=1.4, measurement_now_ns=1_400_000_000)
        return result
    monkeypatch.setattr(ExperimentalLoopSearch, 'update', append_during_update)
    node.publish_preview_once()
    pubs = {p.topic: p for p in node.publishers}
    assert pubs['/preview'].messages
    assert json.loads(pubs['/status'].messages[-1])['stamp_ns'] == '1300000000'


def test_geometry_worker_error_clears_published_pose_and_correction(monkeypatch):
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from vision_stubs import _wait_until
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import make_stream, add_frame
    from plugins.pointcloud_slam_loop_search import ExperimentalLoopSearch
    s = make_stream(); points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    add_frame(s, 1000, 0, points)
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': .1, 'backend': 'geometry_experimental', 'geometry': opts})
    monkeypatch.setattr(ExperimentalLoopSearch, 'update', lambda *args: (_ for _ in ()).throw(ValueError('backend failure')))
    pubs = {p.topic: p for p in node.publishers}
    node.start_preview()
    try:
        assert _wait_until(lambda: pubs['/preview/pose'].messages)
    finally:
        node.stop_preview()
    poses = [json.loads(message) for message in pubs['/preview/pose'].messages]
    corrections = [json.loads(message) for message in pubs['/preview/map_to_odom'].messages]
    assert any(p['state'] == 'error' and p['pose'] is None for p in poses)
    assert any(c['state'] == 'error' and c['transform'] is None for c in corrections)
    assert poses[-1]['state'] == corrections[-1]['state'] == 'stopped'


def test_geometry_stop_clears_pose_without_stopping_sources():
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', stream(), clock=lambda: 1., preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 10,
        'interval_s': .1, 'backend': 'geometry_experimental', 'geometry': opts})
    node.start_preview(); node.stop_preview()
    pubs = {p.topic: p for p in node.publishers}
    assert json.loads(pubs['/preview/pose'].messages[-1])['state'] == 'stopped'
    assert json.loads(pubs['/preview/map_to_odom'].messages[-1])['transform'] is None
    assert {sub.topic for sub in node.subscriptions} == {'/o', '/c'}


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


def test_experimental_relocalization_rebuilds_both_segment_histories():
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_segment_merge import make_two_segments
    s = make_two_segments()
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .1}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.4, preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts,
        'relocalization': True})
    pubs = {p.topic: p for p in node.publishers}
    pubs['/preview'].subscription_count = 1
    node.publish_preview_once()
    status = json.loads(pubs['/status'].messages[-1])
    assert status['relocalization'] == 'unvalidated_hypothesis'
    assert status['map_segments'] == [s.store.segments[0].segment_id, 'b']
    assert status['accepted_loops'] == 0
    assert pubs['/preview'].messages
    correction = json.loads(pubs['/preview/map_to_odom'].messages[-1])
    np.testing.assert_allclose(np.asarray(correction['transform']) @ s.graph()[-1]['poses'][-1],
                               np.asarray(status['map_pose']), atol=1e-6)


def test_lost_during_relocalization_clears_hypothesis_and_outputs(monkeypatch):
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_segment_merge import make_two_segments
    import plugins.pointcloud_slam_relocalization as relocation
    s = make_two_segments()
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .1}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.4, preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts,
        'relocalization': True})
    original = relocation.relocalize_pair
    def lose(*args, **kwargs):
        result = original(*args, **kwargs)
        node._odom(NS(data=json.dumps(odom(1500, status='lost'))))
        return result
    monkeypatch.setattr(relocation, 'relocalize_pair', lose)
    pubs = {p.topic: p for p in node.publishers}
    node.publish_preview_once()
    status = json.loads(pubs['/status'].messages[-1])
    assert status['state'] == 'lost'
    assert 'map_segments' not in status and 'relocalization' not in status
    assert node._segment_links == []
    assert json.loads(pubs['/preview/map_to_odom'].messages[-1])['transform'] is None


def test_live_multigraph_keeps_links_across_three_segments_and_frames(monkeypatch):
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_segment_merge import make_two_segments
    from test_pointcloud_slam_incremental_backend import add_frame
    import plugins.pointcloud_slam_relocalization as relocation
    s = make_two_segments()
    points = s.store.segments[-1].frames[-1].points
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .1}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.6, preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts,
        'relocalization': True})
    pubs = {p.topic: p for p in node.publishers}
    node.publish_preview_once()
    first = json.loads(pubs['/status'].messages[-1])
    assert len(first['map_segments']) == 2
    s.odom(odom(1500, status='lost'), now=1.5)
    add_frame(s, 1600, -3, points, segment='c')
    node.publish_preview_once()
    third = json.loads(pubs['/status'].messages[-1])
    assert third['map_segments'] == [segment.segment_id for segment in s.store.segments]
    assert third['relocalization'] == 'unvalidated_hypothesis'
    assert third['segment_links'] == 2
    monkeypatch.setattr(relocation, 'relocalize_pair', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must retain links')))
    node.publish_preview_once()
    assert json.loads(pubs['/status'].messages[-1])['segment_links'] == 2


def test_submap_loop_constraints_survive_tracking_loss():
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    from test_pointcloud_slam_incremental_backend import make_stream, add_frame
    s = make_stream(); points = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        add_frame(s, t, x, points)
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.5, preview={
        'topic': '/preview', 'status_topic': '/status', 'max_points': 2000,
        'interval_s': 1., 'backend': 'geometry_experimental', 'geometry': opts,
        'relocalization': True})
    node.publish_preview_once()
    first = s.store.segments[0].segment_id
    assert len(node._segment_loops[first]) >= 1
    s.odom(odom(1400, status='lost'), now=1.4)
    node.publish_preview_once()
    assert len(node._segment_loops[first]) >= 1
    add_frame(s, 1500, 9, points, segment='b')
    node.publish_preview_once()
    status = json.loads({p.topic: p for p in node.publishers}['/status'].messages[-1])
    assert status['map_segments'] == [first, 'b']
    assert len(node._segment_loops[first]) >= 1
