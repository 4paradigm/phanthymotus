"""Compatibility with the unmodified Core start arguments and stop deadline."""
import json
import threading
import time
from types import SimpleNamespace as NS

import pytest

from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
from plugins.pointcloud_slam_ros import SlamInputNode
from test_pointcloud_slam_plugin import config
from test_pointcloud_slam_stream import odom
from test_pointcloud_slam_ros import cloud
from vision_stubs import _FakeExecutor, _FakeNode, _wait_until


def legacy_args():
    # Deliberately misleading names and reversed canvas connection order.
    return {'action': 'start', 'input_topics': ['/odom_named_cloud', '/cloud_named_pose'],
            'input_topic': '/odom_named_cloud', 'camera_info': {}}


def native_graph(monkeypatch):
    monkeypatch.setattr(_FakeNode, 'get_topic_names_and_types', lambda self: [
        ('/odom_named_cloud', ['sensor_msgs/msg/PointCloud2', 'std_msgs/msg/String']),
        ('/cloud_named_pose', ['std_msgs/msg/String'])], raising=False)
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic: [
        NS(topic_type='sensor_msgs/msg/PointCloud2' if topic == '/odom_named_cloud' else 'std_msgs/msg/String')], raising=False)


def test_main_core_start_uses_native_publishers_and_runtime_pose_validation(monkeypatch):
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'running'
    try:
        info = p.dispatch('pointcloud_slam', {'action': 'info'})
        assert info['topic_in'] == [
            {'topic': '/odom_named_cloud', 'format': 'sensor/pointcloud2'},
            {'topic': '/cloud_named_pose', 'format': 'state/odom'}]
        assert info['input_status']['odometry'] == 'waiting'
        n = p._node
        n._receipt_clock = lambda: 1.
        n._measurement_clock_ns = lambda: 1_000_000_000
        bad = odom(1000)
        del bad['vendor']['motus_slam']['segment_id']
        n._odom(NS(data=json.dumps(bad)))
        n._cloud(cloud())
        assert not n.stream.store.segments
        n._odom(NS(data=json.dumps(odom(1000, status='lost'))))
        n._cloud(cloud())
        assert not n.stream.store.segments
        n._odom(NS(data=json.dumps(odom(1000))))
        n._cloud(cloud())
        assert len(n.stream.store.segments[0].frames) == 1
    finally:
        p.shutdown()


@pytest.mark.parametrize('extra', [
    {'input_bindings': None},
    {'odom_interfaces': {}},
    {'odom_interfaces': {'/cloud_named_pose': {'schema': 'motus.odom/1', 'slam_pose': {'available': False}}}},
    {'input_metadata': {}},
])
def test_main_core_fallback_never_discards_explicit_invalid_contracts(monkeypatch, extra):
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert p.dispatch('pointcloud_slam', {**legacy_args(), **extra})['state'] == 'error'
    assert p._node is None


def test_main_core_project_ready_reports_both_native_inputs(monkeypatch):
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin({**config(), 'persistent_history': {'reserve_bytes': 0}}, _FakeExecutor())
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'ready'
    info = p.dispatch('pointcloud_slam', {'action': 'info'})
    assert {item['topic'] for item in info['topic_in']} == {'/odom_named_cloud', '/cloud_named_pose'}
    p.shutdown()


def managed_capture(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_plugin as module
    from test_pointcloud_slam_paged_store import store
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    settings = {**config(), 'persistent_history': {'max_disk_bytes': 10_000_000, 'reserve_bytes': 0}}
    p = PointCloudSlamPlugin(settings, _FakeExecutor())
    p._project_inputs = legacy_args()
    p._topics = {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}
    p._map_name = 'Hall'
    p._session_settings = settings
    s = p._stream()
    s.store = store(tmp_path / 'sessions')
    n = SlamInputNode('/o', '/c', s)
    p._executor.add_node(n)
    p._node = n
    n._receipt_clock = lambda: 1.
    n._measurement_clock_ns = lambda: 1_000_000_000
    n._odom(NS(data=json.dumps(odom(1000))))
    n._cloud(cloud())
    return p, n


def call(p, action, **args):
    return p.dispatch('pointcloud_slam', {'action': action, **args})


def test_project_stop_freezes_capture_before_slow_teardown_and_save(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_library as library
    p, n = managed_capture(tmp_path, monkeypatch)
    release, entered = threading.Event(), threading.Event()
    original = library.save_revision
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(library, 'save_revision', slow)
    stopped = []
    t = threading.Thread(target=lambda: stopped.append(call(p, 'stop')))
    t.start()
    try:
        t.join(.5)
        assert not t.is_alive(), 'project stop must acknowledge before persistence finishes'
        assert stopped[0]['state'] == 'idle'
        assert stopped[0]['capture_stopped'] is True
        assert stopped[0]['save_state'] == 'saving'
        assert entered.wait(1)
        n._odom(NS(data=json.dumps(odom(1000, x=5, segment='new'))))
        n._cloud(cloud())
        assert len(n.stream.store.records) == 1
        info = call(p, 'info')
        assert info['state'] == 'idle' and info['save_state'] == 'saving'
        assert call(p, 'stop')['save_state'] == 'saving'
        assert call(p, 'start', **{k: v for k, v in legacy_args().items() if k != 'action'})['state'] == 'error'
        assert call(p, 'config', child_from_cloud=json.dumps(config()['child_from_cloud']))['state'] == 'error'
    finally:
        release.set()
        t.join(2)
        p.shutdown()
    assert call(p, 'info')['save_state'] == 'saved'
    assert len(call(p, 'list_maps')['maps']) == 1
    assert call(p, 'stop')['save_state'] == 'saved'


def test_project_stop_ack_does_not_join_slow_preview_disposal(tmp_path, monkeypatch):
    p, n = managed_capture(tmp_path, monkeypatch)
    release, entered = threading.Event(), threading.Event()
    original = n.stop_preview
    def slow():
        entered.set()
        assert release.wait(5)
        original()
    monkeypatch.setattr(n, 'stop_preview', slow)
    stopped = []
    t = threading.Thread(target=lambda: stopped.append(call(p, 'stop')))
    t.start()
    try:
        assert entered.wait(1)
        t.join(.5)
        assert not t.is_alive(), 'foreground stop must not join preview workers'
        n._odom(NS(data=json.dumps(odom(1000, x=5, segment='new'))))
        n._cloud(cloud())
        assert len(n.stream.store.records) == 1
        assert stopped[0]['save_state'] == 'saving'
    finally:
        release.set()
        t.join(2)
        p.shutdown()


def test_project_save_failure_remains_retryable_and_shutdown_reports_failure(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_library as library
    p, n = managed_capture(tmp_path, monkeypatch)
    directory = n.stream.store.directory
    original = library.save_revision
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(library, 'save_revision', fail)
    assert call(p, 'stop')['state'] == 'idle'
    assert _wait_until(lambda: call(p, 'info').get('save_state') == 'error')
    assert 'disk full' in call(p, 'info')['save_result']['message']
    assert directory.is_dir() and p._pending is n
    with pytest.raises(RuntimeError, match='disk full'):
        p.shutdown()
    monkeypatch.setattr(library, 'save_revision', original)
    assert call(p, 'stop')['save_state'] == 'saving'
    p.shutdown()
    assert call(p, 'info')['save_state'] == 'saved'
    assert not directory.exists()
    assert len(call(p, 'list_maps')['maps']) == 1


def test_shutdown_waits_for_project_save(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_library as library
    p, _ = managed_capture(tmp_path, monkeypatch)
    release, entered, closed = threading.Event(), threading.Event(), threading.Event()
    original = library.save_revision
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(library, 'save_revision', slow)
    caller = threading.Thread(target=lambda: call(p, 'stop'))
    caller.start()
    shutdown = threading.Thread(target=lambda: (p.shutdown(), closed.set()))
    try:
        assert entered.wait(1)
        shutdown.start()
        assert not closed.wait(.1)
        release.set()
        shutdown.join(2)
        assert closed.is_set()
        assert call(p, 'info')['save_state'] == 'saved'
    finally:
        release.set()
        caller.join(2)
        if shutdown.ident is not None:
            shutdown.join(2)


def test_idle_configuration_error_prevents_following_start_until_valid_config(monkeypatch):
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert call(p, 'config', child_from_cloud='invalid')['state'] == 'error'
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'error'
    assert call(p, 'config', child_from_cloud=json.dumps(config()['child_from_cloud']))['state'] == 'configured'
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'running'
    p.shutdown()


def test_ready_project_rejects_mapping_after_invalid_auto_configuration(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_plugin as module
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    native_graph(monkeypatch)
    settings = {**config(), 'preview_backend': 'geometry_experimental', 'preview_geometry': {},
                'persistent_history': {'max_disk_bytes': 10_000_000, 'reserve_bytes': 0}}
    p = PointCloudSlamPlugin(settings, _FakeExecutor())
    original = SlamInputNode.__init__
    monkeypatch.setattr(SlamInputNode, '__init__', lambda self, *args, **kwargs:
        original(self, *args, **{**kwargs, 'preview': None}))
    try:
        assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'ready'
        assert call(p, 'config', child_from_cloud='invalid')['state'] == 'error'
        assert call(p, 'start_mapping', map_name='Hall')['state'] == 'error'
        assert p._node is None
        assert call(p, 'config', child_from_cloud=json.dumps(settings['child_from_cloud']))['state'] == 'configured'
        assert call(p, 'start_mapping', map_name='Hall')['state'] == 'running'
    finally:
        p.shutdown()


def test_running_auto_config_rejection_does_not_poison_next_project_start(monkeypatch):
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'running'
    assert call(p, 'config', child_from_cloud='invalid')['state'] == 'error'
    p.stop()
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'running'
    p.shutdown()


def test_project_stop_cancels_node_whose_constructor_has_not_returned(tmp_path, monkeypatch):
    import plugins.pointcloud_slam_plugin as module
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    settings = {**config(), 'preview_backend': 'geometry_experimental', 'preview_geometry': {},
                'persistent_history': {'max_disk_bytes': 10_000_000, 'reserve_bytes': 0}}
    p = PointCloudSlamPlugin(settings, _FakeExecutor())
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert p.dispatch('pointcloud_slam', args)['state'] == 'ready'
    entered, release = threading.Event(), threading.Event()
    original = SlamInputNode.__init__
    nodes = []
    def delayed(self, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        # Keep real capture and store behavior; this race does not need geometry computation.
        original(self, *args, **{**kwargs, 'preview': None})
        nodes.append(self)
    monkeypatch.setattr(SlamInputNode, '__init__', delayed)
    results = []
    starting = threading.Thread(target=lambda: results.append(call(p, 'start_mapping', map_name='Late')))
    starting.start()
    try:
        assert entered.wait(1)
        before = time.monotonic()
        stopped = call(p, 'stop')
        assert time.monotonic() - before < .5
        assert stopped['state'] == 'idle' and stopped['capture_stopped'] is True
        assert call(p, 'start', **{k: v for k, v in args.items() if k != 'action'})['state'] == 'error'
        release.set()
        starting.join(2)
        p.shutdown()
        assert results[0]['state'] != 'running'
        assert not p._executor.nodes
        n = nodes[0]
        n._measurement_clock_ns = lambda: 1_000_000_000
        n._odom(NS(data=json.dumps(odom(1000))))
        n._cloud(cloud())
        assert n.stream.store.records == []
    finally:
        release.set()
        starting.join(2)
        p.shutdown()


def test_project_stop_joins_existing_view_worker_before_disposal(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_paged_archive as archive
    p, _ = managed_capture(tmp_path, monkeypatch)
    saved = call(p, 'stop_mapping')
    entered, release = threading.Event(), threading.Event()
    original = archive.load_paged_archive
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(archive, 'load_paged_archive', slow)
    assert call(p, 'select_map', map_id=saved['map_id'], background=True, request_id='view')['state'] == 'working'
    try:
        assert entered.wait(1)
        before = time.monotonic()
        assert call(p, 'stop')['state'] == 'idle'
        assert time.monotonic() - before < .5
        assert call(p, 'info')['save_state'] == 'saving'
    finally:
        release.set()
        p.shutdown()
    assert not p._executor.nodes and p._view is None


def test_project_stop_during_manual_background_save_finishes_project_teardown(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_library as library
    p, _ = managed_capture(tmp_path, monkeypatch)
    release, entered = threading.Event(), threading.Event()
    original = library.save_revision
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(library, 'save_revision', slow)
    assert call(p, 'stop_mapping', background=True, request_id='manual')['state'] == 'working'
    try:
        assert entered.wait(1)
        assert call(p, 'stop_mapping', background=True, request_id='manual')['state'] == 'working'
        assert call(p, 'stop')['state'] == 'idle'
        release.set()
        p._worker.join(2)
        assert p._project_inputs is None, 'project stop must finish teardown after the existing save'
        assert call(p, 'info')['save_state'] == 'saved'
    finally:
        release.set()
        p.shutdown()


def test_legacy_live_preview_uses_existing_core_mapping_wire_and_truthful_status(monkeypatch):
    import struct
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin({**config(), 'preview_enabled': True, 'preview_backend': 'raw'}, _FakeExecutor())
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'running'
    n = p._node
    try:
        info = call(p, 'info')
        output = next(o for o in info['topic_out'] if o['topic'] == '/odom_named_cloud/slam_odom_preview')
        assert output['format'] == 'sensor/mapping'
        assert 'snapshot' in output['desc'] and 'last captured' in output['desc']
        assert info['input_compatibility'] == {'mode': 'legacy_core', 'role_source': 'native_ros_publishers',
            'producer_metadata_provided': False, 'validation_source': 'explicit_configuration'}
        n._receipt_clock = lambda: 1.
        n._measurement_clock_ns = lambda: 1_000_000_000
        n._odom(NS(data=json.dumps(odom(1000, x=2))))
        n._cloud(cloud())
        n.publish_preview_once()
        packet = next(pub for pub in n.publishers if pub.topic == output['topic']).messages[-1]
        assert struct.unpack_from('<fffBI', packet) == pytest.approx((2., 0., 0., 3, 1))
        assert len(packet) == 29
        status = json.loads(next(pub for pub in n.publishers if pub.topic.endswith('_status')).messages[-1])
        assert status['accepted_loops'] == 0
        assert status['input_validation']['extrinsic_calibrated'] is True
    finally:
        p.shutdown()


def test_legacy_saved_view_exposes_mapping_wire_without_changing_coordinates(tmp_path, monkeypatch):
    import struct
    p, _ = managed_capture(tmp_path, monkeypatch)
    saved = call(p, 'stop_mapping')
    native_graph(monkeypatch)
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'ready'
    result = call(p, 'select_map', map_id=saved['map_id'])
    assert result['state'] == 'map_selected'
    try:
        assert result['topic_out'][0]['format'] == 'sensor/mapping'
        p._view.publish_preview_once()
        packet = next(pub for pub in p._view.publishers if pub.topic == result['topic_out'][0]['topic']).messages[-1]
        assert struct.unpack_from('<fffBI', packet) == pytest.approx((0., 0., 0., 3, 1))
        assert len(packet) == 29
    finally:
        p.shutdown()


def test_nonpersistent_project_stop_acknowledges_before_slow_preview_teardown(monkeypatch):
    native_graph(monkeypatch)
    p = PointCloudSlamPlugin({**config(), 'preview_enabled': True}, _FakeExecutor())
    assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'running'
    n = p._node
    entered, release = threading.Event(), threading.Event()
    original = n.stop_preview
    def slow():
        entered.set()
        assert release.wait(5)
        original()
    monkeypatch.setattr(n, 'stop_preview', slow)
    results = []
    stopping = threading.Thread(target=lambda: results.append(call(p, 'stop')))
    stopping.start()
    try:
        assert entered.wait(1)
        stopping.join(.5)
        assert not stopping.is_alive(), 'all project stop acknowledgements must meet the Core deadline'
        assert results[0]['state'] == 'idle' and results[0]['capture_stopped'] is True
        assert results[0]['save_state'] == 'idle'
        assert call(p, 'info')['state'] == 'idle'
        assert p.dispatch('pointcloud_slam', legacy_args())['state'] == 'error'
    finally:
        release.set()
        stopping.join(2)
        p.shutdown()
    assert not p._executor.nodes and n.destroyed


def test_shutdown_waits_for_synchronous_manual_save_without_lock_inversion(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_library as library
    p, _ = managed_capture(tmp_path, monkeypatch)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = library.save_revision
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(library, 'save_revision', slow)
    results = []
    manual = threading.Thread(target=lambda: results.append(call(p, 'stop_mapping')), daemon=True)
    shutdown = threading.Thread(target=lambda: (p.shutdown(), closed.set()), daemon=True)
    manual.start()
    assert entered.wait(1)
    shutdown.start()
    assert not closed.wait(.05)
    release.set()
    manual.join(1)
    shutdown.join(1)
    assert not manual.is_alive() and not shutdown.is_alive(), 'shutdown and save must not invert their locks'
    assert results[0]['state'] == 'saved' and closed.is_set()


def saved_after_project_stop(tmp_path, monkeypatch):
    p, _ = managed_capture(tmp_path, monkeypatch)
    assert call(p, 'stop')['state'] == 'idle'
    p._worker.join(2)
    assert not p._worker.is_alive()
    return p, p._last_stop


def test_saved_view_opened_after_project_stop_stays_visible_in_info(tmp_path, monkeypatch):
    p, saved = saved_after_project_stop(tmp_path, monkeypatch)
    try:
        selected = call(p, 'select_map', map_id=saved['map_id'])
        assert selected['state'] == 'map_selected'
        assert len(selected['topic_out']) == 1
        info = call(p, 'info')
        assert info['state'] == 'idle' and info['capture_stopped'] is True
        assert info['topic_out'] == selected['topic_out']
    finally:
        p.shutdown()


def test_repeated_project_stop_disposes_view_opened_after_previous_stop(tmp_path, monkeypatch):
    p, saved = saved_after_project_stop(tmp_path, monkeypatch)
    assert call(p, 'select_map', map_id=saved['map_id'])['state'] == 'map_selected'
    view = p._view
    try:
        before = time.monotonic()
        assert call(p, 'stop')['state'] == 'idle'
        assert time.monotonic() - before < .5
        p._worker.join(2)
        assert p._view is None and not p._executor.nodes
        assert view.destroyed and not view._preview_worker.is_alive()
        assert call(p, 'info')['topic_out'] == []
    finally:
        p.shutdown()


def test_repeated_project_stop_waits_for_late_view_in_background(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_paged_archive as archive
    p, saved = saved_after_project_stop(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = archive.load_paged_archive
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(archive, 'load_paged_archive', slow)
    assert call(p, 'select_map', map_id=saved['map_id'], background=True, request_id='late_view')['state'] == 'working'
    try:
        assert entered.wait(1)
        before = time.monotonic()
        assert call(p, 'stop')['state'] == 'idle'
        assert time.monotonic() - before < .5
        release.set()
        p._worker.join(2)
        assert not p._worker.is_alive()
        assert p._view is None and not p._executor.nodes
    finally:
        release.set()
        p.shutdown()


def test_fresh_plugin_views_saved_map_in_main_core_format_without_live_start(tmp_path, monkeypatch):
    p, saved = saved_after_project_stop(tmp_path, monkeypatch)
    p.shutdown()
    fresh = PointCloudSlamPlugin({'persistent_history': p._config['persistent_history']}, _FakeExecutor())
    try:
        selected = call(fresh, 'select_map', map_id=saved['map_id'])
        assert selected['state'] == 'map_selected'
        assert selected['topic_out'][0]['format'] == 'sensor/mapping'
        info = call(fresh, 'info')
        assert info['topic_out'] == selected['topic_out']
        assert info['input_compatibility']['role_source'] == 'not_resolved'
        assert info['input_compatibility']['producer_metadata_provided'] is False
    finally:
        fresh.shutdown()


def test_repeated_project_stop_disposes_view_from_concurrent_synchronous_selection(tmp_path, monkeypatch):
    from plugins import pointcloud_slam_paged_archive as archive
    p, saved = saved_after_project_stop(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = archive.load_paged_archive
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(archive, 'load_paged_archive', slow)
    results = []
    selecting = threading.Thread(target=lambda: results.append(call(p, 'select_map', map_id=saved['map_id'])))
    selecting.start()
    try:
        assert entered.wait(1)
        before = time.monotonic()
        assert call(p, 'stop')['state'] == 'idle'
        assert time.monotonic() - before < .5
        release.set()
        selecting.join(2)
        p._worker.join(2)
        assert not selecting.is_alive() and not p._worker.is_alive()
        assert results[0]['state'] == 'map_selected'
        assert p._view is None and not p._executor.nodes
        assert call(p, 'info')['topic_out'] == []
    finally:
        release.set()
        selecting.join(2)
        p.shutdown()


@pytest.mark.parametrize('stop_mode', ['dispatch', 'stop', 'shutdown'])
def test_stop_waits_for_selection_already_admitted_before_dispatch(tmp_path, monkeypatch, stop_mode):
    p, saved = saved_after_project_stop(tmp_path, monkeypatch)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = p._dispatch
    def pause_after_admission(action, args):
        if action == 'select_map' and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(action, args)
    monkeypatch.setattr(p, '_dispatch', pause_after_admission)
    selections, stops = [], []
    selecting = threading.Thread(target=lambda: selections.append(call(p, 'select_map', map_id=saved['map_id'])))
    def stop():
        result = (call(p, 'stop') if stop_mode == 'dispatch' else
                  p.stop() if stop_mode == 'stop' else p.shutdown())
        stops.append(result)
        stopped.set()
    stopping = threading.Thread(target=stop)
    selecting.start()
    try:
        assert entered.wait(1)
        stopping.start()
        if stop_mode == 'dispatch':
            assert stopped.wait(.5), 'project stop must promptly acknowledge capture freeze'
            assert stops[0]['state'] == 'idle'
        else:
            assert not stopped.wait(.1), 'blocking teardown must wait for already admitted selection'
        new_selection = call(p, 'select_map', map_id=saved['map_id'])
        release.set()
        selecting.join(2)
        stopping.join(2)
        p._worker.join(2)
        assert not selecting.is_alive() and not stopping.is_alive() and not p._worker.is_alive()
        assert selections[0]['state'] == 'map_selected'
        assert p._view is None and not p._executor.nodes
        assert new_selection['state'] == 'error'
    finally:
        release.set()
        selecting.join(2)
        if stopping.ident is not None:
            stopping.join(2)
        p.shutdown()
