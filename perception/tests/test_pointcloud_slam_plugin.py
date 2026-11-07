"""The experimental card binds both inputs and stops only its own node."""
import numpy as np

from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin, TOOLS
from vision_stubs import _FakeExecutor


def config():
    return {'cloud_frame': 'camera', 'child_frame': 'body', 'clock_id': 'clock',
            'clock_relation_verified': True, 'extrinsic_calibrated': True,
            'child_from_cloud': np.eye(4).tolist(), 'max_gap_ms': 200,
            'max_age_s': 1, 'max_pending': 2, 'max_cloud_points': 100,
            'max_frames': 10, 'max_points': 1000, 'max_measurement_age_ms': 300,
            'max_future_skew_ms': 20}


def test_canvas_configuration_supplies_required_start_settings():
    import json
    props = TOOLS[0]['configSchema']['properties']
    required = set(config())
    assert required <= set(props)
    assert all(props[key]['scope'] == 'instance' for key in required)
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin({}, executor)
    settings = {**config(), 'child_from_cloud': json.dumps(config()['child_from_cloud'])}
    assert plugin.dispatch('pointcloud_slam', {'action': 'config', **settings})['state'] == 'configured'
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    assert plugin.dispatch('pointcloud_slam', {'action': 'config', 'max_frames': 5})['state'] == 'error'
    plugin.dispatch('pointcloud_slam', {'action': 'stop'})
    plugin.shutdown()


def test_canvas_configuration_rejects_malformed_matrix_without_changes():
    plugin = PointCloudSlamPlugin({}, _FakeExecutor())
    result = plugin.dispatch('pointcloud_slam', {'action': 'config', 'child_from_cloud': 'not json'})
    assert result['state'] == 'error'
    assert 'child_from_cloud' not in plugin._config


def test_canvas_configuration_rejects_invalid_partial_limits():
    plugin = PointCloudSlamPlugin({}, _FakeExecutor())
    result = plugin.dispatch('pointcloud_slam', {'action': 'config', 'max_frames': -1})
    assert result['state'] == 'error'
    assert 'max_frames' not in plugin._config


def test_geometry_preview_options_are_received_from_canvas():
    import json
    from test_pointcloud_slam_geometry_gate import thresholds
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    props = TOOLS[0]['configSchema']['properties']
    assert 'geometry_experimental' in props['preview_backend']['enum']
    assert 'preview_geometry' in props
    plugin = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert plugin.dispatch('pointcloud_slam', {'action': 'config',
        'preview_geometry': json.dumps(opts)})['state'] == 'configured'
    assert plugin._config['preview_geometry'] == opts


def test_geometry_preview_starts_and_stops_with_explicit_configuration():
    import pytest
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    from test_pointcloud_slam_geometry_gate import thresholds
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .8}
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin({**config(), 'preview_enabled': True,
        'preview_backend': 'geometry_experimental', 'preview_geometry': opts}, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    node = executor.nodes[0]
    plugin.shutdown()
    assert not executor.nodes and not node._preview_worker.is_alive()


def test_start_uses_formats_regardless_of_order_and_stops_only_own_node():
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin(config(), executor)
    result = plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/c', '/o'],
        'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
        'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
            'available': True, 'profile': 'motus_slam/1'}}}})
    assert result['state'] == 'running'
    node = executor.nodes[0]
    assert {s.topic for s in node.subscriptions} == {'/c', '/o'}
    info = plugin.dispatch('pointcloud_slam', {'action': 'info'})
    assert info['topic_out'] == []
    assert info['input_status']['odometry'] == 'waiting'
    plugin.dispatch('pointcloud_slam', {'action': 'stop'})
    plugin.shutdown()
    assert not executor.nodes and node.destroyed


def test_bundle_shutdown_retires_slam_node_and_preview_worker():
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin({**config(), 'preview_enabled': True}, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    node = executor.nodes[0]
    plugin.shutdown()
    assert not executor.nodes and node.destroyed and not node._preview_worker.is_alive()


def test_start_fails_closed_for_unverified_clock_or_missing_bindings():
    cfg = config()
    cfg['clock_relation_verified'] = False
    plugin = PointCloudSlamPlugin(cfg, _FakeExecutor())
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'error'
    cfg['clock_relation_verified'] = True
    plugin = PointCloudSlamPlugin(cfg, _FakeExecutor())
    assert plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/o', '/c']})['state'] == 'error'


def test_malformed_input_topics_are_rejected_without_exception():
    plugin = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/o', {'invalid': True}],
        'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}})['state'] == 'error'


def test_start_requires_measurement_age_policy():
    cfg = config()
    del cfg['max_measurement_age_ms']
    plugin = PointCloudSlamPlugin(cfg, _FakeExecutor())
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'error'
    cfg['max_measurement_age_ms'] = None
    cfg['max_future_skew_ms'] = None
    plugin = PointCloudSlamPlugin(cfg, _FakeExecutor())
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'error'


def test_start_rejects_velocity_only_odometry_even_with_matching_format():
    plugin = PointCloudSlamPlugin(config(), _FakeExecutor())
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1',
                'slam_pose': {'available': False}}}}
    result = plugin.dispatch('pointcloud_slam', args)
    assert result['state'] == 'error' and 'pose' in result['message']


def test_bundle_exposes_experimental_card_only_when_enabled():
    from main import PerceptionBundle
    executor = _FakeExecutor()
    assert not PerceptionBundle({'plugins': {}}, executor).owns('pointcloud_slam')
    assert PerceptionBundle({'plugins': {'pointcloud_slam': {'enabled': True}}},
                            executor).owns('pointcloud_slam')


def test_missing_optional_slam_module_does_not_stop_perception_bundle(monkeypatch):
    import builtins
    from main import PerceptionBundle
    original = builtins.__import__
    def failing(name, *args, **kwargs):
        if name == 'plugins.pointcloud_slam_plugin':
            raise ImportError('missing optional slam dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', failing)
    bundle = PerceptionBundle({'plugins': {'pointcloud_slam': {'enabled': True}}}, _FakeExecutor())
    assert not bundle.owns('pointcloud_slam')


def test_opt_in_preview_runs_worker_publishes_and_stops_only_its_node():
    import json
    from types import SimpleNamespace as NS
    from test_pointcloud_slam_stream import odom
    from test_pointcloud_slam_ros import cloud
    from vision_stubs import _wait_until
    cfg = {**config(), 'preview_enabled': True, 'preview_backend': 'raw',
           'preview_interval_s': .1, 'preview_max_points': 10}
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin(cfg, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    node = executor.nodes[0]
    node._clock = lambda: 1.
    node._measurement_clock_ns = lambda: 1_000_000_000
    callbacks = {s.topic: s.callback for s in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    info = plugin.dispatch('pointcloud_slam', {'action': 'info'})
    assert any(o['topic'] == '/c/slam_odom_preview' and o['format'] == 'sensor/slam-preview'
               and o['metadata']['map_view']['scope'] == 'live' for o in info['topic_out'])
    pub = next(p for p in node.publishers if p.topic == '/c/slam_odom_preview')
    assert _wait_until(lambda: len(pub.messages) > 0)
    plugin.dispatch('pointcloud_slam', {'action': 'stop'})
    plugin.shutdown()
    assert not executor.nodes and not node._preview_worker.is_alive()


def test_preview_start_failure_removes_node_from_executor(monkeypatch):
    from plugins.pointcloud_slam_ros import SlamInputNode
    monkeypatch.setattr(SlamInputNode, 'start_preview',
                        lambda self: (_ for _ in ()).throw(RuntimeError('worker failed')))
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin({**config(), 'preview_enabled': True}, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'error'
    assert not executor.nodes


def test_gtsam_preview_fails_closed_if_backend_is_missing(monkeypatch):
    import builtins
    original = builtins.__import__
    def failing(name, *args, **kwargs):
        if name == 'gtsam':
            raise ImportError('gtsam unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', failing)
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin({**config(), 'preview_enabled': True,
             'preview_backend': 'gtsam', 'preview_odom_sigmas': [.1] * 6}, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    result = plugin.dispatch('pointcloud_slam', args)
    assert result['state'] == 'error' and 'gtsam' in result['message']
    assert not executor.nodes


def test_relocalization_setting_is_explicit_and_forwarded_to_node():
    import pytest
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    from test_pointcloud_slam_geometry_gate import thresholds
    opts = {'descriptor_voxel_m': .08, 'min_separation_s': .25,
        'candidate_separation_s': .01, 'max_per_frame': 2, 'max_candidates': 4,
        'thresholds': thresholds(), 'odom_sigmas': [.1]*6, 'loop_sigmas': [.05]*6,
        'huber_k': 1., 'voxel_m': .04, 'max_distance_m': .1}
    assert TOOLS[0]['configSchema']['properties']['preview_relocalization']['default'] is False
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin({**config(), 'preview_enabled': True,
        'preview_backend': 'geometry_experimental', 'preview_geometry': opts,
        'preview_relocalization': True}, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    assert executor.nodes[0]._preview['relocalization'] is True
    plugin.shutdown()


def test_map_save_and_explicit_restore_require_new_segment(tmp_path, monkeypatch):
    from test_pointcloud_slam_stream import odom
    import plugins.pointcloud_slam_plugin as module
    monkeypatch.setattr(module, '_ARCHIVE_DIR', str(tmp_path))
    executor = _FakeExecutor()
    cfg = {**config(), 'archive_name': 'example'}
    plugin = PointCloudSlamPlugin(cfg, executor)
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    node = executor.nodes[0]
    node.stream.odom(odom(1000), now=1., measurement_now_ns=1_000_000_000)
    node.stream.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1., measurement_now_ns=1_000_000_000)
    from plugins.pointcloud_slam_archive import save_archive
    save_archive(tmp_path / 'example.json', node.stream.store, [], {}, cfg)
    plugin.stop()
    restored = PointCloudSlamPlugin({**cfg, 'restore_archive': True}, executor)
    assert restored.dispatch('pointcloud_slam', args)['state'] == 'running'
    node = executor.nodes[0]
    assert len(node.stream.store.segments) == 1
    assert node.stream.store.requires_new_segment
    assert node.stream.status(now=1.)['odometry'] != 'tracking'
    node.stream.odom(odom(1100), now=1.1, measurement_now_ns=1_100_000_000)
    node.stream.cloud(1_100_000_000, 'camera', [[1, 2, 3]], now=1.1, measurement_now_ns=1_100_000_000)
    assert len(node.stream.store.segments) == 1
    assert 'new segment' in node.stream.last_error
    node.stream.odom(odom(1200, segment='b'), now=1.2, measurement_now_ns=1_200_000_000)
    node.stream.cloud(1_200_000_000, 'camera', [[1, 2, 3]], now=1.2, measurement_now_ns=1_200_000_000)
    assert [s.segment_id for s in node.stream.store.segments] == ['a', 'b']
    restored.stop()


def test_map_archive_name_rejects_path_traversal():
    plugin = PointCloudSlamPlugin(config(), _FakeExecutor())
    assert plugin.dispatch('pointcloud_slam', {'action': 'config', 'archive_name': '../bad'})['state'] == 'error'


def test_deployed_trial_profile_waits_for_data_without_claiming_calibration():
    cfg = {**config(), 'input_policy': 'experimental',
           'clock_relation_verified': False, 'extrinsic_calibrated': False}
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin(cfg, executor)
    props = plugin.get_tools()[0]['configSchema']['properties']
    assert set(props) == {'child_from_cloud', 'preview_enabled', 'archive_name', 'restore_archive'}
    plugin.dispatch('pointcloud_slam', {'action': 'config', 'max_gap_ms': 9999,
                                      'clock_relation_verified': True})
    assert plugin._config['max_gap_ms'] == 200
    assert plugin._config['clock_relation_verified'] is False
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'},
            'odom_interfaces': {'/o': {'schema': 'motus.odom/1', 'slam_pose': {
                'available': True, 'profile': 'motus_slam/1'}}}}
    assert plugin.dispatch('pointcloud_slam', args)['state'] == 'running'
    info = plugin.dispatch('pointcloud_slam', {'action': 'info'})
    assert info['input_status']['odometry'] == 'waiting'
    assert info['input_validation']['policy'] == 'experimental'
    assert info['input_validation']['clock_relation_verified'] is False
    assert info['input_validation']['extrinsic_calibrated'] is False
    import json
    from types import SimpleNamespace as NS
    from test_pointcloud_slam_stream import odom
    from test_pointcloud_slam_ros import cloud
    node = executor.nodes[0]
    node._measurement_clock_ns = lambda: 1_000_000_000
    node._odom(NS(data=json.dumps(odom(1000))))
    node._cloud(cloud())
    assert len(node.stream.store.segments[0].frames) == 1
    node._odom(NS(data=json.dumps(odom(1010, status='lost'))))
    assert node.status()['odometry'] == 'lost'
    plugin.stop()


def test_managed_paged_history_continues_and_restores_beyond_window(tmp_path, monkeypatch):
    import json
    import pytest
    pytest.importorskip('gtsam'); pytest.importorskip('open3d')
    from types import SimpleNamespace as NS
    import plugins.pointcloud_slam_plugin as module
    import plugins.pointcloud_slam_rolling as rolling
    from plugins.pointcloud_slam_ros import SlamInputNode
    from test_pointcloud_slam_rolling import FakeSearch
    from test_pointcloud_slam_stream import odom
    from test_pointcloud_slam_ros import cloud
    from test_pointcloud_slam_geometry_gate import thresholds
    monkeypatch.setattr(module,'_ARCHIVE_DIR',str(tmp_path))
    monkeypatch.setattr(SlamInputNode,'start_preview',lambda self:None)
    monkeypatch.setattr(rolling,'ExperimentalLoopSearch',FakeSearch)
    settings={**config(),'max_frames':3,'archive_name':'paged','preview_enabled':True,'map_plane':'xy',
        'preview_backend':'geometry_experimental','persistent_history':{
            'max_disk_bytes':10_000_000,'reserve_bytes':0,'min_translation_m':.1,'min_rotation_deg':10},
        'preview_geometry':{'descriptor_voxel_m':.08,'min_separation_s':.25,
            'candidate_separation_s':.01,'max_per_frame':2,'max_candidates':4,
            'thresholds':thresholds(),'odom_sigmas':[.1]*6,'loop_sigmas':[.05]*6,
            'huber_k':1.,'voxel_m':.04,'max_distance_m':.8}}
    args={'action':'start','input_topics':['/o','/c'],
          'input_bindings':{'/o':'state/odom','/c':'sensor/pointcloud2'},
          'odom_interfaces':{'/o':{'schema':'motus.odom/1','slam_pose':{'available':True,'profile':'motus_slam/1'}}}}
    p=PointCloudSlamPlugin(settings,_FakeExecutor())
    def call(action, **kw):
        return p.dispatch('pointcloud_slam', {'action': action, **kw})
    ready = p.dispatch('pointcloud_slam', args)
    assert ready['state'] == 'ready'
    assert len(ready['topic_out']) == 4 and ready['topic_out'] == call('info')['topic_out']
    assert p._node is None
    assert call('start_mapping', map_name='../bad')['state'] == 'error'
    started = call('start_mapping', map_name='走廊')
    assert started['state'] == 'running'
    assert [o['topic'] for o in started['topic_out']] == [o['topic'] for o in ready['topic_out']]
    assert started['topic_out'][0]['metadata']['map_view']['name'] == '走廊'
    n=p._node;now=[1.];n._receipt_clock=lambda:now[0];n._measurement_clock_ns=lambda:int(now[0]*1e9)
    for i in range(12):
        now[0]=(1000+i*100)/1000
        n._odom(NS(data=json.dumps(odom(1000+i*100,x=i))))
        c=cloud();c.header.stamp.sec=int(now[0]);c.header.stamp.nanosec=(i%10)*100_000_000
        n._cloud(c);n.publish_preview_once()
    assert n.status()['keyframes']==12
    assert n.status()['integration']=='collecting_unverified'
    assert n._history_snapshots['a']['poses']
    # A failed automatic save can be retried without appending a second revision.
    from plugins import pointcloud_slam_library as library
    from vision_stubs import _wait_until
    with monkeypatch.context() as patch:
        def fail(*a, **kw):
            raise OSError('disk full')
        patch.setattr(library, 'save_revision', fail)
        assert call('stop_mapping', request_id='retry1')['state'] == 'error'
    assert call('info')['state'] == 'save_pending'
    assert call('start_mapping', map_name='other')['state'] == 'error'
    assert call('stop_mapping', background=True, request_id='after_disk_full')['state'] == 'working'
    assert _wait_until(lambda: call('operation_status', request_id='after_disk_full')['state'] == 'saved')
    saved = call('stop_mapping', request_id='retry1')
    assert saved['revision'] == 1 and saved['keyframes'] == 12
    assert call('start_mapping', map_name='走廊')['state'] == 'error'
    assert not list((tmp_path / 'sessions').iterdir())
    assert call('start_mapping', map_name='Second')['state'] == 'running'
    live = p._node
    # A delayed stop retry for the previous session must leave live capture alone.
    assert call('stop_mapping', request_id='retry1')['map_id'] == saved['map_id']
    assert p._node is live
    entry = {'map_id': saved['map_id'], 'revision': 1}
    assert call('select_map', map_id=saved['map_id'], background=True, request_id='view1')['state'] == 'working'
    assert _wait_until(lambda: call('operation_status', request_id='view1')['state'] == 'map_selected')
    viewed = call('operation_status', request_id='view1')
    assert viewed['tracking'] is False and p._node is live
    assert not p._view.subscriptions and p._view.stream.store.directory is None
    p._view.publish_preview_once()
    topic = viewed['topic_out'][0]['topic']
    assert viewed['topic_out'][0]['segment_id'] == 'a'
    assert topic in [o['topic'] for o in call('info')['topic_out']]
    from test_pointcloud_slam_preview import unpack
    pub = next(pub for pub in p._view.publishers if pub.topic == topic)
    meta, _ = unpack(pub.messages[-1])
    assert meta['captured_frames'] == meta['optimized_frames'] == 12
    tag = call('tag_place', **entry, segment_id='a', tag_name='Door', x=2, y=-1)['tag']
    assert call('list_tags', **entry)['tags'] == [tag]
    assert call('list_maps')['maps'][0]['tag_count'] == 1
    assert call('tag_place', **entry, segment_id='a', tag_name='Entrance', x=1, y=-2,
                yaw=.5, tag_id=tag['id'])['tag']['name'] == 'Entrance'
    assert call('untag_place', **entry, tag_id=tag['id'])['state'] == 'ok'
    assert call('list_tags', **entry)['tags'] == []
    call('close_map_view', map_id='different')
    assert p._view is not None
    assert call('delete_map', map_id=saved['map_id'])['state'] == 'deleted'
    assert p._view is None and p._node is live
    assert call('stop_mapping')['saved'] is False
    assert call('list_maps')['maps'] == []
    assert call('start_mapping', map_name='Third')['state'] == 'running'
    node = p._node
    node.stream.odom(odom(1000), now=1., measurement_now_ns=1_000_000_000)
    node.stream.cloud(1_000_000_000, 'camera', [[1, 2, 3]], now=1., measurement_now_ns=1_000_000_000)
    assert call('stop')['state'] == 'idle'
    assert call('info')['state'] == 'idle'
    assert _wait_until(lambda: call('info')['save_state'] == 'saved')
    assert [e['name'] for e in call('list_maps')['maps']] == ['Third']
    assert call('start_mapping', map_name='Fourth')['state'] == 'error'

    # Existing standalone paged checkpoints can be explicitly imported into a named capture.
    from plugins.pointcloud_slam_paged_archive import save_paged_archive
    from test_pointcloud_slam_paged_store import add, store
    old = store(tmp_path / 'sessions')
    add(old, 0)
    save_paged_archive(tmp_path / 'paged.json', old, {}, settings)
    p = PointCloudSlamPlugin({**settings, 'restore_archive': True}, _FakeExecutor())
    assert p.dispatch('pointcloud_slam', args)['state'] == 'ready'
    assert call('start_mapping', map_name='Imported')['state'] == 'running'
    assert p._node.stream.store.requires_new_segment
    assert call('stop_mapping')['keyframes'] == 1
    assert old.directory.is_dir()


def test_paged_card_exposes_only_extrinsic_and_applies_it():
    import json
    settings = {**config(), 'persistent_history': {'max_disk_bytes': 100000, 'reserve_bytes': 0}}
    card = PointCloudSlamPlugin(settings, _FakeExecutor())
    props = card.get_tools()[0]['configSchema']['properties']
    assert list(props) == ['child_from_cloud']
    assert props['child_from_cloud']['scope'] == 'instance'
    assert json.loads(props['child_from_cloud']['default']) == settings['child_from_cloud']
    matrix = np.eye(4)
    matrix[0, 3] = .12
    assert card.dispatch('pointcloud_slam', {'action': 'config', 'child_from_cloud': json.dumps(matrix.tolist())})['state'] == 'configured'
    np.testing.assert_array_equal(card._stream().extrinsic, matrix)
    assert card.dispatch('pointcloud_slam', {'action': 'config', 'child_from_cloud': '[[1]]'})['state'] == 'error'
