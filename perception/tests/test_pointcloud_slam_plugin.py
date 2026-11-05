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
    assert not executor.nodes and node.destroyed


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
    assert {'topic': '/c/slam_odom_preview', 'format': 'sensor/slam-preview'} in info['topic_out']
    pub = next(p for p in node.publishers if p.topic == '/c/slam_odom_preview')
    assert _wait_until(lambda: len(pub.messages) > 0)
    plugin.dispatch('pointcloud_slam', {'action': 'stop'})
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
