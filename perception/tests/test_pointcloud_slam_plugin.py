"""The experimental card binds both inputs and stops only its own node."""
import numpy as np

from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
from vision_stubs import _FakeExecutor


def config():
    return {'cloud_frame': 'camera', 'child_frame': 'body', 'clock_id': 'clock',
            'clock_relation_verified': True, 'extrinsic_calibrated': True,
            'child_from_cloud': np.eye(4).tolist(), 'max_gap_ms': 200,
            'max_age_s': 1, 'max_pending': 2, 'max_cloud_points': 100,
            'max_frames': 10, 'max_points': 1000, 'max_measurement_age_ms': 300,
            'max_future_skew_ms': 20}


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
