"""The experimental card binds both inputs and stops only its own node."""
import numpy as np

from plugins.pointcloud_slam_plugin import PointCloudSlamPlugin
from vision_stubs import _FakeExecutor


def config():
    return {'cloud_frame': 'camera', 'child_frame': 'body', 'clock_id': 'clock',
            'clock_relation_verified': True, 'extrinsic_calibrated': True,
            'child_from_cloud': np.eye(4).tolist(), 'max_gap_ms': 200,
            'max_age_s': 1, 'max_pending': 2, 'max_cloud_points': 100,
            'max_frames': 10, 'max_points': 1000}


def test_start_uses_formats_regardless_of_order_and_stops_only_own_node():
    executor = _FakeExecutor()
    plugin = PointCloudSlamPlugin(config(), executor)
    result = plugin.dispatch('pointcloud_slam', {'action': 'start',
        'input_topics': ['/c', '/o'],
        'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}})
    assert result['state'] == 'running'
    node = executor.nodes[0]
    assert {s.topic for s in node.subscriptions} == {'/c', '/o'}
    assert plugin.dispatch('pointcloud_slam', {'action': 'info'})['topic_out'] == []
    plugin.dispatch('pointcloud_slam', {'action': 'stop'})
    assert not executor.nodes and node.destroyed


def test_start_fails_closed_for_unverified_clock_or_missing_bindings():
    cfg = config()
    cfg['clock_relation_verified'] = False
    plugin = PointCloudSlamPlugin(cfg, _FakeExecutor())
    args = {'action': 'start', 'input_topics': ['/o', '/c'],
            'input_bindings': {'/o': 'state/odom', '/c': 'sensor/pointcloud2'}}
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


def test_bundle_exposes_experimental_card_only_when_enabled():
    from main import PerceptionBundle
    executor = _FakeExecutor()
    assert not PerceptionBundle({'plugins': {}}, executor).owns('pointcloud_slam')
    assert PerceptionBundle({'plugins': {'pointcloud_slam': {'enabled': True}}},
                            executor).owns('pointcloud_slam')
