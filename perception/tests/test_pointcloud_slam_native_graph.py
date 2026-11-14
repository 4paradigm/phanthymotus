"""Legacy wiring resolves roles from native graph types, never names or order."""
import pytest
from types import SimpleNamespace as NS

import plugins.pointcloud_slam_ros as ros
from vision_stubs import _FakeNode


def test_string_monitor_subscription_does_not_hide_native_cloud_publisher(monkeypatch):
    monkeypatch.setattr(_FakeNode, 'get_topic_names_and_types', lambda self: [
        ('/a', ['sensor_msgs/msg/PointCloud2', 'std_msgs/msg/String']),
        ('/b', ['std_msgs/msg/String']),
    ], raising=False)
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic: [
        NS(topic_type='sensor_msgs/msg/PointCloud2' if topic == '/a' else 'std_msgs/msg/String')
    ], raising=False)
    assert ros.discover_input_bindings(['/b', '/a']) == {
        '/a': 'sensor/pointcloud2', '/b': 'state/odom'}
    assert all(node.destroyed for node in _FakeNode.instances)


def test_conflicting_publisher_types_reject_even_when_graph_union_looks_valid(monkeypatch):
    monkeypatch.setattr(_FakeNode, 'get_topic_names_and_types', lambda self: [
        ('/a', ['sensor_msgs/msg/PointCloud2']), ('/b', ['std_msgs/msg/String']),
    ], raising=False)
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic: [
        NS(topic_type='sensor_msgs/msg/PointCloud2'), NS(topic_type='std_msgs/msg/String')
    ] if topic == '/a' else [NS(topic_type='std_msgs/msg/String')], raising=False)
    with pytest.raises(ValueError, match='native ROS'):
        ros.discover_input_bindings(['/a', '/b'])
    assert all(node.destroyed for node in _FakeNode.instances)


@pytest.mark.parametrize('topics', [
    ['/looks_like_odom', '/looks_like_cloud'],
    ['/looks_like_cloud', '/looks_like_odom'],
])
def test_native_graph_types_override_topic_names_and_connection_order(monkeypatch, topics):
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic: [
        NS(topic_type={
            '/looks_like_odom': 'sensor_msgs/msg/PointCloud2',
            '/looks_like_cloud': 'std_msgs/msg/String',
            '/unconnected': 'sensor_msgs/msg/PointCloud2'}[topic])], raising=False)
    assert ros.discover_input_bindings(topics) == {
        '/looks_like_odom': 'sensor/pointcloud2', '/looks_like_cloud': 'state/odom'}
    assert all(node.destroyed for node in _FakeNode.instances)


def test_late_native_topic_discovery_retries_within_deadline(monkeypatch):
    now = [0.]
    monkeypatch.setattr(ros.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(ros.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic:
        [NS(topic_type='sensor_msgs/msg/PointCloud2')] if topic == '/a' else
        [NS(topic_type='std_msgs/msg/String')] if now[0] >= .1 else [], raising=False)
    assert ros.discover_input_bindings(['/b', '/a']) == {
        '/a': 'sensor/pointcloud2', '/b': 'state/odom'}
    assert .1 <= now[0] <= 1.
    assert all(node.destroyed for node in _FakeNode.instances)


def test_missing_topic_expires_without_guessing_a_role(monkeypatch):
    now = [0.]
    monkeypatch.setattr(ros.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(ros.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic:
        [NS(topic_type='sensor_msgs/msg/PointCloud2')] if topic == '/a' else [], raising=False)
    with pytest.raises(ValueError, match='native ROS.*discovery'):
        ros.discover_input_bindings(['/a', '/b'])
    assert 0 < now[0] <= 1.
    assert all(node.destroyed for node in _FakeNode.instances)


@pytest.mark.parametrize('types', [
    ['sensor_msgs/msg/PointCloud2'],
    ['std_msgs/msg/String', 'sensor_msgs/msg/PointCloud2'],
    ['nav_msgs/msg/Odometry'],
])
def test_ambiguous_or_unsupported_native_graph_does_not_bind(monkeypatch, types):
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', lambda self, topic:
        [NS(topic_type=kind) for kind in (['sensor_msgs/msg/PointCloud2'] if topic == '/a' else types)],
        raising=False)
    with pytest.raises(ValueError, match='native ROS'):
        ros.discover_input_bindings(['/a', '/b'])
    assert all(node.destroyed for node in _FakeNode.instances)


def test_graph_read_failure_still_destroys_discovery_node(monkeypatch):
    def fail(self, topic):
        raise RuntimeError('graph unavailable')
    monkeypatch.setattr(_FakeNode, 'get_publishers_info_by_topic', fail, raising=False)
    with pytest.raises(RuntimeError, match='graph unavailable'):
        ros.discover_input_bindings(['/a', '/b'])
    assert all(node.destroyed for node in _FakeNode.instances)
