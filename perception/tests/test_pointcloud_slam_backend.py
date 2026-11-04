"""GTSAM consumes separate streaming graphs without inventing loop closures."""
import numpy as np
import pytest

pytest.importorskip('gtsam')
from plugins.pointcloud_slam_backend import optimize_odometry_graph
from test_pointcloud_slam_stream import odom, stream


def test_stream_graph_optimizes_with_explicit_noise_and_no_loop():
    s = stream()
    for t, x in [(1000, 0), (1100, 2)]:
        s.odom(odom(t, x), now=t/1000)
        s.cloud(t * 1_000_000, 'camera', [[1, 0, 0]], now=t/1000)
    result = optimize_odometry_graph(s.graph()[0], sigmas=[.1] * 6)
    assert len(result) == 2
    assert result[1][:3, 3] == pytest.approx([2, 0, 0])


def test_ros_input_graph_can_be_passed_to_backend_without_merging_segments():
    import json
    from types import SimpleNamespace as NS
    from plugins.pointcloud_slam_ros import SlamInputNode
    from test_pointcloud_slam_ros import cloud
    s = stream()
    node = SlamInputNode('/o', '/c', s, clock=lambda: 1.)
    callbacks = {sub.topic: sub.callback for sub in node.subscriptions}
    callbacks['/o'](NS(data=json.dumps(odom(1000))))
    callbacks['/c'](cloud())
    result = node.optimize_odometry(sigmas=[.1] * 6)
    assert len(result) == 1 and result[0]['segment_id'] == 'a'
    assert result[0]['poses'][0][:3, 3] == pytest.approx([0, 0, 0])
    assert result[0]['map_points'].tolist() == [[1., 2., 3.]]
    assert result[0]['last_cloud_stamp_ns'] == 1_000_000_000
    assert result[0]['frame_id'] == 'odom'


def test_synthetic_loop_rebuilds_stream_keyframe_history():
    from plugins.pointcloud_slam_map import rebuild_segment
    from tools.pointcloud_slam_graph_experiment import optimize_graph
    s = stream()
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        s.odom(odom(t, x), now=t / 1000)
        s.cloud(t * 1_000_000, 'camera', [[0, 0, 0]], now=t / 1000)
    graph = s.graph()[0]
    edges = [(e['from'], e['to'], e['transform']) for e in graph['edges']]
    corrected = optimize_graph(graph['poses'], edges, [(0, 3, np.eye(4), True)])
    segment = s.store.segments[0]
    before = rebuild_segment(segment, graph['poses'])
    after = rebuild_segment(segment, corrected)
    assert abs(before[1, 0] - after[1, 0]) > .01
    assert abs(after[3, 0]) < abs(before[3, 0])


@pytest.mark.parametrize('sigmas', [None, [0] * 6, [float('nan')] * 6])
def test_noise_must_be_explicit_positive_finite(sigmas):
    with pytest.raises(ValueError):
        optimize_odometry_graph({'poses': [np.eye(4)], 'edges': []}, sigmas=sigmas)
