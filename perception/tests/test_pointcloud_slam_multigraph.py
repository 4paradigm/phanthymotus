"""Provisional inter-segment edges remain in one bounded graph."""
import numpy as np
import pytest
pytest.importorskip('gtsam')
from plugins.pointcloud_slam_multigraph import optimize_multisegment
from test_pointcloud_slam_segment_merge import options
from test_pointcloud_slam_incremental_backend import make_stream, add_frame
from test_pointcloud_slam_stream import odom


def three_segments():
    stream = make_stream()
    points = np.random.default_rng(8).uniform(-1, 1, (100, 3))
    add_frame(stream, 1000, 0, points)
    stream.odom(odom(1100, status='lost'), now=1.1)
    add_frame(stream, 1200, 7, points, segment='b')
    stream.odom(odom(1300, status='lost'), now=1.3)
    add_frame(stream, 1400, -4, points, segment='c')
    return stream


def links():
    return [dict(target='s', source='b', target_index=0, source_index=0, transform=np.eye(4)),
            dict(target='b', source='c', target_index=0, source_index=0, transform=np.eye(4))]


def test_graph_retains_two_links_and_rebuilds_all_three_segments():
    stream = three_segments()
    segments, graphs = stream.store.segments, stream.graph()
    edge_list = links(); edge_list[0]['target'] = segments[0].segment_id
    opts = options()
    result = optimize_multisegment(segments, graphs, edge_list,
        odom_sigmas=opts['odom_sigmas'], loop_sigmas=opts['loop_sigmas'], huber_k=1.,
        max_total_frames=10, max_total_points=4000)
    assert len(result['map_points']) == 300
    assert result['segment_ids'] == [s.segment_id for s in segments]
    for poses in result['poses']:
        np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-5)
    np.testing.assert_allclose(result['map_from_active_odom'] @ graphs[-1]['poses'][-1],
                               result['poses'][-1][-1], atol=1e-6)


def test_disconnected_graph_and_resource_overflow_are_rejected():
    stream = three_segments(); segments, graphs = stream.store.segments, stream.graph()
    edge_list = links(); edge_list[0]['target'] = segments[0].segment_id
    common = dict(odom_sigmas=[.1]*6, loop_sigmas=[.05]*6, huber_k=1.,
                  max_total_frames=10, max_total_points=4000)
    with pytest.raises(ValueError):
        optimize_multisegment(segments, graphs, edge_list[:1], **common)
    with pytest.raises(ValueError):
        optimize_multisegment(segments, graphs, edge_list, **{**common, 'max_total_frames': 2})


def test_multigraph_keeps_within_segment_loop_constraints():
    stream = make_stream(); p = np.random.default_rng(9).uniform(-1, 1, (100, 3))
    add_frame(stream, 1000, 0, p); add_frame(stream, 1100, 1, p); add_frame(stream, 1200, 2, p)
    segments, graphs = stream.store.segments, stream.graph()
    opts = dict(odom_sigmas=[.1]*6, loop_sigmas=[.05]*6, huber_k=1.,
                max_total_frames=10, max_total_points=4000)
    baseline = optimize_multisegment(segments, graphs, [], **opts)
    loop = {'source': 2, 'target': 0, 'transform': np.eye(4)}
    result = optimize_multisegment(segments, graphs, [], segment_loops={segments[0].segment_id: [loop]}, **opts)
    assert result['poses'][0][-1][0, 3] < baseline['poses'][0][-1][0, 3] - .1
    assert not np.allclose(result['map_points'], baseline['map_points'])


def test_three_segment_link_transforms_compose_in_map_frame():
    stream = three_segments(); segments, graphs = stream.store.segments, stream.graph()
    first = np.eye(4); first[0, 3] = 2
    second = np.eye(4); second[1, 3] = 3
    edge_list = [dict(target=segments[0].segment_id, source='b', target_index=0,
                      source_index=0, transform=first),
                 dict(target='b', source='c', target_index=0, source_index=0, transform=second)]
    result = optimize_multisegment(segments, graphs, edge_list, odom_sigmas=[.1]*6,
        loop_sigmas=[.05]*6, huber_k=1., max_total_frames=10, max_total_points=4000)
    np.testing.assert_allclose(result['poses'][1][0], first, atol=1e-5)
    np.testing.assert_allclose(result['poses'][2][0], first @ second, atol=1e-5)
