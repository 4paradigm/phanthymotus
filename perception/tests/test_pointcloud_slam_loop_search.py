"""Synthetic tests for a bounded, unvalidated online candidate scheduler."""
import numpy as np
import pytest

pytest.importorskip('gtsam')
pytest.importorskip('open3d')
from plugins.pointcloud_slam_loop_search import ExperimentalLoopSearch
from test_pointcloud_slam_incremental_backend import make_stream, add_frame
from test_pointcloud_slam_geometry_gate import thresholds


def search():
    return ExperimentalLoopSearch(descriptor_voxel_m=.08, min_separation_s=.25,
        candidate_separation_s=.01, max_per_frame=2, max_candidates=4,
        thresholds=thresholds(), odom_sigmas=[.1]*6, loop_sigmas=[.05]*6,
        huber_k=1., voxel_m=.04, max_distance_m=.8)


def test_searches_keyframes_by_geometry_then_rebuilds_history():
    s = make_stream(); p = np.random.default_rng(42).uniform(-1, 1, (300, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        add_frame(s, t, x, p)
    worker = search()
    result = worker.update(s.store.segments[0], s.graph()[0])
    assert result['candidate_attempts'] >= 1
    assert len(result['passed_candidates']) == 1
    assert result['loop_accepted'] is False
    assert not np.allclose(result['before_map'], result['after_map'])
    assert abs(result['poses'][1][0, 3] - 1.2) > .01
    previous = result['candidate_attempts']
    result = worker.update(s.store.segments[0], s.graph()[0])
    assert result['candidate_attempts'] == previous


def test_cannot_reuse_search_across_segments():
    s = make_stream(); p = np.random.default_rng(1).uniform(-1, 1, (300, 3))
    add_frame(s, 1000, 0, p)
    worker = search(); worker.update(s.store.segments[0], s.graph()[0])
    s.odom(__import__('test_pointcloud_slam_stream').odom(1100, status='lost'), now=1.1)
    add_frame(s, 1200, 0, p, segment='b')
    with pytest.raises(ValueError): worker.update(s.store.segments[1], s.graph()[1])


def test_invalid_descriptor_skips_candidate_search_but_keeps_odometry_history():
    s = make_stream(); p = np.random.default_rng(3).uniform(-1, 1, (300, 3))
    add_frame(s, 1000, 0, [[0, 0, 1]])
    add_frame(s, 1100, 1, p)
    result = search().update(s.store.segments[0], s.graph()[0])
    assert len(result['poses']) == 2
    assert result['descriptor_skips'] == 1
    assert result['candidate_attempts'] == 0
