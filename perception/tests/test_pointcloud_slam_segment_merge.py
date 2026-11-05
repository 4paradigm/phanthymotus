"""Synthetic cross-segment hypotheses must not be mistaken for validated loops."""
import numpy as np
import pytest
pytest.importorskip('gtsam')
pytest.importorskip('open3d')
from plugins.pointcloud_slam_segment_merge import experimental_segment_merge
from test_pointcloud_slam_geometry_gate import thresholds
from test_pointcloud_slam_incremental_backend import make_stream, add_frame
from test_pointcloud_slam_stream import odom


def options():
    return dict(thresholds=thresholds(), odom_sigmas=[.1]*6, loop_sigmas=[.05]*6,
                huber_k=1., voxel_m=.04, max_distance_m=.1,
                max_total_frames=10, max_total_points=4000)


def make_two_segments(mismatch=False):
    s = make_stream(); p = np.random.default_rng(22).uniform(-1, 1, (300, 3))
    add_frame(s, 1000, 0, p); add_frame(s, 1100, 1, p)
    s.odom(odom(1200, status='lost'), now=1.2)
    add_frame(s, 1300, 9, p + (10 if mismatch else 0), segment='b')
    add_frame(s, 1400, 10, p + (10 if mismatch else 0), segment='b')
    return s


def test_geometric_hypothesis_aligns_separate_graphs_and_rebuilds_both_histories():
    s = make_two_segments(); target, source = s.store.segments
    a, b = s.graph(); original = b['poses'][0].copy()
    result = experimental_segment_merge(target, a, source, b,
        target_index=0, source_index=0, initial=np.eye(4), **options())
    assert result['hypothesis_created'] and result['loop_accepted'] is False
    assert abs(result['source_poses'][0][0, 3]) < .05
    assert result['map_points'].shape == (1200, 3)
    np.testing.assert_allclose(result['map_from_source_odom'] @ original,
                               result['source_poses'][0], atol=1e-6)
    np.testing.assert_allclose(s.graph()[1]['poses'][0], original)


def test_failed_cross_segment_geometry_does_not_merge_or_output_map():
    s = make_two_segments(mismatch=True); target, source = s.store.segments; a,b=s.graph()
    result = experimental_segment_merge(target, a, source, b,
        target_index=0, source_index=0, initial=np.eye(4), **options())
    assert not result['hypothesis_created'] and result['loop_accepted'] is False
    assert 'map_points' not in result


def test_cross_segment_merge_rejects_same_segment_and_capacity():
    s = make_two_segments(); target, source = s.store.segments; a,b=s.graph()
    with pytest.raises(ValueError):
        experimental_segment_merge(target, a, target, a, target_index=0, source_index=0,
                                   initial=np.eye(4), **options())
    with pytest.raises(ValueError):
        experimental_segment_merge(target, a, source, b, target_index=0, source_index=0,
                                   initial=np.eye(4), **{**options(), 'max_total_frames': 3})


def test_cross_segment_registration_direction_is_source_to_target():
    s = make_stream(); p = np.random.default_rng(31).uniform(-1, 1, (300, 3))
    add_frame(s, 1000, 0, p + [2, 0, 0])
    s.odom(odom(1100, status='lost'), now=1.1)
    add_frame(s, 1200, 9, p, segment='b')
    target, source = s.store.segments; a, b = s.graph()
    initial = np.eye(4); initial[0, 3] = 2
    result = experimental_segment_merge(target, a, source, b, target_index=0,
                                      source_index=0, initial=initial, **options())
    assert result['hypothesis_created']
    assert result['source_poses'][0][0, 3] == pytest.approx(2, abs=.01)
