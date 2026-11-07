"""Cross-segment coarse registration does not rely on odometry origins."""
import numpy as np
import pytest
pytest.importorskip('open3d')
from plugins.pointcloud_slam_relocalization import coarse_alignment


def test_recovers_transform_without_shared_odometry_origin():
    rng = np.random.default_rng(23)
    source = np.concatenate((rng.normal([0, 0, 0], [.5, .2, .1], (600, 3)),
                             rng.normal([.6, .7, .4], [.1, .3, .2], (400, 3))))
    angle = .45
    transform = np.array([[np.cos(angle), -np.sin(angle), 0, 1.1],
                          [np.sin(angle), np.cos(angle), 0, -.7],
                          [0, 0, 1, .3], [0, 0, 0, 1.]])
    target = source @ transform[:3, :3].T + transform[:3, 3]
    result = coarse_alignment(source, target, voxel_m=.08, max_distance_m=.15)
    assert result is not None
    np.testing.assert_allclose(result, transform, atol=.08)


def test_sparse_cloud_has_no_alignment():
    assert coarse_alignment(np.zeros((2, 3)), np.zeros((2, 3)),
                            voxel_m=.08, max_distance_m=.15) is None


def test_search_links_distinct_segments_without_using_odom_as_initial_guess():
    pytest.importorskip('gtsam')
    from plugins.pointcloud_slam_relocalization import relocalize_pair
    from test_pointcloud_slam_segment_merge import make_two_segments, options
    stream = make_two_segments()
    target, source = stream.store.segments
    target_graph, source_graph = stream.graph()
    result = relocalize_pair(target, target_graph, source, source_graph,
                             descriptor_voxel_m=.08, max_candidates=2, **options())
    assert result['hypothesis_created']
    assert result['loop_accepted'] is False
    assert len(result['map_points']) == 1200
    np.testing.assert_allclose(result['source_poses'][result['source_index']],
                               result['target_poses'][result['target_index']], atol=.05)


def test_sparse_source_does_not_merge():
    pytest.importorskip('gtsam')
    from plugins.pointcloud_slam_relocalization import relocalize_pair
    from plugins.pointcloud_slam_keyframes import Segment, Keyframe
    from test_pointcloud_slam_segment_merge import make_two_segments, options
    stream = make_two_segments()
    target, source = stream.store.segments
    first = source.frames[0]
    source = Segment(source.segment_id, (Keyframe(first.pose, first.cloud_stamp_ns, np.zeros((2, 3))),))
    target_graph, source_graph = stream.graph()
    source_graph = {**source_graph, 'poses': source_graph['poses'][:1], 'edges': []}
    result = relocalize_pair(target, target_graph, source, source_graph,
                             descriptor_voxel_m=.08, max_candidates=2, **options())
    assert not result['hypothesis_created'] and result['loop_accepted'] is False
    assert 'map_points' not in result


def test_relocalization_does_not_merge_when_geometry_gate_rejects():
    pytest.importorskip('gtsam')
    from plugins.pointcloud_slam_relocalization import relocalize_pair
    from test_pointcloud_slam_segment_merge import make_two_segments, options
    stream = make_two_segments()
    target, source = stream.store.segments
    a, b = stream.graph()
    opts = options()
    opts['thresholds']['min_correspondences'] = 100000
    result = relocalize_pair(target, a, source, b, descriptor_voxel_m=.08,
                             max_candidates=2, **opts)
    assert result['attempts'] == 2
    assert not result['hypothesis_created'] and result['loop_accepted'] is False
    assert 'map_points' not in result


@pytest.mark.parametrize('legacy', [False, True])
def test_relocalization_preserves_explicit_model_and_legacy_visible_semantics(legacy):
    from plugins.pointcloud_slam_relocalization import relocalize_pair
    from plugins.pointcloud_slam_observation import legacy_scoring
    from test_pointcloud_slam_segment_merge import modeled_pair
    target, source, a, b, initial, observation, spec, opts = modeled_pair()
    strategy = dict(observation=observation, scoring=spec)
    if legacy:
        opts['thresholds'].pop('max_range_stratified_rmse_m')
        opts['thresholds'].update(max_forward_depth_score_m=.05,
                                   score_contract=legacy_scoring(visible=True))
        strategy = dict(observation=observation, legacy_profile='visible_optical_depth')
    result = relocalize_pair(target, a, source, b, descriptor_voxel_m=.08,
                             max_candidates=1, **strategy, **opts)
    assert result['hypothesis_created'] and result['loop_accepted'] is False
    assert result['attempts'] == 1
    assert result['resolved_scoring']['metric_id'] == ('legacy_visible_optical_depth' if legacy else
                                                      'range_stratified_rmse')


def test_sparse_relocalization_still_validates_and_reports_the_selected_strategy():
    from dataclasses import replace
    from plugins.pointcloud_slam_keyframes import Segment
    from plugins.pointcloud_slam_relocalization import relocalize_pair
    from test_pointcloud_slam_segment_merge import modeled_pair, options
    target, source, a, b, initial, observation, spec, opts = modeled_pair()
    source = Segment(source.segment_id, (replace(source.frames[0], points=np.ones((2, 3))),))
    result = relocalize_pair(target, a, source, b, descriptor_voxel_m=.08,
        max_candidates=1, observation=observation, scoring=spec, **opts)
    assert result['attempts'] == 0 and not result['hypothesis_created']
    assert result['resolved_scoring'] == spec and result['compatibility'] == []
    with pytest.raises(ValueError):
        relocalize_pair(target, a, source, b, descriptor_voxel_m=.08,
            max_candidates=1, observation=observation, scoring=spec, **options())
