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


def modeled_pair(radial=False):
    """Two child frames observe one sensor scene with different capture mounts."""
    from dataclasses import replace
    from plugins.pointcloud_slam_keyframes import Segment
    from test_pointcloud_slam_observation import pinhole, lidar, scoring
    rng = np.random.default_rng(92)
    ranges = np.repeat([1., 3., 5.], 100)
    uv = rng.uniform(-.2, .2, (len(ranges), 2))
    depth = ranges+rng.uniform(-.03, .03, len(ranges))
    raw = np.column_stack((uv*depth[:, None], depth))
    if radial:
        raw = raw[:, [2, 0, 1]]
    a = np.array([[0, 0, 1, 2], [-1, 0, 0, 3], [0, -1, 0, 0], [0, 0, 0, 1.]])
    b = a.copy(); b[0, 3] += 1.
    stream = make_stream()
    add_frame(stream, 1000, 0, raw@a[:3, :3].T+a[:3, 3])
    stream.odom(odom(1100, status='lost'), now=1.1)
    add_frame(stream, 1200, 9, raw@b[:3, :3].T+b[:3, 3], segment='b')
    segments = [Segment(s.segment_id, tuple(replace(f, capture_context={'child_from_cloud': t.tolist()})
                    for f in s.frames)) for s, t in zip(stream.store.segments, (a, b))]
    observation = lidar() if radial else pinhole()
    observation.update(transform_mode='per_capture', sensor_from_child=None)
    spec = scoring(radial)
    opts = options(); opts['thresholds'].pop('max_rmse_m')
    opts['thresholds'].update(max_range_stratified_rmse_m=.05, score_contract=spec)
    return (*segments, *stream.graph(), a@np.linalg.inv(b), observation, spec, opts)


@pytest.mark.parametrize('radial', [False, True])
def test_explicit_cross_segment_models_keep_independent_capture_origins(radial):
    target, source, a, b, initial, observation, spec, opts = modeled_pair(radial)
    result = experimental_segment_merge(target, a, source, b, target_index=0, source_index=0,
        initial=initial, observation=observation, scoring=spec, **opts)
    assert result['hypothesis_created'] and result['loop_accepted'] is False
    assert result['resolved_scoring'] == spec and result['compatibility'] == []
    diagnostic = result['diagnostic']
    assert diagnostic['forward']['score_metadata'] == spec
    assert diagnostic['reverse']['score_metadata'] == spec
    assert diagnostic['forward']['range_stratified_rmse_m'] == pytest.approx(0., abs=1e-10)


def test_explicit_legacy_visible_merge_cannot_fall_back_to_child_x_depth():
    from plugins.pointcloud_slam_observation import legacy_scoring
    target, source, a, b, initial, observation, spec, opts = modeled_pair()
    opts['thresholds'].pop('max_range_stratified_rmse_m')
    opts['thresholds'].update(max_forward_depth_score_m=.05,
                               score_contract=legacy_scoring(visible=True))
    result = experimental_segment_merge(target, a, source, b, target_index=0, source_index=0,
        initial=initial, observation=observation, legacy_profile='visible_optical_depth', **opts)
    assert result['hypothesis_created']
    assert result['diagnostic']['score_metadata']['metric_id'] == 'legacy_visible_optical_depth'
    assert result['compatibility'] and result['resolved_scoring']['range_definition'] == 'sensor_optical_z'


def test_explicit_merge_rejects_unbound_thresholds_and_missing_capture_evidence():
    from dataclasses import replace
    from plugins.pointcloud_slam_keyframes import Segment
    target, source, a, b, initial, observation, spec, opts = modeled_pair()
    call = dict(target_index=0, source_index=0, initial=initial, observation=observation)
    with pytest.raises(ValueError):
        experimental_segment_merge(target, a, source, b, scoring=spec, **call, **options())
    source = Segment(source.segment_id, (replace(source.frames[0], capture_context=None),))
    with pytest.raises(ValueError, match='capture'):
        experimental_segment_merge(target, a, source, b, scoring=spec, **call, **opts)
    with pytest.raises(ValueError):
        experimental_segment_merge(target, a, source, b, scoring=spec,
            legacy_profile='visible_optical_depth', **call, **opts)
