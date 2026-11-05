"""Offline graph experiment uses geometry gates and explicit noise weights."""
import numpy as np
import pytest

pytest.importorskip('gtsam')
pytest.importorskip('open3d')
from plugins.pointcloud_slam_experimental_backend import optimize_candidates
from tools.pointcloud_slam_candidate_graph import experiment_snapshot
from tools.pointcloud_slam_registration_experiment import registration_diagnostics
from test_pointcloud_slam_geometry_gate import thresholds
from test_pointcloud_slam_stream import odom, stream


def test_synthetic_icp_gate_moves_history_while_rejecting_no_overlap():
    s = stream()
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        s.odom(odom(t, x), now=t / 1000)
        s.cloud(t * 1_000_000, 'camera', [[0, 0, 0]], now=t / 1000)
    points = np.random.default_rng(8).uniform(-1, 1, (300, 3))
    good = registration_diagnostics(points, points, np.eye(4), voxel_m=.04, max_distance_m=.1)
    bad = registration_diagnostics(points, points + [10, 0, 0], np.eye(4), voxel_m=.04, max_distance_m=.1)
    result = optimize_candidates(s.graph()[0],
        [{'source': 3, 'target': 0, 'diagnostic': good},
         {'source': 2, 'target': 0, 'diagnostic': bad}],
        thresholds=thresholds(), odom_sigmas=[.1]*6, loop_sigmas=[.05]*6, huber_k=1.)
    assert len(result['passed_candidates']) == 1 and len(result['rejected_candidates']) == 1
    assert abs(result['poses'][3][0, 3]) < .5
    assert abs(result['poses'][1][0, 3] - 1.2) > .01
    assert result['loop_accepted'] is False
    snapshot = experiment_snapshot(s.store.segments[0], s.graph()[0], result)
    assert abs(snapshot['after_map'][1, 0] - snapshot['before_map'][1, 0]) > .01
    np.testing.assert_allclose(snapshot['map_from_odom'] @ s.graph()[0]['poses'][-1],
                               result['poses'][-1], atol=1e-6)
    assert snapshot['loop_accepted'] is False
    from plugins.pointcloud_slam_preview import encode_preview
    from test_pointcloud_slam_preview import unpack
    packet = encode_preview(snapshot['after_map'], result['poses'][-1], max_points=10,
                            metadata={'segment_id': 'a', 'frame_id': 'odom',
                                      'stamp_ns': '1300000000', 'backend': 'raw'})
    _, body = unpack(packet)
    decoded = np.frombuffer(body[17:], dtype='<f4').reshape(-1, 3)
    np.testing.assert_allclose(decoded, snapshot['after_map'], atol=1e-6)


def test_invalid_geometry_transform_is_never_inserted():
    s = stream()
    for t, x in [(1000, 0), (1100, 1), (1200, 2)]:
        s.odom(odom(t, x), now=t / 1000)
        s.cloud(t * 1_000_000, 'camera', [[0, 0, 0]], now=t / 1000)
    from test_pointcloud_slam_geometry_gate import diagnostic
    bad = diagnostic()
    bad['forward']['transform'] = np.zeros((4, 4)).tolist()
    with pytest.raises(ValueError):
        optimize_candidates(s.graph()[0], [{'source': 2, 'target': 0, 'diagnostic': bad}],
            thresholds=thresholds(), odom_sigmas=[.1]*6, loop_sigmas=[.05]*6, huber_k=1.)


def test_registration_transform_is_inverted_for_graph_factor():
    s = stream()
    for t, x in [(1000, 0), (1100, 1), (1200, 2.5)]:
        s.odom(odom(t, x), now=t / 1000)
        s.cloud(t * 1_000_000, 'camera', [[0, 0, 0]], now=t / 1000)
    points = np.random.default_rng(9).uniform(-1, 1, (300, 3))
    initial = np.eye(4)
    initial[0, 3] = 2.
    diag = registration_diagnostics(points, points + [2, 0, 0], initial,
                                    voxel_m=.04, max_distance_m=.1)
    out = optimize_candidates(s.graph()[0], [{'source': 2, 'target': 0, 'diagnostic': diag}],
        thresholds=thresholds(), odom_sigmas=[.1]*6, loop_sigmas=[.05]*6, huber_k=1.)
    assert 1.9 < out['poses'][2][0, 3] < 2.5
