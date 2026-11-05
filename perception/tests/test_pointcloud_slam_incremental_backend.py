"""Synthetic incremental graph behavior; these results are never real loop claims."""
import numpy as np
import pytest

pytest.importorskip('gtsam')
pytest.importorskip('open3d')
from plugins.pointcloud_slam_incremental_backend import ExperimentalLoopGraph
from test_pointcloud_slam_geometry_gate import thresholds
from test_pointcloud_slam_stream import odom, stream


def make_stream():
    return stream(max_cloud_points=400, max_frames=8, max_points=4000)


def add_frame(s, t, x, points, segment='a'):
    s.odom(odom(t, x, segment=segment), now=t / 1000)
    s.cloud(t * 1_000_000, 'camera', points, now=t / 1000)


def backend(max_candidates=3):
    return ExperimentalLoopGraph(thresholds=thresholds(), odom_sigmas=[.1]*6,
                                 loop_sigmas=[.05]*6, huber_k=1., voxel_m=.04,
                                 max_distance_m=.1, max_candidates=max_candidates)


def test_incremental_geometry_gate_corrects_retained_history_and_keeps_loop_on_append():
    s = make_stream()
    points = np.random.default_rng(8).uniform(-1, 1, (200, 3))
    for t, x in [(1000, 0), (1100, 1.2), (1200, 2.2), (1300, .5)]:
        add_frame(s, t, x, points)
    b = backend()
    result = b.update(s.store.segments[0], s.graph()[0], candidate=(3, 0, np.eye(4)))
    assert len(result['passed_candidates']) == 1
    assert result['loop_accepted'] is False
    assert abs(result['poses'][1][0, 3] - 1.2) > .01
    assert not np.allclose(result['before_map'], result['after_map'])
    np.testing.assert_allclose(result['map_from_odom'] @ s.graph()[0]['poses'][-1],
                               result['poses'][-1], atol=1e-6)
    add_frame(s, 1400, .6, points)
    later = b.update(s.store.segments[0], s.graph()[0])
    assert len(later['passed_candidates']) == 1
    assert len(later['poses']) == 5
    assert not np.allclose(later['before_map'], later['after_map'])


def test_rejected_and_duplicate_candidates_never_change_retained_graph():
    s = make_stream()
    p = np.random.default_rng(2).uniform(-1, 1, (200, 3))
    for t, x, cloud in [(1000, 0, p), (1100, 1, p), (1200, 2, p + 10)]:
        add_frame(s, t, x, cloud)
    b = backend(max_candidates=1)
    bad = b.update(s.store.segments[0], s.graph()[0], candidate=(2, 0, np.eye(4)))
    assert not bad['passed_candidates'] and len(bad['rejected_candidates']) == 1
    good = b.update(s.store.segments[0], s.graph()[0], candidate=(1, 0, np.eye(4)))
    assert len(good['passed_candidates']) == 1
    with pytest.raises(ValueError):
        b.update(s.store.segments[0], s.graph()[0], candidate=(0, 1, np.eye(4)))
    assert len(b.update(s.store.segments[0], s.graph()[0])['passed_candidates']) == 1


def test_incremental_backend_refuses_segment_change_or_rewritten_history():
    s = make_stream(); p = np.random.default_rng(3).uniform(-1, 1, (200, 3))
    for t, x in [(1000, 0), (1100, 1)]: add_frame(s, t, x, p)
    b = backend(); b.update(s.store.segments[0], s.graph()[0])
    s.odom(odom(1200, segment='a', status='lost'), now=1.2)
    add_frame(s, 1300, 0, p, segment='b')
    with pytest.raises(ValueError): b.update(s.store.segments[1], s.graph()[1])
    changed = dict(s.graph()[0]); changed['poses'] = changed['poses'][:1]; changed['edges'] = []
    with pytest.raises(ValueError): b.update(s.store.segments[0], changed)
    changed = s.graph()[0]
    changed['edges'][0]['transform'][0, 3] += 4
    with pytest.raises(ValueError): b.update(s.store.segments[0], changed)


def test_optimizer_failure_does_not_retain_candidate(monkeypatch):
    import plugins.pointcloud_slam_incremental_backend as module
    s = make_stream(); p = np.random.default_rng(4).uniform(-1, 1, (200, 3))
    for t, x in [(1000, 0), (1100, 1)]: add_frame(s, t, x, p)
    b = backend(); b.update(s.store.segments[0], s.graph()[0])
    original = module.optimize_candidates
    monkeypatch.setattr(module, 'optimize_candidates', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('failed')))
    with pytest.raises(RuntimeError): b.update(s.store.segments[0], s.graph()[0], candidate=(1, 0, np.eye(4)))
    monkeypatch.setattr(module, 'optimize_candidates', original)
    assert not b.update(s.store.segments[0], s.graph()[0])['passed_candidates']


def test_candidate_capacity_is_bounded_without_losing_previous_constraint():
    s = make_stream(); p = np.random.default_rng(5).uniform(-1, 1, (200, 3))
    for t, x in [(1000, 0), (1100, 1), (1200, 2)]: add_frame(s, t, x, p)
    b = backend(max_candidates=1)
    b.update(s.store.segments[0], s.graph()[0], candidate=(1, 0, np.eye(4)))
    with pytest.raises(ValueError, match='capacity'):
        b.update(s.store.segments[0], s.graph()[0], candidate=(2, 0, np.eye(4)))
    assert len(b.update(s.store.segments[0], s.graph()[0])['passed_candidates']) == 1
