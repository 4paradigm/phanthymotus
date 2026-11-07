"""Candidate scheduling budgets do not depend on a geometry backend being installed."""
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.pointcloud_slam_keyframes import Segment
from plugins.pointcloud_slam_loop_search import ExperimentalLoopSearch
from test_pointcloud_slam_geometry_gate import thresholds
from test_pointcloud_slam_observation import pinhole, scoring


def options():
    return dict(descriptor_voxel_m=.1, min_separation_s=0., candidate_separation_s=.15,
                max_per_frame=1, max_candidates=10, voxel_m=.04, max_distance_m=.3,
                thresholds=thresholds(), odom_sigmas=[.1]*6, loop_sigmas=[.05]*6, huber_k=1.)


class Graph:
    def __init__(self, **kw): pass
    def update(self, segment, graph, **kw):
        return dict(poses=graph['poses'], passed_candidates=[])


def run_search(monkeypatch, worker):
    import plugins.pointcloud_slam_loop_search as module
    monkeypatch.setattr(module, '_descriptor', lambda *a: np.ones(33))
    frames = tuple(SimpleNamespace(points=np.ones((12, 3)), cloud_stamp_ns=1_000_000_000+i*100_000_000)
                   for i in range(6))
    segment = Segment('a', frames)
    graph = dict(segment_id='a', poses=[np.eye(4) for _ in frames], edges=[{}]*5)
    return worker.update(segment, graph, online=True, build_map=False)


def test_revisit_obeys_the_same_attempt_and_candidate_spacing_budgets(monkeypatch):
    import plugins.pointcloud_slam_loop_search as module
    monkeypatch.setattr(module, 'ExperimentalLoopGraph', Graph)
    worker = ExperimentalLoopSearch(**options())
    class Revisit:
        attempts = 0
        confirmation_evaluations = 0
        selections = []
        def update(self, frames, poses, eligible):
            self.selections.append(list(eligible))
            self.attempts += len(eligible)
    worker.revisit = Revisit()
    result = run_search(monkeypatch, worker)
    assert result['candidate_attempts'] == 5
    assert all(len(selected) <= 1 for selected in worker.revisit.selections)
    assert result['resource_profile']['max_per_frame'] == 1
    assert result['compatibility'] and result['resolved_scoring']['metric_id'] == 'legacy_inlier_rmse'


def test_candidate_spacing_is_also_applied_before_revisit(monkeypatch):
    import plugins.pointcloud_slam_loop_search as module
    monkeypatch.setattr(module, 'ExperimentalLoopGraph', Graph)
    worker = ExperimentalLoopSearch(**{**options(), 'max_per_frame': 2})
    class Revisit:
        attempts = 0
        confirmation_evaluations = 0
        selections = []
        def update(self, frames, poses, eligible):
            self.selections.append(list(eligible)); self.attempts += len(eligible)
    worker.revisit = Revisit()
    run_search(monkeypatch, worker)
    assert worker.revisit.selections[-1] == [0, 2]


def test_new_observation_and_old_numeric_depth_threshold_cannot_mix():
    opts = options(); opts['thresholds'].pop('max_rmse_m')
    opts['thresholds']['max_forward_depth_score_m'] = .09
    with pytest.raises(ValueError, match='threshold|contract'):
        ExperimentalLoopSearch(**opts, observation=pinhole(), scoring=scoring())


def test_an_explicit_observation_requires_an_explicit_matching_score():
    with pytest.raises(ValueError, match='scor'):
        ExperimentalLoopSearch(**options(), observation=pinhole())


def test_explicit_strategy_snapshots_are_detached_and_threshold_bound():
    s = scoring(); obs = pinhole(); opts = options()
    opts['thresholds'].pop('max_rmse_m')
    opts['thresholds'].update(max_range_stratified_rmse_m=.09, score_contract=s)
    worker = ExperimentalLoopSearch(**opts, observation=obs, scoring=s, revisit={'enabled': False})
    snapshot = worker.strategy_snapshot()
    assert snapshot['resolved_scoring']['range_definition'] == 'sensor_optical_z'
    assert snapshot['resolved_observation']['sensor_id'] == 'lens'
    assert snapshot['compatibility'] == []
    snapshot['resolved_observation']['parameters']['K'][0] = 2.
    assert worker.strategy_snapshot()['resolved_observation']['parameters']['K'][0] == 100.


def test_explicit_legacy_profile_keeps_original_numeric_threshold_semantics():
    opts = options(); opts['thresholds'].pop('max_rmse_m')
    opts['thresholds']['max_forward_depth_score_m'] = .09
    worker = ExperimentalLoopSearch(**opts, legacy_profile='child_x_depth')
    assert worker.strategy_snapshot()['resolved_scoring']['metric_id'] == 'legacy_child_x_depth'
    assert worker.strategy_snapshot()['compatibility']
    with pytest.raises(ValueError):
        ExperimentalLoopSearch(**opts, observation=pinhole(), scoring=scoring(),
                               legacy_profile='child_x_depth')
    with pytest.raises(ValueError):
        ExperimentalLoopSearch(**opts, legacy_profile='inlier_rmse')
    with pytest.raises(ValueError):
        ExperimentalLoopSearch(**opts, observation=pinhole(), legacy_profile='visible_optical_depth',
                               revisit={'enabled': False})
