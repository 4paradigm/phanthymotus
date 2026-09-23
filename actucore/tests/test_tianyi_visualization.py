import time
import pytest
from types import SimpleNamespace

import numpy as np

from test_tianyi_arm_root import profile
from teleop.kinematics import TianyiIK
from teleop.tianyi_visualization import chains, snapshot


def test_vendor_chain_endpoint_matches_control_fk(tmp_path):
    path, _, _ = profile(tmp_path)
    solver = TianyiIK(path)
    q = np.linspace(-.15, .15, 14)
    lines = chains(solver, q)
    assert [len(line) for line in lines] == [8, 8]
    for line, palm in zip(lines, solver.palms(q)):
        np.testing.assert_allclose(line[-1], palm[:3, 3], atol=1e-12)


def test_snapshot_freshness_and_failed_ik_never_reuses_result(tmp_path):
    path, _, _ = profile(tmp_path)
    solver = TianyiIK(path)
    q = np.zeros(14)
    feedback = {'q': q, 'arm_ns': time.monotonic_ns()}
    adapter = SimpleNamespace(solver=solver, output={}, hardware_output=False,
                              link=SimpleNamespace(feedback=lambda: {'feedback': feedback}))
    value = snapshot(adapter)
    assert value['available'] and value['feedback_fresh']
    assert len(value['body']) == 3 and value['shoulder_height'] > .3
    assert value['ik'] == []
    # Display is lock-independent and does not mutate shared solver FK data.
    targets = solver.palms(q)
    solver.visualization_sample = {'monotonic_ns': time.monotonic_ns(),
                                   'targets': targets, 'ik_q': q.copy()}
    assert len(snapshot(adapter)['ik']) == 2
    solver.visualization_sample = {'monotonic_ns': time.monotonic_ns(),
                                   'targets': targets, 'ik_q': None}
    assert snapshot(adapter)['ik'] == []
    feedback['arm_ns'] -= 200_000_000
    value = snapshot(adapter)
    assert not value['feedback_fresh'] and value['measured'] == []
    assert value['body']


def test_failed_solve_keeps_display_history_but_never_returns_command(tmp_path, monkeypatch):
    import pytest
    path, _, _ = profile(tmp_path)
    solver = TianyiIK(path)
    q = np.zeros(14)
    solver.self_test(q)
    previous = solver.last_valid_visualization
    feedback = {'q': q, 'arm_ns': time.monotonic_ns()}
    adapter = SimpleNamespace(solver=solver, output={}, hardware_output=False,
                              link=SimpleNamespace(feedback=lambda: {'feedback': feedback}))
    def fail(*args, **kwargs):
        raise ValueError('ik_not_converged')
    monkeypatch.setattr(solver, '_solve', fail)
    with pytest.raises(ValueError, match='ik_not_converged'):
        solver.solve(solver.palms(q), q)
    value = snapshot(adapter)
    assert value['ik'] == [] and len(value['held_ik']) == 2
    assert solver.last_valid_visualization is previous
    assert value['reason'] == 'ik_not_converged'
    # Old history remains labelled history; a new failed frame cannot freshen it.
    previous['monotonic_ns'] -= 2_000_000_000
    value = snapshot(adapter)
    assert value['held_ik_age_ms'] >= 2000 and value['intent_age_ms'] < 750
    adapter.link.feedback = lambda: (_ for _ in ()).throw(ValueError('missing'))
    value = snapshot(adapter)
    assert value['available'] and value['measured'] == []
    assert len(value['held_ik']) == 2 and value['reason'] == 'feedback_unavailable'


def test_budget_exhaustion_accepts_only_geometrically_valid_result(tmp_path, monkeypatch):
    import pytest
    path, _, _ = profile(tmp_path)
    solver=TianyiIK(path)
    q=np.zeros(14)
    targets=solver.palms(q)
    result=SimpleNamespace(success=False,status=0,x=q.copy())
    monkeypatch.setattr(solver,'least_squares',lambda *a,**k:result)
    assert len(solver.solve(targets,q))==14
    result.x=np.ones(14)*.2
    with pytest.raises(ValueError,match='ik_target_unreachable'):
        solver.solve(targets,q)
    result.status=-1;result.x=q.copy()
    with pytest.raises(ValueError,match='ik_not_converged'):
        solver.solve(targets,q)


def test_strict_overlay_preserves_raw_target_and_last_valid_arm(tmp_path):
    path,_,_=profile(tmp_path);s=TianyiIK(path);q=np.zeros(14)
    target=s.palms(q);s.solve(target,q)
    target[0][0,3]+=1.5
    with pytest.raises(ValueError):s.solve(target,q)
    adapter=SimpleNamespace(solver=s,output={},hardware_output=False,
        link=SimpleNamespace(feedback=lambda:{'feedback':{'q':q,'arm_ns':time.monotonic_ns()}}))
    v=snapshot(adapter)
    assert len(v['workspace_bounds'])==2
    assert 'projection' not in v and 'requested_targets' not in v
    assert v['targets'][0][0]==pytest.approx(target[0][0,3])
    assert not v['ik'] and v['held_ik'] and v['measured']
