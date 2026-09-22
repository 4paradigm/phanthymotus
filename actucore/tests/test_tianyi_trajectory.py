"""Real Ruckig and Driver sampling with deterministic synthetic joint feedback."""
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from test_tianyi_arm_root import profile
from teleop.kinematics import TianyiIK
from teleop.trajectory import JointTrajectory, execution_state

pytest.importorskip('ruckig', reason='optional trajectory dependencies are not installed')


def smooth_solver(tmp_path):
    path, p, _ = profile(tmp_path)
    p['trajectory_smoothing'] = {'enabled': True, 'max_acceleration_rad_s2': 2., 'max_jerk_rad_s3': 20.}
    path.write_text(json.dumps(p))
    return TianyiIK(path)


def test_online_reversal_respects_derivatives_through_real_driver(tmp_path, monkeypatch):
    source_dir = os.environ.get('TIANYI_DRIVER_SOURCE')
    if not source_dir:
        pytest.skip('set TIANYI_DRIVER_SOURCE to the matching Driver x-humanoid/tianyi2.0 directory')
    source = Path(source_dir)/'motion_stream.py'
    spec = importlib.util.spec_from_file_location('smooth_gate_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = [1_000_000_000]
    measured = np.zeros(14)
    writes = []
    def feedback():
        return {**{k:now[0] for k in ('arm_ns','power_ns','fixed_ns')},
                'q':measured.tolist(), 'dq':[0.]*14, 'power_on':True,
                'estop':False, 'fault':False, 'fixed_body':True}
    solver = smooth_solver(tmp_path)
    lower = solver.model.lowerPositionLimit[solver.indices]
    upper = solver.model.upperPositionLimit[solver.indices]
    gate = module.MotionGate(feedback, lambda q,h:writes.append(q), list(zip(lower, upper)),
        clock=lambda:now[0], live_enabled=True, acceptance_check=lambda:True, hands_enabled=False)
    lease = gate.claim()
    monkeypatch.setattr('teleop.trajectory.time.monotonic_ns', lambda:now[0])
    samples = [measured.copy()]
    for seq in range(140):
        state = gate.status()
        goal = np.zeros(14)
        goal[0] = .12 if seq < 25 else -.12
        command = solver.command_step(goal, measured, state['commanded_q'], lambda:None,
                                      execution_state(state))
        packet = dict(protocol=module.PROTOCOL, boot_id=lease['boot_id'], session_id=lease['session_id'],
            seq=seq, generated_ns=now[0], valid_for_ms=100, q=command.tolist(), hands=[0.,0.])
        gate.accept({**packet, 'mac':module.sign(packet, lease['secret'])})
        now[0] += 20_000_000
        gate.tick()
        assert gate.state == 'active'
        assert not gate.command_state['limited']
        measured[:] = writes[-1]
        samples.append(measured.copy())
    q = np.asarray(samples)
    assert q[:,0].max() > .02 and q[-1,0] < -.1
    assert np.max(np.abs(np.diff(q, axis=0)/.02)) <= .2+1e-8
    assert np.max(np.abs(np.diff(q, n=2, axis=0)/.02**2)) <= 2.+1e-8
    assert np.max(np.abs(np.diff(q, n=3, axis=0)/.02**3)) <= 20.+1e-7
    assert solver.trajectory.diagnostics['state_source'] == 'confirmed_planner_sample'


def test_full_braking_box_is_checked_before_committing_sample(tmp_path, monkeypatch):
    solver = smooth_solver(tmp_path)
    checked = []
    def reject(a, b, budget):
        checked.append((a.copy(), b.copy()))
        raise ValueError('arm_collision')
    monkeypatch.setattr(solver, '_safe_transition', reject)
    goal = np.zeros(14);goal[0] = .2
    with pytest.raises(ValueError, match='arm_collision'):
        solver.command_step(goal, np.zeros(14), np.zeros(14), lambda:None)
    assert checked[0][1][0] == pytest.approx(.02)
    assert solver.trajectory.pending is None


def test_missing_stale_or_mismatched_driver_state_cannot_enable_live_smoothing(tmp_path):
    solver = smooth_solver(tmp_path)
    with pytest.raises(ValueError, match='command_state_missing'):
        execution_state({'state':'active'})
    for change, reason in (({'sample_ns':0}, 'stale'), ({'q':[1.]*14}, 'mismatch')):
        state = {'schema':'motus.command-state.v1','sample_ns':time.monotonic_ns(),
                 'q':[0.]*14,'dq':[0.]*14,'ddq':[0.]*14,'kind':'motion',**change}
        with pytest.raises(ValueError, match=reason):
            solver.command_step(np.zeros(14), np.zeros(14), np.zeros(14), lambda:None, state)
    assert solver.trajectory.pending is None


@pytest.mark.parametrize('config', [dict(enabled='true'), dict(enabled=True),
    dict(enabled=True,max_acceleration_rad_s2=-1,max_jerk_rad_s3=2),
    dict(enabled=True,max_acceleration_rad_s2=1,max_jerk_rad_s3=float('nan'))])
def test_smoothing_requires_explicit_finite_limits(config):
    with pytest.raises(ValueError, match='trajectory'):
        JointTrajectory(config, .2)


def test_hold_reference_requires_new_fresh_stationary_feedback():
    state = {'state':'hold','hold_confirmed':True,
             'command_state':{'schema':'motus.command-state.v1','kind':'hold','sample_ns':0,'q':[0.]*14},
             'feedback':{'arm_ns':time.monotonic_ns(),'dq':[0.]*14}}
    assert execution_state(state)['stationary_confirmed']
    assert 'stationary_confirmed' not in state['command_state']
    state['feedback']['dq'][0] = .03
    with pytest.raises(ValueError, match='stationary_feedback'):
        execution_state(state)


def test_quintic_replay_retimes_limits_and_keeps_phase_boundaries(tmp_path):
    from teleop.trajectory import smooth_schedule
    from teleop.replay import target_at
    solver = smooth_solver(tmp_path)
    original = [
        {'t':0.,'q':[0.]*14,'phase':'baseline'},
        {'t':.05,'q':[.02]+[0.]*13,'phase':'left'},
        {'t':.10,'q':[.06]+[0.]*13,'phase':'left'},
        {'t':.15,'q':[.08]+[0.]*13,'phase':'left'},
        {'t':.3,'q':[.08]+[0.]*13,'phase':'left_plateau'},
        {'t':.35,'q':[.08]+[0.]*6+[.03]+[0.]*6,'phase':'right'}]
    before = json.dumps(original)
    points = smooth_schedule(original, solver.trajectory, solver)
    assert json.dumps(original) == before
    assert points[-1]['t'] > original[-1]['t']
    for a,b in zip(points,points[1:]):
        c = np.asarray(a['polynomial'])
        for t in np.linspace(0,b['t']-a['t'],41):
            q = np.polyval(c,t)
            dq = np.polyval(c[:-1]*np.arange(5,0,-1)[:,None],t)
            ddq = np.polyval(c[:-2]*np.array([20,12,6,2])[:,None],t)
            jerk = np.polyval(c[:-3]*np.array([60,24,6])[:,None],t)
            assert np.max(np.abs(dq)) <= .2+1e-8
            assert np.max(np.abs(ddq)) <= 2.+1e-8
            assert np.max(np.abs(jerk)) <= 20.+1e-8
            sampled,phase = target_at(points,a['t']+t)
            np.testing.assert_allclose(sampled,q,atol=1e-8)
            if b['phase']=='left':np.testing.assert_allclose(q[7:],0,atol=1e-10)
            if b['phase']=='right':assert q[0] == pytest.approx(.08)
    # Internal recorded knots carry nonzero velocity; they are not independent
    # stop/start segments. End of each operation phase is stationary and C2.
    internal = np.asarray(points[1]['polynomial'])
    assert abs(internal[-2,0]) > .001
    for a,b in zip(points,points[1:]):
        if b['phase'] != a['phase'] and a['phase'] != 'baseline':
            c=np.asarray(a['polynomial'])
            np.testing.assert_allclose(c[-2],0,atol=1e-8)
            np.testing.assert_allclose(2*c[-3],0,atol=1e-8)


def test_quintic_curve_checks_intermediate_extrema_not_only_waypoints():
    from types import SimpleNamespace
    from teleop.trajectory import smooth_schedule
    seen=[]
    class Geometry:
        indices=np.arange(14)
        model=SimpleNamespace(lowerPositionLimit=np.full(14,-2.),upperPositionLimit=np.full(14,2.))
        def _safe_transition(self,lo,hi):seen.append((lo.copy(),hi.copy()))
    geometry=Geometry()
    trajectory=JointTrajectory({'enabled':True,'max_acceleration_rad_s2':2,'max_jerk_rad_s3':20},.2)
    points=[{'t':i*.1,'q':[q]+[0.]*13,'phase':'both'} for i,q in enumerate([0.,.1,.1,0.])]
    smooth_schedule(points,trajectory,geometry)
    assert max(hi[0] for _,hi in seen) > .1
    geometry.model.upperPositionLimit[0]=.1001
    with pytest.raises(ValueError,match='curve_joint_limit'):
        smooth_schedule(points,trajectory,geometry)


def test_unsent_proposal_does_not_become_the_next_initial_state(tmp_path):
    solver=smooth_solver(tmp_path)
    seed={'schema':'motus.command-state.v1','kind':'stationary_seed','sample_ns':time.monotonic_ns(),
          'q':[0.]*14,'dq':[0.]*14,'ddq':[0.]*14,'target_sequence':-1}
    goal=np.zeros(14);goal[0]=.1
    first=solver.command_step(goal,np.zeros(14),np.zeros(14),lambda:None,seed)
    second=solver.command_step(-goal,np.zeros(14),np.zeros(14),lambda:None,seed)
    assert first[0]>0 and second[0]<0
    assert first[0] == pytest.approx(-second[0])
    assert solver.trajectory.diagnostics['state_source']=='driver_stationary_reference'


def test_braking_outside_joint_limits_is_rejected_even_if_target_is_inside():
    trajectory=JointTrajectory({'enabled':True,'max_acceleration_rad_s2':2,'max_jerk_rad_s3':20},.2)
    q=np.zeros(14);q[0]=.999
    goal=q.copy();goal[0]=.9995
    state={'schema':'motus.command-state.v1','kind':'motion','sample_ns':time.monotonic_ns(),
           'q':q.tolist(),'dq':[.2]+[0.]*13,'ddq':[0.]*14,'target_sequence':2}
    with pytest.raises(ValueError,match='braking_envelope'):
        trajectory.propose(goal,q,q,state,np.full(14,-1.),np.full(14,1.),lambda:None)
    assert trajectory.pending is None


@pytest.mark.parametrize('change', [dict(limited=True), dict(dt_s=.021), dict(target_sequence=-1)])
def test_limited_delayed_or_unconfirmed_sample_rebases_from_driver(change):
    trajectory=JointTrajectory({'enabled':True,'max_acceleration_rad_s2':2,'max_jerk_rad_s3':20},.2)
    zero=np.zeros(14)
    goal=zero.copy();goal[0]=.1
    seed={'schema':'motus.command-state.v1','kind':'stationary_seed','sample_ns':time.monotonic_ns(),
          'q':zero.tolist(),'target_sequence':-1}
    q,_,_,plan=trajectory.propose(goal,zero,zero,seed,np.full(14,-1.),np.full(14,1.),lambda:None)
    trajectory.commit(plan)
    state={**seed,'kind':'motion','q':q.tolist(),'dq':zero.tolist(),'ddq':zero.tolist(),
           'target_sequence':0,'limited':False,'dt_s':.02,**change}
    _,_,_,next_plan=trajectory.propose(goal,q,q,state,np.full(14,-1.),np.full(14,1.),lambda:None)
    assert next_plan['diagnostics']['state_source']=='driver_finite_difference'


def test_optional_image_build_flag_and_invalid_arguments(tmp_path):
    import subprocess
    script=Path(__file__).parents[2]/'deploy/build_tianyi_actucore.sh'
    docker=tmp_path/'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env={**os.environ,'PATH':str(tmp_path)+':/usr/bin:/bin'}
    for extra,enabled in (([], 'false'), (['--trajectory'], 'true')):
        run=subprocess.run(['bash',str(script),'offline-test',*extra],env=env,capture_output=True,text=True)
        assert run.returncode==0 and 'INSTALL_TRAJECTORY='+enabled in run.stdout
        assert 'linux/arm64' in run.stdout
    assert subprocess.run(['bash',str(script),'offline-test','--bad'],env=env,capture_output=True).returncode==2
