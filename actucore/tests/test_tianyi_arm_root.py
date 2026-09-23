"""Real vendor model FK comparisons; no robot transport or hardware writes."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'plugins'))
from teleop.kinematics import TianyiIK, arm_chain_xml

BASE = Path(__file__).parents[1] / 'plugins/teleop'


def profile(tmp_path):
    p = json.loads((BASE / 'calibration.example.json').read_text())
    urdf = BASE / 'models/tianyi2-official.urdf'
    p.update(urdf_path=str(urdf), urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest())
    p['palm_frames'] = {s: {'position': [0, 0, -.084799], 'orientation': [0, 0, 0, 1]}
                        for s in ('left', 'right')}
    # Workspace is a test fixture, not a physical acceptance record.
    p['workspace'] = {'torso_box': [[-.1, -.1, 0], [.1, .1, .3]],
                      'left': [[-1, -1, -1], [1, 1, 1]],
                      'right': [[-1, -1, -1], [1, 1, 1]],
                      'capsules': [{'from': f'elbow_pitch_{s}_link', 'to': f'wrist_roll_{s}_link',
                                    'radius_m': .05, 'group': s} for s in ('l', 'r')]}
    path = tmp_path / 'site.json'
    path.write_text(json.dumps(p))
    return path, p, urdf


def test_chest_fk_matches_full_model_with_varying_legs(tmp_path):
    path, p, urdf = profile(tmp_path)
    solver = TianyiIK(path)  # null locked joints are deliberately not calibrated
    full = pin.buildModelFromUrdf(str(urdf))
    data = full.createData()
    rng = np.random.default_rng(42)
    for _ in range(20):
        q = pin.neutral(full)
        # Vary all non-arm ancestors too; chest-relative FK must be invariant.
        for joint in full.joints[1:]:
            q[joint.idx_q] = rng.uniform(-.25, .25)
        arms = np.array([q[full.joints[full.getJointId(n)].idx_q] for n in p['arm_joint_names']])
        pin.framesForwardKinematics(full, data, q)
        torso = data.oMf[full.getFrameId(p['torso_frame'])].inverse()
        for index, actual in enumerate(solver.palms(arms)):
            wrist = full.getJointId(p['arm_joint_names'][6 if index == 0 else 13])
            expected = torso * data.oMi[wrist] * pin.SE3(np.eye(3), np.array([0, 0, -.084799]))
            np.testing.assert_allclose(actual, expected.homogeneous, atol=1e-12)
    assert solver.model.nq == 14
    assert not any('leg' in n or 'head' in n or 'thumb' in n for n in solver.model.names)
    assert solver.capsules  # collision checks remain configured


def test_out_of_range_ancestor_does_not_change_arm_fk(tmp_path):
    path, p, _ = profile(tmp_path)
    original = TianyiIK(path).palms(np.zeros(14))
    p['locked_joints'] = {'first_leg_pitch_joint': -.663}
    path.write_text(json.dumps(p))
    np.testing.assert_allclose(TianyiIK(path).palms(np.zeros(14)), original)


def test_extra_movable_joint_on_arm_path_is_rejected(tmp_path):
    _, p, urdf = profile(tmp_path)
    root = ET.fromstring(urdf.read_bytes())
    p['arm_joint_names'][0] = 'head_yaw_joint'
    with pytest.raises(ValueError, match='unexpected_movable_arm_joint'):
        arm_chain_xml(ET.tostring(root), p['torso_frame'], p['arm_joint_names'])


def test_wrong_torso_is_rejected(tmp_path):
    _, p, urdf = profile(tmp_path)
    with pytest.raises(ValueError, match='unexpected_movable_arm_joint'):
        arm_chain_xml(urdf.read_bytes(), 'wrist_roll_l_link', p['arm_joint_names'])


def test_vendor_torso_spheres_keep_collision_and_swept_checks(tmp_path):
    path, p, _ = profile(tmp_path)
    workspace = p['workspace']
    workspace.pop('torso_box')
    workspace['torso_spheres'] = [{'center_m': [0, 0, z], 'radius_m': .16}
                                 for z in (.15, .23, .30)]
    workspace['capsules'] += [
        {'from': f'shoulder_roll_{s}_link', 'to': f'elbow_pitch_{s}_link',
         'radius_m': .05, 'group': s} for s in ('l', 'r')]
    path.write_text(json.dumps(p))
    solver = TianyiIK(path)
    q = np.zeros(14)
    assert solver.self_test(q)['hardware_output'] is False
    with pytest.raises(ValueError, match='torso_collision'):
        solver._safe_configuration(q, excursion=np.full(14, .5))
    # Place a sphere at the middle of an arm segment: collision must reject.
    a, b, _, _ = solver.capsules[0]
    center = (solver.data.oMf[a].translation + solver.data.oMf[b].translation) / 2
    solver.torso_spheres = [(center, .16)]
    with pytest.raises(ValueError, match='torso_collision'):
        solver._safe_configuration(q)


@pytest.mark.parametrize('reordered', [False, True])
def test_analytic_jacobian_matches_original_objective(tmp_path, reordered):
    path, p, _ = profile(tmp_path)
    if reordered:
        p['arm_joint_names'] = p['arm_joint_names'][7:] + p['arm_joint_names'][:7]
        path.write_text(json.dumps(p))
    solver = TianyiIK(path)
    rng = np.random.default_rng(91)
    for _ in range(8):
        measured = rng.uniform(-.2, .2, 14)
        q = rng.uniform(-.3, .3, 14)
        targets = solver.palms(rng.uniform(-.4, .4, 14))
        def original(x):
            actual = solver.palms(x)
            return np.concatenate([np.concatenate((7*(a[:3,3]-b[:3,3]),
                pin.log3(b[:3,:3].T@a[:3,:3]))) for a,b in zip(actual,targets)] + [.01*(x-measured)])
        residual, jacobian = solver._residual_jacobian(q, targets, measured)
        np.testing.assert_allclose(residual, original(q), atol=1e-12)
        epsilon = 1e-6
        numerical = np.column_stack([(original(q+epsilon*axis)-original(q-epsilon*axis))/(2*epsilon)
                                     for axis in np.eye(14)])
        np.testing.assert_allclose(jacobian, numerical, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("phase", ["expired", "residual", "collision"])
def test_input_deadline_limits_entire_solve_without_refreshing_history(tmp_path, monkeypatch, phase):
    from types import SimpleNamespace
    import teleop.kinematics as module
    path, _, _ = profile(tmp_path)
    solver = TianyiIK(path)
    q = np.zeros(14)
    solver.self_test(q)
    history = solver.last_valid_visualization
    targets = solver.palms(q)
    now = [10.]
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=lambda: int(now[0]*1e9)))
    if phase != "expired":
        name = "_residual_jacobian" if phase == "residual" else "_safe_configuration"
        original = getattr(solver, name)
        def late(*args, **kwargs):
            value = original(*args, **kwargs)
            now[0] = 10.04
            return value
        monkeypatch.setattr(solver, name, late)
    with pytest.raises(ValueError, match="ik_timeout"):
        solver.solve(targets, q, deadline_monotonic=10. if phase == "expired" else 10.03)
    assert solver.last_valid_visualization is history
    assert solver.visualization_sample["ik_q"] is None
    monkeypatch.undo()
    solver.solve(targets, q)  # cooperative timeout does not poison the next solve


def test_axis_bounds_cover_vendor_fk_across_joint_configurations(tmp_path):
    path,_,_=profile(tmp_path);s=TianyiIK(path)
    bounds=[(f,b) for f,b in zip(s.frames,s.palm_sweep)]
    bounds += [(f,b) for capsule,b in zip(s.capsules,s.sweep_coefficients) for f in capsule[:2]]
    precise=[b.copy() for b in s.palm_sweep]
    s.axis_aware_sweep=False;s.configure_workspace()
    assert any(np.any(a<b-1e-5) for a,b in zip(precise,s.palm_sweep))
    def positions(q):
        model_q=np.empty(14);model_q[s.indices]=q
        pin.framesForwardKinematics(s.model,s.data,model_q)
        return [s.data.oMf[f].translation.copy() for f,_ in bounds]
    rng=np.random.default_rng(20260922)
    for _ in range(400):
        a=rng.uniform(-2.5,2.5,14);delta=rng.uniform(-.3,.3,14)
        pa,pb=positions(a),positions(a+delta)
        for x,y,(_,bound) in zip(pa,pb,bounds):
            assert np.linalg.norm(x-y)<=bound@np.abs(delta)+1e-9


def test_parent_box_proofs_are_local_and_cover_sampled_children(tmp_path):
    path,_,_=profile(tmp_path);s=TianyiIK(path)
    original=s._safe_configuration
    rng=np.random.default_rng(713);accepted=rejected=0
    for _ in range(100):
        start=rng.uniform(-.4,.4,14);end=start+rng.uniform(-.12,.12,14)
        try:s._safe_transition(start,end)
        except ValueError:
            rejected+=1;continue
        accepted+=1
        for _ in range(12):
            original(rng.uniform(np.minimum(start,end),np.maximum(start,end)))
    assert accepted and rejected
    with pytest.raises(ValueError):
        s._safe_transition(np.full(14,2.),np.full(14,2.1))


def test_interval_balls_cover_independent_vendor_joint_motion(tmp_path):
    path,_,_=profile(tmp_path);s=TianyiIK(path);rng=np.random.default_rng(914)
    frames=list(s._endpoint_paths)
    def fk(q):
        model_q=np.empty(14);model_q[s.indices]=q
        pin.framesForwardKinematics(s.model,s.data,model_q)
        return [s.data.oMf[f].translation.copy() for f in frames]
    for radius_max in (.1,.5,4.):
        for _ in range(40):
            q=rng.uniform(-2.,2.,14);radius=rng.uniform(0,radius_max,14)
            original=fk(q);bounds=[s._endpoint_excursion(f,radius) for f in frames]
            for _ in range(12):
                actual=fk(q+rng.uniform(-radius,radius))
                assert all(np.linalg.norm(a-b)<=r+1e-9 for a,b,r in zip(original,actual,bounds))


@pytest.mark.parametrize('velocity,offset',[(.2,.00383),(1.,.019)])
def test_real_ik_initial_settling_offset_keeps_measured_pose_and_strict_targets(tmp_path,velocity,offset):
    path,p,_=profile(tmp_path);p['joint_velocity_rad_s']=velocity;path.write_text(json.dumps(p));s=TianyiIK(path)
    lower=s.model.lowerPositionLimit[s.indices];upper=s.model.upperPositionLimit[s.indices]
    q=np.zeros(14);q[5]=lower[5]-offset;q[6]=upper[6]+offset
    original=q.copy();targets=s.palms(q)
    result=np.asarray(s.solve(targets,q))
    np.testing.assert_array_equal(q,original)
    assert np.all(result>=lower) and np.all(result<=upper)
    assert np.max(np.abs(result-original))<=s.velocity*.02+1e-9
    q[5]=lower[5]-s.velocity*.02-.00001
    with pytest.raises(ValueError,match='joint_feedback_out_of_bounds'):s.solve(s.palms(q),q)



def test_analytic_segment_distance_against_independent_convex_optimizer():
    from scipy.optimize import lsq_linear
    from teleop.workspace import segment_distance_lower
    cases=[[[0,0,0],[1,0,0],[.5,-1,0],[.5,1,0]],
           [[0,0,0],[1,0,0],[2,.1,0],[3,.1,0]],
           [[0,0,0],[1,0,0],[.2,.1,0],[.8,.1,0]],
           [[0,0,0],[0,0,0],[1,0,0],[1,0,0]],
           [[0,0,0],[1,0,0],[0,1e-8,0],[1,-1e-8,0]]]
    cases+=list(np.random.default_rng(143).normal(size=(200,4,3)))
    for case in cases:
        a,b,c,d=np.asarray(case,float)
        matrix=np.column_stack([b-a,c-d]);offset=a-c
        result=lsq_linear(matrix,-offset,bounds=(0,1),tol=1e-13,max_iter=200)
        reference=float(np.linalg.norm(matrix@result.x+offset))
        bound=segment_distance_lower(a,b,c,d)
        assert 0<=bound<=reference+1e-8
        assert reference-bound<3e-6


def test_tianyi_capsule_distance_preserves_physical_margins(tmp_path):
    from teleop.workspace import segment_distance_lower
    p,_,_=profile(tmp_path);solver=TianyiIK(p)
    assert solver.analytic_capsule_distance
    # Parallel 10 cm capsules with 5 mm safety on EACH arm touch at 21 cm.
    a=np.array([0.,0.,0.]);b=np.array([1.,0.,0.])
    for distance,collision in ((.209,True),(.21,True),(.211,False)):
        c=a+np.array([0.,distance,0.]);d=b+np.array([0.,distance,0.])
        assert (segment_distance_lower(a,b,c,d)<.1+.005+.1+.005)==collision



@pytest.mark.parametrize('case',['continuous','stale','new_reference'])
def test_warm_initial_guess_never_replaces_current_feedback(tmp_path,case):
    import time
    path,_,_=profile(tmp_path);solver=TianyiIK(path)
    measured=np.zeros(14);goal=measured.copy();goal[0]=.08;goal[7]=.04
    targets=solver.palms(goal);solver.solve(targets,measured)
    previous=solver.last_valid_visualization['ik_q'].copy()
    if case=='stale':solver.last_valid_visualization['monotonic_ns']=time.monotonic_ns()-300_000_000
    if case=='new_reference':targets=[t.copy() for t in targets];targets[0][0,3]+=.06
    actual=measured.copy();actual[0]=.003
    initial=[];anchors=[];original=solver.least_squares;residual=solver._residual_jacobian
    def solve(fun,x,**kw):initial.append(x.copy());return original(fun,x,**kw)
    def observe(q,targets,anchor):anchors.append(anchor.copy());return residual(q,targets,anchor)
    solver.least_squares=solve;solver._residual_jacobian=observe
    try:result=solver.solve(targets,actual)
    except ValueError:
        assert case=='new_reference'
    else:assert np.max(np.abs(np.asarray(result)-actual))<=solver.velocity*.02+1e-12
    np.testing.assert_allclose(initial[0],previous if case=='continuous' else actual)
    assert anchors
    for anchor in anchors:np.testing.assert_array_equal(anchor,actual)


@pytest.mark.parametrize('case',['smaller_safe','all_unsafe','timeout','lead_bound'])
def test_safe_advance_revalidates_both_paths_without_relaxing_bounds(tmp_path,case):
    from teleop.workspace import WorkspaceViolation
    path,_,_=profile(tmp_path);s=TianyiIK(path);s.velocity=1.
    measured=np.zeros(14);previous=np.zeros(14);candidate=np.zeros(14)
    previous[0]=.1;candidate[0]=.12;checked=[];ticks=[]
    if case=='lead_bound':previous[0]=.3;candidate[0]=.2
    def budget():
        ticks.append(1)
        if case=='timeout' and len(ticks)>2:raise ValueError('ik_timeout')
    def transition(start,end,budget):
        budget();checked.append((start.copy(),end.copy()))
        if case=='all_unsafe' or end[0]>.111 or (case=='lead_bound' and end[0]>=.2):
            raise WorkspaceViolation('torso_collision',np.ones(14))
    s._safe_transition=transition
    if case in ('all_unsafe','lead_bound','timeout'):
        with pytest.raises(ValueError,match='ik_timeout' if case=='timeout' else 'torso_collision'):
            s._safe_advance(measured,previous,candidate,budget)
        if case=='lead_bound':assert len(checked)==1  # smaller interpolation would exceed lead
    else:
        result,scale=s._safe_advance(measured,previous,candidate,budget)
        assert scale==.5 and result[0]==pytest.approx(.11)
        np.testing.assert_array_equal(checked[-2][0],measured)
        np.testing.assert_array_equal(checked[-1][0],previous)
        np.testing.assert_array_equal(candidate,np.array([.12]+[0.]*13))
        assert max(abs(result-measured))<=.2 and max(abs(result-previous))<=.02


def test_real_ik_preserves_fixed_twenty_ms_increment(tmp_path):
    path,_,_=profile(tmp_path);measured=np.zeros(14);goal=measured.copy()
    goal[0]=.08;goal[7]=.04
    solver=TianyiIK(path)
    result=np.asarray(solver.solve(solver.palms(goal),measured))
    assert np.max(np.abs(result))<=solver.velocity*.02+1e-10
