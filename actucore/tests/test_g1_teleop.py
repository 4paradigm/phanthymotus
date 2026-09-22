"""PR152 model/IK and G1 contracts; no robot I/O.

Only tests using ``g1_numeric_abi`` need the G1 Pinocchio/CasADi environment.
Pure mapping, metadata and Driver contracts also run in the Tianyi environment.
"""
import copy
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.g1 import G1IK,G1IntentAdapter,CAPABILITIES_G1,PROFILE_ID
from teleop.descriptor import capability_digest
from teleop.runtime import TeleopRuntime
from teleop.dispatch import RecordingAdapter,MotionIntent
from test_teleop import frame

ROOT=Path(__file__).parents[1]/'plugins/teleop'


@pytest.fixture(scope='module')
def g1_numeric_abi():
    # A plain Pinocchio import does not prove its optional symbolic ABI exists.
    # Limit skipping to dependency imports; solver/model failures must still fail.
    for module_name in ('casadi', 'pinocchio.casadi'):
        try:
            importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            pytest.skip(
                'G1 numeric test requires the G1 Pinocchio/CasADi ABI; '
                f'{module_name} is unavailable: {exc}. '
                'The Tianyi numeric environment does not provide G1 IK.'
            )


@pytest.mark.parametrize('invalid',[None,'transport_timeout','old','session','unconfirmed','fault'])
def test_pause_accepts_only_matching_post_request_hold_even_before_http_reply(invalid):
    import threading
    from teleop.adapter import DriverLink
    link=DriverLink.__new__(DriverLink);link.condition=threading.Condition()
    link.lease={'boot_id':'boot','session_id':'session'}
    before=time.monotonic_ns()
    def call(action,deadline):
        assert action=='pause'
        link.latest=dict(boot_id='boot',session_id='session',state='hold',hold_confirmed=True,
                         monotonic_ns=time.monotonic_ns())
        if invalid=='old':link.latest['monotonic_ns']=before-1
        if invalid=='session':link.latest['session_id']='previous'
        if invalid=='unconfirmed':link.latest['hold_confirmed']=False
        if invalid=='fault':link.latest['state']='fault'
        if invalid=='transport_timeout':raise TimeoutError('late HTTP response')
    link.call=call
    if invalid in ('old','session'):
        with pytest.raises(ValueError):link.pause(time.monotonic()+.025)
    else:
        assert link.pause(time.monotonic()+.025)==(invalid in (None,'transport_timeout'))


def test_driver_claim_preserves_refusal_reason_without_hardware():
    import io
    from teleop.adapter import DriverLink
    link=object.__new__(DriverLink);link.url='http://127.0.0.1:15701/mcp';link.lease=None
    for reason in ('arms_not_stationary','external_arm_publisher','live_acceptance_missing'):
        reply={'result':{'content':[{'type':'text','text':json.dumps({'state':'error','error':reason})}]}}
        link.opener=SimpleNamespace(open=lambda *a,**kw:io.BytesIO(json.dumps(reply).encode()))
        with pytest.raises(ValueError,match='^'+reason+'$'):link.call('claim',time.monotonic()+1)
        assert link.lease is None

def calibration(tmp_path):
    p=json.loads((ROOT/'g1.calibration.example.json').read_text())
    p['urdf_path']=str(ROOT/'models/g1_body23.urdf')
    p['version']='SYNTHETIC-NOT-ACCEPTED'
    p['locked_joints']=dict.fromkeys(p['locked_joints'],0.)
    for side in ('left','right'):
        p['palm_frames'][side]={'position':[.2,0.,0.],'orientation':[0.,0.,0.,1.]}
        p['controller_to_palm'][side]={'position':[0.,0.,0.],'orientation':[0.,0.,0.,1.]}
    p['workspace']={'torso_box':[[-.05,-.05,-.05],[.05,.05,.05]],
                    'left':[[-1.,-1.,-1.],[1.,1.,1.]],'right':[[-1.,-1.,-1.],[1.,1.,1.]],
                    'capsules':[{'from':s+'_elbow_joint','to':s+'_wrist_roll_joint','radius_m':.025,'group':s} for s in ('left','right')]}
    path=tmp_path/'g1.json';path.write_text(json.dumps(p));return path


@pytest.mark.parametrize('velocity',[.6,5.,5.001,float('inf')])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_configured_speed_ceiling(tmp_path,velocity):
    path=calibration(tmp_path);p=json.loads(path.read_text());p['joint_velocity_rad_s']=velocity
    path.write_text(json.dumps(p))
    if velocity in (.6,5.):assert G1IK(path).velocity==velocity
    else:
        with pytest.raises(ValueError,match='joint_velocity_limit'):G1IK(path)


@pytest.mark.usefixtures("g1_numeric_abi")
def test_reachable_step_is_not_rejected_for_filter_lag(tmp_path):
    solver=G1IK(calibration(tmp_path))
    q=np.array([.243,.297,-.079,.805,-.005,.246,-.298,.083,.777,.007])
    solver.self_test(q)
    targets=solver.palms(q)
    for _ in range(4):solver.solve(targets,q,dq=np.zeros(10))
    for target in targets:target[0,3]+=.04
    result,_=solver.solve(targets,q,dq=np.zeros(10))
    assert max(r['position_m'] for r in solver.residual)>.015
    assert np.max(np.abs(np.array(result)-q))<=.00400001
    for target in targets:target[0,3]+=2
    with pytest.raises((ValueError,RuntimeError)):
        solver.solve(targets,q,dq=np.zeros(10))


@pytest.mark.usefixtures("g1_numeric_abi")
def test_rejected_target_snapshot_survives_hold_and_is_detached(tmp_path):
    solver=G1IK(calibration(tmp_path))
    q=np.array([.257,.297,-.080,.820,-.006,.250,-.303,.082,.780,.007])
    targets=solver.palms(q)
    for target in targets:target[2,3]-=.04
    # Even a solver claiming success outside the constraint must be rejected.
    solver.ik._opti=SimpleNamespace(set_initial=lambda *a:None,set_value=lambda *a:None,
        solve=lambda:SimpleNamespace(value=lambda *a:q.copy()))
    with pytest.raises(ValueError,match='ik_target_unreachable'):
        solver.solve(targets,q,dq=np.zeros(10))
    adapter=G1IntentAdapter(SimpleNamespace(),'shadow');adapter.solver=solver
    diagnostic=adapter.snapshot()['diagnostics']['ik']
    failure=diagnostic['last_rejection']
    assert max(failure['position_error_m'])>.015
    assert failure['targets']==[target.tolist() for target in targets]
    assert failure['measured_q']==q.tolist()
    assert diagnostic['history_depth']==0
    json.dumps(diagnostic,allow_nan=False)
    failure['targets'][0][0][3]=999
    solver.ik.reset(q)
    assert solver.ik.snapshot()['last_rejection']['targets'][0][0][3]!=999


def test_g1_capability_binding_has_no_hands():
    runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),capabilities=CAPABILITIES_G1,auto_watchdog=False)
    try:
        assert runtime.profile_id==PROFILE_ID
        assert runtime.capabilities['outputs']['dual_arm']['joint_count']==10
        assert runtime.capabilities['effectors']==['dual_arm']
        assert runtime.capability_digest!=capability_digest('shadow')
    finally:runtime.close()


@pytest.mark.usefixtures("g1_numeric_abi")
def test_real_pr152_ik_bounded_targets_collision_and_unreachable(tmp_path):
    solver=G1IK(calibration(tmp_path));q=np.zeros(10)
    assert solver.self_test(q)['ready']
    targets=solver.palms(q)
    times=[]
    for _ in range(10):
        result,tau=solver.solve(targets,q,dq=np.zeros(10));times.append(solver.last_ms)
        assert len(result)==len(tau)==10 and np.isfinite(tau).all()
        assert max(abs(np.array(result)-q))<=.00400001
    print('G1 IK P95 ms',np.percentile(times,95))
    solver.workspace['torso_box']=[[-2.,-2.,-2.],[2.,2.,2.]]
    with pytest.raises(ValueError,match='torso_collision'):solver.solve(targets,q,dq=np.zeros(10))
    solver.workspace['torso_box']=[[-.05,-.05,-.05],[.05,.05,.05]]
    targets[0][0,3]+=2
    with pytest.raises((ValueError,RuntimeError)):solver.solve(targets,q,dq=np.zeros(10))


@pytest.mark.usefixtures("g1_numeric_abi")
def test_fixed_hand_tip_collision_cannot_be_omitted_from_profile(tmp_path):
    solver=G1IK(calibration(tmp_path));q=np.zeros(10)
    solver._safe_configuration(q)
    wrist=solver.model.getFrameId('left_wrist_roll_joint')
    pose=solver.data.oMf[solver.torso].inverse()*solver.data.oMf[wrist]
    tip=pose.translation+pose.rotation@np.array([.25,0.,0.])
    solver.workspace['torso_box']=[(tip-.001).tolist(),(tip+.001).tolist()]
    # The previous forearm-only implementation misses this actual rubber-hand region.
    capsules=solver.capsules
    solver.capsules=capsules[:2]
    solver._safe_configuration(q)
    solver.capsules=capsules
    with pytest.raises(ValueError,match='torso_collision'):
        solver._safe_configuration(q)
    with pytest.raises(ValueError,match='torso_collision'):
        solver._safe_configuration(q,np.full(10,.004))


@pytest.mark.usefixtures("g1_numeric_abi")
def test_automatic_hands_do_not_replace_missing_arm_calibration(tmp_path):
    path=calibration(tmp_path);p=json.loads(path.read_text())
    p['workspace']['capsules']=[];path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='collision_calibration_missing'):G1IK(path)


@pytest.mark.usefixtures("g1_numeric_abi")
def test_upper_arm_torso_and_swept_geometry_are_checked(tmp_path):
    from teleop.workspace import ArmWorkspace
    solver=G1IK(calibration(tmp_path));q=np.zeros(10)
    assert len(solver.body_collision.objects)==11
    assert len(solver.body_collision.pairs)==45
    solver._safe_configuration(q)
    q[1]=-.05
    ArmWorkspace._safe_configuration(solver,q)  # The old forearm/hand checks miss this.
    with pytest.raises(ValueError,match='g1_body_collision:torso_link:left_shoulder_yaw_link'):
        solver._safe_configuration(q)
    q[1]=0.;excursion=np.zeros(10);excursion[1]=.02
    solver._safe_configuration(q)
    with pytest.raises(ValueError,match='g1_body_collision'):
        solver._safe_configuration(q,excursion)


class Link:
    def __init__(self):self.lease=None;self.calls=[];self.sha=None;self.stale=False
    def feedback(self):return {'profile_id':PROFILE_ID,'calibration_sha256':self.sha,'feedback':{'q':[0.]*10,'dq':[0.]*10,'arm_ns':time.monotonic_ns()-(200_000_000 if self.stale else 0)}}
    def claim(self,*a):self.calls.append('claim');self.lease={}
    def send(self,*a,**kw):self.calls.append('send')
    def resume(self,*a):self.calls.append('resume')
    def stop(self,*a):self.calls.append('stop');return True


def intent(f,seq=1):
    now=time.monotonic()
    return SimpleNamespace(expires_monotonic=now+.1,received_monotonic=now,admitted_monotonic=now,dispatch_generation=seq,session_generation=1,clutch_sequence=seq,frame=f)


@pytest.mark.usefixtures("g1_numeric_abi")
def test_real_ik_shadow_has_zero_execution_calls_and_stale_feedback_fails(tmp_path):
    link=Link();adapter=G1IntentAdapter(link,'shadow');adapter.calibrate(calibration(tmp_path))
    assert adapter.apply(intent(frame(0,True,1))).ok
    assert len(adapter.output['target_q'])==10 and not link.calls and not link.lease
    # Triggers are irrelevant for G1 arm-only; no mandatory neutral-hand step.
    f=frame(1,True,2);f['controllers']['left']['buttons'][0]=1.
    assert adapter.apply(intent(f,2)).ok
    link.stale=True
    assert not adapter.apply(intent(f,3)).ok
    assert adapter.output['code']=='arm_feedback_stale'


def test_model_digest_and_unknown_profile_rejected(tmp_path):
    path=calibration(tmp_path);p=json.loads(path.read_text());p['urdf_sha256']='0'*64;path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='model_changed'):G1IK(path)


@pytest.mark.usefixtures("g1_numeric_abi")
def test_planning_rejections_never_reach_driver(tmp_path):
    from teleop.dispatch import IK_HOLD_CODES
    link=Link();adapter=G1IntentAdapter(link,'shadow');adapter.calibrate(calibration(tmp_path))
    for code in sorted(IK_HOLD_CODES) + ['invalid_finite_shape']:
        def reject(*args,**kwargs):
            raise ValueError(code+(':torso:left_arm' if code=='g1_body_collision' else ''))
        adapter.solver.solve=reject
        ack=adapter.apply(intent(frame(0,True,1)))
        assert not ack.ok
        assert ack.code==(code if code in IK_HOLD_CODES else 'motion_rejected')
        assert not link.calls and not link.lease


def test_identical_config_does_not_disconnect_active_card():
    from teleop.plugin import TeleopPlugin
    cfg={'robot_profile':'g1_23','mode':'shadow'}
    plugin=TeleopPlugin(cfg,None)
    plugin.runtime=SimpleNamespace(status=lambda:{'authority_valid':True})
    plugin.link=SimpleNamespace(lease={'test':True})
    plugin.info=lambda:{'state':'active'}
    assert plugin.dispatch('teleop',{'action':'config',**cfg})=={'state':'active'}


def test_latest_mailbox_runs_at_50hz_and_pause_preempts_wait():
    from teleop.protocol import bind_rtc_frame_v1
    class Timed(RecordingAdapter):
        def __init__(self):super().__init__();self.times=[]
        def apply(self,intent):
            self.times.append(time.monotonic())
            return super().apply(intent)
    adapter=Timed()
    runtime=TeleopRuntime(mode='shadow',adapter=adapter,motion_interval_ms=20,auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(f):runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='shadow'),source='test')
        send(frame(0));send(frame(1,True,1))
        assert runtime._dispatcher.wait_dispatched(1)
        for seq in range(2,42):send(frame(seq,True,1));time.sleep(.001)
        assert runtime._dispatcher.wait_dispatched(41)
        assert 2<=len(adapter.times)<15
        assert min(b-a for a,b in zip(adapter.times,adapter.times[1:]))>=.018
        send(frame(42,True,1));send(frame(43))
        assert runtime.status()['state']=='hold'
        count=len(adapter.times);time.sleep(.03)
        assert len(adapter.times)==count
    finally:runtime.close()


@pytest.mark.usefixtures("g1_numeric_abi")
def test_position_constraint_handles_captured_orientation_tradeoff(tmp_path):
    solver=G1IK(calibration(tmp_path))
    q=np.array([0.25462883710861206, 0.28413400053977966, -0.07780159264802933, 0.8147715330123901, -0.0055367122404277325, 0.253550261259079, -0.3016189932823181, 0.08356600254774094, 0.7790704965591431, 0.00672315014526248])
    targets=tuple(np.array(t) for t in [[[0.23309103460203612, -0.06150003385450179, 0.9705082768446892, 0.09974294107633096], [0.3313301312048376, 0.943307097207193, -0.019800619038929556, 0.2691383514846893], [-0.9142696067046843, 0.32617398147987675, 0.240253241521582, -0.16025619406088595], [0.0, 0.0, 0.0, 1.0]], [[0.2646870311466649, 0.5688179750362726, 0.7787084735755732, 0.13946358143527407], [-0.7223680605889157, 0.6519062434406506, -0.23065696348497272, -0.27281902985240264], [-0.6388467426415927, -0.5014622229427994, 0.5834470656173473, -0.15298687589578824], [0.0, 0.0, 0.0, 1.0]]])
    result,_=solver.ik.solve(*targets,q,np.zeros(10))
    actual=solver.palms(result)
    assert all(np.linalg.norm(a[:3,3]-b[:3,3]) <= .015 for a,b in zip(actual,targets))
    assert solver.ik.snapshot()["last_rejection"] is None
    targets[0][0,3]+=2
    history_depth=solver.ik.snapshot()['history_depth']
    with pytest.raises(ValueError,match='ik_not_converged'):
        solver.ik.solve(*targets,q,np.zeros(10))
    failure=solver.ik.snapshot()['last_rejection']
    assert failure['code']=='ik_not_converged'
    assert failure['solver_status'] not in ('unavailable','Solve_Succeeded')
    assert isinstance(failure['iterations'],int)
    assert failure['targets']==[t.tolist() for t in targets]
    assert 'raw_q' not in failure
    assert solver.ik.snapshot()['history_depth']==history_depth
    json.dumps(failure,allow_nan=False)


@pytest.mark.parametrize('owned', [False, True])
@pytest.mark.parametrize('reason', ['arms_not_stationary', 'external_arm_publisher', 'timed out'])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_only_unowned_stationary_claim_refusal_is_recoverable(tmp_path, owned, reason):
    link=Link();adapter=G1IntentAdapter(link,'live');adapter.calibrate(calibration(tmp_path))
    link.sha=adapter.solver.profile_sha256
    def refuse(*args):
        if owned:link.lease={'session_id':'test'}
        raise ValueError(reason)
    link.claim=refuse
    ack=adapter.apply(intent(frame(0,True,1)))
    assert not ack.ok
    assert ack.code==('arms_not_stationary' if reason=='arms_not_stationary' and not owned else 'motion_rejected')
    assert 'send' not in link.calls


@pytest.mark.parametrize('reason', ['arms_not_stationary','external_arm_publisher','timed out'])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_owned_resume_settling_is_hold_without_sending(tmp_path,reason):
    link=Link();adapter=G1IntentAdapter(link,'live');adapter.calibrate(calibration(tmp_path))
    link.sha=adapter.solver.profile_sha256
    lease={'session_id':'existing'};link.lease=lease
    def refuse(*args):raise ValueError(reason)
    link.resume=refuse
    ack=adapter.apply(intent(frame(0,True,1)))
    assert not ack.ok
    assert ack.code==('arms_not_stationary' if reason=='arms_not_stationary' else 'motion_rejected')
    assert link.lease is lease and not link.calls and adapter.clutch is None
    link.resume=lambda *a:link.calls.append('resume')
    assert adapter.apply(intent(frame(1,True,2))).ok
    assert 'resume' in link.calls and 'send' not in link.calls


@pytest.mark.parametrize('action', ['send', 'pause', 'stop'])
@pytest.mark.parametrize('fresh_wrong_session', [False, True])
def test_driver_ack_waits_for_post_command_feedback(monkeypatch, action, fresh_wrong_session):
    from teleop.adapter import DriverLink
    monkeypatch.setitem(sys.modules,'std_msgs.msg',SimpleNamespace(String=SimpleNamespace))
    link=object.__new__(DriverLink)
    link.lease={'session_id':'new','boot_id':'boot','secret':'00'*32};link.seq=0
    link.release_requested_ns=0;link.management_request=None
    link.call=lambda *a:None
    old={'monotonic_ns':time.monotonic_ns()-10_000_000,'boot_id':'boot',
         'session_id':'old','state':'hold','hold_confirmed':True,'stop_confirmed':True,
         'ownership_held':False,'applied_sequence':100}
    link.latest=old;link.feedback=lambda:link.latest
    class Condition:
        waits=0
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def wait(self,*a):
            self.waits+=1
            link.latest={**old,'monotonic_ns':time.monotonic_ns(),
                         'boot_id':'other' if fresh_wrong_session else 'boot',
                         'session_id':'new','state':'hold' if action=='pause' else 'active','applied_sequence':1}
    link.condition=Condition();published=[]
    link.publisher=SimpleNamespace(publish=lambda msg:published.append(msg.data))
    def invoke():
        deadline=time.monotonic()+.1
        return link.send([0.]*10,None,deadline) if action=='send' else getattr(link,action)(deadline)
    if fresh_wrong_session:
        with pytest.raises(ValueError,match='driver_(restarted|session_changed)'):invoke()
        assert link.lease is not None
    else:
        invoke()
        if action=='stop':assert link.lease is None
    assert link.condition.waits==(2 if action=='stop' and not fresh_wrong_session else 1)
    assert len(published)==(1 if action=='send' else 0)


def test_old_feedback_cannot_outlive_command_deadline():
    import threading
    from teleop.adapter import DriverLink
    link=object.__new__(DriverLink);link.condition=threading.Condition()
    stamp=time.monotonic_ns();link.feedback=lambda:{'monotonic_ns':stamp-1}
    with pytest.raises(ValueError,match='driver_feedback_ack_timeout'):
        link.feedback_after(stamp,time.monotonic()+.005)


def test_latest_target_send_does_not_wait_but_bounds_unacknowledged_output(monkeypatch):
    from collections import deque
    from teleop.adapter import DriverLink
    monkeypatch.setitem(sys.modules,'std_msgs.msg',SimpleNamespace(String=SimpleNamespace))
    link=object.__new__(DriverLink);link.lease={'session_id':'new','boot_id':'boot','secret':'00'*32}
    link.seq=0;link.pending=deque();link.lease_started_ns=time.monotonic_ns();link.execution_progress_ns=0
    state={'monotonic_ns':link.lease_started_ns-1,'session_id':None,'state':'idle'}
    link.feedback=lambda:state;sent=[]
    link.publisher=SimpleNamespace(publish=lambda msg:sent.append(json.loads(msg.data)))
    link.send([0.]*10,None,time.monotonic()+.1,wait_for_execution=False)
    assert len(sent)==1 and len(link.pending)==1
    state.update(monotonic_ns=time.monotonic_ns(),boot_id='boot',session_id='new',state='active',applied_sequence=1)
    link.send([0.]*10,None,time.monotonic()+.1,wait_for_execution=False)
    assert len(sent)==2 and list(link.pending)[0][0]==2
    link.pending[0]=(2,time.monotonic()-1)
    link.execution_progress_ns=time.monotonic_ns()-101_000_000
    with pytest.raises(ValueError,match='driver_execution_ack_timeout'):
        link.send([0.]*10,None,time.monotonic()+.1,wait_for_execution=False)
    assert len(sent)==2
    state['session_id']='different'
    with pytest.raises(ValueError,match='driver_lease_lost'):
        link.send([0.]*10,None,time.monotonic()+.1,wait_for_execution=False)



def test_coalesced_execution_ack_tracks_progress_not_superseded_deadline(monkeypatch):
    from collections import deque
    from teleop.adapter import DriverLink
    monkeypatch.setitem(sys.modules,'std_msgs.msg',SimpleNamespace(String=SimpleNamespace))
    now=[10.0]
    monkeypatch.setattr(time,'monotonic',lambda:now[0])
    monkeypatch.setattr(time,'monotonic_ns',lambda:int(now[0]*1e9))
    link=object.__new__(DriverLink)
    link.lease={'session_id':'new','boot_id':'boot','secret':'00'*32}
    link.seq=4;link.pending=deque([(3,9.98),(4,10.02)])
    link.lease_started_ns=9_000_000_000;link.execution_progress_ns=0
    state={'monotonic_ns':9_990_000_000,'boot_id':'boot','session_id':'new',
           'state':'active','applied_sequence':3}
    link.feedback=lambda:state;sent=[]
    link.publisher=SimpleNamespace(publish=lambda msg:sent.append(json.loads(msg.data)))
    link.send([0.]*10,None,10.07,wait_for_execution=False)
    now[0]=10.03  # Sequence 4 expired, but execution advanced only 40 ms ago.
    link.send([0.]*10,None,10.1,wait_for_execution=False)
    assert len(sent)==2 and sent[-1]['valid_for_ms']<=70
    link.pending[0]=(4,10.15)  # Even a later deadline cannot extend progress budget.
    state['monotonic_ns']=10_090_000_000  # Fresh feedback alone is not progress.
    now[0]=10.091
    with pytest.raises(ValueError,match='driver_execution_ack_timeout'):
        link.send([0.]*10,None,10.19,wait_for_execution=False)
    assert len(sent)==2

@pytest.mark.usefixtures("g1_numeric_abi")
def test_g1_acquisition_does_not_send_aging_frame(tmp_path):
    link=Link();adapter=G1IntentAdapter(link,'live');adapter.calibrate(calibration(tmp_path))
    link.sha=adapter.solver.profile_sha256
    original_claim=link.claim
    def slow_claim(deadline):
        time.sleep(.045)
        original_claim(deadline)
        link.lease={'session_id':'test'}
    link.claim=slow_claim
    assert adapter.apply(intent(frame(0,True,1))).ok
    assert link.calls==['claim']
    assert adapter.output['state']=='armed_waiting_input'
    assert adapter.apply(intent(frame(1,True,1))).ok
    assert link.calls==['claim','send']
    assert adapter.apply(intent(frame(2,True,2),2)).ok
    assert link.calls==['claim','send','resume','claim']
    assert adapter.output['state']=='armed_waiting_input'
    stale=intent(frame(3,True,2),2)
    stale.received_monotonic-=.2
    assert not adapter.apply(stale).ok
    assert link.calls==['claim','send','resume','claim']


@pytest.mark.usefixtures("g1_numeric_abi")
def test_execution_ack_timeout_requires_confirmed_hold_and_new_clutch(tmp_path):
    link=Link();adapter=G1IntentAdapter(link,'live');adapter.calibrate(calibration(tmp_path))
    link.sha=adapter.solver.profile_sha256
    assert adapter.apply(intent(frame(0,True,1))).ok
    def timeout(*args,**kwargs):raise ValueError('driver_execution_ack_timeout')
    link.send=timeout
    ack=adapter.apply(intent(frame(1,True,2)))
    assert not ack.ok and ack.code=='driver_execution_ack_timeout'
    link.pause=lambda deadline:False
    request=SimpleNamespace(dispatch_generation=2,deadline_monotonic=time.monotonic()+.5,reason=ack.code)
    assert not adapter.safe_stop(request).ok
    link.pause=lambda deadline:True
    assert adapter.safe_stop(request).ok
    assert adapter.clutch is None
    link.send=lambda *a,**kw:link.calls.append('send')
    assert adapter.apply(intent(frame(2,True,3),seq=3)).ok
    assert 'send' not in link.calls  # New clutch only acquires; no stale replay.


@pytest.mark.parametrize('state,reason,expected',[
    ('hold','command_timeout','command_timeout'),
    ('hold','base_not_stationary','base_not_stationary'),
    ('fault','command_timeout','motion_rejected'),
    ('hold','motor_fault','motion_rejected'),
])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_driver_holding_recovers_only_known_pause_reasons(tmp_path,state,reason,expected):
    link=Link();adapter=G1IntentAdapter(link,'live');adapter.calibrate(calibration(tmp_path))
    link.sha=adapter.solver.profile_sha256
    assert adapter.apply(intent(frame(0,True,1))).ok
    original_feedback=link.feedback
    link.feedback=lambda:{**original_feedback(),'state':state,'reason':reason}
    def holding(*args,**kwargs):raise ValueError('driver_holding')
    link.send=holding
    ack=adapter.apply(intent(frame(1,True,2)))
    assert not ack.ok and ack.code==expected


@pytest.mark.usefixtures("g1_numeric_abi")
def test_zero_input_stays_put_and_small_input_advances_bounded_target(tmp_path):
    solver=G1IK(calibration(tmp_path))
    q=np.array([.2455,.2997,-.0784,.8066,-.016,.2507,-.2954,.0944,.774,-.0014])
    solver.self_test(q);targets=solver.palms(q)
    for _ in range(5):
        output,_=solver.solve(targets,q,commanded=q,dq=np.zeros(10))
        assert np.max(np.abs(np.array(output)-q))<1e-5
    for target in targets:target[2,3]+=.002
    previous=q.copy()
    for _ in range(20):
        last=solver.last_solve_at
        output,_=solver.solve(targets,q,commanded=previous,dq=np.zeros(10))
        output=np.array(output)
        elapsed=solver.last_solve_at-last
        elapsed=elapsed if 0<elapsed<=.1 else .02
        assert np.max(np.abs(output-previous))<=solver.velocity*elapsed+1e-8
        previous=output
    assert np.max(np.abs(previous-q))>.0015
    # Removing the gap gate must not bypass the swept-body collision check.
    with pytest.raises(ValueError,match='g1_body_collision'):
        solver.solve(targets,q,commanded=q+.026,dq=np.zeros(10))
    record=solver.last_collision_rejection
    assert record['body']['kind']=='swept_bound'
    assert record['body']['distance_m']<=record['body']['clearance_m']+record['body']['swept_margin_m']
    assert len(record['context']['target_q'])==10
    assert record['context']['measured_q']==q.tolist()


@pytest.mark.parametrize('invalid', [None, 'old', 'boot', 'owned', 'unconfirmed', 'stale'])
def test_late_stop_feedback_reconciles_only_confirmed_requested_release(invalid):
    import threading
    from teleop.adapter import DriverLink
    link=DriverLink.__new__(DriverLink);link.condition=threading.Condition()
    lease={'boot_id':'boot','session_id':'session','secret':'00'*32};link.lease=lease
    link.release_requested_ns=0;link.management_request=None
    link.latest={'monotonic_ns':time.monotonic_ns(),'boot_id':'boot',
                 'ownership_held':True,'stop_confirmed':False}
    def timeout(*a):raise TimeoutError('response delayed')
    link.call=timeout
    # A lost HTTP reply is no longer final: stop waits for independent DDS
    # confirmation. An old sample cannot acknowledge that release request.
    with pytest.raises(ValueError,match='^driver_feedback_ack_timeout$'):
        link.stop(time.monotonic()+.01)
    assert link.lease==lease
    requested=link.release_requested_ns
    state={'monotonic_ns':time.monotonic_ns(),'boot_id':'boot','ownership_held':False,'stop_confirmed':True}
    if invalid=='old':state['monotonic_ns']=requested-1
    if invalid=='boot':state['boot_id']='other'
    if invalid=='owned':state['ownership_held']=True
    if invalid=='unconfirmed':state['stop_confirmed']=False
    if invalid=='stale':state['monotonic_ns']=time.monotonic_ns()-200_000_000
    link.latest=state
    assert link.reconcile_release()==(invalid is None)
    assert (link.lease is None)==(invalid is None)
    if invalid is not None:assert link.lease==lease


@pytest.mark.usefixtures("g1_numeric_abi")
def test_real_near_identity_pose_has_finite_ik_gradient():
    from teleop.g1_ik import G123PinocchioIk
    case=json.loads((Path(__file__).parent/'fixtures/g1_ik_near_identity.json').read_text())
    ik=G123PinocchioIk(ROOT/'models/g1_body23.urdf',palm_frames=case['palm_frames'],locked_joints=case['locked_joints'])
    q=np.asarray(case['measured_q'])
    for targets in ([np.asarray(t) for t in case['targets']],ik.current_targets(q)):
        result,tau=ik.solve(*targets,q,np.zeros(10))
        assert np.isfinite(result).all() and np.isfinite(tau).all()
        assert max(np.linalg.norm(a[:3,3]-b[:3,3]) for a,b in zip(ik.current_targets(result),targets))<.015


def test_short_target_ttl_does_not_shorten_execution_ack_budget(monkeypatch):
    from collections import deque
    from teleop.adapter import DriverLink
    monkeypatch.setitem(sys.modules,'std_msgs.msg',SimpleNamespace(String=SimpleNamespace))
    now=[10.]
    monkeypatch.setattr(time,'monotonic',lambda:now[0])
    monkeypatch.setattr(time,'monotonic_ns',lambda:int(now[0]*1e9))
    link=object.__new__(DriverLink)
    link.lease={'session_id':'s','boot_id':'b','secret':'00'*32}
    link.seq=0;link.pending=deque();link.lease_started_ns=9_000_000_000;link.execution_progress_ns=0
    state={'monotonic_ns':10_000_000_000,'boot_id':'b','session_id':'s','state':'active','applied_sequence':-1}
    link.feedback=lambda:state;sent=[]
    link.publisher=SimpleNamespace(publish=lambda msg:sent.append(json.loads(msg.data)))
    link.send([0.]*10,None,10.045,wait_for_execution=False)
    assert 0<sent[0]['valid_for_ms']<=45
    now[0]=10.06;state['monotonic_ns']=10_060_000_000
    link.send([0.]*10,None,10.13,wait_for_execution=False)
    now[0]=10.101;state['monotonic_ns']=10_101_000_000
    with pytest.raises(ValueError,match='driver_execution_ack_timeout'):
        link.send([0.]*10,None,10.19,wait_for_execution=False)
    assert len(sent)==2


@pytest.mark.usefixtures("g1_numeric_abi")
def test_pr152_approximates_extended_pose_without_legacy_position_gate():
    from teleop.g1_ik import G123PinocchioIk
    ik=G123PinocchioIk(ROOT/'models/g1_body23.urdf',pr152_objective=True)
    q=np.array([.243,.302,-.087,.817,-.005,.251,-.293,.081,.781,.007])
    targets=ik.current_targets(q)
    for target in targets:target[0,3]+=.3
    result,tau=ik.solve(*targets,q,np.zeros(10))
    assert np.isfinite(result).all() and np.isfinite(tau).all()
    assert np.all(result>=ik._model.lowerPositionLimit-1e-6)
    assert np.all(result<=ik._model.upperPositionLimit+1e-6)
    diagnostic=ik.snapshot()['last_solution']
    assert diagnostic['position_policy']=='pr152_weighted_pose'
    assert max(diagnostic['position_error_m'])>.015
    diagnostic['position_error_m'][0]=999
    assert ik.snapshot()['last_solution']['position_error_m'][0]!=999


@pytest.mark.usefixtures("g1_numeric_abi")
def test_pr152_checks_actual_bounded_segment_before_output(tmp_path):
    solver=G1IK(calibration(tmp_path),pr152_objective=True)
    q=np.array([.243,.302,-.087,.817,-.005,.251,-.293,.081,.781,.007])
    target=q.copy();target[0]+=.5
    solver.ik.solve=lambda *a:(target.copy(),np.zeros(10))
    checked=[]
    original=solver._safe_configuration
    def check(point,excursion=None):
        checked.append((point.copy(),None if excursion is None else excursion.copy()))
        return original(point,excursion)
    solver._safe_configuration=check
    result,_=solver.solve(solver.palms(q),q,dq=np.zeros(10))
    assert np.max(np.abs(np.asarray(result)-q))<=.00400001
    assert any(e is not None and np.max(e)>0 for _,e in checked)
    assert all(e is None or np.max(e)<=.00400001 for _,e in checked)
    def reject(*a,**kw):raise ValueError('body_collision')
    solver._safe_configuration=reject
    with pytest.raises(ValueError,match='body_collision'):
        solver.solve(solver.palms(q),q,dq=np.zeros(10))


@pytest.mark.parametrize('invalid',[None,'old','session','fault'])
def test_operator_pause_waits_for_matching_physical_rest(invalid):
    state={'boot_id':'b','session_id':'s','state':'hold','reason':'operator_pause',
           'hold_confirmed':False,'feedback_hold_generation':1,'monotonic_ns':time.monotonic_ns()}
    lease={'boot_id':'b','session_id':'s'}
    def pause(deadline):
        state['monotonic_ns']=time.monotonic_ns()
        if invalid=='old':state['monotonic_ns']=1
        if invalid=='session':state['session_id']='old'
        if invalid=='fault':state['state']='fault'
        return False
    link=SimpleNamespace(lease=lease,feedback=lambda:dict(state),pause=pause)
    adapter=G1IntentAdapter(link,'live')
    request=SimpleNamespace(dispatch_generation=1,deadline_monotonic=time.monotonic()+.1,reason='deadman_released')
    ack=adapter.safe_stop(request)
    assert not ack.ok and ack.code==('feedback_wait' if invalid is None else 'stop_unconfirmed')
    if invalid is None:
        assert not adapter.external_release_signal()['acknowledged']
        state['hold_confirmed']=True
        assert adapter.external_release_signal()=={'generation':1,'reason':'operator_pause','acknowledged':True}


@pytest.mark.parametrize('direction',[-1.,1.])
@pytest.mark.parametrize('wrist',[-.005,-.9643329988281466])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_large_tracking_gap_advances_without_fixed_budget(tmp_path,direction,wrist):
    solver=G1IK(calibration(tmp_path),pr152_objective=True)
    q=np.array([.243,.302,-.087,.817,-.005,.251,-.293,.081,.781,.007])
    q[4]=wrist
    goal=q.copy();goal[4]+=direction*.5
    solver.ik.solve=lambda *a:(goal.copy(),np.zeros(10))
    command=q.copy();command[4]+=direction*.024
    for _ in range(4):
        solver.last_solve_at=time.monotonic()-.02
        result,_=solver.solve(solver.palms(q),q,commanded=command,dq=np.zeros(10))
        command=np.asarray(result)
        assert direction*(command[4]-q[4])>.025
    waiting=command.copy();q[4]+=direction*.01
    solver.last_solve_at=time.monotonic()-.02
    result,_=solver.solve(solver.palms(q),q,commanded=command,dq=np.zeros(10))
    assert direction*(result[4]-waiting[4])>0
    bad=q.copy();bad[4]+=direction*.026
    solver.last_solve_at=time.monotonic()-.02
    result,_=solver.solve(solver.palms(q),q,commanded=bad,dq=np.zeros(10))
    assert direction*(result[4]-bad[4])>0
    assert np.max(np.abs(np.asarray(result)-bad))<=solver.velocity*.03


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('direction',[-1,1])
def test_clutch_relative_translation_and_head_invariance(axis,direction):
    from teleop.g1_mapping import G1ClutchRelativeMapper,G1ControllerPoseMapper
    mapper=G1ClutchRelativeMapper();base=frame(0,True,1)
    palms=[np.eye(4),np.eye(4)];palms[0][:3,3]=[.2,.3,-.1];palms[1][:3,3]=[.2,-.3,-.1]
    mapper.reset(base,palms)
    assert np.allclose(mapper.map_frame(base),palms)
    changed=copy.deepcopy(base);changed['left_controller']['position'][axis]+=.03*direction
    raw=G1ControllerPoseMapper();expected=raw.map_frame(changed)[0][:3,3]-raw.map_frame(base)[0][:3,3]
    target=mapper.map_frame(changed)
    assert np.allclose(target[0][:3,3]-palms[0][:3,3],expected)
    assert np.allclose(target[1],palms[1])
    changed['head']['position']=[.1,1.1,.1];changed['head']['orientation']=[0.,.70710678,0.,.70710678]
    assert np.allclose(mapper.map_frame(changed),target)
    palms[0][:3,3]+=.1
    mapper.reset(changed,palms)
    assert np.allclose(mapper.map_frame(changed),palms)


def test_clutch_relative_rotation():
    from teleop.g1_mapping import G1ClutchRelativeMapper
    mapper=G1ClutchRelativeMapper();f=frame(0,True,1);palms=[np.eye(4),np.eye(4)]
    with pytest.raises(ValueError,match='baseline'):mapper.map_frame(f)
    mapper.reset(f,palms);rotated=copy.deepcopy(f);rotated['left_controller']['orientation']=[0.,0.,.38268343,.92387953]
    target=mapper.map_frame(rotated)
    assert not np.allclose(target[0][:3,:3],palms[0][:3,:3])
    assert np.allclose(target[0][:3,3],palms[0][:3,3])


@pytest.mark.usefixtures("g1_numeric_abi")
def test_clutch_relative_adapter_reclutch(tmp_path):
    f=frame(0,True,1)
    rotated=copy.deepcopy(f);rotated['left_controller']['orientation']=[0.,0.,.38268343,.92387953]
    link=Link();adapter=G1IntentAdapter(link,'shadow',1,mapping_version='pr152_clutch_relative_v1')
    adapter.calibrate(calibration(tmp_path))
    assert adapter.apply(intent(f)).ok
    assert adapter.apply(intent(rotated,2)).ok
    assert not link.calls
    assert adapter.snapshot()['diagnostics']['mapping_version']=='pr152_clutch_relative_v1'


@pytest.mark.parametrize('mapping',['relative_v1','pr152_head_yaw_v1','pr152_clutch_relative_v1'])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_torso_reference_requires_clutch_relative_mapping(tmp_path,mapping):
    path=calibration(tmp_path);profile=json.loads(path.read_text());profile['safety']['waist_reference']='torso';path.write_text(json.dumps(profile))
    adapter=G1IntentAdapter(Link(),'shadow',1,mapping_version=mapping)
    if mapping=='pr152_clutch_relative_v1':adapter.calibrate(path)
    else:
        with pytest.raises(ValueError,match='waist_mapping_mismatch'):adapter.calibrate(path)

@pytest.mark.parametrize('hardware',[False,True])
@pytest.mark.usefixtures("g1_numeric_abi")
def test_collision_hold_preserves_anchor_and_validates_before_resume(tmp_path,hardware):
    from teleop.dispatch import StopRequest
    link=Link();adapter=G1IntentAdapter(link,'shadow',mapping_version='pr152_clutch_relative_v1',scale=1.)
    adapter.calibrate(calibration(tmp_path))
    first=intent(frame(0,True,1),1);assert adapter.apply(first).ok
    anchored=tuple(x.copy() for x in adapter.head_mapper._anchor)
    adapter.hardware_output=hardware
    link.sha=adapter.solver.profile_sha256
    link.lease={'boot_id':'test','session_id':'test'} if hardware else None
    link.pause=lambda deadline:True
    original=adapter.solver.solve
    def reject(*args,**kwargs):raise ValueError('g1_body_collision:a:b')
    adapter.solver.solve=reject
    f=frame(1,True,1);f['left_controller']['position'][1]+=.02
    sample=intent(f,2);sample.clutch_sequence=1
    assert not adapter.apply(sample).ok
    request=SimpleNamespace(dispatch_generation=3,deadline_monotonic=time.monotonic()+.1,reason='g1_body_collision')
    assert adapter.safe_stop(request).ok
    sample=intent(f,3);sample.clutch_sequence=1
    assert not adapter.apply(sample).ok
    assert 'resume' not in link.calls and 'send' not in link.calls
    for before,after in zip(anchored,adapter.head_mapper._anchor):np.testing.assert_array_equal(before,after)
    adapter.solver.solve=original
    # Return to the same acquisition pose, without a new clutch sequence.
    sample=intent(frame(2,True,1),3);sample.clutch_sequence=1
    assert adapter.apply(sample).ok
    assert ('resume' in link.calls)==hardware
    assert 'send' not in link.calls
    for before,after in zip(anchored,adapter.head_mapper._anchor):np.testing.assert_array_equal(before,after)


@pytest.mark.usefixtures("g1_numeric_abi")
def test_collision_broadphase_matches_all_exact_pairs(tmp_path):
    import hppfcl as fcl
    solver=G1IK(calibration(tmp_path));geometry=solver.body_collision
    rng=np.random.default_rng(123)
    rejected=0;accepted=0
    for sample in range(120):
        q=np.zeros(10) if sample<4 else rng.uniform(-.5,.5,10)
        excursion=None if sample%2 else rng.uniform(0,.02,10)
        solver.pin.framesForwardKinematics(solver.model,solver.data,q)
        expected=None
        poses=[]
        for _,frame,placement,_,_ in geometry.objects:
            pose=solver.data.oMf[frame]*placement
            poses.append(fcl.Transform3f(pose.rotation,pose.translation))
        for a,b in geometry.pairs:
            left,right=geometry.objects[a],geometry.objects[b]
            margin=0. if excursion is None else float((left[4]+right[4])@excursion)
            distance=fcl.distance(left[3],poses[a],right[3],poses[b],fcl.DistanceRequest(),fcl.DistanceResult())
            if not np.isfinite(distance) or distance<=.005+margin:
                expected='g1_body_collision:'+left[0]+':'+right[0];break
        if expected:
            with pytest.raises(ValueError,match='^'+expected+'$'):geometry.check(solver.data,excursion)
            rejected+=1
        else:
            geometry.check(solver.data,excursion);accepted+=1
    assert rejected>0 and accepted>0


@pytest.mark.usefixtures("g1_numeric_abi")
def test_submitted_diagnostics_match_display_solution_and_sent_target(tmp_path):
    link=Link();adapter=G1IntentAdapter(link,'shadow');adapter.calibrate(calibration(tmp_path))
    assert adapter.apply(intent(frame(0,True,1))).ok
    visual=adapter.solver.visualization_sample
    assert adapter.output['desired_q']==visual['ik_q']
    assert adapter.output['desired_q'] is not visual['ik_q']
    np.testing.assert_allclose(adapter.solver.palms(adapter.output['desired_q']),adapter.solver.palms(visual['ik_q']))
