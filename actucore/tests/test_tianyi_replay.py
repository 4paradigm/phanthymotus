"""Replay evidence and recording tests; no network or physical devices."""
import copy
import importlib.util
import json
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.recording import PoseRecorder
from teleop.replay import checked_segment,evaluate,schedule,target_at
from teleop.runtime import TeleopRuntime
from teleop.dispatch import RecordingAdapter
from teleop.protocol import bind_rtc_frame_v1
from test_teleop import frame


def test_full_accepted_frames_no_secret_and_no_snapshot_polling(tmp_path):
    recorder=PoseRecorder(tmp_path)
    recorder.start({'profile_sha256':'fixture'},duration=1)
    runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def observe(f,t):recorder.capture(f,t,{'secret':'never-record','feedback':{'q':[0.]*14,'arm_ns':time.monotonic_ns()}})
        runtime.pose_observer=observe
        for i in range(72):
            runtime.submit_frame(bind_rtc_frame_v1(frame(i),authority=binding,expected_mode='shadow'),source='offline',include_status=False)
        with pytest.raises(Exception):runtime.submit_frame({},source='offline')
    finally:runtime.close();result=recorder.stop()
    rows=[json.loads(s) for s in (recorder.path/'poses.jsonl').read_text().splitlines()]
    assert result['complete'] and len(rows)==72 and [r['frame']['sequence'] for r in rows]==list(range(72))
    assert 'fence' not in json.dumps(rows) and 'secret' not in json.dumps(rows)
    assert rows[0]['frame']['schema_version']==1


def test_recorder_full_queue_counts_loss_without_blocking(tmp_path):
    r=PoseRecorder(tmp_path,capacity=1);r.result={'state':'recording','dropped':0};r.until=time.monotonic()+1;r.queue=queue.Queue(1)
    r.capture(frame(),time.monotonic(),{});r.capture(frame(1),time.monotonic(),{})
    assert r.result['dropped']==1 and r.queue.qsize()==1


def test_observer_failure_never_changes_control_authority():
    r=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),auto_watchdog=False)
    try:
        r.prepare_local_session();binding,_=r.rtc_authority_snapshot()
        def broken(*args):raise OSError('full_disk')
        r.pose_observer=broken
        r.submit_frame(bind_rtc_frame_v1(frame(),authority=binding,expected_mode='shadow'),source='offline')
        assert r.status()['counters']['recording_errors']==1
        assert r.status()['state']!='fault'
    finally:r.close()


class Geometry:
    velocity=.2;indices=np.arange(14)
    model=SimpleNamespace(lowerPositionLimit=np.full(14,-1.),upperPositionLimit=np.full(14,1.))
    def _safe_configuration(self,q,excursion=None):
        if np.max(q)>.8:raise ValueError('collision')


def package():
    return {'waypoints':[{'q':[0.]*14,'source_ns':0}, {'q':[.1]*14,'source_ns':500_000_000}]}


def test_trajectory_retimes_and_validates_all_transitions():
    points=schedule(package(),Geometry(),[0.]*14)
    for a,b in zip(points,points[1:]):
        assert max(abs(x-y)/(b['t']-a['t']) for x,y in zip(a['q'],b['q']))<=.2+1e-10
    assert {p['phase'] for p in points}>={'left','right','both','baseline'}
    assert points[-1]['t']<40
    left=[p for p in points if p['phase']=='left'];right=[p for p in points if p['phase']=='right']
    assert all(p['q'][7:]==[0.]*7 for p in left)
    assert all(p['q'][:7]==left[-1]['q'][:7] for p in right)
    q,phase=target_at(points,.5);assert q==[0.]*14 and phase=='baseline'
    with pytest.raises(ValueError,match='collision'):checked_segment(Geometry(),[0.]*14,[.9]*14)
    with pytest.raises(ValueError,match='joint_limit'):checked_segment(Geometry(),[0.]*14,[2.]*14)
    with pytest.raises(ValueError):checked_segment(Geometry(),[0.]*14,[float('nan')]*14)


def evidence(follow=True):
    rows=[]
    for i in range(160):
        phase='baseline' if i<10 else ('left_plateau','right_plateau','both_plateau')[(i-10)//50]
        target=([0.]*14 if phase=='baseline' else
                [.1]*7+[0.]*7 if phase=='left_plateau' else
                [0.]*7+[.1]*7 if phase=='right_plateau' else [.1]*14)
        rows.append({'elapsed':i*.02,'phase':phase,
                     'target_q':target,'q':target if follow else [0.]*14})
    rows.append({'event':'round_end','error':None,'stop_confirmed':True,'paused_and_resumed':True})
    return rows


def test_published_commands_and_moving_flags_do_not_count_as_motion():
    assert evaluate(evidence())['passed']
    rows=evidence(False)
    for r in rows:r.update(output_active=True,applied_sequence=999)
    assert not evaluate(rows)['passed']
    rows=evidence();rows[-1]['stop_confirmed']=False
    assert not evaluate(rows)['passed']
    rows=evidence();rows[-1]['error']='timeout'
    assert not evaluate(rows)['passed']


def test_stable_tracking_error_is_reported_without_an_accuracy_gate():
    rows=evidence()
    for row in rows:
        if 'q' in row:row['q']=[x*.4 for x in row['q']]
    result=evaluate(rows)
    assert result['passed'] and result['tracking_error_policy']=='report_only'
    assert result['tolerance_rad'] is None
    assert result['stable_max_error_rad']==pytest.approx([.06]*14)








def test_report_needs_combined_pass(tmp_path):
    from teleop.acceptance import report
    for i in range(3):(tmp_path/f'round-{i}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in evidence()))
    report(tmp_path);assert not json.loads((tmp_path/'report.json').read_text())['passed']
    (tmp_path/'combined.result.json').write_text('{"passed":true}')
    report(tmp_path);assert json.loads((tmp_path/'report.json').read_text())['passed']
    assert '<svg' in (tmp_path/'report.html').read_text()


def test_real_player_lag_and_finally_release(monkeypatch):
    from teleop.replay import execute
    now=[100.];q=np.zeros(14);goal=np.zeros(14);events=[]
    monkeypatch.setattr('teleop.replay.time.monotonic_ns',lambda:int(now[0]*1e9))
    class Link:
        lease=None;management_request=None;seq=0;pause_count=0;send_tries=0;stop_tries=0
        def prepare_transport(self):pass
        def claim(self,deadline):self.lease={'session_id':'fixture'}
        lease_started_ns=0;stale=False;fresh_waits=0;old_feedback=False
        def resume(self,deadline):
            self.lease={'session_id':'fixture'};self.lease_started_ns=int(now[0]*1e9);self.stale=True
        def feedback_after(self,stamp,deadline):
            assert self.stale and stamp==self.lease_started_ns
            self.stale=False;self.fresh_waits+=1
            return self.feedback()
        def feedback(self):
            if self.stale:
                return {'state':'hold','reason':'operator_pause','monotonic_ns':self.lease_started_ns-1}
            if 100.3<now[0]<100.42:
                self.old_feedback=True
                return {'state':'active','feedback':{'q':q.tolist(),'dq':[0.]*14,'arm_ns':int((now[0]-.15)*1e9)}}
            q[:]+=np.clip(goal-q,-.0005,.0005)
            return {'state':'active','feedback':{'q':q.tolist(),'dq':[0.]*14,'arm_ns':int(now[0]*1e9)}}
        def pause(self,deadline):self.pause_count+=1;return True
        def send(self,target,hands,deadline,**kw):
            self.send_tries+=1
            if self.send_tries==1:raise ValueError('driver_feedback_stale_or_different_clock')
            assert hands==[0.,0.];goal[:]=target;self.seq+=1;return True
        def stop(self,deadline):
            self.stop_tries+=1
            if self.stop_tries==1:raise ValueError('driver_feedback_stale_or_different_clock')
            self.lease=None;return True
    l=Link();points=[{'t':0,'q':[0.]*14,'phase':'baseline'},
                    {'t':1,'q':[0.]*14,'phase':'baseline'},
                    {'t':2,'q':[.04]*14,'phase':'both'},
                    {'t':3,'q':[.04]*14,'phase':'both_plateau'}]
    execute(l,points,Geometry(),events.append,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+max(s,.001)))
    assert l.old_feedback and l.stop_tries==2
    assert any(e.get("event")=="waiting_send_feedback" and not e["hardware_target_sent"] for e in events)
    assert l.fresh_waits==1
    assert l.pause_count==1 and l.lease is None and events[-1]['stop_confirmed']
    assert len(events)>100 and np.max(q)>.03
    assert any(max(abs(x-y) for x,y in zip(e['q'],e['target_q']))>.001 for e in events if 'q' in e)


def test_offline_analysis_preserves_mapping_across_ik_failure(tmp_path,monkeypatch):
    from teleop import replay
    from teleop.kinematics import RelativeMapping
    resets=[]
    class Mapping(RelativeMapping):
        def reset(self,*args):resets.append(1);return super().reset(*args)
    class Solver(Geometry):
        profile={'controller_to_palm':{s:{'position':[0,0,0],'orientation':[0,0,0,1]} for s in ('left','right')}}
        def __init__(self,path):pass
        def palms(self,q):
            values=[np.eye(4),np.eye(4)]
            for i,t in enumerate(values):t[0,3]=q[i*7]
            return values
        def solve(self,targets,initial):
            if abs(targets[0][0,3]-.025)<1e-8:raise ValueError('torso_collision')
            q=np.repeat(targets[0][0,3],14);self.visualization_sample={'ik_q':q};return q.tolist()
    monkeypatch.setattr(replay,'TianyiIK',Solver);monkeypatch.setattr(replay,'RelativeMapping',Mapping)
    profile=tmp_path/'profile.json';profile.write_text('{}')
    recording=tmp_path/'recording';recording.mkdir()
    rows=[]
    for i in range(10):
        f=frame(i,True,1);f['epoch']=1
        for side in ('left_controller','right_controller'):f[side]['position'][2]=-.025*i
        ns=1_000_000_000+i*20_000_000
        rows.append({'frame':f,'received_ns':ns,'driver':{'feedback':{'q':[0.]*14,'arm_ns':ns}}})
    (recording/'poses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (recording/'manifest.json').write_text(json.dumps({'profile_sha256':replay.digest(profile),'position_scale':.5,'sources':{}}))
    (recording/'result.json').write_text(json.dumps({'complete':True,'frames':10,'dropped':0}))
    package=replay.analyze(recording,profile,tmp_path/'analysis')
    assert len(resets)==2  # one per mode, not a hidden clutch after the collision
    assert package['waypoints'][-1]['q'][0]==pytest.approx(.1125)
    details=[json.loads(s) for s in (tmp_path/'analysis/ik.jsonl').read_text().splitlines()]
    assert any(d['code']=='torso_collision' for d in details)
    manifest=json.loads((recording/'manifest.json').read_text());manifest['profile']={}
    (recording/'manifest.json').write_text(json.dumps(manifest))
    profile.write_text('{"joint_velocity_rad_s":1}')
    adjusted=replay.analyze(recording,profile,tmp_path/'speed')
    assert adjusted['profile_sha256']!=adjusted['recorded_profile_sha256']
    profile.write_text('{"joint_velocity_rad_s":1,"workspace":{}}')
    with pytest.raises(ValueError,match='geometry_changed'):replay.analyze(recording,profile,tmp_path/'geometry')
    (recording/'result.json').write_text('{"complete":false,"frames":10,"dropped":1}')
    with pytest.raises(ValueError,match='incomplete'):replay.analyze(recording,profile,tmp_path/'bad')


def test_recovery_journal_written_before_uncertain_rpc(tmp_path):
    from teleop.acceptance import journal_link
    path=tmp_path/'.lease.json'
    old={'boot_id':'boot','session_id':'old','secret':'secret'}
    operation={'id':'req','action':'resume','credentials':{'session_id':'old','secret':'secret'}}
    def call(action,deadline):
        stored=json.loads(path.read_text())
        assert stored['management_request']==operation and stored['lease']==old
        raise TimeoutError('lost reply')
    link=SimpleNamespace(call=call,lease=old,management_request=operation,release_requested_ns=0)
    journal_link(link,path)
    with pytest.raises(TimeoutError):link.call('resume',1)
    assert path.stat().st_mode&0o777==0o600


def test_successful_claim_saved_before_caller_receives_it(tmp_path):
    from teleop.acceptance import journal_link
    result={'boot_id':'boot','session_id':'new','secret':'secret'}
    path=tmp_path/'.lease.json'
    link=SimpleNamespace(call=lambda *a:result,lease=None,management_request={'id':'req'},release_requested_ns=0)
    journal_link(link,path);assert link.call('claim',1)==result
    assert json.loads(path.read_text())['lease']==result


def test_recording_waits_for_first_valid_grip_and_preserves_release(tmp_path):
    r=PoseRecorder(tmp_path);r.start({},duration=1,wait_for_deadman=True)
    try:
        start=time.monotonic();r.capture(frame(0,False),start,{})
        assert r.status()['state']=='armed' and r.status()['frames']==0
        r.capture(frame(1,True),start+.01,{})
        assert r.status()['state']=='recording'
        assert .8<r.until-time.monotonic()<=1
        assert r.status()['trigger_received_ns']==int((start+.01)*1e9)
    finally:r.stop()
    rows=[json.loads(x) for x in (r.path/'poses.jsonl').read_text().splitlines()]
    assert [x['frame']['deadman'] for x in rows]==[False,True]


def test_transition_covers_independent_joint_box_not_only_diagonal():
    from teleop.workspace import ArmWorkspace
    class Box(ArmWorkspace):
        indices=np.arange(2)
        def _safe_configuration(self,q,excursion=None):
            # Both diagonal endpoints are safe, but independent interpolation
            # can put joint 0 high while joint 1 is low.
            if q[0]+excursion[0]>.7 and q[1]-excursion[1]<.3:
                raise ValueError('arm_collision')
    b=Box()
    with pytest.raises(ValueError,match='arm_collision'):
        b._safe_transition([0,0],[1,1])
    b._safe_transition([0,.5],[.4,1])
    calls=[]
    def budget():calls.append(1)
    b._safe_transition([0,.5],[.4,1],budget)
    assert calls


def test_refinement_ignores_large_motion_of_unrelated_distal_joint():
    from teleop.workspace import ArmWorkspace,WorkspaceViolation
    class UpperArm(ArmWorkspace):
        indices=np.arange(2)
        collision=False
        def _safe_configuration(self,q,excursion=None):
            if excursion[0]>.006 or (self.collision and q[0]+excursion[0]>.015):
                raise WorkspaceViolation('torso_collision',np.array([1.,0.]))
    b=UpperArm()
    # Joint 1's much larger travel has no influence on this upper-arm capsule.
    b._safe_transition([0.,0.],[.02,1.])
    b.collision=True
    with pytest.raises(ValueError,match='torso_collision'):
        b._safe_transition([0.,0.],[.02,1.])


@pytest.mark.parametrize('phase', ['left_plateau','right_plateau','both_plateau'])
def test_acceptance_requires_every_complete_stable_phase(phase):
    rows=[r for r in evidence() if r.get('phase')!=phase]
    result=evaluate(rows)
    assert not result['passed'] and phase in result['missing_plateaus']
    rows=evidence()
    keep=[r for r in rows if r.get('phase')==phase][:2]
    rows=[r for r in rows if r.get('phase')!=phase or r in keep]
    assert not evaluate(rows)['passed']


def test_combined_requires_real_direction_accuracy_and_fresh_feedback():
    from teleop.replay import evaluate_combined
    baseline=[{'arm_ns':i,'q':[0.]*14} for i in range(10)]
    rows=[{'arm_ns':i+10,'observed_ns':i+10,'q':[i*.002]*14,
           'target_q':[i*.002]*14} for i in range(50)]
    def check(data,base=baseline,stopped=True):
        return evaluate_combined(data,base,stop_confirmed=stopped,error=None)['passed']
    assert check(rows)
    assert not check(rows,stopped=False)
    assert not check(rows,baseline[:1])
    for mutation in ('static','opposite','stale','same_sample','no_target'):
        data=copy.deepcopy(rows)
        for r in data:
            if mutation=='static':r['q']=[0.]*14
            if mutation=='opposite':r['q']=[-x for x in r['q']]
            if mutation=='stale':r['observed_ns']+=100_000_001
            if mutation=='same_sample':r['arm_ns']=10
            if mutation=='no_target':r['target_q']=None
        assert not check(data),mutation


def test_combined_compares_full_orange_ik_goal_not_just_next_short_command():
    from teleop.replay import evaluate_combined
    baseline=[{'arm_ns':i,'q':[0.]*14} for i in range(10)]
    rows=[{'arm_ns':i+10,'observed_ns':i+10,'q':[i*.002]*14,
           'target_q':[i*.002]*14,'ik_reference_q':[i*.02]*14} for i in range(50)]
    result=evaluate_combined(rows,baseline,stop_confirmed=True,error=None)
    assert result['passed'] and result['comparison_basis']=='full_ik_reference'
    assert result['tracking_error_policy']=='report_only' and result['tolerance_rad'] is None
    assert max(result['max_command_tracking_error_rad'])==0
    assert max(result['max_tracking_error_rad'])>.8
    for row in rows:row['ik_reference_q']=list(row['q'])
    assert evaluate_combined(rows,baseline,stop_confirmed=True,error=None)['passed']
    rows[20]['ik_reference_q']=None
    assert evaluate_combined(rows,baseline,stop_confirmed=True,error=None)['reason']=='ik_reference_samples_missing'


def test_deeper_refinement_preserves_collision_and_time_budget():
    from teleop.workspace import ArmWorkspace
    class Box(ArmWorkspace):
        indices=np.arange(1)
        collision=False
        def _safe_configuration(self,q,excursion=None):
            if self.collision or excursion[0]>.01:raise ValueError('torso_collision')
    b=Box()
    with pytest.raises(ValueError,match='torso_collision'):b._safe_transition([0],[1])
    b.transition_refinement_depth=6;b._safe_transition([0],[1])
    b.collision=True
    with pytest.raises(ValueError,match='torso_collision'):b._safe_transition([0],[1])
    b.collision=False
    def expired():raise ValueError('collision_check_timeout')
    with pytest.raises(ValueError,match='collision_check_timeout'):b._safe_transition([0],[1],expired)


def test_box_proofs_do_not_leak_to_siblings_or_later_traversals():
    from teleop.workspace import ArmWorkspace, WorkspaceViolation
    class Box(ArmWorkspace):
        indices=[0];axis_aware_sweep=True;transition_refinement_depth=6
        def _safe_configuration(self,q,excursion=None,proven=None):
            if 'left_only' in proven:return
            if q[0]+excursion[0]>.5:
                raise WorkspaceViolation('torso_collision',np.ones(1))
            proven.add('left_only')
    b=Box()
    b._safe_transition([0],[.4])
    with pytest.raises(ValueError,match='torso_collision'):b._safe_transition([0],[1])
    with pytest.raises(ValueError,match='torso_collision'):b._safe_transition([.6],[.7])


@pytest.mark.parametrize('case',['recover','four','foreign','unconfirmed','stale','fault','repeat'])
def test_expired_hold_recovery_requires_fresh_owned_confirmed_stop(monkeypatch,case):
    from teleop.replay import execute
    now=[100.];events=[];q=[0.]*14
    monkeypatch.setattr('teleop.replay.time.monotonic_ns',lambda:int(now[0]*1e9))
    class Link:
        lease=None;management_request=None;lease_started_ns=0;seq=0;resumes=0
        def prepare_transport(self):pass
        def claim(self,deadline):self.lease={'boot_id':'boot','session_id':'session'}
        def feedback(self):
            stamp=int(now[0]*1e9)
            held=(not self.resumes or case=='repeat' or (case=='four' and self.resumes<4))
            return {'state':('fault' if case=='fault' else 'hold') if held else 'active',
                'reason':'power_ns_stale','ownership_held':True,'boot_id':'boot',
                'session_id':'other' if case=='foreign' else 'session',
                'hold_confirmed':case!='unconfirmed', 'continuation_allowed':False,
                'feedback':{'arm_ns':stamp,'power_ns':stamp-(200_000_000 if case=='stale' else 0),
                            'fixed_ns':stamp,'q':q,'dq':[0.]*14}}
        def pause(self,deadline):return True
        def resume(self,deadline):self.resumes+=1;now[0]+=.01
        def send(self,target,hands,deadline,**kw):q[:]=target;self.seq+=1;return True
        def stop(self,deadline):self.lease=None;return True
    link=Link();points=[{'t':0,'q':[0.]*14,'phase':'baseline'},
                        {'t':1,'q':[.04]*14,'phase':'both'}]
    def run():execute(link,points,Geometry(),events.append,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+max(s,.001)))
    if case in ('recover','four'):
        run();assert link.resumes==(2 if case=='recover' else 5)  # recovery plus planned mid-round resume
        assert events[0]['event']=='explicit_hold_recovery' and events[0]['elapsed']==0
    else:
        with pytest.raises(ValueError):run()
        if case=='repeat':assert link.resumes>3 and 145<=now[0]<145.1 and 'round_deadline' in events[-1]['error']
        else:assert link.resumes==0
    assert events[-1]['stop_confirmed'] and link.lease is None


def test_refinement_releases_traceback_locals_without_waiting_for_gc():
    import gc,weakref
    from teleop.workspace import ArmWorkspace,WorkspaceViolation
    refs=[]
    class Probe:pass
    class Box(ArmWorkspace):
        indices=[0]
        def _safe_configuration(self,q,excursion=None):
            probe=Probe();refs.append(weakref.ref(probe))
            if excursion[0]>.1:
                errors=[WorkspaceViolation('torso_collision',np.ones(1))]
                raise errors[0]
    enabled=gc.isenabled();gc.disable()
    try:
        Box()._safe_transition([0.],[1.])
        assert not any(r() is not None for r in refs)
    finally:
        if enabled:gc.enable()


def test_initial_measured_settling_offset_never_expands_command_limits():
    raw=[-1.00383,.70383]+[0.]*12
    solver=Geometry();solver.model=SimpleNamespace(lowerPositionLimit=np.full(14,-1.),upperPositionLimit=np.full(14,.7))
    points=schedule(package(),solver,raw)
    assert points[0]['q'][0]==-1.
    assert points[0]['q'][1]==.7
    assert points[0]['initial_limit_adjustment_rad'][:2]==pytest.approx([.00383,-.00383])
    assert all(min(p['q'])>=-1 and max(p['q'])<=.7 for p in points)
    with pytest.raises(ValueError,match='starting_joint_limit'):
        schedule(package(),Geometry(),[-1.00401]+[0.]*13)
    bad=package();bad['waypoints'][0]['q'][0]=-1.00005
    with pytest.raises(ValueError,match='joint_limit'):schedule(bad,Geometry(),[0.]*14)


@pytest.mark.parametrize('case',['transient','stale','duplicate','fault'])
def test_combined_baseline_collects_distinct_fresh_samples_with_fixed_deadline(monkeypatch,case):
    from teleop import acceptance as a
    now=[100.];count=[0]
    monkeypatch.setattr(a.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(a.time,'monotonic_ns',lambda:int(now[0]*1e9))
    monkeypatch.setattr(a.time,'sleep',lambda t:now.__setitem__(0,now[0]+t))
    class Link:
        def feedback(self):
            count[0]+=1
            if case=='fault':raise ValueError('driver_fault')
            if case=='transient' and count[0] in (2,3,6):raise ValueError('driver_feedback_stale_or_different_clock')
            stamp=int((now[0]-(1 if case=='stale' else 0))*1e9)
            if case=='duplicate':stamp=100_000_000_000
            return {'feedback':{'q':[0.]*14,'arm_ns':stamp}}
    if case=='transient':
        rows=a.fresh_baseline(Link());assert len(rows)==10 and len({r['arm_ns'] for r in rows})==10
    else:
        with pytest.raises(ValueError,match='driver_fault' if case=='fault' else 'combined_baseline_missing'):a.fresh_baseline(Link())
    assert now[0]<101.03


@pytest.mark.parametrize('feedback',['fresh','temporary','persistent','oversleep','disk_full'])
def test_combined_replay_does_not_read_blocking_adapter_snapshot(tmp_path,monkeypatch,feedback):
    from teleop import acceptance as a
    now=[100.];submitted=[];snapshots=[];heartbeats=[];shutdown=[]
    monkeypatch.setattr(a.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(a.time,'monotonic_ns',lambda:int(now[0]*1e9))
    monkeypatch.setattr(a.time,'sleep',lambda t:now.__setitem__(0,now[0]+t+(.12 if feedback=='oversleep' else 0)))
    class Adapter:
        input_timeout_ms=300
        dispatch_io_timeout_ms=150
        def __init__(self,*args):pass
        def calibrate(self,path):pass
        def snapshot(self):raise AssertionError('blocking IK snapshot must not be called by input feeder')
    class Runtime:
        def __init__(self,**kwargs):pass
        def prepare_local_session(self):pass
        def rtc_authority_snapshot(self):return {},0
        def heartbeat(self,binding,**kwargs):
            assert kwargs=={'include_status':False};heartbeats.append(now[0])
        def submit_frame(self,f,**kwargs):submitted.append(f['sequence'])
        def status(self):raise AssertionError('heavy public status must not block replay input')
        def control_status(self):
            snapshots.append(1)
            return {'state':'active','dispatch':{'fault_code':None,'adapter':{'output':{'state':'submitted','target_q':[.1]*14}}}}
        def close(self):shutdown.append('runtime_close')
    class Link:
        def prepare_transport(self):pass
        def feedback(self):
            if feedback=='persistent' or (feedback=='temporary' and 20<=len(submitted)<25):raise ValueError('driver_feedback_stale_or_different_clock')
            return {'state':'active','applied_sequence':len(submitted),'feedback':{'q':[0.]*14,'arm_ns':int(now[0]*1e9)}}
        def stop(self,deadline):shutdown.append('link_stop');return True
        def close(self):pass
    class Evidence(a._EvidenceWriter):
        def write(self,text):
            if feedback=='disk_full':raise OSError('disk full')
            return super().write(text)
        def finish(self):
            assert shutdown[:2]==['runtime_close','link_stop']
            shutdown.append('evidence_flush')
            return super().finish()
    monkeypatch.setattr(a,'_EvidenceWriter',Evidence)
    clean=SimpleNamespace(shutdown=lambda:None,join=lambda **kw:None)
    monkeypatch.setattr(a,'ros_link',lambda cfg:(Link(),clean,clean,clean))
    monkeypatch.setattr(a,'idle',lambda:None);monkeypatch.setattr(a,'call',lambda *args:None)
    monkeypatch.setattr(a,'journal_link',lambda *args:None)
    monkeypatch.setattr(a,'fresh_baseline',lambda link:[{'q':[0.]*14,'arm_ns':i} for i in range(10)])
    monkeypatch.setattr(a,'position_recorded_origin',lambda *args:None)
    monkeypatch.setattr(a,'evaluate_combined',lambda *args,**kw:{'passed':kw['error'] is None and kw['stop_confirmed']})
    monkeypatch.setattr('teleop.tianyi.TianyiIntentAdapter',Adapter)
    monkeypatch.setattr('teleop.runtime.TeleopRuntime',Runtime)
    monkeypatch.setattr('teleop.protocol.bind_rtc_frame_v1',lambda f,**kw:f)
    (tmp_path/'poses.jsonl').write_text('\n'.join(json.dumps({'received_ns':i*14_000_000,'frame':{'sequence':i,'deadman':True}}) for i in range(50)))
    d=tmp_path/'result';d.mkdir()
    if feedback=='disk_full':
        with pytest.raises(ValueError,match='disk full'):a.combined({'calibration_path':'unused'},tmp_path,d)
        assert len(submitted)==1
    elif feedback=='oversleep':
        a.combined({'calibration_path':'unused'},tmp_path,d)
        assert 0<len(submitted)<50 and submitted==sorted(set(submitted))
    elif feedback=='persistent':
        with pytest.raises(ValueError,match='combined_feedback_timeout'):a.combined({'calibration_path':'unused'},tmp_path,d)
        assert len(submitted)<30 and now[0]<100.4
    else:
        a.combined({'calibration_path':'unused'},tmp_path,d)
        assert submitted==list(range(50)) and len(snapshots)==50
        assert len(heartbeats)==3 and max(b-a for a,b in zip(heartbeats,heartbeats[1:]))<.3
    rows=[json.loads(s) for s in (d/'combined.jsonl').read_text().splitlines()]
    assert shutdown==['runtime_close','link_stop','evidence_flush','evidence_flush']
    if feedback=='disk_full':
        assert rows==[]
        result=json.loads((d/'combined.result.json').read_text())
        assert result['stop_confirmed'] and not result['passed']
    elif feedback=='oversleep':
        assert len(rows)==len(submitted)
        assert sum(r['observer_timing_ms']['superseded_count'] for r in rows)+len(rows)==50
        assert max(r['observer_timing_ms']['input_lateness_ms'] for r in rows)<100
        assert json.loads((d/'combined.result.json').read_text())['stop_confirmed']
    elif feedback!='persistent':assert len(rows)==50 and rows[-1]['elapsed']==pytest.approx(.686)
    if feedback=='temporary':
        assert len([r for r in rows if r.get('event')=='feedback_unavailable'])==5
        assert json.loads((d/'combined.result.json').read_text())['feedback_gap_samples']==5


@pytest.mark.parametrize('case',['once','always','collision','resume_wait','resume_stuck','rebase'])
def test_geometry_timeout_discards_candidate_and_recomputes_with_bound(monkeypatch,case):
    from teleop.replay import execute
    now=[100.];events=[]
    monkeypatch.setattr('teleop.replay.time.monotonic_ns',lambda:int(now[0]*1e9))
    class Solver(Geometry):
        attempts=0
        def _safe_configuration(self,q,excursion=None):
            self.attempts+=1
            if case in ('always','collision') or (case=='once' and self.attempts==1):
                now[0]+=.061
                raise ValueError('torso_collision' if case=='collision' else 'collision_check_timeout')
    class Link:
        seq=0;lease=None;lease_started_ns=0;management_request=None
        def prepare_transport(self):pass
        def claim(self,deadline):self.lease={'session_id':'fixture'};self.claims=getattr(self,'claims',0)+1
        def feedback(self):return {'state':'active','feedback':{'q':[0.]*14,'dq':[0.]*14,'arm_ns':int(now[0]*1e9)}}
        def pause(self,deadline):return True
        def resume(self,deadline):
            if case=='rebase':
                self.lease=None;raise ValueError('driver_lease_rebased')
            if case in ('resume_wait','resume_stuck'):
                count=getattr(self,'resume_count',0)+1;self.resume_count=count
                if case=='resume_stuck' or count<3:raise ValueError('robot_not_stopped')
        def send(self,target,hands,deadline,**kw):self.seq+=1;return True
        def stop(self,deadline):self.lease=None;return True
    s=Solver();l=Link();points=[{'t':0,'q':[0.]*14,'phase':'baseline'}, {'t':.2,'q':[.01]*14,'phase':'both'}]
    def run():execute(l,points,s,events.append,clock=lambda:now[0],sleep=lambda t:now.__setitem__(0,now[0]+max(.001,t)))
    if case in ('once','resume_wait','rebase'):
        run();assert l.seq>0
        if case=='once':
            assert events[0]['hardware_target_sent'] is False
            assert next(r for r in events if 'q'in r)['elapsed']<.02
        elif case=='rebase':assert l.claims==2
        else:assert l.resume_count==3
    else:
        with pytest.raises(ValueError):run()
        if case=='resume_stuck':assert 3<l.resume_count<252 and now[0]<103
        else:assert l.seq==0 and s.attempts==(3 if case=='always' else 1)
    assert events[-1]['stop_confirmed'] and l.lease is None


def test_geometry_retry_reduces_only_checked_step_not_reference(monkeypatch):
    from teleop.replay import execute
    now=[100.];events=[];actual=np.zeros(14);sent=[]
    monkeypatch.setattr('teleop.replay.time.monotonic_ns',lambda:int(now[0]*1e9))
    class Solver(Geometry):
        def _safe_transition(self,start,end,budget):
            if np.max(np.abs(np.asarray(end)-start))>.0021:
                raise ValueError('collision_check_timeout')
    class Link:
        lease=None;management_request=None;lease_started_ns=0;seq=0
        def prepare_transport(self):pass
        def claim(self,deadline):self.lease={'session_id':'fixture'}
        def feedback(self):return {'state':'active','commanded_q':actual.tolist(),'feedback':{'q':actual.tolist(),'dq':[0.]*14,'arm_ns':int(now[0]*1e9)}}
        def pause(self,deadline):return True
        def resume(self,deadline):pass
        def send(self,target,hands,deadline,**kw):
            assert np.max(np.abs(np.asarray(target)-actual))<=.0021
            sent.append(target);actual[:]=target;self.seq+=1;return True
        def stop(self,deadline):self.lease=None;return True
    points=[{'t':0,'q':[.1]*14,'phase':'both'},{'t':.2,'q':[.1]*14,'phase':'both'}]
    execute(Link(),points,Solver(),events.append,clock=lambda:now[0],sleep=lambda t:now.__setitem__(0,now[0]+max(.001,t)))
    assert sent and any(r.get('event')=='rejected_transition' for r in events)
    assert all(r['target_q']==[.1]*14 for r in events if 'target_q'in r)
    assert events[-1]['stop_confirmed'] and events[-1]['error'] is None


@pytest.mark.parametrize('case',['delayed','slow_elbow','never','regress'])
def test_plateau_duration_does_not_depend_on_tracking_error(monkeypatch,case):
    from teleop.replay import execute
    now=[100.];events=[]
    monkeypatch.setattr('teleop.replay.time.monotonic_ns',lambda:int(now[0]*1e9))
    class Link:
        lease=None;management_request=None;lease_started_ns=0;seq=0
        def prepare_transport(self):pass
        def claim(self,deadline):self.lease={'session_id':'fixture'}
        def feedback(self):
            reached=case!='never' and now[0]>=(108. if case=='slow_elbow' else 102.3) and not(case=='regress' and now[0]>102.55)
            return {'state':'active','feedback':{'q':[.1 if reached else 0.]*14,'dq':[0.]*14,'arm_ns':int(now[0]*1e9)}}
        def pause(self,deadline):return True
        def resume(self,deadline):pass
        def send(self,target,hands,deadline,**kw):self.seq+=1;return True
        def stop(self,deadline):self.lease=None;return True
    l=Link();points=[{'t':0,'q':[0.]*14,'phase':'baseline'},
                    {'t':1,'q':[0.]*14,'phase':'baseline'},
                    {'t':1.5,'q':[.1]*14,'phase':'left'},
                    {'t':2.5,'q':[.1]*14,'phase':'left_plateau'}]
    def run():execute(l,points,Geometry(),events.append,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+max(.001,s)))
    run();stable=[r for r in events if r.get('phase')=='left_plateau']
    assert stable[-1]['elapsed']-stable[0]['elapsed']>.9
    assert now[0]<105 and stable[0]['q']==[0.]*14
    if case in ('never','slow_elbow'):assert all(r['q']==[0.]*14 for r in stable)
    assert not any(r.get('phase')=='left_settling' for r in events)
    assert events[-1]['stop_confirmed'] and l.lease is None


@pytest.mark.parametrize('case',['near','move','lag','collision','changed','invalid'])
def test_recorded_origin_positioning_validates_before_motion_and_measures_result(tmp_path,monkeypatch,case):
    from teleop import acceptance as a
    recording=tmp_path/'recording';recording.mkdir();(recording/'analysis').mkdir()
    directory=tmp_path/'execution';directory.mkdir();profile=tmp_path/'profile';profile.write_text('{}')
    target=[.2]*14
    if case=='invalid':target[0]=float('nan')
    row={'received_ns':100,'observed_ns':100,'frame':{'deadman':True,'tracking':{'head':True,'left':True,'right':True}},
         'driver':{'feedback':{'q':target,'arm_ns':100}}}
    (recording/'poses.jsonl').write_text(json.dumps(row))
    (recording/'analysis/trajectory.json').write_text(json.dumps({'recording_sha256':a.digest(recording/'poses.jsonl'),
        'profile_sha256':'wrong' if case=='changed' else a.digest(profile)}))
    q=[.2]*14 if case=='near' else [0.]*14;actions=[];schedules=[]
    class Solver:
        velocity=1.;indices=np.arange(14)
        model=SimpleNamespace(lowerPositionLimit=np.full(14,-1.),upperPositionLimit=np.full(14,1.))
        def __init__(self,*args):pass
        def _safe_transition(self,*args):
            if case=='collision':raise ValueError('arm_collision')
    class Link:
        lease=None;management_request=None
        def feedback(self):return {'feedback':{'q':q}}
        def close(self):pass
    link=Link();clean=SimpleNamespace(shutdown=lambda:None,join=lambda **kw:None)
    monkeypatch.setattr(a,'TianyiIK',Solver);monkeypatch.setattr(a,'idle',lambda:None)
    monkeypatch.setattr(a,'ros_link',lambda cfg:(link,clean,clean,clean))
    monkeypatch.setattr(a,'journal_link',lambda *args:None)
    monkeypatch.setattr(a,'fresh_baseline',lambda link:[{'q':q.copy()}])
    def call(port,tool,action):
        actions.append(action);return {'state':'ready','first_acceptance_prepared':True}
    monkeypatch.setattr(a,'call',call)
    def execute(link,points,solver,sink):
        schedules.append(points)
        q[:]=[.15]*14 if case=='lag' else points[-1]['q']
        sink({'event':'round_end','stop_confirmed':True})
    monkeypatch.setattr(a,'execute',execute)
    if case in ('lag','collision','changed','invalid'):
        with pytest.raises(ValueError):a.position_recorded_origin({'calibration_path':str(profile)},recording,directory)
        if case!='lag':assert not actions and not schedules
        else:assert not json.loads((directory/'origin.result.json').read_text())['passed']
    else:
        a.position_recorded_origin({'calibration_path':str(profile)},recording,directory)
        result=json.loads((directory/'origin.result.json').read_text())
        assert result['passed'] and result['released'] and result['max_error_rad']==0
        if case=='near':assert not actions and not schedules
        else:
            points=schedules[0]
            assert points[-1]['t']-points[-2]['t']==pytest.approx(3.)
            assert .2/(points[-2]['t']-points[1]['t'])<=.15+1e-9
            assert points[-1]['t']<40


@pytest.mark.parametrize('case',['recover','fault','unknown'])
def test_hold_between_geometry_and_send_discards_target_and_rechecks_state(monkeypatch,case):
    from teleop.replay import execute
    now=[100.];events=[]
    monkeypatch.setattr('teleop.replay.time.monotonic_ns',lambda:int(now[0]*1e9))
    class Link:
        lease=None;management_request=None;lease_started_ns=0;seq=0;attempts=0;state='active';resumes=0
        def prepare_transport(self):pass
        def claim(self,deadline):self.lease={'boot_id':'b','session_id':'s'}
        def feedback(self):return {'state':self.state,'reason':'unexpected' if case=='unknown' else 'command_timeout',
            'boot_id':'b','session_id':'s','ownership_held':True,'hold_confirmed':True,'continuation_allowed':False,
            'feedback':{'q':[0.]*14,'dq':[0.]*14,**{k:int(now[0]*1e9) for k in ('arm_ns','power_ns','fixed_ns')}}}
        def pause(self,deadline):return True
        def resume(self,deadline):self.resumes+=1;self.state='active'
        def send(self,target,hands,deadline,**kw):
            self.attempts+=1
            if self.attempts==1:
                self.state='fault' if case=='fault' else 'hold'
                raise ValueError('driver_holding')
            self.seq+=1;return True
        def stop(self,deadline):self.lease=None;return True
    l=Link();points=[{'t':0,'q':[0.]*14,'phase':'both'},{'t':.1,'q':[.01]*14,'phase':'both'}]
    def run():execute(l,points,Geometry(),events.append,clock=lambda:now[0],sleep=lambda s:now.__setitem__(0,now[0]+max(.001,s)))
    if case=='recover':
        run();assert l.seq>0 and l.resumes>=1
        assert any(r.get('event')=='explicit_hold_recovery' for r in events)
    else:
        with pytest.raises(ValueError,match='driver_fault' if case=='fault' else 'driver_hold'):run()
        assert l.seq==0 and l.attempts==1
    assert events[0]['event']=='waiting_send_feedback' and not events[0]['hardware_target_sent']
    assert events[-1]['stop_confirmed'] and l.lease is None


@pytest.mark.parametrize('failure',['transient','persistent','model_invalid'])
def test_calibration_waits_only_for_transient_feedback_without_authority(monkeypatch,failure):
    from teleop import acceptance as a
    now=[100.];calls=[]
    monkeypatch.setattr(a.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(a.time,'sleep',lambda t:now.__setitem__(0,now[0]+t))
    def calibrate(path):
        calls.append(path)
        if failure=='model_invalid':raise ValueError('calibration_model_changed')
        if failure=='persistent' or len(calls)<3:raise ValueError('driver_feedback_stale_or_different_clock')
        return {'hardware_output':False}
    adapter=SimpleNamespace(calibrate=calibrate)
    if failure=='transient':
        assert a.calibrate_fresh(adapter,'site')['hardware_output'] is False
        assert len(calls)==3 and now[0]<100.03
    else:
        with pytest.raises(ValueError,match='calibration_model_changed' if failure=='model_invalid' else 'driver_feedback_stale'):
            a.calibrate_fresh(adapter,'site')
        if failure=='model_invalid':assert len(calls)==1
        assert now[0]<101.02


def test_html_report_includes_failed_combined_full_ik_and_gaps(tmp_path):
    from teleop.acceptance import report
    rows=[{'elapsed':i*.02,'q':[0.]*14,'target_q':[.01]*14,
           'ik_reference_q':None if i==2 else [1.]*14} for i in range(5)]
    (tmp_path/'combined.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'combined.result.json').write_text(json.dumps({'passed':False,'error':'tracking_failure'}))
    report(tmp_path)
    page=(tmp_path/'report.html').read_text()
    assert 'tracking_failure' in page and 'combined.jsonl' in page
    assert '1.010 rad' in page  # scale uses the full IK target, not 0.01rad command
    assert page.count('data-series="ik_reference_q"')==28  # two pieces per joint
    assert 'data-series="target_q"' not in page
    assert not json.loads((tmp_path/'report.json').read_text())['passed']
