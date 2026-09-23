"""Offline safety/kinematics checks. Synthetic model and feedback, never hardware."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.adapter import IntentAdapter
from teleop.descriptor import CAPABILITIES
from teleop.dispatch import MotionIntent, StopRequest, RecordingAdapter, RECOVERABLE_HOLD_CODES
from teleop.kinematics import ARM_NAMES, RelativeMapping, TianyiIK
from teleop.protocol import ProtocolError, bind_rtc_frame_v1
from teleop.runtime import TeleopRuntime


def test_declared_effectors_match_enabled_outputs():
    enabled={name for name,output in CAPABILITIES['outputs'].items() if output['enabled']}
    assert set(CAPABILITIES['effectors'])==enabled


def frame(seq=0,held=False,clutch=0):
    pose={'position':[0.,1.,0.], 'orientation':[0.,0.,0.,1.]}
    return dict(schema_version=1,sequence=seq,client_monotonic_ns=seq+1,mode='shadow',
        deadman=held,clutch_sequence=clutch,tracking=dict(head=True,left_controller=True,right_controller=True),
        head=copy.deepcopy(pose),left_controller=copy.deepcopy(pose),right_controller=copy.deepcopy(pose),
        controllers={s:{'axes':[0.,0.,0.,0.],'buttons':[0.,float(held)]} for s in ('left','right')})


def test_relative_mapping_reclutch_is_measured_pose_and_head_only_initializes():
    f=frame();palms=[np.eye(4),np.eye(4)];palms[0][1,3]=.3;palms[1][1,3]=-.3
    mapper=RelativeMapping();mapper.reset(f,palms)
    for a,b in zip(mapper.targets(f),palms):np.testing.assert_allclose(a,b)
    f['left_controller']['position'][0]+=.2
    result=mapper.targets(f)
    np.testing.assert_allclose(result[0][:3,3],[0,.2,0],atol=1e-10)
    f['head']['orientation']=[0.,1.,0.,0.]
    np.testing.assert_allclose(mapper.targets(f)[0],result[0])
    mapper.reset(f,palms)
    np.testing.assert_allclose(mapper.targets(f)[0],palms[0])


@pytest.mark.parametrize('stop_ok',[True,False])
@pytest.mark.parametrize('code,auto_collision',[(c,a) for a in (False,True) for c in sorted(RECOVERABLE_HOLD_CODES) if not a or c not in {'torso_collision','arm_collision','g1_body_collision'}])
def test_unreachable_requires_stop_ack_and_physical_reclutch(stop_ok,code,auto_collision):
    from teleop.dispatch import AdapterAck
    class RejectOnce(RecordingAdapter):
        auto_collision_recovery=auto_collision
        auto_ik_recovery=code not in {"ik_target_unreachable","ik_not_converged","ik_timeout"}
        rejected=False
        def apply(self,intent):
            if not self.rejected:
                self.rejected=True
                return AdapterAck(False,code)
            return super().apply(intent)
        def safe_stop(self,request):
            if request.reason==code and not stop_ok:
                return AdapterAck(False,'stop_unconfirmed')
            return super().safe_stop(request)
    adapter=RejectOnce();runtime=TeleopRuntime(mode='shadow',adapter=adapter,auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(f):return runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='shadow'),source='test')
        send(frame(0));send(frame(1,True,1))
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            status=runtime.status()
            if status['state']=='fault' or (status['state']=='hold' and status['dispatch']['stop_acknowledged']):break
            time.sleep(.005)
        assert status['state']==('hold' if stop_ok else 'fault')
        assert not status['output_active']
        if stop_ok:
            assert status['reason']==code
            assert status['counters'][f'dispatch_hold_{code}']==1
            send(frame(2,True,1))
            assert runtime.status()['state']=='hold'
            send(frame(3));send(frame(4,True,2))
            assert runtime._dispatcher.wait_dispatched(4)
            assert runtime.status()['counters'][f'dispatch_hold_{code}']==1
    finally:runtime.close()


def test_background_input_and_watchdog_skip_diagnostics_but_keep_safety(monkeypatch):
    now=[time.monotonic()]
    runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),auto_watchdog=False,clock=lambda:now[0])
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def no_snapshot(*args,**kwargs):raise AssertionError('diagnostics copied on background path')
        with monkeypatch.context() as patch:
            patch.setattr(runtime,'_public_snapshot_locked',no_snapshot)
            patch.setattr(runtime._dispatcher,'snapshot',no_snapshot)
            for seq,held in enumerate((False,True)):
                result=runtime.submit_frame(bind_rtc_frame_v1(frame(seq,held,seq),authority=binding,expected_mode='shadow'),
                                            source='test',include_status=False)
                assert result=={'accepted_sequence':seq}
            bad=frame(2,True,1);bad['controllers']['right']['buttons'][1]=0
            with pytest.raises(ProtocolError,match='both squeeze'):
                runtime.submit_frame(bind_rtc_frame_v1(bad,authority=binding,expected_mode='shadow'),source='test',include_status=False)
            now[0]+=.3
            assert runtime.watchdog_tick(include_status=False)=={}
        assert runtime.status()['reason']=='pose_timeout'
    finally:runtime.close()


def test_runtime_requires_both_grips_neutral_then_new_clutch_and_stops():
    adapter=RecordingAdapter();runtime=TeleopRuntime(mode='shadow',adapter=adapter,auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(f):return runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='shadow'),source='test')
        send(frame(0,True,1));assert runtime.status()['state']=='prepared_shadow'
        send(frame(1));send(frame(2,True,2))
        assert runtime._dispatcher.wait_dispatched(2)
        send(frame(3));assert runtime.status()['state']=='hold'
        bad=frame(4,True,3);bad['controllers']['right']['buttons'][1]=0
        with pytest.raises(ProtocolError,match='both squeeze'):send(bad)
        assert not runtime.status()['output_active'] and not runtime.status()['publisher_present']
    finally:runtime.close()


@pytest.mark.parametrize('interruption', ['rtc', 'tracking'])
@pytest.mark.parametrize('released_frames', [1, 2, 3])
def test_hold_requires_physical_release_not_only_inhibited_deadman(interruption, released_frames):
    runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter(),auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,generation=runtime.rtc_authority_snapshot()
        def send(f):return runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='shadow'),source='test')
        send(frame(0));send(frame(1,True,1))
        assert runtime.status()['state']=='active_shadow'
        if interruption=='rtc':
            runtime.mark_rtc_disconnected(generation)
            runtime.mark_channel(generation,'teleop-control',True)
            runtime.mark_channel(generation,'teleop-pose',True)
        else:
            lost=frame(2,True,1);lost['tracking']['head']=False;lost['head']=None
            send(lost)
        inhibited=frame(3,True,1);inhibited['deadman']=False
        send(inhibited);send(frame(4,True,2))
        assert runtime.status()['state']=='hold'
        for seq in range(5,5+released_frames):send(frame(seq,False,2))
        send(frame(5+released_frames,True,3))
        assert runtime.status()['state']=='active_shadow'
        assert not runtime.status()['output_active']
    finally:runtime.close()


class Link:
    def __init__(self):self.lease=None;self.writes=[];self.claims=0
    def feedback(self):return {'calibration_sha256':'test','feedback':{'q':[0.]*14,'arm_ns':time.monotonic_ns()}}
    def claim(self,deadline):self.claims+=1;self.lease={'session_id':'test'}
    def send(self,q,h,deadline):self.writes.append((q,h))
    def stop(self,deadline):self.lease=None;return True


class Solver:
    profile={'version':'test'};profile_sha256='test';last_ms=1.
    def palms(self,q):return [np.eye(4),np.eye(4)]
    def solve(self,targets,q,**kwargs):return list(q)


def intent(generation=1):
    now=time.monotonic()
    return MotionIntent(1,generation,1,1,now,now+.1,frame(1,True,1),True,now)


def test_shadow_never_claims_or_publishes_and_expired_ik_is_rejected():
    link=Link();adapter=IntentAdapter(link,'shadow');adapter.solver=Solver()
    assert adapter.apply(intent()).ok
    assert link.claims==0 and link.writes==[]
    def late(targets,q,**kwargs):time.sleep(.105);return q
    adapter.solver.solve=late
    assert not adapter.apply(intent()).ok
    assert link.claims==0 and link.writes==[]


def test_stop_fences_a_solver_before_it_can_claim_and_send():
    link=Link();adapter=IntentAdapter(link,'live');adapter.solver=Solver()
    entered=threading.Event();unblock=threading.Event();result=[]
    def slow(targets,q,**kwargs):entered.set();unblock.wait(.5);return q
    adapter.solver.solve=slow
    worker=threading.Thread(target=lambda:result.append(adapter.apply(intent())))
    worker.start();assert entered.wait(.2)
    now=time.monotonic()
    # A blocked solver cannot generate a successful stop acknowledgement.
    ack=adapter.safe_stop(StopRequest(2,'test',now,now+.005))
    assert not ack.ok
    unblock.set();worker.join(.5)
    assert not result[0].ok and link.writes==[] and link.claims==0
    now=time.monotonic();assert adapter.safe_stop(StopRequest(2,'test',now,now+.1)).ok


def synthetic_profile(tmp_path):
    # Anatomically meaningless fixture: exercises real FK/IK and collision code only.
    xml=['<robot name="offline"><link name="torso_link"/>']
    axes=['0 1 0','1 0 0','0 0 1','0 1 0','0 0 1','0 1 0','1 0 0']
    for side,sign in [('left',1),('right',-1)]:
        parent='torso_link'
        for i,name in enumerate(n for n in ARM_NAMES if n.startswith(side)):
            child=name.replace('_joint','_link');xyz=f'0 {sign*.5} 0' if i==0 else '.08 0 0'
            xml.append(f'<link name="{child}"/><joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/><axis xyz="{axes[i]}"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>')
            parent=child
    xml.append('</robot>');urdf=tmp_path/'synthetic.urdf';urdf.write_text(''.join(xml))
    profile=dict(arm_joint_names=list(ARM_NAMES),torso_frame='torso_link',schema='motus.tianyi-calibration.v1',version='offline-only',urdf_path=str(urdf),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),locked_joints={},
        palm_frames={s:dict(position=[.05,0.,0.],orientation=[0.,0.,0.,1.]) for s in ('left','right')},
        workspace={'torso_box':[[-.2,-.1,-.2],[.1,.1,.2]],'left':[[0,.3,-1],[1,1,1]],'right':[[0,-1,-1],[1,-.3,1]],
            'capsules':[{'from':f'{s}_elbow_pitch_link','to':f'{s}_wrist_roll_link','radius_m':.02,'group':s} for s in ('left','right')]})
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile));return path


def test_real_numerical_solver_segment_collision_and_model_hash(tmp_path):
    path=synthetic_profile(tmp_path);solver=TianyiIK(path)
    assert np.all(solver.sweep_coefficients[0][7:]==0)
    assert np.all(solver.sweep_coefficients[1][:7]==0)
    q=np.full(14,.1);result=solver.self_test(q)
    assert result['max_error_rad']<1e-3 and not result['hardware_output']
    desired=q.copy();desired[3]+=.01
    actual=solver.solve(solver.palms(desired),q)
    assert len(actual)==14 and solver.last_ms<=100
    solver.workspace['torso_box']=[[-2,-2,-2],[2,2,2]]
    with pytest.raises(ValueError,match='torso_collision'):solver.solve(solver.palms(q),q)
    profile=json.loads(path.read_text());profile['urdf_sha256']='0'*64;path.write_text(json.dumps(profile))
    with pytest.raises(ValueError,match='model_changed'):TianyiIK(path)


@pytest.mark.parametrize('invalid',[None,'boot','old','owned'])
def test_card_stop_waits_for_real_release_without_resending(monkeypatch,invalid):
    from teleop import plugin
    from teleop.adapter import DriverLink
    card=plugin.TeleopPlugin({},None)
    link=DriverLink.__new__(DriverLink);link.condition=threading.Condition()
    link.lease={'boot_id':'boot'};link.release_requested_ns=time.monotonic_ns()
    link.latest={'boot_id':'boot','monotonic_ns':link.release_requested_ns-1,
                 'ownership_held':True,'stop_confirmed':False}
    card.link=link;sent=[];now=[0.];polls=[]
    card.adapter=SimpleNamespace(close=lambda:(sent.append(1) or SimpleNamespace(ok=False)))
    original=link.reconcile_release
    def reconcile():
        polls.append(1)
        if len(polls)==2:
            link.latest={'boot_id':'other' if invalid=='boot' else 'boot',
                         'monotonic_ns':link.release_requested_ns-1 if invalid=='old' else time.monotonic_ns(),
                         'ownership_held':invalid=='owned','stop_confirmed':True}
        return original()
    link.reconcile_release=reconcile
    monkeypatch.setattr(plugin,'time',SimpleNamespace(monotonic=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n)))
    assert card._release_driver()==(invalid is None)
    assert sent==[1] and (link.lease is None)==(invalid is None)
    assert now[0]<=1.


def test_host_failure_cleans_up_and_config_can_retry(monkeypatch,tmp_path):
    from teleop import plugin
    links=[]
    class HostLink(Link):
        def __init__(self,*args):super().__init__();self.closed=False;links.append(self)
        def pause(self,deadline):raise AssertionError('Shadow must not call hardware pause')
        def close(self):self.closed=True
    monkeypatch.setattr(plugin,'DriverLink',HostLink)
    card=plugin.TeleopPlugin({'capture':{'state_file':str(tmp_path/'state'),
        'public_wss_url':'wss://127.0.0.1:15731/ws/teleop-capture',
        'tls_cert_file':str(tmp_path/'missing-cert')}},None)
    result=card.dispatch('teleop',{'action':'pair_headset'})
    assert result['state']=='error'
    assert links[0].closed and card.runtime is None and card._loop is None
    result=card.dispatch('teleop',{'action':'config','position_scale':.4})
    assert result['state']=='idle' and card.cfg['position_scale']==.4
    assert all(t.name!='actucore-capture' for t in threading.enumerate())


def test_official_model_joint_mapping_matches_direct_fk(tmp_path):
    import pinocchio as pin
    source=Path(__file__).parents[1]/'plugins'/'teleop'
    profile=json.loads((source/'calibration.example.json').read_text())
    urdf=source/'models'/'tianyi2-official.urdf'
    profile['urdf_path']=str(urdf)
    profile['locked_joints']={name:0. for name in profile['locked_joints']}
    profile['palm_frames']={s:dict(position=[0.,0.000035454 if s=='left' else -0.000035405,-0.084799],
        orientation=[0.,0.,0.,1.]) for s in ('left','right')}
    # This fixture tests FK only; its invented envelope is NOT a live calibration.
    profile['workspace']={'torso_box':[[-.1,-.1,-.1],[.1,.1,.1]],
        'capsules':[{'from':'elbow_pitch_l_joint','to':'wrist_roll_l_joint','radius_m':.02,'group':'test'}]}
    path=tmp_path/'offline-official.json';path.write_text(json.dumps(profile))
    solver=TianyiIK(path);model=pin.buildModelFromUrdf(str(urdf));data=model.createData()
    wire=np.linspace(-.1,.1,14);q=pin.neutral(model)
    for name,value in zip(profile['arm_joint_names'],wire):q[model.joints[model.getJointId(name)].idx_q]=value
    pin.framesForwardKinematics(model,data,q)
    torso=data.oMf[model.getFrameId(profile['torso_frame'])].inverse()
    for side,actual in zip(('left','right'),solver.palms(wire)):
        expected=(torso*data.oMf[model.getFrameId(side+'_tcp_link')]).homogeneous
        np.testing.assert_allclose(actual,expected,atol=1e-8)


@pytest.mark.parametrize('site_ready', [True, False])
def test_real_mcp_advertises_card_and_reports_operation_failure_without_ros(tmp_path, monkeypatch, site_ready):
    import ast,logging,urllib.request
    from test_teleop_site import site_configuration
    teleop_cfg=site_configuration(tmp_path,monkeypatch)
    key=tmp_path/'management-key'
    if not site_ready:teleop_cfg.pop('capture')
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    root=Path(__file__).parents[1];sys.path.insert(0,str(root))
    tree=ast.parse((root/'main.py').read_text())
    nodes=[n for n in tree.body if getattr(n,'name','') in ('ActuCoreBundle','make_handler')]
    ns={'BaseHTTPRequestHandler':BaseHTTPRequestHandler,'json':json,'log':logging.getLogger('test'),
        '_brief':repr}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'main.py','exec'),ns)
    bundle=ns['ActuCoreBundle']({'plugins':{'vla':{'enabled':True,'provider':'mock'},'teleop':teleop_cfg}},None)
    ns['_bundle']=bundle
    server=ThreadingHTTPServer(('127.0.0.1',0),ns['make_handler']())
    thread=threading.Thread(target=server.serve_forever);thread.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def rpc(method,params,authenticated=True):
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/mcp',
            json.dumps(dict(jsonrpc='2.0',id=1,method=method,params=params)).encode(),
            {'Content-Type':'application/json', 'X-Teleop-Management':key.read_text() if authenticated else ''})
        with opener.open(request,timeout=1) as response:return json.load(response)['result']
    try:
        assert rpc('initialize',{})['serverInfo']['name']=='actucore-bundle'
        # Core deduplicates by MCP serverInfo.name, not only the registration label.
        dedicated=ns['ActuCoreBundle']({'name':'ActuCore PICO Shadow','plugins':{}},None)
        ns['_bundle']=dedicated
        assert rpc('initialize',{})['serverInfo']['name']=='ActuCore PICO Shadow'
        ns['_bundle']=bundle
        catalog=rpc('tools/list',{})
        tools={t['name']:t for t in catalog['tools']}
        ordinary=rpc('tools/call',{'name':'vla','arguments':{'action':'info'}},False)
        assert json.loads(ordinary['content'][0]['text'])['state']=='idle'
        if not site_ready:
            assert set(tools)=={'vla'}
            assert catalog['_meta']['required_site_config']['teleop']
            info=rpc('tools/call',{'name':'teleop','arguments':{'action':'info'}},False)
            assert info['isError'] is True
            assert json.loads(info['content'][0]['text'])['error']=='required_site_config'
            return
        assert set(tools)=={'vla','teleop'} and tools['teleop']['type']=='processor'
        assert catalog['_meta']['required_site_config']=={}
        denied=rpc('tools/call',{'name':'teleop','arguments':{'action':'open_pairing'}},False)
        assert 'teleop_management_unauthorized' in denied['content'][0]['text']
        result=rpc('tools/call',{'name':'teleop','arguments':{'action':'config','mode':'invalid'}})
        assert result['isError'] is True
        assert bundle._plugins[1].cfg['mode']=='shadow' and bundle._plugins[1].runtime is None
    finally:server.shutdown();thread.join(1);server.server_close()


def test_core_management_key_stays_at_explicit_loopback_endpoint(tmp_path, monkeypatch):
    import ast
    import importlib.util
    from types import SimpleNamespace
    class Denied(Exception): pass
    path=Path(__file__).parents[2]/'agent-core/src/api/mcp_manage.py'
    # Load the real stdlib helper explicitly: the ActuCore image intentionally
    # has no Core src on sys.path. Do not import FastAPI or leak a Core path into
    # unrelated ActuCore tests; monkeypatch restores any existing module entry.
    spec=importlib.util.spec_from_file_location('teleop_management',path.parents[1]/'teleop_management.py')
    management=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(management)
    monkeypatch.setitem(sys.modules,'teleop_management',management)
    node=next(n for n in ast.parse(path.read_text()).body if getattr(n,'name','')=='_teleop_management_headers')
    ns={'fastapi':SimpleNamespace(HTTPException=Denied)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    headers=ns['_teleop_management_headers']
    key=tmp_path/'key';key.write_text('a'*64)
    monkeypatch.setenv('TELEOP_MANAGEMENT_KEY_FILE',str(key))
    monkeypatch.setenv('TELEOP_MANAGEMENT_URL','http://localhost:15740/mcp')
    assert headers('http://localhost:15740/mcp','teleop',{'action':'approve_pairing'})=={'X-Teleop-Management':'a'*64}
    for url in ('http://example.com/mcp','http://localhost:15741/mcp','http://localhost:15740/mcp?x=1'):
        with pytest.raises(Denied):headers(url,'teleop',{'action':'approve_pairing'})
    assert headers('http://example.com/mcp','other',{'action':'start'})=={}
    assert headers('http://example.com/mcp','teleop',{'action':'info'})=={}
    key.unlink()
    with pytest.raises(Denied):headers('http://localhost:15740/mcp','teleop',{'action':'approve_pairing'})

@pytest.mark.parametrize('code',['torso_collision','arm_collision','g1_body_collision','ik_target_unreachable','ik_not_converged','ik_timeout','feedback_unavailable'])
@pytest.mark.parametrize('stop_delay',[.07,.25])
def test_collision_recheck_keeps_held_grips_and_waits_for_stop(code,stop_delay):
    from teleop.dispatch import AdapterAck
    class CollisionAdapter(RecordingAdapter):
        stop_confirmation_timeout_ms=500 if stop_delay>.1 else 100
        auto_shadow_feedback_recovery=True
        auto_collision_recovery=True
        auto_ik_recovery=True
        blocked=True
        entered=threading.Event()
        release=threading.Event()
        def apply(self,intent):
            if self.blocked:return AdapterAck(False,code)
            return super().apply(intent)
        def safe_stop(self,request):
            if request.reason==code:
                self.entered.set();self.release.wait(stop_delay)
            return super().safe_stop(request)
    adapter=CollisionAdapter();runtime=TeleopRuntime(mode='shadow',adapter=adapter,auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(seq,held=True):
            return runtime.submit_frame(bind_rtc_frame_v1(frame(seq,held,0 if seq==0 else 1),authority=binding,expected_mode='shadow'),source='test')
        send(0,False);send(1);assert adapter.entered.wait(.2)
        # A valid frame cannot bypass an unacknowledged hold.
        send(2);assert runtime.status()['state']=='hold'
        if stop_delay>.1:
            time.sleep(.15)
            assert runtime.status()['state']=='hold'
            assert not runtime.status()['dispatch']['stop_acknowledged']
            assert runtime.status()['dispatch']['last_would_apply_sequence'] is None
        adapter.release.set()
        deadline=time.monotonic()+.3
        while not runtime.status()['dispatch']['stop_acknowledged'] and time.monotonic()<deadline:time.sleep(.001)
        send(3)
        deadline=time.monotonic()+.3
        while runtime.status()['state']!='hold' and time.monotonic()<deadline:time.sleep(.001)
        assert runtime.status()['state']=='hold' # still colliding, still no apply
        adapter.blocked=False
        deadline=time.monotonic()+.3
        while not runtime.status()['dispatch']['stop_acknowledged'] and time.monotonic()<deadline:time.sleep(.001)
        send(4);assert runtime._dispatcher.wait_dispatched(4)
        assert runtime.status()['counters']['feedback_rechecks' if code=='feedback_unavailable' else 'ik_rechecks' if code.startswith('ik_') else 'collision_rechecks']>=2
    finally:adapter.release.set();runtime.close()


@pytest.mark.parametrize('code',['g1_body_collision','ik_target_unreachable','ik_not_converged','ik_timeout'])
def test_collision_stop_failure_never_auto_recovers(code):
    from teleop.dispatch import AdapterAck
    class FailedStop(RecordingAdapter):
        auto_collision_recovery=True
        auto_ik_recovery=True
        def apply(self,intent):return AdapterAck(False,code)
        def safe_stop(self,request):
            if request.reason==code:return AdapterAck(False,'stop_unconfirmed')
            return super().safe_stop(request)
    runtime=TeleopRuntime(mode='shadow',adapter=FailedStop(),auto_watchdog=False)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(seq,held,clutch):return runtime.submit_frame(bind_rtc_frame_v1(frame(seq,held,clutch),authority=binding,expected_mode='shadow'),source='test')
        send(0,False,0);send(1,True,1)
        end=time.monotonic()+.3
        while runtime.status()['state']!='fault' and time.monotonic()<end:time.sleep(.001)
        assert runtime.status()['state']=='fault'
        with pytest.raises(ProtocolError):send(2,True,1)
    finally:runtime.close()


def test_longer_stop_confirmation_does_not_extend_motion_deadline():
    class SlowMotion(RecordingAdapter):
        stop_confirmation_timeout_ms=500
        entered=threading.Event()
        def apply(self,intent):
            self.entered.set();time.sleep(.16)
            return super().apply(intent)
    a=SlowMotion();r=TeleopRuntime(mode='shadow',adapter=a,auto_watchdog=False)
    try:
        r.prepare_local_session();binding,_=r.rtc_authority_snapshot()
        for seq,held in ((0,False),(1,True)):
            r.submit_frame(bind_rtc_frame_v1(frame(seq,held,seq),authority=binding,expected_mode='shadow'),source='test')
        assert a.entered.wait(.2)
        time.sleep(.18)
        state=r.status()
        assert state['state']=='fault'
        assert state['dispatch']['fault_code'] in ('adapter_io_stalled','adapter_apply_timeout','adapter_motion_deadline_missed','intent_expired')
    finally:r.close()


def test_card_configuration_readback_and_rejection_preserves_values():
    from teleop.plugin import TeleopPlugin
    card=TeleopPlugin({'mode':'shadow','position_scale':.5,'capture':{'secret':'never-exposed'}},None)
    assert card.info()['configuration']['position_scale']==.5
    assert 'capture' not in card.info()['configuration']
    for values in ({'position_scale':float('nan')},{'position_scale':.001},{'robot_profile':'other'},{'shadow_feedback_source':'other'}):
        assert card.dispatch('teleop',{'action':'config',**values})['error']
        assert card.cfg['position_scale']==.5
    value=card.dispatch('teleop',{'action':'config','position_scale':.8})
    assert value['configuration']['position_scale']==.8


def test_real_configuration_write_error_is_redacted_and_retryable(tmp_path):
    from teleop.plugin import TeleopPlugin
    blocked = tmp_path/'private-site-state'
    blocked.write_text('not a directory')
    card = TeleopPlugin({'mode':'shadow', 'position_scale':.5,
        'capture':{'state_file':str(blocked/'capture.json')}}, None)
    result = card.dispatch('teleop', {'action':'config', 'position_scale':.6})
    assert result == {'state':'error', 'error':'teleop_io_error', 'code':'teleop_io_error'}
    assert str(blocked) not in json.dumps(result) + json.dumps(card.info())
    assert card.info()['reason'] == 'teleop_io_error' and card.cfg['position_scale'] == .5
    assert card.runtime is None and not card._config_file.exists()
    blocked.unlink();blocked.mkdir()
    result = card.dispatch('teleop', {'action':'config', 'position_scale':.6})
    assert result['state'] == 'idle' and result['configuration']['position_scale'] == .6
    assert json.loads(card._config_file.read_text())['values']['position_scale'] == .6


@pytest.mark.parametrize('kind, expected', [
    ('protocol','session_inactive'), ('capture','invalid_capture_id'),
    ('validation','calibrate_before_start'), ('timeout','teleop_timeout'),
    ('io','teleop_io_error'), ('unexpected','teleop_not_ready'),
    ('token_only','teleop_not_ready'), ('bad_code','teleop_not_ready'),
    ('bad_code_type','teleop_not_ready'),
])
def test_public_errors_keep_known_codes_without_exception_details(monkeypatch, kind, expected):
    from teleop.plugin import TeleopPlugin
    from teleop.capture import CaptureError
    detail = 'https://user:credential@private-host/private/path?token=secret'
    errors = {'protocol':ProtocolError('session_inactive', detail),
        'capture':CaptureError('invalid_capture_id'), 'validation':ValueError('calibrate_before_start'),
        'timeout':TimeoutError(detail), 'io':OSError(detail), 'unexpected':RuntimeError(detail),
        'token_only':ValueError('secret_token_0123456789'), 'bad_code':CaptureError(detail), 'bad_code_type':CaptureError(['secret'])}
    card = TeleopPlugin({}, None)
    def fail():raise errors[kind]
    monkeypatch.setattr(card, '_ensure_host', fail)
    reply = card.dispatch('teleop', {'action':'calibrate'})
    assert reply == {'state':'error', 'error':expected, 'code':expected}
    assert card.info()['reason'] == expected
    assert 'credential' not in json.dumps(reply) + json.dumps(card.info())


def test_close_failure_prevents_configuration_success_then_allows_retry(tmp_path):
    from teleop.plugin import TeleopPlugin
    card = TeleopPlugin({'mode':'shadow', 'position_scale':.5,
        'capture':{'state_file':str(tmp_path/'capture.json')}}, None)
    calls = []; failing = [True]
    def fail_close():
        calls.append('close')
        if failing[0]:raise OSError('private-site-path credential=secret')
    card.runtime = SimpleNamespace(status=lambda:{'authority_valid':False}, close=fail_close)
    card.link = SimpleNamespace(lease=None, close=lambda:calls.append('link_close'))
    reply = card.dispatch('teleop', {'action':'config', 'position_scale':.6})
    assert reply == {'state':'error', 'error':'teleop_stop_failed', 'code':'teleop_stop_failed'}
    assert calls == ['close'] and card.runtime is not None and card.link is not None
    assert card.cfg['position_scale'] == .5 and not card._config_file.exists()
    assert card.error == 'teleop_stop_failed'
    # Retry closes the same failed resource; no lease or target is restored.
    failing[0] = False
    reply = card.dispatch('teleop', {'action':'config', 'position_scale':.6})
    assert reply['state'] == 'idle' and reply['configuration']['position_scale'] == .6
    assert card.runtime is None and card.link is None
    assert calls == ['close','close','link_close']


@pytest.mark.parametrize('code', ['arm_ns_stale','power_ns_stale','fixed_ns_stale','hand_ns_stale',
    'hold_not_resumable','invalid_lease','robot_not_stopped','management_request_expired'])
def test_driver_recovery_codes_survive_public_redaction(monkeypatch, code):
    from teleop.plugin import TeleopPlugin
    card=TeleopPlugin({}, None)
    def fail():raise ValueError(code)
    monkeypatch.setattr(card, '_ensure_host', fail)
    result=card.dispatch('teleop', {'action':'calibrate'})
    assert result['error']==result['code']==code
    assert card.info()['reason']==code


@pytest.mark.parametrize('error, code', [(TimeoutError('private password=secret'), 'teleop_timeout'),
    (ProtocolError('dispatch_stop_unconfirmed','private path'), 'dispatch_stop_unconfirmed')])
def test_close_failure_preserves_timeout_and_unconfirmed_receipts(error, code):
    from teleop.plugin import TeleopPlugin
    card=TeleopPlugin({}, None)
    def fail():raise error
    card.runtime=SimpleNamespace(close=fail)
    with pytest.raises(ProtocolError) as result:card.stop()
    assert result.value.code==str(result.value)==card.error==code
    assert card.runtime is not None


@pytest.mark.parametrize('failing_component', ['runtime','recorder'])
@pytest.mark.parametrize('released', [True, False])
def test_close_error_still_attempts_driver_release_and_keeps_retry_state(monkeypatch, failing_component, released):
    from teleop.plugin import TeleopPlugin
    card=TeleopPlugin({}, None)
    calls=[]
    def close(name):
        calls.append(name)
        if name==failing_component:raise OSError('private/site token=secret')
    card.recorder=SimpleNamespace(stop=lambda:close('recorder'))
    card.runtime=SimpleNamespace(close=lambda:close('runtime'))
    card.adapter=object()
    card.link=SimpleNamespace(lease={'test':'only'},close=lambda:close('link'))
    monkeypatch.setattr(card,'_release_driver',lambda:(calls.append('release') or released))
    with pytest.raises(ProtocolError) as result:card.stop()
    assert result.value.code==('teleop_stop_failed' if released else 'stop_unconfirmed')
    assert calls==['recorder','runtime','release']
    assert card.runtime is not None and card.link is not None
