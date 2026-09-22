"""Real MCP JSON/Driver dispatch with lost replies; no network or hardware."""
import io
import json
import os
import time
from types import MethodType, SimpleNamespace
import pytest
from test_tianyi_execution_chain import chain, intent, stop
from teleop.adapter import DriverLink


@pytest.mark.parametrize('mode',['retry','cancel','wrong_nonce','wrong_boot'])
def test_initial_stale_claim_is_retryable_and_cancel_does_not_invent_stop(managed,monkeypatch,mode):
    _,link,gate,writes,requests,_=managed
    original=gate._feedback
    monkeypatch.setattr(gate,'_feedback',lambda:(_ for _ in ()).throw(ValueError('power_ns_stale')))
    with pytest.raises(ValueError,match='driver_management_pending'):link.claim(time.monotonic()+.1)
    assert gate.session_id is None and not writes and not gate.status()['stop_confirmed']
    request_id=link.management_request['id']
    monkeypatch.setattr(gate,'_feedback',original)
    if mode=='retry':
        link.claim(time.monotonic()+.1)
        assert requests[-1]['request_id']==request_id and gate.session_id
        assert link.stop(time.monotonic()+.1)
    else:
        call=link.call
        def altered(action,deadline):
            result=call(action,deadline)
            if action=='release' and mode=='wrong_nonce':result['cancelled_request_id']='wrong'
            if action=='release' and mode=='wrong_boot':result['boot_id']='wrong'
            return result
        link.call=altered
        assert link.stop(time.monotonic()+.025)==(mode=='cancel')
        assert not gate.status()['stop_confirmed']  # Nothing moved; no physical stop fabricated.
        if mode!='cancel':
            assert link.management_request
            link.call=call;assert link.stop(time.monotonic()+.1)
        assert not link.management_request and not link.lease and not writes


@pytest.fixture
def managed(chain,monkeypatch):
    monkeypatch.syspath_prepend(os.environ['TIANYI_DRIVER_SOURCE'])
    from teleop_executor import TeleopExecutor
    a,link,gate,writes=chain
    e=TeleopExecutor({},'isolated',None,SimpleNamespace(_pos_publisher=True),
                     SimpleNamespace(_left_pub=True,_right_pub=True),[])
    e.gate=gate;e.start=lambda:None;e.foreign_publishers=lambda:[];e.profile={'hands_enabled':False}
    link.url='http://127.0.0.1:1/mcp';link.management_retry=True;link.management_request=None
    link.call=MethodType(DriverLink.call,link);link.claim=MethodType(DriverLink.claim,link)
    requests=[];lose=[]
    def request(req,timeout):
        args=json.loads(req.data)['params']['arguments'];requests.append(dict(args))
        value=e.dispatch(args['action'],args)
        if args['action'] in ('pause','release','recoverable_hold'):
            gate.tick();time.sleep(.001);gate.tick()
        if lose and lose[0]==args['action']:
            lose.pop(0);raise TimeoutError('reply lost')
        return io.StringIO(json.dumps({'result':{'isError':bool(value.get('error')),
                            'content':[{'type':'text','text':json.dumps(value)}]}}))
    link.opener=SimpleNamespace(open=request)
    return a,link,gate,writes,requests,lose


@pytest.mark.parametrize('action',['claim','resume'])
def test_lost_reply_recovers_without_reusing_old_lease_or_replaying_target(managed,action):
    a,link,gate,writes,requests,lose=managed
    if action=='resume':
        assert a.apply(intent()).ok
        assert stop(a,'torso_collision').ok
        assert a.apply(intent(2,generation=2,forward=.02)).ok  # validate only
    lose.append(action);before=len(writes)
    assert a.apply(intent(3,generation=2 if action=='resume' else 1)).ok
    assert a.output['state']=='waiting_driver_management'
    owner=gate.session_id
    assert link.management_request and len(writes)==before
    assert a.apply(intent(4,generation=2 if action=='resume' else 1)).ok
    assert gate.session_id==owner and link.lease['session_id']==owner
    assert not link.management_request and len(writes)==before
    calls=[r for r in requests if r['action']==action]
    assert calls[-1]==calls[-2]
    assert a.apply(intent(5,generation=2 if action=='resume' else 1,forward=.04)).ok
    gate.tick();assert gate.applied_seq==1 and len(writes)>before
    assert a.close().ok and not gate.session_id


def test_grip_release_cancels_a_resume_whose_reply_was_lost(managed):
    a,link,gate,writes,requests,lose=managed
    assert a.apply(intent()).ok
    assert stop(a,'torso_collision').ok
    assert a.apply(intent(2,generation=2)).ok
    lose.append('resume');assert a.apply(intent(3,generation=2)).ok
    assert link.management_request
    assert stop(a,'deadman_released',generation=3).ok
    assert not link.management_request and not link.lease and not gate.session_id
    count=len(writes)
    assert not a.apply(intent(4,generation=2,forward=.1)).ok
    assert len(writes)==count


def test_stop_cancels_lost_claim_even_without_a_client_lease(managed):
    a,link,gate,writes,_,lose=managed
    lose.append('claim');assert a.apply(intent()).ok
    assert link.lease is None and gate.session_id and link.management_request
    assert a.close().ok
    assert not gate.session_id and not link.management_request and gate.status()['stop_confirmed']


def test_expired_lost_claim_rebases_and_recovers_with_same_held_grips(managed):
    a,link,gate,writes,_,lose=managed
    lose.append('claim');assert a.apply(intent()).ok
    old=gate.session_id
    time.sleep(.305);gate.tick();time.sleep(.001);gate.tick()
    assert gate.session_id is None and gate.status()['stop_confirmed']
    assert a.apply(intent(2)).ok and a.output['code']=='driver_lease_rebased'
    assert not link.management_request and link.lease is None
    assert a.apply(intent(3)).ok and gate.session_id!=old
    assert a.apply(intent(4,forward=.04)).ok
    gate.tick();assert gate.applied_seq==1
    assert a.close().ok


@pytest.mark.parametrize('change',[{}, {'ownership_held':True}, {'stop_confirmed':False},
                                  {'estop':True}, {'power_ns':0}, {'inflight':True}])
def test_explicit_card_recovery_requires_fresh_confirmed_release(chain,change):
    from teleop.plugin import TeleopPlugin
    from teleop.protocol import ProtocolError
    _,link,gate,_=chain
    state={**gate.status(),'state':'idle','ownership_held':False,'output_active':False,'stop_confirmed':True}
    state['feedback'].update({k:v for k,v in change.items() if k in ('estop','power_ns')})
    state.update({k:v for k,v in change.items() if k in ('ownership_held','stop_confirmed')})
    operations=[]
    def old_release():
        operations.append('invalidate_old_session')
        raise ProtocolError('dispatch_stop_unconfirmed','previous stop failed')
    card=TeleopPlugin({'robot_profile':'tianyi2','calibration_path':'synthetic'},None)
    card.runtime=SimpleNamespace(status=lambda:{'dispatch':{'io_inflight':'apply' if change.get('inflight') else None}},
                                 release_local=old_release)
    card.link=SimpleNamespace(feedback=lambda:state)
    card._release_driver=lambda:operations.append('confirm_release') or True
    card.stop=lambda:operations.append('close_old_host')
    def fresh_host():
        operations.append('new_host')
        card.adapter=SimpleNamespace(calibrate=lambda path:operations.append('calibrate_current_feedback'))
    card._ensure_host=fresh_host
    if change:
        with pytest.raises(ValueError):card._recover_tianyi_host()
        assert 'close_old_host' not in operations
    else:
        card._recover_tianyi_host()
        assert operations==['invalidate_old_session','confirm_release','close_old_host','new_host','calibrate_current_feedback']


def test_reopened_card_waits_for_first_feedback_before_calibration():
    from teleop.plugin import TeleopPlugin
    card=TeleopPlugin({'calibration_path':'synthetic'},None);reads=[];calibrations=[]
    def feedback():
        reads.append(1)
        if len(reads)<3:raise ValueError('driver_feedback_missing')
        return {'fresh':True}
    card.link=SimpleNamespace(feedback=feedback)
    card.adapter=SimpleNamespace(calibrate=lambda path:calibrations.append(path))
    card._calibrate_recovered_host()
    assert len(reads)==3 and calibrations==['synthetic']


@pytest.mark.parametrize('failure',['torso_collision','ik_timeout'])
def test_recovery_in_actual_dispatcher(chain,failure):
    from test_teleop import frame
    from teleop.protocol import bind_rtc_frame_v1
    from teleop.runtime import TeleopRuntime
    a,link,gate,writes=chain
    solve=a.solver.solve;call=link.call;collision=[False];solves=[]
    def slow_solve(*args,**kw):
        solves.append(1)
        if collision[0]:collision[0]=False;raise ValueError(failure)
        time.sleep(.045);return solve(*args,**kw)
    def slow_call(action,deadline):
        if action=='resume':time.sleep(.065)
        return call(action,deadline)
    a.solver.solve=slow_solve;link.call=slow_call
    runtime=TeleopRuntime(mode='live',adapter=a,auto_watchdog=False,pose_timeout_ms=100)
    try:
        runtime.prepare_local_session();binding,_=runtime.rtc_authority_snapshot()
        def send(seq,held=True):
            f=frame(seq,held,1 if held else 0);f['mode']='live'
            runtime.submit_frame(bind_rtc_frame_v1(f,authority=binding,expected_mode='live'),source='test')
        send(0,False);send(1);assert runtime._dispatcher.wait_dispatched(1,.5)
        collision[0]=True;send(2)
        deadline=time.monotonic()+.5
        while time.monotonic()<deadline:
            s=runtime.status()['dispatch']
            if s['state']=='safe_reclutch_required' and s['stop_acknowledged']:break
            time.sleep(.002)
        assert gate.status()['hold_confirmed']
        count=len(writes);owner=gate.session_id
        send(3);assert runtime._dispatcher.wait_dispatched(3,.5)
        if failure=='ik_timeout':
            assert a.output['state']=='submitted' and gate.session_id==owner
            assert 'resume' not in link.calls
            gate.tick();assert gate.applied_seq==1
            send(4,False)
            deadline=time.monotonic()+.5
            while time.monotonic()<deadline and gate.status()['continuation_allowed']:time.sleep(.002)
            assert not gate.status()['continuation_allowed']
            return
        assert a.output['state']=='recovery_validated' and gate.state=='hold'
        n=len(solves);send(4);assert runtime._dispatcher.wait_dispatched(4,.5)
        assert len(solves)==n and gate.state=='ready' and len(writes)==count
        send(5);assert runtime._dispatcher.wait_dispatched(5,.5);gate.tick()
        assert gate.applied_seq==1 and runtime.status()['dispatch']['fault_code'] is None
    finally:runtime.close();a.close()


@pytest.mark.parametrize('persistent',[False,True])
@pytest.mark.parametrize('refusal',['robot_not_stopped','hold_not_resumable'])
def test_resume_residual_velocity_retries_same_bounded_request(managed,persistent,refusal):
    a,link,gate,writes,requests,_=managed
    assert a.apply(intent()).ok
    assert stop(a,'feedback_unavailable').ok
    assert a.apply(intent(2,generation=2)).ok
    call=link.call;attempts=[]
    def transient(action,deadline):
        if action=='resume':
            attempts.append(dict(link.management_request))
            if persistent or len(attempts)<3:raise ValueError(refusal)
        return call(action,deadline)
    link.call=transient;count=len(writes)
    for sequence in (3,4):
        assert a.apply(intent(sequence,generation=2)).ok
        assert a.output['code']=='driver_management_pending' and len(writes)==count
    assert attempts[0]==attempts[1]
    if persistent:
        link.management_request['until_ns']=time.monotonic_ns()-1
        assert a.apply(intent(5,generation=2)).ok
        assert a.output['code']=='driver_lease_rebased'
        assert not link.lease and not link.management_request and not gate.session_id
    else:
        assert a.apply(intent(5,generation=2)).ok
        assert not link.management_request and len(writes)==count
        assert a.apply(intent(6,generation=2,forward=.04)).ok
        gate.tick();assert len(writes)>count
        assert a.close().ok


@pytest.mark.parametrize('change',[{'state':'fault'},{'reason':'unexpected'},
                                  {'ownership_held':False},{'session_id':'foreign'}])
def test_nonresumable_retry_rejects_faults_and_different_owner(managed,change):
    a,link,gate,writes,_,_=managed
    assert a.apply(intent()).ok
    assert stop(a,'feedback_unavailable').ok
    assert a.apply(intent(2,generation=2)).ok
    original_feedback=link.feedback;original_call=link.call;count=len(writes)
    link.feedback=lambda:{**original_feedback(),**change}
    def reject(action,deadline):
        if action=='resume':raise ValueError('hold_not_resumable')
        return original_call(action,deadline)
    link.call=reject
    with pytest.raises(ValueError,match='hold_not_resumable'):
        link.resume(time.monotonic()+.1)
    assert len(writes)==count


def test_expired_resume_keeps_relative_mapping_and_validates_before_reclaim(managed):
    import copy
    import numpy as np
    a,link,gate,writes,_,lose=managed
    assert a.apply(intent()).ok
    clutch=a.clutch;reference=copy.deepcopy(a.mapper.reference)
    assert stop(a,'feedback_unavailable').ok
    assert a.apply(intent(2,generation=2,forward=.04)).ok
    lose.append('resume')
    assert a.apply(intent(3,generation=2,forward=.05)).ok
    assert link.management_request
    time.sleep(.305);gate.tick();time.sleep(.001);gate.tick()
    assert a.apply(intent(4,generation=2,forward=.06)).ok
    assert a.output['code']=='driver_lease_rebased'
    assert not link.lease and not link.management_request
    assert a.clutch==clutch
    count=len(writes)
    assert a.apply(intent(5,generation=2,forward=.07)).ok
    assert a.output['state']=='recovery_validated' and not link.lease
    assert len(writes)==count
    assert a.apply(intent(6,generation=2,forward=.08)).ok
    assert a.output['state']=='armed_waiting_input' and link.lease
    assert len(writes)==count
    for before,after in zip(reference,a.mapper.reference):
        for x,y in zip(before,after):assert np.array_equal(x,y)
    assert a.apply(intent(7,generation=2,forward=.08)).ok
    assert a.output['target_q'][0]>.03  # Current displacement was not silently zeroed.
    gate.tick();assert a.close().ok


@pytest.mark.parametrize('lost_reply',[False,True])
def test_ik_recovery_uses_authenticated_mcp_wire_and_same_lease(managed,lost_reply):
    a,link,gate,writes,requests,lose=managed
    assert a.apply(intent()).ok
    assert a.apply(intent(2,forward=.03)).ok;gate.tick()
    old=gate.session_id
    if lost_reply:lose.append('recoverable_hold')
    assert stop(a,'ik_timeout').ok
    request=requests[-1]
    assert request['action']=='recoverable_hold'
    assert request['session_id']==old and request['secret']==gate.secret
    assert a.apply(intent(3,generation=2,forward=.05)).ok
    assert a.output['state']=='submitted' and gate.session_id==old
    assert not any(r['action']=='resume' for r in requests)
    gate.tick();assert gate.state=='active'
    assert a.close().ok
