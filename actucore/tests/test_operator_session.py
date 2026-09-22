import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.operator_session import OperatorCommands


def test_commands_deduplicate_without_blocking_presence_and_stop_cancels():
    async def run():
        connection=SimpleNamespace(connection_id='new',events=asyncio.Queue())
        manager=SimpleNamespace(_connection=connection,presence_expired=lambda c:False)
        entered=threading.Event();calls=[]
        def execute(action,cancel,valid):
            calls.append(action)
            if action=='finish':
                entered.set()
                assert cancel.wait(2)
                raise ValueError('return_cancelled')
            return {'state':'idle'}
        broker=OperatorCommands(manager,execute)
        msg={'request_id':'1','connection_id':'new','action':'finish'}
        assert (await broker.submit(connection,msg))['state']=='accepted'
        assert await asyncio.to_thread(entered.wait,1)
        assert (await broker.submit(connection,msg))['state']=='accepted'
        with pytest.raises(ValueError,match='operator_busy'):
            await broker.submit(connection,{**msg,'request_id':'2','action':'start'})
        with pytest.raises(ValueError,match='operator_request_conflict'):
            await broker.submit(connection,{**msg,'action':'stop'})
        assert (await broker.submit(connection,{**msg,'request_id':'3','action':'stop'}))['state']=='accepted'
        await broker.task
        assert calls==['finish','stop']
        assert (await broker.submit(connection,msg))['state']=='failed'
        manager._connection=SimpleNamespace(connection_id='replacement')
        with pytest.raises(ValueError,match='operator_connection_changed'):
            await broker.submit(connection,{**msg,'request_id':'4'})
    asyncio.run(run())


def test_queued_start_never_executes_after_disconnect():
    async def run():
        c=SimpleNamespace(connection_id='a',events=asyncio.Queue())
        m=SimpleNamespace(_connection=c,presence_expired=lambda c:False)
        calls=[];broker=OperatorCommands(m,lambda *args:calls.append(args))
        await broker.submit(c,{'request_id':'x','connection_id':'a','action':'start'})
        m._connection=None
        await broker.task
        assert not calls and broker.status['error']=='operator_connection_changed'
    asyncio.run(run())

@pytest.mark.parametrize('failure',[None,'collision','disconnect','cancel','feedback','hold','management_retry','mid_cancel','mid_disconnect'])
def test_return_uses_feedback_settling_and_confirms_hold_on_failure(failure):
    import numpy as np
    import time
    from teleop.operator_session import return_arms
    now=[0.];q=np.ones(14)*.08;writes=[];closed=[];paused=[]
    cancel=threading.Event()
    class Link:
        lease={'session_id':'s'};lease_started_ns=0
        def prepare_transport(self):pass
        def resume(self,deadline):
            if failure=='management_retry' and not getattr(self,'attempted',False):
                self.attempted=True
                raise ValueError('driver_management_pending')
        def feedback_after(self,*args):return self.feedback()
        def feedback(self):
            return {'state':'hold' if failure=='hold' else 'active','hold_confirmed':True,
                'feedback':{'q':q.copy(),'dq':np.zeros(14),
                            'arm_ns':0 if failure=='feedback' else time.monotonic_ns()}}
        def send(self,target,*args,**kwargs):
            writes.append(list(target));q[:]+=(np.asarray(target)-q)*.5
            if failure=='mid_cancel':cancel.set()
        def pause(self,deadline):paused.append(True);return True
    def transition(*args):
        if failure=='collision':raise ValueError('collision')
    solver=SimpleNamespace(hands_enabled=False,velocity=1.,lock=threading.RLock(),
        model=SimpleNamespace(lowerPositionLimit=np.ones(14)*-1,upperPositionLimit=np.ones(14)),
        indices=np.arange(14),_safe_transition=transition)
    adapter=SimpleNamespace(hardware_output=True,link=Link(),solver=solver,lock=threading.RLock(),
        stop_confirmation_timeout_ms=2500,
        close=lambda:(closed.append(now[0]) or SimpleNamespace(ok=True)),
        _confirm_stop=lambda operation,deadline:operation(deadline))
    if failure=='cancel':cancel.set()
    def sleep(t):now[0]+=t
    if failure and failure!='management_retry':
        with pytest.raises(ValueError):return_arms(adapter,cancel,lambda:failure!='disconnect' and not (failure=='mid_disconnect' and writes),clock=lambda:now[0],sleep=sleep)
        assert paused and not closed
        assert len(writes)==(1 if failure in ('mid_cancel','mid_disconnect') else 0)
    else:
        result=return_arms(adapter,cancel,lambda:True,clock=lambda:now[0],sleep=sleep)
        assert result['return_completed'] and result['max_error_rad']<=.02
        assert len(writes)>5 and closed[0]>=.1 and not paused
        assert all(max(abs(x) for x in target)<=.08 for target in writes)

@pytest.mark.parametrize('version',['0.3.14-ikview2','0.3.15-operator1-ikview2'])
def test_real_websocket_operator_control_and_presence(tmp_path,version):
    from aiohttp import web,ClientSession
    from teleop.capture import CaptureManager
    from teleop.capture_server import create_capture_app
    from teleop.protocol import TicketCodec,TicketVerifier
    from teleop.rtc import RtcManager
    from teleop.runtime import TeleopRuntime
    from teleop.dispatch import RecordingAdapter
    from teleop.descriptor import CAPTURE_PROTOCOL,RTC_FRAME_PROTOCOL
    async def run():
        runtime=TeleopRuntime(mode='shadow',adapter=RecordingAdapter())
        codec=TicketCodec('operator-test-signing-secret-123456789')
        rtc=RtcManager(runtime,TicketVerifier(codec))
        manager=CaptureManager(runtime,rtc,codec,state_file=tmp_path/'pair.json',
            public_wss_url='wss://127.0.0.1:15741/ws/teleop-capture',ca_certificate_base64='dGVzdA==')
        entered=threading.Event();calls=[]
        def execute(action,cancel,valid):
            calls.append(action)
            if action=='finish':entered.set();assert cancel.wait(2)
            return {'state':'idle'}
        broker=OperatorCommands(manager,execute);manager.operator_commands=broker
        runner=web.AppRunner(create_capture_app(manager));await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        try:
            pairing=await manager.create_pairing()
            async with ClientSession() as client:
                async with client.ws_connect(f'http://127.0.0.1:{port}/ws/teleop-capture') as ws:
                    await ws.send_json({'type':'pair','pairing_id':pairing['pairing_id'],'pairing_code':pairing['pairing_code'],
                        'capture_protocol':CAPTURE_PROTOCOL,'frame_protocol':RTC_FRAME_PROTOCOL,'client_kind':'native_openxr','app_version':version})
                    ack=await ws.receive_json(timeout=1)
                    assert ack['type']=='paired'
                    if version=='0.3.14-ikview2':
                        assert 'operator_control' not in ack
                        return
                    command={'type':'operator_command','connection_id':ack['operator_control']['connection_id'],
                             'request_id':'one','action':'finish'}
                    await ws.send_json(command);assert (await ws.receive_json(timeout=1))['state']=='accepted'
                    assert await asyncio.to_thread(entered.wait,1)
                    await ws.send_json({'type':'presence','state':'xr_standby','assignment_id':None})
                    assert (await ws.receive_json(timeout=.5))['type']=='presence_ack'
                    await ws.send_json(command);assert (await ws.receive_json(timeout=1))['state']=='accepted'
                    await ws.send_json({**command,'request_id':'two','action':'stop'})
                    assert (await ws.receive_json(timeout=1))['state']=='accepted'
                    await broker.task
                    assert calls==['finish','stop']
        finally:
            broker.cancel.set()
            if broker.task:await broker.task
            await runner.cleanup();await rtc.close_all();runtime.close()
    asyncio.run(run())
