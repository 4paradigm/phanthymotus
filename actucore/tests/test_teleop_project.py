"""Volatile project permission and operation serialization; no ROS or robot."""
import asyncio
import copy
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'plugins'))
from teleop.plugin import TeleopPlugin
from teleop.project import validate_binding
from teleop.runtime import TeleopRuntime
from teleop.dispatch import RecordingAdapter

BINDING={'mcp_id':'driver','tool':'teleop_executor','url':'http://127.0.0.1:15707/mcp',
         'namespace':'test','command_topic':'/test/motion/teleop/command',
         'feedback_topic':'/test/motion/teleop/feedback','robot_profile':'tianyi2','protocol_version':1}


@pytest.mark.parametrize('change',[
    {'protocol_version':True},{'protocol_version':2},{'robot_profile':'g1_23'},
    {'tool':'servo'},{'namespace':'a/b'},{'namespace':None},
    {'command_topic':'/other/command'},{'feedback_topic':'/other/feedback'},
    {'url':'http://192.0.2.1:15707/mcp'},{'mcp_id':''},
])
def test_binding_rejects_incompatible_or_remote_execution(change):
    with pytest.raises(ValueError):validate_binding({**BINDING,**change},'tianyi2')


@pytest.fixture
def card():
    cfg={'robot_profile':'tianyi2','mode':'shadow','namespace':'test',
         'driver_mcp_url':BINDING['url'],'calibration_path':'synthetic'}
    plugin=TeleopPlugin(cfg,None)
    adapter=RecordingAdapter();adapter.hardware_output=False;adapter.solver=object()
    runtime=TeleopRuntime(mode='shadow',adapter=adapter,auto_watchdog=False)
    async def revoke(reason):pass
    assigned=[]
    async def assign():assigned.append(True)
    plugin.runtime=runtime;plugin.adapter=adapter
    plugin.link=SimpleNamespace(lease=None,feedback=lambda:{'feedback':{'arm_ns':time.monotonic_ns()}})
    plugin.capture=SimpleNamespace(_connection=object(),presence_expired=lambda c:False,
        revoke_assignment=revoke,issue_assignment_if_connected=assign)
    plugin.operator_commands=SimpleNamespace(status={})
    plugin.operator_commands.set_status=lambda value:setattr(plugin.operator_commands,'status',value)
    plugin._run=asyncio.run;plugin._ensure_host=lambda:None
    plugin._calibrate_recovered_host=lambda:None
    yield plugin,assigned
    runtime.close()


def test_begin_only_arms_and_shutdown_without_operator_is_zero_output(card):
    c,assigned=card
    assert c.dispatch('teleop',{'action':'start'})['error']=='project_not_armed'
    assert c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})['armed']
    assert not c.runtime.status()['authority_valid'] and not assigned and c.link.lease is None
    result=c.dispatch('teleop',{'action':'project_stop'})
    assert result['return_completed'] and result['authority_released'] and not result['armed']
    assert not assigned
    assert c.dispatch('teleop',{'action':'start'})['error']=='project_not_armed'


def test_shadow_finish_and_restart_keep_project_permission_volatile(card):
    c,assigned=card
    c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})
    assert c.dispatch('teleop',{'action':'start'})['state']=='ready'
    assert c.runtime.status()['authority_valid'] and len(assigned)==1
    result=c._operator_execute('finish',threading.Event(),lambda:True)
    assert result['return_completed'] and result['authority_released']
    assert c._project_armed and not c.runtime.status()['authority_valid']
    assert c.dispatch('teleop',{'action':'start'})['state']=='ready'
    c.capture._connection=None
    assert c.dispatch('teleop',{'action':'project_stop'})['return_completed']
    assert not c.runtime.status()['authority_valid']
    assert not TeleopPlugin(c.cfg,None).info()['project']['armed']


def test_active_project_rejects_rebinding_and_config_mutation(card):
    c,_=card;c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})
    assert c.dispatch('teleop',{'action':'project_start','driver_binding':{**BINDING,'mcp_id':'other'}})['error']=='project_binding_changed'
    assert c.dispatch('teleop',{'action':'config','position_scale':.7})['error']=='project_stop_before_config'
    assert c.cfg.get('position_scale') is None


def test_project_stop_fences_start_during_calibration(card):
    c,assigned=card;c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})
    entered=threading.Event();release=threading.Event();out={}
    def calibrate():entered.set();assert release.wait(2)
    c._calibrate_recovered_host=calibrate
    start=threading.Thread(target=lambda:out.update(start=c.dispatch('teleop',{'action':'start'})))
    stop=threading.Thread(target=lambda:out.update(stop=c.dispatch('teleop',{'action':'project_stop'})))
    start.start();assert entered.wait(1);stop.start()
    deadline=time.monotonic()+1
    while not c._project_stopping and time.monotonic()<deadline:time.sleep(.001)
    assert c._project_stopping
    release.set();start.join(2);stop.join(2)
    assert not start.is_alive() and not stop.is_alive()
    assert out['start']['error'] and out['stop']['return_completed']
    assert not assigned and not c.runtime.status()['authority_valid']


def test_project_stop_fences_start_while_driver_prepare_rpc_returns(card,monkeypatch):
    c,assigned=card
    c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})
    c.adapter.hardware_output=True
    c.cfg['operator_session_enabled']=True
    entered=threading.Event();release=threading.Event();out={};requests=[]
    def call(action,deadline):
        requests.append(action)
        if action=='prepare_operator_session':
            entered.set();assert release.wait(2)
        return {}
    monkeypatch.setattr(c.link,'call',call,raising=False)
    start=threading.Thread(target=lambda:out.update(start=c.dispatch('teleop',{'action':'start'})))
    stop=threading.Thread(target=lambda:out.update(stop=c.dispatch('teleop',{'action':'project_stop'})))
    start.start();assert entered.wait(1);stop.start()
    try:
        deadline=time.monotonic()+1
        while not c._project_stopping and time.monotonic()<deadline:time.sleep(.001)
        assert c._project_stopping
        release.set();start.join(2);stop.join(2)
        assert not start.is_alive() and not stop.is_alive()
        assert out['start'].get('error'),out
        assert out['stop'].get('return_completed'),out
        assert not assigned and not c.runtime.status()['authority_valid']
        assert not c._operator_session_prepared and not c._return_required
        assert requests==['prepare_operator_session','end_operator_session']
    finally:
        release.set();start.join(2);stop.join(2)


def test_immediate_stop_cancels_return_before_new_cancel_event_is_registered(card,monkeypatch):
    from teleop import operator_session
    c,_=card
    c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})
    c._return_required=True
    c.adapter.hardware_output=True
    c.adapter.lock=threading.RLock()
    c.link.lease={'session_id':'synthetic'}
    writes=[];out={}
    class AcquisitionFence:
        """Pause only project_stop after acquiring its real operation lock."""
        def __init__(self):
            self.lock=threading.RLock()
            self.entered=threading.Event();self.proceed=threading.Event()
        def acquire(self,**kwargs):
            result=self.lock.acquire(**kwargs)
            self.entered.set();assert self.proceed.wait(2)
            return result
        def release(self):self.lock.release()
        def __enter__(self):self.lock.acquire();return self
        def __exit__(self,*args):self.lock.release()
    fence=AcquisitionFence()
    monkeypatch.setattr(c,'_operation_lock',fence)
    def release_driver():c.link.lease=None;return True
    monkeypatch.setattr(c,'_release_driver',release_driver)
    def return_arms(adapter,cancel,connected,**kwargs):
        if cancel.is_set():raise ValueError('return_cancelled')
        writes.append('uncancelled_return')
        return {'state':'idle','return_completed':True}
    monkeypatch.setattr(operator_session,'return_arms',return_arms)
    project=threading.Thread(target=lambda:out.update(project=c.dispatch('teleop',{'action':'project_stop'})))
    stop=threading.Thread(target=lambda:out.update(stop=c.dispatch('teleop',{'action':'stop'})))
    project.start()
    try:
        assert fence.entered.wait(1)
        previous_cancel=c._operator_cancel
        stop.start()
        assert previous_cancel.wait(1), 'immediate stop did not register cancellation'
        # The old event is already cancelled. A fresh Event created by the
        # queued return must retain this newer stop intent before it can act.
        fence.proceed.set();project.join(2);stop.join(2)
        assert not project.is_alive() and not stop.is_alive()
        assert not writes
        assert out['project'].get('error'),out
        assert out['project'].get('return_completed') is not True
        assert out['stop'].get('authority_released') is True,out
    finally:
        fence.proceed.set();project.join(2)
        if stop.ident is not None:stop.join(2)


def test_stop_failure_keeps_retry_state_and_blocks_new_start(card,monkeypatch):
    c,_=card;c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})
    actual=c._finish_session
    def fail(*args,**kwargs):raise ValueError('return_driver_fault')
    monkeypatch.setattr(c,'_finish_session',fail)
    result=c.dispatch('teleop',{'action':'project_stop'})
    assert result['error']=='return_driver_fault' and not result.get('authority_released')
    assert not c._project_armed and c.info()['project']['error']=='return_driver_fault'
    assert c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})['error']=='project_stop_required'
    monkeypatch.setattr(c,'_finish_session',actual)
    assert c.dispatch('teleop',{'action':'project_stop'})['return_completed']
    assert c._project_error is None
    assert c.dispatch('teleop',{'action':'project_start','driver_binding':BINDING})['armed']


def test_project_and_operator_failures_do_not_escape_into_public_status(card, monkeypatch):
    c,_ = card
    c.dispatch('teleop', {'action':'project_start', 'driver_binding':BINDING})
    def fail(*args, **kwargs):raise OSError('private-site/path password=not-public')
    monkeypatch.setattr(c, '_finish_session', fail)
    reply = c.dispatch('teleop', {'action':'project_stop'})
    assert reply['error'] == 'teleop_io_error' and not reply.get('authority_released')
    assert c.info()['project']['error'] == 'teleop_io_error'
    assert c.operator_commands.status['error'] == 'teleop_io_error'
    monkeypatch.setattr(c, '_operator_execute_locked', fail)
    from teleop.protocol import ProtocolError
    with pytest.raises(ProtocolError, match='^teleop_io_error$'):
        c._operator_execute('finish', threading.Event(), lambda:True)
    assert c.operator_commands.status['error'] == 'teleop_io_error'
    def bad_feedback():raise ValueError('http://user:secret@internal/path')
    monkeypatch.setattr(c.link, 'feedback', bad_feedback)
    assert c.info()['driver_feedback_error'] == 'teleop_not_ready'
    assert c.info()['driver_feedback_fresh'] is False
