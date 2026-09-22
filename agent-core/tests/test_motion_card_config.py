"""Core config acknowledgement/readback boundary; no ROS or hardware.

The local Driver substitute uses the actual MCP response shape and rejects
missing model, excessive speed and occupied runtime. Numerical validation of
these errors stays in the companion Driver test_motion_control.py.
"""
import asyncio
import copy
import importlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from test_teleop_project_lifecycle import config
from api import canvas

KEY = 'tool_config:driver:motion_control'
OLD = {'calibration_path':'/calibration/old.json', 'joint_velocity_rad_s': .5}
NEW = {'calibration_path':'/calibration/new.json', 'joint_velocity_rad_s': 1.}
TOOL = {'name':'motion_control', 'configSchema':{'properties':{
    'calibration_path':{'type':'string', 'scope':'shared'},
    'joint_velocity_rad_s':{'type':'number', 'scope':'shared'}}}}


class Driver:
    def __init__(self):
        self.current = copy.deepcopy(OLD)
        self.calls = []
        self.busy = False
        self.info_failure = None
        self.receipt_mismatch = False
        self.disconnected = False

    def dispatch(self, arguments):
        self.calls.append(copy.deepcopy(arguments))
        action = arguments['action']
        if action == 'info':
            if self.info_failure == 'missing':return {'state':'idle'}
            if self.info_failure == 'mismatch':return {'state':'idle', 'config':OLD}
            if self.info_failure == 'timeout':raise TimeoutError('offline')
            return {'state':'idle', 'config':copy.deepcopy(self.current)}
        assert action == 'config' and set(arguments) <= {'action', *NEW}
        if self.disconnected:raise OSError('offline')
        if self.busy:return {'state':'error','code':'configuration_requires_idle','error':'configuration_requires_idle'}
        candidate = {**self.current, **{k:v for k,v in arguments.items() if k != 'action'}}
        if candidate['calibration_path'] == '/missing.json':
            return {'state':'error','error':'calibration_missing'}
        speed = candidate['joint_velocity_rad_s']
        if type(speed) not in (int,float) or not 0 < speed <= 1.:
            return {'state':'error','error':'joint_velocity_limit'}
        self.current = candidate
        return {'state':'configured','calibrated':False,
                'config': copy.deepcopy(OLD if self.receipt_mismatch else candidate)}


@pytest.fixture
def subject(monkeypatch):
    module = importlib.import_module('api.mcp_manage')
    driver = Driver()
    storage = {KEY:copy.deepcopy(OLD), 'services':{'mcp':[{'id':'driver','tools':[copy.deepcopy(TOOL)]}]}}
    monkeypatch.setattr(config, 'main', storage)
    monkeypatch.setattr(canvas, '_motion_config_locks', {})
    async def call(mid, request, timeout_s=None):
        assert mid == 'driver' and request.tool == 'motion_control' and timeout_s == 10.
        result = driver.dispatch(request.arguments)
        # mcp_call_tool preserves ordinary Driver error content under code=200.
        return {'code':200, 'data':[{'type':'text','text':json.dumps(result)}]}
    monkeypatch.setattr(module, 'mcp_call_tool', call)
    app = FastAPI()
    app.include_router(canvas.router)
    return SimpleNamespace(driver=driver, storage=storage, app=app, mcp=module)


async def save(subject, values):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=subject.app),base_url='http://core.test') as client:
        return await client.put('/canvas/tool-config/driver/motion_control',json=values)


@pytest.mark.parametrize('case', ['model_missing','speed','boolean_speed','busy','transport',
                                 'readback_missing','readback_mismatch','readback_timeout','receipt_mismatch'])
def test_rejected_or_unconfirmed_config_never_persists(subject, case):
    values = copy.deepcopy(NEW)
    if case == 'model_missing':values['calibration_path']='/missing.json'
    if case == 'speed':values['joint_velocity_rad_s']=2.
    if case == 'boolean_speed':values['joint_velocity_rad_s']=True
    if case == 'busy':subject.driver.busy=True
    if case == 'transport':subject.driver.disconnected=True
    if case.startswith('readback_'):subject.driver.info_failure=case.removeprefix('readback_')
    if case == 'receipt_mismatch':subject.driver.receipt_mismatch=True
    response = asyncio.run(save(subject, values))
    assert response.status_code in (409,503), response.text
    assert '配置' in response.json()['detail']
    assert subject.storage[KEY] == OLD
    assert not response.json().get('applied')
    assert all(r['action'] in ('config','info') for r in subject.driver.calls)


@pytest.mark.parametrize('body', [[], {'action':'start'}, {'instance_id':'other'}])
def test_management_arguments_cannot_be_saved(subject, body):
    response = asyncio.run(save(subject, body))
    assert response.status_code == 400
    assert subject.driver.calls == [] and subject.storage[KEY] == OLD


@pytest.mark.parametrize('body', [{'unknown': True}, {**NEW, 'unknown': True}])
def test_unknown_keys_rejected_before_any_partial_application(subject, body):
    response = asyncio.run(save(subject, body))
    assert response.status_code == 400 and '未下发' in response.json()['detail']
    assert subject.driver.calls == [] and subject.driver.current == OLD
    assert subject.storage[KEY] == OLD


def test_instance_route_cannot_bypass_shared_confirmation(subject):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=subject.app),base_url='http://core.test') as client:
            response=await client.put('/canvas/tool-config/driver/motion_control/instance',json=NEW)
            assert response.status_code == 400
    asyncio.run(run())
    assert 'tool_config:driver:motion_control:instance' not in subject.storage
    assert subject.driver.calls == [] and subject.storage[KEY] == OLD


def test_unregistered_configuration_capability_is_not_called(subject):
    subject.storage['services']['mcp'][0]['tools'] = []
    response = asyncio.run(save(subject, NEW))
    assert response.status_code == 409
    assert not subject.driver.calls and subject.storage[KEY] == OLD


def test_success_is_saved_only_after_matching_info(subject):
    response = asyncio.run(save(subject, {'joint_velocity_rad_s':1.}))
    assert response.status_code == 200
    assert response.json()['applied'] is True
    assert subject.driver.calls == [{'action':'config','joint_velocity_rad_s':1.}, {'action':'info'}]
    assert subject.storage[KEY] == {**OLD,'joint_velocity_rad_s':1.}
    assert response.json()['data'] == subject.driver.current == subject.storage[KEY]


def test_configuration_requests_are_serialized_through_readback(subject, monkeypatch):
    first_entered, release_first = asyncio.Event(), asyncio.Event()
    order = []
    original = subject.mcp.mcp_call_tool
    async def delayed(mid, req, timeout_s=None):
        order.append(req.arguments.copy())
        if len(order)==1:
            first_entered.set()
            await release_first.wait()
        return await original(mid, req, timeout_s)
    monkeypatch.setattr(subject.mcp,'mcp_call_tool',delayed)
    async def run():
        a=asyncio.create_task(save(subject, {'joint_velocity_rad_s':.6}))
        await first_entered.wait()
        b=asyncio.create_task(save(subject, {'joint_velocity_rad_s':.7}))
        await asyncio.sleep(0)
        assert len(order)==1
        release_first.set()
        assert all(r.status_code==200 for r in await asyncio.gather(a,b))
    asyncio.run(run())
    assert [r['action'] for r in order] == ['config','info','config','info']
    assert subject.storage[KEY]['joint_velocity_rad_s'] == .7


def test_reconnect_replays_only_last_confirmed_configuration(subject, monkeypatch):
    async def run():
        assert (await save(subject, NEW)).status_code==200
        assert (await save(subject, {'joint_velocity_rad_s':2.})).status_code==409
        subject.driver.current=copy.deepcopy(OLD)  # A fresh Driver process.
        posts=[]
        class Session:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,url,**kwargs):
                assert url=='http://127.0.0.1:15707/mcp'
                message=kwargs['json'];posts.append(message)
                subject.driver.dispatch(message['params']['arguments'])
        monkeypatch.setattr(subject.mcp.aiohttp,'ClientSession',lambda **kwargs:Session())
        await subject.mcp._restore_saved_configs('driver','http://127.0.0.1:15707/mcp',[TOOL])
        assert len(posts)==1
        assert posts[0]['params']=={'name':'motion_control','arguments':{'action':'config',**NEW}}
        assert subject.driver.current==NEW
    asyncio.run(run())


def test_other_cards_keep_existing_config_behavior(subject, monkeypatch):
    called=[]
    monkeypatch.setattr(canvas,'apply_tool_config',lambda *args:called.append(args))
    result=asyncio.run(canvas.save_tool_config('ordinary','camera',{'fps':30}))
    assert result=={'code':200}
    assert subject.storage['tool_config:ordinary:camera']=={'fps':30}
    assert called==[('ordinary','camera',{'fps':30})]
