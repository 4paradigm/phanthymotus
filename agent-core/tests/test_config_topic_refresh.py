"""A config-completion event must not race the driver's config application."""
import asyncio
import pathlib
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from api import canvas
import tool_config


@pytest.mark.parametrize('instance_id', ['', 'card-a'])
def test_topic_refresh_notification_follows_driver_config(monkeypatch, instance_id):
    events = []
    tasks = []
    monkeypatch.setattr(canvas, '_broadcast', events.append)
    monkeypatch.setattr(tool_config, 'find_tool', lambda *_: {
        'configSchema': {'properties': {'mode': {'type': 'string', 'scope': 'instance'}}},
    })

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def call(mcp_id, request):
            assert mcp_id == 'robot'
            assert request.tool == 'sensor'
            assert request.arguments['action'] == 'config'
            started.set()
            await finish.wait()
            return {'code': 200}

        monkeypatch.setitem(sys.modules, 'api.mcp_manage', types.SimpleNamespace(
            mcp_call_tool=call, MCPCallRequest=lambda **kwargs: types.SimpleNamespace(**kwargs)))
        create_task = asyncio.create_task

        def track(coro):
            task = create_task(coro)
            tasks.append(task)
            return task

        monkeypatch.setattr(canvas.asyncio, 'create_task', track)
        canvas.apply_tool_config('robot', 'sensor', {'mode': 'changed'}, instance_id)
        await started.wait()
        assert events == [], 'querying info before config completes can return old topics'
        finish.set()
        await asyncio.gather(*tasks)
        assert events == [{'type': 'tool_config_updated', 'payload': {
            'mcp_id': 'robot', 'tool': 'sensor', 'instance_id': instance_id}}]

    asyncio.run(run())


@pytest.mark.parametrize('accepted', [True, False])
@pytest.mark.parametrize('key,value', [('wz_limit', .52), ('angular_acceleration', 30.)])
def test_loco_config_refreshes_the_connected_navi_from_accepted_limits(monkeypatch, accepted, key, value):
    events = []
    calls = []
    tasks = []
    layout = {
        'cards': [
            {'id': 'drive', 'mcpId': 'driver', 'toolName': 'loco_servo'},
            {'id': 'other-drive', 'mcpId': 'other-driver', 'toolName': 'loco_servo'},
            {'id': 'tracking', 'mcpId': 'actucore', 'toolName': 'navi'},
            {'id': 'other-tracking', 'mcpId': 'other-actucore', 'toolName': 'navi'},
        ],
        'connections': [
            {'fromCardId': 'tracking', 'toCardId': 'drive', 'format': 'control/velocity'},
            {'fromCardId': 'other-tracking', 'toCardId': 'other-drive', 'format': 'control/velocity'},
        ],
    }
    descriptor = {'control_interface': 'motus.control/1', 'mode': 'twist',
                  'limits': {'upper': [.6, .4, 0, 0, 0, .52],
                             'max_delta_per_step': [.03, .03, 0, 0, 0, 3.]}}
    monkeypatch.setattr(canvas, '_broadcast', events.append)
    monkeypatch.setattr(canvas, 'config', types.SimpleNamespace(main={'canvas_layout': layout}))
    monkeypatch.setattr(tool_config, 'find_tool', lambda *_: {
        'configSchema': {'properties': {key: {'scope': 'instance'}}},
    })

    async def call(mcp_id, request, timeout_s=None):
        args = request.arguments
        calls.append((mcp_id, request.tool, args))
        if mcp_id == 'driver' and args['action'] == 'config':
            return {'code': 200, 'data': {'ok': accepted}}
        if mcp_id == 'driver' and args['action'] == 'info':
            assert timeout_s
            return {'code': 200, 'data': {'control_interface': descriptor}}
        assert timeout_s
        return {'code': 200, 'data': {'status': 'configured', 'control_interface_updated': True}}

    monkeypatch.setitem(sys.modules, 'api.mcp_manage', types.SimpleNamespace(
        mcp_call_tool=call, MCPCallRequest=lambda **kw: types.SimpleNamespace(**kw)))

    async def run():
        create_task = asyncio.create_task

        def track(coro):
            task = create_task(coro)
            tasks.append(task)
            return task

        monkeypatch.setattr(canvas.asyncio, 'create_task', track)
        canvas.apply_tool_config('driver', 'loco_servo', {key: value}, 'drive')
        await asyncio.gather(*tasks)

    asyncio.run(run())
    tail = [
        ('driver', 'loco_servo', {'action': 'info', 'instance_id': 'drive'}),
        ('actucore', 'navi', {'action': 'config', 'instance_id': 'tracking',
                             'control_interface': descriptor}),
    ] if accepted else []
    assert calls == [
        ('driver', 'loco_servo', {'action': 'config', 'instance_id': 'drive', key: value}),
        *tail,
    ]


@pytest.mark.parametrize('instance_id', ['', 'drive'])
@pytest.mark.parametrize('outcome,status', [
    ('driver refused', 422), ('driver offline', 503),
    ('stale schema', 503), ('accepted', 200), ('navi offline', 200),
])
def test_yaw_save_reports_live_result_and_only_persists_accepted_config(
        monkeypatch, instance_id, outcome, status):
    key = canvas.tool_config_key('driver', 'loco_servo', instance_id)
    stored = {key: {'wz_limit': 1.}}
    desired = {'wz_limit': .52, 'angular_acceleration': 30.}
    monkeypatch.setattr(canvas, 'config', types.SimpleNamespace(main=stored))
    monkeypatch.setattr(canvas, '_broadcast', lambda _: None)
    properties = {'angular_acceleration': {}}
    if outcome != 'stale schema':
        properties['wz_limit'] = {'scope': 'instance'}
    monkeypatch.setattr(tool_config, 'find_tool', lambda *_: {
        'configSchema': {'properties': properties},
    })

    async def call(mcp_id, request, timeout_s=None):
        assert mcp_id == 'driver' and timeout_s
        assert outcome != 'stale schema'
        assert request.arguments['wz_limit'] == .52 and request.arguments['angular_acceleration'] == 30.
        assert stored[key] == {'wz_limit': 1.}
        if outcome == 'driver offline':
            return {'code': 500, 'message': 'connection refused'}
        return {'code': 200, 'data': {'ok': outcome != 'driver refused',
                                       'message': 'configured minimum incompatible'}}

    async def refresh(mcp_id, card_instance):
        assert (mcp_id, card_instance) == ('driver', instance_id)
        assert stored[key] == desired
        if outcome == 'navi offline':
            raise RuntimeError('Navi connection refused')

    monkeypatch.setattr(canvas, '_refresh_navi_limits', refresh)
    monkeypatch.setitem(sys.modules, 'api.mcp_manage', types.SimpleNamespace(
        mcp_call_tool=call, MCPCallRequest=lambda **kw: types.SimpleNamespace(**kw)))
    app = FastAPI()
    app.include_router(canvas.router)
    path = '/canvas/tool-config/driver/loco_servo'
    if instance_id:
        path += '/' + instance_id
    with TestClient(app) as client:
        result = client.put(path, json=desired)
    assert result.status_code == status
    if outcome.startswith('driver '):
        assert stored[key] == {'wz_limit': 1.}
        assert ('connection refused' if outcome == 'driver offline' else
                'configured minimum incompatible') in result.json()['detail']
    elif outcome == 'stale schema':
        assert '未声明转向配置项 wz_limit' in result.json()['detail']
        assert stored[key] == {'wz_limit': 1.}
    else:
        assert stored[key] == desired
        if outcome == 'navi offline':
            assert 'Navi connection refused' in result.json()['warning']
        else:
            assert 'warning' not in result.json()
