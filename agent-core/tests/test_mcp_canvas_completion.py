"""Canvas calls must register asynchronous work before replaying early ACP."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'canvas-acp.db'))

import mcp_client
from api import mcp_manage


TOOL = {'name': 'arm_control', 'type': 'actuator', 'inputSchema': {
    'type': 'object', 'properties': {'action': {'type': 'string'}},
    'x-resource': 'arm', 'x-completion': {'actions': ['move'], 'timeout': 60}}}


class Response:
    def __init__(self, value): self.value = value
    async def json(self, **kwargs): return self.value


class Request:
    def __init__(self, response): self.response = response
    def __await__(self):
        async def receive(): return self.response
        return receive().__await__()
    async def __aenter__(self): return self.response
    async def __aexit__(self, *args): pass


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for name in getattr(mcp_client, '_PENDING_TABLE_NAMES', ()):
        monkeypatch.setattr(mcp_client, name, {})
    for name in ('_register_listeners', '_settle_listeners'):
        monkeypatch.setattr(mcp_client, name, [])
    for name in ('_early_completions', '_completion_receipts', '_completion_notification_tasks', '_action_outcomes'):
        if hasattr(mcp_client, name): monkeypatch.setattr(mcp_client, name, type(getattr(mcp_client, name))())


def result(action_id='canvas-1'):
    return {'content': [{'type': 'text', 'text': json.dumps({'state': 'running', 'action_id': action_id})}]}


def test_declared_canvas_action_registers_real_pending_metadata():
    async def scenario():
        registered = []
        mcp_client.on_action_registered(lambda *args: registered.append(args))
        assert mcp_manage._register_canvas_completion(TOOL, {'action': 'move'}, result())
        assert 'canvas-1' in mcp_client._pending_actions
        assert not mcp_client._pending_actions['canvas-1'].is_set()
        assert mcp_client._pending_resources['canvas-1'] == frozenset({'arm'})
        assert mcp_client._pending_timeouts['canvas-1'] == 60
        assert registered[0][:2] == ('canvas-1', 'arm_control')
    asyncio.run(scenario())


@pytest.mark.parametrize('change,args,response', [
    ({}, {'action': 'info'}, result()),
    ({}, {'action': 'move'}, {'content': [{'type': 'text', 'text': '{"state":"completed"}'}]}),
    ({}, {'action': 'move'}, {'isError': True, **result()}),
    ({}, {'action': 'move'}, result(17)),
    ({'inputSchema': {}}, {'action': 'move'}, result()),
    ({}, {'action': 'move'}, {'content': [{'type': 'text', 'text': 'not json'}]}),
])
def test_passive_terminal_invalid_or_undeclared_calls_create_no_waiter(change, args, response):
    async def scenario():
        assert not mcp_manage._register_canvas_completion({**TOOL, **change}, args, response)
        assert not mcp_client._pending_actions
    asyncio.run(scenario())


def test_real_canvas_proxy_registers_the_returned_action_without_changing_arguments(monkeypatch):
    calls = []
    registry = [{'id': 'piper-arm1', 'url': 'http://fake-driver/mcp', 'transport': 'http', 'tools': [TOOL]}]
    monkeypatch.setattr(mcp_manage, '_get_mcp_list', lambda: registry)
    response = result()
    class Session:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def post(self, url, *, json, headers):
            calls.append(deepcopy(json))
            return Request(Response({'result': response} if json['method'] == 'tools/call' else {'result': {}}))
    monkeypatch.setattr(mcp_manage.aiohttp, 'ClientSession', Session)
    async def scenario():
        value = await mcp_manage.mcp_call_tool('piper-arm1', mcp_manage.MCPCallRequest(
            tool='arm_control', arguments={'action': 'move', 'translation_mm': [0, 0, 1]}))
        assert value == {'code': 200, 'data': response['content']}
        assert mcp_client._pending_actions['canvas-1']
        assert calls[-1]['params']['arguments'] == {'action': 'move', 'translation_mm': [0, 0, 1]}
    asyncio.run(scenario())


def test_canvas_registration_replays_an_early_completion_in_listener_order(monkeypatch):
    async def scenario():
        order, emitted = [], []
        async def enqueue(**kwargs): emitted.append(kwargs)
        monkeypatch.setattr(mcp_client.event_bus, 'enqueue', enqueue)
        mcp_client.on_action_registered(lambda *args: order.append('registered'))
        mcp_client.on_action_settled(lambda *args: order.append('settled'))
        # Use the shared entry point selected by the ACP implementation.
        receipt = await mcp_client.receive_action_completion('canvas-1',
            {'action_id': 'canvas-1', 'status': 'completed', 'result': {'executed': True}})
        assert receipt['state'] == 'cached'
        assert order == [] and emitted == []
        mcp_manage._register_canvas_completion(TOOL, {'action': 'move'}, result())
        assert mcp_client._pending_actions['canvas-1'].is_set()
        assert order == ['registered', 'settled']
        await asyncio.sleep(0)
        assert len(emitted) == 1
        completed = await mcp_client.sync(['canvas-1'], timeout=.1)
        assert completed['results']['canvas-1']['status'] == 'completed'
    asyncio.run(scenario())
