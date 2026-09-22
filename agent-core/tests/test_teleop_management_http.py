"""Core HTTP paths against the actual ActuCore management guard, without ROS.

Only ActuCore's bundle dispatch is a recorder; make_handler is compiled verbatim
from main.py to avoid importing ROS/numerical startup during this HTTP test.
The credential is temporary test data, never a site credential.
"""
import ast
import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import aiohttp
import pytest

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))
import mcp_client
from teleop_management import TeleopManagementError, management_headers


@pytest.fixture
def guard(tmp_path, monkeypatch):
    source = Path(__file__).parents[2] / 'actucore/main.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'make_handler')
    calls, requests = [], []

    def dispatch(name, arguments):
        calls.append((name, dict(arguments)))
        return {'state': 'accepted', 'action': arguments.get('action', 'info')}

    namespace = {
        'BaseHTTPRequestHandler': BaseHTTPRequestHandler, 'json': json,
        'log': logging.getLogger('teleop-http-test'), '_brief': str,
        '_bundle': SimpleNamespace(dispatch=dispatch, server_name='test'),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    handler = namespace['make_handler']()

    class RecordedHandler(handler):
        def do_POST(self):
            requests.append(dict(self.headers))
            super().do_POST()

    server = ThreadingHTTPServer(('127.0.0.1', 0), RecordedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/mcp'
    key_file = tmp_path / 'management-key'
    key_file.write_text('temporary-http-test-capability-' + 'x' * 40)
    monkeypatch.setenv('TELEOP_MANAGEMENT_KEY_FILE', str(key_file))
    monkeypatch.setenv('TELEOP_MANAGEMENT_URL', url)
    full_name = 'mcp__local__teleop'
    entry = {
        'url': url, 'online': True, 'tools': ['teleop', 'other'],
        'schemas': {}, 'input_schemas': {}, 'split_map': {}, 'tool_groups': {},
        'tool_meta': {full_name: {'type': 'processor', 'has_config_schema': True}},
    }
    monkeypatch.setattr(mcp_client, 'registry', {'local': entry})
    api = importlib.import_module('api.mcp_manage')
    target = {'id': 'local', 'url': url, 'tools': [{'name': 'teleop'}]}
    monkeypatch.setattr(api, '_get_mcp_list', lambda: [target])
    monkeypatch.setattr(api.config, 'main', {
        'tool_config:local:teleop': {'mode': 'shadow'},
    })
    yield SimpleNamespace(url=url, key=key_file.read_text(), key_file=key_file,
                          calls=calls, requests=requests, api=api, full_name=full_name)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


async def invoke(subject, path, arguments, tool='teleop'):
    if path == 'llm':
        return json.loads(await mcp_client.call_tool(f'mcp__local__{tool}', dict(arguments)))
    if path == 'direct':
        return await mcp_client.call_tool_direct('local', tool, arguments)
    result = await subject.api.mcp_call_tool(
        'local', subject.api.MCPCallRequest(tool=tool, arguments=arguments))
    assert result['code'] == 200, result
    return json.loads(result['data'][0]['text'])


@pytest.mark.parametrize('path', ['api', 'llm', 'direct'])
def test_management_actions_reach_actual_guard(guard, path, monkeypatch):
    def unexpected_replay(*args):
        raise AssertionError('teleop start must not reapply saved config')
    monkeypatch.setattr(mcp_client, '_get_tool_config', unexpected_replay)

    async def run():
        actions = ['config', 'project_start', 'project_stop', 'start', 'pause',
                   'resume', 'finish', 'stop', 'open_pairing', 'approve_pairing',
                   'installation_info', 'create_invitation', 'revoke_invitation']
        for action in actions:
            before = len(guard.calls)
            result = await invoke(guard, path, {'action': action})
            assert result == {'state': 'accepted', 'action': action}
            assert guard.calls[before:] == [('teleop', {'action': action})]
            assert guard.requests[-1].get('X-Teleop-Management') == guard.key
            assert 'Origin' not in guard.requests[-1]
        if path == 'api':
            assert len(guard.requests) == 2 * len(actions)
            assert all('X-Teleop-Management' not in headers
                       for headers in guard.requests[::2])  # initialize is public
        else:
            assert len(guard.requests) == len(actions)
    asyncio.run(run())


def test_split_llm_action_and_system_hook_use_same_guard(guard):
    split_name = 'mcp__local__teleop__pause'
    mcp_client.registry['local']['split_map'][split_name] = {'tool': 'teleop', 'action': 'pause'}

    async def run():
        result = json.loads(await mcp_client.call_tool(split_name, {}))
        assert result == {'state': 'accepted', 'action': 'pause'}
        for barrier_aware in (False, True):
            result = await mcp_client.call_tool_hook(
                'local', 'teleop', {'action': 'stop'}, barrier_aware=barrier_aware)
            assert result == {'state': 'accepted', 'action': 'stop'}
        assert guard.calls == [('teleop', {'action': 'pause'}),
                               ('teleop', {'action': 'stop'}), ('teleop', {'action': 'stop'})]
        assert all(headers.get('X-Teleop-Management') == guard.key for headers in guard.requests)
    asyncio.run(run())


@pytest.mark.parametrize('path', ['api', 'llm', 'direct'])
def test_info_and_ordinary_tools_do_not_receive_capability(guard, path):
    async def run():
        for tool in ('teleop', 'other'):
            result = await invoke(guard, path, {'action': 'info'}, tool)
            assert result['state'] == 'accepted'
            assert 'X-Teleop-Management' not in guard.requests[-1]
        # Ordinary writes must not inherit a teleop capability either.
        result = await invoke(guard, path, {'action': 'stop'}, 'other')
        assert result['state'] == 'accepted'
        assert 'X-Teleop-Management' not in guard.requests[-1]
    asyncio.run(run())


def test_reconnection_configuration_uses_same_guard(guard):
    tools = [{'name': 'teleop', 'configSchema': {'properties': {
        'mode': {'type': 'string', 'scope': 'shared'}}}}]
    asyncio.run(guard.api._restore_saved_configs('local', guard.url, tools))
    assert guard.calls == [('teleop', {'action': 'config', 'mode': 'shadow'})]
    assert guard.requests[-1].get('X-Teleop-Management') == guard.key


def test_api_keeps_http_error_contract_for_missing_capability(guard, monkeypatch):
    from fastapi import HTTPException
    monkeypatch.delenv('TELEOP_MANAGEMENT_KEY_FILE')
    with pytest.raises(HTTPException) as caught:
        guard.api._teleop_management_headers(guard.url, 'teleop', {'action': 'config'})
    assert caught.value.status_code == 503
    assert caught.value.detail == 'Teleop management key is unavailable'
    assert not guard.requests


def test_actual_guard_rejects_missing_wrong_key_and_browser_origin(guard):
    async def run():
        payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                   'params': {'name': 'teleop', 'arguments': {'action': 'start'}}}
        async with aiohttp.ClientSession() as session:
            for headers in ({}, {'X-Teleop-Management': 'wrong'},
                            {'X-Teleop-Management': guard.key, 'Origin': 'https://browser.invalid'}):
                async with session.post(guard.url, json=payload, headers=headers) as response:
                    result = (await response.json())['result']
                assert result['isError']
                assert 'teleop_management_unauthorized' in result['content'][0]['text']
        assert not guard.calls
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['url_mismatch', 'missing_key', 'bad_key'])
def test_new_call_paths_fail_closed_without_exposing_key(guard, monkeypatch, failure, capsys):
    if failure == 'url_mismatch':
        monkeypatch.setenv('TELEOP_MANAGEMENT_URL', guard.url + '?different')
    elif failure == 'missing_key':
        monkeypatch.delenv('TELEOP_MANAGEMENT_KEY_FILE')
    else:
        guard.key_file.write_text(guard.key + '\r\nInjected: secret')

    async def run():
        llm = await mcp_client.call_tool(guard.full_name, {'action': 'start'})
        direct = await mcp_client.call_tool_direct('local', 'teleop', {'action': 'stop'})
        assert 'Teleop management' in llm and '调用失败' in llm
        assert 'Teleop management' in direct['error']
        assert guard.key not in llm + str(direct) + capsys.readouterr().out
        assert not guard.calls and not guard.requests
    asyncio.run(run())


@pytest.mark.parametrize('url', [
    'http://remote.invalid:15730/mcp', 'https://127.0.0.1/mcp',
    'http://user:password@127.0.0.1/mcp', 'http://127.0.0.1/other',
    'http://127.0.0.1/mcp?target=x', 'http://127.0.0.1/mcp#target',
    'http://[invalid/mcp',
])
def test_configured_url_still_must_be_exact_local_management_endpoint(monkeypatch, url):
    monkeypatch.setenv('TELEOP_MANAGEMENT_URL', url)
    with pytest.raises(TeleopManagementError, match='endpoint is not configured') as caught:
        management_headers(url, 'teleop', {'action': 'start'})
    assert caught.value.status_code == 403


def test_redirect_is_not_followed_by_new_call_paths(guard, monkeypatch):
    from aiohttp import web

    async def run():
        seen = []
        async def redirect(request):
            seen.append(request.path)
            assert request.headers.get('X-Teleop-Management') == guard.key
            return web.Response(status=307, headers={'Location': guard.url})
        app = web.Application()
        app.router.add_post('/mcp', redirect)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/mcp'
        monkeypatch.setenv('TELEOP_MANAGEMENT_URL', url)
        mcp_client.registry['local']['url'] = url
        try:
            llm = await mcp_client.call_tool(guard.full_name, {'action': 'start'})
            direct = await mcp_client.call_tool_direct('local', 'teleop', {'action': 'stop'})
            assert 'HTTP request failed (307)' in llm
            assert 'HTTP request failed (307)' in direct['error']
            assert seen == ['/mcp', '/mcp']
            assert not guard.calls and not guard.requests
        finally:
            await runner.cleanup()
    asyncio.run(run())
