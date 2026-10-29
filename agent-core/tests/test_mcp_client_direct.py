"""Regression coverage for direct HTTP responses and hook dispatch contracts."""
import asyncio
import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import mcp_client


class _Response:
    def __init__(self, data):
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self.data


class _Session:
    def __init__(self, data, requests):
        self.data = data
        self.requests = requests

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, *, json, timeout):
        self.requests.append((url, json, timeout.total))
        return _Response(self.data)


def _call_with_response(monkeypatch, response, *, timeout=5):
    requests = []
    monkeypatch.setitem(mcp_client.registry, 'test-http', {
        'online': True, 'url': 'http://test/mcp', 'transport': 'http',
        'tool_meta': {'mcp__test-http__go': {
            'resource': frozenset({'base'}),
            'completion': {'actions': ['go'], 'timeout': 30},
        }},
    })
    monkeypatch.setattr(mcp_client.aiohttp, 'ClientSession',
                        lambda: _Session(response, requests))
    result = asyncio.run(mcp_client.call_tool_hook(
        'test-http', 'go', {'action': 'go'}, barrier_aware=True, timeout_s=timeout))
    return result, requests


@pytest.mark.parametrize('content', [
    [{'type': 'text', 'text': 'rejected'}],
    [{'type': 'text', 'text': '{"action_id":"must-not-register"}'}],
    [],
    [{'type': 'image', 'data': '...'}],
])
def test_mcp_iserror_is_never_unwrapped_into_success(monkeypatch, content):
    result, requests = _call_with_response(monkeypatch, {
        'result': {'isError': True, 'content': content},
    })
    assert result.get('error')
    assert 'must-not-register' not in mcp_client._pending_actions
    assert requests[0][2] == 5


def test_failed_payload_with_action_id_does_not_register_pending(monkeypatch):
    for payload in (
        '{"error":"rejected","action_id":"failed-id"}',
        '{"ok":false,"action_id":"failed-id"}',
        '{"state":"error","action_id":"failed-id"}',
        '{"status":"error","action_id":"failed-id"}',
    ):
        _call_with_response(monkeypatch, {
            'result': {'content': [{'type': 'text', 'text': payload}]},
        })
        assert 'failed-id' not in mcp_client._pending_actions


def test_direct_success_and_custom_hook_timeout(monkeypatch):
    result, requests = _call_with_response(monkeypatch, {
        'result': {'content': [{'type': 'text', 'text': '{"status":"ok"}'}]},
    }, timeout=30)
    assert result == {'status': 'ok'}
    assert requests[0][2] == 30


def test_actual_http_timeout_returns_error_with_custom_budget(monkeypatch):
    requests = []

    class TimeoutResponse(_Response):
        async def __aenter__(self):
            raise asyncio.TimeoutError()

    class TimeoutSession(_Session):
        def post(self, url, *, json, timeout):
            self.requests.append(timeout.total)
            return TimeoutResponse(None)

    monkeypatch.setitem(mcp_client.registry, 'test-timeout', {
        'online': True, 'url': 'http://test/mcp',
    })
    monkeypatch.setattr(mcp_client.aiohttp, 'ClientSession',
                        lambda: TimeoutSession(None, requests))
    result = asyncio.run(mcp_client.call_tool_hook(
        'test-timeout', 'go', {}, timeout_s=0.01))
    assert requests == [0.01]
    assert result['error'] == 'call_tool_direct timed out after 0.01s'


def test_jsonrpc_error_is_propagated(monkeypatch):
    error = {'code': -32601, 'message': 'unknown tool'}
    result, _ = _call_with_response(monkeypatch, {'error': error})
    assert result == {'error': error}


@pytest.mark.parametrize('transport', ['peer', 'internal'])
def test_direct_hook_fails_closed_for_non_http_without_url(monkeypatch, transport):
    monkeypatch.setitem(mcp_client.registry, 'non-http', {
        'online': True, 'transport': transport,
    })
    result = asyncio.run(mcp_client.call_tool_hook('non-http', 'go', {}))
    assert transport in result['error']
    assert 'not directly callable' in result['error']


def test_custom_timeout_is_passed_to_direct_and_failure_does_not_register(monkeypatch):
    monkeypatch.setitem(mcp_client.registry, 'test', {
        'tool_meta': {'mcp__test__go': {
            'resource': frozenset({'base'}), 'completion': {'timeout': 30},
        }},
    })
    calls = []

    async def direct(mcp_id, tool, args, *, timeout_s):
        calls.append(timeout_s)
        return {'error': 'timed out', 'action_id': 'failed-id'}

    monkeypatch.setattr(mcp_client, 'call_tool_direct', direct)
    result = asyncio.run(mcp_client.call_tool_hook(
        'test', 'go', {}, barrier_aware=True, timeout_s=30))
    assert calls == [30]
    assert result['error'] == 'timed out'
    assert 'failed-id' not in mcp_client._pending_actions
