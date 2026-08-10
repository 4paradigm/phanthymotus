from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

import auth
import config
import mcp_client
from api import mcp_manage


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __await__(self):
        async def resolve():
            return self

        return resolve().__await__()

    async def json(self, content_type=None):
        return deepcopy(self._payload)


class RecordingSession:
    def __init__(self, records: list, *args, tools=None, **kwargs):
        self.records = records
        self.tools = tools or [
            {
                'name': 'ordinary_tool',
                'description': 'ordinary',
                'inputSchema': {'type': 'object', 'properties': {}},
            },
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json, headers, **kwargs):
        self.records.append({
            'url': url,
            'method': json['method'],
            'headers': dict(headers),
            'payload': deepcopy(json),
        })
        method = json['method']
        if method == 'initialize':
            result = {'serverInfo': {'name': 'test-driver'}}
        elif method == 'tools/list':
            result = {'tools': deepcopy(self.tools)}
        elif method == 'resources/list':
            result = {'resources': []}
        elif method == 'tools/call':
            result = {'content': [{'type': 'text', 'text': 'ok'}]}
        else:  # pragma: no cover - fails loudly for an unexpected RPC
            raise AssertionError(f'unexpected RPC method {method}')
        payload = {'jsonrpc': '2.0', 'id': json['id'], 'result': result}
        return FakeResponse(payload)


class CancelledSseResponse:
    async def __aenter__(self):
        raise asyncio.CancelledError

    async def __aexit__(self, exc_type, exc, tb):
        return False


class RecordingSseSession:
    def __init__(self, records: list, *args, **kwargs):
        self.records = records

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, *, headers, **kwargs):
        self.records.append({'url': url, 'headers': dict(headers)})
        return CancelledSseResponse()


def _session_factory(records, *, tools=None):
    def factory(*args, **kwargs):
        return RecordingSession(records, *args, tools=tools, **kwargs)

    return factory


@pytest.mark.parametrize('method', ['initialize', 'tools/list', 'tools/call'])
def test_jrpc_sends_driver_bearer_on_every_trusted_rpc(method):
    secret = 'trusted-driver-secret'
    auth.init({'MOTUS_DRIVER_TOKEN': secret})
    records = []
    session = RecordingSession(records)

    asyncio.run(mcp_client._jrpc(
        session,
        'http://trusted.local/mcp',
        method,
        {},
        trusted=True,
        driver_id='legacy-driver',
    ))

    assert len(records) == 1
    assert records[0]['method'] == method
    assert records[0]['headers']['Authorization'] == f'Bearer {secret}'
    assert records[0]['headers']['Content-Type'] == 'application/json'


@pytest.mark.parametrize('method', ['initialize', 'tools/list', 'tools/call'])
def test_jrpc_never_leaks_driver_bearer_to_untrusted_rpc(method):
    secret = 'trusted-driver-secret'
    auth.init({'MOTUS_DRIVER_TOKEN': secret})
    records = []
    session = RecordingSession(records)

    asyncio.run(mcp_client._jrpc(
        session,
        'http://legacy.local/mcp',
        method,
        {},
        trusted=False,
    ))

    assert len(records) == 1
    assert 'Authorization' not in records[0]['headers']
    assert secret not in str(records[0])


@pytest.mark.parametrize('trusted', [True, False])
def test_capability_discovery_authenticates_only_trusted_driver(
    monkeypatch, trusted,
):
    secret = 'trusted-driver-secret'
    auth.init({'MOTUS_DRIVER_TOKEN': secret})
    records = []
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(records),
    )

    result = asyncio.run(mcp_manage._ping_mcp_http(
        'http://driver.local/mcp',
        trusted=trusted,
        driver_id='legacy-driver',
    ))

    assert result['server_name'] == 'test-driver'
    assert [record['method'] for record in records] == [
        'initialize', 'tools/list', 'resources/list',
    ]
    for record in records:
        if trusted:
            assert record['headers']['Authorization'] == f'Bearer {secret}'
        else:
            assert 'Authorization' not in record['headers']


def test_capability_discovery_never_calls_reserved_info_tools(
    monkeypatch, shadow_session_tool,
):
    named_info = deepcopy(shadow_session_tool)
    named_info['name'] = 'teleop_info'
    action_info = deepcopy(shadow_session_tool)
    action_info['inputSchema']['properties']['action']['enum'].append('info')
    records = []
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(records, tools=[named_info, action_info]),
    )

    result = asyncio.run(mcp_manage._ping_mcp_http(
        'http://trusted.local/mcp',
        trusted=True,
        driver_id='legacy-driver',
    ))

    assert [record['method'] for record in records] == [
        'initialize', 'tools/list', 'resources/list',
    ]
    assert all(record['method'] != 'tools/call' for record in records)
    assert [tool['name'] for tool in result['tools']] == [
        'teleop_info', 'teleop_session',
    ]


def test_capability_discovery_selects_only_the_exact_driver_credential(monkeypatch):
    token_a = 'private-driver-token-a-0001'
    token_b = 'private-driver-token-b-0002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': (
            f'{{"driver-a":"{token_a}","driver-b":"{token_b}"}}'
        ),
    })
    records = []
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(records),
    )

    asyncio.run(mcp_manage._ping_mcp_http(
        'http://driver-b.local/mcp',
        trusted=True,
        driver_id='driver-b',
    ))

    assert len(records) == 3
    assert all(
        record['headers'].get('Authorization') == f'Bearer {token_b}'
        for record in records
    )
    assert token_a not in str(records)


def test_sse_selects_only_the_exact_driver_credential(monkeypatch):
    token_a = 'private-driver-token-a-0001'
    token_b = 'private-driver-token-b-0002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': (
            f'{{"driver-a":"{token_a}","driver-b":"{token_b}"}}'
        ),
    })
    records = []
    monkeypatch.setattr(
        mcp_client.aiohttp,
        'ClientSession',
        lambda *args, **kwargs: RecordingSseSession(records),
    )

    asyncio.run(mcp_client._subscribe_sse(
        'driver-b',
        'http://driver-b.local/mcp',
        trusted=True,
    ))

    assert records == [{
        'url': 'http://driver-b.local/mcp/sse',
        'headers': {'Authorization': f'Bearer {token_b}'},
    }]
    assert token_a not in str(records)


def test_trusted_outbound_helpers_fail_before_network_without_exact_identity(
    monkeypatch,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'trusted-driver-secret'})
    jrpc_records = []
    with pytest.raises(PermissionError, match='exact credential'):
        asyncio.run(mcp_client._jrpc(
            RecordingSession(jrpc_records),
            'http://trusted.local/mcp',
            'initialize',
            {},
            trusted=True,
        ))
    assert jrpc_records == []

    ping_records = []
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(ping_records),
    )
    with pytest.raises(PermissionError, match='exact credential'):
        asyncio.run(mcp_manage._ping_mcp_http(
            'http://trusted.local/mcp',
            trusted=True,
        ))
    assert ping_records == []


def test_startup_never_sends_rotated_bearer_to_stale_persisted_endpoint(
    monkeypatch,
):
    old_token = 'private-driver-token-old-0001'
    new_token = 'private-driver-token-new-0002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': f'{{"driver-a":"{old_token}"}}',
    })
    old_binding = auth.driver_credential_binding('driver-a')
    config.main['services'] = {'mcp': [{
        'id': 'driver-a',
        'name': 'Persisted Driver A',
        'url': 'http://stale-driver-a.local/mcp',
        'transport': 'http',
        'trust_state': 'trusted',
        'credential_binding': old_binding,
        'tools': [],
    }]}
    auth.init({
        'MOTUS_DRIVER_TOKENS': f'{{"driver-a":"{new_token}"}}',
    })
    attempted = []

    async def record_connect(*args, **kwargs):
        attempted.append((args, kwargs))

    monkeypatch.setattr(mcp_client, '_connect_one', record_connect)

    asyncio.run(mcp_client.init_all())

    assert attempted == []


def test_config_restore_never_calls_reserved_tool(
    monkeypatch, shadow_session_tool,
):
    records = []
    config.main['services'] = {'mcp': [{
        'id': 'teleop-robot',
        'name': 'Teleop Robot',
        'url': 'http://trusted.local/mcp',
        'transport': 'http',
        'trust_state': 'trusted',
        'authority_domain': 'teleop-robot',
        'tools': [],
    }]}
    config.main['tool_config:teleop-robot:teleop_session'] = {'configured': True}
    reserved = deepcopy(shadow_session_tool)
    reserved['configSchema'] = {'type': 'object', 'properties': {}}
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(records),
    )

    asyncio.run(mcp_manage._restore_saved_configs(
        'teleop-robot',
        'http://trusted.local/mcp',
        [reserved],
        trusted=True,
    ))

    assert records == []


def test_config_restore_selects_only_the_exact_driver_credential(monkeypatch):
    token_a = 'private-driver-token-a-0001'
    token_b = 'private-driver-token-b-0002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': (
            f'{{"driver-a":"{token_a}","driver-b":"{token_b}"}}'
        ),
    })
    tool = {
        'name': 'ordinary_tool',
        'type': 'actuator',
        'configSchema': {'type': 'object', 'properties': {'speed': {'type': 'number'}}},
        'inputSchema': {
            'type': 'object',
            'properties': {'action': {'type': 'string', 'enum': ['config']}},
            'required': ['action'],
        },
    }
    config.main['services'] = {'mcp': [{
        'id': 'driver-b',
        'name': 'Driver B',
        'url': 'http://driver-b.local/mcp',
        'transport': 'http',
        'trust_state': 'trusted',
        'credential_binding': auth.driver_credential_binding('driver-b'),
        'authority_domain': 'driver-b',
        'tools': [tool],
    }]}
    config.main['tool_config:driver-b:ordinary_tool'] = {'speed': 0.5}
    records = []
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(records),
    )

    asyncio.run(mcp_manage._restore_saved_configs(
        'driver-b',
        'http://driver-b.local/mcp',
        [tool],
        trusted=True,
    ))

    assert [record['method'] for record in records] == ['tools/call']
    assert records[0]['headers']['Authorization'] == f'Bearer {token_b}'
    assert token_a not in str(records)


def test_tools_and_rest_call_select_only_the_exact_driver_credential(monkeypatch):
    token_a = 'private-driver-token-a-0001'
    token_b = 'private-driver-token-b-0002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': (
            f'{{"driver-a":"{token_a}","driver-b":"{token_b}"}}'
        ),
    })
    tool = {
        'name': 'ordinary_tool',
        'type': 'sensor',
        'inputSchema': {'type': 'object', 'properties': {}},
    }
    config.main['services'] = {'mcp': [{
        'id': 'driver-b',
        'name': 'Driver B',
        'url': 'http://driver-b.local/mcp',
        'transport': 'http',
        'trust_state': 'trusted',
        'credential_binding': auth.driver_credential_binding('driver-b'),
        'authority_domain': 'driver-b',
        'tools': [tool],
    }]}
    records = []
    monkeypatch.setattr(
        mcp_manage.aiohttp,
        'ClientSession',
        _session_factory(records, tools=[tool]),
    )

    tools_result = asyncio.run(mcp_manage.mcp_get_tools('driver-b'))
    call_result = asyncio.run(mcp_manage.mcp_call_tool(
        'driver-b',
        mcp_manage.MCPCallRequest(tool='ordinary_tool', arguments={}),
    ))

    assert tools_result['code'] == 200
    assert call_result['code'] == 200
    assert [record['method'] for record in records] == [
        'initialize', 'tools/list', 'initialize', 'tools/call',
    ]
    assert all(
        record['headers'].get('Authorization') == f'Bearer {token_b}'
        for record in records
    )
    assert token_a not in str(records)


def test_connect_and_llm_call_keep_trust_on_all_rpc_hops(monkeypatch):
    secret = 'trusted-driver-secret-0001'
    other_secret = 'other-driver-secret-000002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': (
            f'{{"other-robot":"{other_secret}","trusted-robot":"{secret}"}}'
        ),
    })
    config.main['services'] = {'mcp': [{
        'id': 'trusted-robot',
        'name': 'Trusted Robot',
        'url': 'http://trusted.local/mcp',
        'transport': 'http',
        'trust_state': 'trusted',
        'credential_binding': auth.driver_credential_binding('trusted-robot'),
        'tools': [{'name': 'ordinary_tool', 'type': 'actuator'}],
    }]}
    records = []
    monkeypatch.setattr(
        mcp_client.aiohttp,
        'ClientSession',
        _session_factory(records),
    )

    async def no_sse(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_client, '_subscribe_sse', no_sse)

    async def connect_and_call():
        await mcp_client._connect_one(
            'trusted-robot',
            'Trusted Robot',
            'http://trusted.local/mcp',
            '',
            trusted=True,
        )
        await asyncio.sleep(0)
        return await mcp_client.call_tool(
            'mcp__trusted-robot__ordinary_tool', {},
        )

    result = asyncio.run(connect_and_call())

    assert result == 'ok'
    assert [record['method'] for record in records] == [
        'initialize', 'tools/list', 'tools/call',
    ]
    assert all(
        record['headers'].get('Authorization') == f'Bearer {secret}'
        for record in records
    )
    assert other_secret not in str(records)


def test_legacy_runtime_call_does_not_receive_global_driver_secret(monkeypatch):
    secret = 'trusted-driver-secret'
    auth.init({'MOTUS_DRIVER_TOKEN': secret})
    records = []
    monkeypatch.setattr(
        mcp_client.aiohttp,
        'ClientSession',
        _session_factory(records),
    )
    config.main['services'] = {'mcp': [{
        'id': 'legacy-driver',
        'name': 'Legacy Driver',
        'url': 'http://legacy.local/mcp',
        'transport': 'http',
        'trust_state': 'untrusted',
        'tools': [{'name': 'ordinary_tool'}],
    }]}
    mcp_client.registry['legacy-driver'] = {
        'name': 'Legacy Driver',
        'url': 'http://legacy.local/mcp',
        'online': True,
        'trusted': False,
        'schemas': {},
        'input_schemas': {},
        'split_map': {},
        'tool_meta': {
            'mcp__legacy-driver__ordinary_tool': {
                'type': 'actuator',
                'action_enum': None,
                'annotations': {},
            },
        },
    }

    result = asyncio.run(mcp_client.call_tool(
        'mcp__legacy-driver__ordinary_tool', {},
    ))

    assert result == 'ok'
    assert [record['method'] for record in records] == ['tools/call']
    assert 'Authorization' not in records[0]['headers']
    assert secret not in str(records)
