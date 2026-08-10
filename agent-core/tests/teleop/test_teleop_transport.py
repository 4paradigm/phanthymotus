from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import aiohttp
import auth
import config
import mcp_client
import pytest
from aiohttp import web

DRIVER_SECRET = 'driver-token-must-not-leak'
OTHER_DRIVER_SECRET = 'other-driver-token-must-not-leak'
FENCE_SECRET = 'fence-value-must-not-leak-123456'


def _install_target(
    tool: dict,
    url: str,
    *,
    trust_state: str = 'trusted',
    online: bool = True,
    runtime_trusted: bool = True,
) -> None:
    auth.init({
        'MOTUS_DRIVER_TOKENS': (
            f'{{"other-robot":"{OTHER_DRIVER_SECRET}",'
            f'"teleop-robot":"{DRIVER_SECRET}"}}'
        ),
    })
    configured_tool = deepcopy(tool)
    configured_tool['x-teleop']['driver_id'] = 'teleop-robot'
    config.main['services'] = {'mcp': [{
        'id': 'teleop-robot',
        'name': 'Teleop Robot',
        'url': url,
        'transport': 'http',
        'trust_state': trust_state,
        'credential_binding': auth.driver_credential_binding('teleop-robot'),
        'tools': [configured_tool],
    }]}
    mcp_client.registry['teleop-robot'] = {
        'online': online,
        'trusted': runtime_trusted,
        'url': url,
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(configured_tool),
    }


async def _call_through_server(
    shadow_session_tool: dict,
    handler,
    *,
    action: str = 'status',
    arguments: dict | None = None,
) -> dict:
    app = web.Application()
    app.router.add_post('/mcp', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    url = f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'
    _install_target(shadow_session_tool, url)
    try:
        return await mcp_client.call_trusted_shadow_session(
            'teleop-robot',
            action,
            arguments,
            timeout_seconds=1.0,
        )
    finally:
        await runner.cleanup()


def _rpc_success(request_id: str, value: dict) -> web.Response:
    return web.json_response({
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {
            'content': [{
                'type': 'text',
                'text': json.dumps(value, allow_nan=False),
            }],
        },
    })


def test_success_sends_bearer_and_returns_strict_json_object(shadow_session_tool):
    observed = {}
    arguments = {
        'session_id': '9a9d7841-b587-434c-bda2-bf0baeb6381d',
        'epoch': 3,
        'fence': FENCE_SECRET,
    }

    async def handler(request: web.Request):
        observed['authorization'] = request.headers.get('Authorization')
        observed['content_type'] = request.headers.get('Content-Type')
        observed['payload'] = await request.json()
        return _rpc_success(
            observed['payload']['id'],
            {'state': 'shadow', 'boot_id': 'driver-boot'},
        )

    result = asyncio.run(_call_through_server(
        shadow_session_tool,
        handler,
        action='prepare_shadow',
        arguments=arguments,
    ))

    assert result == {'state': 'shadow', 'boot_id': 'driver-boot'}
    assert observed['authorization'] == f'Bearer {DRIVER_SECRET}'
    assert OTHER_DRIVER_SECRET not in json.dumps(observed)
    assert observed['content_type'] == 'application/json'
    assert observed['payload']['method'] == 'tools/call'
    assert observed['payload']['params'] == {
        'name': 'teleop_session',
        'arguments': {'action': 'prepare_shadow', **arguments},
    }
    assert 'action' not in arguments


@pytest.mark.parametrize('status', [401, 500])
def test_http_failures_are_sanitized(shadow_session_tool, status):
    async def handler(request: web.Request):
        request_payload = await request.json()
        leaked_fence = request_payload['params']['arguments']['fence']
        return web.json_response(
            {'error': f'{DRIVER_SECRET}:{leaked_fence}'},
            status=status,
        )

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(_call_through_server(
            shadow_session_tool,
            handler,
            action='prepare_shadow',
            arguments={
                'session_id': '9a9d7841-b587-434c-bda2-bf0baeb6381d',
                'epoch': 1,
                'fence': FENCE_SECRET,
            },
        ))

    assert raised.value.code == 'http_error'
    assert raised.value.http_status == status
    assert DRIVER_SECRET not in str(raised.value)
    assert FENCE_SECRET not in str(raised.value)
    assert DRIVER_SECRET not in repr(raised.value)
    assert FENCE_SECRET not in repr(raised.value)


@pytest.mark.parametrize('driver_code', ['session_mismatch', 'session_expired'])
def test_json_rpc_error_on_http_200_preserves_only_safe_codes(
    shadow_session_tool,
    driver_code,
):
    async def handler(request: web.Request):
        payload = await request.json()
        return web.json_response({
            'jsonrpc': '2.0',
            'id': payload['id'],
            'error': {
                'code': -32602,
                'message': f'echo {DRIVER_SECRET} {FENCE_SECRET}',
                'data': {
                    'code': driver_code,
                    'debug': FENCE_SECRET,
                },
            },
        })

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(_call_through_server(shadow_session_tool, handler))

    assert raised.value.code == 'rpc_error'
    assert raised.value.rpc_code == -32602
    assert raised.value.rpc_data_code == driver_code
    assert DRIVER_SECRET not in str(raised.value)
    assert FENCE_SECRET not in str(raised.value)


def test_unknown_rpc_subcode_cannot_echo_a_secret(shadow_session_tool):
    async def handler(request: web.Request):
        payload = await request.json()
        return web.json_response({
            'jsonrpc': '2.0',
            'id': payload['id'],
            'error': {
                'code': -32603,
                'message': 'internal error',
                'data': {'code': DRIVER_SECRET},
            },
        })

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(_call_through_server(shadow_session_tool, handler))

    assert raised.value.rpc_data_code is None
    assert DRIVER_SECRET not in str(raised.value)


@pytest.mark.parametrize('unsafe_code', [{}, []], ids=['object', 'list'])
def test_unhashable_rpc_subcode_is_safely_discarded(
    shadow_session_tool,
    unsafe_code,
):
    async def handler(request: web.Request):
        payload = await request.json()
        return web.json_response({
            'jsonrpc': '2.0',
            'id': payload['id'],
            'error': {
                'code': -32603,
                'message': 'internal error',
                'data': {'code': unsafe_code},
            },
        })

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(_call_through_server(shadow_session_tool, handler))

    assert raised.value.code == 'rpc_error'
    assert raised.value.rpc_data_code is None


@pytest.mark.parametrize(
    ('body_factory', 'expected_code'),
    [
        (lambda _request_id: b'not-json', 'invalid_response'),
        (lambda _request_id: b'[]', 'invalid_response'),
        (
            lambda request_id: json.dumps({
                'jsonrpc': '2.0',
                'id': f'wrong-{request_id}',
                'result': {'content': []},
            }).encode(),
            'invalid_response',
        ),
        (
            lambda request_id: json.dumps({
                'jsonrpc': '2.0',
                'id': request_id,
                'result': {'content': [
                    {'type': 'text', 'text': '{}'},
                    {'type': 'text', 'text': '{}'},
                ]},
            }).encode(),
            'invalid_result',
        ),
        (
            lambda request_id: json.dumps({
                'jsonrpc': '2.0',
                'id': request_id,
                'result': {'content': [{'type': 'text', 'text': '[]'}]},
            }).encode(),
            'invalid_result',
        ),
        (
            lambda request_id: (
                '{"jsonrpc":"2.0","id":"'
                + request_id
                + '","result":{"content":[{"type":"text",'
                '"text":"{\\"value\\":NaN}"}]}}'
            ).encode(),
            'invalid_result',
        ),
    ],
    ids=[
        'non-json',
        'top-level-list',
        'wrong-request-id',
        'multiple-content-items',
        'json-text-not-object',
        'non-finite-result',
    ],
)
def test_malformed_responses_fail_closed(
    shadow_session_tool,
    body_factory,
    expected_code,
):
    async def handler(request: web.Request):
        payload = await request.json()
        return web.Response(
            body=body_factory(payload['id']),
            content_type='application/json',
        )

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(_call_through_server(shadow_session_tool, handler))

    assert raised.value.code == expected_code


def test_fragmented_response_is_read_to_eof_before_strict_decode(shadow_session_tool):
    async def handler(request: web.Request):
        payload = await request.json()
        response = web.StreamResponse(headers={'Content-Type': 'application/json'})
        await response.prepare(request)
        encoded = json.dumps({
            'jsonrpc': '2.0',
            'id': payload['id'],
            'result': {
                'content': [{'type': 'text', 'text': '{"fragmented":true}'}],
            },
        }).encode()
        for boundary in (1, 7, 19, len(encoded)):
            chunk, encoded = encoded[:boundary], encoded[boundary:]
            if chunk:
                await response.write(chunk)
        if encoded:
            await response.write(encoded)
        await response.write_eof()
        return response

    result = asyncio.run(_call_through_server(shadow_session_tool, handler))
    assert result == {'fragmented': True}


def test_excessively_nested_json_is_a_sanitized_protocol_error(shadow_session_tool):
    nested = (b'{"x":' * 2_000) + b'0' + (b'}' * 2_000)

    async def handler(_request: web.Request):
        return web.Response(body=nested, content_type='application/json')

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(_call_through_server(shadow_session_tool, handler))

    assert raised.value.code == 'invalid_response'


@pytest.mark.parametrize(
    ('trust_state', 'online', 'runtime_trusted', 'expected_code'),
    [
        ('untrusted', True, True, 'target_not_trusted'),
        ('quarantined', True, True, 'target_not_trusted'),
        ('trusted', False, True, 'registry_offline'),
        ('trusted', True, False, 'registry_not_trusted'),
    ],
)
def test_target_requires_trust_and_online_runtime(
    shadow_session_tool,
    trust_state,
    online,
    runtime_trusted,
    expected_code,
):
    url = 'http://127.0.0.1:9/mcp'
    _install_target(
        shadow_session_tool,
        url,
        trust_state=trust_state,
        online=online,
        runtime_trusted=runtime_trusted,
    )

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(mcp_client.call_trusted_shadow_session(
            'teleop-robot',
            'status',
        ))

    assert raised.value.code == expected_code


def test_target_id_must_be_unique(shadow_session_tool):
    url = 'http://127.0.0.1:9/mcp'
    _install_target(shadow_session_tool, url)
    duplicate = deepcopy(config.main['services']['mcp'][0])
    duplicate['url'] = 'http://127.0.0.1:10/mcp'
    config.main['services'] = {
        'mcp': [*config.main['services']['mcp'], duplicate],
    }

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(mcp_client.call_trusted_shadow_session(
            'teleop-robot',
            'status',
        ))

    assert raised.value.code == 'target_ambiguous'


@pytest.mark.parametrize(
    ('url', 'transport', 'runtime_url', 'expected_code'),
    [
        ('ftp://robot.local/mcp', 'http', None, 'invalid_url'),
        ('http://user:password@robot.local/mcp', 'http', None, 'invalid_url'),
        ('http://robot.local/mcp?fence=private', 'http', None, 'invalid_url'),
        ('http://robot.local/mcp#fragment', 'http', None, 'invalid_url'),
        ('http://robot.local/mcp', 'internal', None, 'invalid_transport'),
        (
            'http://127.0.0.1:15711/mcp',
            'http',
            'http://127.0.0.1:15712/mcp',
            'registry_target_mismatch',
        ),
    ],
)
def test_target_transport_url_and_registry_identity_are_exact(
    shadow_session_tool,
    url,
    transport,
    runtime_url,
    expected_code,
):
    _install_target(shadow_session_tool, url)
    target = config.main['services']['mcp'][0]
    target['transport'] = transport
    config.main['services'] = {'mcp': [target]}
    if runtime_url is not None:
        mcp_client.registry['teleop-robot']['url'] = runtime_url

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(mcp_client.call_trusted_shadow_session(
            'teleop-robot',
            'status',
        ))

    assert raised.value.code == expected_code
    assert 'private' not in str(raised.value)
    assert 'password' not in str(raised.value)


@pytest.mark.parametrize(
    ('mutate', 'action', 'expected_code'),
    [
        (
            lambda tool: tool['x-teleop'].__setitem__('mode', 'live'),
            'status',
            'descriptor_invalid',
        ),
        (
            lambda tool: tool['x-teleop'].__setitem__('actuation_enabled', 0),
            'status',
            'descriptor_invalid',
        ),
        (
            lambda tool: tool['x-teleop'].__setitem__('capability_digest', 'A' * 64),
            'status',
            'descriptor_invalid',
        ),
        pytest.param(
            lambda tool: tool['x-teleop'].pop('dispatch_contract'),
            'status',
            'descriptor_invalid',
            id='missing-dispatch-contract',
        ),
        pytest.param(
            lambda tool: tool['x-teleop'].__setitem__(
                'dispatch_contract',
                'motus.teleop.dispatch.live.v1',
            ),
            'status',
            'descriptor_invalid',
            id='wrong-dispatch-contract',
        ),
        pytest.param(
            lambda tool: tool['x-teleop'].pop('signaling'),
            'status',
            'descriptor_invalid',
            id='missing-signaling-contract',
        ),
        pytest.param(
            lambda tool: tool['x-teleop']['signaling'].__setitem__(
                'access',
                'browser-direct',
            ),
            'status',
            'descriptor_invalid',
            id='unsafe-signaling-access',
        ),
        (lambda _tool: None, 'not_declared', 'action_not_declared'),
    ],
)
def test_descriptor_and_action_are_revalidated_before_every_call(
    shadow_session_tool,
    mutate,
    action,
    expected_code,
):
    tool = deepcopy(shadow_session_tool)
    mutate(tool)
    url = 'http://127.0.0.1:9/mcp'
    _install_target(tool, url)

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(mcp_client.call_trusted_shadow_session(
            'teleop-robot',
            action,
        ))

    assert raised.value.code == expected_code


@pytest.mark.parametrize('missing_action', ['heartbeat', 'release', 'stop'])
def test_full_revocation_contract_is_required_before_prepare(
    shadow_session_tool,
    missing_action,
):
    requests_received = 0

    async def handler(request: web.Request):
        nonlocal requests_received
        requests_received += 1
        payload = await request.json()
        return _rpc_success(payload['id'], {'unexpected': True})

    async def scenario() -> None:
        tool = deepcopy(shadow_session_tool)
        action_enum = tool['inputSchema']['properties']['action']['enum']
        action_enum.remove(missing_action)
        tool['inputSchema']['x-action-params'].pop(missing_action)
        app = web.Application()
        app.router.add_post('/mcp', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        url = f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'
        _install_target(tool, url)
        try:
            await mcp_client.call_trusted_shadow_session(
                'teleop-robot',
                'prepare_shadow',
                {
                    'session_id': '9a9d7841-b587-434c-bda2-bf0baeb6381d',
                    'epoch': 1,
                    'fence': FENCE_SECRET,
                },
                timeout_seconds=1.0,
            )
        finally:
            await runner.cleanup()

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(scenario())

    assert raised.value.code == 'descriptor_invalid'
    assert requests_received == 0


def test_descriptor_driver_id_must_match_config_target(shadow_session_tool):
    url = 'http://127.0.0.1:9/mcp'
    _install_target(shadow_session_tool, url)
    target = config.main['services']['mcp'][0]
    target['tools'][0]['x-teleop']['driver_id'] = 'different-driver'
    config.main['services'] = {'mcp': [target]}

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(mcp_client.call_trusted_shadow_session(
            'teleop-robot',
            'status',
        ))

    assert raised.value.code == 'descriptor_invalid'


def test_reuses_supplied_session_connector_without_closing_it(shadow_session_tool):
    observed_transports = set()
    observed_calls = 0

    async def handler(request: web.Request):
        nonlocal observed_calls
        observed_calls += 1
        observed_transports.add(id(request.transport))
        payload = await request.json()
        return _rpc_success(payload['id'], {'sequence': observed_calls})

    async def scenario():
        app = web.Application()
        app.router.add_post('/mcp', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        url = f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'
        _install_target(shadow_session_tool, url)
        connector = aiohttp.TCPConnector(limit=1)
        try:
            async with aiohttp.ClientSession(connector=connector) as shared:
                first = await mcp_client.call_trusted_shadow_session(
                    'teleop-robot',
                    'status',
                    session=shared,
                    timeout_seconds=1.0,
                )
                second = await mcp_client.call_trusted_shadow_session(
                    'teleop-robot',
                    'status',
                    session=shared,
                    timeout_seconds=1.0,
                )
                assert shared.closed is False
                assert shared.connector is connector
            assert connector.closed is True
            return first, second
        finally:
            await runner.cleanup()

    first, second = asyncio.run(scenario())

    assert first == {'sequence': 1}
    assert second == {'sequence': 2}
    assert observed_calls == 2
    assert len(observed_transports) == 1


def test_pinned_live_call_uses_only_runtime_identity_and_survives_offline_hint(
    shadow_session_tool,
    monkeypatch,
):
    observed_calls = 0

    async def handler(request: web.Request):
        nonlocal observed_calls
        observed_calls += 1
        payload = await request.json()
        return _rpc_success(payload['id'], {'heartbeat': 'accepted'})

    async def scenario() -> dict:
        app = web.Application()
        app.router.add_post('/mcp', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        url = f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'
        _install_target(shadow_session_tool, url)
        target = await mcp_client.resolve_trusted_shadow_target('teleop-robot')

        def unexpected_config_resolution(*_args, **_kwargs):
            raise AssertionError('pinned authority call touched config/SQLite')

        monkeypatch.setattr(
            mcp_client,
            '_trusted_shadow_target',
            unexpected_config_resolution,
        )
        mcp_client.registry['teleop-robot']['online'] = False
        try:
            return await mcp_client.call_trusted_shadow_session(
                'teleop-robot',
                'heartbeat',
                {
                    'boot_id': '8ced194d-1677-465b-bd96-8fd17b97582d',
                    'session_id': '9a9d7841-b587-434c-bda2-bf0baeb6381d',
                    'epoch': 3,
                    'fence': FENCE_SECRET,
                },
                target=target,
                timeout_seconds=1.0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(scenario())
    assert result == {'heartbeat': 'accepted'}
    assert observed_calls == 1


@pytest.mark.parametrize('mutation', ['url', 'fingerprint'])
def test_pinned_identity_change_rejects_before_sending_fence(
    shadow_session_tool,
    mutation,
):
    old_endpoint_calls = 0
    new_endpoint_calls = 0

    async def old_handler(request: web.Request):
        nonlocal old_endpoint_calls
        old_endpoint_calls += 1
        payload = await request.json()
        return _rpc_success(payload['id'], {'unexpected': 'old'})

    async def new_handler(request: web.Request):
        nonlocal new_endpoint_calls
        new_endpoint_calls += 1
        payload = await request.json()
        assert FENCE_SECRET not in repr(payload)
        return _rpc_success(payload['id'], {'unexpected': 'new'})

    async def start_server(handler):
        app = web.Application()
        app.router.add_post('/mcp', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        return runner, f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'

    async def scenario() -> None:
        old_runner, old_url = await start_server(old_handler)
        new_runner, new_url = await start_server(new_handler)
        _install_target(shadow_session_tool, old_url)
        target = await mcp_client.resolve_trusted_shadow_target('teleop-robot')
        if mutation == 'url':
            mcp_client.registry['teleop-robot']['url'] = new_url
        else:
            mcp_client.registry['teleop-robot']['teleop_fingerprint'] = 'f' * 64
        try:
            await mcp_client.call_trusted_shadow_session(
                'teleop-robot',
                'heartbeat',
                {
                    'boot_id': '8ced194d-1677-465b-bd96-8fd17b97582d',
                    'session_id': '9a9d7841-b587-434c-bda2-bf0baeb6381d',
                    'epoch': 3,
                    'fence': FENCE_SECRET,
                },
                target=target,
                timeout_seconds=1.0,
            )
        finally:
            await old_runner.cleanup()
            await new_runner.cleanup()

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(scenario())

    assert raised.value.code == 'pinned_target_changed'
    assert old_endpoint_calls == 0
    assert new_endpoint_calls == 0


def test_redirect_is_not_followed_with_driver_authorization(shadow_session_tool):
    redirect_target_hit = False

    async def source(_request: web.Request):
        raise web.HTTPFound('/credential-sink')

    async def sink(_request: web.Request):
        nonlocal redirect_target_hit
        redirect_target_hit = True
        return web.Response(text='unexpected')

    async def scenario():
        app = web.Application()
        app.router.add_post('/mcp', source)
        app.router.add_get('/credential-sink', sink)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        url = f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'
        _install_target(shadow_session_tool, url)
        try:
            await mcp_client.call_trusted_shadow_session(
                'teleop-robot',
                'status',
                timeout_seconds=1.0,
            )
        finally:
            await runner.cleanup()

    with pytest.raises(mcp_client.TrustedShadowTransportError) as raised:
        asyncio.run(scenario())

    assert raised.value.code == 'http_error'
    assert raised.value.http_status == 302
    assert redirect_target_hit is False


def test_generic_call_still_refuses_reserved_teleop_tool(shadow_session_tool):
    url = 'http://127.0.0.1:9/mcp'
    _install_target(shadow_session_tool, url)

    result = asyncio.run(mcp_client.call_tool(
        'mcp__teleop-robot__teleop_session',
        {'action': 'status'},
    ))

    assert result == 'Teleop tools are reserved for the dedicated teleop API'
