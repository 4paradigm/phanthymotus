from __future__ import annotations

import asyncio
import json

import auth
import mcp_client
import pytest
from aiohttp import web

DRIVER_TOKEN = 'driver-bearer-must-stay-server-side'
OTHER_DRIVER_TOKEN = 'other-driver-bearer-must-stay-private'
DIGEST = '0123456789abcdef' * 4


async def _serve(handler):
    app = web.Application()
    app.router.add_post('/offer', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    origin = f'http://127.0.0.1:{sockets[0].getsockname()[1]}'
    target = mcp_client.TrustedShadowTarget(
        mcp_id='teleop-driver',
        url=f'{origin}/mcp',
        capability_digest=DIGEST,
        descriptor_fingerprint='descriptor-fingerprint',
        actions=frozenset({'status'}),
    )
    mcp_client.registry['teleop-driver'] = {
        'trusted': True,
        'url': target.url,
        'teleop_fingerprint': target.descriptor_fingerprint,
    }
    return runner, target, origin


def _call(target: mcp_client.TrustedShadowTarget, **kwargs):
    return mcp_client.call_trusted_shadow_offer(
        'teleop-driver',
        {'type': 'offer', 'sdp': 'v=0\r\no=quest-offer'},
        'private.ticket.signature',
        target=target,
        **kwargs,
    )


def test_offer_uses_same_origin_bearer_and_strict_server_payload():
    observed = {}

    async def scenario():
        async def handler(request: web.Request):
            observed['path'] = request.path
            observed['authorization'] = request.headers.get('Authorization')
            observed['payload'] = await request.json()
            return web.json_response({'type': 'answer', 'sdp': 'v=0\r\na=answer'})

        runner, target, _origin = await _serve(handler)
        auth.init({
            'MOTUS_DRIVER_TOKENS': (
                f'{{"other-driver":"{OTHER_DRIVER_TOKEN}",'
                f'"teleop-driver":"{DRIVER_TOKEN}"}}'
            ),
        })
        try:
            return await _call(target, timeout_seconds=1.0)
        finally:
            await runner.cleanup()

    result = asyncio.run(scenario())
    assert result == {'type': 'answer', 'sdp': 'v=0\r\na=answer'}
    assert observed == {
        'path': '/offer',
        'authorization': f'Bearer {DRIVER_TOKEN}',
        'payload': {
            'sdp': 'v=0\r\no=quest-offer',
            'type': 'offer',
            'ticket': 'private.ticket.signature',
        },
    }
    assert OTHER_DRIVER_TOKEN not in json.dumps(observed)


@pytest.mark.parametrize(
    ('target_url', 'expected_code'),
    [
        ('http://127.0.0.1:9/not-mcp', 'invalid_url'),
        ('http://127.0.0.1:9/mcp/', 'invalid_url'),
        ('http://127.0.0.1:9/mcp?next=/offer', 'invalid_url'),
        ('http://user@127.0.0.1:9/mcp', 'invalid_url'),
    ],
)
def test_offer_url_derivation_rejects_noncanonical_mcp_targets(target_url, expected_code):
    target = mcp_client.TrustedShadowTarget(
        mcp_id='teleop-driver',
        url=target_url,
        capability_digest=DIGEST,
        descriptor_fingerprint='descriptor-fingerprint',
        actions=frozenset({'status'}),
    )
    mcp_client.registry['teleop-driver'] = {
        'trusted': True,
        'url': target.url,
        'teleop_fingerprint': target.descriptor_fingerprint,
    }
    auth.init({'MOTUS_DRIVER_TOKEN': DRIVER_TOKEN})
    with pytest.raises(mcp_client.TrustedShadowTransportError) as captured:
        asyncio.run(_call(target, timeout_seconds=0.25))
    assert captured.value.code == expected_code
    assert DRIVER_TOKEN not in str(captured.value)


def test_redirect_is_rejected_and_never_followed():
    followed = False

    async def scenario():
        nonlocal followed
        app = web.Application()

        async def offer(_request: web.Request):
            raise web.HTTPTemporaryRedirect('/sink')

        async def sink(_request: web.Request):
            nonlocal followed
            followed = True
            return web.json_response({'type': 'answer', 'sdp': 'wrong'})

        app.router.add_post('/offer', offer)
        app.router.add_post('/sink', sink)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        url = f'http://127.0.0.1:{sockets[0].getsockname()[1]}/mcp'
        target = mcp_client.TrustedShadowTarget(
            'teleop-driver', url, DIGEST, 'descriptor-fingerprint', frozenset({'status'}),
        )
        mcp_client.registry['teleop-driver'] = {
            'trusted': True,
            'url': url,
            'teleop_fingerprint': target.descriptor_fingerprint,
        }
        auth.init({'MOTUS_DRIVER_TOKEN': DRIVER_TOKEN})
        try:
            with pytest.raises(mcp_client.TrustedShadowTransportError) as captured:
                await _call(target, timeout_seconds=1.0)
            return captured.value
        finally:
            await runner.cleanup()

    error = asyncio.run(scenario())
    assert error.code == 'redirect_rejected'
    assert error.http_status == 307
    assert followed is False


@pytest.mark.parametrize(
    ('kind', 'expected_code'),
    [
        ('timeout', 'timeout'),
        ('large', 'response_too_large'),
        ('invalid_json', 'invalid_response'),
        ('duplicate_json', 'invalid_response'),
    ],
)
def test_offer_bounds_time_size_and_json(kind, expected_code):
    async def scenario():
        async def handler(_request: web.Request):
            if kind == 'timeout':
                await asyncio.sleep(0.5)
                return web.json_response({'type': 'answer', 'sdp': 'late'})
            if kind == 'large':
                return web.Response(
                    body=json.dumps({'sdp': 'x' * (300 * 1024)}).encode(),
                    content_type='application/json',
                )
            if kind == 'duplicate_json':
                return web.Response(
                    body=b'{"type":"answer","type":"answer","sdp":"v=0"}',
                    content_type='application/json',
                )
            return web.Response(body=b'{"sdp":NaN}', content_type='application/json')

        runner, target, _origin = await _serve(handler)
        auth.init({'MOTUS_DRIVER_TOKEN': DRIVER_TOKEN})
        try:
            with pytest.raises(mcp_client.TrustedShadowTransportError) as captured:
                await _call(target, timeout_seconds=0.25 if kind == 'timeout' else 1.0)
            return captured.value
        finally:
            await runner.cleanup()

    error = asyncio.run(scenario())
    assert error.code == expected_code
    assert DRIVER_TOKEN not in str(error)
    assert 'quest-offer' not in str(error)


def test_offer_rechecks_pinned_registry_identity_after_network_await():
    async def scenario():
        async def handler(_request: web.Request):
            mcp_client.registry['teleop-driver']['url'] = 'http://127.0.0.1:9/mcp'
            return web.json_response({'type': 'answer', 'sdp': 'v=0'})

        runner, target, _origin = await _serve(handler)
        auth.init({'MOTUS_DRIVER_TOKEN': DRIVER_TOKEN})
        try:
            with pytest.raises(mcp_client.TrustedShadowTransportError) as captured:
                await _call(target, timeout_seconds=1.0)
            return captured.value
        finally:
            await runner.cleanup()

    error = asyncio.run(scenario())
    assert error.code == 'pinned_target_changed'


def test_offer_fails_closed_without_driver_bearer():
    async def scenario():
        async def handler(_request: web.Request):
            raise AssertionError('request must not be sent')

        runner, target, _origin = await _serve(handler)
        auth.init({})
        try:
            with pytest.raises(mcp_client.TrustedShadowTransportError) as captured:
                await _call(target, timeout_seconds=1.0)
            return captured.value
        finally:
            await runner.cleanup()

    error = asyncio.run(scenario())
    assert error.code == 'driver_auth_unavailable'
