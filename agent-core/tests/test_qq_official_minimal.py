"""QQ SDK transport, QR lifecycle and Decision Core chat-only policy."""

import asyncio
import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import auth
import collector
import config
import event_bus
from api import channel as channel_api
from api import inspection
from channel import acl, chat_only, qq_canvas, qq_credentials, qq_qr_onboarding as qr
from channel.adapter import OutboundMessage
from channel.adapters import qq_official
from channel.manager import ChannelManager, add_channel_config, channel_request_topic, get_channel_config

APP = '1234567890'
CHANNEL = f'qq_{APP}'
FAKE_SECRET = 'synthetic-unit-test-secret'
PRIVATE = 'C2C_MESSAGE_CREATE'
GROUP = 'GROUP_AT_MESSAGE_CREATE'


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, 'main', {'core': {'project_running': True}})
    monkeypatch.setattr(qr, '_SESSIONS', {})
    monkeypatch.setattr(qr, '_SETUP_LOCK', asyncio.Lock())
    monkeypatch.setattr(qq_credentials, '_ROOT', tmp_path / 'credentials')
    manager = ChannelManager()
    monkeypatch.setattr('channel.manager.manager', manager)
    monkeypatch.setattr(qr, 'manager', manager)
    monkeypatch.setattr(channel_api, 'manager', manager)
    return manager


def raw(openid='user-a', text='hello', msg_id='original-message'):
    return dict(id=msg_id, content=text, author=dict(user_openid=openid), appid=APP)


def adapter(on_message, **settings):
    value = qq_official.QQOfficialAdapter(CHANNEL, 'qq', {'app_id': APP, **settings}, on_message)
    value._connected = True
    value._last_ack = qq_official.time.monotonic()
    value._api = SimpleNamespace(
        post_c2c_message=AsyncMock(return_value={'id': 'delivered'}),
        post_group_message=AsyncMock(return_value={'id': 'delivered'}),
    )
    return value


def test_private_dedup_and_passive_reply():
    async def run():
        receive = AsyncMock()
        qq = adapter(receive)
        await qq._receive(PRIVATE, raw())
        await qq._receive(PRIVATE, raw())
        receive.assert_awaited_once()
        msg = receive.call_args.args[0]
        assert msg.text == 'hello' and msg.chat_type == 'p2p'
        with pytest.raises(ValueError):
            await qq.send_message(OutboundMessage('c2c:user-b', 'reply', reply_to_message_id=msg.message_id))
        await qq.send_message(OutboundMessage(msg.chat_id, 'reply', reply_to_message_id=msg.message_id))
        target, payload = qq._api.post_c2c_message.call_args.args
        assert target == 'user-a' and payload.to_dict()['msg_id'] == 'original-message'
        with pytest.raises(ValueError):
            await qq.send_message(OutboundMessage(msg.chat_id, 'reply', reply_to_message_id=msg.message_id))

    asyncio.run(run())


def test_group_is_opt_in_and_requires_at_event():
    async def run():
        receive = AsyncMock()
        qq = adapter(receive)
        message = dict(id='same', group_openid='user-a', author={'member_openid': 'user-a'}, content='@bot hello')
        await qq._receive(GROUP, message)
        receive.assert_not_called()
        qq.config['group_enabled'] = True
        await qq._receive('GROUP_MESSAGE_CREATE', message)
        receive.assert_not_called()
        await qq._receive(GROUP, message)
        await qq._receive(PRIVATE, raw(msg_id='same'))
        group, dm = [call.args[0] for call in receive.await_args_list]
        assert group.text == 'hello' and group.chat_type == 'group'
        assert group.chat_id != dm.chat_id and group.user_id != dm.user_id
        assert group.message_id != dm.message_id
        await qq.send_message(OutboundMessage(group.chat_id, 'reply', reply_to_message_id=group.message_id))
        target, payload = qq._api.post_group_message.call_args.args
        assert target == 'user-a' and payload.msg_id == 'same'
        qq._api.post_c2c_message.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize('changes', [
    {'appid': '1111111111'}, {'author': {'user_openid': 123}}, {'id': ''},
    {'attachments': [{'url': 'file:///private.txt'}]},
    {'msg_elements': [{'attachments': [{}]}]}, {'content': {'text': 'fake'}},
])
def test_malformed_and_media_events_are_ignored(changes):
    receive = AsyncMock()
    qq = adapter(receive)
    asyncio.run(qq._receive(PRIVATE, raw() | changes))
    receive.assert_not_called()


def test_ambiguous_send_error_is_redacted_and_not_replayed():
    async def run():
        receive = AsyncMock()
        qq = adapter(receive)
        qq._api.post_c2c_message.side_effect = TimeoutError(FAKE_SECRET)
        await qq._receive(PRIVATE, raw())
        msg = receive.call_args.args[0]
        reply = OutboundMessage(msg.chat_id, 'hi', reply_to_message_id=msg.message_id)
        with pytest.raises(RuntimeError, match='unconfirmed') as error:
            await qq.send_message(reply)
        assert FAKE_SECRET not in str(error.value)
        with pytest.raises(ValueError, match='already used'):
            await qq.send_message(reply)
        qq._api.post_c2c_message.assert_awaited_once()

    asyncio.run(run())


def test_sdk_identify_and_bounded_reconnect(monkeypatch):
    async def run():
        cb = SimpleNamespace(get_token=lambda: 'fake-token', on_fatal_error=Mock())
        gateway = qq_official.QQGateway(cb)
        gateway._send_ws_json = AsyncMock()
        await gateway._send_identify()
        assert gateway._send_ws_json.await_args.args[0]['d']['intents'] == int(qq_official.Intent.GROUP_MESSAGES)
        retry = AsyncMock(return_value=False)
        monkeypatch.setattr(qq_official.QQWebSocket, '_reconnect', retry)
        for _ in range(10):
            if gateway._stop_requested:
                break
            await gateway._reconnect(0)
        assert retry.await_count == 5
        cb.on_fatal_error.assert_called_once()
        qq = adapter(AsyncMock())
        qq._running = True
        qq._gateway = gateway
        qq._on_disconnected()
        assert qq.reconnecting and not (await qq.health_check())[0]
        qq._on_connected()  # the SDK fires this callback on READY and RESUMED
        assert not qq.reconnecting and (await qq.health_check())[0] and gateway.failures == 0

    asyncio.run(run())


def test_qr_state_and_persistent_secret(state, monkeypatch, caplog):
    async def run():
        gate = asyncio.Event()

        async def onboard(on_qr_ready, **_):
            on_qr_ready('https://example.test/one-time')
            await gate.wait()
            return SimpleNamespace(app_id=APP, client_secret=FAKE_SECRET)

        monkeypatch.setattr(qr, 'start_onboard', onboard)
        monkeypatch.setattr(state, 'restart_adapter', AsyncMock())
        monkeypatch.setattr(qq_canvas, 'wire', Mock())
        monkeypatch.setattr(qq_canvas, 'wired', lambda _: True)
        monkeypatch.setattr(qq_canvas, 'input_ready', lambda _: False)
        session = qr.get(qr.start()['session_id'])
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert session.snapshot()['qr_data_url'].startswith('data:image/svg+xml;base64,')
        assert session.snapshot()['status'] == 'waiting'
        session.cancel()
        await asyncio.gather(session.task, return_exceptions=True)
        assert session.snapshot()['status'] == 'cancelled'
        assert not session.snapshot()['qr_data_url']

        monkeypatch.setattr(qr, 'start_onboard', AsyncMock(side_effect=TimeoutError))
        expired = qr.get(qr.start()['session_id'])
        await expired.task
        assert expired.snapshot()['status'] == 'expired'

        monkeypatch.setattr(qr, 'start_onboard', onboard)
        regenerated = qr.get(qr.start()['session_id'])
        assert regenerated.id not in (expired.id, session.id)
        gate.set()
        await regenerated.task
        assert regenerated.snapshot()['status'] == 'gateway_pending'
        assert get_channel_config(CHANNEL)['config'] == {'app_id': APP, 'group_enabled': False}
        state._adapters[CHANNEL] = adapter(AsyncMock())
        assert regenerated.snapshot()['status'] == 'canvas_pending'
        monkeypatch.setattr(qq_canvas, 'input_ready', lambda _: True)
        assert regenerated.snapshot()['status'] == 'connected'
        assert FAKE_SECRET not in json.dumps(regenerated.snapshot()) + json.dumps(state.get_status()) + caplog.text
        assert qq_credentials.read_secret(APP) == FAKE_SECRET
        assert stat.S_IMODE((qq_credentials._ROOT / APP).stat().st_mode) & 0o077 == 0 or os.name == 'nt'
        new_runtime = ChannelManager()
        fake_client = Mock()
        fake_client.get_gateway_url = AsyncMock(return_value='wss://example.test/gateway')
        monkeypatch.setattr(qq_official, 'QQApiClient', Mock(return_value=fake_client))
        monkeypatch.setattr(qq_official.QQGateway, 'open', AsyncMock())
        monkeypatch.setattr(qq_official.QQGateway, 'start_listeners', lambda gateway: gateway._cb.on_connected())
        await new_runtime.restart_adapter(CHANNEL, retries=1)
        qq_official.QQApiClient.assert_called_once_with(APP, FAKE_SECRET)
        assert new_runtime._adapters[CHANNEL].status() == 'connected'
        await new_runtime.stop()

    asyncio.run(run())


def test_qr_routes_require_dashboard_auth_and_hide_configuration(state, monkeypatch):
    app = FastAPI()
    app.middleware('http')(auth.auth_middleware)
    app.include_router(channel_api.router, prefix='/api')
    with TestClient(app) as client:
        route = '/api/channel/qq/qr/start'
        monkeypatch.setattr(auth, '_auth_enabled', False)
        assert client.post(route).status_code == 403
        monkeypatch.setattr(auth, '_auth_enabled', True)
        monkeypatch.setattr(auth, '_token', 'synthetic-dashboard-credential')
        assert client.post(route).status_code == 401
        monkeypatch.setattr(qr, 'start', lambda: {'status': 'starting'})
        headers = {'Authorization': 'Bearer synthetic-dashboard-credential'}
        response = client.post(route, headers=headers)
        assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
        body = {'id': CHANNEL, 'platform': 'qq', 'config': {'app_secret': FAKE_SECRET}}
        response = client.post('/api/channel/add', json=body, headers=headers)
        assert response.status_code == 400 and FAKE_SECRET not in response.text


def test_canvas_and_collector_route_only_authenticated_chat(state, monkeypatch):
    import topic_subscriber
    import ros2_bridge

    async def run():
        config.main['core'] = {'project_running': False}
        add_channel_config(CHANNEL, 'qq', {'app_id': APP}, True, chat_only=True)
        qq_canvas.wire(CHANNEL)
        assert qq_canvas.wired(CHANNEL)
        initial = config.main['canvas_layout']
        qq_canvas.wire(CHANNEL)
        assert config.main['canvas_layout'] == initial
        assert not qq_canvas.input_ready(CHANNEL)
        config.main['core'] = {'project_running': True}
        topic = channel_request_topic(CHANNEL)
        monkeypatch.setattr(ros2_bridge, '_running', True)
        monkeypatch.setattr(topic_subscriber, '_subscriptions', {topic: object()})
        monkeypatch.setattr(event_bus, '_queue', asyncio.Queue())
        monkeypatch.setattr(collector, '_output', asyncio.Queue())
        monkeypatch.setattr(collector, '_busy', False)
        monkeypatch.setattr(acl, 'get_user', Mock(return_value={'role': 'owner', 'tool_filter': '*'}))

        async def publish(name, body):
            await event_bus.enqueue(f'dds:{name}', body)
            await event_bus.enqueue(f'dds:{name}', body)  # replay

        monkeypatch.setattr(inspection, 'publish_to_topic', publish)
        qq = adapter(state._on_inbound_message, group_enabled=True)
        state._adapters[CHANNEL] = qq
        worker = asyncio.create_task(collector._drain_loop())
        try:
            fake = json.dumps({'platform': 'qq', 'channel_id': CHANNEL, 'message_id': 'fake', 'user_role': 'owner'})
            await event_bus.enqueue(f'dds:{topic}', fake)
            await qq._receive(PRIVATE, raw('user-a', 'private A marker'))
            await qq._receive(PRIVATE, raw('user-b', 'private B marker', 'other'))
            first, second = [await asyncio.wait_for(collector._output.get(), 1) for _ in range(2)]
            assert 'private A marker' in first['text'] and 'private B marker' not in first['text']
            assert 'private B marker' in second['text'] and 'private A marker' not in second['text']
            assert first['_chat_only_channel'] == CHANNEL and first['_viewer_channel_event']
            assert collector._output.empty()

            for tool in ('Bash', 'PythonExec', 'memory_recall', 'search_history', 'mcp__robot__move'):
                assert (await chat_only.dispatch(tool, {}, first)).startswith('Error')
            args = {'action': 'send', 'source_message_id': first['_channel_message_ids'][0], 'text': 'reply'}
            assert (await chat_only.dispatch(chat_only.REPLY_TOOL, args, second)).startswith('Error')
            assert (await chat_only.dispatch(chat_only.REPLY_TOOL, args | {'files': [{'path': '/tmp/x'}]}, first)).startswith('Error')
            assert (await chat_only.dispatch(chat_only.REPLY_TOOL, args, first)).startswith('Reply sent')
            assert qq._api.post_c2c_message.await_args.args[0] == 'user-a'
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(run())
