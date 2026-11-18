"""
test_wecom_channel.py — 企业微信长连接 ChannelAdapter 一期测试。

全部离线：WS 用 FakeWS / _connect 打桩，不连真实网络、不需要 BotID/Secret。
帧字段名集中在 wecom.py 的常量区，L0 实测回来后只需改那一处与这里的样例帧。

运行：
  cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_wecom_channel.py -q
"""

import asyncio
import base64
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test_wecom.db'))


def _ensure_aiohttp():
    """dev 主机可能没装 aiohttp（仓库用 uv 管理），但适配器顶层 import 了它。

    与 tests/conftest.py 对 docker SDK 的做法一致：只在真的缺失时打桩，
    镜像里跑全量套件时仍使用真实 aiohttp。
    """
    try:
        import aiohttp  # noqa: F401
        return
    except ImportError:
        pass

    stub = types.ModuleType('aiohttp')

    class _WSMsgType:
        TEXT = 1
        BINARY = 2
        CLOSE = 8
        CLOSING = 9
        CLOSED = 256
        ERROR = 258
        PING = 9
        PONG = 10

    class ClientSession:  # noqa: D101
        pass

    class ClientWebSocketResponse:  # noqa: D101
        pass

    stub.WSMsgType = _WSMsgType
    stub.ClientSession = ClientSession
    stub.ClientWebSocketResponse = ClientWebSocketResponse
    sys.modules['aiohttp'] = stub


_ensure_aiohttp()

import config  # noqa: E402
from channel.adapter import Attachment, InboundMessage, OutboundMessage  # noqa: E402
from channel.adapters import wecom  # noqa: E402
from channel.adapters.wecom import WecomAdapter, WecomError  # noqa: E402
from channel.manager import (  # noqa: E402
    ChannelManager, add_channel_config, supports_bot_to_bot, uses_trusted_bots,
)


class FakeWS:
    """最小 WebSocket 替身：记录发出的帧，按需投递收到的帧。"""

    def __init__(self, inbound=None):
        self.sent: list[dict] = []
        self._inbound = asyncio.Queue()
        for frame in inbound or []:
            self._inbound.put_nowait(frame)
        self._closed = False
        self.fail_send = False
        self.auto_reply_ack = True
        self.reply_errcode = 0
        self.subscribe_req_id = ''
        self.reply_sent = asyncio.Event()

    @property
    def closed(self):
        return self._closed

    async def send_json(self, frame):
        if self.fail_send:
            raise RuntimeError('simulated ws send failure')
        self.sent.append(frame)
        if frame.get('cmd') == wecom.CMD_SUBSCRIBE:
            self.subscribe_req_id = frame['headers']['req_id']
        elif frame.get('cmd') == wecom.CMD_RESPOND:
            self.reply_sent.set()
            if self.auto_reply_ack:
                self.push({
                    'errcode': self.reply_errcode,
                    'errmsg': 'ok' if self.reply_errcode == 0 else 'rejected',
                    'headers': {'req_id': frame['headers']['req_id']},
                })

    async def receive(self):
        frame = await self._inbound.get()
        if frame.get('headers', {}).get('req_id') == 'sub_req_1':
            frame = {**frame, 'headers': {'req_id': self.subscribe_req_id}}
        return SimpleNamespace(type=wecom.aiohttp.WSMsgType.TEXT,
                               data=json.dumps(frame, ensure_ascii=False))

    async def close(self):
        self._closed = True

    def push(self, frame):
        self._inbound.put_nowait(frame)

    def frames_with(self, cmd):
        return [f for f in self.sent if f.get('cmd') == cmd]


class FakeSession:
    def __init__(self):
        self._closed = False

    @property
    def closed(self):
        return self._closed

    async def close(self):
        self._closed = True


class DummyAdapter(wecom.ChannelAdapter):
    """记录出站消息的替身，用于验证现有平台不回归。"""

    def __init__(self):
        super().__init__('wecom_test', 'wecom', {}, self._noop)
        self.sent: list[OutboundMessage] = []
        self._running = True

    async def _noop(self, msg):
        pass

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False

    async def send_message(self, msg):
        self.sent.append(msg)


def _make_adapter(on_msg=None, config=None):
    async def _noop(msg):
        pass
    return WecomAdapter('wecom_test', 'wecom',
                        config if config is not None else {'bot_id': 'b', 'secret': 's'},
                        on_msg or _noop)


def _subscribe_ack(errcode=0):
    return {'errcode': errcode, 'errmsg': 'ok' if errcode == 0 else 'bad credentials',
            'headers': {'req_id': 'sub_req_1'}}


def _msg_frame(msgid='m1', req_id='r1', text='你好', chattype='single',
               chatid=None, userid='u1'):
    body = {
        'msgid': msgid,
        'chattype': chattype,
        'from': {'userid': userid},
        'msgtype': 'text',
        'text': {'content': text},
    }
    if chatid:
        body['chatid'] = chatid
    return {'cmd': 'aibot_msg_callback', 'headers': {'req_id': req_id}, 'body': body}


class WecomRegistrationTest(unittest.TestCase):
    """1-2：能力声明驱动的平台自适应，框架无需写 if platform == 'wecom'。"""

    def test_wecom_adapter_registered_and_no_a2a(self):
        self.assertFalse(supports_bot_to_bot('wecom'))
        self.assertFalse(uses_trusted_bots('wecom'))
        self.assertEqual(WecomAdapter.SUPPORTED_FILE_KINDS, (wecom.KIND_IMAGE,))
        self.assertFalse(WecomAdapter.SUPPORTS_BOT_TO_BOT)
        self.assertFalse(WecomAdapter.USES_TRUSTED_BOTS)
        cls = wecom.WecomAdapter
        self.assertTrue(issubclass(cls, wecom.ChannelAdapter))

    def test_add_channel_wecom_strips_a2a_config(self):
        self._db = os.path.join(tempfile.mkdtemp(), 'chan.db')
        with mock.patch.object(config, 'DB_PATH', self._db):
            entry = add_channel_config(
                'wecom_cap_test', 'wecom', {'bot_id': 'b', 'secret': 's'},
                enabled=False, bot_to_bot_enabled=True,
                trusted_bots=[{'id': 'x', 'chat_id': 'oc_x', 'open_id': 'ou_x'}],
            )
        self.assertFalse(entry['bot_to_bot_enabled'])
        self.assertEqual(entry['trusted_bots'], [])


class WecomConnectionTest(unittest.IsolatedAsyncioTestCase):
    """3-4：订阅前置校验与心跳。"""

    def setUp(self):
        self.received: list[InboundMessage] = []

    async def test_connection_uses_proxy_environment(self):
        session = FakeSession()
        ws = FakeWS()
        session.ws_connect = mock.AsyncMock(return_value=ws)
        adapter = _make_adapter()
        with mock.patch.object(wecom.aiohttp, 'ClientSession', return_value=session) as client:
            actual_session, actual_ws = await adapter._connect()
        client.assert_called_once_with(trust_env=True)
        session.ws_connect.assert_awaited_once_with(wecom.WSS_URL, heartbeat=None)
        self.assertIs(actual_session, session)
        self.assertIs(actual_ws, ws)

    async def _start(self, inbound=None, cfg=None):
        async def _on_msg(msg):
            self.received.append(msg)
        ws = FakeWS(inbound if inbound is not None else [_subscribe_ack()])
        session = FakeSession()
        adapter = _make_adapter(_on_msg, cfg)
        with mock.patch.object(adapter, '_connect',
                               new=mock.AsyncMock(return_value=(session, ws))):
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        return adapter, ws

    async def test_start_sends_subscribe_with_credentials(self):
        adapter, ws = await self._start()
        subs = ws.frames_with(wecom.CMD_SUBSCRIBE)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]['body']['bot_id'], 'b')
        self.assertEqual(subs[0]['body']['secret'], 's')
        self.assertTrue(subs[0]['headers']['req_id'])
        self.assertEqual(adapter.status(), 'connected')

    async def test_start_raises_when_credentials_missing(self):
        adapter = _make_adapter(config={})
        with self.assertRaises(WecomError) as ctx:
            await adapter.start()
        self.assertIn('bot_id and secret', str(ctx.exception))
        self.assertFalse(adapter.is_running)

    async def test_start_raises_on_bad_subscribe_ack(self):
        ack = _subscribe_ack(errcode=40001)
        with self.assertRaises(WecomError) as ctx:
            await self._start([ack], cfg={'bot_id': 'b', 'secret': 'UNIQUE_SECRET_XYZ'})
        self.assertIn('errcode=40001', str(ctx.exception))
        # 凭据绝不能出现在异常文本里（会被日志/上层回报带出去）
        self.assertNotIn('UNIQUE_SECRET_XYZ', str(ctx.exception))

    async def test_start_raises_when_subscribe_ack_missing(self):
        # 服务端既不返回 ACK 也不关连接 → 必须超时抛错，不能「起任务就成功」
        with mock.patch.object(wecom, 'SUBSCRIBE_TIMEOUT', 0.05):
            with self.assertRaises(WecomError) as ctx:
                await self._start([])
        self.assertIn('timed out', str(ctx.exception))

    async def test_start_rejects_unmatched_subscribe_ack(self):
        ack = _subscribe_ack()
        ack['headers']['req_id'] = 'another-request'
        with self.assertRaisesRegex(WecomError, 'unmatched ACK'):
            await self._start([ack])

    async def test_heartbeat_sends_ping(self):
        adapter, ws = await self._start()
        with mock.patch.object(wecom, 'HEARTBEAT_INTERVAL', 0.01):
            # 重启心跳任务以使用打桩后的间隔
            adapter._heartbeat_task.cancel()
            adapter._heartbeat_task = asyncio.create_task(adapter._heartbeat_loop())
            await asyncio.sleep(0.05)
        self.assertTrue(ws.frames_with(wecom.CMD_PING))


class WecomInboundTest(unittest.IsolatedAsyncioTestCase):
    """5-6：入站帧归一化与去重。"""

    def setUp(self):
        self.received: list[InboundMessage] = []

    async def _adapter(self):
        async def _on_msg(msg):
            self.received.append(msg)
        return _make_adapter(_on_msg)

    async def test_single_chat_uses_userid_as_chat_id(self):
        adapter = await self._adapter()
        frame = _msg_frame(chattype='single')
        parsed = adapter._extract_inbound(frame, frame['body'])
        self.assertEqual(parsed['chat_type'], 'p2p')
        self.assertEqual(parsed['chat_id'], 'u1')
        self.assertEqual(parsed['message_id'], 'm1')
        self.assertEqual(parsed['text'], '你好')
        self.assertEqual(parsed['user_id'], 'u1')

    async def test_group_chat_and_at_prefix_stripped(self):
        adapter = await self._adapter()
        frame = _msg_frame(chattype='group', chatid='g1', text='@Bumi 挥个手')
        await adapter._process_message(frame, frame['body'])
        self.assertEqual(len(self.received), 1)
        msg = self.received[0]
        self.assertEqual(msg.platform, 'wecom')
        self.assertEqual(msg.chat_type, 'group')
        self.assertEqual(msg.chat_id, 'g1')
        self.assertEqual(msg.text, '挥个手')
        self.assertEqual(msg.message_id, 'm1')

    async def test_non_text_message_gets_placeholder(self):
        adapter = await self._adapter()
        frame = _msg_frame()
        frame['body']['msgtype'] = 'file'
        frame['body']['text'] = None
        parsed = adapter._extract_inbound(frame, frame['body'])
        self.assertIn('不支持的非文本消息', parsed['text'])

    async def test_image_is_prepared_after_authorization(self):
        adapter = await self._adapter()
        frame = _msg_frame()
        frame['body'].update(msgtype='image', text=None, image={'url': 'https://example.test/image', 'aeskey': 'key'})
        attachment = Attachment(kind=wecom.KIND_IMAGE, path='/work/resource/channel_files/x.png')
        adapter._send_progress = mock.AsyncMock()
        adapter._download_image = mock.AsyncMock(return_value=attachment)
        await adapter._process_message(frame, frame['body'])
        msg = self.received[0]
        self.assertEqual(msg.text, '[图片]')
        self.assertEqual(msg.attachments, [])
        adapter._download_image.assert_not_awaited()
        await msg.prepare()
        adapter._send_progress.assert_awaited_once_with('m1')
        self.assertEqual(msg.attachments, [attachment])

    async def test_mixed_message_keeps_text_and_image(self):
        adapter = await self._adapter()
        frame = _msg_frame()
        frame['body'].update(msgtype='mixed', text=None, mixed={'msg_item': [
            {'msgtype': 'text', 'text': {'content': '这是什么？'}},
            {'msgtype': 'image', 'image': {'url': 'https://example.test/image', 'aeskey': 'key'}},
        ]})
        parsed = adapter._extract_inbound(frame, frame['body'])
        self.assertEqual(parsed['text'], '这是什么？')
        self.assertEqual(len(parsed['images']), 1)

    def test_decrypt_image_with_32_byte_padding(self):
        key = bytes(range(32))
        original = b'\x89PNG\r\n\x1a\n' + b'a' * 24
        padding = 32 - len(original) % 32
        ciphertext = wecom.AES.new(key, wecom.AES.MODE_CBC, iv=key[:16]).encrypt(
            original + bytes([padding]) * padding)
        encoded_key = base64.b64encode(key).decode().rstrip('=')
        self.assertEqual(wecom._decrypt_image(ciphertext, encoded_key), original)
        with self.assertRaises(WecomError):
            wecom._decrypt_image(ciphertext, 'invalid')

    async def test_duplicate_msgid_dropped(self):
        adapter = await self._adapter()
        frame = _msg_frame()
        await adapter._process_message(frame, frame['body'])
        await adapter._process_message(frame, frame['body'])
        self.assertEqual(len(self.received), 1)
        self.assertEqual(adapter._duplicate_drops, 1)

    async def test_event_callback_is_not_delivered_as_a_message(self):
        adapter = await self._adapter()
        event = {
            'cmd': wecom.CMD_EVENT_CALLBACK,
            'headers': {'req_id': 'event-req'},
            'body': {'msgid': 'event-id', 'msgtype': 'event',
                     'from': {'userid': 'u1'}, 'event': {'eventtype': 'enter_chat'}},
        }
        await adapter._handle_frame(event)
        self.assertEqual(self.received, [])

    async def test_inbound_handler_can_wait_for_reply_ack(self):
        delivered = asyncio.Event()
        errors = []
        adapter = None

        async def on_message(msg):
            try:
                await adapter.send_message(OutboundMessage(
                    chat_id=msg.chat_id, text='Pending approval.',
                    source_message_id=msg.message_id,
                ))
            except Exception as e:
                errors.append(e)
            finally:
                delivered.set()

        ws = FakeWS([_subscribe_ack()])
        adapter = _make_adapter(on_message)
        with mock.patch.object(adapter, '_connect',
                               new=mock.AsyncMock(return_value=(FakeSession(), ws))):
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        ws.push(_msg_frame())
        await asyncio.wait_for(delivered.wait(), timeout=1)
        self.assertEqual(errors, [])
        self.assertEqual(len(ws.frames_with(wecom.CMD_RESPOND)), 1)


class WecomReqIdTest(unittest.IsolatedAsyncioTestCase):
    """9-11：方案 B 的 req_id 精确映射。"""

    def setUp(self):
        self.received: list[InboundMessage] = []

    async def _running_adapter(self):
        async def _on_msg(msg):
            self.received.append(msg)
        ws = FakeWS([_subscribe_ack()])
        adapter = _make_adapter(_on_msg)
        with mock.patch.object(adapter, '_connect',
                               new=mock.AsyncMock(return_value=(FakeSession(), ws))):
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        return adapter, ws

    async def test_reply_uses_exact_req_id(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])

        msg = OutboundMessage(chat_id='u1', text='收到', source_message_id='m1')
        await adapter.send_message(msg)
        replies = ws.frames_with(wecom.CMD_RESPOND)
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]['headers']['req_id'], 'r1')
        self.assertTrue(replies[0]['body']['stream']['finish'])
        self.assertEqual(replies[0]['body']['stream']['content'], '收到')

    async def test_image_only_reply_uses_source_request(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('image-message', 'image-request')
        await adapter._process_message(frame, frame['body'])
        image = b'\x89PNG\r\n\x1a\nexample'
        await adapter.send_message(OutboundMessage(
            chat_id='u1', image_bytes=image, source_message_id='image-message'))
        reply = ws.frames_with(wecom.CMD_RESPOND)[0]
        self.assertEqual(reply['headers']['req_id'], 'image-request')
        item = reply['body']['stream']['msg_item'][0]['image']
        self.assertEqual(base64.b64decode(item['base64']), image)
        self.assertEqual(item['md5'], hashlib.md5(image, usedforsecurity=False).hexdigest())

    async def test_image_progress_and_final_share_stream_id(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        await adapter._send_progress('m1')
        await adapter.send_message(OutboundMessage(chat_id='u1', text='完成', source_message_id='m1'))
        first, final = ws.frames_with(wecom.CMD_RESPOND)
        self.assertFalse(first['body']['stream']['finish'])
        self.assertTrue(final['body']['stream']['finish'])
        self.assertEqual(first['body']['stream']['id'], final['body']['stream']['id'])

    async def test_concurrent_messages_keep_their_own_req_id(self):
        adapter, ws = await self._running_adapter()
        a = _msg_frame('mA', 'rA', text='A')
        b = _msg_frame('mB', 'rB', text='B')
        await adapter._process_message(a, a['body'])
        await adapter._process_message(b, b['body'])

        await adapter.send_message(OutboundMessage(chat_id='u1', text='回A', source_message_id='mA'))
        await adapter.send_message(OutboundMessage(chat_id='u1', text='回B', source_message_id='mB'))
        replies = ws.frames_with(wecom.CMD_RESPOND)
        self.assertEqual([f['headers']['req_id'] for f in replies], ['rA', 'rB'])
        self.assertEqual([f['body']['stream']['content'] for f in replies], ['回A', '回B'])

    async def test_missing_reply_context_raises(self):
        adapter, _ = await self._running_adapter()
        with self.assertRaises(WecomError) as ctx:
            await adapter.send_message(
                OutboundMessage(chat_id='nobody', text='x', source_message_id='unknown'))
        self.assertIn('No reply context', str(ctx.exception))

    async def test_missing_source_id_does_not_fall_back_to_newer_message(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('new-message', 'new-request')
        await adapter._process_message(frame, frame['body'])
        with self.assertRaisesRegex(WecomError, 'No reply context'):
            await adapter.send_message(OutboundMessage(
                chat_id='u1', text='old reply', source_message_id='old-message'))
        self.assertEqual(ws.frames_with(wecom.CMD_RESPOND), [])

    async def test_reply_without_source_id_does_not_guess_current_chat(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        with self.assertRaisesRegex(WecomError, 'No reply context'):
            await adapter.send_message(OutboundMessage(chat_id='u1', text='Pending approval.'))
        self.assertEqual(ws.frames_with(wecom.CMD_RESPOND), [])

    async def test_reply_rejection_is_reported_to_caller(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        ws.reply_errcode = 40001
        with self.assertRaisesRegex(WecomError, 'reply rejected.*40001'):
            await adapter.send_message(OutboundMessage(
                chat_id='u1', text='x', source_message_id='m1'))

    async def test_reply_ack_timeout_is_reported_to_caller(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        ws.auto_reply_ack = False
        with mock.patch.object(wecom, 'REPLY_ACK_TIMEOUT', 0.05):
            with self.assertRaisesRegex(WecomError, 'ACK timed out'):
                await adapter.send_message(OutboundMessage(
                    chat_id='u1', text='x', source_message_id='m1'))
        self.assertEqual(adapter.status(), 'disconnected')
        self.assertTrue(ws.closed)

    async def test_replies_to_same_request_wait_for_previous_ack(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        ws.auto_reply_ack = False
        first = asyncio.create_task(adapter.send_message(OutboundMessage(
            chat_id='u1', text='first', source_message_id='m1')))
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        ws.reply_sent.clear()
        second = asyncio.create_task(adapter.send_message(OutboundMessage(
            chat_id='u1', text='second', source_message_id='m1')))
        await asyncio.sleep(0)
        self.assertEqual(len(ws.frames_with(wecom.CMD_RESPOND)), 1)
        ws.push({'headers': {'req_id': 'r1'}, 'errcode': 0})
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        self.assertEqual(len(ws.frames_with(wecom.CMD_RESPOND)), 2)
        ws.push({'headers': {'req_id': 'r1'}, 'errcode': 0})
        await asyncio.gather(first, second)

    async def test_replies_to_different_requests_do_not_block_each_other(self):
        adapter, ws = await self._running_adapter()
        for msg_id, req_id in [('m1', 'r1'), ('m2', 'r2')]:
            frame = _msg_frame(msg_id, req_id)
            await adapter._process_message(frame, frame['body'])
        ws.auto_reply_ack = False
        first = asyncio.create_task(adapter.send_message(OutboundMessage(
            chat_id='u1', text='first', source_message_id='m1')))
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        ws.reply_sent.clear()
        second = asyncio.create_task(adapter.send_message(OutboundMessage(
            chat_id='u1', text='second', source_message_id='m2')))
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        self.assertEqual(len(ws.frames_with(wecom.CMD_RESPOND)), 2)
        ws.push({'headers': {'req_id': 'r1'}, 'errcode': 0})
        ws.push({'headers': {'req_id': 'r2'}, 'errcode': 0})
        await asyncio.gather(first, second)

    async def test_stop_unblocks_a_reply_waiting_for_ack(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        ws.auto_reply_ack = False
        pending = asyncio.create_task(adapter.send_message(OutboundMessage(
            chat_id='u1', text='x', source_message_id='m1')))
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        await adapter.stop()
        with self.assertRaisesRegex(WecomError, 'closed before the reply ACK'):
            await pending

    async def test_cancelled_reply_closes_connection_before_retry(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        ws.auto_reply_ack = False
        pending = asyncio.create_task(adapter.send_message(OutboundMessage(
            chat_id='u1', text='x', source_message_id='m1')))
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(adapter.status(), 'disconnected')
        self.assertTrue(ws.closed)

    async def test_expired_reply_window_raises(self):
        adapter, _ = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        adapter._msgid_to_req['m1'] = ('r1', 0.0)
        with self.assertRaises(WecomError) as ctx:
            await adapter.send_message(
                OutboundMessage(chat_id='u1', text='x', source_message_id='m1'))
        self.assertIn('24h', str(ctx.exception))


class WecomFrameworkIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """7-8、12：Canvas 接线、入站发布、出站端到端（含现有平台不回归）。"""

    def setUp(self):
        self._db = os.path.join(tempfile.mkdtemp(), 'wecom_int.db')
        self._patch = mock.patch.object(config, 'DB_PATH', self._db)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        config.main['channel_configs'] = [{
            'id': 'wecom_test', 'platform': 'wecom', 'enabled': True,
            'bot_to_bot_enabled': False, 'trusted_bots': [], 'config': {},
        }]
        config.main['channel_last_context'] = {}
        config.main['channel_message_contexts'] = {}
        self.manager = ChannelManager()
        self.adapter = DummyAdapter()
        self.manager._adapters['wecom_test'] = self.adapter

    def _wire_canvas(self):
        config.main['canvas_layout'] = {'cards': [{
            'mcpId': 'channel', 'toolName': 'channel_request', 'id': 'input_1'}]}
        config.main['tool_config:channel:channel_request:input_1'] = {
            'channel_id': 'wecom_test'}

    async def test_inbound_publishes_channel_request_payload(self):
        self._wire_canvas()
        published = mock.AsyncMock()
        pushed = mock.AsyncMock()
        inspection = types.ModuleType('api.inspection')
        inspection.publish_to_topic = published
        motus_stream = types.ModuleType('api.motus_stream')
        motus_stream.push_event = pushed

        msg = InboundMessage(
            platform='wecom', channel_id='wecom_test', user_id='u1',
            chat_id='u1', display_name='u1', text='挥个手',
            message_id='m1', sender_type='user', chat_type='p2p',
        )
        with mock.patch.dict(sys.modules, {
            'api.inspection': inspection, 'api.motus_stream': motus_stream,
        }), mock.patch('channel.manager.acl.get_user', side_effect=[
                    None,
                    {'role': 'viewer', 'platform': 'wecom', 'user_id': 'u1'},
                ]), \
                mock.patch('channel.manager.acl.upsert_user', return_value={
                    'role': 'viewer', 'platform': 'wecom', 'user_id': 'u1'}), \
                mock.patch('channel.manager.acl.check_permission',
                           return_value=(True, '')):
            await self.manager._on_inbound_message(msg)

        published.assert_awaited_once()
        topic = published.await_args.args[0]
        payload = json.loads(published.await_args.args[1])
        self.assertEqual(topic, '/channel/request/wecom_test')
        self.assertEqual(payload['platform'], 'wecom')
        self.assertEqual(payload['message_id'], 'm1')
        self.assertEqual(payload['user_role'], 'viewer')

    async def test_inbound_discarded_when_no_channel_request_card(self):
        config.main['canvas_layout'] = {'cards': []}
        published = mock.AsyncMock()
        inspection = types.ModuleType('api.inspection')
        inspection.publish_to_topic = published
        motus_stream = types.ModuleType('api.motus_stream')
        motus_stream.push_event = mock.AsyncMock()
        msg = InboundMessage(
            platform='wecom', channel_id='wecom_test', user_id='u1',
            chat_id='u1', display_name='u1', text='hi', message_id='m2',
            sender_type='user', chat_type='p2p',
        )
        with mock.patch.dict(sys.modules, {
            'api.inspection': inspection, 'api.motus_stream': motus_stream,
        }):
            await self.manager._on_inbound_message(msg)
        published.assert_not_awaited()

    async def test_blocked_sender_does_not_prepare_image(self):
        self._wire_canvas()
        prepare = mock.AsyncMock()
        msg = InboundMessage(
            platform='wecom', channel_id='wecom_test', user_id='u1',
            chat_id='u1', display_name='u1', text='[图片]', message_id='m1', prepare=prepare,
        )
        with mock.patch('channel.manager.acl.get_user', return_value={'role': 'blocked'}), \
                mock.patch('channel.manager.acl.check_permission', return_value=(False, 'blocked')):
            await self.manager._on_inbound_message(msg)
        prepare.assert_not_awaited()

    async def test_send_to_channel_forwards_source_message_id(self):
        """方案 B：触发消息的 message_id 必须透传到 OutboundMessage。"""
        self.manager._set_last_context(
            'wecom_test', 'u1', 'u1', message_id='m1',
            sender_type='user', chat_type='p2p',
        )
        result = await self.manager.send_to_channel(
            'wecom_test', text='已挥手', source_message_id='m1')
        self.assertIn('Reply sent', result)
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertEqual(self.adapter.sent[0].source_message_id, 'm1')
        self.assertEqual(self.adapter.sent[0].chat_id, 'u1')

    async def test_manager_reports_platform_rejection(self):
        ws = FakeWS([_subscribe_ack()])
        adapter = _make_adapter()
        with mock.patch.object(adapter, '_connect',
                               new=mock.AsyncMock(return_value=(FakeSession(), ws))):
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        self.manager._adapters['wecom_test'] = adapter
        self.manager._push_error = mock.AsyncMock()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        self.manager._set_last_context('wecom_test', 'u1', 'u1', message_id='m1')
        ws.reply_errcode = 40001
        result = await self.manager.send_to_channel(
            'wecom_test', text='reply', source_message_id='m1')
        self.assertIn('Error sending reply', result)
        self.assertIn('40001', result)
        self.assertNotIn('Reply sent', result)

    async def test_manager_approval_notice_replies_to_source_message(self):
        self._wire_canvas()
        completed = asyncio.Event()
        async def on_message(msg):
            try:
                await self.manager._on_inbound_message(msg)
            finally:
                completed.set()

        ws = FakeWS([_subscribe_ack()])
        adapter = _make_adapter(on_message)
        with mock.patch.object(adapter, '_connect',
                               new=mock.AsyncMock(return_value=(FakeSession(), ws))):
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        self.manager._adapters['wecom_test'] = adapter
        with mock.patch('channel.manager.acl.get_user', return_value=None), \
                mock.patch.object(self.manager, '_get_channel_settings',
                                  return_value={'auto_approve': False}):
            ws.push(_msg_frame('m1', 'r1'))
            await asyncio.wait_for(completed.wait(), timeout=1)
        replies = ws.frames_with(wecom.CMD_RESPOND)
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]['headers']['req_id'], 'r1')

    async def test_legacy_adapters_unaffected_by_new_field(self):
        """新字段带默认值，旧适配器按位置/关键字构造仍可用。"""
        msg = OutboundMessage('chat', 'text')
        self.assertEqual(msg.source_message_id, '')
        self.assertEqual(msg.chat_id, 'chat')
        self.assertEqual(msg.text, 'text')


class WecomLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """13-16：被接管停重连、三态自检、发送失败、stop 幂等。"""

    async def _running_adapter(self):
        ws = FakeWS([_subscribe_ack()])
        adapter = _make_adapter()
        with mock.patch.object(adapter, '_connect',
                               new=mock.AsyncMock(return_value=(FakeSession(), ws))):
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        return adapter, ws

    async def test_disconnected_event_stops_reconnect(self):
        adapter, _ = await self._running_adapter()
        frame = {'cmd': 'aibot_event_callback', 'headers': {'req_id': 'r9'},
                 'body': {'event_type': wecom.EVENT_DISCONNECTED}}
        await adapter._handle_frame(frame)
        self.assertTrue(adapter._taken_over)
        self.assertFalse(adapter.is_running)
        self.assertEqual(adapter.status(), 'disconnected')
        ok, reason = await adapter.health_check()
        self.assertFalse(ok)
        self.assertIn('taken over', reason)

    async def test_watchdog_does_not_restart_after_takeover(self):
        adapter, _ = await self._running_adapter()
        await adapter._handle_frame({
            'cmd': wecom.CMD_EVENT_CALLBACK,
            'body': {'event': {'eventtype': wecom.EVENT_DISCONNECTED}},
        })
        manager = ChannelManager()
        manager._adapters[adapter.channel_id] = adapter
        manager._push_error = mock.AsyncMock()
        manager.restart_adapter = mock.AsyncMock()
        with mock.patch('channel.manager._update_status'):
            await manager._check_one(adapter.channel_id)
        manager.restart_adapter.assert_not_awaited()

    async def test_watchdog_still_restarts_after_normal_disconnect(self):
        adapter = _make_adapter()
        manager = ChannelManager()
        manager._adapters[adapter.channel_id] = adapter
        manager._push_error = mock.AsyncMock()
        manager.restart_adapter = mock.AsyncMock()
        with mock.patch('channel.manager._update_status'):
            await manager._check_one(adapter.channel_id)
        manager.restart_adapter.assert_awaited_once()

    async def test_watchdog_still_restarts_existing_adapter_types(self):
        adapter = DummyAdapter()
        adapter._running = False
        manager = ChannelManager()
        manager._adapters[adapter.channel_id] = adapter
        manager._push_error = mock.AsyncMock()
        manager.restart_adapter = mock.AsyncMock()
        with mock.patch('channel.manager._update_status'):
            await manager._check_one(adapter.channel_id)
        manager.restart_adapter.assert_awaited_once()

    async def test_health_check_states(self):
        adapter, _ = await self._running_adapter()
        ok, _ = await adapter.health_check()
        self.assertTrue(ok)
        self.assertEqual(adapter.status(), 'connected')

        adapter._last_frame_at = 0.0
        adapter._last_frame_at = wecom.time.time() - wecom.STALE_AFTER - 1
        ok, reason = await adapter.health_check()
        self.assertFalse(ok)
        self.assertIn('heartbeat stalled', reason)
        self.assertEqual(adapter.status(), 'degraded')

        adapter._ws._closed = True
        self.assertEqual(adapter.status(), 'disconnected')

    async def test_send_message_failure_raises(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        ws.fail_send = True
        with self.assertRaises(RuntimeError):
            await adapter.send_message(
                OutboundMessage(chat_id='u1', text='x', source_message_id='m1'))

    async def test_send_message_files_rejected(self):
        adapter, _ = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        with self.assertRaises(WecomError) as ctx:
            await adapter.send_message(OutboundMessage(
                chat_id='u1', text='x', source_message_id='m1',
                files=[Attachment(path='/tmp/a.png')]))
        self.assertIn('image attachments only', str(ctx.exception))

    async def test_stop_is_idempotent(self):
        adapter, ws = await self._running_adapter()
        await adapter.stop()
        await adapter.stop()
        self.assertFalse(adapter.is_running)
        self.assertTrue(ws.closed)
        self.assertEqual(adapter.status(), 'disconnected')


if __name__ == '__main__':
    unittest.main()
