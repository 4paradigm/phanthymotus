"""
test_wecom_channel.py — 企业微信长连接 ChannelAdapter 一期测试。

全部离线：使用官方 SDK 公共接口的 FakeClient 替身，不连接真实网络。
这些测试验证适配器行为；SDK 的认证帧、心跳与 ACK 分发由 SDK 自身负责。

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
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test_wecom.db'))


import config  # noqa: E402
from channel.adapter import Attachment, InboundMessage, OutboundMessage  # noqa: E402
from channel.adapters import wecom  # noqa: E402
from channel.adapters.wecom import WecomAdapter, WecomError  # noqa: E402
from channel.manager import (  # noqa: E402
    ChannelManager, add_channel_config, supports_bot_to_bot, uses_trusted_bots,
)


class FakeClient:
    """SDK public-interface double: events, connection state and stream replies."""

    def __init__(self, *, authenticate=True, auth_error=None):
        self.sent = []
        self.handlers = {}
        self.is_connected = False
        self.authenticate = authenticate
        self.auth_error = auth_error
        self.fail_send = False
        self.auto_reply_ack = True
        self.reply_errcode = 0
        self.reply_sent = asyncio.Event()
        self.connected = asyncio.Event()
        self.pending = {}

    @property
    def closed(self):
        return not self.is_connected

    def on(self, event, callback):
        self.handlers[event] = callback

    def emit(self, event, *args):
        callback = self.handlers.get(event)
        if callback:
            callback(*args)

    async def connect(self):
        self.is_connected = True
        self.connected.set()
        if self.auth_error is not None:
            self.emit('error', self.auth_error)
        elif self.authenticate:
            self.emit('authenticated')
        return self

    def disconnect(self):
        self.is_connected = False
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError('Connection manually closed'))

    async def reply_stream(self, frame, stream_id, content, finish=False, msg_item=None):
        if self.fail_send:
            raise RuntimeError('simulated SDK send failure')
        req_id = frame['headers']['req_id']
        stream = {'id': stream_id, 'content': content, 'finish': finish}
        if finish and msg_item:
            stream['msg_item'] = msg_item
        self.sent.append({'cmd': 'aibot_respond_msg', 'headers': frame['headers'],
                          'body': {'msgtype': 'stream', 'stream': stream}})
        self.reply_sent.set()
        if self.auto_reply_ack:
            if self.reply_errcode:
                raise RuntimeError(f'Reply ack error: errcode={self.reply_errcode}, errmsg=rejected')
            return {'headers': frame['headers'], 'errcode': 0}
        future = asyncio.get_running_loop().create_future()
        self.pending[req_id] = future
        try:
            return await future
        finally:
            self.pending.pop(req_id, None)

    def push(self, frame):
        if 'errcode' in frame:
            future = self.pending.get(frame['headers']['req_id'])
            if future is not None and not future.done():
                if frame['errcode']:
                    future.set_exception(RuntimeError(
                        f"Reply ack error: errcode={frame['errcode']}, errmsg=rejected"))
                else:
                    future.set_result(frame)
            return
        self.emit('event' if frame['cmd'] == 'aibot_event_callback' else 'message', frame)

    def frames_with(self, cmd):
        return [frame for frame in self.sent if frame['cmd'] == cmd]


async def _start_fake_sdk(adapter, client):
    with mock.patch.object(adapter, '_create_client', return_value=client), \
            mock.patch.object(wecom.aiohttp, 'ClientSession', return_value=FakeSession()):
        await adapter.start()


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
    """Credential preflight uses SDK events, never raw socket state alone."""

    async def test_sdk_options_and_http_proxy(self):
        adapter = _make_adapter()
        client = FakeClient()
        with mock.patch('aibot.WSClient', return_value=client) as factory, \
                mock.patch.object(wecom.aiohttp, 'ClientSession', return_value=FakeSession()) as session:
            await adapter.start()
        self.addAsyncCleanup(adapter.stop)
        options = factory.call_args.args[0]
        self.assertEqual(options.bot_id, 'b')
        self.assertEqual(options.secret, 's')
        self.assertEqual(options.max_reconnect_attempts, 0)
        self.assertIs(options.logger, adapter._sdk_logger)
        session.assert_called_once_with(trust_env=True)
        self.assertEqual(adapter.status(), 'connected')

    async def test_start_raises_when_credentials_missing(self):
        adapter = _make_adapter(config={})
        with self.assertRaisesRegex(WecomError, 'bot_id and secret'):
            await adapter.start()
        self.assertFalse(adapter.is_running)

    async def test_start_raises_on_sdk_authentication_error(self):
        adapter = _make_adapter(config={'bot_id': 'b', 'secret': 'PRIVATE_SECRET'})
        client = FakeClient(auth_error=Exception(
            'Authentication failed: PRIVATE_SECRET (code: 40001)'))
        with mock.patch('builtins.print') as printed:
            with self.assertRaisesRegex(WecomError, 'errcode=40001') as ctx:
                await _start_fake_sdk(adapter, client)
        self.assertNotIn('PRIVATE_SECRET', str(ctx.exception))
        self.assertNotIn('PRIVATE_SECRET', str(printed.call_args_list))
        self.assertTrue(client.closed)
        self.assertIsNone(adapter._session)

    async def test_start_waits_for_authentication(self):
        adapter = _make_adapter()
        client = FakeClient(authenticate=False)
        task = asyncio.create_task(_start_fake_sdk(adapter, client))
        await client.connected.wait()
        self.assertFalse(task.done())
        self.assertEqual(adapter.status(), 'disconnected')
        client.emit('authenticated')
        await task
        self.addAsyncCleanup(adapter.stop)
        self.assertEqual(adapter.status(), 'connected')

    async def test_start_authentication_timeout_closes_client(self):
        adapter = _make_adapter()
        client = FakeClient(authenticate=False)
        with mock.patch.object(wecom, 'SUBSCRIBE_TIMEOUT', 0.05):
            with self.assertRaisesRegex(WecomError, 'timed out'):
                await _start_fake_sdk(adapter, client)
        self.assertTrue(client.closed)
        self.assertIsNone(adapter._session)

    async def test_cancelled_start_closes_client(self):
        adapter = _make_adapter()
        client = FakeClient(authenticate=False)
        task = asyncio.create_task(_start_fake_sdk(adapter, client))
        await client.connected.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(client.closed)
        self.assertIsNone(adapter._session)

    def test_sdk_logs_are_sampled_without_payloads(self):
        logger = wecom._SdkLogger()
        with mock.patch('builtins.print') as printed:
            for _ in range(100):
                logger.debug('PRIVATE_IMAGE_KEY')
                logger.info('PRIVATE_MESSAGE')
                logger.warn('PRIVATE_SIGNED_URL')
                logger.error('PRIVATE_SECRET')
        self.assertEqual(printed.call_count, 4)
        self.assertNotIn('PRIVATE_', str(printed.call_args_list))


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

    async def test_image_failures_are_sampled_without_logging_secrets(self):
        adapter = await self._adapter()
        adapter._send_progress = mock.AsyncMock(side_effect=WecomError('PRIVATE_VALUE'))
        adapter._download_image = mock.AsyncMock(side_effect=WecomError('PRIVATE_VALUE'))
        with mock.patch('builtins.print') as printed:
            for _ in range(10):
                msg = InboundMessage('wecom', 'wecom_test', 'u1', 'u1', 'u1', '[图片]', 'm1')
                await adapter._prepare_images(msg, [{}] * 10)
                self.assertIn('图片处理失败或超限：10 张', msg.text)
            lines = [call.args[0] for call in printed.call_args_list]
            self.assertEqual(len(lines), 3)  # first progress failure; first and 100th image failures
            self.assertIn('count=1', lines[1])
            self.assertIn('count=100', lines[2])
            self.assertNotIn('PRIVATE_VALUE', '\n'.join(lines))

    def test_decrypt_image_with_32_byte_padding(self):
        key = bytes(range(32))
        original = b'\x89PNG\r\n\x1a\n' + b'a' * 24
        padding = 32 - len(original) % 32
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
        ciphertext = encryptor.update(original + bytes([padding]) * padding) + encryptor.finalize()
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
            'cmd': 'aibot_event_callback',
            'headers': {'req_id': 'event-req'},
            'body': {'msgid': 'event-id', 'msgtype': 'event',
                     'from': {'userid': 'u1'}, 'event': {'eventtype': 'enter_chat'}},
        }
        adapter._handle_event(event)
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

        ws = FakeClient()
        adapter = _make_adapter(on_message)
        await _start_fake_sdk(adapter, ws)
        self.addAsyncCleanup(adapter.stop)
        ws.push(_msg_frame())
        await asyncio.wait_for(delivered.wait(), timeout=1)
        self.assertEqual(errors, [])
        self.assertEqual(len(ws.frames_with('aibot_respond_msg')), 1)


class WecomReqIdTest(unittest.IsolatedAsyncioTestCase):
    """9-11：方案 B 的 req_id 精确映射。"""

    def setUp(self):
        self.received: list[InboundMessage] = []

    async def _running_adapter(self):
        async def _on_msg(msg):
            self.received.append(msg)
        ws = FakeClient()
        adapter = _make_adapter(_on_msg)
        await _start_fake_sdk(adapter, ws)
        self.addAsyncCleanup(adapter.stop)
        return adapter, ws

    async def test_reply_uses_exact_req_id(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])

        msg = OutboundMessage(chat_id='u1', text='收到', source_message_id='m1')
        await adapter.send_message(msg)
        replies = ws.frames_with('aibot_respond_msg')
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
        reply = ws.frames_with('aibot_respond_msg')[0]
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
        first, final = ws.frames_with('aibot_respond_msg')
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
        replies = ws.frames_with('aibot_respond_msg')
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
        self.assertEqual(ws.frames_with('aibot_respond_msg'), [])

    async def test_reply_without_source_id_does_not_guess_current_chat(self):
        adapter, ws = await self._running_adapter()
        frame = _msg_frame('m1', 'r1')
        await adapter._process_message(frame, frame['body'])
        with self.assertRaisesRegex(WecomError, 'No reply context'):
            await adapter.send_message(OutboundMessage(chat_id='u1', text='Pending approval.'))
        self.assertEqual(ws.frames_with('aibot_respond_msg'), [])

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
        self.assertEqual(len(ws.frames_with('aibot_respond_msg')), 1)
        ws.push({'headers': {'req_id': 'r1'}, 'errcode': 0})
        await asyncio.wait_for(ws.reply_sent.wait(), timeout=1)
        self.assertEqual(len(ws.frames_with('aibot_respond_msg')), 2)
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
        self.assertEqual(len(ws.frames_with('aibot_respond_msg')), 2)
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
        ws = FakeClient()
        adapter = _make_adapter()
        await _start_fake_sdk(adapter, ws)
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

        ws = FakeClient()
        adapter = _make_adapter(on_message)
        await _start_fake_sdk(adapter, ws)
        self.addAsyncCleanup(adapter.stop)
        self.manager._adapters['wecom_test'] = adapter
        with mock.patch('channel.manager.acl.get_user', return_value=None), \
                mock.patch.object(self.manager, '_get_channel_settings',
                                  return_value={'auto_approve': False}):
            ws.push(_msg_frame('m1', 'r1'))
            await asyncio.wait_for(completed.wait(), timeout=1)
        replies = ws.frames_with('aibot_respond_msg')
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]['headers']['req_id'], 'r1')

    async def test_legacy_adapters_unaffected_by_new_field(self):
        """新字段带默认值，旧适配器按位置/关键字构造仍可用。"""
        msg = OutboundMessage('chat', 'text')
        self.assertEqual(msg.source_message_id, '')
        self.assertEqual(msg.chat_id, 'chat')
        self.assertEqual(msg.text, 'text')


class WecomLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """被接管停重连、SDK 连接状态、发送失败及 stop 幂等。"""

    async def _running_adapter(self):
        ws = FakeClient()
        adapter = _make_adapter()
        await _start_fake_sdk(adapter, ws)
        self.addAsyncCleanup(adapter.stop)
        return adapter, ws

    async def test_disconnected_event_stops_reconnect(self):
        adapter, _ = await self._running_adapter()
        frame = {'cmd': 'aibot_event_callback', 'headers': {'req_id': 'r9'},
                 'body': {'msgtype': 'event', 'event': {'eventtype': 'disconnected_event'}}}
        adapter._handle_event(frame)
        self.assertTrue(adapter._taken_over)
        self.assertFalse(adapter.is_running)
        self.assertEqual(adapter.status(), 'disconnected')
        ok, reason = await adapter.health_check()
        self.assertFalse(ok)
        self.assertIn('taken over', reason)

    async def test_watchdog_does_not_restart_after_takeover(self):
        adapter, _ = await self._running_adapter()
        adapter._handle_event({
            'cmd': 'aibot_event_callback',
            'body': {'msgtype': 'event', 'event': {'eventtype': 'disconnected_event'}},
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

        adapter._authenticated = False
        ok, _ = await adapter.health_check()
        self.assertFalse(ok)
        self.assertEqual(adapter.status(), 'disconnected')

        adapter._authenticated = True
        adapter._client.is_connected = False
        self.assertEqual(adapter.status(), 'disconnected')

    async def test_sdk_disconnect_is_visible_and_old_callbacks_are_ignored(self):
        adapter, client = await self._running_adapter()
        client.emit('disconnected', 'PRIVATE_REASON')
        self.assertFalse((await adapter.health_check())[0])
        self.assertTrue(adapter.should_auto_reconnect())
        self.assertNotIn('PRIVATE_REASON', adapter._last_error)
        await adapter.stop()
        client.emit('authenticated')
        self.assertEqual(adapter.status(), 'disconnected')
        self.assertFalse(adapter.is_running)

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
