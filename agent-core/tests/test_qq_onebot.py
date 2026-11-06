"""OneBot v11 mock integration (no real QQ credentials or network)."""
import asyncio
import pathlib
import sys
import unittest

from aiohttp import web

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from channel.adapter import OutboundMessage  # noqa: E402
from channel.adapters.qq_onebot import QQOneBotAdapter, validate_qq_config  # noqa: E402


class MockOneBot:
    def __init__(self, token='mock-secret'):
        self.token = token
        self.clients = set()
        self.actions = []
        self.fail_send = False
        self.suppress_send = False
        self.runner = None
        self.url = ''
        self.tasks = set()

    async def start(self):
        app = web.Application()
        app.router.add_get('/', self.websocket)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.url = f'ws://127.0.0.1:{port}/'

    async def stop(self):
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        for ws in list(self.clients):
            await ws.close()
        if self.runner:
            await self.runner.cleanup()

    async def websocket(self, request):
        if request.headers.get('Authorization') != f'Bearer {self.token}':
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            async for item in ws:
                if item.type != web.WSMsgType.TEXT:
                    continue
                payload = item.json()
                self.actions.append(payload)
                action = payload.get('action')
                if action == 'get_login_info':
                    await ws.send_json({'echo': payload['echo'], 'status': 'ok',
                                        'retcode': 0, 'data': {'user_id': 12345}})
                elif action == 'send_private_msg':
                    if self.suppress_send:
                        continue
                    async def answer(payload=payload, ws=ws):
                        if payload['params'].get('user_id') == 666:
                            await asyncio.sleep(0.08)
                        await ws.send_json({
                            'echo': payload['echo'],
                            'status': 'failed' if self.fail_send else 'ok',
                            'retcode': 100 if self.fail_send else 0,
                            'data': {'message_id': 42}})
                    task = asyncio.create_task(answer())
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
        finally:
            self.clients.discard(ws)
        return ws

    async def event(self, **overrides):
        payload = {
            'post_type': 'message', 'message_type': 'private', 'self_id': 12345,
            'user_id': 888, 'message_id': 1, 'sender': {'nickname': 'User'},
            'message': 'hello'}
        payload.update(overrides)
        for ws in list(self.clients):
            await ws.send_json(payload)

    async def raw(self, raw: str):
        for ws in list(self.clients):
            await ws.send_str(raw)


class QQOneBotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = MockOneBot()
        await self.server.start()
        self.messages = []
        async def on_message(message):
            self.messages.append(message)
        self.adapter = QQOneBotAdapter(
            'qq_main', 'qq', {'ws_url': self.server.url,
                             'access_token': self.server.token,
                             'private_only': True, 'request_timeout': 0.2},
            on_message)
        # User-request timeout validation requires >= 1 second.
        self.adapter.config['request_timeout'] = 1

    async def asyncTearDown(self):
        await self.adapter.stop()
        await self.server.stop()

    async def flush(self):
        await asyncio.sleep(0.035)

    async def test_t01_connect_and_health(self):
        await self.adapter.start()
        self.assertEqual(self.adapter.status(), 'connected')
        self.assertEqual(await self.adapter.health_check(), (True, ''))
        self.assertEqual(self.adapter._self_id, '12345')

    async def test_t02_text_and_segments_and_unsupported(self):
        await self.adapter.start()
        await self.server.event(message='   hello   ', message_id=11)
        await self.server.event(message=[
            {'type': 'text', 'data': {'text': '你好'}},
            {'type': 'text', 'data': {'text': '世界'}}], message_id=12)
        await self.server.event(message=[
            {'type': 'text', 'data': {'text': 'delete files'}},
            {'type': 'image', 'data': {'file': 'x'}}], message_id=13)
        await self.server.event(message='[CQ:image,file=x]', message_id=14)
        await self.server.event(message='', message_id=15)
        await self.server.event(message='x' * 2100, message_id=16)
        await self.flush()
        self.assertEqual([x.text for x in self.messages], ['hello', '你好世界'])
        self.assertEqual(self.messages[0].chat_id, '888')
        self.assertEqual(self.messages[0].chat_type, 'p2p')
        self.assertEqual(self.messages[0].platform, 'qq')

    async def test_t03_private_reply(self):
        await self.adapter.start()
        await self.adapter.send_message(OutboundMessage(chat_id='888', text='pong'))
        send = [a for a in self.server.actions if a['action'] == 'send_private_msg']
        self.assertEqual(send[0]['params'], {'user_id': 888, 'message': 'pong'})

    async def test_t04_echo_parallel_routing(self):
        await self.adapter.start()
        await asyncio.gather(
            self.adapter.send_message(OutboundMessage(chat_id='666', text='slow')),
            self.adapter.send_message(OutboundMessage(chat_id='777', text='fast')))
        self.assertEqual(len(self.adapter._pending), 0)
        ids = {a['params']['user_id'] for a in self.server.actions
               if a['action'] == 'send_private_msg'}
        self.assertEqual(ids, {666, 777})

    async def test_t06_duplicate_event(self):
        await self.adapter.start()
        await self.server.event(message_id=99)
        await self.server.event(message_id=99)
        await self.flush()
        self.assertEqual(len(self.messages), 1)

    async def test_identity_required_before_messages_are_delivered(self):
        # A transport-level connection cannot authorize events until the
        # get_login_info response has established this adapter's own QQ ID.
        event = {
            'post_type': 'message', 'message_type': 'private',
            'self_id': 12345, 'user_id': 888, 'message_id': 204,
            'message': 'early frame',
        }
        self.assertIsNone(self.adapter._parse_message(event))
        await self.adapter.start()
        await self.server.event(self_id=54321, message_id=205,
                                message='wrong bot identity')
        await self.server.event(message_id=206, message='verified bot identity')
        await self.flush()
        self.assertEqual([m.text for m in self.messages], ['verified bot identity'])
        await self.adapter.stop()
        self.assertIsNone(self.adapter._parse_message(event))

    async def test_t07_self_and_sent_events(self):
        await self.adapter.start()
        await self.server.event(user_id=12345, message_id=20)
        await self.server.event(post_type='message_sent', message_id=21)
        await self.flush()
        self.assertFalse(self.messages)

    async def test_t08_group_filtered(self):
        await self.adapter.start()
        await self.server.event(message_type='group', message_id=21)
        await self.flush()
        self.assertFalse(self.messages)

    async def test_t09_disconnect_detected_and_restart(self):
        await self.adapter.start()
        for ws in list(self.server.clients):
            await ws.close()
        await self.flush()
        self.assertEqual((await self.adapter.health_check())[0], False)
        await self.adapter.stop()
        await self.adapter.start()
        self.assertEqual((await self.adapter.health_check())[0], True)

    async def test_t10_token_failure(self):
        self.adapter.config['access_token'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'rejected credentials'):
            await self.adapter.start()
        self.assertNotEqual(self.adapter.status(), 'connected')

    async def test_t11_api_failure_and_timeout(self):
        await self.adapter.start()
        self.server.fail_send = True
        with self.assertRaisesRegex(RuntimeError, 'retcode=100'):
            await self.adapter.send_message(OutboundMessage(chat_id='888', text='fail'))
        self.server.fail_send = False
        self.server.suppress_send = True
        with self.assertRaisesRegex(TimeoutError, 'unconfirmed'):
            await self.adapter.send_message(OutboundMessage(chat_id='888', text='timeout'))

    async def test_t12_stop_suppresses_events(self):
        await self.adapter.start()
        await self.adapter.stop()
        await self.server.event(message_id=22)
        await self.flush()
        self.assertFalse(self.messages)

    async def test_t13_multiple_users(self):
        await self.adapter.start()
        await self.server.event(user_id=888, message_id=31, message='one')
        await self.server.event(user_id=999, message_id=32, message='two')
        await self.flush()
        self.assertEqual([(m.chat_id, m.text) for m in self.messages],
                         [('888', 'one'), ('999', 'two')])

    async def test_invalid_json_and_heartbeat(self):
        await self.adapter.start()
        await self.server.raw('{not json')
        await self.server.event(post_type='meta_event', meta_event_type='heartbeat')
        await self.flush()
        self.assertEqual((await self.adapter.health_check())[0], True)

    def test_config_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_qq_config({'ws_url': 'ws://localhost:3001/', 'private_only': False,
                                'access_token': 'placeholder'})
        with self.assertRaises(ValueError):
            validate_qq_config({'ws_url': 'ws://localhost:3001/?access_token=oops',
                                'access_token': 'placeholder'})


if __name__ == '__main__':
    unittest.main()
