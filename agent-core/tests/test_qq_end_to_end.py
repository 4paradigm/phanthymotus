"""Full QQ transport → ACL → Canvas topic → routed reply integration test.

OneBot server is mocked; no credentials or QQ accounts are used by this test.
"""
import asyncio
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from channel.adapters.qq_onebot import QQOneBotAdapter
from channel.manager import ChannelManager
from test_qq_onebot import MockOneBot


class QQEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_message_replies_to_exact_origin(self):
        server = MockOneBot()
        await server.start()
        storage = {
            'canvas_layout': {
                'cards': [
                    {'id': 'in', 'mcpId': 'channel',
                     'toolName': 'channel_request'},
                    {'id': 'out', 'mcpId': 'channel',
                     'toolName': 'channel_reply'},
                ],
            },
            'tool_config:channel:channel_request:in': {'channel_id': 'qq_main'},
            'tool_config:channel:channel_reply:out': {'channel_id': 'qq_main'},
            'channel_configs': [{'id': 'qq_main', 'platform': 'qq',
                                 'enabled': True, 'config': {}}],
        }
        manager = ChannelManager()
        responses = []

        async def publish(topic, raw):
            data = json.loads(raw)
            self.assertEqual(topic, '/channel/request/qq_main')
            self.assertEqual(data['user_role'], 'viewer')
            reply = await manager.send_to_channel(
                'qq_main', text='pong for ' + data['text'],
                source_message_id=data['message_id'],
            )
            responses.append(reply)

        adapter = QQOneBotAdapter(
            'qq_main', 'qq', {'ws_url': server.url, 'private_only': True,
                             'access_token': server.token}, manager._on_inbound_message)
        manager._adapters['qq_main'] = adapter

        try:
            with (mock.patch('channel.manager.config.main', storage),
                  mock.patch('channel.acl.get_user', return_value={
                      'role': 'viewer', 'tool_filter': '*'}),
                  mock.patch('channel.acl.check_permission', return_value=(True, 'ok')),
                  mock.patch('api.inspection.publish_to_topic',
                             new_callable=mock.AsyncMock, side_effect=publish),
                  mock.patch('api.motus_stream.push_event',
                             new_callable=mock.AsyncMock)):
                await adapter.start()
                await server.event(user_id=888, message_id=177, message='one')
                await server.event(user_id=999, message_id=178, message='two')
                await asyncio.sleep(0.13)
                self.assertEqual(len(responses), 2)
                sent = [a['params'] for a in server.actions
                        if a['action'] == 'send_private_msg']
                self.assertEqual(
                    {(a['user_id'], a['message']) for a in sent},
                    {(888, 'pong for one'), (999, 'pong for two')},
                )
                self.assertEqual(len(storage['channel_message_contexts']['qq_main']), 2)
        finally:
            await adapter.stop()
            await server.stop()


if __name__ == '__main__':
    unittest.main()
