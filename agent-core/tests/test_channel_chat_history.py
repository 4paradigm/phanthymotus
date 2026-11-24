"""Per-(channel_id, chat_id) recap stays isolated from other chats and other
channels — it must not depend on the shared global turn history in event/llm.py."""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import chat_history
import collector  # noqa: E402
import config  # noqa: E402
from channel.manager import ChannelManager  # noqa: E402


class ChatHistoryIsolationTest(unittest.TestCase):
    def setUp(self):
        config.main['channel_chat_history'] = {}
        self.manager = ChannelManager()

    def test_different_chats_do_not_see_each_others_recap(self):
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', '帮我查电量',
                                           user_label='Alice')
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'assistant', '电量 80%')
        self.manager._record_chat_exchange('feishu_r1', 'oc_b', 'user', '我刚才说的呢',
                                           user_label='Bob')

        a_history = self.manager.get_chat_history('feishu_r1', 'oc_a')
        b_history = self.manager.get_chat_history('feishu_r1', 'oc_b')

        self.assertEqual([e['text'] for e in a_history], ['帮我查电量', '电量 80%'])
        self.assertEqual([e['text'] for e in b_history], ['我刚才说的呢'])
        self.assertNotIn('电量', json.dumps(b_history, ensure_ascii=False))

    def test_different_channels_with_same_chat_id_do_not_collide(self):
        self.manager._record_chat_exchange('feishu_r1', 'oc_shared', 'user', 'r1 消息')
        self.manager._record_chat_exchange('feishu_g1', 'oc_shared', 'user', 'g1 消息')

        self.assertEqual(
            [e['text'] for e in self.manager.get_chat_history('feishu_r1', 'oc_shared')],
            ['r1 消息'],
        )
        self.assertEqual(
            [e['text'] for e in self.manager.get_chat_history('feishu_g1', 'oc_shared')],
            ['g1 消息'],
        )

    def test_rolling_window_keeps_only_the_most_recent_entries(self):
        for i in range(20):
            self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', f'msg{i}')

        history = self.manager.get_chat_history('feishu_r1', 'oc_a', limit=100)
        self.assertLessEqual(len(history), 12)
        self.assertEqual(history[-1]['text'], 'msg19')
        self.assertNotIn('msg0', [e['text'] for e in history])

    def test_get_chat_history_respects_limit_and_unknown_chat_returns_empty(self):
        for i in range(5):
            self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', f'msg{i}')

        self.assertEqual(len(self.manager.get_chat_history('feishu_r1', 'oc_a', limit=2)), 2)
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_unknown'), [])
        self.assertEqual(self.manager.get_chat_history('feishu_unknown', 'oc_a'), [])

    def test_empty_text_is_not_recorded(self):
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', '')
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a'), [])

    def test_build_trigger_injects_scoped_chat_history_block(self):
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', '帮我查电量',
                                           user_label='Alice')
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'assistant', '电量 80%')
        self.manager._record_chat_exchange('feishu_r1', 'oc_b', 'user', '不相关的话',
                                           user_label='Bob')

        import channel.manager as manager_mod
        original = manager_mod.manager
        manager_mod.manager = self.manager
        try:
            trigger = collector._build_trigger([{
                'source': 'dds:/channel/request/feishu_r1',
                'text': json.dumps({
                    'channel_id': 'feishu_r1', 'chat_id': 'oc_a',
                    'sender_type': 'user', 'message_id': 'om_new',
                    'text': '我刚才说的怎么样了',
                }, ensure_ascii=False),
                'ts': 1,
            }], urgent=True)
        finally:
            manager_mod.manager = original

        self.assertIn('<chat_history channel="feishu_r1" chat_id="oc_a">', trigger['text'])
        self.assertIn('帮我查电量', trigger['text'])
        self.assertIn('电量 80%', trigger['text'])
        self.assertNotIn('不相关的话', trigger['text'])


class InFlightReplyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db = patch.object(config, 'DB_PATH', os.path.join(directory.name, 'test.db'))
        db.start()
        self.addCleanup(db.stop)
        self.addCleanup(config._get_conn().close)
        self.addCleanup(setattr, chat_history, 'last_clear_time', chat_history.last_clear_time)

    async def test_reply_sent_across_clear_does_not_restore_channel_history(self):
        manager = ChannelManager()
        manager._set_last_context('channel', 'chat', 'user', message_id='msg')
        started, resume = asyncio.Event(), asyncio.Event()

        async def delayed_send(message):
            started.set()
            await resume.wait()

        adapter = Mock(send_message=AsyncMock(side_effect=delayed_send))
        manager._adapters['channel'] = adapter
        pending = asyncio.create_task(manager.send_to_channel(
            'channel', text='old reply', source_message_id='msg'))
        try:
            await asyncio.wait_for(started.wait(), 2)
            chat_history.clear_all()
        finally:
            resume.set()
            result = await pending

        self.assertIn('Reply sent to channel', result)
        adapter.send_message.assert_awaited_once()
        self.assertEqual(adapter.send_message.await_args.args[0].text, 'old reply')
        self.assertEqual(manager.get_chat_history('channel', 'chat'), [])
        manager._set_last_context('channel', 'chat', 'user', message_id='new-msg')
        await manager.send_to_channel('channel', text='new reply', source_message_id='new-msg')
        self.assertEqual([item['text'] for item in manager.get_chat_history('channel', 'chat')],
                         ['new reply'])


if __name__ == '__main__':
    unittest.main()
