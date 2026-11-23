"""Per-(channel_id, chat_id) recap stays isolated from other chats and other
channels — it must not depend on the shared global turn history in event/llm.py."""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import collector  # noqa: E402
import config  # noqa: E402
from channel.adapter import InboundMessage  # noqa: E402
from channel.manager import ChannelManager  # noqa: E402


class ChatHistoryIsolationTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_clear_command_requires_operator_and_only_resets_the_sender(self):
        config.main['channel_chat_history'] = {
            'feishu_r1': {
                'oc_a': [{'role': 'user', 'text': '旧版群聊摘要'}],
            },
        }
        for channel_id, chat_id, user_id, text in [
            ('feishu_r1', 'oc_a', 'ou_a', '自己的消息'),
            ('feishu_r1', 'oc_a', 'ou_b', '同群的另一用户'),
            ('feishu_r1', 'oc_b', 'ou_a', '另一会话'),
            ('feishu_g1', 'oc_a', 'ou_a', '另一渠道'),
        ]:
            self.manager._record_chat_exchange(channel_id, chat_id, 'user', text, user_id=user_id)
        self.manager.active_input_channels.add('feishu_r1')
        adapter = mock.Mock(send_message=mock.AsyncMock())
        self.manager._adapters['feishu_r1'] = adapter
        msg = InboundMessage(
            'feishu', 'feishu_r1', 'ou_a', 'oc_a', 'Alice', '@_user_1 /clear',
            message_id='om_clear', chat_type='group',
            mentions=[{'key': '@_user_1', 'is_self': True}],
        )
        with (
            mock.patch.object(self.manager, 'sync_from_canvas'),
            mock.patch.object(self.manager, '_set_last_context') as set_reply_context,
            mock.patch('channel.acl.get_user') as get_user,
        ):
            for role, generation, reply in [
                ('viewer', 0, '没有清除上下文的权限'),
                ('operator', 1, '已清除你在当前会话的上下文'),
                ('owner', 2, '已清除你在当前会话的上下文'),
            ]:
                with self.subTest(role=role):
                    get_user.return_value = {'role': role}
                    await self.manager._on_inbound_message(msg)
                    confirmation = adapter.send_message.await_args.args[0]
                    self.assertEqual(confirmation.reply_to_message_id, 'om_clear')
                    self.assertIn(reply, confirmation.text)
                    self.assertEqual(ChannelManager().chat_generation('feishu_r1', 'oc_a', 'ou_a'),
                                     generation)
                    self.assertEqual(
                        [e['text'] for e in self.manager.get_chat_history('feishu_r1', 'oc_a',
                                                                          user_id='ou_a')],
                        ['自己的消息'] if role == 'viewer' else [],
                    )

        set_reply_context.assert_not_called()
        self.assertEqual(adapter.send_message.await_count, 3)
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a', user_id='ou_b')[0]['text'],
                         '同群的另一用户')
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_b', user_id='ou_a')[0]['text'],
                         '另一会话')
        self.assertEqual(self.manager.get_chat_history('feishu_g1', 'oc_a', user_id='ou_a')[0]['text'],
                         '另一渠道')
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a')[0]['text'], '旧版群聊摘要')

    def test_group_clear_requires_mentioning_this_bot_and_exact_command(self):
        msg = InboundMessage('feishu', 'feishu_r1', 'ou_a', 'oc_a', 'Alice', '/clear',
                             chat_type='group', mentions=[{'key': '@_user_1', 'is_self': True}])
        self.assertFalse(self.manager._is_feishu_reset(msg))
        msg.text = '@_user_1 清除上下文'
        self.assertTrue(self.manager._is_feishu_reset(msg))
        msg.text = '@_user_1 请帮我清除上下文'
        self.assertFalse(self.manager._is_feishu_reset(msg))
        msg.text = '@_user_1 /clear'
        msg.mentions[0]['is_self'] = False
        self.assertFalse(self.manager._is_feishu_reset(msg))
        msg.mentions[0]['is_self'] = True
        msg.sender_type = 'bot'
        self.assertFalse(self.manager._is_feishu_reset(msg))

    async def test_in_flight_reply_cannot_restore_cleared_history(self):
        self.manager._set_last_context('feishu_r1', 'oc_a', 'ou_a',
                                       message_id='om_old', context_user_id='ou_a')
        self.manager._set_last_context('feishu_r1', 'oc_a', 'ou_b',
                                       message_id='om_other', context_user_id='ou_b')

        async def send(_msg):
            self.manager._clear_chat_history('feishu_r1', 'oc_a', 'ou_a')

        self.manager._adapters['feishu_r1'] = mock.Mock(platform='feishu', send_message=send)
        result = await self.manager.send_to_channel('feishu_r1', '旧回复',
                                                    source_message_id='om_old')
        self.assertIn('Reply sent', result)
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a', user_id='ou_a'), [])
        await self.manager.send_to_channel('feishu_r1', '给另一人的回复', source_message_id='om_other')
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a', user_id='ou_b')[0]['text'],
                         '给另一人的回复')

    async def test_queued_trigger_is_discarded_only_for_the_sender_who_cleared(self):
        def event(user_id, generation, text):
            return {
                'source': 'dds:/channel/request/feishu_r1',
                'text': json.dumps({
                    'platform': 'feishu', 'channel_id': 'feishu_r1', 'chat_id': 'oc_a', 'user_id': user_id,
                    'context_generation': generation, 'text': text,
                }, ensure_ascii=False),
                'ts': 1,
            }

        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', '旧上下文', user_id='ou_a')
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', '另一人的上下文', user_id='ou_b')
        with (
            mock.patch('channel.manager.manager', self.manager),
            mock.patch.object(collector, '_output', asyncio.Queue()),
            mock.patch.object(collector, '_steering_queue', asyncio.Queue()),
            mock.patch.object(collector, '_priority_pending', []),
        ):
            old = event('ou_a', 0, '旧请求')
            other = event('ou_b', 0, '同群另一人的请求')
            await collector._emit_batch([old, other], urgent=True)
            collector._steering_queue.put_nowait(old)
            self.manager._clear_chat_history('feishu_r1', 'oc_a', 'ou_a')
            fresh = event('ou_a', 1, '新请求')
            await collector._emit_batch([old, fresh], urgent=True)

            self.assertEqual(await collector.drain_steering(), [])
            with mock.patch.object(collector, '_current_trigger', {'_feishu_chat': ('feishu_r1', 'oc_a', 'ou_a', 0)}):
                self.assertTrue(collector._defer_channel_steering(fresh, ('feishu_r1', 'oc_a', 'ou_a', 1)))
                self.assertTrue(collector._defer_channel_steering(other, ('feishu_r1', 'oc_a', 'ou_b', 0)))
            other_trigger = await asyncio.wait_for(collector.next_trigger(), timeout=1)
            fresh_trigger = await asyncio.wait_for(collector.next_trigger(), timeout=1)
            self.assertIn('同群另一人的请求', other_trigger['text'])
            self.assertIn('另一人的上下文', other_trigger['text'])
            self.assertNotIn('旧上下文', other_trigger['text'])
            self.assertIn('新请求', fresh_trigger['text'])
            self.assertNotIn('旧上下文', fresh_trigger['text'])
            self.assertNotIn('旧请求', fresh_trigger['text'])
            self.assertNotIn('另一人的上下文', fresh_trigger['text'])
            self.assertTrue(collector._output.empty())

    async def test_first_message_after_clear_does_not_send_old_context_to_model(self):
        from event.llm import Event
        from event.desktop import DesktopTools

        core = Event()
        core._desktop_tools = DesktopTools()
        core._turns = [[{'role': 'user', 'content': '上一轮全局历史'}]]
        core._summary = '上一轮全局摘要'
        self.manager._record_chat_exchange('feishu_r1', 'oc_a', 'user', '清除前的消息', user_id='ou_a')
        self.manager._clear_chat_history('feishu_r1', 'oc_a', 'ou_a')
        self.manager.active_input_channels.add('feishu_r1')
        msg = InboundMessage('feishu', 'feishu_r1', 'ou_a', 'oc_a', 'Alice', '新消息',
                             message_id='om_fresh', chat_type='p2p')
        call = mock.AsyncMock(side_effect=[
            {'role': 'assistant', 'content': '回复'},
            {'role': 'assistant', 'content': ''},
        ])
        with (
            mock.patch.object(self.manager, 'sync_from_canvas'),
            mock.patch('channel.acl.get_user', return_value={'role': 'viewer'}),
            mock.patch('channel.acl.check_permission', return_value=(True, '')),
            mock.patch('api.inspection.publish_to_topic', new_callable=mock.AsyncMock) as publish,
            mock.patch('api.motus_stream.push_event', new_callable=mock.AsyncMock),
            mock.patch('channel.manager.manager', self.manager),
        ):
            await self.manager._on_inbound_message(msg)
            topic, raw = publish.await_args.args
            trigger = collector._build_trigger([
                {'source': f'dds:{topic}', 'text': raw, 'ts': 1},
            ], urgent=True)
        self.assertEqual(trigger['_feishu_chat'], ('feishu_r1', 'oc_a', 'ou_a', 1))

        llm_module = sys.modules['event.llm']
        with (
            mock.patch('client.call', call),
            mock.patch.object(llm_module, 'push_event', new_callable=mock.AsyncMock),
            mock.patch('hooks.fire', new_callable=mock.AsyncMock),
            mock.patch('pathlib.Path.write_text'),
            mock.patch.object(llm_module, '_on_task_started'),
            mock.patch.object(llm_module, '_start_countdown'),
            mock.patch.object(llm_module, '_on_all_work_done'),
            mock.patch.object(core, '_flush_current_turn'),
        ):
            await core._one_turn(trigger)

        self.assertEqual(call.await_count, 2)
        for request in call.await_args_list:
            prompt_text = json.dumps(request.kwargs['message_list'], ensure_ascii=False)
            self.assertIn('新消息', prompt_text)
            self.assertNotIn('清除前的消息', prompt_text)
            self.assertNotIn('上一轮全局历史', prompt_text)
            self.assertNotIn('上一轮全局摘要', prompt_text)

    def test_completed_feishu_turn_is_archived_without_entering_shared_history(self):
        import chat_history
        from event.llm import Event

        core = Event()
        core._current_turn = [{'role': 'user', 'content': '普通渠道消息'}]
        core._save_current_turn({'text': '普通渠道消息'})
        self.addCleanup(chat_history.delete_session, core._session_id)

        trigger = {'text': '飞书私聊代号', '_feishu_chat': ('feishu_r1', 'oc_a', 'ou_a', 0)}
        core._current_turn = [{'role': 'user', 'content': trigger['text']}]
        core._flush_current_turn(trigger)
        self.addCleanup(chat_history.delete_session, core._feishu_archive_id)
        core._current_turn.append({'role': 'assistant', 'content': '已收到代号'})
        core._save_current_turn(trigger)
        self.manager._clear_chat_history('feishu_r1', 'oc_a', 'ou_a')

        self.assertEqual(core._build_history(), [{'role': 'user', 'content': '普通渠道消息'}])
        restored = chat_history.get_last_session_turns()
        self.assertEqual(restored['session_id'], core._session_id)
        self.assertEqual(restored['turns'], [core._build_history()])
        archived = chat_history.get_session_messages(core._feishu_archive_id)
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0], core._current_turn)

    def test_legacy_feishu_turn_is_excluded_when_restoring_shared_history(self):
        import chat_history

        session = chat_history.create_session()
        self.addCleanup(chat_history.delete_session, session)
        ordinary = [{'role': 'user', 'content': '普通渠道消息'}]
        feishu = [{'role': 'user', 'content': (
            '<event source="dds:/channel/request/feishu_r1" channel="channel:feishu" ts="time">\n'
            '{"platform":"feishu","text":"飞书私聊代号"}\n</event>'
        )}]
        legacy_feishu = [
            [{'role': 'user', 'content': (
                f'会话开始\n<event source="dds:/channel/request/legacy"{attribute} ts="time">\n'
                '{\n  "platform": "feishu",\n  "text": "旧版飞书消息"\n}\n</event>'
            )}]
            for attribute in ('', ' channel="channel"')
        ]
        telegram = [{'role': 'user', 'content': (
            '<event source="dds:/channel/request/telegram" channel="channel" ts="time">\n'
            '{"platform":"telegram","text":"Telegram 消息"}\n</event>'
        )}]
        steered = [{'role': 'user', 'content': (
            '[system notification source=dds:/channel/request/feishu_r1]\n'
            '{"platform":"feishu","text":"飞书追加消息"}'
        )}]
        turns = [ordinary, feishu, *legacy_feishu, telegram, steered]
        for index, turn in enumerate(turns):
            chat_history.save_turn(session, index, turn)
        restored = chat_history.get_last_session_turns()
        self.assertEqual(restored['turns'], [ordinary, telegram])
        self.assertTrue(restored['filtered_feishu'])
        self.assertEqual(chat_history.get_session_messages(session), turns)


if __name__ == '__main__':
    unittest.main()
