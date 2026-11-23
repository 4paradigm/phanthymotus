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
                ('operator', 1, '已清除你在此飞书渠道的全部上下文'),
                ('owner', 2, '已清除你在此飞书渠道的全部上下文'),
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
                    self.assertEqual(
                        [e['text'] for e in self.manager.get_chat_history('feishu_r1', 'oc_b',
                                                                          user_id='ou_a')],
                        ['另一会话'] if role == 'viewer' else [],
                    )

        set_reply_context.assert_not_called()
        self.assertEqual(adapter.send_message.await_count, 3)
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a', user_id='ou_b')[0]['text'],
                         '同群的另一用户')
        self.assertEqual(self.manager.chat_generation('feishu_r1', 'oc_b', 'ou_a'), 2)
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

    def test_user_reset_supersedes_older_chat_specific_generations(self):
        config.main['channel_chat_history'] = {
            'feishu_r1': {'oc_b': {'users': {'ou_a': {'generation': 4, 'entries': []}}}},
        }
        self.assertEqual(self.manager.chat_generation('feishu_r1', 'oc_b', 'ou_a'), 4)

        self.manager._clear_user_context('feishu_r1', 'ou_a')

        self.assertEqual(self.manager.chat_generation('feishu_r1', 'oc_b', 'ou_a'), 5)
        self.assertEqual(self.manager.chat_generation('feishu_r1', 'oc_new', 'ou_a'), 5)
        self.assertEqual(self.manager.chat_generation('feishu_r1', 'oc_b', 'ou_b'), 0)
        self.manager._record_chat_exchange('feishu_r1', 'oc_b', 'assistant', '旧回复',
                                           user_id='ou_a', source_generation=4)
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_b', user_id='ou_a'), [])

    async def test_in_flight_reply_cannot_restore_cleared_history(self):
        self.manager._set_last_context('feishu_r1', 'oc_b', 'ou_a',
                                       message_id='om_old', context_user_id='ou_a')
        self.manager._set_last_context('feishu_r1', 'oc_a', 'ou_b',
                                       message_id='om_other', context_user_id='ou_b')

        async def send(_msg):
            self.manager._clear_user_context('feishu_r1', 'ou_a')

        self.manager._adapters['feishu_r1'] = mock.Mock(platform='feishu', send_message=send)
        result = await self.manager.send_to_channel('feishu_r1', '旧回复',
                                                    source_message_id='om_old')
        self.assertIn('Reply sent', result)
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_b', user_id='ou_a'), [])
        await self.manager.send_to_channel('feishu_r1', '给另一人的回复', source_message_id='om_other')
        self.assertEqual(self.manager.get_chat_history('feishu_r1', 'oc_a', user_id='ou_b')[0]['text'],
                         '给另一人的回复')

    async def test_clear_stops_its_active_turn_and_sends_receipt_after_in_flight_reply(self):
        self.manager.active_input_channels.add('feishu_r1')
        self.manager._set_last_context('feishu_r1', 'oc_b', 'ou_a', message_id='om_old',
                                       context_user_id='ou_a')
        turn_started = asyncio.Event()
        send_started = asyncio.Event()
        finish_send = asyncio.Event()
        sent_ids = []

        async def send(message):
            if message.reply_to_message_id == 'om_old':
                send_started.set()
                await finish_send.wait()
            sent_ids.append(message.reply_to_message_id)

        from event.llm import Event, TurnCancelled

        core = Event()

        async def run_turn(_trigger, *, cancel_event, reconsider_event):
            turn_started.set()
            await cancel_event.wait()
            raise TurnCancelled('Test Feishu reset')

        self.manager._adapters['feishu_r1'] = mock.Mock(platform='feishu', send_message=send)
        msg = InboundMessage('feishu', 'feishu_r1', 'ou_a', 'oc_a', 'Alice', '/clear',
                             message_id='om_clear', chat_type='p2p')
        trigger = {'_feishu_chat': ('feishu_r1', 'oc_b', 'ou_a', 0)}
        with (
            mock.patch('channel.manager.manager', self.manager),
            mock.patch.object(self.manager, 'sync_from_canvas'),
            mock.patch('channel.acl.get_user', return_value={'role': 'operator'}),
            mock.patch.object(collector, 'next_trigger',
                              new=mock.AsyncMock(side_effect=[trigger, asyncio.CancelledError()])),
            mock.patch.object(core, '_one_turn', side_effect=run_turn),
            mock.patch.object(core, '_save_current_turn'),
            mock.patch.object(core, '_interrupt_active_outputs', new_callable=mock.AsyncMock) as interrupt,
            mock.patch.object(sys.modules['event.llm'], 'push_event', new_callable=mock.AsyncMock),
            mock.patch('hooks.fire', new_callable=mock.AsyncMock),
        ):
            loop = asyncio.create_task(core.run_forever())
            await asyncio.wait_for(turn_started.wait(), timeout=2)
            sending = asyncio.create_task(self.manager.send_to_channel(
                'feishu_r1', '上一轮正在发送的回复', source_message_id='om_old',
            ))
            await asyncio.wait_for(send_started.wait(), timeout=2)
            clearing = asyncio.create_task(self.manager._on_inbound_message(msg))
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(loop, timeout=2)
            self.assertEqual(sent_ids, [])
            interrupt.assert_not_awaited()

            finish_send.set()
            await asyncio.wait_for(asyncio.gather(sending, clearing), timeout=2)
            late = await self.manager.send_to_channel(
                'feishu_r1', '清除后尝试发送的旧回复', source_message_id='om_old',
            )
        self.assertEqual(sent_ids, ['om_old', 'om_clear'])
        self.assertIn('context was cleared', late)

    def test_clear_leaves_another_sender_turn_running(self):
        cancel = asyncio.Event()
        with (
            mock.patch.object(collector, '_cancel_event', cancel),
            mock.patch.object(collector, '_current_trigger', {
                '_feishu_chat': ('feishu_r1', 'oc_a', 'ou_b', 0),
            }),
        ):
            collector.cancel_feishu_turn('feishu_r1', 'ou_a')
        self.assertFalse(cancel.is_set())

    async def test_queued_trigger_is_discarded_only_for_the_sender_who_cleared(self):
        def event(user_id, generation, text, chat_id='oc_a'):
            return {
                'source': 'dds:/channel/request/feishu_r1',
                'text': json.dumps({
                    'platform': 'feishu', 'channel_id': 'feishu_r1', 'chat_id': chat_id, 'user_id': user_id,
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
            old_in_another_chat = event('ou_a', 0, '另一会话的旧请求', chat_id='oc_b')
            other = event('ou_b', 0, '同群另一人的请求')
            await collector._emit_batch([old, other, old_in_another_chat], urgent=True)
            collector._steering_queue.put_nowait(old)
            self.manager._clear_user_context('feishu_r1', 'ou_a')
            fresh = event('ou_a', 1, '新请求')
            await collector._emit_batch([old, old_in_another_chat, fresh], urgent=True)

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
        self.manager._clear_user_context('feishu_r1', 'ou_a')
        self.manager.active_input_channels.add('feishu_r1')
        msg = InboundMessage('feishu', 'feishu_r1', 'ou_a', 'oc_a', 'Alice', '新消息',
                             message_id='om_fresh', chat_type='p2p')
        archive_lookup = mock.AsyncMock(return_value='旧档案内容')
        raw_lookup = mock.AsyncMock(return_value='当前传感器数据')
        core._sys_tools = {
            'search_history': {'schema': {'name': 'search_history'}, 'object': archive_lookup},
            'memory_recall': {'schema': {'name': 'memory_recall'}, 'object': archive_lookup},
            'raw_input_info': {'schema': {'name': 'raw_input_info'}, 'object': raw_lookup},
        }
        call = mock.AsyncMock(side_effect=[
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': f'call_{index}', 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(args)}}
                for index, (name, args) in enumerate([
                    ('search_history', {'query': '明确请求查找归档'}),
                    ('memory_recall', {'query': '明确请求查找记忆'}),
                    ('raw_input_info', {'source': 'dds:/channel/request/feishu_r1'}),
                    ('raw_input_info', {'source': 'dds:/sensors/battery'}),
                ])
            ]},
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
        self.assertNotIn('<chat_history', trigger['text'])

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
        self.assertEqual(archive_lookup.await_count, 2)
        raw_lookup.assert_awaited_once_with(source='dds:/sensors/battery')
        for request in call.await_args_list:
            tool_names = {tool['function']['name'] for tool in request.kwargs['tool_list']}
            self.assertEqual(tool_names, {'raw_input_info', 'search_history', 'memory_recall'})
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
        self.manager._clear_user_context('feishu_r1', 'ou_a')

        self.assertEqual(core._build_history(), [{'role': 'user', 'content': '普通渠道消息'}])
        restored = chat_history.get_last_session_turns()
        self.assertEqual(restored['session_id'], core._session_id)
        self.assertEqual(restored['turns'], [core._build_history()])
        archived = chat_history.get_session_messages(core._feishu_archive_id)
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0], core._current_turn)

    async def test_recall_tools_filter_feishu_scope_before_applying_limit(self):
        import chat_history
        import history_scope
        from event.llm import _memory_recall, _search_history

        def archive(kind, scope, label):
            session = chat_history.create_session(kind, feishu_scope=scope)
            self.addCleanup(chat_history.delete_session, session)
            chat_history.save_turn(session, 0, [{'role': 'user', 'content': f'ctxprobe {label}'}])
            chat_history.save_conclusion(session, 'ctxprobe', label, 'user_task', feishu_scope=scope)

        def clean_conclusions():
            with config._get_conn() as conn:
                conn.execute("DELETE FROM subagent_conclusions WHERE goal='ctxprobe'")
                conn.commit()

        self.addCleanup(clean_conclusions)
        alice_old = ('feishu_r1', 'oc_a', 'ou_a', 0)
        alice_new = ('feishu_r1', 'oc_b', 'ou_a', 1)
        bob = ('feishu_r1', 'oc_a', 'ou_b', 0)
        self.manager._clear_user_context('feishu_r1', 'ou_a')
        archive(chat_history.KIND_FEISHU, history_scope.encode(alice_new), 'alice-current')
        archive(chat_history.KIND_FEISHU, history_scope.encode(bob), 'bob-current')
        archive(chat_history.KIND_FEISHU, history_scope.encode(alice_old), 'alice-late')
        archive(chat_history.KIND_FEISHU, None, 'legacy')

        with mock.patch('channel.manager.manager', self.manager):
            for scope, expected in [(alice_new, 'alice-current'), (bob, 'bob-current')]:
                token = history_scope.current_feishu_scope.set(scope)
                try:
                    for result in (
                        await _search_history('ctxprobe', limit=1),
                        await _memory_recall('ctxprobe', source='conversation', limit=1),
                        await _memory_recall('ctxprobe', source='subagent', limit=1),
                    ):
                        self.assertIn(expected, result)
                        self.assertNotIn('alice-late', result)
                        self.assertNotIn('legacy', result)
                finally:
                    history_scope.current_feishu_scope.reset(token)

        archive(chat_history.KIND_MAIN, history_scope.UNSCOPED, 'ordinary')
        self.assertIn('ordinary', await _search_history('ctxprobe', limit=1))
        self.assertIn('ordinary', await _memory_recall('ctxprobe', source='subagent', limit=1))

    async def test_subagent_keeps_origin_after_checkpoint_and_late_completion(self):
        import chat_history
        import history_scope
        from event.llm import _memory_recall, _search_history
        from subagent.agent import Subagent
        from subagent.manager import SubagentManager
        from subagent.protocol import SubagentResult, SubagentSpec

        scope = ('feishu_r1', 'oc_a', 'ou_a', 0)
        mgr = SubagentManager(llm_client=None)
        token = history_scope.current_feishu_scope.set(scope)
        try:
            with mock.patch('subagent.manager.push_event', new_callable=mock.AsyncMock):
                agent_id = await mgr.spawn(SubagentSpec(goal='handoverprobe'))
        finally:
            history_scope.current_feishu_scope.reset(token)
        spec_json = json.loads(json.dumps(mgr._agents[agent_id].spec.to_dict()))
        restored = Subagent(SubagentSpec.from_dict(spec_json), agent_id=agent_id)
        self.manager._clear_user_context('feishu_r1', 'ou_a')
        restored.persist_turn([{'role': 'assistant', 'content': 'handoverprobe old'}])
        self.addCleanup(chat_history.delete_session, restored.history_session_id)

        def clean_conclusion():
            with config._get_conn() as conn:
                conn.execute('DELETE FROM subagent_conclusions WHERE agent_id=?', (agent_id,))

        self.addCleanup(clean_conclusion)
        with (
            mock.patch('event_bus.enqueue', new_callable=mock.AsyncMock) as enqueue,
            mock.patch('subagent.manager.push_event', new_callable=mock.AsyncMock),
        ):
            await mgr._notify_completion(restored, SubagentResult(agent_id, 'completed', 'old'))
        with config._get_conn() as conn:
            self.assertEqual(
                conn.execute('SELECT feishu_scope FROM subagent_conclusions WHERE agent_id=?',
                             (agent_id,)).fetchone()[0], history_scope.encode(scope),
            )
        event_payload = enqueue.await_args.kwargs['payload']
        notification = {'source': f'subagent:{agent_id}', 'payload': event_payload}
        self.assertEqual(collector._feishu_chat([notification]), scope)
        with mock.patch('channel.manager.manager', self.manager):
            self.assertFalse(collector.feishu_context_is_current(scope))
        self.assertNotIn('handoverprobe old', await _search_history('handoverprobe'))
        self.assertNotIn('[subagent:', await _memory_recall('handoverprobe', source='subagent'))

    async def test_upgrade_backfills_known_shared_archives_only(self):
        import sqlite3
        import chat_history
        from event.llm import _memory_recall, _search_history

        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, 'legacy.db')
            legacy = sqlite3.connect(db_path)
            with legacy:
                legacy.executescript('''
                    CREATE TABLE chat_sessions (
                        id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL,
                        summary TEXT DEFAULT '', turn_count INTEGER, kind TEXT);
                    CREATE TABLE chat_messages (
                        id INTEGER PRIMARY KEY, session_id TEXT, turn_index INTEGER,
                        messages TEXT, created_at REAL);
                    CREATE TABLE subagent_conclusions (
                        id INTEGER PRIMARY KEY, agent_id TEXT, goal TEXT, conclusion TEXT,
                        source_type TEXT, created_at REAL);
                ''')
                chat_history._ensure_fts(legacy)
                for session_id, kind, summary, text, agent_id in [
                    ('ordinary', 'main', '', '<event source="dds:/remote_control/message" '
                     'channel="remote_web" ts="time">\nmessage: upgradeprobe public\n</event>', 'ace00001'),
                    ('mixed', 'main', '', '<event source="dds:/remote_control/message" ts="time">\n'
                     'message: upgradeprobe mixed ordinary message\n</event>\n'
                     '<event source="dds:/channel/request/legacy" ts="time">\n'
                     '{"platform":"feishu","text":"upgradeprobe private"}\n</event>', 'ace00002'),
                    ('feishu', 'feishu', '', 'upgradeprobe unlabelled', None),
                    ('raw-read', 'main', '', 'upgradeprobe archived channel read', None),
                    ('ordinary-subagent', 'subagent', '[subagent:ace00001]', 'upgradeprobe helper', None),
                ]:
                    turn = [{'role': 'user', 'content': text}]
                    if agent_id:
                        turn.extend([
                            {'role': 'assistant', 'tool_calls': [
                                {'id': 'spawn', 'function': {'name': 'subagent_spawn'}},
                            ]},
                            {'role': 'tool', 'tool_call_id': 'spawn',
                             'content': f'子代理已创建: id={agent_id}, priority=2'},
                        ])
                    if session_id == 'raw-read':
                        turn.append({'role': 'assistant', 'tool_calls': [{'id': 'read', 'function': {
                            'name': 'raw_input_info',
                            'arguments': '{"source":"dds:/channel/request/legacy"}',
                        }}]})
                    raw = json.dumps(turn, ensure_ascii=False)
                    legacy.execute('INSERT INTO chat_sessions VALUES (?, 1, ?, ?, 1, ?)',
                                   (session_id, 1 if session_id == 'ordinary' else 2, summary, kind))
                    row = legacy.execute(
                        'INSERT INTO chat_messages (session_id, turn_index, messages, created_at) VALUES (?, 0, ?, 1)',
                        (session_id, raw),
                    )
                    legacy.execute('INSERT INTO chat_messages_fts(rowid, content) VALUES (?, ?)',
                                   (row.lastrowid, raw))
                for agent_id, result in [('ace00001', 'public conclusion'),
                                         ('ace00002', 'private conclusion'), ('unknown', 'unknown conclusion')]:
                    legacy.execute(
                        'INSERT INTO subagent_conclusions (agent_id, goal, conclusion, source_type, created_at) '
                        "VALUES (?, 'upgradeprobe', ?, 'user_task', 1)", (agent_id, result),
                    )
            legacy.close()

            with mock.patch.object(config, 'DB_PATH', db_path):
                result = await _search_history('upgradeprobe')
                self.assertIn('upgradeprobe public', result)
                self.assertIn('upgradeprobe helper', result)
                self.assertNotIn('upgradeprobe mixed ordinary message', result)
                self.assertNotIn('upgradeprobe private', result)
                self.assertNotIn('upgradeprobe unlabelled', result)
                self.assertNotIn('upgradeprobe archived channel read', result)
                conclusions = await _memory_recall('upgradeprobe', source='subagent')
                self.assertIn('public conclusion', conclusions)
                self.assertNotIn('private conclusion', conclusions)
                self.assertNotIn('unknown conclusion', conclusions)
                conn = config._get_conn()
                config._ensure_schema(conn)  # Reopening or retrying must keep the classification.
                self.assertEqual(await _search_history('upgradeprobe'), result)
                self.assertEqual(chat_history.get_last_session_turns()['session_id'], 'ordinary')
                self.assertTrue(chat_history.get_session_messages('mixed'))
                conn.close()

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

    def test_resume_falls_back_when_latest_legacy_session_only_has_feishu_turns(self):
        import chat_history

        ordinary_session = chat_history.create_session()
        self.addCleanup(chat_history.delete_session, ordinary_session)
        ordinary = [{'role': 'user', 'content': '较早的普通渠道消息'}]
        chat_history.save_turn(ordinary_session, 0, ordinary)

        latest_session = chat_history.create_session()
        self.addCleanup(chat_history.delete_session, latest_session)
        feishu = [{'role': 'user', 'content': (
            '<event source="dds:/channel/request/legacy" ts="time">\n'
            '{"platform":"feishu","text":"旧版飞书消息"}\n</event>'
        )}]
        chat_history.save_turn(latest_session, 0, feishu)

        restored = chat_history.get_last_session_turns(limit=1)
        self.assertEqual(restored['session_id'], ordinary_session)
        self.assertEqual(restored['turns'], [ordinary])
        self.assertTrue(restored['filtered_feishu'])
        self.assertEqual(chat_history.get_session_messages(latest_session), [feishu])

        # With an earlier ordinary turn in the latest session, filter before applying the limit.
        latest_ordinary = [{'role': 'user', 'content': '同一会话中的普通渠道消息'}]
        chat_history.save_turn(latest_session, 0, latest_ordinary)
        chat_history.save_turn(latest_session, 1, feishu)
        restored = chat_history.get_last_session_turns(limit=1)
        self.assertEqual(restored['session_id'], latest_session)
        self.assertEqual(restored['turns'], [latest_ordinary])
        self.assertTrue(restored['filtered_feishu'])


if __name__ == '__main__':
    unittest.main()
