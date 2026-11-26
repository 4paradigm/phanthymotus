"""Shared channel recaps coexist with WeCom conversation history."""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import chat_history
import collector
import config
from channel import acl, context
from channel.manager import ChannelManager
from subagent.agent import Subagent
from subagent.manager import SubagentManager
from subagent.protocol import SubagentSpec

llm = importlib.import_module('event.llm')


def trigger(role='viewer', *, user='alice', chat='chat', chat_only=False):
    return collector._build_trigger([{
        'source': 'dds:/channel/request/test', 'ts': 1,
        'text': json.dumps({
            'platform': 'wecom', 'channel_id': 'test', 'chat_id': chat,
            'user_id': user, 'user_role': role, 'sender_type': 'user',
            'chat_only': chat_only, 'message_id': 'm1', 'text': 'CURRENTQUESTION',
        }),
    }], urgent=True)


class ChannelHistoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pool = threading.local()
        self.db_patch = mock.patch.multiple(
            config, DB_PATH=str(pathlib.Path(self.temp.name) / 'test.db'), _conn_local=self.pool)
        self.db_patch.start()
        self.config_patch = mock.patch.object(config, 'main', {
            'event': {'llm': {}},
            'channel_configs': [{'id': 'test', 'platform': 'wecom', 'enabled': True}],
        })
        self.config_patch.start()
        self.previous_event = llm._event_instance
        acl.upsert_user('wecom', 'alice', role='operator')

    def tearDown(self):
        llm._event_instance = self.previous_event
        conn = getattr(self.pool, 'conn', None)
        if conn:
            conn.close()
        self.config_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def save(self, agent, event, text):
        agent._load_channel_history(event)
        agent._current_turn = [
            {'role': 'user', 'content': text}, {'role': 'assistant', 'content': 'DONE'},
        ]
        agent._save_current_turn(event)

    def history_with_turns(self, count=20):
        event = trigger()
        agent = llm.Event()
        agent._load_channel_history(event)
        sid = agent._channel_session_id
        for index in range(count):
            chat_history.save_turn(sid, index, [
                {'role': 'user', 'content': f'CONVERSATION{index}'},
                {'role': 'assistant', 'content': f'ANSWER{index}'},
            ])
        agent._load_channel_history(event)
        return agent, event

    async def test_operator_and_owner_keep_shared_conversation_and_child_memory(self):
        for role in ('operator', 'owner'):
            with self.subTest(role=role):
                agent, event = llm.Event(), trigger(role)
                self.save(agent, event, 'OWNTASK')
                origin = context.capture(event)
                self.assertIsNone(agent._channel_session_id)
                self.assertEqual(origin['_channel_session_id'], agent._session_id)
                self.assertIn('OWNTASK', json.dumps(agent._build_history()))
                worker = Subagent(SubagentSpec(goal='Own task', metadata={'_channel_origin': origin}))
                worker.persist_turn([{'role': 'assistant', 'content': 'OWNCHILDHISTORY'}])
                await worker._dispatch_tool('subagent_report', {'progress': 'OWNCHILDREPORT'})
                token = chat_history.current_channel_context.set(llm._channel_history_key(event))
                try:
                    self.assertIn('OWNTASK', await llm._search_history('OWNTASK'))
                    self.assertIn('OWNCHILDHISTORY', await llm._search_history('OWNCHILDHISTORY'))
                    self.assertIn('[subagent:', await llm._memory_recall('OWNCHILDREPORT', source='subagent'))
                finally:
                    chat_history.current_channel_context.reset(token)
                self.assertIn('OWNTASK', await llm._search_history('OWNTASK'))
                self.assertIn('OWNTASK', json.dumps(chat_history.get_last_session_turns()))

    async def test_worker_keeps_reply_origin_without_losing_shared_memory(self):
        agent, event = llm.Event(), trigger('operator')
        self.save(agent, event, 'SHAREDTASK')
        origin = context.capture(event)
        worker = Subagent(SubagentSpec(goal='Own task', metadata={'_channel_origin': origin}))
        manager = SubagentManager(None)

        async def run(_):
            self.assertIsNone(chat_history.current_channel_context.get())
            self.assertEqual(context.capture(), origin)
            self.assertIn('SHAREDTASK', await llm._search_history('SHAREDTASK'))
            self.assertIn('SHAREDTASK', llm.get_recent_context_rich())

        with (mock.patch.object(llm, '_event_instance', agent),
              mock.patch.object(worker, 'run', side_effect=run),
              mock.patch.object(manager, '_checkpoint'),
              mock.patch.object(manager, '_finalize') as finalize):
            await manager._run_agent(worker)
            finalize.assert_not_called()
        self.assertIsNone(context.current.get())
        self.assertIsNone(chat_history.current_channel_context.get())
        resumed = collector._build_trigger([{
            'source': 'subagent:worker/completed', 'text': 'DONE', 'ts': 2, '_channel_origin': origin,
        }], urgent=True)
        agent._load_channel_history(resumed)
        self.assertEqual(resumed['_channel_session_id'], agent._session_id)
        self.assertEqual(resumed['_channel_message_ids'], ['m1'])

    def test_deleted_or_revoked_main_origin_cannot_resume(self):
        agent, event = llm.Event(), trigger('operator')
        self.save(agent, event, 'TASK')
        origin = context.capture(event)
        self.assertTrue(context.is_active(origin))
        acl.upsert_user('wecom', 'alice', role='blocked')
        self.assertFalse(context.is_active(origin))
        acl.upsert_user('wecom', 'alice', role='operator')
        chat_history.delete_session(agent._session_id)
        self.assertFalse(context.is_active(origin))

    async def test_restricted_history_does_not_enter_main_memory_or_recall_other_contexts(self):
        agent = llm.Event()
        self.save(agent, trigger('operator'), 'SHAREDSECRET')
        for role, chat_only in (('viewer', False), ('operator', True)):
            with self.subTest(role=role, chat_only=chat_only):
                event = trigger(role, chat_only=chat_only)
                self.save(agent, event, 'RESTRICTEDQUESTION')
                self.assertNotIn('RESTRICTEDQUESTION', json.dumps(agent._build_history()))
                token = chat_history.current_channel_context.set(llm._channel_history_key(event))
                try:
                    self.assertIn('RESTRICTEDQUESTION', await llm._search_history('RESTRICTEDQUESTION'))
                    self.assertEqual(await llm._search_history('SHAREDSECRET'), '未找到相关历史记录。')
                finally:
                    chat_history.current_channel_context.reset(token)
                self.assertIn('RESTRICTEDQUESTION', json.dumps(llm.Event()._load_channel_history(event)))
                self.assertNotIn('RESTRICTEDQUESTION', json.dumps(
                    llm.Event()._load_channel_history(trigger(role, user='bob', chat_only=chat_only))))

    def test_unknown_channel_identity_does_not_fall_back_to_shared_history(self):
        event = trigger(user='')
        agent = llm.Event()
        self.assertEqual(llm._channel_history_key(event), '')
        self.assertEqual(agent._load_channel_history(event), [])
        self.assertIsNone(agent._channel_session_id)
        self.assertEqual(chat_history.search('anything', context_key=''), [])

    def test_legacy_main_history_is_not_copied_into_a_restricted_session(self):
        event = trigger()
        old_sid = chat_history.create_session()
        legacy = [
            {'role': 'user', 'content': event['text']},
            {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'call1', 'type': 'function',
                'function': {'name': 'search_history', 'arguments': '{}'},
            }]},
            {'role': 'tool', 'tool_call_id': 'call1', 'content': 'UNSCOPEDTOOLRESULT'},
        ]
        chat_history.save_turn(old_sid, 0, legacy)
        agent = llm.Event()
        self.assertEqual(agent._load_channel_history(event), [])
        self.assertEqual(chat_history.get_session_messages(old_sid), [legacy])
        with config._get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM chat_messages').fetchone()[0], 1)
        chat_history.delete_session(agent._channel_session_id)
        self.assertEqual(llm.Event()._load_channel_history(event), [])
        self.assertEqual(len(chat_history.search('UNSCOPEDTOOLRESULT')), 1)

    def test_deleted_restricted_history_has_no_searchable_copy_or_reimport(self):
        agent, event = llm.Event(), trigger()
        self.save(agent, event, 'DELETESECRET')
        chat_history.delete_session(agent._channel_session_id)
        self.assertEqual(chat_history.search('DELETESECRET'), [])
        self.assertEqual(llm.Event()._load_channel_history(event), [])

    def test_legacy_recap_is_not_duplicated_in_the_trigger(self):
        config.main['channel_chat_history'] = {'test': {'chat': [
            {'role': 'user', 'text': 'OLDRECAP'},
        ]}}
        self.assertNotIn('OLDRECAP', trigger()['text'])
        self.assertIn('CURRENTQUESTION', trigger()['text'])

    async def test_compression_error_or_empty_response_preserves_cursor_and_turns(self):
        agent, event = self.history_with_turns()
        before = chat_history.get_channel_session(context.context_key(event), limit=None)
        for outcome in (RuntimeError('offline'), {'content': ''}, {'content': '   '}):
            with self.subTest(outcome=outcome):
                fake = (mock.AsyncMock(side_effect=outcome) if isinstance(outcome, Exception)
                        else mock.AsyncMock(return_value=outcome))
                with mock.patch.object(llm.client, 'call', fake):
                    await agent._maybe_compress_channel()
                self.assertEqual(chat_history.get_channel_session(context.context_key(event), limit=None), before)
                self.assertEqual(len(agent._channel_turns), 20)

    async def test_merge_failure_preserves_previous_summary_and_cursor(self):
        agent, event = self.history_with_turns(30)
        key = context.context_key(event)
        chat_history.save_channel_summary(agent._channel_session_id, key, 'EARLYSUMMARY',
                                          through=5, expected_through=-1)
        agent._load_channel_history(event)
        before = chat_history.get_channel_session(key, limit=None)
        for outcome in (RuntimeError('offline'), {'content': ''}):
            with self.subTest(outcome=outcome):
                with mock.patch.object(llm.client, 'call', mock.AsyncMock(
                        side_effect=[{'content': 'NEWSUMMARY'}, outcome])):
                    await agent._maybe_compress_channel()
                self.assertEqual(chat_history.get_channel_session(key, limit=None), before)
                self.assertEqual(agent._channel_summary, 'EARLYSUMMARY')
                self.assertEqual(len(agent._channel_turns), 24)

    async def test_successful_summary_survives_restart_without_deleting_archive(self):
        agent, event = self.history_with_turns()
        with mock.patch.object(llm.client, 'call', mock.AsyncMock(return_value={'content': 'EARLYSUMMARY'})):
            await agent._maybe_compress_channel()
        restored = llm.Event()
        history = restored._load_channel_history(event)
        self.assertIn('EARLYSUMMARY', json.dumps(history))
        self.assertEqual(restored._channel_summary_through, 5)
        self.assertEqual(len(restored._channel_turns), 14)
        self.assertEqual(len(chat_history.get_session_messages(restored._channel_session_id)), 20)

    async def test_summary_cannot_resurrect_a_deleted_session(self):
        agent, event = self.history_with_turns()

        async def summarize(**_):
            chat_history.delete_session(agent._channel_session_id)
            return {'content': 'TOOLATE'}

        with mock.patch.object(llm.client, 'call', side_effect=summarize):
            await agent._maybe_compress_channel()
        self.assertIsNone(chat_history.get_channel_session(context.context_key(event)))
        self.assertEqual(agent._channel_summary, '')

    async def test_main_compression_and_merge_failures_preserve_history(self):
        for outcomes in ([RuntimeError('offline')], [{'content': 'NEW'}, RuntimeError('offline')]):
            with self.subTest(outcomes=outcomes):
                agent = llm.Event()
                agent._turns = [[{'role': 'user', 'content': f'OLD{i}'}] for i in range(20)]
                original = list(agent._turns)
                agent._summary = 'EARLYSUMMARY'
                with mock.patch.object(llm.client, 'call', mock.AsyncMock(side_effect=outcomes)):
                    await agent._maybe_compress()
                self.assertEqual(agent._summary, 'EARLYSUMMARY')
                self.assertEqual(agent._turns, original)


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



if __name__ == '__main__':
    unittest.main()
