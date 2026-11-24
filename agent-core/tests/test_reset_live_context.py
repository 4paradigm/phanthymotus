"""Exercise history clearing while work from the previous conversation is pending."""
import asyncio
import importlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import chat_history
import collector
import config
import event
from api.config import ResetRequest, reset_config
from api.history import clear_all

llm = importlib.import_module('event.llm')


class ResetLiveContextTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_path = patch.object(config, 'DB_PATH', os.path.join(directory.name, 'test.db'))
        db_path.start()
        self.addCleanup(db_path.stop)
        self.addCleanup(config._get_conn().close)
        config._seed_defaults()
        self.addCleanup(setattr, chat_history, 'last_clear_time', chat_history.last_clear_time)
        collector.clear_context()
        self.addCleanup(collector.clear_context)

    async def check_reset(self, reset, paused_function):
        agent = llm.Event()
        old = [{'role': 'user', 'content': 'old secret'}]
        new = [{'role': 'user', 'content': 'new conversation'}]
        agent._current_turn = old.copy()
        agent._flush_current_turn()
        agent._turns = [old] * 20
        agent._summary = 'old summary'
        agent._turn_cancel = asyncio.Event()
        config.main['channel_chat_history'] = {'channel': {'chat': old}}
        collector.defer_priority([{'text': 'stale input'}])

        started, resume = asyncio.Event(), asyncio.Event()

        async def delayed(*args):
            started.set()
            await resume.wait()
            return 'stale summary'

        with patch.object(event, 'llm', agent), \
             patch.object(llm, '_compress_turns', AsyncMock(return_value='summary')), \
             patch.object(llm, paused_function, side_effect=delayed):
            pending = asyncio.create_task(agent._maybe_compress())
            try:
                await asyncio.wait_for(started.wait(), 2)
                await reset()
                self.assertTrue(agent._turn_cancel.is_set())
                self.assertEqual(agent._current_turn, [])
                self.assertIsNone(agent._session_id)
                self.assertEqual(agent._build_history(), [])
                self.assertEqual(config.main.get('channel_chat_history', {}), {})
                self.assertFalse(collector.has_steering())
                self.assertIsNone(chat_history.get_last_session_turns())
                agent._current_turn = new.copy()
                agent._save_current_turn({})
            finally:
                resume.set()
                await pending

        self.assertIsNone(agent._summary)
        self.assertEqual(agent._build_history(), new)
        self.assertEqual(chat_history.get_last_session_turns()['turns'], [new])

    async def test_old_subagent_cannot_republish_history(self):
        from subagent.agent import Subagent
        from subagent.manager import SubagentManager
        from subagent.protocol import SubagentResult, SubagentSpec

        old_agent = Subagent(SubagentSpec(goal='old secret'))
        await clear_all()
        result = SubagentResult(old_agent.id, 'completed', 'old conclusion')
        old_agent.persist_turn([{'role': 'user', 'content': 'old secret'}])
        await old_agent._dispatch_tool('subagent_report', {'progress': 'old progress'})
        manager = SubagentManager.__new__(SubagentManager)
        await manager._notify_completion(old_agent, result)
        self.assertEqual(chat_history.list_sessions()[1], 0)
        with config._get_conn() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM subagent_conclusions').fetchone()[0], 0)

    async def test_canvas_reset_during_summary_rewrite(self):
        await self.check_reset(lambda: reset_config(ResetRequest(chat_history=True)),
                               '_rewrite_summary')

    async def test_clear_all_during_initial_summary(self):
        await self.check_reset(clear_all, '_compress_turns')
