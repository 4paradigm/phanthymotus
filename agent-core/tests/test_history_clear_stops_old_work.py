"""清空历史之后，机器人不能接着做旧对话里的事 —— 重启之后也不能。

现场症状是「清了历史、甚至重启了，机器人还在执行以前的任务」。来源不止一个，
这个文件逐个钉住：

1. **「清空全部历史」只删了库。** `DELETE /api/history/sessions` 原来只调
   `chat_history.clear_all()`，内存里的 `_turns` / `_summary` 原封不动，下一轮
   `_build_history()` 照样把旧对话拼进 prompt。
2. **压缩把旧对话写回来。** `_maybe_compress` 先取出 `_turns`、`await` 一次 LLM、
   再无条件写回 `_summary` 和最近几轮。清空落在这个 await 里，旧对话就连同它的摘要
   一起回到内存。
3. **正在跑的 turn 不停。** 它的 history 在开头就拼好了，清空之后还会带着旧上下文
   把剩下的工具调用跑完，最后再把自己存进新历史。
4. **子代理跨重启续跑。** 关机时在跑的子代理被存成 suspended，启动时 `_restore()`
   重新排队、接着做原来的目标；`cancel()` 走正常收尾，又会落结论、唤醒主代理。
5. **静态 scheduler 停不掉。** `config.main['scheduler']` 启动时加载一次，原来没有
   任何办法不重启就停掉其中一条。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_history_clear_stops_old_work.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import chat_history  # noqa: E402
import config  # noqa: E402
import event  # noqa: E402
import event_bus  # noqa: E402
import scheduler  # noqa: E402
from api import history as history_api  # noqa: E402
from subagent.agent import Subagent  # noqa: E402
from subagent.manager import SubagentManager  # noqa: E402
from subagent.protocol import SubagentSpec  # noqa: E402

llm_mod = sys.modules['event.llm']

_OLD_TURN = [{'role': 'user', 'content': '例行检查外卖超时'},
             {'role': 'assistant', 'content': '好的，我去看看外卖'}]


class ClearAllForgetsMemoryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = (event.llm._turns, event.llm._summary, event.llm._session_id)

    def tearDown(self):
        event.llm._turns, event.llm._summary, event.llm._session_id = self._saved

    async def test_clear_all_endpoint_empties_the_in_memory_context(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, _OLD_TURN)
        event.llm._turns = [list(_OLD_TURN)]
        event.llm._summary = '之前一直在盯外卖'
        event.llm._session_id = sid

        await history_api.clear_all()

        self.assertEqual(event.llm._turns, [], '只删库的话，模型下一轮照样看得到旧对话')
        self.assertIsNone(event.llm._summary)
        self.assertIsNone(event.llm._session_id)
        self.assertIsNone(chat_history.get_last_session_turns())

    def test_deleting_the_current_session_forgets_it_too(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, _OLD_TURN)
        event.llm._turns = [list(_OLD_TURN)]
        event.llm._session_id = sid

        history_api.delete_session(sid)

        self.assertEqual(event.llm._turns, [])

    def test_deleting_another_session_keeps_the_live_context(self):
        event.llm._turns = [list(_OLD_TURN)]
        event.llm._session_id = 'live'

        history_api.batch_delete(history_api.BatchDeleteBody(ids=['someone-else']))

        self.assertEqual(len(event.llm._turns), 1)


class CompressionRaceTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_clear_during_compression_is_not_undone(self):
        ev = llm_mod.Event()
        ev._turns = [list(_OLD_TURN) for _ in range(30)]
        ev._summary = '旧摘要'

        async def compress(_turns):
            ev.forget_conversation()          # 用户在压缩进行中点了清空
            return '外卖检查的摘要'

        async def rewrite(old, new, budget):
            return f'{old}+{new}'

        saved = llm_mod._compress_turns, llm_mod._rewrite_summary
        llm_mod._compress_turns, llm_mod._rewrite_summary = compress, rewrite
        try:
            await ev._maybe_compress()
        finally:
            llm_mod._compress_turns, llm_mod._rewrite_summary = saved

        self.assertEqual(ev._turns, [], '压缩写回了清空前的最近几轮')
        self.assertIsNone(ev._summary, '压缩写回了旧对话的摘要')

    async def test_compression_without_a_clear_still_works(self):
        ev = llm_mod.Event()
        ev._turns = [list(_OLD_TURN) for _ in range(30)]

        async def compress(_turns):
            return '摘要'

        saved = llm_mod._compress_turns
        llm_mod._compress_turns = compress
        try:
            await ev._maybe_compress()
        finally:
            llm_mod._compress_turns = saved

        self.assertEqual(ev._summary, '摘要')
        self.assertLess(len(ev._turns), 30)


class InFlightTurnTest(unittest.TestCase):
    def test_forget_cancels_the_running_turn(self):
        ev = llm_mod.Event()
        cancel = asyncio.Event()
        ev._turn_cancel = cancel
        gen = ev._history_gen

        ev.forget_conversation()

        self.assertTrue(cancel.is_set(), '不停下的话，这一轮会带着旧上下文跑完')
        self.assertNotEqual(ev._history_gen, gen,
                            'run_forever 靠它判断不把这一轮存进新历史')


class _NeverEnding:
    """替掉 Subagent.run：一直跑到被取消。"""
    async def __call__(self, llm_client=None):
        await asyncio.sleep(3600)


class DiscardSubagentsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.enqueued = []
        self._saved_enqueue = event_bus.enqueue

        async def enqueue(**kw):
            self.enqueued.append(kw)
        event_bus.enqueue = enqueue

    async def asyncTearDown(self):
        event_bus.enqueue = self._saved_enqueue

    async def test_a_running_subagent_is_stopped_silently_and_not_restored(self):
        mgr = SubagentManager(llm_client=None)
        agent = Subagent(SubagentSpec(goal='每 10 分钟检查外卖超时'))
        agent.run = _NeverEnding()
        agent.status = 'running'
        mgr._agents[agent.id] = agent
        mgr._running[agent.id] = asyncio.create_task(mgr._run_agent(agent))
        mgr._checkpoint(agent)
        await asyncio.sleep(0)

        self.assertEqual(await mgr.discard_all('test'), 1)

        self.assertEqual(mgr._agents, {})
        self.assertEqual(mgr._running, {})
        self.assertEqual(mgr._store.load_active(), [],
                         '取消时的 suspended checkpoint 会让下次启动接着做')
        self.assertEqual(self.enqueued, [], '收尾不能再唤醒主代理')

    async def test_a_suspended_row_from_last_boot_is_removed(self):
        mgr = SubagentManager(llm_client=None)
        spec = SubagentSpec(goal='上次关机前没做完的事')
        mgr._store.save_checkpoint(
            agent_id='old1', spec=spec, status='suspended', turns=[], summary='',
            rounds_completed=3, created_at=time.time(), updated_at=time.time())

        await mgr.discard_all('test')

        fresh = SubagentManager(llm_client=None)
        await fresh._restore()
        self.assertEqual(fresh._agents, {}, '重启后不能再把旧子代理排回队列')


class StaticSchedulerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._saved = config.main.get('scheduler', [])
        config.main['scheduler'] = [
            {'name': 'takeout-timeout-check', 'cron': '*/10 * * * *', 'text': '例行检查外卖超时'},
            {'name': 'hourly', 'cron': '0 * * * *', 'text': '整点检查'},
        ]
        self.runner = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0)

    async def asyncTearDown(self):
        self.runner.cancel()
        try:
            await self.runner
        except asyncio.CancelledError:
            pass
        config.main['scheduler'] = self._saved

    async def test_static_jobs_are_listed(self):
        jobs = {j['name']: j for j in scheduler.list_static()}
        self.assertEqual(set(jobs), {'takeout-timeout-check', 'hourly'})
        self.assertTrue(jobs['takeout-timeout-check']['running'])
        self.assertEqual(jobs['takeout-timeout-check']['text'], '例行检查外卖超时')
        self.assertIsNotNone(jobs['hourly']['next_run'])

    async def test_remove_stops_the_timer_and_drops_the_config_entry(self):
        task = scheduler._static_tasks['takeout-timeout-check']

        self.assertTrue(scheduler.remove_static('takeout-timeout-check'))
        await asyncio.sleep(0)

        self.assertTrue(task.cancelled(), '不重启就要停下')
        self.assertEqual([j['name'] for j in config.main['scheduler']], ['hourly'],
                         '配置里还在的话，下次启动又会加载')
        self.assertIn('hourly', scheduler._static_tasks)

    async def test_removing_an_unknown_name_reports_false(self):
        self.assertFalse(scheduler.remove_static('nope'))
        self.assertEqual(len(config.main['scheduler']), 2)


if __name__ == '__main__':
    unittest.main()
