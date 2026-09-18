"""
test_subagent_substantive_calls.py — "干了活" vs "只是说自己干了活"。

一个 subagent 可以只调 `subagent_finish` 就返回 STATUS_COMPLETED，output 里写一段
"已完成"的散文。在协议层，这和真正执行了动作的运行完全无法区分 —— 而 `peer_delegate`
把这段散文原样交回委派方当成功（peer/delegation.py 的 `result.get('output')`）。

Orin5+Orin6 实测：六个入站委派各要求对端说一句台词，其中四个只跑一轮、没碰 tts 就
报了 completed，两台机器随后都宣布完成了一场十六句的表演 —— 一句都没播。

`substantive_tool_calls()` 是区分这两者的依据，Phase 3 的委派响应会消费它。
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import asyncio  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from subagent.agent import Subagent  # noqa: E402
from subagent.protocol import (  # noqa: E402
    BOOKKEEPING_TOOLS,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    SubagentResult,
    SubagentSpec,
)


def _result(*tool_names: str) -> SubagentResult:
    return SubagentResult(
        agent_id='a1',
        status=STATUS_COMPLETED,
        output='已完成捧哏台词播报。',
        tool_calls_made=[{'name': n, 'round': i} for i, n in enumerate(tool_names)],
    )


class TestSubstantiveToolCalls(unittest.TestCase):
    def test_finish_only_is_not_substantive(self):
        """只调 subagent_finish —— 这正是 Orin6 那四个的形状。"""
        self.assertEqual(_result('subagent_finish').substantive_tool_calls(), [])

    def test_all_bookkeeping_tools_are_excluded(self):
        r = _result(*sorted(BOOKKEEPING_TOOLS))
        self.assertEqual(r.substantive_tool_calls(), [])

    def test_real_tool_counts(self):
        r = _result('mcp__mcp-1__tts', 'subagent_finish')
        self.assertEqual(r.substantive_tool_calls(), ['mcp__mcp-1__tts'])

    def test_report_then_act(self):
        """subagent_report 不算干活，但它后面的真工具算。"""
        r = _result('subagent_report', 'mcp__mcp-1__tts', 'subagent_finish')
        self.assertEqual(r.substantive_tool_calls(), ['mcp__mcp-1__tts'])

    def test_no_tool_calls_at_all(self):
        self.assertEqual(_result().substantive_tool_calls(), [])

    def test_malformed_entries_are_skipped(self):
        """tool_calls_made 来自 LLM 响应解析，条目可能缺 name。"""
        r = SubagentResult(
            agent_id='a1', status=STATUS_COMPLETED, output='',
            tool_calls_made=[{}, {'name': ''}, {'round': 0}, {'name': 'mcp__m__loco'}],
        )
        self.assertEqual(r.substantive_tool_calls(), ['mcp__m__loco'])

    def test_to_dict_exposes_it(self):
        """委派响应要能把这个事实带过机器边界。"""
        d = _result('subagent_finish').to_dict()
        self.assertIn('substantive_tool_calls', d)
        self.assertEqual(d['substantive_tool_calls'], [])

    def test_from_dict_ignores_the_derived_key(self):
        """to_dict 的输出要能喂回 from_dict —— 它不是 dataclass 字段。"""
        d = _result('mcp__mcp-1__tts', 'subagent_finish').to_dict()
        back = SubagentResult.from_dict(d)
        self.assertEqual(back.substantive_tool_calls(), ['mcp__mcp-1__tts'])


class TestConfirmedActions(unittest.TestCase):
    """启动一个异步动作不等于它发生了。

    `speak` 在音频**入队**时就返回，所以"只排了队"和"真播完了"写出来的散文一模一样。
    只有 'completed' 算数 —— 'timeout' 尤其不算，它清 pending 并照常放行。
    """

    def _with(self, *statuses) -> SubagentResult:
        return SubagentResult(
            agent_id='a1', status=STATUS_COMPLETED, output='说完了',
            tool_calls_made=[{'name': 'mcp__m__tts', 'round': 0}],
            actions=[{'action_id': f'speak-{i}', 'status': s, 'tool': 'tts'}
                     for i, s in enumerate(statuses)],
        )

    def test_only_completed_counts(self):
        r = self._with('completed', 'timeout', 'cancelled', 'pending', 'barge_in')
        self.assertEqual([a['action_id'] for a in r.confirmed_actions()], ['speak-0'])

    def test_acted_true_on_confirmed_action(self):
        self.assertTrue(self._with('completed').acted())

    def test_acted_true_on_substantive_tool_without_acp(self):
        """不是每个工具都走 ACP。"""
        r = SubagentResult(agent_id='a1', status=STATUS_COMPLETED, output='',
                           tool_calls_made=[{'name': 'Read', 'round': 0}])
        self.assertTrue(r.acted())

    def test_acted_false_when_only_bookkeeping(self):
        r = SubagentResult(agent_id='a1', status=STATUS_COMPLETED, output='已完成！',
                           tool_calls_made=[{'name': 'subagent_finish', 'round': 0}])
        self.assertFalse(r.acted(), '只调 subagent_finish 不算干活')

    def test_to_dict_carries_both(self):
        d = self._with('completed', 'timeout').to_dict()
        self.assertEqual(len(d['actions']), 2)
        self.assertEqual(len(d['confirmed_actions']), 1)

    def test_from_dict_round_trips_actions(self):
        back = SubagentResult.from_dict(self._with('completed').to_dict())
        self.assertEqual(len(back.confirmed_actions()), 1)


class TestSettleOwnActionsBeforeFinish(unittest.TestCase):
    """subagent_finish 之前必须等自己启动的动作落地。

    实测于 Orin6：一个被委派的 subagent 调了 tts，然后在音频播完**前 1.2 秒**就
    subagent_finish —— 委派因此提前返回，委派方可以压着对端说话，正是这次要消掉的问题。

    更糟的是它让新加的 action 清单说谎。`/api/acp/complete` 只置位 event，终态是被
    barrier 或 sync 清理时才记下的；先 finish 就意味着 `action_outcome()` 还是 None，
    于是清单报 'pending'、`peer_delegate` 宣布"did NOT confirm"—— 而那句话其实播了。
    一个会误报的验证器比没有更糟。
    """

    def setUp(self):
        import mcp_client
        self.mcp_client = mcp_client
        for d in (mcp_client._pending_actions, mcp_client._pending_results,
                  mcp_client._pending_timeouts, mcp_client._pending_tools,
                  mcp_client._pending_resources):
            d.clear()
        mcp_client._action_outcomes.clear()

    def _agent(self):
        a = Subagent(SubagentSpec(goal='说一句捧哏词'), agent_id='settle01')
        a._action_ids = ['speak-x']
        return a

    def test_completed_action_is_reported_as_completed(self):
        """回归：这条以前会报 'pending'。"""
        async def _scenario():
            agent = self._agent()
            ev = asyncio.Event()
            self.mcp_client._pending_actions['speak-x'] = ev
            self.mcp_client._pending_tools['speak-x'] = 'tts'
            self.mcp_client._pending_timeouts['speak-x'] = 30.0
            self.mcp_client._pending_resources['speak-x'] = frozenset({'mouth'})
            # 模拟 driver 的 /api/acp/complete 稍后到达
            async def _late_complete():
                await asyncio.sleep(0.05)
                self.mcp_client._pending_results['speak-x'] = {'status': 'completed'}
                ev.set()
            asyncio.create_task(_late_complete())
            await agent._dispatch_tool('subagent_finish', {'output': '说完了'})
            return agent._action_report()

        report = asyncio.run(_scenario())
        self.assertEqual(report[0]['status'], 'completed',
                         '动作已完成却被报成 pending —— 委派方会误判为没做')

    def test_finish_does_not_return_before_the_action_settles(self):
        async def _scenario():
            agent = self._agent()
            ev = asyncio.Event()
            self.mcp_client._pending_actions['speak-x'] = ev
            self.mcp_client._pending_timeouts['speak-x'] = 30.0
            order = []

            async def _late_complete():
                await asyncio.sleep(0.05)
                order.append('audio_done')
                ev.set()

            asyncio.create_task(_late_complete())
            await agent._dispatch_tool('subagent_finish', {'output': 'x'})
            order.append('finish_returned')
            return order

        self.assertEqual(asyncio.run(_scenario()),
                         ['audio_done', 'finish_returned'])

    def test_no_actions_returns_immediately(self):
        agent = Subagent(SubagentSpec(goal='g'), agent_id='settle02')
        out = asyncio.run(agent._dispatch_tool('subagent_finish', {'output': 'done'}))
        self.assertEqual(out, 'done')

    def test_cancelled_subagent_does_not_wait_forever(self):
        """取消的 subagent 不该抱着 slot 等满整段播放。"""
        async def _scenario():
            agent = self._agent()
            self.mcp_client._pending_actions['speak-x'] = asyncio.Event()
            self.mcp_client._pending_timeouts['speak-x'] = 300.0
            agent._cancel_event.set()
            return await asyncio.wait_for(
                agent._dispatch_tool('subagent_finish', {'output': 'x'}), timeout=3)

        self.assertEqual(asyncio.run(_scenario()), 'x')


class TestCancelCarriesReason(unittest.TestCase):
    """取消一个 running subagent 原来完全静默 —— 日志在半路断掉，和挂死一模一样。

    这条路径不需要 LLM：cancel 信号在 run() 的第一个检查点就命中并返回。
    """

    def _cancelled(self, reason: str) -> SubagentResult:
        agent = Subagent(SubagentSpec(goal='说一句捧哏词'), agent_id='c0ffee')
        agent.cancel(reason)
        return asyncio.run(agent.run())

    def test_reason_lands_on_the_result(self):
        r = self._cancelled('idle timeout (300s without progress)')
        self.assertEqual(r.status, STATUS_CANCELLED)
        self.assertEqual(r.error, 'idle timeout (300s without progress)')

    def test_reasonless_cancel_still_attributable(self):
        r = self._cancelled('')
        self.assertEqual(r.status, STATUS_CANCELLED)
        self.assertEqual(r.error, 'cancelled')

    def test_duration_is_still_recorded(self):
        """加 error= 时曾把 duration_s 挤掉过。"""
        r = self._cancelled('shutting down')
        self.assertIsNotNone(r.duration_s)
        self.assertGreaterEqual(r.duration_s, 0.0)

    def test_cancel_before_any_round_is_not_substantive(self):
        self.assertEqual(self._cancelled('x').substantive_tool_calls(), [])


if __name__ == '__main__':
    unittest.main()
