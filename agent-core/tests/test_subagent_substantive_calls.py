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
