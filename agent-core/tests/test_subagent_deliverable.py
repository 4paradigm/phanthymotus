"""
test_subagent_deliverable.py — 「报告成功却没干活」的告警不能对监控类 subagent 误报。

`BOOKKEEPING_TOOLS` 的守卫是真问题换来的：Orin5+Orin6 上六次入站 peer 委派，每次都要求
对方说一句话，其中四次只跑了一轮、调了 `subagent_finish` 就返回 completed，而
`peer_delegate` 把这份自述当成功交还给委派方 —— 两台机器人于是都宣布了一场根本没发生的
十六行演出。那个守卫必须保留，而且必须保持严格。

但它对**后台监控** subagent 是反的。监控的全部职责就是读传感器数据、判断重要性、用
`subagent_report` 回答（见 `collector._route_to_bg_subagent`，它还用 tool_deny 挡掉了所有
会动的工具）。它正确干完活时**必然**只调用了记账类工具，于是每次都被判成「没干活」。
天轶一份日志里 85 次，全是这一种 —— 一个总在误报的告警等于没有告警。

修的是告警的适用范围，不是判据本身：`substantive_tool_calls()` / `acted()` 的语义一个字
不动，因为 `peer/delegation.py` 用它们核验远端委派真的做了事。

Run: cd agent-core && python3 -m pytest tests/test_subagent_deliverable.py
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from subagent.protocol import (  # noqa: E402
    BOOKKEEPING_TOOLS, STATUS_COMPLETED, SubagentResult, SubagentSpec,
)


class TestSpecField(unittest.TestCase):
    def test_default_is_action(self):
        """不声明就是「该干点什么」—— 什么都不做的任务应当自己声明这件事。"""
        self.assertEqual(SubagentSpec(goal='x').deliverable, 'action')

    def test_deliverable_survives_a_round_trip(self):
        """peer 委派和 checkpoint 都走 to_dict/from_dict。

        `to_dict` 是逐字段列举的，不是遍历 dataclass —— 漏掉新字段的话，它会在
        跨机器委派或重启续跑时被静默丢掉，而默认值 'action' 恰好会让告警回来。
        """
        spec = SubagentSpec(goal='watch sensors', deliverable='report')
        self.assertEqual(spec.to_dict()['deliverable'], 'report')
        self.assertEqual(SubagentSpec.from_dict(spec.to_dict()).deliverable, 'report')

    def test_an_old_dict_without_the_field_still_loads(self):
        """升级前 checkpoint 下来的 spec 里没有这个键。"""
        old = {'goal': 'x', 'priority': 2, 'max_rounds': 10}
        self.assertEqual(SubagentSpec.from_dict(old).deliverable, 'action')


class TestTheStrictSemanticsAreUnchanged(unittest.TestCase):
    """delegation 靠这两个函数判断远端到底做没做事，它们必须保持严格。"""

    def _result(self, tools):
        return SubagentResult(agent_id='a', status=STATUS_COMPLETED, output='done',
                              tool_calls_made=[{'name': t} for t in tools])

    def test_report_only_is_still_not_substantive(self):
        self.assertEqual(self._result(['subagent_report', 'subagent_finish'])
                         .substantive_tool_calls(), [])

    def test_report_only_still_counts_as_not_acted(self):
        self.assertFalse(self._result(['subagent_report', 'subagent_finish']).acted())

    def test_a_real_tool_is_substantive(self):
        self.assertEqual(self._result(['mcp__dev__tts', 'subagent_finish'])
                         .substantive_tool_calls(), ['mcp__dev__tts'])

    def test_bookkeeping_set_is_unchanged(self):
        self.assertEqual(BOOKKEEPING_TOOLS,
                         {'subagent_finish', 'subagent_fail', 'subagent_report'})


class TestWarningScope(unittest.TestCase):
    """告警条件本身 —— 直接对着 agent.py 里那个判据。"""

    @staticmethod
    def _would_warn(spec, tools):
        result = SubagentResult(agent_id='a', status=STATUS_COMPLETED, output='',
                                tool_calls_made=[{'name': t} for t in tools])
        # 与 subagent/agent.py 中的条件保持一致
        return (not result.substantive_tool_calls()
                and getattr(spec, 'deliverable', 'action') != 'report')

    def test_a_monitor_that_only_reported_does_not_warn(self):
        spec = SubagentSpec(goal='[bg] 后台监控', deliverable='report')
        self.assertFalse(self._would_warn(spec, ['subagent_report', 'subagent_finish']))

    def test_a_monitor_that_stayed_silent_does_not_warn(self):
        """「无显著变化 → subagent_finish」是 goal 里明写的正确行为。"""
        spec = SubagentSpec(goal='[bg] 后台监控', deliverable='report')
        self.assertFalse(self._would_warn(spec, ['subagent_finish']))

    def test_a_task_that_did_nothing_still_warns(self):
        """就是这个形状让两台机器人宣布了一场没发生的演出。"""
        spec = SubagentSpec(goal='让对方说一句话')
        self.assertTrue(self._would_warn(spec, ['subagent_finish']))

    def test_a_task_that_acted_does_not_warn(self):
        spec = SubagentSpec(goal='让对方说一句话')
        self.assertFalse(self._would_warn(spec, ['mcp__dev__tts', 'subagent_finish']))

    def test_a_spec_from_before_the_field_existed_still_warns(self):
        """getattr 的兜底：旧 checkpoint 恢复出来的对象可能没有这个属性。"""
        class _Old:
            goal = 'x'
        self.assertTrue(self._would_warn(_Old(), ['subagent_finish']))


class TestBackgroundMonitorIsTaggedReport(unittest.TestCase):
    def test_collector_builds_its_monitor_as_a_report(self):
        """真正的修复点：那个 spec 得真的带上 deliverable='report'。"""
        import collector
        src = pathlib.Path(collector.__file__).read_text(encoding='utf-8')
        spec_block = src[src.index('spec = SubagentSpec('):]
        spec_block = spec_block[:spec_block.index('\n        )')]
        self.assertIn("deliverable='report'", spec_block)
        # 同一个 spec 里 tool_deny 挡掉了所有会动的工具 —— 这正是为什么它只可能
        # 调用记账类工具，两者必须同时成立。
        self.assertIn('tool_deny', spec_block)


if __name__ == '__main__':
    unittest.main()
