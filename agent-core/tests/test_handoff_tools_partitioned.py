"""
test_handoff_tools_partitioned.py — 每一个"让别人动手"的系统工具都必须被 barrier 挡住。

`_needs_barrier` 只看 `mcp__` 前缀的工具，所以 peer_* / subagent_* 这些系统工具**只有**
出现在 `_ACP_BARRIER_SYSTEM_TOOLS` 里才会被拦。漏一个就等于开了一扇没上锁的门。

这条测试的存在是因为真漏过：第一版只加了 `peer_delegate` 和 `subagent_spawn`，而 Orin5
实测那次相声走的是 **`peer_call`** —— 于是每轮都在自己还在说话时就让 Orin6 开口，
间隔 2.7 / 1.7 / 2.9 / 2.5 / 2.6 秒，而自己那句要播 5–14 秒。

按名字猜工具属性是这个仓库栽过的老坑（一份关键词表漏掉了 loco / led / speaker /
switch_mode，让 viewer 能驱动底盘）。所以这里强制**穷举**：任何新增的 peer_* /
subagent_* 工具必须显式落到"会让人动手"或"只读"两边之一，否则这条测试失败。
"""
import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


def _registered_handoff_tool_names() -> set:
    """peer_* / subagent_* 系统工具名，从注册表那段源码里抓。

    读源码而不是构造 Event 实例：注册发生在 __init__ 里，会连带拉起 MCP 连接、
    subagent manager 和 ROS2，测试环境里跑不起来。
    """
    text = (SRC / 'event' / 'llm.py').read_text()
    return set(re.findall(r"\(\s*'((?:peer|subagent)_[a-z_]+)'\s*,", text))


class TestHandoffToolsArePartitioned(unittest.TestCase):
    def setUp(self):
        # `from event import llm` returns an Event *instance* — event/__init__.py
        # rebinds each submodule name to a constructed object. Import the names.
        from event.llm import (
            _HANDOFF_SYSTEM_TOOLS, _READ_ONLY_HANDOFF_TOOLS,
            _sys_tool_needs_barrier,
        )
        self.handoff = _HANDOFF_SYSTEM_TOOLS
        self.read_only = _READ_ONLY_HANDOFF_TOOLS
        self.needs_barrier = _sys_tool_needs_barrier
        self.registered = _registered_handoff_tool_names()

    def test_the_scrape_found_the_tools_at_all(self):
        """先确认抓取本身有效 —— 否则下面几条会空转通过。"""
        self.assertGreaterEqual(len(self.registered), 10,
                                f'只抓到 {self.registered}，注册表写法可能变了')
        for expected in ('peer_call', 'peer_delegate', 'subagent_spawn'):
            self.assertIn(expected, self.registered)

    def test_every_tool_is_classified(self):
        classified = self.handoff | self.read_only
        unclassified = self.registered - classified
        self.assertEqual(unclassified, set(),
                         f'新增的 peer_*/subagent_* 工具未分类: {sorted(unclassified)} —— '
                         f'它会让别人动手吗？是就加进 _HANDOFF_SYSTEM_TOOLS，'
                         f'不是就加进 _READ_ONLY_HANDOFF_TOOLS。')

    def test_the_two_sets_do_not_overlap(self):
        self.assertEqual(
            self.handoff & self.read_only, set())

    def test_classified_tools_actually_exist(self):
        """分类里不该留下已删除工具的名字。"""
        stale = (self.handoff
                 | self.read_only) - self.registered
        self.assertEqual(stale, set(), f'已不存在的工具名: {sorted(stale)}')

    def test_handoff_tools_are_barriered(self):
        for name in self.handoff:
            self.assertTrue(self.needs_barrier(name), name)

    def test_read_only_tools_are_not_barriered(self):
        """查状态不该等一整段音频播完。"""
        for name in self.read_only:
            self.assertFalse(self.needs_barrier(name), name)

    def test_peer_call_specifically(self):
        """回归：这一个漏掉，就是实测那次相声重叠的直接原因。"""
        self.assertTrue(self.needs_barrier('peer_call'))

    def test_finish_still_barriered(self):
        self.assertTrue(self.needs_barrier('finish'))

    def test_a_plain_tool_is_not_barriered(self):
        for name in ('Bash', 'Read', 'WebSearch', 'memory_recall'):
            self.assertFalse(self.needs_barrier(name), name)


class TestNonMcpToolsBypassNeedsBarrier(unittest.TestCase):
    """说明为什么上面那张表是唯一的防线。"""

    def test_needs_barrier_ignores_system_tools(self):
        from event.llm import _needs_barrier
        for name in ('peer_call', 'peer_delegate', 'finish'):
            needed, want = _needs_barrier(name, {})
            self.assertFalse(needed, f'{name}: _needs_barrier 只管 mcp__ 工具')
            self.assertIsNone(want)


if __name__ == '__main__':
    unittest.main()
