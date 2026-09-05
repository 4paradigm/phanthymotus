"""
test_delegated_subagent_scope.py — 被委派的 subagent 不该碰委派方的硬件，也不该唤醒本机主 agent。

Orin5+Orin6 实测出的两个问题，都出在"替对端干活"这个身份没有被区别对待。

**开头混乱**：Orin6 收到"你来当捧哏"的委派后，它的 subagent 选了
`mcp__peer:dd398c73177a__tts` —— **Orin5 的嘴** —— 去说自己的台词，于是 Orin5 的扬声器
说出"你好Orin5，我是Orin6"。自己的 tts 和对端的 tts 在同一张扁平工具表里，描述几乎一样，
它没有任何依据分辨。这既是信任反转（委派方的执行器动了，而委派方的 agent 没参与决定），
也直接听起来像一台机器人在复述另一台的话。

**重复说话**：subagent 说完一句、finish，完成事件唤醒本机主 agent，而通知文本里的摘要
复述了刚说完的台词，于是主 agent 又说一遍：

    16:18:00.966  [subagent] tts("那你说得好的时候再来看。")
    16:18:03.077  subagent_finish
    16:18:05.736  [main]     tts("那你说得好的时候再来看。")   ← 同一句第二遍

六句捧哏台词全部说了两遍。这活是对端要的，结果已经通过 /api/peer/delegate 的响应回去了，
本机主 agent 从没提出请求，也没有什么要决定。
"""
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402
from subagent.agent import Subagent  # noqa: E402
from subagent.protocol import SubagentSpec  # noqa: E402

LOCAL_MCP = 'mcp-1783770461'
PEER_MCP = 'peer:dd398c73177a'


class TestDelegatedSubagentToolScope(unittest.TestCase):
    def setUp(self):
        self._saved = dict(mcp_client.registry)
        mcp_client.registry.clear()
        mcp_client.registry[LOCAL_MCP] = {
            'online': True,
            'schemas': {f'mcp__{LOCAL_MCP}__tts': {'name': f'mcp__{LOCAL_MCP}__tts',
                                                   'description': 'TTS'}},
        }
        mcp_client.registry[PEER_MCP] = {
            'online': True, 'transport': 'peer',
            'schemas': {f'mcp__{PEER_MCP}__tts': {'name': f'mcp__{PEER_MCP}__tts',
                                                  'description': '[Orin5] TTS'}},
        }

    def tearDown(self):
        mcp_client.registry.clear()
        mcp_client.registry.update(self._saved)

    def _names(self, hop_count):
        agent = Subagent(SubagentSpec(goal='说一句捧哏词', hop_count=hop_count),
                         agent_id='scope01')
        return {s.get('name') for s in agent._get_all_mcp_schemas()}

    def test_delegated_subagent_cannot_reach_the_peer(self):
        """回归：这正是 Orin5 的嘴说出 Orin6 台词的原因。"""
        names = self._names(hop_count=1)
        self.assertNotIn(f'mcp__{PEER_MCP}__tts', names)

    def test_delegated_subagent_keeps_its_own_tools(self):
        names = self._names(hop_count=1)
        self.assertIn(f'mcp__{LOCAL_MCP}__tts', names)

    def test_locally_spawned_subagent_still_sees_peers(self):
        """本机自己起的 subagent 是本机 agent 的延伸，行为不变。"""
        names = self._names(hop_count=0)
        self.assertIn(f'mcp__{PEER_MCP}__tts', names)
        self.assertIn(f'mcp__{LOCAL_MCP}__tts', names)

    def test_deeper_hops_are_also_restricted(self):
        self.assertNotIn(f'mcp__{PEER_MCP}__tts', self._names(hop_count=2))

    def test_offline_tools_are_excluded_regardless(self):
        mcp_client.registry[LOCAL_MCP]['online'] = False
        self.assertEqual(self._names(hop_count=1), set())


class TestDelegatedCompletionDoesNotWakeLocalAgent(unittest.TestCase):
    """完成通知只在本机有决策需求时才发。

    直接测那个判定函数，而不是跑 `_finalize` —— 后者还要做 history/DB 记账，桩不全就会
    在到达 event_bus 之前抛异常，于是"没有通知"这个断言会因为错误的原因通过。一条会假
    通过的测试比没有更糟。
    """

    def _reason(self, hop_count, status='completed', is_bg=False):
        from subagent.manager import notify_suppression_reason
        from subagent.protocol import SubagentResult
        spec = SubagentSpec(goal='说一句捧哏词', hop_count=hop_count)
        result = SubagentResult(
            agent_id='fin01', status=status,
            output='已完成第四句捧哏台词播报。台词：那你说得好的时候再来看。',
        )
        return notify_suppression_reason(spec, result, is_bg)

    def test_delegated_completion_is_suppressed(self):
        """回归：这条唤醒让主 agent 把 subagent 刚说的台词又说了一遍。"""
        reason = self._reason(hop_count=1)
        self.assertIsNotNone(reason)
        self.assertIn('delegated', reason)

    def test_local_completion_still_notifies(self):
        self.assertIsNone(self._reason(hop_count=0),
                          '本机自己起的 subagent 完成后仍要通知主 agent')

    def test_delegated_failure_still_notifies(self):
        """失败要说 —— 本机操作者需要知道有活没干成。"""
        self.assertIsNone(self._reason(hop_count=1, status='failed'))

    def test_delegated_timeout_still_notifies(self):
        self.assertIsNone(self._reason(hop_count=1, status='timeout'))

    def test_delegated_cancel_still_notifies(self):
        self.assertIsNone(self._reason(hop_count=1, status='cancelled'))

    def test_bg_behaviour_unchanged(self):
        self.assertEqual(self._reason(hop_count=0, is_bg=True), 'bg')

    def test_bg_failure_still_notifies(self):
        self.assertIsNone(self._reason(hop_count=0, status='failed', is_bg=True))

    def test_deeper_hops_suppressed_too(self):
        self.assertIsNotNone(self._reason(hop_count=2))

    def test_missing_result_notifies(self):
        from subagent.manager import notify_suppression_reason
        self.assertIsNone(notify_suppression_reason(
            SubagentSpec(goal='g', hop_count=1), None, False))

    def test_finalize_consults_the_predicate(self):
        """接线检查：判定函数存在但没被 _finalize 用上，等于没修。"""
        src = (pathlib.Path(__file__).resolve().parents[1]
               / 'src' / 'subagent' / 'manager.py').read_text()
        body = src.split('def _finalize', 1)[1]
        self.assertIn('notify_suppression_reason', body)


if __name__ == '__main__':
    unittest.main()
