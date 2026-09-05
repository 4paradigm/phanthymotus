"""
test_peer_delegation_verified.py — 委派结果必须可验证，而不是相信对端的自述。

`peer_delegate` 原来返回的是远端 subagent 自己写的散文（`result.get('output')`）。
Orin5+Orin6 实测：六个委派各要求对端说一句台词，四个只跑一轮、没调 tts 就返回
"已完成捧哏台词播报" —— 两台机器随后都宣布完成了一场十六句的表演，一句都没播，
其中一台还写进了长期记忆。

这里覆盖三件事：
1. 零动作 + 零实质工具调用 → 明确告诉 LLM 任务**没**完成
2. 只有 'completed' 的 action 算数；'timeout' 尤其不算（它清掉 pending 并照常放行）
3. 旧版 peer 不带这些字段 → 说"无法确认"，既不谎报成功也不谎报失败

以及 Phase 4 的成对再入守卫：hop_count 拦不住 A→B→A。
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import delegation, identity, store  # noqa: E402
from subagent.protocol import SubagentSpec  # noqa: E402

PEER = 'verify_peer'
ENDPOINT = 'https://192.0.2.11:15678'


def _delegate(response: dict) -> str:
    async def _fake_post(endpoints, path, payload, **kw):
        return response, ''
    with mock.patch('peer.transport.post_json', _fake_post), \
         mock.patch('peer.registry.registry.endpoints_for', lambda p: [ENDPOINT]):
        return asyncio.run(delegation.peer_delegate(PEER, 'say one line'))


class TestOutcomeIsVerified(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store.upsert(PEER, identity.public_key_b64(), 'Orin6', role='operator',
                     endpoints=[ENDPOINT])

    def test_success_with_no_action_is_reported_as_not_done(self):
        """正是 Orin6 那四个的形状：一轮结束、只调了 subagent_finish。"""
        out = _delegate({
            'status': 'completed',
            'output': '已完成捧哏台词播报。通过 TTS 说了"进步了嘛。"',
            'actions': [],
            'substantive_tool_calls': [],
        })
        self.assertIn('NOT done', out)
        self.assertIn('no verifiable action', out)
        # 对端的原话要保留，供 LLM 判断，但不能被当成结论
        self.assertIn('已完成捧哏台词播报', out)

    def test_completed_action_counts(self):
        out = _delegate({
            'status': 'completed', 'output': '说完了',
            'actions': [{'action_id': 'speak-1', 'status': 'completed', 'tool': 'tts'}],
            'substantive_tool_calls': ['mcp__m__tts'],
        })
        self.assertIn('1 action(s) confirmed complete', out)
        self.assertNotIn('NOT done', out)

    def test_timeout_action_does_not_count_as_done(self):
        """barrier 超时会清 pending 并照常放行 —— 下游看起来和成功一样。"""
        out = _delegate({
            'status': 'completed', 'output': '说完了',
            'actions': [{'action_id': 'speak-1', 'status': 'timeout', 'tool': 'tts'}],
            'substantive_tool_calls': ['mcp__m__tts'],
        })
        self.assertIn('did NOT confirm', out)
        self.assertIn('speak-1=timeout', out)

    def test_partial_completion_reports_both_halves(self):
        out = _delegate({
            'status': 'completed', 'output': 'x',
            'actions': [
                {'action_id': 'speak-1', 'status': 'completed', 'tool': 'tts'},
                {'action_id': 'speak-2', 'status': 'cancelled', 'tool': 'tts'},
            ],
            'substantive_tool_calls': ['mcp__m__tts'],
        })
        self.assertIn('1 action(s) confirmed complete', out)
        self.assertIn('1 did NOT confirm', out)

    def test_pending_action_does_not_count(self):
        """run 结束时还没落地的 action —— 不能算做完。

        fixture 带上 tts 调用：一个 action 不可能凭空出现，起它必然调过工具。
        """
        out = _delegate({
            'status': 'completed', 'output': 'x',
            'actions': [{'action_id': 'speak-1', 'status': 'pending', 'tool': 'tts'}],
            'substantive_tool_calls': ['mcp__m__tts'],
        })
        self.assertIn('did NOT confirm', out)
        self.assertIn('speak-1=pending', out)

    def test_pending_action_with_no_tool_call_is_the_stronger_verdict(self):
        """两个证据都缺 → 直接说没做，而不是只说"未确认"。"""
        out = _delegate({
            'status': 'completed', 'output': 'x',
            'actions': [{'action_id': 'speak-1', 'status': 'pending', 'tool': 'tts'}],
            'substantive_tool_calls': [],
        })
        self.assertIn('NOT done', out)

    def test_substantive_tool_without_acp_still_counts(self):
        """不是每个工具都走 ACP；调了真工具就不该报"什么都没做"。"""
        out = _delegate({
            'status': 'completed', 'output': 'read the file',
            'actions': [],
            'substantive_tool_calls': ['Read'],
        })
        self.assertNotIn('NOT done', out)
        self.assertIn('tools used: Read', out)

    def test_older_peer_is_unverified_not_failed(self):
        """缺字段 ≠ 空字段。"""
        out = _delegate({'status': 'completed', 'output': 'ok'})
        self.assertIn('unverified', out)
        self.assertNotIn('NOT done', out)

    def test_non_completed_status_unchanged(self):
        out = _delegate({'status': 'timeout', 'error': 'idle'})
        self.assertIn('did not complete', out)


class TestReentrancyGuard(unittest.TestCase):
    """A→B→A：hop_count 拦不住第一条回程腿。

    A 发 hop=0；B 的 subagent 在 hop=1 反向委派，发 hop=1；A 收到 1 ≤ 2 放行。
    """

    @classmethod
    def setUpClass(cls):
        store.upsert('rp', identity.public_key_b64(), 'RP', role='operator',
                     endpoints=[ENDPOINT])

    def tearDown(self):
        delegation._outbound_in_flight.clear()

    def test_inbound_refused_while_outbound_in_flight(self):
        delegation._outbound_in_flight['rp'] = 1
        ok, why = delegation.validate_delegation('rp', SubagentSpec(goal='x', hop_count=1))
        self.assertFalse(ok)
        self.assertIn('reentrant', why)

    def test_inbound_allowed_when_nothing_in_flight(self):
        ok, why = delegation.validate_delegation('rp', SubagentSpec(goal='x', hop_count=1))
        self.assertTrue(ok, why)

    def test_hop_count_alone_would_have_allowed_it(self):
        """说明这条守卫不是多余的。"""
        self.assertLessEqual(1, delegation.MAX_HOP_COUNT)

    def test_counter_survives_concurrent_delegations(self):
        """两个 subagent 同时委派给同一个 peer；先回来的那个不能清掉另一个的标记。"""
        delegation._outbound_in_flight['rp'] = 2
        ok, _ = delegation.validate_delegation('rp', SubagentSpec(goal='x'))
        self.assertFalse(ok)

    def test_in_flight_is_cleared_after_delegation(self):
        async def _fake_post(endpoints, path, payload, **kw):
            self.assertEqual(delegation.outbound_in_flight('rp'), 1,
                             'in-flight marker must be set while the call is open')
            return {'status': 'completed', 'output': 'ok'}, ''
        with mock.patch('peer.transport.post_json', _fake_post), \
             mock.patch('peer.registry.registry.endpoints_for', lambda p: [ENDPOINT]):
            asyncio.run(delegation.peer_delegate('rp', 'g'))
        self.assertEqual(delegation.outbound_in_flight('rp'), 0)

    def test_in_flight_is_cleared_on_failure(self):
        async def _fake_post(endpoints, path, payload, **kw):
            raise RuntimeError('link died')
        with mock.patch('peer.transport.post_json', _fake_post), \
             mock.patch('peer.registry.registry.endpoints_for', lambda p: [ENDPOINT]):
            with self.assertRaises(RuntimeError):
                asyncio.run(delegation.peer_delegate('rp', 'g'))
        self.assertEqual(delegation.outbound_in_flight('rp'), 0,
                         'a failed delegation must not wedge the guard shut forever')

    def test_local_cancel_is_reported_without_claiming_nothing_happened(self):
        async def _fake_post(endpoints, path, payload, **kw):
            return None, 'cancelled'
        with mock.patch('peer.transport.post_json', _fake_post), \
             mock.patch('peer.registry.registry.endpoints_for', lambda p: [ENDPOINT]):
            out = asyncio.run(delegation.peer_delegate('rp', 'g'))
        self.assertIn('interrupted locally', out)
        self.assertIn('may', out)   # 对端可能仍在执行，不能断言它没做


if __name__ == '__main__':
    unittest.main()
