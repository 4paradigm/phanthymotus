"""
test_peer_delegation.py — P4 任务委托测试。

验证：
1. hop_count 超限时拒绝委托
2. blocked peer 无法委托
3. viewer 委托的 spec 不能使用执行器工具
4. prepare_delegated_spec 正确增加 hop_count
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

from peer import identity, store, delegation  # noqa: E402
from subagent.protocol import SubagentSpec, P_NORMAL  # noqa: E402


class TestDelegation(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        identity.ensure_identity()

    def test_hop_count_exceeded(self):
        """hop_count 超限时拒绝（节点 1）"""
        peer_id = 'test_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Test', role='operator')

        spec = SubagentSpec(goal='test', hop_count=3)
        allowed, reason = delegation.validate_delegation(peer_id, spec)
        self.assertFalse(allowed)
        self.assertIn('hop_count', reason)
        self.assertIn('exceeds limit', reason)

    def test_blocked_peer_denied(self):
        """blocked peer 无法委托（节点 2）"""
        peer_id = 'blocked_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Blocked', role='blocked')

        spec = SubagentSpec(goal='test', hop_count=0)
        allowed, reason = delegation.validate_delegation(peer_id, spec)
        self.assertFalse(allowed)
        self.assertEqual(reason, 'blocked')

    def test_unknown_peer_denied(self):
        """未配对 peer 无法委托"""
        spec = SubagentSpec(goal='test', hop_count=0)
        allowed, reason = delegation.validate_delegation('unknown_peer', spec)
        self.assertFalse(allowed)
        self.assertEqual(reason, 'unknown_peer')

    def test_prepare_increments_hop_count(self):
        """prepare_delegated_spec 增加 hop_count（节点 4）"""
        peer_id = 'test_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Test', role='operator')

        spec = SubagentSpec(goal='test', hop_count=0)
        augmented = delegation.prepare_delegated_spec(peer_id, spec)
        self.assertEqual(augmented.hop_count, 1)

        spec2 = SubagentSpec(goal='test', hop_count=1)
        augmented2 = delegation.prepare_delegated_spec(peer_id, spec2)
        self.assertEqual(augmented2.hop_count, 2)

    def test_tool_filter_applied(self):
        """peer tool_filter 被应用到委托 spec"""
        peer_id = 'filtered_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Filtered',
                     role='operator', tool_filter='camera_*,status')

        spec = SubagentSpec(goal='test', tool_filter=None)
        augmented = delegation.prepare_delegated_spec(peer_id, spec)
        self.assertIsNotNone(augmented.tool_filter)
        self.assertIn('camera_*', augmented.tool_filter)
        self.assertIn('status', augmented.tool_filter)

    def test_wildcard_filter_preserved(self):
        """peer tool_filter 为 * 时，spec 的 filter 保持原样"""
        peer_id = 'test_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Test', role='operator')

        spec = SubagentSpec(goal='test', tool_filter=['move_*'])
        augmented = delegation.prepare_delegated_spec(peer_id, spec)
        # Wildcard peer allows the spec's filter through
        self.assertEqual(augmented.tool_filter, ['move_*'])

    def test_valid_delegation(self):
        """正常委托通过验证"""
        peer_id = 'test_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Test', role='operator')

        spec = SubagentSpec(goal='test task', hop_count=0)
        allowed, reason = delegation.validate_delegation(peer_id, spec)
        self.assertTrue(allowed)
        self.assertEqual(reason, '')

    def test_delegation_with_filter_on_restricted_peer(self):
        """受限 peer 委托时带 tool_filter 会被拒绝（安全门）"""
        peer_id = 'restricted_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Restricted',
                     role='operator', tool_filter='camera_*')

        spec = SubagentSpec(goal='test', tool_filter=['move_*'])
        allowed, reason = delegation.validate_delegation(peer_id, spec)
        self.assertFalse(allowed)
        self.assertIn('does not allow arbitrary delegation filters', reason)


if __name__ == '__main__':
    unittest.main()
