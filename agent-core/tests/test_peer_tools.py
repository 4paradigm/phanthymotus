"""
test_peer_tools.py — P2 工具代理测试。

验证：
1. viewer 被拒绝调用执行器工具
2. operator 可以调用所有工具
3. tool_filter glob 正确过滤工具列表
4. 未配对 peer 无法调用任何工具
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

from peer import identity, store, tools  # noqa: E402


class TestToolProxy(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        identity.ensure_identity()

    def test_viewer_denied_actuator(self):
        """viewer 不能调用执行器工具（节点 1）"""
        peer_id = 'test_viewer'
        store.upsert(peer_id, identity.public_key_b64(), 'Viewer', role='viewer')

        allowed, reason = tools.check_tool_permission(peer_id, 'move_forward')
        self.assertFalse(allowed)
        self.assertIn('operator role', reason)

        allowed, reason = tools.check_tool_permission(peer_id, 'grasp_object')
        self.assertFalse(allowed)

    def test_viewer_allowed_sensor(self):
        """viewer 可以调用传感器/查询工具"""
        peer_id = 'test_viewer'
        store.upsert(peer_id, identity.public_key_b64(), 'Viewer', role='viewer')

        allowed, reason = tools.check_tool_permission(peer_id, 'get_status')
        self.assertTrue(allowed)
        self.assertEqual(reason, '')

        allowed, reason = tools.check_tool_permission(peer_id, 'camera_capture')
        self.assertTrue(allowed)

    def test_operator_allowed_all(self):
        """operator 可以调用所有工具（节点 2）"""
        peer_id = 'test_operator'
        store.upsert(peer_id, identity.public_key_b64(), 'Operator', role='operator')

        allowed, reason = tools.check_tool_permission(peer_id, 'move_forward')
        self.assertTrue(allowed)

        allowed, reason = tools.check_tool_permission(peer_id, 'get_status')
        self.assertTrue(allowed)

    def test_tool_filter_glob(self):
        """tool_filter glob 正确过滤（节点 3）"""
        peer_id = 'test_filtered'
        store.upsert(peer_id, identity.public_key_b64(), 'Filtered',
                     role='operator', tool_filter='camera_*,status')

        allowed, reason = tools.check_tool_permission(peer_id, 'camera_capture')
        self.assertTrue(allowed)

        allowed, reason = tools.check_tool_permission(peer_id, 'status')
        self.assertTrue(allowed)

        allowed, reason = tools.check_tool_permission(peer_id, 'move_forward')
        self.assertFalse(allowed)
        self.assertIn('not in filter', reason)

    def test_unknown_peer_denied(self):
        """未配对 peer 无法调用（节点 4）"""
        allowed, reason = tools.check_tool_permission('unknown_peer', 'any_tool')
        self.assertFalse(allowed)
        self.assertEqual(reason, 'unknown_peer')

    def test_blocked_peer_denied(self):
        """blocked peer 无法调用"""
        peer_id = 'blocked_peer'
        store.upsert(peer_id, identity.public_key_b64(), 'Blocked', role='blocked')

        allowed, reason = tools.check_tool_permission(peer_id, 'get_status')
        self.assertFalse(allowed)
        self.assertEqual(reason, 'blocked')

    def test_filter_schemas(self):
        """filter_schemas 返回过滤后的工具列表"""
        peer_id = 'test_viewer'
        store.upsert(peer_id, identity.public_key_b64(), 'Viewer', role='viewer')

        mock_schemas = [
            {'name': 'get_status', 'description': 'Read status'},
            {'name': 'move_forward', 'description': 'Move robot'},
            {'name': 'camera_capture', 'description': 'Take photo'},
        ]
        filtered = tools.filter_schemas(peer_id, mock_schemas)
        names = {s['name'] for s in filtered}
        self.assertIn('get_status', names)
        self.assertIn('camera_capture', names)
        self.assertNotIn('move_forward', names)

    def test_filter_schemas_with_glob(self):
        """filter_schemas 结合 tool_filter"""
        peer_id = 'test_filtered'
        store.upsert(peer_id, identity.public_key_b64(), 'Filtered',
                     role='operator', tool_filter='camera_*')

        mock_schemas = [
            {'name': 'camera_capture', 'description': 'Take photo'},
            {'name': 'camera_stream', 'description': 'Start stream'},
            {'name': 'move_forward', 'description': 'Move robot'},
        ]
        filtered = tools.filter_schemas(peer_id, mock_schemas)
        names = {s['name'] for s in filtered}
        self.assertEqual(names, {'camera_capture', 'camera_stream'})


if __name__ == '__main__':
    unittest.main()
