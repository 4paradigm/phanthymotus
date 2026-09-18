"""
test_peer_tools.py — P2 工具代理测试。

验证：
1. viewer 被拒绝调用执行器工具
2. operator 可以调用所有工具
3. tool_filter glob 正确过滤工具列表
4. 未配对 peer 无法调用任何工具

fixture 原来用的是裸工具名（'get_status'、'move_forward'），直到授权规则改成从 MCP
registry 读工具声明的 type 和层。产品里不存在裸名字 —— 每个工具都是
`mcp__<mcp_id>__<tool>`，而 registry 里查不到的名字现在一律算"会动"（fail closed）。
用产品从不产出的名字做测试，正是之前那个缺口藏身的方式：按名字猜关键词的实现能通过
这些测试，却放行了 viewer 去驱动 loco。这里改成真实形状 + registry 打桩；分类逻辑本身
在 test_peer_tool_authorization.py 里覆盖。
"""
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ['DB_PATH'] = os.path.join(tempfile.mkdtemp(), 'test.db')

from peer import identity, store, tools  # noqa: E402


MCP_ID = 'mcp-test'
SENSOR = f'mcp__{MCP_ID}__get_status'
CAMERA = f'mcp__{MCP_ID}__camera_capture'
ACTUATOR = f'mcp__{MCP_ID}__move_forward'

# Shaped like a real driver's registry entry: the type is declared per tool, which
# is what peer/tools.py reads instead of guessing from the name.
_REGISTRY = {
    MCP_ID: {
        'category': 'driver',
        'tool_meta': {
            SENSOR:   {'type': 'sensor'},
            CAMERA:   {'type': 'sensor'},
            ACTUATOR: {'type': 'actuator'},
        },
    },
}


class TestToolProxy(unittest.TestCase):
    def setUp(self):
        identity.reset_cache()
        identity.ensure_identity()
        import mcp_client
        patch = mock.patch.dict(mcp_client.registry, _REGISTRY, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_viewer_denied_actuator(self):
        """viewer 不能调用执行器工具（节点 1）"""
        peer_id = 'test_viewer'
        store.upsert(peer_id, identity.public_key_b64(), 'Viewer', role='viewer')

        allowed, reason = tools.check_tool_permission(peer_id, ACTUATOR)
        self.assertFalse(allowed)
        self.assertIn('viewer', reason)

        allowed, reason = tools.check_tool_permission(peer_id, 'grasp_object')
        self.assertFalse(allowed)

    def test_viewer_allowed_sensor(self):
        """viewer 可以调用传感器/查询工具"""
        peer_id = 'test_viewer'
        store.upsert(peer_id, identity.public_key_b64(), 'Viewer', role='viewer')

        allowed, reason = tools.check_tool_permission(peer_id, SENSOR)
        self.assertTrue(allowed)
        self.assertEqual(reason, '')

        allowed, reason = tools.check_tool_permission(peer_id, CAMERA)
        self.assertTrue(allowed)

    def test_operator_allowed_all(self):
        """operator 可以调用所有工具（节点 2）"""
        peer_id = 'test_operator'
        store.upsert(peer_id, identity.public_key_b64(), 'Operator', role='operator')

        allowed, reason = tools.check_tool_permission(peer_id, ACTUATOR)
        self.assertTrue(allowed)

        allowed, reason = tools.check_tool_permission(peer_id, SENSOR)
        self.assertTrue(allowed)

    def test_tool_filter_glob(self):
        """tool_filter glob 正确过滤（节点 3）"""
        peer_id = 'test_filtered'
        store.upsert(peer_id, identity.public_key_b64(), 'Filtered',
                     role='operator', tool_filter='camera_*,get_status')

        allowed, reason = tools.check_tool_permission(peer_id, CAMERA)
        self.assertTrue(allowed)

        allowed, reason = tools.check_tool_permission(peer_id, SENSOR)
        self.assertTrue(allowed, '短名 pattern 必须能匹配到全名工具')

        allowed, reason = tools.check_tool_permission(peer_id, ACTUATOR)
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

        allowed, reason = tools.check_tool_permission(peer_id, SENSOR)
        self.assertFalse(allowed)
        self.assertEqual(reason, 'blocked')

    def test_filter_schemas(self):
        """filter_schemas 返回过滤后的工具列表"""
        peer_id = 'test_viewer'
        store.upsert(peer_id, identity.public_key_b64(), 'Viewer', role='viewer')

        mock_schemas = [
            {'name': SENSOR, 'description': 'Read status'},
            {'name': ACTUATOR, 'description': 'Move robot'},
            {'name': CAMERA, 'description': 'Take photo'},
        ]
        filtered = tools.filter_schemas(peer_id, mock_schemas)
        names = {s['name'] for s in filtered}
        self.assertIn(SENSOR, names)
        self.assertIn(CAMERA, names)
        self.assertNotIn(ACTUATOR, names)

    def test_filter_schemas_with_glob(self):
        """filter_schemas 结合 tool_filter"""
        peer_id = 'test_filtered'
        store.upsert(peer_id, identity.public_key_b64(), 'Filtered',
                     role='operator', tool_filter='*camera*')

        mock_schemas = [
            {'name': CAMERA, 'description': 'Take photo'},
            {'name': 'camera_stream', 'description': 'Start stream'},
            {'name': ACTUATOR, 'description': 'Move robot'},
        ]
        filtered = tools.filter_schemas(peer_id, mock_schemas)
        names = {s['name'] for s in filtered}
        self.assertEqual(names, {CAMERA, 'camera_stream'})


if __name__ == '__main__':
    unittest.main()
