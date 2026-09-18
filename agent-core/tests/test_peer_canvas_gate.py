"""
test_peer_canvas_gate.py — 画布绑定门必须覆盖 peer 工具代理。

真机上发现的安全缺口：把 Orin5 提升为 operator 后，它通过
POST /api/peer/tools/call 直接调用了一个**未在画布上绑定**的真实工具并成功
返回。

README 与 CLAUDE.md 都承诺执行器需要同时满足三个条件（角色、画布绑定、本机
LLM 决策），但 /tools/call 是绕过 LLM 的直连代理：当时它只做了角色检查，然后
直接进 mcp_client.call_tool。而我在那个函数的 docstring 里写着"double gate
still applies ... happens in mcp_client.call_tool"——**那句话是错的**，
mcp_client.call_tool 不做任何画布检查。

后果：operator peer 可以直接驱动任意在 tool_filter 内的执行器，一道门都不过。

之前所有测试都用 'move_forward' 这个**并不存在**的工具名，被角色门拦下就通过了，
所以从未触达第二道门。
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import config  # noqa: E402
import canvas_binding  # noqa: E402


def _layout(*pairs):
    """A canvas with an agentcore card wired to the given (mcp_id, tool) pairs."""
    return {
        'cards': [{'id': 'core-1', 'mcpId': 'agentcore'},
                  {'id': 'dev-1', 'mcpId': 'mcp-x'}],
        'execConnections': [
            {'fromCardId': 'core-1', 'toMcpId': m, 'toToolName': t, 'toCardId': 'dev-1'}
            for m, t in pairs
        ],
    }


class TestCanvasGate(unittest.TestCase):
    def setUp(self):
        config.main['canvas_layout'] = _layout(('mcp-x', 'tts'))

    def test_bound_tool_passes(self):
        self.assertTrue(canvas_binding.is_bound('mcp__mcp-x__tts'))

    def test_unbound_tool_blocked(self):
        """未连线的工具必须被拒 —— 这正是真机上漏掉的那一条。"""
        self.assertFalse(canvas_binding.is_bound('mcp__mcp-x__loco'))
        self.assertFalse(canvas_binding.is_bound('mcp__other__tts'))

    def test_empty_canvas_blocks_everything(self):
        """画布为空时不应放行任何工具（fail closed）。"""
        config.main['canvas_layout'] = {'cards': [], 'execConnections': []}
        self.assertFalse(canvas_binding.is_bound('mcp__mcp-x__tts'))

    def test_connection_not_from_core_is_ignored(self):
        """只有从 decision_core 卡片出发的连线才算授权。"""
        config.main['canvas_layout'] = {
            'cards': [{'id': 'core-1', 'mcpId': 'agentcore'},
                      {'id': 'other', 'mcpId': 'mcp-y'}],
            'execConnections': [
                {'fromCardId': 'other', 'toMcpId': 'mcp-x', 'toToolName': 'tts'}],
        }
        self.assertFalse(canvas_binding.is_bound('mcp__mcp-x__tts'))

    def test_split_subtool_inherits_binding(self):
        """x-action-params 拆分出的子工具应继承父工具的绑定。"""
        self.assertTrue(canvas_binding.is_bound('mcp__mcp-x__tts__speak'))

    def test_tools_call_enforces_the_gate(self):
        """/tools/call 必须真的调用画布门，而不只是在注释里声称。"""
        import inspect
        import api.peer as ap
        src = inspect.getsource(ap.call_tool)
        self.assertIn('canvas_binding.is_bound', src,
                      '/tools/call does not enforce the canvas gate; an operator '
                      'peer could invoke unwired tools')
        # 角色门必须仍在
        self.assertIn('check_tool_permission', src)

    def test_tools_list_only_advertises_bound(self):
        """列表也只应给出真能调用的工具，避免对端围绕必然 403 的能力做规划。"""
        import inspect
        import api.peer as ap
        src = inspect.getsource(ap.list_tools)
        self.assertIn('canvas_binding.is_bound', src)


if __name__ == '__main__':
    unittest.main()
