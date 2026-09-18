"""
test_tool_schema_presentation.py — 交给 LLM 的工具 schema 长什么样。

Orin5 实测：用户说"好了别说了"，模型调了 `tts({"action": "stop"})`。它看到的 enum 就是
`["start","stop","speak","info","config","interrupt"]`，描述还写着 "start/stop speech
synthesis" —— 选 stop 是最自然的读法。但两个 action 完全不是一回事：

  interrupt  清队列 + 掐掉当前这句，节点继续跑
  stop       _dispose_node()，把话题订阅节点整个拆掉

`_SYSTEM_ACTIONS` 过滤本来就存在，写在 `mcp_client.all_schemas()` 里，注释写得明明白白
"过滤 processor 系统 action"。但主 agent loop 走的是 `event/llm.py` 的
`_get_bound_tool_schemas()`，直接取 `info['schemas'][name]` 生料；`all_schemas()` 全仓库
只有 `api/peer.py` 一处调用 —— **过滤从来没有在主链路生效过**。

同一处分叉还有第二个后果：`with_parallel_param`（注入 `concurrent`）也只在 all_schemas()
里调，所以主链路上任何工具都没有这个参数，而 system prompt 花了一整段教模型怎么用它。
从真实请求 dump 里确认过：`concurrent` 只在 prompt 正文出现 1 次，没有任何工具带它。

修法是把这两件事收敛成 `mcp_client.present_to_llm()`，两条路共用。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_tool_schema_presentation.py
"""
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402

# perception 的 tts 真实形状（processor、未拆分、action 带完整 enum）
TTS_ACTIONS = ['start', 'stop', 'speak', 'info', 'config', 'interrupt']
TTS_META = {'type': 'processor', 'action_enum': list(TTS_ACTIONS),
            'resource': frozenset({'mouth'})}


def _tts_schema():
    return {
        'name': 'mcp__p__tts',
        'description': 'TTS',
        'parameters': {'type': 'object', 'properties': {
            'action': {'type': 'string', 'enum': list(TTS_ACTIONS)},
            'text': {'type': 'string'}}, 'required': ['action']},
    }


class TestPresentToLLM(unittest.TestCase):
    def test_processor_system_actions_are_hidden(self):
        """这就是 Orin5 那次误调的根因。"""
        out = mcp_client.present_to_llm(_tts_schema(), TTS_META)
        self.assertEqual(out['parameters']['properties']['action']['enum'],
                         ['speak', 'interrupt'])

    def test_does_not_mutate_the_registry_schema(self):
        """注册表里那份是共享的 —— 就地改会污染 peer 路径和后续调用。"""
        schema = _tts_schema()
        mcp_client.present_to_llm(schema, TTS_META)
        self.assertEqual(schema['parameters']['properties']['action']['enum'], TTS_ACTIONS)

    def test_concurrent_param_is_injected(self):
        out = mcp_client.present_to_llm(_tts_schema(), TTS_META)
        self.assertIn('concurrent', out['parameters']['properties'])

    def test_sensor_gets_no_concurrent(self):
        """sensor/resource 只读，相邻的会自动并行，不需要这个参数。"""
        out = mcp_client.present_to_llm(
            {'name': 'mcp__p__cam', 'parameters': {'type': 'object', 'properties': {}}},
            {'type': 'sensor'})
        self.assertNotIn('concurrent', out['parameters']['properties'])

    def test_actuator_actions_are_not_filtered(self):
        """`_SYSTEM_ACTIONS` 只管 processor 的插件生命周期。

        底盘的 `stop` 是一条真正的动作指令（停下来），不是"停掉插件"——把它一起滤掉会
        让机器人失去刹车。
        """
        out = mcp_client.present_to_llm(
            {'name': 'mcp__d__loco', 'parameters': {'type': 'object', 'properties': {
                'action': {'type': 'string', 'enum': ['stop', 'move']}}}},
            {'type': 'actuator'})
        self.assertEqual(out['parameters']['properties']['action']['enum'], ['stop', 'move'])

    def test_processor_with_only_system_actions_is_hidden_entirely(self):
        self.assertIsNone(mcp_client.present_to_llm(
            _tts_schema(), {'type': 'processor', 'action_enum': ['start', 'stop', 'info']}))

    def test_split_subtool_for_a_system_action_is_hidden(self):
        """x-action-params 拆出来的子工具，action 在名字里而不是参数里。"""
        sub = {'name': 'mcp__p__tts__stop', 'parameters': {'type': 'object', 'properties': {}}}
        self.assertIsNone(mcp_client.present_to_llm(sub, TTS_META, split_action='stop'))

    def test_split_subtool_for_a_user_action_survives(self):
        sub = {'name': 'mcp__p__tts__speak', 'parameters': {'type': 'object', 'properties': {}}}
        out = mcp_client.present_to_llm(sub, TTS_META, split_action='speak')
        self.assertIsNotNone(out)
        self.assertIn('concurrent', out['parameters']['properties'])

    def test_missing_meta_is_tolerated(self):
        """没有 tool_meta 时不该炸 —— 按"什么都不知道"处理，原样交出去。"""
        out = mcp_client.present_to_llm(_tts_schema(), None)
        self.assertIsNotNone(out)
        self.assertEqual(out['parameters']['properties']['action']['enum'], TTS_ACTIONS)


class TestBothPathsShareIt(unittest.TestCase):
    """回归护栏：两条"把工具交给模型"的路必须都走 present_to_llm。

    分叉过一次就再分叉一次 —— 主链路那条当初漏了，谁也没发现，直到机器人把自己的 TTS
    节点拆了。
    """

    def _src(self, rel):
        return (pathlib.Path(__file__).resolve().parents[1] / 'src' / rel).read_text()

    def test_all_schemas_uses_it(self):
        body = self._src('mcp_client.py')
        body = body[body.index('def all_schemas('):]
        body = body[:body.index('\n\n\n')]
        self.assertIn('present_to_llm(', body)

    def test_bound_tool_schemas_uses_it(self):
        body = self._src('event/llm.py')
        body = body[body.index('def _get_bound_tool_schemas('):]
        body = body[:body.index('# Peer tools are deliberately')]
        self.assertIn('present_to_llm(', body)
        # 不能再直接把生料塞进去
        self.assertNotIn('schemas.append(schema)\n', body)

    def test_system_actions_set_is_defined_once(self):
        self.assertEqual(self._src('mcp_client.py').count('_SYSTEM_ACTIONS = '), 1)
        self.assertNotIn('_SYSTEM_ACTIONS', self._src('event/llm.py'))


if __name__ == '__main__':
    unittest.main()
