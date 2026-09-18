"""
test_acp_ordering.py — 次序：同一段推理内默认串行，并行必须显式要求。

资源互斥回答"能不能同时"，回答不了"必须先做完哪个"。
「先说'我要起来了'再起身」是**次序**要求：mouth 和 leg 是不同通道，互斥允许它们重叠；
而按资源拆分之前，全局 barrier 是**碰巧**禁止了重叠。两者都不是真答案 —— 同一对工具，
配合讲解的手势必须同时，动作前的安全播报必须先后。工具完全相同，只有意图不同，
而意图只存在于发出这些调用的那一方。

所以次序在**同一个 agent 自己的调用序列内**生效（LLM 按顺序发出，顺序即脚本），
跨 agent 不生效（各自没有共同脚本，无关动作不该互相挡 —— 那正是把 N 个 subagent
压成 1 个的原因）。`concurrent=true` 让发起方为单次调用退出自己的次序约束。

默认 false 是刻意的：模型不被要求就不会思考并发，它写出来的调用读起来就是顺序的。
所以不说就按顺序办，而"警告和它警告的动作重叠"这个不安全方向必须显式索取。
"""
import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import mcp_client  # noqa: E402

MOUTH = frozenset({'mouth'})
LEG = frozenset({'leg'})
MAIN = mcp_client.CONTEXT_MAIN
SUB_A = 'subagent:aaaa'
SUB_B = 'subagent:bbbb'


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._tables = (mcp_client._pending_actions, mcp_client._pending_results,
                        mcp_client._pending_timeouts, mcp_client._pending_tools,
                        mcp_client._pending_resources, mcp_client._pending_owner)
        self._saved = [dict(d) for d in self._tables]
        for d in self._tables:
            d.clear()

    def tearDown(self):
        for d, saved in zip(self._tables, self._saved):
            d.clear()
            d.update(saved)

    def _add(self, aid, resource, owner=MAIN, *, tool='tts', timeout=5.0, done=False):
        ev = asyncio.Event()
        if done:
            ev.set()
        mcp_client._pending_actions[aid] = ev
        mcp_client._pending_tools[aid] = tool
        mcp_client._pending_timeouts[aid] = timeout
        mcp_client._pending_resources[aid] = resource
        mcp_client._pending_owner[aid] = owner
        return ev

    def _wait_for(self, want, owner=MAIN, concurrent=False):
        return mcp_client.pendings_to_wait_for(want, owner=owner, concurrent=concurrent)


class TestOrderingWithinOneSequence(_Fixture):
    def test_announce_then_move_is_serial_by_default(self):
        """回归本条机制存在的理由：安全播报必须先播完再动。"""
        self._add('speak-1', MOUTH)
        self.assertEqual(self._wait_for(LEG), ['speak-1'],
                         '起身动作没有等播报播完')

    def test_gesture_with_speech_when_asked(self):
        """讲解时的手势要和讲解同时 —— 显式声明就放行。"""
        self._add('speak-1', MOUTH)
        self.assertEqual(self._wait_for(LEG, concurrent=True), [])

    def test_same_channel_still_serial_even_when_concurrent_requested(self):
        """concurrent 表达意图，不能凭空多出一张嘴。"""
        self._add('speak-1', MOUTH)
        self.assertEqual(self._wait_for(MOUTH, concurrent=True), ['speak-1'])

    def test_undeclared_caller_waits_regardless(self):
        self._add('speak-1', MOUTH)
        self.assertEqual(self._wait_for(None, concurrent=True), ['speak-1'])

    def test_nothing_pending(self):
        self.assertEqual(self._wait_for(LEG), [])

    def test_registration_order_preserved(self):
        self._add('a', MOUTH)
        self._add('b', LEG)
        self.assertEqual(self._wait_for(frozenset({'head'})), ['a', 'b'])


class TestOrderingDoesNotCrossAgents(_Fixture):
    """跨 agent 不适用次序 —— 各自没有共同脚本。"""

    def test_another_agents_action_is_not_waited_for(self):
        self._add('speak-1', MOUTH, owner=SUB_A)
        self.assertEqual(self._wait_for(LEG, owner=SUB_B), [],
                         '另一个 subagent 的无关动作把我挡住了')

    def test_main_does_not_wait_on_a_subagent(self):
        self._add('speak-1', MOUTH, owner=SUB_A)
        self.assertEqual(self._wait_for(LEG, owner=MAIN), [])

    def test_subagent_does_not_wait_on_main(self):
        self._add('speak-1', MOUTH, owner=MAIN)
        self.assertEqual(self._wait_for(LEG, owner=SUB_A), [])

    def test_but_resource_conflict_still_crosses_agents(self):
        """一张嘴就是一张嘴 —— 谁发起的都得排队。"""
        self._add('speak-1', MOUTH, owner=SUB_A)
        self.assertEqual(self._wait_for(MOUTH, owner=SUB_B), ['speak-1'])

    def test_n_subagents_do_not_serialise_on_unrelated_channels(self):
        """把 N 个 subagent 压成 1 个的那个塌陷不能回来。"""
        self._add('speak-a', MOUTH, owner=SUB_A)
        self._add('arm-b', frozenset({'arm_l'}), owner=SUB_B)
        self.assertEqual(self._wait_for(LEG, owner='subagent:cccc'), [])

    def test_own_action_still_orders_within_a_subagent(self):
        self._add('speak-1', MOUTH, owner=SUB_A)
        self.assertEqual(self._wait_for(LEG, owner=SUB_A), ['speak-1'])

    def test_unknown_owner_counts_as_main(self):
        """旧 pending 没有 owner 记录时按 main 处理，而不是"谁都不是"。"""
        ev = asyncio.Event()
        mcp_client._pending_actions['legacy'] = ev
        mcp_client._pending_resources['legacy'] = MOUTH
        mcp_client._pending_timeouts['legacy'] = 5.0
        self.assertEqual(self._wait_for(LEG, owner=MAIN), ['legacy'])


class TestParallelFlagPlumbing(_Fixture):
    def test_flag_is_popped_not_forwarded(self):
        """driver 的 schema 不认识这个参数，转发过去会被校验拒掉。"""
        args = {'action': 'speak', 'text': 'hi', mcp_client.PARALLEL_PARAM: True}
        self.assertTrue(mcp_client.take_parallel_flag(args))
        self.assertNotIn(mcp_client.PARALLEL_PARAM, args)
        self.assertEqual(args, {'action': 'speak', 'text': 'hi'})

    def test_absent_means_sequential(self):
        args = {'action': 'speak'}
        self.assertFalse(mcp_client.take_parallel_flag(args))

    def test_explicit_false(self):
        args = {mcp_client.PARALLEL_PARAM: False}
        self.assertFalse(mcp_client.take_parallel_flag(args))

    def test_non_dict_is_safe(self):
        self.assertFalse(mcp_client.take_parallel_flag(None))

    def test_owner_is_recorded_from_the_context_var(self):
        token = mcp_client.current_agent_context.set(SUB_A)
        try:
            self.assertEqual(mcp_client.current_agent_context.get(), SUB_A)
        finally:
            mcp_client.current_agent_context.reset(token)
        self.assertEqual(mcp_client.current_agent_context.get(), MAIN)


class TestSchemaInjection(unittest.TestCase):
    def _schema(self, **kw):
        return {'name': 'mcp__m__tts', 'description': 'd',
                'parameters': {'type': 'object',
                               'properties': {'text': {'type': 'string'}},
                               'required': ['text']}, **kw}

    def test_acting_tool_gets_the_parameter(self):
        out = mcp_client.with_parallel_param(self._schema(), 'actuator')
        self.assertIn(mcp_client.PARALLEL_PARAM, out['parameters']['properties'])
        self.assertEqual(
            out['parameters']['properties'][mcp_client.PARALLEL_PARAM]['type'], 'boolean')

    def test_processor_gets_it_too(self):
        out = mcp_client.with_parallel_param(self._schema(), 'processor')
        self.assertIn(mcp_client.PARALLEL_PARAM, out['parameters']['properties'])

    def test_sensor_does_not(self):
        """sensor 从不被 barrier 挡，参数在它的 schema 里只是噪音。"""
        for ty in ('sensor', 'resource'):
            out = mcp_client.with_parallel_param(self._schema(), ty)
            self.assertNotIn(mcp_client.PARALLEL_PARAM, out['parameters']['properties'])

    def test_it_is_not_required(self):
        out = mcp_client.with_parallel_param(self._schema(), 'actuator')
        self.assertNotIn(mcp_client.PARALLEL_PARAM, out['parameters'].get('required', []))

    def test_existing_properties_survive(self):
        out = mcp_client.with_parallel_param(self._schema(), 'actuator')
        self.assertIn('text', out['parameters']['properties'])
        self.assertEqual(out['parameters']['required'], ['text'])

    def test_a_driver_declaring_its_own_is_not_shadowed(self):
        s = self._schema()
        s['parameters']['properties'][mcp_client.PARALLEL_PARAM] = {'type': 'string'}
        out = mcp_client.with_parallel_param(s, 'actuator')
        self.assertEqual(
            out['parameters']['properties'][mcp_client.PARALLEL_PARAM]['type'], 'string')

    def test_the_description_tells_the_model_the_default(self):
        out = mcp_client.with_parallel_param(self._schema(), 'actuator')
        desc = out['parameters']['properties'][mcp_client.PARALLEL_PARAM]['description']
        self.assertIn('false', desc)
        self.assertIn('同一个物理通道', desc)

    def test_original_schema_is_not_mutated(self):
        s = self._schema()
        mcp_client.with_parallel_param(s, 'actuator')
        self.assertNotIn(mcp_client.PARALLEL_PARAM, s['parameters']['properties'])


if __name__ == '__main__':
    unittest.main()
