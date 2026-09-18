"""
test_progress_narration.py — 框架级主动播报 + 沉默计时状态机。

背景：主动播报原本完全建立在「LLM 自愿写 content，系统自动播出去」之上。两个实测问题：
prompt 约束不住，多轮工具调用期间模型经常一句话不写；写了也不合适，那是内部推理而不是
说给等着的人听的进度汇报。现在改成框架保证，content 自动播报同时废除（回归断言见
test_on_notify_barrier.py）。

计时部分经过一轮简化：**定时器的存在性就是状态**，没有时钟变量，也就不需要「播放期间
冻结」——正在说话时根本不存在定时器。

    事件                              动作
    ───────────────────────────────────────────────────────────────
    输出被派发（开始说话）             停止；该输出若无 ACP 跟踪则当场重新计时
    打断                               停止
    说完了（完成 / 超时 / 取消）       有活 → 重新计时；无活 → 停止
    任务开始                           开始计时（已在计时则不动）
    活全干完                           停止
    轮边界                             不在说话且没在计时 → 开始计时（不变式兜底）
    到点                               有活且 gate 全过 → 汇报；没播成 → 重新计时

本文件里标了「防永久静音」的用例是风险最高的一类：新模型下失败模式不是刷屏而是**彻底
没声**，日志里什么都看不到。每条都要能抓到回归。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress_narration.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import client  # noqa: E402
import collector  # noqa: E402
import config  # noqa: E402
import hooks  # noqa: E402
import mcp_client  # noqa: E402
import subagent  # noqa: E402
import event.skills as skills_tools  # noqa: E402
# event/__init__.py rebinds the `event.skills` / `event.llm` *attributes* to instances, so
# those names give you the methods but not the module globals — those must come from
# sys.modules. Same reason event/llm.py reaches get_notify_override through sys.modules.
skills_mod = sys.modules['event.skills']
ell = sys.modules['event.llm']

from event.llm import (  # noqa: E402
    _build_narration_messages,
    _build_system_tools,
    _narration_feedback_message,
    _narration_thresholds,
    _notify_fire_spoke,
    _restricted_channel_tool_allowed,
)

MOUTH_META = {
    'type': 'actuator',
    'action_enum': None,
    'has_config_schema': False,
    'completion': {'actions': ['speak'], 'timeout': 180},
    'resource': frozenset({'mouth'}),
}


class _Fixture(unittest.TestCase):
    """隔离全局注册表、pending、两个运行时覆盖和计时器。"""

    def setUp(self):
        self._saved_registry = dict(mcp_client.registry)
        self._saved_hooks = dict(hooks._registry)
        mcp_client.registry.clear()
        hooks._registry.clear()
        self._saved_pending = tuple(
            dict(d) for d in (mcp_client._pending_actions, mcp_client._pending_tools,
                              mcp_client._pending_resources, mcp_client._pending_timeouts))
        for d in (mcp_client._pending_actions, mcp_client._pending_tools,
                  mcp_client._pending_resources, mcp_client._pending_timeouts):
            d.clear()
        self._saved_notify = skills_mod._notify_override
        self._saved_report = skills_mod._report_override
        skills_mod._notify_override = None
        skills_mod._report_override = None
        self._saved_event = config.main.get('event', {})
        self._saved_mgr = subagent._manager_instance
        self._saved_inst = ell._event_instance
        ell._stop_countdown()
        ell._last_report_text_global = ''
        ell._last_turn_restricted = False
        del ell._pending_narration_feedback[:]
        self._saved_rounds = dict(ell._reported_rounds)
        ell._reported_rounds.clear()
        self._saved_stall = ell._last_progress_ts
        ell._last_progress_ts = None
        self._saved_turn_msgs = ell._reported_turn_msgs
        ell._reported_turn_msgs = 0
        self._saved_busy = collector._busy
        collector._busy = False

    def tearDown(self):
        ell._stop_countdown()
        del ell._pending_narration_feedback[:]
        mcp_client.registry.clear()
        mcp_client.registry.update(self._saved_registry)
        hooks._registry.clear()
        hooks._registry.update(self._saved_hooks)
        for d, saved in zip((mcp_client._pending_actions, mcp_client._pending_tools,
                             mcp_client._pending_resources, mcp_client._pending_timeouts),
                            self._saved_pending):
            d.clear()
            d.update(saved)
        skills_mod._notify_override = self._saved_notify
        skills_mod._report_override = self._saved_report
        config.main['event'] = self._saved_event
        subagent._manager_instance = self._saved_mgr
        ell._event_instance = self._saved_inst
        ell._reported_rounds.clear()
        ell._reported_rounds.update(self._saved_rounds)
        ell._last_progress_ts = self._saved_stall
        ell._reported_turn_msgs = self._saved_turn_msgs
        collector._busy = self._saved_busy

    # -- helpers ---------------------------------------------------------
    def _register_mouth(self, *, resource=frozenset({'mouth'})):
        mcp_client.registry['dev1'] = {
            'name': 'dev1', 'url': 'http://dev1', 'online': True, 'tools': ['tts'],
            'schemas': {'mcp__dev1__tts__speak': {'name': 'mcp__dev1__tts__speak'}},
            'tool_meta': {'mcp__dev1__tts__speak': {**MOUTH_META, 'resource': resource}},
            'split_map': {'mcp__dev1__tts__speak': {'tool': 'tts', 'action': 'speak'}},
            'tool_groups': {}, 'input_schemas': {},
        }
        hooks.register('dev1', 'tts', {'on_notify': {'action': 'speak'}})

    def _cfg(self, **kw):
        ev = dict(config.main.get('event', {}))
        llm = dict(ev.get('llm', {}))
        llm.update(kw)
        ev['llm'] = llm
        config.main['event'] = ev

    def _speaking(self, aid='spk', resource=frozenset({'mouth'})):
        """模拟一次正在播放的输出。返回它的 Event。"""
        evt = asyncio.Event()
        mcp_client._pending_actions[aid] = evt
        mcp_client._pending_resources[aid] = resource
        return evt

    def _work(self, running=True, turns=None, rounds=3):
        """模拟一个在跑的子代理。`turns` 给它一段真实活动记录 —— 汇报器要靠这个才说得出
        "做了什么/发现了什么"，只有目标和轮数的话它只能把目标换个说法念一遍。

        默认给一条：真实场景里"在跑"的子代理总是做过点什么。turns 为空表示刚派出去还没
        动作，那种情况现在会被"没有新进展"直接挡掉、连 LLM 调用都不花，要测那条走
        _work(turns=[])。
        """
        if turns is None:
            turns = [[{'role': 'tool', 'content': 'BASEFINDING'}]]
        st = 'running' if running else 'completed'
        _S = type('S', (), {'id': 'ab12', 'status': st,
                            'rounds_completed': rounds, 'goal': '长任务'})
        _Ctx = type('Ctx', (), {'turns': list(turns or [])})
        _Agent = type('Agent', (), {
            'id': 'ab12', 'status': st, 'rounds_completed': rounds,
            'spec': type('Spec', (), {'goal': '长任务'})(),
            'context': _Ctx()})
        _M = type('M', (), {'_agents': {'ab12': _Agent()},
                            'list_active': lambda self_: [_S()]})
        subagent._manager_instance = _M()
        return _Agent


# ── 纯谓词 ────────────────────────────────────────────────────────────────

class TestNotifyFireSpoke(unittest.TestCase):
    """hooks.fire 的返回值里，到底有没有一个绑定真把话送到设备上。"""

    def test_empty_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([]))
        self.assertFalse(_notify_fire_spoke(None))

    def test_resource_busy_skip_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([{'result': {'skipped': 'resource busy'}}]))

    def test_error_is_not_spoken(self):
        self.assertFalse(_notify_fire_spoke([{'error': 'boom'}]))
        self.assertFalse(_notify_fire_spoke([{'result': {'error': 'device offline'}}]))

    def test_real_result_is_spoken(self):
        self.assertTrue(_notify_fire_spoke([{'result': {'action_id': 'speak-1'}}]))

    def test_mixed_counts_as_spoken(self):
        self.assertTrue(_notify_fire_spoke([
            {'result': {'skipped': 'resource busy'}},
            {'result': {'action_id': 'speak-1'}}]))


class TestBuildNarrationMessages(unittest.TestCase):
    FROZEN = {'role': 'system', 'content': 'you are a robot'}

    def _build(self, context, last='', budget=6000):
        return _build_narration_messages(frozen_system=self.FROZEN, context=context,
                                         last_report_text=last, budget_chars=budget)

    def test_shape_and_system_reuse(self):
        msgs = self._build('ctx')
        self.assertEqual(len(msgs), 2)
        # 同一个对象：汇报继承人设，且与主循环逐字节相同 ⇒ 命中前缀缓存。
        self.assertIs(msgs[0], self.FROZEN)
        self.assertEqual(msgs[1]['role'], 'user')

    def test_keeps_tail_drops_head(self):
        # 与 _compress_turns 的 text[:30000] 相反：进度汇报关心的是近况。
        body = self._build('HEAD' + 'x' * 20000 + '\nTAILMARKER', budget=500)[1]['content']
        self.assertIn('TAILMARKER', body)
        self.assertNotIn('HEAD', body)
        self.assertIn('(前略)', body)

    def test_last_report_text_included(self):
        body = self._build('ctx', last='我已经找完客厅了')[1]['content']
        self.assertIn('我已经找完客厅了', body)

    def test_no_last_report_line_when_empty(self):
        self.assertNotIn('不要重复你上次', self._build('ctx')[1]['content'])


class TestFeedbackMessage(unittest.TestCase):
    def test_shape(self):
        m = _narration_feedback_message('已经找完客厅了，接下来去卧室')
        self.assertEqual(m['role'], 'user')
        self.assertIn('source=narration', m['content'])
        self.assertIn('已经找完客厅了，接下来去卧室', m['content'])


# ── 三个原语 ──────────────────────────────────────────────────────────────

class TestCountdownPrimitives(_Fixture):
    def setUp(self):
        super().setUp()
        self._cfg(narration_silence_seconds=30)

    def test_start_creates_when_idle(self):
        async def go():
            self.assertIsNone(ell._silence_countdown)
            ell._start_countdown()
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_start_does_not_replace_a_running_countdown(self):
        """「已存在不更新」——否则新任务会把上一件事已经攒下的沉默一笔勾销。"""
        async def go():
            ell._start_countdown()
            first = ell._silence_countdown
            ell._start_countdown()
            return first, ell._silence_countdown
        first, second = asyncio.run(go())
        self.assertIs(first, second)
        self.assertFalse(first.cancelled())

    def test_restart_always_replaces(self):
        async def go():
            ell._restart_countdown()
            first = ell._silence_countdown
            ell._restart_countdown()
            second = ell._silence_countdown
            await asyncio.sleep(0)
            return first, second
        first, second = asyncio.run(go())
        self.assertIsNot(first, second)
        self.assertTrue(first.cancelled() or first.done())

    def test_stop_clears(self):
        async def go():
            ell._start_countdown()
            t = ell._silence_countdown
            ell._stop_countdown()
            await asyncio.sleep(0)
            return t
        t = asyncio.run(go())
        self.assertTrue(t.cancelled() or t.done())
        self.assertIsNone(ell._silence_countdown)

    def test_threshold_zero_means_no_countdown(self):
        self._cfg(narration_silence_seconds=0)

        async def go():
            ell._start_countdown()
            ell._restart_countdown()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_no_event_loop_does_not_raise(self):
        """启动早期 / 同步测试里没有运行中的 loop，开始计时不能把调用方炸掉。"""
        ell._start_countdown()
        self.assertIsNone(ell._silence_countdown)


# ── 事件 → 动作 ───────────────────────────────────────────────────────────

class TestStateMachine(_Fixture):
    def setUp(self):
        super().setUp()
        self._register_mouth()
        self._cfg(narration_silence_seconds=30, auto_narration=True)

    def test_speaking_started_stops_countdown_while_tracked(self):
        async def go():
            ell._start_countdown()
            self._speaking()                 # 有 ACP 跟踪，嘴现在忙
            ell._on_speaking_started()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_speaking_started_without_acp_tracking_restarts(self):
        """**防永久静音**：没有 ACP 跟踪的输出永远等不到「说完」事件。

        设备不声明 x-completion 是合法的，那样 call_tool_hook 不注册 pending，嘴从来
        不「忙」，也就永远不会有「说完」来重新计时。只停不启的话播报就此静音。
        """
        async def go():
            ell._start_countdown()
            ell._on_speaking_started()       # 嘴不忙 ⇒ 这次输出已经结束
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_speaking_finished_restarts_when_work_remains(self):
        self._work()

        async def go():
            evt = self._speaking()
            ell._on_speaking_started()
            evt.set()                        # 播完
            ell._on_speaking_finished()
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_speaking_finished_stops_when_nothing_left(self):
        async def go():
            evt = self._speaking()
            ell._on_speaking_started()
            evt.set()
            ell._on_speaking_finished()
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_speaking_finished_still_arms_while_another_mouth_plays(self):
        """多设备：第一个说完时**照样上表**，不能因为嘴还忙就不计时。

        这条原来断言的是相反的（忙 → 不上表），那是一条死路：嘴上可能挂着别人的动作
        （天轶的驱动自己也注册 mouth 动作，barrier 日志里见过 ['28','speak-…','tts-…']
        三个一起 want=mouth），那个动作的完成若没被观察到，就再也没有人来重新计时 ——
        只能等轮边界兜底。实测因此空了 70 秒。

        "正在说话时不该汇报"由到点时的 mouth busy gate 负责，那条**重新计时而不是停表**，
        所以放它过来是安全的。
        """
        self._work()

        async def go():
            a = self._speaking('spk-a')
            self._speaking('spk-b')          # 第二个还在播
            ell._on_speaking_started()
            a.set()
            ell._on_speaking_finished()
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_inflight_skip_rearms_instead_of_dying(self):
        """另一次汇报在飞时早退，**必须重新计时**。

        定时器 task 到此就结束了；不重排的话没有任何东西会再触发汇报 —— 等于永久静音到
        下一个轮边界。这条和上一条是同一次事故的两个嫌疑人，都是无日志的静默死路。
        """
        self._work()

        async def go():
            # inflight 早退是 _report_progress 的第一句，不需要接好线的实例。
            inst = ell.Event.__new__(ell.Event)
            ell._narration_inflight = True
            try:
                ell._stop_countdown()
                await inst._report_progress()
                return ell._silence_countdown
            finally:
                ell._narration_inflight = False
        self.assertIsNotNone(asyncio.run(go()))

    def test_settle_listener_fires_for_user_facing_action(self):
        """完成回调 → 走到 _on_speaking_finished。两处完成入口都经过 mark_action_complete。"""
        self._work()

        async def go():
            self._speaking('spk-9')
            ell._on_speaking_started()
            mcp_client._pending_actions['spk-9'].set()
            ok = mcp_client.mark_action_complete('spk-9', {'status': 'completed'})
            return ok, ell._silence_countdown
        ok, timer = asyncio.run(go())
        self.assertTrue(ok)
        self.assertIsNotNone(timer)

    def test_settle_listener_ignores_non_output_action(self):
        """走路走完了不是「说完话了」，不该触发计时。"""
        self._work()

        async def go():
            mcp_client._pending_actions['move-1'] = asyncio.Event()
            mcp_client._pending_resources['move-1'] = frozenset({'legs'})
            ell._stop_countdown()
            mcp_client.mark_action_complete('move-1', {'status': 'completed'})
            return ell._silence_countdown
        self.assertIsNone(asyncio.run(go()))

    def test_timeout_and_cancel_also_settle(self):
        """**防永久静音**：超时/取消走 _forget_pending，不经过 mark_action_complete。

        不在那里通知的话，一次超时的播报之后再没有任何事件会让计时重新开始。
        """
        self._work()

        async def go():
            self._speaking('spk-t')
            ell._on_speaking_started()
            self.assertIsNone(ell._silence_countdown)
            mcp_client._forget_pending(['spk-t'], 'timeout')
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_task_started_and_all_work_done(self):
        async def go():
            ell._on_task_started()
            started = ell._silence_countdown
            ell._on_all_work_done()
            return started, ell._silence_countdown
        started, after = asyncio.run(go())
        self.assertIsNotNone(started)
        self.assertIsNone(after)

    def test_interrupt_only_stops(self):
        """打断 = 用户在说话，不是沉默的起点。紧接着的新任务才重新计时。"""
        async def go():
            ell._start_countdown()
            ell._stop_countdown()            # 打断路径做的事
            after_interrupt = ell._silence_countdown
            ell._on_task_started()           # 新请求进来
            return after_interrupt, ell._silence_countdown
        after_interrupt, after_task = asyncio.run(go())
        self.assertIsNone(after_interrupt)
        self.assertIsNotNone(after_task)

    def test_round_boundary_invariant_restores(self):
        """轮边界兜底：不在说话却没在计时 ⇒ 补回来。正在说话时**不**补。"""
        async def go():
            ell._stop_countdown()
            if not ell._mouth_busy():        # 轮循环里那两行
                ell._start_countdown()
            restored = ell._silence_countdown

            ell._stop_countdown()
            self._speaking()
            if not ell._mouth_busy():
                ell._start_countdown()
            return restored, ell._silence_countdown
        restored, while_speaking = asyncio.run(go())
        self.assertIsNotNone(restored)
        self.assertIsNone(while_speaking)


# ── 汇报本体 ──────────────────────────────────────────────────────────────

class TestReportProgress(_Fixture):
    def setUp(self):
        super().setUp()
        self._register_mouth()
        self._cfg(auto_narration=True, narration_silence_seconds=30,
                  narration_timeout_s=5, narration_context_chars=6000)
        self._work()
        self._saved_call = client.call
        self._saved_fire = hooks.fire
        self._saved_push = ell.push_event
        # system 段的拼装不是这组用例的主题，而且它要读 prompt_system.md 和一堆 config
        # 键 —— 让状态机测试依赖那些，只会在别的模块改了全局 config 时莫名其妙地挂。
        self._saved_build = ell.prompt_mod.build_system
        ell.prompt_mod.build_system = lambda *a, **k: {'role': 'system', 'content': 'sys'}
        self.fired = []
        self.pushed = []

        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            return [{'result': {'action_id': 'speak-1'}}]

        async def _push(ev):
            self.pushed.append(ev)

        hooks.fire = _fire
        ell.push_event = _push
        self.inst = ell.Event.__new__(ell.Event)
        self.inst._turns = []
        self.inst._current_turn = []
        ell._event_instance = self.inst

    def tearDown(self):
        client.call = self._saved_call
        hooks.fire = self._saved_fire
        ell.push_event = self._saved_push
        ell.prompt_mod.build_system = self._saved_build
        super().tearDown()

    def _stub(self, content=None, *, raises=None, hang=False, record=None):
        async def _call(message_list, tool_list, **kw):
            if record is not None:
                record.append({'messages': message_list, 'tools': tool_list, 'kw': kw})
            if hang:
                await asyncio.sleep(30)
            if raises:
                raise raises
            return {'content': content}
        client.call = _call

    def _run(self):
        return asyncio.run(self.inst._report_progress())

    def test_happy_path(self):
        self._stub('客厅找完了，接下来去卧室')
        self._run()
        self.assertEqual(self.fired[0][0], 'on_notify')
        self.assertEqual(self.fired[0][1], {'text': '客厅找完了，接下来去卧室'})
        # 必须 barrier-aware：既不能盖过正在播的音频，也不能被下一个工具盖掉。
        self.assertTrue(self.fired[0][2].get('barrier_aware'))
        self.assertTrue(any(e['type'] == 'narration' for e in self.pushed))
        self.assertEqual(ell._last_report_text_global, '客厅找完了，接下来去卧室')

    def test_feedback_is_queued_not_appended(self):
        """回灌必须排队：定时器可能在 assistant(tool_calls) 已入列、tool 结果还没入列
        的窗口触发，当场 append 会造出 provider 会拒的消息序列。"""
        self._stub('好了')
        self._run()
        self.assertEqual(self.inst._current_turn, [])
        self.assertEqual(ell._pending_narration_feedback, ['好了'])

    def test_toolless_main_model_call(self):
        rec = []
        self._stub('好了', record=rec)
        self._run()
        self.assertEqual(rec[0]['tools'], [])
        self.assertEqual(rec[0]['messages'][0]['role'], 'system')
        self.assertIsNone(rec[0]['kw'].get('model_override'))
        self.assertIsNone(rec[0]['kw'].get('reconsider_event'))

    def test_out_of_turn_context_carries_subagent_state(self):
        rec = []
        self._stub('好了', record=rec)
        self._run()
        body = rec[0]['messages'][1]['content']
        self.assertIn('ab12', body)
        self.assertIn('长任务', body)

    def test_context_carries_what_the_subagent_actually_did(self):
        """只给目标 + 轮数的话，汇报器只能把目标换个说法念一遍。

        Orin5 实测播出来的就是"投研报告还在调研中，完成后我会第一时间告诉你结果"——
        用户听完仍然不知道进行到哪了。它需要看到子代理具体搜了什么、拿到了什么。
        """
        self._work(turns=[[
            {'role': 'assistant', 'tool_calls': [
                {'function': {'name': 'WebSearch', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': '票房 21.5 亿，豆瓣 8.7'},
        ]])
        rec = []
        self._stub('好了', record=rec)
        self._run()
        body = rec[0]['messages'][1]['content']
        self.assertIn('WebSearch', body)
        self.assertIn('21.5 亿', body)

    def test_only_new_activity_since_last_report_is_sent(self):
        """每次只讲**上次汇报之后**新发生的事。

        不做增量的话，上下文是个前后重叠的滑动窗口，模型只能把累积状态重新总结一遍 ——
        Tianyi 实测连着三条播报越说越像，第三条几乎是第二条加一个词，末尾都靠"马上整理
        成报告"凑数。
        """
        agent = self._work(turns=[[{'role': 'tool', 'content': 'OLDFINDING'}]])
        rec = []
        self._stub('第一条', record=rec)
        self._run()
        self.assertIn('OLDFINDING', rec[0]['messages'][1]['content'])

        # 子代理又往前跑了一轮（rounds 3 → 4），追加一条新 turn
        agent.rounds_completed = 4
        agent.context = type('Ctx', (), {'turns': [
            [{'role': 'tool', 'content': 'OLDFINDING'}],
            [{'role': 'tool', 'content': 'NEWFINDING'}]]})()
        rec2 = []
        self._stub('第二条', record=rec2)
        self._run()
        body = rec2[0]['messages'][1]['content']
        self.assertIn('NEWFINDING', body)
        self.assertNotIn('OLDFINDING', body)

    def test_a_just_spawned_subagent_is_announced(self):
        """刚派出去还没产出 —— 也要说一句"已经着手了"。

        用户刚提完需求接着一片安静时，"已经开始查了、还在等第一批结果"本身就是信息。
        """
        self._work(turns=[], rounds=0)
        rec = []
        self._stub('已经开始查了，还在等第一批结果', record=rec)
        self._run()
        self.assertEqual(len(self.fired), 1)
        # 此时确实什么结果都没有，必须明确禁止编造
        self.assertIn('不要编造任何进展或数据', rec[0]['messages'][1]['content'])

        rec2 = []
        self._stub('还在等结果，已经半分钟了', record=rec2)
        self._run()
        self.assertEqual(len(self.fired), 2, '还没产出时也要继续播，带上等了多久')

    def test_context_compression_does_not_look_like_no_progress(self):
        """子代理压缩自己的上下文时，turns 列表会**变短** —— 不能因此判成没进展。

        Orin5 实测：round 6 时 msgs 从 21 掉到 18。水位若记的是 turns 列表长度，一压缩就
        大于长度、切片为空，于是轮数明明在涨（6→7→8）却连着几条播报都说"没出新结果"、
        还带着荒谬的"已经跑了 0 秒"。水位记轮数就不会 —— rounds_completed 单调递增。
        """
        agent = self._work(turns=[[{'role': 'tool', 'content': f'T{i}'} for i in range(5)]],
                           rounds=5)
        rec = []
        self._stub('第一条', record=rec)
        self._run()
        self.assertEqual(len(self.fired), 1)

        # 轮数前进，但上下文被压缩、turns 变短
        agent.rounds_completed = 6
        agent.context = type('Ctx', (), {'turns': [
            [{'role': 'tool', 'content': 'COMPRESSED_NEW'}]]})()
        rec2 = []
        self._stub('第二条', record=rec2)
        self._run()
        body = rec2[0]['messages'][1]['content']
        self.assertIn('COMPRESSED_NEW', body)
        self.assertNotIn('没有产出新东西', body, '轮数涨了就是有新进展，不该判成卡住')

    def test_stall_keeps_reporting_with_elapsed_time(self):
        """卡住期间照常每个间隔播一句，但必须带上已经卡了多久。

        Tianyi 实测：写一份 454 行报告花了 107 秒，期间一个 turn 都没产出。只播一次的话
        用户要静默 92 秒；每次原句重复又是噪音。时长是真正的新信息 —— 用户想知道的是
        "有没有卡死"，而不是"在做什么"（上一句已经说过了）。
        """
        self._work(turns=[[{'role': 'tool', 'content': 'FINDING'}]])
        self._stub('第一条'); self._run()
        self.assertEqual(len(self.fired), 1)

        rec = []
        self._stub('还在写报告，已经一分钟了', record=rec)
        self._run()
        self.assertEqual(len(self.fired), 2, '卡住的第一句')
        body = rec[0]['messages'][1]['content']
        self.assertIn('已经等了多久', body)
        self.assertIn('距离上一次有新动作已经过去', body)

        rec2 = []
        self._stub('还在写报告，已经两分钟了', record=rec2)
        self._run()
        self.assertEqual(len(self.fired), 3, '卡住期间要继续播，不是只播一次')

    def test_progress_clock_advances_only_on_real_progress(self):
        """"多久没出新结果"要从上次真有进展算起，不是从进入卡顿那一刻算起。

        记成后者的话，第一条卡顿播报会在同一次调用里先把它设成 now、再拿它算时长，
        播出来就是"已经跑了 0 秒没出新结果"（Orin5 实测原话），自相矛盾。
        """
        agent = self._work(turns=[[{'role': 'tool', 'content': 'A'}]])
        self._stub('一'); self._run()
        first = ell._last_progress_ts
        self.assertIsNotNone(first)

        self._stub('二'); self._run()                     # 卡住：轮数没变
        self.assertEqual(ell._last_progress_ts, first, '卡顿期间这个时刻不该被刷新')

        agent.rounds_completed = 5                        # 真的往前跑了
        self._stub('三'); self._run()
        self.assertGreater(ell._last_progress_ts, first, '有新进展后要重新计')

    def test_in_turn_context_is_also_incremental(self):
        """turn 内那条路同样只讲上次汇报之后新增的消息。

        Tianyi 实测：说完"正在从百度百科提取照片"，下一句直接变成"首都之窗的页面抓下来
        了" —— 中间百度百科 403 被拒这条线索用户从没听到，两句听着像两件不相干的事。
        原因是 turn 内每次把整个 turn 喂过去，模型自己挑，就会跳掉中间过程。
        """
        collector._busy = True           # turn 在跑 → 走 turn 内那条路
        self.inst._current_turn = [{'role': 'tool', 'content': 'STEP_ONE'}]
        rec = []
        self._stub('第一步做完了', record=rec)
        self._run()
        self.assertIn('STEP_ONE', rec[0]['messages'][1]['content'])

        self.inst._current_turn.append({'role': 'tool', 'content': 'STEP_TWO'})
        rec2 = []
        self._stub('第二步也做完了', record=rec2)
        self._run()
        body = rec2[0]['messages'][1]['content']
        self.assertIn('STEP_TWO', body)
        self.assertNotIn('STEP_ONE', body, 'turn 内也要只喂增量')

    def test_in_turn_watermark_resets_on_new_turn(self):
        """新 turn 水位要清零 —— 否则上个 turn 的条数会把新 turn 的开头整段吃掉。"""
        collector._busy = True
        self.inst._current_turn = [{'role': 'tool', 'content': 'A'},
                                   {'role': 'tool', 'content': 'B'}]
        self._stub('一'); self._run()
        self.assertEqual(ell._reported_turn_msgs, 2)
        ell._reported_turn_msgs = 0                      # _one_turn 开头做的事
        self.inst._current_turn = [{'role': 'tool', 'content': 'NEWTURN'}]
        rec = []
        self._stub('新一轮', record=rec)
        self._run()
        self.assertIn('NEWTURN', rec[0]['messages'][1]['content'])

    def test_stale_main_history_is_not_in_the_context(self):
        """这条路汇报的是在跑的子代理，主 agent 的旧对话是另一个话题。

        Orin5 实测：11:24:13 刚派出"调研比亚迪海豹"的子代理，11:24:31 播出来的却是
        "理想L6和问界M7的配置对比数据都查到了"——说的是上一个已经结束的任务。旧历史
        混进来不只是噪声，它会直接把汇报带跑偏。
        """
        self.inst._turns = [[{'role': 'user', 'content': 'OLDTOPIC理想L6'}]]
        self._work(turns=[[{'role': 'tool', 'content': 'FRESHFINDING'}]])
        rec = []
        self._stub('好了', record=rec)
        self._run()
        body = rec[0]['messages'][1]['content']
        self.assertIn('FRESHFINDING', body)
        self.assertNotIn('OLDTOPIC', body)

    def test_skip_does_not_fire_but_restarts(self):
        """**防永久静音**：SKIP 也要重新计时，否则这条路就此断掉。"""
        self._stub('SKIP')
        ell._last_report_text_global = '上次说的'
        self._run()
        self.assertEqual(self.fired, [])
        self.assertEqual(ell._last_report_text_global, '上次说的')
        self.assertIsNotNone(ell._silence_countdown)

    def test_empty_response_restarts(self):
        self._stub('')
        self._run()
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_llm_failure_restarts(self):
        """**防永久静音**：失败也要重新计时。"""
        self._stub(raises=RuntimeError('endpoint down'))
        self._run()
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_timeout_is_bounded_and_restarts(self):
        """**防永久静音**：超时也要重新计时。"""
        self._cfg(narration_timeout_s=1)
        self._stub(hang=True)
        t0 = time.time()
        self._run()
        self.assertLess(time.time() - t0, 5)
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_busy_mouth_restarts_without_spending_a_call(self):
        called = []
        self._stub('不该被调用', record=called)
        self._speaking()
        self._run()
        self.assertEqual(called, [])
        self.assertEqual(self.fired, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_all_bindings_skipped_restarts(self):
        """**防永久静音**：gate 时嘴空着、真要说时被占了。"""
        self._stub('说点什么')

        async def _fire(hook_id, params=None, **kw):
            self.fired.append((hook_id, params, kw))
            return [{'result': {'skipped': 'resource busy'}}]
        hooks.fire = _fire
        ell._last_report_text_global = '旧的'
        self._run()
        self.assertEqual(ell._last_report_text_global, '旧的')
        self.assertEqual(ell._pending_narration_feedback, [])
        self.assertIsNotNone(ell._silence_countdown)

    def test_identical_repeat_is_suppressed(self):
        """和上次说的一模一样就别再说一遍。

        prompt 里写了"不要重复上次播报过的"，但 Orin5 实测它照样逐字重复（两次相隔 25 秒，
        子代理近况没变化，它既没说新东西也没 SKIP）。框架能判的就框架判。
        """
        self._stub('理想L6的配置、销量、竞品和口碑都查到了，正在整理成报告。')
        self._run()
        self.assertEqual(len(self.fired), 1)
        self._run()                       # 第二次生成同一句
        self.assertEqual(len(self.fired), 1, '重复的一句不该再播一遍')
        self.assertIsNotNone(ell._silence_countdown)   # 但要继续计时

    def test_repeat_check_ignores_punctuation_and_spacing(self):
        self._stub('找完客厅了，接下来去卧室')
        self._run()
        self._stub('找完客厅了  接下来去卧室。')
        self._run()
        self.assertEqual(len(self.fired), 1)

    def test_a_genuinely_new_report_still_fires(self):
        """归一只做相等判断，不做模糊相似度 —— 阈值调错会把真正的新进展也压掉。"""
        agent = self._work(turns=[[{'role': 'tool', 'content': '客厅'}]])
        self._stub('找完客厅了，接下来去卧室')
        self._run()
        # 有新进展（多了一条 turn），内容也确实变了 → 必须照播
        agent.context = type('Ctx', (), {'turns': [
            [{'role': 'tool', 'content': '客厅'}],
            [{'role': 'tool', 'content': '卧室'}]]})()
        self._stub('卧室也找完了，没找到，去阳台看看')
        self._run()
        self.assertEqual(len(self.fired), 2)

    def test_overlong_report_is_truncated(self):
        self._stub('啊' * 500)
        self._run()
        # 这段话会注册成 ACP pending，超时按长度算，下一个工具调用都得等它播完。
        self.assertLessEqual(len(self.fired[0][1]['text']), ell._NARRATION_MAX_CHARS)

    def test_no_work_stops_and_spends_nothing(self):
        """活干完了就该安静 —— 而且连 LLM 调用都不该花。"""
        called = []
        self._stub('不该被调用', record=called)
        subagent._manager_instance = None
        self._run()
        self.assertEqual(called, [])
        self.assertIsNone(ell._silence_countdown)

    def test_auto_narration_off_stops(self):
        called = []
        self._stub('不该被调用', record=called)
        skills_mod._notify_override = False
        self._run()
        self.assertEqual(called, [])
        self.assertIsNone(ell._silence_countdown)

    def test_no_bindings_stops(self):
        called = []
        self._stub('不该被调用', record=called)
        hooks._registry.clear()
        self._run()
        self.assertEqual(called, [])
        self.assertIsNone(ell._silence_countdown)

    def test_restricted_turn_work_is_not_narrated(self):
        """不可信 bot / viewer 的 turn 本来就不播任何东西，它派出去的活也不该。"""
        called = []
        self._stub('不该被调用', record=called)
        ell._last_turn_restricted = True
        self._run()
        self.assertEqual(called, [])


# ── set_progress_report ───────────────────────────────────────────────────

class TestSetProgressReport(_Fixture):
    def setUp(self):
        super().setUp()
        self._cfg(narration_silence_seconds=15)

    def _call(self, **kw):
        return asyncio.run(skills_tools.set_progress_report(**kw))

    def test_seconds_override(self):
        self._call(seconds=15)
        self.assertEqual(_narration_thresholds()[1], 15)

    def test_no_args_changes_nothing(self):
        out = self._call()
        self.assertIn('没有改动', out)
        self.assertIsNone(skills_mod.get_report_override())

    def test_zero_disables_and_stops_countdown(self):
        """**防意外静音的反面**：关掉之后没有事件会再启动计时，必须当场停掉。"""
        async def go():
            ell._start_countdown()
            out = await skills_tools.set_progress_report(seconds=0)
            return out, ell._silence_countdown
        out, timer = asyncio.run(go())
        self.assertIn('已关闭', out)
        self.assertEqual(_narration_thresholds()[1], 0)
        self.assertIsNone(timer)

    def test_reenabling_restarts_countdown(self):
        """从关闭改回非 0：没有别的事件会来启动它，工具自己要负责。"""
        async def go():
            await skills_tools.set_progress_report(seconds=0)
            await skills_tools.set_progress_report(seconds=20)
            return ell._silence_countdown
        self.assertIsNotNone(asyncio.run(go()))

    def test_restore_default(self):
        self._call(seconds=1)
        out = self._call(restore_default=True)
        self.assertIn('恢复默认', out)
        self.assertIsNone(skills_mod.get_report_override())
        self.assertEqual(_narration_thresholds()[1], 15)

    def test_negative_clamps_to_zero(self):
        self._call(seconds=-5)
        self.assertEqual(_narration_thresholds()[1], 0)

    def test_return_string_states_effective_value(self):
        self.assertIn('15 秒', self._call(seconds=15))

    def test_registers_without_raising(self):
        """_build_system_tools 的类型映射只认 {str,int,float,bool}；联合类型会在
        **启动阶段** KeyError 挂掉整个进程。"""
        d = _build_system_tools([('set_progress_report', skills_tools.set_progress_report)])
        schema = d['set_progress_report']['schema']
        self.assertEqual(schema['parameters']['required'], [])
        self.assertEqual(set(schema['parameters']['properties']), {'seconds', 'restore_default'})
        self.assertIn('主动进展汇报', schema['description'])

    def test_not_reachable_from_restricted_turns(self):
        # bot 身份可伪造；任何群里的 bot 都能关掉进度播报是不可接受的。
        self.assertFalse(_restricted_channel_tool_allowed('set_progress_report', bot_restricted=True))
        self.assertFalse(_restricted_channel_tool_allowed('set_progress_report', bot_restricted=False))


class TestSkillsNoLongerPredeclareNarration(_Fixture):
    """技能不再预先声明要不要播报 —— 播不播由 agent-core 运行时自己判断。"""

    def test_skill_toggle_does_not_clobber_set_auto_narration(self):
        asyncio.run(skills_tools.set_auto_narration(False))
        saved = config.main.get('skills', {})
        try:
            config.main['skills'] = {'installed': [{
                'slug': 'chess', 'name': '下棋', 'active': True,
                'instruction': 'x', 'oneLiner': 'y', 'narrationDefault': True}]}
            asyncio.run(skills_tools.activate_skill(slug='chess'))
            self.assertFalse(ell._narration_enabled())
            asyncio.run(skills_tools.deactivate_skill(slug='chess'))
            self.assertFalse(ell._narration_enabled())
        finally:
            config.main['skills'] = saved

    def test_recompute_helper_is_gone(self):
        self.assertFalse(hasattr(skills_mod, '_recompute_notify_override'))

    def test_legacy_narration_default_field_is_ignored_not_fatal(self):
        saved = config.main.get('skills', {})
        try:
            config.main['skills'] = {'installed': [{
                'slug': 'old', 'name': '旧技能', 'active': True,
                'instruction': 'x', 'oneLiner': 'y', 'narrationDefault': False}]}
            self.assertEqual([s['slug'] for s in skills_mod.visible_skills()], ['old'])
        finally:
            config.main['skills'] = saved


class TestRoundsArmIsGone(unittest.TestCase):
    """轮数维度已删除 —— 纯时间触发，定时器全局负责。"""

    def test_helpers_removed(self):
        for name in ('_narration_decision', '_narration_threshold_hit',
                     '_maybe_report_progress', 'silent_seconds', '_note_user_output'):
            self.assertFalse(hasattr(ell, name), f'{name} 应该已经删掉')

    def test_config_key_removed(self):
        src = pathlib.Path(__file__).resolve().parents[1] / 'src'
        for rel in ('start.py', 'api/mcp_manage.py'):
            self.assertNotIn('narration_silence_rounds', (src / rel).read_text(),
                             f'{rel} 里还有轮数配置项')
        # config.py 里唯一允许出现的地方是迁移的删除清单 —— 它得先知道键名才能删掉它。
        cfg = (src / 'config.py').read_text()
        head = cfg[:cfg.index('def _migrate(')]
        self.assertNotIn('narration_silence_rounds', head,
                         'config.py 的默认值里还有轮数配置项')
        self.assertIn("'narration_silence_rounds'", cfg.split('_narration_removed')[1][:200],
                      '迁移的删除清单里应当有这个废弃键')


if __name__ == '__main__':
    unittest.main()
