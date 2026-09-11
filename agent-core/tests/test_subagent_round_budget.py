"""test_subagent_round_budget.py — 轮次用完时，已经查到的东西必须交回去。

Orin5 实测（subagent 06421b6f，研究英伟达股价）：10 轮里发了 22 次 WebSearch + 1 次
WebFetch，四分钟的检索成果，主 agent 收到的是

    子代理 [06421b6f] ⏱ timeout: 研究英伟达（NVIDIA，股票代码 NVDA）的股价趋势分析
    摘要: (max rounds reached)

—— 一个字都没回来。撞上限时 subagent 几乎总是停在工具调用中间，于是 `content` 为空，
落到那句占位符上。这里钉住三件事：

1. `max_rounds=0` 是"用配置默认值"的哨兵，不是"零轮"。以前 SubagentSpec 硬编码 10，
   配置里的 `subagent.default_max_rounds` 谁都没读（config.py 写 50、manager.py 写 10，
   两个值都是死的）。
2. 用完轮次会再做一次**不带工具**的收尾调用，把结果写出来，状态是 `partial` 而不是
   `timeout` —— 调用方得能分辨 output 是答案还是占位符。
3. 收尾调用失败不能比以前更糟：回退到原来的占位符。
"""
import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import asyncio  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from subagent.agent import Subagent, _FALLBACK_MAX_ROUNDS  # noqa: E402
from subagent.context import SubagentContext  # noqa: E402
from subagent.protocol import (  # noqa: E402
    STATUS_PARTIAL,
    SubagentSpec,
)


class _FakeClient:
    """替掉 `import client as _client`：记录每次调用，按脚本返回。

    subagent 的 LLM 调用都走 `client.call(message_list=..., tool_list=...)`，而且是在
    函数体里 lazy import 的，所以塞进 sys.modules 就能拦住。
    """

    def __init__(self, wrap_up_text='整理结果：NVDA 收盘 180.50 美元。', fail_wrap_up=False):
        self.calls = []
        self._wrap_up_text = wrap_up_text
        self._fail_wrap_up = fail_wrap_up

    async def call(self, message_list=None, tool_list=None, **kw):
        self.calls.append({'messages': message_list, 'tools': tool_list,
                           'trace_id': kw.get('trace_id', '')})
        if not tool_list:  # 收尾调用：没有工具可用
            if self._fail_wrap_up:
                raise RuntimeError('LLM 挂了')
            return {'content': self._wrap_up_text, 'tool_calls': []}
        # 普通一轮：永远再调一个工具，于是永远不会自己停 —— 正是撞上限的形状
        return {
            'content': '',
            'tool_calls': [{
                'id': f'tc{len(self.calls)}',
                'function': {'name': 'subagent_report',
                             'arguments': '{"progress": "查到一条数据"}'},
            }],
        }


def _install_fake_client(fake):
    mod = types.ModuleType('client')
    mod.call = fake.call
    sys.modules['client'] = mod
    return mod


class TestMaxRoundsSentinel(unittest.TestCase):
    def test_zero_means_use_the_default_not_zero_rounds(self):
        """哨兵必须在跑之前解析掉，否则 `range(0)` 直接空转返回。"""
        spec = SubagentSpec(goal='研究 NVDA')
        self.assertEqual(spec.max_rounds, 0, 'dataclass 默认值应当是哨兵')
        agent = Subagent(spec, agent_id='sentinel1')
        self.assertEqual(agent.spec.max_rounds, _FALLBACK_MAX_ROUNDS)

    def test_explicit_value_is_respected(self):
        agent = Subagent(SubagentSpec(goal='g', max_rounds=3), agent_id='sentinel2')
        self.assertEqual(agent.spec.max_rounds, 3)

    def test_prompt_states_the_resolved_budget(self):
        """system prompt 里写的轮数不能还是 0。"""
        agent = Subagent(SubagentSpec(goal='g'), agent_id='sentinel3')
        self.assertIn(f'最多 {_FALLBACK_MAX_ROUNDS} 轮', agent.context.system_prompt)

    def test_manager_resolves_from_config(self):
        from subagent.manager import SubagentManager
        mgr = SubagentManager(llm_client=None)
        mgr._cfg = {**mgr._cfg, 'default_max_rounds': 42, 'default_timeout_s': 777}
        spec = SubagentSpec(goal='研究 NVDA')
        asyncio.run(mgr.spawn(spec))
        self.assertEqual(spec.max_rounds, 42,
                         'subagent.default_max_rounds 又变成死配置了')
        self.assertEqual(spec.timeout_s, 777,
                         'subagent.default_timeout_s 又变成死配置了')


class TestIdleTimeoutSentinel(unittest.TestCase):
    """`default_timeout_s` 和 max_rounds 一样也是死配置：生效的是 SubagentSpec 的 300.0。

    哨兵只能用负数 —— `manager._schedule` 判的是 `timeout_s > 0`，0 已经表示"不装看门狗"。
    """

    def test_negative_is_the_sentinel(self):
        spec = SubagentSpec(goal='g')
        self.assertLess(spec.timeout_s, 0)
        agent = Subagent(spec, agent_id='idle1')
        self.assertEqual(agent.spec.timeout_s, 600.0)

    def test_zero_still_means_no_watchdog(self):
        """别把"不要超时"也当成哨兵吃掉。"""
        agent = Subagent(SubagentSpec(goal='g', timeout_s=0), agent_id='idle2')
        self.assertEqual(agent.spec.timeout_s, 0)

    def test_explicit_value_is_respected(self):
        agent = Subagent(SubagentSpec(goal='g', timeout_s=45), agent_id='idle3')
        self.assertEqual(agent.spec.timeout_s, 45)


class TestWrapUpOnExhaustion(unittest.TestCase):
    def setUp(self):
        self._saved_client = sys.modules.get('client')

    def tearDown(self):
        if self._saved_client is not None:
            sys.modules['client'] = self._saved_client
        else:
            sys.modules.pop('client', None)

    def _run(self, fake, max_rounds=2):
        _install_fake_client(fake)
        agent = Subagent(SubagentSpec(goal='研究 NVDA 股价', max_rounds=max_rounds),
                         agent_id='wrapup01')
        return asyncio.run(agent.run())

    def test_findings_are_written_up_instead_of_discarded(self):
        fake = _FakeClient()
        result = self._run(fake)
        self.assertEqual(result.status, STATUS_PARTIAL)
        self.assertEqual(result.output, '整理结果：NVDA 收盘 180.50 美元。')
        self.assertNotIn('max rounds reached', result.output)
        self.assertEqual(result.rounds_used, 2)
        self.assertIn('max_rounds=2', result.error or '',
                      '停下来的原因仍然要能查到')

    def test_wrap_up_call_has_no_tools(self):
        """预算已经花完了，再给工具只会换来一次没人读的检索。"""
        fake = _FakeClient()
        self._run(fake)
        last = fake.calls[-1]
        self.assertEqual(last['tools'], [])
        self.assertIn('wrap_up', last['trace_id'])
        self.assertIn('不能再调用任何工具', last['messages'][-1]['content'])

    def test_wrap_up_sees_the_history(self):
        fake = _FakeClient()
        self._run(fake)
        roles = [m.get('role') for m in fake.calls[-1]['messages']]
        self.assertIn('tool', roles, '收尾调用没带上工具结果，等于让它凭空编')

    def test_failed_wrap_up_falls_back_to_the_old_placeholder(self):
        """只能变好，不能变坏：收尾挂了就退回原行为。"""
        fake = _FakeClient(fail_wrap_up=True)
        result = self._run(fake)
        self.assertEqual(result.status, STATUS_PARTIAL)
        self.assertEqual(result.output, '(max rounds reached)')


class TestCompressionKeepsFindings(unittest.TestCase):
    """压缩每轮都在烧掉刚查到的东西，所以它一次次重查同一个问题。

    Orin5 那次：支撑位/阻力位搜了 round 2、6、9，分析师目标价搜了 round 0、1、5、8。
    日志里的 msgs= 13→10、13→9、12→9 就是压缩在几乎每轮触发。
    """

    def _ctx(self, turns):
        ctx = SubagentContext(SubagentSpec(goal='g', max_rounds=30))
        ctx.turns = [[{'role': 'tool', 'tool_call_id': f't{i}',
                       'content': f'搜索结果 {i}'}] for i in range(turns)]
        return ctx

    def test_keeps_more_than_two_recent_turns(self):
        ctx = self._ctx(10)
        sys.modules.pop('client', None)  # 让 compress 走 fallback 分支，不打 LLM
        asyncio.run(ctx.compress())
        self.assertEqual(len(ctx.turns), SubagentContext._KEEP_RECENT_TURNS)
        self.assertGreater(SubagentContext._KEEP_RECENT_TURNS, 2)

    def test_nothing_compressed_below_the_keep_window(self):
        ctx = self._ctx(SubagentContext._KEEP_RECENT_TURNS)
        asyncio.run(ctx.compress())
        self.assertIsNone(ctx.summary)
        self.assertEqual(len(ctx.turns), SubagentContext._KEEP_RECENT_TURNS)

    def test_threshold_survives_a_single_search_turn(self):
        """一次 WebSearch 就能几万字符；阈值低于它就等于每轮都压。"""
        ctx = SubagentContext(SubagentSpec(goal='g'))
        ctx.turns = [[{'role': 'tool', 'content': 'x' * 25000}]]
        self.assertFalse(ctx.needs_compression())


class TestStaleConfigRowIsMigrated(unittest.TestCase):
    """已部署机器上的旧默认值不会自己更新。

    `_seed_defaults` 是 `INSERT OR IGNORE`，整行粒度 —— 'subagent' 行一旦存在，改
    `_DB_DEFAULTS` 对它没有任何影响。Orin5 上实测读出来是 20000 / 300，光改代码默认值
    等于这次的压缩修复在现网一台都吃不到。

    config 在 import 时就跑迁移且只跑一次，所以起子进程验。
    """

    def _migrated(self, threshold: int, timeout_s: int) -> dict:
        import json
        import subprocess
        import sqlite3
        db = os.path.join(tempfile.mkdtemp(), 'm.db')
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        conn.execute('INSERT INTO config VALUES (?,?)', (
            'subagent', json.dumps({'max_concurrent': 2, 'default_max_rounds': 50,
                                    'compress_threshold_chars': threshold,
                                    'default_timeout_s': timeout_s})))
        conn.commit()
        conn.close()
        src = str(pathlib.Path(__file__).resolve().parents[1] / 'src')
        out = subprocess.run(
            [sys.executable, '-c',
             'import json, config; print("VALUE", json.dumps(config.main["subagent"]))'],
            cwd=src, env={**os.environ, 'DB_PATH': db, 'PYTHONPATH': src},
            capture_output=True, text=True, timeout=60)
        line = [l for l in out.stdout.splitlines() if l.startswith('VALUE')]
        self.assertTrue(line, f'子进程没跑起来: {out.stdout}\n{out.stderr}')
        return json.loads(line[0][len('VALUE '):])

    def test_stale_defaults_are_bumped(self):
        sa = self._migrated(20000, 300)
        self.assertEqual(sa['compress_threshold_chars'], 40000)
        self.assertEqual(sa['default_timeout_s'], 600)

    def test_unrelated_keys_survive(self):
        self.assertEqual(self._migrated(20000, 300)['default_max_rounds'], 50)

    def test_hand_tuned_values_are_left_alone(self):
        """只认旧默认值这两个数；有人调过就不要覆盖。"""
        sa = self._migrated(35000, 450)
        self.assertEqual(sa['compress_threshold_chars'], 35000)
        self.assertEqual(sa['default_timeout_s'], 450)


class TestSpawnSyncSurfacesPartialOutput(unittest.TestCase):
    """`error or output` 把停下来的原因摆在答案前面，然后把答案整个丢了。"""

    def _call(self, result):
        from subagent.protocol import SubagentResult  # noqa: F401
        from subagent.tools import SubagentTools

        class _Mgr:
            async def spawn_and_wait(self, spec, timeout=120):
                return result

        tools = SubagentTools(_Mgr())
        return asyncio.run(tools.subagent_spawn_sync(goal='研究 NVDA 股价'))

    def test_output_wins_over_error(self):
        from subagent.protocol import SubagentResult
        text = self._call(SubagentResult(
            agent_id='a', status=STATUS_PARTIAL,
            output='NVDA 收盘 180.50 美元。',
            error='Reached max_rounds=30'))
        self.assertIn('NVDA 收盘 180.50 美元。', text)
        self.assertIn('Reached max_rounds=30', text, '停下来的原因不该被吞掉')
        self.assertLess(text.index('NVDA'), text.index('Reached'))

    def test_errorless_output_has_no_dangling_parens(self):
        from subagent.protocol import SubagentResult
        text = self._call(SubagentResult(agent_id='a', status=STATUS_PARTIAL,
                                         output='答案', error=''))
        self.assertEqual(text, '[子代理partial] 答案')

    def test_empty_output_still_reports_the_error(self):
        from subagent.protocol import SubagentResult
        text = self._call(SubagentResult(agent_id='a', status='failed',
                                         output='', error='工具全部不可用'))
        self.assertIn('工具全部不可用', text)


if __name__ == '__main__':
    unittest.main()
