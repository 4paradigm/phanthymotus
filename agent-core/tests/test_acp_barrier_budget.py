"""barrier 的超时预算：同一资源上的动作串行，预算要相加。

天轶 2026-10-15 的展厅导览，最后一站：barrier 同时等一条自动播报和一条 487 字的
讲解，两者都占 `mouth`，于是在驱动的播放队列里是**前后**播的。预算取了 max
（讲解那条的 172s），而真正播完要 208s。barrier 在 13:21:50 放弃，9 秒之后那条才
完成 —— 被挡住的那次调用正是结束语的 `tts.speak`，它于是一个字都没播出去。

`_barrier_budget` 把互相冲突的动作并成一条串行链求和，不冲突的取 max。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest agent-core/tests/test_acp_barrier_budget.py -q
"""
import asyncio
import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import mcp_client  # noqa: E402


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._saved = tuple(dict(d) for d in mcp_client.pending_tables())
        for d in mcp_client.pending_tables():
            d.clear()

    def tearDown(self):
        for d, saved in zip(mcp_client.pending_tables(), self._saved):
            d.clear()
            d.update(saved)

    def _add(self, aid, resource, *, tool='tts', timeout=5.0, done=False):
        ev = asyncio.Event()
        if done:
            ev.set()
        mcp_client._pending_actions[aid] = ev
        mcp_client._pending_tools[aid] = tool
        mcp_client._pending_timeouts[aid] = timeout
        mcp_client._pending_resources[aid] = resource
        return ev


class TestBarrierBudget(_Fixture):
    def test_same_resource_sums(self):
        """一张嘴上排了三段话，预算是三段之和，不是最长的那一段。"""
        self._add('tts-1', frozenset({'mouth'}), timeout=20.0)
        self._add('tts-2', frozenset({'mouth'}), timeout=60.0)
        self._add('tts-3', frozenset({'mouth'}), timeout=172.0)
        self.assertEqual(
            mcp_client._barrier_budget(['tts-1', 'tts-2', 'tts-3'], 120), 252.0)

    def test_the_regression_it_was_written_for(self):
        """自动播报 21s + 讲解 172s：旧算法给 172s，真实要 193s 以上。"""
        self._add('tts-narration', frozenset({'mouth'}), timeout=21.0)
        self._add('tts-p15', frozenset({'mouth'}), timeout=172.0)
        budget = mcp_client._barrier_budget(['tts-narration', 'tts-p15'], 120)
        self.assertEqual(budget, 193.0)
        self.assertGreater(budget, 172.0, '仍然是 max —— 结束语还会被丢掉')

    def test_independent_resources_take_max_not_sum(self):
        """嘴和底盘并行跑：相加等于白等一倍。"""
        self._add('tts-1', frozenset({'mouth'}), timeout=30.0)
        self._add('nav-1', frozenset({'base'}), tool='loco', timeout=180.0)
        self.assertEqual(
            mcp_client._barrier_budget(['tts-1', 'nav-1'], 120), 180.0)

    def test_overlapping_resource_sets_are_one_chain(self):
        """{mouth} 与 {mouth, speaker_mix} 不相等但互相冲突，必须算一条链。

        按「resource 集合相等」分组会把它们分开，于是又回到取 max。
        """
        self._add('tts-1', frozenset({'mouth'}), timeout=30.0)
        self._add('mix-1', frozenset({'mouth', 'speaker_mix'}),
                  tool='audio_mixer', timeout=40.0)
        self._add('nav-1', frozenset({'base'}), tool='loco', timeout=50.0)
        self.assertEqual(
            mcp_client._barrier_budget(['tts-1', 'mix-1', 'nav-1'], 120), 70.0)

    def test_undeclared_resource_joins_every_chain(self):
        """未声明 = 与一切互斥（见 resources_conflict），所以全部并成一条链。"""
        self._add('legacy-1', None, tool='switch_mode', timeout=10.0)
        self._add('tts-1', frozenset({'mouth'}), timeout=30.0)
        self._add('nav-1', frozenset({'base'}), tool='loco', timeout=50.0)
        self.assertEqual(
            mcp_client._barrier_budget(['legacy-1', 'tts-1', 'nav-1'], 120), 90.0)

    def test_infinite_stays_infinite(self):
        """background 且没声明 timeout 的动作是无限等 —— 求和不能把它变成有限。"""
        self._add('music-1', frozenset({'mouth'}), timeout=float('inf'))
        self._add('tts-1', frozenset({'mouth'}), timeout=30.0)
        self.assertEqual(
            mcp_client._barrier_budget(['music-1', 'tts-1'], 120), float('inf'))

    def test_sum_is_capped(self):
        """相加不能把 barrier 变成无上限的等待。"""
        for i in range(50):
            self._add(f'tts-{i}', frozenset({'mouth'}), timeout=180.0)
        self.assertEqual(
            mcp_client._barrier_budget([f'tts-{i}' for i in range(50)], 120),
            mcp_client.MAX_PENDING_TIMEOUT_S)

    def test_missing_timeout_falls_back_to_default(self):
        ev = asyncio.Event()
        mcp_client._pending_actions['orphan'] = ev
        mcp_client._pending_resources['orphan'] = frozenset({'mouth'})
        self.assertEqual(mcp_client._barrier_budget(['orphan'], 7.0), 7.0)

    def test_empty_falls_back_to_default(self):
        self.assertEqual(mcp_client._barrier_budget([], 11.0), 11.0)


class TestBarrierActuallyWaitsLonger(_Fixture):
    def test_two_queued_speaks_are_both_awaited(self):
        """端到端：两条都在预算内完成时，barrier 必须报 completed 而不是 timeout。

        每条 0.15s，旧算法的预算是 max(0.15)=0.15s —— 第二条还没完就超时。
        """
        first = self._add('tts-1', frozenset({'mouth'}), timeout=0.15)
        second = self._add('tts-2', frozenset({'mouth'}), timeout=0.15)

        async def _run():
            async def _finish():
                await asyncio.sleep(0.1)
                first.set()
                await asyncio.sleep(0.12)
                second.set()
            asyncio.create_task(_finish())
            t0 = time.monotonic()
            out = await mcp_client.await_pending(want=frozenset({'mouth'}),
                                                scoped=True, concurrent=True)
            return out, time.monotonic() - t0

        out, elapsed = asyncio.run(_run())
        self.assertEqual(out['status'], 'completed', out)
        self.assertGreater(elapsed, 0.2, '第二条根本没被等到')


if __name__ == '__main__':
    unittest.main()
