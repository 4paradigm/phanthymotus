"""
test_acp_resource_barrier.py — barrier 按物理资源互斥，而不是"所有人等所有人"。

原来的 barrier 是全局的：任何一个 pending 动作挡住任何一个会动的工具。说话时不能导航，
一个 subagent 说话挡住其他 subagent 在无关硬件上的全部调用。这把两件正交的事混成了
一件（因果依赖 vs 资源互斥），而且哪件都不对。

真正互斥的是物理通道 —— 一张嘴、一个底盘、一条左臂 —— driver 用 `x-resource` 声明。
**未声明 = 与一切互斥**，这是保守解读，也是让十四个现存 driver 行为不变的前提。

顺带覆盖 `_forget_pending`：pending 的簿记摊在五个 dict 上，漏掉一个就是永久泄漏。
"""
import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import mcp_client  # noqa: E402


class TestParseResources(unittest.TestCase):
    def test_bare_string(self):
        self.assertEqual(mcp_client.parse_resources('mouth'), frozenset({'mouth'}))

    def test_list(self):
        self.assertEqual(mcp_client.parse_resources(['base', 'arm_l']),
                         frozenset({'base', 'arm_l'}))

    def test_whitespace_is_trimmed(self):
        self.assertEqual(mcp_client.parse_resources('  mouth '), frozenset({'mouth'}))

    def test_undeclared_is_none(self):
        self.assertIsNone(mcp_client.parse_resources(None))

    def test_malformed_falls_back_to_none_not_empty(self):
        """打错的 schema 不能静默解锁并行执行 —— 必须退回"与一切互斥"。"""
        for bad in ({'mouth': True}, 42, '', '   ', [], ['', '  '], [None, 7]):
            self.assertIsNone(mcp_client.parse_resources(bad), msg=repr(bad))


class TestResourcesConflict(unittest.TestCase):
    def test_same_channel_conflicts(self):
        self.assertTrue(mcp_client.resources_conflict(
            frozenset({'mouth'}), frozenset({'mouth'})))

    def test_different_channels_do_not(self):
        """说话 ∥ 导航 —— 这正是全局 barrier 白挡掉的。"""
        self.assertFalse(mcp_client.resources_conflict(
            frozenset({'mouth'}), frozenset({'base'})))

    def test_partial_overlap_conflicts(self):
        self.assertTrue(mcp_client.resources_conflict(
            frozenset({'base', 'arm_l'}), frozenset({'arm_l'})))

    def test_undeclared_caller_waits_for_everything(self):
        self.assertTrue(mcp_client.resources_conflict(None, frozenset({'base'})))

    def test_undeclared_holder_blocks_everything(self):
        self.assertTrue(mcp_client.resources_conflict(frozenset({'base'}), None))

    def test_both_undeclared(self):
        self.assertTrue(mcp_client.resources_conflict(None, None))


class _PendingFixture(unittest.TestCase):
    def setUp(self):
        self._saved = (
            dict(mcp_client._pending_actions), dict(mcp_client._pending_results),
            dict(mcp_client._pending_timeouts), dict(mcp_client._pending_tools),
            dict(mcp_client._pending_resources),
        )
        for d in (mcp_client._pending_actions, mcp_client._pending_results,
                  mcp_client._pending_timeouts, mcp_client._pending_tools,
                  mcp_client._pending_resources):
            d.clear()

    def tearDown(self):
        targets = (mcp_client._pending_actions, mcp_client._pending_results,
                   mcp_client._pending_timeouts, mcp_client._pending_tools,
                   mcp_client._pending_resources)
        for d, saved in zip(targets, self._saved):
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


class TestConflictingPending(_PendingFixture):
    def test_only_conflicting_are_returned(self):
        self._add('speak-1', frozenset({'mouth'}))
        self._add('nav-1', frozenset({'base'}), tool='loco')
        self.assertEqual(mcp_client.conflicting_pending(frozenset({'mouth'})),
                         ['speak-1'])
        self.assertEqual(mcp_client.conflicting_pending(frozenset({'base'})),
                         ['nav-1'])

    def test_undeclared_pending_blocks_all(self):
        self._add('legacy-1', None, tool='switch_mode')
        self.assertEqual(mcp_client.conflicting_pending(frozenset({'mouth'})),
                         ['legacy-1'])

    def test_registration_order_is_preserved(self):
        for i in range(4):
            self._add(f'speak-{i}', frozenset({'mouth'}))
        self.assertEqual(mcp_client.conflicting_pending(frozenset({'mouth'})),
                         ['speak-0', 'speak-1', 'speak-2', 'speak-3'])

    def test_no_pending_at_all(self):
        self.assertEqual(mcp_client.conflicting_pending(frozenset({'mouth'})), [])


class TestForgetPending(_PendingFixture):
    def test_clears_every_side_table(self):
        self._add('speak-1', frozenset({'mouth'}))
        mcp_client._pending_results['speak-1'] = {'status': 'completed'}
        mcp_client._forget_pending(['speak-1'])
        for d in (mcp_client._pending_actions, mcp_client._pending_results,
                  mcp_client._pending_timeouts, mcp_client._pending_tools,
                  mcp_client._pending_resources):
            self.assertNotIn('speak-1', d)

    def test_leaves_others_alone(self):
        self._add('speak-1', frozenset({'mouth'}))
        self._add('nav-1', frozenset({'base'}))
        mcp_client._forget_pending(['speak-1'])
        self.assertIn('nav-1', mcp_client._pending_actions)
        self.assertIn('nav-1', mcp_client._pending_resources)

    def test_unknown_id_is_a_noop(self):
        mcp_client._forget_pending(['nope'])   # must not raise

    def test_accepts_a_generator(self):
        self._add('speak-1', frozenset({'mouth'}))
        mcp_client._forget_pending(a for a in ['speak-1'])
        self.assertNotIn('speak-1', mcp_client._pending_actions)


class TestAwaitPendingScoping(_PendingFixture):
    def test_scoped_ignores_non_conflicting(self):
        """底盘还在跑，但要说话 —— 不该等。"""
        self._add('nav-1', frozenset({'base'}), tool='loco', timeout=60.0)
        out = asyncio.run(mcp_client.await_pending(
            want=frozenset({'mouth'}), scoped=True, timeout=1))
        self.assertEqual(out['status'], 'no_pending')
        # 不相干的 pending 必须原样留着
        self.assertIn('nav-1', mcp_client._pending_actions)

    def test_scoped_waits_for_conflicting(self):
        self._add('speak-1', frozenset({'mouth'}), done=True)
        out = asyncio.run(mcp_client.await_pending(
            want=frozenset({'mouth'}), scoped=True, timeout=1))
        self.assertEqual(out['status'], 'completed')
        self.assertEqual(out['actions'], ['speak-1'])

    def test_scoped_clears_only_what_it_waited_on(self):
        self._add('speak-1', frozenset({'mouth'}), done=True)
        self._add('nav-1', frozenset({'base'}), tool='loco')
        asyncio.run(mcp_client.await_pending(
            want=frozenset({'mouth'}), scoped=True, timeout=1))
        self.assertNotIn('speak-1', mcp_client._pending_actions)
        self.assertIn('nav-1', mcp_client._pending_actions)

    def test_global_still_waits_for_all(self):
        """finish 走全局：结束 turn 前不该有任何动作在飞。"""
        self._add('speak-1', frozenset({'mouth'}), done=True)
        self._add('nav-1', frozenset({'base'}), tool='loco', done=True)
        out = asyncio.run(mcp_client.await_pending(scoped=False, timeout=1))
        self.assertEqual(out['status'], 'completed')
        self.assertEqual(sorted(out['actions']), ['nav-1', 'speak-1'])

    def test_timeout_max_is_over_conflicting_only(self):
        """长动作不该把不相干的调用一起拖住。

        nav 声明 60s，speak 声明 0.05s。想说话时只该看到 0.05s。
        """
        self._add('nav-1', frozenset({'base'}), tool='loco', timeout=60.0)
        self._add('speak-1', frozenset({'mouth'}), timeout=0.05)
        loop_t0 = asyncio.run(self._timed_scoped_wait())
        self.assertLess(loop_t0, 5.0, 'barrier 用了 nav 的 60s timeout')

    async def _timed_scoped_wait(self):
        import time
        t0 = time.monotonic()
        out = await mcp_client.await_pending(want=frozenset({'mouth'}), scoped=True)
        self.assertEqual(out['status'], 'timeout')
        return time.monotonic() - t0


class TestActionOutcomes(_PendingFixture):
    """超时不能和成功长得一样。

    超时路径清掉 pending 并让调用方照常往下走 —— 和成功完全同一条路。所以终态必须在
    pending 消失之后仍然可查，否则"播完了"和"回调没来"在下游无法区分，这正是委派把一句
    没播出去的台词报成已播的原因。
    """

    def setUp(self):
        super().setUp()
        self._saved_outcomes = dict(mcp_client._action_outcomes)
        mcp_client._action_outcomes.clear()

    def tearDown(self):
        mcp_client._action_outcomes.clear()
        mcp_client._action_outcomes.update(self._saved_outcomes)
        super().tearDown()

    def test_completed_is_recorded_after_pending_is_gone(self):
        self._add('speak-1', frozenset({'mouth'}), done=True)
        asyncio.run(mcp_client.await_pending(
            want=frozenset({'mouth'}), scoped=True, timeout=1))
        self.assertNotIn('speak-1', mcp_client._pending_actions)
        self.assertEqual(mcp_client.action_outcome('speak-1')['status'], 'completed')

    def test_timeout_is_distinguishable_from_completed(self):
        self._add('speak-1', frozenset({'mouth'}), timeout=0.05)
        out = asyncio.run(mcp_client.await_pending(
            want=frozenset({'mouth'}), scoped=True))
        self.assertEqual(out['status'], 'timeout')
        self.assertEqual(mcp_client.action_outcome('speak-1')['status'], 'timeout')

    def test_tool_name_is_kept(self):
        """终态记录要在 _pending_tools 被清掉之前抓到 tool 名。"""
        self._add('speak-1', frozenset({'mouth'}), tool='tts', done=True)
        asyncio.run(mcp_client.await_pending(scoped=False, timeout=1))
        self.assertEqual(mcp_client.action_outcome('speak-1')['tool'], 'tts')

    def test_unknown_action_has_no_outcome(self):
        self.assertIsNone(mcp_client.action_outcome('never-existed'))

    def test_record_is_bounded(self):
        """诊断尾巴，不是账本 —— 不能无上限增长。"""
        cap = mcp_client._ACTION_OUTCOME_CAP
        for i in range(cap + 25):
            mcp_client._record_outcome(f'a-{i}', 'completed')
        self.assertLessEqual(len(mcp_client._action_outcomes), cap)
        self.assertIsNone(mcp_client.action_outcome('a-0'), 'oldest should be evicted')
        self.assertIsNotNone(mcp_client.action_outcome(f'a-{cap + 24}'))

    def test_forget_without_outcome_records_nothing(self):
        """没有明确终态时不要猜 —— 记一个错的比不记更糟。"""
        self._add('speak-1', frozenset({'mouth'}))
        mcp_client._forget_pending(['speak-1'])
        self.assertIsNone(mcp_client.action_outcome('speak-1'))


if __name__ == '__main__':
    unittest.main()
