"""
test_acp_background.py — 长时间后台动作（音乐）不该把别人堵住，但仍然占着通道。

音乐是这套系统里第一个「播很久、在后台、不阻塞对话」的动作，而 barrier 原先没有这个
类别。两件事因此是坏的：

1. **时长在注册时不可知。** timeout 只能来自 `len(text)/3`，而一首歌没有 text，于是
   每首歌共用一个静态上限；超了就在播放中途被静默回收，之后 barrier 认为通道空着，
   下一句话直接压着音乐说（超时与成功在 _dispatch 层无法区分，这是有意的设计）。

2. **`pendings_to_wait_for` 有第二条等待理由，和资源无关**：只看是不是同一个 agent
   起的。所以一首在播的歌会把主 agent 的**下一次任意调用**堵到歌放完 —— 比堵住
   finish 严重得多，也是 `background` 这个标志存在的真正原因。

`background` 只改「谁等谁」，不改「忙不忙」：resource_actually_busy 照样算它占用，
否则「现在在放歌吗」就没有答案了。
"""
import asyncio
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import mcp_client  # noqa: E402

MOUTH = frozenset({'mouth'})
MUSIC = frozenset({'music'})


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._saved = tuple(dict(d) for d in mcp_client.pending_tables())
        for d in mcp_client.pending_tables():
            d.clear()

    def tearDown(self):
        for d, saved in zip(mcp_client.pending_tables(), self._saved):
            d.clear()
            d.update(saved)

    def _add(self, aid, resource, *, owner='main', background=False, done=False):
        ev = asyncio.Event()
        if done:
            ev.set()
        mcp_client._pending_actions[aid] = ev
        mcp_client._pending_resources[aid] = resource
        mcp_client._pending_owner[aid] = owner
        mcp_client._pending_background[aid] = background
        mcp_client._pending_timeouts[aid] = 5.0


class TestPendingTimeoutLadder(_Fixture):
    """四级优先：驱动量出来的 → text 字数 → schema 静态值 → 后台不设限。"""

    def test_driver_declared_duration_wins_and_gets_grace(self):
        t = mcp_client._pending_timeout({'timeout': 120}, {}, {'expected_duration_s': 214})
        self.assertEqual(t, 224.0)          # 214 + 默认 10s 余量

    def test_declared_duration_beats_a_text_argument(self):
        """两者都有时听驱动的：它量的是真实播放时长，字数只是估算。"""
        t = mcp_client._pending_timeout({}, {'text': '字' * 300},
                                        {'expected_duration_s': 10})
        self.assertEqual(t, 20.0)

    def test_text_rule_is_unchanged(self):
        self.assertEqual(mcp_client._pending_timeout({}, {'text': '字' * 300}, {}),
                         300 / 3 + 10)

    def test_schema_default_is_unchanged(self):
        self.assertEqual(mcp_client._pending_timeout({'timeout': 60}, {}, {}), 60)
        self.assertEqual(mcp_client._pending_timeout({}, {}, {}), 120)

    def test_background_with_no_timeout_has_no_deadline(self):
        """只有驱动的 /acp/complete 能结束它 —— 一首歌多长没人猜得到。"""
        t = mcp_client._pending_timeout({'background': True, 'timeout': 0}, {}, {})
        self.assertEqual(t, float('inf'))

    def test_background_still_honours_an_explicit_timeout(self):
        t = mcp_client._pending_timeout({'background': True, 'timeout': 90}, {}, {})
        self.assertEqual(t, 90)

    def test_absurd_duration_is_clamped(self):
        """await_pending 取 max(所有要等的 timeout)，所以一条畸形响应会拖垮无关的 finish。"""
        t = mcp_client._pending_timeout({}, {}, {'expected_duration_s': 1e9})
        self.assertEqual(t, mcp_client.MAX_PENDING_TIMEOUT_S)

    def test_nonsense_durations_are_ignored(self):
        """bool 要单独挡：True 是 int 的子类，isinstance 会放它过去。"""
        for bad in (0, -5, float('nan'), float('inf'), True, False, '214', None, [214]):
            with self.subTest(bad=bad):
                self.assertEqual(
                    mcp_client._pending_timeout({'timeout': 77}, {}, {'expected_duration_s': bad}),
                    77, msg=repr(bad))

    def test_nonsense_grace_falls_back_to_ten(self):
        t = mcp_client._pending_timeout({'grace': 'soon'}, {}, {'expected_duration_s': 100})
        self.assertEqual(t, 110.0)

    def test_a_non_dict_result_is_tolerated(self):
        self.assertEqual(mcp_client._pending_timeout({'timeout': 30}, {}, None), 30)


class TestBackgroundIsNotWaitedOn(_Fixture):
    def test_background_is_exempt_from_the_resource_branch(self):
        self._add('music-1', MUSIC, background=True)
        self.assertEqual(
            mcp_client.pendings_to_wait_for(MUSIC, owner='other', concurrent=True), [])

    def test_background_is_exempt_from_the_own_ordering_branch(self):
        """这是最要紧的一条：它和通道无关，只看 owner。

        没有这个豁免，一首在播的歌会把主 agent 的下一次**任意**调用堵到歌放完 ——
        说话、移动、什么都算。
        """
        self._add('music-1', MUSIC, owner='main', background=True)
        self.assertEqual(
            mcp_client.pendings_to_wait_for(MOUTH, owner='main', concurrent=False), [])

    def test_a_blocking_pending_still_serialises(self):
        """豁免只对后台动作生效，别的一切照旧。"""
        self._add('speak-1', MOUTH, owner='main')
        self.assertEqual(
            mcp_client.pendings_to_wait_for(frozenset({'base'}), owner='main',
                                           concurrent=False), ['speak-1'])

    def test_background_still_holds_its_channel(self):
        """「在放歌吗」必须还能问出来 —— 只是没人等它。"""
        self._add('music-1', MUSIC, background=True)
        self.assertTrue(mcp_client.resource_actually_busy(MUSIC))
        self.assertEqual(mcp_client.conflicting_pending(MUSIC), ['music-1'])

    def test_has_blocking_pendings_ignores_background(self):
        self._add('music-1', MUSIC, background=True)
        self.assertFalse(mcp_client.has_blocking_pendings())
        self.assertTrue(mcp_client.get_pending_actions())   # 它确实还在
        self._add('speak-1', MOUTH)
        self.assertTrue(mcp_client.has_blocking_pendings())

    def test_is_background_reports_the_flag(self):
        self._add('music-1', MUSIC, background=True)
        self._add('speak-1', MOUTH)
        self.assertTrue(mcp_client.is_background('music-1'))
        self.assertFalse(mcp_client.is_background('speak-1'))
        self.assertFalse(mcp_client.is_background('never-registered'))


class TestGlobalWaitSkipsBackground(_Fixture):
    """finish / peer_call / subagent_spawn 走 scoped=False，等的是「所有 pending」。"""

    def test_finish_does_not_wait_out_a_track(self):
        self._add('music-1', MUSIC, background=True)
        result = asyncio.run(mcp_client.await_pending(timeout=0.05, scoped=False))
        self.assertEqual(result['status'], 'no_pending')

    def test_global_still_waits_for_a_blocking_pending(self):
        self._add('speak-1', MOUTH)
        result = asyncio.run(mcp_client.await_pending(timeout=0.05, scoped=False))
        self.assertEqual(result['status'], 'timeout')

    def test_a_completed_background_does_not_hide_a_live_one(self):
        self._add('music-1', MUSIC, background=True)
        self._add('speak-1', MOUTH)
        result = asyncio.run(mcp_client.await_pending(timeout=0.05, scoped=False))
        self.assertEqual(result['status'], 'timeout')
        self.assertIn('speak-1', result.get('actions', ['speak-1']))


class TestRegisterPending(_Fixture):
    def test_flags_are_written_explicitly_not_only_when_true(self):
        """action_id 会被复用，留着上一轮的值就是串味 —— 所以两张表都要无条件写。"""
        mcp_client._pending_background['play-1'] = True
        mcp_client._pending_silent['play-1'] = True
        mcp_client._register_pending(
            'play-1', tool_name='tts', meta={'completion': {}, 'resource': MOUTH},
            args={}, result={})
        self.assertFalse(mcp_client._pending_background['play-1'])
        self.assertFalse(mcp_client._pending_silent['play-1'])

    def test_registration_fills_every_table(self):
        mcp_client._register_pending(
            'music-1', tool_name='phanthy_music',
            meta={'completion': {'background': True, 'silent_completion': True,
                                 'timeout': 0},
                  'resource': MUSIC},
            args={}, result={'expected_duration_s': 100})
        self.assertEqual(mcp_client._pending_resources['music-1'], MUSIC)
        self.assertEqual(mcp_client._pending_tools['music-1'], 'phanthy_music')
        self.assertEqual(mcp_client._pending_timeouts['music-1'], 110.0)
        self.assertTrue(mcp_client._pending_background['music-1'])
        self.assertTrue(mcp_client._pending_silent['music-1'])
        self.assertIn('music-1', mcp_client._pending_started)
        self.assertFalse(math.isinf(mcp_client._pending_timeouts['music-1']))

    def test_forget_clears_the_new_tables_too(self):
        self._add('music-1', MUSIC, background=True)
        mcp_client._pending_silent['music-1'] = True
        mcp_client._forget_pending(['music-1'])
        for d in mcp_client.pending_tables():
            self.assertNotIn('music-1', d)


if __name__ == '__main__':
    unittest.main()
