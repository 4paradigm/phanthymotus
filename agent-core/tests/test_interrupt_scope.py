"""
test_interrupt_scope.py — 一次打断只取消它**有权**取消的那些 pending。

打断原先没有范围：`_interrupt_active_outputs` 和显式 interrupt 工具两处都是
`for aid in list(_pending_actions.keys())` 全标 cancelled。只有「嘴」一个长动作时这
看不出问题；一旦有了「播很久的后台动作」（音乐），「别说了」就会把音乐的 pending 也
标掉 —— 播放器还在放，而 agent-core 认为通道空了，之后没有任何东西知道它还在响。

范围不写死通道名：由 `hooks.notify_resources(hook_id)` 从绑定自己申报的 `x-resource`
推出来。这条规则 hooks.py 早就为 on_notify 写下了（「不能按工具名猜，机器人的嘴叫
tts / speaker / audio_play 各有各的叫法」），这里只是终于用上。
"""
import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import mcp_client  # noqa: E402

MOUTH = frozenset({'mouth'})
MUSIC = frozenset({'music'})
BASE = frozenset({'base'})


class TestCancellableBy(unittest.TestCase):
    def setUp(self):
        self._saved = tuple(dict(d) for d in mcp_client.pending_tables())
        for d in mcp_client.pending_tables():
            d.clear()

    def tearDown(self):
        for d, saved in zip(mcp_client.pending_tables(), self._saved):
            d.clear()
            d.update(saved)

    def _add(self, aid, resource, *, background=False):
        mcp_client._pending_actions[aid] = asyncio.Event()
        mcp_client._pending_resources[aid] = resource
        mcp_client._pending_background[aid] = background

    def test_stop_talking_leaves_the_song_alone(self):
        """这就是整个改动的理由。"""
        self._add('speak-1', MOUTH)
        self._add('music-1', MUSIC, background=True)
        self.assertEqual(mcp_client.cancellable_by(MOUTH), ['speak-1'])

    def test_stop_the_music_leaves_speech_alone(self):
        self._add('speak-1', MOUTH)
        self._add('music-1', MUSIC)
        self.assertEqual(mcp_client.cancellable_by(MUSIC), ['music-1'])

    def test_an_unscoped_interrupt_still_takes_everything_blocking(self):
        """急停 / TurnCancelled：scope=None。"""
        self._add('speak-1', MOUTH)
        self._add('nav-1', BASE)
        self.assertEqual(sorted(mcp_client.cancellable_by(None)), ['nav-1', 'speak-1'])

    def test_even_an_unscoped_interrupt_spares_background(self):
        """音乐不是执行危险，急停不该连歌一起掐 —— 停歌走 on_interrupt_music。"""
        self._add('speak-1', MOUTH)
        self._add('music-1', MUSIC, background=True)
        self.assertEqual(mcp_client.cancellable_by(None), ['speak-1'])

    def test_an_undeclared_pending_is_cancelled_by_any_scope(self):
        """没申报 x-resource 的驱动本来就被当作与一切互斥。

        如果打断反而碰不到它们，就会留下一个谁也清不掉的 pending —— 比今天的行为更坏。
        """
        self._add('legacy-1', None)
        self.assertEqual(mcp_client.cancellable_by(MOUTH), ['legacy-1'])
        self.assertEqual(mcp_client.cancellable_by(BASE), ['legacy-1'])

    def test_a_partial_channel_overlap_is_cancelled(self):
        self._add('arms-1', frozenset({'arm_l', 'arm_r'}))
        self.assertEqual(mcp_client.cancellable_by(frozenset({'arm_r'})), ['arms-1'])

    def test_a_disjoint_scope_cancels_nothing(self):
        self._add('nav-1', BASE)
        self.assertEqual(mcp_client.cancellable_by(MOUTH), [])

    def test_an_empty_scope_cancels_only_undeclared(self):
        self._add('speak-1', MOUTH)
        self._add('legacy-1', None)
        self.assertEqual(mcp_client.cancellable_by(frozenset()), ['legacy-1'])


class TestScopeComesFromBindings(unittest.TestCase):
    """范围由绑定申报的通道决定，不由工具名决定。"""

    def setUp(self):
        import hooks
        self.hooks = hooks
        self._saved_registry = dict(mcp_client.registry)
        self._saved_hooks = dict(hooks._registry)
        hooks._registry.clear()
        mcp_client.registry.clear()
        mcp_client.registry['mcp-perception'] = {
            'online': True, 'tools': ['tts'],
            'tool_meta': {'mcp__mcp-perception__tts': {'resource': MOUTH}},
        }
        mcp_client.registry['mcp-music'] = {
            'online': True, 'tools': ['phanthy_music'],
            'tool_meta': {'mcp__mcp-music__phanthy_music': {'resource': MUSIC}},
        }

    def tearDown(self):
        self.hooks._registry.clear()
        self.hooks._registry.update(self._saved_hooks)
        mcp_client.registry.clear()
        mcp_client.registry.update(self._saved_registry)

    def test_speech_scope_is_the_mouth(self):
        self.hooks.register('mcp-perception', 'tts',
                            {'on_interrupt_speak': {'action': 'interrupt'}})
        self.assertEqual(self.hooks.notify_resources('on_interrupt_speak'), MOUTH)

    def test_music_scope_is_the_music_channel(self):
        self.hooks.register('mcp-music', 'phanthy_music',
                            {'on_interrupt_music': {'action': 'stop_music'}})
        self.assertEqual(self.hooks.notify_resources('on_interrupt_music'), MUSIC)

    def test_an_unbound_hook_has_an_empty_scope(self):
        self.assertEqual(self.hooks.notify_resources('on_interrupt_music'), frozenset())

    def test_interrupt_bindings_are_still_barrier_exempt(self):
        """打断本身不能被 barrier 挡住 —— 前缀匹配 on_interrupt 就豁免。"""
        self.hooks.register('mcp-music', 'phanthy_music',
                            {'on_interrupt_music': {'action': 'stop_music'}})
        self.assertTrue(self.hooks.is_interrupt_binding(
            'mcp-music', 'phanthy_music', 'stop_music'))
        self.assertFalse(self.hooks.is_interrupt_binding(
            'mcp-music', 'phanthy_music', 'play_music'))


if __name__ == '__main__':
    unittest.main()
