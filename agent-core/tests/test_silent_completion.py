"""
test_silent_completion.py — 「播完了」是回执，不是新消息，而这件事该由驱动申报。

原先 collector 靠 action_id 的拼写判断：`speak-` / `tts-` 开头的完成事件 P=0（不进
LLM），其余 P=1（唤醒一个 turn）。于是一张把 id 叫 `music-…` 的播放卡每放完一首歌就
唤醒一次 turn —— 而它和 `speak-` 的区别只是驱动作者选了别的前缀。

改成 `x-completion.silent_completion`：/api/acp/complete 与 SSE 两条路都把它写成事件
里的 `silent` 字段，collector 认这个字段。前缀作为**遗留兜底**保留给仓库外的驱动。
"""
import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import collector  # noqa: E402
import mcp_client  # noqa: E402

MOUTH = frozenset({'mouth'})
MUSIC = frozenset({'music'})


def _event(**kw):
    import json
    data = {'type': 'action_complete', 'status': 'completed', **kw}
    return {'source': f'acp:{data.get("action_id", "x")}', 'text': json.dumps(data),
            'payload': dict(data)}


class TestDeclaredFlagDecides(unittest.TestCase):
    def test_a_flagged_completion_does_not_steer(self):
        self.assertEqual(
            collector._extract_priority(_event(action_id='music-7f3a', silent=True)), 0)

    def test_an_unflagged_music_completion_still_steers(self):
        """证明起作用的是申报而不是前缀：同一个 id，只差 silent。"""
        self.assertEqual(
            collector._extract_priority(_event(action_id='music-7f3a')), 1)

    def test_the_legacy_prefixes_still_work(self):
        """仓库外的驱动没升级，删掉前缀会让它们突然开始刷 turn。"""
        for aid in ('speak-abc', 'tts-abc'):
            with self.subTest(aid=aid):
                self.assertEqual(collector._extract_priority(_event(action_id=aid)), 0)

    def test_an_explicit_priority_still_wins(self):
        ev = _event(action_id='music-1', silent=True)
        import json
        ev['text'] = json.dumps({'type': 'action_complete', 'action_id': 'music-1',
                                 'silent': True, 'priority': 2})
        self.assertEqual(collector._extract_priority(ev), 2)

    def test_the_payload_path_honours_the_flag_too(self):
        """两条路都要认：text 不是 JSON 时 collector 退到 payload。"""
        ev = _event(action_id='music-1', silent=True)
        ev['text'] = 'not json'
        self.assertEqual(collector._extract_priority(ev), 0)
        ev2 = _event(action_id='music-1')
        ev2['text'] = 'not json'
        self.assertEqual(collector._extract_priority(ev2), 1)


class TestCompletionIsSilent(unittest.TestCase):
    def setUp(self):
        self._saved = tuple(dict(d) for d in mcp_client.pending_tables())
        for d in mcp_client.pending_tables():
            d.clear()

    def tearDown(self):
        for d, saved in zip(mcp_client.pending_tables(), self._saved):
            d.clear()
            d.update(saved)

    def test_the_flag_comes_from_x_completion(self):
        mcp_client._register_pending(
            'music-1', tool_name='phanthy_music',
            meta={'completion': {'silent_completion': True}, 'resource': MUSIC},
            args={}, result={})
        self.assertTrue(mcp_client.completion_is_silent('music-1'))

    def test_an_undeclared_action_is_not_silent(self):
        mcp_client._register_pending(
            'nav-1', tool_name='navigate',
            meta={'completion': {}, 'resource': frozenset({'base'})},
            args={}, result={})
        self.assertFalse(mcp_client.completion_is_silent('nav-1'))

    def test_a_reaped_action_reads_as_not_silent(self):
        """这正是必须在 mark_action_complete **之前**问的原因。

        一个并发的 barrier 随时可能把这一项回收掉；回收之后就查不到了，于是回执会
        按「不静默」处理并唤醒一个 turn。端点因此先取标志再标记完成。
        """
        mcp_client._register_pending(
            'music-1', tool_name='phanthy_music',
            meta={'completion': {'silent_completion': True}, 'resource': MUSIC},
            args={}, result={})
        self.assertTrue(mcp_client.completion_is_silent('music-1'))
        mcp_client._forget_pending(['music-1'])
        self.assertFalse(mcp_client.completion_is_silent('music-1'))

    def test_the_in_tree_tts_card_declares_it(self):
        """TTS 不该再依赖自己的 action_id 拼成 speak-*。"""
        tts_py = (pathlib.Path(__file__).resolve().parents[2]
                  / 'perception' / 'plugins' / 'tts.py').read_text()
        self.assertIn('silent_completion', tts_py)


if __name__ == '__main__':
    unittest.main()
