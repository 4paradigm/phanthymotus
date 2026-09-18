"""test_finish_deferred_by_new_message.py — 播报期间来了新消息就立刻重新推理。

Orin5 实测，用户在一段 48 秒播报的早期发出下一个问题：

    17:32:02.705  registered pending: speak-67462a1e (tts, timeout=98s)   ← 播报开始
    17:32:04.444  barrier: waiting for ['speak-67462a1e']                 ← 主循环停在这
       （用户发消息，事件自带 ts=17:32:19）
    17:32:52.540  barrier cleared: ['speak-67462a1e']                     ← 播完
    17:32:52.597  received [URGENT] ... ts="2026-09-11T17:32:19"          ← 躺了 33.6 秒
    17:32:52.622  llm request: round=0                                    ← 到这才开始想

`finish` 的 break 在 steering drain 前面，所以 turn 期间没有任何东西会消费队列 ——
播报有多长，延迟就有多长。

修法不是恢复打断（音频必须播完），而是让 finish 的 barrier 和 mcp__ 工具一样吃
`reconsider_event`：等待作废、**pending 不清**（音频继续播）、turn 不结束、带着新消息
再想一遍。barrier 那半在 test_reconsider_barrier.py 里测；这里测"turn 不结束"这半。
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from event.llm import _turn_ends_on_finish  # noqa: E402


def _calls(*names) -> list[dict]:
    return [{'function': {'name': n, 'arguments': '{}'}} for n in names]


class TestTurnEndsOnFinish(unittest.TestCase):
    # ── 正常路径不能被改坏 ────────────────────────────────────────────────

    def test_finish_ends_the_turn(self):
        self.assertTrue(_turn_ends_on_finish(_calls('finish'), 'finish', False))

    def test_finish_alongside_other_calls_still_ends_the_turn(self):
        self.assertTrue(
            _turn_ends_on_finish(_calls('mcp__m1__tts', 'finish'), 'finish', False))

    def test_no_finish_does_not_end_the_turn(self):
        self.assertFalse(
            _turn_ends_on_finish(_calls('mcp__m1__tts'), 'finish', False))

    def test_empty_round_does_not_end_the_turn(self):
        self.assertFalse(_turn_ends_on_finish([], 'finish', False))

    # ── 作废掉的 finish 不能结束 turn ────────────────────────────────────

    def test_deferred_finish_does_not_end_the_turn(self):
        """这条就是 33.6 秒延迟的修复点。"""
        self.assertFalse(_turn_ends_on_finish(_calls('finish'), 'finish', True))

    def test_deferred_finish_with_a_dispatched_tts_in_the_same_round(self):
        """`tts(...) + finish()` 同轮 —— Orin5 日志里的真实形状。

        这一轮 results 非空，`_round_voided` 不成立（它只在本轮还没有任何工具派出去时
        才作废整轮），所以只能靠这个谓词拦住 break。少了它，延迟一点没少。
        """
        self.assertFalse(
            _turn_ends_on_finish(_calls('mcp__m1__tts', 'finish'), 'finish', True))

    def test_deferred_flag_does_not_invent_a_finish(self):
        """没调 finish 的一轮，deferred 也不该让它看起来像调了。"""
        self.assertFalse(
            _turn_ends_on_finish(_calls('mcp__m1__tts'), 'finish', True))


# 没有覆盖到的一段，写在这里而不是假装测了：
#
# 作废 finish 之后 `reconsider_event.clear()`（llm.py，steering drain 之后）没有单测。
# 它在 `_one_turn` 的循环体内联，要测就得把 prompt 构建 / client / mcp registry / DB 全
# 立起来。漏掉它的后果是可观察的，靠 Orin5 实机看：下一次 `llm request` 之后会紧跟一条
# RoundReconsider 重试（日志里的 "round voided by reconsider" 或 "llm call reconsidered"），
# 白烧一次请求 —— 而不是静默错误。


if __name__ == '__main__':
    unittest.main()
