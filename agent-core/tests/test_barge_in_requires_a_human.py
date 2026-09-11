"""test_barge_in_requires_a_human.py — 只有人能把机器人的话打断。

Orin5 实测，用户全程没有做任何打断：

    17:01:48  [acp] registered pending: speak-42c1c65c (tts, timeout=171s)   # 英伟达汇报
    17:02:29  [acp] barrier barge_in before finish: ['speak-42c1c65c']       # 41 秒处被掐
    17:02:29  [decision] received [URGENT] event: source=subagent:7ff3018a   # 坝上 subagent 跑完

第二个后台 subagent 完成，`manager._finalize` 把完成通知塞进 steering 队列，finish 的
ACP barrier 醒来并调 `_abort_pending_for_barge_in()` —— 正在播的英伟达汇报被自己派出去
的后台任务冲掉，而冲掉它的那条通知随后还要再播一遍。

`has_steering()` 对所有 P>0 事件都为真，而这个 wakeup 的唯一后果是掐音频。两者语义不
同，所以拆成 `has_barge_in_steering()`。
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import asyncio  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import collector  # noqa: E402


def _ev(source: str, text: str = 'x') -> dict:
    return {'source': source, 'text': text, 'payload': {}}


class TestBargeInSourceFilter(unittest.TestCase):
    def setUp(self):
        while not collector._steering_queue.empty():
            collector._steering_queue.get_nowait()
        collector._priority_pending.clear()

    tearDown = setUp

    def _queue(self, *sources):
        for s in sources:
            collector._steering_queue.put_nowait(_ev(s))

    # ── 机器自己的通知：进队列，但不掐音频 ────────────────────────────────

    def test_subagent_completion_does_not_barge_in(self):
        """这条就是 Orin5 上掐掉英伟达汇报的那一条。"""
        self._queue('subagent:7ff3018a')
        self.assertTrue(collector.has_steering(), '通知仍然要让主循环尽快看到')
        self.assertFalse(collector.has_barge_in_steering(),
                         '后台任务跑完不该掐掉正在播的汇报')

    def test_other_machine_sources_do_not_barge_in(self):
        for src in ('scheduler:task-1', 'acp', 'peer:dd398c73177a', 'bg_monitor'):
            with self.subTest(src=src):
                self.setUp()
                self._queue(src)
                self.assertFalse(collector.has_barge_in_steering(), src)

    # ── 人：照常能打断 ───────────────────────────────────────────────────

    def test_speech_barges_in(self):
        self._queue('dds:/robot/mic/audio/asr_event')
        self.assertTrue(collector.has_barge_in_steering())

    def test_typed_input_barges_in(self):
        self._queue('dds:/remote_control/message')
        self.assertTrue(collector.has_barge_in_steering())

    def test_channel_message_barges_in(self):
        self._queue('channel:feishu')
        self.assertTrue(collector.has_barge_in_steering())

    def test_unknown_source_barges_in(self):
        """黑名单而非白名单：新渠道忘了登记，最坏是它照常能打断（原行为）。

        反过来（白名单漏登记）就是有人喊停而机器人继续念 —— 不能接受。
        """
        self._queue('some-brand-new-channel')
        self.assertTrue(collector.has_barge_in_steering())

    # ── 混在一起 ─────────────────────────────────────────────────────────

    def test_a_human_behind_machine_notifications_still_barges_in(self):
        """人的消息排在几条通知后面，不能被前面的通知盖住。"""
        self._queue('subagent:a', 'scheduler:b', 'dds:/robot/mic/audio/asr_event')
        self.assertTrue(collector.has_barge_in_steering())

    def test_priority_pending_is_filtered_too(self):
        collector._priority_pending.append(_ev('subagent:7ff3018a'))
        self.assertTrue(collector.has_steering())
        self.assertFalse(collector.has_barge_in_steering())
        collector._priority_pending.append(_ev('dds:/remote_control/message'))
        self.assertTrue(collector.has_barge_in_steering())

    def test_empty_queue_is_not_a_barge_in(self):
        self.assertFalse(collector.has_barge_in_steering())


class TestWaitForSteeringUsesTheFilter(unittest.TestCase):
    """barrier 的 wakeup 必须走过滤后的判据，否则上面那些都白测。"""

    def setUp(self):
        while not collector._steering_queue.empty():
            collector._steering_queue.get_nowait()
        collector._priority_pending.clear()

    tearDown = setUp

    def test_machine_notification_does_not_wake_the_wait(self):
        from event.llm import _wait_for_steering

        async def _scenario():
            collector._steering_queue.put_nowait(_ev('subagent:7ff3018a'))
            try:
                await asyncio.wait_for(_wait_for_steering(poll_s=0.01), timeout=0.3)
                return 'woke'
            except asyncio.TimeoutError:
                return 'kept waiting'

        self.assertEqual(asyncio.run(_scenario()), 'kept waiting')

    def test_human_message_wakes_the_wait(self):
        from event.llm import _wait_for_steering

        async def _scenario():
            collector._steering_queue.put_nowait(_ev('dds:/remote_control/message'))
            await asyncio.wait_for(_wait_for_steering(poll_s=0.01), timeout=2)
            return 'woke'

        self.assertEqual(asyncio.run(_scenario()), 'woke')

    def test_the_notification_is_not_consumed(self):
        """过滤掉不等于丢掉 —— 它要留在队列里，播完之后转成下一轮的触发事件。"""
        collector._steering_queue.put_nowait(_ev('subagent:7ff3018a'))
        collector.has_barge_in_steering()
        self.assertEqual(collector._steering_queue.qsize(), 1)


if __name__ == '__main__':
    unittest.main()
