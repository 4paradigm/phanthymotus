"""历史记录要在 turn 跑的过程中就可见，而不是等它结束。

以前 `save_turn` 只在 turn 结束时调用一次并且是纯 INSERT，一个跑了几分钟的 turn
在这期间在库里完全不存在 —— 历史 modal 手动刷新也刷不出来。现在每轮都写一次、
按 (session_id, turn_index) 覆盖，这几条断言钉住的就是"重复写同一轮不会变成多轮"
以及"分区/排序所依赖的 kind 和最后活动时间是对的"。
"""

import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import chat_history  # noqa: E402


class SaveTurnUpsertTest(unittest.TestCase):
    def setUp(self):
        chat_history.clear_all()

    def test_rewriting_a_turn_updates_it_in_place(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'}])
        chat_history.save_turn(sid, 0, [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ])

        turns = chat_history.get_session_turns(sid)
        self.assertEqual(len(turns), 1, 'a rewritten turn must not become a second turn')
        self.assertEqual(len(turns[0]['messages']), 2)

        sessions, _ = chat_history.list_sessions()
        self.assertEqual([s['turn_count'] for s in sessions if s['id'] == sid], [1])

    def test_turn_timestamps_are_exposed(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'}])
        started = chat_history.get_session_turns(sid)[0]['started_at']
        time.sleep(0.01)
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'},
                                        {'role': 'assistant', 'content': 'yo'}])
        turn = chat_history.get_session_turns(sid)[0]
        self.assertEqual(turn['started_at'], started, 'start time must not move')
        self.assertGreater(turn['updated_at'], started)

    def test_get_session_messages_still_returns_bare_turns(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'hi'}])
        self.assertEqual(chat_history.get_session_messages(sid),
                         [[{'role': 'user', 'content': 'hi'}]])


class SessionOrderingTest(unittest.TestCase):
    def setUp(self):
        chat_history.clear_all()

    def test_kind_round_trips(self):
        main = chat_history.create_session(chat_history.KIND_MAIN)
        bg = chat_history.create_session(chat_history.KIND_BG_SUBAGENT)
        for sid in (main, bg):
            chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'x'}])
        kinds = {s['id']: s['kind'] for s in chat_history.list_sessions()[0]}
        self.assertEqual(kinds[main], chat_history.KIND_MAIN)
        self.assertEqual(kinds[bg], chat_history.KIND_BG_SUBAGENT)

    def test_ordered_by_last_activity_not_by_start(self):
        old = chat_history.create_session()
        chat_history.save_turn(old, 0, [{'role': 'user', 'content': 'first'}])
        time.sleep(0.01)
        new = chat_history.create_session()
        chat_history.save_turn(new, 0, [{'role': 'user', 'content': 'second'}])
        time.sleep(0.01)
        # 老会话又说话了 —— 它应该回到最前面。
        chat_history.save_turn(old, 1, [{'role': 'user', 'content': 'third'}])

        sessions, _ = chat_history.list_sessions()
        self.assertEqual([s['id'] for s in sessions][:2], [old, new])
        self.assertGreaterEqual(sessions[0]['last_at'], sessions[1]['last_at'])


class ResumePicksMainSessionTest(unittest.TestCase):
    """重启续跑只能续主代理自己的会话。

    子代理的对话和主代理的存在同一张表里。子代理改成每轮落盘之后，"最近的一个
    session" 在任何时刻都极可能是某个后台子代理的 —— 续跑到那上面，主代理会把
    一段后台监控的 transcript 当成自己的历史接着写。Orin5 上实测到过：一个
    [daily] 子代理的会话 turn_count 从 4 涨到 5，涨的那一轮是主代理写的。
    """

    def setUp(self):
        chat_history.clear_all()

    def test_subagent_session_is_not_resumed(self):
        main = chat_history.create_session(chat_history.KIND_MAIN)
        chat_history.update_summary(main, '用户：把灯打开')
        chat_history.save_turn(main, 0, [{'role': 'user', 'content': '把灯打开'}])
        time.sleep(0.01)
        bg = chat_history.create_session(chat_history.KIND_BG_SUBAGENT)
        chat_history.update_summary(bg, '[subagent:abc123] [bg] 后台监控')
        chat_history.save_turn(bg, 0, [{'role': 'user', 'content': '监控中'}])

        self.assertEqual(chat_history.get_last_session_turns()['session_id'], main)

    def test_legacy_subagent_rows_are_excluded_too(self):
        # 迁移给老行统一填了 kind='main'，只能靠 summary 前缀排除。
        legacy = chat_history.create_session(chat_history.KIND_MAIN)
        chat_history.update_summary(legacy, '[subagent:old999] 旧的子代理会话')
        chat_history.save_turn(legacy, 0, [{'role': 'user', 'content': 'x'}])
        self.assertIsNone(chat_history.get_last_session_turns())


class SummaryTextTest(unittest.TestCase):
    """会话列表的标题要是人说的那句话，不是事件信封。

    trigger 文本常常是 `<event source=… channel=… ts=…>\\n{"text": "早上好。"}`，
    而 summary 只存前 100 字 —— 整行都是属性，正文被截在外面，列表里看不出这是
    哪一次对话。
    """

    def test_event_envelope_is_unwrapped(self):
        raw = ('<event source="dds:/remote_control/message" channel="remote_web" '
               'ts="2026-09-14T11:05:38">\n'
               '{"text": "早上好，帮我看下电量", "audio_duration_ms": 1676}\n</event>')
        self.assertEqual(chat_history.summary_text(raw), '早上好，帮我看下电量')

    def test_plain_text_is_untouched(self):
        self.assertEqual(chat_history.summary_text('普通文本触发'), '普通文本触发')

    def test_non_json_body_survives(self):
        self.assertEqual(
            chat_history.summary_text('<event source="x">\n不是 JSON 的正文\n</event>'),
            '不是 JSON 的正文')

    def test_empty_body_falls_back_to_the_original(self):
        raw = '<event source="x"></event>'
        self.assertEqual(chat_history.summary_text(raw), raw)

    def test_stored_summary_is_the_unwrapped_text(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'x'}])
        chat_history.update_summary(
            sid, '<event source="dds:/a" ts="t">\n{"text": "把灯打开"}\n</event>')
        summaries = {s['id']: s['summary'] for s in chat_history.list_sessions()[0]}
        self.assertEqual(summaries[sid], '把灯打开')


if __name__ == '__main__':
    unittest.main()
