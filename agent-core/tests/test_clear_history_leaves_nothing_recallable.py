"""清除历史之后，智能体不能还"记得"被删掉的对话。

报上来的不是某处多了一行，而是「感觉清除不干净，之前对话的记忆还在，能从回复
听出来」—— 这种症状没有可看的现场，只能从「谁还存着同一份记忆」反推。

`clear_all()` 原来只删三张 chat 表。但 `subagent_conclusions` 存的是同一份记忆
的上一层抽象：后台监控对刚说过的话的小结、每日摘要，由 `subagent/agent.py` 和
`subagent/manager.py` 写入，再由 `memory_recall`（`event/llm.py`，
`source in ('all','subagent')`）按内容 LIKE 搜回来。清了对话不清结论，
`memory_recall` 照样命中。

Orin5 2026-10-17 实测：`chat_messages` 最老一行是 17:13（刚清过），而
`subagent_conclusions` 还有 1048 行、最老回溯到 09-04 —— 六周的、关于已删对话的
结论。

Run: cd agent-core && python3 -m pytest tests/test_clear_history_leaves_nothing_recallable.py
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
from config import _get_conn  # noqa: E402

SECRET = '用户说他家猫叫毛豆'


def _add_conclusion(source_type='bg_monitor', text=SECRET):
    with _get_conn() as conn:
        conn.execute(
            'INSERT INTO subagent_conclusions (agent_id, goal, conclusion, source_type, created_at) '
            'VALUES (?, ?, ?, ?, ?)',
            ('agent-test', '后台监控', text, source_type, time.time()))
        conn.commit()


def _count(table):
    with _get_conn() as conn:
        return conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]


class ClearHistoryTest(unittest.TestCase):
    def setUp(self):
        chat_history.clear_all()

    def test_clearing_history_also_clears_the_conclusions_drawn_from_it(self):
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': SECRET}])
        _add_conclusion()
        self.assertEqual(_count('chat_messages'), 1)
        self.assertEqual(_count('subagent_conclusions'), 1)

        chat_history.clear_all()

        self.assertEqual(_count('chat_messages'), 0)
        self.assertEqual(_count('chat_sessions'), 0)
        self.assertEqual(
            _count('subagent_conclusions'), 0,
            'memory_recall 会按内容搜这张表，留着它等于留着被删对话的记忆')

    def test_a_daily_summary_is_not_exempt(self):
        """每日摘要也是对话的衍生记忆，不是独立资料。"""
        _add_conclusion(source_type='daily_summary', text=SECRET + '（日报）')
        chat_history.clear_all()
        self.assertEqual(_count('subagent_conclusions'), 0)

    def test_recall_finds_nothing_after_a_clear(self):
        """从检索那一端断言，而不是只看表计数 —— 用户听到的是回复，不是行数。

        这里复刻 `memory_recall` 的 subagent 分支（`event/llm.py`）用的那条
        `conclusion LIKE ?`，因为过表计数证明不了"搜不到"。

        chat 表那一侧不在这里断言：它走 FTS5 而 tokenize='unicode61' 不切中文，
        一句中文整体是一个 token，所以中文子串本来就搜不出来 —— 拿它当探针会把一
        个既有限制混进这次的回归里。那一侧由上面的行数断言覆盖。
        """
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': SECRET}])
        _add_conclusion()

        def recalled():
            with _get_conn() as conn:
                return conn.execute(
                    'SELECT conclusion FROM subagent_conclusions '
                    'WHERE conclusion LIKE ? OR goal LIKE ?',
                    (f'%{SECRET}%', f'%{SECRET}%')).fetchall()

        self.assertTrue(recalled(), 'sanity: 清之前必须搜得到，否则空断言没有意义')
        chat_history.clear_all()
        self.assertEqual(recalled(), [], 'memory_recall 的 subagent 来源仍能命中')

    def test_deleting_one_session_does_not_pretend_to_clear_conclusions(self):
        """按会话删做不到，因为结论表没有 session_id —— 钉住这个已知缺口。

        按时间戳猜会删掉别的会话的结论，所以这里不是待修的 bug，而是
        `clear_all` 必须存在的理由。
        """
        sid = chat_history.create_session()
        chat_history.save_turn(sid, 0, [{'role': 'user', 'content': SECRET}])
        _add_conclusion()

        chat_history.delete_session(sid)

        self.assertEqual(_count('chat_messages'), 0)
        self.assertEqual(_count('subagent_conclusions'), 1)
        with _get_conn() as conn:
            cols = [c[1] for c in conn.execute('PRAGMA table_info(subagent_conclusions)')]
        self.assertNotIn('session_id', cols,
                         '一旦加了 session_id，按会话删就该一起删结论，这条断言会提醒')


if __name__ == '__main__':
    unittest.main()
