"""
test_peer_audit.py — 配对关系的变更要留下查得到的记录。

背景：有人在 2026-09-11 于天轶的面板上解除了和 Orin5 的配对。一周后要查是谁、
什么时候干的 —— 查不到。`peers` 表只回答「现在是什么」，那行删掉之后这段关系
不留任何痕迹；`docker logs` 只回溯到 9/17（容器重启冲掉了）；活动流在内存里。
于是一个再正常不过的问题根本没有答案。

`peer_audit` 回答「发生过什么」，只增不改。覆盖：

  - 四类事件都记（配对、本机解除、被对方解除、角色变更）
  - 审计写失败不能拖垮配对/解除本身 —— 一条查不到的记录远好过一次做不成的操作
  - 有条数上限，且裁剪按 id 而不是按 ts（机器上的 wall clock 会跳变）
  - 迁移对已有库是非破坏的

Run: cd agent-core && python3 -m pytest tests/test_peer_audit.py
"""

import os
import pathlib
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from peer import store  # noqa: E402

PEER = 'a' * 32


class _Base(unittest.TestCase):
    def setUp(self):
        import config
        self._db = os.path.join(tempfile.mkdtemp(), 'audit.db')
        p = mock.patch.object(config, 'DB_PATH', self._db)
        p.start()
        self.addCleanup(p.stop)


class TestRecording(_Base):
    def test_each_event_kind_is_stored_and_readable(self):
        store.record_audit(PEER, store.EVENT_PAIRED, display_name='Orin5',
                           actor='local', detail='role=viewer')
        store.record_audit(PEER, store.EVENT_UNPAIRED_LOCAL, display_name='Orin5',
                           actor='local', detail='peer notified')
        store.record_audit(PEER, store.EVENT_UNPAIRED_BY_PEER, display_name='Orin5',
                           actor=PEER)
        store.record_audit(PEER, store.EVENT_ROLE_CHANGED, display_name='Orin5',
                           actor='local', detail='viewer → operator')

        events = store.list_audit()
        self.assertEqual([e['event'] for e in events], [
            store.EVENT_ROLE_CHANGED,      # newest first
            store.EVENT_UNPAIRED_BY_PEER,
            store.EVENT_UNPAIRED_LOCAL,
            store.EVENT_PAIRED,
        ])

    def test_it_records_who_did_it(self):
        """这正是当初查不出来的那一项：本机操作员，还是对方。"""
        store.record_audit(PEER, store.EVENT_UNPAIRED_LOCAL, actor='local')
        store.record_audit('b' * 32, store.EVENT_UNPAIRED_BY_PEER, actor='b' * 32)
        by_actor = {e['event']: e['actor'] for e in store.list_audit()}
        self.assertEqual(by_actor[store.EVENT_UNPAIRED_LOCAL], 'local')
        self.assertEqual(by_actor[store.EVENT_UNPAIRED_BY_PEER], 'b' * 32)

    def test_it_records_when(self):
        store.record_audit(PEER, store.EVENT_PAIRED)
        self.assertGreater(store.list_audit()[0]['ts'], 1_600_000_000)

    def test_filtering_by_peer(self):
        store.record_audit(PEER, store.EVENT_PAIRED)
        store.record_audit('b' * 32, store.EVENT_PAIRED)
        only = store.list_audit(peer_id=PEER)
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0]['peer_id'], PEER)

    def test_long_detail_is_truncated_not_rejected(self):
        """detail 里会塞进 transport 的失败原因，那串可以很长。"""
        store.record_audit(PEER, store.EVENT_UNPAIRED_LOCAL, detail='x' * 5000)
        self.assertLessEqual(len(store.list_audit()[0]['detail']), 500)


class TestItNeverBreaksTheOperation(_Base):
    def test_a_failing_audit_write_does_not_raise(self):
        """审计是诊断。一条查不到的记录，远好过一次做不成的解除配对。"""
        with mock.patch.object(store, '_conn', side_effect=sqlite3.OperationalError('disk I/O error')):
            store.record_audit(PEER, store.EVENT_UNPAIRED_LOCAL)   # must not raise


class TestRotation(_Base):
    def test_the_table_is_bounded(self):
        for i in range(store.AUDIT_MAX_ROWS + 40):
            store.record_audit(PEER, store.EVENT_PAIRED, detail=str(i))
        with sqlite3.connect(self._db) as conn:
            n = conn.execute('SELECT COUNT(*) FROM peer_audit').fetchone()[0]
        self.assertLessEqual(n, store.AUDIT_MAX_ROWS)

    def test_rotation_keeps_the_newest(self):
        for i in range(store.AUDIT_MAX_ROWS + 10):
            store.record_audit(PEER, store.EVENT_PAIRED, detail=str(i))
        newest = store.list_audit(limit=1)[0]
        self.assertEqual(newest['detail'], str(store.AUDIT_MAX_ROWS + 9))

    def test_rotation_survives_a_clock_that_jumps_backwards(self):
        """裁剪按 id，不按 ts。

        机器上的 wall clock 真的会跳：G1 上两个容器的 StartedAt 是 1970-06-05，
        而宿主机日期是对的。按 ts 排序裁剪，会把时钟正确时写的新记录当成旧的删掉。
        """
        real_time = store.time.time

        def _weird_clock():
            # 先给一串「未来」的时间，再跳回 1970
            return 0.0 if _weird_clock.n % 2 else real_time() + 86400 * _weird_clock.n
        _weird_clock.n = 0

        for i in range(store.AUDIT_MAX_ROWS + 5):
            _weird_clock.n = i
            with mock.patch.object(store.time, 'time', _weird_clock):
                store.record_audit(PEER, store.EVENT_PAIRED, detail=str(i))

        # 最后写进去的那条必须还在，无论它的 ts 是 1970 还是 2030。
        self.assertEqual(store.list_audit(limit=1)[0]['detail'],
                         str(store.AUDIT_MAX_ROWS + 4))


class TestMigration(_Base):
    def test_an_existing_database_is_not_discarded(self):
        """CREATE TABLE IF NOT EXISTS —— 升级不能清掉已有数据。"""
        store.record_audit(PEER, store.EVENT_PAIRED, detail='before upgrade')
        import config
        config._get_conn().close()          # run the schema path again
        rows = store.list_audit()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['detail'], 'before upgrade')

    def test_peers_table_still_works_alongside_it(self):
        from peer import identity
        identity.reset_cache()
        identity.ensure_identity()
        store.upsert(PEER, identity.public_key_b64(), 'Orin5', role='viewer')
        self.assertIsNotNone(store.get(PEER))


if __name__ == '__main__':
    unittest.main()
