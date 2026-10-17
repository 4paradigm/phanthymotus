"""「清除历史」清不掉任务 —— 任务是独立一项，要独立勾选。

任务**故意**比一次对话活得长：`load_all()` 启动时恢复所有活跃任务，`__aenter__`
再把带 cron 的重新注册成定时检查。所以只清历史对话，智能体还在执行一份来自已删
对话的议程 —— 它会继续汇报一个用户已经看不到出处的目标的进展。

两个容易写错的地方，这个文件两头都钉住：

1. **不能用 `done()` 实现。** `done()` 会 `_persist(status='done')` 再从内存删，
   所以 DB 行还在，要等**下一次**启动 `load_all()` 才删。对一个真的干完了的任务
   这是对的（记录比运行活得长），对重置是错的 —— 任务的 `goal`/`progress` 是模型
   或用户写的文本，留着就是和「清了对话却留着 subagent_conclusions」同一类残留。

2. **定时检查必须先取消。** `task_store` 不 import `scheduler`，`task:<id>` 这个
   job 名是 `event/task.py` 的约定。漏了这一步，cron 会继续每隔几分钟发
   「任务定时检查 [id]」把智能体叫醒，而那个任务已经查不到了。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_reset_clears_tasks.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import scheduler  # noqa: E402
import task_store  # noqa: E402
import event  # noqa: E402
from api.config import ResetRequest, reset_config  # noqa: E402

task_mod = sys.modules['event.task']


def _db_rows():
    with task_store._get_conn() as conn:
        return conn.execute('SELECT id, data FROM tasks').fetchall()


class ResetTasksTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(task_store._tasks)
        task_store.clear_all()
        for name in [n for n in list(scheduler._dynamic_tasks) if n.startswith('task:')]:
            scheduler.remove_job(name)

    def tearDown(self):
        task_store.clear_all()
        task_store._tasks.update(self._saved)

    def test_clear_all_removes_the_rows_rather_than_marking_them_done(self):
        a = task_store.create(goal='做一份投研报告')
        task_store.create(goal='盯着充电状态', check_cron='*/5 * * * *')
        self.assertEqual(len(_db_rows()), 2)

        self.assertEqual(task_store.clear_all(), 2)

        self.assertEqual(task_store.active_tasks(), [])
        self.assertEqual(_db_rows(), [],
                         'done() 会留下 status=done 的行，重置不能留 —— goal 是写进去的文本')
        self.assertIsNone(task_store.get(a.id))

    def test_a_finished_tasks_leftover_row_is_taken_too(self):
        """DB 里可能有没对应内存条目的 done/failed 行，重置也该带走。"""
        t = task_store.create(goal='已经干完的事')
        task_store.done(t.id, summary='完成')
        self.assertEqual(task_store.active_tasks(), [], 'done 之后内存里已经没有了')
        self.assertEqual(len(_db_rows()), 1, '但 DB 行还在 —— 这正是要清的东西')

        task_store.clear_all()

        self.assertEqual(_db_rows(), [])

    def test_reset_cancels_the_scheduled_checks(self):
        task_store.create(goal='盯着充电状态', check_cron='*/5 * * * *')
        task_store.create(goal='没有 cron 的任务')

        async def register_then_reset():
            # `add_job` 用 asyncio.ensure_future，所以注册必须在运行中的循环里 ——
            # 和真实调用点一样（`__aenter__` 里的 _register_check）。
            for t in task_store.active_tasks():
                task_mod._register_check(t)
            jobs = [n for n in scheduler._dynamic_tasks if n.startswith('task:')]
            assert len(jobs) == 1, f'只有带 cron 的那个会注册 job，实际 {jobs}'
            await reset_config(ResetRequest(tasks=True))
            return [n for n in scheduler._dynamic_tasks if n.startswith('task:')]

        remaining = asyncio.run(register_then_reset())

        self.assertEqual(task_store.active_tasks(), [])
        self.assertEqual(_db_rows(), [])
        self.assertEqual(
            remaining, [],
            '留下的 job 会继续把智能体叫醒去查一个已经不存在的任务')

    def test_reset_reports_tasks_and_leaves_other_things_alone(self):
        task_store.create(goal='只清任务，别动别的')
        out = asyncio.run(reset_config(ResetRequest(tasks=True)))
        self.assertIn('tasks', out['reset'])
        self.assertNotIn('chat_history', out['reset'])

    def test_clearing_history_alone_does_not_touch_tasks(self):
        """这正是报上来的问题：清了历史，议程还在。

        两件事是两个勾选项，所以也必须是两条互不牵连的路径 —— 把任务捎带进
        chat_history 分支会让「只想清对话、保留在跑的任务」变成做不到。
        """
        t = task_store.create(goal='跨过历史清除继续活着')
        asyncio.run(reset_config(ResetRequest(chat_history=True)))
        self.assertIsNotNone(task_store.get(t.id))
        self.assertEqual(len(_db_rows()), 1)


if __name__ == '__main__':
    unittest.main()
