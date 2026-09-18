"""
test_task_create_no_cron.py — 模型不能再自己建定时检查。

Tianyi 实测：模型每次派子代理都顺手 task_create(check_cron="*/2 * * * *")，于是每 2 分钟
被叫醒一次去 subagent_status + task_update，一次两轮、每轮三万多 token（实测 prompt=35307）。
一小时触发了 12 次。

查下来整份 prompt 里**没有任何一句话**让它这么做 —— `prompt_system.md` 只说"用 task_create
追踪""收到定时检查事件时才 task_update"，从没说过要建 cron。唯一的驱动是 check_cron 这个
参数描述本身：给了现成的 "*/2 * * * *" 可抄，又把"留空"写成"则不自动检查"（听起来像放弃了
什么）。所以修法是把参数拿掉，而不是再加一句 prompt 去劝它别用。

它想解决的三件事现在都有人管：
  · 子代理干完   → manager._notify_completion 推 URGENT（实测 160ms 内到）
  · 子代理卡死   → manager._timeout_watchdog
  · 用户不知进展 → 框架按沉默时长自动播报（event/llm.py）

**但定时检查这个能力本身要留着**：设置页改任务、方案包声明的任务仍然能带 cron，那些是人
明确要的。这个文件两头都锚住。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_task_create_no_cron.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile
import typing
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import scheduler  # noqa: E402
import task_store  # noqa: E402
import event  # noqa: E402
from event.llm import _build_system_tools  # noqa: E402

task_mod = sys.modules['event.task']
_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


class TestLLMCannotCreateCron(unittest.TestCase):
    def setUp(self):
        self._saved = dict(task_store._tasks)
        task_store._tasks.clear()

    def tearDown(self):
        task_store._tasks.clear()
        task_store._tasks.update(self._saved)

    def test_tool_schema_has_no_cron_param(self):
        """模型看到的 task_create 只剩 goal。"""
        d = _build_system_tools([('task_create', task_mod.Tools().task_create)])
        props = d['task_create']['schema']['parameters']['properties']
        self.assertEqual(list(props), ['goal'])
        self.assertNotIn('check_cron', props)

    def test_the_copyable_example_is_gone(self):
        """那个 "*/2 * * * *" 是模型照抄的来源，不能再出现在它读得到的地方。"""
        d = _build_system_tools([('task_create', task_mod.Tools().task_create)])
        schema = d['task_create']['schema']
        blob = repr(schema)
        self.assertNotIn('*/2 * * * *', blob)
        self.assertNotIn('check_cron', blob)

    def test_creating_a_task_registers_no_job(self):
        before = set(scheduler._dynamic_tasks)
        asyncio.run(task_mod.Tools().task_create(goal='做一份投研报告'))
        self.assertEqual(set(scheduler._dynamic_tasks), before,
                         'task_create 不该再注册任何定时 job')

    def test_task_is_still_tracked(self):
        """只去掉轮询，追踪本身要保住 —— 它是模型跨 turn 记得"还有事没完"的唯一凭据。"""
        asyncio.run(task_mod.Tools().task_create(goal='做一份投研报告'))
        active = task_store.active_tasks()
        self.assertEqual([t.goal for t in active], ['做一份投研报告'])


class TestCapabilityIsKeptForHumansAndSolutions(unittest.TestCase):
    """定时检查能力本身保留 —— 删掉会砸掉 solutions（方案包声明的任务）。"""

    def test_task_store_still_accepts_check_cron(self):
        t = task_store.create(goal='每小时巡逻一遍', check_cron='0 * * * *')
        try:
            self.assertEqual(t.check_cron, '0 * * * *')
        finally:
            task_store._tasks.pop(t.id, None)

    def test_register_check_still_exists_for_restart_restore(self):
        """重启时 event/llm.py 要靠它把 API / 方案包建的检查恢复回来。"""
        self.assertTrue(callable(getattr(task_mod, '_register_check', None)))
        llm_src = (_SRC / 'event' / 'llm.py').read_text()
        self.assertIn('from event.task import _register_check', llm_src)

    def test_other_entry_points_untouched(self):
        for rel in ('api/tasks.py', 'api/solutions.py'):
            self.assertIn('check_cron', (_SRC / rel).read_text(),
                          f'{rel} 的 check_cron 入口不该被删')


if __name__ == '__main__':
    unittest.main()
