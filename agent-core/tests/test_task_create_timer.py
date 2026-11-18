"""
test_task_create_timer.py — 明确要求的定时能注册，普通任务仍不自行轮询。

Tianyi 实测：模型每次派子代理都顺手 task_create(check_cron="*/2 * * * *")，于是每 2 分钟
被叫醒一次去 subagent_status + task_update，一次两轮、每轮三万多 token（实测 prompt=35307）。
一小时触发了 12 次。

查下来整份 prompt 里**没有任何一句话**让它这么做 —— `prompt_system.md` 只说"用 task_create
追踪""收到定时检查事件时才 task_update"，从没说过要建 cron。唯一的驱动是 check_cron 这个
参数描述本身：给了现成的 "*/2 * * * *" 可抄，又把"留空"写成"则不自动检查"（听起来像放弃了
什么）。恢复入口时保留默认空，不再提供可照抄的高频示例，只用于用户明确要求的定时。

它想解决的三件事现在都有人管：
  · 子代理干完   → manager._notify_completion 推 URGENT（实测 160ms 内到）
  · 子代理卡死   → manager._timeout_watchdog
  · 用户不知进展 → 框架按沉默时长自动播报（event/llm.py）

**但定时检查这个能力本身要留着**：设置页改任务、方案包声明的任务仍然能带 cron，那些是人
明确要的。这个文件两头都锚住。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_task_create_timer.py
"""
import asyncio
import gc
import os
import pathlib
import sys
import tempfile
import time
import typing
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import scheduler  # noqa: E402
import task_store  # noqa: E402
import event  # noqa: E402
from event.llm import _build_system_tools  # noqa: E402

task_mod = sys.modules['event.task']
_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


class TestExplicitTaskTimer(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_patch = patch.object(task_store.config, 'DB_PATH', os.path.join(directory.name, 'test.db'))
        db_patch.start()
        self.addCleanup(db_patch.stop)
        self.addCleanup(gc.collect)
        self._saved = dict(task_store._tasks)
        task_store._tasks.clear()

    def tearDown(self):
        task_store._tasks.clear()
        task_store._tasks.update(self._saved)

    def test_tool_schema_has_optional_cron_param(self):
        """普通任务只需目标；用户指定定时时才传 check_cron。"""
        d = _build_system_tools([('task_create', task_mod.Tools().task_create)])
        props = d['task_create']['schema']['parameters']['properties']
        self.assertEqual(list(props), ['goal', 'check_cron', 'delay_minutes', 'remind_at'])
        self.assertEqual(d['task_create']['schema']['parameters']['required'], ['goal'])
        self.assertIn(time.strftime('%Z %z'), props['check_cron']['description'])
        self.assertIn('仅用户明确要求', props['check_cron']['description'])

    def test_the_copyable_example_is_gone(self):
        """那个 "*/2 * * * *" 是模型照抄的来源，不能再出现在它读得到的地方。"""
        d = _build_system_tools([('task_create', task_mod.Tools().task_create)])
        schema = d['task_create']['schema']
        blob = repr(schema)
        self.assertNotIn('*/2 * * * *', blob)
        self.assertIn('普通任务留空', blob)

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

    def test_explicit_timer_registers_fires_and_stops_on_done(self):
        async def go():
            fired = asyncio.Event()
            events = []

            async def enqueue(**event):
                events.append(event)
                fired.set()

            with patch.object(scheduler, '_next_run', side_effect=lambda *a: time.time() + 0.02), \
                    patch.object(scheduler.event_bus, 'enqueue', enqueue):
                result = await task_mod.Tools().task_create(goal='报时一次', check_cron='0 8 * * *')
                task = task_store.active_tasks()[0]
                name = f'task:{task.id}'
                try:
                    self.assertIn('定时已注册', result)
                    self.assertIn(name, scheduler._dynamic_tasks)
                    await asyncio.wait_for(fired.wait(), 2)
                    self.assertEqual(events[0]['source'], 'scheduler:' + name)
                    self.assertIn('报时一次', events[0]['text'])
                    await task_mod.Tools().task_done(task.id)
                    await asyncio.sleep(0.05)
                    self.assertNotIn(name, scheduler._dynamic_tasks)
                    self.assertEqual(len(events), 1)
                finally:
                    scheduler.remove_job(name)

        asyncio.run(go())

    def test_explicit_timer_is_persisted_and_can_be_restored(self):
        with patch.object(scheduler, 'add_job') as add:
            result = asyncio.run(task_mod.Tools().task_create(goal='每日提醒', check_cron='0 8 * * *'))
            task = task_store.active_tasks()[0]
            self.assertIn('定时已注册', result)
            task_store._tasks.clear()
            task_store.load_all()
            restored = task_store.get(task.id)
            self.assertEqual(restored.check_cron, '0 8 * * *')
            task_mod._register_check(restored)
            self.assertEqual(add.call_count, 2)
            self.assertEqual(add.call_args.kwargs['cron'], '0 8 * * *')

    def test_invalid_timer_does_not_create_or_register_a_task(self):
        with patch.object(scheduler, 'add_job') as add:
            result = asyncio.run(task_mod.Tools().task_create(goal='报时', check_cron='not a cron'))
        self.assertIn('定时设置失败', result)
        self.assertEqual(task_store.active_tasks(), [])
        self.assertFalse(add.called)

    def test_created_timer_reports_local_time_and_waits_for_its_event(self):
        next_ts = 1794990120.0
        expected_time = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(next_ts))
        with patch.object(scheduler, '_next_run', return_value=next_ts), \
                patch.object(scheduler, 'add_job'):
            result = asyncio.run(task_mod.Tools().task_create(goal='稍后报时', check_cron='0 8 * * *'))
        task = task_store.active_tasks()[0]
        self.assertIn(expected_time, result)
        self.assertIn(f'scheduler:task:{task.id}', result)
        self.assertIn('尚未触发', result)
        self.assertIn('不要现在执行提醒或标记完成', result)

    def test_relative_timer_is_never_earlier_than_requested(self):
        now = 1794990625  # 2026-11-18 16:30:25 +08:00
        with patch.object(time, 'time', return_value=now), patch.object(scheduler, 'add_job'):
            result = asyncio.run(task_mod.Tools().task_create(goal='一分钟后报时', delay_minutes=1))
        task = task_store.active_tasks()[0]
        next_ts = scheduler._next_run(task.check_cron, now)
        self.assertEqual(next_ts, (now // 60 + 2) * 60)
        self.assertGreaterEqual(next_ts, now + 60)
        self.assertIn('北京时间 UTC+08:00', result)

    def test_absolute_timer_converts_user_timezone_in_code(self):
        now = 1794990625
        with patch.object(time, 'time', return_value=now), patch.object(scheduler, 'add_job'):
            result = asyncio.run(task_mod.Tools().task_create(goal='17:30报时', remind_at='17:30'))
        task = task_store.active_tasks()[0]
        next_ts = scheduler._next_run(task.check_cron, now)
        expected = task_mod.datetime.datetime(2026, 11, 18, 17, 30,
                    tzinfo=task_mod.datetime.timezone(task_mod.datetime.timedelta(hours=8))).timestamp()
        self.assertEqual(next_ts, expected)
        self.assertIn('2026-11-18 17:30 北京时间', result)

    def test_invalid_or_conflicting_schedule_does_not_create_task(self):
        for args in ({'delay_minutes': -1}, {'delay_minutes': float('nan')},
                     {'delay_minutes': float('inf')}, {'delay_minutes': True},
                     {'remind_at': '25:90'}, {'remind_at': '2000-01-01T08:00'},
                     {'delay_minutes': 1, 'remind_at': '18:30'},
                     {'check_cron': '0 8 * * *', 'delay_minutes': 1}):
            with self.subTest(args=args):
                result = asyncio.run(task_mod.Tools().task_create(goal='无效定时', **args))
                self.assertIn('定时设置失败', result)
                self.assertEqual(task_store.active_tasks(), [])


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
