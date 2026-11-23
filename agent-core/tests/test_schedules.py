"""
test_schedules.py — 用户要求的定时事项：一次性提醒、每天/每周的例行事务。

872d9c1 把 task_create 的 check_cron 删了（模型拿它建每 2 分钟的轮询），之后"一分钟后
报时""每天晚上巡逻"就没有任何工具能做，模型只能口头答应、到点什么也不发生。

这里锚住三件事：
  · 能做到：一次性与周期事项都能建、到点推事件、重启恢复。
  · 做不出轮询：入口不收 cron，最短周期一天，有数量上限和去重。
  · 不和任务追踪纠缠：task_done / task_force_clear 碰不到它，一次性的触发后自动结束。

以及顺带的两个通用修复：调度器在整分钟边界重复触发；有活跃任务时 status 时间变成 epoch。

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest agent-core/tests/test_schedules.py -q
"""
import asyncio
import datetime
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import scheduler  # noqa: E402
import schedules  # noqa: E402
import task_store  # noqa: E402
import event  # noqa: E402
from event.llm import _build_system_tools  # noqa: E402

task_mod = sys.modules['event.task']
TZ = schedules.TZ
# 2026-11-18 16:30:25 北京时间（周三）
NOW = datetime.datetime(2026, 11, 18, 16, 30, 25, tzinfo=TZ).timestamp()


def bj(*args) -> float:
    return datetime.datetime(*args, tzinfo=TZ).timestamp()


def run(coro):
    return asyncio.run(coro)


class _Isolated(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = patch.object(task_store.config, 'DB_PATH', os.path.join(d.name, 'test.db'))
        p.start()
        self.addCleanup(p.stop)
        self._saved = dict(task_store._tasks)
        task_store._tasks.clear()
        self.addCleanup(lambda: (task_store._tasks.clear(), task_store._tasks.update(self._saved)))


# ── 调度器 ───────────────────────────────────────────────────────────────────

class TestScheduler(unittest.TestCase):
    def test_early_wakeup_does_not_fire_the_same_minute_twice(self):
        """sleep 醒来时墙钟还差半毫秒到整分 —— 从 time.time() 往后算会再选中这一分钟。"""
        async def go():
            clock = [NOW]
            fired = []

            async def sleep(delay):
                # 每次都早醒半毫秒
                clock[0] += delay - 0.0005

            async def enqueue(**ev):
                fired.append(clock[0])
                if len(fired) == 3:
                    raise asyncio.CancelledError

            with patch.object(time, 'time', side_effect=lambda: clock[0]), \
                    patch.object(asyncio, 'sleep', side_effect=sleep), \
                    patch.object(scheduler.event_bus, 'enqueue', side_effect=enqueue):
                with self.assertRaises(asyncio.CancelledError):
                    await scheduler._run_job({'name': 't', 'cron': '* * * * *', 'text': 't'})
            minutes = [round(t / 60) for t in fired]
            self.assertEqual(len(set(minutes)), 3, f'同一分钟触发了多次: {fired}')
            self.assertEqual(minutes[1] - minutes[0], 1)

        run(go())

    def test_cron_can_be_read_in_a_given_timezone(self):
        """容器是 UTC，用户说的 22:00 是北京时间。"""
        nxt = scheduler._next_run('0 22 * * *', NOW, TZ)
        self.assertEqual(nxt, bj(2026, 11, 18, 22, 0))
        utc = datetime.timezone.utc
        self.assertEqual(scheduler._next_run('0 22 * * *', NOW, utc),
                         datetime.datetime(2026, 11, 18, 22, 0, tzinfo=utc).timestamp())

    def test_add_once_fires_once_and_removes_itself(self):
        async def go():
            events, done = [], []

            async def enqueue(**ev):
                events.append(ev)

            with patch.object(scheduler.event_bus, 'enqueue', enqueue):
                scheduler.add_once('once:t', time.time() + 0.02, 'hello',
                                   on_fired=lambda: done.append(1))
                self.assertIn('once:t', scheduler._dynamic_tasks)
                await asyncio.sleep(0.1)
            self.assertEqual([e['source'] for e in events], ['scheduler:once:t'])
            self.assertEqual(done, [1])
            self.assertNotIn('once:t', scheduler._dynamic_tasks)

        run(go())


# ── 参数解析 ─────────────────────────────────────────────────────────────────

class TestBuild(unittest.TestCase):
    def test_delay(self):
        self.assertEqual(schedules.build(1, '', '', NOW), {'at': NOW + 60})

    def test_clock_today_or_tomorrow(self):
        self.assertEqual(schedules.build(0, '17:30', '', NOW)['at'], bj(2026, 11, 18, 17, 30))
        self.assertEqual(schedules.build(0, '08:00', '', NOW)['at'], bj(2026, 11, 19, 8, 0))

    def test_full_datetime_and_offsets(self):
        for v in ('2026-11-18 17:30', '2026-11-18T17:30', '2026-11-18T09:30+00:00',
                  '2026-11-18T04:30-05:00'):
            with self.subTest(v=v):
                self.assertEqual(schedules.build(0, v, '', NOW)['at'], bj(2026, 11, 18, 17, 30))

    def test_recurring(self):
        self.assertEqual(schedules.build(0, '22:00', 'daily', NOW),
                         {'days': list(range(7)), 'hour': 22, 'minute': 0})
        self.assertEqual(schedules.build(0, '08:30', 'mon, wed', NOW)['days'], [0, 2])

    def test_rejected(self):
        for args in [(0, '', ''), (1, '17:30', ''), (-1, '', ''), (float('nan'), '', ''),
                     (True, '', ''), ('1', '', ''), (0, 1730, ''), (0, '25:90', ''),
                     (0, '2000-01-01 08:00', ''), (1, '', 'daily'), (0, '', 'daily'),
                     (0, '2026-11-19 22:00', 'daily'), (0, '22:00', 'hourly'),
                     (0, '22:00', '*/2'), (0, '22:00+00:00', 'daily'),
                     (60 * 24 * 400, '', '')]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    schedules.build(*args, now=NOW)

    def test_describe_and_next_fire(self):
        daily = schedules.build(0, '22:00', 'daily', NOW)
        self.assertEqual(schedules.describe(daily), '每天 22:00')
        self.assertEqual(schedules.next_fire(daily, NOW), bj(2026, 11, 18, 22, 0))
        mon = schedules.build(0, '08:30', 'mon', NOW)
        self.assertEqual(schedules.describe(mon), '每周一 08:30')
        self.assertEqual(schedules.next_fire(mon, NOW), bj(2026, 11, 23, 8, 30))
        wk = schedules.build(0, '09:00', 'weekdays', NOW)
        self.assertEqual(schedules.describe(wk), '工作日每天 09:00')


# ── 工具 ─────────────────────────────────────────────────────────────────────

class TestTools(_Isolated):
    def test_the_model_cannot_hand_over_a_cron(self):
        d = _build_system_tools([('schedule_create', task_mod.Tools().schedule_create),
                                 ('task_create', task_mod.Tools().task_create)])
        props = d['schedule_create']['schema']['parameters']['properties']
        self.assertEqual(list(props), ['text', 'delay_minutes', 'at', 'repeat'])
        self.assertNotIn('*/', repr(d['schedule_create']['schema']))
        self.assertEqual(list(d['task_create']['schema']['parameters']['properties']), ['goal'])

    def test_one_off_fires_and_ends_itself(self):
        async def go():
            events = []

            async def enqueue(**ev):
                events.append(ev)

            with patch.object(scheduler.event_bus, 'enqueue', enqueue), \
                    patch.object(schedules, 'build', return_value={'at': time.time() + 0.02}):
                out = await task_mod.Tools().schedule_create(text='报一下时间', delay_minutes=1)
                tid = task_store.active_tasks()[0].id
                self.assertIn('还没到点', out)
                await asyncio.sleep(0.1)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['source'], f'scheduler:task:{tid}')
            self.assertIn('报一下时间', events[0]['text'])
            self.assertIn('不需要 task_done', events[0]['text'])
            self.assertEqual(task_store.active_tasks(), [])

        run(go())

    def test_recurring_goes_to_the_scheduler_in_beijing_time(self):
        with patch.object(scheduler, 'add_job') as add, patch.object(time, 'time', return_value=NOW):
            out = run(task_mod.Tools().schedule_create(text='巡逻一圈', at='22:00', repeat='daily'))
        self.assertIn('每天 22:00', out)
        self.assertIn('2026-11-18 22:00', out)
        args, kwargs = add.call_args
        self.assertEqual(args[1], '0 22 * * 1,2,3,4,5,6,0')
        self.assertIs(kwargs['tz'], TZ)
        self.assertIn('schedule_cancel', args[2])

    def test_duplicates_and_cap(self):
        with patch.object(scheduler, 'add_job'):
            run(task_mod.Tools().schedule_create(text='巡逻', at='22:00', repeat='daily'))
            out = run(task_mod.Tools().schedule_create(text='巡逻', at='22:00', repeat='daily'))
            self.assertIn('未重复创建', out)
            self.assertEqual(len(schedules.active()), 1)
            for i in range(schedules.MAX_ACTIVE - 1):
                run(task_mod.Tools().schedule_create(text=f'事项{i}', at='22:00', repeat='daily'))
            out = run(task_mod.Tools().schedule_create(text='再来一条', at='22:00', repeat='daily'))
        self.assertIn('没有设定成功', out)
        self.assertEqual(len(schedules.active()), schedules.MAX_ACTIVE)

    def test_invalid_input_creates_nothing(self):
        with patch.object(scheduler, 'add_job') as add, patch.object(scheduler, 'add_once') as once:
            out = run(task_mod.Tools().schedule_create(text='x', at='22:00', repeat='hourly'))
        self.assertIn('没有设定成功', out)
        self.assertEqual(task_store.active_tasks(), [])
        self.assertFalse(add.called or once.called)

    def test_task_tools_cannot_touch_a_schedule(self):
        """「任务完成后及时 task_done」会把模型往周期事项上推 —— 那等于删了用户的每天巡逻。"""
        with patch.object(scheduler, 'add_job'):
            run(task_mod.Tools().schedule_create(text='巡逻', at='22:00', repeat='daily'))
        sid = schedules.active()[0].id
        run(task_mod.Tools().task_create(goal='走到B点'))
        self.assertIn('schedule_cancel', run(task_mod.Tools().task_done(sid)))
        self.assertIn('schedule_cancel', run(task_mod.Tools().task_fail(sid)))
        self.assertNotIn('巡逻', run(task_mod.Tools().task_list()))
        run(task_mod.Tools().task_force_clear())
        self.assertEqual([t.id for t in task_store.active_tasks()], [sid])
        with patch.object(scheduler, 'remove_job') as rm:
            self.assertIn('已取消', run(task_mod.Tools().schedule_cancel(sid)))
        rm.assert_called_once_with(f'task:{sid}')
        self.assertEqual(task_store.active_tasks(), [])

    def test_subagents_cannot_set_or_cancel(self):
        from subagent.agent import _DENIED_TOOLS
        self.assertTrue({'schedule_create', 'schedule_cancel'} <= _DENIED_TOOLS)


# ── 重启恢复 ─────────────────────────────────────────────────────────────────

class TestRestore(_Isolated):
    def _reload(self):
        task_store._tasks.clear()
        task_store.load_all()
        return task_store.active_tasks()

    def test_recurring_survives_restart(self):
        with patch.object(scheduler, 'add_job'):
            run(task_mod.Tools().schedule_create(text='巡逻', at='22:00', repeat='daily'))
        [t] = self._reload()
        with patch.object(scheduler, 'add_job') as add:
            task_mod._register_check(t)
        self.assertEqual(add.call_args.args[1], '0 22 * * 1,2,3,4,5,6,0')

    def test_one_off_missed_briefly_is_delivered_late(self):
        t = task_store.create(goal='报时', metadata={'schedule': {'at': NOW - 120}})
        with patch.object(scheduler, 'add_once') as once, patch.object(time, 'time', return_value=NOW):
            task_mod._register_check(t)
        self.assertEqual(once.call_args.args[1], NOW - 120)

    def test_one_off_missed_long_ago_is_dropped(self):
        t = task_store.create(goal='报时', metadata={'schedule': {'at': NOW - 3600}})
        with patch.object(scheduler, 'add_once') as once, patch.object(time, 'time', return_value=NOW):
            task_mod._register_check(t)
        once.assert_not_called()
        self.assertEqual(task_store.active_tasks(), [])


# ── status 快照 ──────────────────────────────────────────────────────────────

class TestStatus(_Isolated):
    def test_time_stays_a_clock_with_active_tasks_and_schedules_are_listed_apart(self):
        import prompt
        task_store.create(goal='走到B点')
        task_store.create(goal='巡逻', metadata={'schedule': {'days': list(range(7)),
                                                                'hour': 22, 'minute': 0}})
        with patch.object(prompt, 'format_ts', lambda **kw: '2026-11-18 16:30:25'):
            text = prompt._env_dynamic()
        self.assertIn('<status time="2026-11-18 16:30:25">', text)
        tasks = text.split('<active_tasks>')[1].split('</active_tasks>')[0]
        self.assertIn('走到B点', tasks)
        self.assertNotIn('巡逻', tasks)
        self.assertIn('when="每天 22:00"', text)


if __name__ == '__main__':
    unittest.main()
