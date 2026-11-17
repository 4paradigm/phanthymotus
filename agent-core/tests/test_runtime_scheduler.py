"""
test_runtime_scheduler.py — 运行时定时任务（schedule_store / scheduler / 模型工具开关）。

锚住四件事：
  1. 时区：cron 按 Asia/Shanghai 解释，不再要人手算 UTC（容器是 UTC）；不写 tz 的旧配置
     行为不变。
  2. 闸：最短间隔、每日触发上限、条数上限在“建”的时候就拦；运行时每日上限再兜一次。
  3. 运行时增删改立即注册 / 注销 scheduler job，库里持久化，重启（start_all）恢复。
  4. 模型工具默认关：工具表里看不到，被调用也拒绝；打开后只能改删自己建的。

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_runtime_scheduler.py
  （没有 pytest 时：python3 -m unittest tests/test_runtime_scheduler.py）
"""
import asyncio
import datetime
import importlib.util
import json
import os
import pathlib
import sqlite3
import sys
import tempfile
import time
import unittest

_SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(_SRC))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import config  # noqa: E402
import scheduler  # noqa: E402
import schedule_store  # noqa: E402

CN = datetime.timezone(datetime.timedelta(hours=8))


def _ts(s: str) -> float:
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M').replace(tzinfo=CN).timestamp()


def _load_tools_module():
    """直接按文件加载 event/schedule.py —— import event 会把整个 event.llm 拉进来。"""
    spec = importlib.util.spec_from_file_location('event_schedule_under_test', _SRC / 'event' / 'schedule.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeJobs:
    """替换 scheduler.add_job / remove_job，记录注册情况，不真起协程。"""

    def __enter__(self):
        self.jobs = {}
        self._add, self._remove = scheduler.add_job, scheduler.remove_job
        scheduler.add_job = lambda name, cron, text, **kw: self.jobs.__setitem__(name, dict(cron=cron, text=text, **kw))
        scheduler.remove_job = lambda name: self.jobs.pop(name, None)
        return self

    def __exit__(self, *a):
        scheduler.add_job, scheduler.remove_job = self._add, self._remove


class _Base(unittest.TestCase):
    def setUp(self):
        # 全套跑时别的测试文件可能把 config.main 换成普通 dict 且不还原：这里换回 ConfigDB，
        # 结束后恢复原样；scheduler_settings 和 scheduler 里排队的调用也都还原，不影响别的测试。
        if not isinstance(config.main, config.ConfigDB):
            saved_main = config.main
            config.main = config.ConfigDB()
            self.addCleanup(setattr, config, 'main', saved_main)
        saved_settings = config.main.get('scheduler_settings')
        self.addCleanup(lambda: config.main.__setitem__(
            'scheduler_settings', saved_settings if saved_settings is not None
            else dict(schedule_store.DEFAULT_SETTINGS)))
        self.addCleanup(scheduler._pending.clear)
        saved_stopping = scheduler._stopping
        scheduler._stopping = False
        self.addCleanup(setattr, scheduler, '_stopping', saved_stopping)
        schedule_store.clear_all() if schedule_store._loaded else None
        with schedule_store._get_conn() as conn:
            conn.execute('DELETE FROM schedules')
        schedule_store._schedules.clear()
        schedule_store._fire_times.clear()
        schedule_store._last_fire.clear()
        with schedule_store._get_conn() as conn:
            conn.execute('DELETE FROM schedule_fires')
        schedule_store._loaded = True
        config.main['scheduler_settings'] = dict(schedule_store.DEFAULT_SETTINGS)
        schedule_store._settings_snapshot = None

    @staticmethod
    def _far_cron(minute: int = 0) -> str:
        """离现在 11 小时以上的每日 cron：手动触发的“离下次定时太近”不会误伤测试。"""
        return f'{minute} {(datetime.datetime.now(CN).hour + 12) % 24} * * *'

    @staticmethod
    def _due_soon(s) -> float:
        """把一次性定时改到一两秒后（秒级精度），返回 unix 时间，给 scheduler._run_job 的 job['at'] 用。"""
        t = int(time.time()) + 2
        schedule_store._schedules[s.id].at = datetime.datetime.fromtimestamp(t, CN).strftime('%Y-%m-%d %H:%M:%S')
        return float(t)


class TestCronAndTimezone(_Base):
    def test_beijing_8am_is_not_utc_8am(self):
        after = _ts('2026-11-16 07:00')
        nxt = scheduler._next_run('0 8 * * *', after, 'Asia/Shanghai')
        self.assertEqual(nxt, _ts('2026-11-16 08:00'))

    def test_no_tz_keeps_local_behaviour(self):
        after = time.time()
        self.assertEqual(scheduler._next_run('*/30 * * * *', after),
                         scheduler._next_run('*/30 * * * *', after, None))

    def test_range_with_step(self):
        m, h, *_ = scheduler.parse_cron('0-30/10 8-10 * * *')
        self.assertEqual(m, {0, 10, 20, 30})
        self.assertEqual(h, {8, 9, 10})

    def test_bad_cron_is_rejected(self):
        for bad in ('0 8 * *', '61 8 * * *', '0 25 * * *', '*/0 * * * *', 'a b c d e'):
            with self.assertRaises(ValueError, msg=bad):
                scheduler.parse_cron(bad)

    def test_unknown_tz(self):
        with self.assertRaises(ValueError):
            scheduler.get_tz('Mars/Olympus')


class TestValidation(_Base):
    def test_min_interval(self):
        with _FakeJobs():
            with self.assertRaisesRegex(ValueError, '最短'):
                schedule_store.create(name='太频繁', text='x', cron='*/5 * * * *')
            s = schedule_store.create(name='吃药', text='服药提醒定时检查', cron='0,10,20,30 8,12,18 * * *')
            self.assertTrue(s.enabled)

    def test_daily_cap_counts_all_schedules(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_fires_per_day': 12}
        with _FakeJobs():
            schedule_store.create(name='a', text='x', cron='0,10,20,30 8,12,18 * * *')   # 12 次/天
            with self.assertRaisesRegex(ValueError, '超过上限'):
                schedule_store.create(name='b', text='x', cron='0 21 * * *')

    def test_unique_name_and_counts(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_per_skill': 1}
        with _FakeJobs():
            schedule_store.create(name='a', text='x', cron='0 8 * * *', skill='med')
            with self.assertRaisesRegex(ValueError, '已经有'):
                schedule_store.create(name='a', text='y', cron='0 9 * * *')
            with self.assertRaisesRegex(ValueError, '技能 med'):
                schedule_store.create(name='b', text='y', cron='0 9 * * *', skill='med')

    def test_once_in_past_rejected(self):
        with _FakeJobs():
            with self.assertRaisesRegex(ValueError, '过去'):
                schedule_store.create(name='o', text='x', kind='once', at='2020-01-01 08:00')


class TestRegistration(_Base):
    def test_create_update_delete_touch_scheduler(self):
        with _FakeJobs() as fj:
            s = schedule_store.create(name='吃药', text='服药提醒定时检查', cron='0 8 * * *', skill='q5-elder-med-check')
            job = fj.jobs[f'sched:{s.id}']
            self.assertEqual(job['tz'], 'Asia/Shanghai')
            self.assertEqual(job['source'], 'scheduler:吃药')
            self.assertIn('服药提醒定时检查', job['text'])
            self.assertIn('q5-elder-med-check', job['text'])
            schedule_store.update(s.id, enabled=False)
            self.assertNotIn(f'sched:{s.id}', fj.jobs)
            schedule_store.update(s.id, enabled=True, cron='30 8 * * *')
            self.assertEqual(fj.jobs[f'sched:{s.id}']['cron'], '30 8 * * *')
            schedule_store.delete(s.id)
            self.assertNotIn(f'sched:{s.id}', fj.jobs)

    def test_failed_update_keeps_old(self):
        with _FakeJobs():
            s = schedule_store.create(name='a', text='x', cron='0 8 * * *')
            with self.assertRaises(ValueError):
                schedule_store.update(s.id, cron='* * * * *')
            self.assertEqual(schedule_store.find('a').cron, '0 8 * * *')

    def test_persisted_and_restored(self):
        with _FakeJobs() as fj:
            s = schedule_store.create(name='a', text='x', cron='0 8 * * *')
            schedule_store._schedules.clear()
            fj.jobs.clear()
            self.assertEqual(schedule_store.start_all(), 1)
            self.assertIn(f'sched:{s.id}', fj.jobs)
            self.assertEqual(schedule_store.find('a').text, 'x')


class TestFiring(_Base):
    def test_runtime_daily_cap_and_once(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_fires_per_day': 1}
        with _FakeJobs():
            a = schedule_store.create(name='a', text='x', kind='once',
                                      at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))
        check, done = schedule_store._on_fire(a.id), schedule_store._after_fire(a.id)
        self.assertTrue(check({}))
        self.assertTrue(schedule_store.find('a').enabled, '入队之前不改状态')
        done({})
        self.assertFalse(schedule_store.find('a').enabled, '一次性定时成功触发后应停用')
        schedule_store._schedules[a.id].enabled = True
        self.assertFalse(check({}), '超过每日上限应跳过')

    def test_one_shot_job_enqueues_once(self):
        import event_bus
        got = []

        async def fake_enqueue(source, text, payload=None):
            got.append((source, text))

        async def go():
            orig = event_bus.enqueue
            event_bus.enqueue = fake_enqueue
            try:
                await scheduler._run_job({'name': 'once', 'text': 'hi', 'at': time.time() + 0.2,
                                          'source': 'scheduler:once'})
            finally:
                event_bus.enqueue = orig
        asyncio.run(asyncio.wait_for(go(), 5))
        self.assertEqual(got, [('scheduler:once', 'hi')])


class TestModelTools(_Base):
    def setUp(self):
        super().setUp()
        self.mod = _load_tools_module()
        self.tools = self.mod.Tools()

    def test_off_by_default(self):
        self.assertFalse(schedule_store.DEFAULT_SETTINGS['model_tools_enabled'])
        self.assertFalse(config._DB_DEFAULTS['scheduler_settings']['model_tools_enabled'])
        r = asyncio.run(self.tools.schedule_create(name='x', text='y', cron='0 8 * * *'))
        self.assertIn('没有开启', r)
        self.assertEqual(schedule_store.list_all(), [])

    def test_llm_hides_tools_when_off(self):
        src = (_SRC / 'event' / 'llm.py').read_text(encoding='utf-8')
        self.assertIn('_hide_schedule_tools()', src)
        self.assertIn("startswith('schedule_')", src)

    def test_no_copyable_high_frequency_example(self):
        blob = ''.join((getattr(self.tools, n).__doc__ or '') for n in ('schedule_list', 'schedule_create', 'schedule_delete'))
        self.assertNotIn('*/', blob)

    def test_model_cannot_touch_human_schedules(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'model_tools_enabled': True}
        with _FakeJobs():
            schedule_store.create(name='人建的', text='x', cron='0 8 * * *', created_by='web')
            r = asyncio.run(self.tools.schedule_delete(id='人建的'))
            self.assertIn('不能删', r)
            r = asyncio.run(self.tools.schedule_create(name='人建的', text='y', cron='0 9 * * *'))
            self.assertIn('不能改', r)
            r = asyncio.run(self.tools.schedule_create(name='模型建的', text='y', cron='0 9 * * *'))
            self.assertIn('已创建', r)
            self.assertEqual(schedule_store.find('模型建的').created_by, 'model')
            r = asyncio.run(self.tools.schedule_create(name='太快', text='y', cron='*/2 * * * *'))
            self.assertIn('没有建成', r)
            r = asyncio.run(self.tools.schedule_delete(id='模型建的'))
            self.assertIn('已删除', r)


class TestReviewRegressions(_Base):
    """PR #398 审查意见对应的回归：每日上限按日期算、坏数据不拖垮启动、cron 语义、
    同步路由不卡事件循环。"""

    def test_one_shots_on_different_days_do_not_stack(self):
        """49 个一次性定时分在 49 天：每天只有 1 次，不该被每日上限拒绝。"""
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_schedules': 60,
                                             'max_fires_per_day': 48}
        # 从明天起算（写死日期的话过了那天测试就会因为“时间已经过去”失败）
        base = (datetime.datetime.now(CN) + datetime.timedelta(days=1)).replace(hour=8, minute=0,
                                                                              second=0, microsecond=0)
        with _FakeJobs():
            for i in range(49):
                day = (base + datetime.timedelta(days=i)).strftime('%Y-%m-%d %H:%M')
                schedule_store.create(name=f'o{i}', text='x', kind='once', at=day)

    def test_one_shots_on_the_same_day_do_stack(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_schedules': 60,
                                             'max_fires_per_day': 2}
        d1, d2 = ((datetime.datetime.now(CN) + datetime.timedelta(days=n)).strftime('%Y-%m-%d') for n in (2, 3))
        with _FakeJobs():
            schedule_store.create(name='a', text='x', kind='once', at=f'{d1} 08:00')
            schedule_store.create(name='b', text='x', kind='once', at=f'{d1} 09:00')
            with self.assertRaisesRegex(ValueError, '超过上限'):
                schedule_store.create(name='c', text='x', kind='once', at=f'{d1} 10:00')
            schedule_store.create(name='d', text='x', kind='once', at=f'{d2} 10:00')

    def test_corrupt_row_does_not_stop_startup(self):
        with _FakeJobs() as fj:
            good = schedule_store.create(name='好的', text='x', cron='0 8 * * *')
            schedule_store._schedules.clear(); fj.jobs.clear()
            bad = schedule_store.Schedule(name='坏的', text='x', kind='once', at='不是时间')
            schedule_store._schedules[bad.id] = bad
            schedule_store._persist(bad)
            self.assertEqual(schedule_store.start_all(), 1, '坏的跳过，好的照常启动')
            self.assertIn(f'sched:{good.id}', fj.jobs)
            self.assertFalse(schedule_store.find('坏的').enabled, '坏数据停用，不留“启用但没注册”的记录')

    def test_dom_and_dow_are_anded(self):
        """'0 8 13 * 5'：13 号且是周五才触发（本调度器一直如此，不是标准 cron 的“或”）。"""
        after = _ts('2026-11-16 00:00')
        nxt = scheduler._next_run('0 8 13 * 5', after, 'Asia/Shanghai')
        d = datetime.datetime.fromtimestamp(nxt, CN)
        self.assertEqual(d.strftime('%Y-%m-%d %H:%M'), '2027-08-13 08:00')   # 13 号，周五

    def test_add_job_lands_on_the_running_loop(self):
        """从别的线程调 add_job：task 要建在事件循环线程里，不能凭空抛错。"""
        import threading

        async def main():
            scheduler._loop = asyncio.get_running_loop()
            err = []
            t = threading.Thread(target=lambda: _try(lambda: scheduler.add_job('x-thread', '0 8 * * *', 'x')))
            def _try(fn):
                try: fn()
                except Exception as e: err.append(e)
            t.start(); t.join()
            for _ in range(3):
                await asyncio.sleep(0.01)
            self.assertEqual(err, [])
            self.assertIn('x-thread', scheduler._dynamic_tasks)
            scheduler.remove_job('x-thread')
            for _ in range(3):
                await asyncio.sleep(0.01)
            self.assertNotIn('x-thread', scheduler._dynamic_tasks)
        asyncio.run(main())

    def test_api_routes_are_sync(self):
        """SQLite 读写放在 def 路由里（FastAPI 走线程池），不能是 async def。"""
        src = (_SRC / 'api' / 'schedules.py').read_text(encoding='utf-8')
        for name in ('list_schedules', 'create_schedule', 'update_schedule', 'delete_schedule',
                     'patch_settings', 'preview', 'get_settings'):
            self.assertIn(f'\ndef {name}(', src, f'{name} 应该是同步路由')
            self.assertNotIn(f'async def {name}(', src)
        self.assertIn('asyncio.to_thread', (_SRC / 'schedule_store.py').read_text(encoding='utf-8'))



class TestSecondReview(_Base):
    """PR #398 第二轮审查：逐天相加的上限估算、fire_now 快照、一次性入队失败不丢。"""

    @staticmethod
    def _next_weekday(wd: int) -> datetime.datetime:
        d = datetime.datetime.now(CN).replace(hour=0, minute=0, second=0, microsecond=0)
        d += datetime.timedelta(days=1)
        while d.weekday() != wd:
            d += datetime.timedelta(days=1)
        return d

    def test_peaks_from_different_days_are_not_added(self):
        """周一 cron 8 次 + 周二 5 个一次性：哪天都不超过 10，不该被拒；
        周一再加 3 个一次性就是 11 次，要拒。"""
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_fires_per_day': 10,
                                             'max_schedules': 30}
        mon, tue = self._next_weekday(0), self._next_weekday(1)
        with _FakeJobs():
            schedule_store.create(name='周一巡检', text='x', cron='0 8-15 * * 1')
            for i in range(5):
                schedule_store.create(name=f'周二{i}', text='x', kind='once',
                                      at=(tue + datetime.timedelta(hours=9 + i)).strftime('%Y-%m-%d %H:%M'))
            schedule_store.create(name='周一a', text='x', kind='once',
                                  at=(mon + datetime.timedelta(hours=17)).strftime('%Y-%m-%d %H:%M'))
            schedule_store.create(name='周一b', text='x', kind='once',
                                  at=(mon + datetime.timedelta(hours=18)).strftime('%Y-%m-%d %H:%M'))
            with self.assertRaisesRegex(ValueError, '11 次'):
                schedule_store.create(name='周一c', text='x', kind='once',
                                      at=(mon + datetime.timedelta(hours=19)).strftime('%Y-%m-%d %H:%M'))

    def test_fire_now_uses_a_locked_snapshot(self):
        import event_bus
        sent = []

        async def fake(source, text, payload=None):
            sent.append((source, text))

        with _FakeJobs():
            s = schedule_store.create(name='快照', text='旧文字', cron=self._far_cron())
        snap = schedule_store._snapshot(s.id)
        schedule_store._schedules[s.id].text = '被改了'
        self.assertEqual(snap.text, '旧文字', '快照是拷贝，不随后续修改变化')
        orig = event_bus.enqueue
        event_bus.enqueue = fake
        try:
            asyncio.run(schedule_store.fire_now(s.id))
            with _FakeJobs():
                schedule_store.delete(s.id)
            with self.assertRaises(KeyError):
                asyncio.run(schedule_store.fire_now(s.id))
        finally:
            event_bus.enqueue = orig
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], 'scheduler:快照')
        self.assertIn('被改了', sent[0][1], '同一个快照里的名字和文字一致')

    def test_one_shot_survives_enqueue_failure(self):
        import event_bus
        calls = []

        async def flaky(source, text, payload=None):
            calls.append(source)
            if len(calls) == 1:
                raise RuntimeError('总线暂时不可用')

        with _FakeJobs():
            a = schedule_store.create(name='一次', text='x', kind='once',
                                      at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))
        job = {'name': 'sched:' + a.id, 'text': 'x', 'at': self._due_soon(a), 'source': 'scheduler:一次',
               'on_fire': schedule_store._on_fire(a.id), 'after_fire': schedule_store._after_fire(a.id)}

        async def go():
            orig, delay = event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S
            event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S = flaky, 0.05
            try:
                await scheduler._run_job(job)
            finally:
                event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S = orig, delay
        asyncio.run(asyncio.wait_for(go(), 8))
        self.assertEqual(len(calls), 2, '第一次失败后重试一次')
        s = schedule_store.find('一次')
        self.assertFalse(s.enabled)
        self.assertEqual(s.fire_count, 1, '只有成功的那次记账')
        self.assertEqual(schedule_store._fires_today(time.time()), 1, '失败那次的名额已退回')

    def test_one_shot_given_up_is_disabled(self):
        """重试用完还是发不出去：退回名额并停用落盘，不留“启用着但再也不会触发”的记录。"""
        import event_bus

        async def broken(source, text, payload=None):
            raise RuntimeError('down')

        with _FakeJobs():
            a = schedule_store.create(name='一次2', text='x', kind='once',
                                      at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))
        job = {'name': 'sched:' + a.id, 'text': 'x', 'at': self._due_soon(a), 'source': 'scheduler:一次2',
               'on_fire': schedule_store._on_fire(a.id), 'after_fire': schedule_store._after_fire(a.id)}
        tries = []

        async def go():
            orig = (event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S, scheduler.ONCE_MAX_RETRIES)
            event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S, scheduler.ONCE_MAX_RETRIES = broken, 0.02, 2
            try:
                await scheduler._run_job(job)
            finally:
                event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S, scheduler.ONCE_MAX_RETRIES = orig
            tries.append(job.get('retries'))
        asyncio.run(asyncio.wait_for(go(), 8))
        self.assertEqual(tries, [2])
        s = schedule_store.find('一次2')
        self.assertFalse(s.enabled)
        self.assertEqual(s.fire_count, 0)
        self.assertEqual(schedule_store._fires_today(time.time()), 0)
        schedule_store._schedules.clear()
        schedule_store.load_all()
        self.assertFalse(schedule_store.find('一次2').enabled, '停用已落盘')


class TestThirdReview(_Base):
    """PR #398 第三轮审查：每日名额原子占用、收紧设置要重新校验、一次性错过/放弃要停用；
    以及顺带堵上的：重启不丢当天次数、立即触发也占名额、按天跳的 _next_run 与逐分钟扫一致、
    精确的最短间隔、两年都不触发的 cron、模型工具不卡事件循环。"""

    def _in_hour(self, name):
        with _FakeJobs():
            return schedule_store.create(name=name, text='x', kind='once',
                                         at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))

    def _fill_today(self, n):
        schedule_store._fire_times.extend([time.time() - 1] * n)

    def test_concurrent_due_respects_cap(self):
        """差一次到上限时两条同时到点：只放行一条。"""
        import threading
        a, b = self._in_hour('a'), self._in_hour('b')
        self._fill_today(schedule_store.get_settings()['max_fires_per_day'] - 1)
        barrier, got = threading.Barrier(2), []

        def go(sid):
            cb = schedule_store._on_fire(sid)
            barrier.wait()
            got.append(cb({}))
        ts = [threading.Thread(target=go, args=(x.id,)) for x in (a, b)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(sorted(got), [False, True])
        self.assertEqual(schedule_store._fires_today(time.time()), 48)

    def test_concurrent_due_jobs_enqueue_once(self):
        """走完整的 scheduler._run_job：两条一次性同时到点、只剩一个名额 → 只入队一次；
        没轮上的那条当场停用，不留“启用但不会再触发”的记录。"""
        import event_bus
        a, b = self._in_hour('a'), self._in_hour('b')
        self._fill_today(47)
        sent = []

        async def fake(source, text, payload=None):
            await asyncio.sleep(0.01)
            sent.append(source)

        def job(s):
            return {'name': 'sched:' + s.id, 'text': 'x', 'at': self._due_soon(s),
                    'source': 'scheduler:' + s.name,
                    'on_fire': schedule_store._on_fire(s.id), 'after_fire': schedule_store._after_fire(s.id)}

        async def go():
            orig = event_bus.enqueue
            event_bus.enqueue = fake
            try:
                await asyncio.gather(scheduler._run_job(job(a)), scheduler._run_job(job(b)))
            finally:
                event_bus.enqueue = orig
        asyncio.run(asyncio.wait_for(go(), 5))
        self.assertEqual(len(sent), 1)
        self.assertEqual(schedule_store._fires_today(time.time()), 48)
        fa, fb = schedule_store.find('a'), schedule_store.find('b')
        self.assertFalse(fa.enabled or fb.enabled, '发了的和被上限跳过的一次性都停用')
        self.assertEqual(fa.fire_count + fb.fire_count, 1)

    def test_failed_enqueue_releases_slot(self):
        a = self._in_hour('a')
        check, done = schedule_store._on_fire(a.id), schedule_store._after_fire(a.id)
        job = {}
        self.assertTrue(check(job))
        self.assertEqual(schedule_store._fires_today(time.time()), 1)
        done(job, False, True)
        self.assertEqual(schedule_store._fires_today(time.time()), 0, '没发出去要退回名额')
        self.assertTrue(schedule_store.find('a').enabled, '还会重试，保持启用')

    def test_fire_count_survives_restart(self):
        a = self._in_hour('a')
        job = {}
        schedule_store._on_fire(a.id)(job)
        schedule_store._after_fire(a.id)(job, True, False)
        schedule_store._fire_times.clear()
        schedule_store.load_all()
        self.assertEqual(schedule_store._fires_today(time.time()), 1, '当天已触发的次数写在库里')

    def test_once_skipped_by_cap_is_disabled(self):
        a = self._in_hour('a')
        self._fill_today(48)
        self.assertFalse(schedule_store._on_fire(a.id)({}))
        self.assertFalse(schedule_store.find('a').enabled)
        schedule_store._schedules.clear()
        schedule_store.load_all()
        self.assertFalse(schedule_store.find('a').enabled, '停用已落盘')

    def test_tightening_settings_rechecks_existing(self):
        with _FakeJobs():
            schedule_store.create(name='十分钟', text='x', cron='0,10,20,30,40,50 8 * * *', skill='med')
            schedule_store.create(name='早上', text='x', cron='0 9 * * *', skill='med')
        for change, word in (({'min_interval_minutes': 30}, '十分钟'),
                             ({'max_schedules': 1}, '2 条'),
                             ({'max_per_skill': 1}, '技能 med'),
                             ({'max_fires_per_day': 5}, '7 次')):
            with self.assertRaisesRegex(ValueError, word, msg=str(change)):
                schedule_store.update_settings(**change)
        self.assertEqual(schedule_store.get_settings(), schedule_store.DEFAULT_SETTINGS, '被拒时设置不变')
        schedule_store.update_settings(min_interval_minutes=10, max_fires_per_day=7, model_tools_enabled=True)
        self.assertEqual(schedule_store.get_settings()['max_fires_per_day'], 7)

    def test_preexisting_violation_does_not_block_unrelated_change(self):
        with _FakeJobs():
            s = schedule_store.create(name='a', text='x', cron='0,10,20 8 * * *')
        schedule_store._schedules[s.id].cron = '*/5 8 * * *'      # 模拟旧版本留下的数据
        schedule_store.update_settings(model_tools_enabled=True)
        self.assertTrue(schedule_store.get_settings()['model_tools_enabled'])

    def test_fire_now_takes_a_slot(self):
        import event_bus
        sent = []

        async def ok(source, text, payload=None):
            sent.append(source)

        async def broken(source, text, payload=None):
            raise RuntimeError('down')

        with _FakeJobs():
            s = schedule_store.create(name='试', text='x', cron=self._far_cron())
            s2 = schedule_store.create(name='试2', text='x', cron=self._far_cron(30))
        orig = event_bus.enqueue
        try:
            event_bus.enqueue = broken
            with self.assertRaises(RuntimeError):
                asyncio.run(schedule_store.fire_now(s.id))
            self.assertEqual(schedule_store._fires_today(time.time()), 0, '没发出去退回名额')
            event_bus.enqueue = ok
            asyncio.run(schedule_store.fire_now(s.id))       # 失败的那次不算“刚触发过”
            self.assertEqual(schedule_store._fires_today(time.time()), 1)
            self.assertEqual(schedule_store.find('试').fire_count, 1)
            self.assertIsNotNone(schedule_store.find('试').last_run)
            self._fill_today(47)
            with self.assertRaisesRegex(ValueError, '每日上限'):
                asyncio.run(schedule_store.fire_now(s2.id))
        finally:
            event_bus.enqueue = orig
        self.assertEqual(len(sent), 1)

    def test_fire_now_respects_min_interval(self):
        """P1：连点“立即触发”不能在最短间隔里反复叫醒模型。"""
        import event_bus
        sent = []

        async def ok(source, text, payload=None):
            sent.append(source)

        with _FakeJobs():
            s = schedule_store.create(name='间隔', text='x', cron=self._far_cron())
        orig = event_bus.enqueue
        event_bus.enqueue = ok
        try:
            asyncio.run(schedule_store.fire_now(s.id))
            with self.assertRaisesRegex(ValueError, '刚触发过'):
                asyncio.run(schedule_store.fire_now(s.id))
            schedule_store._schedules[s.id].last_run = time.time() - 11 * 60
            schedule_store._last_fire.clear()
            asyncio.run(schedule_store.fire_now(s.id))      # 隔够了就能再触发
        finally:
            event_bus.enqueue = orig
        self.assertEqual(len(sent), 2)

    def test_concurrent_fire_now_only_one_passes(self):
        import event_bus
        sent = []

        async def slow(source, text, payload=None):
            await asyncio.sleep(0.05)
            sent.append(source)

        with _FakeJobs():
            s = schedule_store.create(name='并发', text='x', cron=self._far_cron())

        async def go():
            return await asyncio.gather(*(schedule_store.fire_now(s.id) for _ in range(5)),
                                        return_exceptions=True)
        orig = event_bus.enqueue
        event_bus.enqueue = slow
        try:
            res = asyncio.run(go())
        finally:
            event_bus.enqueue = orig
        self.assertEqual(len(sent), 1)
        self.assertEqual(sum(1 for r in res if not isinstance(r, Exception)), 1)
        self.assertTrue(all(isinstance(r, ValueError) for r in res if isinstance(r, Exception)))
        self.assertEqual(schedule_store._fires_today(time.time()), 1)

    def test_fire_now_close_to_next_scheduled_run_is_refused(self):
        soon = datetime.datetime.now(CN) + datetime.timedelta(minutes=4)
        with _FakeJobs():
            s = schedule_store.create(name='快到点', text='x', cron=f'{soon.minute} {soon.hour} * * *')
        with self.assertRaisesRegex(ValueError, '自己触发'):
            asyncio.run(schedule_store.fire_now(s.id))

    def test_fire_now_disabled_is_refused(self):
        with _FakeJobs():
            s = schedule_store.create(name='停着', text='x', cron=self._far_cron(), enabled=False)
        with self.assertRaisesRegex(ValueError, '已停用'):
            asyncio.run(schedule_store.fire_now(s.id))
        self.assertEqual(schedule_store._fires_today(time.time()), 0)

    def test_next_run_matches_minute_scan(self):
        """按天跳的 _next_run 和原来的逐分钟扫结果一致。"""
        def scan(cron, after, tz):
            tz = scheduler.get_tz(tz) if tz else None
            mi, ho, md, mo, wd = scheduler.parse_cron(cron)
            pw = {(w - 1) % 7 for w in wd}
            dt = datetime.datetime.fromtimestamp(after + 60, tz).replace(second=0, microsecond=0)
            for _ in range(60 * 24 * 400):
                if dt.month in mo and dt.day in md and dt.weekday() in pw and dt.hour in ho and dt.minute in mi:
                    return dt.timestamp()
                dt += datetime.timedelta(minutes=1)
        crons = ('0 8 * * *', '*/10 * * * *', '0,10,20,30 0,4,10 * * *', '30 23 * * 0',
                 '0 8 1 * *', '15 9 13 * 5', '0-30/10 8-10 * * 1-5', '59 23 31 12 *')
        afters = (_ts('2026-11-16 07:59'), _ts('2026-11-16 08:00') + 30, _ts('2026-12-31 23:58'),
                  _ts('2027-02-28 23:59'))
        for c in crons:
            for a in afters:
                for tz in ('Asia/Shanghai', 'UTC', None):
                    self.assertEqual(scheduler._next_run(c, a, tz), scan(c, a, tz), (c, a, tz))

    def test_rare_cron_is_fast(self):
        t0 = time.time()
        scheduler._next_run('0 8 29 2 *', time.time(), 'Asia/Shanghai')
        scheduler.next_runs('0 8 1 1 *', time.time(), 5, 'Asia/Shanghai')
        self.assertLess(time.time() - t0, 0.5)

    def test_min_gap_is_exact(self):
        now = _ts('2026-11-16 00:00')
        self.assertEqual(scheduler.min_gap_minutes('*/10 * * * *', now, 'Asia/Shanghai'), 10)
        self.assertEqual(scheduler.min_gap_minutes('0 8 * * *', now, 'Asia/Shanghai'), 1440)
        self.assertEqual(scheduler.min_gap_minutes('0 8 * * 1', now, 'Asia/Shanghai'), 7 * 1440)
        self.assertEqual(scheduler.min_gap_minutes('0 0,23 30,31 * *', now, 'Asia/Shanghai'), 60,
                         '月底 30 号 23:00 → 31 号 0:00 只隔 1 小时')
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'min_interval_minutes': 61}
        with self.assertRaisesRegex(ValueError, '最短只隔 60 分钟'):
            schedule_store.validate(schedule_store.Schedule(name='月底', text='x', cron='0 0,23 30,31 * *'),
                                    now=now)

    def test_impossible_cron_is_a_value_error(self):
        with _FakeJobs():
            with self.assertRaisesRegex(ValueError, '不会触发'):
                schedule_store.create(name='二月三十', text='x', cron='0 8 30 2 *')

    def test_monthly_dense_day_beyond_a_week_is_counted(self):
        """每月 1 号 48 次：就算 1 号不在接下来 7 天里，也要算进每日上限。"""
        with _FakeJobs():
            schedule_store.create(name='一号', text='x', cron='*/30 * 1 * *')
            with self.assertRaisesRegex(ValueError, '-01 这一天'):
                schedule_store.create(name='一号中午', text='x', cron='15 12 1 * *')

    def test_model_tools_run_store_calls_off_the_loop(self):
        src = (_SRC / 'event' / 'schedule.py').read_text(encoding='utf-8')
        self.assertIn('asyncio.to_thread(', src)
        for call in ('schedule_store.create(', 'schedule_store.update(', 'schedule_store.delete(',
                     'schedule_store.find('):
            self.assertNotIn(call, src, call + ' 应走 model_upsert / model_delete，放进线程池')


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@unittest.skipUnless(_has('fastapi') and _has('httpx'), '需要 fastapi + httpx（容器里有）')
class TestApiRoutes(_Base):
    """/api/schedules 接口契约：鉴权（与 start.py 一样挂 auth.auth_middleware）、参数校验的返回、
    停用 / 最短间隔 / 不存在时“立即触发”的语义、收紧设置被拒。"""

    TOKEN = 'test-token-398'

    def setUp(self):
        super().setUp()
        import fastapi
        from fastapi.testclient import TestClient
        import auth
        import event_bus
        spec = importlib.util.spec_from_file_location('api_schedules_under_test',
                                                      _SRC / 'api' / 'schedules.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        app = fastapi.FastAPI()
        app.include_router(mod.router, prefix='/api')
        app.middleware('http')(auth.auth_middleware)
        saved = (auth._token, auth._auth_enabled)
        auth._token, auth._auth_enabled = self.TOKEN, True
        self.addCleanup(lambda: (setattr(auth, '_token', saved[0]), setattr(auth, '_auth_enabled', saved[1])))
        fj = _FakeJobs().__enter__()
        self.addCleanup(fj.__exit__)
        self.sent = []

        async def fake(source, text, payload=None):
            self.sent.append(source)
        orig = event_bus.enqueue
        event_bus.enqueue = fake
        self.addCleanup(setattr, event_bus, 'enqueue', orig)
        self.c = TestClient(app)
        self.h = {'Authorization': f'Bearer {self.TOKEN}'}

    def _req(self, method, path, json=None, auth=True):
        return self.c.request(method, path, json=json, headers=self.h if auth else {})

    def _create(self, **kw):
        body = {'name': '吃药', 'text': '服药提醒定时检查', 'cron': self._far_cron(), **kw}
        r = self._req('POST', '/api/schedules', body)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()['ok'], r.json())
        return r.json()['schedule']['id']

    def test_requires_token(self):
        for method, path in (('GET', '/api/schedules'), ('POST', '/api/schedules'),
                             ('POST', '/api/schedules/x/run'), ('PATCH', '/api/schedules/settings'),
                             ('DELETE', '/api/schedules/x'), ('POST', '/api/schedules/preview')):
            self.assertEqual(self._req(method, path, {}, auth=False).status_code, 401, path)
        bad = self.c.get('/api/schedules', headers={'Authorization': 'Bearer nope'})
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(self._req('GET', '/api/schedules').status_code, 200)

    def test_create_list_and_validation_errors(self):
        sid = self._create()
        names = [s['name'] for s in self._req('GET', '/api/schedules').json()['schedules']]
        self.assertEqual(names, ['吃药'])
        for body, word in (({'name': '太密', 'text': 'x', 'cron': '*/5 * * * *'}, '最短'),
                           ({'name': '二月三十', 'text': 'x', 'cron': '0 8 30 2 *'}, '不会触发'),
                           ({'name': '吃药', 'text': 'x', 'cron': '0 9 * * *'}, '已经有'),
                           ({'name': '没文字', 'cron': '0 9 * * *'}, '文字')):
            r = self._req('POST', '/api/schedules', body)
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.json()['ok'], body)
            self.assertIn(word, r.json()['error'])
        r = self._req('POST', '/api/schedules', {'name': 'x', 'text': 'y', 'cron': '0 8 * * *',
                                                 'enabled': '不是布尔'})
        self.assertEqual(r.status_code, 422)
        self.assertIn('没有要改的', self._req('PATCH', f'/api/schedules/{sid}', {}).json()['error'])
        self.assertIn('不存在', self._req('PATCH', '/api/schedules/nope', {'text': 'x'}).json()['error'])
        self.assertFalse(self._req('POST', '/api/schedules/preview', {'cron': '61 * * * *'}).json()['ok'])
        self.assertEqual(len(self._req('POST', '/api/schedules/preview',
                                       {'cron': '0 8 * * *', 'count': 3}).json()['runs']), 3)

    def test_run_now_semantics(self):
        sid = self._create()
        r = self._req('POST', f'/api/schedules/{sid}/run').json()
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['schedule']['fire_count'], 1)
        r = self._req('POST', f'/api/schedules/{sid}/run').json()
        self.assertFalse(r['ok'])
        self.assertIn('刚触发过', r['error'])
        self.assertTrue(self._req('PATCH', f'/api/schedules/{sid}', {'enabled': False}).json()['ok'])
        r = self._req('POST', f'/api/schedules/{sid}/run').json()
        self.assertFalse(r['ok'])
        self.assertIn('已停用', r['error'])
        self.assertIn('不存在', self._req('POST', '/api/schedules/nope/run').json()['error'])
        self.assertEqual(len(self.sent), 1, '只有第一次真的叫醒了模型')
        self.assertTrue(self._req('DELETE', f'/api/schedules/{sid}').json()['ok'])
        self.assertIn('不存在', self._req('DELETE', f'/api/schedules/{sid}').json()['error'])

    def test_settings_validation(self):
        self._create(cron='0,10,20 8 * * *')
        r = self._req('PATCH', '/api/schedules/settings', {'min_interval_minutes': 30}).json()
        self.assertFalse(r['ok'])
        self.assertIn('吃药', r['error'])
        self.assertIn('正整数', self._req('PATCH', '/api/schedules/settings',
                                         {'max_schedules': 0}).json()['error'])
        self.assertIn('时区', self._req('PATCH', '/api/schedules/settings',
                                        {'default_tz': 'Mars/Olympus'}).json()['error'])
        self.assertEqual(self._req('GET', '/api/schedules/settings').json()['settings']['min_interval_minutes'], 10)
        r = self._req('PATCH', '/api/schedules/settings', {'model_tools_enabled': True}).json()
        self.assertTrue(r['ok'] and r['settings']['model_tools_enabled'])

    def test_preview_count_out_of_range_is_rejected(self):
        for n in (0, -1, 21):
            r = self._req('POST', '/api/schedules/preview', {'cron': '0 8 * * *', 'count': n}).json()
            self.assertFalse(r['ok'], n)
            self.assertIn('1 到 20', r['error'])
        r = self._req('POST', '/api/schedules/preview', {'cron': '0 8 * * *', 'count': 20}).json()
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['runs']), 20)


class TestFifthReview(_Base):
    """PR #398 第五轮：触发途中被取消也要收好尾、一次性结束后清掉调度登记、内存里的过期
    触发记录要清；以及顺带的：on_fire 出错时一次性会重试、长等待按挂钟分段、启动读库不卡循环。"""

    def _fill_today(self, n):
        schedule_store._fire_times.extend([time.time() - 1] * n)

    def _job(self, s, **kw):
        return {'name': 'sched:' + s.id, 'text': 'x', 'at': self._due_soon(s), 'source': 'scheduler:' + s.name,
                'on_fire': schedule_store._on_fire(s.id), 'after_fire': schedule_store._after_fire(s.id), **kw}

    def _once(self, name):
        with _FakeJobs():
            return schedule_store.create(name=name, text='x', kind='once',
                                         at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))

    def _cancel_mid_enqueue(self, enqueue_ok: bool):
        """job 在推事件途中被取消：这一次照样走完收尾。"""
        import event_bus
        s = self._once('取消' + str(enqueue_ok))
        job = self._job(s)
        started = []

        async def slow(source, text, payload=None):
            started.append(1)
            await asyncio.sleep(0.2)
            if not enqueue_ok:
                raise RuntimeError('down')

        async def go():
            orig = event_bus.enqueue
            event_bus.enqueue = slow
            try:
                task = asyncio.ensure_future(scheduler._run_job(job))
                while not started:
                    await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                for _ in range(100):           # 等留下的那一次收完尾
                    if not scheduler._settling:
                        break
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.05)
            finally:
                event_bus.enqueue = orig
        asyncio.run(asyncio.wait_for(go(), 10))
        self.assertEqual(scheduler._settling, set())
        return schedule_store.find(s.name)

    def test_cancel_after_reserve_settles_delivered(self):
        s = self._cancel_mid_enqueue(True)
        self.assertEqual(s.fire_count, 1, '发出去了就记账')
        self.assertFalse(s.enabled, '一次性发出去了就停用')
        self.assertEqual(schedule_store._fires_today(time.time()), 1)

    def test_cancel_after_reserve_settles_failed(self):
        s = self._cancel_mid_enqueue(False)
        self.assertEqual(schedule_store._fires_today(time.time()), 0, '没发出去退回名额')
        self.assertNotIn(s.id, schedule_store._last_fire, '“正在触发”的标记要撤掉')
        self.assertFalse(s.enabled, '被取消的 job 不再重试，一次性停用，不留启用但没有协程的记录')
        self.assertEqual(s.fire_count, 0)

    def test_cap_skipped_one_shot_clears_registry(self):
        s = self._once('满额')
        self._fill_today(48)

        async def go():
            scheduler._loop = asyncio.get_running_loop()
            live = schedule_store._schedules[s.id]
            self._due_soon(live)
            schedule_store._register(live)
            name = schedule_store.job_name(s.id)
            await asyncio.sleep(0)
            self.assertIsNotNone(scheduler.job_next_ts(name))
            for _ in range(200):
                if name not in scheduler._dynamic_tasks:
                    break
                await asyncio.sleep(0.02)
            self.assertNotIn(name, scheduler._dynamic_tasks)
            self.assertNotIn(name, scheduler._job_info)
            self.assertIsNone(scheduler.job_next_ts(name))
        try:
            asyncio.run(asyncio.wait_for(go(), 10))
        finally:
            scheduler._loop = None
        self.assertFalse(schedule_store.find('满额').enabled)

    def test_finished_job_does_not_remove_its_replacement(self):
        async def go():
            scheduler._loop = asyncio.get_running_loop()
            scheduler.add_job('swap', '', 'x', at=time.time() + 0.05)
            await asyncio.sleep(0)
            old = scheduler._dynamic_tasks['swap']
            scheduler.add_job('swap', '0 8 * * *', 'y')      # 换成新 job，旧的被取消
            await asyncio.sleep(0.2)
            self.assertIn('swap', scheduler._dynamic_tasks)
            self.assertIsNot(scheduler._dynamic_tasks['swap'], old)
            self.assertIsNotNone(scheduler.job_next_ts('swap'))
            scheduler.remove_job('swap')
            await asyncio.sleep(0)
        try:
            asyncio.run(asyncio.wait_for(go(), 5))
        finally:
            scheduler._loop = None

    def test_expired_fire_times_are_pruned_on_check(self):
        schedule_store._fire_times.extend([time.time() - 3 * 86400] * 5)
        schedule_store._reserve(time.time())
        self.assertEqual(len(schedule_store._fire_times), 1)

    def test_on_fire_error_retries_one_shot(self):
        import event_bus
        calls, sent = [], []

        def flaky_check(job):
            calls.append('check')
            if len(calls) == 1:
                raise RuntimeError('db locked')
            return True

        def done(job, delivered, retrying):
            calls.append((delivered, retrying))

        async def ok(source, text, payload=None):
            sent.append(source)

        async def go():
            orig, delay = event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S
            event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S = ok, 0.02
            try:
                await scheduler._run_job({'name': 'r', 'text': 'x', 'at': time.time() + 0.02,
                                          'on_fire': flaky_check, 'after_fire': done})
            finally:
                event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S = orig, delay
        asyncio.run(asyncio.wait_for(go(), 5))
        self.assertEqual(calls, ['check', (False, True), 'check', (True, False)])
        self.assertEqual(len(sent), 1)

    def test_long_waits_are_chunked(self):
        src = (_SRC / 'scheduler.py').read_text(encoding='utf-8')
        self.assertIn('min(wait, _SLEEP_CHUNK_S)', src)
        self.assertIn('asyncio.to_thread(schedule_store.start_all)', src)


class TestLifecycleExtras(_Base):
    """自查补的：被新注册替换的旧 job 不再发事件、记账写库失败时内存状态仍正确、
    和配置文件里的定时同名会被拒。"""

    def test_stale_job_does_not_fire(self):
        with _FakeJobs():
            s = schedule_store.create(name='改时间', text='x', cron='0 8 * * *')
            old_job = {'name': 'sched:' + s.id, 'cron': '0 8 * * *', 'tz': 'Asia/Shanghai', 'at': None}
            self.assertTrue(schedule_store._on_fire(s.id)(dict(old_job)))
            schedule_store._fire_times.clear(); schedule_store._last_fire.clear()
            schedule_store.update(s.id, cron='0 9 * * *')
        self.assertFalse(schedule_store._on_fire(s.id)(dict(old_job)), '旧 cron 的 job 不能再发')
        self.assertEqual(schedule_store._fires_today(time.time()), 0)
        new_job = dict(old_job, cron='0 9 * * *')
        self.assertTrue(schedule_store._on_fire(s.id)(new_job))
        self.assertFalse(schedule_store._on_fire(s.id)(dict(new_job, cancelled=True)))

    def test_bookkeeping_survives_db_error(self):
        with _FakeJobs():
            a = schedule_store.create(name='写库失败', text='x', kind='once',
                                      at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))
        job = {}
        self.assertTrue(schedule_store._on_fire(a.id)(job))
        orig = schedule_store._commit

        def boom(slot):
            raise sqlite3.OperationalError('database is locked')
        schedule_store._commit = boom
        try:
            schedule_store._after_fire(a.id)(job, True, False)
        finally:
            schedule_store._commit = orig
        s = schedule_store.find('写库失败')
        self.assertFalse(s.enabled)
        self.assertEqual(s.fire_count, 1)

    def test_name_clash_with_config_job(self):
        saved = config.main.get('scheduler', [])
        config.main['scheduler'] = [{'name': 'hourly_check', 'cron': '0 * * * *', 'text': 'x'}]
        try:
            with _FakeJobs():
                with self.assertRaisesRegex(ValueError, '配置文件'):
                    schedule_store.create(name='hourly_check', text='x', cron='0 8 * * *')
        finally:
            config.main['scheduler'] = saved


class TestSixthReview(_Base):
    """PR #398 第六轮：名额先落库再放行、写库失败的各条路径、调度循环起来之前建的定时、
    改设置时遇到坏数据不再当成没问题、删除不存在的返回原因。"""

    def _once(self, name):
        with _FakeJobs():
            return schedule_store.create(name=name, text='x', kind='once',
                                         at=schedule_store.fmt_ts(time.time() + 3600, 'Asia/Shanghai'))

    def test_slot_is_durable_before_enqueue(self):
        a = self._once('落库')
        self.assertTrue(schedule_store._on_fire(a.id)({}))
        schedule_store._fire_times.clear()          # 模拟：放行后、记账前进程重启
        schedule_store.load_all()
        self.assertEqual(schedule_store._fires_today(time.time()), 1, '放行之前名额已经在库里')

    def test_released_slot_is_removed_from_db(self):
        a = self._once('退回')
        job = {}
        schedule_store._on_fire(a.id)(job)
        schedule_store._after_fire(a.id)(job, False, True)
        schedule_store._fire_times.clear()
        schedule_store.load_all()
        self.assertEqual(schedule_store._fires_today(time.time()), 0)

    def _broken_db(self):
        orig = schedule_store._get_conn

        def boom():
            raise sqlite3.OperationalError('database is locked')
        schedule_store._get_conn = boom
        self.addCleanup(setattr, schedule_store, '_get_conn', orig)

    def test_db_failure_on_reserve_blocks_the_fire(self):
        import event_bus
        a = self._once('库忙')
        sent = []

        async def ok(source, text, payload=None):
            sent.append(source)
        self._broken_db()
        with self.assertRaises(sqlite3.Error):
            schedule_store._on_fire(a.id)({})
        self.assertEqual(schedule_store._fire_times, [])

        async def go():
            orig = (event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S, scheduler.ONCE_MAX_RETRIES)
            event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S, scheduler.ONCE_MAX_RETRIES = ok, 0.02, 1
            try:
                await scheduler._run_job({'name': 'sched:' + a.id, 'text': 'x', 'at': time.time() + 0.02,
                                          'on_fire': schedule_store._on_fire(a.id),
                                          'after_fire': schedule_store._after_fire(a.id)})
            finally:
                event_bus.enqueue, scheduler.ONCE_RETRY_DELAY_S, scheduler.ONCE_MAX_RETRIES = orig
        asyncio.run(asyncio.wait_for(go(), 5))
        self.assertEqual(sent, [], '记不下名额就不发')

    def test_db_failure_on_manual_fire_is_a_clear_error(self):
        with _FakeJobs():
            s = schedule_store.create(name='手动库忙', text='x', cron=self._far_cron())
        self._broken_db()
        with self.assertRaisesRegex(ValueError, '没有触发'):
            asyncio.run(schedule_store.fire_now(s.id))

    def test_failed_save_leaves_nothing_behind(self):
        with _FakeJobs() as fj:
            s = schedule_store.create(name='原样', text='旧', cron='0 8 * * *')
            self._broken_db()
            with self.assertRaises(sqlite3.Error):
                schedule_store.create(name='新建', text='x', cron='0 9 * * *')
            with self.assertRaises(sqlite3.Error):
                schedule_store.update(s.id, text='新')
            with self.assertRaises(sqlite3.Error):
                schedule_store.delete(s.id)
        self.assertIsNone(schedule_store.find('新建'))
        self.assertEqual(schedule_store.find('原样').text, '旧')
        self.assertEqual(len(fj.jobs), 1, '只有原来那条注册着')

    def test_settings_change_fails_closed_on_broken_row(self):
        bad = schedule_store.Schedule(name='坏时间', text='x', kind='once', at='不是时间')
        schedule_store._schedules[bad.id] = bad
        with self.assertRaises(ValueError):
            schedule_store.update_settings(max_schedules=5)
        self.assertEqual(schedule_store.get_settings()['max_schedules'], 20)

    def test_create_before_scheduler_run_is_registered_on_start(self):
        """API 先起来、调度循环还没记下时建的定时：不报错，run() 起来后补上注册。"""
        scheduler._loop = None
        scheduler._pending.clear()
        s = schedule_store.create(name='早建', text='x', cron='0 8 * * *')   # 真的 add_job
        name = schedule_store.job_name(s.id)
        self.assertNotIn(name, scheduler._dynamic_tasks)

        async def go():
            saved = config.main.get('scheduler', [])
            config.main['scheduler'] = []
            owner = asyncio.create_task(scheduler.run())
            try:
                await asyncio.sleep(0.05)
                self.assertIn(name, scheduler._dynamic_tasks)
                self.assertIsNotNone(scheduler.job_next_ts(name))
                scheduler.remove_job(name)
                await asyncio.sleep(0)
            finally:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
                config.main['scheduler'] = saved
        try:
            asyncio.run(asyncio.wait_for(go(), 5))
        finally:
            scheduler._loop = None
            scheduler._pending.clear()
        self.assertEqual(scheduler._pending, [])


class TestSeventhReview(_Base):
    """PR #398 第七轮：立即触发的请求被取消时照样收尾；设置确实写进 SQLite（config.main 是 ConfigDB）。"""

    def _cancel_manual(self, enqueue_ok: bool):
        import event_bus
        with _FakeJobs():
            s = schedule_store.create(name='手动取消' + str(enqueue_ok), text='x', cron=self._far_cron())
        started = []

        async def slow(source, text, payload=None):
            started.append(1)
            await asyncio.sleep(0.2)
            if not enqueue_ok:
                raise RuntimeError('down')

        async def go():
            orig = event_bus.enqueue
            event_bus.enqueue = slow
            try:
                req = asyncio.ensure_future(schedule_store.fire_now(s.id))
                while not started:
                    await asyncio.sleep(0.01)
                req.cancel()                      # 模拟 HTTP 请求被取消
                with self.assertRaises(asyncio.CancelledError):
                    await req
                for _ in range(100):
                    if not scheduler._settling:
                        break
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.05)
            finally:
                event_bus.enqueue = orig
        asyncio.run(asyncio.wait_for(go(), 10))
        self.assertEqual(scheduler._settling, set())
        return schedule_store.find(s.name)

    def test_cancelled_manual_fire_settles_delivered(self):
        s = self._cancel_manual(True)
        self.assertEqual(s.fire_count, 1)
        self.assertEqual(schedule_store._fires_today(time.time()), 1)

    def test_cancelled_manual_fire_settles_failed(self):
        s = self._cancel_manual(False)
        self.assertEqual(s.fire_count, 0)
        self.assertEqual(schedule_store._fires_today(time.time()), 0, '没发出去：名额退回')
        self.assertNotIn(s.id, schedule_store._last_fire, '没发出去：不挡下一次手动触发')
        schedule_store._fire_times.clear()
        schedule_store.load_all()
        self.assertEqual(schedule_store._fires_today(time.time()), 0, '库里的名额也删了')

    def test_settings_are_written_to_sqlite(self):
        self.assertIsInstance(config.main, config.ConfigDB)
        schedule_store.update_settings(max_fires_per_day=30, model_tools_enabled=True)
        with sqlite3.connect(str(config.DB_PATH)) as conn:
            row = conn.execute("SELECT value FROM config WHERE key = 'scheduler_settings'").fetchone()
        import json
        saved = json.loads(row[0])
        self.assertEqual(saved['max_fires_per_day'], 30)
        self.assertTrue(saved['model_tools_enabled'])
        self.assertEqual(config.ConfigDB()['scheduler_settings']['max_fires_per_day'], 30,
                         '新的 ConfigDB 实例（相当于重启后）读到的也是新值')


class TestEighthReview(_Base):
    """PR #398 第八轮：启动恢复逐行隔离（坏 JSON、写坏的 cron / 时区都只停用它自己），
    测试不依赖全局状态的先后，start.py 的接线。"""

    def _raw_row(self, row_id, data):
        with schedule_store._get_conn() as conn:
            conn.execute('INSERT OR REPLACE INTO schedules (id, data) VALUES (?, ?)', (row_id, data))
            conn.commit()

    def test_malformed_rows_do_not_block_startup(self):
        with _FakeJobs() as fj:
            good = schedule_store.create(name='好的', text='x', cron='0 8 * * *')
            self._raw_row('badjson', '{不是 JSON')
            self._raw_row('badlist', '[1, 2]')
            self._raw_row('badfield', json.dumps({'id': '和行号对不上', 'name': 'x'}))
            schedule_store._schedules.clear(); fj.jobs.clear()
            self.assertEqual(schedule_store.start_all(), 1)
            self.assertEqual(list(fj.jobs), [f'sched:{good.id}'])
        for rid in ('badjson', 'badlist', 'badfield'):
            ph = schedule_store._schedules[rid]
            self.assertFalse(ph.enabled)
            self.assertIn('数据损坏', ph.name)
        schedule_store._schedules.clear()
        schedule_store.load_all()
        self.assertFalse(schedule_store._schedules['badjson'].enabled, '占位记录已落盘')
        self.assertIn('不是 JSON', schedule_store._schedules['badjson'].text, '原始内容留着排查')
        with _FakeJobs():
            self.assertTrue(schedule_store.delete('badjson'), '占位记录能删')

    def test_invalid_cron_or_tz_is_disabled_before_registration(self):
        with _FakeJobs() as fj:
            good = schedule_store.create(name='好的', text='x', cron='0 8 * * *')
            for bad in (schedule_store.Schedule(name='坏cron', text='x', cron='61 8 * * *'),
                        schedule_store.Schedule(name='坏时区', text='x', cron='0 8 * * *', tz='Mars/Olympus'),
                        schedule_store.Schedule(name='不会触发', text='x', cron='0 8 30 2 *'),
                        schedule_store.Schedule(name='坏类型', text='x', kind='weekly', cron='0 8 * * *')):
                schedule_store._persist(bad)
            schedule_store._schedules.clear(); fj.jobs.clear()
            self.assertEqual(schedule_store.start_all(), 1, '坏的不算启动了')
            self.assertEqual(list(fj.jobs), [f'sched:{good.id}'])
        schedule_store._schedules.clear()
        schedule_store.load_all()
        for name in ('坏cron', '坏时区', '不会触发', '坏类型'):
            self.assertFalse(schedule_store.find(name).enabled, name + ' 停用已落盘')
        schedule_store.update_settings(max_schedules=5)      # 停用后不再挡改设置

    def test_tests_survive_config_main_replaced_by_dict(self):
        """模拟 test_peer_name_sync.py：把 config.main 换成 dict 不还原，下一个用例照样隔离。"""
        saved = config.main
        config.main = {'peer_settings': {'display_name': 'TestBot'}}
        try:
            other = TestSeventhReview('test_settings_are_written_to_sqlite')
            result = unittest.TestResult()
            other.run(result)
            self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
            self.assertEqual(config.main, {'peer_settings': {'display_name': 'TestBot'}}, '跑完还原成原来的对象')
        finally:
            config.main = saved

    def test_start_py_wiring(self):
        """start.py 导入时会拉起整个服务，单测里不导入它，这里核对接线：定时路由挂在 app_api 上，
        app_api 挂在 /api，auth 中间件装在外层 app 上（对所有 /api/* 生效）。
        路由本身 + 同一个中间件的行为见 TestApiRoutes。"""
        src = (_SRC / 'start.py').read_text(encoding='utf-8')
        self.assertIn('app_api.include_router(api.schedules.router)', src)
        self.assertIn("app.mount('/api', app_api)", src)
        self.assertIn("app.middleware('http')(auth.auth_middleware)", src)



class _FakeClock:
    """挂钟 = 起点 + 真实经过的时间 + offset；单调时钟用真的。改 offset 就是“校时”。"""

    def __init__(self, start: float):
        self.base, self.m0, self.offset = start, time.monotonic(), 0.0

    def time(self):
        return self.base + (time.monotonic() - self.m0) + self.offset

    def __getattr__(self, k):
        return getattr(time, k)


def _utc(s: str) -> float:
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()


class TestNinthReview(_Base):
    """PR #398 第九轮：挂钟跳变后 cron 按新时钟重算（不补发过期的、不重复已发的），
    设置写库读回核对，last_run 用占名额的时刻，preview 的 count 越界明确报错。"""

    def _run_with_clock(self, job, clock, jumps):
        """跑 job，按 jumps=[(真实秒数, 挂钟跳多少秒)] 依次改挂钟，最后返回 job['next_ts']。"""
        async def go():
            task = asyncio.ensure_future(scheduler._run_job(job))
            for after, jump in jumps:
                await asyncio.sleep(after)
                clock.offset += jump
            await asyncio.sleep(0.3)
            nxt = job.get('next_ts')
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return nxt
        saved = (scheduler.time, scheduler._SLEEP_CHUNK_S)
        scheduler.time, scheduler._SLEEP_CHUNK_S = clock, 0.02
        try:
            return asyncio.run(asyncio.wait_for(go(), 10))
        finally:
            scheduler.time, scheduler._SLEEP_CHUNK_S = saved

    def _job(self, fired, **kw):
        return {'name': 'clock-jump', 'text': 't', 'tz': 'UTC',
                'on_fire': lambda j: (fired.append(time.time()), False)[1], **kw}   # 记下触发、不真推事件

    def test_forward_jump_over_cron_time_skips_stale_fire(self):
        fired = []
        nxt = self._run_with_clock(self._job(fired, cron='0 8 * * *'),
                                   _FakeClock(_utc('2026-11-17 07:59:00')), [(0.2, 120)])
        self.assertEqual(fired, [], '挂钟跳过了 08:00，不按旧时刻补发')
        self.assertEqual(nxt, _utc('2026-11-18 08:00:00'))

    def test_backward_jump_recomputes_earlier_occurrence(self):
        fired = []
        nxt = self._run_with_clock(self._job(fired, cron='0 8 * * *'),
                                   _FakeClock(_utc('2026-11-17 09:00:00')), [(0.2, -5400)])
        self.assertEqual(fired, [])
        self.assertEqual(nxt, _utc('2026-11-17 08:00:00'), '挂钟拨回 07:30，今天 08:00 还没到')

    def test_backward_jump_does_not_repeat_fired_occurrence(self):
        fired = []
        nxt = self._run_with_clock(self._job(fired, cron='0 8 * * *'),
                                   _FakeClock(_utc('2026-11-17 07:59:59') + 0.7),
                                   [(0.8, -300)])
        self.assertEqual(len(fired), 1, '08:00 发过一次')
        self.assertEqual(nxt, _utc('2026-11-18 08:00:00'), '拨回 07:55 也不再发第二次 08:00')

    def test_forward_jump_over_one_shot_fires_late_once(self):
        fired = []
        start = _utc('2026-11-17 07:59:00')
        self._run_with_clock(self._job(fired, at=start + 60), _FakeClock(start), [(0.2, 120)])
        self.assertEqual(len(fired), 1, '一次性的宁晚不漏')

    def test_settings_write_is_read_back(self):
        class Lossy(dict):
            def __setitem__(self, k, v):
                pass                     # 模拟写库静默没生效
        saved = config.main
        config.main = Lossy(scheduler_settings=dict(schedule_store.DEFAULT_SETTINGS))
        try:
            with self.assertRaises(sqlite3.Error):
                schedule_store.update_settings(max_fires_per_day=7)
            self.assertEqual(schedule_store.get_settings()['max_fires_per_day'],
                             schedule_store.DEFAULT_SETTINGS['max_fires_per_day'])
        finally:
            config.main = saved
        schedule_store.update_settings(max_fires_per_day=7)
        self.assertEqual(config.ConfigDB()['scheduler_settings']['max_fires_per_day'], 7, '新连接读得到')

    def test_last_run_is_the_reserved_slot(self):
        with _FakeJobs():
            s = schedule_store.create(name='慢', text='x', cron=self._far_cron(5))
        clock = _FakeClock(time.time())
        saved = schedule_store.time
        schedule_store.time = clock
        try:
            job = {}
            self.assertTrue(schedule_store._on_fire(s.id)(job))
            slot = job['slot']
            clock.offset += 50           # 推事件花了 50 秒
            schedule_store._after_fire(s.id)(job, True, False)
        finally:
            schedule_store.time = saved
        got = schedule_store._schedules[s.id]
        self.assertEqual(got.last_run, slot)
        self.assertEqual(got.fire_count, 1)



class TestTenthReview(_Base):
    """PR #398 第十轮：定时的名字 / 文字 / 技能里的控制字符（NUL、ESC 等）不能进事件和日志。"""

    BAD = ('a\x00b', 'a\x1b[31mb', 'a\x7fb', 'a\x85b', 'a\u2028b')

    def test_control_chars_are_rejected_on_create(self):
        with _FakeJobs():
            for bad in self.BAD:
                with self.assertRaisesRegex(ValueError, '控制字符'):
                    schedule_store.create(name='控制', text=bad, cron=self._far_cron(1))
                with self.assertRaisesRegex(ValueError, '控制字符'):
                    schedule_store.create(name=bad, text='x', cron=self._far_cron(1))
                with self.assertRaisesRegex(ValueError, '控制字符'):
                    schedule_store.create(name='控制', text='x', skill=bad, cron=self._far_cron(1))
            with self.assertRaisesRegex(ValueError, '换行'):
                schedule_store.create(name='两\n行', text='x', cron=self._far_cron(1))
            s = schedule_store.create(name='多行文字', text='第一行\n第二行\t缩进', cron=self._far_cron(1))
        self.assertIn('第一行\n第二行\t缩进', schedule_store.event_text(s), '文字里的换行和制表符照常保留')
        self.assertEqual(schedule_store.list_all(), [s], '被拒的一条都没存')

    def test_event_text_and_source_are_cleaned_for_stored_rows(self):
        """库里的旧数据或绕过校验写进来的：推事件前再清一遍。"""
        s = schedule_store.Schedule(id='x1', name='夜\x00间\x1b[2J', text='吃药\x00\x1b]0;pwn\x07' + 'y' * 600,
                                    skill='med\x00check', kind='cron', cron='0 8 * * *')
        text, source = schedule_store.event_text(s), schedule_store.event_source(s)
        for out in (text, source):
            self.assertIsNone(schedule_store._CTRL_RE.search(out), repr(out[:80]))
        self.assertNotIn('\x00', text + source)
        self.assertLessEqual(len(text.split('\n')[0]), 500, '文字截到 500 字')
        self.assertTrue(source.startswith('scheduler:夜 间'))

    def test_fire_now_never_enqueues_control_chars(self):
        import event_bus
        sent = []

        async def fake(source, text, payload=None):
            sent.append((source, text))
        with _FakeJobs():
            s = schedule_store.create(name='手动', text='ok', cron=self._far_cron(3))
        schedule_store._schedules[s.id].text = '坏\x00数据\x1b[31m'
        orig = event_bus.enqueue
        event_bus.enqueue = fake
        try:
            asyncio.run(schedule_store.fire_now(s.id))
        finally:
            event_bus.enqueue = orig
        self.assertEqual(len(sent), 1)
        self.assertNotIn('\x00', sent[0][1])
        self.assertNotIn('\x1b', sent[0][1])
        self.assertEqual(sent[0][0], 'scheduler:手动')

    def test_api_rejects_nul_text(self):
        try:
            import fastapi  # noqa: F401
            import httpx  # noqa: F401
        except ImportError:
            self.skipTest('需要 fastapi + httpx')
        api = TestApiRoutes('test_preview_count_out_of_range_is_rejected')
        api.setUp()
        try:
            r = api._req('POST', '/api/schedules', {'name': 'nul', 'text': 'a\u0000b', 'cron': self._far_cron(2)}).json()
            self.assertFalse(r['ok'])
            self.assertIn('控制字符', r['error'])
        finally:
            api.doCleanups()



class TestEleventhReview(_Base):
    """PR #398 第十一轮：模型那边每个 turn 的设置读取不碰 SQLite；工具说明和实际用的
    默认时区一致（管理员改了默认时区也一样）。"""

    def setUp(self):
        super().setUp()
        self.mod = _load_tools_module()
        self.tools = self.mod.Tools()

    def test_cached_settings_never_touches_the_db(self):
        class NoDb:
            def get(self, *a, **k):
                raise AssertionError('事件循环上读了数据库')
            __getitem__ = __setitem__ = __contains__ = get
        self.assertFalse(schedule_store.cached_settings()['model_tools_enabled'], '没读过库时按默认（关）')
        schedule_store.update_settings(model_tools_enabled=True, default_tz='Europe/London')
        saved = config.main
        config.main = NoDb()
        try:
            st = schedule_store.cached_settings()
        finally:
            config.main = saved
        self.assertTrue(st['model_tools_enabled'], '改设置后快照立即更新')
        self.assertEqual(st['default_tz'], 'Europe/London')

    def test_start_all_primes_the_snapshot(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'model_tools_enabled': True}
        schedule_store._settings_snapshot = None
        schedule_store.start_all()
        self.assertTrue(schedule_store.cached_settings()['model_tools_enabled'])

    def test_llm_gate_uses_only_the_snapshot(self):
        """event.llm 导入要整套依赖，这里按语法树检查：每个 turn 调的两个函数里不出现 get_settings
        / config 读取，只用 cached_settings；工具表上写默认时区。"""
        import ast
        src = (_SRC / 'event' / 'llm.py').read_text(encoding='utf-8')
        funcs = {n.name: n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
        for name in ('_schedule_settings', '_hide_schedule_tools'):
            body = ast.unparse(funcs[name])
            self.assertNotIn('get_settings', body, name)
            self.assertNotIn('config.', body, name)
        self.assertIn('cached_settings()', ast.unparse(funcs['_schedule_settings']))
        self.assertIn('event.schedule.with_default_tz(t, _sched_tz)', src)

    def test_tool_description_shows_the_configured_tz(self):
        tool = {'object': None, 'schema': {
            'name': 'schedule_create', 'description': '建一个定时任务',
            'parameters': {'type': 'object', 'properties': {'cron': {'type': 'string', 'description': 'cron'},
                                                           'at': {'type': 'string', 'description': 'at'}}}}}
        out = self.mod.with_default_tz(tool, 'America/New_York')
        self.assertIn('America/New_York', out['schema']['description'])
        self.assertIn('America/New_York', out['schema']['parameters']['properties']['cron']['description'])
        self.assertIn('America/New_York', out['schema']['parameters']['properties']['at']['description'])
        self.assertNotIn('America/New_York', tool['schema']['description'], '不改注册表里的原件')
        other = {'schema': {'name': 'schedule_list', 'description': 'x'}}
        self.assertIs(self.mod.with_default_tz(other, 'UTC'), other)
        src = (_SRC / 'event' / 'schedule.py').read_text(encoding='utf-8')
        self.assertNotIn('北京时间', src, '静态说明里不再写死北京时间')

    def test_model_create_uses_the_configured_default_tz(self):
        schedule_store.update_settings(model_tools_enabled=True, default_tz='America/New_York')
        with _FakeJobs():
            r = asyncio.run(self.tools.schedule_create(name='纽约早上', text='x', cron=self._far_cron(7)))
        self.assertIn('已创建', r)
        self.assertIn('America/New_York', r, '返回的描述里写明时区')
        self.assertEqual(schedule_store.find('纽约早上').tz, 'America/New_York')



class TestSchedulerShutdown(_Base):
    """调度器的常驻任务负责动态 job 的退出；旧名称的日志保持转义。"""

    def test_run_owns_dynamic_jobs_without_legacy_config(self):
        from unittest.mock import patch

        async def go():
            ready = asyncio.Event()
            delivered = []

            def start_all():
                scheduler.add_job('shutdown-test', '', 'x', at=time.time() + 0.1)
                asyncio_loop.call_soon_threadsafe(ready.set)

            async def enqueue(**event):
                delivered.append(event)

            asyncio_loop = asyncio.get_running_loop()
            with patch.object(config, 'main', {'scheduler': []}), \
                    patch.object(schedule_store, 'start_all', start_all), \
                    patch.object(scheduler.event_bus, 'enqueue', enqueue):
                owner = asyncio.create_task(scheduler.run())
                try:
                    await asyncio.wait_for(ready.wait(), 2)
                    await asyncio.sleep(0.01)
                    self.assertFalse(owner.done(), '没有旧配置定时也必须保持常驻')
                    job = scheduler._dynamic_tasks['shutdown-test']
                    owner.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await owner
                    self.assertTrue(job.done())
                    self.assertIsNone(scheduler._loop)
                    self.assertEqual(scheduler._dynamic_tasks, {})
                    self.assertEqual(scheduler._job_info, {})
                    await asyncio.sleep(0.12)
                    self.assertEqual(delivered, [])
                finally:
                    owner.cancel()
                    await asyncio.gather(owner, return_exceptions=True)
                    for task in list(scheduler._dynamic_tasks.values()):
                        task.cancel()
                    await asyncio.gather(*scheduler._dynamic_tasks.values(), return_exceptions=True)

        asyncio.run(asyncio.wait_for(go(), 5))

    def test_legacy_name_control_characters_are_escaped_in_logs(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        s = schedule_store.Schedule(name='old\x00\x1b[31m\nname')
        output = StringIO()
        with patch.object(schedule_store, '_persist', side_effect=sqlite3.Error('write failed')), \
                redirect_stdout(output):
            schedule_store._disable_once(s, 'skipped by the daily cap')
        logged = output.getvalue()
        self.assertIn(repr(s.name), logged)
        self.assertNotIn('\x00', logged)
        self.assertNotIn('\x1b', logged)
        self.assertEqual(len(logged.splitlines()), 2, '名称换行不能伪造新日志行')

    def test_shutdown_waits_for_startup_thread_and_ignores_late_registration(self):
        import threading
        from unittest.mock import patch

        async def go():
            started = asyncio.Event()
            finish_startup = threading.Event()
            loop = asyncio.get_running_loop()

            def start_all():
                loop.call_soon_threadsafe(started.set)
                if not finish_startup.wait(2):
                    raise AssertionError('测试未释放启动线程')
                scheduler.add_job('late-startup', '', 'x', at=time.time() + 60)

            with patch.object(config, 'main', {'scheduler': []}), \
                    patch.object(schedule_store, 'start_all', start_all):
                owner = asyncio.create_task(scheduler.run())
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    owner.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done(), '不能遗留仍会注册任务的启动线程')
                    finish_startup.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await owner
                    self.assertIsNone(scheduler._loop)
                    self.assertEqual(scheduler._dynamic_tasks, {})
                    self.assertEqual(scheduler._pending, [])
                finally:
                    finish_startup.set()
                    await asyncio.gather(owner, return_exceptions=True)

        asyncio.run(asyncio.wait_for(go(), 5))

    def test_shutdown_settles_an_admitted_fire_before_returning(self):
        from unittest.mock import patch

        async def go():
            enqueuing = asyncio.Event()
            release = asyncio.Event()
            settled = []

            def start_all():
                scheduler.add_job('in-flight', '', 'x', at=time.time() - 1,
                                  after_fire=lambda job, delivered, retrying: settled.append(delivered))

            async def enqueue(**event):
                enqueuing.set()
                await release.wait()

            with patch.object(config, 'main', {'scheduler': []}), \
                    patch.object(schedule_store, 'start_all', start_all), \
                    patch.object(scheduler.event_bus, 'enqueue', enqueue):
                owner = asyncio.create_task(scheduler.run())
                try:
                    await asyncio.wait_for(enqueuing.wait(), 2)
                    owner.cancel()
                    while scheduler._loop is not None:
                        await asyncio.sleep(0)
                    scheduler.add_job('late-api', '', 'x', at=time.time() - 1)
                    self.assertFalse(owner.done(), '已准入的触发必须先结算')
                    self.assertNotIn('late-api', scheduler._dynamic_tasks)
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await owner
                    self.assertEqual(settled, [True])
                    self.assertEqual(scheduler._dynamic_tasks, {})
                    self.assertFalse(scheduler._settling)
                finally:
                    release.set()
                    await asyncio.gather(owner, return_exceptions=True)

        asyncio.run(asyncio.wait_for(go(), 5))

    def test_shutdown_cancels_legacy_jobs_together_with_dynamic_jobs(self):
        from unittest.mock import patch

        async def go():
            started = {name: asyncio.Event() for name in ('legacy', 'dynamic')}
            exited = set()

            async def wait_job(job, name):
                started[name].set()
                try:
                    await asyncio.Event().wait()
                finally:
                    exited.add(name)

            def start_all():
                scheduler.add_job('dynamic', '', 'x', at=time.time() + 60)

            with patch.object(config, 'main', {'scheduler': [{'name': 'legacy'}]}), \
                    patch.object(schedule_store, 'start_all', start_all), \
                    patch.object(scheduler, '_run_job_loop', wait_job):
                owner = asyncio.create_task(scheduler.run())
                try:
                    await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 2)
                    owner.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await owner
                    self.assertEqual(exited, {'legacy', 'dynamic'})
                    self.assertIsNone(scheduler._loop)
                    self.assertEqual(scheduler._dynamic_tasks, {})
                finally:
                    owner.cancel()
                    await asyncio.gather(owner, return_exceptions=True)

        asyncio.run(asyncio.wait_for(go(), 5))


class TestWebIds(unittest.TestCase):
    """定时页签生成的元素 id 不能和页面上别的地方重复：技能编辑表单占了 sf-name 等，
    撞了以后 getElementById 拿到的是那边隐藏的空输入框，保存时报“要有名字”。"""

    def test_schedule_tab_ids_are_unique_in_the_web_ui(self):
        import re
        web = _SRC.parent / 'web'
        hist = (web / 'js' / 'history.js').read_text(encoding='utf-8')
        sec = hist[hist.index('// ── Schedules Tab'):]
        ids = set(re.findall(r'id="([A-Za-z][\w-]*)"', sec))      # 这一段自己生成的元素
        self.assertIn('schf-name', ids)
        others = [p for p in web.rglob('*') if p.suffix in ('.html', '.js') and p.name != 'history.js']
        others_text = '\n'.join(p.read_text(encoding='utf-8', errors='ignore') for p in others)
        rest_of_history = hist[:hist.index('// ── Schedules Tab')]
        for i in sorted(ids):
            pat = re.compile(r'["\'#]' + re.escape(i) + r'["\'\s)]')
            self.assertIsNone(pat.search(others_text), f'{i} 和别的页面元素撞 id')
            self.assertIsNone(pat.search(rest_of_history), f'{i} 和 history.js 里别的元素撞 id')


if __name__ == '__main__':
    unittest.main()
