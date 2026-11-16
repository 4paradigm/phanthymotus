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
        base = datetime.datetime(2026, 11, 17, 8, 0, tzinfo=CN)
        with _FakeJobs():
            for i in range(49):
                day = (base + datetime.timedelta(days=i)).strftime('%Y-%m-%d %H:%M')
                schedule_store.create(name=f'o{i}', text='x', kind='once', at=day)

    def test_one_shots_on_the_same_day_do_stack(self):
        config.main['scheduler_settings'] = {**schedule_store.DEFAULT_SETTINGS, 'max_schedules': 60,
                                             'max_fires_per_day': 2}
        with _FakeJobs():
            schedule_store.create(name='a', text='x', kind='once', at='2026-12-01 08:00')
            schedule_store.create(name='b', text='x', kind='once', at='2026-12-01 09:00')
            with self.assertRaisesRegex(ValueError, '超过上限'):
                schedule_store.create(name='c', text='x', kind='once', at='2026-12-01 10:00')
            schedule_store.create(name='d', text='x', kind='once', at='2026-12-02 10:00')

    def test_corrupt_row_does_not_stop_startup(self):
        with _FakeJobs() as fj:
            good = schedule_store.create(name='好的', text='x', cron='0 8 * * *')
            schedule_store._schedules.clear(); fj.jobs.clear()
            bad = schedule_store.Schedule(name='坏的', text='x', kind='once', at='不是时间')
            schedule_store._schedules[bad.id] = bad
            schedule_store._persist(bad)
            self.assertEqual(schedule_store.start_all(), 1, '坏的跳过，好的照常启动')
            self.assertIn(f'sched:{good.id}', fj.jobs)

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
        self.assertFalse(self._req('DELETE', f'/api/schedules/{sid}').json()['ok'])

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


if __name__ == '__main__':
    unittest.main()