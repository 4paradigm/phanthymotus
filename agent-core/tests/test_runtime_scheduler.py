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
        schedule_store._fires_by_day.clear()
        schedule_store._loaded = True
        config.main['scheduler_settings'] = dict(schedule_store.DEFAULT_SETTINGS)


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
            with self.assertRaisesRegex(ValueError, '一天最多'):
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
        cb = schedule_store._on_fire(a.id)
        self.assertTrue(cb({}))
        self.assertFalse(schedule_store.find('a').enabled, '一次性定时触发后应停用')
        schedule_store._schedules[a.id].enabled = True
        self.assertFalse(cb({}), '超过每日上限应跳过')

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


if __name__ == '__main__':
    unittest.main()