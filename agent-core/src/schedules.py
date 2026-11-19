"""
schedules.py — 用户要求的定时事项（一次性提醒、每天/每周的例行事务）。

存储复用 task_store：一条定时事项就是一个 metadata 里带 `schedule` 的 Task，于是设置页
能看到、能删，重置会清掉，重启会恢复，这些都不用另做一份。但它不是"正在做的任务"——
不由 task_done 收尾，也不进 task_force_clear；一次性的触发后由框架自动结束，周期的直到
用户取消。

给模型的入口刻意不收 cron 字符串，只收结构化的「几分钟后 / 几点 / 每天·工作日·周几」。
原因见 event/task.py 的 Tools 注释：模型手里有 cron 就会顺手建每 2 分钟的轮询。这里能
表达的最短周期是一天，轮询从结构上建不出来。时间换算也全在代码里：用户说的时刻按北京
时间理解，cron 以北京时区交给调度器，不经过容器时区，也不让模型做加减法。
"""

import datetime
import math
import time

import scheduler
import task_store

TZ = datetime.timezone(datetime.timedelta(hours=8))   # 与 prompt._TZ_CN 一致
MAX_ACTIVE = 10
# 重启错过的一次性提醒：这么久以内补发，再久就不补了（半夜补一句"该吃药了"没有意义）。
LATE_GRACE_S = 600

_DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']   # 下标 = Python weekday()
_DAYS_ZH = ['一', '二', '三', '四', '五', '六', '日']
_REPEAT_ALIASES = {
    'daily': list(range(7)),
    'weekdays': list(range(5)),
    'weekends': [5, 6],
}


def is_schedule(task: task_store.Task) -> bool:
    return isinstance(task.metadata, dict) and 'schedule' in task.metadata


def active() -> list[task_store.Task]:
    return [t for t in task_store.active_tasks() if is_schedule(t)]


def job_name(task_id: str) -> str:
    # 与 task 的定时检查同名：设置页删除、重置都按 `task:<id>` 移除 job，这样不用改它们。
    return f'task:{task_id}'


# ── 解析 ──────────────────────────────────────────────────────────────────────

def _parse_repeat(repeat: str) -> list[int]:
    """'' → []；'daily' / 'weekdays' / 'weekends' / 'mon,wed,fri' → 星期下标（0=周一）。"""
    repeat = repeat.strip().lower()
    if not repeat:
        return []
    if repeat in _REPEAT_ALIASES:
        return _REPEAT_ALIASES[repeat]
    days = []
    for part in repeat.replace(' ', '').split(','):
        if part not in _DAYS:
            raise ValueError(f'repeat 只能是 daily / weekdays / weekends 或 mon..sun 的逗号列表，收到 {repeat!r}')
        days.append(_DAYS.index(part))
    return sorted(set(days))


def _parse_at(at: str, now: float) -> datetime.datetime:
    """'HH:MM' 或 'YYYY-MM-DD HH:MM'（可带 ISO 时区偏移）；不带时区按北京时间。"""
    at = at.strip()
    try:
        when = datetime.datetime.fromisoformat(at)
    except ValueError:
        try:
            clock = datetime.time.fromisoformat(at)
        except ValueError:
            raise ValueError(f'at 格式应为 HH:MM 或 YYYY-MM-DD HH:MM，收到 {at!r}') from None
        today = datetime.datetime.fromtimestamp(now, TZ).date()
        when = datetime.datetime.combine(today, clock.replace(tzinfo=None),
                                         tzinfo=clock.tzinfo or TZ)
        if when.timestamp() <= now:
            when += datetime.timedelta(days=1)   # 只说了钟点且今天已过 → 明天
        return when
    return when if when.tzinfo else when.replace(tzinfo=TZ)


def build(delay_minutes: float, at: str, repeat: str, now: float) -> dict:
    """把模型给的参数变成存进 metadata 的 spec。参数不合法抛 ValueError。"""
    if not isinstance(at, str) or not isinstance(repeat, str):
        raise ValueError('at 和 repeat 必须是字符串')
    if (isinstance(delay_minutes, bool) or not isinstance(delay_minutes, (int, float))
            or not math.isfinite(delay_minutes) or delay_minutes < 0):
        raise ValueError('delay_minutes 必须是非负数')
    days = _parse_repeat(repeat)

    if days:
        if delay_minutes:
            raise ValueError('周期事项用 at 指定每天几点，不能用 delay_minutes')
        if not at:
            raise ValueError('周期事项需要 at（HH:MM，北京时间）')
        try:
            clock = datetime.time.fromisoformat(at.strip())
        except ValueError:
            raise ValueError(f'周期事项的 at 只能是 HH:MM，收到 {at!r}') from None
        if clock.tzinfo is not None:
            raise ValueError('周期事项的 at 按北京时间填，不带时区')
        return {'days': days, 'hour': clock.hour, 'minute': clock.minute}

    if bool(delay_minutes) == bool(at):
        raise ValueError('一次性事项 delay_minutes 和 at 填且只填一个')
    if delay_minutes:
        fire_at = now + delay_minutes * 60
    else:
        fire_at = _parse_at(at, now).timestamp()
        if fire_at <= now:
            raise ValueError(f'{at} 已经过去了')
    if fire_at - now > 366 * 86400:
        raise ValueError('最远只能定到一年以内')
    return {'at': round(fire_at, 3)}


# ── 描述 / 下次触发 ────────────────────────────────────────────────────────────

def _cron(spec: dict) -> str:
    # cron 的星期 0=周日；Python weekday() 0=周一。
    dows = ','.join(str((d + 1) % 7) for d in spec['days'])
    return f'{spec["minute"]} {spec["hour"]} * * {dows}'


def next_fire(spec: dict, now: float | None = None) -> float:
    now = time.time() if now is None else now
    if 'at' in spec:
        return spec['at']
    return scheduler._next_run(_cron(spec), now, TZ)


def fmt(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, TZ).strftime('%Y-%m-%d %H:%M')


def describe(spec: dict) -> str:
    """'一次，2026-11-19 22:00' / '每天 22:00' / '每周一、三 08:30'。"""
    if 'at' in spec:
        return f'一次，{fmt(spec["at"])}'
    clock = f'{spec["hour"]:02d}:{spec["minute"]:02d}'
    days = spec['days']
    if days == list(range(7)):
        when = '每天'
    elif days == list(range(5)):
        when = '工作日每天'
    elif days == [5, 6]:
        when = '周末每天'
    else:
        when = '每周' + '、'.join(_DAYS_ZH[d] for d in days)
    return f'{when} {clock}'


def _fire_text(task: task_store.Task) -> str:
    spec = task.metadata['schedule']
    set_at = fmt(task.created_at)
    if 'at' in spec:
        tail = '这是一次性事项，框架已自动结束它，不需要 task_done。'
    else:
        tail = ('这是周期事项，下次到点还会再来；做完不要 task_done，'
                f'用户要取消时用 schedule_cancel("{task.id}")。')
    return (f'定时事项 [{task.id}] 到点了（用户于 {set_at} 设定，{describe(spec)}）：'
            f'{task.goal}。现在去做。{tail}')


# ── 注册 ──────────────────────────────────────────────────────────────────────

def register(task: task_store.Task, now: float | None = None) -> None:
    """把一条定时事项挂到调度器上。创建时和重启恢复时都走这里。"""
    now = time.time() if now is None else now
    spec = task.metadata['schedule']
    name = job_name(task.id)
    if 'at' in spec:
        if spec['at'] < now - LATE_GRACE_S:
            print(f'[schedules] {task.id} 错过了触发时刻 {fmt(spec["at"])}（停机期间），不再补发')
            task_store.done(task.id, summary=f'错过触发时刻 {fmt(spec["at"])}')
            return
        scheduler.add_once(name, spec['at'], _fire_text(task),
                           on_fired=lambda: task_store.done(task.id, summary='已触发'))
    else:
        scheduler.add_job(name, _cron(spec), _fire_text(task), tz=TZ)


def create(text: str, spec: dict) -> tuple[task_store.Task, bool]:
    """返回 (task, 是否新建)。同样内容、同样时间的已存在就直接返回那条，不重复建。"""
    for t in active():
        if t.goal == text and t.metadata['schedule'] == spec:
            return t, False
    if len(active()) >= MAX_ACTIVE:
        raise ValueError(f'已有 {MAX_ACTIVE} 条定时事项，先用 schedule_list 看看、取消不要的')
    task = task_store.create(goal=text, metadata={'schedule': spec})
    register(task)
    return task, True


def cancel(task_id: str) -> task_store.Task | None:
    task = task_store.get(task_id)
    if not task or not is_schedule(task):
        return None
    scheduler.remove_job(job_name(task_id))
    return task_store.done(task_id, summary='已取消')
