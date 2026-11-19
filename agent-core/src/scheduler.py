"""
scheduler.py — 定时任务调度器。

读取 config.main 的 [[scheduler]] 列表，每个条目：
    [[scheduler]]
    name = 'hourly_check'
    cron = '0 * * * *'      # 标准 5 字段 cron（分 时 日 月 周）
    text = '一小时定时检查，请确认环境状态。'

到期后把 text 作为事件推入 event_bus（source = 'scheduler:<name>'）。

cron 默认按本机时区解释；add_job 可传 tz 指定别的时区（容器多半是 UTC，而用户说的
「晚上十点」是北京时间）。add_once 是只触发一次的 job，触发完自己移除。
"""

import asyncio
import datetime
import time
import typing

import config
import event_bus


def _next_run(cron: str, after: float, tz: datetime.tzinfo | None = None) -> float:
    """
    计算 cron 表达式在 `after` 之后的下一个触发时间（unix timestamp）。

    `tz` 为 None 时按本机时区解释 cron 的各字段。

    支持标准 5 字段 cron：分 时 日(月) 月 周
    字段值：数字、*、*/N、逗号列表（不支持 L/W/#）
    """
    def parse_field(expr: str, lo: int, hi: int) -> set[int]:
        values: set[int] = set()
        for part in expr.split(','):
            if part == '*':
                values.update(range(lo, hi + 1))
            elif part.startswith('*/'):
                step = int(part[2:])
                values.update(range(lo, hi + 1, step))
            elif '-' in part:
                a, b = part.split('-')
                values.update(range(int(a), int(b) + 1))
            else:
                values.add(int(part))
        return values

    fields = cron.strip().split()
    if len(fields) != 5:
        raise ValueError(f'Invalid cron expression: {cron!r}')

    minutes = parse_field(fields[0], 0, 59)
    hours   = parse_field(fields[1], 0, 23)
    mdays   = parse_field(fields[2], 1, 31)
    months  = parse_field(fields[3], 1, 12)
    wdays   = parse_field(fields[4], 0, 6)   # 0=Sunday

    # Iterate minute-by-minute from `after+60` up to 2 years
    dt = datetime.datetime.fromtimestamp(after + 60, tz).replace(second=0, microsecond=0)
    limit = dt + datetime.timedelta(days=366 * 2)

    while dt < limit:
        if (dt.month in months
                and dt.day in mdays
                and dt.weekday() in {(w - 1) % 7 for w in wdays}   # map: 0=Mon in Python, 0=Sun in cron
                and dt.hour in hours
                and dt.minute in minutes):
            return dt.timestamp()
        dt += datetime.timedelta(minutes=1)

    raise RuntimeError(f'Cannot find next run for cron: {cron!r}')


# ── 动态 job 管理 ─────────────────────────────────────────────────────────────

_dynamic_tasks: dict[str, asyncio.Task] = {}   # name → asyncio.Task


def add_job(name: str, cron: str, text: str,
            tz: datetime.tzinfo | None = None) -> None:
    """动态添加一个定时 job（如果同名已存在则先移除）。"""
    remove_job(name)
    job = {'name': name, 'cron': cron, 'text': text, 'tz': tz}
    task = asyncio.ensure_future(_run_job(job))
    _dynamic_tasks[name] = task


def add_once(name: str, at: float, text: str,
             on_fired: typing.Callable[[], None] | None = None) -> None:
    """动态添加一个只触发一次的 job：到 `at`（unix timestamp）推一次事件后自行移除。

    `at` 已过去则立即触发。`on_fired` 在事件推出后调用，给调用方收尾用。
    """
    remove_job(name)
    task = asyncio.ensure_future(_run_once(name, at, text, on_fired))
    _dynamic_tasks[name] = task


def remove_job(name: str) -> None:
    """移除一个动态 job。"""
    task = _dynamic_tasks.pop(name, None)
    if task and not task.done():
        task.cancel()


# ── 静态配置启动 ──────────────────────────────────────────────────────────────

async def run() -> None:
    """常驻 async task：加载配置并驱动所有定时任务。"""
    jobs = list(config.main.get('scheduler', []))
    if not jobs:
        return

    # 每个 job 独立运行一个协程
    await asyncio.gather(*[_run_job(job) for job in jobs])


async def _run_job(job: dict) -> None:
    name = job.get('name', 'unnamed')
    cron = job.get('cron', '')
    text = job.get('text', f'[scheduler:{name}] 触发')
    tz = job.get('tz')

    now = time.time()
    try:
        next_ts = _next_run(cron, now, tz)
    except Exception as e:
        print(f'[scheduler] invalid cron for {name!r}: {e}')
        return

    while True:
        wait = next_ts - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        await event_bus.enqueue(
            source  = f'scheduler:{name}',
            text    = text,
        )

        try:
            # 从刚触发的这一次往后算，而不是从 time.time()。asyncio 判定到期时会放宽一个
            # clock resolution，sleep 又走单调时钟，醒来时墙钟可能还差几毫秒才到整分；
            # 从 time.time() 算会再次选中刚触发的那一分钟，同一时刻推两次事件。
            next_ts = _next_run(cron, max(next_ts, time.time()), tz)
        except Exception:
            return   # 无法计算下次，退出


async def _run_once(name: str, at: float, text: str,
                    on_fired: typing.Callable[[], None] | None) -> None:
    wait = at - time.time()
    if wait > 0:
        await asyncio.sleep(wait)
    # 先摘掉自己再推事件：事件处理里若有人 add/remove 同名 job，不能被这里误删。
    if _dynamic_tasks.get(name) is asyncio.current_task():
        del _dynamic_tasks[name]
    await event_bus.enqueue(source=f'scheduler:{name}', text=text)
    if on_fired:
        try:
            on_fired()
        except Exception as e:
            print(f'[scheduler] on_fired for {name!r} failed: {e}')
