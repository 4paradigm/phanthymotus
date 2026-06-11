"""
scheduler.py — 定时任务调度器。

读取 config.main 的 [[scheduler]] 列表，每个条目：
    [[scheduler]]
    name = 'hourly_check'
    cron = '0 * * * *'      # 标准 5 字段 cron（分 时 日 月 周）
    text = '一小时定时检查，请确认环境状态。'

到期后把 text 作为事件推入 event_bus（source = 'scheduler:<name>'）。
"""

import asyncio
import datetime
import time

import config
import event_bus


def _next_run(cron: str, after: float) -> float:
    """
    计算 cron 表达式下一个触发时间（unix timestamp）。

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
    dt = datetime.datetime.fromtimestamp(after + 60).replace(second=0, microsecond=0)
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


def add_job(name: str, cron: str, text: str) -> None:
    """动态添加一个定时 job（如果同名已存在则先移除）。"""
    remove_job(name)
    job = {'name': name, 'cron': cron, 'text': text}
    task = asyncio.ensure_future(_run_job(job))
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

    now = time.time()
    try:
        next_ts = _next_run(cron, now)
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
            next_ts = _next_run(cron, time.time())
        except Exception:
            return   # 无法计算下次，退出
