"""
scheduler.py — 定时任务调度器。

三类定时 job，到期后都把一段文字作为事件推入 event_bus（source = 'scheduler:<name>'），
collector 把 scheduler 视为优先来源，会叫醒主模型：

1. 配置项 [[scheduler]]（config.main['scheduler']），启动时读一次：
       [[scheduler]]
       name = 'hourly_check'
       cron = '0 * * * *'      # 标准 5 字段 cron（分 时 日 月 周）
       text = '一小时定时检查，请确认环境状态。'
       tz   = 'Asia/Shanghai'  # 可选；不写按容器本地时区（线上是 UTC），与旧行为一致
2. 任务的 check_cron（api/tasks.py、api/solutions.py、启动恢复），job 名 'task:<id>'。
3. 运行时定时任务（schedule_store，网页 / API / 可选的模型工具建），job 名 'sched:<id>'，
   增删改立即生效、不用重启，存在 SQLite 里跨重启保留。每次触发都要跑一轮模型，所以
   这一类有最短间隔和每日触发上限（config.main['scheduler_settings']），见 schedule_store。
"""

import asyncio
import datetime
import functools
import time

import config
import event_bus

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover — Python < 3.9
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception


# ── cron 解析 ─────────────────────────────────────────────────────────────────

_FIELD_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
_FIELD_NAMES = ('分', '时', '日', '月', '周')


def _parse_field(expr: str, lo: int, hi: int, label: str = '') -> set[int]:
    """解析一个 cron 字段。支持 *、*/N、a-b、a-b/N、逗号列表；越界或写错就报 ValueError。"""
    values: set[int] = set()
    for part in expr.split(','):
        part = part.strip()
        if not part:
            raise ValueError(f'cron 字段「{label}」有空项')
        step = 1
        if '/' in part:
            part, step_s = part.split('/', 1)
            step = int(step_s)
            if step <= 0:
                raise ValueError(f'cron 字段「{label}」的步长必须大于 0')
        if part == '*':
            a, b = lo, hi
        elif '-' in part:
            a_s, b_s = part.split('-', 1)
            a, b = int(a_s), int(b_s)
        else:
            a = b = int(part)
        if a < lo or b > hi or a > b:
            raise ValueError(f'cron 字段「{label}」取值 {a}-{b} 超出范围 {lo}-{hi}')
        values.update(range(a, b + 1, step))
    return values


def parse_cron(cron: str) -> tuple[set[int], ...]:
    """校验并解析 5 字段 cron，返回 (分, 时, 日, 月, 周) 五个集合。周：0 = 周日。"""
    fields = (cron or '').strip().split()
    if len(fields) != 5:
        raise ValueError(f'cron 要正好 5 个字段（分 时 日 月 周），收到 {cron!r}')
    try:
        return tuple(_parse_field(f, lo, hi, name)
                     for f, (lo, hi), name in zip(fields, _FIELD_RANGES, _FIELD_NAMES))
    except ValueError as e:
        if '字段' in str(e):
            raise
        raise ValueError(f'cron 里有不是数字的写法：{cron!r}') from e


def get_tz(name: str | None):
    """时区名 → tzinfo。空 = 容器本地时区（返回 None）。未知时区报 ValueError。"""
    if not name:
        return None
    if ZoneInfo is None:
        raise ValueError('这台机器的 Python 不支持时区（缺 zoneinfo）')
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f'未知时区：{name!r}') from e


def _tz_arg(tz):
    if isinstance(tz, str) or tz is None:
        return get_tz(tz) if tz else None
    return tz


def _compile(cron: str):
    """cron → (当天的触发分钟（排好序，时*60+分）, 日集合, 月集合, Python 星期集合)。"""
    minutes, hours, mdays, months, wdays = parse_cron(cron)
    day_mins = sorted(h * 60 + m for h in hours for m in minutes)
    py_wdays = {(w - 1) % 7 for w in wdays}   # cron 0=Sunday → Python 0=Monday
    return day_mins, mdays, months, py_wdays


def _date_ok(d: datetime.date, mdays, months, py_wdays) -> bool:
    return d.month in months and d.day in mdays and d.weekday() in py_wdays


def _wall(d: datetime.date, mins: int, tz) -> datetime.datetime:
    return datetime.datetime(d.year, d.month, d.day, mins // 60, mins % 60, tzinfo=tz)


def _next_run(cron: str, after: float, tz=None) -> float:
    """
    计算 cron 表达式在 after 之后的下一个触发时间（unix timestamp）。

    注意：日(月) 和 周 两个字段**都**限定时按“且”匹配（例如 '0 8 13 * 5' 只在
    13 号又恰好是周五时触发），与标准 cron 的“或”不同。这是本调度器一直以来的
    行为，配置里已有的 job 依赖它，所以保持不变（tests/test_runtime_scheduler.py 锁住）。

    支持标准 5 字段 cron：分 时 日(月) 月 周
    字段值：数字、*、*/N、a-b、a-b/N、逗号列表（不支持 L/W/#）
    tz：tzinfo 或时区名；None 按容器本地时区（旧行为）。

    结果与“从 after+60 的整分钟起按挂钟逐分钟找、最多找两年”完全一样，但按天跳：
    日期不匹配的整天直接跳过，匹配的那天只看列出来的时、分。一年一次的 cron 也只要几百步，
    不会在事件循环里扫几十万分钟。
    """
    tz = _tz_arg(tz)
    day_mins, mdays, months, py_wdays = _compile(cron)
    start = datetime.datetime.fromtimestamp(after + 60, tz).replace(second=0, microsecond=0)
    limit = start + datetime.timedelta(days=366 * 2)
    start_min = start.hour * 60 + start.minute
    d = start.date()
    first = True
    while d <= limit.date():
        if _date_ok(d, mdays, months, py_wdays):
            for mm in day_mins:
                if first and mm < start_min:
                    continue
                cand = _wall(d, mm, tz)
                if cand >= limit:
                    break
                return cand.timestamp()
        first = False
        d += datetime.timedelta(days=1)

    raise RuntimeError(f'Cannot find next run for cron: {cron!r}')


def next_runs(cron: str, after: float, count: int, tz=None) -> list[float]:
    """接下来 count 次触发时间。"""
    out, t = [], after
    for _ in range(count):
        t = _next_run(cron, t, tz)
        out.append(t)
    return out


def fires_between(cron: str, start: float, end: float, tz=None, limit: int | None = None) -> int:
    """[start, end) 里触发几次（按挂钟时间，与 _next_run 一致）。limit：数到这么多就停。"""
    if limit is not None and limit <= 0:
        return 0
    tz = _tz_arg(tz)
    day_mins, mdays, months, py_wdays = _compile(cron)
    d = datetime.datetime.fromtimestamp(start, tz).date()
    last = datetime.datetime.fromtimestamp(end, tz).date()
    n = 0
    while d <= last:
        if _date_ok(d, mdays, months, py_wdays):
            for mm in day_mins:
                t = _wall(d, mm, tz).timestamp()
                if start <= t < end:
                    n += 1
                    if limit is not None and n >= limit:
                        return n
        d += datetime.timedelta(days=1)
    return n


def matching_dates(cron: str, after: float, days: int, tz=None) -> list[datetime.date] | None:
    """接下来 days 天里（cron 的时区）日期会匹配的那些天。

    日、月两个字段都是 * 时返回 None：这种 cron 只按星期重复，看一个星期就够了。"""
    _, mdays, months, py_wdays = _compile(cron)
    if mdays == set(range(1, 32)) and months == set(range(1, 13)):
        return None
    d = datetime.datetime.fromtimestamp(after, _tz_arg(tz)).date()
    out = []
    for _ in range(days):
        if _date_ok(d, mdays, months, py_wdays):
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def min_gap_minutes(cron: str, after: float, tz=None) -> int | None:
    """相邻两次触发最短隔几分钟（按挂钟时间，看接下来两年）；两年里不到两次返回 None。

    每个匹配日里的触发分钟都一样，所以一天之内的最短间隔是定值；跨天的间隔只取决于
    相邻两个匹配日隔几天：隔 k 天 → k*1440 - 当天最后一次 + 当天第一次。
    比逐次算“接下来 50 次”准：像 '0 0,23 30,31 * *' 这种月底才出现的 1 小时间隔也能算到。"""
    day_mins, mdays, months, py_wdays = _compile(cron)
    intra = min((b - a for a, b in zip(day_mins, day_mins[1:])), default=None)
    d = datetime.datetime.fromtimestamp(after, _tz_arg(tz)).date()
    prev, span = None, None
    for _ in range(366 * 2):
        if _date_ok(d, mdays, months, py_wdays):
            if prev is not None:
                k = (d - prev).days
                span = k if span is None else min(span, k)
                if span == 1:
                    break
            prev = d
        d += datetime.timedelta(days=1)
    gaps = [g for g in (intra,) if g is not None]
    if span is not None:
        gaps.append(span * 1440 - day_mins[-1] + day_mins[0])
    return min(gaps) if gaps else None


# ── 动态 job 管理 ─────────────────────────────────────────────────────────────

_dynamic_tasks: dict[str, asyncio.Task] = {}   # name → asyncio.Task
_job_info: dict[str, dict] = {}                # name → job（含 next_ts），供 API 展示
_loop: asyncio.AbstractEventLoop | None = None  # run() 记下的事件循环
ONCE_RETRY_DELAY_S = 30                         # 一次性事件入队失败后多久重试
ONCE_MAX_RETRIES = 10


def _on_loop(fn) -> None:
    """在事件循环线程里执行 fn。

    api/schedules.py 的路由是同步的（SQLite 读写不能卡事件循环，FastAPI 把它们放进
    线程池），但 asyncio 的 task 只能在循环线程里建和取消，所以从别的线程来的调用
    用 call_soon_threadsafe 转回循环；按 FIFO 执行，先 remove 后 add 的顺序不会乱。
    """
    loop = _loop
    if loop is None or not loop.is_running():
        fn()                     # 启动前或测试里没有循环：直接执行
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        fn()                     # 已经在循环线程里
    else:
        loop.call_soon_threadsafe(fn)


def _add_job_now(name, cron, text, tz, at, source, on_fire, after_fire=None) -> None:
    _remove_job_now(name)
    job = {'name': name, 'cron': cron, 'text': text, 'tz': tz, 'at': at,
           'source': source, 'on_fire': on_fire, 'after_fire': after_fire}
    task = asyncio.ensure_future(_run_job(job))
    _dynamic_tasks[name] = task
    _job_info[name] = job


def _remove_job_now(name: str) -> None:
    _job_info.pop(name, None)
    task = _dynamic_tasks.pop(name, None)
    if task and not task.done():
        task.cancel()


def add_job(name: str, cron: str, text: str, *, tz=None, at: float | None = None,
            source: str | None = None, on_fire=None, after_fire=None) -> None:
    """动态添加一个定时 job（如果同名已存在则先移除）。可以从任何线程调用。

    cron 和 at 二选一：at 是一次性触发的 unix 时间。
    on_fire(job) -> bool：推事件前调用，返回 False 就跳过这一次。schedule_store 在这里
      持锁检查并**占一个当天名额**（检查和占用是同一步）。
    after_fire(job, delivered, retrying)：on_fire 放行之后一定会调用一次。delivered=True
      表示事件已入队（名额转正、记账、一次性任务收尾）；False 表示没发出去（退回名额），
      retrying 表示 scheduler 还会再试（只有一次性 job 会重试，最多 ONCE_MAX_RETRIES 次）。
    两个回调都在线程池里跑（它们会读写 SQLite，不能卡事件循环）。job 在推事件途中被取消时
    占着的名额不退回——算它发过了，宁可少发不多发。
    """
    _on_loop(functools.partial(_add_job_now, name, cron, text, tz, at, source, on_fire, after_fire))


def remove_job(name: str) -> None:
    """移除一个动态 job。可以从任何线程调用。"""
    _on_loop(functools.partial(_remove_job_now, name))


def job_next_ts(name: str) -> float | None:
    """某个动态 job 的下一次触发时间（没在跑返回 None）。"""
    job = _job_info.get(name)
    return job.get('next_ts') if job else None


# ── 启动 ──────────────────────────────────────────────────────────────────────

async def run() -> None:
    """常驻 async task：加载配置里的 job 和运行时定时任务并驱动它们。"""
    global _loop
    _loop = asyncio.get_running_loop()
    try:
        import schedule_store
        schedule_store.start_all()
    except Exception as e:  # 运行时定时任务坏了不能拖垮配置里的 job
        print(f'[scheduler] failed to start runtime schedules: {e}')

    jobs = list(config.main.get('scheduler', []))
    if not jobs:
        return

    # 每个 job 独立运行一个协程
    await asyncio.gather(*[_run_job(job) for job in jobs])


async def _run_job(job: dict) -> None:
    name = job.get('name', 'unnamed')
    cron = job.get('cron', '')
    text = job.get('text', f'[scheduler:{name}] 触发')
    source = job.get('source') or f'scheduler:{name}'
    at = job.get('at')
    on_fire = job.get('on_fire')
    after_fire = job.get('after_fire')
    try:
        tz = get_tz(job['tz']) if isinstance(job.get('tz'), str) else job.get('tz')
    except ValueError as e:
        print(f'[scheduler] invalid tz for {name!r}: {e}')
        return

    now = time.time()
    try:
        next_ts = at if at else _next_run(cron, now, tz)
    except Exception as e:
        print(f'[scheduler] invalid cron for {name!r}: {e}')
        return

    while True:
        job['next_ts'] = next_ts
        wait = next_ts - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        fire = True
        if on_fire is not None:
            try:
                fire = (await asyncio.to_thread(on_fire, job)) is not False
            except Exception as e:
                # 检查本身出错就不发：宁可漏一次，也不越过每日上限
                print(f'[scheduler] on_fire for {name!r} failed: {e}')
                fire = False
        delivered = False
        if fire:
            try:
                await event_bus.enqueue(
                    source  = source,
                    text    = text,
                )
                delivered = True
            except Exception as e:
                print(f'[scheduler] enqueue for {name!r} failed: {e}')
        # 一次性的没发出去：过一会儿再试，直到重试用完
        retrying = bool(at) and fire and not delivered and job.get('retries', 0) < ONCE_MAX_RETRIES
        if fire and after_fire is not None:
            try:
                await asyncio.to_thread(after_fire, job, delivered, retrying)
            except Exception as e:
                print(f'[scheduler] after_fire for {name!r} failed: {e}')

        if at:   # 一次性
            if retrying:
                job['retries'] = job.get('retries', 0) + 1
                next_ts = time.time() + ONCE_RETRY_DELAY_S
                continue
            job['next_ts'] = None
            return
        try:
            next_ts = _next_run(cron, time.time(), tz)
        except Exception:
            return   # 无法计算下次，退出