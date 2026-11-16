"""
schedule_store.py — 运行时定时任务：存储、校验、注册到 scheduler。

和配置项 [[scheduler]] 的区别：这里的定时任务在网页 / API（以及管理员打开后的模型工具）
里增删改，**立即生效、不用重启**，存在 SQLite（与 config.py 共用 DB_PATH）里跨重启保留。

每次触发都会叫醒主模型跑一轮（Tianyi 实测一轮三万多 token），所以这里管着几道闸，
数值在 config.main['scheduler_settings']，设置页可改：
  · min_interval_minutes —— 同一个定时任务两次触发的最短间隔；
  · max_fires_per_day    —— 所有运行时定时任务（含网页上的“立即触发”）加起来一天最多触发
                            几次。建 / 改的时候按 cron 逐天估算拦一次；到点时再按实际次数卡
                            一次：推事件之前在锁里**先占一个名额**（检查和占用是同一步，几条
                            同时到点也不会一起越过上限），没发出去就退回名额，发出去了写进
                            schedule_fires 表（重启后当天已触发的次数还在）；
  · max_schedules / max_per_skill —— 条数上限；
  · model_tools_enabled  —— 模型能不能自己建定时（默认关，见 event/schedule.py）。
改设置时拿新的限制把现有启用的定时任务重新校验一遍，有不合规的就拒绝这次修改并列出是哪几条，
不会出现“设置收紧了、老定时照旧按原频率跑”。

job 名统一是 'sched:<id>'，事件 source 是 'scheduler:<name>'。
"""

from __future__ import annotations

import contextlib
import datetime
import functools
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

import config
import scheduler

DEFAULT_SETTINGS = {
    'model_tools_enabled': False,
    'min_interval_minutes': 10,
    'max_schedules': 20,
    'max_per_skill': 6,
    'max_fires_per_day': 48,
    'default_tz': 'Asia/Shanghai',
}

KINDS = ('cron', 'once')
CREATORS = ('web', 'api', 'model')
_AT_FORMATS = ('%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S')
_HORIZON_DAYS = 367          # 估算每日触发数时往后看多远（一次性定时最多设到一年后）
_FIRE_KEEP_S = 2 * 86400     # 触发记录保留多久（算“今天”够用）


@dataclass
class Schedule:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ''
    text: str = ''
    kind: str = 'cron'            # cron | once
    cron: str = ''                # kind=cron：5 字段 cron，按 tz 解释
    at: str = ''                  # kind=once：'YYYY-MM-DD HH:MM'，按 tz 解释
    tz: str = 'Asia/Shanghai'
    skill: str = ''               # 对应的技能 slug（可空），写进事件文字提示模型先激活
    enabled: bool = True
    created_by: str = 'web'       # web | api | model
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_run: float | None = None
    fire_count: int = 0


_schedules: dict[str, Schedule] = {}
# 最近两天的触发时间（unix 秒）：已发出的（也存在 schedule_fires 表里）+ 正在发、占着名额的。
# “今天”按 default_tz 现算，所以改时区不会把计数清零。
_fire_times: list[float] = []
_loaded = False
# API 路由跑在线程池里，可能并发；增删改、设置、名额和内存表的读写都串行。
_lock = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with _lock:
            return fn(*a, **kw)
    return wrapper


# ── SQLite ──────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    import pathlib
    db_path = pathlib.Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute('CREATE TABLE IF NOT EXISTS schedules (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS schedule_fires (ts REAL NOT NULL)')
    conn.commit()
    return conn


@contextlib.contextmanager
def _db():
    """连接用完就关；with 块正常结束时提交。"""
    conn = _get_conn()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _persist(s: Schedule) -> None:
    with _db() as conn:
        conn.execute('INSERT OR REPLACE INTO schedules (id, data) VALUES (?, ?)',
                     (s.id, json.dumps(asdict(s), ensure_ascii=False)))


def _delete_row(sid: str) -> None:
    with _db() as conn:
        conn.execute('DELETE FROM schedules WHERE id = ?', (sid,))


@_locked
def load_all() -> None:
    global _loaded
    cutoff = time.time() - _FIRE_KEEP_S
    with _db() as conn:
        rows = conn.execute('SELECT data FROM schedules').fetchall()
        fires = conn.execute('SELECT ts FROM schedule_fires WHERE ts >= ?', (cutoff,)).fetchall()
    _schedules.clear()
    for (data,) in rows:
        d = json.loads(data)
        s = Schedule(**{k: v for k, v in d.items() if k in Schedule.__dataclass_fields__})
        _schedules[s.id] = s
    # 库里的是已发出的；内存里可能还有正在发、占着名额的，合在一起
    _fire_times[:] = sorted({t for (t,) in fires} | set(_fire_times))
    _loaded = True


def _ensure_loaded() -> None:
    with _lock:
        if not _loaded:
            load_all()


# ── 时间 ─────────────────────────────────────────────────────────────────────

def parse_at(at: str, tz_name: str) -> float:
    tz = scheduler.get_tz(tz_name)
    for fmt in _AT_FORMATS:
        try:
            dt = datetime.datetime.strptime(at.strip(), fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f'时间要写成 YYYY-MM-DD HH:MM，收到 {at!r}')
    return dt.replace(tzinfo=tz).timestamp() if tz else dt.timestamp()


def fmt_ts(ts: float | None, tz_name: str | None = None) -> str:
    if not ts:
        return ''
    tz = scheduler.get_tz(tz_name or get_settings()['default_tz'])
    return datetime.datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d %H:%M')


def next_run_ts(s: Schedule, now: float | None = None) -> float | None:
    if not s.enabled:
        return None
    now = time.time() if now is None else now
    if s.kind == 'once':
        t = parse_at(s.at, s.tz)
        return t if t > now else None
    return scheduler._next_run(s.cron, now, s.tz)


def _midnight(d: datetime.date, tz) -> float:
    return datetime.datetime.combine(d, datetime.time(), tzinfo=tz).timestamp()


def _day_bounds(ts: float, tz_name: str) -> tuple[float, float]:
    """ts 所在那一天（tz_name 时区）的 [0 点, 次日 0 点)。"""
    tz = scheduler.get_tz(tz_name)
    d = datetime.datetime.fromtimestamp(ts, tz).date()
    return _midnight(d, tz), _midnight(d + datetime.timedelta(days=1), tz)


def _day_total(d: datetime.date, crons: list[Schedule], once_n: int, tz, now: float,
               limit: int | None = None) -> int:
    """d 这一天（tz）还会触发几次：当天的一次性定时 + 各条 cron 在当天（从 now 起）的实际触发数。
    limit：数到这么多就停（只想知道超没超）。"""
    start = max(_midnight(d, tz), now)
    end = _midnight(d + datetime.timedelta(days=1), tz)
    n = once_n
    for c in crons:
        if limit is not None and n >= limit:
            break
        n += scheduler.fires_between(c.cron, start, end, c.tz,
                                     None if limit is None else limit - n)
    return n


def _peak_daily_fires(items: list[Schedule], now: float, day_tz: str,
                      cap: int | None = None) -> tuple[int, str]:
    """所有启用的定时任务合起来，按真实日期逐天相加，返回 (最忙那天的次数, 那一天)。

    要看的日子：
      · 今天剩下的部分 + 往后 7 个整天 —— 日、月都是 * 的 cron 只按星期重复，每个星期几都看到；
      · 每个一次性定时所在的那天；
      · 日或月有限定的 cron（比如每月 1 号），接下来一年里它真正会触发的那些天。
    每一天都把当天 cron 的实际触发数和当天的一次性定时加在一起，不拿不同日期的峰值相加。
    给了 cap 时，碰到第一个超过 cap 的日子就返回它（次数是那天的准确值）。
    """
    tz = scheduler.get_tz(day_tz)
    today = datetime.datetime.fromtimestamp(now, tz).date()
    days = {today + datetime.timedelta(days=i) for i in range(8)}
    once_by_day: dict = {}
    for x in items:
        if x.enabled and x.kind == 'once':
            t = parse_at(x.at, x.tz)
            if t <= now:
                continue
            d = datetime.datetime.fromtimestamp(t, tz).date()
            once_by_day[d] = once_by_day.get(d, 0) + 1
            days.add(d)
    crons = [x for x in items if x.enabled and x.kind == 'cron']
    for c in crons:
        ctz = scheduler.get_tz(c.tz)
        for cd in scheduler.matching_dates(c.cron, now, _HORIZON_DAYS, ctz) or ():
            # cron 时区的一天落在 day_tz 里可能跨两天，两天都看
            a = _midnight(cd, ctz)
            b = _midnight(cd + datetime.timedelta(days=1), ctz) - 1
            days.add(datetime.datetime.fromtimestamp(a, tz).date())
            days.add(datetime.datetime.fromtimestamp(b, tz).date())
    best, best_day = 0, ''
    for d in sorted(x for x in days if x >= today):
        n = _day_total(d, crons, once_by_day.get(d, 0), tz, now, None if cap is None else cap + 1)
        if cap is not None and n > cap:
            return _day_total(d, crons, once_by_day.get(d, 0), tz, now), d.isoformat()
        if n > best:
            best, best_day = n, d.isoformat()
    return best, best_day


# ── 设置 ─────────────────────────────────────────────────────────────────────

def get_settings() -> dict:
    saved = config.main.get('scheduler_settings', {}) or {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def _conflicts(st: dict, now: float | None = None) -> list[str]:
    """按设置 st，现在启用的定时任务有哪些不合规（中文短句）。"""
    now = time.time() if now is None else now
    on = [s for s in _schedules.values() if s.enabled]
    out = []
    for s in on:
        if s.kind != 'cron':
            continue
        try:
            gap = scheduler.min_gap_minutes(s.cron, now, s.tz)
        except ValueError:
            continue
        if gap is not None and gap < st['min_interval_minutes']:
            out.append(f'「{s.name}」最短 {gap} 分钟触发一次')
    if len(on) > st['max_schedules']:
        out.append(f'启用中的定时任务有 {len(on)} 条')
    if st['max_per_skill']:
        per: dict[str, int] = {}
        for s in on:
            if s.skill:
                per[s.skill] = per.get(s.skill, 0) + 1
        out += [f'技能 {k} 有 {n} 条' for k, n in sorted(per.items()) if n > st['max_per_skill']]
    try:
        total, day = _peak_daily_fires(on, now, st['default_tz'], cap=st['max_fires_per_day'])
    except ValueError:
        total, day = 0, ''
    if total > st['max_fires_per_day']:
        out.append(f'{day} 这一天会触发 {total} 次')
    return out


@_locked
def update_settings(**changes) -> dict:
    """改设置。新限制会拿来重新校验现有启用的定时任务：这次修改会让哪条变得不合规，
    就整个拒绝并列出来，请先改或停用那几条（不替人偷偷停用定时）。原本就不合规的不算在内，
    所以打开模型工具这类无关的修改不会被历史数据卡住。"""
    _ensure_loaded()
    cur = get_settings()
    new = dict(cur)
    for k, v in changes.items():
        if k not in DEFAULT_SETTINGS or v is None:
            continue
        if k == 'model_tools_enabled':
            new[k] = bool(v)
        elif k == 'default_tz':
            scheduler.get_tz(v)          # 未知时区直接报错
            new[k] = v
        else:
            v = int(v)
            if v < (1 if k != 'max_per_skill' else 0):
                raise ValueError(f'{k} 必须是正整数')
            new[k] = v
    now = time.time()
    before = set(_conflicts(cur, now))
    problems = [p for p in _conflicts(new, now) if p not in before]
    if problems:
        raise ValueError('按新的设置，现有的定时任务不合规：' + '；'.join(problems)
                         + '。请先改或停用这些定时任务，再改设置')
    config.main['scheduler_settings'] = new
    return new


# ── 校验 ─────────────────────────────────────────────────────────────────────

def validate(s: Schedule, *, now: float | None = None) -> None:
    """不合规就抛 ValueError（中文原因，直接给网页和模型看）。"""
    now = time.time() if now is None else now
    st = get_settings()
    s.name = (s.name or '').strip()
    s.text = (s.text or '').strip()
    s.skill = (s.skill or '').strip()
    if not s.name:
        raise ValueError('定时任务要有名字')
    if len(s.name) > 40:
        raise ValueError('名字不要超过 40 个字')
    if not s.text:
        raise ValueError('要写触发时推给机器人的文字')
    if len(s.text) > 500:
        raise ValueError('触发文字不要超过 500 个字')
    if s.kind not in KINDS:
        raise ValueError('kind 只能是 cron（重复）或 once（一次）')
    s.tz = (s.tz or st['default_tz']).strip()
    tz = scheduler.get_tz(s.tz)
    for other in _schedules.values():
        if other.id != s.id and other.name == s.name:
            raise ValueError(f'已经有叫「{s.name}」的定时任务了（id {other.id}）')

    if s.kind == 'once':
        s.cron = ''
        if not s.at:
            raise ValueError('一次性定时要写触发时间 at（YYYY-MM-DD HH:MM）')
        t = parse_at(s.at, s.tz)
        if s.enabled and t <= now + 30:
            raise ValueError(f'触发时间 {s.at}（{s.tz}）已经过去了')
        if t > now + 366 * 86400:
            raise ValueError('一次性定时最多设到一年以后')
    else:
        s.at = ''
        scheduler.parse_cron(s.cron)
        try:
            scheduler._next_run(s.cron, now, tz)
        except RuntimeError:
            raise ValueError(f'cron {s.cron!r} 接下来两年一次都不会触发（日期不存在？）') from None
        gap = scheduler.min_gap_minutes(s.cron, now, tz)
        if gap is not None and gap < st['min_interval_minutes']:
            raise ValueError(
                f'两次触发最短只隔 {gap} 分钟，低于下限 {st["min_interval_minutes"]} 分钟。'
                '每次触发都要跑一轮模型，请把间隔放宽')

    if not s.enabled:
        return
    others = [o for o in _schedules.values() if o.enabled and o.id != s.id]
    if len(others) + 1 > st['max_schedules']:
        raise ValueError(f'启用中的定时任务最多 {st["max_schedules"]} 条')
    if s.skill and st['max_per_skill'] and \
            sum(1 for o in others if o.skill == s.skill) + 1 > st['max_per_skill']:
        raise ValueError(f'技能 {s.skill} 的定时任务最多 {st["max_per_skill"]} 条')
    cap = st['max_fires_per_day']
    total, day = _peak_daily_fires([s, *others], now, st['default_tz'], cap=cap)
    if total > cap:
        raise ValueError(f'加上这条，{day} 这一天所有定时任务会触发 {total} 次，超过上限 {cap} 次')


# ── 每日名额（调用方持锁）────────────────────────────────────────────────────

def _fires_today(now: float) -> int:
    start, end = _day_bounds(now, get_settings()['default_tz'])
    return sum(1 for t in _fire_times if start <= t < end)


def _reserve(now: float) -> float | None:
    """占一个当天名额，返回名额（就是时间戳）；满了返回 None。"""
    if _fires_today(now) >= get_settings()['max_fires_per_day']:
        return None
    _fire_times.append(now)
    return now


def _release(slot: float | None) -> None:
    """没发出去：退回名额。"""
    if slot is not None:
        with contextlib.suppress(ValueError):
            _fire_times.remove(slot)


def _commit(slot: float | None) -> None:
    """发出去了：名额写进库（重启后当天的次数还在），顺手清掉两天前的记录。"""
    if slot is None:
        slot = time.time()
        _fire_times.append(slot)
    cutoff = slot - _FIRE_KEEP_S
    with _db() as conn:
        conn.execute('INSERT INTO schedule_fires (ts) VALUES (?)', (slot,))
        conn.execute('DELETE FROM schedule_fires WHERE ts < ?', (cutoff,))
    _fire_times[:] = [t for t in _fire_times if t >= cutoff]


def _cap_msg() -> str:
    return f'今天所有定时任务已经触发满 {get_settings()["max_fires_per_day"]} 次（每日上限）'


# ── 注册到 scheduler ─────────────────────────────────────────────────────────

def job_name(sid: str) -> str:
    return f'sched:{sid}'


def event_text(s: Schedule) -> str:
    hint = f'（定时任务「{s.name}」'
    if s.skill:
        hint += f'，对应技能 {s.skill}：先调 activate_skill(slug="{s.skill}")'
    return f'{s.text}\n{hint}）'


def _is_current(s: Schedule, job: dict) -> bool:
    """job 是不是这条一次性定时当前的注册：触发途中被改了时间就有新 job，旧 job 不能停用新的。"""
    at = job.get('at')
    if s.kind != 'once' or at is None:
        return True
    try:
        return abs(parse_at(s.at, s.tz) - at) < 1
    except ValueError:
        return False


def _disable_once(s: Schedule, why: str) -> None:
    print(f'[schedule] one-shot {s.name!r} {why}, disabled')
    s.enabled = False
    s.updated_at = time.time()
    _persist(s)


def _on_fire(sid: str):
    """推事件之前（线程池里、持锁）：还启用着就占一个当天名额，返回 True。

    名额满了返回 False 跳过这一次；一次性的错过就是错过了，当场停用落盘，不留一条
    “启用着但再也不会触发”的记录。检查和占名额在同一把锁里，几条同时到点的定时不会
    都读到“还差一次”而一起放行。"""
    def _cb(job: dict) -> bool:
        with _lock:
            s = _schedules.get(sid)
            if not s or not s.enabled:
                return False
            slot = _reserve(time.time())
            if slot is None:
                print(f'[schedule] daily cap {get_settings()["max_fires_per_day"]} reached, skip {s.name!r}')
                if s.kind == 'once' and _is_current(s, job):
                    _disable_once(s, 'skipped by the daily cap')
                return False
            job['slot'] = slot
            return True
    return _cb


def _after_fire(sid: str):
    """每次放行之后（线程池里、持锁）收尾。

    发出去了：名额转正写库，记触发次数、上次触发时间，一次性的停用落盘。
    没发出去：退回名额；一次性的如果 scheduler 不再重试（重试用完），停用落盘。"""
    def _cb(job: dict, delivered: bool = True, retrying: bool = False) -> None:
        with _lock:
            slot = job.pop('slot', None)
            s = _schedules.get(sid)
            if not delivered:
                _release(slot)
                if s and s.kind == 'once' and s.enabled and not retrying and _is_current(s, job):
                    _disable_once(s, 'could not be delivered after retries')
                return
            _commit(slot)                 # 发出去之后才被删的，这次照样算进当天次数
            if not s:
                return
            s.last_run = time.time()
            s.fire_count += 1
            if s.kind == 'once' and _is_current(s, job):
                s.enabled = False
            _persist(s)
    return _cb


def _register(s: Schedule) -> None:
    scheduler.remove_job(job_name(s.id))
    if not s.enabled:
        return
    if s.kind == 'once':
        t = parse_at(s.at, s.tz)
        if t <= time.time():
            return
        scheduler.add_job(job_name(s.id), '', event_text(s), at=t, source=f'scheduler:{s.name}',
                          on_fire=_on_fire(s.id), after_fire=_after_fire(s.id))
    else:
        scheduler.add_job(job_name(s.id), s.cron, event_text(s), tz=s.tz, source=f'scheduler:{s.name}',
                          on_fire=_on_fire(s.id), after_fire=_after_fire(s.id))


@_locked
def start_all() -> int:
    """启动时调用（scheduler.run）：从库里读出来，注册所有启用的定时任务。

    每条单独兜错：库里有一条坏数据（比如 at 写坏了）只跳过它，不能拖垮其他定时。
    """
    load_all()
    n = 0
    for s in _schedules.values():
        if not s.enabled:
            continue
        try:
            if s.kind == 'once' and parse_at(s.at, s.tz) <= time.time():
                s.enabled = False      # 关机期间错过的一次性定时：不补发，标记停用
                _persist(s)
                continue
            _register(s)
            n += 1
        except Exception as e:
            print(f'[schedule] skip {s.name!r}: {e}')
    if n:
        print(f'[schedule] started {n} runtime schedule(s)')
    return n


# ── 公开接口（都是同步的、会读写 SQLite：async 调用方要放进线程池）────────────

@_locked
def list_all() -> list[Schedule]:
    _ensure_loaded()
    return sorted(_schedules.values(), key=lambda s: (not s.enabled, s.created_at))


@_locked
def find(key: str) -> Schedule | None:
    """按 id 或名字找。"""
    _ensure_loaded()
    key = (key or '').strip()
    if key in _schedules:
        return _schedules[key]
    for s in _schedules.values():
        if s.name == key:
            return s
    return None


@_locked
def create(*, name: str, text: str, kind: str = 'cron', cron: str = '', at: str = '',
           tz: str = '', skill: str = '', enabled: bool = True, created_by: str = 'web') -> Schedule:
    _ensure_loaded()
    s = Schedule(name=name, text=text, kind=kind, cron=cron, at=at, tz=tz, skill=skill,
                 enabled=enabled, created_by=created_by if created_by in CREATORS else 'api')
    validate(s)
    _schedules[s.id] = s
    _persist(s)
    _register(s)
    return s


@_locked
def update(sid: str, **changes) -> Schedule:
    _ensure_loaded()
    s = _schedules.get(sid)
    if not s:
        raise KeyError(sid)
    trial = Schedule(**asdict(s))
    for k, v in changes.items():
        if k in ('name', 'text', 'kind', 'cron', 'at', 'tz', 'skill', 'enabled') and v is not None:
            setattr(trial, k, v)
    validate(trial)
    trial.updated_at = time.time()
    _schedules[sid] = trial
    _persist(trial)
    _register(trial)
    return trial


@_locked
def delete(sid: str) -> bool:
    _ensure_loaded()
    scheduler.remove_job(job_name(sid))
    s = _schedules.pop(sid, None)
    _delete_row(sid)
    return s is not None


@_locked
def clear_all() -> int:
    """删掉所有定时任务。当天已触发的次数不清（不能靠清空绕过每日上限）。"""
    _ensure_loaded()
    n = len(_schedules)
    for sid in list(_schedules):
        scheduler.remove_job(job_name(sid))
    _schedules.clear()
    with _db() as conn:
        conn.execute('DELETE FROM schedules')
    return n


@_locked
def model_upsert(*, name: str, text: str, kind: str, cron: str, at: str, tz: str,
                 skill: str) -> tuple[str, Schedule]:
    """模型工具用：按名字建或改它自己的定时。查重、权限判断和写入在同一把锁里。
    同名的是人建的 → PermissionError。返回 ('已创建' | '已更新', 定时任务)。"""
    _ensure_loaded()
    key = (name or '').strip()
    old = next((s for s in _schedules.values() if s.name == key), None)
    if old:
        if old.created_by != 'model':
            raise PermissionError(key)
        return '已更新', update(old.id, text=text, kind=kind, cron=cron, at=at, tz=tz,
                                skill=skill, enabled=True)
    return '已创建', create(name=name, text=text, kind=kind, cron=cron, at=at, tz=tz,
                            skill=skill, created_by='model')


@_locked
def model_delete(key: str) -> Schedule:
    """模型工具用：删它自己建的定时。找不到 → KeyError；人建的 → PermissionError。"""
    s = find(key)
    if not s:
        raise KeyError(key)
    if s.created_by != 'model':
        raise PermissionError(s.name)
    delete(s.id)
    return s


@_locked
def _snapshot(sid: str) -> Schedule | None:
    _ensure_loaded()
    s = _schedules.get(sid)
    return Schedule(**asdict(s)) if s else None


@_locked
def _snapshot_and_reserve(sid: str) -> tuple[Schedule | None, float | None]:
    s = _snapshot(sid)
    if s is None:
        return None, None
    slot = _reserve(time.time())
    if slot is None:
        raise ValueError(_cap_msg() + '，明天再试或在设置里调高上限')
    return s, slot


@_locked
def _settle(slot: float, delivered: bool) -> None:
    (_commit if delivered else _release)(slot)


async def fire_now(sid: str) -> Schedule:
    """立即触发一次（网页“试一下”用）。和到点触发一样占当天名额：满了抛 ValueError，
    没发出去就退回。不改 fire_count / last_run，也不改一次性任务的状态。

    拍快照、占名额、记账都在线程池里（要持锁、读写库），推事件在循环里。用拍下的快照
    推事件：名字和文字来自同一时刻；与删除的先后以拍快照为准。"""
    import asyncio
    import event_bus
    s, slot = await asyncio.to_thread(_snapshot_and_reserve, sid)
    if s is None:
        raise KeyError(sid)
    try:
        await event_bus.enqueue(source=f'scheduler:{s.name}', text=event_text(s))
    except Exception:
        await asyncio.to_thread(_settle, slot, False)
        raise
    await asyncio.to_thread(_settle, slot, True)
    return s


def to_dict(s: Schedule) -> dict:
    d = asdict(s)
    try:
        nxt = next_run_ts(s)
    except (ValueError, RuntimeError):
        nxt = None
    d['next_run'] = nxt
    d['next_run_text'] = fmt_ts(nxt, s.tz)
    d['last_run_text'] = fmt_ts(s.last_run, s.tz)
    return d


def describe(s: Schedule) -> str:
    when = f'cron {s.cron}' if s.kind == 'cron' else f'一次 {s.at}'
    state = '启用' if s.enabled else '停用'
    nxt = to_dict(s)['next_run_text'] or '无'
    skill = f' | 技能 {s.skill}' if s.skill else ''
    return f'[{s.id}] {s.name} | {when}（{s.tz}）| {state} | 下次 {nxt}{skill} | 文字：{s.text}'
