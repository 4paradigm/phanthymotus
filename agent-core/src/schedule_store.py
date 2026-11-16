"""
schedule_store.py — 运行时定时任务：存储、校验、注册到 scheduler。

和配置项 [[scheduler]] 的区别：这里的定时任务在网页 / API（以及管理员打开后的模型工具）
里增删改，**立即生效、不用重启**，存在 SQLite（与 config.py 共用 DB_PATH）里跨重启保留。

每次触发都会叫醒主模型跑一轮（Tianyi 实测一轮三万多 token），所以这里管着几道闸，
数值在 config.main['scheduler_settings']，设置页可改：
  · min_interval_minutes —— 同一个定时任务两次触发的最短间隔；
  · max_fires_per_day    —— 所有运行时定时任务加起来一天最多触发几次（建的时候按 cron 估算
                            拦一次，运行时再按实际触发数兜底一次）；
  · max_schedules / max_per_skill —— 条数上限；
  · model_tools_enabled  —— 模型能不能自己建定时（默认关，见 event/schedule.py）。

job 名统一是 'sched:<id>'，事件 source 是 'scheduler:<name>'。
"""

from __future__ import annotations

import datetime
import json
import sqlite3
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
_fires_by_day: dict[str, int] = {}     # 'YYYY-MM-DD'（default_tz）→ 当天实际触发数
_loaded = False


# ── 设置 ─────────────────────────────────────────────────────────────────────

def get_settings() -> dict:
    saved = config.main.get('scheduler_settings', {}) or {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def update_settings(**changes) -> dict:
    cur = get_settings()
    for k, v in changes.items():
        if k not in DEFAULT_SETTINGS or v is None:
            continue
        if k == 'model_tools_enabled':
            cur[k] = bool(v)
        elif k == 'default_tz':
            scheduler.get_tz(v)          # 未知时区直接报错
            cur[k] = v
        else:
            v = int(v)
            if v < (1 if k != 'max_per_skill' else 0):
                raise ValueError(f'{k} 必须是正整数')
            cur[k] = v
    config.main['scheduler_settings'] = cur
    return cur


# ── SQLite ──────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    import pathlib
    db_path = pathlib.Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute('CREATE TABLE IF NOT EXISTS schedules (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
    conn.commit()
    return conn


def _persist(s: Schedule) -> None:
    with _get_conn() as conn:
        conn.execute('INSERT OR REPLACE INTO schedules (id, data) VALUES (?, ?)',
                     (s.id, json.dumps(asdict(s), ensure_ascii=False)))
        conn.commit()


def _delete_row(sid: str) -> None:
    with _get_conn() as conn:
        conn.execute('DELETE FROM schedules WHERE id = ?', (sid,))
        conn.commit()


def load_all() -> None:
    global _loaded
    with _get_conn() as conn:
        rows = conn.execute('SELECT data FROM schedules').fetchall()
    _schedules.clear()
    for (data,) in rows:
        d = json.loads(data)
        s = Schedule(**{k: v for k, v in d.items() if k in Schedule.__dataclass_fields__})
        _schedules[s.id] = s
    _loaded = True


def _ensure_loaded() -> None:
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


def _daily_estimate(s: Schedule, now: float) -> int:
    """按 cron 估算这个定时任务一天最多触发几次（看接下来 7 天里最多的一天）。"""
    if s.kind == 'once' or not s.enabled:
        return 1 if (s.enabled and s.kind == 'once') else 0
    tz = scheduler.get_tz(s.tz)
    horizon = now + 7 * 86400
    per_day: dict[str, int] = {}
    t = now
    for _ in range(2000):
        t = scheduler._next_run(s.cron, t, tz)
        if t > horizon:
            break
        day = datetime.datetime.fromtimestamp(t, tz).strftime('%Y-%m-%d')
        per_day[day] = per_day.get(day, 0) + 1
    return max(per_day.values()) if per_day else 0


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
        runs = scheduler.next_runs(s.cron, now, 50, tz)
        min_gap = min((b - a for a, b in zip(runs, runs[1:])), default=None)
        if min_gap is not None and min_gap < st['min_interval_minutes'] * 60:
            raise ValueError(
                f'两次触发最短只隔 {int(min_gap // 60)} 分钟，低于下限 {st["min_interval_minutes"]} 分钟。'
                '每次触发都要跑一轮模型，请把间隔放宽')

    if not s.enabled:
        return
    others = [o for o in _schedules.values() if o.enabled and o.id != s.id]
    if len(others) + 1 > st['max_schedules']:
        raise ValueError(f'启用中的定时任务最多 {st["max_schedules"]} 条')
    if s.skill and st['max_per_skill'] and \
            sum(1 for o in others if o.skill == s.skill) + 1 > st['max_per_skill']:
        raise ValueError(f'技能 {s.skill} 的定时任务最多 {st["max_per_skill"]} 条')
    total = _daily_estimate(s, now) + sum(_daily_estimate(o, now) for o in others)
    if total > st['max_fires_per_day']:
        raise ValueError(
            f'加上这条，所有定时任务一天最多会触发 {total} 次，超过上限 {st["max_fires_per_day"]} 次')


# ── 注册到 scheduler ─────────────────────────────────────────────────────────

def job_name(sid: str) -> str:
    return f'sched:{sid}'


def event_text(s: Schedule) -> str:
    hint = f'（定时任务「{s.name}」'
    if s.skill:
        hint += f'，对应技能 {s.skill}：先调 activate_skill(slug="{s.skill}")'
    return f'{s.text}\n{hint}）'


def _on_fire(sid: str):
    def _cb(_job: dict) -> bool:
        s = _schedules.get(sid)
        if not s or not s.enabled:
            return False
        st = get_settings()
        day = fmt_ts(time.time(), st['default_tz'])[:10]
        if _fires_by_day.get(day, 0) >= st['max_fires_per_day']:
            print(f'[schedule] daily cap {st["max_fires_per_day"]} reached, skip {s.name!r}')
            return False
        if day not in _fires_by_day:
            _fires_by_day.clear()      # 只留当天的计数
        _fires_by_day[day] = _fires_by_day.get(day, 0) + 1
        s.last_run = time.time()
        s.fire_count += 1
        if s.kind == 'once':
            s.enabled = False
        _persist(s)
        return True
    return _cb


def _register(s: Schedule) -> None:
    scheduler.remove_job(job_name(s.id))
    if not s.enabled:
        return
    if s.kind == 'once':
        t = parse_at(s.at, s.tz)
        if t <= time.time():
            return
        scheduler.add_job(job_name(s.id), '', event_text(s), at=t,
                          source=f'scheduler:{s.name}', on_fire=_on_fire(s.id))
    else:
        scheduler.add_job(job_name(s.id), s.cron, event_text(s), tz=s.tz,
                          source=f'scheduler:{s.name}', on_fire=_on_fire(s.id))


def start_all() -> int:
    """启动时调用（scheduler.run）：从库里读出来，注册所有启用的定时任务。"""
    load_all()
    n = 0
    for s in _schedules.values():
        if s.enabled and s.kind == 'once' and parse_at(s.at, s.tz) <= time.time():
            s.enabled = False          # 关机期间错过的一次性定时：不补发，标记停用
            _persist(s)
            continue
        if s.enabled:
            try:
                _register(s)
                n += 1
            except ValueError as e:
                print(f'[schedule] skip {s.name!r}: {e}')
    if n:
        print(f'[schedule] started {n} runtime schedule(s)')
    return n


# ── 公开接口 ─────────────────────────────────────────────────────────────────

def list_all() -> list[Schedule]:
    _ensure_loaded()
    return sorted(_schedules.values(), key=lambda s: (not s.enabled, s.created_at))


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


def delete(sid: str) -> bool:
    _ensure_loaded()
    scheduler.remove_job(job_name(sid))
    s = _schedules.pop(sid, None)
    _delete_row(sid)
    return s is not None


def clear_all() -> int:
    _ensure_loaded()
    n = len(_schedules)
    for sid in list(_schedules):
        scheduler.remove_job(job_name(sid))
    _schedules.clear()
    with _get_conn() as conn:
        conn.execute('DELETE FROM schedules')
        conn.commit()
    return n


async def fire_now(sid: str) -> Schedule:
    """立即触发一次（网页“试一下”用），不计入每日上限、不改一次性任务的状态。"""
    import event_bus
    _ensure_loaded()
    s = _schedules.get(sid)
    if not s:
        raise KeyError(sid)
    await event_bus.enqueue(source=f'scheduler:{s.name}', text=event_text(s))
    return s


def to_dict(s: Schedule) -> dict:
    d = asdict(s)
    try:
        nxt = next_run_ts(s)
    except ValueError:
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