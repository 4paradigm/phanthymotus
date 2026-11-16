"""
api/schedules.py — 运行时定时任务 REST API（网页“历史 → 定时”页用）。

除了“立即触发”，路由都是普通 def：schedule_store 要同步读写 SQLite，写成 async def
会在事件循环线程上执行、慢盘或锁库时卡住整个 agent-core。def 路由由 FastAPI 放进线程池；
scheduler.add_job / remove_job 会自己转回事件循环线程。

增删改立即生效、不用重启；校验（最短间隔、每日上限、条数上限，以及改设置时对现有定时的
重新校验）全在 schedule_store。
出错返回 {'ok': False, 'error': '<中文原因>'}，与 api/tasks.py 的风格一致。
"""

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

import config
import scheduler
import schedule_store

router = APIRouter(prefix='/schedules', tags=['schedules'])


class ScheduleBody(BaseModel):
    name: Optional[str] = None
    text: Optional[str] = None
    kind: Optional[str] = None       # cron | once
    cron: Optional[str] = None
    at: Optional[str] = None         # YYYY-MM-DD HH:MM
    tz: Optional[str] = None
    skill: Optional[str] = None
    enabled: Optional[bool] = None


class SettingsBody(BaseModel):
    model_tools_enabled: Optional[bool] = None
    min_interval_minutes: Optional[int] = None
    max_schedules: Optional[int] = None
    max_per_skill: Optional[int] = None
    max_fires_per_day: Optional[int] = None
    default_tz: Optional[str] = None


class PreviewBody(BaseModel):
    cron: str = ''
    tz: Optional[str] = None
    count: int = 5


def _config_jobs() -> list[dict]:
    """配置项 [[scheduler]] 里的 job：只读展示（改它们仍要改配置并重启）。"""
    out = []
    for job in config.main.get('scheduler', []) or []:
        item = {k: job.get(k, '') for k in ('name', 'cron', 'text', 'tz')}
        try:
            nxt = scheduler._next_run(job.get('cron', ''), time.time(), job.get('tz') or None)
            item['next_run_text'] = schedule_store.fmt_ts(nxt, job.get('tz') or None)
        except Exception as e:
            item['next_run_text'] = ''
            item['error'] = str(e)
        out.append(item)
    return out


@router.get('')
def list_schedules():
    return {
        'schedules': [schedule_store.to_dict(s) for s in schedule_store.list_all()],
        'settings': schedule_store.get_settings(),
        'config_jobs': _config_jobs(),
    }


@router.get('/settings')
def get_settings():
    return {'ok': True, 'settings': schedule_store.get_settings()}


@router.patch('/settings')
def patch_settings(body: SettingsBody):
    try:
        st = schedule_store.update_settings(**body.dict())
    except ValueError as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'settings': st}


@router.post('/preview')
def preview(body: PreviewBody):
    """算出 cron 接下来几次触发时间，给网页表单即时提示。"""
    tz = body.tz or schedule_store.get_settings()['default_tz']
    try:
        runs = scheduler.next_runs(body.cron, time.time(), max(1, min(body.count, 20)), tz)
    except (ValueError, RuntimeError) as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'runs': [schedule_store.fmt_ts(t, tz) for t in runs]}


@router.post('')
def create_schedule(body: ScheduleBody):
    data = {k: v for k, v in body.dict().items() if v is not None}
    try:
        s = schedule_store.create(created_by='web', **data)
    except (ValueError, TypeError) as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'schedule': schedule_store.to_dict(s)}


@router.patch('/{sid}')
def update_schedule(sid: str, body: ScheduleBody):
    data = {k: v for k, v in body.dict().items() if v is not None}
    if not data:
        return {'ok': False, 'error': '没有要改的内容'}
    try:
        s = schedule_store.update(sid, **data)
    except KeyError:
        return {'ok': False, 'error': f'定时任务 {sid} 不存在'}
    except ValueError as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'schedule': schedule_store.to_dict(s)}


@router.delete('/{sid}')
def delete_schedule(sid: str):
    return {'ok': schedule_store.delete(sid)}


@router.post('/{sid}/run')
async def run_now(sid: str):
    """立即触发一次，用来试效果。停用的不能触发；离上一次、下一次触发都要隔开最短间隔；
    和到点触发一样占当天的触发名额，满了会拒绝。被拒时返回 ok=False 和中文原因。"""
    try:
        s = await schedule_store.fire_now(sid)
    except KeyError:
        return {'ok': False, 'error': f'定时任务 {sid} 不存在'}
    except ValueError as e:
        return {'ok': False, 'error': str(e)}
    except Exception as e:
        return {'ok': False, 'error': f'事件没发出去：{e}'}
    return {'ok': True, 'schedule': schedule_store.to_dict(s)}