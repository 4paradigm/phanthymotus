"""
event/task.py — 任务管理系统工具。

提供 LLM 可调用的任务生命周期管理工具。
"""

import datetime
import math
import time
import typing

import log
import task_store
import scheduler


def _elapsed_str(created_at: float) -> str:
    """格式化已用时间。"""
    elapsed = time.time() - created_at
    if elapsed < 60:
        return f'{int(elapsed)}s'
    elif elapsed < 3600:
        return f'{int(elapsed / 60)}min'
    else:
        return f'{elapsed / 3600:.1f}h'


def _register_check(task: task_store.Task) -> None:
    """为任务注册定时检查 job。"""
    if not task.check_cron:
        return
    scheduler.add_job(
        name=f'task:{task.id}',
        cron=task.check_cron,
        text=f'任务定时检查 [{task.id}]：{task.goal}。请查询实际状态并更新进展。',
    )


def _unregister_check(task_id: str) -> None:
    """移除任务的定时检查 job。"""
    scheduler.remove_job(f'task:{task_id}')


class Tools:
    # 只恢复用户明确要求的定时入口。默认不轮询，也不给模型可照抄的高频 cron 示例；
    # 子代理完成通知、超时检测和进度播报仍沿用现有机制。
    @log.function_(call=True)
    async def task_create(self,
        goal: typing.Annotated[str, '任务目标描述（如"走到B点"）'],
        check_cron: typing.Annotated[str, f'仅用户明确要求周期提醒时填写的5字段cron（分 时 日 月 周）；按本机时区 {time.strftime("%Z %z")} 计算。一次性提醒使用 delay_minutes 或 remind_at。普通任务留空，禁止为追踪进度自行设置轮询。'] = '',
        delay_minutes: typing.Annotated[float, '一次性提醒：多少分钟后执行，直接填写用户要求的分钟数；工具负责计算触发时刻，不能同时填 remind_at 或 check_cron。普通任务为0。'] = 0,
        remind_at: typing.Annotated[str, '一次性提醒：用户指定的时刻，如HH:MM或YYYY-MM-DDTHH:MM；默认北京时间，可附ISO时区偏移。直接填用户时刻，工具负责时区换算；不能同时填 delay_minutes 或 check_cron。普通任务留空。'] = '',
    ):
        """创建一个长时间任务并开始追踪。适用于预计超过30秒的动作（导航、巡逻、等待等）。

        默认只追踪，不自动定时检查。一次性报时或提醒填写 delay_minutes 或 remind_at，
        周期提醒才填写 check_cron；时间加法、分钟取整和时区转换交给工具计算。
        注册成功仅表示排期完成，当前回合只确认排期后结束，不要立即执行提醒或调用 task_done。
        等收到对应 scheduler:task:<id> 事件后才执行；一次性提醒执行后调用 task_done，避免重复。
        cron 精度为分钟，确认时使用返回的触发时间，不要承诺秒级倒计时。
        完成或失败时用 task_done / task_fail 收尾；子代理完成通知和进度播报由框架处理。
        """
        try:
            if sum(bool(v) for v in (check_cron, delay_minutes, remind_at)) > 1:
                raise ValueError('check_cron、delay_minutes、remind_at 只能填写一个')
            now = time.time()
            target = None
            if delay_minutes:
                if (isinstance(delay_minutes, bool) or not math.isfinite(delay_minutes)
                        or delay_minutes <= 0):
                    raise ValueError('delay_minutes 必须是大于0的有限分钟数')
                target = now + delay_minutes * 60
            elif remind_at:
                user_tz = datetime.timezone(datetime.timedelta(hours=8))
                try:
                    when = datetime.datetime.fromisoformat(remind_at)
                except ValueError:
                    clock = datetime.time.fromisoformat(remind_at)
                    when = datetime.datetime.combine(
                        datetime.datetime.fromtimestamp(now, user_tz).date(), clock)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=user_tz)
                    if when.timestamp() <= now:
                        when += datetime.timedelta(days=1)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=user_tz)
                target = when.timestamp()
                if target <= now:
                    raise ValueError('remind_at 指定时刻已过去')
            if target is not None:
                # 沿用现有分钟精度 cron，向上取整，不能早于用户要求的延迟。
                target = math.ceil(target / 60) * 60
                check_cron = time.strftime('%M %H %d %m *', time.localtime(target))
            next_ts = scheduler._next_run(check_cron, time.time()) if check_cron else None
        except (ValueError, RuntimeError, OverflowError) as e:
            return f'定时设置失败，未创建任务：{e}'
        task = task_store.create(goal=goal, check_cron=check_cron)
        _register_check(task)
        user_when = (datetime.datetime.fromtimestamp(next_ts, datetime.timezone(datetime.timedelta(hours=8)))
                     .strftime('%Y-%m-%d %H:%M 北京时间 UTC+08:00')) if next_ts is not None else ''
        timing = (f'；定时已注册，下次触发 {user_when}'
                  f'（调度器 {time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(next_ts))}）。'
                  f'尚未触发：本轮只确认排期并结束，等待 scheduler:task:{task.id} 事件；'
                  '不要现在执行提醒或标记完成。') if next_ts is not None else '（未设置定时）'
        return f'任务已创建：[{task.id}] {task.goal}{timing}'

    @log.function_(call=True)
    async def task_update(self,
        id: typing.Annotated[str, '任务 ID（8位短码）'],
        progress: typing.Annotated[str, '当前进展描述'] = '',
    ):
        """更新任务进展。收到任务检查事件后用此工具记录最新状态。"""
        task = task_store.update(id, progress=progress if progress else None)
        if not task:
            active = task_store.active_tasks()
            if active:
                ids = ', '.join(t.id for t in active)
                return f'任务 {id} 不存在。当前活跃任务 ID: {ids}'
            return f'任务 {id} 不存在，当前无活跃任务。'
        return f'已更新：[{task.id}] {task.progress}'

    @log.function_(call=True)
    async def task_done(self,
        id: typing.Annotated[str, '任务 ID（8位短码）'],
        summary: typing.Annotated[str, '完成总结'] = '',
    ):
        """标记任务完成。完成后定时检查自动停止。"""
        _unregister_check(id)
        task = task_store.done(id, summary=summary)
        if not task:
            active = task_store.active_tasks()
            if active:
                ids = ', '.join(t.id for t in active)
                return f'任务 {id} 不存在。当前活跃任务 ID: {ids}'
            return f'任务 {id} 不存在，当前无活跃任务。'
        return f'任务完成：[{task.id}] {task.goal}'

    @log.function_(call=True)
    async def task_fail(self,
        id: typing.Annotated[str, '任务 ID（8位短码）'],
        reason: typing.Annotated[str, '失败原因'] = '',
    ):
        """标记任务失败。失败后定时检查自动停止。"""
        _unregister_check(id)
        task = task_store.fail(id, reason=reason)
        if not task:
            active = task_store.active_tasks()
            if active:
                ids = ', '.join(t.id for t in active)
                return f'任务 {id} 不存在。当前活跃任务 ID: {ids}'
            return f'任务 {id} 不存在，当前无活跃任务。'
        return f'任务失败：[{task.id}] {reason or task.goal}'

    @log.function_(call=True)
    async def task_force_clear(self):
        """强制清除所有活跃任务及其定时检查。当任务无法正常关闭时使用此工具。"""
        tasks = task_store.active_tasks()
        if not tasks:
            return '当前没有活跃任务需要清除。'
        for t in tasks:
            _unregister_check(t.id)
            task_store.done(t.id, summary='强制清除')
        return f'已强制清除 {len(tasks)} 个任务。'

    @log.function_(call=True)
    async def task_list(self):
        """列出所有活跃任务（运行中/暂停）。"""
        tasks = task_store.active_tasks()
        if not tasks:
            return '当前没有活跃任务。'
        lines = []
        for t in tasks:
            lines.append(f'[{t.id}] {t.goal} | {t.status} | {t.progress or "无进展记录"} | {_elapsed_str(t.created_at)}')
        return '\n'.join(lines)
