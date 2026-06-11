"""
event/task.py — 任务管理系统工具。

提供 LLM 可调用的任务生命周期管理工具。
"""

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
    @log.function_(call=True)
    async def task_create(self,
        goal: typing.Annotated[str, '任务目标描述（如"走到B点"）'],
        check_cron: typing.Annotated[str, '定时检查 cron 表达式（如 "*/2 * * * *" 每2分钟），留空则不自动检查'] = '',
    ):
        """创建一个长时间任务并开始追踪。适用于预计超过30秒的动作（导航、巡逻、等待等）。"""
        task = task_store.create(goal=goal, check_cron=check_cron)
        _register_check(task)
        return f'任务已创建：[{task.id}] {task.goal}'

    @log.function_(call=True)
    async def task_update(self,
        id: typing.Annotated[str, '任务 ID（8位短码）'],
        progress: typing.Annotated[str, '当前进展描述'] = '',
    ):
        """更新任务进展。收到任务检查事件后用此工具记录最新状态。"""
        task = task_store.update(id, progress=progress if progress else None)
        if not task:
            return f'任务 {id} 不存在'
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
            return f'任务 {id} 不存在'
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
            return f'任务 {id} 不存在'
        return f'任务失败：[{task.id}] {reason or task.goal}'

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
