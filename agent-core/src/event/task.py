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
    # task_create 原来还有一个 check_cron 参数，已删。
    #
    # 注意：下面这些 docstring 会**原样送进模型的 prompt**（_build_system_tools 拿
    # fn.__doc__ 当 tool description），所以设计理由只能写在这种注释里 —— 写进 docstring
    # 等于把刚拿掉的东西又塞回模型眼前。这条是写测试时才发现的：解释性 docstring 里带着
    # "*/2 * * * *"，测试直接判它还在 prompt 里。
    #
    # 为什么删：整份 prompt 里没有任何一句话让模型去建定时检查，唯一的驱动就是那个参数
    # 描述本身 —— 它给了现成的 "*/2 * * * *" 可抄，又把"留空"写成"则不自动检查"（听起来
    # 像放弃了什么）。Tianyi 实测模型基本每次都设，于是每 2 分钟被叫醒一次去
    # subagent_status + task_update，一次两轮、每轮三万多 token，一小时 12 次。
    #
    # 它想解决的三件事现在都有人管：子代理干完会自己推 URGENT 通知
    # （subagent/manager.py 的 _notify_completion，实测 160ms 内到）；卡死有
    # _timeout_watchdog 兜；用户那边的进度由框架按沉默时长自动播报（event/llm.py）。
    # 而写进去的 progress 只被环境快照读回来给模型自己看 —— 花两轮 LLM 调用写一句自己
    # 刚说过的话。
    #
    # 能力本身保留，只是不再由模型来建：设置页改任务（api/tasks.py）和方案包声明的任务
    # （api/solutions.py）仍然可以带 cron，那些是人明确要的；_register_check 也留着，
    # 重启时要靠它把那些检查恢复回来（event/llm.py 的启动恢复段）。
    @log.function_(call=True)
    async def task_create(self,
        goal: typing.Annotated[str, '任务目标描述（如"走到B点"）'],
    ):
        """创建一个长时间任务并开始追踪。适用于预计超过30秒的动作（导航、巡逻、等待等）。

        纯追踪：任务会出现在你的环境快照里提醒你还有事没完，完成或失败时用 task_done /
        task_fail 收尾。不会定期叫醒你查进度 —— 子代理干完会自动通知你，进度也会自动
        播报给用户。
        """
        task = task_store.create(goal=goal)
        return f'任务已创建：[{task.id}] {task.goal}'

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
