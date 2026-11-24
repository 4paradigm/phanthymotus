"""
event/task.py — 任务管理系统工具。

提供 LLM 可调用的任务生命周期管理工具。
"""

import time
import typing

import log
import schedules
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
    """为任务注册定时检查 job；定时事项（schedule_create 建的）交给 schedules。"""
    if schedules.is_schedule(task):
        schedules.register(task)
        return
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


def _plain_tasks() -> list[task_store.Task]:
    return [t for t in task_store.active_tasks() if not schedules.is_schedule(t)]


def _refuse_schedule(task_id: str) -> str:
    # 周期事项做完一次就 task_done，等于把用户的「每天巡逻」删了 —— prompt 里"任务完成后
    # 及时 task_done"会把模型往这上面推。
    task = task_store.get(task_id)
    if task and schedules.is_schedule(task):
        return f'[{task_id}] 是定时事项，不用 task_done/task_fail；要取消用 schedule_cancel。'
    return ''


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
    #
    # 删完之后"一分钟后提醒我""每天晚上巡逻"就没有工具能做了，模型只能口头答应。用户
    # 明确要的定时由下面的 schedule_* 承接 —— 它和 task_create 分开（建任务时顺手带定时
    # 正是当初的触发路径），也不收 cron 字符串，只收结构化的时刻，最短周期一天。见
    # schedules.py。
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
        if (refused := _refuse_schedule(id)):
            return refused
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
        if (refused := _refuse_schedule(id)):
            return refused
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
        """强制清除所有活跃任务及其定时检查。当任务无法正常关闭时使用此工具。不影响 schedule_create 建的定时事项。"""
        tasks = _plain_tasks()
        if not tasks:
            return '当前没有活跃任务需要清除。'
        for t in tasks:
            _unregister_check(t.id)
            task_store.done(t.id, summary='强制清除')
        return f'已强制清除 {len(tasks)} 个任务。'

    @log.function_(call=True)
    async def task_list(self):
        """列出所有活跃任务（运行中/暂停）。定时事项用 schedule_list 看。"""
        tasks = _plain_tasks()
        if not tasks:
            return '当前没有活跃任务。'
        lines = []
        for t in tasks:
            lines.append(f'[{t.id}] {t.goal} | {t.status} | {t.progress or "无进展记录"} | {_elapsed_str(t.created_at)}')
        return '\n'.join(lines)

    @log.function_(call=True)
    async def schedule_create(self,
        text: typing.Annotated[str, '到点要做的事（如"报一下时间""巡逻一圈"）'],
        delay_minutes: typing.Annotated[float, '一次性：用户说的"几分钟后"，直接填分钟数'] = 0,
        at: typing.Annotated[str, '用户说的时刻，北京时间，HH:MM 或 YYYY-MM-DD HH:MM；周期事项只填 HH:MM'] = '',
        repeat: typing.Annotated[str, '留空=一次性；daily / weekdays / weekends，或 mon..sun 的逗号列表（如 "mon,wed"）'] = '',
    ):
        """用户要求将来某个时刻、或每天/每周固定时刻做某件事时使用（提醒、报时、定时巡逻）。

        到点时你会收到 scheduler:task:<id> 事件，那时再去做；现在只需告诉用户约好的时间。
        时间换算交给工具：直接填用户说的分钟数或钟点。一次性事项触发后自动结束。
        """
        try:
            spec = schedules.build(delay_minutes, at, repeat, time.time())
            task, created = schedules.create(text, spec)
        except (ValueError, RuntimeError, OverflowError) as e:
            return f'没有设定成功：{e}'
        next_at = schedules.fmt(schedules.next_fire(spec))
        head = '已设定' if created else '已有同样的定时事项，未重复创建'
        return (f'{head}：[{task.id}] {task.goal}，{schedules.describe(spec)}，'
                f'下次 {next_at}（北京时间）。还没到点，现在不要去做。')

    @log.function_(call=True)
    async def schedule_list(self):
        """列出用户设定的定时事项。"""
        items = schedules.active()
        if not items:
            return '当前没有定时事项。'
        return '\n'.join(
            f'[{t.id}] {t.goal} | {schedules.describe(t.metadata["schedule"])} | '
            f'下次 {schedules.fmt(schedules.next_fire(t.metadata["schedule"]))}'
            for t in items)

    @log.function_(call=True)
    async def schedule_cancel(self,
        id: typing.Annotated[str, '定时事项 ID（8位短码）'],
    ):
        """取消一条定时事项。"""
        task = schedules.cancel(id)
        if not task:
            ids = ', '.join(t.id for t in schedules.active())
            return f'定时事项 {id} 不存在。' + (f'当前有: {ids}' if ids else '当前没有定时事项。')
        return f'已取消：[{task.id}] {task.goal}'
