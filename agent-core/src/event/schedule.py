"""
event/schedule.py — 让模型自己管理定时任务的工具（默认关闭）。

为什么默认关：task_create 的 check_cron 参数被删过一次（见 event/task.py 的长注释）——
Tianyi 上模型几乎每次都顺手建一个每 2 分钟的检查，一小时被叫醒 12 次。这里重新开放
"模型建定时"，前提是：
  · 管理员在设置里打开 scheduler_settings.model_tools_enabled 才会出现在模型的工具表里
    （event/llm.py 每个 turn 现读配置过滤），关着时即使被调用也直接拒绝；
  · 走和网页完全相同的校验（schedule_store.validate）：最短间隔、每日触发上限、条数上限；
  · 模型只能改、删自己建的（created_by='model'），人建的只读。

下面工具的 docstring 会原样进模型的 prompt，所以只写用法，不写例子 cron（别给它现成的
高频写法照抄）。
"""

import typing

import log
import schedule_store


def _enabled() -> bool:
    return bool(schedule_store.get_settings().get('model_tools_enabled'))


_OFF = '定时工具没有开启（管理员可在设置的“定时任务”里打开）。需要定时的话，请告诉用户在网页上设置。'


class Tools:
    @log.function_(call=True)
    async def schedule_list(self):
        """列出所有定时任务（含人在网页上建的）：名字、触发时间、下次触发、触发时推给你的文字。"""
        if not _enabled():
            return _OFF
        items = schedule_store.list_all()
        if not items:
            return '当前没有定时任务。'
        return '\n'.join(schedule_store.describe(s) for s in items)

    @log.function_(call=True)
    async def schedule_create(self,
        name: typing.Annotated[str, '定时任务名字，同名会覆盖你之前建的那条'],
        text: typing.Annotated[str, '到点时推给你的事件文字，写清到点要做什么'],
        cron: typing.Annotated[str, '重复触发：5 字段 cron（分 时 日 月 周），按北京时间解释；和 at 二选一'] = '',
        at: typing.Annotated[str, '只触发一次：YYYY-MM-DD HH:MM（北京时间）；和 cron 二选一'] = '',
        skill: typing.Annotated[str, '到点要用的技能 slug（可空），事件里会提示先激活它'] = '',
    ):
        """建一个定时任务：到点时系统会把 text 作为事件推给你，叫醒你去做事。

        只在用户或当前技能指令明确要求定时（例如“每天 8 点提醒吃药”）时使用，不要为了
        “过一会儿检查一下进度”而建。每次触发都会让你完整跑一轮，花费很大；系统有最短间隔
        和每天触发次数上限，超了会拒绝。建好后把触发时间复述给用户。
        """
        if not _enabled():
            return _OFF
        if bool(cron) == bool(at):
            return '要么写 cron（重复），要么写 at（一次），两个只能写一个。'
        tz = schedule_store.get_settings()['default_tz']
        kind = 'once' if at else 'cron'
        old = schedule_store.find(name)
        try:
            if old and old.name == name:
                if old.created_by != 'model':
                    return f'「{name}」是人在网页上建的，你不能改。换个名字，或请用户在网页上改。'
                s = schedule_store.update(old.id, text=text, kind=kind, cron=cron, at=at,
                                          tz=tz, skill=skill, enabled=True)
                verb = '已更新'
            else:
                s = schedule_store.create(name=name, text=text, kind=kind, cron=cron, at=at,
                                          tz=tz, skill=skill, created_by='model')
                verb = '已创建'
        except ValueError as e:
            return f'没有建成：{e}'
        return f'{verb}：' + schedule_store.describe(s)

    @log.function_(call=True)
    async def schedule_delete(self,
        id: typing.Annotated[str, '定时任务的 id 或名字'],
    ):
        """删除一个你自己建的定时任务。人在网页上建的不能删。"""
        if not _enabled():
            return _OFF
        s = schedule_store.find(id)
        if not s:
            names = '、'.join(x.name for x in schedule_store.list_all()) or '无'
            return f'没有找到定时任务 {id}。现有：{names}'
        if s.created_by != 'model':
            return f'「{s.name}」是人在网页上建的，你不能删。请用户在网页上处理。'
        schedule_store.delete(s.id)
        return f'已删除定时任务「{s.name}」。'