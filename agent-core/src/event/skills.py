"""
event/skills.py — 技能系统（混合模式）。

两层状态：
  1. DB `active` 字段 — UI 控制技能对 LLM 的可见性（出现在 <skills> 列表中）
  2. 内存 `_runtime_activated` — LLM 调用 activate_skill 后才注入完整 instruction

提供 activate_skill / deactivate_skill / set_auto_narration / set_progress_report 系统工具。
"""

import typing

import config


# ── 运行时状态 ─────────────────────────────────────────────────────────────────

# LLM 按需激活的 slugs（内存态，重启清空）
_runtime_activated: set[str] = set()

# 自动播报运行时覆盖（内存态，重启清空）。None = 跟随全局配置 event.llm.auto_narration。
# 唯一写者是 set_auto_narration 工具 —— 技能不再预先声明要不要播报（原先的
# narrationDefault + _recompute_notify_override 已删除）：播不播由 agent-core 在运行时
# 自己判断，而技能切换会把模型刚设好的状态冲掉，本身就是个 bug。
_notify_override: bool | None = None

# 主动进展汇报节奏的运行时覆盖（内存态，重启清空）。None = 跟随 DB 配置
# event.llm.narration_silence_seconds。
#
# 刻意不写 DB，三个理由：与 _notify_override 对称（同一类口头指令）；"别老打断我"
# 说的是这个任务而不是长期方针；以及 ConfigDB 没有嵌套 setter，工具写入要整块
# read-modify-write `event` 这一个 key，而 api/mcp_manage.py 的 config 分支和
# topic_subscriber 改的是同一个 key 且无锁 —— 那几个写者都是人力节奏，一个 LLM 每轮
# 都能调的工具不是。持久设置的正规入口仍然是 decision_core 卡片。
_report_override: dict | None = None      # {'seconds': int}


def get_notify_override() -> bool | None:
    return _notify_override


def get_report_override() -> dict | None:
    return _report_override


def installed_skills() -> list[dict]:
    """获取已安装技能列表。"""
    return config.main.get('skills', {}).get('installed', [])


def visible_skills() -> list[dict]:
    """获取对 LLM 可见的技能（UI 激活的）。"""
    return [s for s in installed_skills() if s.get('active', False)]


def get_active_skills() -> list[dict]:
    """获取 LLM 已加载的技能（可见 + runtime activated，含完整 instruction）。"""
    return [s for s in installed_skills()
            if s.get('active', False) and s['slug'] in _runtime_activated]


# ── DB 激活状态操作（供 API 调用） ──────────────────────────────────────────────

class active_skills:
    """兼容旧 API 调用的命名空间（操作 DB active 字段）。"""

    @staticmethod
    def add(slug):
        skills_cfg = config.main.get('skills', {'installed': []})
        for s in skills_cfg['installed']:
            if s['slug'] == slug:
                s['active'] = True
                break
        config.main['skills'] = skills_cfg

    @staticmethod
    def discard(slug):
        skills_cfg = config.main.get('skills', {'installed': []})
        for s in skills_cfg['installed']:
            if s['slug'] == slug:
                s['active'] = False
                break
        config.main['skills'] = skills_cfg
        # 同时从 runtime 移除
        _runtime_activated.discard(slug)


# ── 系统工具 ───────────────────────────────────────────────────────────────────

class Tools:
    async def activate_skill(self,
        slug: typing.Annotated[str, '要激活的技能 slug（从 <skills> 列表中选择）'],
    ):
        """激活一个技能，将其完整指令注入上下文。当你需要使用某个技能的详细步骤时调用。"""
        avail = visible_skills()
        skill = next((s for s in avail if s['slug'] == slug), None)
        if not skill:
            return f'技能 "{slug}" 不可用。可用技能: {", ".join(s["slug"] for s in avail)}'
        _runtime_activated.add(slug)
        return f'已激活技能「{skill["name"]}」。完整指令已注入，请立即根据指令执行任务，不要 finish。'

    async def deactivate_skill(self,
        slug: typing.Annotated[str, '要停用的技能 slug'],
    ):
        """停用一个已激活的技能，从上下文中移除其指令以节省空间。当你不再需要某个技能时调用。"""
        if slug not in _runtime_activated:
            return f'技能 "{slug}" 未处于激活状态。'
        _runtime_activated.discard(slug)
        return f'已停用技能「{slug}」，其指令已从上下文移除。'

    async def set_auto_narration(self,
        enabled: typing.Annotated[bool, '是否允许系统在你长时间不出声时自动替你播报进展'],
    ):
        """临时开启/关闭系统的自动进展播报。默认开启；进入不希望每步都被听到/看到的
        场景（下棋、表演、需要沉浸感的角色扮演）前调用 false，结束后调用 true 恢复。

        关掉之后系统不会再替你说任何话，但你自己调播报工具说的仍然会出声。"""
        global _notify_override
        _notify_override = bool(enabled)
        return f'自动播报已{"开启" if enabled else "关闭"}。'

    async def set_progress_report(self,
        seconds: typing.Annotated[int, '距上次说话多少秒就自动汇报一次进展。0 = 关闭自动汇报，-1 = 保持不变'] = -1,
        restore_default: typing.Annotated[bool, '恢复默认节奏，忽略上面的参数'] = False,
    ):
        """调整主动进展汇报的节奏。长任务里你长时间不说话时，系统会自动替你生成并播报
        一句进展汇报；这个工具改的是"多久算长时间不说话"。

        什么时候调用：用户口头提了意见时。
        - "多汇报一点" / "我不知道你在干嘛" → 调小，如 set_progress_report(seconds=15)
        - "别老打断我" / "太吵了" → 调大，如 set_progress_report(seconds=120)
        - "别自动汇报了" → set_progress_report(seconds=0)
        - "恢复正常" → set_progress_report(restore_default=True)

        计时从你**说完话**那一刻开始算，说话过程本身不计入，所以一段很长的播报不会刚说完
        就又触发一次。设置在本次运行内有效，重启后回到默认。

        注意这只改节奏、不改开关：用户要的是"一句话都别说"（下棋、表演、沉浸式角色扮演）时，
        用 set_auto_narration(false)。
        """
        global _report_override
        import sys as _sys
        _ell = _sys.modules['event.llm']
        if restore_default:
            _report_override = None
            _ell._restart_countdown()
            return '主动汇报节奏已恢复默认（跟随设置页的配置）。'
        if seconds == -1:
            return '没有改动任何设置（seconds 是 -1）。要恢复默认请传 restore_default=true。'
        _report_override = {'seconds': max(0, seconds)}
        _, s_eff = _ell._narration_thresholds()
        if s_eff <= 0:
            # 关掉之后没有任何事件会再启动计时，必须当场停掉正在跑的那个。
            _ell._stop_countdown()
            return '主动汇报已关闭，之后不会再自动替你播报进展。'
        # 改成非 0 时同样要当场重新计时：如果刚才是关闭状态，没有别的事件会来启动它。
        _ell._restart_countdown()
        return f'主动汇报节奏已调整为：{s_eff} 秒不说话就自动汇报一次。'
