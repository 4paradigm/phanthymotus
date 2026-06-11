"""
event/skills.py — 技能系统（混合模式）。

两层状态：
  1. DB `active` 字段 — UI 控制技能对 LLM 的可见性（出现在 <skills> 列表中）
  2. 内存 `_runtime_activated` — LLM 调用 activate_skill 后才注入完整 instruction

提供 activate_skill / deactivate_skill 系统工具。
"""

import typing

import config


# ── 运行时状态 ─────────────────────────────────────────────────────────────────

# LLM 按需激活的 slugs（内存态，重启清空）
_runtime_activated: set[str] = set()


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
        return f'已激活技能「{skill["name"]}」。完整指令将在下一轮推理中可见。'

    async def deactivate_skill(self,
        slug: typing.Annotated[str, '要停用的技能 slug'],
    ):
        """停用一个已激活的技能，从上下文中移除其指令以节省空间。当你不再需要某个技能时调用。"""
        if slug not in _runtime_activated:
            return f'技能 "{slug}" 未处于激活状态。'
        _runtime_activated.discard(slug)
        return f'已停用技能「{slug}」，其指令已从上下文移除。'
