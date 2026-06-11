"""
prompt.py — 分层 Prompt 构建器。

四层结构（参考 Claude Code 设计，适配具身智能场景）：

L1  系统定义       从 prompt_system.md 读取，基本不变。
                   身份 / IO 规则 / 安全围栏 / 工具约定 / 何时 finish。

L2  环境快照       每轮调用 LLM 前动态生成。
                   当前时间、已注册 MCP（在线/离线/工具列表）、最近事件来源统计。

L3  对话历史       message_list（含工具调用及结果），由调用方传入。
                   必要时由外层做 trim/compaction（暂留给 llm.py 处理）。

L4  当前触发       本轮触发该次推理的事件，用 XML 格式标注来源和时间。
                   始终作为最后一条 user 消息。
"""

import datetime
import pathlib

import config
import event_bus


# ── L1 ────────────────────────────────────────────────────────────────────────

def _system_definition() -> str:
    """读取 L1 base prompt（含 system prompt + 身份定义 + 长期记忆）。"""
    system = pathlib.Path(config.main['event']['llm']['prompt_system']).read_text()

    # 身份定义（不可由 LLM 修改，仅通过前端编辑）
    identity_path = pathlib.Path('./resource/memory/identity.md')
    identity = identity_path.read_text() if identity_path.exists() else ''

    memory = pathlib.Path(config.main['event']['llm']['prompt_memory']).read_text()

    parts = [system]
    if identity.strip():
        parts.append("\n\n---以下是你的身份定义（不可修改）---\n\n" + identity)
    parts.append("\n\n---以下是你的长期记忆，可通过记忆工具修改---\n\n" + memory)
    return ''.join(parts)


# ── L2 ────────────────────────────────────────────────────────────────────────

def _env_snapshot(mcp_registry: dict, bound_tools: set | None = None) -> str:
    """生成 L2 环境快照（每轮重建）。

    Args:
        mcp_registry: MCP 注册表
        bound_tools: 已绑定的工具全名集合 (如 {'mcp__id__tool'})，
                     若为 None 则显示所有工具，否则只显示绑定的。
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 设备列表（只显示有绑定工具的设备）
    device_lines = []
    for mcp_id, info in mcp_registry.items():
        online  = "true" if info.get('online') else "false"
        render  = info.get('render_hint', '')
        name    = info.get('name', mcp_id)

        # 只列出绑定的工具
        if bound_tools is not None:
            visible = [t for t in info.get('tools', []) if f'mcp__{mcp_id}__{t}' in bound_tools]
        else:
            visible = info.get('tools', [])

        # 没有绑定工具的设备不显示
        if not visible:
            continue

        # 构建带描述的工具列表
        schemas = info.get('schemas', {})
        tool_descs = []
        for t in visible:
            full_name = f'mcp__{mcp_id}__{t}'
            desc = schemas.get(full_name, {}).get('description', '')
            if desc:
                tool_descs.append(f'        - {full_name}: {desc}')
            else:
                tool_descs.append(f'        - {full_name}')
        tools_block = '\n'.join(tool_descs)

        device_lines.append(
            f'    <device id="{mcp_id}" online="{online}" render="{render}">\n'
            f'      {name}\n'
            f'      <tools>\n{tools_block}\n      </tools>\n'
            f'    </device>'
        )
    devices_xml = '\n'.join(device_lines) if device_lines else '    (无已注册设备)'

    # 最近事件来源统计
    recents = event_bus.recent(10)
    if recents:
        from collections import Counter
        counts = Counter(e['source'] for e in recents)
        recent_str = ', '.join(f'{src} ×{n}' for src, n in counts.most_common())
    else:
        recent_str = '暂无历史事件'

    # 活跃任务
    import task_store
    import time as _time
    active = task_store.active_tasks()
    if active:
        task_lines = []
        for t in active:
            elapsed = _time.time() - t.created_at
            if elapsed < 60:
                elapsed_str = f'{int(elapsed)}s'
            elif elapsed < 3600:
                elapsed_str = f'{int(elapsed / 60)}min'
            else:
                elapsed_str = f'{elapsed / 3600:.1f}h'
            task_lines.append(f'    <task id="{t.id}" status="{t.status}" elapsed="{elapsed_str}">{t.goal}{" — " + t.progress if t.progress else ""}</task>')
        tasks_xml = '\n'.join(task_lines)
    else:
        tasks_xml = ''

    tasks_section = f'\n  <active_tasks>\n{tasks_xml}\n  </active_tasks>\n' if active else ''

    # 技能列表（混合模式：仅展示 UI 激活的技能，LLM 按需加载 instruction）
    import sys, event.skills
    skills_mod = sys.modules['event.skills']
    visible = skills_mod.visible_skills()
    skills_section = ''
    if visible:
        skill_lines = []
        for s in visible:
            skill_lines.append(f'    <skill slug="{s["slug"]}">{s["name"]} — {s["oneLiner"]}</skill>')
        skills_xml = '\n'.join(skill_lines)

        # 已激活技能的完整指令
        active_skill_lines = []
        for s in skills_mod.get_active_skills():
            active_skill_lines.append(
                f'    <skill_instruction slug="{s["slug"]}">\n'
                f'      {s["instruction"]}\n'
                f'    </skill_instruction>'
            )
        active_skills_xml = '\n'.join(active_skill_lines)

        skills_section = f'  <skills>\n{skills_xml}\n  </skills>\n'
        if active_skill_lines:
            skills_section += f'  <active_skills>\n{active_skills_xml}\n  </active_skills>\n'

    return (
        f'<environment>\n'
        f'  <time>{now}</time>\n'
        f'  <devices>\n{devices_xml}\n  </devices>\n'
        f'  <recent_sources>最近 {len(recents)} 条事件来自: {recent_str}</recent_sources>\n'
        f'{tasks_section}'
        f'{skills_section}'
        f'</environment>'
    )


# ── L4 ────────────────────────────────────────────────────────────────────────

def _trigger_message(event: dict) -> str:
    """把触发事件格式化为 L4 触发消息（user 角色）。

    如果来源是 collector，text 已经是批量格式化的 XML，直接使用。
    否则按单事件格式化。
    """
    if event.get('source') == 'collector':
        return event['text']
    ts  = datetime.datetime.fromtimestamp(event['ts']).strftime('%Y-%m-%dT%H:%M:%S')
    src = event['source']
    txt = event['text']
    return f'<event source="{src}" ts="{ts}">\n{txt}\n</event>'


# ── 公共入口 ──────────────────────────────────────────────────────────────────

def build(
    message_list:  list[dict],
    trigger_event: dict,
    mcp_registry:  dict,
    bound_tools:   set | None = None,
) -> list[dict]:
    """
    返回完整的 messages 列表，可直接传给 client.llm()。

    参数：
        message_list:  L3 历史（已经过 sanitize/trim），不含本轮 trigger。
        trigger_event: L4 触发事件（从 collector 取到的 dict）。
        mcp_registry:  {mcp_id: {name, online, tools, render_hint}} 的快照。
        bound_tools:   已绑定工具全名集合，用于过滤 L2 中展示的工具。
    """
    return [
        # L1 + L2 合并为单条 system（部分模型如 Qwen 要求 system 仅出现一次且在开头）
        {'role': 'system', 'content': _system_definition() + '\n\n' + _env_snapshot(mcp_registry, bound_tools)},

        # L3 — 历史对话（可能为空）
        *message_list,

        # L4 — 本轮触发事件
        {'role': 'user', 'content': _trigger_message(trigger_event)},
    ]


def build_continuation(
    message_list: list[dict],
    mcp_registry: dict,
    bound_tools:  set | None = None,
) -> list[dict]:
    """
    后续轮次（工具调用后继续推理）：不加新 user message，
    LLM 直接基于 tool result 做下一步决策。
    """
    return [
        # L1 + L2 合并为单条 system
        {'role': 'system', 'content': _system_definition() + '\n\n' + _env_snapshot(mcp_registry, bound_tools)},

        # L3 — 完整历史（含本轮 trigger + assistant + tool results）
        *message_list,
    ]
