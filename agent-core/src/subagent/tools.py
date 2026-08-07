"""
tools.py — System tools exposed to the main agent for subagent management.

These tools are registered in event/llm.py's _sys_tools and allow the main agent
to spawn, manage, and query subagents via its normal tool-calling interface.
"""

from __future__ import annotations
import json
import typing
from typing import Annotated

from .protocol import SubagentSpec, P_CRITICAL, P_HIGH, P_NORMAL, P_LOW
from .manager import SubagentManager


class SubagentTools:
    """Wrapper that produces tool functions bound to a SubagentManager instance."""

    def __init__(self, manager: SubagentManager):
        self._mgr = manager

    def _build_context_with_history(self, explicit_context: str, goal: str) -> str:
        """构建 subagent 初始上下文。

        注入确定性信息（任务列表）+ 最近决策摘要。
        subagent 可通过 memory_recall 自行按需检索更多历史。
        """
        sections = []

        # 1. 活跃任务列表（确定性上下文，保证任务意图传递）
        try:
            import task_store
            active = task_store.active_tasks()
            if active:
                task_lines = [f'  - [{t.id}] {t.goal}{" — " + t.progress if t.progress else ""}' for t in active]
                sections.append(f'[当前活跃任务]\n' + '\n'.join(task_lines))
        except (ImportError, AttributeError):
            pass

        # 2. 最近 10 轮 main agent 对话（足够理解上下文）
        try:
            from event.llm import get_recent_context
            recent = get_recent_context(max_turns=10)
            if recent:
                sections.append(f'[主代理最近对话]\n{recent}')
        except (ImportError, AttributeError):
            pass

        # 3. LLM 显式传入的 context
        if explicit_context:
            sections.append(f'[额外上下文]\n{explicit_context}')

        # 4. 提示 subagent 可使用 memory_recall
        sections.append('[提示] 如需更多历史信息，可使用 memory_recall 工具检索。')

        return '\n\n'.join(sections) if sections else ''

    async def subagent_spawn(
        self,
        goal: Annotated[str, "子代理的任务目标描述"],
        priority: Annotated[int, "优先级: 0=紧急 1=高 2=普通 3=低, -1=自动继承"] = -1,
        tools: Annotated[str, "工具过滤: '*'=全部, 逗号分隔的 fnmatch 模式"] = '*',
        model: Annotated[str, "LLM模型覆盖,为空则用默认模型"] = '',
        max_rounds: Annotated[int, "最大推理轮数"] = 10,
        context: Annotated[str, "传递给子代理的初始上下文信息"] = '',
    ) -> str:
        """创建子代理异步执行任务。返回 agent_id 用于后续查询。适合不需要立即结果的后台任务。"""
        # priority=-1 表示自动继承: 反转映射 turn priority → subagent priority
        if priority < 0:
            import collector
            priority = max(0, min(3, 3 - collector._current_turn_priority))
        else:
            priority = max(0, min(3, priority))
        context_seed = self._build_context_with_history(context, goal)
        tool_filter = None if tools == '*' else [t.strip() for t in tools.split(',')]

        # 默认 deny MCP 工具（subagent 通常不需要 tts/asr/channel 等硬件工具）
        # 除非 tools 参数显式包含 mcp 相关模式
        tool_deny = None
        if tools == '*' or (tool_filter and not any('mcp' in t for t in tool_filter)):
            tool_deny = ['mcp__*']

        spec = SubagentSpec(
            goal=goal,
            priority=priority,
            model=model or None,
            tool_filter=tool_filter,
            tool_deny=tool_deny,
            max_rounds=max_rounds,
            context_seed=context_seed,
        )
        agent_id = await self._mgr.spawn(spec)
        return f'子代理已创建: id={agent_id}, priority={spec.priority}, 任务: {goal[:80]}'

    async def subagent_spawn_sync(
        self,
        goal: Annotated[str, "子代理的任务目标描述"],
        priority: Annotated[int, "优先级: 0=紧急 1=高 2=普通 3=低"] = 0,
        tools: Annotated[str, "工具过滤: '*'=全部, 逗号分隔的 fnmatch 模式"] = '*',
        model: Annotated[str, "LLM模型覆盖,为空则用默认模型"] = '',
        max_rounds: Annotated[int, "最大推理轮数"] = 10,
        context: Annotated[str, "传递给子代理的初始上下文信息"] = '',
        timeout: Annotated[int, "等待超时秒数"] = 120,
    ) -> str:
        """创建子代理并等待结果返回。适合需要立即获得结果的查询类任务。会阻塞当前轮直到完成。"""
        context_seed = self._build_context_with_history(context, goal)
        tool_filter = None if tools == '*' else [t.strip() for t in tools.split(',')]

        # 默认 deny MCP 工具
        tool_deny = None
        if tools == '*' or (tool_filter and not any('mcp' in t for t in tool_filter)):
            tool_deny = ['mcp__*']

        spec = SubagentSpec(
            goal=goal,
            priority=max(0, min(3, priority)),
            model=model or None,
            tool_filter=tool_filter,
            tool_deny=tool_deny,
            max_rounds=max_rounds,
            context_seed=context_seed,
        )
        result = await self._mgr.spawn_and_wait(spec, timeout=float(timeout))
        if result.status == 'completed':
            return f'[子代理完成] {result.output}'
        else:
            return f'[子代理{result.status}] {result.error or result.output or "(无输出)"}'

    async def subagent_status(
        self,
        id: Annotated[str, "子代理 ID（留空则列出所有活跃子代理）"] = '',
    ) -> str:
        """查看子代理状态。不传 id 则列出所有活跃子代理。"""
        if id:
            status = self._mgr.get_status(id)
            if not status:
                return f'未找到子代理 {id}'
            return status.to_display()
        else:
            active = self._mgr.list_active()
            if not active:
                return '当前没有活跃的子代理。'
            lines = [s.to_display() for s in active]
            return f'活跃子代理 ({len(active)}):\n' + '\n'.join(lines)

    async def subagent_cancel(
        self,
        id: Annotated[str, "要取消的子代理 ID"],
        reason: Annotated[str, "取消原因"] = '',
    ) -> str:
        """取消一个运行中或排队中的子代理。"""
        ok = await self._mgr.cancel(id, reason)
        if ok:
            return f'子代理 {id} 已取消。'
        return f'无法取消子代理 {id}（可能已完成或不存在）。'

    async def subagent_message(
        self,
        id: Annotated[str, "目标子代理 ID"],
        text: Annotated[str, "发送给子代理的消息内容"],
    ) -> str:
        """向运行中的子代理发送消息/指令。消息会追加到其上下文中。"""
        ok = self._mgr.send_message(id, text)
        if ok:
            return f'消息已发送给子代理 {id}。'
        return f'无法发送消息（子代理 {id} 未在运行）。'

    async def subagent_result(
        self,
        id: Annotated[str, "子代理 ID"],
    ) -> str:
        """获取已完成子代理的结果。"""
        result = self._mgr.get_result(id)
        if not result:
            # Check if still active
            status = self._mgr.get_status(id)
            if status:
                return f'子代理 {id} 尚未完成 (status={status.status}, round={status.rounds_completed})'
            return f'未找到子代理 {id} 的结果。'
        parts = [f'状态: {result.status}']
        if result.output:
            parts.append(f'输出: {result.output}')
        if result.error:
            parts.append(f'错误: {result.error}')
        parts.append(f'用时: {result.duration_s:.1f}s, 轮数: {result.rounds_used}')
        return '\n'.join(parts)
