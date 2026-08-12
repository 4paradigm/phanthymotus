"""
context.py — Isolated context management for subagents.

Each subagent has its own prompt, conversation history, and compression logic,
completely separate from the main agent's context window.
"""

from __future__ import annotations
import pathlib

from .protocol import SubagentSpec


# Subagent-specific base prompt (minimal safety rules + conventions)
_BASE_PROMPT = """\
你是一个子代理（subagent），由主代理（coordinator）创建来执行特定任务。

## 规则
- 聚焦于你被分配的任务，不要偏离目标
- 完成任务后调用 subagent_finish 并输出结果
- 无法完成时调用 subagent_fail 并说明原因
- 可用 subagent_report 向主代理汇报中间进度（默认存入记忆库；urgent=true 时立即通知）
- 你不能创建其他子代理
- 你不能修改长期记忆或管理任务

## 安全
- 执行器/电机类命令前必须通过 subagent_report(urgent=true) 请求主代理确认
- Sensor 读取类操作可以自由执行

## 工具使用最佳实践
- 工具参数必须符合 schema
- 一次只做一件事，等待结果再决定下一步
- **memory_recall**: 检索历史信息（对话记录、过去的任务结果、监控结论）。参数：query(关键词), source('all'/'subagent'/'conversation'), time_range('1h'/'1d'/'7d'/空)
- **WebSearch**: 搜索结果可能很长。提取关键信息后继续，不要反复搜索相同内容
- **PythonExec**: 适合数据计算和格式化处理
- 如果一个工具调用失败，换一种方式尝试，不要重复相同调用

## 输出规范
- subagent_finish 的 output 应简洁有结构（使用 markdown）
- 控制在 2000 字以内，抓重点
- 如果信息量大，用标题/列表组织
"""


class SubagentContext:
    """Manages isolated context for a single subagent."""

    def __init__(self, spec: SubagentSpec, compress_threshold: int = 20000):
        self._spec = spec
        self._compress_threshold = compress_threshold
        self._system_prompt = self._build_system(spec)
        self._turns: list[list[dict]] = []
        self._summary: str | None = None

    def _build_system(self, spec: SubagentSpec) -> str:
        """Build subagent system prompt."""
        parts = [_BASE_PROMPT]
        parts.append(f'\n## 你的任务\n{spec.goal}')
        if spec.system_prompt_extra:
            parts.append(f'\n## 附加说明\n{spec.system_prompt_extra}')
        parts.append(f'\n## 约束\n- 最多 {spec.max_rounds} 轮对话完成任务')
        return '\n'.join(parts)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def turns(self) -> list[list[dict]]:
        return self._turns

    @turns.setter
    def turns(self, value: list[list[dict]]):
        self._turns = value

    @property
    def summary(self) -> str | None:
        return self._summary

    @summary.setter
    def summary(self, value: str | None):
        self._summary = value

    def build_messages(self, inbox_messages: list[dict] | None = None) -> list[dict]:
        """Build full message list for LLM call.

        Returns: [system, (summary if exists), history turns..., context_seed, inbox messages...]
        Note: context_seed is placed AFTER stable prefix (system+summary+turns) to maximize
        prefix cache hit rate. Dynamic sensor data in context_seed would break caching if first.
        """
        messages = [{'role': 'system', 'content': self._system_prompt}]

        # Compressed summary of old turns (stable prefix)
        if self._summary:
            messages.append({
                'role': 'user',
                'content': f'[历史摘要]\n{self._summary}'
            })
            messages.append({
                'role': 'assistant',
                'content': '了解，继续执行任务。'
            })

        # History turns (stable prefix)
        for turn in self._turns:
            messages.extend(turn)

        # Context seed placed after stable content — dynamic data won't break prefix cache
        if self._spec.context_seed and not self._turns:
            messages.append({
                'role': 'user',
                'content': f'[上下文信息]\n{self._spec.context_seed}'
            })

        # Inbox messages from parent
        if inbox_messages:
            for msg in inbox_messages:
                messages.append({
                    'role': 'user',
                    'content': f'[来自主代理] {msg["text"]}'
                })

        return messages

    def add_turn(self, messages: list[dict]) -> None:
        """Add a completed round's messages to history."""
        self._turns.append(messages)

    def needs_compression(self) -> bool:
        """Check if total history exceeds compression threshold."""
        total = sum(
            len(str(msg.get('content', '')))
            for turn in self._turns
            for msg in turn
        )
        return total > self._compress_threshold

    async def compress(self, llm_client=None, model_override: str | None = None) -> None:
        """Compress old turns into a summary, keeping recent 2 turns."""
        if len(self._turns) <= 2:
            return

        old_turns = self._turns[:-2]
        self._turns = self._turns[-2:]

        # Build text representation of old turns
        text_parts = []
        for turn in old_turns:
            for msg in turn:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content:
                    text_parts.append(f'[{role}] {content[:500]}')
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get('function', {})
                        text_parts.append(f'[tool_call] {fn.get("name", "?")}')

        old_text = '\n'.join(text_parts)

        # Ask LLM to compress
        compress_messages = [
            {'role': 'system', 'content': '将以下对话历史压缩为简洁的摘要，保留关键信息（工具调用结果、决策、发现）。用中文输出。'},
            {'role': 'user', 'content': old_text[:8000]},  # cap input
        ]

        try:
            import client as _client
            result = await _client.call(compress_messages, [], model_override=model_override, trace_id='subagent:compress')
            new_summary = result.get('content', '')
            if self._summary:
                self._summary = f'{self._summary}\n\n{new_summary}'
            else:
                self._summary = new_summary
        except Exception:
            # Compression LLM call failed — fallback to simple text truncation
            # Keep key info (tool results, decisions) without LLM summarization
            fallback_parts = []
            for turn in old_turns:
                for msg in turn:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'tool' and content:
                        fallback_parts.append(content[:200])
                    elif role == 'assistant' and content:
                        fallback_parts.append(content[:100])
            fallback = '\n'.join(fallback_parts)[:3000]
            if self._summary:
                self._summary = f'{self._summary}\n\n[简要回顾] {fallback}'
            else:
                self._summary = f'[简要回顾] {fallback}'

    def to_checkpoint(self) -> dict:
        """Serialize context state for persistence."""
        return {
            'turns': self._turns,
            'summary': self._summary,
        }

    def restore_from_checkpoint(self, turns: list, summary: str) -> None:
        """Restore context from persisted state."""
        self._turns = turns
        self._summary = summary or None
