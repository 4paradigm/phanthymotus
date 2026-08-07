"""
agent.py — Subagent class with isolated LLM loop.

Each subagent runs its own multi-round reasoning loop with:
- Isolated context (own history, compression)
- Filtered tool access (no nesting, no memory/task management)
- Checkpoint/resume support
- Cancel/pause signals
"""

from __future__ import annotations
import asyncio
import fnmatch
import json
import time
import typing
from uuid import uuid4

import mcp_client
from .protocol import (
    SubagentSpec, SubagentResult, SubagentStatus,
    STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED,
    STATUS_TIMEOUT, STATUS_CANCELLED, STATUS_PAUSED, STATUS_SUSPENDED,
)
from .context import SubagentContext


# Tools that subagents are NEVER allowed to use (enforced at code level)
_DENIED_TOOLS = {
    'subagent_spawn', 'subagent_spawn_sync', 'subagent_status',
    'subagent_cancel', 'subagent_message', 'subagent_result',
    'update_memory', 'activate_skill', 'deactivate_skill',
    'task_create', 'task_update', 'task_done', 'task_fail',
}


class Subagent:
    """An isolated agent instance with its own LLM loop and context."""

    def __init__(self, spec: SubagentSpec, agent_id: str | None = None,
                 compress_threshold: int = 20000):
        self.id = agent_id or uuid4().hex[:8]
        self.spec = spec
        self.status: str = 'pending'
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.rounds_completed: int = 0
        self.result: SubagentResult | None = None

        # Context isolation
        self._context = SubagentContext(spec, compress_threshold)
        self._inbox: asyncio.Queue = asyncio.Queue(maxsize=16)

        # Control signals
        self._cancel_event = asyncio.Event()
        self._pause_event = asyncio.Event()

        # Tracking
        self._tool_calls_made: list[dict] = []
        self._progress_reports: list[str] = []

    @property
    def context(self) -> SubagentContext:
        return self._context

    def get_status(self) -> SubagentStatus:
        return SubagentStatus(
            id=self.id,
            goal=self.spec.goal,
            status=self.status,
            priority=self.spec.priority,
            model=self.spec.model,
            rounds_completed=self.rounds_completed,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def send_message(self, text: str) -> None:
        """Queue a message from the parent agent."""
        try:
            self._inbox.put_nowait({'text': text, 'ts': time.time()})
        except asyncio.QueueFull:
            pass  # drop if inbox full

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancel_event.set()

    def pause(self) -> None:
        """Signal pause (for voluntary pause or preemption)."""
        self._pause_event.set()

    # ── Tool Filtering ─────────────────────────────────────────────────────────

    def _get_allowed_tools(self) -> list[dict]:
        """Build tool list for this subagent based on spec filters."""
        # Get all bound tools from canvas (same as main agent sees)
        all_schemas = self._get_all_mcp_schemas()

        # Also include desktop tools (system tools available to subagents)
        all_schemas.extend(self._get_desktop_tool_schemas())

        # Apply whitelist filter
        if self.spec.tool_filter is not None:
            filtered = []
            for schema in all_schemas:
                name = schema.get('name', '')
                if any(fnmatch.fnmatch(name, pat) for pat in self.spec.tool_filter):
                    filtered.append(schema)
            all_schemas = filtered

        # Apply blacklist deny
        if self.spec.tool_deny:
            all_schemas = [
                s for s in all_schemas
                if not any(fnmatch.fnmatch(s.get('name', ''), pat) for pat in self.spec.tool_deny)
            ]

        # Remove absolutely denied tools
        all_schemas = [s for s in all_schemas if s.get('name', '') not in _DENIED_TOOLS]

        # Add subagent-specific system tools
        sys_tools = self._build_subagent_sys_tools()

        tool_list = (
            [{'type': 'function', 'function': s} for s in sys_tools]
            + [{'type': 'function', 'function': s} for s in all_schemas]
        )
        return tool_list

    def _get_all_mcp_schemas(self) -> list[dict]:
        """Get all online MCP tool schemas (unfiltered by canvas binding for subagent)."""
        # Subagents use all online tools from registry, filtered by spec
        schemas = []
        for mcp_id, info in mcp_client.registry.items():
            if not info.get('online'):
                continue
            for name, schema in info.get('schemas', {}).items():
                schemas.append(schema)
        return schemas

    def _get_desktop_tool_schemas(self) -> list[dict]:
        """Get desktop tool schemas from the main event loop's sys_tools."""
        from event.llm import _event_instance
        if not _event_instance:
            return []
        _DESKTOP_TOOLS = {'Bash', 'PythonExec', 'Read', 'Write', 'Edit', 'Glob', 'Grep', 'WebFetch', 'WebSearch', 'memory_recall'}
        return [
            info['schema']
            for name, info in _event_instance._sys_tools.items()
            if name in _DESKTOP_TOOLS and name not in _DENIED_TOOLS
        ]

    def _build_subagent_sys_tools(self) -> list[dict]:
        """Build the 3 subagent-specific system tools."""
        return [
            {
                'name': 'subagent_finish',
                'description': '任务完成时调用。output 为最终结果文本。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'output': {'type': 'string', 'description': '任务结果/输出'},
                    },
                    'required': ['output'],
                },
            },
            {
                'name': 'subagent_fail',
                'description': '无法完成任务时调用。说明失败原因。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'reason': {'type': 'string', 'description': '失败原因'},
                    },
                    'required': ['reason'],
                },
            },
            {
                'name': 'subagent_report',
                'description': '向主代理汇报。默认存入记忆库供按需检索。仅紧急情况（安全/硬件告警）设 urgent=true 立即中断主代理。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'progress': {'type': 'string', 'description': '进度/结论描述'},
                        'urgent': {'type': 'boolean', 'description': '是否紧急（默认false存DB，true立即通知主代理）'},
                    },
                    'required': ['progress'],
                },
            },
        ]

    # ── LLM Loop ───────────────────────────────────────────────────────────────

    async def run(self, llm_client=None) -> SubagentResult:
        """Execute the subagent's LLM loop until completion, failure, or interruption.

        Args:
            llm_client: Deprecated, kept for backward compat. Uses client.call() directly.

        Returns:
            SubagentResult with final status and output
        """
        self.status = STATUS_RUNNING
        self.updated_at = time.time()
        t0 = time.time()
        _trace_id = f'subagent:{self.id}'

        tool_list = self._get_allowed_tools()
        finish_output: str | None = None
        fail_reason: str | None = None
        _spans = []

        try:
            for round_idx in range(self.spec.max_rounds):
                # Check cancel/pause signals
                if self._cancel_event.is_set():
                    self.result = SubagentResult(
                        agent_id=self.id,
                        status=STATUS_CANCELLED,
                        output='',
                        tool_calls_made=self._tool_calls_made,
                        rounds_used=self.rounds_completed,
                        duration_s=time.time() - t0,
                    )
                    self.status = STATUS_CANCELLED
                    return self.result

                if self._pause_event.is_set():
                    self.status = STATUS_PAUSED if not self._cancel_event.is_set() else STATUS_SUSPENDED
                    self.updated_at = time.time()
                    # Return None to indicate pause (manager handles)
                    return None

                # Drain inbox
                inbox_msgs = []
                while not self._inbox.empty():
                    try:
                        inbox_msgs.append(self._inbox.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Build messages
                messages = self._context.build_messages(inbox_msgs if inbox_msgs else None)

                # Call LLM
                print(f'[subagent:{self.id}] round {round_idx} | msgs={len(messages)} tools={len(tool_list)}')
                _round_start = time.time()

                try:
                    import client as _client
                    response = await _client.call(
                        message_list=messages,
                        tool_list=tool_list,
                        cancel_event=self._cancel_event,
                        model_override=self.spec.model,
                        trace_id=_trace_id,
                        caller_info={'agent_type': 'subagent'},
                    )
                    _spans.append({'span': f'llm_round_{round_idx}', 'component': 'subagent',
                                   'start_ts': _round_start, 'end_ts': time.time()})
                except Exception as e:
                    from client.llm import LLMErrorKind, _classify_error
                    kind, _ = _classify_error(e)
                    if kind == LLMErrorKind.CONTEXT_OVERFLOW:
                        # Try compression
                        await self._context.compress(None, self.spec.model)
                        continue
                    raise

                # Process response
                round_messages = []
                content = response.get('content', '')
                tool_calls = response.get('tool_calls', [])

                # Record assistant message
                assistant_msg = {'role': 'assistant'}
                if content:
                    assistant_msg['content'] = content
                if tool_calls:
                    assistant_msg['tool_calls'] = tool_calls
                if response.get('_usage'):
                    assistant_msg['_usage'] = response['_usage']
                round_messages.append(assistant_msg)

                # Dispatch tool calls
                if tool_calls:
                    for tc in tool_calls:
                        tc_id = tc.get('id', '')
                        fn = tc.get('function', {})
                        fn_name = fn.get('name', '')
                        fn_args_str = fn.get('arguments', '{}')

                        try:
                            fn_args = json.loads(fn_args_str)
                        except json.JSONDecodeError:
                            fn_args = {}

                        # Dispatch
                        result_text = await self._dispatch_tool(fn_name, fn_args)

                        # Check for terminal tools
                        if fn_name == 'subagent_finish':
                            finish_output = fn_args.get('output', result_text)
                        elif fn_name == 'subagent_fail':
                            fail_reason = fn_args.get('reason', result_text)

                        # Record tool result
                        round_messages.append({
                            'role': 'tool',
                            'tool_call_id': tc_id,
                            'content': result_text,
                        })

                        # Track
                        self._tool_calls_made.append({
                            'name': fn_name,
                            'round': round_idx,
                        })

                # Add round to context
                self._context.add_turn(round_messages)
                self.rounds_completed = round_idx + 1
                self.updated_at = time.time()

                # Check terminal conditions
                if finish_output is not None:
                    self.result = SubagentResult(
                        agent_id=self.id,
                        status=STATUS_COMPLETED,
                        output=finish_output,
                        tool_calls_made=self._tool_calls_made,
                        rounds_used=self.rounds_completed,
                        duration_s=time.time() - t0,
                    )
                    self.status = STATUS_COMPLETED
                    return self.result

                if fail_reason is not None:
                    self.result = SubagentResult(
                        agent_id=self.id,
                        status=STATUS_FAILED,
                        output='',
                        tool_calls_made=self._tool_calls_made,
                        rounds_used=self.rounds_completed,
                        duration_s=time.time() - t0,
                        error=fail_reason,
                    )
                    self.status = STATUS_FAILED
                    return self.result

                # No tool calls and no finish = natural stop
                if not tool_calls:
                    self.result = SubagentResult(
                        agent_id=self.id,
                        status=STATUS_COMPLETED,
                        output=content or '(no output)',
                        tool_calls_made=self._tool_calls_made,
                        rounds_used=self.rounds_completed,
                        duration_s=time.time() - t0,
                    )
                    self.status = STATUS_COMPLETED
                    return self.result

                # Compression check
                if self._context.needs_compression():
                    await self._context.compress(llm_client, self.spec.model)

                # Checkpoint check (done by manager externally)

            # Max rounds reached
            self.result = SubagentResult(
                agent_id=self.id,
                status=STATUS_TIMEOUT,
                output=content if content else '(max rounds reached)',
                tool_calls_made=self._tool_calls_made,
                rounds_used=self.rounds_completed,
                duration_s=time.time() - t0,
                error=f'Reached max_rounds={self.spec.max_rounds}',
            )
            self.status = STATUS_TIMEOUT
            return self.result

        except asyncio.CancelledError:
            self.result = SubagentResult(
                agent_id=self.id,
                status=STATUS_CANCELLED,
                output='',
                tool_calls_made=self._tool_calls_made,
                rounds_used=self.rounds_completed,
                duration_s=time.time() - t0,
            )
            self.status = STATUS_CANCELLED
            return self.result

        except Exception as e:
            self.result = SubagentResult(
                agent_id=self.id,
                status=STATUS_FAILED,
                output='',
                tool_calls_made=self._tool_calls_made,
                rounds_used=self.rounds_completed,
                duration_s=time.time() - t0,
                error=f'{type(e).__name__}: {e}',
            )
            self.status = STATUS_FAILED
            return self.result

        finally:
            # Commit perf spans for this subagent run
            if _spans:
                _spans.append({'span': 'subagent_total', 'component': 'subagent',
                               'start_ts': t0, 'end_ts': time.time()})
                try:
                    import perf_log
                    perf_log.commit_spans(
                        trace_id=_trace_id,
                        spans=_spans,
                        source=f'subagent:{self.id}',
                        trigger_text=self.spec.goal[:200],
                    )
                except Exception:
                    pass

    async def _dispatch_tool(self, name: str, args: dict) -> str:
        """Dispatch a tool call and return result text."""
        # Subagent system tools
        if name == 'subagent_finish':
            return args.get('output', 'done')
        if name == 'subagent_fail':
            return args.get('reason', 'failed')
        if name == 'subagent_report':
            progress = args.get('progress', '')
            urgent = args.get('urgent', False)
            if not progress.strip():
                return 'ok'
            self._progress_reports.append(progress)
            print(f'[subagent:{self.id}] progress{"(urgent)" if urgent else ""}: {progress[:100]}')
            if urgent:
                # 紧急：直接进 event_bus → 触发 main agent
                import event_bus
                await event_bus.enqueue(
                    source=f'subagent:{self.id}/report',
                    text=progress,
                )
            else:
                # 非紧急：存入 DB 供 memory_recall 检索，不触发 main agent
                try:
                    import time as _time
                    from config import _get_conn
                    with _get_conn() as conn:
                        conn.execute(
                            'INSERT INTO subagent_conclusions (agent_id, goal, conclusion, source_type, created_at) '
                            'VALUES (?, ?, ?, ?, ?)',
                            (self.id, self.spec.goal[:100], progress, 'bg_monitor', _time.time())
                        )
                        conn.commit()
                except Exception as e:
                    print(f'[subagent:{self.id}] save report to DB failed: {e}')
            return 'ok'

        # MCP tool call
        if name.startswith('mcp__'):
            try:
                result = await mcp_client.call_tool(name, args)
                if isinstance(result, dict):
                    return json.dumps(result, ensure_ascii=False)
                return str(result)
            except Exception as e:
                return f'[tool error] {type(e).__name__}: {e}'

        # Desktop tool call (system tools from main agent)
        from event.llm import _event_instance
        if _event_instance and name in _event_instance._sys_tools:
            try:
                fn = _event_instance._sys_tools[name]['object']
                result = await fn(**args)
                result_str = str(result) if result else '(no output)'
                # 截断大结果（WebSearch/WebFetch 等可能返回 6K+ chars）
                _MAX_TOOL_RESULT = 2500
                if len(result_str) > _MAX_TOOL_RESULT:
                    result_str = result_str[:_MAX_TOOL_RESULT] + '\n...(结果已截断，如需更多请细化查询)'
                return result_str
            except Exception as e:
                return f'[tool error] {type(e).__name__}: {e}'

        return f'[unknown tool] {name}'

    # ── Checkpoint/Restore ─────────────────────────────────────────────────────

    def to_checkpoint(self) -> dict:
        """Serialize full state for persistence."""
        return {
            'id': self.id,
            'spec': self.spec.to_dict(),
            'status': self.status,
            'rounds_completed': self.rounds_completed,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'context': self._context.to_checkpoint(),
            'tool_calls_made': self._tool_calls_made,
            'progress_reports': self._progress_reports,
        }

    @classmethod
    def from_checkpoint(cls, data: dict) -> Subagent:
        """Restore a subagent from persisted checkpoint data."""
        spec = SubagentSpec.from_dict(data['spec'])
        agent = cls(spec=spec, agent_id=data['id'])
        agent.status = data['status']
        agent.rounds_completed = data['rounds_completed']
        agent.created_at = data['created_at']
        agent.updated_at = data['updated_at']
        agent._tool_calls_made = data.get('tool_calls_made', [])
        agent._progress_reports = data.get('progress_reports', [])

        # Restore context
        ctx = data.get('context', {})
        agent._context.restore_from_checkpoint(
            turns=ctx.get('turns', []),
            summary=ctx.get('summary', ''),
        )
        return agent
