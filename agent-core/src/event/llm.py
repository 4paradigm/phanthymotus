"""
event/llm.py — 事件驱动的 Agent Loop。

职责：
  - 通过 collector 批量获取事件（带触发间隔）
  - 构建分层 prompt（L1~L4，由 prompt.py 完成）
  - 调用 LLM（支持多轮工具调用）
  - 分发 MCP 工具调用（mcp_client）及系统工具（finish / update_memory）
  - 把每一步广播到 /ws/motus 供前端可视化

工具命名约定：
  - 系统工具：短名如 'finish', 'update_memory'
  - MCP 工具 ：'mcp__<mcp_id>__<tool_name>'（由 mcp_client.py 生成）
"""

import asyncio
import json
import pathlib
import time
import typing

import log
import config
import client
import event
import event_bus
import collector
import mcp_client
import perf_log
import prompt as prompt_mod
from api.motus_stream import push_event


# ── Turn 取消异常 ────────────────────────────────────────────────────────────────

class TurnCancelled(Exception):
    """用户消息抢占时抛出，中断正在进行的 sensor turn。"""
    pass


# ── 系统工具注册（静态，仅 finish / memory）──────────────────────────────────

def _build_system_tools(named_functions: list[tuple[str, callable]]) -> dict:
    """把 (name, fn) 列表转成 tool_dict，使用简短常规命名。"""
    import inspect
    tool_dict: dict = {}
    for tool_name, fn in named_functions:
        param_list = [
            (name, typing.get_args(tp)[0], typing.get_args(tp)[1])
            for name, tp in typing.get_type_hints(fn, include_extras=True).items()
            if name not in ('self', 'cls', 'return')
        ]
        # 检测哪些参数有默认值（即可选）
        sig = inspect.signature(fn)
        optional_params = {
            k for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        tool_dict[tool_name] = {
            'object': fn,
            'schema': {
                'name':        tool_name,
                'description': fn.__doc__ or '',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        n: {
                            'type':        {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}[t],
                            'description': d,
                        }
                        for n, t, d in param_list
                    },
                    'required': [n for n, _, _ in param_list if n not in optional_params],
                },
            },
        }
    return tool_dict


# ── History helpers ────────────────────────────────────────────────────────────

def _sanitize(message_list: list[dict]) -> list[dict]:
    """移除末尾未被 tool 结果回应的 tool_calls（避免 API 报错）。"""
    responded_ids: set[str] = set()
    for i in range(len(message_list) - 1, -1, -1):
        msg = message_list[i]
        if msg.get('role') == 'tool':
            responded_ids.add(msg.get('tool_call_id'))
        elif msg.get('role') == 'assistant' and msg.get('tool_calls'):
            expected = {call['id'] for call in msg['tool_calls']}
            if not expected.issubset(responded_ids):
                return message_list[:i]
            break
    return message_list


def _trim(message_list: list[dict], max_messages: int = 100, max_images: int = 5) -> list[dict]:
    """裁剪历史：超限图片替换为占位符，超限条数从头截断。"""
    image_count = 0
    result = []
    for msg in reversed(message_list):
        if msg.get('role') == 'tool' and isinstance(msg.get('content'), list):
            image_count += 1
            if image_count > max_images:
                msg = {**msg, 'content': '（此处原为图片，已压缩以节省上下文）'}
        result.append(msg)
    message_list = list(reversed(result))

    if len(message_list) > max_messages:
        start = len(message_list) - max_messages
        while start < len(message_list) and message_list[start].get('role') == 'tool':
            start += 1
        message_list = message_list[start:]

    return message_list


def _estimate_chars(turns: list[list[dict]]) -> int:
    """粗估 turns 的总字符数（用于判断是否需要压缩）。"""
    total = 0
    for turn in turns:
        for msg in turn:
            content = msg.get('content', '')
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += 200  # multimodal 粗估
            # tool_calls 的 arguments 也计入
            for tc in (msg.get('tool_calls') or []):
                total += len(tc.get('function', {}).get('arguments', ''))
    return total


def _turns_to_text(turns: list[list[dict]]) -> str:
    """把 turns 转为文本摘要素材（供压缩用）。"""
    lines = []
    for i, turn in enumerate(turns):
        for msg in turn:
            role = msg.get('role', '?')
            content = msg.get('content', '')
            if isinstance(content, list):
                content = '[图片/多模态内容]'
            if role == 'assistant' and msg.get('tool_calls'):
                tool_names = [tc['function']['name'] for tc in msg['tool_calls']]
                lines.append(f'[assistant] 调用工具: {", ".join(tool_names)}')
                if content:
                    lines.append(f'[assistant] {content[:300]}')
            elif role == 'tool':
                # 工具结果只保留前200字符
                lines.append(f'[tool_result] {str(content)[:200]}')
            elif content:
                lines.append(f'[{role}] {content[:500]}')
    return '\n'.join(lines)


# ── Event class ────────────────────────────────────────────────────────────────

_COMPRESS_PROMPT = """你是一个对话历史压缩器。请将以下对话历史精炼为一段简洁的摘要。

要求：
- 保留关键事实、决策、工具调用结果中的重要信息
- 保留未完成的任务和待处理事项
- 去除重复的传感器数据和冗余的工具调用细节
- 使用简洁的中文，控制在 500 字以内
- 以「[历史摘要]」开头

对话历史：
"""


async def _compress_turns(turns: list[list[dict]]) -> str:
    """用 LLM 压缩旧的 turns 为文本摘要。"""
    text = _turns_to_text(turns)
    # 截断过长的输入（避免压缩请求本身溢出）
    if len(text) > 30000:
        text = text[:30000] + '\n...(已截断)'

    try:
        summary_response = await client.call(
            message_list=[
                {'role': 'system', 'content': '你是一个高效的对话摘要助手。'},
                {'role': 'user', 'content': _COMPRESS_PROMPT + text},
            ],
            tool_list=[],
        )
        return summary_response.get('content', '') or '[历史摘要] （压缩失败，无内容）'
    except Exception as e:
        print(f'[decision] compress failed: {e}')
        # 压缩失败时，回退到简单截断
        return f'[历史摘要] 之前有 {len(turns)} 轮对话，因压缩失败仅保留最近内容。'


# ── detailed_info 系统工具实现 ────────────────────────────────────────────────────

import datetime as _dt


async def _search_history(
    query: typing.Annotated[str, "搜索关键词（支持中文）"],
    limit: typing.Annotated[int, "返回最多 N 条结果，默认 5"] = 5,
) -> str:
    """搜索历史对话记录。当需要回忆过去的对话内容、查找之前讨论过的话题时使用。"""
    import chat_history
    results = chat_history.search(query, limit=limit)
    if not results:
        return '未找到相关历史记录。'
    lines = []
    for r in results:
        ts = _dt.datetime.fromtimestamp(r['ts']).strftime('%m-%d %H:%M')
        lines.append(f'[{ts}] {r["preview"]}')
    return '\n---\n'.join(lines)


async def _raw_input_info(
    source: typing.Annotated[str, "要查看详情的信息源名称（可通过摘要中的 source name 获得）"],
    limit: typing.Annotated[int, "返回最近 N 条原始事件，默认 20"] = 20,
) -> str:
    """获取指定信息源的原始输入数据。当摘要信息不足以做决策时使用此工具深入查看原始事件。"""
    events = collector.get_source_detail(source, limit=limit)
    if not events:
        available = collector.get_available_sources()
        return f'未找到 source={source} 的数据。当前可用 sources: {", ".join(available) if available else "(无)"}'
    # 格式化为详细 XML
    lines = []
    for ev in events:
        ts = _dt.datetime.fromtimestamp(ev['ts']).strftime('%Y-%m-%dT%H:%M:%S')
        text = ev.get('text', '')
        lines.append(f'<event ts="{ts}">\n{text}\n</event>')
    return '\n'.join(lines)


# ── Module-level reference for bg subagent context sync ───────────────────────

_event_instance: 'Event | None' = None


def get_recent_context(max_turns: int = 5) -> str:
    """返回最近 N 轮 main agent 的 assistant 输出摘要，供 bg subagent 同步上下文。"""
    if not _event_instance or not _event_instance._turns:
        return ''
    recent = _event_instance._turns[-max_turns:]
    lines = []
    for turn in recent:
        for msg in turn:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if content:
                    lines.append(content[:200])
            elif msg.get('role') == 'user':
                content = msg.get('content', '')
                if content and not content.startswith('<status'):
                    lines.append(f'[用户] {content[:100]}')
    return '\n'.join(lines[-10:])  # 最多 10 行


def get_recent_context_rich(max_turns: int = 20, max_chars: int = 6000) -> str:
    """返回 main agent 最近对话的原始片段，供 subagent 理解完整上下文。

    策略：20 轮内，纯字符串提取，不额外调 LLM。包含用户消息、
    assistant 决策文本、tool 结果（含 subagent_result 返回值）。
    """
    if not _event_instance or not _event_instance._turns:
        return ''
    recent = _event_instance._turns[-max_turns:]
    parts = []
    total = 0
    truncated = False
    for turn in recent:
        if truncated:
            break
        for msg in turn:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if not content:
                continue
            # 跳过 <status 开头的环境快照（噪音大）
            if role == 'user' and content.startswith('<status'):
                continue
            if role == 'user':
                line = f'[用户] {content[:500]}'
            elif role == 'assistant':
                line = f'[助手] {content[:500]}'
            elif role == 'tool':
                line = f'[工具结果] {content[:800]}'
            else:
                continue
            if total + len(line) > max_chars:
                parts.append('...(更早历史已截断)')
                truncated = True
                break
            parts.append(line)
            total += len(line)
    return '\n'.join(parts)


class Event:
    def __init__(self):
        self._turns: list[list[dict]] = []  # 每轮对话的消息列表
        self._sys_tools:   dict       = {}
        self._summary: str | None     = None  # 压缩后的历史摘要
        self._session_id: str | None  = None  # chat history session
        self._current_turn: list[dict] = []   # 当前轮消息（供 run_forever 保存）
        self._subagent_mgr = None             # SubagentManager instance

    async def __aenter__(self):
        global _event_instance
        _event_instance = self
        # 初始化子代理管理器
        from subagent.manager import SubagentManager
        from subagent.tools import SubagentTools
        from subagent import _set_manager
        self._subagent_mgr = SubagentManager(llm_client=client.llm)
        _set_manager(self._subagent_mgr)
        _sa_tools = SubagentTools(self._subagent_mgr)

        # 注册桌面工具（文件操作 / Shell / Python / 搜索 / Web）
        from event.desktop import DesktopTools
        self._desktop_tools = DesktopTools()

        # 注册系统工具（finish / memory / task / detailed_info / subagent / desktop）
        self._sys_tools = _build_system_tools([
            ('finish', event.finish.__call__),
            ('update_memory', event.memory.update),
            ('activate_skill', event.skills.activate_skill),
            ('deactivate_skill', event.skills.deactivate_skill),
            ('task_create', event.task.task_create),
            ('task_update', event.task.task_update),
            ('task_done', event.task.task_done),
            ('task_fail', event.task.task_fail),
            ('task_list', event.task.task_list),
            ('task_force_clear', event.task.task_force_clear),
            ('raw_input_info', _raw_input_info),
            ('search_history', _search_history),
            ('subagent_spawn', _sa_tools.subagent_spawn),
            ('subagent_spawn_sync', _sa_tools.subagent_spawn_sync),
            ('subagent_status', _sa_tools.subagent_status),
            ('subagent_cancel', _sa_tools.subagent_cancel),
            ('subagent_message', _sa_tools.subagent_message),
            ('subagent_result', _sa_tools.subagent_result),
            # Desktop tools (Claude Code 风格)
            ('Bash', self._desktop_tools.Bash),
            ('PythonExec', self._desktop_tools.PythonExec),
            ('Read', self._desktop_tools.Read),
            ('Write', self._desktop_tools.Write),
            ('Edit', self._desktop_tools.Edit),
            ('Glob', self._desktop_tools.Glob),
            ('Grep', self._desktop_tools.Grep),
            ('WebFetch', self._desktop_tools.WebFetch),
            ('WebSearch', self._desktop_tools.WebSearch),
        ])
        # 连接并注册所有 MCP 工具
        await mcp_client.init_all()
        # 恢复持久化的活跃任务及其定时检查
        import task_store
        from event.task import _register_check
        task_store.load_all()
        for task in task_store.active_tasks():
            _register_check(task)
        # 启动子代理调度器（restore + scheduler loop）
        await self._subagent_mgr.start()
        # 重启续跑：加载上一个 session 的最近 turns
        import chat_history
        last = chat_history.get_last_session_turns(limit=10)
        if last:
            self._turns = last['turns']
            self._session_id = last['session_id']
            print(f'[startup] resumed session {last["session_id"][:8]}... ({len(last["turns"])} turns)')
        else:
            self._session_id = None
        return self

    async def __aexit__(self, *args):
        # 关闭子代理管理器（checkpoint all running）
        if self._subagent_mgr:
            await self._subagent_mgr.shutdown()
        return False

    def _get_bound_tool_schemas(self) -> list[dict]:
        """从画布 executor connections 获取绑定到 decision_core 的工具 schemas。"""
        layout = config.main.get('canvas_layout', {})
        cards = layout.get('cards', [])
        exec_conns = layout.get('execConnections', [])

        # 找到 agentcore 卡片的 cardId
        core_card_ids = {c['id'] for c in cards if c.get('mcpId') == 'agentcore'}

        # 从 executor connections 直接收集绑定的工具 schemas
        schemas = []
        for ec in exec_conns:
            if ec.get('fromCardId') not in core_card_ids:
                continue
            mcp_id = ec.get('toMcpId', '')
            tool_name = ec.get('toToolName', '')
            if not mcp_id or not tool_name:
                continue
            # 从 mcp_client registry 中取该工具的 schema
            info = mcp_client.registry.get(mcp_id)
            if not info or not info.get('online'):
                continue
            full_name = f"mcp__{mcp_id}__{tool_name}"
            schema = info.get('schemas', {}).get(full_name)
            if schema:
                schemas.append(schema)
            else:
                # 检查是否有拆分的子工具（x-action-params 拆分）
                for split_name in info.get('tool_groups', {}).get(tool_name, []):
                    s = info.get('schemas', {}).get(split_name)
                    if s:
                        schemas.append(s)

        if not schemas:
            # 没有绑定任何工具时，仅使用系统工具（不暴露全部 MCP 工具）
            return []

        return schemas

    # ── 主循环 ───────────────────────────────────────────────────────────────

    async def run_forever(self):
        """事件驱动：通过 collector 批量获取事件，每批跑一轮推理。"""
        while True:
            ev = await collector.next_trigger()
            self._current_turn = []  # 本轮消息，无论成功失败都会保存
            # 注册取消信号（用户消息可通过此信号中断 sensor turn）
            cancel_ev = asyncio.Event()
            collector.set_cancel_event(cancel_ev)
            collector.set_turn_priority(1 if ev.get('_urgent') else 0)
            collector.set_busy(True)
            try:
                await self._one_turn(ev, cancel_event=cancel_ev)
            except TurnCancelled:
                print(f'[decision] turn cancelled by user message')
                self._current_turn.append({
                    'role': 'assistant',
                    'content': '[turn interrupted by user message]',
                })
                await push_event({'type': 'turn_cancelled', 'payload': {}})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f'[decision] error in _one_turn: {e}')
                # 把错误也记入本轮消息
                self._current_turn.append({
                    'role': 'assistant',
                    'content': f'[错误] {type(e).__name__}: {e}',
                })
                await push_event({'type': 'error', 'payload': {'message': str(e)}})
            finally:
                collector.set_cancel_event(None)
                collector.set_busy(False)
                # 无论成功失败，只要有消息就持久化
                if self._current_turn:
                    self._save_current_turn(ev)

    def _save_current_turn(self, trigger_event: dict):
        """保存 _current_turn 到内存历史 + SQLite。"""
        turn = self._current_turn
        self._turns.append(turn)
        # 持久化（延迟创建 session）
        import chat_history
        try:
            if not self._session_id:
                self._session_id = chat_history.create_session()
            chat_history.save_turn(self._session_id, len(self._turns) - 1, turn)
            summary_text = trigger_event.get('text', '') or trigger_event.get('source', '')
            if summary_text:
                chat_history.update_summary(self._session_id, summary_text)
        except Exception as e:
            print(f'[chat_history] save_turn failed: {e}')
        # 裁剪
        max_turns = config.main.get('event', {}).get('llm', {}).get('history_turns', 30)
        if len(self._turns) > max_turns:
            self._turns = self._turns[-max_turns:]

    # ── 单轮推理 ─────────────────────────────────────────────────────────────

    def _build_history(self) -> list[dict]:
        """从 _turns 构建 L3 历史（取最近 N 轮 flatten）。若有摘要则前置。"""
        max_turns = config.main.get('event', {}).get('llm', {}).get('history_turns', 30)
        recent_turns = self._turns[-max_turns:] if len(self._turns) > max_turns else self._turns
        history = []
        # 前置历史摘要（如果有）
        if self._summary:
            history.append({'role': 'user', 'content': self._summary})
            history.append({'role': 'assistant', 'content': '好的，我已了解之前的对话背景。'})
        for turn in recent_turns:
            history.extend(turn)
        return _sanitize(history)

    async def _maybe_compress(self):
        """检查历史是否超过阈值，如果是则压缩旧轮次为摘要。"""
        llm_cfg = config.main.get('event', {}).get('llm', {})
        threshold = llm_cfg.get('compress_threshold_chars', 80000)
        keep_recent = llm_cfg.get('compress_keep_recent', 6)

        total_chars = _estimate_chars(self._turns)
        if total_chars <= threshold:
            return
        if len(self._turns) <= keep_recent:
            return  # 不够分割，跳过

        # 分割：压缩旧的，保留最近的
        old_turns = self._turns[:-keep_recent]
        recent_turns = self._turns[-keep_recent:]

        print(f'[decision] compressing history: {len(old_turns)} old turns ({total_chars} chars > {threshold} threshold)')
        summary = await _compress_turns(old_turns)
        # 合并旧摘要
        if self._summary:
            summary = self._summary + '\n\n' + summary

        self._summary = summary
        self._turns = recent_turns
        print(f'[decision] compressed: kept {len(recent_turns)} recent turns, summary={len(summary)} chars')

    async def _one_turn(self, trigger_event: dict, cancel_event: asyncio.Event | None = None):
        import time as _time
        from uuid import uuid4
        _turn_t0 = _time.perf_counter()

        # Reset Python sandbox namespace for this turn
        self._desktop_tools.reset_python_namespace()

        # 性能追踪（开放 span 式）
        _trace_id = str(uuid4())
        _turn_start_ts = time.time()
        _spans = []  # 收集所有 span
        _tool_names_collected = []

        # 从 trigger_event 中提取 perception 上报的 spans
        _perf_spans_from_perception = trigger_event.get('_perf_spans', [])
        for ps in _perf_spans_from_perception:
            ps['component'] = ps.get('component', 'perception')
            _spans.append(ps)

        # collector_wait span
        _collector_receive = trigger_event.get('ts')
        _trigger_emit = trigger_event.get('_perf_trigger_emit_ts')
        if _collector_receive and _trigger_emit:
            _spans.append({'span': 'event_queue', 'component': 'core',
                           'start_ts': _collector_receive, 'end_ts': _trigger_emit})

        # Log incoming event
        _urgent_tag = ' [URGENT]' if trigger_event.get('_urgent') else ''
        print(f'[decision] received{_urgent_tag} event: source={trigger_event.get("source", "?")} text={trigger_event.get("text", "")[:300]}')
        # Subagent status in log
        if self._subagent_mgr:
            _sa_active = self._subagent_mgr.list_active()
            if _sa_active:
                _sa_summary = ', '.join(f'{s.id}(P{s.priority}/{s.status})' for s in _sa_active[:5])
                print(f'[decision] subagents: {_sa_summary}')

        # 广播触发事件到前端
        await push_event({
            'type':    'trigger',
            'mcp_id':  trigger_event.get('source', ''),
            'payload': {'text': trigger_event.get('text', '')[:200]},
        })

        # 合并工具表：系统工具 + 画布上绑定的 MCP 工具（通过 executor connections）
        bound_schemas = self._get_bound_tool_schemas()
        all_tool_list = (
            [{'type': 'function', 'function': t['schema']} for t in self._sys_tools.values()]
            + [{'type': 'function', 'function': s} for s in bound_schemas]
        )
        # 绑定工具全名集合，用于 L2 环境快照过滤
        bound_tool_names = {s['name'] for s in bound_schemas}

        # ── 冻结 system message（turn 内复用，保证 prefix caching 命中）────
        frozen_system = prompt_mod.build_system(mcp_client.registry, bound_tool_names)

        finish_tool = 'finish'
        max_rounds  = 20
        response    = None
        decisions   = []
        turn_messages = self._current_turn  # alias for brevity
        _turn_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cached_tokens': 0}

        for round_idx in range(max_rounds):
            # ── 构建分层 prompt ────────────────────────────────────────────
            history = self._build_history()
            # 本轮已产生的消息也要加入历史（多轮工具调用场景）
            current_history = history + _sanitize(turn_messages)

            if round_idx == 0:
                # 首轮：加入 L4 触发事件（含 L2 动态快照）
                messages = prompt_mod.build(
                    system_msg    = frozen_system,
                    message_list  = current_history,
                    trigger_event = trigger_event,
                )
                # 把 trigger user message 记入 turn_messages，后续轮次能看到
                trigger_user_msg = messages[-1]  # build() 最后一条是 L4 user
                turn_messages.append(trigger_user_msg)
            else:
                # 后续轮：不加新的 user message，复用冻结的 system
                messages = prompt_mod.build_continuation(
                    system_msg   = frozen_system,
                    message_list = current_history,
                )

            await push_event({'type': 'llm_request', 'payload': {'round': round_idx}})

            # 保存请求日志
            pathlib.Path('./resource/log').mkdir(parents=True, exist_ok=True)
            pathlib.Path('./resource/log/llm.json').write_text(
                json.dumps(messages, ensure_ascii=False)
            )
            pathlib.Path('./resource/log/llm_tools.json').write_text(
                json.dumps(all_tool_list, ensure_ascii=False, indent=2)
            )

            # Log LLM request summary
            msg_count = len(messages)
            tool_count = len(all_tool_list)
            # Estimate prompt size (rough: 1 token ≈ 3 chars for CJK)
            prompt_chars = sum(len(m.get('content') or '') for m in messages)
            last_user = next((m.get('content', '')[:200] for m in reversed(messages) if m.get('role') == 'user'), '')
            print(f'[decision] llm request: round={round_idx} messages={msg_count} tools={tool_count} ~chars={prompt_chars} last_user={last_user}')

            # ── 调用 LLM（含上下文溢出恢复 + 取消检查）──────────────────────
            # 取消检查点：在耗时的 LLM 调用前检查是否被用户消息中断
            if cancel_event and cancel_event.is_set():
                raise TurnCancelled("Interrupted before LLM call")

            _round_t0 = _time.perf_counter()
            _round_start_ts = time.time()
            try:
                response = await client.call(
                    message_list = messages,
                    tool_list    = all_tool_list,
                    cancel_event = cancel_event,
                    trace_id     = _trace_id,
                    caller_info  = {'agent_type': 'main_agent'},
                )
            except TurnCancelled:
                raise
            except Exception as e:
                from client.llm import LLMErrorKind, _classify_error
                kind, _ = _classify_error(e)
                if kind == LLMErrorKind.CONTEXT_OVERFLOW and round_idx == 0:
                    # 上下文溢出：强制压缩后重试一次
                    print(f'[decision] context overflow — force compressing history')
                    if len(self._turns) > 2:
                        old = self._turns[:-2]
                        summary = await _compress_turns(old)
                        self._summary = (self._summary + '\n\n' + summary) if self._summary else summary
                        self._turns = self._turns[-2:]
                        # 重建 history 并重试（复用冻结的 system）
                        history = self._build_history()
                        current_history = history + _sanitize(turn_messages)
                        messages = prompt_mod.build(
                            system_msg    = frozen_system,
                            message_list  = current_history,
                            trigger_event = trigger_event,
                        )
                        trigger_user_msg = messages[-1]
                        turn_messages.clear()
                        turn_messages.append(trigger_user_msg)
                        response = await client.call(
                            message_list = messages,
                            tool_list    = all_tool_list,
                            trace_id     = _trace_id,
                            caller_info  = {'agent_type': 'main_agent'},
                        )
                    else:
                        raise
                else:
                    raise
            turn_messages.append(response)

            # Log LLM response
            _round_elapsed = _time.perf_counter() - _round_t0
            _round_end_ts = time.time()
            _spans.append({'span': f'llm_round_{round_idx}', 'component': 'core',
                           'start_ts': _round_start_ts, 'end_ts': _round_end_ts})
            resp_text = (response.get('content') or '')[:300]
            resp_tools = []
            for c in (response.get('tool_calls') or []):
                name = c['function']['name']
                args_str = c['function'].get('arguments', '')[:300]
                resp_tools.append(f'{name}({args_str})')
            print(f'[decision] llm response: round_time={_round_elapsed:.2f}s text={resp_text!r}')
            if resp_tools:
                for t in resp_tools:
                    print(f'[decision]   tool_call: {t}')

            # ── 文字输出 ──────────────────────────────────────────────────
            text = response.get('content') or ''
            if text:
                await push_event({'type': 'agent_thought', 'payload': {'text': text}})

            # ── 用量广播 ──────────────────────────────────────────────────
            _usage = response.get('_usage')
            if _usage:
                _turn_usage['prompt_tokens'] += _usage.get('prompt_tokens', 0)
                _turn_usage['completion_tokens'] += _usage.get('completion_tokens', 0)
                _turn_usage['total_tokens'] += _usage.get('total_tokens', 0)
                _turn_usage['cached_tokens'] += _usage.get('cached_tokens', 0)
                await push_event({'type': 'llm_usage', 'payload': _usage})

            # ── 工具调用 ──────────────────────────────────────────────────
            tool_calls = response.get('tool_calls') or []

            async def _dispatch(call: dict) -> dict:
                name   = call['function']['name']
                args   = json.loads(call['function']['arguments'] or '{}')

                # 性能追踪：记录工具时间
                _t_before = time.time()
                _tool_names_collected.append(name)

                await push_event({
                    'type':    'mcp_call',
                    'mcp_id':  name.split('__')[1] if name.startswith('mcp__') else '',
                    'payload': {'tool': name, 'args': args},
                })

                if name in self._sys_tools:
                    result = await self._sys_tools[name]['object'](**args)
                elif name.startswith('mcp__'):
                    args['_trace_id'] = _trace_id
                    result = await mcp_client.call_tool(name, args)
                else:
                    result = f'未知工具: {name}'

                # 性能追踪：记录工具完成
                _t_after = time.time()
                # 工具 span 名称：mcp__mcp-123__tool_name → tool:tool_name
                _short = name.split('__')[-1] if name.startswith('mcp__') else name
                _span_name = f'tool:{_short}'
                _spans.append({'span': _span_name, 'component': 'core',
                               'start_ts': _t_before, 'end_ts': _t_after})

                await push_event({
                    'type':    'mcp_result',
                    'mcp_id':  name.split('__')[1] if name.startswith('mcp__') else '',
                    'payload': {'tool': name, 'result': result if isinstance(result, str) else '[multimodal]'},
                })

                return {'id': call['id'], 'result': result}

            # 顺序执行工具调用（尊重 LLM 输出顺序），连续 sensor 工具批量并行
            def _is_sensor(name: str) -> bool:
                if not name.startswith('mcp__'):
                    return False
                mcp_id = name.split('__')[1]
                entry = mcp_client.registry.get(mcp_id)
                if not entry:
                    return False
                meta = entry.get('tool_meta', {}).get(name)
                return bool(meta and meta.get('type') == 'sensor')

            results = []
            _batch = []
            for c in tool_calls:
                if _is_sensor(c['function']['name']):
                    _batch.append(c)
                else:
                    if _batch:
                        results.extend(await asyncio.gather(*[_dispatch(b) for b in _batch]))
                        _batch = []
                    results.append(await _dispatch(c))
            if _batch:
                results.extend(await asyncio.gather(*[_dispatch(b) for b in _batch]))

            # ── 把工具结果加入本轮消息 ────────────────────────────────────
            if results:
                decisions.append({
                    'round': round_idx,
                    'text': text,
                    'tool_calls': [
                        {'name': c['function']['name'], 'args': c['function'].get('arguments', '{}'),
                         'result': next((r['result'] for r in results if r['id'] == c['id']), None)}
                        for c in tool_calls
                    ],
                })
                turn_messages += [
                    {
                        'role':         'tool',
                        'tool_call_id': r['id'],
                        'content':      r['result'],
                    }
                    for r in results
                ]
            else:
                decisions.append({'round': round_idx, 'text': text, 'tool_calls': []})
                break

            # ── finish 检测 ───────────────────────────────────────────────
            if finish_tool in [c['function']['name'] for c in tool_calls]:
                break

            # ── 取消检查点：工具执行完毕后，下一轮 LLM 调用前 ────────────────
            if cancel_event and cancel_event.is_set():
                raise TurnCancelled("Interrupted after tool dispatch")

        # 检查是否需要压缩（保存由 run_forever 的 finally 统一处理）
        await self._maybe_compress()

        # 发布决策到 /decision_core DDS topic
        import ros2_bridge
        decision = {
            'text': response.get('content', '') if response else '',
            'decisions': decisions,
            'source': trigger_event.get('source', ''),
            'ts': time.time(),
        }
        ros2_bridge.publish('/decision_core', json.dumps(decision, ensure_ascii=False))

        await push_event({'type': 'turn_end', 'payload': {
            'rounds': round_idx + 1,
            'duration_s': round(_time.perf_counter() - _turn_t0, 2),
            'usage': _turn_usage,
        }})
        _turn_elapsed = _time.perf_counter() - _turn_t0
        print(f'[decision] turn complete: {_turn_elapsed:.2f}s total, {round_idx + 1} rounds')

        # 性能追踪：提交 spans
        _turn_end_ts = time.time()
        _spans.append({'span': 'turn_total', 'component': 'core',
                       'start_ts': _turn_start_ts, 'end_ts': _turn_end_ts})
        try:
            perf_log.commit_spans(
                trace_id=_trace_id,
                spans=_spans,
                source=trigger_event.get('source', ''),
                trigger_text=trigger_event.get('text', '')[:300],
            )
        except Exception as _pe:
            print(f'[perf_log] commit error: {_pe}')
