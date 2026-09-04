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

def _scrub(message_list: list[dict]) -> list[dict]:
    """丢弃结构非法的 tool_call，以及随之失去归属的 tool 结果。

    glm 偶发在正常 tool_call 后追加一条 id/name 均为空串的条目。它一旦落盘，
    历史续跑会把它一路带回去，之后每次请求都被服务端 400
    （messages[N].tool_calls[M].function missing required field "name"），
    直到那一轮从 tier1/tier2 里滚出去为止——表现就是「这台机器总是报错」。
    """
    from client.llm import _valid_tool_call
    orphaned: set = set()
    out: list[dict] = []
    for msg in message_list:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            good = [tc for tc in msg['tool_calls'] if _valid_tool_call(tc)]
            if len(good) != len(msg['tool_calls']):
                orphaned.update(
                    tc.get('id') for tc in msg['tool_calls']
                    if isinstance(tc, dict) and not _valid_tool_call(tc)
                )
                msg = {k: v for k, v in msg.items() if k != 'tool_calls'}
                if good:
                    msg['tool_calls'] = good
                elif not msg.get('content'):
                    continue  # 既无文本也无有效调用，整条丢掉
        elif msg.get('role') == 'tool' and msg.get('tool_call_id') in orphaned:
            continue
        out.append(msg)
    return out


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


def _trigger_channel_ids(trigger_event: dict) -> list[str]:
    """本轮触发事件里涉及的 channel_id 列表（消息平台来源）。

    collector 从原始 Channel 事件 JSON 携带真实 ID；不能从 ROS-safe topic
    反推，因为中文、空格等字符会被 slug/hash 转换。
    """
    channel_ids = (trigger_event.get('payload') or {}).get('channel_ids')
    if not isinstance(channel_ids, list):
        return []
    ids = []
    for channel_id in channel_ids:
        if isinstance(channel_id, str) and channel_id and channel_id not in ids:
            ids.append(channel_id)
    return ids


def _channel_tool_retry_message(trigger_event: dict, round_idx: int, text: str,
                                retry_consumed: bool = False) -> str:
    """首轮渠道回复漏掉工具调用时，给模型一次纠正机会。"""
    channel_ids = _trigger_channel_ids(trigger_event)
    if retry_consumed or round_idx != 0 or not channel_ids or not text.strip():
        return ''
    return (
        '[system correction]\n'
        f'This turn came from messaging channel(s): '
        f'{json.dumps(channel_ids, ensure_ascii=False)}. '
        'You produced reply text but called no tool, so nothing was delivered. '
        'If a reply is warranted, call the bound channel_reply tool now with action="send" '
        'and the reply text. If no reply is warranted, call finish. '
        'Do not return content-only text again.'
    )


def _missed_channel_reply_warning(trigger_event: dict, replied_message_ids: set[str]) -> str:
    """本轮触发涉及的 Channel 消息里，有没有一条都没被 channel_reply 覆盖到。

    只检测「整轮零工具调用」（_channel_tool_retry_message）覆盖不了多人合批的场景：
    一批里有 A、B 两条消息，模型只回复了 A，日志里看不出 B 被漏了。这里只产出一条
    告警文本，不强制重试——模型判断"B 这条不需要回复"也是合法结果。
    """
    channel_ids = trigger_event.get('_channel_message_ids')
    if not isinstance(channel_ids, list):
        return ''
    missed = [mid for mid in channel_ids if mid not in replied_message_ids]
    if not missed:
        return ''
    return (f'{len(missed)} channel message(s) in this batch got no channel_reply: '
            f'{json.dumps(missed, ensure_ascii=False)}')


def _acp_barrier_log(name: str, acp_result: dict | None) -> None:
    """记录 barrier 结果。timeout 与 completed 走同一条路（都清 pending 并放行），
    所以不打这一行的话，一次静默超时看起来跟成功完全一样 —— ACP 回调发不出去
    （自签证书、AGENT_CORE_URL 配错）时无从归因。"""
    if not isinstance(acp_result, dict):
        return
    status = acp_result.get('status')
    if status in ('timeout', 'cancelled', 'barge_in'):
        actions = acp_result.get('actions')
        detail = f": {actions}" if actions else ''
        print(f"[acp] barrier {status} before {name}{detail}")


# barrier 等待期间检查 steering_queue 的间隔。相对于它守护的动作时长（TTS 讲解
# 数十秒、导航更久）足够细，又不会把事件循环打满。
_STEERING_POLL_S = 0.1


# finish 结束 turn 前必须等 pending 动作完成，否则 speak → finish 会在音频播完前
# 结束本轮 —— 讲解被截断。其余系统工具不挡：task_update 等应能在播放期间调用，
# 否则每站都要多等一整段音频。
_ACP_BARRIER_SYSTEM_TOOLS = frozenset({'finish'})


def _sys_tool_needs_barrier(name: str) -> bool:
    return name in _ACP_BARRIER_SYSTEM_TOOLS


async def _acp_barrier(name: str, cancel_event, *, barge_in: bool = False,
                       interrupt_fallback=None) -> dict | None:
    """有 pending ACP 动作时等待其完成，并记录非正常结果。无 pending 则直接返回。

    `barge_in=True` 时，等待还会被新的用户消息唤醒。这条路只给 finish 用，因为
    只有 finish 的 barrier 会横跨整段播放：`cancel_event` 仅在 interrupt /
    followup 模式下置位（collector.py 的 `_interrupt_mode` 分支），默认的 steer
    模式把 busy 期间的用户消息塞进 steering_queue 就完事、不动 cancel_event，而
    finish 检测在 steering drain（`:1377`）之前就 break 了 —— 队列里的消息在本轮
    永远等不到消费，用户要一直等到讲解播完。唤醒后中止播放并放行，turn 正常结束，
    消息由 `_flush_all_pending()` 转成下一轮的触发事件；这正是加 barrier 之前的打
    断行为，只是不再依赖 "finish 提前返回" 这个副作用。

    turn 中段的 mcp 工具 barrier 不走这条：那里 steering 的语义是注入当前 turn
    （`:1377` 之后就会 drain），不是中止动作。
    """
    if not mcp_client.get_pending_actions():
        return None

    if not barge_in:
        result = await mcp_client.await_pending(cancel_event, timeout=120)
        _acp_barrier_log(name, result)
        return result

    barrier = asyncio.create_task(mcp_client.await_pending(cancel_event, timeout=120))
    steering = asyncio.create_task(_wait_for_steering())
    try:
        done, _ = await asyncio.wait(
            [barrier, steering], return_when=asyncio.FIRST_COMPLETED)
        if barrier in done:
            result = barrier.result()
        else:
            result = await _abort_pending_for_barge_in(interrupt_fallback)
    finally:
        # Reap, don't just request: cancel() alone leaves these pending until the
        # loop resumes them, and dropping the reference here is what orphaned
        # await_pending's inner tasks on every barge-in.
        await mcp_client.cancel_and_reap((barrier, steering))
    _acp_barrier_log(name, result)
    return result


async def _wait_for_steering(poll_s: float = _STEERING_POLL_S) -> None:
    """轮询到 steering_queue 非空为止。asyncio.Queue 没有 "非破坏性等待"，
    而 drain_steering() 会把消息取走 —— 取走了本轮 finish 之后就没人再放回去。"""
    while not collector.has_steering():
        await asyncio.sleep(poll_s)


async def _abort_pending_for_barge_in(interrupt_fallback=None) -> dict:
    """用户在播放期间说话：停掉正在进行的输出，清 pending，放 finish 过去。

    `interrupt_fallback` 是没有 on_interrupt_all 绑定时的兜底（硬编码找 tts/loco），
    与 run_forever 的 TurnCancelled 路径共用同一套逻辑。
    """
    import hooks
    actions = mcp_client.get_pending_actions()
    fired = await hooks.fire('on_interrupt_all')
    if not fired and interrupt_fallback is not None:
        await interrupt_fallback()
    for aid in actions:
        mcp_client._pending_actions.pop(aid, None)
        mcp_client._pending_results.pop(aid, None)
        mcp_client._pending_timeouts.pop(aid, None)
        mcp_client._pending_tools.pop(aid, None)
    return {"status": "barge_in", "actions": actions}


_BOT_READ_ONLY_SYSTEM_TOOLS = frozenset({'finish'})

# viewer 是已注册的合法 ACL 用户（不是可以随便伪造身份的群里 Bot），"只读" 的定义
# 可以松一档：纯信息检索、没有副作用的系统工具（联网搜索、历史/记忆检索、原始输入
# 查询）都算读，不算 mutate。Bot 门禁维持只给 finish，因为群里任何人都能顶一个
# Bot 身份，是更弱的信任边界。
_VIEWER_READ_ONLY_SYSTEM_TOOLS = _BOT_READ_ONLY_SYSTEM_TOOLS | frozenset({
    'WebSearch', 'search_history', 'memory_recall', 'raw_input_info',
})


def _restricted_channel_tool_allowed(name: str, *, bot_restricted: bool = True) -> bool:
    """Shared allow-list for turns that must not mutate or actuate: untrusted-bot
    Channel turns, and (see _viewer_channel_restricted) viewer-role human Channel
    turns. May read state and reply, nothing else — viewer additionally gets the
    read-only system tools (search, history/memory recall)."""
    if not name.startswith('mcp__'):
        allowed = _BOT_READ_ONLY_SYSTEM_TOOLS if bot_restricted else _VIEWER_READ_ONLY_SYSTEM_TOOLS
        return name in allowed
    if name == 'mcp__channel__channel_reply':
        return True
    parts = name.split('__')
    entry = mcp_client.registry.get(parts[1] if len(parts) > 1 else '', {})
    meta = entry.get('tool_meta', {}).get(name)
    return bool(meta and meta.get('type') in ('sensor', 'resource'))


def _bot_channel_restricted(trigger_event: dict) -> bool:
    return bool(trigger_event.get('_bot_channel_event')) and not bool(
        trigger_event.get('_trusted_bot_channel_event')
    )


def _viewer_channel_restricted(trigger_event: dict) -> bool:
    """True when every human Channel message in this turn's batch came from a
    'viewer' (or role-less) ACL user — no operator/owner present to justify
    unlocking actuator/processor tools.

    Before this, channel/acl.py's role levels were never enforced anywhere in
    the dispatch path — `user_role` was only informational text handed to the
    LLM. Any auto-approved Channel user could ask the model to call any bound
    tool, actuators included, regardless of their recorded ACL role.
    """
    return bool(trigger_event.get('_viewer_channel_event'))


def _channel_tool_restricted(trigger_event: dict) -> bool:
    return _bot_channel_restricted(trigger_event) or _viewer_channel_restricted(trigger_event)


def _bot_channel_reply_allowed(args: dict, source_message_ids: set[str]) -> bool:
    """Keep bot replies on the current inbound message and text-only."""
    return (
        args.get('source_message_id') in source_message_ids
        and not args.get('files')
    )


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


# ── Tiered Retention helpers ──────────────────────────────────────────────────

def _degrade_turn(turn: list[dict]) -> list[dict]:
    """降质 turn：tool results 截短，tool_calls 只留名称列表。用于 tier2 历史。"""
    degraded = []
    for msg in turn:
        if msg.get('role') == 'tool':
            content = msg.get('content', '')
            if isinstance(content, str) and len(content) > 80:
                msg = {**msg, 'content': content[:80] + '...'}
            elif isinstance(content, list):
                msg = {**msg, 'content': '(多模态内容已省略)'}
        elif msg.get('role') == 'assistant' and msg.get('tool_calls'):
            names = [tc['function']['name'] for tc in msg['tool_calls']]
            text = msg.get('content', '') or ''
            msg = {'role': 'assistant', 'content': (text + '\n[调用: ' + ', '.join(names) + ']').strip()}
        degraded.append(msg)
    return degraded


def _compact_turn_messages(turn_messages: list[dict], keep_recent: int = 12) -> None:
    """Turn 内 compaction：保留最近 keep_recent 条完整，早期消息的 tool results 截短。
    直接修改 turn_messages（in-place）。"""
    if len(turn_messages) <= keep_recent:
        return
    # 只压缩 [0 : -keep_recent] 范围内的消息
    compact_end = len(turn_messages) - keep_recent
    for i in range(compact_end):
        msg = turn_messages[i]
        if msg.get('role') == 'tool':
            content = msg.get('content', '')
            if isinstance(content, str) and len(content) > 150:
                turn_messages[i] = {**msg, 'content': content[:150] + '...(compacted)'}
            elif isinstance(content, list):
                turn_messages[i] = {**msg, 'content': '(多模态内容已省略)'}
        elif msg.get('role') == 'assistant' and msg.get('tool_calls'):
            # 保留 tool_calls 结构（API 需要），但截短 arguments
            new_calls = []
            for tc in msg['tool_calls']:
                args = tc.get('function', {}).get('arguments', '')
                if len(args) > 100:
                    new_tc = {**tc, 'function': {**tc['function'], 'arguments': args[:100] + '...'}}
                else:
                    new_tc = tc
                new_calls.append(new_tc)
            turn_messages[i] = {**msg, 'tool_calls': new_calls}


_REWRITE_SUMMARY_PROMPT = """将以下两段历史摘要合并为一段简洁摘要。
要求：保留活跃任务、关键决策、未完成事项。去除已完成/过时的细节。
最终控制在 {budget} 字以内，以「[历史摘要]」开头。

旧摘要：
{old}

新摘要：
{new}
"""


async def _rewrite_summary(old: str, new: str, budget: int = 5000) -> str:
    """合并两段摘要为固定预算内的单一摘要。"""
    try:
        resp = await client.call(
            message_list=[
                {'role': 'system', 'content': '你是高效的信息压缩器。'},
                {'role': 'user', 'content': _REWRITE_SUMMARY_PROMPT.format(budget=budget, old=old, new=new)},
            ],
            tool_list=[],
        )
        result = resp.get('content', '') or new
        # 硬上限兜底
        if len(result) > budget * 2:
            result = result[:budget * 2]
        return result
    except Exception as e:
        print(f'[decision] rewrite_summary failed: {e}')
        return new  # 失败时只保留新摘要


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


async def _memory_recall(
    query: typing.Annotated[str, "搜索关键词"],
    source: typing.Annotated[str, "来源过滤: 'all'=全部, 'subagent'=子代理结论, 'conversation'=对话历史"] = 'all',
    limit: typing.Annotated[int, "返回最多 N 条结果，默认 5"] = 5,
    time_range: typing.Annotated[str, "时间范围: '1h'/'6h'/'1d'/'7d'/'' (不限)"] = '',
) -> str:
    """从记忆库检索历史信息。包含过去的对话、subagent 分析结论等。当需要回顾历史状态、查找之前的任务结果时使用。"""
    import time as _time
    from config import _get_conn

    results = []
    now = _time.time()

    # 解析时间范围
    time_cutoff = 0
    if time_range:
        multipliers = {'h': 3600, 'd': 86400}
        unit = time_range[-1]
        try:
            num = int(time_range[:-1])
            time_cutoff = now - num * multipliers.get(unit, 3600)
        except (ValueError, IndexError):
            pass

    # 搜索 subagent_conclusions
    if source in ('all', 'subagent'):
        try:
            with _get_conn() as conn:
                # 分词搜索：将 query 按空格拆分，每个关键词都必须匹配（AND 逻辑）
                keywords = [k.strip() for k in query.split() if k.strip()]
                if not keywords:
                    keywords = [query]
                where_clauses = ' AND '.join(['(conclusion LIKE ? OR goal LIKE ?)'] * len(keywords))
                params = []
                for kw in keywords:
                    params.extend([f'%{kw}%', f'%{kw}%'])
                if time_cutoff > 0:
                    sql = (f'SELECT agent_id, goal, conclusion, source_type, created_at '
                           f'FROM subagent_conclusions WHERE ({where_clauses}) AND created_at > ? '
                           f'ORDER BY created_at DESC LIMIT ?')
                    params.extend([time_cutoff, limit])
                else:
                    sql = (f'SELECT agent_id, goal, conclusion, source_type, created_at '
                           f'FROM subagent_conclusions WHERE ({where_clauses}) '
                           f'ORDER BY created_at DESC LIMIT ?')
                    params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                for agent_id, goal, conclusion, source_type, ts in rows:
                    time_str = _dt.datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')
                    results.append({
                        'ts': ts,
                        'text': f'[{time_str}] [subagent:{agent_id}/{source_type}] {goal[:40]}\n{conclusion[:300]}',
                    })
        except Exception as e:
            print(f'[memory_recall] conclusions search error: {e}')

    # 搜索对话历史
    if source in ('all', 'conversation'):
        try:
            import chat_history
            hist_results = chat_history.search(query, limit=limit)
            for r in hist_results:
                if time_cutoff > 0 and r['ts'] < time_cutoff:
                    continue
                time_str = _dt.datetime.fromtimestamp(r['ts']).strftime('%m-%d %H:%M')
                results.append({
                    'ts': r['ts'],
                    'text': f'[{time_str}] [conversation] {r["preview"][:300]}',
                })
        except Exception as e:
            print(f'[memory_recall] history search error: {e}')

    if not results:
        return f'未找到与 "{query}" 相关的记忆。'

    # 按时间排序（最新在前），去重截断
    results.sort(key=lambda x: x['ts'], reverse=True)
    results = results[:limit]
    return '\n---\n'.join(r['text'] for r in results)


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
        self._bound_instance_ids: dict = {}   # full_name → card_id (canvas binding)

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
        from peer import delegation as _peer_delegation

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
            ('memory_recall', _memory_recall),
            ('subagent_spawn', _sa_tools.subagent_spawn),
            ('subagent_spawn_sync', _sa_tools.subagent_spawn_sync),
            ('subagent_status', _sa_tools.subagent_status),
            ('subagent_cancel', _sa_tools.subagent_cancel),
            ('subagent_message', _sa_tools.subagent_message),
            ('subagent_result', _sa_tools.subagent_result),
            ('peer_list', _peer_delegation.peer_list),
            ('peer_state', _peer_delegation.peer_state),
            ('peer_tools', _peer_delegation.peer_tools),
            ('peer_call', _peer_delegation.peer_call),
            ('peer_delegate', _peer_delegation.peer_delegate),
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
        self._bound_instance_ids = {}  # full_name → card_id (for multiInstance tools)
        for ec in exec_conns:
            if ec.get('fromCardId') not in core_card_ids:
                continue
            mcp_id = ec.get('toMcpId', '')
            tool_name = ec.get('toToolName', '')
            card_id = ec.get('toCardId', '')
            if not mcp_id or not tool_name:
                continue
            # 从 mcp_client registry 中取该工具的 schema
            info = mcp_client.registry.get(mcp_id)
            if not info or not info.get('online'):
                continue
            full_name = f"mcp__{mcp_id}__{tool_name}"
            if card_id:
                self._bound_instance_ids[full_name] = card_id
            schema = info.get('schemas', {}).get(full_name)
            if schema:
                schemas.append(schema)
            else:
                # 检查是否有拆分的子工具（x-action-params 拆分）
                for split_name in info.get('tool_groups', {}).get(tool_name, []):
                    s = info.get('schemas', {}).get(split_name)
                    if s:
                        schemas.append(s)
                        if card_id:
                            self._bound_instance_ids[split_name] = card_id

        # Peer tools are deliberately **not** added here. They reach the model
        # through the `peer_tools` / `peer_call` pair instead (peer/delegation.py),
        # which costs two schemas no matter how large the fleet gets.
        #
        # Expanding them was tried and does not scale: one schema measured 613
        # chars (~200 tokens), a fully-wired robot binds 8–15 tools, so ten peers
        # would add ~100 tools and ~20k tokens **per request**. Two things break
        # before the context limit does — tool-choice accuracy across a list that
        # long, and the prompt cache, since the tool list sits in the cached prefix
        # and a peer going offline or re-advertising rewrites it.
        if not schemas:
            # 没有绑定任何工具时，仅使用系统工具（不暴露全部 MCP 工具）
            return []

        return schemas

    # ── 打断：中止正在进行的输出 ─────────────────────────────────────────────

    async def _interrupt_active_outputs(self):
        """中止所有正在进行的输出（TTS + 动作）。在 TurnCancelled 时调用。
        优先使用 hook 系统；fallback 到硬编码查找。"""
        import hooks
        results = await hooks.fire('on_interrupt_all')
        if results:
            # Hook handled it — also clear pending ACP
            for aid in list(mcp_client._pending_actions.keys()):
                mcp_client._pending_actions[aid].set()
            print(f'[decision] interrupted via on_interrupt_all hook ({len(results)} binding(s))')
            return

        # Fallback: hardcoded lookup (no hook registered)
        #
        # registry[mcp_id]['tools'] holds *bare* plugin names ('tts', 'loco') --
        # mcp_client._connect_one does `tools.append(tool['name'])`. This used to
        # hand those straight to call_tool(), which parses its argument as a full
        # `mcp__<id>__<tool>` name: it split 'loco' into one segment, failed the
        # `len(parts) != 3` check and returned the *string* '工具名格式错误: loco'.
        # A returned string is not an exception, so the error check below never
        # fired and the log still announced "interrupted N active output(s)" --
        # while nothing had been interrupted, on any robot.
        #
        # call_tool_direct takes (mcp_id, bare_tool_name) and is the intended entry
        # point for hooks: it also skips the ACP barrier, which an interrupt must.
        #
        # Deliberately not interrupting switch_mode: aborting a posture change
        # partway is how a controlled descent becomes a fall, so a running
        # stand-up/lie-down is left to finish.
        tasks = []
        for mcp_id, info in mcp_client.registry.items():
            if not info.get('online'):
                continue
            tools = info.get('tools', [])
            for short_name, action in (('tts', 'interrupt'), ('loco', 'stop_move')):
                if short_name in tools:
                    tasks.append((f'{mcp_id}:{short_name}',
                                  mcp_client.call_tool_direct(mcp_id, short_name,
                                                              {'action': action})))
        if not tasks:
            print('[decision] interrupt_active_outputs: no tts/loco tool registered')
            return

        labels = [label for label, _ in tasks]
        results = await asyncio.gather(*[coro for _, coro in tasks],
                                       return_exceptions=True)
        ok = 0
        for label, r in zip(labels, results):
            if isinstance(r, Exception):
                print(f'[decision] interrupt_active_outputs: {label} raised: {r}')
            elif isinstance(r, dict) and r.get('error'):
                print(f'[decision] interrupt_active_outputs: {label} failed: {r["error"]}')
            else:
                ok += 1
        print(f'[decision] interrupted {ok}/{len(tasks)} active output(s) (fallback)')


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
                # 中止正在进行的 TTS 播放和动作
                await self._interrupt_active_outputs()
                await push_event({'type': 'turn_cancelled', 'payload': {}})
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f'[decision] error in _one_turn: {e}')
                # Fire on_error hook (LED feedback etc.)
                if not ev.get('_bot_channel_event'):
                    import hooks
                    asyncio.create_task(hooks.fire('on_error'))
                # 把错误也记入本轮消息
                self._current_turn.append({
                    'role': 'assistant',
                    'content': f'[错误] {type(e).__name__}: {e}',
                })
                await push_event({'type': 'error', 'payload': {'message': str(e)}})
            finally:
                collector.set_cancel_event(None)
                collector.set_busy(False)
                # Fire on_idle hook (LED state reset etc.)
                if not ev.get('_bot_channel_event'):
                    import hooks as _hooks_idle
                    asyncio.create_task(_hooks_idle.fire('on_idle'))
                # Bot 输入不进入共享历史，避免在后续人工 turn 中延迟执行。
                if self._current_turn and not ev.get('_bot_channel_event'):
                    self._save_current_turn(ev)

    def _save_current_turn(self, trigger_event: dict):
        """保存 _current_turn 到内存历史 + SQLite。"""
        turn = self._current_turn
        # 保存前 compact：截断大 tool results，减少 tier1 历史占用
        llm_cfg = config.main.get('event', {}).get('llm', {})
        save_compact_limit = llm_cfg.get('save_compact_chars', 500)
        for i, msg in enumerate(turn):
            if msg.get('role') == 'tool':
                content = msg.get('content', '')
                if isinstance(content, str) and len(content) > save_compact_limit:
                    turn[i] = {**msg, 'content': content[:save_compact_limit] + '...(trimmed)'}
                elif isinstance(content, list):
                    turn[i] = {**msg, 'content': '(多模态内容已省略)'}
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
        # 裁剪：保留 tier1 + tier2 + 少量缓冲（压缩在 _maybe_compress 中处理）
        llm_cfg = config.main.get('event', {}).get('llm', {})
        tier1 = llm_cfg.get('tier1_turns', 6)
        tier2 = llm_cfg.get('tier2_turns', 8)
        max_turns = llm_cfg.get('history_turns', tier1 + tier2 + 4)
        if len(self._turns) > max_turns:
            self._turns = self._turns[-max_turns:]

    # ── 单轮推理 ─────────────────────────────────────────────────────────────

    def _build_history(self) -> list[dict]:
        """从 _turns 构建 L3 历史（tiered retention: tier1 全量 + tier2 降质 + summary）。"""
        llm_cfg = config.main.get('event', {}).get('llm', {})
        tier1 = llm_cfg.get('tier1_turns', 6)
        tier2 = llm_cfg.get('tier2_turns', 8)

        n = len(self._turns)
        recent = self._turns[-tier1:] if n > tier1 else self._turns
        medium = self._turns[max(0, n - tier1 - tier2):max(0, n - tier1)]

        history = []
        # 前置历史摘要（如果有）
        if self._summary:
            history.append({'role': 'user', 'content': self._summary})
            history.append({'role': 'assistant', 'content': '好的，我已了解之前的对话背景。'})
        for turn in medium:
            history.extend(_degrade_turn(turn))
        for turn in recent:
            history.extend(turn)
        return _sanitize(_scrub(history))

    async def _maybe_compress(self):
        """检查历史是否需要压缩（基于轮数或字符数），压缩旧轮次为 rolling summary。"""
        llm_cfg = config.main.get('event', {}).get('llm', {})
        tier1 = llm_cfg.get('tier1_turns', 6)
        tier2 = llm_cfg.get('tier2_turns', 8)
        max_kept = tier1 + tier2
        threshold = llm_cfg.get('compress_threshold_chars', 80000)
        summary_budget = llm_cfg.get('summary_max_chars', 5000)

        # 触发条件1: 轮数超限
        need_compress = len(self._turns) > max_kept + 2
        # 触发条件2: 字符超限（兜底）
        if not need_compress:
            need_compress = _estimate_chars(self._turns) > threshold
        if not need_compress:
            return
        if len(self._turns) <= max_kept:
            return  # 不够分割，跳过

        # 分割：压缩旧的，保留最近的
        old_turns = self._turns[:-max_kept]
        recent_turns = self._turns[-max_kept:]

        print(f'[decision] compressing history: {len(old_turns)} old turns, keeping {max_kept} recent')
        summary = await _compress_turns(old_turns)
        # Rolling summary: 合并旧摘要（固定预算重写，而非无限拼接）
        if self._summary:
            summary = await _rewrite_summary(self._summary, summary, summary_budget)

        self._summary = summary
        self._turns = recent_turns
        print(f'[decision] compressed: kept {len(recent_turns)} recent turns, summary={len(summary)} chars')

    async def _one_turn(self, trigger_event: dict, cancel_event: asyncio.Event | None = None):
        import time as _time
        from uuid import uuid4
        _turn_t0 = _time.perf_counter()
        bot_restricted = _bot_channel_restricted(trigger_event)
        viewer_restricted = _viewer_channel_restricted(trigger_event)
        tool_restricted = bot_restricted or viewer_restricted
        bot_reply_source_ids = set(trigger_event.get('_bot_channel_message_ids', []))
        replied_message_ids: set[str] = set()

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

        # Fire on_thinking hook (non-blocking LED feedback etc.)
        import hooks
        if not bot_restricted:
            asyncio.create_task(hooks.fire('on_thinking'))

        # ── Auto-interrupt: 新用户 turn 开始时清除旧的 pending ACP ──
        if mcp_client.get_pending_actions() and not bot_restricted:
            _int_results = await hooks.fire('on_interrupt_all')
            if _int_results:
                for aid in list(mcp_client._pending_actions.keys()):
                    mcp_client._pending_results[aid] = {
                        "status": "cancelled",
                        "reason": "auto-interrupted by new user message",
                    }
                    mcp_client._pending_actions[aid].set()
                print(f'[decision] auto-interrupt: {len(_int_results)} hook(s) fired, pending cleared')
            else:
                # Fallback: 没有 hook 注册时用硬编码查找
                await self._interrupt_active_outputs()

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
        system_tools = list(self._sys_tools.values())
        if tool_restricted:
            system_tools = [
                tool for tool in system_tools
                if _restricted_channel_tool_allowed(tool['schema']['name'], bot_restricted=bot_restricted)
            ]
            bound_schemas = [
                schema for schema in bound_schemas
                if _restricted_channel_tool_allowed(schema['name'], bot_restricted=bot_restricted)
            ]
        all_tool_list = (
            [{'type': 'function', 'function': t['schema']} for t in system_tools]
            + [{'type': 'function', 'function': s} for s in bound_schemas]
        )
        # 绑定工具全名集合，用于 L2 环境快照过滤
        bound_tool_names = {s['name'] for s in bound_schemas}

        # ── 冻结 system message（turn 内复用，保证 prefix caching 命中）────
        frozen_system = prompt_mod.build_system(mcp_client.registry, bound_tool_names)

        finish_tool = 'finish'
        llm_cfg = config.main.get('event', {}).get('llm', {})
        max_rounds  = llm_cfg.get('max_rounds', 100)
        truncate_keep = llm_cfg.get('truncate_keep_rounds', 50)
        absolute_max = max_rounds * 5  # 绝对上限防死循环
        response    = None
        decisions   = []
        turn_messages = self._current_turn  # alias for brevity
        _turn_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cached_tokens': 0}

        round_idx = 0
        total_rounds = 0
        channel_reply_retry_consumed = False
        while True:
            # ── 绝对上限检查 ──────────────────────────────────────────────
            if total_rounds >= absolute_max:
                print(f'[decision] absolute max {absolute_max} reached, forcing end')
                break

            # ── 截断续跑：达到 max_rounds 时截断 turn_messages ────────────
            if round_idx >= max_rounds:
                if len(turn_messages) > truncate_keep:
                    turn_messages_new = [turn_messages[0]] + turn_messages[-truncate_keep:]
                    turn_messages.clear()
                    turn_messages.extend(turn_messages_new)
                round_idx = 0
                print(f'[decision] hit max_rounds={max_rounds}, truncated turn_messages to {len(turn_messages)}, continuing')
                await push_event({'type': 'turn_truncated', 'payload': {'kept': len(turn_messages), 'total_rounds': total_rounds}})

            # ── Turn 内 compaction：消息过多时压缩早期 tool results ────────────
            compact_threshold = llm_cfg.get('turn_compact_threshold', 30)
            compact_keep_recent = llm_cfg.get('turn_compact_keep_recent', 12)
            if len(turn_messages) > compact_threshold:
                _compact_turn_messages(turn_messages, compact_keep_recent)

            # ── 构建分层 prompt ────────────────────────────────────────────
            history = [] if bot_restricted else self._build_history()
            # 本轮已产生的消息也要加入历史（多轮工具调用场景）
            current_history = history + _sanitize(_scrub(turn_messages))

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
                    if len(self._turns) > 2 and not bot_restricted:
                        old = self._turns[:-2]
                        summary = await _compress_turns(old)
                        if self._summary:
                            llm_cfg = config.main.get('event', {}).get('llm', {})
                            budget = llm_cfg.get('summary_max_chars', 5000)
                            summary = await _rewrite_summary(self._summary, summary, budget)
                        self._summary = summary
                        self._turns = self._turns[-2:]
                        # 重建 history 并重试（复用冻结的 system）
                        history = self._build_history()
                        current_history = history + _sanitize(_scrub(turn_messages))
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
                _turn_usage['prompt_tokens'] += _usage.get('prompt_tokens') or 0
                _turn_usage['completion_tokens'] += _usage.get('completion_tokens') or 0
                _turn_usage['total_tokens'] += _usage.get('total_tokens') or 0
                _turn_usage['cached_tokens'] += _usage.get('cached_tokens') or 0
                await push_event({'type': 'llm_usage', 'payload': _usage})

            # ── 工具调用 ──────────────────────────────────────────────────
            tool_calls = response.get('tool_calls') or []

            def _needs_barrier(name: str, call_args: dict = None) -> bool:
                """actuator/processor 类型的 MCP 工具需要 ACP barrier。
                例外：在 on_interrupt_* hook 中注册的 tool+action 免 barrier。"""
                if not name.startswith('mcp__'):
                    return False
                parts = name.split('__')
                mcp_id = parts[1] if len(parts) > 1 else ''
                # 从 split_map 获取原始 tool name + action
                entry = mcp_client.registry.get(mcp_id)
                if not entry:
                    return False
                split_info = entry.get('split_map', {}).get(name, {})
                if split_info:
                    # Split tool: action is encoded in schema name
                    tool_name = split_info.get('tool', '')
                    action_name = split_info.get('action', '')
                else:
                    # Non-split tool: action comes from call args
                    tool_name = parts[-1] if len(parts) > 2 else ''
                    action_name = (call_args or {}).get('action', '')
                # 在 interrupt hook 中注册的 → 免 barrier
                import hooks
                if hooks.is_interrupt_binding(mcp_id, tool_name, action_name):
                    return False
                meta = entry.get('tool_meta', {}).get(name)
                if not meta:
                    return True  # 无 meta 默认 barrier（安全）
                return meta.get('type') not in ('sensor', 'resource')

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

                if tool_restricted and not _restricted_channel_tool_allowed(name, bot_restricted=bot_restricted):
                    reason = 'an untrusted bot source' if bot_restricted else 'a viewer-role source'
                    result = (
                        f'Error: this turn is tool-restricted ({reason}) and cannot call '
                        'mutating, actuator, processor, or delegated execution tools.'
                    )
                elif (bot_restricted and name == 'mcp__channel__channel_reply'
                      and not _bot_channel_reply_allowed(args, bot_reply_source_ids)):
                    result = (
                        'Error: bot-triggered replies must use the current source_message_id '
                        'and cannot send files.'
                    )
                elif name in self._sys_tools:
                    # ACP barrier: finish 之前等音频播完（见 _ACP_BARRIER_SYSTEM_TOOLS）。
                    # 等待期间用户开口要能立刻打断 —— 故 barge_in=True。
                    if _sys_tool_needs_barrier(name):
                        await _acp_barrier(
                            name, cancel_event, barge_in=True,
                            interrupt_fallback=self._interrupt_active_outputs)
                    result = await self._sys_tools[name]['object'](**args)
                elif name.startswith('mcp__'):
                    # ACP barrier: 有 pending 时，非 sensor/resource 工具等待所有 pending 完成
                    if _needs_barrier(name, args):
                        await _acp_barrier(name, cancel_event)
                    args['_trace_id'] = _trace_id
                    args['_cancel_event'] = cancel_event
                    # Inject instance_id from canvas binding (multiInstance tools need it)
                    if name in self._bound_instance_ids and 'instance_id' not in args:
                        args['instance_id'] = self._bound_instance_ids[name]
                    result = await mcp_client.call_tool(name, args)
                    # interrupt hook 绑定的工具执行后：清 pending + 通知其他绑定方
                    if not _needs_barrier(name, args) and mcp_client.get_pending_actions():
                        import hooks as _hooks
                        parts = name.split('__')
                        _mcp_id = parts[1] if len(parts) > 1 else ''
                        _entry = mcp_client.registry.get(_mcp_id, {})
                        _split = _entry.get('split_map', {}).get(name, {})
                        _tool = _split.get('tool', parts[-1] if len(parts) > 2 else '')
                        _act = _split.get('action', args.get('action', ''))
                        if _hooks.is_interrupt_binding(_mcp_id, _tool, _act):
                            for aid in list(mcp_client._pending_actions.keys()):
                                mcp_client._pending_results[aid] = {
                                    "status": "cancelled",
                                    "reason": "interrupted by user instruction",
                                }
                                mcp_client._pending_actions[aid].set()
                            # Fire hook to notify ALL registered parties (e.g. perception TTS)
                            _hook_id = _hooks.get_hook_for_binding(_mcp_id, _tool, _act)
                            if _hook_id:
                                asyncio.create_task(_hooks.fire(_hook_id, exclude_mcp_id=_mcp_id))
                            print(f'[acp] interrupt: cancelled pending + fired {_hook_id} (source: {_tool}.{_act})')
                else:
                    result = f'未知工具: {name}'

                if (name == 'mcp__channel__channel_reply' and args.get('source_message_id')
                        and isinstance(result, str) and not result.startswith('Error')):
                    replied_message_ids.add(args['source_message_id'])

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
                # 本轮由某个消息渠道触发，却一个工具都没调 → 用户那边是纯沉默：
                # content 不会送达任何人。曾经因为这个丢过回复，而日志里只有
                # 「turn complete: 1 rounds」看不出异常，只能去翻 llm_recent_request。
                retry_message = _channel_tool_retry_message(
                    trigger_event, round_idx, text, channel_reply_retry_consumed,
                )
                if retry_message:
                    channel_reply_retry_consumed = True
                    turn_messages.append({'role': 'user', 'content': retry_message})
                    print('[decision] channel-triggered turn produced content without a tool call; retrying once')
                    await push_event({'type': 'llm_retry', 'payload': {
                        'reason': 'channel_reply_tool_missing',
                    }})
                elif (channel_ids := _trigger_channel_ids(trigger_event)):
                    warn = (f'[decision] WARNING: channel-triggered turn produced no tool call — '
                            f'nothing was delivered to '
                            f'{json.dumps(channel_ids, ensure_ascii=False)}. '
                            f'content was: {(text or "")[:200]}')
                    print(warn)
                    await push_event({'type': 'error', 'payload': {'message': warn}})
                    break
                else:
                    break

            # ── finish 检测 ───────────────────────────────────────────────
            if finish_tool in [c['function']['name'] for c in tool_calls]:
                break

            # ── Rebuild frozen_system if skill state changed (activate/deactivate) ─
            skill_tools = {'activate_skill', 'deactivate_skill'}
            if any(c['function']['name'] in skill_tools for c in tool_calls):
                frozen_system = prompt_mod.build_system(mcp_client.registry, bound_tool_names)

            # ── Steering: 检查是否有用户消息需要注入 ─────────────────────────
            steered = await collector.drain_steering()
            if steered:
                deferred = [
                    sev for sev in steered
                    if bot_restricted or collector.has_bot_channel_event([sev])
                ]
                if deferred:
                    collector.defer_priority(deferred)
                    steered = [sev for sev in steered if sev not in deferred]
                if steered:
                    for sev in steered:
                        s_text = sev.get('text', '')
                        s_source = sev.get('source', '')
                        turn_messages.append({
                            'role': 'user',
                            'content': f'[system notification source={s_source}]\n{s_text}',
                        })
                    await push_event({'type': 'turn_steered', 'payload': {
                        'count': len(steered),
                        'sources': [s.get('source', '') for s in steered],
                    }})
                    print(f'[decision] steered {len(steered)} user message(s) into current turn')

            # ── 取消检查点：工具执行完毕后，下一轮 LLM 调用前 ────────────────
            if cancel_event and cancel_event.is_set():
                raise TurnCancelled("Interrupted after tool dispatch")

            round_idx += 1
            total_rounds += 1

        # ── 漏回复检测：这一批触发里有 Channel 消息，但没被任何一次 channel_reply
        # 覆盖到 —— 之前只检测「整轮零工具调用」，多人合批时漏回复其中一人完全
        # 无法从日志看出来，这里补一条可见的告警（不强制重试，避免误伤「确实不
        # 需要回复」的场景）。
        _missed_warn = _missed_channel_reply_warning(trigger_event, replied_message_ids)
        if _missed_warn:
            print(f'[decision] WARNING: {_missed_warn}')
            await push_event({'type': 'error', 'payload': {'message': f'[decision] {_missed_warn}'}})

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
            'rounds': total_rounds + 1,
            'duration_s': round(_time.perf_counter() - _turn_t0, 2),
            'usage': _turn_usage,
        }})
        _turn_elapsed = _time.perf_counter() - _turn_t0
        print(f'[decision] turn complete: {_turn_elapsed:.2f}s total, {total_rounds + 1} rounds')

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
