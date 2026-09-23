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


# 活动面板里一条触发事件能有多长。
#
# **原先是 200 字，而且只留开头。** 对一句人说的话够用，对 ACP 的完成事件正好切在
# 最有用的地方 —— `{"type": "action_complete", "action_id": …, "status": "failed",
# "result": {"status": "failed", "reason": …}}` 的样板部分就占掉一百八十几个字符，
# `reason` 在最后面，于是面板上永远停在 `"reaso`。而 `reason` 恰恰是唯一不能从别处
# 推出来的东西（「目标丢失」和「被挡住十几秒动不了」该做的下一步完全相反）。
#
# 这和 `collector._slim_acp_result` 修过的是同一类错，只是那次修的是送进 LLM 的那
# 一份，面板这一份被漏掉了。所以这里不只是把数字调大：超长时**两头都留**，并写明
# 中间省了多少 —— 不写明的话，半句话看起来就是全部。
_ACTIVITY_TEXT_CHARS = 2000
_ACTIVITY_TEXT_TAIL = 600


def _activity_text(text) -> str:
    text = str(text or '')
    if len(text) <= _ACTIVITY_TEXT_CHARS:
        return text
    head = _ACTIVITY_TEXT_CHARS - _ACTIVITY_TEXT_TAIL
    dropped = len(text) - _ACTIVITY_TEXT_CHARS
    return (f'{text[:head]}\n…（中间省略 {dropped} 字）…\n'
            f'{text[-_ACTIVITY_TEXT_TAIL:]}')


def _decoded_json(value):
    """Unwrap a JSON string into the value it encodes, or return it unchanged.

    A tool call's `arguments` is a *string* by the OpenAI schema, and providers
    emit it with non-ASCII escaped — so publishing it verbatim put things like
    `\\u4f60\\u597d` on /decision_core where `你好` was meant, defeating the
    `ensure_ascii=False` the surrounding dump already uses. Decoding it once here
    means the bus carries structured data rather than source text.

    Anything that is not a JSON object or array is returned as-is: a model that
    emits a malformed argument string should still show up on the bus as what it
    actually sent.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not (text.startswith('{') and text.endswith('}')) and \
       not (text.startswith('[') and text.endswith(']')):
        return value
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return value
    return parsed if isinstance(parsed, (dict, list)) else value


def _tool_content(result):
    """一个 tool role 消息的 `content`，保证是服务端收得下的形状。

    OpenAI 的 schema 里 tool 消息的 content 只能是字符串（或多模态 content part
    数组）。而 `_dispatch` 的返回值并不都是字符串：barrier 被打断时它返回
    `{"status": "not_dispatched", "reason": ...}` 这样的 dict，之前原样塞进
    content，请求就被上游以 400 拒掉。

    这不是偶发抖动，而是确定性的、并且会粘住：这条坏消息一旦进了 turn_messages，
    本轮后面每一次请求都带着它，于是整轮持续 400 直到 turn 结束。天轶上 24356 次
    历史请求里，带 dict content 的有 35 次，**没有一次拿到过回复**；日志里只有
    `UPSTREAM_PASSTHROUGH: The model service rejected this request.`，网关把上游
    的具体字段名吞掉了，所以从错误信息本身看不出是哪条消息的问题。

    子代理那条路径（`subagent/agent.py::_dispatch_tool`）一直是 json.dumps 收口的，
    这里补上同样的收口，而不是去逐个改 `_dispatch` 的返回值 —— 收口放在唯一的出口
    上，以后任何新的非字符串返回值都不会再把整轮打掉。
    """
    if isinstance(result, str):
        return result
    # 多模态 content part 数组（图片等）是 schema 允许的另一种形状，原样放行。
    if isinstance(result, list):
        return result
    if result is None:
        return ''
    if isinstance(result, (dict, int, float, bool)):
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return str(result)


def _sanitise_turn(turn: list) -> list:
    """把一轮历史消息里不合法的 tool content 收干净。

    修复之前写进 `chat_messages` 的 dict content 还躺在每台已部署机器的 data.db
    里，而 `__aenter__` 的重启续跑会把最近 10 轮原样读回 `_turns` —— 也就是把同一个
    400 一起续跑回来，升级并不能自愈。这里在读回来的那一侧再过一遍。
    """
    if not isinstance(turn, list):
        return turn
    out = []
    for msg in turn:
        if (isinstance(msg, dict) and msg.get('role') == 'tool'
                and not isinstance(msg.get('content'), (str, list, type(None)))):
            msg = {**msg, 'content': _tool_content(msg.get('content'))}
        out.append(msg)
    return out


# ── Turn 取消异常 ────────────────────────────────────────────────────────────────

class TurnCancelled(Exception):
    """用户消息抢占时抛出，中断正在进行的 sensor turn。"""
    pass


class RoundReconsider(Exception):
    """`reconsider_event` 打断了一次还在飞的 LLM 请求时抛出。

    跟 `TurnCancelled` 不同：不结束这个 turn，`turn_messages` 也不清空——调用方
    （`_one_turn` 的主循环）捕获后原地把新到的 steering drain 进上下文，直接发起
    新一轮请求，不把这次失败的请求记进历史。"""
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

def _compact_turn(turn: list[dict]) -> list[dict]:
    """返回一份截断了大 tool result 的副本，用于存历史（内存 + SQLite）。

    返回副本而不是原地改：turn 跑到一半也要落盘一次，那时原 turn 还要接着喂给 LLM，
    不能被截断过的内容替换掉。
    """
    llm_cfg = config.main.get('event', {}).get('llm', {})
    limit = llm_cfg.get('save_compact_chars', 500)
    out = []
    for msg in turn:
        if msg.get('role') == 'tool':
            content = msg.get('content', '')
            if isinstance(content, str) and len(content) > limit:
                out.append({**msg, 'content': content[:limit] + '...(trimmed)'})
                continue
            if isinstance(content, list):
                out.append({**msg, 'content': '(多模态内容已省略)'})
                continue
        out.append(msg)
    return out


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
#
# 让**别人**去动手的系统工具同样必须挡，理由和 finish 一样但更隐蔽：交出去之后，对方第
# 一件事往往就是占用同一个物理资源。这些工具不是 `mcp__` 前缀，所以 `_needs_barrier` 在
# 第一行就返回 False —— 只有出现在这个集合里才会被拦。
#
# Orin5 实测，五轮相声每轮同一个形状：
#
#   16:00:20.863  registered pending: speak-406f7423   ← 自己开始说（实播 14.1s）
#   16:00:23.579  peer_call(让 Orin6 说)               ← 2.7s 后就叫对端开口
#   16:00:31.186  barrier: waiting ['speak-406f7423']   ← 直到自己下一句才拦
#
# 间隔 2.7 / 1.7 / 2.9 / 2.5 / 2.6 秒，自己那句要播 5–14 秒：两张嘴每轮都重叠。
# 它对自己的嘴一直是守 barrier 的，只是从没把"让别人说"算成一次输出。
#
# 这里**穷举**而不是按名字猜：上一版只加了 peer_delegate 和 subagent_spawn，漏掉的
# peer_call 恰好就是这次实测用的那扇门。`test_handoff_tools_are_partitioned` 会强制新增的
# peer_* / subagent_* 工具二选一落到这里或下面的只读集合，免得再漏第三扇。
_HANDOFF_SYSTEM_TOOLS = frozenset({
    'peer_call',            # 直接调对端的工具 —— 对端立刻执行
    'peer_delegate',        # 对端起 subagent 干活
    'subagent_spawn',       # 本机起 subagent
    'subagent_spawn_sync',
    'subagent_message',     # 可能唤醒一个在等指令的 subagent 去动手
})

# 只读 / 纯管理，不会让任何人开始动作 —— 不挡，否则每次查状态都要等一整段音频。
_READ_ONLY_HANDOFF_TOOLS = frozenset({
    'peer_list', 'peer_state', 'peer_tools',
    'subagent_status', 'subagent_result', 'subagent_cancel',
})

_ACP_BARRIER_SYSTEM_TOOLS = frozenset({'finish'}) | _HANDOFF_SYSTEM_TOOLS


def _sys_tool_needs_barrier(name: str) -> bool:
    return name in _ACP_BARRIER_SYSTEM_TOOLS


def _turn_ends_on_finish(tool_calls: list, finish_tool: str, finish_deferred: bool) -> bool:
    """`finish` 出现在 tool_calls 里，是否意味着本轮 turn 该结束。

    不等价 —— 出现不代表执行了。新消息在 finish 的 ACP barrier 期间到达时这次 finish
    被作废（`_dispatch` 的 sys-tool 分支），turn 必须接着走，否则就退回"播报多长、延迟
    多长"：Orin5 实测一条消息在队列里躺了 33.6 秒，正好是播报的剩余时长。

    做成模块级谓词而不是内联条件，是为了能在不搭起整个 `_one_turn`（prompt 构建、
    client、mcp registry、DB）的情况下测到它 —— 与 `subagent/manager.py` 的
    `notify_suppression_reason` 同样的理由。
    """
    if finish_deferred:
        return False
    return finish_tool in [c['function']['name'] for c in tool_calls]


async def _acp_barrier(name: str, cancel_event, *,
                       want: frozenset | None = None,
                       scoped: bool = False,
                       concurrent: bool = False,
                       reconsider_event=None) -> dict | None:
    """有 pending ACP 动作时等待其完成，并记录非正常结果。无 pending 则直接返回。

    `scoped=True` 时只等与 `want`（本次调用要占用的物理资源）冲突的 pending；
    `scoped=False` 等全部。系统工具走全局：`finish` 结束 turn 前不该有任何动作在飞，
    `peer_delegate` / `subagent_spawn` 把活交给另一个执行者、无法预知对方会碰什么。

    **只有显式 interrupt 能中止播放。** 这个 barrier 不会因为"来了新消息"就掐掉正在
    播的音频：打断走 `cancel_event`（collector 的 interrupt / followup 模式置位，或
    interrupt 工具）。

    finish 的 barrier 以前额外被 steering 队列唤醒（barge_in），本意是"用户开口就停"。
    实际上队列里什么都算数：Orin5 上一个后台 subagent 跑完的完成通知掐掉了主 agent
    正在播的上一份汇报；改成只认人发的消息之后，用户在网页里打下一个问题同样会把上
    一份汇报冲掉 —— 因为在这条路上"发消息"和"要求停止"根本无法区分。既然架构里已经
    有 interrupt 这个显式入口，隐式打断整条去掉了。

    新消息走的是 `reconsider_event`，语义完全不同、**不掐音频**：它只放弃"这次调用还
    没发出去、还在排队"的这一次等待，返回 `{"status": "reconsidering"}` 而**不**
    `_forget_pending`（见 `mcp_client.await_pending`），正在播的东西原样继续。调用方
    据此作废这次调用、把新消息 drain 进当前 turn 重新推理 —— 即"播边想"。finish 走这条
    时还要额外阻止 turn 结束，见 `_dispatch` 的 sys-tool 分支与 finish 检测处。
    """
    if not mcp_client.get_pending_actions():
        return None

    result = await mcp_client.await_pending(cancel_event, timeout=120,
                                           want=want, scoped=scoped,
                                           concurrent=concurrent,
                                           reconsider_event=reconsider_event)
    _acp_barrier_log(name, result)
    return result


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


def _round_already_notified(tool_calls: list) -> bool:
    """True if this round's tool_calls already exercised whatever action
    on_notify would also fire — the LLM replied itself, so auto-notify should
    not double-say it.

    Must resolve through `mcp_client.resolve_tool_binding` rather than parsing
    the tool_call name directly: a device using x-action-params (e.g. a `tts`
    tool split into `tts__speak`/`tts__interrupt`/...) never has a literal
    `action` argument or a name ending in the base tool name, so a name-suffix
    or args["action"] check silently never matches for it — every round then
    looks "not yet notified" even when the LLM just spoke, and on_notify fires
    a second, redundant call on top of the LLM's own.
    """
    for call in tool_calls:
        name = call.get('function', {}).get('name', '')
        try:
            args = json.loads(call['function'].get('arguments') or '{}')
        except (json.JSONDecodeError, TypeError):
            args = {}
        resolved = mcp_client.resolve_tool_binding(name, args)
        if not resolved:
            continue
        mcp_id, tool_name, action = resolved
        import hooks
        if hooks.get_hook_for_binding(mcp_id, tool_name, action) == 'on_notify':
            return True
    return False


def _notify_fire_spoke(results: list | None) -> bool:
    """True if at least one on_notify binding actually reached a device.

    hooks.fire returns one entry per binding: {'error': ...} on exception, else
    {'result': <tool return>}. With barrier_aware=True a binding whose mouth was
    already held comes back as the literal {"skipped": "resource busy"}
    (mcp_client.call_tool_hook) — the user heard nothing new from *this* fire,
    so it must not be counted as an interaction.
    """
    for r in results or []:
        if not isinstance(r, dict) or 'error' in r:
            continue
        res = r.get('result')
        if isinstance(res, dict) and ('skipped' in res or 'error' in res):
            continue
        return True
    return False


def _narration_enabled() -> bool:
    """自动播报（框架代为出声）的总开关。

    键名是 auto_narration 而不是旧的 auto_notify：旧键管的是"把 content 念出来"，那个
    功能已经废除。沿用旧键会让当初为了"别念内部推理"而关掉它的机器人，静默地把这个新
    功能也一起关死 —— 见 config.py 里 _narration_removed 那段。

    event.skills (attribute) is rebound to a Tools() instance by
    event/__init__.py, shadowing the submodule — get_notify_override is a
    module-level function, so it must come from sys.modules.
    """
    import sys as _sys
    override = _sys.modules['event.skills'].get_notify_override()
    if override is not None:
        return bool(override)
    return bool(config.main.get('event', {}).get('llm', {}).get('auto_narration', True))


def _narration_thresholds() -> tuple[int, int]:
    """(_, seconds) —— 运行时覆盖（set_progress_report）优先于 DB 配置。

    第一个元素保留成 0 只是为了不改调用点的解包形状；轮数维度已经删掉（纯时间触发，
    定时器全局负责，turn 在不在跑都一样）。
    """
    import sys as _sys
    llm_cfg = config.main.get('event', {}).get('llm', {})
    seconds = int(llm_cfg.get('narration_silence_seconds', 15))
    ov = _sys.modules['event.skills'].get_report_override()
    if ov:
        seconds = int(ov.get('seconds', seconds))
    return 0, seconds


def _normalize_report(text: str) -> str:
    """比"说过没说过"时用的归一形式：去掉空白和标点，只看内容。

    只做精确/归一后相等的判断，不做模糊相似度 —— 后者需要一个阈值，而阈值调错的代价是
    把真正的新进展也压掉（用户就此听不到），比偶尔重复一句更糟。
    """
    return ''.join(ch for ch in (text or '') if ch.isalnum())


def _same_as_last_report(text: str) -> bool:
    return bool(text) and _normalize_report(text) == _normalize_report(_last_report_text_global)


_last_narration_gate: str = ''


def _narration_gate(reason: str, *, stop: bool) -> None:
    """记下这次为什么没播，并按 reason 决定停表还是重新计时。

    只在**原因变化时**打印 —— 每 15 秒刷一行同样的话没人看，但一次都不打印就会像
    Orin5 这次一样：子代理跑了 9 轮、一条 narration 没有、也没有任何 skipped 记录，
    完全无从判断是"计时死了"还是"被 gate 挡了"。
    """
    global _last_narration_gate
    if reason != _last_narration_gate:
        _last_narration_gate = reason
        print(f'[decision] narration gate: {reason}')
    if stop:
        _stop_countdown()
    else:
        _restart_countdown()


def _human_duration(seconds: float) -> str:
    """把秒数说成人话。播报是念出来的 —— "107 秒"没人这么讲。"""
    s = max(0, int(seconds))
    if s < 60:
        return f'{s} 秒'
    m, rest = divmod(s, 60)
    if m < 60:
        return f'{m} 分钟' if rest < 15 else f'{m} 分半' if rest < 45 else f'{m + 1} 分钟'
    return f'{m // 60} 小时 {m % 60} 分钟' if m % 60 else f'{m // 60} 小时'


def _narration_feedback_message(report: str) -> dict:
    """回灌给主 LLM 的一条消息：你已经说过了，别再说一遍。

    role 用 user 而不是 assistant —— 一条没有 tool_calls 的 assistant 消息插在工具结果
    和下一次请求之间虽然合法，但把作者标错了。`[system notification source=...]` 这个
    形状本文件已有四处，_scrub/_sanitize 和保存路径都认。
    """
    return {'role': 'user', 'content':
            f'[system notification source=narration]\n'
            f'已自动替你向用户播报了一句进展：「{report}」。'
            f'不要重复这些内容；继续手头的工作。'}


# 哑火护栏的提示语。以前模型只写 content 就 finish 还能被 content 自动播报救回来，
# 现在那条路没了，这种 turn 会彻底没声 —— 而它恰恰是最常见的短问答形态。
_NO_OUTPUT_RETRY_MESSAGE = (
    '[system notification source=no_output]\n'
    '你这一轮只写了 content，没有通过任何工具把它说出口，用户什么也没听到。\n'
    '如果这句话是要给用户的，现在调用播报工具说出来；如果不是，直接 finish。'
)


def _needs_barrier(name: str, call_args: dict = None) -> tuple[bool, frozenset | None]:
    """这个 MCP 工具调用要不要 barrier，以及它想占用哪些物理资源。

    返回 `(needs, want)`。`needs=False` 时 `want` 无意义。`want=None` 表示工具没声明
    `x-resource` —— 按保守解读当作与一切互斥，等于旧的全局行为。

    actuator/processor 类型才挡；sensor/resource 放行。例外：在 on_interrupt_* hook
    中注册的 tool+action 免 barrier（打断本身不能被 barrier 挡住）。

    模块级函数，因为主循环和 subagent 都要用同一套判定 —— 它原来是 `_dispatch` 里的
    嵌套函数，subagent 那条路（subagent/agent.py 的 MCP 分支）因此完全没有 barrier。
    """
    if not name.startswith('mcp__'):
        return False, None
    parts = name.split('__')
    mcp_id = parts[1] if len(parts) > 1 else ''
    # 从 split_map 获取原始 tool name + action
    entry = mcp_client.registry.get(mcp_id)
    if not entry:
        return False, None
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
        return False, None
    meta = entry.get('tool_meta', {}).get(name)
    if not meta:
        return True, None    # 无 meta 默认 barrier 且当全局独占（安全）
    if meta.get('type') in ('sensor', 'resource'):
        return False, None
    return True, meta.get('resource')


# Sources whose events are this machine talking to itself: a subagent finishing, an
# ACP completion callback, a scheduled tick. None of them is somebody asking us to
# stop. Vocabulary matches collector.py's `_PRIORITY_SOURCES`, which is what decides
# who gets to wake the main loop at all — the two lists are meant to be read together.
_SELF_ORIGINATED_SOURCE_KEYS = ('subagent', 'acp', 'scheduler')


def _trigger_is_self_originated(trigger_event: dict) -> bool:
    """True when every event in this turn's batch came from us, not from outside.

    Used to decide whether starting a turn should abort whatever is currently
    playing. The auto-interrupt below is documented as firing on a "new user turn",
    but its only guard was `get_pending_actions() and not bot_restricted` — it never
    looked at *who* triggered the turn. A subagent's own completion event arrives
    URGENT the instant it finishes, so the main loop woke up and interrupted the TTS
    that same subagent had just queued. On Orin6 that is most of why a robot asked to
    speak stayed silent: `interrupted 1/2 active output(s)`, once per line.

    Deliberately conservative in the other direction. This returns True only when we
    can positively account for every source as self-originated; an empty or unknown
    `sources` list falls through to interrupting, as before. Failing to interrupt when
    a person actually speaks is the worse bug of the two, so an unenumerated barge-in
    path keeps its old behaviour rather than silently losing it.
    """
    sources = (trigger_event.get('payload') or {}).get('sources')
    if not isinstance(sources, list) or not sources:
        return False
    for src in sources:
        low = str(src).lower()
        if not any(key in low for key in _SELF_ORIGINATED_SOURCE_KEYS):
            return False
    return True


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


# ── 主动播报（framework-generated progress narration）─────────────────────────

_NARRATION_PROMPT = """[系统] 你已经有一段时间没有对用户说过任何话了，用户正在等你，
而且他看不到你的屏幕、不知道你在做什么。用**一句话**告诉他进展。

只输出那一句话。不要调用任何工具，不要写别的内容。

这句话每 15 秒左右才有一次，它的作用是**承上启下**：让还在等的人知道事情在往前走、
下一步是什么。所以重心放在**当前和接下来**，已经做完的事只作铺垫，一句带过。

按这个顺序（没有的就跳过，不要硬凑）：
1. 正在做什么 / 接下来要做什么 —— 这是重点
2. 支撑它的那件已完成的事，以及其中值得说的发现（具体的数字、名称、结论）

不要把已完成的事说成"刚刚完成"的样子 —— 它可能是十几秒前的事了，听起来会像机器人
在原地复述。

下面的"过程记录"只包含**上次汇报之后新发生的事**，所以直接讲这些新东西就行，
不用再把之前说过的重复一遍、也不用做总结。

反面例子（这几种都等于没说，不要输出）：
- "还在处理中，完成后告诉你"
- "正在整理信息，马上就好"
- "任务进行中，请稍候"
共同毛病是：把目标换个说法念了一遍，用户听完还是不知道进行到哪了。

结尾也不要凑话。"马上整理成报告""很快就好""稍后告诉你"这类收尾没有任何信息量，
说完最后一件具体的事就停住。

正面例子（都落在"接下来"上，已完成的部分只是铺垫）：
- "营收和机构评级查到了，正在算估值这一块"
- "客厅和厨房都没有，接着去卧室找"
- "第一家店关门了，正在看附近还有哪几家开着"

其他要求：
- 口语，会被直接念出来。不要 markdown、编号、括号注释、工具名、文件路径。
- 一两句话，把事情说清楚就停。**不要为了短而把名字、地点砍掉或缩写** —— 说完整的
  名字比省几个字重要。
- **时长只能用下面「当前等待」里给出的数，一个字都不要自己算。** 过程记录里那些
  时间戳不是拿来做减法的 —— 真机上照着它们算，算出过"大约等了 25 秒"这种数。
  「当前等待」里没有的，就别提时长。
- 用用户的语言。
- 只说过程记录里**真实发生过**的事，没查到的别编。
{last}
如果过程记录里确实还没有任何具体进展（比如刚开始、还没拿到任何结果），
只输出 SKIP 三个字母 —— 这种时候沉默比说一句空话好。

当前等待（这是唯一可靠的时长来源）：
{waiting}

过程记录：
{context}
"""

# 播出去之前的上限。这段话会注册成一个 ACP pending，超时按 len(text)/3 + 10 算，下一个
# 需要 barrier 的工具调用（含 finish）都得等它播完 —— 所以不能无限长。
#
# 但**从中间砍是错的**：真机上把人名、展位名砍成半截播了出去，听的人只会以为机器人
# 出故障了。宁可多播几个字，也不要播一个断掉的词。所以放宽上限，并且只在句读处收尾。
_NARRATION_MAX_CHARS = 200
_SENTENCE_ENDS = '。！？!?；;…'


def _trim_narration(text: str, limit: int = _NARRATION_MAX_CHARS) -> str:
    """超长时在**句读处**收尾，不从字中间砍。

    截到一半的名字比长一点的句子糟得多：用户听到的是「我们到了算力工」然后戛然而止。
    找不到句读就整句退回上一个逗号；再找不到，才认了硬截 —— 但那时至少已经尽力。
    """
    text = (text or '').strip()
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max((window.rfind(ch) for ch in _SENTENCE_ENDS), default=-1)
    if cut < limit // 3:                       # 句读太靠前，整句都没了，退而求其次
        cut = max((window.rfind(ch) for ch in '，,、 '), default=-1)
    return window[:cut + 1].strip() if cut > 0 else window

# 「一台机器上没有任何 on_notify 绑定」只值得说一次，但必须说 —— 见 _report_progress。
_warned_no_notify = False


# ── 沉默计时状态机 ──────────────────────────────────────────────────────────
#
# 量的是一件事：**用户在有活干的情况下，连续多久没听到任何东西**。
#
# **定时器的存在性就是状态**，没有第二个变量。没有时钟、没有"播放期间冻结"——正在说话
# 时根本不存在定时器，自然就不会把播放时长算进沉默。
#
#   事件                              动作
#   ───────────────────────────────────────────────────────────────────────
#   面向用户的输出被派发（开始说话）   _stop_countdown()；该输出若无 ACP 跟踪
#                                      （不会有"说完"事件）则当场 _restart_countdown()
#   打断（on_interrupt_* / barge-in）  _stop_countdown()
#   说完了（ACP 完成 / 超时 / 取消）   有活 → _restart_countdown()；无活 → _stop_countdown()
#   任务开始（turn 开始）              _start_countdown()  —— 已在计时则不动
#   活全干完（turn 结束且无子代理）    _stop_countdown()
#   轮边界                             不在说话且没在计时 → _start_countdown()（不变式兜底）
#   到点                               有活且 gate 全过 → 汇报；汇报没播成 → _restart_countdown()
#
# 打断归入"停止计时"而不是"嘴空了→开始计时"：被打断时用户正在说话，机器人根本不处于
# 沉默等待状态；紧接着到来的新请求会通过"任务开始"重新计时。这也省掉了三处打断路径
# 各自去开始计时的需要。
#
# 代价是留下两个口子——TurnCancelled 时的 _interrupt_active_outputs（turn 没了但子代理
# 可能还在跑）、以及模型自己调 interrupt 类工具（turn 继续走）——两者都是"取消了但没有
# 新任务来重新计时"。与其枚举每一条"谁负责重新计时"（本次已数出 4 条绕过监听的路径，
# 说明枚举法不可靠），轮循环末尾放一条不变式恢复兜底。
_silence_countdown: 'asyncio.Task | None' = None    # 存在 ⇒ 正在计时
_narration_inflight: bool = False
# 最后一次汇报的内容，跨 turn 保留 —— "别重复上次说过的话"要跨 turn 成立（turn 结束后
# 子代理接着跑，是同一件事的延续）。
_last_report_text_global: str = ''
# 上一个 turn 是不是受限（不可信 bot / viewer）。受限 turn 本来就不播任何东西，它派出去
# 的活也不该在 turn 结束后被代为出声。
_last_turn_restricted: bool = False
# 汇报播出后要告诉主 LLM"这句话已经替你说了"。不能当场 append 进 turn_messages：定时器
# 可能在 assistant(tool_calls) 已入列、它的 tool 结果还没入列的窗口触发，那时插一条 user
# 消息会造出 assistant(tool_calls) → user → tool 的序列，多数 provider 直接拒。改成排队，
# 由轮循环在排空 steering 的同一个安全点一起灌进去。
_pending_narration_feedback: list = []
# 上次汇报时每个子代理跑到第几轮。只讲那之后的新进展 —— 不然每次都是一个前后重叠的滑动
# 窗口，模型只能把累积状态重新总结一遍，越说越像（Tianyi 实测连着三条播报，第三条几乎是
# 第二条加一个词，末尾都靠"马上整理成报告"凑数）。
#
# 记轮数而不是 turns 列表长度：后者会被子代理自己的上下文压缩改短（Orin5 实测 round 6 时
# msgs 从 21 掉到 18），一压缩水位就大于长度、切片为空，轮数明明在涨却被判成"没出新结果"。
_reported_rounds: dict = {}
# turn 内已经喂过汇报器的消息条数。和子代理那边的水位是同一件事的两半 —— 之前只做了
# 子代理那半，turn 内每次仍把整个 turn 喂过去，模型就从里面自己挑，两次播报之间会跳掉
# 中间过程（Tianyi 实测：说完"正在从百度百科提取照片"，下一句直接变成"首都之窗的页面
# 抓下来了"，而百度百科 403 被拒这条线索用户从没听到，听着很割裂）。
_reported_turn_msgs: int = 0
# **上次看到新进展的时刻**。卡顿时用它算"多久没出新结果"。
#
# 不能记成"进入 idle 的时刻"：那样第一条卡顿播报会在同一次调用里先把它设成 now、再拿它
# 算时长，播出来就是"已经跑了 0 秒没出新结果"（Orin5 实测原话），自相矛盾。
_last_progress_ts: float | None = None


def _remember_report(text: str) -> None:
    global _last_report_text_global
    _last_report_text_global = text


def _mouth_busy() -> bool:
    """嘴（或任何 on_notify 绑定占用的物理通道）此刻是不是真被占着。"""
    import hooks
    return hooks.notify_resource_busy('on_notify')


# ── 三个原语：别处不许直接碰 _silence_countdown ────────────────────────────

def _stop_countdown() -> None:
    """解除计时。"""
    global _silence_countdown
    if _silence_countdown is not None and not _silence_countdown.done():
        _silence_countdown.cancel()
    _silence_countdown = None


# 沉默是从哪一刻开始的。播报要说「已经等了多久」，这是唯一说得准的那个数 ——
# 播报那次 LLM 调用看不到当前时刻，让它自己从历史时间戳里凑，凑出来的就是天轶上
# 那句「大约等了 25 秒」。
_silence_since: float | None = None


def _start_countdown() -> None:
    """开始计时；**已在计时则不动**（不把正在跑的 deadline 往后推）。"""
    global _silence_countdown
    if _silence_countdown is not None and not _silence_countdown.done():
        return
    _spawn_countdown()


def _restart_countdown() -> None:
    """重新计时；无论如何都从现在重新计满一个间隔。"""
    _stop_countdown()
    _spawn_countdown()


def _spawn_countdown() -> None:
    global _silence_countdown, _silence_since
    _silence_since = time.time()      # 用户从这一刻起没再听到任何东西
    _, seconds_thr = _narration_thresholds()
    if seconds_thr <= 0:
        _silence_countdown = None
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _silence_countdown = None    # 没有运行中的事件循环（测试 / 启动早期）
        return                       # 先判再建协程，否则留下一个 never awaited
    _silence_countdown = loop.create_task(_countdown_body(seconds_thr))


async def _countdown_body(seconds: float) -> None:
    """一次性：睡够就汇报。没有循环、没有轮询、没有剩余时间计算。"""
    try:
        await asyncio.sleep(seconds)
        inst = _event_instance
        if inst is not None:
            await inst._report_progress()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f'[decision] narration countdown error: {e}')
        _restart_countdown()


# ── 事件回调：名字照着上面那张表，代码读起来就是状态机 ──────────────────────

def _on_speaking_started() -> None:
    """一次面向用户的输出被派发出去了。"""
    _stop_countdown()
    # 设备没声明 x-completion 时 call_tool_hook 不注册 pending，嘴从来不"忙"，
    # 也就永远不会有"说完"事件来重新计时 —— 那样播报会永久静音。这种输出本来就是
    # 即时的（文本送达 / 无跟踪），当场重新计时。
    if not _mouth_busy():
        _restart_countdown()


def _on_speaking_finished() -> None:
    """某个动作播完了（完成回调 / 超时 / 取消都会走到这里）。

    **嘴还忙也照样上表**，不要在这里 return。这条早先是"忙就直接返回"，留下一条死路：
    嘴上可能还挂着别人的动作（天轶的驱动自己也注册 mouth 动作，barrier 日志里见过
    ['28', 'speak-…', 'tts-…'] 三个一起 want=mouth），那个动作的完成若没被观察到，就
    再也没有人来重新计时 —— 只能等轮边界兜底。实测因此空了 70 秒。

    "正在说话时不该汇报"这件事由到点时的 mouth busy gate 负责，那条会重新计时而不是
    停表，所以放它过来是安全的：计时继续跑，说完了自然就播。
    """
    if _active_work_summary() or _turn_running():
        _restart_countdown()
    elif not _mouth_busy():
        _stop_countdown()


def _on_task_started() -> None:
    _start_countdown()


def _on_all_work_done() -> None:
    _stop_countdown()


def _on_action_settled(action_id: str, resource=None) -> None:
    """mcp_client 的 pending 有了结局（完成 / 超时 / 取消）。

    只关心面向用户的那些 —— 走路走完了不是"说完话了"。按**物理通道**比，不按工具名猜
    （机器人的嘴叫 tts / speaker / audio_play 各有各的叫法）。
    """
    import hooks
    if resource is None:
        resource = mcp_client._pending_resources.get(action_id)
    if resource and hooks.notify_resources() & resource:
        _on_speaking_finished()


mcp_client.on_action_settled(_on_action_settled)


def _turn_running() -> bool:
    import collector
    return bool(getattr(collector, '_busy', False))


def _active_work_summary() -> list[str]:
    """当前还有什么活在干。空列表 = 闲着，没什么可汇报的。

    给 gate 判"有没有活"，也当上下文的兜底 —— 所以带上目标：拿不到 _active_work_detail()
    时（子代理刚起、manager 形状不对），这行是汇报器仅有的素材，连目标都没有就只能说
    "还在处理中"。
    """
    lines = []
    try:
        import subagent
        for s in subagent._get_active_subagents():
            if s.status in ('running', 'pending'):
                lines.append(f'子代理 [{s.id}] {s.status}，已跑 {s.rounds_completed} 轮：'
                             f'{(s.goal or "")[:120]}')
    except Exception:
        pass
    return lines


def _active_work_detail() -> tuple:
    """还在跑的活的**近况**，喂给汇报器。

    只喂目标 + 轮数的话，汇报器手里除了目标本身什么都没有，只能把目标换个说法念一遍。
    Orin5 实测播出来的是"投研报告还在调研中，完成后告诉你结果"——用户听完仍然不知道
    进行到哪了。要说清"做了什么 / 发现了什么 / 接下来做什么"，就得让它看到子代理具体
    搜了什么、拿到了什么。
    """
    try:
        import subagent
        digests = subagent._get_active_digests(since=_reported_rounds)
    except Exception:
        digests = []
    if not digests:
        return '', 'none'
    blocks = []
    for d in digests:
        head = f"子代理 [{d['id']}] 已跑 {d['rounds']} 轮，目标：{(d['goal'] or '')[:120]}"
        body = _turns_to_text(d['turns']) if d['turns'] else ''
        label = '它最近做的事（这段时间没有新动作，仍在同一步上）：' if d.get('stalled') \
            else '它最近做的事：'
        blocks.append(head + ('\n' + label + '\n' + body if body else ''))
    text = '\n\n'.join(blocks)

    # 只分两种，调用方据此决定"这次说不说"：
    #   new   有新动作 —— 照播，讲增量
    #   idle  一个 turn 都没新增 —— 也要播（刚派出去、或卡在一个长单步上，用户都该知道
    #         还在进行），但**只播一次**，否则就回到每 15 秒念一遍同样的话。
    #
    # 不再细分"刚开始"和"卡住了"：两者给模型的约束是同一条（别编造），而它们的区别
    # 上下文本身已经写明 —— 刚开始的没有活动记录，卡住的带着"仍在同一步上"的标注。
    return text, ('idle' if any(d.get('stalled') for d in digests) else 'new')



def _build_narration_messages(*, frozen_system: dict, context: str,
                              last_report_text: str, budget_chars: int) -> list[dict]:
    """构造汇报调用的消息。

    system 段直接复用主循环那一份 frozen_system：汇报因此继承机器人的人设和语言规则
    （一个光秃秃的"你是进度播报器"会说出一个跟用户聊了半天的那个实体不像的语气），
    而且逐字节相同意味着 provider 前缀缓存几乎免费。

    上下文**取尾不取头** —— 与 _compress_turns 的 text[:30000] 相反，是刻意的：进度
    汇报关心的是近况，历史摘要才必须保住开头。
    """
    if len(context) > budget_chars:
        context = '...(前略)\n' + context[-budget_chars:]
    last = (f'- 不要重复你上次已经播报过的：「{last_report_text}」' if last_report_text else '')
    return [
        frozen_system,
        {'role': 'user', 'content': _NARRATION_PROMPT.format(
            last=last, context=context, waiting=_waiting_facts())},
    ]


def _waiting_facts() -> str:
    """这一刻等了多久、在等什么 —— 由框架算好交给模型。

    模型没有别的办法拿到这个数：播报那次调用里既没有当前时刻，也没有 pending 列表，
    过程记录里只有几条历史时间戳。让它自己减，就减出了「大约等了 25 秒」。
    """
    lines = []
    if _silence_since is not None:
        lines.append(f'- 距你上次开口：{int(time.time() - _silence_since)} 秒')
    try:
        import mcp_client
        for wait in mcp_client.pending_waits()[:3]:
            tool = wait['tool'] or '某个动作'
            lines.append(f'- 正在等 {tool} 完成：已等 {wait["seconds"]} 秒')
    except Exception:
        pass
    if not lines:
        return '（这一刻没有在等任何东西，也就没有"等了多久"可说）'
    return '\n'.join(lines)


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
            if not isinstance(content, str):
                # `content[:800]` 上一个 dict 会抛 `unhashable type: 'slice'`，而这个
                # 函数是从 `_bg_trigger_loop` 里调的 —— 那个 while True 没有任何
                # try/except，异常直接把 task 打死，后台传感器监控就此静默停摆，
                # 日志里只留一行 "Task exception was never retrieved"。天轶上实测
                # 一次这样死了 40 分钟，直到容器重启才恢复。
                content = _tool_content(content)
                if not isinstance(content, str):
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
            ('set_auto_narration', event.skills.set_auto_narration),
            ('set_progress_report', event.skills.set_progress_report),
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
            # 落盘的历史可能是修复之前写的：那时 tool 消息的 content 会是 dict，
            # 读回来就等于把同一个 400 一起续跑回来。所以在 restore 这一侧也过一遍。
            self._turns = [_sanitise_turn(t) for t in last['turns']]
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
            # present_to_llm 而不是直接用生料：它负责滤掉 processor 的系统 action
            # （start/stop/config/info —— 插件生命周期归画布管，不归 LLM 管）并注入
            # concurrent 参数。这两件事原本只写在 mcp_client.all_schemas() 里，而
            # all_schemas() 全仓库只有 peer 那一处调用，主链路走的是这里 —— 于是
            # Orin5 上用户说"别说了"时模型调了 tts(action="stop")（把话题订阅节点整个
            # 拆了，本该是 interrupt），而 system prompt 教了一整段的 concurrent 参数
            # 在任何工具上都不存在。
            _meta = info.get('tool_meta', {})
            _split = info.get('split_map', {})
            schema = info.get('schemas', {}).get(full_name)
            if schema:
                presented = mcp_client.present_to_llm(schema, _meta.get(full_name))
                if presented is not None:
                    schemas.append(presented)
            else:
                # 检查是否有拆分的子工具（x-action-params 拆分）
                for split_name in info.get('tool_groups', {}).get(tool_name, []):
                    s = info.get('schemas', {}).get(split_name)
                    if not s:
                        continue
                    presented = mcp_client.present_to_llm(
                        s, _meta.get(split_name),
                        split_action=_split.get(split_name, {}).get('action'))
                    if presented is None:
                        continue
                    schemas.append(presented)
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

    async def _interrupt_active_outputs(self, reason: str = ''):
        """中止所有正在进行的输出（TTS + 动作）。在 TurnCancelled / 新用户 turn 时调用。

        **hook 与硬编码兜底都跑，不是二选一。** 这两条路覆盖的是**不相交**的两组
        卡片：hook 覆盖「自己声明了 `on_interrupt_all` 绑定」的卡，兜底覆盖「叫
        `tts`/`loco` 但没声明绑定」的卡。当成二选一，另一组就永远碰不到。

        以前是 `if results: return`，而 `hooks.fire` 对**每一个执行过的绑定**都追加
        一条结果，**包括抛异常的和什么都没做的**。于是「存在绑定」被当成了「打断已
        处理」，一张卡的绑定替全机队关掉了兜底。

        Orin6 实测（2026-09-19）：actucore 的 `vla` 卡绑着 `on_interrupt_all`，卡片
        没在跑时返回 `{"state":"idle","message":"卡片未在运行"}` —— 什么都没做，却
        让 `results` 非空。它是那台机器上**唯一**的绑定，而 actucore 跑在每一台机器人
        上。直接探未修复的这个函数，它打印
        「interrupted via on_interrupt_all hook (1 binding(s))」，而实际只叫了 vla，
        `tts` 和 `loco` 一次都没被调用。

        受影响的是**运动**。语音未必：ASR 的 barge-in 另有一条 `on_interrupt_speak`
        会停住 TTS（Orin6 上实测确实停了），所以按打断来源不同，语音可能侥幸得救。
        而底盘/导航只有这一条路，兜底被跳过就真的不停。

        去重按 `(mcp_id, tool)`：hook 已经成功叫停过的那张卡不再叫第二次。叫停本身
        是幂等的，所以重复调用无害，但日志会变得难读。
        """
        import hooks
        from peer import mcp_bridge

        try:
            hook_results = await hooks.fire('on_interrupt_all')
        except Exception as exc:
            # 打断路径是最不该抛异常的地方：它跑在 TurnCancelled 处理里，抛出去就
            # 连兜底一起丢了 —— 而兜底恰恰是这时候唯一还能停住机器人的东西。
            print(f'[decision] on_interrupt_all hook raised: {exc}')
            hook_results = []
        # 只有**成功**的绑定才算覆盖到：抛异常的那张卡并没有被叫停，若它恰好也叫
        # tts/loco，兜底该再试一次。非 dict 的条目当成「不知道覆盖了谁」，宁可让
        # 兜底多叫一次（叫停是幂等的），也不要漏。
        covered = {(r.get('mcp_id'), r.get('tool')) for r in hook_results
                   if isinstance(r, dict) and 'error' not in r}
        hook_failures = [r for r in hook_results if isinstance(r, dict) and 'error' in r]
        tasks = []

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
        for mcp_id, info in mcp_client.registry.items():
            if not info.get('online'):
                continue
            # Skip peers. peer/mcp_bridge.py registers a synthetic entry per paired
            # peer so its tools travel the normal mcp__ path, which means this loop
            # over the whole registry was reaching across the network to silence
            # *another robot's* mouth on every local barge-in. Observed on
            # Orin5+Orin6: both machines logged
            # `interrupt_active_outputs: peer:<id>:tts` at each other, and it only
            # failed because a peer entry carries no url. Someone else's speech is
            # not our output to abort — and a robot cutting off its partner
            # mid-sentence is exactly the behaviour we are trying to fix here.
            if mcp_bridge.is_peer_mcp(mcp_id):
                continue
            tools = info.get('tools', [])
            for short_name, action in (('tts', 'interrupt'), ('loco', 'stop_move')):
                if short_name in tools and (mcp_id, short_name) not in covered:
                    tasks.append((f'{mcp_id}:{short_name}',
                                  mcp_client.call_tool_direct(mcp_id, short_name,
                                                              {'action': action})))

        ok = 0
        if tasks:
            labels = [label for label, _ in tasks]
            results = await asyncio.gather(*[coro for _, coro in tasks],
                                           return_exceptions=True)
            for label, r in zip(labels, results):
                if isinstance(r, Exception):
                    print(f'[decision] interrupt_active_outputs: {label} raised: {r}')
                elif isinstance(r, dict) and r.get('error'):
                    print(f'[decision] interrupt_active_outputs: {label} failed: {r["error"]}')
                else:
                    ok += 1

        # Unconditionally, on both paths. Previously this ran only when a hook was
        # bound, so a robot with no binding kept its pending ACP actions blocked —
        # the barrier never released — and the silence countdown kept running
        # through a barge-in.
        for aid in list(mcp_client._pending_actions.keys()):
            if reason:
                mcp_client._pending_results[aid] = {'status': 'cancelled', 'reason': reason}
            mcp_client._pending_actions[aid].set()
        _stop_countdown()   # 打断 = 用户在说话，不是沉默的起点

        if not hook_results and not tasks:
            print('[decision] interrupt_active_outputs: nothing to interrupt '
                  '(no on_interrupt_all binding, no tts/loco tool)')
            return
        print(f'[decision] interrupted: {len(covered)} via hook, {ok}/{len(tasks)} via fallback'
              + (f', {len(hook_failures)} hook binding(s) failed' if hook_failures else ''))


    # ── 主循环 ───────────────────────────────────────────────────────────────

    async def run_forever(self):
        """事件驱动：通过 collector 批量获取事件，每批跑一轮推理。"""
        while True:
            ev = await collector.next_trigger()
            self._current_turn = []  # 本轮消息，无论成功失败都会保存
            # 注册取消信号（用户消息可通过此信号中断 sensor turn）
            cancel_ev = asyncio.Event()
            collector.set_cancel_event(cancel_ev)
            # 更窄的信号：高优先级 steering 到达时，打断"正在飞的 LLM 请求"或者
            # "还没发出去、卡在排队里的 tool_call"，但不结束这个 turn（跟 cancel_ev
            # 的区别见 event/llm.py 的 RoundReconsider 和 mcp_client.await_pending）。
            reconsider_ev = asyncio.Event()
            collector.set_reconsider_event(reconsider_ev)
            collector.set_turn_priority(1 if ev.get('_urgent') else 0)
            collector.set_busy(True)
            # Publish this turn's cancel signal so tools that block for a long time
            # can honour it. peer_delegate holds an HTTP connection for the remote
            # task's whole duration, and the checks below only run between rounds.
            from peer.delegation import current_cancel_event as _cancel_ctx
            _cancel_token = _cancel_ctx.set(cancel_ev)
            try:
                await self._one_turn(ev, cancel_event=cancel_ev, reconsider_event=reconsider_ev)
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
                _cancel_ctx.reset(_cancel_token)
                collector.set_cancel_event(None)
                collector.set_reconsider_event(None)
                collector.set_busy(False)
                # Fire on_idle hook (LED state reset etc.)
                if not ev.get('_bot_channel_event'):
                    import hooks as _hooks_idle
                    asyncio.create_task(_hooks_idle.fire('on_idle'))
                # Bot 输入不进入共享历史，避免在后续人工 turn 中延迟执行。
                if self._current_turn and not ev.get('_bot_channel_event'):
                    self._save_current_turn(ev)

    def _flush_current_turn(self, trigger_event: dict | None = None):
        """把正在跑的 turn 写进 SQLite（每轮一次），不动内存历史。

        以前只有 turn 结束时写一次，一个跑了几分钟的 turn 在这期间对历史 modal 完全
        不存在 —— 手动刷新也刷不出来，因为要刷的行还没有。这里按当前 turn 索引 upsert，
        turn 结束时 `_save_current_turn` 再写同一个索引覆盖掉（那一份已 compact）。
        """
        if not self._current_turn:
            return
        import chat_history
        try:
            if not self._session_id:
                self._session_id = chat_history.create_session(chat_history.KIND_MAIN)
            # 存盘的是截断过的副本 —— 和 turn 结束时那一份同一个规则，也避免把整轮
            # 未截断的 tool result 每轮重写一遍。原 turn 不能动，LLM 还要用完整内容。
            chat_history.save_turn(
                self._session_id, len(self._turns), _compact_turn(self._current_turn))
            if trigger_event:
                summary_text = trigger_event.get('text', '') or trigger_event.get('source', '')
                if summary_text:
                    chat_history.update_summary(self._session_id, summary_text)
        except Exception as e:
            print(f'[chat_history] flush_turn failed: {e}')

    def _save_current_turn(self, trigger_event: dict):
        """保存 _current_turn 到内存历史 + SQLite。"""
        # 保存前 compact：截断大 tool results，减少 tier1 历史占用
        turn = _compact_turn(self._current_turn)
        self._current_turn[:] = turn
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

    async def _report_progress(self) -> None:
        """生成一句进度汇报并播出去。**唯一的触发点是沉默计时到点。**

        跑的是一次**跳出 agent loop、也跳出 subagent 体系**的一次性 LLM 调用（形状同
        _compress_turns / subagent 的 _wrap_up）：无工具、主模型、短上下文，产出一句话。

        上下文按"当前有没有 turn 在跑"二选一 —— 有 turn 就用它活着的消息列表，没有就
        用最近几轮历史加上还在跑的子代理状态。两种情况下播报通道、gate、退避完全一样。

        任何"没播成"的分支都必须 _restart_countdown() 退满一个间隔。漏掉任何一条，
        播报就**永久静音** —— 没有别的东西会来重新计时，而日志里什么都看不到。
        """
        global _narration_inflight
        import hooks
        if _narration_inflight:
            # 另一次汇报正在飞（LLM 调用最长 20 秒，期间完成回调可能又上了一次表）。
            # **必须重新计时**：这个 task 到此就结束了，不重排的话没有任何东西会再触发
            # 汇报，等于永久静音到下一个轮边界。
            _narration_gate('another report in flight', stop=False)
            return
        # gate。顺序按从便宜到贵排。
        _, seconds_thr = _narration_thresholds()
        if seconds_thr <= 0:
            _narration_gate('disabled (seconds=0)', stop=True)
            return
        work = _active_work_summary()
        turn_alive = _turn_running()
        if not work and not turn_alive:
            _narration_gate('no active work', stop=True)   # 活干完了，没什么好报的
            return
        if _last_turn_restricted and not turn_alive:
            # 受限 turn 派出去的活，不该由我们代为出声
            _narration_gate('last turn was tool-restricted', stop=True)
            return
        if not _narration_enabled():
            _narration_gate('auto_narration off', stop=True)
            return
        if not hooks.has_bindings('on_notify'):
            global _warned_no_notify
            if not _warned_no_notify:
                # 否则"没有任何播报设备"和"阈值还没到"在日志里长得一模一样。
                _warned_no_notify = True
                print('[decision] narration disabled: no on_notify binding registered')
            _narration_gate('no on_notify binding', stop=True)
            return
        if _mouth_busy():
            # 正在说话，这一轮不该到点；重新计
            _narration_gate('mouth busy', stop=False)
            return

        # 上下文
        global _last_progress_ts, _reported_turn_msgs
        phase = 'new'
        live = self._current_turn if turn_alive else None
        if live:
            # 只讲**上次汇报之后新增的那几条**，理由同子代理那边：喂整个 turn 的话模型
            # 会从里面自己挑，两次之间跳掉中间过程，听着像两件不相干的事。
            #
            # turn_messages 会被 compaction / truncate 改短，那时水位大于长度、切片为空
            # —— 退化成"这次没有新动作"，和子代理那边同样的安全降级。
            fresh = list(live)[_reported_turn_msgs:]
            if fresh:
                phase = 'new'
                context = _turns_to_text([fresh])
            else:
                phase = 'idle'
                context = _turns_to_text([list(live)[-3:]])
            if phase == 'new' or _last_progress_ts is None:
                _last_progress_ts = time.time()
        else:
            # **只给当前这件活的材料，不给主 agent 的历史。**
            #
            # 这条路汇报的对象就是还在跑的子代理，而 self._turns 里装的是主 agent 之前
            # 聊过的（往往是已经做完的上一个任务）—— 那是另一个话题，放进来纯属噪声，
            # 而且实测会把汇报带跑偏：Orin5 上 11:24:13 刚派出"调研比亚迪海豹"的子代理，
            # 11:24:31 播出来的却是"理想L6和问界M7的配置对比数据都查到了"，说的是上一个
            # 已经结束的任务。
            #
            # 子代理刚起步、digest 还空时，上下文就只剩目标 —— 那时按 prompt 的要求应当
            # 输出 SKIP（没有具体进展就别说），这比报一个陈旧话题好。
            detail, phase = _active_work_detail()
            if not detail:
                _narration_gate('no work detail', stop=False)
                return
            if phase == 'new' or _last_progress_ts is None:
                _last_progress_ts = time.time()
            context = '当前还在进行的工作：\n' + detail

        llm_cfg = config.main.get('event', {}).get('llm', {})
        _narration_inflight = True
        try:
            if phase == 'new':
                _extra = ''
            else:
                _waited = _human_duration(time.time() - (_last_progress_ts or time.time()))
                _extra = ('\n注意：这段时间它没有产出新东西 —— 可能刚着手，也可能卡在同一步上。'
                          f'距离上一次有新动作已经过去 {_waited}。用一句话让用户知道'
                          '**还在进行、正在做哪一步、已经等了多久**，'
                          '把时长说出来（用户想知道的是有没有卡死），'
                          '不要编造任何进展或数据。')
            messages = _build_narration_messages(
                frozen_system=prompt_mod.build_system(
                    mcp_client.registry, self._bound_tool_names()),
                context=context + _extra, last_report_text=_last_report_text_global,
                budget_chars=int(llm_cfg.get('narration_context_chars', 6000)))
            response = await asyncio.wait_for(
                # reconsider_event 不传 —— 汇报很短、重跑也是一样的代价，而 steer
                # 下一轮就会被排空。cancel_event 也不传：这里不在某个 turn 的调用栈上，
                # 抛 TurnCancelled 没有接的人。
                client.call(message_list=messages, tool_list=[],
                            caller_info={'agent_type': 'main_agent'}),
                timeout=float(llm_cfg.get('narration_timeout_s', 20)))
            report = (response.get('content') or '').strip()
        except asyncio.TimeoutError:
            print('[decision] narration: LLM call timed out, backing off')
            _restart_countdown()
            return
        except Exception as e:
            print(f'[decision] narration failed: {e}')
            _restart_countdown()
            return
        finally:
            _narration_inflight = False

        # SKIP 逃生口：传感器轮询那种确实没进展的时段，连说三遍"我还在查看"比沉默更糟。
        if not report or report.upper().startswith('SKIP'):
            _restart_countdown()
            return
        # 和上次说的一模一样就别再说一遍。prompt 里已经写了"不要重复上次播报过的"，但
        # Orin5 实测它照样逐字重复（两次相隔 25 秒，子代理近况没什么变化，它既没说新东西
        # 也没按要求 SKIP）。这种事不该指望模型自觉 —— 框架能判的就框架判。
        if _same_as_last_report(report):
            print(f'[decision] narration: same as last, skipped → "{report}"')
            _restart_countdown()
            return
        report = _trim_narration(report)

        results = await hooks.fire('on_notify', {'text': report}, barrier_aware=True)
        if not _notify_fire_spoke(results):
            # 竞态：gate 判定时嘴还空着，真要说的时候被占了。LLM 的钱已经花了，退避。
            print(f'[decision] narration: all bindings skipped (busy) → "{report}"')
            _restart_countdown()
            return

        global _last_narration_gate
        _last_narration_gate = ''
        _reported_turn_msgs = len(self._current_turn or [])
        _remember_report(report)
        # 推进水位：下次只讲这之后新发生的事。
        try:
            import subagent as _sa
            _live = {_d['id']: _d['rounds']
                     for _d in _sa._get_active_digests(max_turns=0)}
            _reported_rounds.update(_live)
            # 干完的子代理没必要一直留在水位表里
            for _gone in [k for k in _reported_rounds if k not in _live]:
                _reported_rounds.pop(_gone, None)
        except Exception:
            pass
        # 这次播报本身就是一次"开始说话"，走同一个回调：停止计时；若它没有 ACP 跟踪
        # （设备没声明 x-completion），当场重新计时，否则等它的"说完"事件。
        _on_speaking_started()
        # 告诉主 LLM 这句话已经替它说了，否则它下一轮很可能显式调播报工具再说一遍。
        # 排队而不是当场 append —— 见 _pending_narration_feedback 的注释。
        _pending_narration_feedback.append(report)
        print(f'[decision] narration: {int(seconds_thr)}s silent, '
              f'{len(work)} active, in_turn={bool(live)} → "{report}"')
        await push_event({'type': 'narration', 'payload': {
            'text': report, 'silent_seconds': int(seconds_thr),
            'in_turn': bool(live)}})

    def _bound_tool_names(self) -> set:
        return {s['name'] for s in self._get_bound_tool_schemas()}

    async def _one_turn(self, trigger_event: dict, cancel_event: asyncio.Event | None = None,
                        reconsider_event: asyncio.Event | None = None):
        import time as _time
        from uuid import uuid4
        _turn_t0 = _time.perf_counter()
        bot_restricted = _bot_channel_restricted(trigger_event)
        viewer_restricted = _viewer_channel_restricted(trigger_event)
        tool_restricted = bot_restricted or viewer_restricted
        bot_reply_source_ids = set(trigger_event.get('_bot_channel_message_ids', []))
        replied_message_ids: set[str] = set()

        global _reported_turn_msgs
        _reported_turn_msgs = 0    # 新 turn，水位清零，否则上个 turn 的条数会吃掉开头

        # 任务开始 → 开始计时。用 _start_countdown（"已在计时则不动"）而不是重新计：
        # 上一件事已经沉默了 20 秒的话，新任务不该把它再往后推一个完整间隔。
        _on_task_started()
        global _last_turn_restricted
        _last_turn_restricted = tool_restricted

        # ── 主动播报的 turn 内状态 ──────────────────────────────────────────
        # 计时、上次汇报内容、定时器全是模块级的 —— 活会跨过 turn 边界继续跑，沉默也就
        # 跨过 turn 边界继续算。这里只剩哑火护栏用的两个标志。
        _turn_interacted = False              # 整个 turn 有没有出过一次声（哑火护栏用）
        _no_output_retried = False            # 哑火护栏只触发一次，防死循环

        # Reset Python sandbox namespace for this turn
        self._desktop_tools.reset_python_namespace()

        # 性能追踪（开放 span 式）
        _trace_id = str(uuid4())
        _turn_start_ts = time.time()
        _spans = []  # 收集所有 span
        _spans_committed = 0  # 已落盘的 span 数（turn 跑到一半也提交，见 _flush_spans）
        _tool_names_collected = []

        def _flush_spans():
            """把本轮新产生的 span 落盘，让性能面板能看到正在跑的 turn。

            以前只有 turn 结束时提交一次，所以面板上最新的一条永远是上一个 turn ——
            正在跑的那个（也就是唯一有人想看的那个）还不存在。
            """
            nonlocal _spans_committed
            new = _spans[_spans_committed:]
            if not new:
                return
            try:
                perf_log.commit_spans(
                    trace_id=_trace_id,
                    spans=new,
                    source=trigger_event.get('source', ''),
                    trigger_text=trigger_event.get('text', '')[:300],
                )
                _spans_committed = len(_spans)
            except Exception as _pe:
                print(f'[perf_log] incremental commit error: {_pe}')

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
        # 自己触发的 turn（subagent 完成 / ACP 回调 / 定时器）不打断：那会掐掉
        # 刚刚由自己排上的音频。见 _trigger_is_self_originated。
        if (mcp_client.get_pending_actions() and not bot_restricted
                and not _trigger_is_self_originated(trigger_event)):
            # 一处调用，两条路都走。这里原本自己 fire 一次 hook，再在 else 分支调
            # _interrupt_active_outputs —— 那个函数里又 fire 一次，所以没有绑定时
            # hook 被触发两遍；而有绑定时兜底被整个跳过。两个问题同源：把 hook 和
            # 兜底当成了二选一，它们覆盖的其实是不相交的两组卡片。
            await self._interrupt_active_outputs(
                reason='auto-interrupted by new user message')

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
            'payload': {'text': _activity_text(trigger_event.get('text', ''))},
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

        async def _no_output_guard(text: str) -> bool:
            """turn 要结束了，但它从头到尾一次都没出过声 —— 再给一轮。

            废除 content 自动播报的配套护栏。以前"只写 content 就 finish"还能被自动播报
            救回来，现在那条路没了，这种 turn 会彻底没声，而它恰恰是最常见的短问答形态。
            形状照搬 _channel_tool_retry_message 那条一次性自纠：**只触发一次**，否则模型
            顽固不调工具时会死循环。

            返回 True 表示"别 break，接着走"。
            """
            nonlocal _no_output_retried
            if _no_output_retried or _turn_interacted or not text:
                return False
            # 受限 turn 本来就不该播任何东西；没有播报通道时催也没用。
            if tool_restricted or not _narration_enabled():
                return False
            import hooks as _h
            if not _h.has_bindings('on_notify'):
                return False
            _no_output_retried = True
            turn_messages.append({'role': 'user', 'content': _NO_OUTPUT_RETRY_MESSAGE})
            print('[decision] turn produced no user-facing output; retrying once')
            await push_event({'type': 'llm_retry', 'payload': {'reason': 'no_user_output'}})
            return True

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
            if reconsider_event is not None and reconsider_event.is_set():
                # 已经有高优先级 steering 在等——不用真的发一次请求再被打断，
                # 直接原地重新构建（走下面 except RoundReconsider 同一条处理）。
                reconsider_event.clear()
                _steered_pre = await collector.drain_steering()
                if _steered_pre:
                    for sev in _steered_pre:
                        turn_messages.append({'role': 'user', 'content':
                            f'[system notification source={sev.get("source", "")}]\n{sev.get("text", "")}'})
                    print(f'[decision] reconsider pending before request, steered {len(_steered_pre)} message(s)')
                continue

            _round_t0 = _time.perf_counter()
            _round_start_ts = time.time()
            try:
                response = await client.call(
                    message_list = messages,
                    tool_list    = all_tool_list,
                    cancel_event = cancel_event,
                    reconsider_event = reconsider_event,
                    trace_id     = _trace_id,
                    caller_info  = {'agent_type': 'main_agent'},
                )
            except TurnCancelled:
                raise
            except RoundReconsider:
                # 高优先级 steering 在这次请求还没返回时就到了：这次请求整个作废，
                # 不记进 turn_messages（本来就还没 append），不结束 turn——把新消息
                # drain 进去，原地重新发起请求。跟 finish 的 barge_in 是两回事：那
                # 条会中止播放、结束整轮；这里只是换一次更知情的推理，turn 接着走。
                reconsider_event.clear()
                _steered_inflight = await collector.drain_steering()
                if _steered_inflight:
                    for sev in _steered_inflight:
                        turn_messages.append({'role': 'user', 'content':
                            f'[system notification source={sev.get("source", "")}]\n{sev.get("text", "")}'})
                    print(f'[decision] llm call reconsidered, steered {len(_steered_inflight)} message(s) into retry')
                round_idx += 1
                total_rounds += 1
                continue
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
            self._flush_current_turn(trigger_event)
            _flush_spans()

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
            # content 不再自动播报。它曾经被 hooks.fire('on_notify') 直接念出去，两个
            # 实测问题：模型经常整段沉默（prompt 约束不住），而它真写了的时候那是内部
            # 推理，不是说给等着的人听的进度汇报。现在只有两条出声路径 —— 模型显式调
            # 播报工具，或框架在连续沉默后生成的进度汇报（_report_progress）。
            # agent_thought 照发：仪表盘仍要看到模型在想什么，只是不再念出来。
            text = response.get('content') or ''
            if text:
                await push_event({'type': 'agent_thought', 'payload': {'text': text}})

            # 本轮是否发生了面向用户的输出。_round_already_notified 原本用于去重（模型
            # 自己说过就别再自动播一遍），现在改为交互判定 —— 语义是同一个：这轮有没有
            # 东西经由注册了 on_notify 的工具送到用户那里。
            tool_calls = response.get('tool_calls') or []
            _notified_by_tool = _round_already_notified(tool_calls)

            # ── 用量广播 ──────────────────────────────────────────────────
            _usage = response.get('_usage')
            if _usage:
                _turn_usage['prompt_tokens'] += _usage.get('prompt_tokens') or 0
                _turn_usage['completion_tokens'] += _usage.get('completion_tokens') or 0
                _turn_usage['total_tokens'] += _usage.get('total_tokens') or 0
                _turn_usage['cached_tokens'] += _usage.get('cached_tokens') or 0
                await push_event({'type': 'llm_usage', 'payload': _usage})

            # ── 工具调用 ──────────────────────────────────────────────────

            async def _dispatch(call: dict) -> dict:
                nonlocal _finish_deferred, _channel_replied
                name   = call['function']['name']
                args   = json.loads(call['function']['arguments'] or '{}')
                # Set by either barrier branch below, when a barrier wait was cut short
                # by reconsider_event rather than genuinely completing. Lets the caller
                # tell "interrupted before it ever reached the device, still had made
                # no progress" apart from every other outcome.
                _reconsidered = False

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
                    # 和下面的 mcp__ 分支同形：等待被 reconsider_event 截断时**不派发**
                    # 这次调用，由调用方决定作废整轮还是记成"没派出去"。
                    #
                    # 这条以前不传 reconsider_event 也不看返回值，于是播报期间的新消息
                    # 谁都看不到：finish 的 break 在 steering drain 前面，turn 里没有任何
                    # 东西会消费队列。Orin5 实测，一条消息在队列里躺了 33.6 秒 —— 正好是
                    # 播报的剩余时长（事件自带 ts=17:32:19，received 在 17:32:52.597）。
                    _barrier_result = None
                    if _sys_tool_needs_barrier(name):
                        _barrier_result = await _acp_barrier(
                            name, cancel_event, reconsider_event=reconsider_event)
                    if (_barrier_result is not None
                            and _barrier_result.get('status') not in ('completed', 'no_pending')):
                        _reconsidered = (_barrier_result.get('status') == 'reconsidering')
                        if name == finish_tool and _reconsidered:
                            # 关键：barrier 没放行不代表 turn 该结束。把这次 finish 作废，
                            # 音频继续播（reconsidering 刻意不 _forget_pending），循环回到
                            # steering drain 带着新消息再想一遍 —— 也就是"播边想"。
                            _finish_deferred = True
                            result = {
                                "status": "not_dispatched",
                                "reason": "新消息到达，本次 finish 已取消，turn 继续。"
                                          "当前语音仍在正常播放、不会被打断，你新说的话会排在它后面。"
                                          "请结合新消息继续处理，不要重复已经说过的内容。",
                            }
                        else:
                            result = {
                                "status": "not_dispatched",
                                "reason": f"barrier wait interrupted before dispatch "
                                          f"(status={_barrier_result.get('status')})",
                            }
                    else:
                        result = await self._sys_tools[name]['object'](**args)
                elif name.startswith('mcp__'):
                    # ACP barrier: 只等与本次调用资源冲突的 pending（见 _needs_barrier）
                    # Pop `concurrent` before _needs_barrier looks at args and before
                    # the call leaves for the device — it is harness-injected and the
                    # driver's schema does not know it.
                    _parallel = mcp_client.take_parallel_flag(args)
                    _bar_needed, _bar_want = _needs_barrier(name, args)
                    _barrier_result = None
                    if _bar_needed:
                        _barrier_result = await _acp_barrier(name, cancel_event, want=_bar_want,
                                           scoped=True, concurrent=_parallel,
                                           reconsider_event=reconsider_event)
                    if _barrier_result is not None and _barrier_result.get('status') not in ('completed', 'no_pending'):
                        # Barrier wait ended without ever reaching the device — do NOT
                        # dispatch it anyway (this used to be silently ignored: a
                        # cancelled/reconsidering wait still fell through to
                        # call_tool below). Whatever it was waiting on is untouched,
                        # still legitimately in flight; the caller decides whether to
                        # void this whole round (reconsidering, nothing dispatched
                        # yet) or just record it as not-dispatched.
                        result = {"status": "not_dispatched",
                                  "reason": f"barrier wait interrupted before dispatch (status={_barrier_result.get('status')})"}
                        _reconsidered = (_barrier_result.get('status') == 'reconsidering')
                    else:
                        args['_trace_id'] = _trace_id
                        args['_cancel_event'] = cancel_event
                        # Inject instance_id from canvas binding (multiInstance tools need it)
                        if name in self._bound_instance_ids and 'instance_id' not in args:
                            args['instance_id'] = self._bound_instance_ids[name]
                        result = await mcp_client.call_tool(name, args)
                        # interrupt hook 绑定的工具执行后：清 pending + 通知其他绑定方
                        if not _bar_needed and mcp_client.get_pending_actions():
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
                                _stop_countdown()   # 打断 = 用户在说话，不是沉默的起点
                                # Fire hook to notify ALL registered parties (e.g. perception TTS)
                                _hook_id = _hooks.get_hook_for_binding(_mcp_id, _tool, _act)
                                if _hook_id:
                                    asyncio.create_task(_hooks.fire(_hook_id, exclude_mcp_id=_mcp_id))
                                print(f'[acp] interrupt: cancelled pending + fired {_hook_id} (source: {_tool}.{_act})')
                else:
                    result = f'未知工具: {name}'

                if (name == 'mcp__channel__channel_reply'
                        and isinstance(result, str) and not result.startswith('Error')):
                    # 面向用户的输出 —— 与 on_notify 绑定的工具等价，一起喂给沉默计数器。
                    # source_message_id 只管 replied_message_ids 那本账（哪条消息回过了），
                    # 主动发起的 channel 消息没有它，但用户一样收到了。
                    _channel_replied = True
                    if args.get('source_message_id'):
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

                return {'id': call['id'], 'result': result, '_reconsidered': _reconsidered}

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
            _round_voided = False
            # Reset per round, before any _dispatch runs. Set by the sys-tool branch
            # when finish's barrier was cut short by a new message: the finish never
            # executed, so the finish detection below must not end the turn.
            _finish_deferred = False
            # Same lifecycle: set by _dispatch when a channel_reply actually landed.
            _channel_replied = False
            for c in tool_calls:
                if _is_sensor(c['function']['name']):
                    _batch.append(c)
                else:
                    if _batch:
                        results.extend(await asyncio.gather(*[_dispatch(b) for b in _batch]))
                        _batch = []
                    r = await _dispatch(c)
                    if r.get('_reconsidered') and not results:
                        # Nothing in this round has actually reached the device yet
                        # (this is the first non-sensor call, and it never got past
                        # its own barrier wait) — safe to treat the whole round as
                        # if it never happened, same as reconsider firing while the
                        # LLM call itself was still in flight. If something earlier
                        # in this round *had* already dispatched, voiding here would
                        # erase the model's only memory of a real, already-running
                        # action — so that case falls through and keeps this result
                        # as a plain "not dispatched" entry instead (see below).
                        _round_voided = True
                        break
                    results.append(r)
            if _batch and not _round_voided:
                results.extend(await asyncio.gather(*[_dispatch(b) for b in _batch]))

            if _round_voided:
                # Discard this round entirely: same treatment as a reconsidered
                # in-flight LLM call (RoundReconsider below) — pop the response we
                # appended before dispatch, don't record any tool result, drain the
                # steering that triggered this, and retry with fresh context.
                turn_messages.pop()
                _steered_void = await collector.drain_steering()
                if _steered_void:
                    for sev in _steered_void:
                        turn_messages.append({'role': 'user', 'content':
                            f'[system notification source={sev.get("source", "")}]\n{sev.get("text", "")}'})
                    print(f'[decision] round voided by reconsider, steered {len(_steered_void)} message(s) into retry')
                if reconsider_event is not None:
                    reconsider_event.clear()
                round_idx += 1
                total_rounds += 1
                continue

            # ── 把工具结果加入本轮消息 ────────────────────────────────────
            if results:
                decisions.append({
                    'round': round_idx,
                    'text': text,
                    'tool_calls': [
                        {'name': c['function']['name'],
                         'args': _decoded_json(c['function'].get('arguments', '{}')),
                         'result': next((r['result'] for r in results if r['id'] == c['id']), None)}
                        for c in tool_calls
                    ],
                })
                turn_messages += [
                    {
                        'role':         'tool',
                        'tool_call_id': r['id'],
                        'content':      _tool_content(r['result']),
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
                elif await _no_output_guard(text):
                    pass          # 护栏已经把提示写进 turn_messages，继续下一轮
                else:
                    break

            # 工具结果已入列 —— 再落一次盘，这样历史 modal 里这一轮是完整的（调用 + 结果），
            # 而不是等整个 turn 结束才一起出现。
            self._flush_current_turn(trigger_event)
            _flush_spans()

            # 本轮的交互状态要在 finish 检测**之前**就记上。`speak(...) + finish()` 同轮
            # 是常见形状，而计数器那段在循环体末尾 —— 等到那里再更新，finish 分支看到的
            # _turn_interacted 还是 False，哑火护栏会凭空多插一轮。
            if _notified_by_tool or _channel_replied:
                _turn_interacted = True
                _on_speaking_started()

            # ── finish 检测 ───────────────────────────────────────────────
            #
            # 出现在 tool_calls 里不等于执行了。新消息在 finish 的 ACP barrier 期间到达时
            # 这次 finish 被作废（见 _dispatch 的 sys-tool 分支），turn 必须接着走 —— 否则
            # 就退回"播报多长、延迟多长"。
            #
            # 这里单独判而不是靠 _round_voided：后者只在本轮还没有任何工具派出去时才成立，
            # 而 `tts(...) + finish()` 同轮是常见形状（Orin5 日志里就有），那时 results 非空、
            # 不 void，会一路落到这个 break 上。
            if _turn_ends_on_finish(tool_calls, finish_tool, _finish_deferred):
                # 整个 turn 一次都没出过声就别急着结束 —— 见 _no_output_guard。
                if not await _no_output_guard(text):
                    break
                round_idx += 1
                total_rounds += 1
                continue
            if _finish_deferred:
                print('[decision] finish deferred: new message arrived during playback, '
                      'turn continues (audio keeps playing)')

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

            # 汇报的回灌在这里排空 —— 和 steering 同一个安全点。定时器可能在
            # assistant(tool_calls) 已入列、tool 结果还没入列的窗口触发，那时当场 append
            # 会造出 provider 会拒的消息序列。
            while _pending_narration_feedback:
                turn_messages.append(
                    _narration_feedback_message(_pending_narration_feedback.pop(0)))

            # 消息已经进 turn_messages 了，这次 reconsider 的目的达成 —— 不清的话下一次
            # client.call 会立刻抛 RoundReconsider 再 drain 一次空队列，白烧一次请求。
            # 放在 drain 之后而不是作废 finish 的当场，是为了尽量少丢"clear 与 drain 之间
            # 新到的消息"；真撞上也只是多一次 reconsider 重试，那条路本来就能吃下。
            # （_round_voided 那条路有自己的 clear，见上面。）
            if _finish_deferred and reconsider_event is not None:
                reconsider_event.clear()

            # ── 取消检查点：工具执行完毕后，下一轮 LLM 调用前 ────────────────
            if cancel_event and cancel_event.is_set():
                raise TurnCancelled("Interrupted after tool dispatch")

            # ── 沉默计时的不变式兜底 ─────────────────────────────────────
            #
            # 正常路径上"开始/重新/停止计时"都由事件回调驱动（输出派发、说完、打断、
            # 任务起止）。但打断改成只取消之后留了两个口子：TurnCancelled 时的
            # _interrupt_active_outputs（turn 没了但子代理可能还在跑），以及模型自己调
            # interrupt 类工具（turn 继续走）—— 都是"取消了却没有新任务来重新计时"。
            #
            # 与其去枚举每一条"谁负责重新计时"（本次数出 4 条绕过监听的路径，说明枚举
            # 法不可靠），这里放一条不变式恢复：只要循环还在转，计时就不会凭空消失。
            # _start_countdown 的"已在计时则不动"保证它不会把正在跑的 deadline 往后推。
            if not _mouth_busy():
                _start_countdown()

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

        # Turn 结束但活未必干完 —— 主 agent 派个异步子代理就 finish 是常见形状
        # （Orin5：17:22:31 spawn → 17:22:48 turn complete → 子代理又跑了 2 分半），
        # 那之后计时要继续，否则子代理跑多久用户就听不到多久。
        #
        # 活全干完时必须**显式取消**，不能只是"不重新计时"：turn 里那个定时器还在睡，
        # 它会一直活到下一个任务，然后对着刚起步几秒的新任务播一句进度。
        if not _active_work_summary():
            _on_all_work_done()

        # 性能追踪：提交 spans
        _turn_end_ts = time.time()
        _spans.append({'span': 'turn_total', 'component': 'core',
                       'start_ts': _turn_start_ts, 'end_ts': _turn_end_ts})
        # 只提交尚未落盘的部分 —— 轮内已经提交过的不能再写一遍。
        _flush_spans()
