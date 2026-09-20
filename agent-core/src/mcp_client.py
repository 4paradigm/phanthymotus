"""
mcp_client.py — MCP HTTP transport 客户端。

每个配置的 MCP（transport='http'）在启动时：
  1. initialize — 握手
  2. tools/list — 获取工具列表并注册到 tool_dict
  3. (可选) 订阅 SSE 通知流，把 notifications/message 推到 event_bus

工具调用：
  call_tool(mcp_id, tool_name, args) → 返回 MCP result 内容

注册表格式（module-level dict，供 prompt.py / event/llm.py 读取）：
    registry[mcp_id] = {
        'name':        str,
        'url':         str,
        'online':      bool,
        'tools':       [tool_name, ...],
        'render_hint': str,
        'schemas':     { tool_name: openai_function_schema },
    }
"""

import asyncio
from collections import OrderedDict
import collections
import contextvars
import json
import time
import uuid

import aiohttp
import jsonschema

import config
import event_bus

# ── 全局注册表 ─────────────────────────────────────────────────────────────────
registry: dict[str, dict] = {}   # mcp_id → info

# ── ACP: 异步动作完成协议 ──────────────────────────────────────────────────────
_pending_actions: dict[str, asyncio.Event] = {}   # action_id → Event (set on completion)
_pending_results: OrderedDict[str, dict] = OrderedDict()  # includes early SSE results
_pending_timeouts: dict[str, float] = {}          # action_id → dynamic timeout (seconds)
_pending_tools: dict[str, str] = {}               # action_id → tool_name (资源冲突检测用)
_MAX_EARLY_COMPLETIONS = 1024
_pending_resources: dict[str, frozenset | None] = {}  # action_id → 占用的物理通道
_pending_owner: dict[str, str] = {}               # action_id → 发起它的 agent 上下文

# 动作"有结局了"的订阅者。原本没有单一观察点：完成回调分散在 start.py 的
# /api/acp/complete 和下面 WS 分支两处，各自直接 .set() 那个 Event；而超时/取消走的是
# _forget_pending，跟前两者毫无关系。想知道"嘴什么时候空出来"只能去轮询。
#
# 沉默计时需要精确的"说完了"时刻（从**开始**播算的话，一段 134 秒的播报刚说完就会立刻
# 判超时），所以三条路统一在这里发通知：完成走 mark_action_complete，超时/取消走
# _forget_pending。订阅者自己再判一次资源是否真空了 —— 通知可能早于实际空闲，多个嘴
# 同时在播时第一个完成也不等于说完了。
_settle_listeners: list = []


def on_action_settled(fn) -> None:
    """注册"某个动作有结局了"的回调（完成 / 超时 / 取消都会触发）。

    回调签名 fn(action_id, resource)。异常会被吞掉并打日志 —— 一个订阅者出问题不该
    影响 ACP 本身的解锁。
    """
    _settle_listeners.append(fn)


def _notify_settled(action_id: str, resource=None) -> None:
    """`resource` 显式传入，因为 _forget_pending 是先拆表再通知的（见那里的注释），
    这时候已经查不到这个 action 占用过什么了。"""
    if resource is None:
        resource = _pending_resources.get(action_id)
    for fn in _settle_listeners:
        try:
            fn(action_id, resource)
        except Exception as e:
            print(f'[acp] settle listener failed: {e}')


def mark_action_complete(action_id: str, payload: dict) -> bool:
    """把一个 pending 标记为完成：记结果、解锁等待者、通知订阅者。

    两处完成入口（HTTP 回调 / WS action_complete 事件）都必须走这里，否则订阅者会漏掉
    其中一条路上的完成事件。注意**不删** _pending_actions 那一项 —— 晚到的 waiter 还要
    读 _pending_results，回收是 barrier 的事（见 resource_actually_busy）。
    """
    if action_id not in _pending_actions:
        _pending_results.pop(action_id, None)
        _pending_results[action_id] = payload
        early = [aid for aid in _pending_results if aid not in _pending_actions]
        for aid in early[:-_MAX_EARLY_COMPLETIONS]:
            _pending_results.pop(aid, None)
        return False
    _pending_results[action_id] = payload
    _pending_actions[action_id].set()
    _notify_settled(action_id)
    return True


# ── 次序：谁发起的动作 ────────────────────────────────────────────────────────
#
# Resource exclusion answers "may these two run at once"; it cannot answer "must
# this one finish first". Those are different questions and only the second one
# knows about intent.
#
# "先说'我要起来了'再起身" is an *ordering* requirement. mouth and leg are different
# channels, so exclusion permits the overlap — and before the barrier was scoped by
# resource, the global barrier forbade it by accident. Neither is a real answer: the
# same pair of tools must overlap when it is a gesture accompanying speech and must
# not when it is a warning preceding motion. The tools are identical; only the intent
# differs, and the intent lives in whoever emitted the calls.
#
# So ordering is enforced *within one agent's own sequence of calls* — the LLM emitted
# them in an order and that order is the script — and NOT across independent agents,
# which share no intent and whose unrelated actions must not block each other.
# `PARALLEL_PARAM` lets the emitter opt a single call out of its own ordering.
CONTEXT_MAIN = 'main'
current_agent_context: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    'acp_current_agent_context', default=CONTEXT_MAIN
)

# Parameter the harness injects into every acting tool's schema. Not declared by
# drivers: it is a property of *this call*, not of the hardware, so asking 14 drivers
# to carry it would be putting caller intent in the wrong layer. Stripped before the
# call leaves for the device.
#
# Default false, deliberately. A model does not reason about concurrency unless made
# to; left to itself it writes calls as if they were sequential, because that is how
# the text reads. So sequential is what it gets unless it says otherwise, and the
# unsafe direction — a warning overlapping the motion it warns about — is the one that
# requires an explicit request.
PARALLEL_PARAM = 'concurrent'

# ── ACP: 物理资源互斥 ─────────────────────────────────────────────────────────
#
# The barrier used to be global: any pending action blocked any acting tool. That
# conflates two unrelated things — "I need X's result before Y" (causality) and "X
# and Y both need the mouth" (exclusion) — and implements neither, arriving instead
# at "everyone waits for everyone". Speaking blocked navigating; one subagent
# speaking blocked every other subagent's every actuator call, on unrelated
# hardware. With subagents newly honouring the barrier at all, that would have
# collapsed N concurrent agents into an effective 1.
#
# What is genuinely mutually exclusive on a robot is a *physical channel* — one
# mouth, one chassis, one left arm — not "all actuators". Drivers declare theirs as
# `x-resource` next to `x-completion`; robotera/q5_bundle already splits base, arm,
# leg and waist into separate tools, which the global barrier serialised for no
# reason.
#
# Undeclared (`None`) means exclusive against everything. That is the conservative
# reading and it is deliberate: every driver that has not declared keeps exactly its
# old behaviour, so this can land without touching all fourteen of them at once.


def parse_resources(raw) -> frozenset | None:
    """Normalise a tool's `x-resource` into a set of channel names.

    Accepts a bare string (`"mouth"`) or a list (`["base", "arm_l"]`). Returns None
    for anything undeclared, empty or malformed — all of which mean "assume this
    conflicts with everything" rather than "conflicts with nothing", because a
    typo in a driver schema must not silently unlock parallel actuation.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        names = list(raw)
    else:
        return None
    clean = {n.strip() for n in names if isinstance(n, str) and n.strip()}
    return frozenset(clean) or None


def resources_conflict(want: frozenset | None, held: frozenset | None) -> bool:
    """Whether a call wanting `want` must wait for a pending action holding `held`."""
    if want is None or held is None:
        return True          # 任一侧未声明 → 保守当作互斥
    return bool(want & held)


def conflicting_pending(want: frozenset | None) -> list[str]:
    """Pending action_ids whose resources clash with `want`, in registration order."""
    return [
        aid for aid in _pending_actions
        if resources_conflict(want, _pending_resources.get(aid))
    ]


def resource_actually_busy(want: frozenset | None) -> bool:
    """这些资源**此刻**是不是真被占着 —— 用于「现在能不能说话」这类即时判断。

    与 `conflicting_pending` 的区别很关键：完成回调（/api/acp/complete 或 WS 的
    action_complete）只做 `_pending_actions[aid].set()`，**故意不删**这一项，好让晚到的
    waiter 还能读到结果；真正的删除发生在某个 barrier 等到它的时候。所以一个早就播完的
    action 会一直留在 `_pending_actions` 里，直到下一次 barrier 顺手回收它。

    对 barrier 本身没问题（它 await 一个已经 set 的 Event，瞬间返回并完成回收），但对
    「嘴现在空不空」就是错的。Orin5 上实测：一句播报 17:03:19 发出、17:03:24 完成回调就
    到了，可那一项直到 17:05:58 才被回收 —— 中间 2 分 34 秒里嘴明明是空的，却一直被判成
    忙，期间所有自动播报被静默跳过。

    因此这里要跳过 Event 已经 set 的那些：它们已经完成，只是还没被回收。
    """
    for aid in conflicting_pending(want):
        ev = _pending_actions.get(aid)
        if ev is not None and not ev.is_set():
            return True
    return False


def pendings_to_wait_for(want: frozenset | None, *, owner: str,
                         concurrent: bool) -> list[str]:
    """Everything a call must wait for, in registration order.

    Two independent reasons to wait, and they answer different questions:

    * **resource conflict** — the hardware cannot do both. Always enforced; a caller
      cannot opt out, because `concurrent=True` is a statement about intent, not a
      claim that one chassis can drive two ways at once.
    * **own ordering** — this agent already started something and has not asked for
      overlap, so the order it emitted its calls in is honoured. Skipped for actions
      started by a *different* context: independent agents share no script, and
      making them wait on each other is what collapsed N subagents into one.
    """
    out = []
    for aid in _pending_actions:
        if resources_conflict(want, _pending_resources.get(aid)):
            out.append(aid)
        elif not concurrent and _pending_owner.get(aid, CONTEXT_MAIN) == owner:
            out.append(aid)
    return out


def take_parallel_flag(args: dict) -> bool:
    """Pop `concurrent` out of a call's arguments and return it.

    Popped, not read: the parameter is injected by the harness and means nothing to
    the device, so forwarding it would show up as an unexpected field in a driver's
    schema validation.
    """
    if not isinstance(args, dict):
        return False
    return bool(args.pop(PARALLEL_PARAM, False))


def _forget_pending(aids, outcome: str | None = None) -> None:
    """Drop every per-action side table for `aids`, recording how each ended.

    One helper rather than the same four pops repeated at each exit: the pending
    bookkeeping is spread over five dicts now, and a cleanup path that forgets one
    of them leaks it for the lifetime of the process.

    `outcome` ('completed' | 'timeout' | 'cancelled' | 'barge_in') is remembered in
    `_action_outcomes` after the pending itself is gone. Without that the two
    outcomes are indistinguishable a moment later: the timeout path clears pending
    and lets the caller proceed exactly as success does, so an action whose
    completion callback never arrived looks, downstream, like one that played. That
    is what lets a delegation report a line as spoken when nothing was heard.
    """
    settled = []
    for aid in aids:
        if outcome:
            _record_outcome(aid, outcome)
        settled.append((aid, _pending_resources.get(aid)))
        _pending_actions.pop(aid, None)
        _pending_results.pop(aid, None)
        _pending_timeouts.pop(aid, None)
        _pending_tools.pop(aid, None)
        _pending_resources.pop(aid, None)
        _pending_owner.pop(aid, None)

    # 超时 / 取消也是"这个动作有结局了" —— 不在这里通知的话，一次超时的播报之后再没有
    # 任何事件会让沉默计时重新开始，播报就永久静音了。
    #
    # **必须在上面那些 pop 之后通知**：订阅者会去判"资源现在真空了吗"，而超时的 Event
    # 从来没有被 set 过，只要这一项还挂在 _pending_actions 里就仍然算占用 —— 在 pop
    # 之前通知，订阅者看到的是"还在说话"，于是什么都不做。
    for aid, res in settled:
        _notify_settled(aid, res)


# Terminal state of recently finished actions, so a caller can still ask "did that
# actually play?" after the pending is gone. Bounded — this is a diagnostic tail,
# not a ledger, and an agent that never asks must not grow it without limit.
_ACTION_OUTCOME_CAP = 512
_action_outcomes: "collections.OrderedDict[str, dict]" = collections.OrderedDict()


def _record_outcome(action_id: str, status: str) -> None:
    entry = {'status': status, 'tool': _pending_tools.get(action_id, '')}
    _action_outcomes.pop(action_id, None)
    _action_outcomes[action_id] = entry
    while len(_action_outcomes) > _ACTION_OUTCOME_CAP:
        _action_outcomes.popitem(last=False)


def action_outcome(action_id: str) -> dict | None:
    """Terminal state of a finished action, or None if still pending / evicted."""
    return _action_outcomes.get(action_id)


# ── 内部 JSON-RPC 助手 ─────────────────────────────────────────────────────────

# Key under which _jrpc reports a JSON-RPC `error` object. Callers that only look
# for their own keys (`content`, `tools`, ...) behave exactly as before — they see
# a dict without those keys, which is what `{}` gave them. Callers that care read
# this key.
JRPC_ERROR_KEY = '_jrpc_error'


async def _jrpc(session: aiohttp.ClientSession, url: str, method: str, params: dict, req_id: int = 1) -> dict:
    """Send one JSON-RPC request. On an `error` response, report it, don't drop it.

    This used to be `return data.get('result', {})`, which discarded the `error`
    object outright — so a driver that correctly answered
    `-32601 Unknown tool: move` came back as `{}`, and `call_tool` handed the
    model `"{}"`: indistinguishable from a successful call returning nothing.
    Observed on R1 with locomotion: the robot never moved, the model announced
    "好的，我要转身了", and when told it had not moved it retried the identical
    bad call, because nothing in the transcript said anything had failed.
    """
    payload = {'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params}
    async with session.post(url, json=payload) as resp:
        data = await resp.json(content_type=None)
    if isinstance(data, dict) and data.get('error') is not None and 'result' not in data:
        return {JRPC_ERROR_KEY: data['error']}
    return data.get('result', {})


def _to_openai_schema(mcp_id: str, tool: dict) -> list[dict]:
    """把 MCP tool 定义转成 OpenAI function calling schema。

    如果 inputSchema 包含 x-action-params，则拆分为每个 action 一个独立 schema。
    返回 list[dict]，无拆分时为单元素 list。
    """
    input_schema = tool.get('inputSchema') or {'type': 'object', 'properties': {}}
    action_params = input_schema.get('x-action-params')

    if not action_params:
        # 无拆分，保持原有行为
        name = f'mcp__{mcp_id}__{tool["name"]}'
        return [{
            'name':        name,
            'description': tool.get('description', ''),
            'parameters':  input_schema,
        }]

    # 按 action 拆分：每个 action 生成独立的 function schema
    all_props = input_schema.get('properties', {})
    all_required = set(input_schema.get('required', []))
    tool_desc = tool.get('description', '')
    schemas = []

    for action_name, action_def in action_params.items():
        param_keys = action_def.get('params', [])
        action_desc = action_def.get('description', action_name)

        # 只保留该 action 对应的参数（不含 action 字段本身）
        props = {k: all_props[k] for k in param_keys if k in all_props}
        required = [k for k in param_keys if k in all_required]

        schemas.append({
            'name':        f'mcp__{mcp_id}__{tool["name"]}__{action_name}',
            'description': f'{tool_desc} — {action_desc}',
            'parameters':  {
                'type': 'object',
                'properties': props,
                'required': required,
            },
        })

    return schemas


# ── 连接单个 MCP ───────────────────────────────────────────────────────────────

async def _connect_one(mcp_id: str, name: str, url: str, render_hint: str) -> None:
    timeout = aiohttp.ClientTimeout(total=8)
    schemas: dict[str, dict] = {}
    tools:   list[str]       = []
    tool_meta: dict[str, dict] = {}   # schema_name → {type, action_enum}
    split_map:  dict[str, dict] = {}  # split_schema_name → {tool, action}
    tool_groups: dict[str, list] = {} # original_tool_name → [split_schema_names]
    input_schemas: dict[str, dict] = {}  # schema_name → 原始 MCP inputSchema（用于参数校验）
    tool_definitions: list[dict] = []    # 原始定义（含 x-topic-actions 等扩展）

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # 1. initialize
            await _jrpc(session, url, 'initialize', {
                'protocolVersion': '2024-11-05',
                'capabilities':    {},
                'clientInfo':      {'name': 'phanthy-motus', 'version': '1.0'},
            })

            # 2. tools/list
            result = await _jrpc(session, url, 'tools/list', {})
            for tool in result.get('tools', []):
                tool_definitions.append(tool)
                tool_schemas = _to_openai_schema(mcp_id, tool)
                tools.append(tool['name'])

                if len(tool_schemas) == 1:
                    # 未拆分：保持原有行为
                    schema = tool_schemas[0]
                    schemas[schema['name']] = schema
                    raw_input_schema = tool.get('inputSchema') or {'type': 'object', 'properties': {}}
                    input_schemas[schema['name']] = raw_input_schema
                    action_enum = raw_input_schema.get('properties', {}).get('action', {}).get('enum')
                    tool_meta[schema['name']] = {
                        'type': tool.get('type'),
                        'action_enum': action_enum,
                        'has_config_schema': bool(tool.get('configSchema')),
                        'completion': raw_input_schema.get('x-completion'),
                        'resource': parse_resources(raw_input_schema.get('x-resource')),
                    }
                else:
                    # 拆分：多个 sub-schemas
                    group = []
                    for schema in tool_schemas:
                        schemas[schema['name']] = schema
                        # 拆分后用 schema 中的 parameters 作为 inputSchema
                        input_schemas[schema['name']] = schema.get('parameters', {'type': 'object', 'properties': {}})
                        tool_meta[schema['name']] = {
                            'type': tool.get('type'),
                            'action_enum': None,
                            'has_config_schema': bool(tool.get('configSchema')),
                            'completion': (tool.get('inputSchema') or {}).get('x-completion'),
                            'resource': parse_resources(
                                (tool.get('inputSchema') or {}).get('x-resource')),
                        }
                        # 解析 action name（最后一段 __）
                        action_name = schema['name'].split('__')[-1]
                        split_map[schema['name']] = {
                            'tool': tool['name'],
                            'action': action_name,
                        }
                        group.append(schema['name'])
                    tool_groups[tool['name']] = group

            online = True
        except Exception as e:
            online = False

    registry[mcp_id] = {
        'name':          name,
        'url':           url,
        'online':        online,
        'tools':         tools,
        'render_hint':   render_hint,
        'schemas':       schemas,
        'tool_meta':     tool_meta,
        'split_map':     split_map,
        'tool_groups':   tool_groups,
        'input_schemas': input_schemas,
        'tool_definitions': tool_definitions,
        # "A full connect has happened for this device." The heartbeat in
        # api/mcp_manage.py reads this to decide whether to call us, because it
        # is the *connect* that matters, not any one key it leaves behind —
        # notably the SSE subscription started just below, which nothing else
        # sets up. Only set when the connect actually reached the device:
        # a failed attempt must be retried on the next heartbeat.
        'connected':     online,
    }

    # 3. 后台订阅 SSE 事件流（非阻塞）
    if online:
        _start_sse(mcp_id, url)


# mcp_id → 该设备当前的 SSE 订阅 task。
#
# 必须有人记着它：`_connect_one` 不止在启动时跑一次，`api/mcp_manage.py` 的心跳
# 分支在 registry 缺 input_schemas 时也会再调一次。原来每调一次就 create_task 一个
# 新的订阅循环，而旧的谁也不认识、永远不退出，于是每次心跳泄漏一个 task。
#
# 天轶实测：驱动日志里 `GET /mcp/sse → 404` 的速率逐小时递增 39k → 52k → 60k →
# 67k → 74k 每小时（约 20 req/s），agent-core 一重启立刻归零再重新爬。心跳 30s 一次
# ⇒ 每小时多 120 个 task，每个退避到 60s 上限 ⇒ 每小时多约 2 req/s，和实测吻合。
# 这些 404 占了那台机器驱动日志的 98.5%（310262 / 315088 行）。
_sse_tasks: dict[str, asyncio.Task] = {}

# mcp_id → the sse url we already reported as absent, so "this server has no SSE
# endpoint" is said once per device rather than once per reconnect. Keyed by url
# too: a device that moves to a new port deserves to be reported again.
_sse_absent: dict[str, str] = {}


def _start_sse(mcp_id: str, url: str) -> None:
    """(重)启动一个设备的 SSE 订阅，先取消上一个。"""
    old = _sse_tasks.pop(mcp_id, None)
    if old is not None and not old.done():
        old.cancel()
    _sse_tasks[mcp_id] = asyncio.create_task(
        _subscribe_sse(mcp_id, url), name=f'sse:{mcp_id}')


def stop_sse(mcp_id: str | None = None) -> None:
    """取消 SSE 订阅：给 mcp_id 就取消那一个，不给就全部。"""
    ids = [mcp_id] if mcp_id else list(_sse_tasks)
    for i in ids:
        task = _sse_tasks.pop(i, None)
        if task is not None and not task.done():
            task.cancel()


async def _subscribe_sse(mcp_id: str, url: str) -> None:
    """长连接订阅 MCP 的 SSE 事件流，推到 event_bus。重连策略：指数退避最多 60s。"""
    sse_url   = url.rstrip('/') + '/sse'
    delay     = 2.0
    timeout   = aiohttp.ClientTimeout(total=None, sock_read=60)
    # 404 = 这个 server 根本没有 SSE 端点。15 个驱动里只有 4 个实现了 `/mcp/sse`，
    # 所以这是常态而不是故障，重试多少次都不会变成 200。连着两次就收工，等下一次
    # `_connect_one`（重连、重新注册）再决定要不要重新订阅。
    #
    # 5xx 之类不在此列：那是「现在不行」，退避重试是对的。
    missing = 0

    while True:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(sse_url) as resp:
                    if resp.status == 404:
                        missing += 1
                        if missing >= 2:
                            # Once per device, not once per reconnect. This line
                            # was the second-biggest source of noise in
                            # agent-core's own log (206 lines in 1642) because
                            # `_connect_one` re-ran every 30s — the churn is
                            # fixed separately, but "this server has no SSE" is
                            # a standing fact either way and does not improve by
                            # being repeated.
                            if _sse_absent.get(mcp_id) != sse_url:
                                _sse_absent[mcp_id] = sse_url
                                print(f'[mcp] {mcp_id}: no SSE endpoint at {sse_url} '
                                      f'(404) — not subscribing')
                            return
                        await asyncio.sleep(delay)
                        continue
                    if resp.status >= 400:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60)
                        continue
                    delay = 2.0
                    missing = 0
                    async for line in resp.content:
                        line = line.decode().strip()
                        if not line.startswith('data:'):
                            continue
                        raw = line[5:].strip()
                        try:
                            msg = json.loads(raw)
                            text    = msg.get('text') or msg.get('message') or raw
                            payload = msg.get('payload', {})
                        except json.JSONDecodeError:
                            text    = raw
                            payload = {}

                        # ACP: action_complete 事件 → 解锁 sync() 等待
                        msg_type = msg.get('type') if isinstance(msg, dict) else None
                        if msg_type == 'action_complete':
                            action_id = msg.get('action_id') or payload.get('action_id')
                            if action_id:
                                mark_action_complete(action_id, msg)

                        await event_bus.enqueue(
                            source  = f'mcp:{mcp_id}',
                            text    = text,
                            payload = payload,
                        )
        except asyncio.CancelledError:
            return
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


# ── 初始化所有配置的 MCP ───────────────────────────────────────────────────────

async def init_all() -> None:
    """在启动时并行连接所有 services.mcp 配置项。"""
    mcp_list = config.main.get('services', {}).get('mcp', [])
    tasks = [
        _connect_one(
            mcp_id      = m['id'],
            name        = m.get('name', m['id']),
            url         = m.get('url', ''),
            render_hint = m.get('render_hint', ''),
        )
        for m in mcp_list
        if m.get('transport', 'http') == 'http' and m.get('url')
    ]
    if tasks:
        await asyncio.gather(*tasks)

    # Register internal MCPs (transport='internal') into registry for tool schema lookup
    _register_internal_mcps()


def _register_internal_mcps():
    """Register internal MCPs (agentcore, channel) into registry so their
    tool schemas are available for _get_bound_tool_schemas() in llm.py."""
    mcp_list = config.main.get('services', {}).get('mcp', [])
    for m in mcp_list:
        if m.get('transport') != 'internal':
            continue
        mcp_id = m.get('id', '')
        if not mcp_id or mcp_id in registry:
            continue
        tools = m.get('tools', [])
        schemas = {}
        input_schemas = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = tool.get('name', '')
            full_name = f'mcp__{mcp_id}__{tool_name}'
            # Build schema in the format LLM expects
            schema = {
                'name': full_name,
                'description': tool.get('description', ''),
                'parameters': tool.get('inputSchema', {'type': 'object', 'properties': {}}),
            }
            schemas[full_name] = schema
            input_schemas[full_name] = tool.get('inputSchema', {})
        registry[mcp_id] = {
            'online': True,
            'transport': 'internal',
            'schemas': schemas,
            'input_schemas': input_schemas,
            'tool_groups': {},
            'split_map': {},
            'tool_definitions': tools,
        }


# ── 工具调用 ────────────────────────────────────────────────────────────────────

def _get_tool_config(mcp_id: str, tool_name: str) -> dict | None:
    """查找 per-tool 持久化 config（由前端 sidebar 保存）。"""
    return config.main.get(f'tool_config:{mcp_id}:{tool_name}', None)


async def _transfer_file_args(
    mcp_id: str, url: str, input_schema: dict, args: dict
) -> tuple[dict, str | None]:
    """Send any `format: file` argument to the target service; rewrite the path.

    Returns `(args, error)`. `error` is a message for the model when the file
    cannot be handed over — it must not fall through to the tool, or the tool
    reports "cannot read <path>" and the model retries with another path it
    invented, which is the loop this exists to break.

    Already-remote values are left alone: an operator who used the canvas picker
    has a path in the *target's* namespace, and re-sending it would fail (it does
    not exist here). The test is simply whether the file exists locally.
    """
    properties = (input_schema.get('properties') or {})
    file_keys = [
        key for key, spec in properties.items()
        if isinstance(spec, dict) and spec.get('format') == 'file'
    ]
    if not file_keys:
        return args, None

    import os

    updated = dict(args)
    for key in file_keys:
        local_path = updated.get(key)
        if not local_path or not isinstance(local_path, str):
            continue
        if not os.path.isfile(local_path):
            # Not a local file. Either it is already the target's path (the
            # canvas picker's output, or a second call reusing an earlier
            # result), or the model invented it. Let the tool answer — it knows
            # its own filesystem, and its error names the upload mechanism.
            continue
        try:
            remote_path = await _push_file(mcp_id, url, local_path)
        except Exception as error:  # noqa: BLE001 - reported to the model
            return updated, (
                f'Error: could not send {local_path!r} to {mcp_id}: {error}. '
                f'The service may be running an image without the /file/upload '
                f'endpoint; check its version, or pass a URL if the tool takes one.'
            )
        print(f'[mcp] {key}: sent {local_path} → {mcp_id}:{remote_path}')
        updated[key] = remote_path
    return updated, None


async def _push_file(mcp_id: str, url: str, local_path: str) -> str:
    """Upload one local file to a service's /file/upload; return its path there.

    Streams from disk rather than reading the file in: a photo can be tens of
    megabytes and this process runs on a 7.4 GB robot alongside everything else.
    """
    import os

    base = url.rsplit('/mcp', 1)[0] if url.endswith('/mcp') else url.rstrip('/')
    endpoint = f'{base}/file/upload'

    import auth
    token = auth.get_token()
    headers = {'X-Access-Token': token} if token else {}

    form = aiohttp.FormData()
    with open(local_path, 'rb') as handle:
        form.add_field('file', handle, filename=os.path.basename(local_path))
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, data=form, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f'HTTP {resp.status}: {text[:300]}')
                payload = json.loads(text)
    if not payload.get('ok') or not payload.get('path'):
        raise RuntimeError(payload.get('error') or 'upload rejected')
    return payload['path']


async def call_tool(full_name: str, args: dict) -> str:
    """
    调用 MCP 工具。full_name 格式: 'mcp__<mcp_id>__<tool_name>'
    或拆分后的格式: 'mcp__<mcp_id>__<tool_name>__<action>'

    返回工具结果的文本表示（用于填入 tool role 消息）。
    图片内容返回 OpenAI multi-modal list。
    """
    # 优先查找 split_map（拆分工具的反向解析）
    mcp_id = None
    tool_name = None
    for mid, info in registry.items():
        split = info.get('split_map', {}).get(full_name)
        if split:
            mcp_id = mid
            tool_name = split['tool']
            args = {**args, 'action': split['action']}
            break

    if mcp_id is None:
        # 原有逻辑：3-part split
        parts = full_name.split('__', 2)
        if len(parts) != 3:
            return f'工具名格式错误: {full_name}'
        _, mcp_id, tool_name = parts

        # A split tool's real name has four segments
        # (`mcp__<id>__loco__move`), and models sometimes emit the action
        # without the tool segment (`mcp__<id>__move`). That used to fall
        # through here as tool_name='move' with no `action` injected, so the
        # driver got a tool it does not have and the call silently did nothing.
        # Recover when the action name is unambiguous, and say so rather than
        # guessing quietly.
        info_probe = registry.get(mcp_id) or {}
        if tool_name not in (info_probe.get('tools') or []):
            matches = {
                (s['tool'], s['action'])
                for s in (info_probe.get('split_map') or {}).values()
                if s.get('action') == tool_name
            }
            if len(matches) == 1:
                real_tool, real_action = matches.pop()
                print(f'[mcp] {full_name} is not a tool name — resolved to '
                      f'{real_tool}(action={real_action}); the model dropped the tool segment')
                tool_name = real_tool
                args = {**args, 'action': real_action}
            elif matches:
                opts = ', '.join(sorted(f'mcp__{mcp_id}__{t}__{a}' for t, a in matches))
                return (f'工具名 {full_name} 不明确：{len(matches)} 个工具都有 '
                        f'{tool_name} 动作。请使用完整名称之一：{opts}')

    info = registry.get(mcp_id)
    if not info:
        return f'MCP {mcp_id} 未注册'

    # Internal tools (agentcore) — dispatch locally
    if info.get('transport') == 'internal':
        return await _dispatch_internal(mcp_id, tool_name, args)

    # A paired peer's tools, reached over its signed link instead of local HTTP.
    # Routed here so a remote tool travels the same path as any other: same
    # schema plumbing, same history, same ACP handling. See peer/mcp_bridge.py.
    if info.get('transport') == 'peer':
        from peer import mcp_bridge
        return await mcp_bridge.call(mcp_id, tool_name, args)

    url     = info['url']
    # Actuator/processor tools (e.g. load_map, navigate) may need longer than 30s
    meta = info.get('tool_meta', {}).get(full_name, {})
    tool_type = meta.get('type', '')
    if tool_type in ('actuator', 'processor'):
        timeout = aiohttp.ClientTimeout(total=60)
    else:
        timeout = aiohttp.ClientTimeout(total=30)

    # ── ACP: 提取内部控制参数（不送给 driver）──────────────────────────────────
    cancel_event = args.pop('_cancel_event', None)
    trace_id = args.pop('_trace_id', None)
    if trace_id:
        args['_trace_id'] = trace_id  # _trace_id 保留给 driver（driver 需要）

    # ── 文件参数：本地路径 → 目标服务内的路径 ──────────────────────────────
    # Runs before validation, because the value the LLM supplied is a path in
    # *this* container and the tool needs one in its own.
    #
    # The browser has had this since `format: file` existed: the canvas renders a
    # picker and uploads through /api/mcp/<id>/file/upload. The LLM had no
    # equivalent, so it could only guess — and it guessed wrong twice in
    # production, passing /work/dai_wenyuan_1.jpeg for a file it had just
    # downloaded to /tmp. Nothing about that is specific to face recognition, so
    # the transfer belongs here, at the one point every MCP call passes through,
    # rather than in any one card: a future tool that declares `format: file`
    # gets it without knowing this code exists.
    input_schema = info.get('input_schemas', {}).get(full_name)
    if input_schema:
        args, transfer_error = await _transfer_file_args(
            mcp_id, url, input_schema, args)
        if transfer_error:
            return transfer_error

    # ── 参数校验：按工具声明的 inputSchema 验证 LLM 生成的参数 ──────────────
    if input_schema:
        try:
            jsonschema.validate(instance=args, schema=input_schema)
        except jsonschema.ValidationError as ve:
            msg = f'参数校验失败: {ve.message}'
            if ve.schema_path:
                msg += f' (schema path: {"/".join(str(p) for p in ve.schema_path)})'
            print(f'[mcp] {full_name} validation error: {msg}')
            return msg

    # Auto-config: start 前自动 apply 已保存的 config
    action = args.get('action')
    if action == 'start':
        meta = info.get('tool_meta', {}).get(full_name, {})
        if meta.get('has_config_schema'):
            saved_cfg = _get_tool_config(mcp_id, tool_name)
            if saved_cfg:
                # Drop keys the current schema no longer advertises; a stale row
                # would otherwise be replayed on every start. See tool_config.
                from tool_config import find_tool, split_config_by_scope
                _shared, _inst = split_config_by_scope(find_tool(mcp_id, tool_name), saved_cfg)
                saved_cfg = {**_shared, **_inst}
            if saved_cfg:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    cfg_result = await _jrpc(session, url, 'tools/call', {
                        'name':      tool_name,
                        'arguments': {'action': 'config', **saved_cfg},
                    })
                # 检查 config 结果，adapter_ok=false 说明凭据无效
                try:
                    cfg_text = (cfg_result.get('content') or [{}])[0].get('text', '{}')
                    cfg_parsed = json.loads(cfg_text)
                    if not cfg_parsed.get('adapter_ok', True):
                        return f'[{tool_name}] 配置无效（缺少 url/key），请在设备面板中检查配置后再启动。'
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
            else:
                return f'[{tool_name}] 尚未配置，请先在设备面板中完成配置（provider/url/key）后再启动。'

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await _jrpc(session, url, 'tools/call', {
                'name':      tool_name,
                'arguments': args,
            })
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
        msg = f'[{tool_name}] MCP 调用超时（{int(timeout.total)}s），设备可能正在执行耗时操作（如地图上传/定位）。请稍后重试。'
        print(f'[mcp] {full_name} timeout after {timeout.total}s')
        return msg

    # The driver answered with a JSON-RPC error. Hand it to the model verbatim —
    # this is the difference between "the model retries correctly" and "the model
    # believes it moved the robot".
    jrpc_err = result.get(JRPC_ERROR_KEY)
    if jrpc_err is not None:
        code = jrpc_err.get('code') if isinstance(jrpc_err, dict) else None
        emsg = jrpc_err.get('message') if isinstance(jrpc_err, dict) else str(jrpc_err)
        print(f'[mcp] {full_name} → error {code}: {emsg}')
        valid = sorted((info.get('split_map') or {}).keys()) or sorted(info.get('tools') or [])
        hint = f' 该设备可用工具：{", ".join(valid)}' if code == -32601 and valid else ''
        return f'[{tool_name}] 调用失败（{code}）：{emsg}。此次调用未执行任何动作。{hint}'

    # MCP call result: list of content items
    content_items = result.get('content', [])
    if not content_items:
        return result.get('text', str(result))

    # 图片 → multimodal list
    images = [c for c in content_items if c.get('type') == 'image']
    texts  = [c.get('text', '') for c in content_items if c.get('type') == 'text']

    if images:
        # 与 Read 读图同一个开关。关闭时按**失败**返回，不要返回「成功但没有图像」——
        # 模型会把成功结果当成「我拿到了这张图」，然后凭 mime 和字节数编出画面内容。
        from event.desktop import vision_input_enabled
        if not vision_input_enabled():
            lines = list(texts)
            for img in images:
                mime = img.get('mimeType', 'image/jpeg')
                approx = len(img.get('data', '')) * 3 // 4
                lines.append(
                    f'Error: cannot parse image contents — this model does not accept '
                    f'image input. Image: [{mime} | {approx} bytes]')
            return '\n'.join(lines)
        parts_list = []
        for img in images:
            data   = img.get('data', '')
            mime   = img.get('mimeType', 'image/jpeg')
            parts_list.append({'type': 'image_url', 'image_url': f'data:{mime};base64,{data}'})
        if texts:
            parts_list.insert(0, {'type': 'text', 'text': '\n'.join(texts)})
        return parts_list   # type: ignore[return-value]  — LLM client accepts list too

    text_result = '\n'.join(texts) or str(result)

    # 更新动态 topic 信息（如 start 工具返回了 topic_out/topic_in）
    if texts:
        try:
            parsed = json.loads(texts[0])
            for key in ('topic_out', 'topic_in'):
                dyn_topics = parsed.get(key)
                if isinstance(dyn_topics, list):
                    existing = registry[mcp_id].setdefault(key, [])
                    for t in dyn_topics:
                        if t.get('topic'):
                            for ex in existing:
                                if ex.get('topic') == t['topic']:
                                    ex.update(t)
                                    break
                            else:
                                existing.append(t)
        except Exception:
            pass

    # ── ACP: 异步工具 — 注册 pending，立即返回（barrier 在 _dispatch 层）────────
    action = args.get('action')
    meta = info.get('tool_meta', {}).get(full_name, {})
    completion_spec = meta.get('completion')
    if completion_spec and _should_await_completion(completion_spec, action):
        try:
            parsed_result = json.loads(texts[0]) if texts else {}
            action_id = parsed_result.get('action_id')
            if action_id:
                _pending_actions[action_id] = asyncio.Event()
                # 记录该 pending 属于哪个工具（用于 barrier 资源冲突判断）
                _pending_tools[action_id] = tool_name
                _pending_resources[action_id] = meta.get('resource')
                _pending_owner[action_id] = current_agent_context.get()
                # 动态 timeout：有 text 参数时按字数算（合成+播放: 字数/3 + 10s余量），否则用 schema 默认值
                text_arg = args.get('text', '')
                default_timeout = completion_spec.get('timeout', 120)
                if text_arg:
                    dynamic_timeout = len(text_arg) / 3 + 10
                else:
                    dynamic_timeout = default_timeout
                _pending_timeouts[action_id] = dynamic_timeout
                if action_id in _pending_results:
                    mark_action_complete(action_id, _pending_results[action_id])
                _res = _pending_resources.get(action_id)
                _res_txt = ','.join(sorted(_res)) if _res else 'undeclared/exclusive'
                print(f'[acp] registered pending: {action_id} (tool={tool_name}, '
                      f'timeout={dynamic_timeout:.0f}s, resource={_res_txt})')
        except (json.JSONDecodeError, IndexError):
            pass

    return text_result


# ── 便捷查询 ─────────────────────────────────────────────────────────────────────

_SYSTEM_ACTIONS = {'start', 'stop', 'info', 'config'}


def present_to_llm(schema: dict, meta: dict | None, split_action: str | None = None) -> dict | None:
    """把一个注册表里的原始 schema 变成可以交给 LLM 的形状。返回 None = 不该暴露。

    两件事：**滤掉 processor 的系统 action**，以及**注入 concurrent 参数**。

    必须是所有"把工具交给模型"的路径共用的唯一入口。这个过滤原本只写在 all_schemas()
    里，而主 agent loop 走的是 event/llm.py 的 _get_bound_tool_schemas()（直接取
    info['schemas'][name] 生料），all_schemas() 全仓库只有 peer 那一处调用 —— 于是过滤
    **从来没有在主链路生效过**。

    后果实测到两个：
    · Orin5 上用户说"好了别说了"，模型调了 tts(action="stop")。它看到的 enum 就是
      ["start","stop","speak","info","config","interrupt"]，描述还写着 "start/stop
      speech synthesis" —— 选 stop 是最自然的读法。但 stop 会 _dispose_node() 把
      话题订阅节点整个拆掉，而"停住这句话"应该是 interrupt。
    · concurrent 参数只在 all_schemas() 里注入，所以主链路上**任何工具都没有这个参数**，
      而 system prompt 花了一整段教模型怎么用它。

    `split_action`：x-action-params 拆出来的子工具，action 编码在 schema 名里而不是
    参数里，只能靠它判断这个子工具是不是系统 action。
    """
    meta = meta or {}
    tool_type = meta.get('type')

    if tool_type == 'processor':
        if split_action is not None:
            if split_action in _SYSTEM_ACTIONS:
                return None          # 拆分子工具本身就是系统 action，不暴露
        elif meta.get('action_enum'):
            user_actions = [a for a in meta['action_enum'] if a not in _SYSTEM_ACTIONS]
            if not user_actions:
                return None          # 无用户 action，整个工具不暴露给 LLM
            props = schema.get('parameters', {}).get('properties', {})
            if 'action' in props:
                schema = {**schema, 'parameters': {
                    **schema['parameters'],
                    'properties': {**props,
                                   'action': {**props['action'], 'enum': user_actions}},
                }}
    return with_parallel_param(schema, tool_type)


def all_schemas() -> list[dict]:
    """返回所有在线 MCP 工具的 OpenAI function calling schema 列表。"""
    schemas = []
    for info in registry.values():
        if not info.get('online'):
            continue
        tool_meta = info.get('tool_meta', {})
        split_map = info.get('split_map', {})
        for name, schema in info['schemas'].items():
            presented = present_to_llm(
                schema, tool_meta.get(name),
                split_action=split_map.get(name, {}).get('action') if name in split_map else None)
            if presented is not None:
                schemas.append(presented)
    return schemas


_PARALLEL_PARAM_DESC = (
    '默认 false：这次调用会等你自己此前发起的动作先完成，也就是按你写出的先后顺序执行。'
    '只有当这个动作**本来就该和上一个同时发生**时才设 true（例如讲解时配合的手势）。'
    '安全播报之类"必须先说完再动"的场景不要设 true。'
    '注意：占用同一个物理通道的动作永远串行，设 true 也不会并行。'
)


def with_parallel_param(schema: dict, tool_type: str | None) -> dict:
    """Add the `concurrent` parameter to an acting tool's schema.

    Injected by the harness rather than declared by drivers: whether a call should
    overlap the previous one is a property of the *intent behind this call*, not of
    the hardware. Asking every driver to carry it would put caller intent in the
    device layer, and the flag would then be absent from any driver that forgot.

    Only acting tools get it. `sensor`/`resource` tools are never barriered, so the
    parameter would be noise in their schema and an invitation to set it meaninglessly.
    """
    if tool_type in ('sensor', 'resource'):
        return schema
    params = schema.get('parameters') or {}
    props = params.get('properties') or {}
    if PARALLEL_PARAM in props:
        return schema                      # driver declared its own; do not shadow it
    return {**schema, 'parameters': {
        **params,
        'properties': {**props, PARALLEL_PARAM: {
            'type': 'boolean',
            'description': _PARALLEL_PARAM_DESC,
        }},
    }}


async def _dispatch_internal(mcp_id: str, tool_name: str, args: dict) -> str:
    """Dispatch tool call for internal (agentcore/channel) tools."""
    if tool_name == 'channel_reply':
        action = args.get('action', '')
        if action == 'send':
            text = args.get('text', '') or ''
            files = args.get('files', []) or []
            if not text and not files:
                return 'Error: provide "text" and/or "files".'
            from channel.manager import manager as channel_mgr
            # instance_id 由 llm.py 从画布绑定注入（_bound_instance_ids），
            # 用它解析卡片上选的 channel —— 卡片配置必须真正决定回复去向
            return await channel_mgr.send_reply(
                instance_id=args.get('instance_id', ''),
                text=text,
                files=files,
                mention_open_id=args.get('mention_open_id', ''),
                source_message_id=args.get('source_message_id', ''),
                expect_reply=args.get('expect_reply', False),
                trusted_bot_id=args.get('trusted_bot_id', ''),
            )
        return f'Error: Unknown action "{action}". Use action="send" with "text" and/or "files".'

    # Default: return info for other internal tools
    return json.dumps({'status': 'ok', 'tool': tool_name})


# ── ACP: 异步动作完成协议 ─────────────────────────────────────────────────────

def _should_await_completion(completion_spec: dict, action: str | None) -> bool:
    """判断当前 action 是否为异步动作（需要注册 pending）。"""
    actions_list = completion_spec.get('actions', [])
    if not actions_list:
        return True  # 无 filter → 所有 action 都是异步的
    return action in actions_list


async def cancel_and_reap(tasks) -> None:
    """Cancel tasks and wait for the cancellation to actually be delivered.

    `Task.cancel()` only *requests* cancellation — the task stays pending until
    the loop gets to resume it and raise CancelledError inside it. Cancelling and
    then dropping the reference is what filled the log with

        Task was destroyed but it is pending!
        task: <Task pending ... coro=<Event.wait() ...>>
        task: <Task pending ... coro=<await_pending.<locals>._wait_all() ...>>

    on every barge-in: the barrier's outer task and the inner `Event.wait()`
    children of its `gather` were both abandoned mid-cancellation.
    """
    live = [t for t in tasks if not t.done()]
    for task in live:
        task.cancel()
    if live:
        await asyncio.gather(*live, return_exceptions=True)


async def await_pending(cancel_event: asyncio.Event | None = None, timeout: float = 120,
                        tool_name: str | None = None,
                        want: frozenset | None = None,
                        scoped: bool = False,
                        concurrent: bool = False,
                        owner: str | None = None,
                        reconsider_event: asyncio.Event | None = None) -> dict:
    """等待与 `want` 冲突的 pending actions 完成。

    `scoped=False`（默认）保持全局语义：等所有 pending。`finish` 走这条 —— 结束 turn
    前不该有任何动作还在飞，跟资源无关。

    `reconsider_event` 是比 `cancel_event` 更窄的信号：`cancel_event` 触发时这次等待
    在等的 pending 会被当作"不再关心"直接遗忘（`_forget_pending`），因为调用方
    （interrupt/followup 模式）打算让整个 turn 作废。`reconsider_event` 触发时，正在
    等的那个动作**仍然合法地在别处跑着**——只是这次调用（还没排到号、还没真的发给
    设备的那个）放弃继续等，把协程还给主循环去问一次新的 LLM，而不是连带把别人的
    pending 记账也抹掉。两者互不影响，可以同时传。

    `scoped=True` 时只等资源冲突的那些（见 `resources_conflict`），且 `effective_timeout`
    只对冲突项取 max —— 原来对全部 pending 取 max，一个长动作会把不相干的调用一起拖住。
    清理也只清等到的那几个，不再连带把别人的 pending 抹掉。

    `tool_name` 仅用于日志归因。
    """
    if scoped:
        aids = pendings_to_wait_for(
            want, owner=owner if owner is not None else current_agent_context.get(),
            concurrent=concurrent)
    else:
        aids = list(_pending_actions.keys())
    if not aids:
        return {"status": "no_pending"}

    events = [_pending_actions[aid] for aid in aids if aid in _pending_actions]
    if not events:
        return {"status": "no_pending"}

    # 只对实际要等的 action 取最大 timeout
    effective_timeout = max(_pending_timeouts.get(aid, timeout) for aid in aids)
    _scope_txt = ''
    if scoped:
        _want_txt = ','.join(sorted(want)) if want else 'undeclared/exclusive'
        _held = len(_pending_actions)
        _mode = 'concurrent' if concurrent else 'sequential'
        _scope_txt = (f' want={_want_txt}, {_mode}, '
                      f'{len(aids)}/{_held} pending to wait for;')
    print(f'[acp] barrier: waiting for {aids}'
          f'{_scope_txt} (timeout={effective_timeout:.0f}s)')

    async def _wait_all():
        await asyncio.gather(*[ev.wait() for ev in events])

    try:
        if cancel_event or reconsider_event:
            wait_task = asyncio.create_task(_wait_all())
            tasks = [wait_task]
            cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event else None
            reconsider_task = asyncio.create_task(reconsider_event.wait()) if reconsider_event else None
            if cancel_task:
                tasks.append(cancel_task)
            if reconsider_task:
                tasks.append(reconsider_task)
            try:
                done, _unfinished = await asyncio.wait(
                    tasks,
                    timeout=effective_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                # Also runs when *this* coroutine is cancelled from outside, which
                # is the common case: on barge-in `_acp_barrier` cancels the task
                # running await_pending as soon as a steering message arrives.
                # Without the finally, CancelledError propagated straight out and
                # left _wait_all and its Event.wait() children orphaned — the
                # "Task was destroyed but it is pending!" pair in the R1 logs.
                await cancel_and_reap(tasks)
            if cancel_task and cancel_task in done:
                # 用户打断：只清本次等待的那些，不连带抹掉不相干的 pending
                _forget_pending(aids, 'cancelled')
                return {"status": "cancelled"}
            if reconsider_task and reconsider_task in done:
                # 跟上面不一样：这次等的东西仍然合法地在别处跑着（比如还在导航的
                # navigate），不是"不再关心"，只是这次调用放弃继续排队——所以
                # 不 _forget_pending，aids 的记账原样留着，该谁的完成通知还是谁的。
                return {"status": "reconsidering", "actions": aids}
            if wait_task not in done:
                # Unlike wait_for, asyncio.wait() does not raise on timeout — it
                # returns with an empty `done`. Falling through from here reported
                # a silent timeout as {"status": "completed"}, so a barrier that
                # waited out its full 120s looked identical to one that succeeded.
                # That is precisely the case _acp_barrier_log was added to make
                # attributable: an ACP completion callback that never arrives
                # (self-signed cert rejected, AGENT_CORE_URL misconfigured).
                # Converge on the TimeoutError path below, which already clears
                # pending and reports "timeout".
                raise asyncio.TimeoutError()
        else:
            # Not reached from the agent loop: it creates a cancel_event for every
            # turn (event/llm.py:887), so the branch above is the live one. Kept
            # for direct callers.
            await asyncio.wait_for(_wait_all(), timeout=effective_timeout)

        # 清理已完成的
        _forget_pending(aids, 'completed')
        print(f'[acp] barrier cleared: {aids}')
        return {"status": "completed", "actions": aids}
    except asyncio.TimeoutError:
        _forget_pending(aids, 'timeout')
        print(f'[acp] barrier timeout: {aids}')
        return {"status": "timeout", "actions": aids}


async def sync(action_ids: list[str] | None = None, timeout: float = 120,
               cancel_event: asyncio.Event | None = None) -> dict:
    """等待指定异步动作完成。不指定 ids 则等待所有 pending actions。

    返回: {"status": "completed"|"timeout"|"cancelled", "results": {...}}
    """
    targets = action_ids or list(_pending_actions.keys())
    if not targets:
        return {"status": "no_pending_actions"}

    events = [(aid, _pending_actions[aid]) for aid in targets if aid in _pending_actions]
    if not events:
        return {"status": "no_pending_actions", "note": f"action_ids {targets} not found in pending"}

    async def _wait_all():
        await asyncio.gather(*[ev.wait() for _, ev in events])

    async def _wait_with_cancel():
        """等待完成或取消。"""
        wait_task = asyncio.create_task(_wait_all())
        if cancel_event:
            cancel_task = asyncio.create_task(cancel_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    [wait_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                await cancel_and_reap([wait_task, cancel_task])
            if cancel_task in done:
                raise asyncio.CancelledError()
        else:
            await wait_task

    try:
        await asyncio.wait_for(_wait_with_cancel(), timeout=timeout)
        # 收集结果并清理。先取走 payload，再统一清副表 —— 这里原来只 pop 了
        # _pending_results 和 _pending_actions，把 _pending_timeouts / _pending_tools
        # 永久留在进程里；Phase 3 的 peer 回调正是走这条路，每次委派都会漏一份。
        results = {}
        for aid, _ in events:
            results[aid] = _pending_results.pop(aid, {"status": "completed"})
        _forget_pending((aid for aid, _ in events), 'completed')
        return {"status": "completed", "results": results}
    except asyncio.TimeoutError:
        completed = {aid: _pending_results.pop(aid, {}) for aid, ev in events if ev.is_set()}
        still_pending = [aid for aid, ev in events if not ev.is_set()]
        # 只清理已完成的；还在飞的留着，它们的 barrier 语义没变
        _forget_pending(completed, 'completed')
        return {"status": "timeout", "completed": completed, "pending": still_pending}
    except asyncio.CancelledError:
        return {"status": "cancelled", "pending": [aid for aid, _ in events]}


def get_pending_actions() -> list[str]:
    """返回当前所有 pending action_ids（供 prompt 展示）。"""
    return list(_pending_actions.keys())


def get_pending_for_tool(tool_name: str) -> list[str]:
    """返回指定工具的 pending action_ids。

    资源冲突判定现在用 `conflicting_pending(want)` —— 按物理通道，而不是按工具名。
    工具名太粗也太细：两个 driver 的 `tts` 是两个名字但可能是同一间屋子的同一个声学
    空间，而一个 `arm` 工具可能同时代表 arm_l 和 arm_r 两个独立通道。

    留着这个函数只为按工具归因（日志/诊断）；它在此之前是零调用者，连同
    `await_pending(tool_name=...)` 一起构成了一套搭好却从未接上的资源冲突骨架。
    """
    return [aid for aid, tn in _pending_tools.items() if tn == tool_name and aid in _pending_actions]


# ── Direct Tool Call (bypass barrier/ACP) ────────────────────────────────────

async def call_tool_direct(mcp_id: str, tool_name: str, args: dict) -> dict:
    """Direct MCP tool call — bypasses barrier, ACP, and schema validation.

    Used by system hooks for immediate execution (e.g. interrupt, LED effects).
    Does NOT register pending actions or check barriers.
    """
    entry = registry.get(mcp_id)
    if not entry:
        return {"error": f"device {mcp_id} not registered"}
    if not entry.get('online'):
        return {"error": f"device {mcp_id} offline"}
    url = entry['url']
    if not url:
        # A peer's synthetic entry (peer/mcp_bridge.py) carries no url — its tools
        # reach the remote over the signed /api/peer/tools/call path, not by POSTing
        # here. Falling through posted to the empty string, and aiohttp's failure
        # for that surfaced as `call_tool_direct failed: ` with nothing after the
        # colon, which is what the Orin5/Orin6 logs showed. Say what is wrong.
        return {"error": f"device {mcp_id} has no url — not directly callable "
                         f"(transport={entry.get('transport', '?')!r}); use the "
                         f"transport's own call path"}
    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000) % 1_000_000,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                if "error" in data:
                    return {"error": data["error"]}
                result = data.get("result", {})
                # Extract text content from MCP response
                content = result.get("content", [])
                if content and isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    if text_parts:
                        try:
                            return json.loads(text_parts[0])
                        except (json.JSONDecodeError, IndexError):
                            return {"raw": text_parts[0]}
                return result
    except Exception as e:
        return {"error": f"call_tool_direct failed: {e}"}


def resolve_tool_binding(name: str, args: dict) -> tuple[str, str, str | None] | None:
    """Resolve an LLM-facing tool_call name back to (mcp_id, tool_name, action).

    A device's real action can reach the LLM two ways: unsplit (tool name is
    `mcp__{id}__{tool}`, action is an `args["action"]` value) or split via
    x-action-params (tool name is `mcp__{id}__{tool}__{action}`, no `action`
    arg). Callers that need to compare an LLM tool_call against an x-hooks
    binding — which is always declared against the raw device tool+action —
    must go through this instead of guessing from the name string, or they
    silently stop matching for whichever convention they didn't hardcode.
    Returns None if `name` isn't a currently-registered tool of any device.
    """
    for mcp_id, info in registry.items():
        split = info.get('split_map', {}).get(name)
        if split:
            return mcp_id, split['tool'], split['action']
        prefix = f'mcp__{mcp_id}__'
        if name in info.get('schemas', {}) and name.startswith(prefix):
            return mcp_id, name[len(prefix):], args.get('action')
    return None


async def call_tool_hook(mcp_id: str, tool_name: str, args: dict, *,
                          barrier_aware: bool = False) -> dict:
    """Like `call_tool_direct`, with an opt-in barrier-aware mode for hooks that
    must not behave like an interrupt.

    True interrupt hooks (on_interrupt_*, e-stop) are supposed to bypass
    everything immediately — that's the correct semantics for "stop now no
    matter what". `on_notify` (narrate LLM content so the user isn't left in
    silence during a long tool-calling turn) is not that: firing it should not
    cut off whatever the robot is already saying or doing, and once it does
    speak, the next barrier-respecting tool call shouldn't cut *it* off either.
    `barrier_aware=True` gets both: skip the call if the tool's declared
    x-resource is already held by a pending ACP action, and — if the call
    returns an action_id under a completion spec — register it as pending the
    same way `call_tool`'s normal ACP dispatch does, so it participates in the
    barrier like any LLM-issued call would.
    """
    entry = registry.get(mcp_id)
    if not entry:
        return {"error": f"device {mcp_id} not registered"}

    meta = {}
    if barrier_aware:
        action = args.get('action')
        candidates = [f'mcp__{mcp_id}__{tool_name}__{action}'] if action else []
        candidates.append(f'mcp__{mcp_id}__{tool_name}')
        tool_meta = entry.get('tool_meta', {})
        for candidate in candidates:
            if candidate in tool_meta:
                meta = tool_meta[candidate]
                break
        resource = meta.get('resource')
        # resource_actually_busy 而不是 conflicting_pending：已完成但还没被 barrier
        # 回收的 action 不算占用，否则一句播报会把嘴"锁"到下一次 barrier 为止。
        if resource and resource_actually_busy(resource):
            return {"skipped": "resource busy"}

    result = await call_tool_direct(mcp_id, tool_name, args)

    if barrier_aware and isinstance(result, dict):
        completion_spec = meta.get('completion')
        action = args.get('action')
        if completion_spec and _should_await_completion(completion_spec, action):
            action_id = result.get('action_id')
            if action_id:
                resource = meta.get('resource')
                _pending_actions[action_id] = asyncio.Event()
                _pending_tools[action_id] = tool_name
                _pending_resources[action_id] = resource
                _pending_owner[action_id] = current_agent_context.get()
                text_arg = args.get('text', '')
                default_timeout = completion_spec.get('timeout', 120)
                dynamic_timeout = len(text_arg) / 3 + 10 if text_arg else default_timeout
                _pending_timeouts[action_id] = dynamic_timeout
                if action_id in _pending_results:
                    mark_action_complete(action_id, _pending_results[action_id])
                _res_txt = ','.join(sorted(resource)) if resource else 'undeclared/exclusive'
                print(f'[acp] registered pending: {action_id} (tool={tool_name}, '
                      f'timeout={dynamic_timeout:.0f}s, resource={_res_txt}) [via hook]')

    return result


def cleanup_stale_actions(max_age_s: float = 300):
    """清理超时的 pending actions（防泄漏，由定时器调用）。"""
    # 简单实现：如果 action 超过 max_age 仍未完成，移除
    # 实际超时由 sync() 的 timeout 参数处理，这里作为安全网
    stale = [aid for aid, ev in _pending_actions.items() if ev.is_set()]
    _forget_pending(stale)
