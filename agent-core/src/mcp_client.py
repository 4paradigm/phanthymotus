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
_pending_results: dict[str, dict] = {}            # action_id → completion payload
_pending_timeouts: dict[str, float] = {}          # action_id → dynamic timeout (seconds)
_pending_tools: dict[str, str] = {}               # action_id → tool_name (资源冲突检测用)


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
    }

    # 3. 后台订阅 SSE 事件流（非阻塞）
    if online:
        asyncio.create_task(_subscribe_sse(mcp_id, url))


async def _subscribe_sse(mcp_id: str, url: str) -> None:
    """长连接订阅 MCP 的 SSE 事件流，推到 event_bus。重连策略：指数退避最多 60s。"""
    sse_url   = url.rstrip('/') + '/sse'
    delay     = 2.0
    timeout   = aiohttp.ClientTimeout(total=None, sock_read=60)

    while True:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(sse_url) as resp:
                    if resp.status >= 400:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60)
                        continue
                    delay = 2.0
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
                            if action_id and action_id in _pending_actions:
                                _pending_results[action_id] = msg
                                _pending_actions[action_id].set()

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
        }


# ── 工具调用 ────────────────────────────────────────────────────────────────────

def _get_tool_config(mcp_id: str, tool_name: str) -> dict | None:
    """查找 per-tool 持久化 config（由前端 sidebar 保存）。"""
    return config.main.get(f'tool_config:{mcp_id}:{tool_name}', None)


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

    # ── 参数校验：按工具声明的 inputSchema 验证 LLM 生成的参数 ──────────────
    input_schema = info.get('input_schemas', {}).get(full_name)
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
                # 动态 timeout：有 text 参数时按字数算（合成+播放: 字数/3 + 10s余量），否则用 schema 默认值
                text_arg = args.get('text', '')
                default_timeout = completion_spec.get('timeout', 120)
                if text_arg:
                    dynamic_timeout = len(text_arg) / 3 + 10
                else:
                    dynamic_timeout = default_timeout
                _pending_timeouts[action_id] = dynamic_timeout
                print(f'[acp] registered pending: {action_id} (tool={tool_name}, timeout={dynamic_timeout:.0f}s)')
        except (json.JSONDecodeError, IndexError):
            pass

    return text_result


# ── 便捷查询 ─────────────────────────────────────────────────────────────────────

_SYSTEM_ACTIONS = {'start', 'stop', 'info', 'config'}


def all_schemas() -> list[dict]:
    """返回所有在线 MCP 工具的 OpenAI function calling schema 列表（过滤 processor 系统 action）。"""
    schemas = []
    for info in registry.values():
        if not info.get('online'):
            continue
        tool_meta = info.get('tool_meta', {})
        for name, schema in info['schemas'].items():
            meta = tool_meta.get(name, {})
            # Processor 类型：过滤系统 action
            if meta.get('type') == 'processor' and meta.get('action_enum'):
                user_actions = [a for a in meta['action_enum'] if a not in _SYSTEM_ACTIONS]
                if not user_actions:
                    continue  # 无用户 action，不暴露给 LLM
                # 复制 schema，修改 action enum 只保留用户可调用的
                schema = {**schema, 'parameters': {
                    **schema['parameters'],
                    'properties': {
                        **schema['parameters']['properties'],
                        'action': {**schema['parameters']['properties']['action'], 'enum': user_actions}
                    }
                }}
            schemas.append(schema)
    return schemas


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
                        tool_name: str | None = None) -> dict:
    """等待 pending actions 完成。全局 barrier：等所有 pending。"""
    aids = list(_pending_actions.keys())
    if not aids:
        return {"status": "no_pending"}

    events = [_pending_actions[aid] for aid in aids if aid in _pending_actions]
    if not events:
        return {"status": "no_pending"}

    # 取所有 pending action 中最大的 timeout
    effective_timeout = max(_pending_timeouts.get(aid, timeout) for aid in aids)
    print(f'[acp] barrier: waiting for {aids} (timeout={effective_timeout:.0f}s)')

    async def _wait_all():
        await asyncio.gather(*[ev.wait() for ev in events])

    try:
        if cancel_event:
            wait_task = asyncio.create_task(_wait_all())
            cancel_task = asyncio.create_task(cancel_event.wait())
            try:
                done, _unfinished = await asyncio.wait(
                    [wait_task, cancel_task],
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
                await cancel_and_reap([wait_task, cancel_task])
            if cancel_task in done:
                # 用户打断：清理所有 pending
                for aid in aids:
                    _pending_actions.pop(aid, None)
                    _pending_results.pop(aid, None)
                    _pending_timeouts.pop(aid, None)
                    _pending_tools.pop(aid, None)
                return {"status": "cancelled"}
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
        for aid in aids:
            _pending_actions.pop(aid, None)
            _pending_results.pop(aid, None)
            _pending_timeouts.pop(aid, None)
            _pending_tools.pop(aid, None)
        print(f'[acp] barrier cleared: {aids}')
        return {"status": "completed", "actions": aids}
    except asyncio.TimeoutError:
        for aid in aids:
            _pending_actions.pop(aid, None)
            _pending_results.pop(aid, None)
            _pending_timeouts.pop(aid, None)
            _pending_tools.pop(aid, None)
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
        # 收集结果并清理
        results = {}
        for aid, _ in events:
            results[aid] = _pending_results.pop(aid, {"status": "completed"})
            _pending_actions.pop(aid, None)
        return {"status": "completed", "results": results}
    except asyncio.TimeoutError:
        completed = {aid: _pending_results.pop(aid, {}) for aid, ev in events if ev.is_set()}
        still_pending = [aid for aid, ev in events if not ev.is_set()]
        # 清理已完成的
        for aid in completed:
            _pending_actions.pop(aid, None)
        return {"status": "timeout", "completed": completed, "pending": still_pending}
    except asyncio.CancelledError:
        return {"status": "cancelled", "pending": [aid for aid, _ in events]}


def get_pending_actions() -> list[str]:
    """返回当前所有 pending action_ids（供 prompt 展示）。"""
    return list(_pending_actions.keys())


def get_pending_for_tool(tool_name: str) -> list[str]:
    """返回指定工具的 pending action_ids（barrier 资源冲突用）。"""
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


def cleanup_stale_actions(max_age_s: float = 300):
    """清理超时的 pending actions（防泄漏，由定时器调用）。"""
    # 简单实现：如果 action 超过 max_age 仍未完成，移除
    # 实际超时由 sync() 的 timeout 参数处理，这里作为安全网
    stale = [aid for aid, ev in _pending_actions.items() if ev.is_set()]
    for aid in stale:
        _pending_actions.pop(aid, None)
        _pending_results.pop(aid, None)
