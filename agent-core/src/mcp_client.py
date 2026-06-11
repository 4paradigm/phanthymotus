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

import aiohttp
import jsonschema

import config
import event_bus

# ── 全局注册表 ─────────────────────────────────────────────────────────────────
registry: dict[str, dict] = {}   # mcp_id → info


# ── 内部 JSON-RPC 助手 ─────────────────────────────────────────────────────────

async def _jrpc(session: aiohttp.ClientSession, url: str, method: str, params: dict, req_id: int = 1) -> dict:
    payload = {'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params}
    async with session.post(url, json=payload) as resp:
        data = await resp.json(content_type=None)
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

    info = registry.get(mcp_id)
    if not info:
        return f'MCP {mcp_id} 未注册'

    url     = info['url']
    timeout = aiohttp.ClientTimeout(total=30)

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

    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await _jrpc(session, url, 'tools/call', {
            'name':      tool_name,
            'arguments': args,
        })

    # MCP call result: list of content items
    content_items = result.get('content', [])
    if not content_items:
        return result.get('text', str(result))

    # 图片 → multimodal list
    images = [c for c in content_items if c.get('type') == 'image']
    texts  = [c.get('text', '') for c in content_items if c.get('type') == 'text']

    if images:
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
