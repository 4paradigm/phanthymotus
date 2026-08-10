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
import hashlib
import ipaddress
import json
import math
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import auth
import config
import event_bus
import jsonschema
from mcp_protocol import rpc_response_error, tool_result_error
from teleop.command_broker import (
    InvalidAuthorityBinding,
    TeleopCommandBlocked,
    authority_domain_for_mcp,
    classify_tool_access,
)
from teleop.contracts import TeleopContractError, project_teleop_descriptor

# ── 全局注册表 ─────────────────────────────────────────────────────────────────
registry: dict[str, dict] = {}   # mcp_id → info

# ── ACP: 异步动作完成协议 ──────────────────────────────────────────────────────
_pending_actions: dict[str, asyncio.Event] = {}   # action_id → Event (set on completion)
_pending_results: dict[str, dict] = {}            # action_id → completion payload
_pending_timeouts: dict[str, float] = {}          # action_id → dynamic timeout (seconds)
_pending_tools: dict[str, str] = {}               # action_id → tool_name (资源冲突检测用)


_TELEOP_RESPONSE_LIMIT = 1024 * 1024
_TELEOP_SIGNALING_RESPONSE_LIMIT = 256 * 1024
_TELEOP_TIMEOUT_MIN = 0.25
_TELEOP_TIMEOUT_MAX = 10.0
_SAFE_DRIVER_ERROR_CODES = {
    'boot_mismatch',
    'epoch_mismatch',
    'fence_mismatch',
    'frame_too_large',
    'invalid_arguments',
    'invalid_control',
    'invalid_encoding',
    'invalid_epoch',
    'invalid_fence',
    'invalid_json',
    'invalid_length',
    'invalid_params',
    'invalid_quaternion',
    'invalid_session_id',
    'invalid_type',
    'invalid_value',
    'message_too_large',
    'missing_action',
    'missing_field',
    'missing_identity',
    'non_finite',
    'out_of_range',
    'session_inactive',
    'session_expired',
    'session_mismatch',
    'session_paused',
    'stale_epoch',
    'tracking_mismatch',
    'unknown_action',
    'unknown_field',
    'unknown_tool',
    'unsupported_mode',
    'unsupported_version',
}


class TrustedShadowTransportError(RuntimeError):
    """Sanitized failure from the dedicated Core → Shadow Driver boundary.

    The exception deliberately retains only bounded machine codes.  Driver
    response text, request arguments, service credentials and target URLs are
    never included because teleop arguments contain the active fence token.
    """

    def __init__(
        self,
        code: str,
        *,
        http_status: int | None = None,
        rpc_code: int | None = None,
        rpc_data_code: str | None = None,
    ) -> None:
        self.code = code
        self.http_status = http_status
        self.rpc_code = rpc_code
        self.rpc_data_code = rpc_data_code
        details = [f'code={code}']
        if http_status is not None:
            details.append(f'http_status={http_status}')
        if rpc_code is not None:
            details.append(f'rpc_code={rpc_code}')
        if rpc_data_code is not None:
            details.append(f'rpc_data_code={rpc_data_code}')
        super().__init__(f'trusted shadow driver call failed ({", ".join(details)})')


@dataclass(frozen=True)
class TrustedShadowTarget:
    """Immutable authority endpoint pinned before a fence is issued."""

    mcp_id: str
    url: str
    capability_digest: str
    descriptor_fingerprint: str
    actions: frozenset[str]


def teleop_tool_fingerprint(tool: object) -> str | None:
    """Return a stable fingerprint for the authority-relevant tool contract."""

    if not isinstance(tool, dict) or tool.get('name') != 'teleop_session':
        return None
    try:
        encoded = json.dumps(
            {
                'name': tool.get('name'),
                'type': tool.get('type'),
                'inputSchema': tool.get('inputSchema'),
                'x-teleop': tool.get('x-teleop'),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def teleop_tools_fingerprint(tools: object) -> str | None:
    if not isinstance(tools, list):
        return None
    matches = [
        tool for tool in tools
        if isinstance(tool, dict) and tool.get('name') == 'teleop_session'
    ]
    return teleop_tool_fingerprint(matches[0]) if len(matches) == 1 else None


def teleop_url_safety_code(url: object) -> str | None:
    """Return a stable rejection code, or ``None`` for a safe endpoint URL."""

    if not isinstance(url, str) or len(url) > 2048 or any(char.isspace() for char in url):
        return 'invalid_url'
    try:
        parsed_url = urlsplit(url)
        _ = parsed_url.port
    except ValueError:
        return 'invalid_url'
    if (
        parsed_url.scheme not in ('http', 'https')
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        return 'invalid_url'
    if parsed_url.scheme == 'http':
        hostname = parsed_url.hostname.lower()
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = hostname == 'localhost'
        if not is_loopback:
            return 'insecure_transport'
    return None


def _is_reserved_teleop_tool(mcp_id: str, tool_name: str) -> bool:
    if tool_name == 'teleop_session':
        return True
    for mcp in config.main.get('services', {}).get('mcp', []):
        if mcp.get('id') != mcp_id:
            continue
        return any(
            isinstance(tool, dict)
            and tool.get('name') == tool_name
            and 'x-teleop' in tool
            for tool in mcp.get('tools', [])
        )
    return False


def _configured_tool_descriptor(mcp_id: str, tool_name: str) -> dict | None:
    matches = [
        mcp for mcp in config.main.get('services', {}).get('mcp', [])
        if isinstance(mcp, dict) and mcp.get('id') == mcp_id
    ]
    if len(matches) != 1:
        return None
    tools = [
        tool for tool in matches[0].get('tools', [])
        if isinstance(tool, dict) and tool.get('name') == tool_name
    ]
    return tools[0] if len(tools) == 1 else None


# ── 内部 JSON-RPC 助手 ─────────────────────────────────────────────────────────

async def _jrpc(
    session: aiohttp.ClientSession,
    url: str,
    method: str,
    params: dict,
    req_id: int = 1,
    *,
    trusted: bool = False,
    driver_id: str | None = None,
) -> dict:
    payload = {'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params}
    headers = {'Content-Type': 'application/json'}
    if trusted:
        driver_headers = auth.driver_request_headers(driver_id)
        if not driver_headers:
            raise PermissionError(
                f'trusted Driver request requires an exact credential: {driver_id!r}'
            )
        headers.update(driver_headers)
    async with session.post(
        url,
        json=payload,
        headers=headers,
        allow_redirects=False,
    ) as resp:
        data = await resp.json(content_type=None)
        response_error = rpc_response_error(
            data,
            request_id=req_id,
            http_status=resp.status,
        )
    if response_error:
        raise ConnectionError(f'MCP {method} failed: {response_error}')
    return data['result']


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

async def _connect_one(
    mcp_id: str,
    name: str,
    url: str,
    render_hint: str,
    *,
    trusted: bool = False,
) -> None:
    timeout = aiohttp.ClientTimeout(total=8)
    schemas: dict[str, dict] = {}
    tools:   list[str]       = []
    tool_meta: dict[str, dict] = {}   # schema_name → {type, action_enum}
    split_map:  dict[str, dict] = {}  # split_schema_name → {tool, action}
    tool_groups: dict[str, list] = {} # original_tool_name → [split_schema_names]
    input_schemas: dict[str, dict] = {}  # schema_name → 原始 MCP inputSchema（用于参数校验）
    teleop_fingerprint: str | None = None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # 1. initialize
            await _jrpc(session, url, 'initialize', {
                'protocolVersion': '2024-11-05',
                'capabilities':    {},
                'clientInfo':      {'name': 'phanthy-motus', 'version': '1.0'},
            }, trusted=trusted, driver_id=mcp_id)

            # 2. tools/list
            result = await _jrpc(
                session,
                url,
                'tools/list',
                {},
                trusted=trusted,
                driver_id=mcp_id,
            )
            discovered_tools = result.get('tools', [])
            teleop_fingerprint = teleop_tools_fingerprint(discovered_tools)
            for tool in discovered_tools:
                if 'x-teleop' in tool:
                    # Dedicated teleop descriptors remain discoverable through
                    # /api/teleop only; never expose them to ordinary agents.
                    continue
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
                        'annotations': tool.get('annotations') or {},
                        'x_teleop': tool.get('x-teleop'),
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
                            'annotations': tool.get('annotations') or {},
                            'x_teleop': tool.get('x-teleop'),
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
        except Exception:
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
        'trusted':       trusted,
        'teleop_fingerprint': teleop_fingerprint,
    }

    # 3. 后台订阅 SSE 事件流（非阻塞）
    if online:
        asyncio.create_task(_subscribe_sse(mcp_id, url, trusted=trusted))


async def _subscribe_sse(mcp_id: str, url: str, *, trusted: bool = False) -> None:
    """长连接订阅 MCP 的 SSE 事件流，推到 event_bus。重连策略：指数退避最多 60s。"""
    sse_url   = url.rstrip('/') + '/sse'
    delay     = 2.0
    timeout   = aiohttp.ClientTimeout(total=None, sock_read=60)

    while True:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = auth.driver_request_headers(mcp_id) if trusted else {}
                if trusted and not headers:
                    return
                async with session.get(sse_url, headers=headers) as resp:
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

    def _runtime_allowed(mcp: dict) -> bool:
        state = mcp.get('trust_state')
        if state == 'trusted':
            return auth.driver_record_credential_available(
                mcp.get('id', ''),
                mcp.get('credential_binding'),
            )
        # Before enforcement, preserve legacy MCP runtime behavior.  Once
        # enforcement is enabled, missing/untrusted records remain discovery-
        # only until they re-register with the driver token.
        return not auth.is_driver_auth_enforced() and state in (None, 'untrusted')

    tasks = [
        _connect_one(
            mcp_id      = m['id'],
            name        = m.get('name', m['id']),
            url         = m.get('url', ''),
            render_hint = m.get('render_hint', ''),
            trusted     = m.get('trust_state') == 'trusted',
        )
        for m in mcp_list
        if (
            m.get('transport', 'http') == 'http'
            and m.get('url')
            and _runtime_allowed(m)
        )
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


def _safe_saved_config(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key not in {'action', 'instance_id'}
    }


def _trusted_shadow_target(mcp_id: str, action: str) -> TrustedShadowTarget:
    """Resolve one exact, trusted Shadow Driver target from config + runtime."""
    services = config.main.get('services', {})
    entries = services.get('mcp', []) if isinstance(services, dict) else []
    matches = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get('id') == mcp_id
    ]
    if not matches:
        raise TrustedShadowTransportError('target_not_found')
    if len(matches) != 1:
        raise TrustedShadowTransportError('target_ambiguous')

    target = matches[0]
    if target.get('trust_state') != 'trusted':
        raise TrustedShadowTransportError('target_not_trusted')
    if not auth.driver_record_credential_available(
        mcp_id,
        target.get('credential_binding'),
    ):
        raise TrustedShadowTransportError('driver_auth_unavailable')
    if target.get('transport') != 'http':
        raise TrustedShadowTransportError('invalid_transport')

    url = target.get('url')
    url_rejection = teleop_url_safety_code(url)
    if url_rejection is not None:
        raise TrustedShadowTransportError(url_rejection)
    assert isinstance(url, str)

    runtime = registry.get(mcp_id)
    if not isinstance(runtime, dict) or runtime.get('online') is not True:
        raise TrustedShadowTransportError('registry_offline')
    if runtime.get('trusted') is not True:
        raise TrustedShadowTransportError('registry_not_trusted')
    if runtime.get('url') != url:
        raise TrustedShadowTransportError('registry_target_mismatch')

    tools = target.get('tools')
    if not isinstance(tools, list):
        raise TrustedShadowTransportError('descriptor_invalid')
    session_tools = [
        tool for tool in tools
        if isinstance(tool, dict) and tool.get('name') == 'teleop_session'
    ]
    if len(session_tools) != 1:
        raise TrustedShadowTransportError('descriptor_invalid')
    session_tool = session_tools[0]
    try:
        descriptor = project_teleop_descriptor(
            session_tool,
            expected_driver_id=mcp_id,
        )
    except TeleopContractError:
        raise TrustedShadowTransportError('descriptor_invalid')
    digest = descriptor['capability_digest']

    input_schema = session_tool.get('inputSchema')
    properties = input_schema.get('properties') if isinstance(input_schema, dict) else None
    action_schema = properties.get('action') if isinstance(properties, dict) else None
    action_enum = action_schema.get('enum') if isinstance(action_schema, dict) else None
    action_params = input_schema.get('x-action-params') if isinstance(input_schema, dict) else None
    if not isinstance(action_enum, list) or not isinstance(action_params, dict):
        raise TrustedShadowTransportError('descriptor_invalid')
    declared_actions = frozenset(
        item for item in action_enum
        if isinstance(item, str) and item
    )
    if (
        len(declared_actions) != len(action_enum)
        or not set(action_params).issuperset(declared_actions)
    ):
        raise TrustedShadowTransportError('descriptor_invalid')
    if (
        not isinstance(action, str)
        or action not in declared_actions
        or action not in action_params
    ):
        raise TrustedShadowTransportError('action_not_declared')
    descriptor_fingerprint = teleop_tool_fingerprint(session_tool)
    if (
        descriptor_fingerprint is None
        or runtime.get('teleop_fingerprint') != descriptor_fingerprint
    ):
        raise TrustedShadowTransportError('registry_descriptor_mismatch')

    return TrustedShadowTarget(
        mcp_id=mcp_id,
        url=url,
        capability_digest=digest,
        descriptor_fingerprint=descriptor_fingerprint,
        actions=declared_actions,
    )


def _validate_pinned_shadow_target(
    mcp_id: str,
    action: str,
    target: TrustedShadowTarget,
) -> str:
    """Validate only in-memory identity before reusing an issued fence."""

    if target.mcp_id != mcp_id or action not in target.actions:
        raise TrustedShadowTransportError('pinned_target_changed')
    runtime = registry.get(mcp_id)
    if not auth.driver_runtime_credential_available(mcp_id):
        raise TrustedShadowTransportError('driver_auth_unavailable')
    if (
        not isinstance(runtime, dict)
        or runtime.get('trusted') is not True
        or runtime.get('url') != target.url
        or runtime.get('teleop_fingerprint') != target.descriptor_fingerprint
    ):
        raise TrustedShadowTransportError('pinned_target_changed')
    return target.url


async def resolve_trusted_shadow_target(
    mcp_id: str,
    *,
    timeout_seconds: float = 2.0,
) -> TrustedShadowTarget:
    """Resolve one immutable target before creating a Driver session."""

    timeout_value = _bounded_teleop_timeout(timeout_seconds)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_trusted_shadow_target, mcp_id, 'status'),
            timeout=timeout_value,
        )
    except TrustedShadowTransportError:
        raise
    except asyncio.TimeoutError:
        raise TrustedShadowTransportError('timeout') from None
    except Exception:  # noqa: BLE001 -- keep ConfigDB details private
        raise TrustedShadowTransportError('target_resolution_error') from None


def _bounded_teleop_timeout(timeout_seconds: float) -> float:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
    ):
        raise TrustedShadowTransportError('invalid_timeout')
    return min(_TELEOP_TIMEOUT_MAX, max(_TELEOP_TIMEOUT_MIN, float(timeout_seconds)))


def _reject_non_finite_json(value: str):
    raise ValueError(f'non-finite JSON token: {value}')


def _decode_strict_json_object(raw: bytes, error_code: str) -> dict:
    try:
        value = json.loads(
            raw.decode('utf-8'),
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise TrustedShadowTransportError(error_code) from None
    if not isinstance(value, dict):
        raise TrustedShadowTransportError(error_code)
    return value


def _decode_strict_signaling_json_object(raw: bytes) -> dict:
    def reject_duplicate_fields(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON field')
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode('utf-8'),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise TrustedShadowTransportError('invalid_response') from None
    if not isinstance(value, dict):
        raise TrustedShadowTransportError('invalid_response')
    return value


def _safe_rpc_data_code(error) -> str | None:
    if not isinstance(error, dict):
        return None
    data = error.get('data')
    value = data.get('code') if isinstance(data, dict) else None
    return value if isinstance(value, str) and value in _SAFE_DRIVER_ERROR_CODES else None


async def call_trusted_shadow_session(
    mcp_id: str,
    action: str,
    arguments: dict | None = None,
    *,
    timeout_seconds: float = 5.0,
    session: aiohttp.ClientSession | None = None,
    target: TrustedShadowTarget | None = None,
) -> dict:
    """Call a declared ``teleop_session`` action on one trusted Shadow Driver.

    This is intentionally separate from :func:`call_tool`: reserved teleop
    tools stay unavailable to LLM, Canvas and generic MCP callers.  Successful
    calls return the Driver's single JSON text object; every failure uses a
    sanitized :class:`TrustedShadowTransportError`.  Coordinators should pass
    their lifespan-owned ``session`` so heartbeat calls reuse its connector;
    the per-call timeout remains bounded and the supplied session is not closed.
    """
    timeout_value = _bounded_teleop_timeout(timeout_seconds)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict) or 'action' in arguments:
        raise TrustedShadowTransportError('invalid_arguments')

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_value
    if target is not None:
        url = _validate_pinned_shadow_target(mcp_id, action, target)
    else:
        # ConfigDB is SQLite-backed.  Non-session calls resolve off-loop within
        # the same total budget.  Live authority calls always pass a pinned
        # target and therefore never touch SQLite on the 4 Hz heartbeat path.
        try:
            resolved = await asyncio.wait_for(
                asyncio.to_thread(_trusted_shadow_target, mcp_id, action),
                timeout=timeout_value,
            )
            url = resolved.url
        except TrustedShadowTransportError:
            raise
        except asyncio.TimeoutError:
            raise TrustedShadowTransportError('timeout') from None
        except Exception:  # noqa: BLE001 -- never expose ConfigDB/runtime details
            raise TrustedShadowTransportError('target_resolution_error') from None

    request_id = uuid.uuid4().hex
    payload = {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'tools/call',
        'params': {
            'name': 'teleop_session',
            'arguments': {'action': action, **arguments},
        },
    }
    try:
        encoded_payload = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError, RecursionError):
        raise TrustedShadowTransportError('invalid_arguments') from None

    driver_headers = auth.driver_request_headers(mcp_id)
    authorization = driver_headers.get('Authorization')
    if (
        not isinstance(authorization, str)
        or not authorization.startswith('Bearer ')
        or not authorization[7:]
    ):
        raise TrustedShadowTransportError('driver_auth_unavailable')
    headers = {
        'Authorization': authorization,
        'Content-Type': 'application/json',
    }
    remaining_timeout = deadline - loop.time()
    if remaining_timeout <= 0:
        raise TrustedShadowTransportError('timeout')
    request_timeout = aiohttp.ClientTimeout(total=remaining_timeout)

    async def _read_bounded(response: aiohttp.ClientResponse) -> bytes:
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            if len(body) + len(chunk) > _TELEOP_RESPONSE_LIMIT:
                raise TrustedShadowTransportError('response_too_large')
            body.extend(chunk)
        return bytes(body)

    async def _post(client: aiohttp.ClientSession) -> bytes:
        async with client.post(
            url,
            data=encoded_payload,
            headers=headers,
            allow_redirects=False,
            timeout=request_timeout,
        ) as response:
            status = response.status
            if status < 200 or status >= 300:
                raise TrustedShadowTransportError('http_error', http_status=status)
            return await _read_bounded(response)

    try:
        if session is not None:
            if session.closed:
                raise TrustedShadowTransportError('client_session_closed')
            raw = await _post(session)
        else:
            async with aiohttp.ClientSession() as owned_session:
                raw = await _post(owned_session)
    except TrustedShadowTransportError:
        raise
    except asyncio.TimeoutError:
        raise TrustedShadowTransportError('timeout') from None
    except (aiohttp.ClientError, OSError, RuntimeError, ValueError, TypeError):
        raise TrustedShadowTransportError('network_error') from None

    response_payload = _decode_strict_json_object(raw, 'invalid_response')
    if (
        response_payload.get('jsonrpc') != '2.0'
        or response_payload.get('id') != request_id
    ):
        raise TrustedShadowTransportError('invalid_response')

    if 'error' in response_payload:
        rpc_error = response_payload.get('error')
        raw_rpc_code = rpc_error.get('code') if isinstance(rpc_error, dict) else None
        rpc_code = raw_rpc_code if isinstance(raw_rpc_code, int) and not isinstance(raw_rpc_code, bool) else None
        raise TrustedShadowTransportError(
            'rpc_error',
            rpc_code=rpc_code,
            rpc_data_code=_safe_rpc_data_code(rpc_error),
        )

    result = response_payload.get('result')
    content = result.get('content') if isinstance(result, dict) else None
    if not isinstance(content, list) or len(content) != 1:
        raise TrustedShadowTransportError('invalid_result')
    item = content[0]
    if (
        not isinstance(item, dict)
        or item.get('type') != 'text'
        or not isinstance(item.get('text'), str)
    ):
        raise TrustedShadowTransportError('invalid_result')
    is_error = result.get('isError', False)
    if not isinstance(is_error, bool):
        raise TrustedShadowTransportError('invalid_result')
    if is_error:
        raise TrustedShadowTransportError('tool_result_error')

    return _decode_strict_json_object(item['text'].encode('utf-8'), 'invalid_result')


def _trusted_shadow_offer_url(target: TrustedShadowTarget) -> str:
    """Derive the Driver's same-origin RTC endpoint from its pinned MCP URL."""

    if teleop_url_safety_code(target.url) is not None:
        raise TrustedShadowTransportError('invalid_url')
    try:
        parsed = urlsplit(target.url)
        # The generic teleop Driver advertises exactly its root MCP endpoint.
        # Requiring that contract prevents surprising path-prefix, query, or
        # encoded-path interpretations at the separate signaling boundary.
        if parsed.path != '/mcp':
            raise TrustedShadowTransportError('invalid_url')
        offer_url = urlunsplit((parsed.scheme, parsed.netloc, '/offer', '', ''))
        offer = urlsplit(offer_url)
    except (TypeError, ValueError):
        raise TrustedShadowTransportError('invalid_url') from None
    if (
        teleop_url_safety_code(offer_url) is not None
        or (offer.scheme, offer.hostname, offer.port)
        != (parsed.scheme, parsed.hostname, parsed.port)
    ):
        raise TrustedShadowTransportError('invalid_url')
    return offer_url


async def call_trusted_shadow_offer(
    mcp_id: str,
    offer: dict,
    ticket: str,
    *,
    timeout_seconds: float = 2.0,
    session: aiohttp.ClientSession | None = None,
    target: TrustedShadowTarget,
) -> dict:
    """Exchange one ticket-bound SDP offer with the pinned Shadow Driver.

    The ticket and Driver bearer are intentionally accepted only by this
    server-to-server boundary.  Errors retain machine codes, never response
    bodies, URLs, credentials, ticket claims, or SDP.
    """

    timeout_value = _bounded_teleop_timeout(timeout_seconds)
    if (
        not isinstance(offer, dict)
        or set(offer) != {'sdp', 'type'}
        or offer.get('type') != 'offer'
        or not isinstance(offer.get('sdp'), str)
        or not offer['sdp']
        or not isinstance(ticket, str)
        or not ticket
    ):
        raise TrustedShadowTransportError('invalid_arguments')
    if not isinstance(target, TrustedShadowTarget):
        raise TrustedShadowTransportError('pinned_target_changed')

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_value
    mcp_url = _validate_pinned_shadow_target(mcp_id, 'status', target)
    if mcp_url != target.url:
        raise TrustedShadowTransportError('pinned_target_changed')
    url = _trusted_shadow_offer_url(target)

    try:
        encoded_payload = json.dumps(
            {'sdp': offer['sdp'], 'type': 'offer', 'ticket': ticket},
            ensure_ascii=False,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError, RecursionError):
        raise TrustedShadowTransportError('invalid_arguments') from None

    driver_headers = auth.driver_request_headers(mcp_id)
    authorization = driver_headers.get('Authorization')
    if (
        not isinstance(authorization, str)
        or not authorization.startswith('Bearer ')
        or not authorization[7:]
    ):
        raise TrustedShadowTransportError('driver_auth_unavailable')
    headers = {
        'Authorization': authorization,
        'Content-Type': 'application/json',
    }
    remaining_timeout = deadline - loop.time()
    if remaining_timeout <= 0:
        raise TrustedShadowTransportError('timeout')
    request_timeout = aiohttp.ClientTimeout(total=remaining_timeout)

    async def _read_bounded(response: aiohttp.ClientResponse) -> bytes:
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            if len(body) + len(chunk) > _TELEOP_SIGNALING_RESPONSE_LIMIT:
                raise TrustedShadowTransportError('response_too_large')
            body.extend(chunk)
        return bytes(body)

    async def _post(client: aiohttp.ClientSession) -> bytes:
        async with client.post(
            url,
            data=encoded_payload,
            headers=headers,
            allow_redirects=False,
            timeout=request_timeout,
        ) as response:
            if 300 <= response.status < 400:
                raise TrustedShadowTransportError(
                    'redirect_rejected',
                    http_status=response.status,
                )
            if response.status < 200 or response.status >= 300:
                raise TrustedShadowTransportError(
                    'http_error',
                    http_status=response.status,
                )
            return await _read_bounded(response)

    try:
        if session is not None:
            if session.closed:
                raise TrustedShadowTransportError('client_session_closed')
            raw = await _post(session)
        else:
            async with aiohttp.ClientSession() as owned_session:
                raw = await _post(owned_session)
    except TrustedShadowTransportError:
        raise
    except asyncio.TimeoutError:
        raise TrustedShadowTransportError('timeout') from None
    except (aiohttp.ClientError, OSError, RuntimeError, ValueError, TypeError):
        raise TrustedShadowTransportError('network_error') from None

    # Registry identity must still match after the network await.  A concurrent
    # Driver retarget may not turn an old ticket into authority on a new target.
    if loop.time() > deadline:
        raise TrustedShadowTransportError('timeout')
    _validate_pinned_shadow_target(mcp_id, 'status', target)
    response = _decode_strict_signaling_json_object(raw)
    if loop.time() > deadline:
        raise TrustedShadowTransportError('timeout')
    return response


async def call_tool(full_name: str, args: dict) -> str:
    """
    调用 MCP 工具。full_name 格式: 'mcp__<mcp_id>__<tool_name>'
    或拆分后的格式: 'mcp__<mcp_id>__<tool_name>__<action>'

    返回工具结果的文本表示（用于填入 tool role 消息）。
    图片内容返回 OpenAI multi-modal list。
    """
    args = dict(args)

    # 优先查找 split_map（拆分工具的反向解析）
    mcp_id = None
    tool_name = None
    split_action = None
    for mid, info in registry.items():
        split = info.get('split_map', {}).get(full_name)
        if split:
            mcp_id = mid
            tool_name = split['tool']
            split_action = split['action']
            args = {**args, 'action': split_action}
            break

    if mcp_id is None:
        # 原有逻辑：3-part split
        parts = full_name.split('__', 2)
        if len(parts) != 3:
            return f'工具名格式错误: {full_name}'
        _, mcp_id, tool_name = parts

    if _is_reserved_teleop_tool(mcp_id, tool_name):
        return 'Teleop tools are reserved for the dedicated teleop API'

    info = registry.get(mcp_id)
    if not info:
        return f'MCP {mcp_id} 未注册'

    # Internal tools (agentcore) — dispatch locally
    if info.get('transport') == 'internal':
        return await _dispatch_internal(mcp_id, tool_name, args)

    configured_tool = _configured_tool_descriptor(mcp_id, tool_name)
    if configured_tool is None:
        return 'MCP tool not found'

    action = args.get('action')
    meta = info.get('tool_meta', {}).get(full_name)
    if not isinstance(meta, dict):
        return 'MCP tool not found'
    configured_schema = configured_tool.get('inputSchema')
    configured_schema = configured_schema if isinstance(configured_schema, dict) else {}
    properties = configured_schema.get('properties')
    properties = properties if isinstance(properties, dict) else {}
    action_schema = properties.get('action')
    action_schema = action_schema if isinstance(action_schema, dict) else {}
    action_enum = action_schema.get('enum')
    action_declared = isinstance(action_enum, list) and action in action_enum
    access = classify_tool_access(
        tool_type=configured_tool.get('type'),
        annotations=configured_tool.get('annotations'),
        action=action,
        action_declared=action_declared,
    )
    # Imported lazily to keep the trusted teleop transport independent from
    # ordinary MCP dispatch during module initialization.
    from teleop.service import coordinator

    try:
        authority_domain = authority_domain_for_mcp(mcp_id)
    except InvalidAuthorityBinding:
        return json.dumps(
            {
                'error': {
                    'code': 'authority_binding_invalid',
                    'reason': 'core_authority_binding_invalid',
                    'robot_id': mcp_id,
                },
            },
            ensure_ascii=False,
            separators=(',', ':'),
        )

    try:
        async with coordinator.command_broker.ordinary_command(
            authority_domain,
            read_only=access.read_only,
            source='mcp_client',
            tool=tool_name,
            action=action if isinstance(action, str) else '',
            tool_verified=True,
            action_verified=action_declared,
        ):
            return await _call_external_tool(
                full_name,
                mcp_id,
                tool_name,
                args,
                info,
            )
    except TeleopCommandBlocked as error:
        return json.dumps(
            {'error': error.public_detail()},
            ensure_ascii=False,
            separators=(',', ':'),
        )


async def _call_external_tool(
    full_name: str,
    mcp_id: str,
    tool_name: str,
    args: dict,
    info: dict,
):
    """Execute one already-admitted ordinary MCP call."""

    url     = info['url']
    timeout = aiohttp.ClientTimeout(total=30)

    # ── ACP: 提取内部控制参数（不送给 driver）──────────────────────────────────
    args.pop('_cancel_event', None)
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
            saved_cfg = _safe_saved_config(
                _get_tool_config(mcp_id, tool_name)
            )
            if saved_cfg:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    cfg_result = await _jrpc(session, url, 'tools/call', {
                        'name':      tool_name,
                        'arguments': {**saved_cfg, 'action': 'config'},
                    }, trusted=bool(info.get('trusted')), driver_id=mcp_id)
                cfg_error = tool_result_error(
                    cfg_result,
                    require_structured_ack=True,
                )
                if cfg_error:
                    return f'[{tool_name}] Driver 拒绝配置，启动已取消。'
            else:
                return f'[{tool_name}] 尚未配置，请先在设备面板中完成配置（provider/url/key）后再启动。'

    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await _jrpc(session, url, 'tools/call', {
            'name':      tool_name,
            'arguments': args,
        }, trusted=bool(info.get('trusted')), driver_id=mcp_id)

    result_error = tool_result_error(
        result,
        require_structured_ack=(
            isinstance(action, str) and action in {'config', 'start', 'stop'}
        ),
    )
    if result_error:
        return f'[{tool_name}] Driver 调用失败: {result_error}'

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
            text = args.get('text', '')
            if not text:
                return 'Error: "text" field is required.'
            from channel.manager import manager as channel_mgr
            channels_with_context = list(channel_mgr._get_last_context().keys())
            if not channels_with_context:
                return (
                    'Error: No active conversation context. '
                    'A user must send a message to the bot first before it can reply. '
                    'Ask the user to send a message in Feishu/Telegram/Slack.'
                )
            channel_id = channels_with_context[-1]
            return await channel_mgr.send_to_channel(channel_id, text)
        return f'Error: Unknown action "{action}". Use action="send" with a "text" field.'

    # Default: return info for other internal tools
    return json.dumps({'status': 'ok', 'tool': tool_name})


# ── ACP: 异步动作完成协议 ─────────────────────────────────────────────────────

def _should_await_completion(completion_spec: dict, action: str | None) -> bool:
    """判断当前 action 是否为异步动作（需要注册 pending）。"""
    actions_list = completion_spec.get('actions', [])
    if not actions_list:
        return True  # 无 filter → 所有 action 都是异步的
    return action in actions_list


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
            done, pending = await asyncio.wait(
                [wait_task, cancel_task],
                timeout=effective_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            if cancel_task in done:
                # 用户打断：清理所有 pending
                for aid in aids:
                    _pending_actions.pop(aid, None)
                    _pending_results.pop(aid, None)
                    _pending_timeouts.pop(aid, None)
                    _pending_tools.pop(aid, None)
                return {"status": "cancelled"}
        else:
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
            done, pending = await asyncio.wait(
                [wait_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
            if cancel_task in done:
                wait_task.cancel()
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
