import asyncio
import json
import re
import uuid

import aiohttp
import fastapi
from pydantic import BaseModel

import auth
import config
import mcp_client
from mcp_protocol import (
    rpc_response_error as _rpc_response_error,
    tool_result_error as _tool_result_error,
)
from teleop import audit, authority_guard
from teleop.command_broker import (
    InvalidAuthorityBinding,
    TeleopCommandBlocked,
    authority_domain_for_target,
    classify_tool_access,
)

router = fastapi.APIRouter(prefix='/mcp', tags=['mcp'])

_mcp_write_lock = authority_guard.target_mutation_lock
_RESERVED_CONFIG_ARGUMENTS = frozenset({'action', 'instance_id'})


def _get_inspector_url() -> str:
    return ''  # Inspector is now embedded in agent-core; no external URL needed


async def _notify_inspector(mcp_id: str, topics: list) -> None:
    """Register topics with the embedded inspection module (process-internal call)."""
    from api.inspection import register_topic_internal
    for t in topics:
        topic     = t.get('topic', '')
        fmt       = t.get('format', '')
        if not topic:
            continue
        try:
            await register_topic_internal(topic, fmt, mcp_id)
        except Exception:
            pass


def _get_mcp_list() -> list:
    return list(config.main.get('services', {}).get('mcp', []))


def _save_mcp_list(mcp_list: list):
    services = config.main.get('services', {})
    services['mcp'] = mcp_list
    config.main['services'] = services


def _effective_trust_state(mcp: dict) -> str:
    state = mcp.get('trust_state')
    if (
        state == 'trusted'
        and not auth.driver_record_credential_available(
            mcp.get('id', ''),
            mcp.get('credential_binding'),
        )
    ):
        return 'quarantined'
    if state:
        return state
    # Records created before C1 are legacy-untrusted.  Once a deployment opts
    # into driver authentication they are quarantined until re-registration.
    return 'quarantined' if auth.is_driver_auth_enforced() else 'untrusted'


def _runtime_registration_allowed(mcp: dict) -> bool:
    if mcp.get('transport') == 'internal':
        return True
    state = _effective_trust_state(mcp)
    return state == 'trusted' or (state == 'untrusted' and not auth.is_driver_auth_enforced())


def _is_reserved_teleop_tool(tool) -> bool:
    return (
        isinstance(tool, dict)
        and (tool.get('name') == 'teleop_session' or 'x-teleop' in tool)
    )


def _ordinary_tools(tools: list) -> list:
    """Hide teleop-only tools from Canvas, LLM, and generic MCP callers."""
    return [tool for tool in (tools or []) if not _is_reserved_teleop_tool(tool)]


def _descriptor_action_declared(tool: object, arguments: object) -> bool:
    descriptor = tool if isinstance(tool, dict) else {}
    safe_arguments = arguments if isinstance(arguments, dict) else {}
    action = safe_arguments.get('action')
    input_schema = descriptor.get('inputSchema')
    input_schema = input_schema if isinstance(input_schema, dict) else {}
    properties = input_schema.get('properties')
    properties = properties if isinstance(properties, dict) else {}
    action_schema = properties.get('action')
    action_schema = action_schema if isinstance(action_schema, dict) else {}
    action_enum = action_schema.get('enum')
    return isinstance(action_enum, list) and action in action_enum


def _descriptor_tool_access(tool: object, arguments: object):
    descriptor = tool if isinstance(tool, dict) else {}
    safe_arguments = arguments if isinstance(arguments, dict) else {}
    action = safe_arguments.get('action')

    return classify_tool_access(
        tool_type=descriptor.get('type'),
        annotations=descriptor.get('annotations'),
        action=action,
        action_declared=_descriptor_action_declared(descriptor, safe_arguments),
    )


def _safe_saved_config(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key not in _RESERVED_CONFIG_ARGUMENTS
    }


def _outbound_headers(target: dict) -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    if _effective_trust_state(target) == 'trusted':
        driver_headers = auth.driver_request_headers(target.get('id', ''))
        if not driver_headers:
            raise PermissionError('trusted Driver credential is unavailable')
        headers.update(driver_headers)
    return headers


# Cache of last-seen tool names per mcp_id (for change detection)
_last_tool_names: dict[str, list[str]] = {}
_ping_generations: dict[str, int] = {}


def _capability_fingerprint(value: object) -> str:
    """Return a conservative, deterministic snapshot for persisted capabilities."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        # Invalid/non-JSON persisted data must not make two distinct snapshots
        # look equal.  repr is intentionally conservative for this corrupt case.
        return repr(value)


def _ping_target_fingerprint(target: dict) -> tuple:
    """Fields that must remain unchanged while a capability ping is in flight."""

    return (
        target.get('id'),
        target.get('url', ''),
        target.get('transport', 'http'),
        _effective_trust_state(target),
        target.get('name', ''),
        target.get('depends_on', ''),
        target.get('authority_domain', ''),
        target.get('pending_authority_domain', ''),
        bool(target.get('authority_binding_required', False)),
        target.get('authority_binding_error', ''),
        target.get('reported_robot_id', ''),
        bool(target.get('capability_refresh_required', False)),
        _capability_fingerprint(target.get('tools', [])),
    )


def _stale_ping_result(mcp_id: str) -> dict:
    return {
        'online': False,
        'error': 'ping target changed; stale result discarded',
        'tools': [],
        'resources': [],
        'mcp_id': mcp_id,
    }


async def _project_target_mutation_error(
    current_targets: list[dict],
    proposed_targets: list[dict],
) -> dict | None:
    from api import config as config_api

    return await config_api._project_target_mutation_error(
        current_targets,
        proposed_targets,
    )


def _project_target_locked_response(error: dict):
    return fastapi.responses.JSONResponse(
        status_code=409,
        content={
            'code': 409,
            'message': 'Execution target is locked until safe Stop or recovery completes',
            'data': error,
        },
    )


def _project_target_locked_ping_result(
    mcp_id: str,
    target: dict,
    error: dict,
) -> dict:
    return {
        'online': bool(mcp_client.registry.get(mcp_id, {}).get('online', False)),
        'error': 'project execution target is locked until Stop completes',
        'detail': error,
        'tools': target.get('tools', []),
        'resources': target.get('resources', []),
        'render_hint': target.get('render_hint', ''),
        'server_name': target.get('server_name', ''),
        'topic_out': target.get('topic_out', []),
        'topic_in': target.get('topic_in', []),
    }


# ── Models ───────────────────────────────────────────────────────────────────

class MCPAddRequest(BaseModel):
    id:        str = ''
    name:      str
    transport: str = 'http'
    url:       str = ''
    render_hint: str = ''
    category:  str = ''
    robot_id: str = ''

    model_config = {'extra': 'forbid'}


class MCPAuthorityBindingRequest(BaseModel):
    root_mcp_id: str

    model_config = {'extra': 'forbid'}


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _ping_mcp_http(
    url: str,
    *,
    trusted: bool = False,
    driver_id: str | None = None,
) -> dict:
    """Connect to an MCP HTTP server, initialize, list tools and resources."""
    headers = {'Content-Type': 'application/json'}
    if trusted:
        driver_headers = auth.driver_request_headers(driver_id)
        if not driver_headers:
            raise PermissionError(
                f'trusted Driver request requires an exact credential: {driver_id!r}'
            )
        headers.update(driver_headers)
    timeout = aiohttp.ClientTimeout(total=5)
    tools = []
    resources = []
    server_name = ''
    device_type = ''
    topic_out: list = []
    topic_in:  list = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Initialize
        init_payload = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'phanthy-motus', 'version': '1.0'},
            }
        }
        async with session.post(
            url,
            json=init_payload,
            headers=headers,
            allow_redirects=False,
        ) as resp:
            init_data = await resp.json(content_type=None)
            init_error = _rpc_response_error(
                init_data,
                request_id=1,
                http_status=resp.status,
            )
            if init_error:
                raise ConnectionError(f'MCP initialize failed: {init_error}')
            init_result = init_data['result']
            if not isinstance(init_result, dict):
                raise ConnectionError('MCP initialize returned an invalid result')
            server_info = init_result.get('serverInfo')
            server_name = (
                server_info.get('name', '')
                if isinstance(server_info, dict)
                else ''
            )

        # List tools
        tools_payload = {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}
        async with session.post(
            url,
            json=tools_payload,
            headers=headers,
            allow_redirects=False,
        ) as resp:
            data = await resp.json(content_type=None)
            tools_error = _rpc_response_error(
                data,
                request_id=2,
                http_status=resp.status,
            )
            if tools_error:
                raise ConnectionError(f'MCP tools/list failed: {tools_error}')
            tools_result = data['result']
            raw_tools = (
                tools_result.get('tools')
                if isinstance(tools_result, dict)
                else None
            )
            if not isinstance(raw_tools, list) or any(
                not isinstance(tool, dict) for tool in raw_tools
            ):
                raise ConnectionError('MCP tools/list returned an invalid result')
            tools = [
                {k: v for k, v in tool.items() if k in (
                    'name', 'description', 'type', 'multiInstance',
                    'inputSchema', 'configSchema', 'topic_out', 'topic_in',
                    'annotations', 'x-teleop',
                )}
                for tool in raw_tools
            ]

        # Call all *_info / info tools in parallel — device self-reports type and topics.
        # Bundles expose per-plugin tools like mic_info, loco_info; single devices use bare 'info'.
        # Tools with action enum containing 'info' are called with {action: "info"}.
        # Reserved teleop descriptors are discovery metadata only.  They must
        # never enter auto-info, even if named ``*_info`` or if their action
        # enum advertises ``info``.
        info_candidates = _ordinary_tools(tools)
        info_tools = [
            tool.get('name', '')
            for tool in info_candidates
            if (
                isinstance(tool, dict)
                and (
                    tool.get('name') == 'info'
                    or tool.get('name', '').endswith('_info')
                )
                and _descriptor_tool_access(tool, {}).read_only
            )
        ]
        # Also detect tools with action schema containing 'info'
        action_info_tools = []
        for t in info_candidates:
            if not isinstance(t, dict): continue
            name = t.get('name', '')
            if name in info_tools: continue
            props = (t.get('inputSchema') or {}).get('properties', {})
            action_def = props.get('action', {})
            if 'info' in (action_def.get('enum') or []):
                action_info_tools.append(name)

        req_id = 4

        async def _call_info(tool_name, arguments, rid):
            """Call a single info tool and return parsed info_obj or None."""
            payload = {
                'jsonrpc': '2.0', 'id': rid,
                'method': 'tools/call',
                'params': {'name': tool_name, 'arguments': arguments},
            }
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    data = await resp.json(content_type=None)
                    response_error = _rpc_response_error(
                        data,
                        request_id=rid,
                        http_status=resp.status,
                    )
                    if response_error:
                        return None
                    result = data['result']
                    if _tool_result_error(result):
                        return None
                    content = result['content']
                    for item in content:
                        text = item.get('text', '')
                        if text:
                            try:
                                return json.loads(text)
                            except Exception:
                                return text.strip()
            except Exception as e:
                print(f'[mcp/info] {tool_name} error: {e}')
            return None

        # Build all info calls and execute in parallel
        info_calls = []
        info_call_names = []
        for info_tool in info_tools:
            info_calls.append(_call_info(info_tool, {}, req_id))
            info_call_names.append(info_tool)
            req_id += 1
        for info_tool in action_info_tools:
            info_calls.append(_call_info(info_tool, {'action': 'info'}, req_id))
            info_call_names.append(info_tool)
            req_id += 1

        results = await asyncio.gather(*info_calls, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, (Exception, type(None))):
                continue
            if isinstance(result, dict):
                if not device_type:
                    device_type = result.get('type', '') or result.get('device_type', '')
                for t in result.get('topic_out', []):
                    if t.get('topic') and not any(e.get('topic') == t['topic'] for e in topic_out):
                        topic_out.append(t)
                for t in result.get('topic_in', []):
                    if t.get('topic') and not any(e.get('topic') == t['topic'] for e in topic_in):
                        topic_in.append(t)
                # Back-fill topic paths into the corresponding tool definition
                info_name = info_call_names[idx]
                # Match tool: for "xxx_info" → tool "xxx"; for action-based → same name
                tool_prefix = info_name.removesuffix('_info') if info_name.endswith('_info') else info_name
                for t in tools:
                    if not isinstance(t, dict):
                        continue
                    if t.get('name') != tool_prefix:
                        continue
                    # Merge topic paths from info result into tool's topic_in/topic_out.
                    # Only back-fill when info() returns real (non-empty) topic paths;
                    # idle multiInstance tools report empty strings which must not overwrite
                    # the static format-only schema declarations.
                    # multiInstance tools have per-instance topics tracked on canvas cards;
                    # aggregated info() mixes all instances and must not pollute the static schema.
                    if t.get('multiInstance'):
                        break
                    info_tin  = [ti for ti in result.get('topic_in',  []) if ti.get('topic')]
                    info_tout = [ti for ti in result.get('topic_out', []) if ti.get('topic')]
                    if info_tin:
                        t['topic_in'] = info_tin
                    if info_tout:
                        t['topic_out'] = info_tout
                    break
            elif isinstance(result, str) and not device_type:
                device_type = result

        # Collect topic_out/topic_in declared in tool definitions
        for t in tools:
            if isinstance(t, dict):
                for tp in t.get('topic_out', []):
                    if tp.get('topic') and not any(e.get('topic') == tp['topic'] for e in topic_out):
                        topic_out.append(tp)
                for tp in t.get('topic_in', []):
                    if tp.get('topic') and not any(e.get('topic') == tp['topic'] for e in topic_in):
                        topic_in.append(tp)

        # List resources
        try:
            res_payload = {'jsonrpc': '2.0', 'id': 3, 'method': 'resources/list', 'params': {}}
            async with session.post(
                url,
                json=res_payload,
                headers=headers,
                allow_redirects=False,
            ) as resp:
                data = await resp.json(content_type=None)
                resources_error = _rpc_response_error(
                    data,
                    request_id=3,
                    http_status=resp.status,
                )
                if resources_error:
                    raise ConnectionError(resources_error)
                resources_result = data['result']
                raw_resources = (
                    resources_result.get('resources')
                    if isinstance(resources_result, dict)
                    else None
                )
                if not isinstance(raw_resources, list):
                    raise ConnectionError('invalid resources/list result')
                resources = [
                    resource.get('name')
                    for resource in raw_resources
                    if isinstance(resource, dict)
                ]
        except Exception:
            pass

    return {'tools': tools, 'resources': resources, 'server_name': server_name, 'device_type': device_type,
            'topic_out': topic_out, 'topic_in': topic_in}


def _guess_data_type(tools: list, resources: list, name: str) -> str:
    """Infer data bus type (category/format).
    Returns one of the standard bus types or 'data/json' as fallback.
    See README § Data Bus Types for the full type table.
    """
    tool_names = [t.get('name', '') if isinstance(t, dict) else t for t in (tools or [])]
    descs = [t.get('description', '') if isinstance(t, dict) else '' for t in (tools or [])]
    combined = ' '.join(tool_names + descs + (resources or []) + [name]).lower()

    checks = [
        # ── audio ─────────────────────────────────────────────────────────────
        ('audio/pcm-16k',    ('pcm_16k', 'pcm16k', 'asr', 'microphone', 'mic', 'record_audio', 'capture_audio')),
        ('audio/pcm-48k',    ('pcm_48k', 'pcm48k', 'speaker', 'tts', 'play_audio', 'speak')),
        ('audio/opus',       ('opus',)),
        ('audio/pcm',        ('pcm', 'audio')),
        # ── video ─────────────────────────────────────────────────────────────
        ('video/depth',      ('depth', 'rgbd', 'depth_image')),
        ('video/ir',         ('infrared', 'thermal', '_ir', 'ir_')),
        ('video/stereo',     ('stereo', 'binocular', 'left_image', 'right_image')),
        ('video/mjpeg',      ('mjpeg', 'jpeg_stream')),
        ('video/h265',       ('h265', 'h.265', 'hevc')),
        ('video/h264',       ('h264', 'h.264', 'avc')),
        ('video/yuv',        ('yuv', 'nv12', 'i420')),
        ('video/rgb',        ('rgb', 'raw_frame', 'capture_frame')),
        ('video/mjpeg',      ('video', 'stream', 'camera', 'cam', 'frame')),
        # ── sensor ────────────────────────────────────────────────────────────
        ('sensor/lidar-3d',  ('lidar_3d', 'point_cloud', 'pointcloud', 'velodyne', 'livox')),
        ('sensor/lidar-2d',  ('lidar_2d', 'laser_scan', 'lidar', 'laser', 'rplidar')),
        ('sensor/rtk',       ('rtk', 'gnss')),
        ('sensor/gps',       ('gps', 'nmea', 'geolocation')),
        ('sensor/odometry',  ('odometry', 'odom', 'wheel_encoder', 'encoder')),
        ('sensor/imu',       ('imu', 'gyro', 'accelerometer', 'magnetometer', 'ahrs')),
        ('sensor/force-torque', ('force_torque', 'force_sensor', 'ft_sensor', 'wrench')),
        ('sensor/tactile',   ('tactile', 'touch', 'fingertip')),
        ('sensor/battery',   ('battery', 'voltage', 'current', 'power_state')),
        ('sensor/env',       ('temperature', 'humidity', 'pressure', 'air_quality', 'env')),
        ('sensor/ultrasonic',('ultrasonic', 'sonar', 'proximity')),
        # ── control ───────────────────────────────────────────────────────────
        ('control/gripper',  ('gripper', 'clamp', 'end_effector')),
        ('control/joint-torque', ('torque_control', 'joint_torque')),
        ('control/joint-velocity', ('joint_velocity',)),
        ('control/joint',    ('joint', 'joint_position', 'arm', 'servo', 'actuator')),
        ('control/attitude', ('attitude', 'roll', 'pitch', 'yaw', 'setpoint')),
        ('control/waypoint', ('waypoint', 'navigate_to', 'goto')),
        ('control/velocity', ('velocity', 'cmd_vel', 'wheel', 'drive', 'locomotion', 'motion', 'motor')),
        # ── state ─────────────────────────────────────────────────────────────
        ('state/joint',      ('joint_state', 'joint_status')),
        ('state/pose',       ('pose', 'localization', 'amcl', 'robot_pose')),
        ('state/velocity',   ('state_velocity', 'body_velocity')),
        ('state/power',      ('power_status', 'motor_temp', 'system_health')),
        ('state/error',      ('error_code', 'fault', 'alarm', 'estop')),
        # ── text / data ───────────────────────────────────────────────────────
        ('text/asr',         ('asr_result', 'transcript', 'speech_text')),
        ('text/plain',       ('text', 'chat', 'message', 'keyboard')),
        ('data/ros-topic',   ('ros_topic', 'rostopic', 'ros2')),
        ('data/canbus',      ('canbus', 'can_frame', 'can_bus')),
        ('data/modbus',      ('modbus', 'holding_register', 'coil')),
    ]
    for data_type, keywords in checks:
        if any(k in combined for k in keywords):
            return data_type

    return 'data/json'


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get('')
async def mcp_list():
    items = [
        {
            'id':          m.get('id', ''),
            'name':        m.get('name', ''),
            'transport':   m.get('transport', 'http'),
            'url':         m.get('url', ''),
            'render_hint': m.get('render_hint', ''),
            'server_name': m.get('server_name', ''),
            'tools':       _ordinary_tools(m.get('tools', [])),
            'resources':   m.get('resources', []),
            'topic_out':   m.get('topic_out', []),
            'topic_in':    m.get('topic_in',  []),
            'category':    m.get('category', ''),
            'reported_robot_id': m.get('reported_robot_id', ''),
            'authority_domain': m.get('authority_domain', m.get('id', '')),
            'pending_authority_domain': m.get('pending_authority_domain', ''),
            'authority_binding_error': m.get('authority_binding_error', ''),
            'restart_required': 'pending_authority_domain' in m,
            'trust_state': (
                'internal' if m.get('transport') == 'internal'
                else _effective_trust_state(m)
            ),
            'depends_on':  m.get('depends_on', ''),
            'ws_path':     ('/ws/bus' + (m.get('topic_out') or [{}])[0].get('topic', '')) if m.get('topic_out') else '',
            'online':      None,
        }
        for m in _get_mcp_list()
    ]
    return {'code': 200, 'data': items}


_MCP_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$')
_RESERVED_INTERNAL_MCP_IDS = frozenset({'agentcore', 'channel'})


@router.post('')
async def mcp_add(req: MCPAddRequest, request: fastapi.Request):
    """Register or heartbeat an MCP service.

    Legacy services remain discoverable, but only callers presenting the
    credential selected for their exact stable id receive
    ``trust_state=trusted``.  An untrusted heartbeat is never allowed to mutate
    a trusted record, and a dedicated credential cannot claim another id.
    """
    requested_id = req.id.strip()
    presented_token = auth.extract_driver_token(request)
    dedicated_identity = auth.driver_token_identity(presented_token)
    if (
        auth.is_known_driver_token(presented_token)
        and not _MCP_ID_RE.fullmatch(requested_id)
    ):
        raise fastapi.HTTPException(
            status_code=422,
            detail='trusted Driver registration requires a valid stable MCP id',
        )
    if dedicated_identity is not None and dedicated_identity != requested_id:
        return fastapi.responses.JSONResponse(
            status_code=403,
            content={
                'code': 403,
                'message': 'Driver credential is bound to a different stable MCP id',
                'data': {'id': requested_id, 'trust_state': 'quarantined'},
            },
        )
    if (
        auth.has_dedicated_driver_token(requested_id)
        and not auth.verify_driver_token(presented_token, requested_id)
    ):
        return fastapi.responses.JSONResponse(
            status_code=401,
            content={
                'code': 401,
                'message': 'Dedicated Driver credential required for this stable MCP id',
                'data': {'id': requested_id, 'trust_state': 'quarantined'},
            },
        )
    trusted = auth.verify_driver_token(presented_token, requested_id)
    credential_binding = (
        auth.driver_credential_binding(requested_id) if trusted else ''
    )
    trust_state = (
        'trusted' if trusted
        else 'quarantined' if auth.is_driver_auth_enforced()
        else 'untrusted'
    )

    requested_robot_id = req.robot_id.strip()
    if trusted and not requested_id:
        raise fastapi.HTTPException(
            status_code=422,
            detail='trusted driver registration requires a stable MCP id',
        )
    if requested_id and trusted and not _MCP_ID_RE.fullmatch(requested_id):
        raise fastapi.HTTPException(status_code=422, detail='invalid MCP id')
    if requested_robot_id and trusted and not _MCP_ID_RE.fullmatch(requested_robot_id):
        raise fastapi.HTTPException(status_code=422, detail='invalid robot id')

    async with _mcp_write_lock:
        mcps = _get_mcp_list()
        internal_collision = next(
            (
                m for m in mcps
                if m.get('transport') == 'internal'
                and (
                    (requested_id and m.get('id') == requested_id)
                    or (req.url and m.get('url') == req.url)
                    or (req.name and m.get('name') == req.name)
                    or (req.name and m.get('server_name') == req.name)
                )
            ),
            None,
        )
        if requested_id in _RESERVED_INTERNAL_MCP_IDS or internal_collision:
            conflict_id = (
                internal_collision.get('id', '')
                if internal_collision else requested_id
            )
            return fastapi.responses.JSONResponse(
                status_code=409,
                content={
                    'code': 409,
                    'message': 'MCP identity is reserved for an internal service',
                    'data': {'id': conflict_id, 'trust_state': 'internal'},
                },
            )

        # A trusted service's stable configured id is its authority key.  Name,
        # URL, and initialize.serverInfo.name are descriptive and may be shared
        # by multiple robots, so they must never merge two trusted identities.
        if trusted and requested_id:
            existing = next((m for m in mcps if m.get('id') == requested_id), None)
        else:
            # Preserve legacy upsert behavior only for callers without a stable
            # trusted identity.  Any collision with a trusted record is rejected
            # below before the record can be mutated.
            existing = next(
                (m for m in mcps if (m.get('url') == req.url and req.url)
                 or (m.get('name') == req.name and req.name)
                 or (m.get('server_name') and m.get('server_name') == req.name)),
                None,
            )

        if existing and existing.get('trust_state') == 'trusted' and not trusted:
            return fastapi.responses.JSONResponse(
                status_code=403,
                content={
                    'code': 403,
                    'message': 'Untrusted registration cannot replace a trusted service',
                    'data': {'id': existing.get('id', ''), 'trust_state': 'trusted'},
                },
            )

        if (
            trusted
            and existing
            and existing.get('trust_state') == 'trusted'
            and existing.get('url')
            and req.url != existing.get('url')
        ):
            return fastapi.responses.JSONResponse(
                status_code=409,
                content={
                    'code': 409,
                    'message': 'Trusted Driver target URL is owner-controlled',
                    'data': {'id': existing.get('id', '')},
                },
            )

        expected_reported_robot_id = (
            existing.get('authority_domain')
            or existing.get('reported_robot_id')
            if existing else ''
        )
        if trusted and expected_reported_robot_id and not requested_robot_id:
            return fastapi.responses.JSONResponse(
                status_code=409,
                content={
                    'code': 409,
                    'message': 'Trusted Driver registration must echo its robot identity',
                    'data': {'id': existing.get('id', '')},
                },
            )

        existing_robot_id = existing.get('reported_robot_id', '') if existing else ''
        if (
            trusted
            and requested_robot_id
            and existing_robot_id
            and existing_robot_id != requested_robot_id
        ):
            return fastapi.responses.JSONResponse(
                status_code=409,
                content={
                    'code': 409,
                    'message': 'Trusted Driver-reported robot identity is immutable',
                    'data': {
                        'id': existing.get('id', ''),
                        'robot_id': existing_robot_id,
                    },
                },
            )
        if (
            trusted
            and requested_robot_id
            and existing
            and existing.get('authority_domain')
            and existing.get('authority_domain') != requested_robot_id
        ):
            return fastapi.responses.JSONResponse(
                status_code=409,
                content={
                    'code': 409,
                    'message': 'Driver-reported robot identity conflicts with owner binding',
                    'data': {
                        'id': existing.get('id', ''),
                        'authority_domain': existing.get('authority_domain'),
                    },
                },
            )

        if existing:
            prospective = dict(existing)
            prospective['name'] = req.name
            prospective['transport'] = req.transport
            prospective['url'] = req.url
            prospective['render_hint'] = req.render_hint
            if trusted:
                prospective['trust_state'] = 'trusted'
                if credential_binding:
                    prospective['credential_binding'] = credential_binding
                else:
                    prospective.pop('credential_binding', None)
            else:
                prospective['trust_state'] = trust_state
            if req.category:
                prospective['category'] = req.category
            if trusted and requested_robot_id:
                prospective['reported_robot_id'] = requested_robot_id
            proposed_mcps = [
                prospective if target is existing else target
                for target in mcps
            ]
            mutation_error = await _project_target_mutation_error(
                mcps,
                proposed_mcps,
            )
            if mutation_error is not None:
                return _project_target_locked_response(mutation_error)
            if trusted and not credential_binding:
                existing.pop('credential_binding', None)
            existing.update(prospective)
            _save_mcp_list(mcps)
            mcp_id = existing['id']
        else:
            mcp_id = requested_id if trusted and requested_id else f'mcp-{uuid.uuid4().hex[:12]}'
            record = {
                'id':          mcp_id,
                'name':        req.name,
                'transport':   req.transport,
                'url':         req.url,
                'render_hint': req.render_hint,
                'category':    req.category,
                'trust_state': trust_state,
            }
            if credential_binding:
                record['credential_binding'] = credential_binding
            if trusted and requested_robot_id:
                record['reported_robot_id'] = requested_robot_id
            mutation_error = await _project_target_mutation_error(
                mcps,
                [*mcps, record],
            )
            if mutation_error is not None:
                return _project_target_locked_response(mutation_error)
            mcps.append(record)
            _save_mcp_list(mcps)
    if trust_state == 'quarantined':
        # Discovery-only record: it must not contribute LLM schemas, restored
        # configuration, or runtime authority.
        mcp_client.registry.pop(mcp_id, None)
    else:
        asyncio.create_task(_do_ping(mcp_id))
    return {'code': 200, 'data': {'id': mcp_id, 'trust_state': trust_state}}


@router.delete('/{mcp_id}')
async def mcp_delete(mcp_id: str):
    async with _mcp_write_lock:
        current = _get_mcp_list()
        mcps = [m for m in current if m.get('id') != mcp_id]
        mutation_error = await _project_target_mutation_error(current, mcps)
        if mutation_error is not None:
            return _project_target_locked_response(mutation_error)
        dependants = [
            m.get('id', '') for m in current
            if (
                m.get('authority_domain') == mcp_id
                or m.get('pending_authority_domain') == mcp_id
            )
            and m.get('id') != mcp_id
        ]
        if dependants:
            return fastapi.responses.JSONResponse(
                status_code=409,
                content={
                    'code': 409,
                    'message': 'MCP is an active authority root',
                    'data': {'dependants': dependants},
                },
            )
        _save_mcp_list(mcps)
    mcp_client.registry.pop(mcp_id, None)
    return {'code': 200}


def _authority_binding_problem(target: dict, root: dict, root_mcp_id: str) -> str | None:
    if (
        _effective_trust_state(target) != 'trusted'
        or _effective_trust_state(root) != 'trusted'
        or target.get('transport') != 'http'
        or root.get('transport') != 'http'
        or target.get('category') != 'driver'
        or root.get('category') != 'driver'
    ):
        return 'binding_requires_trusted_http_drivers'
    if root.get('authority_domain') not in (None, '', root_mcp_id):
        return 'authority_root_is_alias'
    session_tools = [
        tool for tool in target.get('tools', [])
        if isinstance(tool, dict) and tool.get('name') == 'teleop_session'
    ]
    if len(session_tools) != 1 or not isinstance(session_tools[0].get('x-teleop'), dict):
        return 'source_missing_teleop_descriptor'
    ordinary_actuator = any(
        isinstance(tool, dict)
        and tool.get('type') == 'actuator'
        and not _is_reserved_teleop_tool(tool)
        for tool in root.get('tools', [])
    )
    if not ordinary_actuator:
        return 'root_missing_ordinary_actuator'
    reported_robot_id = target.get('reported_robot_id')
    descriptor_robot_id = session_tools[0]['x-teleop'].get('robot_id')
    if reported_robot_id and reported_robot_id != root_mcp_id:
        return 'reported_robot_id_mismatch'
    if descriptor_robot_id != root_mcp_id:
        return 'descriptor_robot_id_mismatch'
    return None


def _require_binding_owner(request: fastapi.Request) -> auth.Principal:
    if not auth.is_enabled():
        raise fastapi.HTTPException(
            status_code=503,
            detail='Owner authentication must be configured for authority binding',
        )
    return auth.require_role(request, 'owner')


async def activate_pending_authority_bindings() -> None:
    """Freeze validated owner bindings before the application accepts traffic."""

    audit_events = []
    async with _mcp_write_lock:
        try:
            persisted_guards = await asyncio.to_thread(authority_guard.list_guards)
        except Exception:  # noqa: BLE001 -- leave every pending binding untouched
            print('[teleop] authority guard store unavailable; pending bindings deferred')
            return
        guard_locked_ids = {
            guarded_id
            for guard in persisted_guards
            for guarded_id in (guard.driver_id, guard.robot_id)
        }
        mcps = _get_mcp_list()
        changed = False
        for target in mcps:
            if not isinstance(target, dict):
                continue
            mcp_id = target.get('id')
            if not isinstance(mcp_id, str):
                continue
            pending = target.get('pending_authority_domain')
            active = target.get('authority_domain')
            has_pending = 'pending_authority_domain' in target
            desired = pending if isinstance(pending, str) else active
            if not isinstance(desired, str) or not desired:
                continue
            old_root = active or mcp_id
            if has_pending and {mcp_id, old_root, desired} & guard_locked_ids:
                audit_events.append({
                    'mcp_id': mcp_id,
                    'old_root': old_root,
                    'new_root': desired,
                    'decision': 'deferred',
                    'reason': 'persistent_authority_guard_requires_stable_target',
                })
                continue
            changed = True
            if desired == mcp_id:
                target.pop('authority_domain', None)
                target.pop('pending_authority_domain', None)
                target.pop('authority_binding_error', None)
                target.pop('authority_binding_required', None)
                if has_pending:
                    audit_events.append({
                        'mcp_id': mcp_id,
                        'old_root': active or mcp_id,
                        'new_root': mcp_id,
                        'decision': 'activated',
                        'reason': 'owner_unbind_activated',
                    })
                continue
            roots = [
                root for root in mcps
                if isinstance(root, dict) and root.get('id') == desired
            ]
            problem = (
                'authority_root_not_unique'
                if len(roots) != 1
                else _authority_binding_problem(target, roots[0], desired)
            )
            if problem is not None:
                target.pop('authority_domain', None)
                target.pop('pending_authority_domain', None)
                target['authority_binding_required'] = True
                target['authority_binding_error'] = problem
                mcp_client.registry.pop(mcp_id, None)
                if has_pending:
                    audit_events.append({
                        'mcp_id': mcp_id,
                        'old_root': active or mcp_id,
                        'new_root': desired,
                        'decision': 'rejected',
                        'reason': problem,
                    })
                continue
            target['authority_domain'] = desired
            target['authority_binding_required'] = True
            target.pop('pending_authority_domain', None)
            target.pop('authority_binding_error', None)
            runtime = mcp_client.registry.get(mcp_id)
            if runtime is not None:
                runtime['authority_domain'] = desired
            if has_pending:
                audit_events.append({
                    'mcp_id': mcp_id,
                    'old_root': active or mcp_id,
                    'new_root': desired,
                    'decision': 'activated',
                    'reason': 'owner_binding_activated',
                })
        if changed:
            _save_mcp_list(mcps)
    for event in audit_events:
        await audit.emit(
            'teleop.authority_binding',
            robot_id=event['new_root'],
            principal_id='core:startup',
            source='core',
            decision=event['decision'],
            reason=event['reason'],
            details={
                'mcp_id': event['mcp_id'],
                'old_root': event['old_root'],
                'new_root': event['new_root'],
            },
        )


@router.put('/{mcp_id}/authority-domain')
async def mcp_bind_authority_domain(
    mcp_id: str,
    req: MCPAuthorityBindingRequest,
    request: fastapi.Request,
):
    """Stage an owner-approved alias for activation on the next Core start."""

    principal = _require_binding_owner(request)
    root_mcp_id = req.root_mcp_id.strip()
    if not _MCP_ID_RE.fullmatch(mcp_id) or not _MCP_ID_RE.fullmatch(root_mcp_id):
        raise fastapi.HTTPException(status_code=422, detail='invalid MCP id')
    if root_mcp_id == mcp_id:
        raise fastapi.HTTPException(
            status_code=422,
            detail='Use DELETE to remove an authority alias',
        )

    async with _mcp_write_lock:
        mcps = _get_mcp_list()
        targets = [m for m in mcps if m.get('id') == mcp_id]
        roots = [m for m in mcps if m.get('id') == root_mcp_id]
        if len(targets) != 1 or len(roots) != 1:
            raise fastapi.HTTPException(status_code=404, detail='MCP not found')
        problem = _authority_binding_problem(targets[0], roots[0], root_mcp_id)
        if problem is not None:
            await audit.emit(
                'teleop.authority_binding',
                robot_id=root_mcp_id,
                principal_id=principal.id,
                source='api',
                decision='rejected',
                reason=problem,
                details={'mcp_id': mcp_id, 'new_root': root_mcp_id},
            )
            raise fastapi.HTTPException(
                status_code=409,
                detail={'code': 'authority_binding_invalid', 'reason': problem},
            )
        old_root = targets[0].get('authority_domain') or mcp_id
        prospective = dict(targets[0])
        prospective['pending_authority_domain'] = root_mcp_id
        proposed_mcps = [
            prospective if target is targets[0] else target
            for target in mcps
        ]
        mutation_error = await _project_target_mutation_error(mcps, proposed_mcps)
        if mutation_error is not None:
            return _project_target_locked_response(mutation_error)
        targets[0]['pending_authority_domain'] = root_mcp_id
        _save_mcp_list(mcps)

    await audit.emit(
        'teleop.authority_binding',
        robot_id=root_mcp_id,
        principal_id=principal.id,
        source='api',
        decision='staged',
        reason='owner_binding_staged',
        details={
            'mcp_id': mcp_id,
            'old_root': old_root,
            'new_root': root_mcp_id,
            'restart_required': True,
        },
    )

    return fastapi.responses.JSONResponse(
        status_code=202,
        content={
            'code': 202,
            'data': {
                'mcp_id': mcp_id,
                'authority_domain': root_mcp_id,
                'restart_required': True,
            },
        },
    )


@router.delete('/{mcp_id}/authority-domain')
async def mcp_unbind_authority_domain(
    mcp_id: str,
    request: fastapi.Request,
):
    """Stage removal of an authority alias for the next Core start."""

    principal = _require_binding_owner(request)
    async with _mcp_write_lock:
        mcps = _get_mcp_list()
        targets = [m for m in mcps if m.get('id') == mcp_id]
        if len(targets) != 1:
            raise fastapi.HTTPException(status_code=404, detail='MCP not found')
        old_root = targets[0].get('authority_domain') or mcp_id
        prospective = dict(targets[0])
        prospective['pending_authority_domain'] = mcp_id
        proposed_mcps = [
            prospective if target is targets[0] else target
            for target in mcps
        ]
        mutation_error = await _project_target_mutation_error(mcps, proposed_mcps)
        if mutation_error is not None:
            return _project_target_locked_response(mutation_error)
        targets[0]['pending_authority_domain'] = mcp_id
        _save_mcp_list(mcps)
    await audit.emit(
        'teleop.authority_binding',
        robot_id=old_root,
        principal_id=principal.id,
        source='api',
        decision='staged',
        reason='owner_unbind_staged',
        details={
            'mcp_id': mcp_id,
            'old_root': old_root,
            'new_root': mcp_id,
            'restart_required': True,
        },
    )
    return fastapi.responses.JSONResponse(
        status_code=202,
        content={
            'code': 202,
            'data': {
                'mcp_id': mcp_id,
                'authority_domain': mcp_id,
                'restart_required': True,
            },
        },
    )


async def _restore_saved_configs(
    mcp_id: str,
    url: str,
    tools: list,
    *,
    trusted: bool = False,
    authority_domain: str | None = None,
    target_fingerprint: tuple | None = None,
    ping_generation: int | None = None,
) -> None:
    """Re-send saved tool configs to a device that just came online.

    Only sends shared (non-instance) configs for tools that have configSchema.
    Called once when a device transitions from offline → online.
    """
    headers = {'Content-Type': 'application/json'}
    if trusted:
        driver_headers = auth.driver_request_headers(mcp_id)
        if not driver_headers:
            return
        headers.update(driver_headers)
    timeout = aiohttp.ClientTimeout(total=5)
    sent = []
    from teleop.service import coordinator

    def current_target() -> dict | None:
        records = _get_mcp_list()
        targets = [record for record in records if record.get('id') == mcp_id]
        if len(targets) != 1:
            return None
        current = targets[0]
        if (
            (
                ping_generation is not None
                and _ping_generations.get(mcp_id) != ping_generation
            )
            or current.get('url', '') != url
            or current.get('transport', 'http') != 'http'
            or not _runtime_registration_allowed(current)
            or (
                target_fingerprint is not None
                and _ping_target_fingerprint(current) != target_fingerprint
            )
        ):
            return None
        return current

    def current_configurable_tool(target: dict, tool_name: str) -> dict | None:
        matches = [
            tool
            for tool in (target.get('tools') or [])
            if isinstance(tool, dict) and tool.get('name') == tool_name
        ]
        if len(matches) != 1:
            return None
        tool = matches[0]
        config_schema = tool.get('configSchema')
        if _is_reserved_teleop_tool(tool) or not (
            isinstance(config_schema, dict) and config_schema
        ):
            return None
        return tool

    initial_target = current_target()
    if initial_target is None:
        return
    if target_fingerprint is None:
        target_fingerprint = _ping_target_fingerprint(initial_target)
    if authority_domain is None:
        records = _get_mcp_list()
        try:
            authority_domain = authority_domain_for_target(
                mcp_id,
                initial_target,
                targets=records,
            )
        except InvalidAuthorityBinding:
            return

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            seen_tool_names = set()
            for discovered_tool in tools:
                if not isinstance(discovered_tool, dict):
                    continue
                if _is_reserved_teleop_tool(discovered_tool):
                    continue
                tool_name = discovered_tool.get('name', '')
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                if tool_name in seen_tool_names:
                    continue
                seen_tool_names.add(tool_name)
                target_snapshot = current_target()
                if target_snapshot is None:
                    return
                tool = current_configurable_tool(target_snapshot, tool_name)
                if tool is None:
                    continue
                saved_cfg = _safe_saved_config(
                    config.main.get(f'tool_config:{mcp_id}:{tool_name}', None)
                )
                if not saved_cfg:
                    continue
                cfg_payload = {
                    'jsonrpc': '2.0', 'id': 99,
                    'method': 'tools/call',
                    'params': {
                        'name': tool_name,
                        'arguments': {**saved_cfg, 'action': 'config'},
                    },
                }
                access = _descriptor_tool_access(tool, cfg_payload['params']['arguments'])
                try:
                    async with coordinator.command_broker.ordinary_command(
                        authority_domain,
                        read_only=access.read_only,
                        source='config_restore',
                        tool=tool_name,
                        action='config',
                        tool_verified=True,
                        action_verified=_descriptor_action_declared(
                            tool,
                            cfg_payload['params']['arguments'],
                        ),
                    ):
                        latest_target = current_target()
                        if latest_target is None:
                            return
                        latest_tool = current_configurable_tool(
                            latest_target,
                            tool_name,
                        )
                        if latest_tool is None:
                            continue
                        try:
                            latest_authority = authority_domain_for_target(
                                mcp_id,
                                latest_target,
                                targets=_get_mcp_list(),
                            )
                        except InvalidAuthorityBinding:
                            return
                        if latest_authority != authority_domain:
                            return
                        latest_access = _descriptor_tool_access(
                            latest_tool,
                            cfg_payload['params']['arguments'],
                        )
                        if latest_access.read_only != access.read_only:
                            continue
                        await session.post(
                            url,
                            json=cfg_payload,
                            headers=headers,
                            allow_redirects=False,
                        )
                except TeleopCommandBlocked:
                    continue
                sent.append(tool_name)
    except Exception as e:
        print(f'[mcp/config-restore] {mcp_id} error: {e}')
        return

    if sent:
        print(f'[mcp/config-restore] {mcp_id}: restored config for {sent}')


async def _do_ping(mcp_id: str) -> dict:
    """Core ping logic — fetch capabilities, persist, notify inspector.
    Returns the same dict as the ping endpoint's data field.
    Raises HTTPException(404) if mcp_id not found."""
    mcps = _get_mcp_list()
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    if not target:
        raise fastapi.HTTPException(status_code=404, detail='MCP not found')

    if not _runtime_registration_allowed(target):
        mcp_client.registry.pop(mcp_id, None)
        return {
            'online': False,
            'error': 'registration quarantined: valid Driver credential required',
            'tools': [],
            'resources': [],
            'render_hint': target.get('render_hint', ''),
            'server_name': target.get('server_name', ''),
            'topic_out': [],
            'topic_in': [],
        }

    try:
        authority_domain = authority_domain_for_target(
            mcp_id,
            target,
            targets=mcps,
        )
    except InvalidAuthorityBinding:
        mcp_client.registry.pop(mcp_id, None)
        return {
            'online': False,
            'error': 'authority_binding_invalid',
            'tools': [],
            'resources': [],
        }

    transport = target.get('transport', 'http')
    url       = target.get('url', '')

    if transport != 'http' or not url:
        is_internal = transport == 'internal'
        # Register topics for internal MCPs (so inspection/monitoring works)
        if is_internal:
            topics = target.get('topic_out', []) + target.get('topic_in', [])
            if topics:
                asyncio.create_task(_notify_inspector(mcp_id, topics))
        return {
            'online':      is_internal and target.get('online', False),
            'tools':       target.get('tools', []),
            'resources':   target.get('resources', []),
            'render_hint': target.get('render_hint', ''),
            'server_name': target.get('server_name', ''),
            'topic_out':   target.get('topic_out', []),
            'topic_in':    target.get('topic_in', []),
        }

    target_fingerprint = _ping_target_fingerprint(target)
    ping_generation = _ping_generations.get(mcp_id, 0) + 1
    _ping_generations[mcp_id] = ping_generation

    # 记录 ping 前的 online 状态，用于判断是否需要重新下发 config
    was_online = mcp_client.registry.get(mcp_id, {}).get('online', False)

    try:
        caps = await _ping_mcp_http(
            url,
            trusted=_effective_trust_state(target) == 'trusted',
            driver_id=mcp_id,
        )
    except Exception as e:
        # Legacy duplicate cleanup is intentionally limited to two untrusted
        # identities.  A descriptive server_name must never merge or delete a
        # stable trusted identity, even while it is offline.
        async with _mcp_write_lock:
            mcps = _get_mcp_list()
            this_entry = next((m for m in mcps if m.get('id') == mcp_id), None)
            if (
                _ping_generations.get(mcp_id) != ping_generation
                or this_entry is None
                or _ping_target_fingerprint(this_entry) != target_fingerprint
            ):
                return _stale_ping_result(mcp_id)
            if mcp_id in mcp_client.registry:
                mcp_client.registry[mcp_id]['online'] = False
            if (
                this_entry
                and this_entry.get('server_name')
                and _effective_trust_state(this_entry) != 'trusted'
            ):
                dup = next(
                    (m for m in mcps if m.get('server_name') == this_entry['server_name'] and m.get('id') != mcp_id),
                    None,
                )
                if dup:
                    proposed_mcps = [
                        m for m in mcps if m.get('id') != mcp_id
                    ]
                    mutation_error = await _project_target_mutation_error(
                        mcps,
                        proposed_mcps,
                    )
                    if mutation_error is None:
                        _save_mcp_list(proposed_mcps)
                        print(
                            f'[mcp/ping] dedup: removed offline {mcp_id} '
                            '(same legacy server_name)'
                        )
        return {'online': False, 'error': str(e), 'tools': [], 'resources': []}

    # render_hint priority:
    # 1. topic_out[0].format (most authoritative — comes from driver's info())
    # 2. device self-reported type field
    # 3. heuristic from tool names
    topic_fmt = (caps.get('topic_out') or [{}])[0].get('format', '')
    render_hint = (
        topic_fmt
        or caps.get('device_type')
        or _guess_data_type(caps['tools'], caps['resources'], target.get('name', ''))
    )

    # Resolve empty topics from depends_on relationship
    topic_in  = [dict(t) for t in caps.get('topic_in',  [])]
    topic_out = [dict(t) for t in caps.get('topic_out', [])]

    upstream_topic = ''
    depends_on = target.get('depends_on', '')
    if depends_on:
        upstream = next((m for m in mcps if m.get('id') == depends_on), None)
        upstream_topic = ((upstream or {}).get('topic_out') or [{}])[0].get('topic', '')

    # Fill empty topic_in from upstream (depends_on relationship)
    if upstream_topic:
        for t in topic_in:
            if not t.get('topic'):
                t['topic'] = upstream_topic

    # Prepare change logging only after the target snapshot is revalidated.
    current_tool_names = [t.get('name', '') if isinstance(t, dict) else t for t in caps['tools']]

    # Persist on every successful ping; server_name only set once (not overwritten).
    # server_name is never an authority key for trusted services.  Legacy
    # untrusted registrations retain historical same-name de-duplication.
    async with _mcp_write_lock:
        mcps = _get_mcp_list()  # re-read under lock to avoid race condition
        new_server_name = caps.get('server_name', '')
        current = next((m for m in mcps if m.get('id') == mcp_id), None)
        if (
            _ping_generations.get(mcp_id) != ping_generation
            or current is None
            or _ping_target_fingerprint(current) != target_fingerprint
        ):
            return _stale_ping_result(mcp_id)
        published_target = dict(current)

        proposed_capabilities = []
        for record in mcps:
            if record is current:
                updated = dict(record)
                updated['tools'] = caps['tools']
                updated.pop('capability_refresh_required', None)
                proposed_capabilities.append(updated)
            else:
                proposed_capabilities.append(record)
        mutation_error = await _project_target_mutation_error(
            mcps,
            proposed_capabilities,
        )
        if mutation_error is not None:
            return _project_target_locked_ping_result(
                mcp_id,
                current,
                mutation_error,
            )

        internal_collision = next(
            (
                m for m in mcps
                if m.get('transport') == 'internal'
                and m.get('server_name') == new_server_name
                and m.get('id') != mcp_id
            ),
            None,
        ) if new_server_name else None
        if current and current.get('transport') != 'internal' and internal_collision:
            proposed_mcps = [m for m in mcps if m.get('id') != mcp_id]
            mutation_error = await _project_target_mutation_error(
                mcps,
                proposed_mcps,
            )
            if mutation_error is not None:
                return _project_target_locked_ping_result(
                    mcp_id,
                    current,
                    mutation_error,
                )
            _save_mcp_list(proposed_mcps)
            mcp_client.registry.pop(mcp_id, None)
            print(f'[mcp/ping] rejected external duplicate {mcp_id} of internal {internal_collision["id"]}')
            return {
                'online': False,
                'error': 'external duplicate of internal service',
                'tools': [],
                'resources': [],
                'render_hint': '',
                'server_name': new_server_name,
                'topic_out': [],
                'topic_in': [],
            }

        if new_server_name and current and _effective_trust_state(current) != 'trusted':
            same_name = [
                m for m in mcps
                if m.get('server_name') == new_server_name and m.get('id') != mcp_id
            ]
            trusted_collision = next(
                (m for m in same_name if _effective_trust_state(m) == 'trusted'),
                None,
            )
            if trusted_collision:
                proposed_mcps = [m for m in mcps if m.get('id') != mcp_id]
                mutation_error = await _project_target_mutation_error(
                    mcps,
                    proposed_mcps,
                )
                if mutation_error is not None:
                    return _project_target_locked_ping_result(
                        mcp_id,
                        current,
                        mutation_error,
                    )
                # A legacy endpoint spoofing a trusted server name loses without
                # touching that trusted record or its runtime identity.
                _save_mcp_list(proposed_mcps)
                mcp_client.registry.pop(mcp_id, None)
                print(f'[mcp/ping] rejected untrusted duplicate {mcp_id} of trusted {trusted_collision["id"]}')
                return {
                    'online': False,
                    'error': 'untrusted duplicate of trusted service',
                    'tools': [],
                    'resources': [],
                    'render_hint': '',
                    'server_name': new_server_name,
                    'topic_out': [],
                    'topic_in': [],
                }

            legacy_duplicate = next(
                (m for m in same_name if _effective_trust_state(m) != 'trusted'),
                None,
            )
            if legacy_duplicate:
                proposed_mcps = []
                for record in mcps:
                    if record is current:
                        continue
                    if record is legacy_duplicate:
                        updated_duplicate = dict(record)
                        updated_duplicate['url'] = current.get(
                            'url',
                            record.get('url', ''),
                        )
                        proposed_mcps.append(updated_duplicate)
                    else:
                        proposed_mcps.append(record)
                mutation_error = await _project_target_mutation_error(
                    mcps,
                    proposed_mcps,
                )
                if mutation_error is not None:
                    return _project_target_locked_ping_result(
                        mcp_id,
                        current,
                        mutation_error,
                    )
                print(f'[mcp/ping] dedup: removed {mcp_id}, merged into {legacy_duplicate["id"]} (server_name={new_server_name})')
                _save_mcp_list(proposed_mcps)
                return {'online': True, 'tools': caps['tools'], 'resources': caps['resources'],
                        'render_hint': render_hint, 'server_name': new_server_name,
                        'topic_out': topic_out, 'topic_in': topic_in}

        for m in mcps:
            if m.get('id') == mcp_id:
                m['render_hint'] = render_hint
                m['tools']       = caps['tools']
                m['resources']   = caps['resources']
                m['topic_out']   = topic_out
                m['topic_in']    = topic_in
                m.pop('capability_refresh_required', None)
                if not m.get('server_name'):
                    m['server_name'] = new_server_name
                published_target = dict(m)
                break
        _save_mcp_list(mcps)

    prev_tool_names = _last_tool_names.get(mcp_id)
    if prev_tool_names != current_tool_names:
        _last_tool_names[mcp_id] = current_tool_names
        print(f'[mcp/ping] {mcp_id}: server={caps.get("server_name", "?")} tools={current_tool_names}')

    # 同步更新内存中的 mcp_client.registry（LLM 决策依赖此数据）
    schemas = {}
    tool_meta_map = {}
    split_map = {}
    tool_groups = {}
    ordinary_tools = _ordinary_tools(caps['tools'])
    for tool in ordinary_tools:
        tool_schemas = mcp_client._to_openai_schema(mcp_id, tool)

        if len(tool_schemas) == 1:
            schema = tool_schemas[0]
            schemas[schema['name']] = schema
            raw_input_schema = tool.get('inputSchema') or {}
            action_enum = raw_input_schema.get('properties', {}).get('action', {}).get('enum')
            tool_meta_map[schema['name']] = {
                'type': tool.get('type'),
                'action_enum': action_enum,
                'has_config_schema': bool(tool.get('configSchema')),
                'completion': raw_input_schema.get('x-completion'),
                'annotations': tool.get('annotations') or {},
                'x_teleop': tool.get('x-teleop'),
            }
        else:
            group = []
            for schema in tool_schemas:
                schemas[schema['name']] = schema
                tool_meta_map[schema['name']] = {
                    'type': tool.get('type'),
                    'action_enum': None,
                    'has_config_schema': bool(tool.get('configSchema')),
                    'completion': (tool.get('inputSchema') or {}).get('x-completion'),
                    'annotations': tool.get('annotations') or {},
                    'x_teleop': tool.get('x-teleop'),
                }
                action_name = schema['name'].split('__')[-1]
                split_map[schema['name']] = {
                    'tool': tool.get('name', ''),
                    'action': action_name,
                }
                group.append(schema['name'])
            tool_name = tool.get('name', '')
            if tool_name:
                tool_groups[tool_name] = group

    mcp_client.registry[mcp_id] = {
        'name':        published_target.get('name', mcp_id),
        'url':         url,
        'online':      True,
        'tools':       [t.get('name', '') if isinstance(t, dict) else t for t in ordinary_tools],
        'render_hint': render_hint,
        'schemas':     schemas,
        'tool_meta':   tool_meta_map,
        'split_map':   split_map,
        'tool_groups': tool_groups,
        'trusted':     _effective_trust_state(published_target) == 'trusted',
        'authority_domain': authority_domain,
        'teleop_fingerprint': mcp_client.teleop_tools_fingerprint(caps['tools']),
    }

    # Register system hooks from x-hooks declarations
    import hooks
    for tool in caps['tools']:
        x_hooks = (tool.get('inputSchema') or {}).get('x-hooks')
        if x_hooks and isinstance(x_hooks, dict):
            hooks.register(mcp_id, tool.get('name', ''), x_hooks)

    # Notify inspection module about all topics from this device
    asyncio.create_task(_notify_inspector(mcp_id, topic_out + topic_in))

    # Auto-restore saved configs when device comes online (first ping or after offline)
    if not was_online:
        asyncio.create_task(_restore_saved_configs(
            mcp_id,
            url,
            ordinary_tools,
            trusted=_effective_trust_state(published_target) == 'trusted',
            authority_domain=authority_domain,
            target_fingerprint=_ping_target_fingerprint(published_target),
            ping_generation=ping_generation,
        ))

    ws_path = ('/ws/bus' + topic_out[0].get('topic', '')) if topic_out else ''
    return {
        'online':      True,
        'tools':       caps['tools'],
        'resources':   caps['resources'],
        'render_hint': render_hint,
        'server_name': caps.get('server_name', ''),
        'topic_out':   topic_out,
        'topic_in':    topic_in,
        'ws_path':     ws_path,
    }


@router.post('/{mcp_id}/ping')
async def mcp_ping(mcp_id: str):
    data = await _do_ping(mcp_id)
    return {'code': 200, 'data': data}


@router.get('/{mcp_id}/tools')
async def mcp_get_tools(mcp_id: str):
    """Return full tool list with inputSchema for the capability modal."""
    mcps = _get_mcp_list()
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    if not target:
        raise fastapi.HTTPException(status_code=404, detail='MCP not found')

    if not _runtime_registration_allowed(target):
        return {'code': 200, 'data': _ordinary_tools(target.get('tools', []))}

    url = target.get('url', '')
    if not url or target.get('transport', 'http') != 'http':
        return {'code': 200, 'data': _ordinary_tools(target.get('tools', []))}

    headers = _outbound_headers(target)
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            init_payload = {
                'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05', 'capabilities': {},
                    'clientInfo': {'name': 'phanthy-motus', 'version': '1.0'},
                }
            }
            async with session.post(
                url,
                json=init_payload,
                headers=headers,
                allow_redirects=False,
            ) as resp:
                init_data = await resp.json(content_type=None)
                init_error = _rpc_response_error(
                    init_data,
                    request_id=1,
                    http_status=resp.status,
                )
                if init_error:
                    return {'code': 502, 'message': init_error, 'data': None}
                if not isinstance(init_data.get('result'), dict):
                    return {
                        'code': 502,
                        'message': 'Driver returned an invalid initialize result',
                        'data': None,
                    }
            tools_payload = {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}
            async with session.post(
                url,
                json=tools_payload,
                headers=headers,
                allow_redirects=False,
            ) as resp:
                data = await resp.json(content_type=None)
                tools_error = _rpc_response_error(
                    data,
                    request_id=2,
                    http_status=resp.status,
                )
                if tools_error:
                    return {'code': 502, 'message': tools_error, 'data': None}
                result = data['result']
                tools = result.get('tools') if isinstance(result, dict) else None
                if not isinstance(tools, list) or any(
                    not isinstance(tool, dict) for tool in tools
                ):
                    return {
                        'code': 502,
                        'message': 'Driver returned an invalid tools/list result',
                        'data': None,
                    }
        return {'code': 200, 'data': _ordinary_tools(tools)}
    except Exception:
        return {'code': 200, 'data': _ordinary_tools(target.get('tools', []))}


class MCPCallRequest(BaseModel):
    tool:      str
    arguments: dict = {}


async def _handle_agentcore_call(req: MCPCallRequest):
    """Handle tool calls for the internal agentcore MCP (decision_core)."""
    import topic_subscriber

    action = req.arguments.get('action', '')
    input_topic = req.arguments.get('input_topic', '')
    input_topics = req.arguments.get('input_topics', [])
    # Merge single + list params
    all_topics = list(input_topics) if input_topics else []
    if input_topic and input_topic not in all_topics:
        all_topics.append(input_topic)

    if action == 'start':
        # Auto-apply saved config before start (same pattern as HTTP MCPs)
        saved_cfg = _safe_saved_config(
            config.main.get(f'tool_config:agentcore:{req.tool}', None)
        )
        if saved_cfg:
            await _handle_agentcore_call(MCPCallRequest(
                tool=req.tool, arguments={**saved_cfg, 'action': 'config'}
            ))

        # Subscribe to requested topics (additive — cleanup is done by prior 'stop' call)
        if all_topics:
            event_cfg = config.main.get('event', {})
            topics = event_cfg.get('subscribe_topics', [])
            for t in all_topics:
                if t not in topics:
                    topics.append(t)
            event_cfg['subscribe_topics'] = topics
            config.main['event'] = event_cfg
            for t in all_topics:
                topic_subscriber.subscribe(t)
        return {
            'code': 200,
            'data': {
                'state': 'running',
                'subscribed_topics': all_topics,
            },
        }

    elif action == 'stop':
        event_cfg = config.main.get('event', {})
        topics = event_cfg.get('subscribe_topics', [])
        print(f'[agentcore] stop: all_topics={all_topics!r}, current_topics={topics}')
        if all_topics:
            # 指定 topic(s)：逐个退订
            for t in all_topics:
                if t in topics:
                    topics.remove(t)
                topic_subscriber.unsubscribe(t)
            event_cfg['subscribe_topics'] = topics
            config.main['event'] = event_cfg
            return {
                'code': 200,
                'data': {
                    'state': 'idle',
                    'unsubscribed_topics': all_topics,
                },
            }
        else:
            # 未指定 topic：退订全部（项目停止时的清理）
            for t in list(topics):
                topic_subscriber.unsubscribe(t)
            event_cfg['subscribe_topics'] = []
            config.main['event'] = event_cfg
            return {
                'code': 200,
                'data': {
                    'state': 'idle',
                    'unsubscribed_topics': list(topics),
                },
            }

    elif action == 'info':
        event_cfg = config.main.get('event', {})
        sub_topics = event_cfg.get('subscribe_topics', [])
        llm_cfg = event_cfg.get('llm', {})
        trigger_interval_ms = llm_cfg.get('trigger_interval_ms', 1000)
        topic_in_list = [{'topic': t, 'format': 'data/json'} for t in sub_topics] if sub_topics else [{'topic': '', 'format': 'data/json'}]
        return {'code': 200, 'data': {
            'description': '决策核心 — 接收多路 DDS 输入，LLM 推理后执行动作',
            'topic_in': topic_in_list,
            'topic_out': [{'topic': '/decision_core', 'format': 'data/json'}],
            'trigger_interval_ms': trigger_interval_ms,
        }}

    elif action == 'config':
        changes = {}
        candidate_llm = None
        client_mod = None

        # Stage and validate the LLM runtime before any durable write.
        llm_url = req.arguments.get('llm_url', '')
        llm_key = req.arguments.get('llm_key', '')
        llm_model = req.arguments.get('llm_model', '')
        think_mode = req.arguments.get('think_mode', False)
        if llm_url and llm_key:
            client_cfg = config.main.get('client', {})
            client_cfg['llm'] = [{
                'url': llm_url,
                'key': llm_key,
                'model': llm_model,
                'think_mode': think_mode,
            }]
            import client as client_mod

            candidate_llm = client_mod.llm.__class__(configs=client_cfg['llm'])
            changes['client'] = client_cfg

        # Save trigger_interval_ms to event.llm config
        trigger_interval = req.arguments.get('trigger_interval_ms')
        if trigger_interval is not None:
            event_cfg = config.main.get('event', {})
            llm_cfg = event_cfg.get('llm', {})
            llm_cfg['trigger_interval_ms'] = int(trigger_interval)
            event_cfg['llm'] = llm_cfg
            changes['event'] = event_cfg

        # Save search config to desktop_tools.search
        search_type = req.arguments.get('search_type')
        if search_type is not None:
            dt = config.main.get('desktop_tools', {})
            search_cfg = dt.get('search', {})
            search_cfg['type'] = search_type
            search_base = req.arguments.get('search_base_url', '')
            search_key = req.arguments.get('search_api_key', '')
            if search_base and search_base != '****':
                search_cfg['base_url'] = search_base
            if search_key and search_key != '****':
                search_cfg['api_key'] = search_key
            dt['search'] = search_cfg
            changes['desktop_tools'] = dt

        try:
            if changes:
                config.main.set_many(changes)
        except Exception:
            if candidate_llm is not None:
                await candidate_llm.aclose()
            raise
        if candidate_llm is not None and client_mod is not None:
            client_mod.llm = candidate_llm
        return {'code': 200, 'data': {'configured': True}}

    return {'code': 200, 'data': None}


@router.post('/{mcp_id}/call')
async def mcp_call_tool(mcp_id: str, req: MCPCallRequest):
    """Call a tool on an MCP server and return the result."""
    # ── Handle internal agentcore MCP (no HTTP transport) ──
    if mcp_id == 'agentcore':
        # remote_mic and remote_message — simple internal tools
        if req.tool == 'remote_mic':
            action = req.arguments.get('action', 'start')
            if action == 'start':
                # Self-check: ensure publisher exists + wait for real browser audio data
                from start import _ensure_mic_pub
                import start as _start_mod
                pub = _ensure_mic_pub()
                if pub is None:
                    return {'code': 200, 'data': {'state': 'error', 'message': 'ROS2 mic publisher not available'}}
                # Wait up to 10s for browser to connect and send audio chunks
                # (browser mic is started in parallel by frontend before this API call)
                import asyncio
                initial_count = _start_mod._mic_chunk_count
                for _ in range(20):  # 20 × 0.5s = 10s
                    if _start_mod._mic_chunk_count > initial_count:
                        return {'code': 200, 'data': {'state': 'running', 'ws_path': '/ws/mic',
                                                       'chunks_received': _start_mod._mic_chunk_count}}
                    await asyncio.sleep(0.5)
                # Timeout — no audio received
                if not _start_mod._mic_ws_connected:
                    return {'code': 200, 'data': {'state': 'error', 'message': '等待浏览器麦克风连接超时（10s）— 请在 dashboard 开启麦克风'}}
                else:
                    return {'code': 200, 'data': {'state': 'error', 'message': '浏览器已连接但未收到音频数据 — 请检查麦克风权限'}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'info':
                import ros2_bridge, start as _start_mod
                topic_visible = '/remote_control/mic' in ros2_bridge.get_dds_topics()
                return {'code': 200, 'data': {'state': 'running' if _start_mod._mic_chunk_count > 0 else 'idle',
                                               'ws_path': '/ws/mic',
                                               'topic_out': [{'topic': '/remote_control/mic', 'format': 'audio/pcm-16k'}],
                                               'topic_visible': topic_visible,
                                               'ws_connected': _start_mod._mic_ws_connected,
                                               'chunks_received': _start_mod._mic_chunk_count}}
            return {'code': 200, 'data': None}
        if req.tool == 'remote_message':
            action = req.arguments.get('action', 'start')
            if action == 'start':
                return {'code': 200, 'data': {'state': 'running'}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'send_message':
                text = req.arguments.get('text', '')
                if text:
                    import json as _json
                    import time as _time
                    import ros2_bridge
                    ros2_bridge.publish('/remote_control/message', _json.dumps({'text': text, 'ts': _time.time()}, ensure_ascii=False))
                    return {'code': 200, 'data': {'status': 'sent', 'text': text}}
                return {'code': 200, 'data': {'error': 'Missing text'}}
            return {'code': 200, 'data': None}
        if req.tool == 'remote_audio':
            action = req.arguments.get('action', 'start')
            if action == 'start':
                return {'code': 200, 'data': {'state': 'running'}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'send_audio':
                audio_file = req.arguments.get('audio_file', '')
                if not audio_file:
                    return {'code': 400, 'message': '缺少 audio_file 参数', 'data': None}
                from start import publish_audio_file
                return await publish_audio_file(audio_file)
            elif action == 'info':
                return {'code': 200, 'data': {'state': 'running', 'topic_out': [{'topic': '/remote_control/audio', 'format': 'audio/pcm-16k'}]}}
            return {'code': 200, 'data': None}
        return await _handle_agentcore_call(req)

    # ── Handle internal channel MCP ──
    if mcp_id == 'channel':
        if req.tool == 'channel_request':
            action = req.arguments.get('action', 'start')
            if action == 'start':
                # Self-check: verify channel adapter is connected (with retry wait)
                from channel.manager import manager as channel_mgr
                import asyncio
                instance_id = req.arguments.get('instance_id', '')
                channel_id = ''
                if instance_id:
                    cfg = config.main.get(f'tool_config:channel:channel_request:{instance_id}', None)
                    if cfg:
                        channel_id = cfg.get('channel_id', '')
                if channel_id:
                    # Wait up to 10s for adapter to connect
                    for _ in range(20):  # 20 × 0.5s = 10s
                        if channel_id in channel_mgr._adapters:
                            adapter = channel_mgr._adapters[channel_id]
                            if adapter.status() == 'connected':
                                return {'code': 200, 'data': {'state': 'running', 'channel': channel_id}}
                        await asyncio.sleep(0.5)
                    return {'code': 200, 'data': {'state': 'error', 'message': f'channel {channel_id} adapter not connected (10s timeout)'}}
                return {'code': 200, 'data': {'state': 'running'}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'info':
                channel_id = req.arguments.get('channel_id', '')
                if not channel_id:
                    instance_id = req.arguments.get('instance_id', '')
                    if instance_id:
                        cfg = config.main.get(f'tool_config:channel:channel_request:{instance_id}', None)
                        if cfg:
                            channel_id = cfg.get('channel_id', '')
                topic_id = channel_id.replace(' ', '_') if channel_id else ''
                topic = f'/channel/request/{topic_id}' if topic_id else '/channel/request'
                return {'code': 200, 'data': {'topic_out': [{'topic': topic, 'format': 'data/json'}]}}
            return {'code': 200, 'data': None}
        if req.tool == 'channel_reply':
            action = req.arguments.get('action', 'send')
            if action == 'start':
                # Self-check: send a greeting to verify channel is working
                from channel.manager import manager as channel_mgr
                try:
                    import asyncio
                    result = await channel_mgr.send_to_channel_any("我上线啦！我可以通过飞书与您交流。")
                    if result:
                        return {'code': 200, 'data': {'state': 'running', 'self_check': 'greeting sent'}}
                    else:
                        return {'code': 200, 'data': {'state': 'error', 'message': 'failed to send greeting — channel not connected'}}
                except Exception as e:
                    return {'code': 200, 'data': {'state': 'error', 'message': f'channel send failed: {e}'}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'send':
                text = req.arguments.get('text', '')
                if not text:
                    return {'code': 200, 'data': {'error': 'text is required'}}
                from channel.manager import manager as channel_mgr
                result = await channel_mgr.send_to_channel_any(text)
                return {'code': 200, 'data': {'result': result}}
            return {'code': 200, 'data': None}
        return {'code': 200, 'data': None}

    mcps = _get_mcp_list()
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    if not target:
        raise fastapi.HTTPException(status_code=404, detail='MCP not found')

    if not _runtime_registration_allowed(target):
        raise fastapi.HTTPException(status_code=403, detail='MCP registration is quarantined')

    if req.tool == 'teleop_session':
        raise fastapi.HTTPException(
            status_code=403,
            detail='Teleop tools are reserved for the dedicated teleop API',
        )

    tool_descriptor = next(
        (
            tool for tool in target.get('tools', [])
            if isinstance(tool, dict) and tool.get('name') == req.tool
        ),
        None,
    )
    if tool_descriptor is None:
        raise fastapi.HTTPException(status_code=404, detail='MCP tool not found')
    if _is_reserved_teleop_tool(tool_descriptor):
        raise fastapi.HTTPException(
            status_code=403,
            detail='Teleop tools are reserved for the dedicated teleop API',
        )

    url = target.get('url', '')
    if not url or target.get('transport', 'http') != 'http':
        raise fastapi.HTTPException(status_code=400, detail='MCP not reachable via HTTP')

    headers = _outbound_headers(target)
    timeout = aiohttp.ClientTimeout(total=30)
    access = _descriptor_tool_access(tool_descriptor, req.arguments)
    action = req.arguments.get('action')
    from teleop.service import coordinator
    try:
        authority_domain = authority_domain_for_target(
            mcp_id,
            target,
            targets=mcps,
        )
    except InvalidAuthorityBinding:
        raise fastapi.HTTPException(
            status_code=409,
            detail={
                'code': 'authority_binding_invalid',
                'reason': 'core_authority_binding_invalid',
                'robot_id': mcp_id,
            },
        ) from None
    admission = coordinator.command_broker.ordinary_command(
        authority_domain,
        read_only=access.read_only,
        source='mcp_api',
        tool=req.tool,
        action=action if isinstance(action, str) else '',
        tool_verified=True,
        action_verified=_descriptor_action_declared(tool_descriptor, req.arguments),
    )
    try:
        await admission.__aenter__()
    except TeleopCommandBlocked as error:
        raise fastapi.HTTPException(
            status_code=409,
            detail=error.public_detail(),
        ) from None
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Initialize first (required by MCP protocol)
            init_payload = {
                'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05', 'capabilities': {},
                    'clientInfo': {'name': 'phanthy-motus', 'version': '1.0'},
                }
            }
            async with session.post(
                url,
                json=init_payload,
                headers=headers,
                allow_redirects=False,
            ) as resp:
                init_data = await resp.json(content_type=None)
                init_error = _rpc_response_error(
                    init_data,
                    request_id=1,
                    http_status=resp.status,
                )
                if init_error:
                    return {'code': 502, 'message': init_error, 'data': None}
                if not isinstance(init_data.get('result'), dict):
                    return {
                        'code': 502,
                        'message': 'Driver returned an invalid initialize result',
                        'data': None,
                    }

            # Auto-config: start 前自动 apply 已保存的 config (shared + instance merged)
            # Also send config for non-system actions (set_*/get_*) so driver can resolve device_path after restart
            action = req.arguments.get('action')
            _SYSTEM_ACTIONS_NO_CONFIG = {'info', 'stop', 'config'}
            if (
                not access.read_only
                and action
                and action not in _SYSTEM_ACTIONS_NO_CONFIG
            ):
                # Check if this tool has configSchema (requires config before start)
                tools = target.get('tools') or []
                tool_obj = next((t for t in tools if isinstance(t, dict) and t.get('name') == req.tool), None)
                has_config_schema = bool(tool_obj and tool_obj.get('configSchema'))

                instance_id = req.arguments.get('instance_id', '')
                shared_cfg = _safe_saved_config(
                    config.main.get(f'tool_config:{mcp_id}:{req.tool}', None)
                )
                instance_cfg = {}
                if instance_id:
                    instance_cfg = _safe_saved_config(
                        config.main.get(
                            f'tool_config:{mcp_id}:{req.tool}:{instance_id}',
                            None,
                        )
                    )
                merged_cfg = {**shared_cfg, **instance_cfg}

                if merged_cfg:
                    cfg_args = {**merged_cfg, 'action': 'config'}
                    if instance_id:
                        cfg_args['instance_id'] = instance_id
                    cfg_payload = {
                        'jsonrpc': '2.0', 'id': 2,
                        'method': 'tools/call',
                        'params': {'name': req.tool, 'arguments': cfg_args},
                    }
                    async with session.post(
                        url,
                        json=cfg_payload,
                        headers=headers,
                        allow_redirects=False,
                    ) as resp:
                        cfg_data = await resp.json(content_type=None)
                        cfg_rpc_error = _rpc_response_error(
                            cfg_data,
                            request_id=2,
                            http_status=resp.status,
                        )
                        if cfg_rpc_error:
                            return {
                                'code': 502,
                                'message': cfg_rpc_error,
                                'data': None,
                            }
                        cfg_result = cfg_data['result']
                        cfg_result_error = _tool_result_error(
                            cfg_result,
                            require_structured_ack=True,
                        )
                        if cfg_result_error:
                            return {
                                'code': 502,
                                'message': f'[{req.tool}] {cfg_result_error}',
                                'data': None,
                            }
                elif has_config_schema:
                    return {'code': 400, 'message': f'[{req.tool}] 尚未配置，请先完成配置后再启动。', 'data': None}

            call_payload = {
                'jsonrpc': '2.0', 'id': 3,
                'method': 'tools/call',
                'params': {'name': req.tool, 'arguments': req.arguments},
            }
            async with session.post(
                url,
                json=call_payload,
                headers=headers,
                allow_redirects=False,
            ) as resp:
                data = await resp.json(content_type=None)
                rpc_error = _rpc_response_error(
                    data,
                    request_id=3,
                    http_status=resp.status,
                )
                if rpc_error:
                    return {'code': 502, 'message': rpc_error, 'data': None}
                result = data['result']
                result_error = _tool_result_error(
                    result,
                    require_structured_ack=(
                        isinstance(action, str)
                        and action in {'config', 'start', 'stop'}
                    ),
                )
                if result_error:
                    return {'code': 502, 'message': result_error, 'data': None}
                # Auto-register any instance-specific topics returned by the tool
                content_items = result.get('content') or []
                if isinstance(content_items, list):
                    for item in content_items:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            try:
                                parsed = json.loads(item.get('text', ''))
                                if isinstance(parsed, dict):
                                    topics_to_reg = parsed.get('topic_out', []) + parsed.get('topic_in', [])
                                    if any(t.get('topic') for t in topics_to_reg):
                                        asyncio.create_task(_notify_inspector(mcp_id, topics_to_reg))
                            except Exception:
                                pass
                return {'code': 200, 'data': result.get('content', result)}
    except Exception as e:
        return {'code': 500, 'message': str(e), 'data': None}
    finally:
        await admission.__aexit__(None, None, None)
