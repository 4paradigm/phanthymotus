import asyncio
import json
import time
from typing import Optional

import aiohttp
import fastapi
from pydantic import BaseModel

import config
import mcp_client

router = fastapi.APIRouter(prefix='/mcp', tags=['mcp'])

_mcp_write_lock = asyncio.Lock()  # 防止并发 ping 的 read-modify-write race condition


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


# Cache of last-seen tool names per mcp_id (for change detection)
_last_tool_names: dict[str, list[str]] = {}


# ── Models ───────────────────────────────────────────────────────────────────

class MCPAddRequest(BaseModel):
    name:      str
    transport: str = 'http'
    url:       str = ''
    render_hint: str = ''
    category:  str = ''


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _ping_mcp_http(url: str) -> dict:
    """Connect to an MCP HTTP server, initialize, list tools and resources."""
    headers = {'Content-Type': 'application/json'}
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
        async with session.post(url, json=init_payload, headers=headers) as resp:
            if resp.status >= 400:
                raise ConnectionError(f'MCP initialize failed: HTTP {resp.status}')
            init_data = await resp.json(content_type=None)
            server_name = init_data.get('result', {}).get('serverInfo', {}).get('name', '')

        # List tools
        try:
            tools_payload = {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}
            async with session.post(url, json=tools_payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                tools = [
                    {k: v for k, v in t.items() if k in ('name', 'description', 'type', 'multiInstance', 'inputSchema', 'configSchema', 'topic_out', 'topic_in')}
                    for t in data.get('result', {}).get('tools', [])
                ]
        except Exception as e:
            print(f'[mcp/tools] error: {e}')
            pass

        # Call all *_info / info tools in parallel — device self-reports type and topics.
        # Bundles expose per-plugin tools like mic_info, loco_info; single devices use bare 'info'.
        # Tools with action enum containing 'info' are called with {action: "info"}.
        tool_names = [t.get('name', '') if isinstance(t, dict) else t for t in tools]
        info_tools = [n for n in tool_names if n == 'info' or n.endswith('_info')]
        # Also detect tools with action schema containing 'info'
        action_info_tools = []
        for t in tools:
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
                async with session.post(url, json=payload, headers=headers) as resp:
                    data = await resp.json(content_type=None)
                    content = data.get('result', {}).get('content', [])
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
            async with session.post(url, json=res_payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                resources = [r.get('name') for r in data.get('result', {}).get('resources', [])]
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
            'tools':       m.get('tools', []),
            'resources':   m.get('resources', []),
            'topic_out':   m.get('topic_out', []),
            'topic_in':    m.get('topic_in',  []),
            'category':    m.get('category', ''),
            'depends_on':  m.get('depends_on', ''),
            'ws_path':     ('/ws/bus' + (m.get('topic_out') or [{}])[0].get('topic', '')) if m.get('topic_out') else '',
            'online':      None,
        }
        for m in _get_mcp_list()
    ]
    return {'code': 200, 'data': items}


@router.post('')
async def mcp_add(req: MCPAddRequest):
    async with _mcp_write_lock:
        mcps = _get_mcp_list()
        # Upsert: match by URL, name, or server_name (prevents duplicate device bundles)
        existing = next(
            (m for m in mcps if (m.get('url') == req.url and req.url)
             or (m.get('name') == req.name and req.name)
             or (m.get('server_name') and m.get('server_name') == req.name)),
            None,
        )
        if existing:
            existing['name']        = req.name
            existing['transport']   = req.transport
            existing['url']         = req.url
            existing['render_hint'] = req.render_hint
            if req.category:
                existing['category'] = req.category
            _save_mcp_list(mcps)
            mcp_id = existing['id']
        else:
            mcp_id = f'mcp-{int(time.time())}'
            mcps.append({
                'id':          mcp_id,
                'name':        req.name,
                'transport':   req.transport,
                'url':         req.url,
                'render_hint': req.render_hint,
                'category':    req.category,
            })
            _save_mcp_list(mcps)
    # Auto-ping to discover tools
    asyncio.create_task(_do_ping(mcp_id))
    return {'code': 200, 'data': {'id': mcp_id}}


@router.delete('/{mcp_id}')
async def mcp_delete(mcp_id: str):
    mcps = [m for m in _get_mcp_list() if m.get('id') != mcp_id]
    _save_mcp_list(mcps)
    return {'code': 200}


async def _do_ping(mcp_id: str) -> dict:
    """Core ping logic — fetch capabilities, persist, notify inspector.
    Returns the same dict as the ping endpoint's data field.
    Raises HTTPException(404) if mcp_id not found."""
    mcps = _get_mcp_list()
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    if not target:
        raise fastapi.HTTPException(status_code=404, detail='MCP not found')

    transport = target.get('transport', 'http')
    url       = target.get('url', '')

    if transport != 'http' or not url:
        is_internal = transport == 'internal'
        return {
            'online':      is_internal and target.get('online', False),
            'tools':       target.get('tools', []),
            'resources':   target.get('resources', []),
            'render_hint': target.get('render_hint', ''),
            'server_name': target.get('server_name', ''),
            'topic_out':   target.get('topic_out', []),
            'topic_in':    target.get('topic_in', []),
        }

    try:
        caps = await _ping_mcp_http(url)
    except Exception as e:
        # 标记 registry 中该设备离线
        if mcp_id in mcp_client.registry:
            mcp_client.registry[mcp_id]['online'] = False
        # Dedup: if this offline MCP has same server_name as another entry, remove it
        async with _mcp_write_lock:
            mcps = _get_mcp_list()
            this_entry = next((m for m in mcps if m.get('id') == mcp_id), None)
            if this_entry and this_entry.get('server_name'):
                dup = next((m for m in mcps if m.get('server_name') == this_entry['server_name'] and m.get('id') != mcp_id), None)
                if dup:
                    mcps = [m for m in mcps if m.get('id') != mcp_id]
                    _save_mcp_list(mcps)
                    print(f'[mcp/ping] dedup: removed offline {mcp_id} (same server_name as {dup["id"]})')
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

    # Log only when tools change (first ping or tool list updated)
    current_tool_names = [t.get('name', '') if isinstance(t, dict) else t for t in caps['tools']]
    prev_tool_names = _last_tool_names.get(mcp_id)
    if prev_tool_names != current_tool_names:
        _last_tool_names[mcp_id] = current_tool_names
        print(f'[mcp/ping] {mcp_id}: server={caps.get("server_name", "?")} tools={current_tool_names}')

    # Persist on every successful ping; server_name only set once (not overwritten)
    # Also deduplicate: if another MCP with the same server_name exists, remove this one (keep the earlier entry)
    async with _mcp_write_lock:
        mcps = _get_mcp_list()  # re-read under lock to avoid race condition
        new_server_name = caps.get('server_name', '')

        # Check for duplicate server_name — keep the first registered entry, remove this one
        if new_server_name:
            existing_with_same_name = next(
                (m for m in mcps if m.get('server_name') == new_server_name and m.get('id') != mcp_id),
                None,
            )
            if existing_with_same_name:
                # This is a duplicate — remove current entry, update existing one's URL
                target = next((m for m in mcps if m.get('id') == mcp_id), None)
                if target:
                    existing_with_same_name['url'] = target.get('url', existing_with_same_name.get('url', ''))
                    mcps = [m for m in mcps if m.get('id') != mcp_id]
                    print(f'[mcp/ping] dedup: removed {mcp_id}, merged into {existing_with_same_name["id"]} (server_name={new_server_name})')
                    _save_mcp_list(mcps)
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
                if not m.get('server_name'):
                    m['server_name'] = new_server_name
                break
        _save_mcp_list(mcps)

    # 同步更新内存中的 mcp_client.registry（LLM 决策依赖此数据）
    schemas = {}
    tool_meta_map = {}
    split_map = {}
    tool_groups = {}
    for tool in caps['tools']:
        tool_schemas = mcp_client._to_openai_schema(mcp_id, tool)

        if len(tool_schemas) == 1:
            schema = tool_schemas[0]
            schemas[schema['name']] = schema
            action_enum = (tool.get('inputSchema') or {}).get('properties', {}).get('action', {}).get('enum')
            tool_meta_map[schema['name']] = {
                'type': tool.get('type'),
                'action_enum': action_enum,
                'has_config_schema': bool(tool.get('configSchema')),
            }
        else:
            group = []
            for schema in tool_schemas:
                schemas[schema['name']] = schema
                tool_meta_map[schema['name']] = {
                    'type': tool.get('type'),
                    'action_enum': None,
                    'has_config_schema': bool(tool.get('configSchema')),
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
        'name':        target.get('name', mcp_id),
        'url':         url,
        'online':      True,
        'tools':       [t.get('name', '') if isinstance(t, dict) else t for t in caps['tools']],
        'render_hint': render_hint,
        'schemas':     schemas,
        'tool_meta':   tool_meta_map,
        'split_map':   split_map,
        'tool_groups': tool_groups,
    }

    # Notify inspection module about all topics from this device
    asyncio.create_task(_notify_inspector(mcp_id, topic_out + topic_in))

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

    url = target.get('url', '')
    if not url or target.get('transport', 'http') != 'http':
        return {'code': 200, 'data': target.get('tools', [])}

    headers = {'Content-Type': 'application/json'}
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
            await session.post(url, json=init_payload, headers=headers)
            tools_payload = {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}
            async with session.post(url, json=tools_payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                tools = data.get('result', {}).get('tools', [])
        return {'code': 200, 'data': tools}
    except Exception:
        return {'code': 200, 'data': target.get('tools', [])}


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
        saved_cfg = config.main.get(f'tool_config:agentcore:{req.tool}', None)
        if saved_cfg:
            await _handle_agentcore_call(MCPCallRequest(
                tool=req.tool, arguments={'action': 'config', **saved_cfg}
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
        return {'code': 200, 'data': f'subscribed to {all_topics}' if all_topics else 'started'}

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
            return {'code': 200, 'data': f'unsubscribed from {all_topics}'}
        else:
            # 未指定 topic：退订全部（项目停止时的清理）
            for t in list(topics):
                topic_subscriber.unsubscribe(t)
            event_cfg['subscribe_topics'] = []
            config.main['event'] = event_cfg
            return {'code': 200, 'data': 'unsubscribed all topics'}

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
        # Save LLM config to client.llm (list format used by client/llm.py)
        llm_url = req.arguments.get('llm_url', '')
        llm_key = req.arguments.get('llm_key', '')
        llm_model = req.arguments.get('llm_model', '')
        if llm_url and llm_key:
            client_cfg = config.main.get('client', {})
            client_cfg['llm'] = [{'url': llm_url, 'key': llm_key, 'model': llm_model}]
            config.main['client'] = client_cfg
            # Reinitialize the LLM client with new config
            import client as client_mod
            client_mod.llm = client_mod.llm.__class__()
        # Save trigger_interval_ms to event.llm config
        trigger_interval = req.arguments.get('trigger_interval_ms')
        if trigger_interval is not None:
            event_cfg = config.main.get('event', {})
            llm_cfg = event_cfg.get('llm', {})
            llm_cfg['trigger_interval_ms'] = int(trigger_interval)
            event_cfg['llm'] = llm_cfg
            config.main['event'] = event_cfg
        return {'code': 200, 'data': 'config saved'}

    return {'code': 200, 'data': None}


@router.post('/{mcp_id}/call')
async def mcp_call_tool(mcp_id: str, req: MCPCallRequest):
    """Call a tool on an MCP server and return the result."""
    # ── Handle internal agentcore MCP (no HTTP transport) ──
    if mcp_id == 'agentcore':
        return await _handle_agentcore_call(req)

    mcps = _get_mcp_list()
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    if not target:
        raise fastapi.HTTPException(status_code=404, detail='MCP not found')

    url = target.get('url', '')
    if not url or target.get('transport', 'http') != 'http':
        raise fastapi.HTTPException(status_code=400, detail='MCP not reachable via HTTP')

    headers = {'Content-Type': 'application/json'}
    timeout = aiohttp.ClientTimeout(total=10)
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
            await session.post(url, json=init_payload, headers=headers)

            # Auto-config: start 前自动 apply 已保存的 config (shared + instance merged)
            # Also send config for non-system actions (set_*/get_*) so driver can resolve device_path after restart
            action = req.arguments.get('action')
            _SYSTEM_ACTIONS_NO_CONFIG = {'info', 'stop', 'config'}
            if action and action not in _SYSTEM_ACTIONS_NO_CONFIG:
                # Check if this tool has configSchema (requires config before start)
                tools = target.get('tools') or []
                tool_obj = next((t for t in tools if isinstance(t, dict) and t.get('name') == req.tool), None)
                has_config_schema = bool(tool_obj and tool_obj.get('configSchema'))

                instance_id = req.arguments.get('instance_id', '')
                shared_cfg = config.main.get(f'tool_config:{mcp_id}:{req.tool}', None) or {}
                instance_cfg = {}
                if instance_id:
                    instance_cfg = config.main.get(f'tool_config:{mcp_id}:{req.tool}:{instance_id}', None) or {}
                merged_cfg = {**shared_cfg, **instance_cfg}

                if merged_cfg:
                    cfg_args = {'action': 'config', **merged_cfg}
                    if instance_id:
                        cfg_args['instance_id'] = instance_id
                    cfg_payload = {
                        'jsonrpc': '2.0', 'id': 2,
                        'method': 'tools/call',
                        'params': {'name': req.tool, 'arguments': cfg_args},
                    }
                    async with session.post(url, json=cfg_payload, headers=headers) as resp:
                        cfg_data = await resp.json(content_type=None)
                        cfg_result = cfg_data.get('result', {})
                        cfg_content = (cfg_result.get('content') or [{}])[0].get('text', '{}')
                        try:
                            parsed = json.loads(cfg_content)
                            if not parsed.get('adapter_ok', True):
                                return {'code': 400, 'message': f'[{req.tool}] 配置无效（缺少 url/key），请检查配置。', 'data': None}
                        except (json.JSONDecodeError, IndexError):
                            pass
                elif has_config_schema:
                    return {'code': 400, 'message': f'[{req.tool}] 尚未配置，请先完成配置后再启动。', 'data': None}

            call_payload = {
                'jsonrpc': '2.0', 'id': 3,
                'method': 'tools/call',
                'params': {'name': req.tool, 'arguments': req.arguments},
            }
            async with session.post(url, json=call_payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                result = data.get('result', {})
                error  = data.get('error')
                if error:
                    return {'code': 500, 'message': error.get('message', 'Tool call error'), 'data': None}
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
