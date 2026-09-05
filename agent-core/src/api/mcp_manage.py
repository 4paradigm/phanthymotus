import asyncio
import json
import time
from typing import Optional

import aiohttp
import fastapi
from pydantic import BaseModel

import config
import mcp_client
from tool_config import (missing_required_config, plan_config_calls,
                         split_config_by_scope)

router = fastapi.APIRouter(prefix='/mcp', tags=['mcp'])

_mcp_write_lock = asyncio.Lock()  # 防止并发 ping 的 read-modify-write race condition


async def _notify_inspector(mcp_id: str, topic_out: list, topic_in: list | None = None) -> None:
    """Register topics with the embedded inspection module (process-internal call).

    生产者（topic_out）与消费者（topic_in）分开登记：一个 topic 的数据格式由**发布方**
    决定。消费者声明的 format 只是「我要吃什么」，不能拿它改写别人发布的总线格式 ——
    否则 ocr 的 image/jpeg 输入会把 /decision_core 注册成图片总线，仪表盘按图片渲染、
    DDS 订阅也会用错消息类型。
    """
    from api.inspection import register_topic_internal
    for producer, topics in ((True, topic_out or []), (False, topic_in or [])):
        for t in topics:
            topic = t.get('topic', '')
            fmt   = t.get('format', '')
            if not topic:
                continue
            try:
                await register_topic_internal(topic, fmt, mcp_id, producer=producer)
            except Exception:
                pass


def _fmt_match(required: str, available: str) -> bool:
    """数据格式是否兼容。与前端 setup.js 的 _fmtMatch 保持一致：精确相等，或 `audio/*` 前缀通配。"""
    if not required or not available:
        return False
    if required == available:
        return True
    if required.endswith('/*'):
        return available.startswith(required[:-1])
    return False


# 已经提示过「上游没有匹配格式」的 (mcp_id, format)，避免每次 ping 都刷日志
_unmatched_logged: set = set()


def _upstream_topic_for(fmt: str, upstream_out: list) -> str:
    """在上游的 topic_out 里找格式匹配的那条 topic。"""
    for t in upstream_out:
        if t.get('topic') and _fmt_match(fmt, t.get('format', '')):
            return t['topic']
    return ''


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

@router.get('/hooks/status')
async def hooks_status():
    """Return hook registry and recent fire log for diagnostics."""
    import hooks
    return {'code': 200, 'data': hooks.get_status()}


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


async def _restore_saved_configs(mcp_id: str, url: str, tools: list) -> None:
    """Re-send saved tool configs to a device that just came online.

    Only sends shared (non-instance) configs for tools that have configSchema.
    Called once when a device transitions from offline → online.
    """
    headers = {'Content-Type': 'application/json'}
    timeout = aiohttp.ClientTimeout(total=5)
    sent = []

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                tool_name = tool.get('name', '')
                if not tool.get('configSchema'):
                    continue
                saved_cfg = config.main.get(f'tool_config:{mcp_id}:{tool_name}', None)
                if not saved_cfg:
                    continue
                # Drop keys the current schema no longer advertises, same as the
                # start path — a stale row must not be replayed on every reconnect.
                shared_cfg, instance_cfg = split_config_by_scope(tool, saved_cfg)
                restore_cfg = {**shared_cfg, **instance_cfg}
                if not restore_cfg:
                    continue
                cfg_payload = {
                    'jsonrpc': '2.0', 'id': 99,
                    'method': 'tools/call',
                    'params': {'name': tool_name, 'arguments': {'action': 'config', **restore_cfg}},
                }
                await session.post(url, json=cfg_payload, headers=headers)
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

    transport = target.get('transport', 'http')
    url       = target.get('url', '')

    if transport != 'http' or not url:
        is_internal = transport == 'internal'
        # Register topics for internal MCPs (so inspection/monitoring works)
        if is_internal:
            t_out = target.get('topic_out', []) or []
            t_in  = target.get('topic_in', []) or []
            if t_out or t_in:
                asyncio.create_task(_notify_inspector(mcp_id, t_out, t_in))
        return {
            'online':      is_internal and target.get('online', False),
            'tools':       target.get('tools', []),
            'resources':   target.get('resources', []),
            'render_hint': target.get('render_hint', ''),
            'server_name': target.get('server_name', ''),
            'topic_out':   target.get('topic_out', []),
            'topic_in':    target.get('topic_in', []),
        }

    # 记录 ping 前的 online 状态，用于判断是否需要重新下发 config
    was_online = mcp_client.registry.get(mcp_id, {}).get('online', False)

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

    upstream_out = []
    depends_on = target.get('depends_on', '')
    if depends_on:
        upstream = next((m for m in mcps if m.get('id') == depends_on), None)
        upstream_out = (upstream or {}).get('topic_out') or []

    # Fill empty topic_in from upstream — **按格式匹配**，不是无脑取 topic_out[0]。
    # agentcore 的 topic_out[0] 是 /decision_core (data/json)，取 [0] 会把 ocr/vop 的
    # image/jpeg 输入挂到决策总线上，进而把 /decision_core 注册成图片格式：仪表盘按图片
    # 渲染，DDS 订阅也会拿 CompressedImage 去订阅 std_msgs/String。
    # 匹配不到就留空 —— UI 上显示「未连接」是对的，硬塞一个格式不符的总线不是。
    for t in topic_in:
        if t.get('topic'):
            continue
        fmt = t.get('format', '')
        picked = _upstream_topic_for(fmt, upstream_out)
        if picked:
            t['topic'] = picked
        elif upstream_out and (mcp_id, fmt) not in _unmatched_logged:
            _unmatched_logged.add((mcp_id, fmt))
            avail = ', '.join(f'{x.get("topic", "?")}({x.get("format", "?")})' for x in upstream_out)
            print(f'[mcp/ping] {mcp_id}: upstream {depends_on} has no {fmt!r} topic — '
                  f'input left unbound (upstream offers: {avail})')

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
            raw_input_schema = tool.get('inputSchema') or {}
            action_enum = raw_input_schema.get('properties', {}).get('action', {}).get('enum')
            tool_meta_map[schema['name']] = {
                'type': tool.get('type'),
                'action_enum': action_enum,
                'has_config_schema': bool(tool.get('configSchema')),
                'completion': raw_input_schema.get('x-completion'),
                'resource': mcp_client.parse_resources(raw_input_schema.get('x-resource')),
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
                    'resource': mcp_client.parse_resources(
                        (tool.get('inputSchema') or {}).get('x-resource')),
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

    # Register system hooks from x-hooks declarations
    import hooks
    for tool in caps['tools']:
        x_hooks = (tool.get('inputSchema') or {}).get('x-hooks')
        if x_hooks and isinstance(x_hooks, dict):
            hooks.register(mcp_id, tool.get('name', ''), x_hooks)

    # Notify inspection module about all topics from this device
    asyncio.create_task(_notify_inspector(mcp_id, topic_out, topic_in))

    # Auto-restore saved configs when device comes online (first ping or after offline)
    if not was_online:
        asyncio.create_task(_restore_saved_configs(mcp_id, url, caps['tools']))

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
    # A paired peer's tools live only in mcp_client.registry (peer/mcp_bridge.py),
    # never in the configured device list, so the lookup below would 404 on them.
    if mcp_id.startswith('peer:'):
        import mcp_client as _mc
        entry = _mc.registry.get(mcp_id)
        if not entry:
            raise fastapi.HTTPException(status_code=404, detail='peer not offering tools')
        return {'code': 200, 'data': {
            'tools': [
                {'name': name.split('__', 2)[-1],
                 'description': schema.get('description', ''),
                 'inputSchema': schema.get('parameters', {})}
                for name, schema in (entry.get('schemas') or {}).items()
            ],
        }}

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
            'vision_input': bool(llm_cfg.get('vision_input', False)),
        }}

    elif action == 'config':
        # Save LLM config to client.llm (list format used by client/llm.py)
        llm_url = req.arguments.get('llm_url', '')
        llm_key = req.arguments.get('llm_key', '')
        llm_model = req.arguments.get('llm_model', '')
        think_mode = req.arguments.get('think_mode', False)
        if llm_url and llm_key:
            client_cfg = config.main.get('client', {})
            client_cfg['llm'] = [{'url': llm_url, 'key': llm_key, 'model': llm_model, 'think_mode': think_mode}]
            config.main['client'] = client_cfg

            # Mirror into services.llm as well. The dashboard has two places
            # that configure the LLM — Settings (api/config.py) and this
            # decision_core card — and only Settings used to write both keys.
            # Configuring from the card therefore left services.llm empty, so
            # the Settings page showed blank fields for a working setup, and
            # saving that blank form wrote the emptiness back through to
            # client.llm and killed the LLM. Keep the two entry points
            # symmetric; services.llm is what the Settings form reads.
            services_cfg = config.main.get('services', {})
            existing_llm = services_cfg.get('llm', {}) or {}
            services_cfg['llm'] = {
                'url': llm_url,
                'key': llm_key,
                'model': llm_model,
                # Preserve fields this card does not expose rather than
                # dropping them; think_mode is set here, others are not.
                'think_mode': think_mode,
                **{k: v for k, v in existing_llm.items()
                   if k not in ('url', 'key', 'model', 'think_mode')},
            }
            config.main['services'] = services_cfg

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
        # 图片输入开关：模型不支持时把图像内容内联进请求，只会换来一个 400 和一轮失败
        vision_input = req.arguments.get('vision_input')
        if vision_input is not None:
            event_cfg = config.main.get('event', {})
            llm_cfg = event_cfg.get('llm', {})
            llm_cfg['vision_input'] = bool(vision_input)
            event_cfg['llm'] = llm_cfg
            config.main['event'] = event_cfg
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
            config.main['desktop_tools'] = dt
        return {'code': 200, 'data': 'config saved'}

    return {'code': 200, 'data': None}


@router.post('/{mcp_id}/call')
async def mcp_call_tool(mcp_id: str, req: MCPCallRequest):
    """Call a tool on an MCP server and return the result."""
    # A paired peer's tool is reached over its signed link, not local HTTP, and this
    # handler builds JSON-RPC itself — so hand it to call_tool, which knows how to
    # route `transport: 'peer'` (peer/mcp_bridge.py). Without this the dashboard has
    # no way to exercise a peer tool at all.
    if mcp_id.startswith('peer:'):
        import mcp_client as _mc
        if mcp_id not in _mc.registry:
            raise fastapi.HTTPException(status_code=404, detail='peer not offering tools')
        result = await _mc.call_tool(f'mcp__{mcp_id}__{req.tool}', dict(req.arguments))
        return {'code': 200, 'data': {'result': result}}

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
        if req.tool == 'remote_image':
            action = req.arguments.get('action', 'start')
            if action == 'start':
                return {'code': 200, 'data': {'state': 'running'}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'send_image':
                image_file = req.arguments.get('image_file', '')
                if not image_file:
                    return {'code': 400, 'message': '缺少 image_file 参数', 'data': None}
                from start import publish_image_file
                return await publish_image_file(image_file)
            elif action == 'info':
                return {'code': 200, 'data': {'state': 'running', 'topic_out': [{'topic': '/remote_control/image', 'format': 'image/jpeg'}]}}
            return {'code': 200, 'data': None}
        return await _handle_agentcore_call(req)

    # ── Handle internal channel MCP ──
    if mcp_id == 'channel':
        from channel.manager import channel_request_topic, manager as channel_mgr

        def _card_channel(tool: str, instance_id: str) -> str:
            if not instance_id:
                return ''
            cfg = config.main.get(f'tool_config:channel:{tool}:{instance_id}', None)
            return (cfg or {}).get('channel_id', '')

        if req.tool == 'channel_request':
            action = req.arguments.get('action', 'start')
            instance_id = req.arguments.get('instance_id', '')
            if action == 'start':
                # 自检：卡片必须选了一个存在且启用的 channel，且 adapter 真的连通。
                # ensure_connected 会顺手把被 Stop 过的 adapter 拉起来（自愈）。
                channel_id = _card_channel('channel_request', instance_id)
                ok, reason = await channel_mgr.ensure_connected(channel_id)
                if not ok:
                    print(f'[channel_request] self-check failed: {reason}')
                    return {'code': 200, 'data': {'state': 'error', 'message': reason}}
                return {'code': 200, 'data': {'state': 'running', 'channel': channel_id}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'info':
                channel_id = req.arguments.get('channel_id', '') or _card_channel(
                    'channel_request', instance_id)
                topic = channel_request_topic(channel_id)
                return {'code': 200, 'data': {'topic_out': [{'topic': topic, 'format': 'data/json'}]}}
            return {'code': 200, 'data': None}
        if req.tool == 'channel_reply':
            action = req.arguments.get('action', 'send')
            instance_id = req.arguments.get('instance_id', '')
            if action == 'start':
                # 自检：静默探测（校验配置 + adapter 健康），不给用户发问候消息。
                # 旧实现发「我上线啦！」并用 `if result:` 判断，而返回值恒为非空字符串 —— 永远「通过」。
                channel_id = _card_channel('channel_reply', instance_id)
                ok, reason = await channel_mgr.ensure_connected(channel_id)
                if not ok:
                    print(f'[channel_reply] self-check failed: {reason}')
                    return {'code': 200, 'data': {'state': 'error', 'message': reason}}
                return {'code': 200, 'data': {'state': 'running', 'channel': channel_id}}
            elif action == 'stop':
                return {'code': 200, 'data': {'state': 'idle'}}
            elif action == 'send':
                text = req.arguments.get('text', '')
                files = req.arguments.get('files', []) or []
                mention_open_id = req.arguments.get('mention_open_id', '')
                source_message_id = req.arguments.get('source_message_id', '')
                trusted_bot_id = req.arguments.get('trusted_bot_id', '')
                expect_reply = req.arguments.get('expect_reply', False)
                if not text and not files:
                    return {'code': 200, 'data': {'error': 'text or files is required'}}
                result = await channel_mgr.send_reply(
                    instance_id=instance_id,
                    text=text,
                    files=files,
                    mention_open_id=mention_open_id,
                    source_message_id=source_message_id,
                    expect_reply=expect_reply,
                    trusted_bot_id=trusted_bot_id,
                )
                return {'code': 200, 'data': {'result': result}}
            return {'code': 200, 'data': None}
        return {'code': 200, 'data': None}

    mcps = _get_mcp_list()
    target = next((m for m in mcps if m.get('id') == mcp_id), None)
    if not target:
        raise fastapi.HTTPException(status_code=404, detail='MCP not found')

    url = target.get('url', '')
    if not url or target.get('transport', 'http') != 'http':
        raise fastapi.HTTPException(status_code=400, detail='MCP not reachable via HTTP')

    headers = {'Content-Type': 'application/json'}
    timeout = aiohttp.ClientTimeout(total=None)
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

            # Auto-config: start 前自动 apply 已保存的 config
            # Also send config for non-system actions (set_*/get_*) so driver can resolve device_path after restart
            action = req.arguments.get('action')
            _SYSTEM_ACTIONS_NO_CONFIG = {'info', 'stop', 'config'}
            if action and action not in _SYSTEM_ACTIONS_NO_CONFIG:
                tools = target.get('tools') or []
                tool_obj = next((t for t in tools if isinstance(t, dict) and t.get('name') == req.tool), None)

                instance_id = req.arguments.get('instance_id', '')
                saved_shared = config.main.get(f'tool_config:{mcp_id}:{req.tool}', None) or {}
                saved_instance = {}
                if instance_id:
                    saved_instance = config.main.get(f'tool_config:{mcp_id}:{req.tool}:{instance_id}', None) or {}

                # Partition both rows by declared scope, dropping keys the
                # current schema no longer advertises. Either row can hold
                # either kind of key: older UI builds saved without filtering.
                shared_cfg, instance_cfg = split_config_by_scope(tool_obj, saved_shared)
                inst_shared, inst_own = split_config_by_scope(tool_obj, saved_instance)
                merged_for_required = {**shared_cfg, **inst_shared, **instance_cfg, **inst_own}

                missing = missing_required_config(tool_obj, merged_for_required)
                if missing:
                    return {'code': 400,
                            'message': f'[{req.tool}] 尚未配置：缺少 {"、".join(missing)}，请先在卡片配置里填写后再启动。',
                            'data': None}

                for cfg_body, extra_args in plan_config_calls(
                        tool_obj, saved_shared, saved_instance, instance_id):
                    cfg_payload = {
                        'jsonrpc': '2.0', 'id': 2,
                        'method': 'tools/call',
                        'params': {'name': req.tool,
                                   'arguments': {'action': 'config', **cfg_body, **extra_args}},
                    }
                    async with session.post(url, json=cfg_payload, headers=headers) as resp:
                        cfg_data = await resp.json(content_type=None)
                        cfg_error = cfg_data.get('error')
                        if cfg_error:
                            return {'code': 400,
                                    'message': f'[{req.tool}] 配置失败：{cfg_error.get("message", "unknown error")}',
                                    'data': None}
                        cfg_result = cfg_data.get('result', {})
                        cfg_content = (cfg_result.get('content') or [{}])[0].get('text', '{}')
                        try:
                            parsed = json.loads(cfg_content)
                            if not parsed.get('adapter_ok', True):
                                return {'code': 400, 'message': f'[{req.tool}] 配置无效（缺少 url/key），请检查配置。', 'data': None}
                        except (json.JSONDecodeError, IndexError):
                            pass

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
                                    t_out = parsed.get('topic_out', []) or []
                                    t_in  = parsed.get('topic_in', []) or []
                                    if any(t.get('topic') for t in t_out + t_in):
                                        asyncio.create_task(_notify_inspector(mcp_id, t_out, t_in))
                            except Exception:
                                pass
                return {'code': 200, 'data': result.get('content', result)}
    except Exception as e:
        return {'code': 500, 'message': str(e), 'data': None}
