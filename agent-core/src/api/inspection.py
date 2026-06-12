"""
inspection.py — DDS topic monitoring & WebSocket relay, embedded in agent-core.

Previously a separate `dds_inspection` service; merged here to eliminate the proxy hop.

HTTP endpoints (mounted under /api via start.py):
  POST /monitor/start      — (legacy, no-op) 保留兼容
  POST /monitor/stop       — (legacy, no-op) 保留兼容
  GET  /monitor            — 返回当前监控状态（是否有活跃订阅）
  POST /topics/register    — 注册 topic
  GET  /topics             — 返回已注册 topic 列表（含 status）
  GET  /topics/status      — 返回 {topic: status} 字典

WebSocket endpoint (mounted on root app in start.py):
  WS   /ws/bus/{topic:path}  — 订阅并实时推送该 topic 的数据帧（按需自动启动 DDS 订阅）
"""

import asyncio
import json
import time

import fastapi

import ros2_bridge

router    = fastapi.APIRouter(tags=['inspection'])
ws_router = fastapi.APIRouter(tags=['inspection'])

# ── Module-level state ─────────────────────────────────────────────────────────

# topic → {format, mcp_id, registered_at}
_topic_registry: dict[str, dict] = {}

# per-topic fan-out queues: topic → list of asyncio.Queue (one per WS consumer)
_topic_queues: dict[str, list] = {}

# Active primary subscriptions (topic paths with live DDS sub)
_active_primary_subs: set[str] = set()

# Last frame cache per topic (for initial snapshot push on new WS connect)
_last_frame: dict[str, bytes] = {}


# ── Status helpers ─────────────────────────────────────────────────────────────

def _topic_status(topic: str) -> str:
    if topic in _active_primary_subs:
        if time.time() - ros2_bridge.get_last_seen(topic) < 10:
            return 'active'
        return 'online'
    if topic in ros2_bridge.get_dds_topics():
        return 'online'
    return 'offline'


# ── Primary subscription management ──────────────────────────────────────────

def _push_factory(topic: str):
    """Create a push callback for a topic that fans out to all WS consumers."""
    async def _push(data: bytes, msg_fmt: str):
        # Cache latest frame for snapshot push on new connections
        _last_frame[topic] = data
        queues = _topic_queues.get(topic, [])
        for q in list(queues):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass  # drop frame for slow consumer
    return _push


def _ensure_primary_sub(topic: str, fmt: str, loop: asyncio.AbstractEventLoop):
    """Start primary ROS2 subscription only if not already active. Once started, stays forever."""
    if topic in _active_primary_subs:
        return  # already subscribed, no DDS discovery delay

    key = f'__primary__#{topic}'
    ros2_bridge.subscribe(key, topic, fmt, loop, _push_factory(topic))
    _active_primary_subs.add(topic)
    print(f'[inspection] started primary sub: {topic}')


# ── Internal API (called by mcp_manage directly) ───────────────────────────────

async def register_topic_internal(topic: str, fmt: str, mcp_id: str) -> None:
    """Register a topic in the registry; if consumers exist, start primary sub immediately."""
    if not topic:
        return
    existing = _topic_registry.get(topic)
    if existing and existing.get('format') == fmt and existing.get('mcp_id') == mcp_id:
        return  # already registered with same params, skip
    _topic_registry[topic] = {
        'format':        fmt,
        'mcp_id':        mcp_id,
        'registered_at': time.time(),
    }
    print(f'[inspection] registered topic={topic} format={fmt} mcp_id={mcp_id}')
    # Start primary sub immediately on registration (stays forever)
    loop = asyncio.get_event_loop()
    _ensure_primary_sub(topic, fmt, loop)


# ── HTTP: Monitor (legacy, kept for compatibility) ────────────────────────────

@router.post('/monitor/start')
async def monitor_start():
    # Legacy no-op: subscriptions are now managed automatically per-consumer.
    return {'code': 200, 'message': 'auto-managed'}


@router.post('/monitor/stop')
async def monitor_stop():
    # Legacy no-op: primary subs stay alive as long as topic is registered.
    return {'code': 200, 'message': 'auto-managed'}


@router.get('/monitor')
async def monitor_status():
    return {'code': 200, 'data': {'monitoring': len(_active_primary_subs) > 0}}


# ── HTTP: Topics ───────────────────────────────────────────────────────────────

@router.post('/topics/register')
async def register_topic(payload: dict):
    topic     = payload.get('topic', '').strip()
    fmt       = payload.get('format', '')
    mcp_id    = payload.get('mcp_id', '')
    if not topic:
        raise fastapi.HTTPException(status_code=400, detail='topic is required')
    await register_topic_internal(topic, fmt, mcp_id)
    return {'code': 200, 'topic': topic}


@router.get('/topics')
async def list_topics():
    items = [
        {'topic': t, **info, 'status': _topic_status(t)}
        for t, info in _topic_registry.items()
    ]
    return {'code': 200, 'data': items}


@router.get('/topics/status')
async def topics_status():
    data = {t: _topic_status(t) for t in _topic_registry}
    return {'code': 200, 'data': data}


@router.get('/topics/subscriptions')
async def topics_subscriptions():
    """Debug: 返回当前 topic_subscriber 的实际订阅列表和 config 中的 subscribe_topics。"""
    import topic_subscriber
    import config as cfg
    return {
        'code': 200,
        'data': {
            'active_subscriptions': list(topic_subscriber._subscriptions.keys()),
            'config_subscribe_topics': cfg.main.get('event', {}).get('subscribe_topics', []),
        }
    }


# ── WebSocket relay ────────────────────────────────────────────────────────────

@ws_router.websocket('/ws/bus/{topic:path}')
async def bus_ws(websocket: fastapi.WebSocket, topic: str):
    await websocket.accept()

    topic = '/' + topic  # restore leading /

    info = _topic_registry.get(topic)
    if not info:
        await websocket.send_text(json.dumps({
            'type':    'error',
            'message': f'Topic {topic} not registered',
        }))
        await websocket.close()
        return

    fmt = info['format']
    loop = asyncio.get_event_loop()

    # Ensure DDS subscription is active (no-op if already running)
    _ensure_primary_sub(topic, fmt, loop)

    await websocket.send_text(json.dumps({
        'type':   'meta',
        'ts':     time.time(),
        'topic':  topic,
        'format': fmt,
    }))

    # Push cached snapshot immediately (so page refresh shows current map)
    snapshot = _last_frame.get(topic)
    if snapshot:
        if fmt in ('sensor/pointcloud', 'sensor/mapping'):
            await websocket.send_bytes(snapshot)
        elif fmt.startswith('data/') or fmt.startswith('text/') or fmt.startswith('sensor/'):
            await websocket.send_text(snapshot.decode('utf-8', errors='replace'))
        else:
            await websocket.send_bytes(snapshot)

    q: asyncio.Queue = asyncio.Queue(maxsize=4096)
    _topic_queues.setdefault(topic, []).append(q)

    try:
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=5.0)
                if data is None:
                    break  # signal to stop
                if fmt in ('sensor/pointcloud', 'sensor/mapping'):
                    await websocket.send_bytes(data)
                elif fmt.startswith('data/') or fmt.startswith('text/') or fmt.startswith('sensor/'):
                    await websocket.send_text(data.decode('utf-8', errors='replace'))
                else:
                    await websocket.send_bytes(data)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({'type': 'ping', 'ts': time.time()}))
                except Exception:
                    break
    except (fastapi.WebSocketDisconnect, Exception):
        pass
    finally:
        queues = _topic_queues.get(topic, [])
        if q in queues:
            queues.remove(q)
