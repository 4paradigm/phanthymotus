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
    _push_count = [0]
    async def _push(data: bytes, msg_fmt: str):
        # Cache latest frame for snapshot push on new connections
        _last_frame[topic] = data
        queues = _topic_queues.get(topic, [])
        _push_count[0] += 1
        if topic == '/remote_control/mic' and _push_count[0] <= 5:
            print(f'[inspection] DEBUG push #{_push_count[0]} to {topic}: {len(data)}B, {len(queues)} consumers')
        for q in list(queues):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass  # drop frame for slow consumer
        # 处理 perf_span 类型消息（来自 perception TTS 等组件）
        if topic == '/perception/perf_spans':
            try:
                import json, perf_log, config
                text = data.decode('utf-8') if isinstance(data, bytes) else data
                span_data = json.loads(text)
                if span_data.get('type') == 'perf_span':
                    trace_id = span_data.pop('trace_id', None)
                    if trace_id:
                        # 直接归属（trace_id 由 agent-core 透传）
                        perf_log.commit_spans(trace_id, [span_data], source='perception')
                    else:
                        # 兼容旧版本 perception: fallback 到时间窗口匹配
                        span_start = span_data.get('start_ts', 0)
                        conn = config._get_conn()
                        row = conn.execute(
                            '''SELECT trace_id FROM perf_spans
                               WHERE span LIKE 'tool:tts%' AND start_ts <= ? AND start_ts >= ? - 30
                               ORDER BY start_ts DESC LIMIT 1''',
                            (span_start, span_start)
                        ).fetchone()
                        conn.close()
                        if row:
                            perf_log.commit_spans(row[0], [span_data], source='perception')
                    if row:
                        perf_log.commit_spans(row[0], [span_data], source='perception')
            except Exception:
                pass
    return _push


def _ensure_primary_sub(topic: str, fmt: str, loop: asyncio.AbstractEventLoop):
    """Start primary ROS2 subscription only if not already active."""
    key = f'__primary__#{topic}'

    if topic in _active_primary_subs:
        return  # already subscribed

    _active_primary_subs.add(topic)  # mark immediately to prevent race
    ros2_bridge.subscribe(key, topic, fmt, loop, _push_factory(topic))
    print(f'[inspection] started primary sub: {topic}')


# ── Internal API (called by mcp_manage directly) ───────────────────────────────

async def register_topic_internal(topic: str, fmt: str, mcp_id: str) -> None:
    """Register a topic in the registry; always check primary sub health."""
    if not topic:
        return
    existing = _topic_registry.get(topic)
    if not (existing and existing.get('format') == fmt and existing.get('mcp_id') == mcp_id):
        _topic_registry[topic] = {
            'format':        fmt,
            'mcp_id':        mcp_id,
            'registered_at': time.time(),
        }
        print(f'[inspection] registered topic={topic} format={fmt} mcp_id={mcp_id}')
    # Always check health of primary sub (even on duplicate registration)
    loop = asyncio.get_event_loop()
    _ensure_primary_sub(topic, fmt, loop)


async def publish_to_topic(topic: str, data: str | bytes) -> None:
    """Publish data to a topic via DDS (ros2_bridge).
    Data reaches dashboard via DDS → inspection primary sub → WebSocket.
    Data reaches decision_core via DDS → topic_subscriber → event_bus."""
    if topic not in _topic_registry:
        await register_topic_internal(topic, 'data/json', 'agentcore')
    text = data.decode('utf-8', errors='replace') if isinstance(data, bytes) else data

    # Publish to DDS — inspection sub will push to WebSocket, topic_subscriber to event_bus
    import ros2_bridge
    ros2_bridge.publish(topic, text)


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
    import auth
    if not auth.check_ws_token(websocket):
        await websocket.close(code=4003, reason='Owner role required')
        return
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
    except (fastapi.WebSocketDisconnect, asyncio.CancelledError, Exception):
        pass
    finally:
        queues = _topic_queues.get(topic, [])
        if q in queues:
            queues.remove(q)
