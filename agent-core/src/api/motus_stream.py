"""
Phanthy Motus activity stream — WebSocket endpoint at /ws/motus.

Other modules push events via:
    from api.motus_stream import push_event
    await push_event({...})

Event schema:
    {
        "type":    "mcp_call | mcp_result | agent_thought | render | status",
        "ts":      <unix float>,
        "mcp_id":  "<optional mcp id>",
        "render":  "<optional: video|text|image|audio|lidar|activity>",
        "payload": { ... }
    }
"""

import asyncio
import json
import time
from typing import Set

import fastapi

router = fastapi.APIRouter(tags=['motus'])

# Set of per-client queues; each connected WebSocket gets its own queue.
_clients: Set[asyncio.Queue] = set()


async def push_event(event: dict):
    """Push an event to all connected WebSocket clients."""
    if 'ts' not in event:
        event['ts'] = time.time()
    message = json.dumps(event, ensure_ascii=False)
    dead = set()
    for q in _clients:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.add(q)
    _clients.difference_update(dead)


@router.websocket('/ws/motus')
async def motus_ws(websocket: fastapi.WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _clients.add(queue)

    # Send a welcome / connection-established event
    await websocket.send_text(json.dumps({
        'type': 'status',
        'ts':   time.time(),
        'payload': {'connected': True},
    }))

    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_text(message)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({'type': 'ping', 'ts': time.time()}))
                except Exception:
                    break
    except (fastapi.WebSocketDisconnect, Exception):
        pass
    finally:
        _clients.discard(queue)
