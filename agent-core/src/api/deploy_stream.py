"""
deploy_stream.py — WebSocket endpoint for real-time deployment progress.

Provides /ws/deploy/{driver_id} that streams:
- Pre-flight checks (disk space, network, registry auth)
- Pull progress (layer download %, speed, ETA)
- Deployment steps (compose merge, container start)
- Errors with actionable recovery suggestions

Usage from drivers.py:
    from api.deploy_stream import DeployProgress

    async with DeployProgress(driver_id) as progress:
        await progress.check('disk', '检查磁盘空间…')
        await progress.update('pull', '拉取镜像层…', percent=45, speed='2.3 MB/s')
        await progress.error('disk', '磁盘空间不足', suggestion='运行 docker image prune -a')
        await progress.done('部署完成')
"""

import asyncio
import json
import time
from typing import Set, Optional
from contextlib import asynccontextmanager

import fastapi

router = fastapi.APIRouter(tags=['deploy'])

# Per-driver deployment streams: {driver_id: set(queue)}
_streams: dict[str, Set[asyncio.Queue]] = {}

# Active deployments: {driver_id: {'status': 'deploying', 'image': '...', 'start_ts': ...}}
_active_deployments: dict[str, dict] = {}


class DeployProgress:
    """Context manager for deployment progress streaming."""

    def __init__(self, driver_id: str, image: str = ''):
        self.driver_id = driver_id
        self.image = image
        self.start_ts = time.time()

    async def __aenter__(self):
        # Mark deployment as active
        mark_deployment_start(self.driver_id, self.image)

        await self._push({
            'type': 'start',
            'message': '开始部署…',
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clear deployment state
        mark_deployment_end(self.driver_id)

        if exc_type:
            await self.error('deploy', f'部署异常: {exc_val}')
        return False

    async def _push(self, data: dict):
        """Push event to all clients watching this driver."""
        if 'ts' not in data:
            data['ts'] = time.time()
        data['elapsed'] = round(data['ts'] - self.start_ts, 1)

        message = json.dumps(data, ensure_ascii=False)
        queues = _streams.get(self.driver_id, set())
        dead = set()
        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                dead.add(q)
        queues.difference_update(dead)

    async def check(self, check_id: str, message: str, **extra):
        """Report a pre-flight check."""
        await self._push({
            'type': 'check',
            'check_id': check_id,
            'message': message,
            **extra,
        })

    async def update(self, stage: str, message: str, percent: Optional[float] = None, **extra):
        """Report progress update (pull, extract, start)."""
        data = {
            'type': 'progress',
            'stage': stage,
            'message': message,
            **extra,
        }
        if percent is not None:
            data['percent'] = round(percent, 1)
        await self._push(data)

    async def error(self, error_type: str, message: str, suggestion: Optional[str] = None, **extra):
        """Report an error with optional recovery suggestion."""
        data = {
            'type': 'error',
            'error_type': error_type,
            'message': message,
            **extra,
        }
        if suggestion:
            data['suggestion'] = suggestion
        await self._push(data)

    async def done(self, message: str = '部署完成'):
        """Report successful completion."""
        await self._push({
            'type': 'done',
            'message': message,
        })


@asynccontextmanager
async def progress_stream(driver_id: str):
    """Context manager variant that can be used without async with."""
    progress = DeployProgress(driver_id)
    await progress.__aenter__()
    try:
        yield progress
    finally:
        await progress.__aexit__(None, None, None)


@router.websocket('/ws/deploy/{driver_id}')
async def deploy_ws(driver_id: str, websocket: fastapi.WebSocket):
    """WebSocket endpoint for streaming deployment progress."""
    # Token auth check
    import auth
    if not auth.check_ws_token(websocket):
        await websocket.close(code=4001, reason='Unauthorized')
        return

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)

    # Register this client
    if driver_id not in _streams:
        _streams[driver_id] = set()
    _streams[driver_id].add(queue)

    # Send connection confirmation
    await websocket.send_text(json.dumps({
        'type': 'connected',
        'ts': time.time(),
        'driver_id': driver_id,
    }))

    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_text(message)
            except asyncio.TimeoutError:
                # Keepalive ping
                try:
                    await websocket.send_text(json.dumps({'type': 'ping', 'ts': time.time()}))
                except Exception:
                    break
    except (fastapi.WebSocketDisconnect, asyncio.CancelledError, Exception):
        pass
    finally:
        if driver_id in _streams:
            _streams[driver_id].discard(queue)
            if not _streams[driver_id]:
                del _streams[driver_id]


@router.get('/api/deploying')
async def get_active_deployments():
    """Get list of currently active deployments.

    Returns:
        {
            'deployments': [
                {
                    'driver_id': 'perception',
                    'status': 'deploying',
                    'image': 'bj-warehouse.../perception:release...',
                    'start_ts': 1234567890,
                    'elapsed': 45.2
                }
            ]
        }
    """
    now = time.time()
    deployments = []
    for driver_id, info in _active_deployments.items():
        deployments.append({
            'driver_id': driver_id,
            'status': info.get('status', 'deploying'),
            'image': info.get('image', ''),
            'start_ts': info.get('start_ts', 0),
            'elapsed': round(now - info.get('start_ts', now), 1),
        })
    return {'deployments': deployments}


def mark_deployment_start(driver_id: str, image: str):
    """Mark a deployment as started (called from drivers_async.py)."""
    _active_deployments[driver_id] = {
        'status': 'deploying',
        'image': image,
        'start_ts': time.time(),
    }


def mark_deployment_end(driver_id: str):
    """Mark a deployment as finished (called from drivers_async.py)."""
    if driver_id in _active_deployments:
        del _active_deployments[driver_id]
