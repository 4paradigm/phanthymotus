"""
deploy_stream.py — WebSocket endpoint for real-time deployment progress.

Provides /ws/deploy/{driver_id} that streams:
- Pre-flight checks (disk space, network, registry auth)
- Pull progress (layer download %, speed, ETA)
- Deployment steps (compose merge, container start)
- Errors with actionable recovery suggestions

Usage from drivers.py:
    run_id = begin_run(driver_id, image)        # in the request handler
    async with DeployProgress(driver_id, image, run_id=run_id) as progress:
        await progress.check('disk', '检查磁盘空间…')
        await progress.update('pull', '拉取镜像层…', percent=45, speed='2.3 MB/s')
        await progress.error('disk', '磁盘空间不足', suggestion='运行 docker image prune -a')
        await progress.done('部署完成')

Why runs exist
--------------
Every event is also appended to the run's own buffer and replayed to a client
that connects late. Without this, a client raced the deployment it had just
asked for: the browser opens the socket and fires the POST concurrently, and a
deploy whose image is already local finishes in ~2s — so the preflight checks,
and sometimes everything, were emitted before the socket had registered and
were dropped on the floor. The same window then looked "simple" (a bar and one
line) purely as a function of timing.

Replay needs the run id to stay honest: a client that asks for run N must not be
shown run N-1's buffered `done` (it would auto-close over a deploy that is still
starting), so both the replay and the live feed are filtered on the id the
client asked for. A client that passes no id (the page-load restore path) takes
whatever the driver's current run is.
"""

import asyncio
import itertools
import json
import time
from typing import Set, Optional
from contextlib import asynccontextmanager

import fastapi

router = fastapi.APIRouter(tags=['deploy'])

# Per-driver deployment streams: {driver_id: set(queue)}. Queue items are
# (run_id, message) so the socket can drop events from a run this client did
# not ask about.
_streams: dict[str, Set[asyncio.Queue]] = {}

# Latest run per driver: {driver_id: {'run_id', 'image', 'start_ts', 'events',
# 'active'}}. One record per driver — a finished run is kept (so a window
# opened right after completion still shows how it went) until the next run on
# the same driver replaces it.
_runs: dict[str, dict] = {}

_run_seq = itertools.count(1)

# Cap on replayed events per run. Pull progress is already throttled to 2/s, so
# this holds several minutes of a slow pull; the oldest lines are dropped first.
HISTORY_MAX = 600


def begin_run(driver_id: str, image: str = '') -> str:
    """Open a run and return its id. Call this in the request handler, before
    the background task starts, so the id can be handed to the client in the
    POST response and used to filter its socket."""
    run_id = f'{driver_id}:{int(time.time() * 1000)}:{next(_run_seq)}'
    _runs[driver_id] = {
        'run_id':   run_id,
        'image':    image,
        'start_ts': time.time(),
        'events':   [],
        'active':   True,
    }
    return run_id


def end_run(driver_id: str) -> None:
    """Mark the driver's current run finished. The buffer is kept for replay."""
    run = _runs.get(driver_id)
    if run:
        run['active'] = False


def emit(driver_id: str, data: dict, run_id: str = '') -> None:
    """Fan out one event. Must run on the event loop thread — use
    emit_threadsafe from a worker thread."""
    run = _runs.get(driver_id)
    if not run_id:
        run_id = run['run_id'] if run else ''

    data.setdefault('ts', time.time())
    if run and run['run_id'] == run_id:
        data.setdefault('elapsed', round(data['ts'] - run['start_ts'], 1))
    data['run_id'] = run_id

    message = json.dumps(data, ensure_ascii=False)

    if run and run['run_id'] == run_id:
        run['events'].append(message)
        if len(run['events']) > HISTORY_MAX:
            del run['events'][:len(run['events']) - HISTORY_MAX]

    queues = _streams.get(driver_id, set())
    dead = set()
    for q in queues:
        try:
            q.put_nowait((run_id, message))
        except asyncio.QueueFull:
            dead.add(q)
    queues.difference_update(dead)


def emit_threadsafe(loop, driver_id: str, data: dict, run_id: str = '') -> None:
    """emit() from a worker thread. asyncio.Queue.put_nowait wakes a getter
    future, which is not thread-safe, so the call has to be handed back."""
    try:
        loop.call_soon_threadsafe(emit, driver_id, data, run_id)
    except RuntimeError:
        pass  # loop already closed — the client is gone anyway


class DeployProgress:
    """Context manager for deployment progress streaming."""

    def __init__(self, driver_id: str, image: str = '', run_id: str = ''):
        self.driver_id = driver_id
        self.image = image
        self.run_id = run_id
        self.start_ts = time.time()

    async def __aenter__(self):
        if not self.run_id:
            # No handler allocated one (tests, legacy callers) — open it here.
            self.run_id = begin_run(self.driver_id, self.image)
        run = _runs.get(self.driver_id)
        if run and run['run_id'] == self.run_id:
            self.start_ts = run['start_ts']

        await self._push({
            'type': 'start',
            'message': '开始部署…',
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.error('deploy', f'部署异常: {exc_val}')
        end_run(self.driver_id)
        return False

    async def _push(self, data: dict):
        """Push event to all clients watching this driver."""
        emit(self.driver_id, data, self.run_id)

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
async def deploy_ws(driver_id: str, websocket: fastapi.WebSocket, run: str = ''):
    """WebSocket endpoint for streaming deployment progress.

    `run` is the id returned by the deploy POST. Pass it to receive exactly that
    run — including whatever it already emitted before this socket opened. Omit
    it to follow whatever the driver is doing now.
    """
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

    # Replay what this run already emitted. Registering the queue first means an
    # event landing mid-replay is buffered rather than lost; it may duplicate a
    # replayed line, which the UI tolerates (both carry the same text) where a
    # hole in the sequence would not be.
    run_record = _runs.get(driver_id)
    if run_record and (not run or run == run_record['run_id']):
        for message in list(run_record['events']):
            await websocket.send_text(message)

    try:
        while True:
            try:
                event_run, message = await asyncio.wait_for(queue.get(), timeout=5.0)
                if run and event_run and event_run != run:
                    continue
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


@router.get('/deploying')
async def get_active_deployments():
    """Get list of currently active deployments.

    Returns:
        {
            'deployments': [
                {
                    'driver_id': 'perception',
                    'run_id': 'perception:1758000000000:3',
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
    for driver_id, run in _runs.items():
        if not run.get('active'):
            continue
        deployments.append({
            'driver_id': driver_id,
            'run_id':    run.get('run_id', ''),
            'status':    'deploying',
            'image':     run.get('image', ''),
            'start_ts':  run.get('start_ts', 0),
            'elapsed':   round(now - run.get('start_ts', now), 1),
        })
    return {'deployments': deployments}
