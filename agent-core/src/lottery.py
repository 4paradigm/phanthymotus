"""Server-authoritative lottery action and display state.

The browser is deliberately never allowed to choose the winner.  One draw is
decided here, then announced as two snapshots on the same topic: ``spinning``
for the animation and ``result`` for the reveal.  The inspection bus caches the
last snapshot, so a display opened halfway through a draw (or reconnected after
a network gap) resumes from server state instead of inventing its own answer.
"""

import asyncio
import json
import secrets
import time
import uuid
from typing import Optional


TOPIC = '/ui/lottery'
FORMAT = 'data/lottery'
DEFAULT_OPTIONS = ['1', '2', '3', '4', '5', '6']
MIN_DURATION_MS = 500
MAX_DURATION_MS = 30_000

_draw_lock = asyncio.Lock()
_state: dict = {
    'state': 'idle',
    'mode': 'dice',
    'options': DEFAULT_OPTIONS,
}


def _settings(arguments: dict) -> tuple[str, list[str], int]:
    import config

    saved = config.main.get('tool_config:agentcore:lottery_draw', None) or {}
    values = {**saved, **{k: v for k, v in arguments.items() if v is not None}}

    mode = str(values.get('mode') or 'dice')
    if mode not in {'dice', 'card', 'number', 'options'}:
        mode = 'options'

    raw_options = values.get('options') or DEFAULT_OPTIONS
    if isinstance(raw_options, str):
        raw_options = raw_options.split(',')
    options = [str(item).strip() for item in raw_options if str(item).strip()]
    if not options:
        options = list(DEFAULT_OPTIONS)

    try:
        duration_ms = int(values.get('duration_ms', 3000))
    except (TypeError, ValueError):
        duration_ms = 3000
    duration_ms = max(MIN_DURATION_MS, min(MAX_DURATION_MS, duration_ms))
    return mode, options, duration_ms


async def _publish(payload: dict) -> None:
    from api.inspection import publish_to_topic, register_topic_internal

    await register_topic_internal(TOPIC, FORMAT, 'agentcore')
    await publish_to_topic(TOPIC, json.dumps(payload, ensure_ascii=False))


def snapshot() -> dict:
    return dict(_state)


async def handle(arguments: Optional[dict] = None) -> dict:
    """Handle the internal ``lottery_draw`` tool and return its canonical state."""
    global _state
    arguments = arguments or {}
    action = arguments.get('action', 'draw')

    if action == 'info':
        return {
            **snapshot(),
            'topic_out': [{'topic': TOPIC, 'format': FORMAT}],
        }
    if action == 'start':
        await _publish(snapshot())
        return {
            'state': 'running',
            'topic_out': [{'topic': TOPIC, 'format': FORMAT}],
        }
    if action == 'stop':
        _state = {'state': 'idle', 'mode': _state.get('mode', 'dice'),
                  'options': _state.get('options', DEFAULT_OPTIONS)}
        await _publish(_state)
        return snapshot()
    if action == 'config':
        # Persistence is owned by api.canvas. Acknowledge the same action shape
        # as every other internal tool so config application has no side effect.
        return {'state': 'configured'}
    if action != 'draw':
        return {'state': 'error', 'error': f'unknown action: {action}'}

    if _draw_lock.locked():
        return {
            'state': 'busy',
            'message': 'A lottery draw is already in progress.',
            'current': snapshot(),
        }

    async with _draw_lock:
        mode, options, duration_ms = _settings(arguments)

        # Decide once, on the server, before any animation starts.  The value is
        # intentionally absent from the spinning frame; the renderer gets a seed
        # for decoration, never authority over the outcome.
        result_index = secrets.randbelow(len(options))
        result = options[result_index]
        draw_id = f'draw-{uuid.uuid4().hex}'
        started_at = int(time.time() * 1000)
        reveal_at = started_at + duration_ms

        _state = {
            'state': 'spinning',
            'draw_id': draw_id,
            'mode': mode,
            'options': options,
            'animation_seed': secrets.randbits(32),
            'started_at': started_at,
            'reveal_at': reveal_at,
            'duration_ms': duration_ms,
        }
        await _publish(_state)
        await asyncio.sleep(duration_ms / 1000)

        _state = {
            'state': 'result',
            'draw_id': draw_id,
            'mode': mode,
            'options': options,
            'result': result,
            'result_index': result_index,
            'revealed_at': int(time.time() * 1000),
        }
        await _publish(_state)
        # MCP/canvas callers receive the exact object the displays received.
        return snapshot()
