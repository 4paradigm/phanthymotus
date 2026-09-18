"""Regression tests for the deploy progress stream's run buffer.

The bug: DeployProgress fanned events out only to the sockets registered at the
instant of the push, and kept no history. The browser opens the socket and fires
the deploy POST concurrently, so a deploy whose image is already present locally
— which finishes in ~2s — emitted its preflight checks, its pull lines and
sometimes its `done` before the socket had registered. Those events were dropped
and the window showed a bar and a couple of lines: the "simple" rendering users
reported, purely as a function of timing.

Replaying a buffer fixes that but introduces a second trap: replaying the
*previous* run's buffer to a client waiting on a new one hands it a stale `done`,
and the UI auto-closes over a deploy that is still starting. Hence run ids, and
hence the second and fourth tests here.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest agent-core/tests/test_deploy_run_replay.py -q

(pytest-asyncio is not in play — PYTEST_DISABLE_PLUGIN_AUTOLOAD is required for
this suite to collect at all — so each test drives its own loop, as the rest of
the async tests in this directory do.)
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import deploy_stream  # noqa: E402
from api.deploy_stream import DeployProgress, begin_run, emit, emit_threadsafe  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    deploy_stream._runs.clear()
    deploy_stream._streams.clear()
    yield
    deploy_stream._runs.clear()
    deploy_stream._streams.clear()


def _types(messages):
    return [json.loads(m)['type'] for m in messages]


def _run_events(driver_id):
    return deploy_stream._runs[driver_id]['events']


def test_events_survive_a_client_that_never_connected():
    """A whole deploy that runs before any socket registers is still readable."""
    async def scenario():
        async with DeployProgress('perception', 'img:tag') as progress:
            await progress.check('disk', '✓ 磁盘空间充足', status='pass')
            await progress.update('pull', '镜像拉取完成', percent=100)
            await progress.done('部署完成')

    asyncio.run(scenario())
    assert _types(_run_events('perception')) == ['start', 'check', 'progress', 'done']


def test_new_run_does_not_inherit_the_previous_runs_buffer():
    """Otherwise a client attaching to run 2 replays run 1's `done` and closes."""
    async def scenario():
        async with DeployProgress('perception') as first:
            await first.done('部署完成')
        assert 'done' in _types(_run_events('perception'))

        run_id = begin_run('perception', 'img:new')
        assert _run_events('perception') == []

        async with DeployProgress('perception', 'img:new', run_id=run_id) as second:
            await second.update('pull', '拉取中', percent=10)
        assert _types(_run_events('perception')) == ['start', 'progress']

    asyncio.run(scenario())


def test_finished_run_is_kept_for_replay():
    """A window opened just after a fast deploy still gets the whole story —
    the run is only dropped when the next one on that driver replaces it."""
    async def scenario():
        async with DeployProgress('perception') as progress:
            await progress.done('部署完成')

    asyncio.run(scenario())
    run = deploy_stream._runs['perception']
    assert run['active'] is False
    assert _types(run['events']) == ['start', 'done']


def test_live_events_carry_the_run_id_the_socket_filters_on():
    async def scenario():
        run_id = begin_run('perception', 'img:tag')
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        deploy_stream._streams['perception'] = {queue}

        emit('perception', {'type': 'progress', 'message': 'x'}, run_id)
        event_run, message = queue.get_nowait()

        assert event_run == run_id
        assert json.loads(message)['run_id'] == run_id

    asyncio.run(scenario())


def test_stale_run_id_is_filtered_out():
    """A socket asking for a run that never started must stay empty rather than
    be handed whatever the driver happens to be doing."""
    async def scenario():
        async with DeployProgress('perception') as old:
            await old.done('部署完成')

        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        deploy_stream._streams['perception'] = {queue}
        wanted = 'perception:0:999'   # never begun

        current = deploy_stream._runs['perception']
        # The replay condition, as the websocket handler applies it.
        assert current['run_id'] != wanted

        emit('perception', {'type': 'done', 'message': '部署完成'}, current['run_id'])
        event_run, _ = queue.get_nowait()
        assert event_run != wanted     # handler skips it

    asyncio.run(scenario())


def test_emit_threadsafe_delivers_from_a_worker_thread():
    """The pull loop runs in an executor; asyncio.Queue.put_nowait wakes a
    getter future and is not thread-safe, so it has to be handed back."""
    async def scenario():
        run_id = begin_run('core', 'img:tag')
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        deploy_stream._streams['core'] = {queue}
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(
            None, emit_threadsafe, loop, 'core',
            {'type': 'progress', 'stage': 'pull', 'message': '拉取镜像', 'percent': 42}, run_id,
        )
        event_run, message = await asyncio.wait_for(queue.get(), timeout=2)

        assert event_run == run_id
        assert json.loads(message)['percent'] == 42

    asyncio.run(scenario())
