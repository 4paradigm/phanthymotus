"""
test_sse_subscription_lifecycle.py — one SSE subscription per device, and none at
all against a server that has no SSE endpoint.

Background (Tianyi, 2026-09-18): `_connect_one` ended with a bare
`asyncio.create_task(_subscribe_sse(...))`. Nothing held the task, so nothing
could cancel it — and `_connect_one` is not a startup-only call: the heartbeat
branch in `api/mcp_manage.py` re-runs it whenever the registry entry is missing
`input_schemas`. Every repeat leaked one immortal polling loop.

The driver log is where it showed up: `GET /mcp/sse → 404` at a rate that climbed
every hour — 39k, 53k, 60k, 67k, 74k per hour — and collapsed to zero the moment
agent-core restarted, then climbed again. 310262 of that container's 315088 log
lines (98.5%) were this one request. Heartbeat every 30s ⇒ +120 tasks/hour, each
backing off to a 60s poll ⇒ +2 req/s per hour, which is the slope observed.

Two independent things were wrong, so both are tested:
  1. nobody cancelled the previous subscription (the leak)
  2. a 404 was retried forever, and only 4 of the 15 drivers in
     phanthymotus-driver serve `/mcp/sse` at all, so 404 is the normal answer,
     not a fault

Run: cd agent-core && python3 -m pytest tests/test_sse_subscription_lifecycle.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_tasks():
    mcp_client._sse_tasks.clear()
    yield
    mcp_client.stop_sse()
    mcp_client._sse_tasks.clear()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _drain():
    """Cancel every subscription and let the loop reap it.

    Without this the tasks are still pending when run_until_complete closes the
    loop, and asyncio prints "Task was destroyed but it is pending!" — noise that
    makes a passing run look broken.
    """
    tasks = [t for t in mcp_client._sse_tasks.values() if not t.done()]
    mcp_client.stop_sse()
    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass


# ── the leak ─────────────────────────────────────────────────────────────────

def test_restarting_a_subscription_cancels_the_previous_one(monkeypatch):
    async def go():
        started = []

        async def _never_ends(mcp_id, url):
            started.append(url)
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_client, '_subscribe_sse', _never_ends)
        mcp_client._start_sse('dev-1', 'http://x/mcp')
        first = mcp_client._sse_tasks['dev-1']
        await asyncio.sleep(0)

        mcp_client._start_sse('dev-1', 'http://x/mcp')
        second = mcp_client._sse_tasks['dev-1']
        await asyncio.sleep(0)

        assert first is not second
        assert first.cancelled() or first.done(), 'the old loop was left running'
        assert len(mcp_client._sse_tasks) == 1, 'one device must hold one task'
        assert len(started) == 2
        await _drain()

    _run(go())


def test_many_reconnects_leave_exactly_one_live_task(monkeypatch):
    """The heartbeat calls this ~120 times an hour. It must not accumulate."""
    async def go():
        async def _never_ends(mcp_id, url):
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_client, '_subscribe_sse', _never_ends)
        for _ in range(50):
            mcp_client._start_sse('dev-1', 'http://x/mcp')
            await asyncio.sleep(0)

        live = [t for t in asyncio.all_tasks() if (t.get_name() or '').startswith('sse:')
                and not t.done()]
        assert len(live) == 1, f'{len(live)} subscription loops still running'
        await _drain()

    _run(go())


def test_two_devices_keep_their_own_subscriptions(monkeypatch):
    async def go():
        async def _never_ends(mcp_id, url):
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_client, '_subscribe_sse', _never_ends)
        mcp_client._start_sse('dev-1', 'http://a/mcp')
        mcp_client._start_sse('dev-2', 'http://b/mcp')
        await asyncio.sleep(0)
        assert set(mcp_client._sse_tasks) == {'dev-1', 'dev-2'}
        assert not mcp_client._sse_tasks['dev-1'].done()
        await _drain()

    _run(go())


def test_stop_sse_cancels_one_or_all(monkeypatch):
    async def go():
        async def _never_ends(mcp_id, url):
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_client, '_subscribe_sse', _never_ends)
        mcp_client._start_sse('dev-1', 'http://a/mcp')
        mcp_client._start_sse('dev-2', 'http://b/mcp')
        await asyncio.sleep(0)

        mcp_client.stop_sse('dev-1')
        assert set(mcp_client._sse_tasks) == {'dev-2'}

        mcp_client.stop_sse()
        assert mcp_client._sse_tasks == {}

    _run(go())


# ── 404 means "this server has no SSE endpoint" ──────────────────────────────

class _Resp:
    def __init__(self, status):
        self.status = status
        self.content = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    """Counts GETs and always answers with `status`."""

    def __init__(self, status, counter):
        self._status = status
        self._counter = counter

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, url):
        self._counter.append(url)
        return _Resp(self._status)


def _patch_session(monkeypatch, status, counter):
    monkeypatch.setattr(mcp_client.aiohttp, 'ClientSession',
                        lambda *a, **k: _Session(status, counter))
    # Collapse the backoff, but still yield to the loop — a sleep that never
    # suspends would spin the while-loop forever and starve the test's own timeout.
    real_sleep = asyncio.sleep

    async def _fast(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(mcp_client.asyncio, 'sleep', _fast)


def test_404_gives_up_instead_of_polling_forever(monkeypatch):
    calls = []
    _patch_session(monkeypatch, 404, calls)
    _run(asyncio.wait_for(mcp_client._subscribe_sse('dev-1', 'http://x/mcp'), timeout=5))
    assert len(calls) == 2, f'expected to stop after 2 probes, made {len(calls)}'
    assert calls[0] == 'http://x/mcp/sse'


def test_500_keeps_retrying(monkeypatch):
    """A server error is 'not right now', not 'no such endpoint'."""
    calls = []
    _patch_session(monkeypatch, 503, calls)

    async def go():
        task = asyncio.ensure_future(mcp_client._subscribe_sse('dev-1', 'http://x/mcp'))
        for _ in range(20):          # let it go round a few times
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _run(go())
    assert len(calls) > 2, f'gave up on a 5xx after {len(calls)} probes'
