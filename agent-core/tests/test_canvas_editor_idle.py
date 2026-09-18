"""Regression: the canvas edit lock must auto-release after 60s of idleness.

Reported 2026-09-14 — "抢到锁不释放". A tab that claimed the canvas held it until it
was closed, and everyone else stayed read-only indefinitely.

`_EDITOR_TIMEOUT = 60.0` existed the whole time but could never fire, for two
independent reasons, either one sufficient:

  1. `_check_editor_expired` returned early whenever the holder had a live
     /ws/motus connection — i.e. whenever its tab was open.
  2. The frontend's unconditional 10s `GET /edit-status` poll refreshed
     `_editor_last_seen`, so even a WS-less holder renewed forever.

The lock is now idle-timed: only real actions (claim, layout write, /keep-edit)
push `_editor_last_seen`, and nothing vetoes expiry. WS liveness still exists but
only to release *faster* than the TTL when a tab goes away.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_canvas_editor_idle.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import canvas  # noqa: E402

HOLDER = 'sess-holder'
OTHER = 'sess-other'


class Clock:
    """Fake monotonic clock — the TTL is 60s and the suite must not sleep it.

    Substituted for canvas's `time` *module reference* rather than patching
    time.monotonic itself: that attribute is global, and asyncio's event loop reads
    it, so patching it would freeze the real sleeps the grace-period tests need.
    """

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def env(monkeypatch):
    """Fresh lock state, a fake clock, and a record of every broadcast."""
    events = []

    clock = Clock()
    monkeypatch.setattr(canvas, 'time', clock)
    monkeypatch.setattr(canvas, '_broadcast', lambda ev: events.append(ev))
    monkeypatch.setattr(canvas, '_editor_session', None)
    monkeypatch.setattr(canvas, '_editor_last_seen', 0.0)
    monkeypatch.setattr(canvas, '_last_release_reason', '')
    monkeypatch.setattr(canvas, '_live_sessions', {})

    return clock, events


def reasons(events):
    return [e['payload']['reason'] for e in events if e['type'] == 'canvas_editor']


def claim(session_id):
    return asyncio.run(canvas.claim_edit({'session_id': session_id}))


def keep(session_id):
    return asyncio.run(canvas.keep_edit({'session_id': session_id}))


def status(session_id=''):
    return asyncio.run(canvas.edit_status(session_id))


def test_idle_holder_with_live_ws_expires(env):
    """The actual bug: an open tab is not an active editor."""
    clock, events = env
    claim(HOLDER)
    canvas.session_connected(HOLDER)        # tab is open, WS is up
    assert canvas.current_editor() == HOLDER

    clock.advance(61)

    assert canvas.current_editor() is None
    assert 'idle' in reasons(events)
    # The tab is still connected — liveness must not have been used to veto expiry.
    assert canvas._live_sessions.get(HOLDER) == 1


def test_status_poll_does_not_renew(env):
    """The 10s poll is a read. If it renewed, an open tab would never time out."""
    clock, _ = env
    claim(HOLDER)

    for _ in range(6):                      # six polls across 54s, as the UI does
        clock.advance(9)
        assert status(HOLDER)['editor'] == HOLDER

    clock.advance(7)                        # 61s since the claim, zero real actions
    assert canvas.current_editor() is None


def test_keep_edit_renews(env):
    """Real interaction holds the lock indefinitely."""
    clock, _ = env
    claim(HOLDER)

    for _ in range(3):
        clock.advance(30)
        assert keep(HOLDER)['editor'] == HOLDER

    assert canvas.current_editor() == HOLDER   # 90s elapsed, still editing

    clock.advance(61)                          # stop interacting
    assert canvas.current_editor() is None


def test_keep_edit_from_non_holder_is_rejected(env):
    """A ping cannot steal the lock, nor refresh someone else's."""
    clock, _ = env
    claim(HOLDER)
    clock.advance(40)

    resp = keep(OTHER)
    assert resp.status_code == 409
    assert canvas._editor_session == HOLDER

    clock.advance(21)                       # 61s since HOLDER's last real action
    assert canvas.current_editor() is None   # OTHER's ping did not renew it


def test_expired_holder_cannot_ping_its_way_back(env):
    """After expiry the client must re-claim; a late ping must not resurrect it."""
    clock, _ = env
    claim(HOLDER)
    clock.advance(61)

    resp = keep(HOLDER)
    assert resp.status_code == 409
    assert canvas._editor_session is None

    assert claim(HOLDER)['editor'] == HOLDER   # re-claiming still works


def test_status_reports_release_reason(env):
    """A client whose WS is down learns *why* it lost the lock from the poll."""
    clock, _ = env
    claim(HOLDER)
    clock.advance(61)

    body = status(HOLDER)
    assert body['editor'] is None
    assert body['reason'] == 'idle'


def test_disconnect_grace_release_still_works(env, monkeypatch):
    """Closing the tab must still free the lock well before the 60s TTL."""
    _, events = env
    monkeypatch.setattr(canvas, '_DISCONNECT_GRACE', 0.05)

    async def scenario():
        await canvas.claim_edit({'session_id': HOLDER})
        canvas.session_connected(HOLDER)
        canvas.session_disconnected(HOLDER)
        await asyncio.sleep(0)              # let _grace_release start
        assert canvas._editor_session == HOLDER   # still held inside the grace window
        await asyncio.sleep(0.15)
        return canvas._editor_session

    assert asyncio.run(scenario()) is None
    assert 'disconnect' in reasons(events)


def test_reconnect_within_grace_keeps_the_lock(env, monkeypatch):
    """A WS flap must not drop an active editor's lock."""
    _, events = env
    monkeypatch.setattr(canvas, '_DISCONNECT_GRACE', 0.05)

    async def scenario():
        await canvas.claim_edit({'session_id': HOLDER})
        canvas.session_connected(HOLDER)
        canvas.session_disconnected(HOLDER)
        canvas.session_connected(HOLDER)    # reconnected immediately
        await asyncio.sleep(0.15)
        return canvas._editor_session

    assert asyncio.run(scenario()) == HOLDER
    assert 'disconnect' not in reasons(events)
