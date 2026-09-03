"""Regression: `speak` → `finish` must not end the turn before the audio plays.

Observed on G1 in an exhibition tour: a ~47s narration was cut off because the
model called `finish` about 4s after `speak`. The reported root cause — perception
firing ACP `completed` at push-EOF instead of playback end — accounts for about
1s (perception's 500ms prebuffer plus the speaker's 300ms PREFILL_BYTES and 240ms
MAX_LEAD_S); the push loop itself is paced to the audio clock, so pushing 47s of
audio takes 47s.

The real hole is that `finish` never reached the ACP barrier. `_dispatch` matches
`name in self._sys_tools` *before* `name.startswith('mcp__')`, and the only
`await_pending` call lived in the latter branch — `_needs_barrier` also returns
False for any non-`mcp__` name. There is no end-of-turn barrier either, so
`speak` → `finish` waited for nothing regardless of when `completed` arrived.

Run: cd agent-core && python3 -m pytest tests/test_acp_barrier_finish.py
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
import collector  # noqa: E402
from event.llm import (  # noqa: E402
    _abort_pending_for_barge_in,
    _acp_barrier,
    _acp_barrier_log,
    _sys_tool_needs_barrier,
)

SPEAK_ID = 'speak-deadbeef'


@pytest.fixture(autouse=True)
def clean_pending():
    """Every test starts with an empty ACP pending table and leaves one behind."""
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools):
        d.clear()
    yield
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools):
        d.clear()


@pytest.fixture(autouse=True)
def clean_steering():
    """The steering queue is module-global too — a leaked message would make the
    next test's barrier return `barge_in` immediately."""
    def _drain():
        while not collector._steering_queue.empty():
            collector._steering_queue.get_nowait()
        collector._priority_pending.clear()
    _drain()
    yield
    _drain()


@pytest.fixture(autouse=True)
def no_interrupt_hooks(monkeypatch):
    """`hooks.fire` reaches the real binding registry; keep it out of these tests."""
    import hooks
    async def _fire(hook_id, *a, **kw):
        return []
    monkeypatch.setattr(hooks, 'fire', _fire)


def _arm(action_id=SPEAK_ID, timeout=30.0, tool='tts'):
    """Register a pending ACP action the way mcp_client.call_tool would."""
    mcp_client._pending_actions[action_id] = asyncio.Event()
    mcp_client._pending_tools[action_id] = tool
    mcp_client._pending_timeouts[action_id] = timeout
    return mcp_client._pending_actions[action_id]


def _complete(action_id=SPEAK_ID, status='completed'):
    """What POST /api/acp/complete does to unblock the barrier."""
    mcp_client._pending_results[action_id] = {'action_id': action_id, 'status': status}
    mcp_client._pending_actions[action_id].set()


# ── which tools the barrier gates ────────────────────────────────────────────

def test_finish_is_gated():
    assert _sys_tool_needs_barrier('finish')


@pytest.mark.parametrize('name', [
    'task_update',      # 每站都要调；挡住它就等于每站多等一整段音频
    'task_create',
    'task_done',
    'update_memory',
    'subagent_status',
])
def test_other_system_tools_are_not_gated(name):
    assert not _sys_tool_needs_barrier(name)


# ── the barrier itself ───────────────────────────────────────────────────────
#
# finish's real call site passes barge_in=True, so these exercise that path.

def _finish_barrier(cancel_event=None, interrupt_fallback=None):
    return _acp_barrier('finish', cancel_event, barge_in=True,
                        interrupt_fallback=interrupt_fallback)


def test_no_pending_does_not_block():
    """The common case — a turn with no audio in flight must not pay anything."""
    async def scenario():
        return await asyncio.wait_for(_finish_barrier(), timeout=1)

    assert asyncio.run(scenario()) is None


def test_finish_waits_until_playback_completes():
    """The regression: finish must not return while a speak is still pending."""
    async def scenario():
        ev = _arm()
        barrier = asyncio.create_task(_finish_barrier())

        # Give the barrier a chance to run and block.
        await asyncio.sleep(0.05)
        assert not barrier.done(), 'finish returned before the speak completed'
        assert not ev.is_set()

        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    result = asyncio.run(scenario())
    assert result['status'] == 'completed'
    assert result['actions'] == [SPEAK_ID]
    assert not mcp_client._pending_actions, 'pending table must be cleared'


def test_barge_in_cancels_the_wait():
    """"别说了" must still cut through — the wait honours cancel_event."""
    async def scenario():
        _arm()
        cancel = asyncio.Event()
        barrier = asyncio.create_task(_finish_barrier(cancel))

        await asyncio.sleep(0.05)
        assert not barrier.done()

        cancel.set()
        return await asyncio.wait_for(barrier, timeout=1)

    result = asyncio.run(scenario())
    assert result['status'] == 'cancelled'
    assert not mcp_client._pending_actions


def test_barge_in_leaves_no_task_pending():
    """The barge-in path must reap the tasks it cancels, not just request it.

    `Task.cancel()` only schedules the CancelledError. `_acp_barrier` cancels its
    `await_pending` task the moment a steering message wins the race, and
    `await_pending` had no cleanup of its own for that case — so CancelledError
    propagated straight out of it and left its two inner tasks orphaned. Every
    barge-in on R1 logged:

        Task was destroyed but it is pending!
        task: <Task pending ... coro=<Event.wait() ...>>
        task: <Task pending ... coro=<await_pending.<locals>._wait_all() ...>>

    Note this needs the *steering* path, not `cancel_event`: with cancel_event set
    `await_pending` returns 'cancelled' under its own control and cleans up on the
    way out.
    """
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())

        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': '别说了'})
        result = await asyncio.wait_for(barrier, timeout=2)
        # all_tasks() is the set of *unfinished* tasks, so anything left here
        # besides this coroutine is a task the barrier abandoned mid-cancel.
        leaked = asyncio.all_tasks() - {asyncio.current_task()}
        return result, sorted(t.get_coro().__qualname__ for t in leaked)

    result, leaked = asyncio.run(scenario())
    assert result['status'] == 'barge_in'
    assert leaked == [], f'barrier abandoned cancelled tasks: {leaked}'


def test_missing_acp_callback_times_out_and_releases():
    """A completion that never arrives must not wedge the turn forever.

    This is the ACP-callback-failed path (self-signed cert, wrong
    AGENT_CORE_URL): perception logs a warning and moves on, so the barrier is
    all that is left. It has to release, and say so.
    """
    async def scenario():
        _arm(timeout=0.1)
        return await asyncio.wait_for(_finish_barrier(), timeout=2)

    result = asyncio.run(scenario())
    assert result['status'] == 'timeout'
    assert result['actions'] == [SPEAK_ID]
    assert not mcp_client._pending_actions


def test_timeout_with_a_cancel_event_is_still_reported_as_timeout():
    """Same silent timeout, but on the branch that takes a cancel_event.

    `asyncio.wait()` — unlike `wait_for` — does not raise on timeout; it returns
    with an empty `done`. The code fell through from there to the success path, so
    a barrier that waited out its full timeout returned {"status": "completed"}
    and `_acp_barrier_log` stayed silent.

    This is the branch production always takes, in every interrupt mode:
    `_run`/`_one_turn` create a fresh `cancel_event` per turn unconditionally
    (llm.py:887), and `await_pending` branches on whether that argument is None,
    not on whether it is set. The `wait_for` branch above — the one that does
    raise — is only reachable from a direct call, so
    test_missing_acp_callback_times_out_and_releases was passing on a path no
    robot runs.
    """
    async def scenario():
        _arm(timeout=0.1)
        cancel = asyncio.Event()  # armed, but nothing ever sets it
        return await asyncio.wait_for(_finish_barrier(cancel), timeout=2)

    result = asyncio.run(scenario())
    assert result['status'] == 'timeout'
    assert result['actions'] == [SPEAK_ID]
    assert not mcp_client._pending_actions


def test_barrier_uses_the_longest_pending_timeout():
    """Two utterances in flight: the barrier must outlast the slower one."""
    async def scenario():
        _arm('speak-short', timeout=0.05)
        _arm('speak-long', timeout=5.0)
        barrier = asyncio.create_task(_finish_barrier())

        # Past the short action's own timeout — the barrier must still be waiting,
        # because effective_timeout is max(...) over every pending action.
        await asyncio.sleep(0.3)
        assert not barrier.done()

        _complete('speak-short')
        _complete('speak-long')
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


# ── barge-in during the finish barrier ───────────────────────────────────────
#
# The default interrupt_mode is "steer" (collector.py:44), which parks a user
# message in _steering_queue and does NOT set cancel_event. finish's `break`
# (llm.py:1367) is ahead of the steering drain (llm.py:1377), so nothing in the
# turn will ever consume it. Without this wake-up the barrier makes the user wait
# out the whole narration — the very thing the barrier was added to protect.

def test_steering_message_releases_the_barrier():
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())

        await asyncio.sleep(0.05)
        assert not barrier.done()

        collector._steering_queue.put_nowait({'source': 'asr', 'text': '别说了'})
        return await asyncio.wait_for(barrier, timeout=2)

    result = asyncio.run(scenario())
    assert result['status'] == 'barge_in'
    assert result['actions'] == [SPEAK_ID]
    assert not mcp_client._pending_actions, 'barge-in must clear pending too'


def test_steering_message_is_left_in_the_queue():
    """The barrier only peeks. _flush_all_pending() turns it into the next
    turn's trigger — draining it here would drop the user's message on the floor."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())
        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': '别说了'})
        await asyncio.wait_for(barrier, timeout=2)

    asyncio.run(scenario())
    assert collector._steering_queue.qsize() == 1


def test_barge_in_falls_back_when_no_hook_is_bound():
    """No on_interrupt_all binding → the hardcoded tts/loco lookup must still run,
    otherwise barge-in clears pending but leaves the audio playing."""
    called = []

    async def _fallback():
        called.append(True)

    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier(interrupt_fallback=_fallback))
        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': '停'})
        return await asyncio.wait_for(barrier, timeout=2)

    assert asyncio.run(scenario())['status'] == 'barge_in'
    assert called == [True]


def test_deferred_priority_events_also_release_the_barrier():
    """`_priority_pending` holds bot-channel and queue-full events — same story."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())
        await asyncio.sleep(0.05)
        collector._priority_pending.append({'source': 'channel', 'text': 'hi'})
        return await asyncio.wait_for(barrier, timeout=2)

    assert asyncio.run(scenario())['status'] == 'barge_in'


def test_pre_existing_steering_short_circuits_immediately():
    """A message that landed before finish must not wait a poll interval either."""
    async def scenario():
        _arm(timeout=30.0)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': '停'})
        return await asyncio.wait_for(_finish_barrier(), timeout=1)

    assert asyncio.run(scenario())['status'] == 'barge_in'


def test_mid_turn_mcp_barrier_ignores_steering():
    """barge_in defaults to False: mid-turn steering is injected into the current
    turn (llm.py:1377), not treated as an abort signal. Only finish opts in."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_acp_barrier('mcp__x__navigate', None))
        collector._steering_queue.put_nowait({'source': 'asr', 'text': 'hi'})

        await asyncio.sleep(0.3)
        assert not barrier.done(), 'mid-turn barrier must not honour steering'

        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


def test_mid_turn_mcp_barrier_only_waits_for_its_tool():
    async def scenario():
        _arm('nav-action', tool='navigation')
        _arm('tts-action', tool='tts')
        _complete('nav-action')
        return await asyncio.wait_for(
            _acp_barrier(
                'mcp__x__navigate', None, tool_name='navigation'
            ),
            timeout=1,
        )

    result = asyncio.run(scenario())
    assert result['actions'] == ['nav-action']
    assert 'tts-action' in mcp_client._pending_actions


def test_abort_clears_every_pending_table():
    """A leftover in _pending_timeouts would inflate the next barrier's
    effective_timeout (mcp_client.py:610, max over all pending)."""
    async def scenario():
        _arm('speak-a', timeout=30.0)
        _arm('speak-b', timeout=40.0)
        return await _abort_pending_for_barge_in()

    result = asyncio.run(scenario())
    assert sorted(result['actions']) == ['speak-a', 'speak-b']
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools):
        assert not d


# ── attribution ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('status', ['timeout', 'cancelled', 'barge_in'])
def test_abnormal_results_are_logged(status, capsys):
    _acp_barrier_log('finish', {'status': status, 'actions': [SPEAK_ID]})
    out = capsys.readouterr().out
    assert f'barrier {status} before finish' in out
    assert SPEAK_ID in out


def test_cancelled_is_logged_without_a_trailing_none(capsys):
    """await_pending's cancel branch returns no `actions` key."""
    _acp_barrier_log('finish', {'status': 'cancelled'})
    out = capsys.readouterr().out
    assert 'barrier cancelled before finish' in out
    assert 'None' not in out


@pytest.mark.parametrize('result', [
    {'status': 'completed', 'actions': [SPEAK_ID]},
    {'status': 'no_pending'},
    None,
    'not-a-dict',
])
def test_normal_results_stay_quiet(result, capsys):
    _acp_barrier_log('finish', result)
    assert capsys.readouterr().out == ''
