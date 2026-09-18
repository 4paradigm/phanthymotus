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
# finish's real call site is now just `_acp_barrier(name, cancel_event)` — the
# barrier takes no barge-in options at all. Only an explicit interrupt
# (cancel_event) cuts a wait short; see _acp_barrier's docstring.

def _finish_barrier(cancel_event=None):
    return _acp_barrier('finish', cancel_event)


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


def test_explicit_interrupt_cancels_the_wait():
    """"别说了" must still cut through — the wait honours cancel_event.

    This is the *only* way to stop playback now. cancel_event is set by the
    interrupt/followup collector modes and by the interrupt tool; nothing else.
    """
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


def test_interrupt_leaves_no_task_pending():
    """The wait must reap whatever it abandons, not merely request cancellation.

    `Task.cancel()` only schedules the CancelledError. This used to be a race
    between the barrier and a steering watcher, and cancelling the loser orphaned
    `await_pending`'s two inner tasks on every barge-in — R1 logged
    `Task was destroyed but it is pending!` each time. The steering watcher is
    gone; this keeps the leak assertion on the surviving abort path.
    """
    async def scenario():
        _arm(timeout=30.0)
        cancel = asyncio.Event()
        barrier = asyncio.create_task(_finish_barrier(cancel))

        await asyncio.sleep(0.05)
        cancel.set()
        result = await asyncio.wait_for(barrier, timeout=2)
        # all_tasks() is the set of *unfinished* tasks, so anything left here
        # besides this coroutine is a task the barrier abandoned mid-cancel.
        leaked = asyncio.all_tasks() - {asyncio.current_task()}
        return result, sorted(t.get_coro().__qualname__ for t in leaked)

    result, leaked = asyncio.run(scenario())
    assert result['status'] == 'cancelled'
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


# ── steering must NOT cut playback short ─────────────────────────────────────
#
# There used to be a barge-in here: the finish barrier raced `await_pending`
# against a watcher on `_steering_queue`, so any queued P>0 event aborted the
# audio. Two measurements on Orin5 killed it.
#
#   17:01:48  registered pending: speak-42c1c65c (tts, 171s)   # 英伟达汇报
#   17:02:29  barrier barge_in before finish                   # 41s in, cut
#   17:02:29  received [URGENT] source=subagent:7ff3018a       # a *subagent* finishing
#
# Filtering to human sources only moved the problem: typing the next question in
# the web console then cut the previous briefing off mid-sentence. On this path
# "sent a message" and "asked it to stop" are indistinguishable — and the
# architecture already has an explicit entry point for the latter.
#
# So: messages queue. `set_busy(False)` → `_flush_all_pending()` turns them into
# the next turn's trigger when the turn ends normally. Nothing is dropped; the
# user waits out the current utterance.

def test_steering_message_does_not_release_the_barrier():
    """The regression: a queued message must not abort playback."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())

        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': '再帮我查个别的'})
        await asyncio.sleep(0.3)
        assert not barrier.done(), 'a new message cut the audio short'

        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    result = asyncio.run(scenario())
    assert result['status'] == 'completed'
    assert result['actions'] == [SPEAK_ID]


def test_subagent_notification_does_not_release_the_barrier():
    """The Orin5 case: a background task finishing cut off the previous report."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())

        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait(
            {'source': 'subagent:7ff3018a', 'text': '子代理 [7ff3018a] ✓ completed'})
        await asyncio.sleep(0.3)
        assert not barrier.done(), 'a subagent finishing cut the audio short'

        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


def test_steering_message_is_left_in_the_queue():
    """The barrier never touches the queue. `_flush_all_pending()` turns the
    message into the next turn's trigger — draining it here would drop it."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())
        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': 'hi'})
        _complete()
        await asyncio.wait_for(barrier, timeout=1)

    asyncio.run(scenario())
    assert collector._steering_queue.qsize() == 1


def test_deferred_priority_events_do_not_release_the_barrier():
    """`_priority_pending` holds bot-channel and queue-full events — same rule."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_finish_barrier())
        await asyncio.sleep(0.05)
        collector._priority_pending.append({'source': 'channel', 'text': 'hi'})
        await asyncio.sleep(0.3)
        assert not barrier.done()
        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


def test_pre_existing_steering_does_not_short_circuit():
    """A message that landed before finish must not abort the audio either."""
    async def scenario():
        _arm(timeout=30.0)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': '停'})
        barrier = asyncio.create_task(_finish_barrier())
        await asyncio.sleep(0.3)
        assert not barrier.done()
        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


def test_interrupt_still_wins_over_a_queued_message():
    """Queued messages wait; an explicit interrupt does not."""
    async def scenario():
        _arm(timeout=30.0)
        cancel = asyncio.Event()
        barrier = asyncio.create_task(_finish_barrier(cancel))
        await asyncio.sleep(0.05)
        collector._steering_queue.put_nowait({'source': 'asr', 'text': 'hi'})
        await asyncio.sleep(0.2)
        assert not barrier.done()

        cancel.set()
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'cancelled'


def test_mid_turn_mcp_barrier_ignores_steering():
    """Unchanged: mid-turn steering is injected into the current turn
    (llm.py's drain after the tool batch), never treated as an abort signal."""
    async def scenario():
        _arm(timeout=30.0)
        barrier = asyncio.create_task(_acp_barrier('mcp__x__navigate', None))
        collector._steering_queue.put_nowait({'source': 'asr', 'text': 'hi'})

        await asyncio.sleep(0.3)
        assert not barrier.done(), 'mid-turn barrier must not honour steering'

        _complete()
        return await asyncio.wait_for(barrier, timeout=1)

    assert asyncio.run(scenario())['status'] == 'completed'


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
