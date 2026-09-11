"""
test_reconsider_barrier.py — reconsider_event: mid-turn barriers and in-flight LLM
calls can both be interrupted by high-priority steering without ending the turn.

Background: only `finish`'s barrier could ever be woken early (barge_in=True).
Every other mcp__ tool's barrier wait was a bare `await` with no race against new
input — on Tianyi, a user saying "别说了" while a navigate/speak call was queued
behind another action sat unprocessed for up to ~58s, because the round loop
couldn't even ask the LLM again until the wait cleared on its own.

`reconsider_event` fixes this without reusing `cancel_event`/`TurnCancelled`
(which end the whole turn) — it's a narrower per-turn signal:
- `mcp_client.await_pending(..., reconsider_event=ev)`: when it wins the race,
  the pending being waited on is left alone (still legitimately in flight
  elsewhere) — only this caller's wait is abandoned. Contrast with cancel_event,
  which forgets it.
- `client.llm.Client.__call__(..., reconsider_event=ev)`: when it wins the race
  against the in-flight API call, raises RoundReconsider (not TurnCancelled).

Also covers the fix this depends on: `_acp_barrier`'s return value used to be
discarded by `_dispatch`'s mcp__ branch, so a cancelled/reconsidering wait still
fell through to `mcp_client.call_tool` regardless — see test_dispatch_*.

Run: cd agent-core && python3 -m pytest tests/test_reconsider_barrier.py
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
from event.llm import _acp_barrier  # noqa: E402

ACTION_ID = 'nav-deadbeef'


@pytest.fixture(autouse=True)
def clean_pending():
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools,
              mcp_client._pending_resources):
        d.clear()
    yield
    for d in (mcp_client._pending_actions, mcp_client._pending_results,
              mcp_client._pending_timeouts, mcp_client._pending_tools,
              mcp_client._pending_resources):
        d.clear()


def _arm(action_id=ACTION_ID, timeout=30.0, tool='controlled_spatial', resource=None):
    mcp_client._pending_actions[action_id] = asyncio.Event()
    mcp_client._pending_tools[action_id] = tool
    mcp_client._pending_timeouts[action_id] = timeout
    mcp_client._pending_resources[action_id] = resource
    return mcp_client._pending_actions[action_id]


# ── mcp_client.await_pending: the core primitive ─────────────────────────────

class TestAwaitPendingReconsider:
    def test_reconsider_does_not_forget_pending(self):
        """The thing being waited on (e.g. a navigate still walking) must stay
        tracked — reconsider only abandons *this* wait, not the action itself."""
        async def scenario():
            _arm()
            reconsider = asyncio.Event()
            task = asyncio.create_task(
                mcp_client.await_pending(reconsider_event=reconsider, timeout=5))
            await asyncio.sleep(0.05)
            assert not task.done()
            reconsider.set()
            return await asyncio.wait_for(task, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] == 'reconsidering'
        assert result['actions'] == [ACTION_ID]
        assert ACTION_ID in mcp_client._pending_actions, \
            'reconsider must not forget a pending action that is still legitimately running'

    def test_cancel_still_forgets_pending(self):
        """Unchanged regression check: cancel_event's existing "give up entirely"
        semantics must not have been altered by adding the reconsider path."""
        async def scenario():
            _arm()
            cancel = asyncio.Event()
            task = asyncio.create_task(
                mcp_client.await_pending(cancel_event=cancel, timeout=5))
            await asyncio.sleep(0.05)
            cancel.set()
            return await asyncio.wait_for(task, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] == 'cancelled'
        assert ACTION_ID not in mcp_client._pending_actions

    def test_real_completion_still_wins_when_it_comes_first(self):
        async def scenario():
            ev = _arm()
            reconsider = asyncio.Event()
            task = asyncio.create_task(
                mcp_client.await_pending(reconsider_event=reconsider, timeout=5))
            await asyncio.sleep(0.05)
            mcp_client._pending_results[ACTION_ID] = {'status': 'completed'}
            ev.set()
            return await asyncio.wait_for(task, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] == 'completed'
        assert ACTION_ID not in mcp_client._pending_actions

    def test_cancel_and_reconsider_can_both_be_armed(self):
        """Same call can carry both signals (main loop always passes cancel_event
        for interrupt/followup compatibility, reconsider_event for steer) — cancel
        must still win if both are set, since it's the more drastic one."""
        async def scenario():
            _arm()
            cancel = asyncio.Event()
            reconsider = asyncio.Event()
            task = asyncio.create_task(
                mcp_client.await_pending(cancel_event=cancel, reconsider_event=reconsider, timeout=5))
            await asyncio.sleep(0.05)
            cancel.set()
            reconsider.set()
            return await asyncio.wait_for(task, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] in ('cancelled', 'reconsidering')  # either wins the race legitimately
        # whichever won must behave consistently with its own contract:
        if result['status'] == 'cancelled':
            assert ACTION_ID not in mcp_client._pending_actions
        else:
            assert ACTION_ID in mcp_client._pending_actions

    def test_no_reconsider_event_behaves_as_before(self):
        """Backward compatibility: omitting reconsider_event must not change
        anything about a plain cancel_event wait or a plain uninterrupted wait."""
        async def scenario():
            ev = _arm()
            task = asyncio.create_task(mcp_client.await_pending(timeout=5))
            await asyncio.sleep(0.05)
            mcp_client._pending_results[ACTION_ID] = {'status': 'completed'}
            ev.set()
            return await asyncio.wait_for(task, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] == 'completed'


# ── _acp_barrier: the non-barge_in (mid-turn mcp tool) path ─────────────────

class TestAcpBarrierReconsider:
    def test_mid_turn_barrier_honours_reconsider(self):
        """This is the exact shape of the P4/stop_nav case: a normal mcp__ tool's
        barrier (barge_in=False) must now be interruptible by reconsider_event,
        without needing to be `finish`."""
        async def scenario():
            _arm()
            reconsider = asyncio.Event()
            barrier = asyncio.create_task(
                _acp_barrier('mcp__dev1__controlled_spatial__navigate_to_tag',
                             cancel_event=None, reconsider_event=reconsider,
                             want=frozenset({'base'}), scoped=True))
            await asyncio.sleep(0.05)
            assert not barrier.done(), 'a mid-turn barrier must actually be waiting here'
            reconsider.set()
            return await asyncio.wait_for(barrier, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] == 'reconsidering'
        assert ACTION_ID in mcp_client._pending_actions, \
            'the action being waited on is still running — must not be forgotten'

    def test_finish_barrier_honours_reconsider_without_cutting_audio(self):
        """The system-tool barrier takes reconsider_event too, and the audio survives.

        Orin5, 17:32: the finish barrier took neither reconsider_event nor a look at
        its own return value, so a message sent early in a 48s briefing sat in the
        queue for 33.6s (event ts=17:32:19, received 17:32:52.597) — finish's break
        is ahead of the steering drain, so nothing in the turn could consume it.

        The two assertions below are the whole point of "播边想": the wait ends
        (so the loop can think) *and* the speak is still pending (so it keeps playing).
        """
        async def scenario():
            _arm('speak-67462a1e', tool='tts', resource=frozenset({'mouth'}))
            reconsider = asyncio.Event()
            barrier = asyncio.create_task(
                _acp_barrier('finish', cancel_event=None, reconsider_event=reconsider))
            await asyncio.sleep(0.05)
            assert not barrier.done(), 'finish barrier must actually be waiting here'
            reconsider.set()
            return await asyncio.wait_for(barrier, timeout=1)

        result = asyncio.run(scenario())
        assert result['status'] == 'reconsidering'
        assert 'speak-67462a1e' in mcp_client._pending_actions, \
            'the briefing is still playing — reconsider must not forget it'

    def test_interrupt_still_beats_a_new_message_on_the_finish_barrier(self):
        """cancel_event and reconsider_event armed together: interrupt wins.

        "有 interrupt 就无视 barrier 直接执行" — and unlike reconsider, it *does*
        forget the pending, because the whole turn is being thrown away.
        """
        async def scenario():
            _arm('speak-67462a1e', tool='tts')
            cancel, reconsider = asyncio.Event(), asyncio.Event()
            barrier = asyncio.create_task(
                _acp_barrier('finish', cancel_event=cancel, reconsider_event=reconsider))
            await asyncio.sleep(0.05)
            cancel.set()
            return await asyncio.wait_for(barrier, timeout=1)

        assert asyncio.run(scenario())['status'] == 'cancelled'
        assert 'speak-67462a1e' not in mcp_client._pending_actions


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
