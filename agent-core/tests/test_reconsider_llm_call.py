"""
test_reconsider_llm_call.py — an in-flight LLM completion can be aborted by
reconsider_event without being mistaken for a full turn cancellation.

`client.llm.Client.__call__` already raced `cancel_event` against the live API
call and raised TurnCancelled (ends the whole turn — event/llm.py's run_forever
discards the turn's history and waits for a fresh trigger). reconsider_event is
the narrower sibling: same race, but raises RoundReconsider so the caller can
retry within the same turn instead.

No real endpoints are configured (config.main['client']['llm'] defaults to []),
so `task_list` is always empty and the only thing in the race is the sentinel —
this isolates the racing logic from needing to mock an actual OpenAI client.

Run: cd agent-core && python3 -m pytest tests/test_reconsider_llm_call.py
"""
import asyncio
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import client.llm  # noqa: E402
from event.llm import RoundReconsider, TurnCancelled  # noqa: E402

# `client/__init__.py` does `llm = client.llm.Client()`, rebinding the `llm`
# attribute on the `client` package to a Client *instance* — so `client.llm`
# after `import client` is no longer the submodule. Reach the actual module
# through sys.modules, same workaround event/llm.py already uses for the
# analogous event.skills shadowing.
import sys  # noqa: E402
_llm_module = sys.modules['client.llm']

import config  # noqa: E402


@pytest.fixture(autouse=True)
def empty_llm_config():
    """`Client.__call__` reads `config.main['client']['llm']` directly (not just
    at construction) — config.main is a process-wide singleton shared with every
    other test file, so pin this explicitly rather than trust whatever default
    or leftover state it's in by the time this file runs in the full suite."""
    config.main['client'] = {'llm': []}
    yield


def _client():
    """A Client with zero configured endpoints, without going through __init__ —
    config.main is a process-wide singleton shared with every other test file in
    the suite, so relying on its default ['client']['llm'] == [] is brittle
    (whichever test runs first and mutates it wins). Bypassing _init_clients()
    isolates this test from that entirely; the race being tested only needs
    task_list to be empty, not a real client construction path."""
    c = _llm_module.Client.__new__(_llm_module.Client)
    c.client_list = []
    c._endpoint_dead = []
    return c


def test_reconsider_event_raises_round_reconsider_not_turn_cancelled():
    async def scenario():
        c = _client()
        reconsider = asyncio.Event()
        reconsider.set()  # already armed before the call starts
        with pytest.raises(RoundReconsider):
            await c(message_list=[], tool_list=[], reconsider_event=reconsider)
    asyncio.run(scenario())


def test_cancel_event_still_raises_turn_cancelled():
    """Regression: adding reconsider_event must not change cancel_event's own
    contract — it still ends the whole turn."""
    async def scenario():
        c = _client()
        cancel = asyncio.Event()
        cancel.set()
        with pytest.raises(TurnCancelled):
            await c(message_list=[], tool_list=[], cancel_event=cancel)
    asyncio.run(scenario())


def test_reconsider_fires_mid_wait_not_just_when_pre_armed():
    async def scenario():
        c = _client()
        reconsider = asyncio.Event()

        async def _arm_soon():
            await asyncio.sleep(0.05)
            reconsider.set()

        asyncio.create_task(_arm_soon())
        with pytest.raises(RoundReconsider):
            await c(message_list=[], tool_list=[], reconsider_event=reconsider)
    asyncio.run(scenario())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
