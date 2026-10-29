"""Local generation fences; mocks are not remote or physical stop evidence."""
import asyncio
import importlib
import os
import pathlib
import sys
import tempfile
from collections import deque

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.gettempdir(), 'project-stop-characterization.db'))

import collector  # noqa: E402
import config  # noqa: E402
import event_bus  # noqa: E402
import hooks  # noqa: E402
import mcp_client  # noqa: E402
from api import config as api_config  # noqa: E402
from event.llm import Event as DecisionLoop  # noqa: E402


def _setup(monkeypatch):
    monkeypatch.setattr(config, 'main', {'core': {'project_running': True}, 'canvas_layout': {'cards': []}})
    monkeypatch.setattr(api_config, '_stop_teardown_task', None)
    monkeypatch.setattr(api_config, '_start_project_task', None)
    monkeypatch.setattr(api_config, 'layout_lifecycle_lock', asyncio.Lock())
    monkeypatch.setattr(collector, '_output', asyncio.Queue())
    monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
    monkeypatch.setattr(collector, '_priority_pending', deque())
    monkeypatch.setattr(collector, '_bg_buffer', deque())
    monkeypatch.setattr(collector, '_busy', False)
    monkeypatch.setattr(collector, '_cancel_event', None)
    monkeypatch.setattr(collector, '_reconsider_event', None)
    monkeypatch.setattr(collector, '_current_turn_priority', 0)
    monkeypatch.setattr(importlib.import_module('topic_actions').manager, 'deactivate', lambda: None)

    async def noop(*args, **kwargs):
        return []
    monkeypatch.setattr('api.motus_stream.push_event', noop)
    monkeypatch.setattr(hooks, 'fire', noop)


async def _stop():
    result = await asyncio.wait_for(api_config.api_stop_project(), 1)
    await asyncio.wait_for(api_config._stop_teardown_task, 1)
    assert result['ok'] and not collector.project_running()


def _trigger():
    return {'_urgent': True, '_project_epoch': collector.project_epoch()}


@pytest.mark.parametrize('active_before_stop', [False, True], ids=['queued', 'active'])
def test_stop_and_restart_fence_queued_and_active_turn(monkeypatch, active_before_stop):
    async def scenario():
        _setup(monkeypatch)
        entered, release, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []
        agent = DecisionLoop.__new__(DecisionLoop)
        agent._turns = []
        saved = []
        monkeypatch.setattr(agent, '_save_current_turn', lambda event: saved.append(list(agent._current_turn)))

        async def one_turn(_event, *, cancel_event, reconsider_event):
            agent._current_turn.append({'role': 'assistant', 'content': 'old tool request'})
            entered.set()
            await release.wait()
            calls.append((mcp_client._dispatch_is_admitted(), cancel_event.is_set()))
            attempted.set()

        monkeypatch.setattr(agent, '_one_turn', one_turn)
        collector._output.put_nowait(_trigger())
        task = None
        try:
            if active_before_stop:
                task = asyncio.create_task(agent.run_forever())
                await asyncio.wait_for(entered.wait(), 1)
            await _stop()
            config.main['core']['project_running'] = True
            if not active_before_stop:
                collector._output.put_nowait(_trigger())
                task = asyncio.create_task(agent.run_forever())
                await asyncio.wait_for(entered.wait(), 1)
            release.set()
            await asyncio.wait_for(attempted.wait(), 1)
            assert calls == [(not active_before_stop, False)]
            await asyncio.sleep(0)  # run_forever's finalization follows the mock turn
            assert bool(saved) is not active_before_stop
        finally:
            release.set()
            if task:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
    asyncio.run(scenario())


def test_event_bus_saturated_enqueue_keeps_original_epoch(monkeypatch):
    async def scenario():
        _setup(monkeypatch)
        monkeypatch.setattr(event_bus, '_queue', asyncio.Queue(maxsize=1))
        monkeypatch.setattr(event_bus, '_recent', [])
        await event_bus.enqueue('message', 'first')
        old_epoch = collector.project_epoch()
        blocked = asyncio.create_task(event_bus.enqueue('message', 'blocked'))
        await asyncio.sleep(0)
        assert not blocked.done()
        await _stop()
        config.main['core']['project_running'] = True
        assert (await event_bus.dequeue())['_project_epoch'] == old_epoch
        await asyncio.wait_for(blocked, 1)
        assert (await event_bus.dequeue())['_project_epoch'] == old_epoch
        await event_bus.enqueue('message', 'fresh')
        assert (await event_bus.dequeue())['_project_epoch'] == collector.project_epoch()
    asyncio.run(scenario())


def test_pending_and_steering_drop_old_without_losing_new(monkeypatch):
    async def scenario():
        _setup(monkeypatch)
        old = {'source': 'message', 'text': 'old', 'payload': {}, 'ts': 1, '_project_epoch': collector.project_epoch()}
        collector._priority_pending.append(old)
        collector._steering_queue.put_nowait(old)
        await _stop()
        config.main['core']['project_running'] = True
        fresh = {**old, 'text': 'new', '_project_epoch': collector.project_epoch()}
        collector._priority_pending.append(fresh)
        collector._steering_queue.put_nowait(fresh)
        assert await collector.drain_steering() == [fresh]
        result = await asyncio.wait_for(collector.next_trigger(), 1)
        assert result['_project_epoch'] == fresh['_project_epoch'] and 'new' in result['text']
        assert 'old' not in result['text']
    asyncio.run(scenario())
