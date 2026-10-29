"""Offline lifecycle interleavings; no MCP server, hardware or remote calls."""
import asyncio
import importlib
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import client
import event_bus
import mcp_client
from subagent.agent import Subagent
from subagent.manager import SubagentManager
from subagent.protocol import SubagentResult, SubagentSpec

llm = importlib.import_module('event.llm')


def lifecycle():
    state = {'epoch': 1, 'running': True}
    def predicate():
        return state['running'] and state['epoch'] == 1
    def restart():
        state['running'] = False
        state['epoch'] += 1
        state['running'] = True
    return predicate, restart


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr('subagent.manager.push_event', AsyncMock())
    monkeypatch.setattr('subagent.agent.Subagent.persist_turn', lambda *_: None)
    monkeypatch.setattr(Subagent, '_get_allowed_tools', lambda self: [{}])
    monkeypatch.setattr(event_bus, 'enqueue', AsyncMock())
    monkeypatch.setattr('subagent.manager.SubagentManager._save_subagent_history', lambda *_: None)
    monkeypatch.setattr('subagent.manager.SubagentStore.save_result', lambda *_: None)


async def spawn(mgr, predicate):
    token = mcp_client.dispatch_admission.set(predicate)
    try:
        return await mgr.spawn(SubagentSpec(goal='test', max_rounds=1, timeout_s=0))
    finally:
        mcp_client.dispatch_admission.reset(token)


def test_queued_stop_restart_and_independent_task(isolated, monkeypatch):
    async def scenario():
        predicate, restart = lifecycle()
        mgr = SubagentManager(None)
        old = await spawn(mgr, predicate)
        independent = await spawn(mgr, None)
        calls = AsyncMock(return_value={'content': 'independent result', 'tool_calls': []})
        monkeypatch.setattr(client, 'call', calls)
        restart()
        await mgr._schedule_next()
        await mgr._running[old]
        await asyncio.sleep(0)
        assert mgr.get_result(old).status == 'cancelled'
        calls.assert_not_awaited()
        event_bus.enqueue.assert_not_awaited()
        await mgr._schedule_next()
        await mgr._running[independent]
        await asyncio.sleep(0)
        assert mgr.get_result(independent).status == 'completed'
        calls.assert_awaited_once()
        event_bus.enqueue.assert_awaited_once()
    asyncio.run(scenario())


@pytest.mark.parametrize('error', [False, True])
def test_model_await_spanning_restart_cannot_dispatch_or_notify(isolated, monkeypatch, error):
    async def scenario():
        predicate, restart = lifecycle()
        mgr = SubagentManager(None)
        agent_id = await spawn(mgr, predicate)
        started, release = asyncio.Event(), asyncio.Event()
        async def model(**kwargs):
            assert mcp_client.dispatch_admission.get() is predicate
            started.set()
            await release.wait()
            if error:
                raise RuntimeError('model failed after restart')
            return {'tool_calls': [{'id': '1', 'function': {'name': 'peer_call', 'arguments': '{}'}}]}
        tool = AsyncMock()
        monkeypatch.setattr(client, 'call', model)
        monkeypatch.setattr(llm, '_event_instance', SimpleNamespace(_sys_tools={'peer_call': {'object': tool}}))
        await mgr._schedule_next()
        task = mgr._running[agent_id]
        await started.wait()
        restart()
        release.set()
        await task
        await asyncio.sleep(0)
        tool.assert_not_awaited()
        event_bus.enqueue.assert_not_awaited()
        assert mgr.get_result(agent_id).status in ('cancelled', 'failed')
        assert mcp_client.dispatch_admission.get() is None
    asyncio.run(scenario())


@pytest.mark.parametrize('name', ['mcp__dev__move', 'peer_call'])
def test_barrier_wait_spanning_restart_cannot_dispatch(isolated, monkeypatch, name):
    async def scenario():
        predicate, restart = lifecycle()
        agent = Subagent(SubagentSpec(goal='test'))
        agent.dispatch_admission = predicate
        started, release = asyncio.Event(), asyncio.Event()
        async def barrier(*args, **kwargs):
            started.set()
            await release.wait()
            return {'status': 'completed'}
        tool = AsyncMock()
        monkeypatch.setattr(llm, '_acp_barrier', barrier)
        monkeypatch.setattr(llm, '_needs_barrier', lambda *_: (True, None))
        monkeypatch.setattr(mcp_client, 'call_tool', tool)
        monkeypatch.setattr(llm, '_event_instance', SimpleNamespace(_sys_tools={'peer_call': {'object': tool}}))
        task = asyncio.create_task(agent._dispatch_tool(name, {}))
        await started.wait()
        restart()
        release.set()
        assert 'not dispatched' in await task
        tool.assert_not_awaited()
    asyncio.run(scenario())


def test_running_acp_result_remains_pending_after_restart(isolated, monkeypatch):
    async def scenario():
        predicate, restart = lifecycle()
        mgr = SubagentManager(None)
        agent_id = await spawn(mgr, predicate)
        agent = mgr._agents[agent_id]
        started, release = asyncio.Event(), asyncio.Event()
        pending = {'own-action': asyncio.Event()}
        monkeypatch.setattr(mcp_client, '_pending_actions', pending)
        async def tool(*_args, **_kwargs):
            started.set()
            await release.wait()
            return {'action_id': 'own-action'}
        monkeypatch.setattr(mcp_client, 'call_tool', tool)
        monkeypatch.setattr(llm, '_needs_barrier', lambda *_: (False, None))
        call = {'id': '1', 'function': {'name': 'mcp__dev__move', 'arguments': '{}'}}
        monkeypatch.setattr(client, 'call', AsyncMock(return_value={'tool_calls': [call]}))
        await mgr._schedule_next()
        task = mgr._running[agent_id]
        await started.wait()
        restart()
        release.set()
        await task
        await asyncio.sleep(0)
        assert 'own-action' in pending and not pending['own-action'].is_set()
        assert agent._action_ids == ['own-action']
        assert mgr.get_result(agent_id).status == 'cancelled'
        event_bus.enqueue.assert_not_awaited()
    asyncio.run(scenario())


def test_running_tool_result_after_restart_cannot_launch_next_tool(isolated, monkeypatch):
    async def scenario():
        predicate, restart = lifecycle()
        mgr = SubagentManager(None)
        agent_id = await spawn(mgr, predicate)
        started, release = asyncio.Event(), asyncio.Event()
        async def first():
            started.set()
            await release.wait()
            return 'done'
        second = AsyncMock()
        monkeypatch.setattr(llm, '_sys_tool_needs_barrier', lambda *_: False)
        monkeypatch.setattr(llm, '_event_instance', SimpleNamespace(_sys_tools={
            'first': {'object': first}, 'second': {'object': second}}))
        calls = [{'id': str(i), 'function': {'name': name, 'arguments': '{}'}}
                 for i, name in enumerate(('first', 'second'))]
        monkeypatch.setattr(client, 'call', AsyncMock(return_value={'tool_calls': calls}))
        await mgr._schedule_next()
        task = mgr._running[agent_id]
        await started.wait()
        restart()
        release.set()
        await task
        await asyncio.sleep(0)
        second.assert_not_awaited()
        event_bus.enqueue.assert_not_awaited()
        assert mgr.get_result(agent_id).status == 'cancelled'
    asyncio.run(scenario())


def test_stale_report_completion_and_message_fail_closed(isolated):
    async def scenario():
        predicate, restart = lifecycle()
        mgr = SubagentManager(None)
        agent_id = await spawn(mgr, predicate)
        agent = mgr._agents[agent_id]
        agent.status = 'running'
        assert mgr.accepts_message(agent_id)
        restart()
        assert not mgr.accepts_message(agent_id)
        assert not mgr.send_message(agent_id, 'new epoch data')
        assert 'not dispatched' in await agent._dispatch_tool('subagent_report', {'progress': 'stale', 'urgent': True})
        await mgr._notify_completion(agent, SubagentResult(agent_id=agent_id, status='failed', output='', error='stale'))
        event_bus.enqueue.assert_not_awaited()
        agent.dispatch_admission = lambda: 1 / 0
        assert not mgr.accepts_message(agent_id)
        # Management remains available even after admission is lost.
        assert await mgr.cancel(agent_id, 'operator request')
        assert agent._cancel_event.is_set()
    asyncio.run(scenario())
