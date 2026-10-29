"""Subagent tool dispatch must honour the same ACP results as the main loop."""
import asyncio
import importlib
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import mcp_client  # noqa: E402
from subagent.agent import Subagent  # noqa: E402
from subagent.protocol import SubagentSpec  # noqa: E402

llm = importlib.import_module('event.llm')
MCP = 'mcp__dev__move'
SYSTEM = 'peer_call'


@pytest.fixture(params=['mcp', 'system'])
def dispatch(request, monkeypatch):
    agent = Subagent(SubagentSpec(goal='test'), agent_id='barrier-test')
    tool = AsyncMock(return_value={'action_id': 'new-action'} if request.param == 'mcp' else 'ok')
    if request.param == 'mcp':
        name = MCP
        monkeypatch.setattr(mcp_client, 'call_tool', tool)
        monkeypatch.setattr(llm, '_needs_barrier', Mock(return_value=(True, frozenset({'base'}))))
    else:
        name = SYSTEM
        monkeypatch.setattr(llm, '_event_instance', SimpleNamespace(
            _sys_tools={SYSTEM: {'object': tool}}))
    return agent, name, tool


@pytest.mark.parametrize('status', ['cancelled', 'timeout', 'reconsidering', 'unknown'])
def test_noncompleted_barrier_does_not_dispatch(dispatch, monkeypatch, status):
    agent, name, tool = dispatch
    barrier_result = {'status': status, 'actions': ['previous-action']}
    barrier = AsyncMock(return_value=barrier_result)
    monkeypatch.setattr(llm, '_acp_barrier', barrier)

    result = json.loads(asyncio.run(agent._dispatch_tool(name, {'value': 1})))

    assert result == llm._not_dispatched(name, barrier_result)
    assert result['status'] == 'not_dispatched'
    assert result['dispatched'] is False
    if status == 'timeout':
        assert result['retry'] is True
    tool.assert_not_awaited()
    assert agent._action_ids == []
    expected = {'want': frozenset({'base'}), 'scoped': True, 'concurrent': False} if name == MCP else {'scoped': False}
    barrier.assert_awaited_once_with(f'subagent:{agent.id}/{name}', agent._cancel_event, **expected)


@pytest.mark.parametrize('barrier_result', [None, {'status': 'completed'}, {'status': 'no_pending'}])
def test_completed_or_no_pending_dispatches(dispatch, monkeypatch, barrier_result):
    agent, name, tool = dispatch
    barrier = AsyncMock(return_value=barrier_result)
    monkeypatch.setattr(llm, '_acp_barrier', barrier)

    result = asyncio.run(agent._dispatch_tool(name, {'value': 1}))

    tool.assert_awaited_once()
    if name == MCP:
        assert json.loads(result) == {'action_id': 'new-action'}
        assert agent._action_ids == ['new-action']
    else:
        assert result == 'ok'
    barrier.assert_awaited_once()


def test_real_barrier_no_pending_allows_already_cancelled_event(dispatch, monkeypatch):
    """A set cancellation event alone does not change no-pending barrier semantics."""
    agent, name, tool = dispatch
    monkeypatch.setattr(mcp_client, 'get_pending_actions', lambda: [])
    agent._cancel_event.set()

    result = asyncio.run(agent._dispatch_tool(name, {'value': 1}))

    tool.assert_awaited_once()
    assert result == (json.dumps({'action_id': 'new-action'}, ensure_ascii=False)
                      if name == MCP else 'ok')


def test_non_barriered_tools_keep_dispatching(dispatch, monkeypatch):
    agent, name, tool = dispatch
    barrier = AsyncMock(side_effect=AssertionError('should not wait'))
    monkeypatch.setattr(llm, '_acp_barrier', barrier)
    if name == MCP:
        monkeypatch.setattr(llm, '_needs_barrier', lambda *_: (False, None))
    else:
        monkeypatch.setattr(llm, '_sys_tool_needs_barrier', lambda *_: False)
    agent._cancel_event.set()

    asyncio.run(agent._dispatch_tool(name, {'value': 1}))

    tool.assert_awaited_once()
    barrier.assert_not_awaited()
