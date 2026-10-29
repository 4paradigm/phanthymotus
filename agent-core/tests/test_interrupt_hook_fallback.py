"""Interrupt failures must reach the fallback; no physical stop is inferred."""
import asyncio
import importlib
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'interrupt-hook.db'))

import hooks
import mcp_client
from event.llm import Event


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(hooks, '_registry', {})
    monkeypatch.setattr(hooks, '_fire_log', hooks.deque(maxlen=30))
    monkeypatch.setattr(mcp_client, 'registry', {'dev': {
        'online': True, 'tools': ['tts'], 'url': 'http://example.invalid'}})
    for name in mcp_client._PENDING_TABLE_NAMES:
        monkeypatch.setattr(mcp_client, name, {})
    monkeypatch.setattr(importlib.import_module('event.llm'), '_stop_countdown', lambda: None)
    return monkeypatch


@pytest.mark.parametrize('failed', [
    {'error': 'transport failed'}, {'status': 'error'}, {'state': 'error'},
    {'ok': False}, {'isError': True}, {'skipped': 'resource busy'},
    RuntimeError('device failed'),
])
def test_failed_real_hook_dispatch_retries_fallback(rig, failed, capsys):
    calls = []
    async def direct(mid, tool, args):
        calls.append((mid, tool, args['action']))
        if len(calls) == 1:
            if isinstance(failed, Exception):
                raise failed
            return failed
        return {'state': 'idle'}
    rig.setattr(mcp_client, 'call_tool_direct', direct)
    hooks.register('dev', 'tts', {'on_interrupt_all': {'action': 'interrupt'}})
    asyncio.run(Event.__new__(Event)._interrupt_active_outputs())
    assert calls == [('dev', 'tts', 'interrupt'), ('dev', 'tts', 'interrupt')]
    assert '1/1 via fallback' in capsys.readouterr().out
    assert hooks._fire_log[-1]['errors']
    if not isinstance(failed, Exception):
        assert hooks._fire_log[-1]['errors'][0]['result'] == failed


@pytest.mark.parametrize('failed', [
    {'error': 'transport failed'}, {'status': 'error'}, {'state': 'error'},
    {'ok': False}, {'isError': True}, {'skipped': 'resource busy'},
    RuntimeError('device failed'),
])
def test_fallback_failure_not_counted_as_success_and_local_pending_only(rig, failed, capsys):
    async def direct(*args):
        if isinstance(failed, Exception):
            raise failed
        return failed
    rig.setattr(mcp_client, 'call_tool_direct', direct)
    event = asyncio.Event()
    mcp_client._pending_actions['x'] = event
    asyncio.run(Event.__new__(Event)._interrupt_active_outputs(reason='interrupt'))
    output = capsys.readouterr().out
    assert '0/1 via fallback' in output
    assert 'failed:' in output or 'raised:' in output
    assert event.is_set() and mcp_client._pending_results['x']['status'] == 'cancelled'


def test_success_hook_does_not_repeat_same_tool(rig):
    calls = []
    async def direct(mid, tool, args):
        calls.append((mid, tool, args))
        return {'state': 'idle'}
    rig.setattr(mcp_client, 'call_tool_direct', direct)
    hooks.register('dev', 'tts', {'on_interrupt_all': {'action': 'interrupt'}})
    asyncio.run(Event.__new__(Event)._interrupt_active_outputs())
    assert len(calls) == 1


def test_legacy_wrapped_failure_still_triggers_fallback(rig):
    calls = []
    async def fire(hid):
        if hid == 'on_interrupt_all':
            return [{'mcp_id': 'dev', 'tool': 'tts', 'result': {'ok': False}}]
        return []
    async def direct(*args):
        calls.append(args)
        return {'state': 'idle'}
    rig.setattr(hooks, 'fire', fire)
    rig.setattr(mcp_client, 'call_tool_direct', direct)
    asyncio.run(Event.__new__(Event)._interrupt_active_outputs())
    assert len(calls) == 1


def test_one_hook_exception_does_not_skip_other_hook_or_fallback(rig):
    fired, calls = [], []
    async def fire(hid):
        fired.append(hid)
        if hid == 'on_interrupt_all':
            raise RuntimeError('hook failure')
        return []
    async def direct(*args):
        calls.append(args)
        return {'state': 'idle'}
    rig.setattr(hooks, 'fire', fire)
    rig.setattr(mcp_client, 'call_tool_direct', direct)
    asyncio.run(Event.__new__(Event)._interrupt_active_outputs())
    assert fired == ['on_interrupt_all', 'on_interrupt_speak']
    assert len(calls) == 1
