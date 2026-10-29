"""Characterize the existing hook check/dispatch race; no hardware is contacted.

This test intentionally asserts the *unsafe current behavior*. A future reservation
fix should replace this characterization with the desired mutual-exclusion contract.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import mcp_client


def test_concurrent_same_resource_hooks_can_both_dispatch(monkeypatch):
    for name in mcp_client._PENDING_TABLE_NAMES:
        monkeypatch.setattr(mcp_client, name, {})
    monkeypatch.setattr(mcp_client, 'registry', {
        'test-device': {'tool_meta': {'mcp__test-device__move': {
            'resource': frozenset({'base'}),
            'completion': {'actions': ['move'], 'timeout': 30},
        }}},
    })

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def direct(mcp_id, tool_name, args, *, timeout_s):
            calls.append(args['request'])
            if len(calls) == 2:
                entered.set()
            await release.wait()
            return {'action_id': args['request']}

        monkeypatch.setattr(mcp_client, 'call_tool_direct', direct)
        first = asyncio.create_task(mcp_client.call_tool_hook(
            'test-device', 'move', {'action': 'move', 'request': 'first'},
            barrier_aware=True, conservative_resources=True))
        second = asyncio.create_task(mcp_client.call_tool_hook(
            'test-device', 'move', {'action': 'move', 'request': 'second'},
            barrier_aware=True, conservative_resources=True))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert calls == ['first', 'second']
            assert not mcp_client.resource_actually_busy(frozenset({'base'}))
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
            assert results == [{'action_id': 'first'}, {'action_id': 'second'}]
            assert set(mcp_client._pending_actions) == {'first', 'second'}
            assert mcp_client.resource_actually_busy(frozenset({'base'}))
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(exercise())
