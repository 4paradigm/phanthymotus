"""await_pending only cleans the Event generation it actually waited for."""
import asyncio

import pytest

import mcp_client


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in mcp_client._PENDING_TABLE_NAMES:
        monkeypatch.setattr(mcp_client, name, mcp_client.OrderedDict() if name == '_pending_results' else {})
    monkeypatch.setattr(mcp_client, '_action_outcomes', mcp_client.OrderedDict())
    monkeypatch.setattr(mcp_client, '_settle_listeners', [])
    monkeypatch.setattr(mcp_client, '_register_listeners', [])


def register(aid, tool, resource):
    mcp_client._register_pending(
        aid, tool_name=tool,
        meta={'completion': {'timeout': 20}, 'resource': frozenset({resource})},
        args={}, result={}, mcp_id=tool,
    )
    return mcp_client._pending_actions[aid]


@pytest.mark.parametrize('outcome', ['completed', 'timeout', 'cancelled'])
def test_barrier_only_cleans_captured_events(monkeypatch, outcome):
    async def run():
        old = register('reused', 'old-tool', 'base')
        unchanged = register('unchanged', 'steady-tool', 'mouth')
        removed = register('removed', 'removed-tool', 'camera')
        notifications = []
        mcp_client._settle_listeners.append(lambda aid, resource: notifications.append((aid, resource)))
        entered = asyncio.Event()
        reached_cleanup = asyncio.Event()
        release_cleanup = asyncio.Event()
        cancel = asyncio.Event()
        original_wait = asyncio.wait
        original_reap = mcp_client.cancel_and_reap

        async def controlled_wait(tasks, **kwargs):
            entered.set()  # all three events have already been captured
            if outcome == 'timeout':
                return set(), set(tasks)
            return await original_wait(tasks, **kwargs)

        async def controlled_reap(tasks):
            await original_reap(tasks)
            reached_cleanup.set()
            await release_cleanup.wait()

        monkeypatch.setattr(mcp_client.asyncio, 'wait', controlled_wait)
        monkeypatch.setattr(mcp_client, 'cancel_and_reap', controlled_reap)
        barrier = asyncio.create_task(mcp_client.await_pending(cancel_event=cancel))
        await entered.wait()
        if outcome == 'completed':
            old.set()
            unchanged.set()
            removed.set()
        elif outcome == 'cancelled':
            cancel.set()
        await reached_cleanup.wait()

        # Simulate another waiter completing and removing two entries, followed
        # by a new dispatch reusing one ID while the old barrier is suspended.
        mcp_client._forget_pending(['reused', 'removed'], 'completed')
        new = register('reused', 'new-tool', 'vision')
        payload = {'status': 'failed', 'result': {'detail': 'new generation'}}
        mcp_client._pending_results['reused'] = payload
        new.set()
        before = {name: getattr(mcp_client, name)['reused']
                  for name in mcp_client._PENDING_TABLE_NAMES}
        before_outcome = dict(mcp_client.action_outcome('reused'))
        notifications.clear()
        release_cleanup.set()
        result = await barrier

        assert result['status'] == outcome
        if outcome != 'cancelled':
            assert result['actions'] == ['reused', 'unchanged', 'removed']
        assert mcp_client._pending_actions['reused'] is new
        assert new is not old
        assert {name: getattr(mcp_client, name)['reused']
                for name in mcp_client._PENDING_TABLE_NAMES} == before
        assert mcp_client._pending_results['reused'] is payload
        assert mcp_client.action_outcome('reused') == before_outcome
        assert mcp_client.action_outcome('removed') == {
            'status': 'completed', 'tool': 'removed-tool'}
        assert 'unchanged' not in mcp_client._pending_actions
        assert mcp_client.action_outcome('unchanged') == {
            'status': outcome, 'tool': 'steady-tool'}
        assert notifications == [('unchanged', frozenset({'mouth'}))]
        for table in mcp_client.pending_tables():
            assert 'unchanged' not in table
            assert 'removed' not in table

    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['completed', 'timeout', 'cancelled'])
def test_barrier_normal_outcomes_keep_cleanup_and_notification(monkeypatch, outcome):
    async def run():
        event = register('action', 'tool', 'base')
        notifications = []
        mcp_client._settle_listeners.append(lambda aid, resource: notifications.append((aid, resource)))
        cancel = asyncio.Event()
        if outcome == 'completed':
            event.set()
        elif outcome == 'cancelled':
            cancel.set()
        else:
            async def timed_out(tasks, **kwargs):
                return set(), set(tasks)
            monkeypatch.setattr(mcp_client.asyncio, 'wait', timed_out)
        result = await mcp_client.await_pending(cancel_event=cancel)
        assert result['status'] == outcome
        assert mcp_client.action_outcome('action') == {'status': outcome, 'tool': 'tool'}
        assert notifications == [('action', frozenset({'base'}))]
        for table in mcp_client.pending_tables():
            assert 'action' not in table

    asyncio.run(run())
