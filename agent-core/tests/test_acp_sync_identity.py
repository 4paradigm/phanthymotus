"""sync must not consume or clear a different Event generation."""
import asyncio
from collections import OrderedDict

import pytest

import mcp_client


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in mcp_client._PENDING_TABLE_NAMES:
        monkeypatch.setattr(mcp_client, name, OrderedDict() if name == '_pending_results' else {})
    monkeypatch.setattr(mcp_client, '_action_outcomes', OrderedDict())
    monkeypatch.setattr(mcp_client, '_settle_listeners', [])
    monkeypatch.setattr(mcp_client, '_register_listeners', [])


def register(aid, tool, resource):
    mcp_client._register_pending(
        aid, tool_name=tool,
        meta={'completion': {'timeout': 20}, 'resource': frozenset({resource})},
        args={}, result={}, mcp_id=tool,
    )
    return mcp_client._pending_actions[aid]


def snapshot(aid):
    return {name: (aid in getattr(mcp_client, name), getattr(mcp_client, name).get(aid))
            for name in mcp_client._PENDING_TABLE_NAMES}


def install_wait_gate(monkeypatch):
    entered, reached, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_wait = asyncio.wait
    original_reap = mcp_client.cancel_and_reap

    async def controlled_wait(tasks, **kwargs):
        entered.set()
        return await original_wait(tasks, **kwargs)

    async def controlled_reap(tasks):
        await original_reap(tasks)
        reached.set()
        await release.wait()

    monkeypatch.setattr(mcp_client.asyncio, 'wait', controlled_wait)
    monkeypatch.setattr(mcp_client, 'cancel_and_reap', controlled_reap)
    return entered, reached, release


def test_success_identity_loss_mixed_with_matched_and_removed(monkeypatch):
    async def run():
        old = register('reused', 'old', 'base')
        same = register('same', 'same-tool', 'mouth')
        removed = register('removed', 'removed-tool', 'camera')
        cancel = asyncio.Event()
        notifications = []
        mcp_client._settle_listeners.append(lambda aid, res: notifications.append((aid, res)))
        entered, reached, release = install_wait_gate(monkeypatch)
        task = asyncio.create_task(mcp_client.sync(['reused', 'same', 'removed'], cancel_event=cancel))
        await entered.wait()
        old.set()
        same.set()
        removed.set()
        mcp_client._pending_results['same'] = {'status': 'failed', 'detail': 'same'}
        await reached.wait()
        mcp_client._forget_pending(['reused', 'removed'], 'completed')
        new = register('reused', 'new', 'vision')
        payload = {'status': 'failed', 'detail': 'new'}
        mcp_client._pending_results['reused'] = payload
        new.set()
        before = snapshot('reused')
        before_outcome = dict(mcp_client.action_outcome('reused'))
        notifications.clear()
        release.set()
        out = await task
        assert out == {'status': 'unknown', 'results': {'same': {'status': 'failed', 'detail': 'same'}},
                       'unknown': ['reused', 'removed']}
        assert mcp_client._pending_actions['reused'] is new
        assert snapshot('reused') == before
        assert mcp_client._pending_results['reused'] is payload
        assert mcp_client.action_outcome('reused') == before_outcome
        assert mcp_client.action_outcome('removed') == {'status': 'completed', 'tool': 'removed-tool'}
        assert mcp_client.action_outcome('same') == {'status': 'completed', 'tool': 'same-tool'}
        assert notifications == [('same', frozenset({'mouth'}))]
        for table in mcp_client.pending_tables():
            assert 'same' not in table and 'removed' not in table

    asyncio.run(run())


def test_two_sync_waiters_do_not_invent_second_result(monkeypatch):
    async def run():
        event = register('action', 'tool', 'base')
        cancel = asyncio.Event()
        both_waiting = asyncio.Event()
        count = 0
        original_wait = asyncio.wait

        async def observed_wait(tasks, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                both_waiting.set()
            return await original_wait(tasks, **kwargs)

        monkeypatch.setattr(mcp_client.asyncio, 'wait', observed_wait)
        first = asyncio.create_task(mcp_client.sync(['action'], cancel_event=cancel))
        second = asyncio.create_task(mcp_client.sync(['action'], cancel_event=cancel))
        await both_waiting.wait()
        payload = {'status': 'failed', 'result': {'reason': 'failure'}}
        mcp_client._pending_results['action'] = payload
        event.set()
        outcomes = await asyncio.gather(first, second)
        assert sorted(out['status'] for out in outcomes) == ['completed', 'unknown']
        completed = next(out for out in outcomes if out['status'] == 'completed')
        unknown = next(out for out in outcomes if out['status'] == 'unknown')
        assert completed == {'status': 'completed', 'results': {'action': payload}}
        assert unknown == {'status': 'unknown', 'results': {}, 'unknown': ['action']}
        assert mcp_client.action_outcome('action') == {'status': 'completed', 'tool': 'tool'}

    asyncio.run(run())


def test_timeout_identity_loss_and_matched_states(monkeypatch):
    async def run():
        old_set = register('old-set', 'old-set-tool', 'base')
        old_unset = register('old-unset', 'old-unset-tool', 'arm')
        complete = register('complete', 'complete-tool', 'mouth')
        pending = register('pending', 'pending-tool', 'camera')
        old_set.set()
        complete.set()
        mcp_client._pending_results['complete'] = {'status': 'failed', 'detail': 'original'}
        entered, release = asyncio.Event(), asyncio.Event()

        async def controlled_wait_for(awaitable, timeout):
            child = asyncio.create_task(awaitable)
            entered.set()
            await release.wait()
            child.cancel()
            try:
                await child
            except asyncio.CancelledError:
                pass
            raise asyncio.TimeoutError()

        monkeypatch.setattr(mcp_client.asyncio, 'wait_for', controlled_wait_for)
        notifications = []
        mcp_client._settle_listeners.append(lambda aid, res: notifications.append((aid, res)))
        task = asyncio.create_task(mcp_client.sync(['old-set', 'old-unset', 'complete', 'pending']))
        await entered.wait()
        # wait_for is entered after the Event snapshot, before the child runs.
        mcp_client._forget_pending(['old-set', 'old-unset'], 'timeout')
        new_set = register('old-set', 'new-set', 'vision')
        new_unset = register('old-unset', 'new-unset', 'speech')
        for aid, ev in [('old-set', new_set), ('old-unset', new_unset)]:
            mcp_client._pending_results[aid] = {'status': 'failed', 'detail': aid}
            ev.set()
        before = {aid: snapshot(aid) for aid in ('old-set', 'old-unset', 'pending')}
        outcomes = {aid: dict(mcp_client.action_outcome(aid)) for aid in ('old-set', 'old-unset')}
        notifications.clear()
        release.set()
        out = await task
        assert out == {'status': 'timeout', 'completed': {'complete': {'status': 'failed', 'detail': 'original'}},
                       'pending': ['pending'], 'unknown': ['old-set', 'old-unset']}
        for aid in before:
            assert snapshot(aid) == before[aid]
        assert mcp_client._pending_actions['pending'] is pending
        assert mcp_client._pending_actions['old-set'] is new_set
        assert mcp_client._pending_actions['old-unset'] is new_unset
        for aid in outcomes:
            assert mcp_client.action_outcome(aid) == outcomes[aid]
        assert notifications == [('complete', frozenset({'mouth'}))]
        for table in mcp_client.pending_tables():
            assert 'complete' not in table

    asyncio.run(run())


def test_normal_success_deduplicates_ids_in_order():
    async def run():
        first = register('first', 'first-tool', 'base')
        second = register('second', 'second-tool', 'mouth')
        first.set()
        second.set()
        mcp_client._pending_results.update({'first': {'status': 'failed'}, 'second': {'status': 'completed'}})
        notifications = []
        mcp_client._settle_listeners.append(lambda aid, res: notifications.append((aid, res)))
        out = await mcp_client.sync(['second', 'first', 'second', 'first'])
        assert out == {'status': 'completed', 'results': {'second': {'status': 'completed'}, 'first': {'status': 'failed'}}}
        assert list(out['results']) == ['second', 'first']
        assert notifications == [('second', frozenset({'mouth'})), ('first', frozenset({'base'}))]
        for table in mcp_client.pending_tables():
            assert 'first' not in table and 'second' not in table

    asyncio.run(run())


def test_normal_timeout_and_cancel_remain_compatible(monkeypatch):
    async def run():
        complete = register('complete', 'complete-tool', 'base')
        pending = register('pending', 'pending-tool', 'mouth')
        complete.set()
        mcp_client._pending_results['complete'] = {'status': 'failed'}
        before = snapshot('pending')
        original_wait_for = asyncio.wait_for

        async def timeout_now(awaitable, timeout):
            awaitable.close()
            raise asyncio.TimeoutError()

        monkeypatch.setattr(mcp_client.asyncio, 'wait_for', timeout_now)
        out = await mcp_client.sync(['complete', 'pending', 'pending'])
        assert out == {'status': 'timeout', 'completed': {'complete': {'status': 'failed'}},
                       'pending': ['pending']}
        assert snapshot('pending') == before
        assert mcp_client._pending_actions['pending'] is pending
        monkeypatch.setattr(mcp_client.asyncio, 'wait_for', original_wait_for)
        cancel = asyncio.Event()
        cancel.set()
        out = await mcp_client.sync(['pending', 'pending'], cancel_event=cancel)
        assert out == {'status': 'cancelled', 'pending': ['pending']}
        assert snapshot('pending') == before

    asyncio.run(run())
