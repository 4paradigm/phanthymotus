"""ACP windows: local attribution only, not device authentication or fencing."""
import asyncio
import json

import pytest
import mcp_client


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in mcp_client._PENDING_TABLE_NAMES:
        monkeypatch.setattr(mcp_client, name, mcp_client.OrderedDict() if name == '_pending_results' else {})
    monkeypatch.setattr(mcp_client, '_early_dispatches', {})
    monkeypatch.setattr(mcp_client, '_register_listeners', [])
    monkeypatch.setattr(mcp_client, '_settle_listeners', [])


def register(aid, mcp='a', token=None, silent=False):
    mcp_client._register_pending(aid, tool_name='move',
        meta={'completion': {'silent_completion': silent}, 'resource': frozenset({'base'})},
        args={}, result={}, mcp_id=mcp, dispatch_token=token)


def test_outside_window_history_and_duplicates_are_not_consumed():
    assert mcp_client.receive_action_completion('x', {'status': 'old'}) == 'rejected'
    register('x')
    assert not mcp_client._pending_actions['x'].is_set()
    assert mcp_client.receive_action_completion('x', {'status': 'new'}) == 'completed'
    assert mcp_client.receive_action_completion('x', {'status': 'duplicate'}) == 'duplicate'
    assert mcp_client.pending_result('x')['status'] == 'new'


def test_late_completion_after_interrupt_does_not_overwrite_cancel_or_notify():
    register('x')
    settled = []
    mcp_client._settle_listeners.append(lambda *args: settled.append(args))
    # event.llm's interrupt path writes cancellation and sets the event itself.
    mcp_client._pending_results['x'] = {'status': 'cancelled'}
    mcp_client._pending_actions['x'].set()
    assert mcp_client.receive_action_completion('x', {'status': 'completed'}) == 'duplicate'
    assert mcp_client.pending_result('x') == {'status': 'cancelled'}
    assert settled == []


def test_sse_source_and_ambiguous_unsourced_window(capsys):
    a = mcp_client._begin_completion_dispatch('a')
    b = mcp_client._begin_completion_dispatch('b')
    assert mcp_client.receive_action_completion('x', {'status': 'early'}) == 'rejected'
    assert 'may time out' in capsys.readouterr().out
    assert mcp_client.receive_action_completion('x', {'status': 'bad'}, source_mcp='c') == 'rejected'
    assert mcp_client.receive_action_completion('x', {'status': 'ok'}, source_mcp='a') == 'early'
    mcp_client._end_completion_dispatch(b)
    assert list(mcp_client._early_dispatches[a]['receipts']) == ['x']
    mcp_client._end_completion_dispatch(a)
    register('x', mcp='a')
    assert mcp_client.receive_action_completion('x', {}, source_mcp='b') == 'rejected'
    assert not mcp_client._pending_actions['x'].is_set()
    assert mcp_client.receive_action_completion('x', {}, source_mcp='a') == 'completed'


def test_early_registration_order_silent_and_one_notification(monkeypatch):
    async def run():
        seen, events = [], []
        async def enqueue(**kw):
            events.append(kw)
        monkeypatch.setattr(mcp_client.event_bus, 'enqueue', enqueue)
        mcp_client._register_listeners.append(lambda aid, *_: seen.append(('register', aid)))
        mcp_client._settle_listeners.append(lambda aid, *_: seen.append(('settle', aid)))
        token = mcp_client._begin_completion_dispatch('a')
        assert mcp_client.receive_action_completion('x', {'status': 'completed'}) == 'early'
        assert mcp_client.receive_action_completion('x', {'status': 'other'}) == 'duplicate'
        register('x', token=token, silent=True)
        mcp_client._end_completion_dispatch(token)
        await asyncio.sleep(0)
        assert seen == [('register', 'x'), ('settle', 'x')]
        assert len(events) == 1 and json.loads(events[0]['text'])['silent'] is True
        assert mcp_client.pending_result('x')['status'] == 'completed'
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['exception', 'cancel', 'timeout'])
def test_dispatch_window_closes_on_failure(monkeypatch, failure):
    async def run():
        name = 'mcp__a__move'
        monkeypatch.setattr(mcp_client, 'registry', {'a': {
            'url': 'http://localhost/mcp', 'tools': ['move'], 'split_map': {},
            'tool_meta': {name: {'completion': {'timeout': 3}}},
            'input_schemas': {}, 'schemas': {}, 'tool_groups': {}}})
        async def jrpc(*args, **kwargs):
            assert mcp_client.receive_action_completion('x', {}) == 'early'
            if failure == 'cancel':
                raise asyncio.CancelledError()
            if failure == 'timeout':
                raise asyncio.TimeoutError()
            raise RuntimeError('network failure')
        monkeypatch.setattr(mcp_client, '_jrpc', jrpc)
        if failure == 'cancel':
            with pytest.raises(asyncio.CancelledError):
                await mcp_client.call_tool(name, {})
        elif failure == 'exception':
            with pytest.raises(RuntimeError):
                await mcp_client.call_tool(name, {})
        else:
            assert '超时' in await mcp_client.call_tool(name, {})
        assert not mcp_client._early_dispatches
        assert mcp_client.receive_action_completion('x', {}) == 'rejected'
    asyncio.run(run())


def test_hook_early_receipt_registers_then_settles_even_if_listener_raises(monkeypatch):
    async def run():
        seen = []
        async def enqueue(**kw):
            seen.append(('event', kw['payload']['silent']))
        monkeypatch.setattr(mcp_client.event_bus, 'enqueue', enqueue)
        monkeypatch.setattr(mcp_client, 'registry', {'a': {
            'tool_meta': {'mcp__a__move': {'completion': {'silent_completion': True},
                                         'resource': frozenset({'base'})}}}})
        def broken(*args):
            raise RuntimeError('observer failure')
        def registered(aid, tool, args, resource):
            assert aid in mcp_client._pending_actions
            seen.append(('register', aid, tool, args, resource))
        mcp_client._register_listeners.extend([broken, registered])
        mcp_client._settle_listeners.append(lambda aid, res: seen.append(('settle', aid, res)))
        async def direct(*args, **kwargs):
            assert mcp_client.receive_action_completion('x', {'status': 'completed'}) == 'early'
            return {'action_id': 'x'}
        monkeypatch.setattr(mcp_client, 'call_tool_direct', direct)
        out = await mcp_client.call_tool_hook('a', 'move', {}, barrier_aware=True)
        await asyncio.sleep(0)
        assert out == {'action_id': 'x'}
        assert seen == [('register', 'x', 'move', {}, frozenset({'base'})),
                        ('settle', 'x', frozenset({'base'})), ('event', True)]
    asyncio.run(run())


def test_registration_listener_can_complete_reentrantly_without_duplicate(monkeypatch):
    async def run():
        seen = []
        async def enqueue(**kw):
            seen.append('event')
        monkeypatch.setattr(mcp_client.event_bus, 'enqueue', enqueue)
        token = mcp_client._begin_completion_dispatch('a')
        assert mcp_client.receive_action_completion('x', {'status': 'early'}) == 'early'
        def registered(aid, *_):
            seen.append('registered')
            assert mcp_client.receive_action_completion(aid, {'status': 'reentrant'}) == 'completed'
        mcp_client._register_listeners.append(registered)
        mcp_client._settle_listeners.append(lambda *_: seen.append('settled'))
        register('x', token=token)
        mcp_client._end_completion_dispatch(token)
        await asyncio.sleep(0)
        assert seen == ['registered', 'settled']
        assert mcp_client.pending_result('x')['status'] == 'reentrant'
    asyncio.run(run())


def test_early_replay_does_not_enqueue_after_turn_invalidated(monkeypatch):
    async def run():
        alive = [True]
        emitted = []
        async def enqueue(**kw):
            emitted.append(kw)
        monkeypatch.setattr(mcp_client.event_bus, 'enqueue', enqueue)
        ctx = mcp_client.dispatch_admission.set(lambda: alive[0])
        try:
            token = mcp_client._begin_completion_dispatch('a')
            assert mcp_client.receive_action_completion('x', {'status': 'completed'}) == 'early'
            register('x', token=token)
            mcp_client._end_completion_dispatch(token)
            # Delivery task has been created but not yet scheduled.
            alive[0] = False
            await asyncio.sleep(0)
            assert not emitted
            assert mcp_client._pending_actions['x'].is_set()
        finally:
            mcp_client.dispatch_admission.reset(ctx)
    asyncio.run(run())


def test_admission_checked_after_file_transfer_and_before_dispatch(monkeypatch):
    async def run():
        name = 'mcp__a__move'
        monkeypatch.setattr(mcp_client, 'registry', {'a': {
            'url': 'http://localhost/mcp', 'tools': ['move'], 'split_map': {},
            'tool_meta': {}, 'input_schemas': {name: {'type': 'object'}},
            'schemas': {}, 'tool_groups': {}}})
        alive, sent = [True], []
        async def transfer(mid, url, schema, args):
            alive[0] = False
            return args, None
        async def jrpc(*args, **kwargs):
            sent.append(True)
            return {}
        monkeypatch.setattr(mcp_client, '_transfer_file_args', transfer)
        monkeypatch.setattr(mcp_client, '_jrpc', jrpc)
        token = mcp_client.dispatch_admission.set(lambda: alive[0])
        try:
            result = await mcp_client.call_tool(name, {})
        finally:
            mcp_client.dispatch_admission.reset(token)
        assert result == mcp_client._DISPATCH_BLOCKED and not sent
    asyncio.run(run())


def test_admission_exception_fails_closed_and_unbound_allows():
    def broken():
        raise RuntimeError('bad predicate')
    assert mcp_client._dispatch_is_admitted()
    token = mcp_client.dispatch_admission.set(broken)
    try:
        assert not mcp_client._dispatch_is_admitted()
    finally:
        mcp_client.dispatch_admission.reset(token)
    assert mcp_client._dispatch_is_admitted()


def test_same_id_collision_currently_overwrites_original_event():
    # Characterization, NOT a safety guarantee. Another action may already run;
    # ID-only tables cannot safely hold both or recover a quarantined resource.
    register('x', mcp='a')
    original = mcp_client._pending_actions['x']
    register('x', mcp='b')
    assert mcp_client._pending_actions['x'] is not original
