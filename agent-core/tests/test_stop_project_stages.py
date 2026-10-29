"""Mock-based tests for local project-stop ordering and failure reporting.

Closing local admission does not stop existing LLM turns, hooks, remote requests,
or physical devices. These tests do not establish a remote lifecycle fence.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config as config_mod  # noqa: E402
from api import config as api_config  # noqa: E402


CARDS = [
    {'id': 'c-mic', 'mcpId': 'agentcore', 'toolName': 'remote_mic'},
    {'id': 'c-asr', 'mcpId': 'mcp-1', 'toolName': 'asr'},
    {'id': 'c-tts', 'mcpId': 'mcp-1', 'toolName': 'tts'},
]


class Recorder:
    """记下调用顺序 —— 顺序正是这里要守的东西。"""

    def __init__(self, hang_for=(), fail_for=()):
        self.calls = []
        self.events = []
        self.hang_for = set(hang_for)
        self.fail_for = set(fail_for)
        self.concurrent_peak = 0
        self._live = 0

    async def mcp_call_tool(self, mcp_id, req, timeout_s=None):
        self.calls.append((mcp_id, req.tool, req.arguments.get('action'), timeout_s))
        self._live += 1
        self.concurrent_peak = max(self.concurrent_peak, self._live)
        try:
            if req.tool in self.hang_for:
                # 没死线就永远挂着 —— 调用方必须自己带 timeout_s
                await asyncio.sleep(3600)
            if req.tool in self.fail_for:
                raise RuntimeError('mcp said no')
            await asyncio.sleep(0.01)
            return {'code': 200, 'data': {'result': {'state': 'idle'}}}
        finally:
            self._live -= 1

    async def push_event(self, event):
        self.events.append(event)

    def types(self):
        return [e['type'] for e in self.events]


@pytest.fixture
def rig(monkeypatch):
    rec = Recorder()
    config_mod.main['canvas_layout'] = {'cards': list(CARDS)}
    config_mod.main['core'] = {'project_running': True}

    def _install(recorder):
        monkeypatch.setattr('api.mcp_manage.mcp_call_tool', recorder.mcp_call_tool,
                            raising=False)
        monkeypatch.setattr('api.motus_stream.push_event', recorder.push_event,
                            raising=False)
        return recorder

    _install(rec)
    rec.install = _install
    monkeypatch.setattr(api_config, 'STOP_CARD_TIMEOUT_S', 0.2)
    monkeypatch.setattr(api_config, 'layout_lifecycle_lock', asyncio.Lock())
    monkeypatch.setattr(api_config, '_stop_teardown_task', None, raising=False)
    monkeypatch.setattr(api_config, '_start_project_task', None, raising=False)
    yield rec
    config_mod.main['canvas_layout'] = {}


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ── 第 1 段:agent loop ───────────────────────────────────────────────────────

def test_local_state_is_broadcast_before_device_calls(rig):
    """The state marker is not evidence existing dispatches have stopped."""
    run(api_config._do_stop_project())
    assert rig.events[0]['type'] == 'project_state'
    assert rig.events[0]['payload']['running'] is False
    # 第一条 MCP 调用必须发生在那条广播之后
    assert rig.types().index('project_state') < rig.types().index('project_stop_begin')
    assert config_mod.main['core']['project_running'] is False


def test_a_hanging_device_does_not_delay_the_agent_stop(rig):
    """一张卡片挂死,`project_running` 仍然要立刻为 False。"""
    rig.install(Recorder(hang_for={'tts'}))

    async def scenario():
        await api_config._do_stop_project_agent()
        assert config_mod.main['core']['project_running'] is False
        # 收尾单独跑,慢是它的事
        await api_config._do_stop_project_teardown()

    run(asyncio.wait_for(scenario(), 5))


def test_project_stop_does_not_wait_for_inflight_topic_request(rig, monkeypatch):
    """Mock ordering only: neither HTTP completion nor stop proves physical safety."""
    import json
    from dataclasses import replace
    import topic_actions

    async def scenario():
        manager = topic_actions.TopicActionManager()
        callbacks = []
        unsubscribed = []
        entered = asyncio.Event()
        release = asyncio.Event()
        device_stop = asyncio.Event()
        calls = []
        cancelled = False
        route = topic_actions.TopicActionRoute(
            key='goal', topic='/goal', fmt='data/json', mcp_id='mcp-1',
            tool_name='nav2', instance_id='c-nav', port='goal',
            action='navigate_to_pose', schema='goal-v1', id_field='goal_id',
            id_argument='_control_nav_id', allowed_fields=('x',),
            required_fields=('x',), input_schema={'type': 'object'},
        )

        def subscribe(_key, _topic, _fmt, _loop, callback):
            callbacks.append(callback)

        async def call_hook(_mcp, _tool, args, **_kwargs):
            nonlocal cancelled
            calls.append(args['_control_nav_id'])
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise
            return {'status': 'navigating'}

        async def stop_card(_mcp, _req, timeout_s=None):
            assert not release.is_set()
            device_stop.set()
            return {'code': 200, 'data': {'result': {'state': 'idle'}}}

        def unsubscribe(key):
            unsubscribed.append(key)
            if key == 'goal':
                raise RuntimeError('simulated unsubscribe failure')

        monkeypatch.setattr(topic_actions, 'manager', manager)
        monkeypatch.setattr(topic_actions.ros2_bridge, 'subscribe', subscribe)
        monkeypatch.setattr(topic_actions.ros2_bridge, 'unsubscribe', unsubscribe)
        monkeypatch.setattr(topic_actions.mcp_client, 'call_tool_hook', call_hook)
        monkeypatch.setattr('api.mcp_manage.mcp_call_tool', stop_card)
        other = replace(route, key='goal-other', topic='/other')
        manager._activate({route.key: topic_actions.RouteRuntime(route=route),
                           other.key: topic_actions.RouteRuntime(route=other)})
        payload = {'schema': 'goal-v1', 'goal_id': 'first', 'x': 1}
        encode = lambda p: json.dumps(p).encode()
        first = asyncio.create_task(callbacks[0](encode(payload), 'data/json'))
        await asyncio.wait_for(entered.wait(), 1)
        queued = asyncio.create_task(callbacks[0](
            encode({**payload, 'goal_id': 'queued'}), 'data/json'
        ))
        await asyncio.sleep(0)
        assert queued in manager._inflight
        try:
            result = await asyncio.wait_for(api_config.api_stop_project(), 1)
            assert result['ok'] is True
            assert unsubscribed == ['goal', 'goal-other']
            assert config_mod.main['core']['project_running'] is False
            assert any(e['type'] == 'project_state' and not e['payload']['running']
                       for e in rig.events)
            await asyncio.wait_for(device_stop.wait(), 1)
            assert not first.done() and first in manager._inflight
            assert not cancelled
            assert calls == ['first']
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(first, queued), 1)
            if api_config._stop_teardown_task is not None:
                await asyncio.wait_for(api_config._stop_teardown_task, 1)
        assert calls == ['first']
        assert not manager._inflight

    asyncio.run(scenario())


# ── 第 2 段:并行 + 死线 ──────────────────────────────────────────────────────

def test_every_card_gets_a_deadline(rig):
    run(api_config._stop_cards(CARDS))
    assert rig.calls, '一张卡片都没停'
    for _mcp, _tool, action, timeout_s in rig.calls:
        assert action == 'stop'
        assert timeout_s == pytest.approx(0.2), \
            '没传 timeout_s 的话默认是不限时,一个不回应的 MCP 会永远挂住停止'


def test_cards_stop_in_parallel(rig):
    run(api_config._stop_cards(CARDS))
    assert rig.concurrent_peak == len(CARDS), \
        f'串行了(峰值并发 {rig.concurrent_peak}),总耗时会是各卡之和'


def test_a_hanging_card_does_not_block_the_others(rig):
    rec = rig.install(Recorder(hang_for={'tts'}))
    results = run(asyncio.wait_for(api_config._stop_cards(CARDS), 5))
    ok = {r['tool'] for r in results if r['ok']}
    assert ok == {'remote_mic', 'asr'}, '坏掉的那张卡片不能带走别的'
    bad = [r for r in results if not r['ok']]
    assert len(bad) == 1 and bad[0]['tool'] == 'tts'
    assert '死线' in bad[0]['error'], bad[0]['error']
    assert rec.concurrent_peak == 3


def test_a_failing_card_is_reported_not_swallowed(rig):
    rig.install(Recorder(fail_for={'asr'}))
    results = run(api_config._stop_cards(CARDS))
    bad = [r for r in results if not r['ok']]
    assert len(bad) == 1
    assert bad[0]['tool'] == 'asr'
    assert bad[0]['error'], '错误信息不能是空的 —— 界面要显示它'


@pytest.mark.parametrize('reply', [
    {'code': 500, 'message': 'release_result_unknown'},
    {'code': 200, 'data': {'error': 'release_result_unknown'}},
    {'code': 200, 'data': {'state': 'failed', 'reason': 'release_result_unknown'}},
    {'code': 200, 'data': [{'type': 'text', 'text':
        '{"state":"error","error":"release_result_unknown"}'}]},
])
def test_unsuccessful_device_receipt_is_not_counted_as_stopped(rig, monkeypatch, reply):
    async def call(*args, **kwargs):
        return reply
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    results = run(api_config._stop_cards(CARDS[:1], progress=True))
    assert results[0]['ok'] is False
    assert 'release_result_unknown' in results[0]['error']
    assert rig.events[-1]['payload']['status'] == 'error'


def test_a_declined_stop_is_not_a_failed_stop(rig, monkeypatch):
    """A stateless actuator answers the verb it does not know, and that is fine.

    Twenty-four of phanthymotus-driver's thirty bundles still answer `stop`
    raw, so their gestures, dock commands and speakers reply
    `{"error": "unknown action: stop"}`. Nothing was armed. Reading that as a
    failed stop is the same misreading that once rolled an entire robot's
    start-project back over a dock command — see that repo's common/lifecycle.py.
    """
    async def call(*args, **kwargs):
        return {'code': 200, 'data': {'error': 'unknown action: stop'}}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    assert run(api_config._stop_cards(CARDS[:1]))[0]['ok'] is True


def test_a_stale_telemetry_read_is_not_a_failed_stop(rig, monkeypatch):
    """`unknown` means "I cannot tell", not "I refused".

    engineai/t800 answers `state: "unknown"` when its telemetry has gone stale.
    Counting that as a failed stop reports a device that stopped fine as stuck.
    """
    async def call(*args, **kwargs):
        return {'code': 200, 'data': {'state': 'unknown'}}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    assert run(api_config._stop_cards(CARDS[:1]))[0]['ok'] is True


def test_core_does_not_invent_a_release_vocabulary(rig, monkeypatch):
    """Undeclared device-specific fields must not decide a generic stop.

    `ownership_held`, `authority_released`, `hold`, `handback`, `sdk_pending`
    and `awaiting_feedback` were once hardcoded as "still stopping". Not one of
    those strings exists anywhere in phanthymotus-driver — so the guess applied
    to nobody, while `active` (19 occurrences, what pndbotics/adam answers for a
    powered arm) was hardcoded too and applied to every card on every robot.
    """
    async def call(*args, **kwargs):
        return {'code': 200, 'data': {'state': 'active', 'ownership_held': True,
                                      'authority_released': False}}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    assert run(api_config._stop_cards(CARDS[:1]))[0]['ok'] is True


def _declare_stop_rules(declaration):
    """Put an `x-stop` declaration on the card CARDS[:1] points at."""
    config_mod.main['services'] = {'mcp': [{'id': 'agentcore', 'transport': 'http', 'tools': [
        {'name': 'remote_mic', 'inputSchema': {'x-stop': declaration}}]}]}


@pytest.mark.parametrize('complete', [True, False])
def test_a_declared_pending_flag_polls_info_without_reissuing_stop(rig, monkeypatch, complete):
    """A device with a release protocol says so, and then Core waits for it."""
    _declare_stop_rules({'pending_flags': {'ownership_held': True}})
    calls = []

    async def call(mcp_id, req, timeout_s=None):
        action = req.arguments['action']
        calls.append(action)
        held = not (action == 'info' and complete)
        return {'code': 200, 'data': {'state': 'idle', 'ownership_held': held}}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    result = run(api_config._stop_cards(CARDS[:1]))[0]
    assert result['ok'] is complete
    assert calls[0] == 'stop' and calls.count('stop') == 1 and 'info' in calls
    if not complete:
        assert '停止未确认' in result['error']


def test_a_declared_pending_state_is_honoured(rig, monkeypatch):
    _declare_stop_rules({'pending_states': ['handing_back']})
    seen = []

    async def call(mcp_id, req, timeout_s=None):
        seen.append(req.arguments['action'])
        state = 'handing_back' if req.arguments['action'] == 'stop' else 'idle'
        return {'code': 200, 'data': {'state': state}}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    assert run(api_config._stop_cards(CARDS[:1]))[0]['ok'] is True
    assert seen == ['stop', 'info']


def test_a_declared_failed_state_fails_the_stop(rig, monkeypatch):
    _declare_stop_rules({'failed_states': ['refused']})

    async def call(*args, **kwargs):
        return {'code': 200, 'data': {'state': 'refused', 'error': 'operator override'}}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    result = run(api_config._stop_cards(CARDS[:1]))[0]
    assert result['ok'] is False and 'operator override' in result['error']


@pytest.mark.parametrize('declaration', ['nonsense', None, {'pending_states': 'running'},
                                         {'pending_flags': ['busy']}, {}])
def test_a_malformed_declaration_falls_back_to_the_generic_rules(declaration):
    """A driver cannot break every card's stop by sending junk in `x-stop`."""
    rules = api_config.stop_rules({'inputSchema': {'x-stop': declaration}})
    assert rules == {'failed': api_config._STOP_FAILED_STATES,
                     'pending': api_config._STOP_PENDING_STATES, 'flags': {},
                     'timeout': api_config.STOP_CARD_TIMEOUT_S}


def test_a_warning_alongside_a_live_state_is_not_a_failure(rig, monkeypatch):
    """`error` decides only when `state` is absent — the rule tool_state_of
    already documents a few hundred lines up, and which an earlier version of
    the stop path contradicted by raising on any `error` key at all."""
    _declare_stop_rules({'pending_states': ['winding_down']})
    replies = iter([{'state': 'winding_down', 'error': 'last frame dropped'},
                    {'state': 'idle'}])

    async def call(*args, **kwargs):
        return {'code': 200, 'data': next(replies)}
    monkeypatch.setattr('api.mcp_manage.mcp_call_tool', call)
    assert run(api_config._stop_cards(CARDS[:1]))[0]['ok'] is True


def test_cards_without_an_mcp_are_skipped(rig):
    results = run(api_config._stop_cards(
        CARDS + [{'id': 'note', 'mcpId': '', 'toolName': ''}]))
    assert len(results) == len(CARDS)


# ── 进度事件 ─────────────────────────────────────────────────────────────────

def test_progress_events_are_only_for_the_stop_button(rig):
    """删卡片触发的清理走同一个函数,但不该在界面上报「设备收尾」。"""
    run(api_config._stop_cards(CARDS))
    assert 'project_stop_item' not in rig.types()
    rig.events = []
    run(api_config._stop_cards(CARDS, progress=True))
    assert rig.types().count('project_stop_item') == len(CARDS)


def test_the_done_event_carries_what_failed(rig):
    rec = rig.install(Recorder(fail_for={'tts'}))
    run(api_config._do_stop_project())
    done = [e for e in rec.events if e['type'] == 'project_stop_done'][-1]
    assert done['payload']['total'] == 3
    assert done['payload']['stopped'] == 2
    assert [f['tool'] for f in done['payload']['failed']] == ['tts']
    assert done['payload']['failed'][0]['card_id'] == 'c-tts', \
        '界面要靠 card_id 做「重试这张」'


def test_begin_event_lists_the_cards_up_front(rig):
    """列表要在收尾开始前就给出来,否则界面只能一项项往上堆,看不出还剩几台。"""
    run(api_config._do_stop_project())
    begin = [e for e in rig.events if e['type'] == 'project_stop_begin'][0]
    assert [c['tool'] for c in begin['payload']['cards']] == \
        [c['toolName'] for c in CARDS]


# ── 在飞守卫 ─────────────────────────────────────────────────────────────────

def test_a_second_stop_does_not_start_a_second_teardown(rig):
    async def scenario():
        rig.install(Recorder(hang_for={'tts'}))
        first = await api_config.api_stop_project()
        assert first['ok'] is True
        assert api_config.teardown_running() is True
        second = await api_config.api_stop_project()
        assert second.get('teardown_running') is True
        task = api_config._stop_teardown_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(scenario())


def test_start_is_rejected_while_stop_broadcast_is_pending(rig, monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        original_push = rig.push_event
        started = []

        async def blocked_push(event):
            if event['type'] == 'project_state':
                entered.set()
                await release.wait()
            await original_push(event)

        async def start_impl():
            started.append(True)
            return True

        monkeypatch.setattr('api.motus_stream.push_event', blocked_push)
        monkeypatch.setattr(api_config, '_do_start_project_impl', start_impl)
        stop = asyncio.create_task(api_config.api_stop_project())
        await asyncio.wait_for(entered.wait(), 1)
        assert config_mod.main['core']['project_running'] is False
        assert api_config.teardown_running()
        second = await api_config.api_stop_project()
        assert second['teardown_running'] is True
        response = await api_config.api_start_project()
        assert response.status_code == 409
        assert await api_config._do_start_project() is False
        assert not started
        assert not stop.done()
        release.set()
        assert (await asyncio.wait_for(stop, 1))['ok'] is True
        await asyncio.wait_for(api_config._stop_teardown_task, 1)
        assert await api_config._do_start_project() is True
        assert started == [True]

    asyncio.run(scenario())


def test_stop_continues_if_request_is_cancelled_during_broadcast(rig, monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        original_push = rig.push_event

        async def blocked_push(event):
            if event['type'] == 'project_state':
                entered.set()
                await release.wait()
            await original_push(event)

        monkeypatch.setattr('api.motus_stream.push_event', blocked_push)
        stop = asyncio.create_task(api_config.api_stop_project())
        await asyncio.wait_for(entered.wait(), 1)
        managed = api_config._stop_teardown_task
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop
        assert not managed.cancelled() and not managed.done()
        release.set()
        await asyncio.wait_for(managed, 1)
        assert rig.calls

    asyncio.run(scenario())


def test_stop_snapshot_survives_layout_replacement_and_includes_current(rig):
    async def scenario():
        # Simulate an already-running canvas/solution transaction holding the lock.
        await api_config.layout_lifecycle_lock.acquire()
        try:
            result = await api_config.api_stop_project()
            assert result['ok'] is True
            config_mod.main['canvas_layout'] = {'cards': [
                {'id': 'c-mic', 'mcpId': 'different', 'toolName': 'replacement'},
                CARDS[1],  # Identical target must not be stopped twice.
            ]}
        finally:
            api_config.layout_lifecycle_lock.release()
        await asyncio.wait_for(api_config._stop_teardown_task, 1)
        targets = [(mcp, tool) for mcp, tool, _action, _timeout in rig.calls]
        assert sorted(targets) == sorted([
            ('agentcore', 'remote_mic'), ('mcp-1', 'asr'), ('mcp-1', 'tts'),
            ('different', 'replacement'),
        ])

    asyncio.run(scenario())


def test_project_running_true_is_rejected_during_teardown(rig):
    async def scenario():
        rig.install(Recorder(hang_for={'tts'}))
        await api_config.api_stop_project()
        response = await api_config.set_project_running(api_config.ProjectRunningRequest(running=True))
        assert response.status_code == 409
        assert config_mod.main['core']['project_running'] is False
        assert await api_config.set_project_running(api_config.ProjectRunningRequest(running=False)) == {'ok': True}
        task = api_config._stop_teardown_task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Existing behavior allows later restart; cancellation is not a safety confirmation.
        assert await api_config.set_project_running(api_config.ProjectRunningRequest(running=True)) == {'ok': True}

    asyncio.run(scenario())


@pytest.mark.parametrize('event_type', ['project_state', 'project_stop_begin'])
def test_broadcast_exception_does_not_block_device_stop(rig, monkeypatch, event_type):
    async def scenario():
        original = rig.push_event
        async def push(event):
            if event['type'] == event_type:
                raise RuntimeError('broadcast failed')
            await original(event)
        monkeypatch.setattr('api.motus_stream.push_event', push)
        await api_config.api_stop_project()
        await asyncio.wait_for(api_config._stop_teardown_task, 1)
        assert len(rig.calls) == len(CARDS)
    asyncio.run(scenario())


def test_project_running_false_invalidates_local_epoch_without_device_stop(rig):
    async def scenario():
        before = api_config.stop_project_epoch()
        assert await api_config.set_project_running(api_config.ProjectRunningRequest(running=False)) == {'ok': True}
        assert api_config.stop_project_epoch() == before + 1
        assert not rig.calls
        assert await api_config.set_project_running(api_config.ProjectRunningRequest(running=False)) == {'ok': True}
        assert api_config.stop_project_epoch() == before + 2
        assert await api_config.set_project_running(api_config.ProjectRunningRequest(running=True)) == {'ok': True}
        assert api_config.stop_project_epoch() == before + 2

    asyncio.run(scenario())


def test_the_endpoint_returns_before_the_teardown_finishes(rig):
    """响应只等第 1 段 —— 让它等完收尾,就等于让已经为真的「已停止」被一台坏设备拖着。"""
    async def scenario():
        rig.install(Recorder(hang_for={'tts'}))
        result = await asyncio.wait_for(api_config.api_stop_project(), 1.0)
        assert result['ok'] is True
        assert config_mod.main['core']['project_running'] is False
        assert api_config.teardown_running() is True
        task = api_config._stop_teardown_task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
