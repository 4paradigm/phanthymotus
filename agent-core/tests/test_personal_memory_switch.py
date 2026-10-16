"""Switch modes at a stopped project's startup boundary, without restarting Core."""
import asyncio
import sys
from unittest.mock import AsyncMock, Mock

import pytest
import auth
import chat_history
import collector
import config
import event_bus
import personal_memory as pm
import prompt
import subagent
from event.llm import Event
from subagent.manager import SubagentManager
from test_personal_memory import store


@pytest.fixture
def runtime(store, monkeypatch):
    monkeypatch.setattr(auth, '_auth_enabled', True)
    monkeypatch.setattr(sys.modules['event.llm'], '_event_instance', None)
    monkeypatch.setattr(subagent, '_manager_instance', None)
    monkeypatch.setattr(SubagentManager, 'start', AsyncMock())
    monkeypatch.setattr(SubagentManager, 'shutdown', AsyncMock())
    monkeypatch.setattr(sys.modules['mcp_client'], 'init_all', AsyncMock())
    monkeypatch.setattr(sys.modules['task_store'], 'load_all', Mock())
    monkeypatch.setattr(sys.modules['task_store'], 'active_tasks', Mock(return_value=[]))
    for name in ('_output', '_steering_queue'):
        monkeypatch.setattr(collector, name, asyncio.Queue())
    monkeypatch.setattr(collector, '_busy', False)
    import scheduler
    monkeypatch.setattr(scheduler, '_dynamic_tasks', {})
    collector._priority_pending.clear()
    collector._bg_buffer.clear()
    config.main['personal_memory'] = {'enabled': False}
    pm.pin_settings()
    yield Event()
    # Do not leave queues/caches populated for other collector tests.
    if hasattr(collector, 'reset_context'):
        collector.reset_context()


def desired(enabled):
    settings = pm.configured_settings()
    settings['enabled'] = enabled
    config.main['personal_memory'] = settings


def test_switch_rebuilds_tools_and_backgrounds_without_crossing_histories(runtime):
    user = pm.create_user('Alice', remember=True)
    pm.bind_context(user['id'])
    item = pm.save_memory('private fact', 'fact', 'private-turn')
    pm.bind_context(None)
    sid = chat_history.create_session()
    chat_history.save_turn(sid, 0, [{'role': 'user', 'content': 'ordinary history'}])

    async def run():
        await runtime.__aenter__()
        original_manager = runtime._subagent_mgr
        assert 'ordinary history' in str(runtime._turns)
        collector._source_ring['old'] = ['ordinary input']
        collector._priority_pending.append({'text': 'old queued input'})
        collector._output.put_nowait({'text': 'old trigger'})
        collector._steering_queue.put_nowait({'text': 'old steering'})
        collector._bg_buffer.append({'text': 'old background'})
        event_bus._recent.append({'text': 'old event'})
        event_bus._queue.put_nowait({'text': 'old event'})
        runtime._summary = 'ordinary summary'
        desired(True)
        await runtime.apply_memory_mode()
        assert pm.enabled()
        assert runtime._subagent_mgr is not original_manager
        assert SubagentManager.shutdown.await_count == 1
        assert SubagentManager.start.await_count == 1  # only ordinary startup
        assert runtime._turns == [] and runtime._current_turn == []
        assert runtime._summary is None and runtime._session_id is None
        assert pm.current_user() is None
        assert 'remember_personal' in runtime._sys_tools
        assert not pm.tool_allowed('Bash')
        assert not collector._source_ring and not collector._priority_pending
        assert collector._output.empty() and collector._steering_queue.empty()
        assert not collector._bg_buffer and not event_bus._recent and event_bus._queue.empty()
        runtime._turns = [[{'role': 'user', 'content': 'private transcript'}]]
        runtime._current_turn = [{'role': 'user', 'content': 'private current'}]
        runtime._transient_turns = list(runtime._turns)
        event_bus._recent.append({'text': 'private event'})
        desired(False)
        await runtime.apply_memory_mode()
        assert not pm.enabled()
        assert 'ordinary history' in str(runtime._turns)
        assert 'private' not in str(runtime._turns)
        assert runtime._current_turn == [] and runtime._transient_turns == []
        assert 'remember_personal' not in runtime._sys_tools
        assert pm.tool_allowed('Bash')
        assert SubagentManager.start.await_count == 2
        assert not event_bus._recent
        sys.modules['mcp_client'].init_all.assert_awaited_once()
        pm.bind_context(user['id'])
        assert pm.recall('private fact')[0]['id'] == item['id']
        pm.bind_context(None)
        await runtime.__aexit__()

    asyncio.run(run())


def test_same_mode_preserves_existing_runtime(runtime):
    async def run():
        await runtime.__aenter__()
        manager = runtime._subagent_mgr
        runtime._summary = 'keep ordinary state'
        await runtime.apply_memory_mode()
        assert runtime._summary == 'keep ordinary state'
        assert runtime._subagent_mgr is manager
        SubagentManager.shutdown.assert_not_awaited()
        await runtime.__aexit__()
    asyncio.run(run())


def test_switch_waits_for_old_turn_and_discards_pending_input(runtime, monkeypatch):
    async def run():
        await runtime.__aenter__()
        entered, release = asyncio.Event(), asyncio.Event()
        async def turn(ev, cancel_event, reconsider_event):
            entered.set()
            await cancel_event.wait()
            assert not pm.enabled()
            await release.wait()
        monkeypatch.setattr(runtime, '_one_turn', turn)
        loop = asyncio.create_task(runtime.run_forever())
        collector._output.put_nowait({'text': 'ordinary turn'})
        await asyncio.wait_for(entered.wait(), 2)
        desired(True)
        switch = asyncio.create_task(runtime.apply_memory_mode())
        await asyncio.sleep(0)
        assert not switch.done() and not pm.enabled()
        collector._priority_pending.append({'source': 'dds:/asr', 'text': 'stale', 'ts': 1.0})
        release.set()
        await switch
        assert pm.enabled() and not collector._priority_pending and collector._output.empty()
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await runtime.__aexit__()
    asyncio.run(run())


def test_running_project_rejects_runtime_switch(runtime):
    config.main['core'] = {'project_running': True}
    desired(True)
    with pytest.raises(ValueError, match='停止'):
        asyncio.run(runtime.apply_memory_mode())
    assert not pm.enabled()


from test_start_project_resolution import driver, R1_LAYOUT, CORE


@pytest.mark.parametrize('missing_tts', [False, True])
def test_project_start_applies_pending_mode_after_live_route_validation(runtime, driver, monkeypatch, missing_tts):
    import copy
    import mcp_client
    from api import config as project
    graph = copy.deepcopy(R1_LAYOUT)
    next(c for c in graph['cards'] if c['id'] == CORE)['mcpId'] = 'agentcore'
    if not missing_tts:
        graph['cards'].append({'id': 'tts', 'mcpId': 'mcp-perception', 'toolName': 'tts'})
        graph['execConnections'] = [{'fromCardId': CORE, 'toCardId': 'tts',
                                    'toMcpId': 'mcp-perception', 'toToolName': 'tts'}]
    monkeypatch.setattr(mcp_client, 'registry', {'mcp-perception': {
        'online': True, 'schemas': {'mcp__mcp-perception__tts': {}}}})
    config.main['canvas_layout'] = graph

    async def run():
        await runtime.__aenter__()
        config.main['personal_memory'] = {'enabled': True, 'use_canvas_routes': True}
        result = await project._do_start_project_impl()
        assert result is not missing_tts
        assert pm.enabled() is not missing_tts
        assert config.main['core']['project_running'] is not missing_tts
        if not missing_tts:
            assert pm.settings()['voice_sources'] == ['dds:/ubuntu/mic/audio/asr']
            assert 'remember_personal' in runtime._sys_tools
        else:
            assert 'remember_personal' not in runtime._sys_tools
        await runtime.__aexit__()
    asyncio.run(run())


def test_failed_mode_initialization_can_be_retried(runtime):
    async def run():
        await runtime.__aenter__()
        desired(True)
        await runtime.apply_memory_mode()
        desired(False)
        SubagentManager.start.side_effect = RuntimeError('restore failed')
        with pytest.raises(RuntimeError, match='restore failed'):
            await runtime.apply_memory_mode()
        assert not runtime._mode_ready and not runtime._mode_switching
        assert not config.main.get('core', {}).get('project_running', False)
        SubagentManager.start.side_effect = None
        from personal_memory_config import apply_runtime_mode
        await apply_runtime_mode()
        assert runtime._mode_ready and not pm.enabled()
        await runtime.__aexit__()
    asyncio.run(run())


def test_config_cannot_change_while_project_is_starting(runtime, monkeypatch):
    from api import config as project
    from personal_memory_config import save_enabled
    task = Mock()
    task.done.return_value = False
    monkeypatch.setattr(project, '_start_project_task', task)
    with pytest.raises(ValueError, match='停止'):
        save_enabled(True)
    assert not pm.configured_settings()['enabled']


def test_ordinary_task_checks_pause_in_personal_mode_and_resume(runtime, monkeypatch):
    from types import SimpleNamespace
    import scheduler
    monkeypatch.setattr(scheduler, '_dynamic_tasks', {})
    task = SimpleNamespace(id='ordinary-task', goal='ordinary goal', check_cron='*/2 * * * *')
    monkeypatch.setattr(sys.modules['task_store'], 'active_tasks', Mock(return_value=[task]))

    async def run():
        await runtime.__aenter__()
        old_job = scheduler._dynamic_tasks['task:ordinary-task']
        desired(True)
        await runtime.apply_memory_mode()
        assert 'task:ordinary-task' not in scheduler._dynamic_tasks
        assert old_job.done()
        desired(False)
        await runtime.apply_memory_mode()
        assert 'task:ordinary-task' in scheduler._dynamic_tasks
        assert scheduler._dynamic_tasks['task:ordinary-task'] is not old_job
        await runtime.__aexit__()
        scheduler.remove_job('task:ordinary-task')
    asyncio.run(run())


def test_blocked_private_event_cannot_reappear_after_switch(runtime, monkeypatch):
    monkeypatch.setattr(event_bus, '_queue', asyncio.Queue(maxsize=1))
    monkeypatch.setattr(event_bus, '_recent', [])
    async def run():
        await runtime.__aenter__()
        desired(True)
        await runtime.apply_memory_mode()
        await event_bus.enqueue('dds:/asr', 'private first')
        blocked = asyncio.create_task(event_bus.enqueue('dds:/asr', 'private delayed'))
        await asyncio.sleep(0)
        assert not blocked.done()
        desired(False)
        await runtime.apply_memory_mode()
        await blocked
        assert not event_bus.recent()
        reader = asyncio.create_task(event_bus.dequeue())
        await event_bus.enqueue('dds:/asr', 'ordinary new')
        assert (await reader)['text'] == 'ordinary new'
        assert [ev['text'] for ev in event_bus.recent()] == ['ordinary new']
        await runtime.__aexit__()
    asyncio.run(run())


@pytest.mark.parametrize('personal', [True, False])
def test_shared_background_notifications_only_reach_ordinary_mode(runtime, monkeypatch, personal):
    desired(personal)
    pm.pin_settings()
    config.main['core'] = {'project_running': True}
    events = iter([{'source': source, 'text': 'shared task result', 'ts': 1.0}
                   for source in ('scheduler:task:old', 'subagent:old')])
    async def dequeue():
        try:
            return next(events)
        except StopIteration:
            raise asyncio.CancelledError
    emit = AsyncMock()
    monkeypatch.setattr(event_bus, 'dequeue', dequeue)
    monkeypatch.setattr(collector, '_emit_batch', emit)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collector._drain_loop())
    assert emit.await_count == (0 if personal else 2)


def test_blocked_collector_trigger_is_discarded_after_switch(runtime, monkeypatch):
    monkeypatch.setattr(collector, '_output', asyncio.Queue(maxsize=1))
    async def run():
        await runtime.__aenter__()
        desired(True)
        await runtime.apply_memory_mode()
        collector._output.put_nowait({'text': 'private first'})
        blocked = asyncio.create_task(collector._emit_batch([
            {'source': 'dds:/asr', 'text': 'private delayed', 'ts': 1.0}]))
        await asyncio.sleep(0)
        assert not blocked.done()
        desired(False)
        await runtime.apply_memory_mode()
        await blocked
        received, done = [], asyncio.Event()
        async def turn(ev, **kwargs):
            received.append(ev['text'])
            if ev['text'] == 'ordinary new':
                done.set()
        monkeypatch.setattr(runtime, '_one_turn', turn)
        loop = asyncio.create_task(runtime.run_forever())
        try:
            await collector._output.put({'text': 'ordinary new'})
            await asyncio.wait_for(done.wait(), 5)
            assert received == ['ordinary new']
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
            await runtime.__aexit__()
    asyncio.run(run())
