"""Event/turn fences tested with in-process mocks, not devices."""
import asyncio
import importlib
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.gettempdir(), 'project-stop-lifecycle.db'))

import collector
import config
import event_bus
import hooks
import mcp_client
from api import config as api_config
from event.llm import Event

llm = importlib.import_module('event.llm')


def test_collector_discards_bus_backlog_from_old_epoch(monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, 'main', {'core': {'project_running': True}})
        monkeypatch.setattr(event_bus, '_queue', asyncio.Queue())
        monkeypatch.setattr(event_bus, '_recent', [])
        monkeypatch.setattr(collector, '_output', asyncio.Queue())
        monkeypatch.setattr(collector, '_busy', False)
        await event_bus.enqueue('message', 'old')
        config.main['core']['project_running'] = False
        monkeypatch.setattr(api_config, '_stop_project_epoch', api_config._stop_project_epoch + 1)
        await event_bus.enqueue('message', 'during-stop')
        config.main['core']['project_running'] = True
        await event_bus.enqueue('message', 'new')
        task = asyncio.create_task(collector._drain_loop())
        try:
            trigger = await asyncio.wait_for(collector.next_trigger(), 1)
            assert 'new' in trigger['text']
            assert 'old' not in trigger['text'] and 'during-stop' not in trigger['text']
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())


def test_wakeup_hook_keeps_event_epoch_when_task_runs_after_restart(monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, 'main', {'core': {'project_running': True}})
        monkeypatch.setattr(event_bus, '_queue', asyncio.Queue())
        monkeypatch.setattr(event_bus, '_recent', [])
        monkeypatch.setattr(collector, '_output', asyncio.Queue())
        monkeypatch.setattr(collector, '_busy', False)
        captured, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def fire(name):
            captured.set()
            await release.wait()
            seen.append((name, mcp_client._dispatch_is_admitted()))
        monkeypatch.setattr(hooks, 'fire', fire)
        await event_bus.enqueue('asr', 'wake', {'kws_triggered': True})
        task = asyncio.create_task(collector._drain_loop())
        try:
            await asyncio.wait_for(captured.wait(), 1)
            config.main['core']['project_running'] = False
            monkeypatch.setattr(api_config, '_stop_project_epoch', api_config._stop_project_epoch + 1)
            config.main['core']['project_running'] = True
            release.set()
            await asyncio.sleep(0)
            assert seen == [('on_kws_wakeup', False)]
            assert mcp_client._dispatch_is_admitted()  # unrelated caller is unbound
        finally:
            release.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())


def test_kws_hook_real_dispatch_blocked_after_restart(monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, 'main', {'core': {'project_running': True}})
        monkeypatch.setattr(event_bus, '_queue', asyncio.Queue())
        monkeypatch.setattr(event_bus, '_recent', [])
        monkeypatch.setattr(collector, '_output', asyncio.Queue())
        monkeypatch.setattr(collector, '_busy', False)
        monkeypatch.setattr(hooks, '_registry', {})
        hooks.register('fake', 'indicator', {'on_kws_wakeup': {'action': 'wake'}})
        monkeypatch.setattr(mcp_client, 'registry', {'fake': {'online': True, 'url': 'http://example.invalid'}})
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        posts = []
        class Session:
            async def __aenter__(self):
                entered.set()
                await release.wait()
                return self
            async def __aexit__(self, *args):
                finished.set()
            def post(self, *args, **kwargs):
                posts.append(args)
                raise AssertionError('stale wakeup must not reach transport')
        monkeypatch.setattr(mcp_client.aiohttp, 'ClientSession', Session)
        await event_bus.enqueue('asr', 'wake', {'kws_triggered': True})
        task = asyncio.create_task(collector._drain_loop())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            config.main['core']['project_running'] = False
            monkeypatch.setattr(api_config, '_stop_project_epoch', api_config._stop_project_epoch + 1)
            config.main['core']['project_running'] = True
            release.set()
            await asyncio.wait_for(finished.wait(), 1)
            assert posts == []
        finally:
            release.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    asyncio.run(scenario())


def test_progress_ignores_old_busy_turn_after_restart(monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, 'main', {'core': {'project_running': True}})
        monkeypatch.setattr(collector, '_busy', True)
        monkeypatch.setattr(llm, '_narration_inflight', False)
        monkeypatch.setattr(llm, '_narration_thresholds', lambda: (0, 1))
        monkeypatch.setattr(llm, '_active_work_summary', lambda: ['new work'])
        monkeypatch.setattr(llm, '_active_work_detail', lambda: ('new work detail', 'new'))
        monkeypatch.setattr(llm, '_narration_enabled', lambda: True)
        monkeypatch.setattr(llm, '_mouth_busy', lambda: False)
        monkeypatch.setattr(llm, '_last_turn_restricted', False)
        monkeypatch.setattr(llm, '_same_as_last_report', lambda _text: False)
        monkeypatch.setattr(llm, '_notify_fire_spoke', lambda _results: True)
        monkeypatch.setattr(llm, '_on_speaking_started', lambda: None)
        monkeypatch.setattr(hooks, 'has_bindings', lambda _name: True)
        monkeypatch.setattr(llm.prompt_mod, 'build_system', lambda *a: 'system')
        agent = Event.__new__(Event)
        agent._current_turn_epoch = collector.project_epoch()
        agent._current_turn = [{'role': 'user', 'content': 'old work'}]
        monkeypatch.setattr(agent, '_bound_tool_names', lambda: [])
        calls, spoken, retries = [], [], []
        async def call(**kwargs):
            calls.append(kwargs)
            return {'content': 'new progress'}
        async def fire(*args, **kwargs):
            spoken.append(args)
            return [{}]
        async def push(*args, **kwargs):
            pass
        monkeypatch.setattr(llm.client, 'call', call)
        monkeypatch.setattr(hooks, 'fire', fire)
        monkeypatch.setattr(llm, 'push_event', push)
        monkeypatch.setattr(llm, '_restart_countdown', lambda: retries.append(True))
        config.main['core']['project_running'] = False
        await agent._report_progress()
        assert retries == []  # stopped projects do not reschedule
        monkeypatch.setattr(api_config, '_stop_project_epoch', api_config._stop_project_epoch + 1)
        config.main['core']['project_running'] = True
        await agent._report_progress()
        assert calls == [] and retries == [True]
        collector._busy = False  # old turn ended before the rescheduled deadline
        await agent._report_progress()
        assert len(calls) == 1 and spoken[0][0] == 'on_notify'
        assert 'new work detail' in str(calls[0]) and 'old work' not in str(calls[0])
    asyncio.run(scenario())


def test_delayed_llm_response_never_dispatches_after_restart(monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, 'main', {'core': {'project_running': True}, 'event': {'llm': {}}})
        monkeypatch.setattr(collector, '_steering_queue', asyncio.Queue())
        entered, release = asyncio.Event(), asyncio.Event()
        dispatched = []
        agent = Event.__new__(Event)
        agent._desktop_tools = SimpleNamespace(reset_python_namespace=lambda: None)
        agent._subagent_mgr = None
        agent._current_turn = []
        agent._sys_tools = {}
        agent._turns = []
        agent._summary = None
        agent._session_id = None
        monkeypatch.setattr(agent, '_get_bound_tool_schemas', lambda: [])
        monkeypatch.setattr(agent, '_build_history', lambda: [])
        monkeypatch.setattr(llm.prompt_mod, 'build_system', lambda *a, **kw: 'system')
        monkeypatch.setattr(llm.prompt_mod, 'build', lambda **kw: [{'role': 'user', 'content': 'hello'}])
        monkeypatch.setattr(llm, '_on_task_started', lambda: None)
        monkeypatch.setattr(mcp_client, 'has_blocking_pendings', lambda: False)
        async def noop(*args, **kwargs):
            return []
        async def call(**kwargs):
            entered.set()
            await release.wait()
            return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'function': {'name': 'mcp__test__move', 'arguments': '{}'}}]}
        async def dispatch(*args, **kwargs):
            dispatched.append(args)
        monkeypatch.setattr(llm, 'push_event', noop)
        monkeypatch.setattr(hooks, 'fire', noop)
        monkeypatch.setattr(llm.client, 'call', call)
        monkeypatch.setattr(mcp_client, 'call_tool', dispatch)
        epoch = collector.project_epoch()
        task = asyncio.create_task(agent._one_turn({'source': 'message', 'text': 'hello', '_project_epoch': epoch}))
        await asyncio.wait_for(entered.wait(), 1)
        config.main['core']['project_running'] = False
        monkeypatch.setattr(api_config, '_stop_project_epoch', api_config._stop_project_epoch + 1)
        config.main['core']['project_running'] = True
        release.set()
        await asyncio.wait_for(task, 1)
        assert dispatched == []
    asyncio.run(scenario())
