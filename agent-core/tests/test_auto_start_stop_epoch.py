"""Mock-only tests for stop invalidating a pending boot auto-start."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import start  # noqa: E402
from api import config as api_config  # noqa: E402


@pytest.mark.parametrize('stop_before_task_runs', [True, False])
def test_stop_invalidates_scheduled_auto_start(monkeypatch, stop_before_task_runs):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        started = []
        monkeypatch.setattr(start.mcp_client, 'registry', {'device': {'online': False}})
        monkeypatch.setattr(api_config, '_stop_teardown_task', None)
        monkeypatch.setattr(api_config, '_start_project_task', None)
        monkeypatch.setattr(api_config, '_stop_project_epoch', 0)
        monkeypatch.setattr(api_config, '_deactivate_project_agent', lambda: None)
        monkeypatch.setattr(api_config, '_snapshot_stop_cards', lambda: [])

        async def wait_for_device(_seconds):
            entered.set()
            await release.wait()
            start.mcp_client.registry['device']['online'] = True

        async def broadcast():
            pass

        async def teardown(**_kwargs):
            pass

        async def start_impl():
            started.append(True)
            return True

        monkeypatch.setattr(start.asyncio, 'sleep', wait_for_device)
        monkeypatch.setattr(api_config, '_broadcast_project_stopped', broadcast)
        monkeypatch.setattr(api_config, '_do_stop_project_teardown', teardown)
        monkeypatch.setattr(api_config, '_do_start_project_impl', start_impl)
        captured = api_config.stop_project_epoch()
        task = asyncio.create_task(start._auto_start_project(captured))
        if not stop_before_task_runs:
            await asyncio.wait_for(entered.wait(), 1)
        await api_config.api_stop_project()
        await asyncio.wait_for(api_config._stop_teardown_task, 1)
        release.set()
        await asyncio.wait_for(task, 1)
        assert started == []

    asyncio.run(scenario())


@pytest.mark.parametrize('stop_mode', ['explicit', 'rollback'])
def test_loading_watcher_discards_info_returning_after_stop(monkeypatch, stop_mode):
    """Mock MCP only: no claim about its real inspector side effects."""
    import config as config_mod
    import types

    async def scenario():
        original_sleep = asyncio.sleep
        info_entered = asyncio.Event()
        release_info = asyncio.Event()
        watcher_done = asyncio.Event()
        registrations, events, watchers = [], [], []
        info_calls = 0

        async def sleep_fast(seconds):
            if seconds == 3:
                await original_sleep(0)
            else:
                await original_sleep(seconds)

        async def call(_mcp, req, timeout_s=None):
            nonlocal info_calls
            action = req.arguments['action']
            if action == 'start' and req.tool == 'loading':
                return {'code': 200, 'data': {'state': 'loading'}}
            if action == 'start':
                await asyncio.wait_for(info_entered.wait(), 1)
                return {'code': 200, 'data': {'state': 'error', 'error': 'simulated'}}
            if action == 'info' and req.tool == 'loading':
                info_calls += 1
                if info_calls == 2:
                    info_entered.set()
                    await release_info.wait()
                    try:
                        return {'code': 200, 'data': {'state': 'running',
                            'topic_out': [{'topic': '/old', 'format': 'data/json'}]}}
                    finally:
                        watcher_done.set()
                return {'code': 200, 'data': {'state': 'loading'}}
            return {'code': 200, 'data': {'state': 'idle'}}

        class Req:
            def __init__(self, tool, arguments):
                self.tool, self.arguments = tool, arguments

        async def push(event):
            events.append(event)

        async def register(*args):
            registrations.append(args)

        def create_task(coro):
            task = asyncio.get_running_loop().create_task(coro)
            watchers.append(task)
            return task

        mcp = types.ModuleType('api.mcp_manage')
        mcp.mcp_call_tool, mcp.MCPCallRequest = call, Req
        stream = types.ModuleType('api.motus_stream')
        stream.push_event = push
        inspection = types.ModuleType('api.inspection')
        inspection.register_topic_internal = register
        chan = types.ModuleType('channel.manager')
        chan.manager = types.SimpleNamespace(sync_from_canvas=lambda: None, _adapters={})
        chan._get_channel_configs = lambda: []
        for name, module in (('api.mcp_manage', mcp), ('api.motus_stream', stream),
                             ('api.inspection', inspection), ('channel.manager', chan)):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(api_config, '_stop_project_epoch', 0)
        monkeypatch.setattr(api_config, '_deactivate_project_agent', lambda: None)
        monkeypatch.setattr(asyncio, 'sleep', sleep_fast)
        monkeypatch.setattr(asyncio, 'create_task', create_task)
        cards = [{'id': 'a', 'mcpId': 'mcp', 'toolName': 'loading'}]
        if stop_mode == 'rollback':
            cards.append({'id': 'b', 'mcpId': 'mcp', 'toolName': 'fails'})
        monkeypatch.setitem(config_mod.main, 'canvas_layout', {'cards': cards, 'connections': []})
        monkeypatch.setitem(config_mod.main, 'core', {'project_running': False})
        if stop_mode == 'explicit':
            async def start_topics(_layout):
                pass
            monkeypatch.setitem(sys.modules, 'topic_actions',
                types.SimpleNamespace(manager=types.SimpleNamespace(start=start_topics)))

        result = await api_config._do_start_project_impl()
        assert result is (stop_mode == 'explicit')
        if stop_mode == 'explicit':
            await asyncio.wait_for(info_entered.wait(), 1)
            await api_config._do_stop_project(_lock_held=True)
        release_info.set()
        await asyncio.wait_for(watcher_done.wait(), 1)
        await asyncio.wait_for(asyncio.gather(*watchers), 1)
        assert registrations == []
        assert not any(e['type'] == 'project_start_item' and
                       e['payload']['tool'] == 'loading' and
                       e['payload']['status'] == 'ready' for e in events)

    asyncio.run(scenario())
