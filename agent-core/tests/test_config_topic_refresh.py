"""A config-completion event must not race the driver's config application."""
import asyncio
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from api import canvas
import tool_config


@pytest.mark.parametrize('instance_id', ['', 'card-a'])
def test_topic_refresh_notification_follows_driver_config(monkeypatch, instance_id):
    events = []
    tasks = []
    monkeypatch.setattr(canvas, '_broadcast', events.append)
    monkeypatch.setattr(tool_config, 'find_tool', lambda *_: {
        'configSchema': {'properties': {'mode': {'type': 'string', 'scope': 'instance'}}},
    })

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()

        async def call(mcp_id, request):
            assert mcp_id == 'robot'
            assert request.tool == 'sensor'
            assert request.arguments['action'] == 'config'
            started.set()
            await finish.wait()
            return {'code': 200}

        monkeypatch.setitem(sys.modules, 'api.mcp_manage', types.SimpleNamespace(
            mcp_call_tool=call, MCPCallRequest=lambda **kwargs: types.SimpleNamespace(**kwargs)))
        create_task = asyncio.create_task

        def track(coro):
            task = create_task(coro)
            tasks.append(task)
            return task

        monkeypatch.setattr(canvas.asyncio, 'create_task', track)
        canvas.apply_tool_config('robot', 'sensor', {'mode': 'changed'}, instance_id)
        await started.wait()
        assert events == [], 'querying info before config completes can return old topics'
        finish.set()
        await asyncio.gather(*tasks)
        assert events == [{'type': 'tool_config_updated', 'payload': {
            'mcp_id': 'robot', 'tool': 'sensor', 'instance_id': instance_id}}]

    asyncio.run(run())
