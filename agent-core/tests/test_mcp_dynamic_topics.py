"""SLAM calls preserve their live session and register resolved outputs."""
import asyncio
import json
import os
import pathlib
import sys
import tempfile
import types
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import mcp_manage
import config


@pytest.fixture
def http_tool(monkeypatch):
    calls, replies = [], {}
    monkeypatch.setattr(config, 'main', {})

    class Response:
        def __init__(self, payload):
            self.payload = payload

        async def json(self, **kwargs):
            return {'result': {'content': [{'type': 'text', 'text': json.dumps(self.payload)}]}}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __await__(self):
            return self.__aenter__().__await__()

    class Session(Response):
        def post(self, url, **kwargs):
            request = kwargs['json']
            if request['method'] == 'tools/call':
                arguments = request['params']['arguments']
                calls.append(arguments)
                return Response(replies.get(arguments['action'], {'state': 'idle'}))
            return Response({})

    monkeypatch.setattr(mcp_manage.aiohttp, 'ClientSession', lambda **kwargs: Session({}))

    def install(tool):
        monkeypatch.setattr(mcp_manage, '_get_mcp_list', lambda: [
            {'id': 'perception', 'url': 'http://test/mcp', 'tools': [
                {'name': tool, 'configSchema': {'properties': {
                    'child_from_cloud': {'type': 'string', 'scope': 'instance'}}}}]}])
        monkeypatch.setitem(config.main, f'tool_config:perception:{tool}:map',
                            {'child_from_cloud': 'saved-extrinsic'})

    return types.SimpleNamespace(calls=calls, replies=replies, install=install)


def _call(tool, action):
    return asyncio.run(mcp_manage.mcp_call_tool('perception',
        mcp_manage.MCPCallRequest(tool=tool, arguments={'action': action, 'instance_id': 'map'})))


def test_slam_start_applies_saved_extrinsic_before_start(http_tool):
    http_tool.install('pointcloud_slam')
    assert _call('pointcloud_slam', 'start')['code'] == 200
    assert http_tool.calls == [
        {'action': 'config', 'child_from_cloud': 'saved-extrinsic', 'instance_id': 'map'},
        {'action': 'start', 'instance_id': 'map'}]


def test_invalid_slam_config_prevents_start(http_tool):
    http_tool.install('pointcloud_slam')
    http_tool.replies['config'] = {'state': 'error', 'message': 'extrinsic must be a 4x4 matrix'}
    response = _call('pointcloud_slam', 'start')
    assert response['code'] == 400
    assert response['message'] == 'extrinsic must be a 4x4 matrix'
    assert [args['action'] for args in http_tool.calls] == ['config']


@pytest.mark.parametrize('action', ['start_mapping', 'stop_mapping', 'operation_status', 'select_map'])
def test_slam_session_actions_do_not_reapply_config(http_tool, action):
    http_tool.install('pointcloud_slam')
    http_tool.replies['config'] = {'state': 'error', 'message': 'stop before changing config'}
    assert _call('pointcloud_slam', action)['code'] == 200
    assert http_tool.calls == [{'action': action, 'instance_id': 'map'}]


def test_ordinary_tool_action_keeps_main_auto_config_behavior(http_tool):
    http_tool.install('visual_depth')
    assert _call('visual_depth', 'get_status')['code'] == 200
    assert [args['action'] for args in http_tool.calls] == ['config', 'get_status']


def test_completed_map_selection_registers_topic(monkeypatch, http_tool):
    http_tool.install('pointcloud_slam')
    topic = {'topic': '/pointcloud_slam/maps/test/history/segment',
             'format': 'sensor/slam-preview'}
    http_tool.replies['operation_status'] = {'state': 'map_selected', 'topic_out': [topic]}
    notify = AsyncMock()
    monkeypatch.setattr(mcp_manage, '_notify_inspector', notify)

    async def run():
        response = await mcp_manage.mcp_call_tool('perception',
            mcp_manage.MCPCallRequest(tool='pointcloud_slam',
                arguments={'action': 'operation_status', 'request_id': 'selected'}))
        assert response['code'] == 200
        await asyncio.sleep(0)
        notify.assert_awaited_once_with('perception', [topic], [])

    asyncio.run(run())
