"""A canvas map selection must register its output for the monitor."""
import asyncio
import json
import os
import pathlib
import sys
import tempfile
from unittest.mock import AsyncMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

from api import mcp_manage


def test_completed_map_selection_registers_topic(monkeypatch):
    topic = {'topic': '/pointcloud_slam/maps/test/history/segment',
             'format': 'sensor/slam-preview'}
    result = {'result': {'content': [{'type': 'text', 'text': json.dumps({
        'state': 'map_selected', 'topic_out': [topic]})}]}}

    class Response:
        async def json(self, **kwargs):
            return result

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __await__(self):
            return self.__aenter__().__await__()

    class Session(Response):
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(mcp_manage.aiohttp, 'ClientSession', lambda **kwargs: Session())
    monkeypatch.setattr(mcp_manage, '_get_mcp_list', lambda: [
        {'id': 'perception', 'url': 'http://test/mcp', 'tools': []}])
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
