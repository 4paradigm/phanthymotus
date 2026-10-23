"""Image producers must send content accepted by OpenAI-compatible endpoints.

MiniMax M3.1 rejected the former string-valued image_url with HTTP 400.
Exercise both real producers through the agent's tool-content boundary.

Run: python3 -m pytest tests/test_image_content.py -q
"""
import asyncio
import base64
import importlib
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('DB_PATH', os.path.join(tempfile.mkdtemp(), 'test.db'))

import config  # noqa: E402
import mcp_client  # noqa: E402
from event.llm import _tool_content  # noqa: E402

desktop = importlib.import_module('event.desktop')
PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AA'
    'AAMBAQDJ/pLvAAAAAElFTkSuQmCC')


@pytest.fixture
def image_sources(tmp_path, monkeypatch):
    path = tmp_path / 'camera.png'
    path.write_bytes(PNG)
    monkeypatch.setattr(desktop, '_ALLOWED_DIRS', [str(tmp_path.resolve())])
    monkeypatch.setattr(config, 'main', {
        'event': {'llm': {'vision_input': True}},
    })
    monkeypatch.setattr(mcp_client, 'registry', {'camera': {
        'url': 'http://fixture.invalid/mcp', 'tools': ['capture'],
        'schemas': {}, 'input_schemas': {},
    }})

    async def camera_reply(session, url, method, params, req_id=1):
        assert method == 'tools/call'
        assert params == {'name': 'capture', 'arguments': {}}
        return {'content': [
            {'type': 'text', 'text': 'Camera frame'},
            {'type': 'image', 'mimeType': 'image/png',
             'data': base64.b64encode(PNG).decode('ascii')},
        ]}

    monkeypatch.setattr(mcp_client, '_jrpc', camera_reply)
    return {
        'read': lambda: desktop.DesktopTools().Read(str(path)),
        'mcp': lambda: mcp_client.call_tool('mcp__camera__capture', {}),
    }


@pytest.mark.parametrize('source', ['read', 'mcp'])
def test_image_reaches_tool_message_as_a_url_object(image_sources, source):
    content = _tool_content(asyncio.run(image_sources[source]()))
    images = [part for part in content if part['type'] == 'image_url']
    assert len(images) == 1
    url_object = images[0]['image_url']
    assert isinstance(url_object, dict), 'image_url must be an object, not a string'
    prefix, encoded = url_object['url'].split(',', 1)
    assert prefix == 'data:image/png;base64'
    assert base64.b64decode(encoded, validate=True) == PNG
    assert any(part['type'] == 'text' and part['text'] for part in content)


@pytest.mark.parametrize('source', ['read', 'mcp'])
def test_disabled_vision_still_reports_failure_without_image_data(image_sources, source):
    config.main['event']['llm']['vision_input'] = False
    content = _tool_content(asyncio.run(image_sources[source]()))
    assert isinstance(content, str)
    assert 'Error: cannot parse image contents' in content
    assert 'base64' not in content
