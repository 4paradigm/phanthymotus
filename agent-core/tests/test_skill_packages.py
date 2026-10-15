"""The Core transfers files only to a selected registered device card."""
import base64
import hashlib
import io
import json
import zipfile

import pytest

from test_skill_import import client, DEFINITION
import config
from api import skill_packages as packages


@pytest.fixture
def target(client):
    config.main['services'] = {'mcp': [{'id': 'robot-a', 'transport': 'http', 'tools': [{
        'name': 'g1_teaching', 'inputSchema': {'x-skill-package': {'schema': packages.SCHEMA},
            'x-workbench': {'protocol': 'motus.workbench/1'}}}]}]}
    config.main['canvas_layout'] = {'cards': [{'id': 'teach-a', 'mcpId': 'robot-a', 'toolName': 'g1_teaching'}]}
    return packages.targets()[0]


def package():
    out = io.BytesIO()
    manifest = {'schema': packages.SCHEMA, 'skill': {'name': '抬手'},
                'compatibility': {'model': 'G1', 'variant': 'G1_23', 'tool': 'g1_teaching'}}
    with zipfile.ZipFile(out, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('trajectory.jsonl', b'validated by the driver')
    return out.getvalue()


def test_import_uses_canvas_target_and_does_not_activate(client, target, monkeypatch):
    data = package()
    calls, received = [], bytearray()
    async def call(actual, action, **args):
        assert actual == target
        calls.append(action)
        if action == 'package_begin':
            assert args == {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            return {'transfer_id': 'local-transfer'}
        assert args['transfer_id'] == 'local-transfer'
        if action == 'package_write':
            assert args['offset'] == len(received)
            received.extend(base64.b64decode(args['data']))
            return {}
        return {'draft': {'id': 'new-local-draft'}, 'requires_local_trial': True, 'installed': False}
    monkeypatch.setattr(packages, 'call', call)
    response = client.post('/api/skills/package/inspect', files={'file': ('action.motus-skill', data)})
    assert response.json()['data']['targets'][0]['instance_id'] == 'teach-a'
    response = client.post('/api/skills/package/install', data={'card_id': 'teach-a'}, files={'file': ('action.motus-skill', data)})
    assert response.status_code == 200 and not response.json()['data']['installed']
    assert bytes(received) == data
    assert calls[0] == 'package_begin' and calls[-1] == 'package_finish'
    assert config.main['skills']['installed'] == []
    assert client.post('/api/skills/package/install', data={'card_id': 'not-on-canvas'}, files={'file': ('x', data)}).status_code == 409


def test_export_is_pinned_to_origin_and_checks_transfer_hash(client, target, monkeypatch):
    data = package()
    skill = {**DEFINITION, 'slug': 'taught-'+'a'*32, 'requiredTools': ['g1_teaching'],
             'compatibility': {'model': 'G1'},
             'attachment': {'schema': packages.SCHEMA, 'tool': 'g1_teaching', 'asset_id': 'a'*32},
             'origin': {'mcp_id': 'robot-a', 'instance_id': 'teach-a'}}
    assert client.post('/api/skills/import', json=skill).status_code == 200
    assert client.get('/api/skills').json()['data'][0]['compatibility'] == {'model': 'G1'}
    digest = hashlib.sha256(data).hexdigest()
    async def call(actual, action, **args):
        assert actual == target
        if action == 'package_export':
            assert args['asset_id'] == 'a'*32
            return {'transfer_id': 'download', 'bytes': len(data), 'sha256': digest}
        return {'data': base64.b64encode(data[args['offset']:args['offset']+24576]).decode()}
    monkeypatch.setattr(packages, 'call', call)
    url = '/api/skills/package/export/' + skill['slug']
    response = client.get(url)
    assert response.content == data and response.headers['content-type'] == 'application/zip'
    digest = '0'*64
    assert client.get(url).status_code == 502
    config.main['canvas_layout']['cards'][0]['id'] = 'replacement-card'
    assert client.get(url).status_code == 409


@pytest.mark.parametrize('data', [b'not a zip', b'', b'x' * (packages.MAX_BYTES + 1)])
def test_invalid_upload_rejected_before_contacting_robot(client, target, data, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail('invalid package must not contact driver')
    monkeypatch.setattr(packages, 'call', unexpected)
    assert client.post('/api/skills/package/install', data={'card_id': 'teach-a'},
                       files={'file': ('bad.zip', data)}).status_code in (413, 422)


def test_refresh_migrates_legacy_instruction_preserving_visibility(client, target, monkeypatch):
    old = {**DEFINITION, 'slug': 'taught-'+'a'*32, 'requiredTools': ['g1_teaching']}
    client.post('/api/skills/import', json=old)
    config.main['skills']['installed'][0]['active'] = False
    updated = {**old, 'instruction': 'Complete preview and run arguments',
               'attachment': {'schema': packages.SCHEMA, 'tool': 'g1_teaching', 'asset_id': 'a'*32},
               'compatibility': {'variant': 'G1_23'}}
    async def call(actual, action, **args):
        assert actual == target and action == 'package_definition' and args == {'asset_id': 'a'*32}
        return {'skill': updated}
    monkeypatch.setattr(packages, 'call', call)
    assert client.post('/api/skills/package/refresh/'+old['slug']).status_code == 200
    installed = config.main['skills']['installed']
    assert len(installed) == 1 and not installed[0]['active']
    assert installed[0]['instruction'] == updated['instruction']
    assert installed[0]['origin'] == {'mcp_id': 'robot-a', 'instance_id': 'teach-a'}
