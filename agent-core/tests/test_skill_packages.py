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
        'name': 'g1_teaching', 'inputSchema': {'x-skill-package': {'schema': packages.SCHEMA, 'pending': True, 'session_authorization': True},
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


def test_execution_policy_is_local_bound_and_card_overrides_skill(client, target):
    from api.skill_packages import execution_key, execution_setting
    device, asset, recording = 'd'*64, 'a'*32, 'b'*64
    config.main['teaching_execution_policies'] = {execution_key(target)+':'+asset: {
        'enabled': True, 'device_binding': device, 'asset_binding': recording}}
    assert execution_setting(target, device, asset, recording)['source'] == 'skill'
    assert not execution_setting(target, device, asset, 'changed')['enabled']
    assert not execution_setting(target, 'other-device', asset, recording)['enabled']
    key = 'tool_config:'+execution_key(target)
    config.main[key] = {'auto_execute_verified': True, '_execution_binding': device}
    assert execution_setting(target, device, 'new-validated-asset', 'new')['source'] == 'card'
    config.main[key] = {'auto_execute_verified': False}
    assert execution_setting(target, device, asset, recording)['source'] == 'skill'
    assert not execution_setting(target, device, 'new-validated-asset', 'new')['enabled']


def test_execution_settings_require_browser_and_driver_reads_are_loopback_only(client, target, monkeypatch):
    import auth
    monkeypatch.setattr(auth, '_token', 'browser-only-token')
    assert client.post('/api/skills/package/execution/taught-'+'a'*32,
                       json={'enabled': True}).status_code == 401
    assert client.post('/api/skills/package/execution/read', json={}).status_code == 403
    config.main['services']['mcp'][0]['tools'][0]['inputSchema']['x-skill-package']['persistent_execution'] = True
    url = '/api/canvas/tool-config/robot-a/g1_teaching/teach-a'
    assert client.put(url, json={'auto_execute_verified': True, '_execution_binding': 'forged'}).status_code == 401


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


def test_session_authorization_requires_browser_token_and_one_use_local_receipt(client, target, monkeypatch):
    import auth
    from starlette.requests import Request
    from fastapi import HTTPException
    monkeypatch.setattr(auth, '_token', 'browser-only-token')
    packages._approvals.clear()
    definition = {**DEFINITION, 'slug': 'taught-'+'a'*32, 'requiredTools': ['g1_teaching'],
                  'attachment': {'schema': packages.SCHEMA, 'tool': 'g1_teaching', 'asset_id': 'a'*32}}
    client.post('/api/skills/import', json=definition)
    seen = []
    async def call(actual, action, **args):
        seen.append(action)
        if action == 'session_challenge': return {'nonce': 'driver-one-use-nonce'}
        if action == 'session_authorize':
            assert args['asset_id'] == 'a'*32
            req = Request({'type': 'http', 'client': ('127.0.0.1', 1234)})
            body = {'approval_id': args['approval_id']}
            receipt = (await packages.consume_approval(req, body))['data']
            assert receipt == {'nonce': 'driver-one-use-nonce', 'asset_id': 'a'*32,
                               'tool': 'g1_teaching', 'instance_id': 'teach-a', 'seconds': 1800}
            with pytest.raises(HTTPException): await packages.consume_approval(req, body)
            return {'active': True}
        if action == 'package_definition': return {'skill': definition}
        if action == 'session_revoke': return {'active': False}
        pytest.fail(action)
    monkeypatch.setattr(packages, 'call', call)
    url = '/api/skills/package/session/'+definition['slug']
    body = {'enabled': True, 'workspace_clear': True, 'take_control': True}
    assert client.post(url, json=body).status_code == 401
    assert client.post(url, json=body, headers={'Authorization':'Bearer model-forgery'}).status_code == 401
    headers = {'Authorization': 'Bearer browser-only-token'}
    assert client.post(url, json={'enabled': True}, headers=headers).status_code == 422
    assert not seen
    assert client.post(url, json=body, headers=headers).json()['data']['active']
    assert not packages._approvals
    assert client.post(url, json={'enabled': False}, headers=headers).json()['data']['active'] is False
    assert seen == ['session_challenge', 'session_authorize', 'package_definition', 'session_revoke']


def test_approval_consume_is_loopback_only_and_expires(client, monkeypatch):
    import asyncio
    from starlette.requests import Request
    from fastapi import HTTPException
    ident = 'a'*64
    packages._approvals[ident] = {'expires': 0, 'receipt': {}}
    # Even a valid browser token must not turn the driver receipt API into a grant API.
    assert client.post('/api/skills/package/approval/consume', json={'approval_id':ident}).status_code == 403
    local = Request({'type':'http', 'client':('127.0.0.1', 1234)})
    with pytest.raises(HTTPException, match='授权凭据'):
        asyncio.run(packages.consume_approval(local, {'approval_id':ident}))
    assert ident not in packages._approvals


def test_pending_resume_uses_original_card_without_installing(client, target, monkeypatch):
    seen = []
    async def call(actual, action, **args):
        assert actual == target
        seen.append(action)
        if action == 'draft_list': return {'drafts': [{'id':'a'*32, 'name':'待验证动作'}]}
        if action == 'draft_open':
            assert args == {'draft_id':'a'*32}
            return {'draft': {'id':'a'*32}}
        if action == 'ui': return {'protocol':'motus.workbench/1', 'html':'<p>检查</p>'}
        pytest.fail(action)
    monkeypatch.setattr(packages, 'call', call)
    result = client.get('/api/skills/package/pending').json()['data']
    assert result['drafts'][0]['target'] == target
    response = client.post('/api/skills/package/resume', json={'card_id':'teach-a', 'draft_id':'a'*32})
    assert response.json()['data']['ui']['html'] == '<p>检查</p>'
    assert seen == ['draft_list','draft_open','ui']
    assert config.main['skills']['installed'] == []


def test_reimport_reuses_local_receipt_without_reenabling_existing_skill(client, target, monkeypatch):
    definition = {**DEFINITION, 'slug':'taught-'+'a'*32, 'requiredTools':['g1_teaching'],
                  'attachment': {'schema':packages.SCHEMA, 'tool':'g1_teaching','asset_id':'a'*32},
                  'origin':{'mcp_id':'robot-a','instance_id':'teach-a'}}
    client.post('/api/skills/import', json=definition)
    config.main['skills']['installed'][0]['active'] = False
    async def call(actual, action, **args):
        if action == 'package_begin': return {'transfer_id':'t'}
        if action == 'package_write': return {}
        return {'reused':True, 'asset_id':'a'*32, 'skill':definition}
    monkeypatch.setattr(packages, 'call', call)
    response = client.post('/api/skills/package/install', data={'card_id':'teach-a'}, files={'file':('action.motus-skill',package())})
    assert response.json()['data']['reused']
    assert len(config.main['skills']['installed']) == 1
    assert not config.main['skills']['installed'][0]['active']
