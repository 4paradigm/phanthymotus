"""Core transfers a package to one selected card, and knows nothing more.

Every identifier in this file is deliberately **not** one any shipped driver
uses: the protocol is `example.device.package/1`, the legacy slug prefix is
`recorded-`, the asset id is not 32 hex and the binding is not 64 chars. If any
of these have to match something spelled inside agent-core, that is the
coupling these tests exist to catch.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
         python3 -m pytest tests/test_skill_packages.py -q
"""
import base64
import hashlib
import io
import json
import zipfile

import fastapi
import pytest
from fastapi.testclient import TestClient

from test_skill_import import client, DEFINITION          # noqa: F401 - fixture
import config
from api import skill_packages as packages

SCHEMA = 'example.device.package/1'
PREFIX = 'recorded-'
ASSET = 'asset-7'                    # not 32 hex, on purpose
BINDING = 'device-binding-v3'         # not 64 chars, on purpose
SLUG = PREFIX + ASSET
TOOL = 'example_recorder'

CAPABILITIES = {'schema': SCHEMA, 'pending': True, 'session_authorization': True,
                'legacy_slug_prefix': PREFIX}


def _services(**extra):
    return {'mcp': [{'id': 'robot-a', 'transport': 'http', 'tools': [{
        'name': TOOL, 'inputSchema': {'x-skill-package': {**CAPABILITIES, **extra},
                                      'x-workbench': {'protocol': 'motus.workbench/1'}}}]}]}


@pytest.fixture
def target(client):
    config.main['services'] = _services()
    config.main['canvas_layout'] = {'cards': [{'id': 'card-a', 'mcpId': 'robot-a',
                                               'toolName': TOOL}]}
    return packages.targets()[0]


@pytest.fixture
def full_client(monkeypatch):
    """An app with the canvas router too.

    The `client` fixture mounts only `skills.router`, so a request to
    `/api/canvas/...` through it 404s. An assertion written against it that
    expected a 401 therefore passed no judgement on anything -- which is how
    the forged-binding gate shipped with no coverage at all.
    """
    monkeypatch.setattr(config, 'main', {'skills': {'installed': []}})
    from api import canvas, skills
    app = fastapi.FastAPI()
    app.include_router(skills.router, prefix='/api')
    app.include_router(canvas.router, prefix='/api')
    config.main['services'] = _services(persistent_execution=True)
    config.main['canvas_layout'] = {'cards': [{'id': 'card-a', 'mcpId': 'robot-a',
                                               'toolName': TOOL}]}
    with TestClient(app) as test_client:
        yield test_client


def package(schema=SCHEMA, entries=(('payload.bin', b'opaque to Core'),)):
    out = io.BytesIO()
    manifest = {'schema': schema, 'skill': {'name': '示例技能'},
                'compatibility': {'labels': ['Example', 'Model-9'], 'tool': TOOL}}
    with zipfile.ZipFile(out, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        for name, body in entries:
            archive.writestr(name, body)
    return out.getvalue()


def installed(**extra):
    return {**DEFINITION, 'slug': SLUG, 'requiredTools': [TOOL],
            'attachment': {'schema': SCHEMA, 'tool': TOOL, 'asset_id': ASSET},
            'origin': {'mcp_id': 'robot-a', 'instance_id': 'card-a'}, **extra}


# ── Core holds no application vocabulary ─────────────────────────────────────

def test_core_spells_no_protocol_identifier_of_its_own(target):
    """A card's declared schema is the only identifier in play."""
    assert target['schema'] == SCHEMA
    assert not hasattr(packages, 'SCHEMA')
    source = io.open(packages.__file__, encoding='utf-8').read()
    for word in ('teaching', 'taught', 'trajectory.jsonl', '示教', 'dual_arm', 'fixed_hand'):
        assert word not in source.replace(
            source[source.index('"""'):source.index('"""', 3) + 3], ''), word


def test_any_payload_entry_name_is_accepted(target):
    """Core required a literal `trajectory.jsonl` and nothing else. A second
    kind of package was therefore an edit to this file."""
    for entries in [(('trajectory.jsonl', b'x'),), (('frames/0001.npy', b'x'), ('notes.md', b'y')),
                    ()]:
        assert packages.manifest_from(package(entries=entries))['schema'] == SCHEMA


def test_a_package_for_another_protocol_is_refused_at_the_card(client, target, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail('mismatched package must not reach the device')
    monkeypatch.setattr(packages, 'call', unexpected)
    wrong = package(schema='other.vendor.package/2')
    assert client.post('/api/skills/package/install', data={'card_id': 'card-a'},
                       files={'file': ('p.motus-skill', wrong)}).status_code == 409
    assert client.post('/api/skills/package/inspect',
                       files={'file': ('p.motus-skill', wrong)}).json()['data']['targets'] == []


def test_legacy_slug_resolution_follows_the_declared_prefix(client, target):
    """The prefix is the card's, not Core's."""
    client.post('/api/skills/import', json={**DEFINITION, 'slug': SLUG, 'requiredTools': [TOOL]})
    assert packages.resolve_skill(SLUG)[2] == ASSET
    config.main['services'] = _services(legacy_slug_prefix='captured-')
    with pytest.raises(fastapi.HTTPException):
        packages.resolve_skill(SLUG)


@pytest.mark.parametrize('entries', [
    (('../escape.bin', b'x'),),                       # path traversal
    (('/abs.bin', b'x'),),
    (('big.bin', b'0' * (packages.MAX_ENTRY_BYTES + 1)),),
])
def test_archive_boundaries_are_enforced(entries):
    with pytest.raises(fastapi.HTTPException):
        packages.manifest_from(package(entries=entries))


# ── the binding contract ─────────────────────────────────────────────────────

def test_a_device_may_return_more_than_core_reads():
    """`execution_setting(**binding)` used to splat the device's reply into a
    three-parameter function, so the first driver to include the model and
    calibration summary its own protocol calls for got a 500."""
    parsed = packages.read_binding({'device_binding': BINDING, 'asset_binding': 'rec-1',
                                    'model': 'Model-9', 'calibration': {'sha': 'abc'}},
                                   asset_id=ASSET)
    assert parsed == {'device_binding': BINDING, 'asset_id': ASSET, 'asset_binding': 'rec-1'}


@pytest.mark.parametrize('reply', [{}, None, 'text', {'device_binding': ''},
                                   {'device_binding': 'x' * 200},
                                   {'device_binding': BINDING, 'asset_binding': 5}])
def test_a_missing_or_malformed_binding_is_a_protocol_error(reply):
    with pytest.raises(fastapi.HTTPException) as caught:
        packages.read_binding(reply)
    assert caught.value.status_code == 502


def test_execution_policy_is_local_bound_and_card_overrides_skill(client, target):
    from api.skill_packages import execution_key, execution_setting
    recording = 'rec-1'
    config.main[packages._POLICY_KEY] = {execution_key(target) + ':' + ASSET: {
        'enabled': True, 'device_binding': BINDING, 'asset_binding': recording}}
    assert execution_setting(target, BINDING, ASSET, recording)['source'] == 'skill'
    assert not execution_setting(target, BINDING, ASSET, 'changed')['enabled']
    assert not execution_setting(target, 'other-device', ASSET, recording)['enabled']
    key = 'tool_config:' + execution_key(target)
    config.main[key] = {'auto_execute_verified': True, '_execution_binding': BINDING}
    assert execution_setting(target, BINDING, 'new-asset', 'new')['source'] == 'card'
    config.main[key] = {'auto_execute_verified': False}
    assert execution_setting(target, BINDING, ASSET, recording)['source'] == 'skill'
    assert not execution_setting(target, BINDING, 'new-asset', 'new')['enabled']


# ── who may grant automatic execution ────────────────────────────────────────

def test_a_forged_binding_on_the_generic_config_route_is_rejected(full_client, monkeypatch):
    import auth
    monkeypatch.setattr(auth, '_token', 'browser-only-token')
    url = '/api/canvas/tool-config/robot-a/' + TOOL + '/card-a'
    body = {'auto_execute_verified': True, '_execution_binding': 'forged'}
    assert full_client.put(url, json=body).status_code == 401
    assert full_client.put(url, json=body,
                           headers={'Authorization': 'Bearer wrong'}).status_code == 401

    async def call(actual, action, **args):
        assert action == 'execution_binding'
        return {'device_binding': BINDING}
    monkeypatch.setattr(packages, 'call', call)
    headers = {'Authorization': 'Bearer browser-only-token'}
    assert full_client.put(url, json=body, headers=headers).status_code == 200
    saved = config.main['tool_config:robot-a:' + TOOL + ':card-a']
    assert saved['_execution_binding'] == BINDING, '调用方送来的绑定必须被丢弃'
    assert full_client.put(url, json={'auto_execute_verified': 'yes'},
                           headers=headers).status_code == 422


def test_an_unrelated_card_config_save_is_untouched(full_client, monkeypatch):
    """The gate hangs off a declaration, so every other tool saves as before."""
    import auth
    monkeypatch.setattr(auth, '_token', 'browser-only-token')
    config.main['services'] = {'mcp': [{'id': 'robot-a', 'transport': 'http',
                                        'tools': [{'name': 'asr', 'inputSchema': {}}]}]}

    async def unexpected(*args, **kwargs):
        pytest.fail('a plain config save must not contact a device')
    monkeypatch.setattr(packages, 'call', unexpected)
    assert full_client.put('/api/canvas/tool-config/robot-a/asr',
                           json={'provider': 'local'}).status_code == 200


def test_execution_settings_require_browser_and_driver_reads_are_loopback_only(
        client, target, monkeypatch):
    import auth
    monkeypatch.setattr(auth, '_token', 'browser-only-token')
    assert client.post('/api/skills/package/execution/' + SLUG,
                       json={'enabled': True}).status_code == 401
    assert client.post('/api/skills/package/execution/read', json={}).status_code == 403


def test_an_auth_disabled_host_says_so_instead_of_asking_for_a_login(client, target, monkeypatch):
    """`get_token()` is `''` when the deployment has no ACCESS_TOKEN, so the
    gate can never open. Telling the operator to log in sends them after a
    login that does not exist on that machine."""
    import auth
    monkeypatch.setattr(auth, '_token', '')
    response = client.post('/api/skills/package/execution/' + SLUG, json={'enabled': True})
    assert response.status_code == 403 and 'ACCESS_TOKEN' in response.json()['detail']


def test_disabling_revokes_before_it_persists(client, target, monkeypatch):
    """A device that refuses the revoke must leave the stored setting alone.

    Persisting first reported a 409 to the page *and* left the old temporary
    grant alive -- the one thing this call exists to prevent.
    """
    import auth
    monkeypatch.setattr(auth, '_token', 'tok')
    client.post('/api/skills/import', json=installed())
    config.main['services'] = _services(persistent_execution=True)
    key = packages.execution_key(packages.targets()[0]) + ':' + ASSET
    config.main[packages._POLICY_KEY] = {key: {'enabled': True, 'device_binding': BINDING,
                                               'asset_binding': 'rec-1'}}

    async def call(actual, action, **args):
        if action == 'execution_binding':
            return {'device_binding': BINDING, 'asset_binding': 'rec-1'}
        if action == 'session_revoke':
            raise fastapi.HTTPException(409, '设备正在执行，暂不能撤销')
        pytest.fail(action)
    monkeypatch.setattr(packages, 'call', call)
    response = client.post('/api/skills/package/execution/' + SLUG, json={'enabled': False},
                           headers={'Authorization': 'Bearer tok'})
    assert response.status_code == 409
    assert config.main[packages._POLICY_KEY][key]['enabled'] is True, '拒绝时不得改写设置'


def test_enabling_a_whole_card_migrates_bounded(client, target, monkeypatch):
    import auth
    monkeypatch.setattr(auth, '_token', 'tok')
    config.main['services'] = _services(persistent_execution=True)
    for index in range(3):
        client.post('/api/skills/import', json=installed(
            slug=f'{PREFIX}asset-{index}',
            attachment={'schema': SCHEMA, 'tool': TOOL, 'asset_id': f'asset-{index}'}))
    refreshed = []

    async def call(actual, action, **args):
        if action == 'execution_binding':
            return {'device_binding': BINDING}
        if action == 'package_definition':
            asset = args['asset_id']
            refreshed.append(asset)
            if asset == 'asset-1':
                raise fastapi.HTTPException(409, '设备忙')
            skill = installed(slug=PREFIX + asset, instruction='新的调用说明',
                              attachment={'schema': SCHEMA, 'tool': TOOL, 'asset_id': asset})
            return {'skill': skill}
        pytest.fail(action)
    monkeypatch.setattr(packages, 'call', call)
    monkeypatch.setattr(packages, 'prepare_execution_config', packages.prepare_execution_config)

    from starlette.requests import Request
    import asyncio
    request = Request({'type': 'http', 'client': ('127.0.0.1', 1),
                       'headers': [(b'authorization', b'Bearer tok')]})
    body = asyncio.run(packages.prepare_execution_config(
        'robot-a', TOOL, {'auto_execute_verified': True}, 'card-a', request))
    assert body['_execution_binding'] == BINDING
    assert sorted(refreshed) == ['asset-0', 'asset-1', 'asset-2']
    # One device refusal must not abort the save, nor the other migrations.
    instructions = {s['slug']: s['instruction'] for s in config.main['skills']['installed']}
    assert instructions[PREFIX + 'asset-0'] == '新的调用说明'
    assert instructions[PREFIX + 'asset-2'] == '新的调用说明'
    assert instructions[PREFIX + 'asset-1'] == DEFINITION['instruction']


def test_already_bound_card_does_not_re_migrate(client, target, monkeypatch):
    import asyncio
    import auth
    from starlette.requests import Request
    monkeypatch.setattr(auth, '_token', 'tok')
    config.main['services'] = _services(persistent_execution=True)
    client.post('/api/skills/import', json=installed())
    config.main['tool_config:robot-a:' + TOOL + ':card-a'] = {
        'auto_execute_verified': True, '_execution_binding': BINDING}

    async def call(actual, action, **args):
        if action == 'execution_binding':
            return {'device_binding': BINDING}
        pytest.fail(f'already bound: {action} must not be called again')
    monkeypatch.setattr(packages, 'call', call)
    request = Request({'type': 'http', 'client': ('127.0.0.1', 1),
                       'headers': [(b'authorization', b'Bearer tok')]})
    asyncio.run(packages.prepare_execution_config(
        'robot-a', TOOL, {'auto_execute_verified': True}, 'card-a', request))


# ── transfer ─────────────────────────────────────────────────────────────────

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
        return {'draft': {'id': 'new-local-draft'}, 'requires_local_trial': True,
                'installed': False}
    monkeypatch.setattr(packages, 'call', call)
    response = client.post('/api/skills/package/inspect',
                           files={'file': ('p.motus-skill', data)})
    assert response.json()['data']['targets'][0]['instance_id'] == 'card-a'
    response = client.post('/api/skills/package/install', data={'card_id': 'card-a'},
                           files={'file': ('p.motus-skill', data)})
    assert response.status_code == 200 and not response.json()['data']['installed']
    assert bytes(received) == data
    assert calls[0] == 'package_begin' and calls[-1] == 'package_finish'
    assert config.main['skills']['installed'] == []
    assert client.post('/api/skills/package/install', data={'card_id': 'not-on-canvas'},
                       files={'file': ('x', data)}).status_code == 409


def test_a_transfer_that_never_finishes_is_bounded(client, target, monkeypatch):
    """16 MiB is up to 683 chunks. Without a deadline over the loop an upload
    holds the request open indefinitely, which is what every other
    device-facing path here already refuses to do."""
    import asyncio
    monkeypatch.setattr(packages, 'TRANSFER_DEADLINE_S', 0.2)

    async def call(actual, action, **args):
        if action == 'package_begin':
            return {'transfer_id': 't'}
        await asyncio.sleep(10)
    monkeypatch.setattr(packages, 'call', call)
    with pytest.raises(asyncio.TimeoutError):
        client.post('/api/skills/package/install', data={'card_id': 'card-a'},
                    files={'file': ('p.motus-skill', package())})


def test_export_is_pinned_to_origin_and_checks_transfer_hash(client, target, monkeypatch):
    data = package()
    client.post('/api/skills/import', json=installed(compatibility={'labels': ['Model-9']}))
    assert client.get('/api/skills').json()['data'][0]['compatibility'] == {'labels': ['Model-9']}
    digest = hashlib.sha256(data).hexdigest()

    async def call(actual, action, **args):
        assert actual == target
        if action == 'package_export':
            assert args['asset_id'] == ASSET
            return {'transfer_id': 'download', 'bytes': len(data), 'sha256': digest}
        return {'data': base64.b64encode(
            data[args['offset']:args['offset'] + packages.CHUNK_BYTES]).decode()}
    monkeypatch.setattr(packages, 'call', call)
    url = '/api/skills/package/export/' + SLUG
    response = client.get(url)
    assert response.content == data and response.headers['content-type'] == 'application/zip'
    assert 'filename="recorded-asset-7.motus-skill"' in response.headers['content-disposition']
    digest = '0' * 64
    assert client.get(url).status_code == 502
    config.main['canvas_layout']['cards'][0]['id'] = 'replacement-card'
    assert client.get(url).status_code == 409


@pytest.mark.parametrize('data', [b'not a zip', b'', b'x' * (packages.MAX_BYTES + 1)])
def test_invalid_upload_rejected_before_contacting_robot(client, target, data, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail('invalid package must not contact driver')
    monkeypatch.setattr(packages, 'call', unexpected)
    assert client.post('/api/skills/package/install', data={'card_id': 'card-a'},
                       files={'file': ('bad.zip', data)}).status_code in (413, 422)


def test_refresh_migrates_legacy_instruction_preserving_visibility(client, target, monkeypatch):
    old = {**DEFINITION, 'slug': SLUG, 'requiredTools': [TOOL]}
    client.post('/api/skills/import', json=old)
    config.main['skills']['installed'][0]['active'] = False
    updated = {**old, 'instruction': 'Complete preview and run arguments',
               'attachment': {'schema': SCHEMA, 'tool': TOOL, 'asset_id': ASSET},
               'compatibility': {'labels': ['Model-9']}}

    async def call(actual, action, **args):
        assert actual == target and action == 'package_definition' and args == {'asset_id': ASSET}
        return {'skill': updated}
    monkeypatch.setattr(packages, 'call', call)
    assert client.post('/api/skills/package/refresh/' + SLUG).status_code == 200
    rows = config.main['skills']['installed']
    assert len(rows) == 1 and not rows[0]['active']
    assert rows[0]['instruction'] == updated['instruction']
    assert rows[0]['origin'] == {'mcp_id': 'robot-a', 'instance_id': 'card-a'}


def test_session_authorization_requires_browser_token_and_one_use_local_receipt(
        client, target, monkeypatch):
    import auth
    from starlette.requests import Request
    from fastapi import HTTPException
    monkeypatch.setattr(auth, '_token', 'browser-only-token')
    packages._approvals.clear()
    definition = installed()
    client.post('/api/skills/import', json=definition)
    seen = []

    async def call(actual, action, **args):
        seen.append(action)
        if action == 'session_challenge':
            return {'nonce': 'driver-one-use-nonce'}
        if action == 'session_authorize':
            assert args['asset_id'] == ASSET
            req = Request({'type': 'http', 'client': ('127.0.0.1', 1234)})
            body = {'approval_id': args['approval_id']}
            receipt = (await packages.consume_approval(req, body))['data']
            assert receipt == {'nonce': 'driver-one-use-nonce', 'asset_id': ASSET,
                               'tool': TOOL, 'instance_id': 'card-a', 'seconds': 1800}
            with pytest.raises(HTTPException):
                await packages.consume_approval(req, body)
            return {'active': True}
        if action == 'package_definition':
            return {'skill': definition}
        if action == 'session_revoke':
            return {'active': False}
        pytest.fail(action)
    monkeypatch.setattr(packages, 'call', call)
    url = '/api/skills/package/session/' + SLUG
    body = {'enabled': True, 'workspace_clear': True, 'take_control': True}
    assert client.post(url, json=body).status_code == 401
    assert client.post(url, json=body,
                       headers={'Authorization': 'Bearer model-forgery'}).status_code == 401
    headers = {'Authorization': 'Bearer browser-only-token'}
    assert client.post(url, json={'enabled': True}, headers=headers).status_code == 422
    assert not seen
    assert client.post(url, json=body, headers=headers).json()['data']['active']
    assert not packages._approvals
    assert client.post(url, json={'enabled': False},
                       headers=headers).json()['data']['active'] is False
    assert seen == ['session_challenge', 'session_authorize', 'package_definition',
                    'session_revoke']


def test_approval_consume_is_loopback_only_and_expires(client, monkeypatch):
    import asyncio
    from starlette.requests import Request
    from fastapi import HTTPException
    ident = 'a' * 64
    packages._approvals[ident] = {'expires': 0, 'receipt': {}}
    # Even a valid browser token must not turn the driver receipt API into a grant API.
    assert client.post('/api/skills/package/approval/consume',
                       json={'approval_id': ident}).status_code == 403
    local = Request({'type': 'http', 'client': ('127.0.0.1', 1234)})
    with pytest.raises(HTTPException, match='授权凭据'):
        asyncio.run(packages.consume_approval(local, {'approval_id': ident}))
    assert ident not in packages._approvals


def test_pending_resume_uses_original_card_without_installing(client, target, monkeypatch):
    seen = []

    async def call(actual, action, **args):
        assert actual == target
        seen.append(action)
        if action == 'draft_list':
            return {'drafts': [{'id': 'draft-1', 'name': '待验证技能'}]}
        if action == 'draft_open':
            assert args == {'draft_id': 'draft-1'}
            return {'draft': {'id': 'draft-1'}}
        if action == 'ui':
            return {'protocol': 'motus.workbench/1', 'html': '<p>检查</p>'}
        pytest.fail(action)
    monkeypatch.setattr(packages, 'call', call)
    result = client.get('/api/skills/package/pending').json()['data']
    assert result['drafts'][0]['target'] == target
    response = client.post('/api/skills/package/resume',
                           json={'card_id': 'card-a', 'draft_id': 'draft-1'})
    assert response.json()['data']['ui']['html'] == '<p>检查</p>'
    assert seen == ['draft_list', 'draft_open', 'ui']
    assert config.main['skills']['installed'] == []


def test_reimport_reuses_local_receipt_without_reenabling_existing_skill(
        client, target, monkeypatch):
    definition = installed()
    client.post('/api/skills/import', json=definition)
    config.main['skills']['installed'][0]['active'] = False

    async def call(actual, action, **args):
        if action == 'package_begin':
            return {'transfer_id': 't'}
        if action == 'package_write':
            return {}
        return {'reused': True, 'asset_id': ASSET, 'skill': definition}
    monkeypatch.setattr(packages, 'call', call)
    response = client.post('/api/skills/package/install', data={'card_id': 'card-a'},
                           files={'file': ('p.motus-skill', package())})
    assert response.json()['data']['reused']
    assert len(config.main['skills']['installed']) == 1
    assert not config.main['skills']['installed'][0]['active']


def test_the_loopback_read_resolves_an_asset_to_its_owning_card(client, target, monkeypatch):
    """What the device asks on every preview/run, over the loopback exemption."""
    import asyncio
    from starlette.requests import Request
    config.main['services'] = _services(persistent_execution=True)
    owner = packages.targets()[0]
    client.post('/api/skills/import', json=installed())
    config.main[packages._POLICY_KEY] = {
        packages.execution_key(owner) + ':' + ASSET: {
            'enabled': True, 'device_binding': BINDING, 'asset_binding': 'rec-1'}}
    local = Request({'type': 'http', 'client': ('127.0.0.1', 1)})

    def read(**body):
        return asyncio.run(packages.read_execution(local, body))['data']

    assert read(tool=TOOL, device_binding=BINDING, asset_id=ASSET,
                asset_binding='rec-1')['source'] == 'skill'
    # A different recording of the same skill is a different grant.
    assert not read(tool=TOOL, device_binding=BINDING, asset_id=ASSET,
                    asset_binding='rec-2')['enabled']
    # A device that cannot be matched gets `manual`, never an accidental yes.
    assert read(tool='other_tool', device_binding=BINDING)['source'] == 'manual'
    assert read(tool=TOOL, device_binding=BINDING, asset_id='unknown-asset',
                asset_binding='rec-1')['source'] == 'manual'


def test_a_legacy_slug_resolves_over_the_loopback_read_too(client, target, monkeypatch):
    import asyncio
    from starlette.requests import Request
    config.main['services'] = _services(persistent_execution=True)
    owner = packages.targets()[0]
    # Installed before packages carried `attachment`: no attachment at all.
    client.post('/api/skills/import', json={**DEFINITION, 'slug': SLUG,
                                            'requiredTools': [TOOL]})
    config.main[packages._POLICY_KEY] = {
        packages.execution_key(owner) + ':' + ASSET: {
            'enabled': True, 'device_binding': BINDING, 'asset_binding': 'rec-1'}}
    local = Request({'type': 'http', 'client': ('127.0.0.1', 1)})
    data = asyncio.run(packages.read_execution(local, {
        'tool': TOOL, 'device_binding': BINDING, 'asset_id': ASSET,
        'asset_binding': 'rec-1'}))['data']
    assert data['source'] == 'skill'


@pytest.mark.parametrize('device', [None, '', 'x' * 200, 5])
def test_the_loopback_read_still_validates_its_input(client, target, device):
    import asyncio
    from starlette.requests import Request
    config.main['services'] = _services(persistent_execution=True)
    local = Request({'type': 'http', 'client': ('127.0.0.1', 1)})
    with pytest.raises(fastapi.HTTPException) as caught:
        asyncio.run(packages.read_execution(local, {'tool': TOOL, 'device_binding': device}))
    assert caught.value.status_code == 422
