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
PREFIX = 'recorded-'          # just a naming habit in these fixtures, not a convention Core knows
ASSET = 'asset-7'                    # not 32 hex, on purpose
BINDING = 'device-binding-v3'         # not 64 chars, on purpose
SLUG = PREFIX + ASSET
TOOL = 'example_recorder'

CAPABILITIES = {'schema': SCHEMA, 'pending': True}


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
    body = source.replace(source[source.index('"""'):source.index('"""', 3) + 3], '')
    for word in ('teaching', 'taught', 'trajectory.jsonl', '示教', 'dual_arm', 'fixed_hand'):
        assert word not in body, word


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


@pytest.mark.parametrize('name', [
    '../escape.bin', 'a/../../escape.bin', '/abs.bin',
    # Both separators, because the extractor may be on either platform. A
    # `'..' in name.split('/')` check lets this one straight through -- found by
    # feeding the resource centre's TypeScript validator a zip built with
    # Python's zipfile, which does not normalise entry names on write.
    '..\\escape.bin', 'C:\\x.bin', '\\\\unc\\x',
])
def test_an_entry_that_escapes_its_directory_is_refused(name):
    with pytest.raises(fastapi.HTTPException) as caught:
        packages.manifest_from(package(entries=((name, b'x'),)))
    assert caught.value.status_code == 422


@pytest.mark.parametrize('entries', [
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


def test_execution_policy_is_per_skill_per_card_per_recording(client, target):
    """There is deliberately no card-wide form.

    One existed: a single checkbox authorised every skill the card would
    *later* verify. It was the widest grant in the feature, and buying it cost
    an interception of the generic `PUT /api/canvas/tool-config` endpoint,
    device I/O inside a config save, and a private field stored inside the
    user's own config blob. Per-skill consent covers the real need.
    """
    from api.skill_packages import execution_key, execution_setting
    recording = 'rec-1'
    config.main[packages._POLICY_KEY] = {execution_key(target) + ':' + ASSET: {
        'enabled': True, 'device_binding': BINDING, 'asset_binding': recording}}
    assert execution_setting(target, BINDING, ASSET, recording)['source'] == 'skill'
    # A re-recording, another device, or another skill is a separate decision.
    assert not execution_setting(target, BINDING, ASSET, 'changed')['enabled']
    assert not execution_setting(target, 'other-device', ASSET, recording)['enabled']
    assert not execution_setting(target, BINDING, 'another-asset', recording)['enabled']
    assert execution_setting(target, BINDING, ASSET, 'changed')['binding_changed']
    # Nothing in the card's own tool config can turn it on.
    config.main['tool_config:' + execution_key(target)] = {'auto_execute_verified': True}
    assert not execution_setting(target, BINDING, 'another-asset', 'new')['enabled']


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


def test_refresh_updates_the_instruction_preserving_visibility(client, target, monkeypatch):
    client.post('/api/skills/import', json=installed())
    config.main['skills']['installed'][0]['active'] = False
    updated = installed(instruction='Complete preview and run arguments',
                        compatibility={'labels': ['Model-9']})

    async def call(actual, action, **args):
        assert actual == target and action == 'package_definition' and args == {'asset_id': ASSET}
        return {'skill': updated}
    monkeypatch.setattr(packages, 'call', call)
    assert client.post('/api/skills/package/refresh/' + SLUG).status_code == 200
    rows = config.main['skills']['installed']
    assert len(rows) == 1 and not rows[0]['active']
    assert rows[0]['instruction'] == updated['instruction']
    assert rows[0]['origin'] == {'mcp_id': 'robot-a', 'instance_id': 'card-a'}


def test_a_skill_without_an_attachment_cannot_be_resolved(client, target):
    """Identity is `attachment`, full stop.

    A slug-prefix fallback for skills predating that field lived in
    `resolve_skill`; exactly one test rig had such a row. A naming convention
    is not an identity mechanism -- fix the row, not the resolver.
    """
    client.post('/api/skills/import', json={**DEFINITION, 'slug': SLUG,
                                            'requiredTools': [TOOL]})
    with pytest.raises(fastapi.HTTPException) as caught:
        packages.resolve_skill(SLUG)
    assert caught.value.status_code == 409


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


@pytest.mark.parametrize('device', [None, '', 'x' * 200, 5])
def test_the_loopback_read_still_validates_its_input(client, target, device):
    import asyncio
    from starlette.requests import Request
    config.main['services'] = _services(persistent_execution=True)
    local = Request({'type': 'http', 'client': ('127.0.0.1', 1)})
    with pytest.raises(fastapi.HTTPException) as caught:
        asyncio.run(packages.read_execution(local, {'tool': TOOL, 'device_binding': device}))
    assert caught.value.status_code == 422
