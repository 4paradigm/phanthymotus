"""Whether a skill may run unattended, and which device it belongs to.

Every identifier in this file is deliberately **not** one any shipped driver
uses: the protocol is `example.device.package/1`, the legacy slug prefix is
`recorded-`, the asset id is not 32 hex and the binding is not 64 chars. If any
of these have to match something spelled inside agent-core, that is the
coupling these tests exist to catch.

Run: cd agent-core && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
         python3 -m pytest tests/test_skill_packages.py -q
"""
import io

import io

import fastapi
import pytest
from fastapi.testclient import TestClient

from test_skill_import import client, DEFINITION          # noqa: F401 - fixture
import config
from api import skill_execution as packages

SCHEMA = 'example.device.package/1'
PREFIX = 'recorded-'          # just a naming habit in these fixtures, not a convention Core knows
ASSET = 'asset-7'                    # not 32 hex, on purpose
BINDING = 'device-binding-v3'         # not 64 chars, on purpose
SLUG = PREFIX + ASSET
TOOL = 'example_recorder'

CAPABILITIES = {'schema': SCHEMA}


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


def installed(**extra):
    return {**DEFINITION, 'slug': SLUG, 'requiredTools': [TOOL],
            'attachment': {'schema': SCHEMA, 'tool': TOOL, 'asset_id': ASSET},
            'origin': {'mcp_id': 'robot-a', 'instance_id': 'card-a'}, **extra}


# ── Core holds no application vocabulary ─────────────────────────────────────

def test_core_holds_no_protocol_identifier_and_no_second_transport(target):
    """The card's declared schema is the only identifier, and there is one
    transport.

    This module used to carry a chunked `package_begin`/`write`/`finish` copy of
    file transfer, plus the zip validation that goes with forwarding a whole
    archive, plus a bespoke import dialog. All of it existed because no driver
    implemented `POST /file/upload`. That endpoint is wired now
    (phanthymotus-driver#417), so the generic `format: file` path applies -- the
    canvas already renders a picker for such a parameter, and
    `mcp_client._transfer_file_args` pushes the file to whichever service the
    called tool belongs to. Two transports for one job is the thing not to ship.
    """
    assert target['schema'] == SCHEMA
    assert not hasattr(packages, 'SCHEMA')
    source = io.open(packages.__file__, encoding='utf-8').read()
    body = source.replace(source[source.index('"""'):source.index('"""', 3) + 3], '')
    for word in ('teaching', 'taught', 'trajectory.jsonl', '示教', 'dual_arm', 'fixed_hand'):
        assert word not in body, word
    for gone in ('package_begin', 'package_write', 'package_finish', 'package_read',
                 'zipfile', 'b64decode', 'manifest_from', 'UploadFile'):
        assert gone not in body, f'{gone} 应随分块传输一起移除'


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
    from api.skill_execution import execution_key, execution_setting
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
