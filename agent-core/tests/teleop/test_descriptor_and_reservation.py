from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import auth
import config
import mcp_client
import pytest
from api import mcp_manage, teleop
from teleop.service import TeleopServiceError, _driver_descriptor


def _trusted_robot(tool: dict, **overrides) -> dict:
    tool = deepcopy(tool)
    record = {
        'id': 'teleop-robot',
        'name': 'Teleop Robot',
        'server_name': 'teleop-shadow',
        'url': 'http://127.0.0.1:15711/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [tool],
    }
    record.update(overrides)
    if isinstance(tool.get('x-teleop'), dict):
        tool['x-teleop']['driver_id'] = record['id']
        tool['x-teleop']['robot_id'] = record['id']
    return record


def _invalidate(tool: dict, case: str) -> None:
    descriptor = tool.get('x-teleop')
    if case == 'missing-descriptor':
        tool.pop('x-teleop')
    elif case == 'null-descriptor':
        tool['x-teleop'] = None
    elif case == 'missing-protocol':
        descriptor.pop('protocol')
    elif case == 'wrong-protocol':
        descriptor['protocol'] = 'motus.teleop.live.v1'
    elif case == 'missing-dispatch-contract':
        descriptor.pop('dispatch_contract')
    elif case == 'wrong-dispatch-contract':
        descriptor['dispatch_contract'] = 'motus.teleop.dispatch.live.v1'
    elif case == 'wrong-dry-run-profile':
        descriptor['dry_run_profile'] = 'go1_live'
    elif case == 'non-string-dry-run-profile':
        descriptor['dry_run_profile'] = ['unitree_go1_high_level']
    elif case == 'missing-signaling':
        descriptor.pop('signaling')
    elif case == 'wrong-signaling-protocol':
        descriptor['signaling']['protocol'] = 'motus.teleop.websocket.v1'
    elif case == 'wrong-signaling-path':
        descriptor['signaling']['path'] = '/unsafe-offer'
    elif case == 'wrong-signaling-access':
        descriptor['signaling']['access'] = 'browser-direct'
    elif case == 'wrong-mode':
        descriptor['mode'] = 'live'
    elif case == 'missing-actuation-flag':
        descriptor.pop('actuation_enabled')
    elif case == 'actuation-enabled':
        descriptor['actuation_enabled'] = True
    elif case == 'falsey-integer-is-not-false':
        descriptor['actuation_enabled'] = 0
    elif case == 'missing-digest':
        descriptor.pop('capability_digest')
    elif case == 'short-digest':
        descriptor['capability_digest'] = 'a' * 63
    elif case == 'uppercase-digest':
        descriptor['capability_digest'] = 'A' * 64
    elif case == 'old-prepare-action':
        actions = tool['inputSchema']['properties']['action']['enum']
        actions[actions.index('prepare_shadow')] = 'prepare'
    elif case == 'input-schema-string':
        tool['inputSchema'] = 'not-an-object'
    elif case == 'properties-list':
        tool['inputSchema']['properties'] = []
    elif case == 'action-schema-string':
        tool['inputSchema']['properties']['action'] = 'not-an-object'
    elif case == 'wrong-tool-type':
        tool['type'] = 'resource'
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError(f'unknown invalid descriptor case: {case}')


def test_driver_producer_fixture_satisfies_exact_shadow_contract(shadow_session_tool):
    assert teleop._valid_shadow_descriptor(shadow_session_tool) is True


@pytest.mark.parametrize(
    'case',
    [
        'missing-descriptor',
        'null-descriptor',
        'missing-protocol',
        'wrong-protocol',
        'missing-dispatch-contract',
        'wrong-dispatch-contract',
        'wrong-dry-run-profile',
        'non-string-dry-run-profile',
        'missing-signaling',
        'wrong-signaling-protocol',
        'wrong-signaling-path',
        'wrong-signaling-access',
        'wrong-mode',
        'missing-actuation-flag',
        'actuation-enabled',
        'falsey-integer-is-not-false',
        'missing-digest',
        'short-digest',
        'uppercase-digest',
        'old-prepare-action',
        'input-schema-string',
        'properties-list',
        'action-schema-string',
        'wrong-tool-type',
    ],
)
def test_shadow_descriptor_rejects_every_malformed_contract_field(
    shadow_session_tool, case,
):
    tool = deepcopy(shadow_session_tool)
    _invalidate(tool, case)
    assert teleop._valid_shadow_descriptor(tool) is False


@pytest.mark.parametrize(
    ('trust_state', 'online', 'ready', 'reason'),
    [
        ('trusted', True, True, 'ready'),
        ('trusted', False, False, 'driver_offline'),
        ('untrusted', True, False, 'driver_registration_not_trusted'),
        ('quarantined', True, False, 'driver_registration_not_trusted'),
    ],
)
def test_teleop_ready_requires_valid_descriptor_trust_and_runtime_online(
    shadow_session_tool, trust_state, online, ready, reason,
):
    record = _trusted_robot(deepcopy(shadow_session_tool), trust_state=trust_state)
    mcp_client.registry[record['id']] = {
        'online': online,
        'trusted': True,
        'url': record['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
    }

    view = teleop._robot_view(record)

    assert view['descriptor_valid'] is True
    assert view['teleop_ready'] is ready
    assert view['reason'] == reason


def test_invalid_descriptor_is_visible_but_never_ready(shadow_session_tool):
    tool = deepcopy(shadow_session_tool)
    tool['x-teleop']['mode'] = 'active'
    record = _trusted_robot(tool)
    mcp_client.registry[record['id']] = {
        'online': True,
        'trusted': True,
        'url': record['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
    }

    view = teleop._robot_view(record)

    assert view['teleop_declared'] is True
    assert view['descriptor_valid'] is False
    assert view['teleop_ready'] is False
    assert view['reason'] == 'teleop_descriptor_invalid'


def test_missing_exact_ticket_secret_is_visible_and_blocks_acquire_preflight(
    shadow_session_tool,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    record = _trusted_robot(deepcopy(shadow_session_tool))
    config.main['services'] = {'mcp': [record]}
    mcp_client.registry[record['id']] = {
        'online': True,
        'trusted': True,
        'url': record['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
    }

    view = teleop._robot_view(record)

    assert view['teleop_ready'] is False
    assert view['reason'] == 'teleop_signaling_unavailable'
    with pytest.raises(TeleopServiceError) as raised:
        _driver_descriptor(record['id'])
    assert raised.value.code == 'teleop_signaling_unavailable'


def test_teleop_directory_sanitizes_untrusted_metadata_and_internal_url(
    client, auth_headers, shadow_session_tool,
):
    tool = deepcopy(shadow_session_tool)
    tool['x-teleop'].update({
        'access_token': 'do-not-return-token',
        'nested': {
            'fence': 'do-not-return-fence',
            'private_key': 'do-not-return-key',
            'privateKey': 'do-not-return-camel-key',
            'apiKey': 'do-not-return-api-key',
            'api-key-id': 'do-not-return-hyphen-key',
            'public_label': 'safe',
        },
    })
    record = _trusted_robot(tool)
    config.main['services'] = {'mcp': [record]}
    mcp_client.registry[record['id']] = {
        'online': True,
        'trusted': True,
        'url': record['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
    }

    response = client.get('/api/teleop/robots', headers=auth_headers['viewer'])

    assert response.status_code == 200
    robot = response.json()['data'][0]
    assert robot['descriptor_valid'] is False
    assert robot['teleop_ready'] is False
    assert robot['reason'] == 'teleop_descriptor_invalid'
    assert 'url' not in robot
    assert robot['teleop']['nested'] == {'public_label': 'safe'}
    assert 'access_token' not in robot['teleop']
    assert 'do-not-return' not in response.text


def test_non_finite_metadata_cannot_take_down_other_robot_directory_entries(
    client, auth_headers, shadow_session_tool,
):
    malformed = deepcopy(shadow_session_tool)
    malformed['x-teleop']['diagnostics'] = {
        'nan': float('nan'),
        'positive_infinity': float('inf'),
        'negative_infinity': float('-inf'),
    }
    malformed['annotations']['non_finite'] = [
        float('nan'), float('inf'), float('-inf'),
    ]
    bad_record = _trusted_robot(
        malformed,
        id='robot-with-non-finite-metadata',
        name='A Robot With Bad Metadata',
    )
    good_record = _trusted_robot(
        deepcopy(shadow_session_tool),
        id='unaffected-robot',
        name='B Unaffected Robot',
    )
    config.main['services'] = {'mcp': [bad_record, good_record]}
    for record in (bad_record, good_record):
        mcp_client.registry[record['id']] = {
            'online': True,
            'trusted': True,
            'url': record['url'],
            'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
        }

    response = client.get('/api/teleop/robots', headers=auth_headers['viewer'])

    assert response.status_code == 200
    robots = {robot['id']: robot for robot in response.json()['data']}
    assert set(robots) == {bad_record['id'], good_record['id']}
    assert robots[bad_record['id']]['teleop']['diagnostics'] == {
        'nan': None,
        'positive_infinity': None,
        'negative_infinity': None,
    }
    assert robots[bad_record['id']]['descriptor_valid'] is False
    assert robots[bad_record['id']]['reason'] == 'teleop_descriptor_invalid'
    assert robots[bad_record['id']]['annotations']['non_finite'] == [None, None, None]
    assert robots[good_record['id']]['teleop_ready'] is True


def test_teleop_directory_uses_bound_robot_domain_for_busy_session(
    client,
    auth_headers,
    shadow_session_tool,
    monkeypatch,
):
    adapter_id = 'teleop-shadow-lab-a'
    robot_id = 'g1-lab-a'
    adapter = _trusted_robot(
        deepcopy(shadow_session_tool),
        id=adapter_id,
        name='Standalone Teleop Adapter',
        reported_robot_id=robot_id,
        authority_domain=robot_id,
        authority_binding_required=True,
    )
    adapter['tools'][0]['x-teleop']['robot_id'] = robot_id
    root = {
        'id': robot_id,
        'name': 'G1 Lab A',
        'url': 'http://g1.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{'name': 'locomotion', 'type': 'actuator'}],
    }
    config.main['services'] = {'mcp': [adapter, root]}
    mcp_client.registry[adapter_id] = {
        'online': True,
        'trusted': True,
        'url': adapter['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(
            adapter['tools'][0],
        ),
    }
    looked_up = []
    session = SimpleNamespace(
        id='bound-session',
        principal_id='operator:alice',
        client_id='other-client',
        state='active',
    )

    async def active_for_robot(requested_robot_id):
        looked_up.append(requested_robot_id)
        return session if requested_robot_id == robot_id else None

    monkeypatch.setattr(
        teleop.coordinator.manager,
        'active_for_robot',
        active_for_robot,
    )
    monkeypatch.setattr(
        teleop.coordinator.manager,
        'public_dict',
        lambda current: {
            'id': current.id,
            'robot_id': robot_id,
            'remaining_seconds': 10.0,
        },
    )

    response = client.get('/api/teleop/robots', headers=auth_headers['viewer'])

    assert response.status_code == 200
    robots = {robot['id']: robot for robot in response.json()['data']}
    view = robots[adapter_id]
    assert view['driver_id'] == adapter_id
    assert view['robot_id'] == robot_id
    assert view['teleop_ready'] is True
    assert view['session']['busy'] is True
    assert robot_id in looked_up


def test_canvas_and_generic_mcp_endpoints_hide_reserved_tools(
    client, auth_headers, shadow_session_tool,
):
    ordinary = {
        'name': 'camera_info',
        'type': 'resource',
        'inputSchema': {'type': 'object', 'properties': {}},
    }
    record = _trusted_robot(
        deepcopy(shadow_session_tool),
        transport='internal',
        url='',
    )
    record['tools'] = [ordinary, deepcopy(shadow_session_tool)]
    config.main['services'] = {'mcp': [record]}

    listing = client.get('/api/mcp', headers=auth_headers['owner'])
    details = client.get(
        f'/api/mcp/{record["id"]}/tools', headers=auth_headers['owner'],
    )
    rejected = client.post(
        f'/api/mcp/{record["id"]}/call',
        headers=auth_headers['owner'],
        json={'tool': 'teleop_session', 'arguments': {'action': 'status'}},
    )

    assert [tool['name'] for tool in listing.json()['data'][0]['tools']] == ['camera_info']
    assert [tool['name'] for tool in details.json()['data']] == ['camera_info']
    assert rejected.status_code == 403
    assert rejected.json()['detail'] == 'Teleop tools are reserved for the dedicated teleop API'


def test_any_x_teleop_presence_is_reserved_even_when_descriptor_is_null(
    client, auth_headers,
):
    suspicious = {
        'name': 'maybe_teleop',
        'type': 'actuator',
        'inputSchema': {'type': 'object', 'properties': {}},
        'x-teleop': None,
    }
    record = _trusted_robot(suspicious, transport='internal', url='')
    config.main['services'] = {'mcp': [record]}

    listing = client.get('/api/mcp', headers=auth_headers['owner'])
    rejected = client.post(
        f'/api/mcp/{record["id"]}/call',
        headers=auth_headers['owner'],
        json={'tool': 'maybe_teleop', 'arguments': {}},
    )

    assert listing.json()['data'][0]['tools'] == []
    assert rejected.status_code == 403


def test_ping_preserves_raw_descriptor_but_excludes_it_from_llm_registry(
    monkeypatch, shadow_session_tool,
):
    ordinary = {
        'name': 'camera_info',
        'description': 'ordinary camera status',
        'type': 'resource',
        'inputSchema': {'type': 'object', 'properties': {}},
        'annotations': {'readOnlyHint': True},
    }
    robot = _trusted_robot(deepcopy(shadow_session_tool))
    config.main['services'] = {'mcp': [robot]}

    async def fake_ping(url, *, trusted=False, driver_id=''):
        assert trusted is True
        return {
            'server_name': 'teleop-shadow',
            'tools': [ordinary, deepcopy(shadow_session_tool)],
            'resources': [],
            'device_type': '',
            'topic_out': [],
            'topic_in': [],
        }

    async def no_op(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_manage, '_ping_mcp_http', fake_ping)
    monkeypatch.setattr(mcp_manage, '_notify_inspector', no_op)
    monkeypatch.setattr(mcp_manage, '_restore_saved_configs', no_op)

    async def run_ping():
        result = await mcp_manage._do_ping(robot['id'])
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run_ping())

    persisted = config.main['services']['mcp'][0]['tools']
    persisted_teleop = next(tool for tool in persisted if tool['name'] == 'teleop_session')
    assert persisted_teleop['x-teleop']['protocol'] == 'motus.teleop.shadow.v1'
    assert persisted_teleop['annotations']['destructiveHint'] is False
    assert result['tools'][1]['name'] == 'teleop_session'

    runtime = mcp_client.registry[robot['id']]
    assert runtime['tools'] == ['camera_info']
    assert set(runtime['schemas']) == {'mcp__teleop-robot__camera_info'}
    assert all('teleop_session' not in schema['name'] for schema in mcp_client.all_schemas())


def test_llm_transport_refuses_reserved_tool_before_network(shadow_session_tool):
    robot = _trusted_robot(deepcopy(shadow_session_tool))
    config.main['services'] = {'mcp': [robot]}
    mcp_client.registry[robot['id']] = {
        'online': True,
        'url': robot['url'],
        'trusted': True,
        'schemas': {},
        'split_map': {},
    }

    result = asyncio.run(mcp_client.call_tool(
        'mcp__teleop-robot__teleop_session', {'action': 'status'},
    ))

    assert result == 'Teleop tools are reserved for the dedicated teleop API'
