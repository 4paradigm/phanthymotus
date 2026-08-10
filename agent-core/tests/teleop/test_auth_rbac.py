from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.websockets import WebSocketDisconnect

import auth


@pytest.mark.parametrize('role', ['owner', 'operator', 'viewer'])
@pytest.mark.parametrize('path', ['/api/teleop/me', '/api/teleop/robots'])
def test_all_human_roles_can_read_teleop_directory(client, auth_headers, role, path):
    response = client.get(path, headers=auth_headers[role])
    assert response.status_code == 200


@pytest.mark.parametrize('role', ['operator', 'viewer'])
@pytest.mark.parametrize(
    ('method', 'path', 'json_body'),
    [
        ('get', '/api/mcp', None),
        ('get', '/api/canvas/layout', None),
        (
            'post',
            '/api/mcp/agentcore/call',
            {'tool': 'remote_message', 'arguments': {'action': 'stop'}},
        ),
        (
            'post',
            '/api/mcp',
            {
                'id': 'human-must-not-register-driver',
                'name': 'Human Registration',
                'url': 'http://human.invalid/mcp',
            },
        ),
    ],
)
def test_non_owners_cannot_reach_existing_api_authority(
    client, auth_headers, role, method, path, json_body,
):
    response = client.request(method, path, headers=auth_headers[role], json=json_body)
    assert response.status_code == 403
    assert response.json()['detail'] == 'Owner role required'


def test_owner_keeps_existing_api_and_canvas_compatibility(client, auth_headers):
    assert client.get('/api/mcp', headers=auth_headers['owner']).status_code == 200
    assert client.get('/api/canvas/layout', headers=auth_headers['owner']).status_code == 200

    call = client.post(
        '/api/mcp/agentcore/call',
        headers=auth_headers['owner'],
        json={'tool': 'remote_message', 'arguments': {'action': 'stop'}},
    )
    assert call.status_code == 200
    assert call.json()['data']['state'] == 'idle'


def test_missing_or_invalid_human_token_is_rejected(client):
    assert client.get('/api/teleop/robots').status_code == 401
    assert client.get(
        '/api/teleop/robots',
        headers={'Authorization': 'Bearer not-a-token'},
    ).status_code == 401
    assert client.get('/api/mcp').status_code == 401


@pytest.mark.parametrize(
    ('role', 'token', 'principal_id'),
    [
        ('owner', 'owner-token', 'owner:legacy'),
        ('operator', 'operator-token', 'operator:alice'),
        ('viewer', 'viewer-token', 'viewer:auditor'),
    ],
)
def test_auth_verify_returns_role_namespaced_principal(
    client, role, token, principal_id,
):
    response = client.get(
        '/api/auth/verify', headers={'Authorization': f'Bearer {token}'},
    )
    assert response.status_code == 200
    assert response.json()['principal'] == {'id': principal_id, 'role': role}


def test_invalid_auth_verify_is_401(client):
    response = client.get('/api/auth/verify')
    assert response.status_code == 401
    assert response.json() == {
        'valid': False, 'auth_required': True, 'principal': None,
    }


@pytest.mark.parametrize(
    'service_settings',
    [
        {'MOTUS_DRIVER_TOKEN': 'legacy-driver-service-token'},
        {
            'MOTUS_DRIVER_TOKENS': json.dumps({
                'driver-a': 'dedicated-driver-service-token-a',
            }),
        },
        {'MOTUS_TELEOP_TICKET_SECRET': 't' * 32},
        {
            'MOTUS_TELEOP_TICKET_SECRETS': json.dumps({
                'driver-a': 's' * 32,
            }),
        },
    ],
)
def test_service_credentials_without_human_principal_lock_management_api(
    client, service_settings,
):
    auth.init(service_settings)

    assert auth.is_enabled() is True
    assert client.get('/api/auth/verify').status_code == 401
    assert client.get('/api/mcp').status_code == 401
    assert client.post('/api/config', json={'mcp_list': []}).status_code == 401
    assert client.post('/api/mcp/driver-a/ping').status_code == 401


def test_duplicate_human_token_fails_closed():
    with pytest.raises(ValueError, match='assigned to both'):
        auth.init({
            'MOTUS_OPERATOR_TOKENS': '{"alice":"same-token"}',
            'MOTUS_VIEWER_TOKENS': '{"bob":"same-token"}',
        })


def test_duplicate_role_principal_id_fails_closed():
    with pytest.raises(ValueError, match='principal id is configured more than once'):
        auth.init({'MOTUS_OPERATOR_TOKENS': 'alice:first,alice:second'})


def test_exact_duplicate_token_assignment_also_fails_closed():
    with pytest.raises(ValueError, match='authentication token is configured more than once'):
        auth.init({'MOTUS_OPERATOR_TOKENS': 'alice:same,alice:same'})


def test_duplicate_json_principal_name_fails_closed():
    with pytest.raises(ValueError, match='principal id is configured more than once'):
        auth.init({'MOTUS_VIEWER_TOKENS': '{"alice":"first","alice":"second"}'})


def test_same_name_in_distinct_role_namespaces_is_valid():
    auth.init({
        'MOTUS_OPERATOR_TOKENS': '{"sam":"operator-secret"}',
        'MOTUS_VIEWER_TOKENS': '{"sam":"viewer-secret"}',
    })
    assert auth.authenticate('operator-secret').id == 'operator:sam'
    assert auth.authenticate('viewer-secret').id == 'viewer:sam'


def test_driver_token_cannot_reuse_human_token():
    with pytest.raises(ValueError, match='must not reuse a human access token'):
        auth.init({'ACCESS_TOKEN': 'collision', 'MOTUS_DRIVER_TOKEN': 'collision'})


def test_rejected_auth_reload_atomically_preserves_last_valid_snapshot():
    auth.init({
        'ACCESS_TOKEN': 'old-owner-token',
        'MOTUS_DRIVER_TOKEN': 'old-driver-token',
        'MOTUS_TELEOP_TICKET_SECRET': 'o' * 32,
    })

    with pytest.raises(ValueError, match='must not reuse a human access token'):
        auth.init({
            'ACCESS_TOKEN': 'rejected-collision-token',
            'MOTUS_DRIVER_TOKEN': 'rejected-collision-token',
            'MOTUS_TELEOP_TICKET_SECRET': 'n' * 32,
        })

    assert auth.authenticate('old-owner-token').id == 'owner:legacy'
    assert auth.verify_driver_token('old-driver-token', 'legacy-driver') is True
    assert auth.teleop_ticket_secret('legacy-driver') == b'o' * 32
    assert auth.authenticate('rejected-collision-token') is None
    assert auth.verify_driver_token(
        'rejected-collision-token', 'legacy-driver',
    ) is False


def test_driver_enforcement_requires_driver_token():
    with pytest.raises(ValueError, match='requires MOTUS_DRIVER_TOKEN'):
        auth.init({'MOTUS_ENFORCE_DRIVER_AUTH': 'true'})


def test_dedicated_driver_and_ticket_credentials_select_exact_identity():
    driver_a = 'driver-a-token-that-is-private'
    driver_b = 'driver-b-token-that-is-private'
    ticket_a = 'a' * 32
    ticket_b = 'b' * 32
    auth.init({
        'MOTUS_DRIVER_TOKEN': 'legacy-driver-fallback',
        'MOTUS_DRIVER_TOKENS': json.dumps({
            'driver-a': driver_a,
            'driver-b': driver_b,
        }),
        'MOTUS_TELEOP_TICKET_SECRET': 'l' * 32,
        'MOTUS_TELEOP_TICKET_SECRETS': json.dumps({
            'driver-a': ticket_a,
            'driver-b': ticket_b,
        }),
    })

    assert auth.verify_driver_token(driver_a, 'driver-a') is True
    assert auth.verify_driver_token(driver_a, 'driver-b') is False
    assert auth.driver_token_identity(driver_a) == 'driver-a'
    assert auth.driver_request_headers('driver-a') == {
        'Authorization': f'Bearer {driver_a}',
    }
    assert auth.driver_request_headers('driver-b') == {
        'Authorization': f'Bearer {driver_b}',
    }
    assert auth.driver_request_headers('legacy-driver') == {
        'Authorization': 'Bearer legacy-driver-fallback',
    }
    assert auth.teleop_ticket_secret('driver-a') == ticket_a.encode()
    assert auth.teleop_ticket_secret('driver-b') == ticket_b.encode()
    assert auth.teleop_ticket_secret('legacy-driver') == b'l' * 32
    assert auth.teleop_ticket_credential_available('driver-a') is True
    assert auth.teleop_ticket_credential_available('legacy-driver') is True


def test_dedicated_driver_credentials_enable_enforcement_without_global_token():
    auth.init({
        'MOTUS_DRIVER_TOKENS': '{"driver-a":"private-driver-a-token-0001"}',
        'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
    })

    assert auth.is_driver_auth_enforced() is True
    assert auth.driver_runtime_credential_available('driver-a') is True
    assert auth.driver_runtime_credential_available('driver-b') is False


def test_dedicated_driver_bearer_requires_24_url_safe_ascii_characters():
    with pytest.raises(ValueError, match='at least 24 bytes'):
        auth.init({'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': 'a' * 23})})

    auth.init({'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': 'a' * 24})})
    assert auth.verify_driver_token('a' * 24, 'driver-a') is True


def test_removing_every_driver_credential_quarantines_persisted_trust():
    auth.init({})

    assert auth.has_any_driver_credentials() is False
    assert auth.driver_runtime_credential_available('persisted-driver') is False


def test_driver_credential_selectors_reject_missing_or_invalid_identity():
    auth.init({
        'MOTUS_DRIVER_TOKEN': 'legacy-driver-fallback',
        'MOTUS_TELEOP_TICKET_SECRET': 't' * 32,
    })

    for invalid_id in ('', ' bad-id', 'driver/id', None):
        assert auth.is_driver_auth_configured(invalid_id) is False
        assert auth.driver_runtime_credential_available(invalid_id) is False
        assert auth.driver_request_headers(invalid_id) == {}
        assert auth.teleop_ticket_secret(invalid_id) == b''
        assert auth.teleop_ticket_credential_available(invalid_id) is False
        assert auth.verify_driver_token('legacy-driver-fallback', invalid_id) is False


def test_ticket_readiness_is_exact_per_driver_and_does_not_use_another_map_entry():
    auth.init({
        'MOTUS_TELEOP_TICKET_SECRETS': json.dumps({'driver-a': 'a' * 32}),
    })

    assert auth.teleop_ticket_credential_available('driver-a') is True
    assert auth.teleop_ticket_credential_available('driver-b') is False


def test_dedicated_credential_binding_requires_fresh_registration_after_rotation():
    old_token = 'private-driver-a-token-old-0001'
    new_token = 'private-driver-a-token-new-0002'
    auth.init({
        'MOTUS_DRIVER_TOKEN': 'legacy-driver-fallback',
        'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': old_token}),
    })
    old_binding = auth.driver_credential_binding('driver-a')

    assert old_binding.startswith('sha256:')
    assert old_token not in old_binding
    assert auth.driver_record_credential_available('driver-a', old_binding) is True
    assert auth.driver_record_credential_available('driver-a', '') is False

    auth.init({
        'MOTUS_DRIVER_TOKEN': 'legacy-driver-fallback',
        'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': new_token}),
    })
    new_binding = auth.driver_credential_binding('driver-a')
    assert new_binding != old_binding
    assert auth.driver_record_credential_available('driver-a', old_binding) is False
    assert auth.driver_record_credential_available('driver-a', new_binding) is True

    auth.init({'MOTUS_DRIVER_TOKEN': 'legacy-driver-fallback'})
    assert auth.driver_record_credential_available('driver-a', old_binding) is False
    assert auth.driver_record_credential_available('driver-a', None) is True


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        (
            {
                'MOTUS_DRIVER_TOKENS': (
                    '{"driver-a":"same-secret-value-1234567890",'
                    '"driver-b":"same-secret-value-1234567890"}'
                ),
            },
            'more than one id',
        ),
        (
            {
                'ACCESS_TOKEN': 'same-secret-value-1234567890',
                'MOTUS_DRIVER_TOKENS': (
                    '{"driver-a":"same-secret-value-1234567890"}'
                ),
            },
            'human access token',
        ),
        (
            {
                'MOTUS_DRIVER_TOKEN': 'same-secret-value-1234567890',
                'MOTUS_DRIVER_TOKENS': (
                    '{"driver-a":"same-secret-value-1234567890"}'
                ),
            },
            'reuse dedicated credential',
        ),
        (
            {'MOTUS_DRIVER_TOKENS': '{"bad id":"private"}'},
            'invalid driver credential id',
        ),
        (
            {
                'MOTUS_TELEOP_TICKET_SECRETS': (
                    '{"driver-a":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                    '"driver-b":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
                ),
            },
            'more than one id',
        ),
        (
            {
                'MOTUS_TELEOP_TICKET_SECRET': 'a' * 32,
                'MOTUS_TELEOP_TICKET_SECRETS': (
                    '{"driver-a":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
                ),
            },
            'reuse dedicated secret',
        ),
        (
            {
                'MOTUS_DRIVER_TOKENS': (
                    '{"driver-a":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
                ),
                'MOTUS_TELEOP_TICKET_SECRETS': (
                    '{"driver-b":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
                ),
            },
            'reuse dedicated Driver credential',
        ),
        (
            {
                'ACCESS_TOKEN': 'h' * 32,
                'MOTUS_TELEOP_TICKET_SECRET': 'h' * 32,
            },
            'reuse a human access token',
        ),
        (
            {
                'MOTUS_DRIVER_TOKEN': 'l' * 32,
                'MOTUS_TELEOP_TICKET_SECRET': 'l' * 32,
            },
            'reuse legacy Driver credential',
        ),
        (
            {
                'MOTUS_DRIVER_TOKEN': '遥' * 11,
                'MOTUS_TELEOP_TICKET_SECRET': '遥' * 11,
            },
            'reuse legacy Driver credential',
        ),
    ],
)
def test_driver_credential_maps_reject_ambiguous_or_reused_secrets(
    settings,
    message,
):
    with pytest.raises(ValueError, match=message):
        auth.init(settings)


def test_teleop_ticket_secret_requires_at_least_32_utf8_bytes():
    with pytest.raises(ValueError, match='at least 32 bytes'):
        auth.init({'MOTUS_TELEOP_TICKET_SECRET': 'x' * 31})

    auth.init({'MOTUS_TELEOP_TICKET_SECRET': '遥' * 11})
    assert auth.teleop_ticket_secret('legacy-driver') == ('遥' * 11).encode('utf-8')


def test_dedicated_bearer_rejects_unicode_but_ticket_secret_supports_it():
    with pytest.raises(ValueError, match='restricted ASCII Bearer'):
        auth.init({
            'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': '密' * 24}),
        })

    unicode_ticket = '遥' * 11
    auth.init({
        'MOTUS_TELEOP_TICKET_SECRETS': json.dumps({
            'driver-a': unicode_ticket,
        }),
    })
    assert auth.teleop_ticket_secret('driver-a') == unicode_ticket.encode('utf-8')

    with pytest.raises(ValueError, match='reuse a human access token'):
        auth.init({
            'ACCESS_TOKEN': unicode_ticket,
            'MOTUS_TELEOP_TICKET_SECRETS': json.dumps({
                'driver-a': unicode_ticket,
            }),
        })


def test_teleop_ticket_secret_loads_from_deployment_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    deployment_env = tmp_path / 'deployment.env'
    deployment_env.write_text(
        f'MOTUS_TELEOP_TICKET_SECRET={"s" * 32}\n',
        encoding='utf-8',
    )
    monkeypatch.delenv('MOTUS_TELEOP_TICKET_SECRET', raising=False)
    monkeypatch.setattr(auth, '_ENV_PATH', deployment_env)
    monkeypatch.setattr(auth, '_ENV_PATH_DEV', tmp_path / 'missing.env')

    auth.init()

    assert auth.teleop_ticket_secret('legacy-driver') == b's' * 32


def test_per_driver_credentials_load_from_deployment_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    deployment_env = tmp_path / 'deployment.env'
    deployment_env.write_text(
        'MOTUS_DRIVER_TOKENS={"driver-a":"private-driver-a-token-0001"}\n'
        f'MOTUS_TELEOP_TICKET_SECRETS={{"driver-a":"{"t" * 32}"}}\n'
        'MOTUS_ENFORCE_DRIVER_AUTH=true\n',
        encoding='utf-8',
    )
    for key in (
        'MOTUS_DRIVER_TOKEN',
        'MOTUS_DRIVER_TOKENS',
        'MOTUS_TELEOP_TICKET_SECRET',
        'MOTUS_TELEOP_TICKET_SECRETS',
        'MOTUS_ENFORCE_DRIVER_AUTH',
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(auth, '_ENV_PATH', deployment_env)
    monkeypatch.setattr(auth, '_ENV_PATH_DEV', tmp_path / 'missing.env')

    auth.init()

    assert auth.driver_request_headers('driver-a') == {
        'Authorization': 'Bearer private-driver-a-token-0001',
    }
    assert auth.teleop_ticket_secret('driver-a') == b't' * 32
    assert auth.is_driver_auth_enforced() is True


@pytest.mark.parametrize('path', ['/ws/motus', '/ws/bus/unregistered'])
def test_non_owner_websocket_is_closed(client, path):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f'{path}?token=viewer-token'):
            pass
    assert exc.value.code in (4001, 4003)


def test_owner_websocket_remains_usable(client):
    with client.websocket_connect('/ws/motus?token=owner-token') as websocket:
        message = websocket.receive_json()
    assert message['type'] == 'status'
    assert message['payload']['connected'] is True


def test_owner_can_open_bus_websocket(client):
    with client.websocket_connect('/ws/bus/unregistered?token=owner-token') as websocket:
        message = websocket.receive_json()
    assert message['type'] == 'error'
    assert 'not registered' in message['message']
