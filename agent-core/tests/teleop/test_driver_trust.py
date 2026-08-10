from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import fastapi
import pytest

import auth
import config
import mcp_client
from api import config as config_api
from api import mcp_manage
from teleop import authority_guard


def _persist_authority_guard(driver_id: str, robot_id: str):
    return authority_guard.create_guard(authority_guard.AuthorityGuard(
        robot_id=robot_id,
        driver_id=driver_id,
        session_id='f43d94e5-fdea-44d9-a11e-61723387b3a4',
        boot_id='8ee5e121-78d9-47cb-91f6-cf9134973677',
        epoch=7,
        capability_digest='0' * 64,
        target_fingerprint='a' * 64,
        dispatch_generation=3,
        phase='recovery_required',
        created_at=1_000.0,
        updated_at=1_001.0,
    ))


def _registration(**overrides):
    body = {
        'id': 'robot-stable-id',
        'name': 'Robot Driver',
        'transport': 'http',
        'url': 'http://robot.local/mcp',
        'category': 'driver',
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize(
    ('settings', 'expected_state', 'should_schedule'),
    [
        ({}, 'untrusted', True),
        ({'MOTUS_DRIVER_TOKEN': 'driver-secret'}, 'untrusted', True),
        (
            {
                'MOTUS_DRIVER_TOKEN': 'driver-secret',
                'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
            },
            'quarantined',
            False,
        ),
    ],
)
def test_unauthenticated_driver_three_state_rollout(
    client, monkeypatch, settings, expected_state, should_schedule,
):
    auth.init(settings)
    scheduled = []

    def fake_ping(mcp_id):
        scheduled.append(mcp_id)

        async def done():
            return {}

        return done()

    monkeypatch.setattr(mcp_manage, '_do_ping', fake_ping)
    response = client.post('/api/mcp', json=_registration())

    assert response.status_code == 200
    data = response.json()['data']
    assert data['trust_state'] == expected_state
    assert bool(scheduled) is should_schedule
    record = config.main['services']['mcp'][0]
    assert record['trust_state'] == expected_state
    assert mcp_manage._runtime_registration_allowed(record) is should_schedule
    if expected_state == 'quarantined':
        assert data['id'] not in mcp_client.registry


def test_valid_driver_token_creates_trusted_stable_identity(client, monkeypatch):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    scheduled = []

    def fake_ping(mcp_id):
        scheduled.append(mcp_id)

        async def done():
            return {}

        return done()

    monkeypatch.setattr(mcp_manage, '_do_ping', fake_ping)
    response = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(),
    )

    assert response.status_code == 200
    assert response.json()['data'] == {
        'id': 'robot-stable-id', 'trust_state': 'trusted',
    }
    assert scheduled == ['robot-stable-id']
    assert config.main['services']['mcp'][0]['trust_state'] == 'trusted'


def test_driver_registration_never_accepts_credential_in_query_string(
    client,
    monkeypatch,
):
    token = 'private-driver-a-token-0001'
    auth.init({
        'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': token}),
        'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
    })
    scheduled = []
    monkeypatch.setattr(
        mcp_manage,
        '_do_ping',
        lambda mcp_id: scheduled.append(mcp_id) or asyncio.sleep(0),
    )

    response = client.post(
        f'/api/mcp?token={token}',
        json=_registration(id='driver-a', url='http://driver-a.local/mcp'),
    )

    assert response.status_code == 401
    assert config.main['services']['mcp'] == []
    assert scheduled == []
    assert token not in json.dumps(response.json())


def test_dedicated_driver_token_cannot_register_another_driver_identity(
    client,
    monkeypatch,
):
    token_a = 'private-driver-a-token-0001'
    token_b = 'private-driver-b-token-0002'
    auth.init({
        'MOTUS_DRIVER_TOKENS': json.dumps({
            'driver-a': token_a,
            'driver-b': token_b,
        }),
        'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
    })
    scheduled = []

    def fake_ping(mcp_id):
        scheduled.append(mcp_id)
        return asyncio.sleep(0)

    monkeypatch.setattr(mcp_manage, '_do_ping', fake_ping)
    accepted_a = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': token_a},
        json=_registration(
            id='driver-a',
            url='http://driver-a.local/mcp',
        ),
    )
    before_rejections = deepcopy(config.main['services']['mcp'])

    cross_identity = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': token_a},
        json=_registration(
            id='driver-b',
            url='http://driver-b.local/mcp',
        ),
    )
    wrong_for_dedicated = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'unknown-token'},
        json=_registration(
            id='driver-b',
            url='http://driver-b.local/mcp',
        ),
    )

    assert accepted_a.status_code == 200
    assert accepted_a.json()['data']['trust_state'] == 'trusted'
    assert cross_identity.status_code == 403
    assert wrong_for_dedicated.status_code == 401
    assert config.main['services']['mcp'] == before_rejections
    assert scheduled == ['driver-a']
    public_scan = json.dumps(
        [
            cross_identity.json(),
            wrong_for_dedicated.json(),
            config.main['services'],
        ],
        ensure_ascii=False,
    )
    assert token_a not in public_scan
    assert token_b not in public_scan

    accepted_b = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': token_b},
        json=_registration(
            id='driver-b',
            url='http://driver-b.local/mcp',
        ),
    )
    assert accepted_b.status_code == 200
    assert accepted_b.json()['data']['trust_state'] == 'trusted'
    assert scheduled == ['driver-a', 'driver-b']


def test_dedicated_credential_rotation_quarantines_until_new_inbound_registration(
    client,
    monkeypatch,
):
    old_token = 'private-driver-a-token-old-0001'
    new_token = 'private-driver-a-token-new-0002'
    scheduled = []
    monkeypatch.setattr(
        mcp_manage,
        '_do_ping',
        lambda mcp_id: scheduled.append(mcp_id) or asyncio.sleep(0),
    )
    auth.init({
        'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': old_token}),
        'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
    })
    first = client.post(
        '/api/mcp',
        headers={'Authorization': f'Bearer {old_token}'},
        json=_registration(id='driver-a', url='http://driver-a.local/mcp'),
    )
    assert first.status_code == 200
    persisted = config.main['services']['mcp'][0]
    old_binding = persisted['credential_binding']
    assert old_token not in old_binding

    auth.init({
        'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': new_token}),
        'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
    })
    assert mcp_manage._effective_trust_state(persisted) == 'quarantined'
    assert mcp_manage._runtime_registration_allowed(persisted) is False
    before_revalidation = deepcopy(config.main['services']['mcp'])
    stale = client.post(
        '/api/mcp',
        headers={'Authorization': f'Bearer {old_token}'},
        json=_registration(id='driver-a', url='http://driver-a.local/mcp'),
    )
    assert stale.status_code == 401
    assert config.main['services']['mcp'] == before_revalidation

    refreshed = client.post(
        '/api/mcp',
        headers={'Authorization': f'Bearer {new_token}'},
        json=_registration(id='driver-a', url='http://driver-a.local/mcp'),
    )
    assert refreshed.status_code == 200
    rebound = config.main['services']['mcp'][0]
    assert rebound['credential_binding'] != old_binding
    assert mcp_manage._effective_trust_state(rebound) == 'trusted'
    assert scheduled == ['driver-a', 'driver-a']


def test_stale_dedicated_binding_cannot_silently_downgrade_to_legacy_fallback(
    client,
    monkeypatch,
):
    dedicated = 'private-driver-a-token-old-0001'
    legacy = 'legacy-driver-fallback'
    scheduled = []
    monkeypatch.setattr(
        mcp_manage,
        '_do_ping',
        lambda mcp_id: scheduled.append(mcp_id) or asyncio.sleep(0),
    )
    auth.init({
        'MOTUS_DRIVER_TOKEN': legacy,
        'MOTUS_DRIVER_TOKENS': json.dumps({'driver-a': dedicated}),
    })
    assert client.post(
        '/api/mcp',
        headers={'Authorization': f'Bearer {dedicated}'},
        json=_registration(id='driver-a', url='http://driver-a.local/mcp'),
    ).status_code == 200
    bound = config.main['services']['mcp'][0]
    assert 'credential_binding' in bound

    auth.init({'MOTUS_DRIVER_TOKEN': legacy})
    assert mcp_manage._effective_trust_state(bound) == 'quarantined'

    rebound = client.post(
        '/api/mcp',
        headers={'Authorization': f'Bearer {legacy}'},
        json=_registration(id='driver-a', url='http://driver-a.local/mcp'),
    )
    assert rebound.status_code == 200
    record = config.main['services']['mcp'][0]
    assert record['trust_state'] == 'trusted'
    assert 'credential_binding' not in record
    assert mcp_manage._effective_trust_state(record) == 'trusted'
    assert scheduled == ['driver-a', 'driver-a']


def test_persisted_trusted_driver_without_selected_credential_is_quarantined(
    client,
    auth_headers,
    shadow_session_tool,
):
    auth.init({
        'ACCESS_TOKEN': 'owner-token',
        'MOTUS_DRIVER_TOKENS': '{"driver-a":"private-driver-a-token-0001"}',
        'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
    })
    tool = deepcopy(shadow_session_tool)
    tool['x-teleop']['driver_id'] = 'driver-b'
    tool['x-teleop']['robot_id'] = 'driver-b'
    target = {
        'id': 'driver-b',
        'name': 'Persisted Driver B',
        'url': 'http://localhost:15712/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [tool],
    }
    config.main['services'] = {'mcp': [target]}
    mcp_client.registry['driver-b'] = {
        'online': True,
        'trusted': True,
        'url': target['url'],
        'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(tool),
    }

    response = client.get('/api/teleop/robots', headers=auth_headers['owner'])

    assert response.status_code == 200
    robot = response.json()['data'][0]
    assert robot['driver_id'] == 'driver-b'
    assert robot['trust_state'] == 'quarantined'
    assert robot['teleop_ready'] is False
    assert robot['reason'] == 'driver_credential_unavailable'
    assert mcp_manage._runtime_registration_allowed(target) is False
    with pytest.raises(
        mcp_client.TrustedShadowTransportError,
        match='driver_auth_unavailable',
    ):
        mcp_client._trusted_shadow_target('driver-b', 'status')


def test_driver_reported_robot_id_cannot_create_an_authority_binding(
    client,
    monkeypatch,
):
    auth.init({
        'ACCESS_TOKEN': 'owner-token',
        'MOTUS_DRIVER_TOKEN': 'driver-secret',
    })
    monkeypatch.setattr(mcp_manage, '_do_ping', lambda _mcp_id: asyncio.sleep(0))

    rejected = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(
            id='standalone-teleop',
            robot_id='victim-robot',
            authority_domain='victim-robot',
        ),
    )
    assert rejected.status_code == 422
    assert config.main['services']['mcp'] == []

    response = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(
            id='standalone-teleop',
            robot_id='victim-robot',
        ),
    )

    assert response.status_code == 200
    record = config.main['services']['mcp'][0]
    assert record['reported_robot_id'] == 'victim-robot'
    assert 'authority_domain' not in record
    assert 'pending_authority_domain' not in record


def test_authority_binding_is_owner_only_and_activates_at_restart_boundary(
    client,
    auth_headers,
    shadow_session_tool,
):
    root_id = 'g1-lab-a'
    adapter_id = 'teleop-shadow-lab-a'
    teleop_tool = deepcopy(shadow_session_tool)
    teleop_tool['x-teleop']['driver_id'] = adapter_id
    teleop_tool['x-teleop']['robot_id'] = root_id
    config.main['services'] = {'mcp': [{
        'id': adapter_id,
        'name': 'Standalone Teleop Adapter',
        'url': 'http://teleop.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': root_id,
        'tools': [teleop_tool],
    }, {
        'id': root_id,
        'name': 'G1 Lab A',
        'url': 'http://g1.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{
            'name': 'locomotion',
            'type': 'actuator',
            'inputSchema': {'type': 'object'},
        }],
    }]}

    forbidden = client.put(
        f'/api/mcp/{adapter_id}/authority-domain',
        headers=auth_headers['operator'],
        json={'root_mcp_id': root_id},
    )
    assert forbidden.status_code == 403

    staged = client.put(
        f'/api/mcp/{adapter_id}/authority-domain',
        headers=auth_headers['owner'],
        json={'root_mcp_id': root_id},
    )
    assert staged.status_code == 202
    assert staged.json()['data']['restart_required'] is True
    record = config.main['services']['mcp'][0]
    assert record['pending_authority_domain'] == root_id
    assert 'authority_domain' not in record

    asyncio.run(mcp_manage.activate_pending_authority_bindings())
    record = config.main['services']['mcp'][0]
    assert record['authority_domain'] == root_id
    assert record['authority_binding_required'] is True
    assert 'pending_authority_domain' not in record

    auth.init({})
    unavailable = client.delete(f'/api/mcp/{adapter_id}/authority-domain')
    assert unavailable.status_code == 503


def test_persistent_guard_locks_delete_retarget_and_binding_changes(
    client,
    auth_headers,
    shadow_session_tool,
    monkeypatch,
):
    adapter_id = 'teleop-shadow-guarded'
    root_id = 'robot-guarded'
    teleop_tool = deepcopy(shadow_session_tool)
    teleop_tool['x-teleop'].update({
        'driver_id': adapter_id,
        'robot_id': root_id,
    })
    original_targets = [{
        'id': adapter_id,
        'name': 'Guarded Teleop Adapter',
        'url': 'http://guarded-teleop.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': root_id,
        'authority_domain': root_id,
        'tools': [teleop_tool],
    }, {
        'id': root_id,
        'name': 'Guarded Robot',
        'url': 'http://guarded-robot.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{
            'name': 'locomotion',
            'type': 'actuator',
            'inputSchema': {'type': 'object'},
        }],
    }]
    config.main['services'] = {'mcp': deepcopy(original_targets)}
    _persist_authority_guard(adapter_id, root_id)

    for locked_id in (adapter_id, root_id):
        response = client.delete(
            f'/api/mcp/{locked_id}',
            headers=auth_headers['owner'],
        )
        assert response.status_code == 409
        assert response.json()['data'] == {
            'code': 'authority_target_locked',
            'reason': 'persistent_authority_guard_requires_stable_target',
            'project_state': 'running',
            'mcp_ids': [locked_id],
        }

    unbind = client.delete(
        f'/api/mcp/{adapter_id}/authority-domain',
        headers=auth_headers['owner'],
    )
    assert unbind.status_code == 409
    assert unbind.json()['data']['code'] == 'authority_target_locked'
    assert unbind.json()['data']['mcp_ids'] == [adapter_id]

    auth.init({
        'ACCESS_TOKEN': 'owner-token',
        'MOTUS_DRIVER_TOKEN': 'driver-secret',
    })

    async def forbidden_ping(_mcp_id):
        raise AssertionError('locked retarget must not schedule discovery')

    monkeypatch.setattr(mcp_manage, '_do_ping', forbidden_ping)
    retarget = client.post(
        '/api/mcp',
        headers={
            'Authorization': 'Bearer owner-token',
            'X-Motus-Driver-Token': 'driver-secret',
        },
        json=_registration(
            id=adapter_id,
            name='Guarded Teleop Adapter',
            url='http://guarded-teleop.invalid/mcp',
            transport='stdio',
            robot_id=root_id,
        ),
    )
    assert retarget.status_code == 409
    assert retarget.json()['data']['code'] == 'authority_target_locked'
    assert retarget.json()['data']['mcp_ids'] == [adapter_id]
    assert config.main['services']['mcp'] == original_targets


def test_startup_defers_pending_unbind_until_persistent_guard_is_cleared(
    shadow_session_tool,
):
    async def scenario() -> None:
        adapter_id = 'teleop-shadow-deferred-unbind'
        root_id = 'robot-deferred-unbind'
        teleop_tool = deepcopy(shadow_session_tool)
        teleop_tool['x-teleop'].update({
            'driver_id': adapter_id,
            'robot_id': root_id,
        })
        config.main['services'] = {'mcp': [{
            'id': adapter_id,
            'name': 'Deferred Unbind Adapter',
            'url': 'http://deferred-unbind.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'reported_robot_id': root_id,
            'authority_domain': root_id,
            'pending_authority_domain': adapter_id,
            'tools': [teleop_tool],
        }, {
            'id': root_id,
            'name': 'Deferred Unbind Robot',
            'url': 'http://deferred-robot.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [{
                'name': 'locomotion',
                'type': 'actuator',
                'inputSchema': {'type': 'object'},
            }],
        }]}
        guard = _persist_authority_guard(adapter_id, root_id)

        await mcp_manage.activate_pending_authority_bindings()
        deferred = config.main['services']['mcp'][0]
        assert deferred['authority_domain'] == root_id
        assert deferred['pending_authority_domain'] == adapter_id

        assert authority_guard.delete_guard(root_id, guard.session_id) is True
        await mcp_manage.activate_pending_authority_bindings()
        activated = config.main['services']['mcp'][0]
        assert 'authority_domain' not in activated
        assert 'pending_authority_domain' not in activated

    asyncio.run(scenario())


def test_config_save_cannot_retarget_guarded_authority_root(
    shadow_session_tool,
):
    async def scenario() -> None:
        adapter_id = 'teleop-shadow-config-guarded'
        root_id = 'robot-config-guarded'
        teleop_tool = deepcopy(shadow_session_tool)
        teleop_tool['x-teleop'].update({
            'driver_id': adapter_id,
            'robot_id': root_id,
        })
        services = {'mcp': [{
            'id': adapter_id,
            'name': 'Config Guarded Adapter',
            'url': 'http://config-guarded-adapter.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'reported_robot_id': root_id,
            'authority_domain': root_id,
            'tools': [teleop_tool],
        }, {
            'id': root_id,
            'name': 'Config Guarded Root',
            'url': 'http://config-guarded-root.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [{
                'name': 'locomotion',
                'type': 'actuator',
                'inputSchema': {'type': 'object'},
            }],
        }]}
        config.main['services'] = deepcopy(services)
        _persist_authority_guard(adapter_id, root_id)
        request = config_api.ConfigSaveRequest(mcp_list=[{
            'id': adapter_id,
            'name': 'Config Guarded Adapter',
            'transport': 'http',
            'url': 'http://config-guarded-adapter.invalid/mcp',
        }, {
            'id': root_id,
            'name': 'Config Guarded Root',
            'transport': 'http',
            'url': 'http://replacement-root.invalid/mcp',
        }])

        with pytest.raises(fastapi.HTTPException) as raised:
            await config_api.config_save(request)
        assert raised.value.status_code == 409
        assert raised.value.detail == {
            'code': 'authority_target_locked',
            'reason': 'persistent_authority_guard_requires_stable_target',
            'project_state': 'running',
            'mcp_ids': [root_id],
        }
        assert config.main['services'] == services

    asyncio.run(scenario())


def test_persistent_guard_does_not_lock_unrelated_target():
    async def scenario() -> None:
        guarded_id = 'guarded-standalone'
        unrelated_id = 'unrelated-sensor'
        guarded = {
            'id': guarded_id,
            'name': 'Guarded Standalone',
            'url': 'http://guarded-standalone.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [],
        }
        unrelated = {
            'id': unrelated_id,
            'name': 'Unrelated Sensor',
            'url': 'http://unrelated.invalid/mcp',
            'transport': 'http',
            'category': 'sensor',
            'trust_state': 'trusted',
            'tools': [],
        }
        config.main['services'] = {'mcp': [guarded, unrelated]}
        _persist_authority_guard(guarded_id, guarded_id)

        response = await mcp_manage.mcp_delete(unrelated_id)
        assert response == {'code': 200}
        assert config.main['services']['mcp'] == [guarded]
        assert authority_guard.get_guard(guarded_id) is not None

    asyncio.run(scenario())


def test_unreadable_guard_store_rejects_target_mutation_and_defers_activation(
    shadow_session_tool,
    monkeypatch,
):
    async def scenario() -> None:
        adapter_id = 'teleop-shadow-store-down'
        root_id = 'robot-store-down'
        teleop_tool = deepcopy(shadow_session_tool)
        teleop_tool['x-teleop'].update({
            'driver_id': adapter_id,
            'robot_id': root_id,
        })
        targets = [{
            'id': adapter_id,
            'name': 'Store Down Adapter',
            'url': 'http://store-down.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'reported_robot_id': root_id,
            'authority_domain': root_id,
            'pending_authority_domain': adapter_id,
            'tools': [teleop_tool],
        }, {
            'id': root_id,
            'name': 'Store Down Robot',
            'url': 'http://store-down-robot.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [{
                'name': 'locomotion',
                'type': 'actuator',
                'inputSchema': {'type': 'object'},
            }],
        }]
        config.main['services'] = {'mcp': deepcopy(targets)}

        def unavailable():
            raise OSError('simulated authority guard store failure')

        monkeypatch.setattr(authority_guard, 'list_guards', unavailable)
        renamed = deepcopy(targets)
        renamed[0]['name'] = 'Changed While Guard Store Is Unknown'
        assert await config_api._project_target_mutation_error(targets, renamed) == {
            'code': 'authority_guard_persistence_error',
            'reason': 'authority_guard_store_unavailable',
            'project_state': 'running',
            'mcp_ids': [adapter_id],
        }
        response = await mcp_manage.mcp_delete(adapter_id)
        assert response.status_code == 409
        assert json.loads(response.body)['data'] == {
            'code': 'authority_guard_persistence_error',
            'reason': 'authority_guard_store_unavailable',
            'project_state': 'running',
            'mcp_ids': [adapter_id],
        }
        await mcp_manage.activate_pending_authority_bindings()
        assert config.main['services']['mcp'] == targets

    asyncio.run(scenario())


def test_guard_store_contention_does_not_block_event_loop():
    async def scenario() -> None:
        target_id = 'contention-target'
        config.main['services'] = {'mcp': [{
            'id': target_id,
            'name': 'Contention Target',
            'url': 'http://contention.invalid/mcp',
            'transport': 'http',
            'category': 'sensor',
            'trust_state': 'trusted',
            'tools': [],
        }]}
        blocker = config._get_conn()
        blocker.execute('BEGIN IMMEDIATE')
        ticks = 0
        ticker_done = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not ticker_done.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        ticker_task = asyncio.create_task(ticker())
        delete_task = asyncio.create_task(mcp_manage.mcp_delete(target_id))
        try:
            await asyncio.sleep(0.06)
            assert ticks >= 5
            assert not delete_task.done()
            blocker.rollback()
            response = await asyncio.wait_for(delete_task, timeout=0.5)
            assert response == {'code': 200}
        finally:
            ticker_done.set()
            blocker.rollback()
            blocker.close()
            await asyncio.gather(ticker_task, return_exceptions=True)
            if not delete_task.done():
                delete_task.cancel()
            await asyncio.gather(delete_task, return_exceptions=True)

    asyncio.run(scenario())


def test_startup_migration_preserves_mcp_snapshot_while_guard_is_pending():
    guarded_id = 'guarded-migration-target'
    duplicate_url = 'http://duplicate-during-recovery.invalid/mcp'
    targets = [{
        'id': guarded_id,
        'name': 'Guarded Migration Target',
        'url': duplicate_url,
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [],
    }, {
        'id': 'legacy-duplicate-target',
        'name': 'Legacy Duplicate Target',
        'url': duplicate_url,
        'transport': 'http',
        'category': 'sensor',
        'trust_state': 'untrusted',
        'tools': [],
    }]
    config.main['services'] = {'mcp': deepcopy(targets)}
    _persist_authority_guard(guarded_id, guarded_id)

    config._migrate()

    assert config.main['services']['mcp'] == targets


def test_staged_unbind_keeps_active_binding_valid_for_driver_heartbeats(
    client,
    monkeypatch,
):
    adapter_id = 'teleop-shadow-lab-a'
    root_id = 'g1-lab-a'
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    monkeypatch.setattr(mcp_manage, '_do_ping', lambda _mcp_id: asyncio.sleep(0))
    config.main['services'] = {'mcp': [{
        'id': adapter_id,
        'name': 'Standalone Teleop Adapter',
        'url': 'http://teleop.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': root_id,
        'authority_domain': root_id,
        'pending_authority_domain': adapter_id,
        'tools': [],
    }]}

    response = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(
            id=adapter_id,
            name='Standalone Teleop Adapter',
            url='http://teleop.invalid/mcp',
            robot_id=root_id,
        ),
    )

    assert response.status_code == 200
    record = config.main['services']['mcp'][0]
    assert record['authority_domain'] == root_id
    assert record['pending_authority_domain'] == adapter_id


def test_shared_driver_token_cannot_retarget_or_omit_bound_identity(
    client,
    monkeypatch,
):
    adapter_id = 'teleop-shadow-secure'
    root_id = 'robot-secure'
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    monkeypatch.setattr(mcp_manage, '_do_ping', lambda _mcp_id: asyncio.sleep(0))
    original = {
        'id': adapter_id,
        'name': 'Secure Adapter',
        'url': 'http://secure-adapter.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': root_id,
        'authority_domain': root_id,
        'tools': [],
    }
    config.main['services'] = {'mcp': [deepcopy(original)]}

    retarget = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(
            id=adapter_id,
            name='Attacker',
            url='http://attacker.invalid/mcp',
        ),
    )
    assert retarget.status_code == 409
    assert config.main['services']['mcp'][0] == original

    omitted_identity = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(
            id=adapter_id,
            name='Attacker',
            url=original['url'],
        ),
    )
    assert omitted_identity.status_code == 409
    assert config.main['services']['mcp'][0] == original


def test_cross_id_binding_requires_descriptor_robot_identity(
    client,
    auth_headers,
    shadow_session_tool,
):
    adapter_id = 'teleop-shadow-missing-robot'
    root_id = 'robot-root'
    teleop_tool = deepcopy(shadow_session_tool)
    teleop_tool['x-teleop']['driver_id'] = adapter_id
    teleop_tool['x-teleop'].pop('robot_id')
    config.main['services'] = {'mcp': [{
        'id': adapter_id,
        'name': 'Missing Robot Adapter',
        'url': 'http://missing-robot.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'reported_robot_id': root_id,
        'tools': [teleop_tool],
    }, {
        'id': root_id,
        'name': 'Robot Root',
        'url': 'http://robot-root.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{
            'name': 'locomotion',
            'type': 'actuator',
            'inputSchema': {'type': 'object'},
        }],
    }]}

    response = client.put(
        f'/api/mcp/{adapter_id}/authority-domain',
        headers=auth_headers['owner'],
        json={'root_mcp_id': root_id},
    )

    assert response.status_code == 409
    assert response.json()['detail'] == {
        'code': 'authority_binding_invalid',
        'reason': 'descriptor_robot_id_mismatch',
    }
    assert 'pending_authority_domain' not in config.main['services']['mcp'][0]


def test_config_save_cannot_remove_active_root_while_unbind_is_pending(
    shadow_session_tool,
):
    async def scenario() -> None:
        adapter_id = 'teleop-shadow-pending-unbind'
        root_id = 'robot-active-root'
        teleop_tool = deepcopy(shadow_session_tool)
        teleop_tool['x-teleop'].update({
            'driver_id': adapter_id,
            'robot_id': root_id,
        })
        config.main['services'] = {'mcp': [{
            'id': adapter_id,
            'name': 'Pending Unbind Adapter',
            'url': 'http://pending-unbind.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'reported_robot_id': root_id,
            'authority_domain': root_id,
            'pending_authority_domain': adapter_id,
            'tools': [teleop_tool],
        }, {
            'id': root_id,
            'name': 'Active Root',
            'url': 'http://active-root.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [{
                'name': 'locomotion',
                'type': 'actuator',
                'inputSchema': {'type': 'object'},
            }],
        }]}
        request = config_api.ConfigSaveRequest(mcp_list=[{
            'id': adapter_id,
            'name': 'Pending Unbind Adapter',
            'transport': 'http',
            'url': 'http://pending-unbind.invalid/mcp',
        }])

        with pytest.raises(fastapi.HTTPException) as error:
            await config_api.config_save(request)

        assert getattr(error.value, 'status_code', None) == 409
        assert error.value.detail['reason'] == 'authority_root_would_be_removed'
        assert config.main['services']['mcp'][1]['id'] == root_id

    asyncio.run(scenario())


def test_owner_target_change_invalidates_old_capability_snapshot():
    async def scenario() -> None:
        mcp_id = 'owner-retargeted-driver'
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Owner Retargeted Driver',
            'url': 'http://old-target.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [{
                'name': 'locomotion',
                'type': 'actuator',
                'inputSchema': {'type': 'object'},
            }],
            'resources': [{'uri': 'robot://state'}],
        }]}
        mcp_client.registry[mcp_id] = {
            'url': 'http://old-target.invalid/mcp',
            'online': True,
        }
        request = config_api.ConfigSaveRequest(mcp_list=[{
            'id': mcp_id,
            'name': 'Owner Retargeted Driver',
            'transport': 'http',
            'url': 'http://new-target.invalid/mcp',
        }])

        response = await config_api.config_save(request)

        assert response['code'] == 200
        record = config.main['services']['mcp'][0]
        assert record['url'] == 'http://new-target.invalid/mcp'
        assert record['tools'] == []
        assert record['resources'] == []
        assert record['capability_refresh_required'] is True
        assert mcp_id not in mcp_client.registry

    asyncio.run(scenario())


def test_settings_save_preserves_internal_credential_binding():
    async def scenario() -> None:
        import client as client_mod

        mcp_id = 'driver-with-dedicated-binding'
        auth.init({
            'MOTUS_DRIVER_TOKENS': json.dumps({
                mcp_id: 'private-driver-token-bound-0001',
            }),
        })
        binding = auth.driver_credential_binding(mcp_id)
        config.main['services'] = {'mcp': [{
            'id': mcp_id,
            'name': 'Bound Driver',
            'url': 'http://bound-driver.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'credential_binding': binding,
            'tools': [],
        }]}
        previous_client = client_mod.llm
        try:
            response = await config_api.config_save(config_api.ConfigSaveRequest(
                mcp_list=[{
                    'id': mcp_id,
                    'name': 'Bound Driver',
                    'url': 'http://bound-driver.invalid/mcp',
                    'transport': 'http',
                }],
            ))
            assert response['code'] == 200
            record = config.main['services']['mcp'][0]
            assert record['credential_binding'] == binding
            assert mcp_manage._effective_trust_state(record) == 'trusted'
        finally:
            replacement = client_mod.llm
            client_mod.llm = previous_client
            if replacement is not previous_client:
                await replacement.aclose()

    asyncio.run(scenario())


def test_retargeted_authority_root_cannot_reuse_old_actuator_proof(
    shadow_session_tool,
):
    async def scenario() -> None:
        adapter_id = 'retarget-binding-adapter'
        root_id = 'retarget-binding-root'
        teleop_tool = deepcopy(shadow_session_tool)
        teleop_tool['x-teleop'].update({
            'driver_id': adapter_id,
            'robot_id': root_id,
        })
        services = {'mcp': [{
            'id': adapter_id,
            'name': 'Retarget Binding Adapter',
            'url': 'http://binding-adapter.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'reported_robot_id': root_id,
            'authority_domain': root_id,
            'tools': [teleop_tool],
        }, {
            'id': root_id,
            'name': 'Retarget Binding Root',
            'url': 'http://binding-root-old.invalid/mcp',
            'transport': 'http',
            'category': 'driver',
            'trust_state': 'trusted',
            'tools': [{
                'name': 'locomotion',
                'type': 'actuator',
                'inputSchema': {'type': 'object'},
            }],
        }]}
        config.main['services'] = services
        request = config_api.ConfigSaveRequest(mcp_list=[{
            'id': adapter_id,
            'name': 'Retarget Binding Adapter',
            'transport': 'http',
            'url': 'http://binding-adapter.invalid/mcp',
        }, {
            'id': root_id,
            'name': 'Retarget Binding Root',
            'transport': 'http',
            'url': 'http://binding-root-new.invalid/mcp',
        }])

        with pytest.raises(fastapi.HTTPException) as error:
            await config_api.config_save(request)

        assert error.value.status_code == 409
        assert error.value.detail['reason'] == 'authority_binding_would_be_invalidated'
        assert error.value.detail['bindings'] == [{
            'mcp_id': adapter_id,
            'root_mcp_id': root_id,
            'reason': 'root_missing_ordinary_actuator',
        }]
        assert config.main['services'] == services

    asyncio.run(scenario())


def test_active_project_registration_cannot_change_stop_target(
    client,
    monkeypatch,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    mcp_id = 'active-registration-driver'
    target = {
        'id': mcp_id,
        'name': 'Active Registration Driver',
        'url': 'http://active-registration.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{
            'name': 'locomotion',
            'type': 'actuator',
            'inputSchema': {'type': 'object'},
        }],
    }
    config.main['services'] = {'mcp': [target]}
    config_api._set_project_state('running', cards=[{
        'id': 'active-registration-card',
        'mcpId': mcp_id,
        'toolName': 'locomotion',
    }])

    async def forbidden_ping(_mcp_id):
        raise AssertionError('rejected registration must not schedule a ping')

    monkeypatch.setattr(mcp_manage, '_do_ping', forbidden_ping)
    response = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(
            id=mcp_id,
            name='Active Registration Driver',
            url=target['url'],
            transport='stdio',
        ),
    )

    assert response.status_code == 409
    assert response.json()['data'] == {
        'code': 'project_target_locked',
        'reason': 'active_project_requires_stable_stop_targets',
        'project_state': 'running',
        'mcp_ids': [mcp_id],
    }
    assert config.main['services']['mcp'] == [target]


@pytest.mark.parametrize(
    ('reserved_id', 'settings', 'headers'),
    [
        (
            'agentcore',
            {'MOTUS_DRIVER_TOKEN': 'driver-secret'},
            {'X-Motus-Driver-Token': 'driver-secret'},
        ),
        (
            'channel',
            {'MOTUS_DRIVER_TOKEN': 'driver-secret'},
            {'X-Motus-Driver-Token': 'driver-secret'},
        ),
        ('agentcore', {}, {}),
        (
            'agentcore',
            {
                'MOTUS_DRIVER_TOKEN': 'driver-secret',
                'MOTUS_ENFORCE_DRIVER_AUTH': 'true',
            },
            {},
        ),
    ],
    ids=['trusted-agentcore', 'trusted-channel', 'untrusted', 'quarantined'],
)
def test_external_registration_cannot_claim_reserved_internal_id(
    client, monkeypatch, reserved_id, settings, headers,
):
    auth.init(settings)
    internal_records = [
        {
            'id': 'agentcore',
            'name': 'Agent Core',
            'transport': 'internal',
            'url': '',
            'server_name': 'AgentCore',
            'category': 'controller',
            'trust_state': 'internal',
            'online': True,
            'tools': [{'name': 'decision_core'}],
        },
        {
            'id': 'channel',
            'name': 'Channel',
            'transport': 'internal',
            'url': '',
            'server_name': 'Channel',
            'category': 'controller',
            'trust_state': 'internal',
            'online': True,
            'tools': [{'name': 'channel_request'}],
        },
    ]
    config.main['services'] = {'mcp': deepcopy(internal_records)}

    def forbidden_ping(mcp_id):  # pragma: no cover - invocation fails the test
        raise AssertionError(f'internal collision scheduled ping for {mcp_id}')

    monkeypatch.setattr(mcp_manage, '_do_ping', forbidden_ping)
    before = json.dumps(config.main['services'], sort_keys=True)

    response = client.post(
        '/api/mcp',
        headers=headers,
        json=_registration(
            id=reserved_id,
            name='External Driver',
            url='http://external.local/mcp',
        ),
    )

    assert response.status_code == 409
    assert response.json()['data'] == {
        'id': reserved_id, 'trust_state': 'internal',
    }
    assert json.dumps(config.main['services'], sort_keys=True) == before
    assert config.main['services']['mcp'] == internal_records


@pytest.mark.parametrize(
    'registration_fields',
    [
        {'name': 'Agent Core', 'url': 'http://external.local/mcp'},
        {'name': 'External Driver', 'url': 'internal://agent-core'},
        {'name': 'AgentCore', 'url': 'http://external.local/mcp'},
    ],
    ids=['same-name', 'same-url', 'same-server-name'],
)
def test_external_registration_descriptors_cannot_merge_internal_record(
    client, monkeypatch, registration_fields,
):
    internal = {
        'id': 'custom-internal',
        'name': 'Agent Core',
        'transport': 'internal',
        'url': 'internal://agent-core',
        'server_name': 'AgentCore',
        'category': 'controller',
        'trust_state': 'internal',
        'tools': [{'name': 'decision_core'}],
    }
    config.main['services'] = {'mcp': [deepcopy(internal)]}
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})

    def forbidden_ping(mcp_id):  # pragma: no cover - invocation fails the test
        raise AssertionError(f'internal collision scheduled ping for {mcp_id}')

    monkeypatch.setattr(mcp_manage, '_do_ping', forbidden_ping)
    response = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': 'driver-secret'},
        json=_registration(id='external-driver', **registration_fields),
    )

    assert response.status_code == 409
    assert config.main['services']['mcp'] == [internal]


@pytest.mark.parametrize('bad_id', ['', 'spaces are invalid', '../robot'])
def test_trusted_registration_requires_valid_stable_id(client, bad_id):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    response = client.post(
        '/api/mcp',
        headers={'Authorization': 'Bearer driver-secret'},
        json=_registration(id=bad_id),
    )
    assert response.status_code == 422
    assert config.main['services']['mcp'] == []


@pytest.mark.parametrize(
    'attacker_overrides',
    [
        {'name': 'Trusted Robot', 'url': 'http://attacker.local/mcp'},
        {'name': 'Attacker', 'url': 'http://trusted.local/mcp'},
        {'name': 'TrustedServerName', 'url': 'http://attacker.local/mcp'},
    ],
    ids=['same-name', 'same-url', 'same-server-name'],
)
def test_untrusted_registration_cannot_merge_trusted_identity(
    client, attacker_overrides,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    trusted = {
        'id': 'trusted-robot',
        'name': 'Trusted Robot',
        'server_name': 'TrustedServerName',
        'url': 'http://trusted.local/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{'name': 'safe-tool'}],
    }
    config.main['services'] = {'mcp': [deepcopy(trusted)]}

    response = client.post('/api/mcp', json=_registration(**attacker_overrides))

    assert response.status_code == 403
    assert config.main['services']['mcp'] == [trusted]


def test_untrusted_initialize_name_spoof_cannot_touch_trusted_record(
    monkeypatch,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    trusted = {
        'id': 'trusted-robot',
        'name': 'Trusted Robot',
        'server_name': 'SharedServerName',
        'url': 'http://trusted.local/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [{'name': 'safe-tool'}],
    }
    attacker = {
        'id': 'legacy-attacker',
        'name': 'Different Registration Name',
        'url': 'http://attacker.local/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'untrusted',
    }
    config.main['services'] = {'mcp': [deepcopy(trusted), attacker]}
    trusted_runtime = {'online': True, 'url': trusted['url'], 'sentinel': 'unchanged'}
    mcp_client.registry.update({
        trusted['id']: deepcopy(trusted_runtime),
        attacker['id']: {'online': False, 'url': attacker['url']},
    })

    async def fake_ping(url, *, trusted=False, driver_id=''):
        assert url == attacker['url']
        assert trusted is False
        return {
            'server_name': trusted_record_name,
            'tools': [{'name': 'attacker-tool'}],
            'resources': [],
            'device_type': '',
            'topic_out': [],
            'topic_in': [],
        }

    trusted_record_name = trusted['server_name']
    monkeypatch.setattr(mcp_manage, '_ping_mcp_http', fake_ping)
    result = asyncio.run(mcp_manage._do_ping(attacker['id']))

    assert result['online'] is False
    assert result['error'] == 'untrusted duplicate of trusted service'
    assert config.main['services']['mcp'] == [trusted]
    assert mcp_client.registry[trusted['id']] == trusted_runtime
    assert attacker['id'] not in mcp_client.registry


def test_external_ping_server_name_cannot_affect_internal_identity(
    monkeypatch, shadow_session_tool,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    internal = {
        'id': 'agentcore',
        'name': 'Agent Core',
        'server_name': 'AgentCore',
        'url': '',
        'transport': 'internal',
        'category': 'controller',
        'trust_state': 'internal',
        'tools': [{'name': 'decision_core'}],
    }
    external = {
        'id': 'external-driver',
        'name': 'Non-Colliding Registration Name',
        'url': 'http://external.local/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
    }
    config.main['services'] = {'mcp': [deepcopy(internal), external]}
    internal_runtime = {'online': True, 'transport': 'internal', 'sentinel': 'unchanged'}
    mcp_client.registry.update({
        internal['id']: deepcopy(internal_runtime),
        external['id']: {'online': False, 'url': external['url']},
    })

    async def spoofed_ping(url, *, trusted=False, driver_id=''):
        assert trusted is True
        return {
            'server_name': 'AgentCore',
            'tools': [deepcopy(shadow_session_tool)],
            'resources': [],
            'device_type': '',
            'topic_out': [],
            'topic_in': [],
        }

    monkeypatch.setattr(mcp_manage, '_ping_mcp_http', spoofed_ping)
    result = asyncio.run(mcp_manage._do_ping(external['id']))

    assert result['online'] is False
    assert result['error'] == 'external duplicate of internal service'
    assert config.main['services']['mcp'] == [internal]
    assert mcp_client.registry[internal['id']] == internal_runtime
    assert external['id'] not in mcp_client.registry


def test_two_trusted_instances_with_same_server_name_remain_distinct(
    client, monkeypatch, shadow_session_tool,
):
    auth.init({'MOTUS_DRIVER_TOKEN': 'driver-secret'})
    scheduled = []

    def registration_ping(mcp_id):
        scheduled.append(mcp_id)

        async def done():
            return {}

        return done()

    real_do_ping = mcp_manage._do_ping
    monkeypatch.setattr(mcp_manage, '_do_ping', registration_ping)
    headers = {'X-Motus-Driver-Token': 'driver-secret'}
    for robot_id in ('robot-a', 'robot-b'):
        response = client.post(
            '/api/mcp',
            headers=headers,
            json=_registration(
                id=robot_id,
                name='Generic Teleop Shadow Diagnostics',
                url=f'http://{robot_id}.local/mcp',
            ),
        )
        assert response.status_code == 200
        assert response.json()['data']['id'] == robot_id

    assert scheduled == ['robot-a', 'robot-b']
    assert [item['id'] for item in config.main['services']['mcp']] == ['robot-a', 'robot-b']

    async def same_server_name(url, *, trusted=False, driver_id=''):
        assert trusted is True
        return {
            'server_name': 'teleop-shadow',
            'tools': [deepcopy(shadow_session_tool)],
            'resources': [],
            'device_type': '',
            'topic_out': [],
            'topic_in': [],
        }

    async def no_op(*args, **kwargs):
        return None

    monkeypatch.setattr(mcp_manage, '_do_ping', real_do_ping)
    monkeypatch.setattr(mcp_manage, '_ping_mcp_http', same_server_name)
    monkeypatch.setattr(mcp_manage, '_notify_inspector', no_op)
    monkeypatch.setattr(mcp_manage, '_restore_saved_configs', no_op)

    async def ping_both():
        await mcp_manage._do_ping('robot-a')
        await mcp_manage._do_ping('robot-b')
        await asyncio.sleep(0)

    asyncio.run(ping_both())

    records = config.main['services']['mcp']
    assert {record['id'] for record in records} == {'robot-a', 'robot-b'}
    assert {record['server_name'] for record in records} == {'teleop-shadow'}
    assert {record['url'] for record in records} == {
        'http://robot-a.local/mcp', 'http://robot-b.local/mcp',
    }
    assert mcp_client.registry['robot-a']['online'] is True
    assert mcp_client.registry['robot-b']['online'] is True


def test_driver_secret_is_neither_persisted_nor_returned_or_logged(
    client, monkeypatch, capsys,
):
    secret = 'driver-secret-never-persist'
    auth.init({
        'ACCESS_TOKEN': 'owner-token-for-secret-proof',
        'MOTUS_DRIVER_TOKEN': secret,
    })

    def fake_ping(mcp_id):
        async def done():
            return {}

        return done()

    monkeypatch.setattr(mcp_manage, '_do_ping', fake_ping)
    response = client.post(
        '/api/mcp',
        headers={'X-Motus-Driver-Token': secret},
        json=_registration(),
    )
    listing = client.get(
        '/api/mcp',
        headers={'Authorization': 'Bearer owner-token-for-secret-proof'},
    )

    assert response.status_code == 200
    assert listing.status_code == 200
    assert secret not in json.dumps(config.main['services'])
    assert secret not in response.text
    assert secret not in listing.text
    assert secret not in capsys.readouterr().out
