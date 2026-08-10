from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy

import pytest

import auth
import config
import mcp_client
from api import teleop
from teleop import audit
from teleop.models import ShadowSession
from teleop.service import AcquireResult, TeleopServiceError
from teleop.session_manager import (
    SessionClientMismatch,
    SessionConflict,
    SessionForbidden,
    SessionNotFound,
    SessionStateConflict,
)

DRIVER_TOKEN = 'driver-token-that-must-never-cross-the-api-boundary'
CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'
OTHER_CLIENT_ID = '68991413-d37a-4603-9f07-3e25219d6d96'
ALICE_SESSION_ID = 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3'
BOB_SESSION_ID = 'b6f55f8c-0989-48ef-bf9a-55ed0bbb60df'
ACQUIRED_SESSION_ID = '0ae6c709-36f8-4730-a388-62160ee742a8'
FENCE_BY_SESSION = {
    ALICE_SESSION_ID: 'alice-real-session-fence-0123456789abcdef',
    BOB_SESSION_ID: 'bob-real-session-fence-0123456789abcdef',
    ACQUIRED_SESSION_ID: 'acquired-real-session-fence-0123456789abcdef',
}


def _session(
    session_id: str,
    robot_id: str,
    principal_id: str,
    *,
    state: str = 'active',
) -> ShadowSession:
    return ShadowSession(
        id=session_id,
        robot_id=robot_id,
        driver_id=robot_id,
        principal_id=principal_id,
        boot_id='72559c63-e2a7-46ed-a50f-e8128ed8aa2b',
        epoch=7,
        capability_digest='0123456789abcdef' * 4,
        client_id=CLIENT_ID if principal_id != 'operator:bob' else OTHER_CLIENT_ID,
        fence=FENCE_BY_SESSION.get(
            session_id,
            f'generated-real-session-fence-{session_id}',
        ),
        state=state,
        operation_generation=3,
        operation_state='succeeded',
        created_at=time.time() - 1.0,
        lease_seconds=15.0,
        deadline_monotonic=time.monotonic() + 15.0,
    )


class _FakeManager:
    def __init__(self, sessions: list[ShadowSession]):
        self.sessions = {session.id: session for session in sessions}
        self.robot_sessions = {
            session.robot_id: session.id
            for session in sessions
            if session.state not in {'released', 'expired', 'faulted'}
        }
        self.get_authorized_calls: list[tuple[str, str, bool, bool]] = []

    async def active_for_robot(self, robot_id: str) -> ShadowSession | None:
        session_id = self.robot_sessions.get(robot_id)
        return self.sessions.get(session_id) if session_id else None

    async def get(self, session_id: str) -> ShadowSession | None:
        return self.sessions.get(session_id)

    async def get_authorized(
        self,
        session_id: str,
        principal_id: str,
        owner: bool = False,
        include_terminal: bool = False,
    ) -> ShadowSession:
        self.get_authorized_calls.append(
            (session_id, principal_id, owner, include_terminal),
        )
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        if session.state in {'released', 'expired', 'faulted'} and not include_terminal:
            raise SessionNotFound(session_id)
        if not owner and session.principal_id != principal_id:
            raise SessionForbidden(session_id)
        return session

    def public_dict(self, session: ShadowSession) -> dict:
        return session.public_dict()


class _FakeCoordinator:
    def __init__(self):
        self.alice = _session(
            ALICE_SESSION_ID,
            'robot-a',
            'operator:alice',
        )
        self.bob = _session(
            BOB_SESSION_ID,
            'robot-b',
            'operator:bob',
        )
        self.manager = _FakeManager([self.alice, self.bob])
        self.heartbeat_calls: list[tuple[str, str]] = []
        self.signaling_calls: list[tuple[str, str, str, dict]] = []
        self.confirm_calls: list[tuple[str, str, str, str]] = []
        self.live_prepare_count = 0
        self.authority_guards: dict[str, dict] = {}
        self.reconcile_calls: list[tuple[str, str]] = []

    def authority_guard_for_robot(self, robot_id: str) -> dict | None:
        return self.authority_guards.get(robot_id)

    def list_authority_guards(self) -> list[dict]:
        return [self.authority_guards[key] for key in sorted(self.authority_guards)]

    async def reconcile_authority_guard(
        self,
        robot_id: str,
        *,
        principal_id: str,
    ) -> dict:
        self.reconcile_calls.append((robot_id, principal_id))
        guard = self.authority_guards.pop(robot_id, None)
        return {
            'state': 'clear',
            'robot_id': robot_id,
            'driver_id': guard['driver_id'] if guard else robot_id,
            'old_session_restored': False,
            'reacquire_required': True,
            'already_clear': guard is None,
        }

    async def acquire(
        self,
        driver_id: str,
        principal_id: str,
        client_id: str,
        mode: str = 'shadow',
    ) -> AcquireResult:
        assert client_id == CLIENT_ID
        assert mode in {'shadow', 'live'}
        if driver_id == 'robot-busy':
            raise SessionConflict(self.bob)
        service_errors = {
            'robot-not-ready': ('driver_not_ready', 503),
            'robot-timeout': ('driver_timeout', 504),
            'robot-protocol-error': ('driver_protocol_error', 502),
        }
        if driver_id in service_errors:
            code, status = service_errors[driver_id]
            raise TeleopServiceError(code, status)

        disposition = {
            'robot-new': 'created',
            'robot-existing': 'existing',
            'robot-preparing': 'preparing',
        }.get(driver_id, 'created')
        if mode == 'live':
            disposition = 'confirmation_required'
        state = (
            'awaiting_confirmation'
            if mode == 'live'
            else 'preparing'
            if disposition == 'preparing'
            else 'active'
        )
        session = _session(
            ACQUIRED_SESSION_ID,
            driver_id,
            principal_id,
            state=state,
        )
        if mode == 'live':
            self._configure_live(session)
        return AcquireResult(session=session, disposition=disposition)

    @staticmethod
    def _configure_live(session: ShadowSession) -> None:
        session.mode = 'live'
        session.profile_id = 'dual_arm_profile_v1'
        session.capabilities = {
            'profile_id': session.profile_id,
            'input_bindings': {},
            'outputs': {'dual_arm': {'enabled': True, 'joint_count': 10}},
            'effectors': ['dual_arm'],
        }
        session.effectors = ['dual_arm']
        session.signaling_audience = 'motus-teleop-rtc'
        session.live_confirmed = False

    async def confirm_live(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        profile_id: str,
    ) -> AcquireResult:
        session = self.manager.sessions.get(session_id)
        if session is None or session.state in {'released', 'expired', 'faulted'}:
            raise SessionNotFound(session_id)
        if session.principal_id != principal_id:
            raise SessionForbidden(session_id)
        if session.client_id != client_id:
            raise SessionClientMismatch(session_id)
        if session.mode != 'live' or session.profile_id != profile_id:
            raise TeleopServiceError('live_confirmation_mismatch', 409)
        self.confirm_calls.append((session_id, principal_id, client_id, profile_id))
        if session.state == 'active' and session.live_confirmed:
            return AcquireResult(session=session, disposition='existing')
        if session.state != 'awaiting_confirmation' or session.live_confirmed:
            raise SessionStateConflict(session_id)
        self.live_prepare_count += 1
        session.live_confirmed = True
        session.state = 'active'
        session.operation_state = 'succeeded'
        session.epoch = 8
        return AcquireResult(session=session, disposition='created')

    def public_session(self, session: ShadowSession) -> dict:
        result = self.manager.public_dict(session)
        result['driver'] = {
            'driver_id': session.driver_id,
            'mode': session.mode,
            'actuation_enabled': session.mode == 'live',
            'profile_id': session.profile_id,
            'state': {
                'active': f'prepared_{session.mode}',
                'paused': 'paused',
                'hold': 'hold',
                'released': 'released',
            }.get(session.state, session.state),
            'authority_valid': session.state != 'released',
        }
        result['driver_heartbeat'] = {
            'state': 'healthy' if session.state != 'released' else 'stopped',
            'last_confirmed_at': time.time(),
            'consecutive_failures': 0,
        }
        return result

    async def sessions_for(self, principal_id: str, *, owner: bool) -> list[dict]:
        sessions = list(self.manager.sessions.values())
        if not owner:
            sessions = [session for session in sessions if session.principal_id == principal_id]
        return [self.public_session(session) for session in sessions]

    async def session_for(
        self,
        session_id: str,
        principal_id: str,
        *,
        owner: bool,
    ) -> dict:
        session = await self.manager.get_authorized(
            session_id,
            principal_id,
            owner=owner,
            include_terminal=True,
        )
        return self.public_session(session)

    async def status(
        self,
        session_id: str,
        principal_id: str,
        *,
        owner: bool,
    ) -> dict:
        session = await self.manager.get_authorized(
            session_id,
            principal_id,
            owner=owner,
        )
        return self.public_session(session)

    async def heartbeat(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
    ) -> ShadowSession:
        assert client_id == CLIENT_ID
        # This intentionally mirrors the real coordinator: owner status does not
        # bypass heartbeat ownership.
        session = await self.manager.get_authorized(
            session_id,
            principal_id,
            owner=False,
        )
        self.heartbeat_calls.append((session_id, principal_id))
        session.deadline_monotonic = time.monotonic() + session.lease_seconds
        return session

    async def signaling_offer(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        offer: dict,
    ) -> dict:
        session = self.manager.sessions.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        if session.principal_id != principal_id:
            raise SessionForbidden(session_id)
        if session.client_id != client_id:
            raise SessionClientMismatch(session_id)
        if session.state in {'released', 'expired', 'faulted'}:
            raise SessionNotFound(session_id)
        if session.state != 'active':
            raise SessionStateConflict(session_id)
        self.signaling_calls.append((session_id, principal_id, client_id, offer))
        return {'type': 'answer', 'sdp': 'v=0\r\na=fake-driver-answer'}

    async def pause(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> ShadowSession:
        assert client_id == CLIENT_ID
        session = await self.manager.get_authorized(session_id, principal_id, owner=owner)
        if session.state not in {'active', 'paused', 'hold'}:
            raise SessionStateConflict(session_id)
        session.state = 'paused'
        return session

    async def soft_stop(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> ShadowSession:
        assert client_id == CLIENT_ID
        session = await self.manager.get_authorized(session_id, principal_id, owner=owner)
        if session.state not in {'active', 'hold'}:
            raise SessionStateConflict(session_id)
        session.state = 'hold'
        return session

    async def release(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> tuple[ShadowSession, bool]:
        assert client_id == CLIENT_ID
        session = await self.manager.get_authorized(
            session_id,
            principal_id,
            owner=owner,
            include_terminal=True,
        )
        if session.state == 'released':
            return session, True
        if session.state in {'expired', 'faulted'}:
            return session, False
        session.state = 'released'
        session.operation_state = 'cancelled'
        self.manager.robot_sessions.pop(session.robot_id, None)
        return session, True


class _ApiHarness:
    def __init__(self, client, auth_headers, coordinator: _FakeCoordinator):
        self.client = client
        self.headers = auth_headers
        self.coordinator = coordinator

    def request(self, method: str, path: str, *, role: str = 'operator', **kwargs):
        response = self.client.request(
            method,
            path,
            headers=self.headers[role],
            **kwargs,
        )
        _assert_secret_free(response)
        return response


def _walk_json(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)
    elif value is not None:
        yield str(value)


def _assert_secret_free(response) -> None:
    payload = response.json()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    leaves = list(_walk_json(payload))
    for secret in (*FENCE_BY_SESSION.values(), DRIVER_TOKEN):
        assert secret not in serialized
        assert all(secret not in leaf for leaf in leaves)


def _robot_record(shadow_session_tool: dict, robot_id: str, name: str) -> dict:
    tool = deepcopy(shadow_session_tool)
    tool['x-teleop']['driver_id'] = robot_id
    tool['x-teleop']['robot_id'] = robot_id
    return {
        'id': robot_id,
        'name': name,
        'server_name': f'{robot_id}-shadow',
        'url': f'https://{robot_id}.invalid/mcp',
        'transport': 'http',
        'category': 'driver',
        'trust_state': 'trusted',
        'tools': [tool],
    }


@pytest.fixture
def api_harness(
    client,
    auth_headers,
    shadow_session_tool,
    monkeypatch: pytest.MonkeyPatch,
):
    auth.init({
        'ACCESS_TOKEN': 'owner-token',
        'MOTUS_OPERATOR_TOKENS': '{"alice":"operator-token"}',
        'MOTUS_VIEWER_TOKENS': '{"auditor":"viewer-token"}',
        'MOTUS_DRIVER_TOKEN': DRIVER_TOKEN,
        'MOTUS_TELEOP_TICKET_SECRET': 'test-session-api-ticket-secret-0001',
    })
    coordinator = _FakeCoordinator()
    monkeypatch.setattr(teleop, 'coordinator', coordinator)

    records = [
        _robot_record(shadow_session_tool, 'robot-a', 'A Owned Robot'),
        _robot_record(shadow_session_tool, 'robot-b', 'B Other Robot'),
        _robot_record(shadow_session_tool, 'robot-idle', 'C Idle Robot'),
    ]
    config.main['services'] = {'mcp': records}
    for record in records:
        mcp_client.registry[record['id']] = {
            'online': True,
            'trusted': True,
            'url': record['url'],
            'teleop_fingerprint': mcp_client.teleop_tool_fingerprint(record['tools'][0]),
        }
    return _ApiHarness(client, auth_headers, coordinator)


def test_me_control_permission_matches_role(api_harness: _ApiHarness):
    expectations = {'viewer': False, 'operator': True, 'owner': True}
    for role, expected in expectations.items():
        response = api_harness.request('GET', '/api/teleop/me', role=role)
        assert response.status_code == 200
        assert response.json()['code'] == 200
        assert response.json()['data']['permissions'] == {
            'view_devices': True,
            'control': expected,
        }


def test_robot_busy_state_is_deidentified_except_for_owner_or_session_operator(
    api_harness: _ApiHarness,
):
    viewer = api_harness.request('GET', '/api/teleop/robots', role='viewer')
    operator = api_harness.request('GET', '/api/teleop/robots', role='operator')
    owner = api_harness.request('GET', '/api/teleop/robots', role='owner')

    def robots(response):
        return {item['id']: item for item in response.json()['data']}

    viewer_robots = robots(viewer)
    operator_robots = robots(operator)
    owner_robots = robots(owner)

    assert viewer_robots['robot-idle']['session'] == {
        'busy': False,
        'owned_by_me': False,
        'owned_by_client': False,
    }
    for robot_id in ('robot-a', 'robot-b'):
        visible = viewer_robots[robot_id]['session']
        assert visible['busy'] is True
        assert visible['owned_by_me'] is False
        assert 'id' not in visible
        assert 'principal_id' not in visible

    assert operator_robots['robot-a']['session']['owned_by_me'] is True
    assert operator_robots['robot-a']['session']['owned_by_client'] is True
    assert operator_robots['robot-a']['session']['id'] == ALICE_SESSION_ID
    assert operator_robots['robot-a']['session']['principal_id'] == 'operator:alice'
    assert 'id' not in operator_robots['robot-b']['session']
    assert 'principal_id' not in operator_robots['robot-b']['session']
    assert operator_robots['robot-b']['session']['owned_by_client'] is False
    assert owner_robots['robot-a']['session']['id'] == ALICE_SESSION_ID
    assert owner_robots['robot-b']['session']['id'] == BOB_SESSION_ID


def test_directory_rejects_driver_without_lifecycle_stop_parameters(
    api_harness: _ApiHarness,
):
    services = deepcopy(config.main['services'])
    record = next(item for item in services['mcp'] if item['id'] == 'robot-idle')
    record['tools'][0]['inputSchema']['x-action-params'].pop('stop')
    config.main['services'] = services
    mcp_client.registry['robot-idle']['teleop_fingerprint'] = (
        mcp_client.teleop_tool_fingerprint(record['tools'][0])
    )

    directory = api_harness.request('GET', '/api/teleop/robots', role='viewer')
    robot = next(
        item for item in directory.json()['data'] if item['id'] == 'robot-idle'
    )
    assert robot['descriptor_valid'] is False
    assert robot['teleop_ready'] is False
    assert robot['reason'] == 'teleop_descriptor_invalid'


def test_restart_guard_is_visible_but_only_owner_can_reconcile(
    api_harness: _ApiHarness,
):
    api_harness.coordinator.authority_guards['robot-idle'] = {
        'state': 'recovery_required',
        'phase': 'recovery_required',
        'driver_id': 'robot-idle',
        'robot_id': 'robot-idle',
        'retryable': True,
        'created_at': 1.0,
        'updated_at': 2.0,
    }

    directory = api_harness.request('GET', '/api/teleop/robots', role='viewer')
    guarded = next(
        item for item in directory.json()['data'] if item['id'] == 'robot-idle'
    )
    assert guarded['teleop_ready'] is False
    assert guarded['reason'] == 'authority_recovery_required'
    assert guarded['session'] == {
        'busy': True,
        'owned_by_me': False,
        'owned_by_client': False,
        'state': 'recovery_required',
    }
    assert guarded['authority_guard'] == {
        'state': 'recovery_required',
        'phase': 'recovery_required',
        'driver_id': 'robot-idle',
        'robot_id': 'robot-idle',
        'retryable': True,
        'created_at': 1.0,
        'updated_at': 2.0,
    }

    path = '/api/teleop/authority-guards/robot-idle/reconcile'
    for role in ('viewer', 'operator'):
        denied = api_harness.request('POST', path, role=role)
        assert denied.status_code == 403
    cleared = api_harness.request('POST', path, role='owner')
    assert cleared.status_code == 200
    assert cleared.json()['data']['state'] == 'clear'
    assert cleared.json()['data']['old_session_restored'] is False
    assert api_harness.coordinator.reconcile_calls == [
        ('robot-idle', 'owner:legacy'),
    ]

    after = api_harness.request('GET', '/api/teleop/robots', role='viewer')
    idle = next(item for item in after.json()['data'] if item['id'] == 'robot-idle')
    assert idle['teleop_ready'] is True
    assert idle['session']['busy'] is False


def test_orphan_restart_guard_stays_visible_when_driver_record_is_missing(
    api_harness: _ApiHarness,
):
    api_harness.coordinator.authority_guards['robot-orphan'] = {
        'state': 'recovery_required',
        'phase': 'recovery_required',
        'driver_id': 'missing-teleop-driver',
        'robot_id': 'robot-orphan',
        'retryable': True,
        'created_at': 3.0,
        'updated_at': 4.0,
    }

    directory = api_harness.request('GET', '/api/teleop/robots', role='viewer')
    assert directory.status_code == 200
    orphan = next(
        item for item in directory.json()['data']
        if item['id'] == 'recovery:robot-orphan'
    )
    assert orphan['driver_id'] == 'missing-teleop-driver'
    assert orphan['robot_id'] == 'robot-orphan'
    assert orphan['online'] is False
    assert orphan['teleop_ready'] is False
    assert orphan['reason'] == 'authority_recovery_required'
    assert orphan['session'] == {
        'busy': True,
        'owned_by_me': False,
        'owned_by_client': False,
        'state': 'recovery_required',
    }
    assert set(orphan['authority_guard']) == {
        'state',
        'phase',
        'driver_id',
        'robot_id',
        'retryable',
        'created_at',
        'updated_at',
    }

    reconcile = api_harness.request(
        'POST',
        '/api/teleop/authority-guards/robot-orphan/reconcile',
        role='owner',
    )
    assert reconcile.status_code == 200
    assert reconcile.json()['data']['old_session_restored'] is False


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'driver_id': ''},
        {'driver_id': 'contains a space'},
        {'driver_id': 'robot-a', 'extra': 'not-accepted'},
    ],
)
def test_acquire_body_accepts_only_a_valid_driver_id(api_harness: _ApiHarness, body):
    response = api_harness.request('POST', '/api/teleop/sessions', json=body)
    assert response.status_code == 422


def test_viewer_cannot_create_a_session(api_harness: _ApiHarness):
    response = api_harness.request(
        'POST',
        '/api/teleop/sessions',
        role='viewer',
        json={'driver_id': 'robot-new'},
    )
    assert response.status_code == 403
    assert response.json() == {'detail': 'operator role required'}


@pytest.mark.parametrize(
    ('driver_id', 'expected_status', 'disposition'),
    [
        ('robot-new', 201, 'created'),
        ('robot-existing', 200, 'existing'),
        ('robot-preparing', 202, 'preparing'),
    ],
)
def test_acquire_uses_http_and_json_envelopes_for_each_disposition(
    api_harness: _ApiHarness,
    driver_id: str,
    expected_status: int,
    disposition: str,
):
    response = api_harness.request(
        'POST',
        '/api/teleop/sessions',
        json={'driver_id': driver_id},
    )
    body = response.json()
    assert response.status_code == expected_status
    assert body['code'] == expected_status
    assert body['data']['disposition'] == disposition
    assert body['data']['session']['driver_id'] == driver_id
    assert body['data']['session']['mode'] == 'shadow'


def test_live_acquire_requires_confirmation_before_becoming_active(
    api_harness: _ApiHarness,
):
    response = api_harness.request(
        'POST',
        '/api/teleop/sessions',
        json={'driver_id': 'robot-new', 'mode': 'live'},
    )

    assert response.status_code == 202
    assert response.json()['data']['disposition'] == 'confirmation_required'
    session = response.json()['data']['session']
    assert session['mode'] == 'live'
    assert session['state'] == 'awaiting_confirmation'
    assert session['live_confirmed'] is False
    assert session['driver']['actuation_enabled'] is True


def test_confirm_live_route_binds_profile_and_is_idempotent_for_same_tab(
    api_harness: _ApiHarness,
):
    session = api_harness.coordinator.alice
    api_harness.coordinator._configure_live(session)
    session.state = 'awaiting_confirmation'
    body = {
        'confirm_live_actuation': True,
        'profile_id': 'dual_arm_profile_v1',
    }
    path = f'/api/teleop/sessions/{session.id}/confirm-live'

    confirmed = api_harness.request('POST', path, json=body)
    retried = api_harness.request('POST', path, json=body)

    assert confirmed.status_code == 200
    assert confirmed.json()['data']['disposition'] == 'created'
    assert confirmed.json()['data']['session']['mode'] == 'live'
    assert confirmed.json()['data']['session']['live_confirmed'] is True
    assert retried.status_code == 200
    assert retried.json()['data']['disposition'] == 'existing'
    assert api_harness.coordinator.live_prepare_count == 1
    assert api_harness.coordinator.confirm_calls == [
        (session.id, 'operator:alice', CLIENT_ID, 'dual_arm_profile_v1'),
        (session.id, 'operator:alice', CLIENT_ID, 'dual_arm_profile_v1'),
    ]


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'confirm_live_actuation': False, 'profile_id': 'dual_arm_profile_v1'},
        {'confirm_live_actuation': 1, 'profile_id': 'dual_arm_profile_v1'},
        {'confirm_live_actuation': True},
        {'confirm_live_actuation': True, 'profile_id': 'contains a space'},
        {
            'confirm_live_actuation': True,
            'profile_id': 'dual_arm_profile_v1',
            'extra': 'rejected',
        },
    ],
)
def test_confirm_live_body_is_strict_and_never_reaches_coordinator(
    api_harness: _ApiHarness,
    body,
):
    session = api_harness.coordinator.alice
    api_harness.coordinator._configure_live(session)
    session.state = 'awaiting_confirmation'

    response = api_harness.request(
        'POST',
        f'/api/teleop/sessions/{session.id}/confirm-live',
        json=body,
    )

    assert response.status_code == 422
    assert api_harness.coordinator.confirm_calls == []
    assert api_harness.coordinator.live_prepare_count == 0


def test_confirm_live_rejects_viewer_owner_other_tab_and_wrong_profile(
    api_harness: _ApiHarness,
):
    session = api_harness.coordinator.alice
    api_harness.coordinator._configure_live(session)
    session.state = 'awaiting_confirmation'
    path = f'/api/teleop/sessions/{session.id}/confirm-live'
    body = {
        'confirm_live_actuation': True,
        'profile_id': 'dual_arm_profile_v1',
    }

    viewer = api_harness.request('POST', path, role='viewer', json=body)
    owner = api_harness.request('POST', path, role='owner', json=body)
    other_tab = api_harness.client.post(
        path,
        headers={
            'Authorization': 'Bearer operator-token',
            'X-Motus-Teleop-Client': OTHER_CLIENT_ID,
        },
        json=body,
    )
    _assert_secret_free(other_tab)
    wrong_profile = api_harness.request(
        'POST',
        path,
        json={**body, 'profile_id': 'different_profile'},
    )

    assert viewer.status_code == 403
    assert owner.status_code == 403
    assert owner.json() == {'detail': {'code': 'session_forbidden'}}
    assert other_tab.status_code == 403
    assert other_tab.json() == {'detail': {'code': 'session_client_mismatch'}}
    assert wrong_profile.status_code == 409
    assert wrong_profile.json() == {
        'detail': {'code': 'live_confirmation_mismatch'},
    }
    assert api_harness.coordinator.live_prepare_count == 0


@pytest.mark.parametrize(
    ('driver_id', 'expected_status', 'expected_code'),
    [
        ('robot-busy', 409, 'robot_busy'),
        ('robot-not-ready', 503, 'driver_not_ready'),
        ('robot-timeout', 504, 'driver_timeout'),
        ('robot-protocol-error', 502, 'driver_protocol_error'),
    ],
)
def test_acquire_maps_conflicts_and_service_errors_to_stable_codes(
    api_harness: _ApiHarness,
    driver_id: str,
    expected_status: int,
    expected_code: str,
):
    response = api_harness.request(
        'POST',
        '/api/teleop/sessions',
        json={'driver_id': driver_id},
    )
    assert response.status_code == expected_status
    assert response.json() == {'detail': {'code': expected_code}}


def test_operator_lists_only_own_sessions_while_owner_lists_all(
    api_harness: _ApiHarness,
):
    operator = api_harness.request('GET', '/api/teleop/sessions')
    owner = api_harness.request('GET', '/api/teleop/sessions', role='owner')

    assert operator.status_code == 200
    assert [item['id'] for item in operator.json()['data']] == [ALICE_SESSION_ID]
    assert {item['id'] for item in owner.json()['data']} == {
        ALICE_SESSION_ID,
        BOB_SESSION_ID,
    }

    forbidden = api_harness.request(
        'GET',
        f'/api/teleop/sessions/{BOB_SESSION_ID}',
    )
    owner_read = api_harness.request(
        'GET',
        f'/api/teleop/sessions/{BOB_SESSION_ID}',
        role='owner',
    )
    assert forbidden.status_code == 403
    assert forbidden.json() == {'detail': {'code': 'session_forbidden'}}
    assert owner_read.status_code == 200
    assert owner_read.json()['data']['id'] == BOB_SESSION_ID


def test_owner_cannot_renew_another_principals_browser_lease(
    api_harness: _ApiHarness,
):
    original_deadline = api_harness.coordinator.bob.deadline_monotonic
    response = api_harness.request(
        'POST',
        f'/api/teleop/sessions/{BOB_SESSION_ID}/heartbeat',
        role='owner',
    )
    assert response.status_code == 403
    assert response.json() == {'detail': {'code': 'session_forbidden'}}
    assert api_harness.coordinator.bob.deadline_monotonic == original_deadline
    assert api_harness.coordinator.heartbeat_calls == []

    own = api_harness.request(
        'POST',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/heartbeat',
    )
    assert own.status_code == 200
    assert own.json()['code'] == 200
    assert api_harness.coordinator.heartbeat_calls == [
        (ALICE_SESSION_ID, 'operator:alice'),
    ]


def test_signaling_offer_returns_only_sdp_answer_for_original_tab(
    api_harness: _ApiHarness,
):
    deadline = api_harness.coordinator.alice.deadline_monotonic
    offer = {'type': 'offer', 'sdp': 'v=0\r\no=quest-browser'}
    response = api_harness.request(
        'POST',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/signaling/offer',
        json=offer,
    )
    assert response.status_code == 200
    assert response.json() == {
        'code': 200,
        'data': {'type': 'answer', 'sdp': 'v=0\r\na=fake-driver-answer'},
    }
    assert response.headers['cache-control'] == 'no-store'
    assert api_harness.coordinator.signaling_calls == [(
        ALICE_SESSION_ID,
        'operator:alice',
        CLIENT_ID,
        offer,
    )]
    assert api_harness.coordinator.alice.deadline_monotonic == deadline


def test_owner_and_other_browser_tab_cannot_take_over_signaling(
    api_harness: _ApiHarness,
):
    path = f'/api/teleop/sessions/{ALICE_SESSION_ID}/signaling/offer'
    offer = {'type': 'offer', 'sdp': 'v=0'}
    viewer = api_harness.request('POST', path, role='viewer', json=offer)
    assert viewer.status_code == 403
    owner = api_harness.request('POST', path, role='owner', json=offer)
    assert owner.status_code == 403
    assert owner.json() == {'detail': {'code': 'session_forbidden'}}

    other_tab_headers = {
        'Authorization': 'Bearer operator-token',
        'X-Motus-Teleop-Client': OTHER_CLIENT_ID,
    }
    other_tab = api_harness.client.post(path, headers=other_tab_headers, json=offer)
    _assert_secret_free(other_tab)
    assert other_tab.status_code == 403
    assert other_tab.json() == {'detail': {'code': 'session_client_mismatch'}}
    assert api_harness.coordinator.signaling_calls == []


@pytest.mark.parametrize(
    ('content', 'content_type', 'expected_status', 'expected_code'),
    [
        (
            b'{"type":"offer","sdp":"v=0","extra":true}',
            'application/json',
            400,
            'invalid_signaling_offer',
        ),
        (
            b'{"type":"answer","sdp":"v=0"}',
            'application/json',
            400,
            'invalid_signaling_offer',
        ),
        (
            b'{"type":"offer","type":"offer","sdp":"v=0"}',
            'application/json',
            400,
            'invalid_signaling_offer',
        ),
        (
            b'{"type":"offer","sdp":NaN}',
            'application/json',
            400,
            'invalid_signaling_offer',
        ),
        (
            b'{"type":"offer","sdp":"v=0"}',
            'text/plain',
            415,
            'signaling_content_type_required',
        ),
    ],
)
def test_signaling_offer_body_is_strict_and_stably_rejected(
    api_harness: _ApiHarness,
    content,
    content_type,
    expected_status,
    expected_code,
):
    response = api_harness.client.request(
        'POST',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/signaling/offer',
        content=content,
        headers={**api_harness.headers['operator'], 'Content-Type': content_type},
    )
    _assert_secret_free(response)
    assert response.status_code == expected_status
    assert response.json() == {'detail': {'code': expected_code}}
    assert api_harness.coordinator.signaling_calls == []


def test_signaling_offer_raw_body_has_a_hard_byte_limit(api_harness: _ApiHarness):
    response = api_harness.client.request(
        'POST',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/signaling/offer',
        content=b'{"type":"offer","sdp":"' + b'x' * (129 * 1024) + b'"}',
        headers={**api_harness.headers['operator'], 'Content-Type': 'application/json'},
    )
    _assert_secret_free(response)
    assert response.status_code == 413
    assert response.json() == {
        'detail': {'code': 'signaling_offer_too_large'},
    }
    assert api_harness.coordinator.signaling_calls == []


@pytest.mark.parametrize(
    ('path_suffix', 'expected_state'),
    [
        ('pause', 'paused'),
        ('soft-stop', 'hold'),
    ],
)
def test_pause_and_soft_stop_return_visible_states(
    api_harness: _ApiHarness,
    path_suffix: str,
    expected_state: str,
):
    response = api_harness.request(
        'POST',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/{path_suffix}',
    )
    assert response.status_code == 200
    assert response.json()['code'] == 200
    assert response.json()['data']['state'] == expected_state


def test_release_is_idempotent_and_terminal_control_routes_return_410(
    api_harness: _ApiHarness,
):
    path = f'/api/teleop/sessions/{ALICE_SESSION_ID}'
    first = api_harness.request('DELETE', path)
    repeated = api_harness.request('DELETE', path)

    assert first.status_code == 200
    assert first.json()['data']['session']['state'] == 'released'
    assert first.json()['data']['driver_acknowledged'] is True
    assert repeated.status_code == 200
    assert repeated.json()['code'] == 200
    assert repeated.json()['data']['session']['id'] == ALICE_SESSION_ID
    assert repeated.json()['data']['session']['state'] == 'released'
    assert repeated.json()['data']['driver_acknowledged'] is True

    # The historical read remains available, while every route that requires
    # live authority reports the stable 410 terminal code.
    history = api_harness.request('GET', path)
    assert history.status_code == 200
    assert history.json()['data']['state'] == 'released'

    terminal_requests = [
        ('GET', f'{path}/driver-status'),
        ('POST', f'{path}/heartbeat'),
        ('POST', f'{path}/pause'),
        ('POST', f'{path}/soft-stop'),
    ]
    for method, terminal_path in terminal_requests:
        response = api_harness.request(method, terminal_path)
        assert response.status_code == 410
        assert response.json() == {'detail': {'code': 'session_released'}}
    signaling = api_harness.request(
        'POST',
        f'{path}/signaling/offer',
        json={'type': 'offer', 'sdp': 'v=0'},
    )
    assert signaling.status_code == 410
    assert signaling.json() == {'detail': {'code': 'session_released'}}


def test_unknown_and_invalid_state_errors_have_stable_envelopes(
    api_harness: _ApiHarness,
):
    missing = api_harness.request(
        'GET',
        '/api/teleop/sessions/00000000-0000-0000-0000-000000000000',
    )
    assert missing.status_code == 404
    assert missing.json() == {'detail': {'code': 'session_not_found'}}

    api_harness.coordinator.alice.state = 'paused'
    conflict = api_harness.request(
        'POST',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/soft-stop',
    )
    assert conflict.status_code == 409
    assert conflict.json() == {'detail': {'code': 'session_state_conflict'}}


def test_get_routes_never_renew_browser_lease(api_harness: _ApiHarness):
    deadline = api_harness.coordinator.alice.deadline_monotonic
    calls = list(api_harness.coordinator.heartbeat_calls)
    paths = [
        '/api/teleop/robots',
        '/api/teleop/sessions',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/driver-status',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/events',
    ]
    for path in paths:
        response = api_harness.request('GET', path)
        assert response.status_code == 200
    assert api_harness.coordinator.alice.deadline_monotonic == deadline
    assert api_harness.coordinator.heartbeat_calls == calls


def test_session_events_are_authorized_and_isolated_by_exact_session(
    api_harness: _ApiHarness,
):
    async def seed_events():
        await audit.emit(
            'teleop.session.alice',
            session_id=ALICE_SESSION_ID,
            robot_id='robot-a',
            principal_id='operator:alice',
            details={
                'safe': 'alice-event',
                'fence': FENCE_BY_SESSION[ALICE_SESSION_ID],
                'driver_token': DRIVER_TOKEN,
            },
        )
        await audit.emit(
            'teleop.session.same-robot-other-session',
            session_id='80f42326-4555-4481-92b1-dba426d2f4c4',
            robot_id='robot-a',
            principal_id='operator:alice',
        )
        await audit.emit(
            'teleop.session.bob',
            session_id=BOB_SESSION_ID,
            robot_id='robot-b',
            principal_id='operator:bob',
        )

    asyncio.run(seed_events())

    alice = api_harness.request(
        'GET',
        f'/api/teleop/sessions/{ALICE_SESSION_ID}/events?limit=200',
    )
    assert alice.status_code == 200
    assert [event['event_type'] for event in alice.json()['data']] == [
        'teleop.session.alice',
    ]
    assert alice.json()['data'][0]['details'] == {'safe': 'alice-event'}

    forbidden = api_harness.request(
        'GET',
        f'/api/teleop/sessions/{BOB_SESSION_ID}/events',
    )
    owner = api_harness.request(
        'GET',
        f'/api/teleop/sessions/{BOB_SESSION_ID}/events',
        role='owner',
    )
    assert forbidden.status_code == 403
    assert forbidden.json() == {'detail': {'code': 'session_forbidden'}}
    assert [event['event_type'] for event in owner.json()['data']] == [
        'teleop.session.bob',
    ]


def test_every_session_route_returns_json_without_private_authority(
    api_harness: _ApiHarness,
):
    # This is an explicit response-surface inventory in addition to the scan
    # automatically performed by every request made through the harness.
    requests = [
        ('GET', '/api/teleop/me', 'operator', None),
        ('GET', '/api/teleop/robots', 'viewer', None),
        ('GET', '/api/teleop/sessions', 'operator', None),
        ('GET', f'/api/teleop/sessions/{ALICE_SESSION_ID}', 'operator', None),
        (
            'GET',
            f'/api/teleop/sessions/{ALICE_SESSION_ID}/driver-status',
            'operator',
            None,
        ),
        ('GET', f'/api/teleop/sessions/{ALICE_SESSION_ID}/events', 'operator', None),
        ('POST', f'/api/teleop/sessions/{ALICE_SESSION_ID}/heartbeat', 'operator', None),
        ('POST', f'/api/teleop/sessions/{ALICE_SESSION_ID}/pause', 'operator', None),
        (
            'POST',
            f'/api/teleop/sessions/{ALICE_SESSION_ID}/signaling/offer',
            'operator',
            {'type': 'offer', 'sdp': 'v=0'},
        ),
        ('DELETE', f'/api/teleop/sessions/{ALICE_SESSION_ID}', 'operator', None),
        ('POST', '/api/teleop/sessions', 'operator', {'driver_id': 'robot-timeout'}),
        ('POST', '/api/teleop/sessions', 'operator', {'driver_id': 'bad id'}),
    ]
    for method, path, role, body in requests:
        kwargs = {'json': body} if body is not None else {}
        response = api_harness.request(method, path, role=role, **kwargs)
        assert response.headers['content-type'].startswith('application/json')
