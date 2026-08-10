from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from teleop.capture_manager import MAX_CAPTURE_RECORDS, CaptureError, CaptureManager

PRINCIPAL = 'operator:alice'
CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'
OTHER_CLIENT_ID = '68991413-d37a-4603-9f07-3e25219d6d96'
SESSION_ID = 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3'
DIGEST = '0123456789abcdef' * 4
CLIENT_METADATA = {
    'capture_protocol': 'motus.teleop.capture.v1',
    'frame_protocol': 'motus.teleop.rtc-frame.v1',
    'client_kind': 'browser_webxr',
    'app_version': '1.0.0-test',
}


@dataclass
class _Clock:
    monotonic: float = 10.0
    wall: float = 1_700_000_000.0

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall += seconds


async def _connected_capture(
    manager: CaptureManager,
    *,
    label: str = 'Quest 3',
):
    pairing = await manager.create_pairing(PRINCIPAL, CLIENT_ID, label=label)
    connection = await manager.connect_with_pairing(
        pairing.pairing_id,
        pairing.pairing_code,
        **CLIENT_METADATA,
    )
    return pairing, connection


async def _pairing_is_single_use_bounded_and_secrets_never_enter_repr():
    clock = _Clock()
    manager = CaptureManager(monotonic=lambda: clock.monotonic, wall_clock=lambda: clock.wall)

    pairing = await manager.create_pairing(PRINCIPAL, CLIENT_ID, label=' Quest 3 ')
    assert pairing.expires_at == clock.wall + 60.0
    assert pairing.pairing_code not in repr(pairing)

    connection = await manager.connect_with_pairing(
        pairing.pairing_id,
        pairing.pairing_code,
        **CLIENT_METADATA,
    )
    assert connection.capture_credential is not None
    assert connection.capture_credential not in repr(connection)
    with pytest.raises(CaptureError, match='capture_pairing_invalid'):
        await manager.connect_with_pairing(
            pairing.pairing_id,
            pairing.pairing_code,
            **CLIENT_METADATA,
        )

    captures = await manager.list_for_supervisor(PRINCIPAL)
    assert captures == [{
        'id': connection.capture_id,
        'label': 'Quest 3',
        **CLIENT_METADATA,
        'connected': True,
        'observed_state': 'browser_ready',
        'created_at': clock.wall,
        'connected_at': clock.wall,
        'last_seen_at': clock.wall,
        'assignment': None,
    }]
    serialized = repr(captures)
    assert CLIENT_ID not in serialized
    assert pairing.pairing_code not in serialized
    assert connection.capture_credential not in serialized


async def _expired_pairing_and_wrong_supervisor_fail_closed():
    clock = _Clock()
    manager = CaptureManager(monotonic=lambda: clock.monotonic, wall_clock=lambda: clock.wall)
    pairing = await manager.create_pairing(PRINCIPAL, CLIENT_ID, label='Quest 3')
    clock.advance(60.0)

    with pytest.raises(CaptureError, match='capture_pairing_invalid'):
        await manager.connect_with_pairing(
            pairing.pairing_id,
            pairing.pairing_code,
            **CLIENT_METADATA,
        )

    _, connection = await _connected_capture(manager)
    assert len(await manager.list_for_supervisor(PRINCIPAL)) == 1
    with pytest.raises(CaptureError, match='capture_forbidden'):
        await manager.revoke_capture(connection.capture_id, 'operator:bob')


async def _pairing_limit_is_principal_scoped_across_refreshed_clients():
    manager = CaptureManager()
    for index in range(4):
        await manager.create_pairing(
            PRINCIPAL,
            f'00000000-0000-4000-8000-{index:012d}',
            label=f'Quest {index}',
        )
    with pytest.raises(CaptureError, match='capture_pairing_limit'):
        await manager.create_pairing(
            PRINCIPAL,
            OTHER_CLIENT_ID,
            label='Quest after refresh',
        )

    # Another principal retains an independent bounded enrollment window.
    await manager.create_pairing('operator:bob', OTHER_CLIENT_ID, label='Bob Quest')


async def _operator_revoke_deletes_tombstone_and_releases_global_capacity():
    manager = CaptureManager()
    first_capture_id = ''
    first_credential = ''
    for index in range(MAX_CAPTURE_RECORDS + 1):
        pairing = await manager.create_pairing(
            PRINCIPAL,
            CLIENT_ID,
            label=f'Quest {index}',
        )
        connection = await manager.connect_with_pairing(
            pairing.pairing_id,
            pairing.pairing_code,
            **CLIENT_METADATA,
        )
        if index == 0:
            first_capture_id = connection.capture_id
            first_credential = connection.capture_credential or ''
        revoked = await manager.revoke_capture(connection.capture_id, PRINCIPAL)
        assert revoked['revoked'] is True
        assert await connection.events.get() == {
            'type': 'capture_revoked',
            'reason': 'operator_revoked',
        }

    assert await manager.list_for_supervisor(PRINCIPAL) == []
    with pytest.raises(CaptureError, match='capture_auth_invalid'):
        await manager.connect_with_credential(
            first_capture_id,
            first_credential,
            **CLIENT_METADATA,
        )

    # A different principal can still enroll after more than the former global
    # tombstone limit worth of successful pair+revoke cycles.
    pairing = await manager.create_pairing(
        'operator:bob',
        OTHER_CLIENT_ID,
        label='Bob Quest',
    )
    connection = await manager.connect_with_pairing(
        pairing.pairing_id,
        pairing.pairing_code,
        **CLIENT_METADATA,
    )
    assert connection.capture_id


async def _attach_requires_xr_standby_and_binds_exact_session_contract():
    manager = CaptureManager()
    _, connection = await _connected_capture(manager)

    with pytest.raises(CaptureError, match='capture_not_ready'):
        await manager.attach(
            capture_id=connection.capture_id,
            principal_id=PRINCIPAL,
            session_id=SESSION_ID,
            operation_generation=3,
            mode='live',
            profile_id='dual_arm_v1',
            capability_digest=DIGEST,
            capabilities={'profile_id': 'dual_arm_v1'},
            effectors=['dual_arm'],
        )

    await manager.update_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_standby',
        assignment_id=None,
    )
    assignment = await manager.attach(
        capture_id=connection.capture_id,
        principal_id=PRINCIPAL,
        session_id=SESSION_ID,
        operation_generation=3,
        mode='live',
        profile_id='dual_arm_v1',
        capability_digest=DIGEST,
        capabilities={'profile_id': 'dual_arm_v1'},
        effectors=['dual_arm'],
    )
    pushed = connection.events.get_nowait()
    assert pushed['type'] == 'assignment'
    assert pushed['assignment']['id'] == assignment.id
    assert pushed['assignment']['mode'] == 'live'
    assert 'operation_generation' not in pushed['assignment']

    duplicate = await manager.attach(
        capture_id=connection.capture_id,
        principal_id=PRINCIPAL,
        session_id=SESSION_ID,
        operation_generation=3,
        mode='live',
        profile_id='dual_arm_v1',
        capability_digest=DIGEST,
        capabilities={'profile_id': 'dual_arm_v1'},
        effectors=['dual_arm'],
    )
    assert duplicate.id == assignment.id
    assert connection.events.empty()


async def _offer_failure_creates_loss_fence_instead_of_same_session_retry():
    manager = CaptureManager()
    _, connection = await _connected_capture(manager)
    await manager.update_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_standby',
        assignment_id=None,
    )
    values = {
        'capture_id': connection.capture_id,
        'principal_id': PRINCIPAL,
        'session_id': SESSION_ID,
        'operation_generation': 3,
        'mode': 'shadow',
        'profile_id': 'recording',
        'capability_digest': DIGEST,
        'capabilities': {'profile_id': 'recording'},
        'effectors': ['dual_arm'],
    }
    assignment = await manager.attach(**values)
    connection.events.get_nowait()

    claimed = await manager.claim_offer(
        connection.capture_id,
        connection.connection_id,
        assignment.id,
    )
    assert claimed.state == 'offer_consumed'
    with pytest.raises(CaptureError, match='capture_offer_already_consumed'):
        await manager.claim_offer(
            connection.capture_id,
            connection.connection_id,
            assignment.id,
        )

    lost = await manager.fail_offer(connection.capture_id, assignment.id)
    assert lost is not None
    assert lost.id == assignment.id
    assert lost.failure_code == 'capture_signaling_failed'
    revoked = connection.events.get_nowait()
    assert revoked == {
        'type': 'assignment_revoked',
        'assignment_id': assignment.id,
        'reason': 'capture_signaling_failed',
    }
    assert await manager.assignment_loss_is_pending(lost) is True
    with pytest.raises(CaptureError, match='capture_loss_pending'):
        await manager.attach(**values)
    assert await manager.complete_assignment_loss(lost) is True


async def _disconnect_and_session_terminalization_revoke_assignment():
    manager = CaptureManager()
    _, connection = await _connected_capture(manager)
    await manager.update_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_standby',
        assignment_id=None,
    )
    assignment = await manager.attach(
        capture_id=connection.capture_id,
        principal_id=PRINCIPAL,
        session_id=SESSION_ID,
        operation_generation=3,
        mode='shadow',
        profile_id='recording',
        capability_digest=DIGEST,
        capabilities={'profile_id': 'recording'},
        effectors=[],
    )
    connection.events.get_nowait()

    await manager.revoke_for_session(SESSION_ID, 'operator_pause')
    event = connection.events.get_nowait()
    assert event == {
        'type': 'assignment_revoked',
        'assignment_id': assignment.id,
        'reason': 'operator_pause',
    }
    captures = await manager.list_for_supervisor(PRINCIPAL)
    assert captures[0]['assignment'] is None

    await manager.disconnect(connection.capture_id, connection.connection_id)
    captures = await manager.list_for_supervisor(PRINCIPAL)
    assert captures[0]['connected'] is False
    assert captures[0]['observed_state'] == 'offline'


async def _reset_drops_all_ephemeral_capture_authority():
    manager = CaptureManager()
    pairing, connection = await _connected_capture(manager)
    credential = connection.capture_credential
    await manager.reset()

    assert await manager.list_for_supervisor(PRINCIPAL) == []
    with pytest.raises(CaptureError, match='capture_auth_invalid'):
        await manager.connect_with_credential(
            connection.capture_id,
            credential or '',
            **CLIENT_METADATA,
        )
    with pytest.raises(CaptureError, match='capture_pairing_invalid'):
        await manager.connect_with_pairing(
            pairing.pairing_id,
            pairing.pairing_code,
            **CLIENT_METADATA,
        )
    # The reset notification is queued before the in-memory connection owner is dropped.
    assert await asyncio.wait_for(connection.events.get(), timeout=0.1) == {
        'type': 'capture_revoked',
        'reason': 'core_stopped',
    }


async def _terminal_presence_and_stale_heartbeat_revoke_assignment():
    clock = _Clock()
    manager = CaptureManager(monotonic=lambda: clock.monotonic, wall_clock=lambda: clock.wall)
    _, connection = await _connected_capture(manager)
    await manager.update_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_standby',
        assignment_id=None,
    )
    assignment = await manager.attach(
        capture_id=connection.capture_id,
        principal_id=PRINCIPAL,
        session_id=SESSION_ID,
        operation_generation=3,
        mode='shadow',
        profile_id='recording',
        capability_digest=DIGEST,
        capabilities={'profile_id': 'recording'},
        effectors=[],
    )
    connection.events.get_nowait()

    _presence, ended_loss = await manager.update_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_ended',
        assignment_id=None,
    )
    assert ended_loss is not None
    assert connection.events.get_nowait() == {
        'type': 'assignment_revoked',
        'assignment_id': assignment.id,
        'reason': 'capture_xr_ended',
    }
    assert (await manager.list_for_supervisor(PRINCIPAL))[0]['assignment'] is None
    with pytest.raises(CaptureError, match='capture_loss_pending'):
        await manager.attach(
            capture_id=connection.capture_id,
            principal_id=PRINCIPAL,
            session_id=SESSION_ID,
            operation_generation=3,
            mode='shadow',
            profile_id='recording',
            capability_digest=DIGEST,
            capabilities={'profile_id': 'recording'},
            effectors=[],
        )
    assert await manager.complete_assignment_loss(ended_loss) is True

    await manager.update_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_standby',
        assignment_id=None,
    )
    replacement = await manager.attach(
        capture_id=connection.capture_id,
        principal_id=PRINCIPAL,
        session_id=SESSION_ID,
        operation_generation=3,
        mode='shadow',
        profile_id='recording',
        capability_digest=DIGEST,
        capabilities={'profile_id': 'recording'},
        effectors=[],
    )
    connection.events.get_nowait()
    clock.advance(5.0)
    stale, lost_assignment = await manager.expire_stale_connection(
        connection.capture_id,
        connection.connection_id,
    )
    assert stale is True
    assert lost_assignment is not None
    assert lost_assignment.id == replacement.id
    assert lost_assignment.failure_code == 'capture_presence_timeout'
    assert await manager.assignment_loss_is_pending(lost_assignment) is True
    assert connection.events.get_nowait() == {
        'type': 'assignment_revoked',
        'assignment_id': replacement.id,
        'reason': 'capture_presence_timeout',
    }
    assert connection.events.get_nowait() == {
        'type': 'capture_stale',
        'reason': 'capture_presence_timeout',
    }
    public = (await manager.list_for_supervisor(PRINCIPAL))[0]
    assert public['connected'] is False
    assert public['observed_state'] == 'offline'
    assert public['assignment'] is None
    assert await manager.complete_assignment_loss(lost_assignment) is True


def test_pairing_is_single_use_bounded_and_secrets_never_enter_repr():
    asyncio.run(_pairing_is_single_use_bounded_and_secrets_never_enter_repr())


def test_expired_pairing_and_wrong_supervisor_fail_closed():
    asyncio.run(_expired_pairing_and_wrong_supervisor_fail_closed())


def test_pairing_limit_is_principal_scoped_across_refreshed_clients():
    asyncio.run(_pairing_limit_is_principal_scoped_across_refreshed_clients())


def test_operator_revoke_deletes_tombstone_and_releases_global_capacity():
    asyncio.run(_operator_revoke_deletes_tombstone_and_releases_global_capacity())


def test_attach_requires_xr_standby_and_binds_exact_session_contract():
    asyncio.run(_attach_requires_xr_standby_and_binds_exact_session_contract())


def test_offer_failure_creates_loss_fence_instead_of_same_session_retry():
    asyncio.run(_offer_failure_creates_loss_fence_instead_of_same_session_retry())


def test_disconnect_and_session_terminalization_revoke_assignment():
    asyncio.run(_disconnect_and_session_terminalization_revoke_assignment())


def test_reset_drops_all_ephemeral_capture_authority():
    asyncio.run(_reset_drops_all_ephemeral_capture_authority())


def test_terminal_presence_and_stale_heartbeat_revoke_assignment():
    asyncio.run(_terminal_presence_and_stale_heartbeat_revoke_assignment())
