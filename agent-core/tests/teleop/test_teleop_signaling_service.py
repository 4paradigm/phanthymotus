from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
from copy import deepcopy

import pytest

import auth
import mcp_client
from teleop import audit
from teleop.capture_manager import CaptureError
from teleop.models import ShadowSession
from teleop.service import (
    TeleopCoordinator,
    TeleopServiceError,
    _project_driver_snapshot,
)
from teleop.session_manager import (
    SessionClientMismatch,
    SessionForbidden,
    SessionNotFound,
    SessionStateConflict,
    ShadowSessionManager,
)

SESSION_ID = 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3'
BOOT_ID = '72559c63-e2a7-46ed-a50f-e8128ed8aa2b'
CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'
OTHER_CLIENT_ID = '68991413-d37a-4603-9f07-3e25219d6d96'
PRINCIPAL_ID = 'operator:alice'
FENCE = 'private-fence-token-0123456789abcdef'
DIGEST = '0123456789abcdef' * 4
TICKET_SECRET = 'ticket-secret-that-is-at-least-32-bytes-long'
OFFER = {'type': 'offer', 'sdp': 'v=0\r\no=quest-3-offer'}
CAPTURE_METADATA = {
    'capture_protocol': 'motus.teleop.capture.v1',
    'frame_protocol': 'motus.teleop.rtc-frame.v1',
    'client_kind': 'browser_webxr',
    'app_version': '1.0.0-test',
}


class _Clock:
    def __init__(self):
        self.monotonic = 100.0
        self.wall = 1_800_000_000.75

    def monotonic_now(self) -> float:
        return self.monotonic

    def wall_now(self) -> float:
        return self.wall


class _OpenSession:
    closed = False


def _answer(**changes) -> dict:
    value = {
        'sdp': 'v=0\r\na=driver-answer',
        'type': 'answer',
        'boot_id': BOOT_ID,
        'session_id': SESSION_ID,
        'epoch': 7,
        'capability_digest': DIGEST,
        'mode': 'shadow',
        'actuation_enabled': False,
    }
    value.update(changes)
    return value


def _build(signaler, *, state: str = 'active', caller=None):
    clock = _Clock()
    manager = ShadowSessionManager(
        monotonic=clock.monotonic_now,
        wall_clock=clock.wall_now,
    )
    session = ShadowSession(
        id=SESSION_ID,
        robot_id='robot-a',
        driver_id='teleop-driver-a',
        principal_id=PRINCIPAL_ID,
        boot_id=BOOT_ID,
        epoch=7,
        capability_digest=DIGEST,
        client_id=CLIENT_ID,
        fence=FENCE,
        state=state,
        operation_generation=3,
        operation_state='succeeded',
        created_at=clock.wall - 1,
        lease_seconds=15.0,
        deadline_monotonic=clock.monotonic + 15.0,
    )
    manager._sessions[session.id] = session
    if state not in {'released', 'expired', 'faulted'}:
        manager._robot_sessions[session.robot_id] = session.id
    coordinator_kwargs = {
        'session_manager': manager,
        'signaler': signaler,
        'monotonic': clock.monotonic_now,
        'wall_clock': clock.wall_now,
    }
    if caller is not None:
        coordinator_kwargs['caller'] = caller
    coordinator = TeleopCoordinator(
        **coordinator_kwargs,
    )
    coordinator._http_session = _OpenSession()  # type: ignore[assignment]
    target = mcp_client.TrustedShadowTarget(
        mcp_id=session.driver_id,
        url='https://teleop-driver.invalid/mcp',
        capability_digest=DIGEST,
        descriptor_fingerprint='descriptor-fingerprint',
        actions=frozenset({'status'}),
    )
    coordinator._pinned_targets[session.id] = target
    return coordinator, manager, session, clock, target


def _recording_snapshot(
    *,
    state: str,
    reason: str | None,
    generation: int,
) -> dict:
    authority_valid = state in {'active_shadow', 'hold', 'paused'}
    terminal = state == 'released'
    dispatch_state = {
        'active_shadow': 'motion_eligible',
        'hold': 'safe_latched',
        'paused': 'safe_latched',
        'released': 'safe_revoked',
    }[state]
    decision = {
        'active_shadow': 'admitted',
        'hold': 'would_stop:soft_stop',
        'paused': 'would_stop:operator_pause',
        'released': 'would_stop:operator_release',
    }[state]
    return {
        'driver': 'teleop-driver-a',
        'driver_id': 'teleop-driver-a',
        'driver_name': 'Generic Teleop Shadow Diagnostics',
        'driver_type': 'teleop-shadow',
        'robot_id': 'robot-a',
        'mode': 'shadow',
        'actuation_enabled': False,
        'boot_id': BOOT_ID,
        'session_id': None if terminal else SESSION_ID,
        'epoch': 7,
        'state': state,
        'reason': reason,
        'authority_valid': authority_valid,
        'capability_digest': DIGEST,
        'capabilities': {
            'pose_transport': ['webrtc-datachannel'],
            'control_transport': ['mcp'],
        },
        'lease': {
            'source': 'agent-core-mcp-heartbeat-only',
            'timeout_ms': 10_000,
            'age_ms': 0.0 if authority_valid else None,
            'fresh': authority_valid,
            'authority_valid': authority_valid,
            'expired_latched': False,
        },
        'pose': {
            'timeout_ms': 250.0,
            'age_ms': None,
            'fresh': False,
            'latest_sequence': None,
            'latest': None,
        },
        'rtc': {
            'connected': False,
            'channels': {'teleop-control': False, 'teleop-pose': False},
            'renews_lease': False,
        },
        'dispatch': {
            'kind': 'recording',
            'state': dispatch_state,
            'ready': True,
            'generation': generation,
            'mailbox_depth': 0,
            'stop_queue_depth': 0,
            'last_admitted_sequence': 1,
            'last_would_apply_sequence': None,
            'last_decision': decision,
            'stop_acknowledged': True,
            'fault_code': None,
            'io_inflight': None,
            'counters': {'startup_safe_acks': 1, 'stop_acks': generation + 1},
            'adapter': {
                'kind': 'recording',
                'closed': False,
                'current': {'kind': 'safe', 'reason': 'not_started'},
                'records': [],
            },
        },
        'counters': {'lease_heartbeats': 1},
    }


class _CaptureLossCaller:
    def __init__(self, *, fail_soft_stop: bool = False):
        self.actions: list[str] = []
        self.generation = 1
        self.fail_soft_stop = fail_soft_stop

    async def __call__(
        self,
        _driver_id,
        action,
        _arguments,
        **_kwargs,
    ):
        self.actions.append(action)
        if action == 'soft_stop':
            if self.fail_soft_stop:
                raise mcp_client.TrustedShadowTransportError('network_error')
            self.generation += 1
            return _recording_snapshot(
                state='hold',
                reason='soft_stop',
                generation=self.generation,
            )
        raise AssertionError(f'unexpected capture-loss action: {action}')


def _seed_active_driver_snapshot(
    coordinator: TeleopCoordinator,
    session: ShadowSession,
) -> None:
    projected, _ = _project_driver_snapshot(
        _recording_snapshot(
            state='active_shadow',
            reason=None,
            generation=1,
        ),
        driver_id=session.driver_id,
        robot_id=session.robot_id,
        capability_digest=session.capability_digest,
        action='status',
        session=session,
    )
    coordinator._driver_snapshots[session.id] = projected
    coordinator._driver_snapshot_revisions[session.id] = 1


def _decode_ticket(ticket: str) -> tuple[dict, str, str]:
    payload, signature = ticket.split('.', 1)
    decoded = base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4))
    return json.loads(decoded), payload, signature


async def _ready_capture(coordinator: TeleopCoordinator, *, label: str = 'Quest 3'):
    pairing = await coordinator.create_capture_pairing(
        PRINCIPAL_ID,
        CLIENT_ID,
        label=label,
    )
    connection = await coordinator.connect_capture_with_pairing(
        pairing.pairing_id,
        pairing.pairing_code,
        **CAPTURE_METADATA,
    )
    await coordinator.capture_presence(
        connection.capture_id,
        connection.connection_id,
        state='xr_standby',
        assignment_id=None,
    )
    return connection


async def _attach_ready_capture(
    coordinator: TeleopCoordinator,
    *,
    label: str = 'Quest 3',
):
    connection = await _ready_capture(coordinator, label=label)
    assignment = await coordinator.attach_capture(
        SESSION_ID,
        PRINCIPAL_ID,
        CLIENT_ID,
        capture_id=connection.capture_id,
        mode='shadow',
        profile_id='recording',
        capability_digest=DIGEST,
    )
    assert (await connection.events.get())['assignment']['id'] == assignment.id
    return connection, assignment


def test_offer_ticket_is_driver_compatible_private_and_does_not_renew_lease(
    monkeypatch: pytest.MonkeyPatch,
):
    observed = {}
    events = []

    async def signaler(driver_id, offer, ticket, **kwargs):
        observed.update({
            'driver_id': driver_id,
            'offer': deepcopy(offer),
            'ticket': ticket,
            'kwargs': kwargs,
        })
        return _answer()

    async def capture(event_type, **fields):
        events.append({'event_type': event_type, **deepcopy(fields)})
        return events[-1]

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', capture)
    coordinator, _manager, session, clock, target = _build(signaler)
    deadline_before = session.deadline_monotonic

    result = asyncio.run(coordinator.signaling_offer(
        SESSION_ID,
        PRINCIPAL_ID,
        CLIENT_ID,
        OFFER,
    ))

    assert result == {'sdp': 'v=0\r\na=driver-answer', 'type': 'answer'}
    assert session.deadline_monotonic == deadline_before
    assert observed['driver_id'] == session.driver_id
    assert observed['offer'] == OFFER
    assert observed['kwargs'] == {
        'timeout_seconds': 2.0,
        'session': coordinator._http_session,
        'target': target,
    }

    claims, payload, signature = _decode_ticket(observed['ticket'])
    expected_signature = base64.urlsafe_b64encode(
        hmac.new(
            TICKET_SECRET.encode(),
            payload.encode('ascii'),
            hashlib.sha256,
        ).digest(),
    ).rstrip(b'=').decode('ascii')
    assert hmac.compare_digest(signature, expected_signature)
    assert claims == {
        'aud': 'teleop-shadow-rtc',
        'boot_id': BOOT_ID,
        'capability_digest': DIGEST,
        'epoch': 7,
        'exp': int(clock.wall) + 20,
        'fence': FENCE,
        'iat': int(clock.wall),
        'jti': claims['jti'],
        'sdp_sha256': hashlib.sha256(OFFER['sdp'].encode()).hexdigest(),
        'session_id': SESSION_ID,
        'v': 1,
    }
    assert re.fullmatch(r'[A-Za-z0-9_-]{16,128}', claims['jti'])
    public_surface = json.dumps({'result': result, 'events': events}, sort_keys=True)
    assert FENCE not in public_surface
    assert observed['ticket'] not in public_surface
    assert OFFER['sdp'] not in public_surface
    assert events[-1]['event_type'] == 'teleop.signaling.offer.accepted'


def test_offer_uses_ticket_secret_loaded_by_auth_when_process_env_is_absent(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_ticket = ''

    async def signaler(_driver_id, _offer, ticket, **_kwargs):
        nonlocal observed_ticket
        observed_ticket = ticket
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    monkeypatch.delenv('MOTUS_TELEOP_TICKET_SECRET', raising=False)
    auth.init({'MOTUS_TELEOP_TICKET_SECRET': TICKET_SECRET})
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, *_ = _build(signaler)

    result = asyncio.run(coordinator.signaling_offer(
        SESSION_ID,
        PRINCIPAL_ID,
        CLIENT_ID,
        OFFER,
    ))

    assert result['type'] == 'answer'
    claims, _payload, _signature = _decode_ticket(observed_ticket)
    assert claims['fence'] == FENCE


def test_offer_selects_dedicated_ticket_secret_by_exact_driver_id(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_ticket = ''
    dedicated = 'd' * 32
    other = 'e' * 32

    async def signaler(_driver_id, _offer, ticket, **_kwargs):
        nonlocal observed_ticket
        observed_ticket = ticket
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    auth.init({
        'MOTUS_TELEOP_TICKET_SECRETS': json.dumps({
            'teleop-driver-a': dedicated,
            'teleop-driver-b': other,
        }),
    })
    # A dedicated mapping must take precedence over the legacy process-wide
    # fallback, including in focused deployments that still export it.
    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, *_ = _build(signaler)

    result = asyncio.run(coordinator.signaling_offer(
        SESSION_ID,
        PRINCIPAL_ID,
        CLIENT_ID,
        OFFER,
    ))

    assert result['type'] == 'answer'
    _claims, payload, signature = _decode_ticket(observed_ticket)
    expected = base64.urlsafe_b64encode(
        hmac.new(dedicated.encode(), payload.encode('ascii'), hashlib.sha256).digest(),
    ).rstrip(b'=').decode('ascii')
    wrong_driver = base64.urlsafe_b64encode(
        hmac.new(other.encode(), payload.encode('ascii'), hashlib.sha256).digest(),
    ).rstrip(b'=').decode('ascii')
    assert hmac.compare_digest(signature, expected)
    assert not hmac.compare_digest(signature, wrong_driver)


@pytest.mark.parametrize(
    ('principal_id', 'client_id', 'error_type'),
    [
        ('owner:root', CLIENT_ID, SessionForbidden),
        (PRINCIPAL_ID, OTHER_CLIENT_ID, SessionClientMismatch),
    ],
)
def test_original_principal_and_client_are_both_required(
    monkeypatch: pytest.MonkeyPatch,
    principal_id,
    client_id,
    error_type,
):
    called = False

    async def signaler(*_args, **_kwargs):
        nonlocal called
        called = True
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, *_ = _build(signaler)
    with pytest.raises(error_type):
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            principal_id,
            client_id,
            OFFER,
        ))
    assert called is False


@pytest.mark.parametrize(
    'state',
    ['preparing', 'paused', 'hold', 'released', 'expired', 'faulted'],
)
def test_only_live_shadow_session_states_may_signal(
    monkeypatch: pytest.MonkeyPatch,
    state,
):
    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaler must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, *_ = _build(signaler, state=state)
    expected = (
        SessionStateConflict
        if state in {'preparing', 'paused', 'hold'}
        else SessionNotFound
    )
    with pytest.raises(expected):
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))


@pytest.mark.parametrize('secret', [None, 'short-secret'])
def test_missing_or_short_ticket_secret_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    secret,
):
    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaler must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    auth.init({})
    if secret is None:
        monkeypatch.delenv('MOTUS_TELEOP_TICKET_SECRET', raising=False)
    else:
        monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', secret)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, *_ = _build(signaler)
    with pytest.raises(TeleopServiceError) as captured:
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
    assert (captured.value.code, captured.value.status_code) == (
        'teleop_signaling_unavailable',
        503,
    )
    assert secret is None or secret not in str(captured.value)


@pytest.mark.parametrize('pin_change', ['missing', 'driver', 'digest'])
def test_signaling_requires_the_exact_session_pinned_target(
    monkeypatch: pytest.MonkeyPatch,
    pin_change,
):
    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaler must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, _manager, session, _clock, target = _build(signaler)
    if pin_change == 'missing':
        coordinator._pinned_targets.pop(session.id)
    elif pin_change == 'driver':
        coordinator._pinned_targets[session.id] = mcp_client.TrustedShadowTarget(
            'other-driver', target.url, DIGEST, target.descriptor_fingerprint, target.actions,
        )
    else:
        coordinator._pinned_targets[session.id] = mcp_client.TrustedShadowTarget(
            session.driver_id,
            target.url,
            'f' * 64,
            target.descriptor_fingerprint,
            target.actions,
        )
    with pytest.raises(TeleopServiceError) as captured:
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
    assert captured.value.code == 'driver_not_ready'


@pytest.mark.parametrize(
    'change',
    [
        {'type': 'offer'},
        {'boot_id': 'e4314745-5d67-4745-9c9f-fe4615778a4c'},
        {'session_id': '09259f38-bd2f-42e6-80f8-bc5bd931c8b7'},
        {'epoch': True},
        {'epoch': 8},
        {'capability_digest': 'f' * 64},
        {'mode': 'active'},
        {'actuation_enabled': True},
        {'sdp': FENCE},
        {'unexpected': 'field'},
    ],
)
def test_driver_answer_identity_and_shadow_mode_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    change,
):
    async def signaler(*_args, **_kwargs):
        return _answer(**change)

    async def quiet(*_args, **_kwargs):
        return {}

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, *_ = _build(signaler)
    with pytest.raises(TeleopServiceError) as captured:
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
    assert (captured.value.code, captured.value.status_code) == (
        'driver_protocol_error',
        502,
    )
    assert FENCE not in str(captured.value)


def test_driver_cannot_reflect_the_private_offer_ticket_to_the_browser(
    monkeypatch: pytest.MonkeyPatch,
):
    reflected_ticket = ''
    events = []

    async def signaler(_driver_id, _offer, ticket, **_kwargs):
        nonlocal reflected_ticket
        reflected_ticket = ticket
        return _answer(sdp=ticket)

    async def capture(event_type, **fields):
        events.append({'event_type': event_type, **deepcopy(fields)})
        return events[-1]

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', capture)
    coordinator, *_ = _build(signaler)
    with pytest.raises(TeleopServiceError) as captured:
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
    assert captured.value.code == 'driver_protocol_error'
    assert reflected_ticket
    serialized = json.dumps(events, sort_keys=True)
    assert reflected_ticket not in serialized
    assert FENCE not in serialized


@pytest.mark.parametrize(
    ('transport_error', 'expected'),
    [
        (mcp_client.TrustedShadowTransportError('timeout'), ('driver_timeout', 504)),
        (
            mcp_client.TrustedShadowTransportError('pinned_target_changed'),
            ('driver_session_lost', 409),
        ),
        (
            mcp_client.TrustedShadowTransportError('http_error', http_status=401),
            ('driver_auth_rejected', 502),
        ),
        (
            mcp_client.TrustedShadowTransportError('network_error'),
            ('driver_unreachable', 503),
        ),
        (
            mcp_client.TrustedShadowTransportError('response_too_large'),
            ('driver_protocol_error', 502),
        ),
        (
            mcp_client.TrustedShadowTransportError('redirect_rejected', http_status=307),
            ('driver_protocol_error', 502),
        ),
    ],
)
def test_transport_errors_are_stably_mapped_and_audited_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
    transport_error,
    expected,
):
    issued_ticket = ''
    events = []

    async def signaler(_driver_id, _offer, ticket, **_kwargs):
        nonlocal issued_ticket
        issued_ticket = ticket
        raise transport_error

    async def capture(event_type, **fields):
        events.append({'event_type': event_type, **deepcopy(fields)})
        return events[-1]

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', capture)
    coordinator, *_ = _build(signaler)
    with pytest.raises(TeleopServiceError) as captured:
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
    assert (captured.value.code, captured.value.status_code) == expected
    assert events[-1]['reason'] == expected[0]
    serialized = json.dumps(events, sort_keys=True)
    assert issued_ticket not in serialized
    assert FENCE not in serialized
    assert OFFER['sdp'] not in serialized


def test_release_preempts_offer_network_wait_and_stale_answer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    entered = asyncio.Event()
    finish_offer = asyncio.Event()

    async def signaler(*_args, **_kwargs):
        entered.set()
        await finish_offer.wait()
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, _clock, _target = _build(signaler)

        async def complete_release(_released, _principal_id, *, origin_task):
            assert origin_task is not None
            return True

        coordinator._complete_release_cleanup = complete_release  # type: ignore[method-assign]
        offer_task = asyncio.create_task(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
        await entered.wait()
        release_task = asyncio.create_task(coordinator.release(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            owner=False,
        ))
        for _ in range(20):
            if release_task.done():
                break
            await asyncio.sleep(0)
        assert release_task.done() is True
        assert offer_task.done() is False
        released, acknowledged = await release_task
        assert session.state == 'released'
        finish_offer.set()
        with pytest.raises(SessionNotFound):
            await offer_task
        return released.state, acknowledged

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    assert asyncio.run(scenario()) == ('released', True)


def test_expiry_during_offer_is_rejected_without_lease_renewal(
    monkeypatch: pytest.MonkeyPatch,
):
    async def quiet(*_args, **_kwargs):
        return {}

    coordinator = None

    async def signaler(*_args, **_kwargs):
        assert coordinator is not None
        clock.monotonic += 16.0
        return _answer()

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, _manager, session, clock, _target = _build(signaler)
    deadline = session.deadline_monotonic
    with pytest.raises(SessionNotFound):
        asyncio.run(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
    assert deadline == 115.0
    assert session.deadline_monotonic == deadline
    assert session.state == 'expired'


def test_capture_source_blocks_direct_offer_and_consumes_one_capture_offer(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    async def signaler(driver_id, offer, ticket, **kwargs):
        calls.append((driver_id, deepcopy(offer), ticket, kwargs))
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, *_ = _build(signaler)
        connection, assignment = await _attach_ready_capture(coordinator)

        with pytest.raises(TeleopServiceError, match='signaling_source_conflict'):
            await coordinator.signaling_offer(
                SESSION_ID,
                PRINCIPAL_ID,
                CLIENT_ID,
                OFFER,
            )
        answer = await coordinator.capture_signaling_offer(
            connection.capture_id,
            connection.connection_id,
            assignment.id,
            OFFER,
        )
        with pytest.raises(CaptureError, match='capture_offer_already_consumed'):
            await coordinator.capture_signaling_offer(
                connection.capture_id,
                connection.connection_id,
                assignment.id,
                OFFER,
            )
        return coordinator, connection, answer

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, connection, answer = asyncio.run(scenario())

    assert answer == {'sdp': 'v=0\r\na=driver-answer', 'type': 'answer'}
    assert len(calls) == 1
    assert coordinator._signaling_sources[SESSION_ID] == (
        'capture',
        connection.capture_id,
    )
    claims, _payload, _signature = _decode_ticket(calls[0][2])
    assert claims['fence'] == FENCE
    assert FENCE not in json.dumps(answer, sort_keys=True)


def test_completed_direct_offer_blocks_capture_attachment(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    async def signaler(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, *_ = _build(signaler)
        connection = await _ready_capture(coordinator)
        await coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        )
        with pytest.raises(TeleopServiceError, match='signaling_source_conflict'):
            await coordinator.attach_capture(
                SESSION_ID,
                PRINCIPAL_ID,
                CLIENT_ID,
                capture_id=connection.capture_id,
                mode='shadow',
                profile_id='recording',
                capability_digest=DIGEST,
            )
        return coordinator, connection

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, connection = asyncio.run(scenario())

    assert calls == 1
    assert coordinator._signaling_sources[SESSION_ID] == ('direct', CLIENT_ID)
    captures = asyncio.run(coordinator.list_captures(PRINCIPAL_ID))
    assert captures[0]['id'] == connection.capture_id
    assert captures[0]['assignment'] is None


def test_inflight_direct_offer_claims_source_before_capture_can_attach(
    monkeypatch: pytest.MonkeyPatch,
):
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def signaler(*_args, **_kwargs):
        entered.set()
        await finish.wait()
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, *_ = _build(signaler)
        connection = await _ready_capture(coordinator)
        direct = asyncio.create_task(coordinator.signaling_offer(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            OFFER,
        ))
        await entered.wait()
        with pytest.raises(TeleopServiceError, match='signaling_source_conflict'):
            await coordinator.attach_capture(
                SESSION_ID,
                PRINCIPAL_ID,
                CLIENT_ID,
                capture_id=connection.capture_id,
                mode='shadow',
                profile_id='recording',
                capability_digest=DIGEST,
            )
        finish.set()
        return coordinator, await direct

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, answer = asyncio.run(scenario())

    assert answer == {'sdp': 'v=0\r\na=driver-answer', 'type': 'answer'}
    assert coordinator._signaling_sources[SESSION_ID] == ('direct', CLIENT_ID)


def test_failed_capture_offer_holds_session_and_requires_new_pc_session(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    caller = _CaptureLossCaller()

    async def signaler(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise mcp_client.TrustedShadowTransportError('network_error')
        return _answer()

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, *_ = _build(signaler, caller=caller)
        _seed_active_driver_snapshot(coordinator, session)
        connection, assignment = await _attach_ready_capture(coordinator)
        with pytest.raises(TeleopServiceError, match='driver_unreachable'):
            await coordinator.capture_signaling_offer(
                connection.capture_id,
                connection.connection_id,
                assignment.id,
                OFFER,
            )
        assert session.state == 'hold'
        assert caller.actions == ['soft_stop']
        assert await connection.events.get() == {
            'type': 'assignment_revoked',
            'assignment_id': assignment.id,
            'reason': 'capture_signaling_failed',
        }
        with pytest.raises(SessionStateConflict):
            await coordinator.attach_capture(
                SESSION_ID,
                PRINCIPAL_ID,
                CLIENT_ID,
                capture_id=connection.capture_id,
                mode='shadow',
                profile_id='recording',
                capability_digest=DIGEST,
            )
        # WSS finally disconnects after returning the stable offer error. The
        # already revoked assignment must not issue a second Driver stop.
        await coordinator.disconnect_capture(
            connection.capture_id,
            connection.connection_id,
        )
        return coordinator, session, connection

    monkeypatch.setenv('MOTUS_TELEOP_TICKET_SECRET', TICKET_SECRET)
    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, session, connection = asyncio.run(scenario())

    assert calls == 1
    assert session.state == 'hold'
    assert caller.actions == ['soft_stop']
    assert coordinator._signaling_sources[SESSION_ID] == (
        'capture',
        connection.capture_id,
    )


def test_unconfirmed_live_session_cannot_attach_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaler must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, *_ = _build(signaler)
        session.mode = 'live'
        session.profile_id = 'dual_arm_profile_v1'
        session.signaling_audience = 'motus-teleop-rtc'
        session.live_confirmed = False
        connection = await _ready_capture(coordinator)
        with pytest.raises(SessionStateConflict):
            await coordinator.attach_capture(
                SESSION_ID,
                PRINCIPAL_ID,
                CLIENT_ID,
                capture_id=connection.capture_id,
                mode='live',
                profile_id=session.profile_id,
                capability_digest=DIGEST,
            )
        return coordinator

    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator = asyncio.run(scenario())
    assert coordinator._signaling_sources == {}


@pytest.mark.parametrize(
    ('terminal_method', 'terminal_state', 'reason'),
    [
        ('pause', 'paused', 'operator_pause'),
        ('soft_stop', 'hold', 'operator_soft_stop'),
        ('release', 'released', 'operator_release'),
    ],
)
def test_operator_lifecycle_revokes_capture_assignment_before_return(
    monkeypatch: pytest.MonkeyPatch,
    terminal_method,
    terminal_state,
    reason,
):
    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaler must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, *_ = _build(signaler)
        connection, assignment = await _attach_ready_capture(coordinator)
        session.state = terminal_state
        if terminal_method == 'release':
            coordinator._release_results[session.id] = True
        method = getattr(coordinator, terminal_method)
        await method(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            owner=False,
        )
        return coordinator, connection, assignment

    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, connection, assignment = asyncio.run(scenario())
    assert asyncio.run(connection.events.get()) == {
        'type': 'assignment_revoked',
        'assignment_id': assignment.id,
        'reason': reason,
    }
    assert asyncio.run(coordinator.list_captures(PRINCIPAL_ID))[0]['assignment'] is None


def test_fault_and_expiry_cleanup_revoke_capture_assignment(
    monkeypatch: pytest.MonkeyPatch,
):
    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaler must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    async def fault_scenario():
        coordinator, _manager, session, *_ = _build(signaler)
        connection, assignment = await _attach_ready_capture(coordinator)

        async def cleanup(*_args, **_kwargs):
            return None

        coordinator._complete_fault_cleanup = cleanup  # type: ignore[method-assign]
        await coordinator._fault_session(session, 'test_fault')
        return coordinator, connection, assignment

    async def expiry_scenario():
        coordinator, manager, _session, clock, *_ = _build(signaler)
        connection, assignment = await _attach_ready_capture(coordinator)
        clock.monotonic += 16.0
        expired = (await manager.expire_due())[0]

        async def cleanup(*_args, **_kwargs):
            return None

        async def acknowledged(*_args, **_kwargs):
            return True

        coordinator._cancel_heartbeat = cleanup  # type: ignore[method-assign]
        coordinator._update_command_claim = cleanup  # type: ignore[method-assign]
        coordinator._best_effort_soft_stop_and_release = acknowledged  # type: ignore[method-assign]
        coordinator._finalize_or_quarantine_release = cleanup  # type: ignore[method-assign]
        await coordinator._complete_expiry_cleanup(expired, origin_task=None)
        return coordinator, connection, assignment

    monkeypatch.setattr(audit, 'emit', quiet)
    faulted = asyncio.run(fault_scenario())
    expired = asyncio.run(expiry_scenario())
    for coordinator, connection, assignment in (faulted, expired):
        event = asyncio.run(connection.events.get())
        assert event['type'] == 'assignment_revoked'
        assert event['assignment_id'] == assignment.id
        assert asyncio.run(
            coordinator.list_captures(PRINCIPAL_ID),
        )[0]['assignment'] is None
    assert faulted[0]._signaling_sources[SESSION_ID][0] == 'capture'


@pytest.mark.parametrize(
    ('loss_kind', 'presence_state'),
    [
        ('disconnect', None),
        ('timeout', None),
        ('presence', 'xr_ended'),
        ('presence', 'error'),
        ('presence', 'browser_ready'),
        ('presence', 'xr_standby'),
    ],
)
def test_selective_capture_control_loss_forces_driver_hold(
    monkeypatch: pytest.MonkeyPatch,
    loss_kind,
    presence_state,
):
    caller = _CaptureLossCaller()

    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaling is already peer-to-peer')

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, manager, session, clock, *_ = _build(
            signaler,
            caller=caller,
        )
        _seed_active_driver_snapshot(coordinator, session)
        connection, assignment = await _attach_ready_capture(coordinator)

        # The PC ownership heartbeat remains healthy while only the capture
        # control WSS is lost. This must not leave the Driver RTC pose path live.
        clock.monotonic += 1.0
        await coordinator.heartbeat(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
        )
        assert session.state == 'active'

        if loss_kind == 'disconnect':
            await coordinator.disconnect_capture(
                connection.capture_id,
                connection.connection_id,
            )
        elif loss_kind == 'timeout':
            clock.monotonic += 5.0
            assert await coordinator.expire_capture_connection(
                connection.capture_id,
                connection.connection_id,
            ) is True
        else:
            await coordinator.capture_presence(
                connection.capture_id,
                connection.connection_id,
                state=presence_state,
                assignment_id=assignment.id if presence_state == 'error' else None,
            )
        return coordinator, manager, session, connection

    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, _manager, session, connection = asyncio.run(scenario())

    assert session.state == 'hold'
    assert caller.actions == ['soft_stop']
    assert coordinator._driver_snapshots[SESSION_ID]['state'] == 'hold'
    capture = asyncio.run(coordinator.list_captures(PRINCIPAL_ID))[0]
    assert capture['id'] == connection.capture_id
    assert capture['assignment'] is None


def test_capture_loss_pending_blocks_racing_reconnect_assignment_until_hold(
    monkeypatch: pytest.MonkeyPatch,
):
    caller = _CaptureLossCaller()

    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaling must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, *_ = _build(
            signaler,
            caller=caller,
        )
        _seed_active_driver_snapshot(coordinator, session)
        old_connection, _old_assignment = await _attach_ready_capture(coordinator)
        credential = old_connection.capture_credential
        assert credential is not None

        # Block the fail-close task before it can take the session lock. The
        # Manager must fence a racing reconnect/attach as soon as disconnect
        # commits, independent of which waiter gets the session lock next.
        session_lock = coordinator._session_lock(SESSION_ID)
        await session_lock.acquire()
        lost = await coordinator.capture_manager.disconnect(
            old_connection.capture_id,
            old_connection.connection_id,
        )
        assert lost is not None

        new_connection = await coordinator.connect_capture_with_credential(
            old_connection.capture_id,
            credential,
            **CAPTURE_METADATA,
        )
        await coordinator.capture_presence(
            new_connection.capture_id,
            new_connection.connection_id,
            state='xr_standby',
            assignment_id=None,
        )
        attach_task = asyncio.create_task(coordinator.attach_capture(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            capture_id=new_connection.capture_id,
            mode='shadow',
            profile_id='recording',
            capability_digest=DIGEST,
        ))
        await asyncio.sleep(0)
        loss_task = asyncio.create_task(
            coordinator._fail_close_capture_assignment_loss(lost),
        )
        session_lock.release()
        with pytest.raises(CaptureError, match='capture_loss_pending'):
            await attach_task
        await loss_task
        return coordinator, session, new_connection

    monkeypatch.setattr(audit, 'emit', quiet)
    coordinator, session, new_connection = asyncio.run(scenario())

    assert session.state == 'hold'
    assert caller.actions == ['soft_stop']
    assert coordinator._driver_snapshots[SESSION_ID]['state'] == 'hold'
    assert asyncio.run(coordinator.list_captures(PRINCIPAL_ID))[0]['assignment'] is None
    assert new_connection.events.empty()


def test_capture_loss_soft_stop_failure_uses_existing_fault_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    caller = _CaptureLossCaller(fail_soft_stop=True)
    cleanup_calls = []

    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaling must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, *_ = _build(
            signaler,
            caller=caller,
        )
        _seed_active_driver_snapshot(coordinator, session)
        connection, _assignment = await _attach_ready_capture(coordinator)

        async def cleanup(faulted, reason, *, origin_task):
            cleanup_calls.append((faulted.id, reason, origin_task is not None))

        coordinator._complete_fault_cleanup = cleanup  # type: ignore[method-assign]
        await coordinator.disconnect_capture(
            connection.capture_id,
            connection.connection_id,
        )
        return session

    monkeypatch.setattr(audit, 'emit', quiet)
    session = asyncio.run(scenario())

    assert session.state == 'faulted'
    assert caller.actions == ['soft_stop']
    assert cleanup_calls == [(SESSION_ID, 'driver_soft_stop_unconfirmed', True)]


def test_operator_soft_stop_then_wss_disconnect_does_not_stop_twice(
    monkeypatch: pytest.MonkeyPatch,
):
    caller = _CaptureLossCaller()

    async def signaler(*_args, **_kwargs):
        raise AssertionError('signaling must not run')

    async def quiet(*_args, **_kwargs):
        return {}

    async def scenario():
        coordinator, _manager, session, *_ = _build(
            signaler,
            caller=caller,
        )
        _seed_active_driver_snapshot(coordinator, session)
        connection, _assignment = await _attach_ready_capture(coordinator)
        held = await coordinator.soft_stop(
            SESSION_ID,
            PRINCIPAL_ID,
            CLIENT_ID,
            owner=False,
        )
        await coordinator.disconnect_capture(
            connection.capture_id,
            connection.connection_id,
        )
        return held

    monkeypatch.setattr(audit, 'emit', quiet)
    held = asyncio.run(scenario())

    assert held.state == 'hold'
    assert caller.actions == ['soft_stop']
