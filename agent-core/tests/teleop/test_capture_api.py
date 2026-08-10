from __future__ import annotations

import asyncio
import base64
import gc
import json

import pytest

from api import teleop
from teleop.capture_manager import CaptureConnection, CaptureManager
from teleop.service import TeleopServiceError
from teleop.session_manager import SessionClientMismatch, SessionStateConflict

CLIENT_ID = '7dbabfca-15c1-43ca-b600-75e7682c21d0'
REFRESHED_CLIENT_ID = '68991413-d37a-4603-9f07-3e25219d6d96'
SESSION_ID = 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3'
DIGEST = '0123456789abcdef' * 4
CAPTURE_METADATA = {
    'capture_protocol': 'motus.teleop.capture.v1',
    'frame_protocol': 'motus.teleop.rtc-frame.v1',
    'client_kind': 'browser_webxr',
    'app_version': '1.0.0-test',
}


class _CaptureApiCoordinator:
    def __init__(self, *, fail_signaling: bool = False):
        self.capture_manager = CaptureManager()
        self.fail_signaling = fail_signaling
        self.session_state = 'active'
        self.soft_stop_count = 0

    async def _complete_loss(self, lost_assignment):
        if lost_assignment is None:
            return
        self.soft_stop_count += 1
        self.session_state = 'hold'
        assert await self.capture_manager.complete_assignment_loss(
            lost_assignment,
        ) is True

    async def create_capture_pairing(self, principal_id, client_id, *, label):
        return await self.capture_manager.create_pairing(
            principal_id,
            client_id,
            label=label,
        )

    async def list_captures(self, principal_id):
        return await self.capture_manager.list_for_supervisor(principal_id)

    async def revoke_capture(self, capture_id, principal_id):
        return await self.capture_manager.revoke_capture(capture_id, principal_id)

    async def attach_capture(
        self,
        session_id,
        principal_id,
        client_id,
        *,
        capture_id,
        mode,
        profile_id,
        capability_digest,
    ):
        if principal_id != 'operator:alice':
            raise SessionStateConflict(session_id)
        if client_id != CLIENT_ID:
            raise SessionClientMismatch(session_id)
        if (
            self.session_state != 'active'
            or session_id != SESSION_ID
            or mode != 'shadow'
            or profile_id != 'recording'
            or capability_digest != DIGEST
        ):
            raise SessionStateConflict(session_id)
        return await self.capture_manager.attach(
            capture_id=capture_id,
            principal_id=principal_id,
            session_id=session_id,
            operation_generation=3,
            mode=mode,
            profile_id=profile_id,
            capability_digest=capability_digest,
            capabilities={
                'profile_id': 'recording',
                'input_bindings': {},
                'outputs': {'dual_arm': {'enabled': True, 'joint_count': 10}},
                'effectors': ['dual_arm'],
            },
            effectors=['dual_arm'],
        )

    async def connect_capture_with_pairing(
        self,
        pairing_id,
        pairing_code,
        **metadata,
    ):
        return await self.capture_manager.connect_with_pairing(
            pairing_id,
            pairing_code,
            **metadata,
        )

    async def connect_capture_with_credential(
        self,
        capture_id,
        capture_credential,
        **metadata,
    ):
        return await self.capture_manager.connect_with_credential(
            capture_id,
            capture_credential,
            **metadata,
        )

    async def disconnect_capture(self, capture_id, connection_id):
        lost_assignment = await self.capture_manager.disconnect(
            capture_id,
            connection_id,
        )
        await self._complete_loss(lost_assignment)

    async def capture_presence(
        self,
        capture_id,
        connection_id,
        *,
        state,
        assignment_id,
    ):
        presence, lost_assignment = await self.capture_manager.update_presence(
            capture_id,
            connection_id,
            state=state,
            assignment_id=assignment_id,
        )
        await self._complete_loss(lost_assignment)
        return presence

    async def expire_capture_connection(self, capture_id, connection_id):
        stale, lost_assignment = await self.capture_manager.expire_stale_connection(
            capture_id,
            connection_id,
        )
        await self._complete_loss(lost_assignment)
        return stale

    async def capture_signaling_offer(
        self,
        capture_id,
        connection_id,
        assignment_id,
        offer,
    ):
        assert offer == {'type': 'offer', 'sdp': 'v=0\r\no=quest-capture'}
        await self.capture_manager.claim_offer(
            capture_id,
            connection_id,
            assignment_id,
        )
        if self.fail_signaling:
            lost_assignment = await self.capture_manager.fail_offer(
                capture_id,
                assignment_id,
            )
            await self._complete_loss(lost_assignment)
            raise TeleopServiceError('driver_unreachable', 503)
        assert await self.capture_manager.complete_offer(
            capture_id,
            connection_id,
            assignment_id,
        ) is True
        return {'type': 'answer', 'sdp': f'v=0\r\na={assignment_id}'}


def _operator_headers(client_id: str = CLIENT_ID) -> dict[str, str]:
    return {
        'Authorization': 'Bearer operator-token',
        'X-Motus-Teleop-Client': client_id,
    }


def _pairing(client, *, label: str = 'Quest 3') -> dict:
    response = client.post(
        '/api/teleop/capture-pairings',
        headers=_operator_headers(),
        json={'label': label},
    )
    assert response.status_code == 201
    assert response.headers['cache-control'] == 'no-store'
    return response.json()['data']


def _pair_message(pairing: dict) -> dict:
    return {
        'type': 'pair',
        'pairing_id': pairing['pairing_id'],
        'pairing_code': pairing['pairing_code'],
        **CAPTURE_METADATA,
    }


def test_pairing_returns_only_bounded_public_tls_bootstrap(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    coordinator = _CaptureApiCoordinator()
    monkeypatch.setattr(teleop, 'coordinator', coordinator)

    pairing = _pairing(client)

    assert pairing['websocket_path'] == '/ws/teleop-capture'
    assert pairing['ca_certificate_pem'].startswith(
        '-----BEGIN CERTIFICATE-----\n'
    )
    assert pairing['ca_certificate_pem'].endswith(
        '-----END CERTIFICATE-----\n'
    )
    assert base64.b64decode(
        pairing['ca_certificate_base64'],
        validate=True,
    ).decode('ascii') == pairing['ca_certificate_pem']
    serialized = json.dumps(pairing, sort_keys=True)
    assert 'PRIVATE KEY' not in serialized
    assert 'cert.pem' not in serialized
    assert 'key.pem' not in serialized


def test_pairing_tls_bootstrap_is_not_visible_to_viewers(client):
    response = client.post(
        '/api/teleop/capture-pairings',
        headers={
            'Authorization': 'Bearer viewer-token',
            'X-Motus-Teleop-Client': CLIENT_ID,
        },
        json={'label': 'Quest 3'},
    )

    assert response.status_code == 403
    serialized = response.text
    assert 'BEGIN CERTIFICATE' not in serialized
    assert 'pairing_code' not in serialized


def test_pairing_fails_before_issuing_secret_without_safe_tls_bootstrap(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    coordinator = _CaptureApiCoordinator()
    monkeypatch.setattr(teleop, 'coordinator', coordinator)
    api_mount = next(route for route in client.app.routes if route.path == '/api')
    api_mount.app.state.teleop_capture_ca_certificate_pem = (
        '-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n'
    )

    response = client.post(
        '/api/teleop/capture-pairings',
        headers=_operator_headers(),
        json={'label': 'Quest 3'},
    )

    assert response.status_code == 503
    assert response.headers['cache-control'] == 'no-store'
    assert response.json()['detail']['code'] == 'capture_tls_bootstrap_unavailable'
    assert coordinator.capture_manager._pairings == {}


def test_capture_secret_models_hide_credentials_from_repr():
    pairing_code = 'pairing-secret-' + ('x' * 32)
    credential = 'credential-secret-' + ('y' * 32)
    pair = teleop.CapturePairMessage.model_validate({
        'type': 'pair',
        'pairing_id': 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3',
        'pairing_code': pairing_code,
        **CAPTURE_METADATA,
    })
    reconnect = teleop.CaptureCredentialMessage.model_validate({
        'type': 'credential',
        'capture_id': '79def5a3-85e9-48b0-9220-8e730e2944c1',
        'capture_credential': credential,
        **CAPTURE_METADATA,
    })

    assert pairing_code not in repr(pair)
    assert credential not in repr(reconnect)


def test_pairing_is_in_band_and_enrollment_survives_pc_tab_refresh(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    coordinator = _CaptureApiCoordinator()
    monkeypatch.setattr(teleop, 'coordinator', coordinator)
    pairing = _pairing(client)

    with client.websocket_connect('/ws/teleop-capture') as websocket:
        websocket.send_json(_pair_message(pairing))
        paired = websocket.receive_json()
        credential = paired['capture_credential']
        capture_id = paired['capture_id']
        assert paired == {
            'type': 'paired',
            'capture_id': capture_id,
            'capture_credential': credential,
            'capture_protocol': CAPTURE_METADATA['capture_protocol'],
            'frame_protocol': CAPTURE_METADATA['frame_protocol'],
            'presence_interval_ms': 2_000,
            'presence_timeout_ms': 5_000,
        }

        websocket.send_json({
            'type': 'presence',
            'state': 'xr_standby',
            'assignment_id': None,
        })
        assert websocket.receive_json() == {
            'type': 'presence_ack',
            'state': 'xr_standby',
        }

        refreshed = client.get(
            '/api/teleop/captures',
            headers=_operator_headers(REFRESHED_CLIENT_ID),
        )
        assert refreshed.status_code == 200
        public_capture = refreshed.json()['data'][0]
        assert public_capture['id'] == capture_id
        assert public_capture['connected'] is True
        assert public_capture['observed_state'] == 'xr_standby'
        for key, value in CAPTURE_METADATA.items():
            assert public_capture[key] == value

        serialized = json.dumps(public_capture, sort_keys=True)
        for secret in (
            pairing['pairing_code'],
            credential,
            CLIENT_ID,
            REFRESHED_CLIENT_ID,
        ):
            assert secret not in serialized

        other_principal = client.get(
            '/api/teleop/captures',
            headers={
                'Authorization': 'Bearer owner-token',
                'X-Motus-Teleop-Client': REFRESHED_CLIENT_ID,
            },
        )
        assert other_principal.status_code == 200
        assert other_principal.json()['data'] == []

        revoked = client.delete(
            f'/api/teleop/captures/{capture_id}',
            headers=_operator_headers(REFRESHED_CLIENT_ID),
        )
        assert revoked.status_code == 200
        assert websocket.receive_json() == {
            'type': 'capture_revoked',
            'reason': 'operator_revoked',
        }


def test_original_session_tab_attaches_and_wss_receives_only_safe_assignment(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    coordinator = _CaptureApiCoordinator()
    monkeypatch.setattr(teleop, 'coordinator', coordinator)
    pairing = _pairing(client)

    with client.websocket_connect('/ws/teleop-capture') as websocket:
        websocket.send_json(_pair_message(pairing))
        paired = websocket.receive_json()
        websocket.send_json({
            'type': 'presence',
            'state': 'xr_standby',
            'assignment_id': None,
        })
        assert websocket.receive_json()['type'] == 'presence_ack'

        wrong_tab = client.post(
            f'/api/teleop/sessions/{SESSION_ID}/capture-attachment',
            headers=_operator_headers(REFRESHED_CLIENT_ID),
            json={
                'capture_id': paired['capture_id'],
                'mode': 'shadow',
                'profile_id': 'recording',
                'capability_digest': DIGEST,
            },
        )
        assert wrong_tab.status_code == 403
        assert wrong_tab.json()['detail']['code'] == 'session_client_mismatch'

        attached = client.post(
            f'/api/teleop/sessions/{SESSION_ID}/capture-attachment',
            headers=_operator_headers(),
            json={
                'capture_id': paired['capture_id'],
                'mode': 'shadow',
                'profile_id': 'recording',
                'capability_digest': DIGEST,
            },
        )
        assert attached.status_code == 200
        assignment = websocket.receive_json()
        assert assignment['type'] == 'assignment'
        assert assignment['assignment'] == attached.json()['data']
        assignment_id = assignment['assignment']['id']

        public_wire = json.dumps(assignment, sort_keys=True)
        for secret in (
            paired['capture_credential'],
            pairing['pairing_code'],
            CLIENT_ID,
            'private-fence',
            'teleop-ticket',
        ):
            assert secret not in public_wire

        websocket.send_json({
            'type': 'signaling_offer',
            'assignment_id': assignment_id,
            'offer': {'type': 'offer', 'sdp': 'v=0\r\no=quest-capture'},
        })
        assert websocket.receive_json() == {
            'type': 'signaling_answer',
            'assignment_id': assignment_id,
            'answer': {
                'type': 'answer',
                'sdp': f'v=0\r\na={assignment_id}',
            },
        }

        still_attached = client.delete(
            f"/api/teleop/captures/{paired['capture_id']}",
            headers=_operator_headers(),
        )
        assert still_attached.status_code == 409
        assert still_attached.json()['detail']['code'] == 'capture_attached'

        websocket.send_json({
            'type': 'presence',
            'state': 'xr_ended',
            'assignment_id': None,
        })
        assert websocket.receive_json() == {
            'type': 'presence_ack',
            'state': 'xr_ended',
        }
        assert websocket.receive_json() == {
            'type': 'assignment_revoked',
            'assignment_id': assignment_id,
            'reason': 'capture_xr_ended',
        }

        revoked = client.delete(
            f"/api/teleop/captures/{paired['capture_id']}",
            headers=_operator_headers(),
        )
        assert revoked.status_code == 200
        assert websocket.receive_json() == {
            'type': 'capture_revoked',
            'reason': 'operator_revoked',
        }


def test_real_wss_offer_error_closes_and_holds_exactly_once(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    coordinator = _CaptureApiCoordinator(fail_signaling=True)
    monkeypatch.setattr(teleop, 'coordinator', coordinator)
    pairing = _pairing(client)

    with client.websocket_connect('/ws/teleop-capture') as websocket:
        websocket.send_json(_pair_message(pairing))
        paired = websocket.receive_json()
        websocket.send_json({
            'type': 'presence',
            'state': 'xr_standby',
            'assignment_id': None,
        })
        assert websocket.receive_json()['type'] == 'presence_ack'
        attached = client.post(
            f'/api/teleop/sessions/{SESSION_ID}/capture-attachment',
            headers=_operator_headers(),
            json={
                'capture_id': paired['capture_id'],
                'mode': 'shadow',
                'profile_id': 'recording',
                'capability_digest': DIGEST,
            },
        )
        assert attached.status_code == 200
        assignment = websocket.receive_json()['assignment']

        websocket.send_json({
            'type': 'signaling_offer',
            'assignment_id': assignment['id'],
            'offer': {'type': 'offer', 'sdp': 'v=0\r\no=quest-capture'},
        })
        assert websocket.receive_json() == {
            'type': 'error',
            'code': 'driver_unreachable',
        }

    assert coordinator.session_state == 'hold'
    assert coordinator.soft_stop_count == 1
    captures = asyncio.run(coordinator.list_captures('operator:alice'))
    assert captures[0]['assignment'] is None


class _SimultaneousTerminalWebSocket:
    def __init__(self, first_message: dict):
        self._first_message = first_message
        self._receive_count = 0
        self.peer_disconnect_ready = asyncio.Event()
        self.sent: list[dict] = []
        self.closed: list[tuple[int, str]] = []

    async def accept(self):
        return None

    async def receive(self):
        self._receive_count += 1
        if self._receive_count == 1:
            return {
                'type': 'websocket.receive',
                'text': json.dumps(self._first_message),
            }
        self.peer_disconnect_ready.set()
        return {'type': 'websocket.disconnect', 'code': 1000}

    async def send_json(self, value):
        self.sent.append(value)

    async def close(self, *, code, reason):
        self.closed.append((code, reason))


class _TerminalAfterPeerDisconnectEvents:
    def __init__(self, websocket: _SimultaneousTerminalWebSocket):
        self.websocket = websocket

    async def get(self):
        await self.websocket.peer_disconnect_ready.wait()
        # Let the receive task turn the disconnect packet into its terminal
        # exception before the event task completes in the same wait cycle.
        await asyncio.sleep(0)
        return {'type': 'capture_revoked', 'reason': 'operator_revoked'}


class _SimultaneousTerminalCoordinator:
    def __init__(self, websocket: _SimultaneousTerminalWebSocket):
        self.websocket = websocket
        self.capture_manager = CaptureManager()
        self.disconnects = []

    async def connect_capture_with_pairing(self, *_args, **_kwargs):
        return CaptureConnection(
            capture_id='79def5a3-85e9-48b0-9220-8e730e2944c1',
            connection_id='29656917-6e93-40ae-9476-0a532787d752',
            events=_TerminalAfterPeerDisconnectEvents(self.websocket),
            capture_credential='capture-credential-that-is-private-000000',
        )

    async def disconnect_capture(self, capture_id, connection_id):
        self.disconnects.append((capture_id, connection_id))


def test_simultaneous_terminal_event_and_peer_disconnect_consumes_both_tasks(
    monkeypatch: pytest.MonkeyPatch,
):
    async def scenario():
        websocket = _SimultaneousTerminalWebSocket({
            'type': 'pair',
            'pairing_id': 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3',
            'pairing_code': 'x' * 32,
            **CAPTURE_METADATA,
        })
        coordinator = _SimultaneousTerminalCoordinator(websocket)
        monkeypatch.setattr(teleop, 'coordinator', coordinator)
        loop = asyncio.get_running_loop()
        unhandled = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda _loop, context: unhandled.append(context),
        )
        try:
            await teleop.teleop_capture_websocket(websocket)
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)
        return websocket, coordinator, unhandled

    websocket, coordinator, unhandled = asyncio.run(scenario())

    assert [message['type'] for message in websocket.sent] == [
        'paired',
        'capture_revoked',
    ]
    assert websocket.closed == [(4403, 'operator_revoked')]
    assert coordinator.disconnects == [(
        '79def5a3-85e9-48b0-9220-8e730e2944c1',
        '29656917-6e93-40ae-9476-0a532787d752',
    )]
    assert unhandled == []


@pytest.mark.parametrize(
    'message',
    [
        {
            'type': 'pair',
            'pairing_id': 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3',
            'pairing_code': 'x' * 32,
        },
        {
            'type': 'pair',
            'pairing_id': 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3',
            'pairing_code': 'x' * 32,
            **CAPTURE_METADATA,
            'frame_protocol': 'motus.teleop.rtc-frame.v2',
        },
        {
            'type': 'pair',
            'pairing_id': 'd097eb8f-b386-455f-9e2b-23f1ad6a1ee3',
            'pairing_code': 'x' * 32,
            **CAPTURE_METADATA,
            'app_version': 'x' * 33,
        },
    ],
)
def test_wss_rejects_capture_with_unknown_frame_contract(
    client,
    monkeypatch: pytest.MonkeyPatch,
    message,
):
    monkeypatch.setattr(teleop, 'coordinator', _CaptureApiCoordinator())
    with client.websocket_connect('/ws/teleop-capture') as websocket:
        websocket.send_json(message)
        assert websocket.receive_json() == {
            'type': 'error',
            'code': 'capture_message_invalid',
        }


def test_capture_secrets_are_never_accepted_in_url(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(teleop, 'coordinator', _CaptureApiCoordinator())
    pairing = _pairing(client)
    with client.websocket_connect(
        f"/ws/teleop-capture?pairing_code={pairing['pairing_code']}",
    ) as websocket:
        websocket.send_json(_pair_message(pairing))
        assert websocket.receive_json() == {
            'type': 'error',
            'code': 'capture_query_forbidden',
        }
