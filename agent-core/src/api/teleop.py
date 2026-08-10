"""Dedicated teleoperation identity, session, and signaling API."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import uuid
from typing import Literal

import fastapi
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import auth
import config
import mcp_client
from teleop import audit
from teleop.capture_manager import CaptureError
from teleop.command_broker import (
    InvalidAuthorityBinding,
    authority_domain_for_target,
)
from teleop.contracts import (
    SHADOW_MODE,
    TeleopContractError,
    project_teleop_descriptor,
)
from teleop.service import TeleopServiceError, coordinator
from teleop.session_manager import (
    SessionClientMismatch,
    SessionConflict,
    SessionForbidden,
    SessionNotFound,
    SessionStateConflict,
)
from tls_config import TLSConfigurationError, project_public_certificate_chain

router = fastapi.APIRouter(prefix='/teleop', tags=['teleop'])
ws_router = fastapi.APIRouter(tags=['teleop-capture'])

_SENSITIVE_KEY_PARTS = (
    'apikey',
    'credential',
    'fence',
    'password',
    'privatekey',
    'secret',
    'token',
)
_SIGNALING_OFFER_BODY_LIMIT = 128 * 1024
_SIGNALING_OFFER_SDP_LIMIT = 120 * 1024
_CAPTURE_WS_AUTH_TIMEOUT_SECONDS = 5.0
_CAPTURE_WS_HEALTH_TICK_SECONDS = 1.0
_CAPTURE_WS_MESSAGE_LIMIT = 128 * 1024


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    driver_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]*$')
    mode: Literal['shadow', 'live'] = SHADOW_MODE


class LiveConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    confirm_live_actuation: Literal[True]
    profile_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]*$',
    )

    @field_validator('confirm_live_actuation', mode='before')
    @classmethod
    def _require_exact_true(cls, value):
        if value is not True:
            raise ValueError('confirm_live_actuation must be the boolean true')
        return value


class SignalingOfferRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    type: Literal['offer']
    sdp: str = Field(min_length=1, max_length=_SIGNALING_OFFER_SDP_LIMIT)


class CapturePairingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    label: str = Field(min_length=1, max_length=64)


class CaptureAttachmentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    capture_id: str = Field(
        min_length=36,
        max_length=36,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    )
    mode: Literal['shadow', 'live']
    profile_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]*$',
    )
    capability_digest: str = Field(pattern=r'^[0-9a-f]{64}$')


class CapturePairMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    type: Literal['pair']
    pairing_id: str = Field(
        min_length=36,
        max_length=36,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    )
    pairing_code: str = Field(min_length=32, max_length=128, repr=False)
    capture_protocol: Literal['motus.teleop.capture.v1']
    frame_protocol: Literal['motus.teleop.rtc-frame.v1']
    client_kind: Literal['browser_webxr', 'native_openxr']
    app_version: str = Field(
        min_length=1,
        max_length=32,
        pattern=r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$',
    )


class CaptureCredentialMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    type: Literal['credential']
    capture_id: str = Field(
        min_length=36,
        max_length=36,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    )
    capture_credential: str = Field(min_length=32, max_length=128, repr=False)
    capture_protocol: Literal['motus.teleop.capture.v1']
    frame_protocol: Literal['motus.teleop.rtc-frame.v1']
    client_kind: Literal['browser_webxr', 'native_openxr']
    app_version: str = Field(
        min_length=1,
        max_length=32,
        pattern=r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$',
    )


class CapturePresenceMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    type: Literal['presence']
    state: Literal[
        'browser_ready',
        'error',
        'rtc_connecting',
        'streaming',
        'xr_ended',
        'xr_standby',
    ]
    assignment_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    )


class CaptureSignalingOfferMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    type: Literal['signaling_offer']
    assignment_id: str = Field(
        min_length=36,
        max_length=36,
        pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    )
    offer: SignalingOfferRequest


class _DuplicateJsonField(ValueError):
    pass


def _reject_duplicate_json_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonField(key)
        value[key] = item
    return value


def _reject_non_finite_json(_value: str):
    raise ValueError('non-finite JSON is not allowed')


async def _signaling_offer_request(request: fastapi.Request) -> SignalingOfferRequest:
    media_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
    if media_type != 'application/json':
        raise fastapi.HTTPException(
            status_code=415,
            detail={'code': 'signaling_content_type_required'},
        )
    content_length = request.headers.get('content-length')
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = -1
        if declared_length < 0:
            raise fastapi.HTTPException(
                status_code=400,
                detail={'code': 'invalid_signaling_offer'},
            )
        if declared_length > _SIGNALING_OFFER_BODY_LIMIT:
            raise fastapi.HTTPException(
                status_code=413,
                detail={'code': 'signaling_offer_too_large'},
            )

    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > _SIGNALING_OFFER_BODY_LIMIT:
            raise fastapi.HTTPException(
                status_code=413,
                detail={'code': 'signaling_offer_too_large'},
            )
        raw.extend(chunk)
    try:
        payload = json.loads(
            raw.decode('utf-8'),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_non_finite_json,
        )
        body = SignalingOfferRequest.model_validate(payload)
        if len(body.sdp.encode('utf-8')) > _SIGNALING_OFFER_SDP_LIMIT:
            raise ValueError('SDP exceeds byte limit')
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError,
        RecursionError,
    ):
        raise fastapi.HTTPException(
            status_code=400,
            detail={'code': 'invalid_signaling_offer'},
        ) from None
    return body


def _capture_ws_payload(
    raw: str,
    *,
    first_message: bool,
) -> BaseModel:
    try:
        if len(raw.encode('utf-8')) > _CAPTURE_WS_MESSAGE_LIMIT:
            raise ValueError('capture message too large')
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_non_finite_json,
        )
        if not isinstance(payload, dict):
            raise TypeError('capture message must be an object')
        message_type = payload.get('type')
        if first_message:
            model = {
                'credential': CaptureCredentialMessage,
                'pair': CapturePairMessage,
            }.get(message_type)
        else:
            model = {
                'presence': CapturePresenceMessage,
                'signaling_offer': CaptureSignalingOfferMessage,
            }.get(message_type)
        if model is None:
            raise ValueError('capture message type is not supported')
        return model.model_validate(payload)
    except (
        UnicodeEncodeError,
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError,
        RecursionError,
    ):
        raise CaptureError('capture_message_invalid', 400) from None


async def _capture_ws_text(websocket: fastapi.WebSocket) -> str:
    message = await websocket.receive()
    if message.get('type') == 'websocket.disconnect':
        raise fastapi.WebSocketDisconnect(message.get('code', 1000))
    raw = message.get('text')
    if not isinstance(raw, str):
        raise CaptureError('capture_message_invalid', 400)
    return raw


def _capture_ws_error_code(error: BaseException) -> str:
    if isinstance(error, (CaptureError, TeleopServiceError)):
        return error.code
    if isinstance(error, SessionClientMismatch):
        return 'session_client_mismatch'
    if isinstance(error, SessionForbidden):
        return 'session_forbidden'
    if isinstance(error, SessionNotFound):
        return 'session_not_found'
    if isinstance(error, SessionStateConflict):
        return 'session_state_conflict'
    if isinstance(error, asyncio.TimeoutError):
        return 'capture_auth_timeout'
    return 'capture_internal_error'


async def _close_capture_ws(
    websocket: fastapi.WebSocket,
    *,
    code: int,
    reason: str,
) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001, S110 -- the peer may already be gone
        pass


def _client_id(request: fastapi.Request) -> str:
    value = request.headers.get('x-motus-teleop-client', '')
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise fastapi.HTTPException(
            status_code=400,
            detail={'code': 'teleop_client_required'},
        ) from None
    if str(parsed) != value.lower():
        raise fastapi.HTTPException(
            status_code=400,
            detail={'code': 'teleop_client_required'},
        )
    return str(parsed)


def _safe_metadata(value, depth: int = 0):
    """Bound untrusted descriptor metadata and strip credential-like fields."""
    if depth > 5:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        return [_safe_metadata(item, depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        safe = {}
        for raw_key, item in list(value.items())[:100]:
            key = str(raw_key)[:100]
            normalized = ''.join(
                character for character in key.lower()
                if character.isalnum()
            )
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                continue
            safe[key] = _safe_metadata(item, depth + 1)
        return safe
    return str(value)[:1000]


def _tool_name(tool) -> str:
    return tool.get('name', '') if isinstance(tool, dict) else str(tool)


def _project_descriptor(
    tool: dict | None,
    expected_driver_id: str | None = None,
) -> dict | None:
    try:
        return project_teleop_descriptor(
            tool,
            expected_driver_id=expected_driver_id,
        )
    except TeleopContractError:
        return None


def _valid_teleop_descriptor(
    tool: dict | None,
    expected_driver_id: str | None = None,
) -> bool:
    return _project_descriptor(tool, expected_driver_id) is not None


def _valid_shadow_descriptor(
    tool: dict | None,
    expected_driver_id: str | None = None,
) -> bool:
    """Compatibility predicate retained for callers that require Shadow."""

    descriptor = _project_descriptor(tool, expected_driver_id)
    return descriptor is not None and descriptor['mode'] == SHADOW_MODE


def _robot_view(mcp: dict, mcps: list | None = None) -> dict:
    tools = mcp.get('tools') or []
    session_tool = next(
        (
            tool for tool in tools
            if isinstance(tool, dict) and tool.get('name') == 'teleop_session'
        ),
        None,
    )
    explicitly_declared = bool(session_tool and 'x-teleop' in session_tool)
    projected_descriptor = _project_descriptor(session_tool, mcp.get('id', ''))
    descriptor_valid = projected_descriptor is not None
    try:
        robot_id = authority_domain_for_target(
            mcp.get('id', ''),
            mcp,
            targets=mcps,
        )
        binding_valid = True
    except InvalidAuthorityBinding:
        robot_id = ''
        binding_valid = False
    descriptor = session_tool.get('x-teleop') if isinstance(session_tool, dict) else {}
    descriptor_robot_id = (
        descriptor.get('robot_id') if isinstance(descriptor, dict) else None
    )
    reported_robot_id = mcp.get('reported_robot_id')
    robot_identity_matches = binding_valid and (
        descriptor_robot_id in (None, robot_id)
        and reported_robot_id in (None, '', robot_id)
        and (robot_id == mcp.get('id') or descriptor_robot_id == robot_id)
    )
    trust_state = mcp.get('trust_state') or (
        'quarantined' if auth.is_driver_auth_enforced() else 'untrusted'
    )
    credential_missing = (
        trust_state == 'trusted'
        and not auth.driver_record_credential_available(
            mcp.get('id', ''),
            mcp.get('credential_binding'),
        )
    )
    if credential_missing:
        trust_state = 'quarantined'
    signaling_missing = not auth.teleop_ticket_credential_available(
        mcp.get('id', '')
    )
    registry_entry = mcp_client.registry.get(mcp.get('id', ''), {})
    online = bool(registry_entry.get('online', False))
    runtime_trusted = registry_entry.get('trusted') is True
    target_matches = registry_entry.get('url') == mcp.get('url')
    descriptor_matches = (
        registry_entry.get('teleop_fingerprint')
        == mcp_client.teleop_tool_fingerprint(session_tool)
    )

    if credential_missing:
        reason = 'driver_credential_unavailable'
    elif trust_state != 'trusted':
        reason = 'driver_registration_not_trusted'
    elif not binding_valid:
        reason = 'authority_binding_invalid'
    elif not explicitly_declared:
        reason = 'teleop_session_not_declared'
    elif not descriptor_valid:
        reason = 'teleop_descriptor_invalid'
    elif signaling_missing:
        reason = 'teleop_signaling_unavailable'
    elif not robot_identity_matches:
        reason = 'authority_binding_required'
    elif (
        mcp.get('transport') != 'http'
        or mcp_client.teleop_url_safety_code(mcp.get('url')) is not None
    ):
        reason = 'driver_transport_invalid'
    elif not runtime_trusted:
        reason = 'driver_runtime_not_trusted'
    elif not target_matches or not descriptor_matches:
        reason = 'driver_runtime_target_mismatch'
    elif not online:
        reason = 'driver_offline'
    else:
        reason = 'ready'

    return {
        'id': mcp.get('id', ''),
        'driver_id': mcp.get('id', ''),
        'robot_id': robot_id,
        'name': mcp.get('name', '') or mcp.get('server_name', ''),
        'server_name': mcp.get('server_name', ''),
        'online': online,
        'trust_state': trust_state,
        'teleop_declared': explicitly_declared,
        'descriptor_valid': descriptor_valid,
        'teleop_ready': reason == 'ready',
        'reason': reason,
        'teleop': _safe_metadata(session_tool.get('x-teleop')) if session_tool else None,
        'annotations': _safe_metadata(session_tool.get('annotations') or {}) if session_tool else {},
        'tools': sorted({_tool_name(tool) for tool in tools if _tool_name(tool)}),
    }


@router.get('/me')
async def teleop_me(request: fastapi.Request):
    principal = auth.require_role(request, 'viewer')
    return {
        'code': 200,
        'data': {
            'principal': principal.as_dict(),
            'permissions': {
                'view_devices': True,
                'control': principal.role in ('operator', 'owner'),
            },
        },
    }


@router.get('/robots')
async def teleop_robots(request: fastapi.Request):
    principal = auth.require_role(request, 'viewer')
    client_id = request.headers.get('x-motus-teleop-client', '')
    mcps = await asyncio.to_thread(
        lambda: config.main.get('services', {}).get('mcp', []),
    )
    robots = []
    represented_guard_ids: set[str] = set()
    for mcp in mcps:
        if mcp.get('category') != 'driver':
            continue
        robot = _robot_view(mcp, mcps)
        session = (
            await coordinator.manager.active_for_robot(robot['robot_id'])
            if robot['robot_id'] else None
        )
        authority_guard = (
            coordinator.authority_guard_for_robot(robot['robot_id'])
            if robot['robot_id'] else None
        )
        if authority_guard is not None:
            represented_guard_ids.add(authority_guard['robot_id'])
            robot['teleop_ready'] = False
            robot['reason'] = 'authority_recovery_required'
            robot['authority_guard'] = authority_guard
            robot['session'] = {
                'busy': True,
                'owned_by_me': False,
                'owned_by_client': False,
                'state': 'recovery_required',
            }
        elif session is None:
            robot['session'] = {
                'busy': False,
                'owned_by_me': False,
                'owned_by_client': False,
            }
        else:
            owned_by_me = session.principal_id == principal.id
            owned_by_client = owned_by_me and session.client_id == client_id
            session_view = coordinator.manager.public_dict(session)
            robot['session'] = {
                'busy': True,
                'owned_by_me': owned_by_me,
                'owned_by_client': owned_by_client,
                'state': session.state,
                'remaining_seconds': session_view['remaining_seconds'],
            }
            if owned_by_me or principal.role == 'owner':
                robot['session'].update({
                    'id': session.id,
                    'principal_id': session.principal_id,
                })
        robots.append(robot)
    for authority_guard in coordinator.list_authority_guards():
        robot_id = authority_guard['robot_id']
        if robot_id in represented_guard_ids:
            continue
        robots.append({
            'id': f'recovery:{robot_id}',
            'driver_id': authority_guard['driver_id'],
            'robot_id': robot_id,
            'name': f'{robot_id} · recovery',
            'server_name': '',
            'online': False,
            'trust_state': 'unknown',
            'teleop_declared': False,
            'descriptor_valid': False,
            'teleop_ready': False,
            'reason': 'authority_recovery_required',
            'teleop': None,
            'annotations': {},
            'tools': [],
            'authority_guard': authority_guard,
            'session': {
                'busy': True,
                'owned_by_me': False,
                'owned_by_client': False,
                'state': 'recovery_required',
            },
        })
    robots.sort(key=lambda item: (not item['teleop_ready'], item['name'], item['id']))
    return {'code': 200, 'data': robots}


def _owner(principal: auth.Principal) -> bool:
    return principal.role == 'owner'


def _capture_ca_certificate_pem(request: fastapi.Request) -> str:
    raw_pem = getattr(
        request.app.state,
        'teleop_capture_ca_certificate_pem',
        None,
    )
    try:
        return project_public_certificate_chain(raw_pem)
    except TLSConfigurationError as exc:
        raise fastapi.HTTPException(
            status_code=503,
            detail={'code': 'capture_tls_bootstrap_unavailable'},
            headers={'Cache-Control': 'no-store'},
        ) from exc


@router.post('/authority-guards/{robot_id}/reconcile')
async def reconcile_authority_guard(robot_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'owner')
    try:
        result = await coordinator.reconcile_authority_guard(
            robot_id,
            principal_id=principal.id,
        )
    except Exception as error:
        await _raise_session_error(error)
        raise
    return {'code': 200, 'data': result}


@router.post('/capture-pairings')
async def create_capture_pairing(
    body: CapturePairingRequest,
    request: fastapi.Request,
    response: fastapi.Response,
):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    ca_certificate_pem = _capture_ca_certificate_pem(request)
    try:
        pairing = await coordinator.create_capture_pairing(
            principal.id,
            client_id,
            label=body.label,
        )
    except Exception as error:
        await _raise_session_error(error)
        raise
    response.status_code = 201
    response.headers['Cache-Control'] = 'no-store'
    return {
        'code': 201,
        'data': {
            'pairing_id': pairing.pairing_id,
            'pairing_code': pairing.pairing_code,
            'expires_at': pairing.expires_at,
            'websocket_path': '/ws/teleop-capture',
            'ca_certificate_pem': ca_certificate_pem,
            'ca_certificate_base64': base64.b64encode(
                ca_certificate_pem.encode('ascii')
            ).decode('ascii'),
        },
    }


@router.get('/captures')
async def list_captures(request: fastapi.Request, response: fastapi.Response):
    principal = auth.require_role(request, 'operator')
    _client_id(request)
    try:
        captures = await coordinator.list_captures(principal.id)
    except Exception as error:
        await _raise_session_error(error)
        raise
    response.headers['Cache-Control'] = 'no-store'
    return {'code': 200, 'data': captures}


@router.delete('/captures/{capture_id}')
async def revoke_capture(
    capture_id: str,
    request: fastapi.Request,
    response: fastapi.Response,
):
    principal = auth.require_role(request, 'operator')
    _client_id(request)
    try:
        capture = await coordinator.revoke_capture(capture_id, principal.id)
    except Exception as error:
        await _raise_session_error(error)
        raise
    response.headers['Cache-Control'] = 'no-store'
    return {'code': 200, 'data': capture}


@router.post('/sessions/{session_id}/capture-attachment')
async def attach_capture(
    session_id: str,
    body: CaptureAttachmentRequest,
    request: fastapi.Request,
    response: fastapi.Response,
):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        assignment = await coordinator.attach_capture(
            session_id,
            principal.id,
            client_id,
            capture_id=body.capture_id,
            mode=body.mode,
            profile_id=body.profile_id,
            capability_digest=body.capability_digest,
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    response.headers['Cache-Control'] = 'no-store'
    return {
        'code': 200,
        'data': coordinator.capture_manager.public_assignment(assignment),
    }


async def _raise_session_error(error: Exception, session_id: str = '') -> None:
    if isinstance(error, CaptureError):
        raise fastapi.HTTPException(
            status_code=error.status_code,
            detail={'code': error.code},
        ) from None
    if isinstance(error, TeleopServiceError):
        raise fastapi.HTTPException(
            status_code=error.status_code,
            detail={'code': error.code},
        ) from None
    if isinstance(error, SessionConflict):
        raise fastapi.HTTPException(
            status_code=409,
            detail={'code': 'robot_busy'},
        ) from None
    if isinstance(error, SessionForbidden):
        code = (
            'session_client_mismatch'
            if isinstance(error, SessionClientMismatch)
            else 'session_forbidden'
        )
        raise fastapi.HTTPException(
            status_code=403,
            detail={'code': code},
        ) from None
    if isinstance(error, SessionStateConflict):
        raise fastapi.HTTPException(
            status_code=409,
            detail={'code': 'session_state_conflict'},
        ) from None
    if isinstance(error, SessionNotFound):
        known = await coordinator.manager.get(session_id) if session_id else None
        if known is not None and known.state in ('released', 'expired', 'faulted'):
            raise fastapi.HTTPException(
                status_code=410,
                detail={'code': f'session_{known.state}'},
            ) from None
        raise fastapi.HTTPException(
            status_code=404,
            detail={'code': 'session_not_found'},
        ) from None
    raise error


@router.post('/sessions')
async def acquire_session(
    body: SessionCreateRequest,
    request: fastapi.Request,
    response: fastapi.Response,
):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        result = await coordinator.acquire(
            body.driver_id,
            principal.id,
            client_id,
            mode=body.mode,
        )
    except Exception as error:
        await _raise_session_error(error)
        raise

    response.status_code = {
        'created': 201,
        'existing': 200,
        'preparing': 202,
        'confirmation_required': 202,
    }[result.disposition]
    return {
        'code': response.status_code,
        'data': {
            'disposition': result.disposition,
            'session': coordinator.public_session(result.session),
        },
    }


@router.post('/sessions/{session_id}/confirm-live')
async def confirm_live_session(
    session_id: str,
    body: LiveConfirmationRequest,
    request: fastapi.Request,
):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        result = await coordinator.confirm_live(
            session_id,
            principal.id,
            client_id,
            profile_id=body.profile_id,
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {
        'code': 200,
        'data': {
            'disposition': result.disposition,
            'session': coordinator.public_session(result.session),
        },
    }


@router.get('/sessions')
async def list_sessions(request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    sessions = await coordinator.sessions_for(principal.id, owner=_owner(principal))
    return {'code': 200, 'data': sessions}


@router.get('/sessions/{session_id}')
async def get_session(session_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    try:
        session = await coordinator.session_for(
            session_id,
            principal.id,
            owner=_owner(principal),
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {'code': 200, 'data': session}


@router.get('/sessions/{session_id}/driver-status')
async def get_driver_status(session_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    try:
        session = await coordinator.status(
            session_id,
            principal.id,
            owner=_owner(principal),
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {'code': 200, 'data': session}


@router.post('/sessions/{session_id}/heartbeat')
async def heartbeat_session(session_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        session = await coordinator.heartbeat(session_id, principal.id, client_id)
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {'code': 200, 'data': coordinator.public_session(session)}


@router.post('/sessions/{session_id}/signaling/offer')
async def signaling_offer(
    session_id: str,
    request: fastapi.Request,
    response: fastapi.Response,
):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        body = await _signaling_offer_request(request)
    except fastapi.HTTPException as error:
        detail = error.detail if isinstance(error.detail, dict) else {}
        code = detail.get('code')
        reason = code if isinstance(code, str) else 'invalid_signaling_offer'
        await audit.emit(
            'teleop.signaling.offer.rejected',
            session_id=session_id,
            principal_id=principal.id,
            source='api',
            decision='rejected',
            reason=reason,
        )
        raise
    try:
        answer = await coordinator.signaling_offer(
            session_id,
            principal.id,
            client_id,
            body.model_dump(),
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    response.headers['Cache-Control'] = 'no-store'
    return {'code': 200, 'data': answer}


@router.post('/sessions/{session_id}/pause')
async def pause_session(session_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        session = await coordinator.pause(
            session_id,
            principal.id,
            client_id,
            owner=_owner(principal),
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {'code': 200, 'data': coordinator.public_session(session)}


@router.post('/sessions/{session_id}/soft-stop')
async def soft_stop_session(session_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        session = await coordinator.soft_stop(
            session_id,
            principal.id,
            client_id,
            owner=_owner(principal),
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {'code': 200, 'data': coordinator.public_session(session)}


@router.delete('/sessions/{session_id}')
async def release_session(session_id: str, request: fastapi.Request):
    principal = auth.require_role(request, 'operator')
    client_id = _client_id(request)
    try:
        session, acknowledged = await coordinator.release(
            session_id,
            principal.id,
            client_id,
            owner=_owner(principal),
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    return {
        'code': 200,
        'data': {
            'session': coordinator.public_session(session),
            'driver_acknowledged': acknowledged,
        },
    }


@router.get('/sessions/{session_id}/events')
async def session_events(
    session_id: str,
    request: fastapi.Request,
    limit: int = fastapi.Query(default=50, ge=1, le=200),
):
    principal = auth.require_role(request, 'operator')
    try:
        session = await coordinator.manager.get_authorized(
            session_id,
            principal.id,
            owner=_owner(principal),
            include_terminal=True,
        )
    except Exception as error:
        await _raise_session_error(error, session_id)
        raise
    stored_events = await asyncio.to_thread(
        audit.list_events,
        limit,
        session.robot_id,
        session.id,
    )
    return {'code': 200, 'data': stored_events}


@ws_router.websocket('/ws/teleop-capture')
async def teleop_capture_websocket(websocket: fastapi.WebSocket):
    """Authenticate a capture in-band; never accept secrets in the URL or logs."""

    connection = None
    receive_task: asyncio.Task | None = None
    event_task: asyncio.Task | None = None
    await websocket.accept()
    if getattr(websocket, 'scope', {}).get('query_string', b''):
        await websocket.send_json({
            'type': 'error',
            'code': 'capture_query_forbidden',
        })
        await _close_capture_ws(
            websocket,
            code=4400,
            reason='capture_query_forbidden',
        )
        return
    try:
        raw_first = await asyncio.wait_for(
            _capture_ws_text(websocket),
            timeout=_CAPTURE_WS_AUTH_TIMEOUT_SECONDS,
        )
        first = _capture_ws_payload(raw_first, first_message=True)
        if isinstance(first, CapturePairMessage):
            connection = await coordinator.connect_capture_with_pairing(
                first.pairing_id,
                first.pairing_code,
                capture_protocol=first.capture_protocol,
                frame_protocol=first.frame_protocol,
                client_kind=first.client_kind,
                app_version=first.app_version,
            )
            assert connection.capture_credential is not None
            await websocket.send_json({
                'type': 'paired',
                'capture_id': connection.capture_id,
                'capture_credential': connection.capture_credential,
                'capture_protocol': first.capture_protocol,
                'frame_protocol': first.frame_protocol,
                'presence_interval_ms': 2_000,
                'presence_timeout_ms': 5_000,
            })
        elif isinstance(first, CaptureCredentialMessage):
            connection = await coordinator.connect_capture_with_credential(
                first.capture_id,
                first.capture_credential,
                capture_protocol=first.capture_protocol,
                frame_protocol=first.frame_protocol,
                client_kind=first.client_kind,
                app_version=first.app_version,
            )
            await websocket.send_json({
                'type': 'connected',
                'capture_id': connection.capture_id,
                'capture_protocol': first.capture_protocol,
                'frame_protocol': first.frame_protocol,
                'presence_interval_ms': 2_000,
                'presence_timeout_ms': 5_000,
            })
        else:  # pragma: no cover -- parser only returns the two exact models
            raise CaptureError('capture_message_invalid', 400)

        receive_task = asyncio.create_task(_capture_ws_text(websocket))
        event_task = asyncio.create_task(connection.events.get())
        while True:
            done, _ = await asyncio.wait(
                {receive_task, event_task},
                timeout=_CAPTURE_WS_HEALTH_TICK_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                stale = await coordinator.expire_capture_connection(
                    connection.capture_id,
                    connection.connection_id,
                )
                if stale:
                    await websocket.send_json({
                        'type': 'error',
                        'code': 'capture_presence_timeout',
                    })
                    await _close_capture_ws(
                        websocket,
                        code=4408,
                        reason='capture_presence_timeout',
                    )
                    return
                continue

            if event_task in done:
                event = event_task.result()
                await websocket.send_json(event)
                if event.get('type') in {'capture_revoked', 'capture_stale'}:
                    await _close_capture_ws(
                        websocket,
                        code=4403,
                        reason=str(event.get('reason', 'capture_revoked')),
                    )
                    return
                event_task = asyncio.create_task(connection.events.get())

            if receive_task in done:
                incoming = _capture_ws_payload(
                    receive_task.result(),
                    first_message=False,
                )
                if isinstance(incoming, CapturePresenceMessage):
                    presence = await coordinator.capture_presence(
                        connection.capture_id,
                        connection.connection_id,
                        state=incoming.state,
                        assignment_id=incoming.assignment_id,
                    )
                    await websocket.send_json({
                        'type': 'presence_ack',
                        'state': presence['observed_state'],
                    })
                elif isinstance(incoming, CaptureSignalingOfferMessage):
                    answer = await coordinator.capture_signaling_offer(
                        connection.capture_id,
                        connection.connection_id,
                        incoming.assignment_id,
                        incoming.offer.model_dump(),
                    )
                    await websocket.send_json({
                        'type': 'signaling_answer',
                        'assignment_id': incoming.assignment_id,
                        'answer': answer,
                    })
                else:  # pragma: no cover -- parser only returns the two exact models
                    raise CaptureError('capture_message_invalid', 400)
                receive_task = asyncio.create_task(_capture_ws_text(websocket))
    except fastapi.WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        # Test clients and some ASGI servers cancel an authenticated handler
        # to represent a normal peer disconnect. Once in-band authentication
        # succeeded, falling through to ``finally`` is the required revoke
        # path. Preserve cancellation before authentication completes.
        if connection is None:
            raise
    except Exception as error:  # noqa: BLE001 -- WSS returns only stable error codes
        code = _capture_ws_error_code(error)
        try:
            await websocket.send_json({'type': 'error', 'code': code})
        except Exception:  # noqa: BLE001, S110 -- peer may already be disconnected
            pass
        await _close_capture_ws(
            websocket,
            code=4401 if connection is None else 4400,
            reason=code,
        )
    finally:
        tasks = [
            task for task in (receive_task, event_task)
            if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        # Consume exceptions from both tasks even when a terminal event and a
        # peer disconnect become ready in the same wait cycle. Gathering only
        # pending tasks leaves the completed receive exception unobserved.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if connection is not None:
            await coordinator.disconnect_capture(
                connection.capture_id,
                connection.connection_id,
            )
