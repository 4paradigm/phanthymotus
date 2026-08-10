from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace

import aiohttp

import auth
import config
import mcp_client
from teleop import audit, authority_guard
from teleop.capture_manager import (
    CaptureAssignment,
    CaptureConnection,
    CaptureError,
    CaptureManager,
    CapturePairingResult,
)
from teleop.command_broker import (
    AuthorityAlreadyClaimed,
    CommandBroker,
    CommandDrainTimeout,
    InvalidAuthorityBinding,
    authority_domain_for_target,
)
from teleop.command_broker import broker as global_command_broker
from teleop.contracts import (
    LIVE_MODE,
    PREPARE_ACTION_BY_MODE,
    SHADOW_MODE,
    TeleopContractError,
    project_teleop_descriptor,
)
from teleop.models import ShadowSession
from teleop.session_manager import (
    MAX_DRIVER_EPOCH,
    MIN_LEASE_SECONDS,
    EpochExhausted,
    SessionClientMismatch,
    SessionConflict,
    SessionForbidden,
    SessionNotFound,
    SessionStateConflict,
    ShadowSessionManager,
    manager,
)

_AUTHORITY_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$')
_SAFE_COUNTER_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
_RECORDING_DISPATCH_CONTRACT = 'motus.teleop.dispatch.recording.v1'
_HARDWARE_DISPATCH_CONTRACT = 'motus.teleop.dispatch.hardware.v1'
_WEBRTC_SIGNALING_PROTOCOL = 'motus.teleop.webrtc-offer-answer.v1'
_WEBRTC_SIGNALING_PATH = '/offer'
_WEBRTC_SIGNALING_ACCESS = 'authenticated-core-proxy-only'
_MAX_WIRE_INTEGER = 2**63 - 1
_AUTHORITY_DRIVER_STATES = frozenset({
    'prepared_shadow', 'active_shadow', 'prepared_live', 'active_live', 'hold', 'paused',
})
_STATE_REASONS = {
    'idle': frozenset({None}),
    'prepared_shadow': frozenset({None}),
    'active_shadow': frozenset({None}),
    'prepared_live': frozenset({None}),
    'active_live': frozenset({None}),
    'paused': frozenset({'operator_pause'}),
    'hold': frozenset({
        'deadman_released',
        'command_timeout',
        'intent_expired',
        'lease_timeout',
        'pose_timeout',
        'rtc_closed',
        'rtc_disconnected',
        'rtc_failed',
        'rtc_not_ready',
        'soft_stop',
        'tracking_lost',
    }),
    'released': frozenset({'lifecycle_stop', 'operator_release'}),
    'fault': frozenset({'dispatch_fault'}),
}
_PUBLIC_COUNTERS = frozenset({
    'explicit_reclutches',
    'frames_accepted',
    'frames_held_without_reclutch',
    'frames_rejected',
    'lease_heartbeats',
    'lease_timeouts',
    'pose_timeouts',
    'protocol_errors',
    'rtc_disconnects',
    'sessions_prepared',
    'sessions_released',
    'soft_stops',
})
_PUBLIC_DISPATCH_COUNTERS = frozenset({
    'adapter_faults',
    'adapter_io_stalls',
    'dispatch_reclutches',
    'late_adapter_returns',
    'mailbox_cleared_by_stop',
    'mailbox_replacements',
    'motion_admitted',
    'motion_applied',
    'motion_dropped_expired',
    'motion_dropped_stale',
    'sessions_armed',
    'startup_safe_acks',
    'stop_acks',
    'stop_requests',
})
_DISPATCH_STATES = frozenset({
    'safe_unarmed',
    'safe_waiting_frame',
    'motion_eligible',
    'safe_reclutch_required',
    'safe_latched',
    'safe_revoked',
    'fault_latched',
})
_ACTIVE_DISPATCH_DECISIONS = frozenset({
    'admitted',
    'motion_committed',
    'published',
    'would_apply',
})
_STOP_DISPATCH_STATES = frozenset({
    'safe_reclutch_required',
    'safe_latched',
    'safe_revoked',
    'fault_latched',
})
_RETRYABLE_TRANSPORT_CODES = {'timeout', 'network_error'}
_TERMINAL_RPC_CODES = {
    'boot_mismatch',
    'epoch_mismatch',
    'fence_mismatch',
    'session_expired',
    'session_inactive',
    'session_mismatch',
}
_MIN_DRIVER_LEASE_SECONDS = 0.75
_MAX_DRIVER_LEASE_SECONDS = 10.0
_DEFAULT_CONTROL_TIMEOUT_SECONDS = 2.0
_MAX_HEARTBEAT_INTERVAL_SECONDS = 0.25
_REAPER_INTERVAL_SECONDS = 0.1
_LOCK_STRIPE_COUNT = 256
_COMMAND_DRAIN_TIMEOUT_SECONDS = 2.0
_SIGNALING_TIMEOUT_SECONDS = 2.0
_SIGNALING_TICKET_TTL_SECONDS = 20
_MAX_SIGNALING_SDP_BYTES = 128 * 1024
# Core Pause and HOLD are both non-resuming latches: Driver closes every peer
# and accepts no more Pose frames until release + a new prepare_shadow. Driver
# may report a recoverable transport HOLD while Core still remains active, so
# only the Core active state may open or replace an RTC peer.
_LIVE_SIGNALING_STATES = frozenset({'active'})
_CAPTURE_FAIL_CLOSE_REASONS = frozenset({
    'capture_browser_ready',
    'capture_disconnected',
    'capture_error',
    'capture_presence_timeout',
    'capture_signaling_failed',
    'capture_xr_ended',
    'capture_xr_standby',
})
_SIGNALING_ANSWER_FIELDS = frozenset({
    'actuation_enabled',
    'boot_id',
    'capability_digest',
    'epoch',
    'mode',
    'sdp',
    'session_id',
    'type',
})


class TeleopServiceError(RuntimeError):
    """Stable, secret-free error suitable for the dedicated REST API."""

    def __init__(self, code: str, status_code: int):
        self.code = code
        self.status_code = status_code
        super().__init__(f'teleop service error: {code}')


@dataclass(frozen=True)
class AcquireResult:
    session: ShadowSession
    disposition: str


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')


def _signaling_ticket(session: ShadowSession, sdp: str, *, wall_now: float) -> str:
    """Issue a Driver-compatible, one-use RTC offer ticket.

    The Driver owns replay consumption.  Core deliberately keeps the ticket and
    its private fence claim on the server-to-server hop.
    """

    # Focused tests and explicit process configuration may override the cached
    # deployment setting.  Production normally loads the same value from
    # /opt/phanthy-motus/.env during auth.init().
    process_secret = os.environ.get('MOTUS_TELEOP_TICKET_SECRET')
    try:
        secret_bytes = (
            auth.teleop_ticket_secret(session.driver_id)
            if auth.has_dedicated_teleop_ticket_secret(session.driver_id)
            else (
                process_secret.encode('utf-8')
                if process_secret is not None
                else auth.teleop_ticket_secret(session.driver_id)
            )
        )
    except UnicodeEncodeError:
        raise TeleopServiceError('teleop_signaling_unavailable', 503) from None
    if len(secret_bytes) < 32:
        raise TeleopServiceError('teleop_signaling_unavailable', 503)
    if (
        not isinstance(sdp, str)
        or not sdp
        or not isinstance(wall_now, (int, float))
        or isinstance(wall_now, bool)
        or not math.isfinite(float(wall_now))
        or wall_now < 0
    ):
        raise TeleopServiceError('teleop_signaling_unavailable', 503)

    issued_at = int(wall_now)
    claims = {
        'v': 1,
        'aud': session.signaling_audience,
        'boot_id': session.boot_id,
        'session_id': session.id,
        'epoch': session.epoch,
        'fence': session.fence,
        'capability_digest': session.capability_digest,
        'sdp_sha256': hashlib.sha256(sdp.encode('utf-8')).hexdigest(),
        'iat': issued_at,
        'exp': issued_at + _SIGNALING_TICKET_TTL_SECONDS,
        'jti': _base64url(uuid.uuid4().bytes),
    }
    try:
        canonical = json.dumps(
            claims,
            ensure_ascii=False,
            allow_nan=False,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')
    except (TypeError, ValueError, RecursionError):
        raise TeleopServiceError('teleop_signaling_unavailable', 503) from None
    payload = _base64url(canonical)
    signature = _base64url(
        hmac.new(secret_bytes, payload.encode('ascii'), hashlib.sha256).digest(),
    )
    return f'{payload}.{signature}'


def _validated_signaling_offer_sdp(offer: object) -> str:
    if (
        not isinstance(offer, dict)
        or set(offer) != {'sdp', 'type'}
        or offer.get('type') != 'offer'
        or not isinstance(offer.get('sdp'), str)
        or not offer['sdp']
    ):
        raise TeleopServiceError('invalid_signaling_offer', 400)
    try:
        offer_sdp_size = len(offer['sdp'].encode('utf-8'))
    except UnicodeEncodeError:
        raise TeleopServiceError('invalid_signaling_offer', 400) from None
    if offer_sdp_size > _MAX_SIGNALING_SDP_BYTES:
        raise TeleopServiceError('invalid_signaling_offer', 400)
    return offer['sdp']


def _project_signaling_answer(
    raw: object,
    session: ShadowSession,
    ticket: str,
) -> dict:
    if not isinstance(raw, dict) or set(raw) != _SIGNALING_ANSWER_FIELDS:
        raise TeleopServiceError('driver_protocol_error', 502)
    sdp = raw.get('sdp')
    try:
        sdp_size = len(sdp.encode('utf-8')) if isinstance(sdp, str) else -1
    except UnicodeEncodeError:
        sdp_size = -1
    if (
        raw.get('type') != 'answer'
        or not isinstance(sdp, str)
        or not sdp
        or sdp_size < 1
        or sdp_size > _MAX_SIGNALING_SDP_BYTES
        or raw.get('boot_id') != session.boot_id
        or raw.get('session_id') != session.id
        or isinstance(raw.get('epoch'), bool)
        or not isinstance(raw.get('epoch'), int)
        or raw.get('epoch') != session.epoch
        or raw.get('capability_digest') != session.capability_digest
        or raw.get('mode') != session.mode
        or raw.get('actuation_enabled') is not (session.mode == LIVE_MODE)
        or session.fence in sdp
        or ticket in sdp
    ):
        raise TeleopServiceError('driver_protocol_error', 502)
    projection = {'sdp': sdp, 'type': 'answer'}
    if session.fence in repr(projection):
        raise TeleopServiceError('driver_protocol_error', 502)
    return projection


@dataclass
class _HeartbeatHealth:
    last_confirmed_monotonic: float
    last_confirmed_at: float
    consecutive_failures: int = 0
    state: str = 'healthy'

    def public_dict(self) -> dict:
        return {
            'state': self.state,
            'last_confirmed_at': self.last_confirmed_at,
            'consecutive_failures': self.consecutive_failures,
        }


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise TeleopServiceError('driver_response_invalid', 502)
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise TeleopServiceError('driver_response_invalid', 502) from None
    if str(parsed) != value.lower():
        raise TeleopServiceError('driver_response_invalid', 502)
    return str(parsed)


def _strict_int(
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    return value


def _finite_number(value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeleopServiceError('driver_response_invalid', 502)
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise TeleopServiceError('driver_response_invalid', 502) from None
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise TeleopServiceError('driver_response_invalid', 502)
    return number


def _bounded_text(value: object, maximum: int = 256) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TeleopServiceError('driver_response_invalid', 502)
    return value[:maximum]


def _driver_descriptor(driver_id: str) -> dict:
    services = config.main.get('services', {})
    entries = services.get('mcp', []) if isinstance(services, dict) else []
    matches = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get('id') == driver_id
    ]
    if len(matches) != 1:
        raise TeleopServiceError('driver_not_found', 404)
    target = matches[0]
    tools = target.get('tools')
    session_tools = [
        tool for tool in tools or []
        if isinstance(tool, dict) and tool.get('name') == 'teleop_session'
    ]
    try:
        descriptor = project_teleop_descriptor(
            session_tools[0] if len(session_tools) == 1 else None,
            expected_driver_id=driver_id,
        )
    except TeleopContractError:
        raise TeleopServiceError('driver_not_ready', 503) from None
    descriptor_robot_id = descriptor['robot_id']
    reported_robot_id = target.get('reported_robot_id')
    try:
        robot_id = authority_domain_for_target(
            driver_id,
            target,
            targets=entries,
        )
    except InvalidAuthorityBinding:
        raise TeleopServiceError('driver_not_ready', 503) from None
    if not auth.teleop_ticket_credential_available(driver_id):
        raise TeleopServiceError('teleop_signaling_unavailable', 503)
    if (
        target.get('trust_state') != 'trusted'
        or not auth.driver_record_credential_available(
            driver_id,
            target.get('credential_binding'),
        )
        or target.get('transport') != 'http'
        or not isinstance(robot_id, str)
        or not _AUTHORITY_ID_RE.fullmatch(robot_id)
        or (robot_id != driver_id and descriptor_robot_id != robot_id)
        or (
            reported_robot_id is not None
            and reported_robot_id != ''
            and reported_robot_id != robot_id
        )
        or (
            descriptor_robot_id is not None
            and descriptor_robot_id != robot_id
        )
    ):
        raise TeleopServiceError('driver_not_ready', 503)
    return {
        **descriptor,
        'robot_id': robot_id,
    }


def _safe_submapping(value: object, allowed: Mapping[str, str]) -> dict:
    if not isinstance(value, dict):
        raise TeleopServiceError('driver_response_invalid', 502)
    result = {}
    for key, kind in allowed.items():
        item = value.get(key)
        if kind == 'bool':
            if not isinstance(item, bool):
                raise TeleopServiceError('driver_response_invalid', 502)
            result[key] = item
        elif kind == 'number_or_none':
            if item is None:
                result[key] = None
            else:
                result[key] = _finite_number(item, minimum=0.0, maximum=86_400_000.0)
        elif kind == 'int_or_none':
            result[key] = None if item is None else _strict_int(item)
        elif kind == 'text':
            result[key] = _bounded_text(item)
        else:
            raise RuntimeError('invalid projection schema')
    return result


def _expected_dispatch_state(state: str, reason: str | None) -> str:
    if state == 'idle':
        return 'safe_unarmed'
    if state in {'prepared_shadow', 'prepared_live'}:
        return 'safe_waiting_frame'
    if state in {'active_shadow', 'active_live'}:
        return 'motion_eligible'
    if state == 'paused' or (state == 'hold' and reason == 'soft_stop'):
        return 'safe_latched'
    if state == 'hold' and reason == 'lease_timeout':
        return 'safe_revoked'
    if state == 'hold':
        return 'safe_reclutch_required'
    if state == 'released':
        return 'safe_revoked'
    if state == 'fault' and reason == 'dispatch_fault':
        return 'fault_latched'
    raise TeleopServiceError('driver_response_invalid', 502)


def _terminal_fault_decision_valid(
    last_decision: str,
    *,
    stop_acknowledged: bool,
) -> bool:
    if stop_acknowledged:
        return last_decision == 'would_stop:adapter_fault'
    return last_decision.startswith((
        'apply_failed:',
        'async_fault:',
        'io_stalled:',
        'stop_failed:',
    ))


def _project_dry_run_adapter(
    value: object,
) -> dict:
    """Project only adapter-neutral recording evidence.

    Core deliberately does not decode vendor SDK calls, joint mappings, gait
    policy, or robot-specific limits. Those remain Driver-owned diagnostics.
    """

    if not isinstance(value, dict):
        raise TeleopServiceError('driver_response_invalid', 502)
    kind = value.get('kind')
    if kind == 'recording':
        current = value.get('current')
        records = value.get('records')
        if (
            value.get('closed') is not False
            or (
                'hardware_output' in value
                and value.get('hardware_output') is not False
            )
            or (
                'actuation_enabled' in value
                and value.get('actuation_enabled') is not False
            )
            or not isinstance(current, dict)
            or not isinstance(current.get('kind'), str)
            or current.get('kind') not in {'safe', 'would_apply', 'would_stop'}
            or not isinstance(records, list)
            or len(records) > 64
        ):
            raise TeleopServiceError('driver_response_invalid', 502)
        return {
            'profile': 'recording',
            'hardware_output': False,
            'actuation_enabled': False,
            'operation': 'recording',
            'sequence': None,
            'requested': None,
            'effective': None,
            'clamped_axes': [],
            'stop_reason': None,
            'gait_policy': None,
        }
    if kind == 'unavailable':
        reason = value.get('reason')
        if not isinstance(reason, str) or reason not in {
            'startup_in_progress',
            'invalid_adapter_snapshot',
            'adapter_snapshot_exception',
        }:
            raise TeleopServiceError('driver_response_invalid', 502)
        return {
            'profile': 'unavailable',
            'hardware_output': None,
            'actuation_enabled': None,
            'operation': 'unavailable',
            'sequence': None,
            'requested': None,
            'effective': None,
            'clamped_axes': [],
            'stop_reason': reason,
            'gait_policy': None,
        }
    raise TeleopServiceError('driver_response_invalid', 502)


def _project_recording_dispatch(
    value: object,
    *,
    state: str,
    reason: str | None,
    allow_stop_pending: bool,
) -> dict:
    if not isinstance(value, dict) or value.get('kind') != 'recording':
        raise TeleopServiceError('driver_response_invalid', 502)
    dispatch_state = value.get('state')
    ready = value.get('ready')
    stop_acknowledged = value.get('stop_acknowledged')
    io_inflight = value.get('io_inflight')
    fault_code = value.get('fault_code')
    last_decision = value.get('last_decision')
    if (
        not isinstance(dispatch_state, str)
        or dispatch_state not in _DISPATCH_STATES
        or not isinstance(ready, bool)
        or not isinstance(stop_acknowledged, bool)
        or (
            io_inflight is not None
            and (
                not isinstance(io_inflight, str)
                or io_inflight not in {'apply', 'safe_stop'}
            )
        )
        or (
            fault_code is not None
            and (
                not isinstance(fault_code, str)
                or not 1 <= len(fault_code) <= 96
            )
        )
        or not isinstance(last_decision, str)
        or not 1 <= len(last_decision) <= 96
    ):
        raise TeleopServiceError('driver_response_invalid', 502)

    generation = _strict_int(value.get('generation'), maximum=_MAX_WIRE_INTEGER)
    mailbox_depth = _strict_int(value.get('mailbox_depth'), maximum=1)
    stop_queue_depth = _strict_int(value.get('stop_queue_depth'), maximum=64)
    admitted_raw = value.get('last_admitted_sequence')
    applied_raw = value.get('last_would_apply_sequence')
    admitted = (
        None
        if admitted_raw is None
        else _strict_int(admitted_raw, maximum=_MAX_WIRE_INTEGER)
    )
    applied = (
        None
        if applied_raw is None
        else _strict_int(applied_raw, maximum=_MAX_WIRE_INTEGER)
    )
    if applied is not None and (admitted is None or applied > admitted):
        raise TeleopServiceError('driver_response_invalid', 502)

    expected_state = _expected_dispatch_state(state, reason)
    if dispatch_state != expected_state:
        raise TeleopServiceError('driver_response_invalid', 502)
    terminal_fault = state == 'fault' and reason == 'dispatch_fault'
    if terminal_fault:
        if (
            ready is not False
            or mailbox_depth != 0
            or stop_queue_depth != 0
            or io_inflight is not None
            or fault_code is None
            or not _terminal_fault_decision_valid(
                last_decision,
                stop_acknowledged=stop_acknowledged,
            )
        ):
            raise TeleopServiceError('driver_response_invalid', 502)
    elif ready is not True or fault_code is not None:
        raise TeleopServiceError('driver_response_invalid', 502)
    stop_pending = (
        not terminal_fault
        and allow_stop_pending
        and reason is not None
        and expected_state != 'motion_eligible'
        and stop_acknowledged is False
        and last_decision == f'stop_requested:{reason}'
    )
    if terminal_fault:
        pass
    elif stop_pending:
        if mailbox_depth != 0 or io_inflight not in {None, 'safe_stop'}:
            raise TeleopServiceError('driver_response_invalid', 502)
    elif stop_acknowledged is not True or stop_queue_depth != 0:
        raise TeleopServiceError('driver_response_invalid', 502)
    elif expected_state == 'motion_eligible':
        if io_inflight not in {None, 'apply'} or admitted is None:
            raise TeleopServiceError('driver_response_invalid', 502)
        if last_decision not in _ACTIVE_DISPATCH_DECISIONS:
            raise TeleopServiceError('driver_response_invalid', 502)
    else:
        if mailbox_depth != 0 or io_inflight is not None:
            raise TeleopServiceError('driver_response_invalid', 502)
        if expected_state == 'safe_unarmed':
            expected_decision = 'startup_safe_ack'
        elif expected_state == 'safe_waiting_frame':
            expected_decision = 'prepared_after_stop_ack'
        else:
            expected_decision = f'would_stop:{reason}'
        if last_decision != expected_decision:
            raise TeleopServiceError('driver_response_invalid', 502)
    if expected_state in {'safe_unarmed', 'safe_waiting_frame'} and (
        admitted is not None or applied is not None
    ):
        raise TeleopServiceError('driver_response_invalid', 502)

    counters_raw = value.get('counters')
    if not isinstance(counters_raw, dict):
        raise TeleopServiceError('driver_response_invalid', 502)
    counters = {}
    for key in _PUBLIC_DISPATCH_COUNTERS:
        if key not in counters_raw:
            continue
        counter = counters_raw[key]
        counters[key] = _strict_int(counter, maximum=_MAX_WIRE_INTEGER)

    dry_run = _project_dry_run_adapter(value.get('adapter'))

    return {
        'contract': _RECORDING_DISPATCH_CONTRACT,
        'kind': 'recording',
        'state': dispatch_state,
        'ready': ready,
        'generation': generation,
        'mailbox_depth': mailbox_depth,
        'stop_queue_depth': stop_queue_depth,
        'last_admitted_sequence': admitted,
        'last_would_apply_sequence': applied,
        'last_decision': last_decision,
        'stop_acknowledged': stop_acknowledged,
        'fault_code': fault_code,
        'io_inflight': io_inflight,
        'counters': dict(sorted(counters.items())),
        'dry_run': dry_run,
    }


def _project_hardware_dispatch(
    value: object,
    *,
    state: str,
    reason: str | None,
    allow_stop_pending: bool,
) -> dict:
    if not isinstance(value, dict) or value.get('kind') != 'hardware':
        raise TeleopServiceError('driver_response_invalid', 502)
    dispatch_state = value.get('state')
    ready = value.get('ready')
    stop_acknowledged = value.get('stop_acknowledged')
    io_inflight = value.get('io_inflight')
    last_decision = value.get('last_decision')
    fault_code = value.get('fault_code')
    if (
        dispatch_state not in _DISPATCH_STATES
        or not isinstance(ready, bool)
        or not isinstance(stop_acknowledged, bool)
        or (
            io_inflight is not None
            and io_inflight not in {'apply', 'safe_stop'}
        )
        or not isinstance(last_decision, str)
        or not 1 <= len(last_decision) <= 96
        or (
            fault_code is not None
            and (
                not isinstance(fault_code, str)
                or not 1 <= len(fault_code) <= 96
            )
        )
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    generation = _strict_int(value.get('generation'), maximum=_MAX_WIRE_INTEGER)
    mailbox_depth = _strict_int(value.get('mailbox_depth'), maximum=1)
    stop_queue_depth = _strict_int(value.get('stop_queue_depth'), maximum=64)
    admitted_raw = value.get('last_admitted_sequence')
    if (
        'last_published_sequence' not in value
        or 'last_applied_sequence' in value
        or 'last_would_apply_sequence' in value
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    published_raw = value.get('last_published_sequence')
    admitted = None if admitted_raw is None else _strict_int(
        admitted_raw,
        maximum=_MAX_WIRE_INTEGER,
    )
    published = None if published_raw is None else _strict_int(
        published_raw,
        maximum=_MAX_WIRE_INTEGER,
    )
    if published is not None and (admitted is None or published > admitted):
        raise TeleopServiceError('driver_response_invalid', 502)

    expected_state = _expected_dispatch_state(state, reason)
    if dispatch_state != expected_state:
        raise TeleopServiceError('driver_response_invalid', 502)
    terminal_fault = state == 'fault' and reason == 'dispatch_fault'
    if terminal_fault:
        if (
            ready is not False
            or mailbox_depth != 0
            or stop_queue_depth != 0
            or io_inflight is not None
            or fault_code is None
            or not _terminal_fault_decision_valid(
                last_decision,
                stop_acknowledged=stop_acknowledged,
            )
        ):
            raise TeleopServiceError('driver_response_invalid', 502)
    elif ready is not True or fault_code is not None:
        raise TeleopServiceError('driver_response_invalid', 502)
    stop_pending = (
        not terminal_fault
        and allow_stop_pending
        and reason is not None
        and expected_state != 'motion_eligible'
        and stop_acknowledged is False
        and last_decision == f'stop_requested:{reason}'
    )
    if terminal_fault:
        pass
    elif stop_pending:
        if mailbox_depth != 0 or io_inflight not in {None, 'safe_stop'}:
            raise TeleopServiceError('driver_response_invalid', 502)
    elif stop_acknowledged is not True or stop_queue_depth != 0:
        raise TeleopServiceError('driver_response_invalid', 502)
    elif expected_state == 'motion_eligible':
        if io_inflight not in {None, 'apply'} or admitted is None:
            raise TeleopServiceError('driver_response_invalid', 502)
        if last_decision not in _ACTIVE_DISPATCH_DECISIONS:
            raise TeleopServiceError('driver_response_invalid', 502)
    elif mailbox_depth != 0 or io_inflight is not None:
        raise TeleopServiceError('driver_response_invalid', 502)

    counters_raw = value.get('counters')
    if not isinstance(counters_raw, dict):
        raise TeleopServiceError('driver_response_invalid', 502)
    counters = {
        key: _strict_int(counters_raw[key], maximum=_MAX_WIRE_INTEGER)
        for key in _PUBLIC_DISPATCH_COUNTERS
        if key in counters_raw
    }
    return {
        'contract': _HARDWARE_DISPATCH_CONTRACT,
        'kind': 'hardware',
        'state': dispatch_state,
        'ready': ready,
        'generation': generation,
        'mailbox_depth': mailbox_depth,
        'stop_queue_depth': stop_queue_depth,
        'last_admitted_sequence': admitted,
        'last_published_sequence': published,
        'last_decision': last_decision,
        'stop_acknowledged': stop_acknowledged,
        'fault_code': fault_code,
        'io_inflight': io_inflight,
        'counters': dict(sorted(counters.items())),
    }


_TRANSPORT_DIAGNOSTIC_FIELDS = {
    'rtc_rtt_ms': 'number_or_none',
    'pose_age_ms': 'number_or_none',
    'frame_rate_hz': 'number_or_none',
    'frames_received': 'int',
    'frames_rejected': 'int',
    'sequence_gaps': 'int',
    'mailbox_replacements': 'int',
}
_LATENCY_STAGES = (
    'receive_to_admit',
    'mailbox_wait',
    'ik',
    'adapter_apply',
    'robot_follow',
)


def _project_diagnostics(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {'transport', 'latency_ms'}:
        raise TeleopServiceError('driver_response_invalid', 502)
    transport_raw = value.get('transport')
    if not isinstance(transport_raw, dict) or set(transport_raw) != set(
        _TRANSPORT_DIAGNOSTIC_FIELDS,
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    transport: dict[str, int | float | None] = {}
    for key, kind in _TRANSPORT_DIAGNOSTIC_FIELDS.items():
        item = transport_raw[key]
        if kind == 'int':
            transport[key] = _strict_int(item, maximum=_MAX_WIRE_INTEGER)
        else:
            transport[key] = None if item is None else _finite_number(
                item,
                minimum=0.0,
                maximum=86_400_000.0,
            )

    latency_raw = value.get('latency_ms')
    if not isinstance(latency_raw, dict) or set(latency_raw) != set(_LATENCY_STAGES):
        raise TeleopServiceError('driver_response_invalid', 502)
    latency: dict[str, dict] = {}
    for stage in _LATENCY_STAGES:
        sample = latency_raw[stage]
        if not isinstance(sample, dict) or set(sample) != {
            'last', 'p50', 'p95', 'p99', 'count',
        }:
            raise TeleopServiceError('driver_response_invalid', 502)
        count = _strict_int(sample['count'], maximum=_MAX_WIRE_INTEGER)
        timings = {
            key: None if sample[key] is None else _finite_number(
                sample[key],
                minimum=0.0,
                maximum=86_400_000.0,
            )
            for key in ('last', 'p50', 'p95', 'p99')
        }
        percentiles = [timings[key] for key in ('p50', 'p95', 'p99')]
        if (
            count == 0
            and any(item is not None for item in timings.values())
        ) or (
            count > 0
            and any(item is None for item in timings.values())
        ) or (
            all(item is not None for item in percentiles)
            and percentiles != sorted(percentiles)
        ):
            raise TeleopServiceError('driver_response_invalid', 502)
        latency[stage] = {**timings, 'count': count}
    return {'transport': transport, 'latency_ms': latency}


def _project_joint_vector(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) > 128:
        raise TeleopServiceError('driver_response_invalid', 502)
    return [
        _finite_number(item, minimum=-1_000.0, maximum=1_000.0)
        for item in value
    ]


def _project_output(
    value: object,
    *,
    mode: str,
    profile_id: str,
    capabilities: Mapping[str, object],
    driver_state: str,
) -> dict:
    expected_fields = {
        'profile_id',
        'hardware_output',
        'state',
        'target_joint_positions_rad',
        'measured_joint_positions_rad',
        'max_abs_error_rad',
        'arm_sdk_weight',
        'command_age_ms',
        'fault_reason',
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise TeleopServiceError('driver_response_invalid', 502)
    state = value.get('state')
    fault_reason = value.get('fault_reason')
    if (
        value.get('profile_id') != profile_id
        or value.get('hardware_output') is not (mode == LIVE_MODE)
        or not isinstance(state, str)
        or not _SAFE_COUNTER_RE.fullmatch(state)
        or (
            fault_reason is not None
            and (not isinstance(fault_reason, str) or not 1 <= len(fault_reason) <= 256)
        )
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    target = _project_joint_vector(value.get('target_joint_positions_rad'))
    measured = _project_joint_vector(value.get('measured_joint_positions_rad'))
    if len(target) != len(measured):
        raise TeleopServiceError('driver_response_invalid', 502)
    outputs = capabilities.get('outputs')
    if not isinstance(outputs, Mapping):
        raise TeleopServiceError('driver_response_invalid', 502)
    declared_joint_count = 0
    for output in outputs.values():
        if not isinstance(output, Mapping):
            raise TeleopServiceError('driver_response_invalid', 502)
        if output.get('enabled') is True:
            count = output.get('joint_count', 0)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TeleopServiceError('driver_response_invalid', 502)
            declared_joint_count += count
    allowed_counts = (
        {declared_joint_count}
        if driver_state in _AUTHORITY_DRIVER_STATES or driver_state == 'fault'
        else {0, declared_joint_count}
    )
    if len(target) not in allowed_counts:
        raise TeleopServiceError('driver_response_invalid', 502)
    nullable_numbers = {}
    for key, maximum in (
        ('max_abs_error_rad', 1_000.0),
        ('arm_sdk_weight', 1.0),
        ('command_age_ms', 86_400_000.0),
    ):
        item = value.get(key)
        nullable_numbers[key] = None if item is None else _finite_number(
            item,
            minimum=0.0,
            maximum=maximum,
        )
    if (
        mode == SHADOW_MODE
        and nullable_numbers['arm_sdk_weight'] is not None
    ) or (
        mode == LIVE_MODE
        and nullable_numbers['arm_sdk_weight'] is None
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    if driver_state == 'fault' and (
        state != 'fault'
        or fault_reason is None
        or (mode == LIVE_MODE and nullable_numbers['arm_sdk_weight'] != 0.0)
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    return {
        'profile_id': profile_id,
        'hardware_output': mode == LIVE_MODE,
        'state': state,
        'target_joint_positions_rad': target,
        'measured_joint_positions_rad': measured,
        **nullable_numbers,
        'fault_reason': fault_reason,
    }


def _project_driver_snapshot(
    raw: Mapping[str, object],
    *,
    driver_id: str,
    robot_id: str,
    capability_digest: str,
    action: str,
    session: ShadowSession | None = None,
    expected_mode: str | None = None,
    expected_profile_id: str | None = None,
    expected_capabilities: Mapping[str, object] | None = None,
    expected_dry_run_profile: str | None = None,
    allow_stop_pending: bool = False,
) -> tuple[dict, float]:
    """Validate one trusted Driver response into an adapter-neutral projection."""

    if not isinstance(raw, dict):
        raise TeleopServiceError('driver_response_invalid', 502)
    mode = session.mode if session is not None else (expected_mode or SHADOW_MODE)
    profile_id = (
        session.profile_id
        if session is not None
        else (expected_profile_id or expected_dry_run_profile or 'recording')
    )
    capabilities = (
        session.capabilities
        if session is not None
        else dict(expected_capabilities or {})
    )
    driver_type = raw.get('driver_type')
    if (
        raw.get('driver_id') != driver_id
        or (
            raw.get('robot_id') != robot_id
            and not (raw.get('robot_id') is None and robot_id == driver_id)
        )
        or driver_type not in {'teleop-shadow', 'teleop'}
        or (driver_type == 'teleop-shadow' and mode != SHADOW_MODE)
        or raw.get('mode') != mode
        or raw.get('actuation_enabled') is not (mode == LIVE_MODE)
        or raw.get('capability_digest') != capability_digest
        or not isinstance(raw.get('authority_valid'), bool)
    ):
        raise TeleopServiceError('driver_response_invalid', 502)

    boot_id = _canonical_uuid(raw.get('boot_id'))
    epoch = _strict_int(raw.get('epoch'), maximum=MAX_DRIVER_EPOCH)
    raw_session_id = raw.get('session_id')
    session_id = None if raw_session_id is None else _canonical_uuid(raw_session_id)
    state = raw.get('state')
    reason = raw.get('reason')
    if (
        not isinstance(state, str)
        or (reason is not None and not isinstance(reason, str))
        or state not in _STATE_REASONS
        or reason not in _STATE_REASONS[state]
        or (state == 'fault' and driver_type != 'teleop')
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    authority_valid = raw['authority_valid']

    lease = _safe_submapping(raw.get('lease'), {
        'source': 'text',
        'timeout_ms': 'number_or_none',
        'age_ms': 'number_or_none',
        'fresh': 'bool',
        'authority_valid': 'bool',
        'expired_latched': 'bool',
    })
    if lease['source'] != 'agent-core-mcp-heartbeat-only':
        raise TeleopServiceError('driver_response_invalid', 502)
    timeout_ms = lease['timeout_ms']
    if timeout_ms is None:
        raise TeleopServiceError('driver_response_invalid', 502)
    driver_lease_seconds = timeout_ms / 1000.0
    if not _MIN_DRIVER_LEASE_SECONDS <= driver_lease_seconds <= _MAX_DRIVER_LEASE_SECONDS:
        raise TeleopServiceError('driver_lease_unsafe', 502)
    if lease['authority_valid'] is not authority_valid:
        raise TeleopServiceError('driver_response_invalid', 502)
    lease_age = lease['age_ms']
    expected_fresh = lease_age is not None and lease_age <= timeout_ms
    if lease['fresh'] is not expected_fresh:
        raise TeleopServiceError('driver_response_invalid', 502)

    if authority_valid:
        if (
            session_id is None
            or state not in _AUTHORITY_DRIVER_STATES
            or lease['fresh'] is not True
            or lease['expired_latched'] is not False
        ):
            raise TeleopServiceError('driver_response_invalid', 502)
    else:
        terminal_consistent = (
            state in {'idle', 'released'}
            and lease['expired_latched'] is False
        ) or (
            state == 'hold'
            and reason == 'lease_timeout'
            and lease['expired_latched'] is True
        ) or (
            state == 'fault'
            and reason == 'dispatch_fault'
            and lease['expired_latched'] is False
            and lease['age_ms'] is None
        )
        if session_id is not None or lease['fresh'] is not False or not terminal_consistent:
            raise TeleopServiceError('driver_response_invalid', 502)

    if session is not None:
        identity_changed = boot_id != session.boot_id or epoch != session.epoch
        if action != 'release' and state != 'fault':
            identity_changed = identity_changed or session_id != session.id
        if identity_changed:
            raise TeleopServiceError('driver_identity_changed', 409)

    prepare_action = PREPARE_ACTION_BY_MODE[mode]
    prepared_state = f'prepared_{mode}'
    active_state = f'active_{mode}'
    if action == prepare_action:
        if (
            state != prepared_state
            or reason is not None
            or authority_valid is not True
            or lease['fresh'] is not True
            or lease['expired_latched'] is not False
        ):
            raise TeleopServiceError('driver_prepare_rejected', 502)
    elif action == 'heartbeat' and state != 'fault':
        if (
            state not in {prepared_state, active_state, 'hold', 'paused'}
            or authority_valid is not True
            or lease['fresh'] is not True
        ):
            raise TeleopServiceError('driver_session_lost', 409)
    elif action == 'pause':
        if state != 'paused' or reason != 'operator_pause' or authority_valid is not True:
            raise TeleopServiceError('driver_pause_rejected', 502)
    elif action == 'soft_stop':
        if state != 'hold' or reason != 'soft_stop' or authority_valid is not True:
            raise TeleopServiceError('driver_soft_stop_rejected', 502)
    elif (
        action == 'release'
        and (
            state != 'released'
            or reason not in {'operator_release', 'lifecycle_stop'}
            or authority_valid is not False
            or session_id is not None
        )
    ):
        raise TeleopServiceError('driver_release_invalid', 502)

    if mode == SHADOW_MODE:
        dispatch = _project_recording_dispatch(
            raw.get('dispatch'),
            state=state,
            reason=reason,
            allow_stop_pending=allow_stop_pending,
        )
    else:
        dispatch = _project_hardware_dispatch(
            raw.get('dispatch'),
            state=state,
            reason=reason,
            allow_stop_pending=allow_stop_pending,
        )

    pose = _safe_submapping(raw.get('pose'), {
        'timeout_ms': 'number_or_none',
        'age_ms': 'number_or_none',
        'fresh': 'bool',
        'latest_sequence': 'int_or_none',
    })
    rtc_raw = raw.get('rtc')
    if not isinstance(rtc_raw, dict) or not isinstance(rtc_raw.get('channels'), dict):
        raise TeleopServiceError('driver_response_invalid', 502)
    if (
        not isinstance(rtc_raw.get('connected'), bool)
        or rtc_raw.get('renews_lease') is not False
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    channels = rtc_raw['channels']
    if set(channels) != {'teleop-control', 'teleop-pose'} or not all(
        isinstance(item, bool) for item in channels.values()
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    if rtc_raw['connected'] is not all(channels.values()):
        raise TeleopServiceError('driver_response_invalid', 502)
    rtc = {
        'connected': rtc_raw['connected'],
        'channels': dict(channels),
        'renews_lease': False,
    }
    if state == 'fault' and (
        pose['age_ms'] is not None
        or pose['fresh'] is not False
        or pose['latest_sequence'] is not None
        or rtc['connected'] is not False
        or any(rtc['channels'].values())
    ):
        raise TeleopServiceError('driver_response_invalid', 502)
    counters_raw = raw.get('counters')
    counters: dict[str, int] = {}
    if isinstance(counters_raw, dict):
        for key, value in list(counters_raw.items())[:100]:
            if (
                isinstance(key, str)
                and key in _PUBLIC_COUNTERS
                and _SAFE_COUNTER_RE.fullmatch(key)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 2**63 - 1
            ):
                counters[key] = value

    projection = {
        'driver_id': driver_id,
        'driver_name': driver_id,
        'robot_id': robot_id,
        'mode': mode,
        'actuation_enabled': mode == LIVE_MODE,
        'profile_id': profile_id,
        'capabilities': capabilities,
        'boot_id': boot_id,
        'session_id': session_id,
        'epoch': epoch,
        'state': state,
        'reason': reason,
        'authority_valid': authority_valid,
        'capability_digest': capability_digest,
        'lease': lease,
        'pose': pose,
        'rtc': rtc,
        'counters': counters,
    }
    projection['dispatch'] = dispatch
    if driver_type == 'teleop':
        projection['diagnostics'] = _project_diagnostics(raw.get('diagnostics'))
        output = _project_output(
            raw.get('output'),
            mode=mode,
            profile_id=profile_id,
            capabilities=capabilities,
            driver_state=state,
        )
        if state == 'fault' and output['fault_reason'] != dispatch['fault_code']:
            raise TeleopServiceError('driver_response_invalid', 502)
        projection['output'] = output
    else:
        projection['diagnostics'] = None
        projection['output'] = {
            'profile_id': profile_id,
            'hardware_output': False,
            'state': dispatch['state'],
            'target_joint_positions_rad': [],
            'measured_joint_positions_rad': [],
            'max_abs_error_rad': None,
            'arm_sdk_weight': None,
            'command_age_ms': None,
            'fault_reason': dispatch.get('fault_code'),
        }
    if session is not None and session.fence in repr(projection):
        raise TeleopServiceError('driver_secret_reflected', 502)
    return projection, driver_lease_seconds


class TeleopCoordinator:
    def __init__(
        self,
        *,
        session_manager: ShadowSessionManager = manager,
        caller: Callable[..., Awaitable[dict]] = mcp_client.call_trusted_shadow_session,
        signaler: Callable[..., Awaitable[dict]] = mcp_client.call_trusted_shadow_offer,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        command_broker: CommandBroker = global_command_broker,
        capture_manager: CaptureManager | None = None,
    ):
        self.manager = session_manager
        self.command_broker = command_broker
        self._caller = caller
        self._signaler = signaler
        self._uses_pinned_targets = caller is mcp_client.call_trusted_shadow_session
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self.capture_manager = capture_manager or CaptureManager(
            monotonic=monotonic,
            wall_clock=wall_clock,
        )
        self._http_session: aiohttp.ClientSession | None = None
        self._heartbeat_tasks: dict[str, asyncio.Task] = {}
        self._supervisor_tasks: set[asyncio.Task] = set()
        self._safety_tasks: set[asyncio.Task] = set()
        self._acquire_tasks: set[asyncio.Task] = set()
        self._acquire_locks = tuple(asyncio.Lock() for _ in range(_LOCK_STRIPE_COUNT))
        self._operation_locks = tuple(asyncio.Lock() for _ in range(_LOCK_STRIPE_COUNT))
        self._signaling_locks = tuple(asyncio.Lock() for _ in range(_LOCK_STRIPE_COUNT))
        self._signaling_sources: dict[str, tuple[str, str]] = {}
        self._heartbeat_health: dict[str, _HeartbeatHealth] = {}
        self._driver_snapshots: dict[str, dict] = {}
        self._driver_snapshot_revisions: dict[str, int] = {}
        self._driver_lease_seconds: dict[str, float] = {}
        self._pinned_targets: dict[str, mcp_client.TrustedShadowTarget] = {}
        self._release_results: dict[str, bool] = {}
        self._terminal_cleanup_done: set[str] = set()
        self._command_claims: dict[str, tuple[str, str]] = {}
        self._authority_guards: dict[str, authority_guard.AuthorityGuard] = {}
        self._recovery_claims: dict[str, tuple[str, str]] = {}
        self._guard_store_loaded = False
        self._release_reconcile_tasks: dict[str, asyncio.Task] = {}
        self._reaper_task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        if not self._guard_store_loaded:
            await self._restore_authority_guards()
            self._guard_store_loaded = True
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        self._stopping = False
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(
                self._reaper_loop(),
                name='teleop-session-reaper',
            )

    @staticmethod
    def _authority_root_fingerprint(robot_id: str) -> str:
        services = config.main.get('services', {})
        entries = services.get('mcp', []) if isinstance(services, dict) else []
        matches = [
            entry for entry in entries
            if isinstance(entry, dict) and entry.get('id') == robot_id
        ]
        if len(matches) != 1:
            raise TeleopServiceError('driver_not_ready', 503)
        target = matches[0]
        payload = {
            'id': target.get('id'),
            'url': target.get('url', ''),
            'transport': target.get('transport', 'http'),
            'trust_state': target.get('trust_state', ''),
            'category': target.get('category', ''),
            'authority_domain': target.get('authority_domain', ''),
            'pending_authority_domain': target.get('pending_authority_domain', ''),
            'authority_binding_error': target.get('authority_binding_error', ''),
            'authority_binding_required': bool(
                target.get('authority_binding_required', False),
            ),
            'capability_refresh_required': bool(
                target.get('capability_refresh_required', False),
            ),
            'reported_robot_id': target.get('reported_robot_id', ''),
            'tools': target.get('tools', []),
        }
        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(',', ':'),
                sort_keys=True,
            ).encode('utf-8')
        except (TypeError, ValueError, RecursionError) as error:
            raise TeleopServiceError('driver_not_ready', 503) from error
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _target_fingerprint(
        driver_id: str,
        robot_id: str,
        capability_digest: str,
        target: mcp_client.TrustedShadowTarget | None,
        authority_root_fingerprint: str,
    ) -> str:
        payload = {
            'driver_id': driver_id,
            'robot_id': robot_id,
            'capability_digest': capability_digest,
            'authority_root_fingerprint': authority_root_fingerprint,
            'url': target.url if target is not None else '',
            'descriptor_fingerprint': (
                target.descriptor_fingerprint if target is not None else ''
            ),
            'actions': sorted(target.actions) if target is not None else [],
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def authority_guard_for_robot(self, robot_id: str) -> dict | None:
        """Return the public, secret-free restart quarantine for one robot."""

        guard = self._authority_guards.get(robot_id)
        if guard is None or guard.phase not in {'recovery_required', 'reconciling'}:
            return None
        return {
            'state': 'recovery_required',
            'phase': guard.phase,
            'driver_id': guard.driver_id,
            'robot_id': guard.robot_id,
            'retryable': True,
            'created_at': guard.created_at,
            'updated_at': guard.updated_at,
        }

    def list_authority_guards(self) -> list[dict]:
        """Return every public restart quarantine in deterministic order."""

        return [
            public
            for robot_id in sorted(self._authority_guards)
            if (public := self.authority_guard_for_robot(robot_id)) is not None
        ]

    async def _restore_authority_guards(self) -> None:
        """Rebuild deny-only command claims; never recover old authority."""

        try:
            stored = await asyncio.to_thread(authority_guard.list_guards)
        except Exception as error:
            raise TeleopServiceError('authority_guard_persistence_error', 503) from error

        acquired: list[tuple[str, str]] = []
        try:
            for guard in stored:
                recovered = await asyncio.to_thread(
                    authority_guard.update_guard,
                    guard.robot_id,
                    guard.session_id,
                    phase='recovery_required',
                    dispatch_generation=guard.dispatch_generation,
                )
                token = str(uuid.uuid4())
                await self._begin_command_claim(recovered.robot_id, token)
                await self.command_broker.update_authority(
                    recovered.robot_id,
                    token,
                    session_id=recovered.session_id,
                    principal_id='core:recovery',
                    state='recovery_required',
                )
                acquired.append((recovered.robot_id, token))
                self._authority_guards[recovered.robot_id] = recovered
                self._recovery_claims[recovered.robot_id] = (
                    recovered.session_id,
                    token,
                )
        except Exception:
            for robot_id, token in acquired:
                await self.command_broker.release_authority(robot_id, token)
            self._authority_guards.clear()
            self._recovery_claims.clear()
            raise

        for guard in stored:
            try:
                await audit.emit(
                    'teleop.authority_guard.restored',
                    session_id=guard.session_id,
                    robot_id=guard.robot_id,
                    principal_id='core:recovery',
                    source='core',
                    decision='blocked',
                    reason='core_restart_recovery_required',
                    details={'driver_id': guard.driver_id},
                )
            except Exception:  # noqa: BLE001, S110 -- observability cannot open gate
                pass

    async def _create_authority_guard(
        self,
        session: ShadowSession,
        status: dict,
        target: mcp_client.TrustedShadowTarget | None,
    ) -> None:
        commit = asyncio.create_task(
            self._commit_authority_guard(session, status, target),
            name=f'teleop-authority-guard-commit-{session.id}',
        )
        await self._await_safety_completion(commit)

    async def _commit_authority_guard(
        self,
        session: ShadowSession,
        status: dict,
        target: mcp_client.TrustedShadowTarget | None,
    ) -> None:
        async with authority_guard.target_mutation_lock:
            try:
                current_descriptor = await asyncio.to_thread(
                    _driver_descriptor,
                    session.driver_id,
                )
            except TeleopServiceError as error:
                raise TeleopServiceError('driver_not_ready', 503) from error
            if (
                current_descriptor['robot_id'] != session.robot_id
                or current_descriptor['capability_digest']
                != session.capability_digest
                or current_descriptor['mode'] != session.mode
                or current_descriptor['profile_id'] != session.profile_id
                or current_descriptor['capabilities'] != session.capabilities
            ):
                raise TeleopServiceError('driver_not_ready', 503)
            authority_root_fingerprint = self._authority_root_fingerprint(
                session.robot_id,
            )

            current_target = None
            if self._uses_pinned_targets:
                try:
                    current_target = await mcp_client.resolve_trusted_shadow_target(
                        session.driver_id,
                        timeout_seconds=_DEFAULT_CONTROL_TIMEOUT_SECONDS,
                    )
                except mcp_client.TrustedShadowTransportError as error:
                    raise self._transport_service_error(error) from None
                if target is None or self._target_fingerprint(
                    session.driver_id,
                    session.robot_id,
                    session.capability_digest,
                    current_target,
                    authority_root_fingerprint,
                ) != self._target_fingerprint(
                    session.driver_id,
                    session.robot_id,
                    session.capability_digest,
                    target,
                    authority_root_fingerprint,
                ):
                    raise TeleopServiceError('driver_not_ready', 503)

            now = self._wall_clock()
            guard = authority_guard.AuthorityGuard(
                robot_id=session.robot_id,
                driver_id=session.driver_id,
                session_id=session.id,
                boot_id=session.boot_id,
                epoch=session.epoch,
                capability_digest=session.capability_digest,
                target_fingerprint=self._target_fingerprint(
                    session.driver_id,
                    session.robot_id,
                    session.capability_digest,
                    current_target,
                    authority_root_fingerprint,
                ),
                dispatch_generation=status['dispatch']['generation'],
                phase='preparing',
                created_at=now,
                updated_at=now,
            )
            try:
                created = await asyncio.to_thread(authority_guard.create_guard, guard)
            except Exception as error:
                raise TeleopServiceError(
                    'authority_guard_persistence_error',
                    503,
                ) from error
        self._authority_guards[created.robot_id] = created

    async def _advance_authority_guard(
        self,
        session: ShadowSession,
        *,
        phase: str,
        dispatch_generation: int,
    ) -> None:
        current = self._authority_guards.get(session.robot_id)
        if current is None or current.session_id != session.id:
            return
        try:
            updated = await asyncio.to_thread(
                authority_guard.update_guard,
                session.robot_id,
                session.id,
                phase=phase,
                dispatch_generation=dispatch_generation,
            )
        except Exception as error:
            raise TeleopServiceError('authority_guard_persistence_error', 503) from error
        self._authority_guards[session.robot_id] = updated

    async def _delete_authority_guard_after_safe(self, session: ShadowSession) -> bool:
        current = self._authority_guards.get(session.robot_id)
        if current is None:
            return True
        if current.session_id != session.id:
            return False
        try:
            deleted = await asyncio.to_thread(
                authority_guard.delete_guard,
                session.robot_id,
                session.id,
            )
        except Exception:  # noqa: BLE001 -- DB uncertainty keeps command admission shut
            return False
        if not deleted:
            return False
        self._authority_guards.pop(session.robot_id, None)
        return True

    @staticmethod
    def _recovery_status_proves_safe(
        guard: authority_guard.AuthorityGuard,
        projected: dict,
    ) -> bool:
        if projected['authority_valid'] is not False or projected['session_id'] is not None:
            return False
        dispatch = projected.get('dispatch')
        if (
            not isinstance(dispatch, dict)
            or dispatch['stop_acknowledged'] is not True
            or dispatch['stop_queue_depth'] != 0
            or dispatch['io_inflight'] is not None
        ):
            return False
        if projected['boot_id'] != guard.boot_id:
            return (
                projected['state'] == 'idle'
                and dispatch['state'] == 'safe_unarmed'
                and dispatch['last_decision'] == 'startup_safe_ack'
            )
        return (
            projected['epoch'] <= guard.epoch
            and dispatch['state'] == 'safe_revoked'
            and dispatch['generation'] > guard.dispatch_generation
        )

    async def _clear_recovery_guard(
        self,
        guard: authority_guard.AuthorityGuard,
        *,
        principal_id: str,
        reason: str,
    ) -> dict:
        cleanup = asyncio.create_task(
            self._complete_recovery_guard_clear(
                guard,
                principal_id=principal_id,
                reason=reason,
            ),
            name=f'teleop-authority-guard-clear-{guard.robot_id}',
        )
        return await self._await_safety_completion(cleanup)

    async def _complete_recovery_guard_clear(
        self,
        guard: authority_guard.AuthorityGuard,
        *,
        principal_id: str,
        reason: str,
    ) -> dict:
        claim = self._recovery_claims.get(guard.robot_id)
        if claim is None or claim[0] != guard.session_id:
            raise TeleopServiceError('authority_guard_persistence_error', 503)
        try:
            deleted = await asyncio.to_thread(
                authority_guard.delete_guard,
                guard.robot_id,
                guard.session_id,
            )
        except Exception as error:
            raise TeleopServiceError('authority_guard_persistence_error', 503) from error
        if not deleted:
            try:
                persisted = await asyncio.to_thread(
                    authority_guard.get_guard,
                    guard.robot_id,
                )
            except Exception as error:
                raise TeleopServiceError(
                    'authority_guard_persistence_error',
                    503,
                ) from error
            if persisted is not None:
                raise TeleopServiceError('authority_guard_persistence_error', 503)
        released = await self.command_broker.release_authority(
            guard.robot_id,
            claim[1],
        )
        if not released:
            current_claim = await self.command_broker.authority_for(guard.robot_id)
            if current_claim is not None:
                raise TeleopServiceError('authority_guard_persistence_error', 503)
        self._authority_guards.pop(guard.robot_id, None)
        self._recovery_claims.pop(guard.robot_id, None)
        try:
            await audit.emit(
                'teleop.authority_guard.cleared',
                session_id=guard.session_id,
                robot_id=guard.robot_id,
                principal_id=principal_id,
                source='api',
                decision='released',
                reason=reason,
                details={'driver_id': guard.driver_id, 'old_session_restored': False},
            )
        except Exception:  # noqa: BLE001, S110 -- proof and deletion already succeeded
            pass
        return {
            'state': 'clear',
            'robot_id': guard.robot_id,
            'driver_id': guard.driver_id,
            'old_session_restored': False,
            'reacquire_required': True,
        }

    async def reconcile_authority_guard(
        self,
        robot_id: str,
        *,
        principal_id: str,
    ) -> dict:
        """Owner-triggered proof that a restarted Core may reopen writes."""

        if not _AUTHORITY_ID_RE.fullmatch(robot_id):
            raise TeleopServiceError('authority_guard_not_found', 404)
        if not self._guard_store_loaded:
            await self.start()
        lock = self._driver_acquire_lock(robot_id)
        async with lock:
            guard = self._authority_guards.get(robot_id)
            if guard is None:
                return {
                    'state': 'clear',
                    'robot_id': robot_id,
                    'old_session_restored': False,
                    'reacquire_required': True,
                    'already_clear': True,
                }
            if guard.phase not in {'recovery_required', 'reconciling'}:
                raise TeleopServiceError('authority_guard_not_safe', 409)

            try:
                descriptor = await asyncio.to_thread(_driver_descriptor, guard.driver_id)
                authority_root_fingerprint = self._authority_root_fingerprint(
                    guard.robot_id,
                )
            except TeleopServiceError as error:
                raise TeleopServiceError('authority_guard_target_changed', 409) from error
            if (
                descriptor['robot_id'] != guard.robot_id
                or descriptor['capability_digest'] != guard.capability_digest
            ):
                raise TeleopServiceError('authority_guard_target_changed', 409)

            target = None
            if self._uses_pinned_targets:
                try:
                    target = await mcp_client.resolve_trusted_shadow_target(
                        guard.driver_id,
                        timeout_seconds=_DEFAULT_CONTROL_TIMEOUT_SECONDS,
                    )
                except mcp_client.TrustedShadowTransportError as error:
                    raise self._transport_service_error(error) from None
            if self._target_fingerprint(
                guard.driver_id,
                guard.robot_id,
                guard.capability_digest,
                target,
                authority_root_fingerprint,
            ) != guard.target_fingerprint:
                raise TeleopServiceError('authority_guard_target_changed', 409)

            try:
                raw_status = await self._call(
                    guard.driver_id,
                    'status',
                    timeout_seconds=0.5,
                    pinned_target=target,
                )
            except mcp_client.TrustedShadowTransportError as error:
                raise self._transport_service_error(error) from None
            projected, _ = _project_driver_snapshot(
                raw_status,
                driver_id=guard.driver_id,
                robot_id=guard.robot_id,
                capability_digest=guard.capability_digest,
                action='status',
                expected_mode=descriptor['mode'],
                expected_profile_id=descriptor['profile_id'],
                expected_capabilities=descriptor['capabilities'],
                allow_stop_pending=True,
            )
            if self._recovery_status_proves_safe(guard, projected):
                return await self._clear_recovery_guard(
                    guard,
                    principal_id=principal_id,
                    reason=(
                        'driver_restart_safe_unarmed'
                        if projected['boot_id'] != guard.boot_id
                        else 'driver_watchdog_safe_revoked'
                    ),
                )
            if (
                projected['boot_id'] == guard.boot_id
                and (
                    projected['epoch'] > guard.epoch
                    or projected['dispatch']['generation'] < guard.dispatch_generation
                )
            ):
                raise TeleopServiceError('authority_guard_not_safe', 409)

            try:
                reconciling = await asyncio.to_thread(
                    authority_guard.update_guard,
                    guard.robot_id,
                    guard.session_id,
                    phase='reconciling',
                    dispatch_generation=max(
                        guard.dispatch_generation,
                        projected['dispatch']['generation'],
                    ),
                )
            except Exception as error:
                raise TeleopServiceError('authority_guard_persistence_error', 503) from error
            self._authority_guards[robot_id] = reconciling
            await self.command_broker.update_authority(
                robot_id,
                self._recovery_claims[robot_id][1],
                state='reconciling',
            )

            try:
                raw_stopped = await self._call(
                    guard.driver_id,
                    'stop',
                    timeout_seconds=0.5,
                    pinned_target=target,
                )
            except mcp_client.TrustedShadowTransportError as error:
                raise self._transport_service_error(error) from None
            stopped, _ = _project_driver_snapshot(
                raw_stopped,
                driver_id=guard.driver_id,
                robot_id=guard.robot_id,
                capability_digest=guard.capability_digest,
                action='release',
                expected_mode=descriptor['mode'],
                expected_profile_id=descriptor['profile_id'],
                expected_capabilities=descriptor['capabilities'],
            )
            same_observed_boot = stopped['boot_id'] == projected['boot_id']
            if (
                (same_observed_boot and stopped['dispatch']['generation']
                 <= projected['dispatch']['generation'])
                or (stopped['boot_id'] == guard.boot_id and stopped['epoch'] > guard.epoch)
            ):
                raise TeleopServiceError('authority_guard_not_safe', 409)
            return await self._clear_recovery_guard(
                reconciling,
                principal_id=principal_id,
                reason='driver_lifecycle_stop_acknowledged',
            )

    async def stop(self) -> None:
        self._stopping = True
        await self.capture_manager.reset('core_stopped')
        current_task = asyncio.current_task()
        acquire_tasks = [
            task for task in self._acquire_tasks
            if task is not current_task and not task.done()
        ]
        if acquire_tasks:
            await asyncio.gather(*acquire_tasks, return_exceptions=True)
        await self._drain_safety_tasks()

        reaper = self._reaper_task
        self._reaper_task = None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)

        tasks = list(self._heartbeat_tasks.values())
        self._heartbeat_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        supervisor_tasks = list(self._supervisor_tasks)
        self._supervisor_tasks.clear()
        for task in supervisor_tasks:
            task.cancel()
        if supervisor_tasks:
            await asyncio.gather(*supervisor_tasks, return_exceptions=True)

        reconcile_tasks = list(self._release_reconcile_tasks.values())
        self._release_reconcile_tasks.clear()
        for task in reconcile_tasks:
            task.cancel()
        if reconcile_tasks:
            await asyncio.gather(*reconcile_tasks, return_exceptions=True)

        sessions = await self.manager.list_visible('', owner=True)
        for session in sessions:
            lock = self._session_lock(session.id)
            async with lock:
                await self._shutdown_session_locked(session)
        await self._drain_safety_tasks()

        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None
        await self.manager.reset()
        self._safety_tasks.clear()
        self._acquire_tasks.clear()
        self._heartbeat_health.clear()
        self._driver_snapshots.clear()
        self._driver_snapshot_revisions.clear()
        self._driver_lease_seconds.clear()
        self._pinned_targets.clear()
        self._signaling_sources.clear()
        self._release_results.clear()
        self._terminal_cleanup_done.clear()
        self._release_reconcile_tasks.clear()

    async def _shutdown_session_locked(self, session: ShadowSession) -> None:
        if session.mode == LIVE_MODE and not session.live_confirmed:
            if session.state == 'awaiting_confirmation':
                try:
                    session = await self.manager.release(
                        session.id,
                        'core:shutdown',
                        session.client_id,
                        owner=True,
                    )
                    await self._update_command_claim(session, 'releasing')
                except (SessionNotFound, SessionStateConflict):
                    pass
            self._terminal_cleanup_done.add(session.id)
            await self._cleanup_terminal_runtime(session.id)
            return
        was_live = session.state in {'preparing', 'active', 'paused', 'hold'}
        acknowledged = False
        if was_live:
            try:
                session = await self.manager.release(
                    session.id,
                    'core:shutdown',
                    session.client_id,
                    owner=True,
                )
                await self._update_command_claim(session, 'releasing')
            except (SessionNotFound, SessionStateConflict):
                was_live = False

        if session.id not in self._terminal_cleanup_done:
            target_available = (
                not self._uses_pinned_targets
                or session.id in self._pinned_targets
            )
            if target_available:
                if session.state == 'released':
                    acknowledged = await self._best_effort_release(session)
                    self._release_results[session.id] = acknowledged
                else:
                    acknowledged = await self._best_effort_soft_stop_and_release(session)
                driver_acknowledged = acknowledged
                if acknowledged:
                    acknowledged = await self._delete_authority_guard_after_safe(session)
                if acknowledged:
                    self._terminal_cleanup_done.add(session.id)
            else:
                driver_acknowledged = False
        else:
            driver_acknowledged = acknowledged

        if was_live:
            await audit.emit(
                'teleop.session.shutdown',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=session.principal_id,
                source='core',
                decision='released' if acknowledged else 'quarantined',
                reason=(
                    'core_shutdown_release_acknowledged'
                    if acknowledged else 'core_shutdown_release_unconfirmed'
                ),
                details={
                    'driver_acknowledged': driver_acknowledged,
                    'guard_deleted': acknowledged,
                },
            )
        if acknowledged:
            await self._cleanup_terminal_runtime(session.id)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._operation_locks[hash(session_id) % _LOCK_STRIPE_COUNT]

    def _signaling_lock(self, session_id: str) -> asyncio.Lock:
        return self._signaling_locks[hash(session_id) % _LOCK_STRIPE_COUNT]

    def _driver_acquire_lock(self, robot_id: str) -> asyncio.Lock:
        return self._acquire_locks[hash(robot_id) % _LOCK_STRIPE_COUNT]

    async def _begin_command_claim(self, driver_id: str, token: str) -> None:
        try:
            await self.command_broker.begin_authority(
                driver_id,
                token,
                drain_timeout_seconds=_COMMAND_DRAIN_TIMEOUT_SECONDS,
            )
        except CommandDrainTimeout:
            raise TeleopServiceError('driver_command_busy', 409) from None
        except AuthorityAlreadyClaimed:
            raise TeleopServiceError('driver_authority_busy', 409) from None
        except asyncio.CancelledError:
            # Cancellation can arrive after Broker committed the claim but before
            # this wrapper returns. Releasing by token is safe even if the claim
            # was never committed, and the shielded cleanup survives re-cancel.
            cleanup = asyncio.create_task(
                self.command_broker.release_authority(driver_id, token),
                name=f'teleop-cancelled-command-claim-{driver_id}',
            )
            await self._await_safety_completion(cleanup)
            raise

    async def _ensure_command_claim(self, session: ShadowSession) -> None:
        claim = self._command_claims.get(session.id)
        if claim is None:
            token = str(uuid.uuid4())
            await self._begin_command_claim(session.robot_id, token)
            self._command_claims[session.id] = (session.robot_id, token)
            claim = (session.robot_id, token)
        await self.command_broker.update_authority(
            claim[0],
            claim[1],
            session_id=session.id,
            principal_id=session.principal_id,
            state=session.state,
        )

    async def _update_command_claim(self, session: ShadowSession, state: str) -> None:
        claim = self._command_claims.get(session.id)
        if claim is None:
            return
        await self.command_broker.update_authority(
            claim[0],
            claim[1],
            session_id=session.id,
            principal_id=session.principal_id,
            state=state,
        )

    async def _call(
        self,
        driver_id: str,
        action: str,
        arguments: dict | None = None,
        *,
        timeout_seconds: float = _DEFAULT_CONTROL_TIMEOUT_SECONDS,
        pinned_target: mcp_client.TrustedShadowTarget | None = None,
    ) -> dict:
        if self._http_session is None or self._http_session.closed:
            if self._stopping:
                raise TeleopServiceError('coordinator_stopping', 503)
            await self.start()
        assert self._http_session is not None
        kwargs = {
            'timeout_seconds': timeout_seconds,
            'session': self._http_session,
        }
        if self._uses_pinned_targets:
            if pinned_target is None and isinstance(arguments, dict):
                session_id = arguments.get('session_id')
                if isinstance(session_id, str):
                    pinned_target = self._pinned_targets.get(session_id)
            kwargs['target'] = pinned_target
        try:
            return await self._caller(driver_id, action, arguments, **kwargs)
        except (mcp_client.TrustedShadowTransportError, asyncio.CancelledError):
            raise
        except Exception:  # noqa: BLE001 -- normalize injected/unknown transports
            raise mcp_client.TrustedShadowTransportError(
                'internal_transport_error',
            ) from None

    async def _signal_offer(
        self,
        session: ShadowSession,
        offer: dict,
        ticket: str,
        target: mcp_client.TrustedShadowTarget,
    ) -> dict:
        if self._http_session is None or self._http_session.closed:
            if self._stopping:
                raise TeleopServiceError('coordinator_stopping', 503)
            await self.start()
        assert self._http_session is not None
        try:
            return await self._signaler(
                session.driver_id,
                offer,
                ticket,
                timeout_seconds=_SIGNALING_TIMEOUT_SECONDS,
                session=self._http_session,
                target=target,
            )
        except (mcp_client.TrustedShadowTransportError, asyncio.CancelledError):
            raise
        except Exception:  # noqa: BLE001 -- normalize injected/unknown transports
            raise mcp_client.TrustedShadowTransportError(
                'internal_transport_error',
            ) from None

    @staticmethod
    def _transport_service_error(error: mcp_client.TrustedShadowTransportError) -> TeleopServiceError:
        if error.code == 'timeout':
            return TeleopServiceError('driver_timeout', 504)
        if error.code in {
            'target_not_found', 'target_ambiguous', 'target_not_trusted',
            'invalid_transport', 'invalid_url', 'registry_offline',
            'registry_not_trusted', 'registry_target_mismatch',
            'registry_descriptor_mismatch', 'insecure_transport',
            'descriptor_invalid', 'action_not_declared', 'driver_auth_unavailable',
            'client_session_closed', 'target_resolution_error',
        }:
            return TeleopServiceError('driver_not_ready', 503)
        if error.code == 'pinned_target_changed':
            return TeleopServiceError('driver_session_lost', 409)
        if error.code == 'http_error' and error.http_status in {401, 403}:
            return TeleopServiceError('driver_auth_rejected', 502)
        if error.code == 'rpc_error' and error.rpc_data_code in _TERMINAL_RPC_CODES:
            return TeleopServiceError('driver_session_lost', 409)
        if error.code == 'network_error':
            return TeleopServiceError('driver_unreachable', 503)
        return TeleopServiceError('driver_protocol_error', 502)

    async def acquire(
        self,
        driver_id: str,
        principal_id: str,
        client_id: str,
        mode: str = SHADOW_MODE,
    ) -> AcquireResult:
        if self._stopping:
            raise TeleopServiceError('coordinator_stopping', 503)
        task = asyncio.current_task()
        if task is None:
            raise TeleopServiceError('coordinator_stopping', 503)
        self._acquire_tasks.add(task)
        try:
            # The authenticated teleop descriptor binds a potentially standalone
            # adapter id to the physical robot authority domain.
            descriptor = await asyncio.to_thread(_driver_descriptor, driver_id)
            if descriptor['mode'] != mode:
                raise TeleopServiceError('teleop_mode_unavailable', 409)
            robot_id = descriptor['robot_id']
            lock = self._driver_acquire_lock(robot_id)
            async with lock:
                return await self._acquire_locked(
                    driver_id,
                    robot_id,
                    principal_id,
                    client_id,
                    descriptor,
                )
        finally:
            self._acquire_tasks.discard(task)

    async def confirm_live(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        profile_id: str,
    ) -> AcquireResult:
        """Turn a memory-only live reservation into Driver authority exactly once."""

        lock = self._session_lock(session_id)
        async with lock:
            session = await self.manager.get_authorized(
                session_id,
                principal_id,
                owner=False,
                client_id=client_id,
                require_client=True,
            )
            if session.mode != LIVE_MODE or session.profile_id != profile_id:
                raise TeleopServiceError('live_confirmation_mismatch', 409)
            if session.state == 'active' and session.live_confirmed:
                return AcquireResult(session, 'existing')
            if session.state != 'awaiting_confirmation' or session.live_confirmed:
                raise SessionStateConflict(session_id)

            descriptor = await asyncio.to_thread(_driver_descriptor, session.driver_id)
            if (
                descriptor['mode'] != LIVE_MODE
                or descriptor['profile_id'] != session.profile_id
                or descriptor['capability_digest'] != session.capability_digest
                or descriptor['capabilities'] != session.capabilities
                or descriptor['robot_id'] != session.robot_id
            ):
                raise TeleopServiceError('driver_not_ready', 503)
            pinned_target = None
            if self._uses_pinned_targets:
                try:
                    pinned_target = await mcp_client.resolve_trusted_shadow_target(
                        session.driver_id,
                        timeout_seconds=_DEFAULT_CONTROL_TIMEOUT_SECONDS,
                    )
                except mcp_client.TrustedShadowTransportError as error:
                    raise self._transport_service_error(error) from None
                if pinned_target.capability_digest != session.capability_digest:
                    raise TeleopServiceError('driver_not_ready', 503)
            try:
                raw_status = await self._call(
                    session.driver_id,
                    'status',
                    pinned_target=pinned_target,
                )
            except mcp_client.TrustedShadowTransportError as error:
                raise self._transport_service_error(error) from None
            status, _ = _project_driver_snapshot(
                raw_status,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=session.capability_digest,
                action='status',
                expected_mode=LIVE_MODE,
                expected_profile_id=session.profile_id,
                expected_capabilities=session.capabilities,
            )
            if status['authority_valid'] is True or status['session_id'] is not None:
                raise TeleopServiceError('driver_authority_busy', 409)

            await audit.emit(
                'teleop.session.live_confirmed',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=principal_id,
                source='api',
                decision='confirmed',
                reason='operator_confirmed_hardware_actuation',
                details={
                    'mode': LIVE_MODE,
                    'profile_id': session.profile_id,
                    'capability_digest': session.capability_digest,
                    'hardware_output': True,
                },
            )

            identity_bound = False
            guard_created = False
            prepare_started = False
            generation = session.operation_generation
            claim = self._command_claims.get(session.id)
            if claim is None:
                raise TeleopServiceError('driver_session_lost', 409)
            try:
                session = await self.manager.confirm_live_identity(
                    session.id,
                    principal_id,
                    client_id,
                    boot_id=status['boot_id'],
                    minimum_epoch=status['epoch'] + 1,
                )
                identity_bound = True
                self._store_driver_snapshot(session, status)
                if pinned_target is not None:
                    self._pinned_targets[session.id] = pinned_target
                await self._create_authority_guard(session, status, pinned_target)
                guard_created = True
                await self._update_command_claim(session, 'preparing')
                prepare_started = True
                result = await self._activate_reserved_session(
                    session,
                    generation=generation,
                    digest=session.capability_digest,
                    principal_id=principal_id,
                )
                await self._update_command_claim(result.session, 'active')
                return result
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(
                    self._cleanup_live_confirmation_interruption(
                        session,
                        generation=generation,
                        claim_token=claim[1],
                        identity_bound=identity_bound,
                        guard_created=guard_created,
                        prepare_started=prepare_started,
                        reason='request_cancelled',
                    ),
                    name=f'teleop-cancelled-live-confirm-{session.id}',
                )
                await self._await_safety_completion(cleanup)
                raise
            except Exception:
                await self._cleanup_live_confirmation_interruption(
                    session,
                    generation=generation,
                    claim_token=claim[1],
                    identity_bound=identity_bound,
                    guard_created=guard_created,
                    prepare_started=prepare_started,
                    reason='prepare_failed',
                )
                raise

    async def _cleanup_live_confirmation_interruption(
        self,
        session: ShadowSession,
        *,
        generation: int,
        claim_token: str,
        identity_bound: bool,
        guard_created: bool,
        prepare_started: bool,
        reason: str,
    ) -> None:
        """Close every post-confirm commit/return race without opening authority."""

        current = await self.manager.get(session.id)
        identity_bound = identity_bound or (
            current is session
            and session.mode == LIVE_MODE
            and session.live_confirmed
            and session.epoch > 0
            and bool(session.boot_id)
        )
        if not identity_bound or session.id not in self._command_claims:
            return
        if session.robot_id in self._recovery_claims:
            return
        committed = self._authority_guards.get(session.robot_id)
        guard_created = guard_created or (
            committed is not None and committed.session_id == session.id
        )
        if guard_created and prepare_started:
            await self._fail_prepare_locked(session, generation, reason)
        elif guard_created:
            await self._discard_guard_before_prepare(
                session,
                generation,
                claim_token,
                reason,
            )
        else:
            await self._abandon_before_prepare(session, generation, reason)

    async def _acquire_locked(
        self,
        driver_id: str,
        robot_id: str,
        principal_id: str,
        client_id: str,
        descriptor: dict,
    ) -> AcquireResult:
        guard = self._authority_guards.get(robot_id)
        if guard is not None and guard.phase in {'recovery_required', 'reconciling'}:
            raise TeleopServiceError('robot_recovery_required', 409)
        existing = await self.manager.active_for_robot(robot_id)
        if existing is not None:
            if (
                existing.driver_id != driver_id
                or existing.principal_id != principal_id
                or existing.client_id != client_id
            ):
                raise SessionConflict(existing)
            if (
                existing.mode != descriptor['mode']
                or existing.profile_id != descriptor['profile_id']
                or existing.capability_digest != descriptor['capability_digest']
            ):
                raise TeleopServiceError('driver_not_ready', 503)
            await self._ensure_command_claim(existing)
            disposition = (
                'confirmation_required'
                if existing.state == 'awaiting_confirmation'
                else 'preparing'
                if existing.state == 'preparing'
                else 'existing'
            )
            return AcquireResult(existing, disposition)

        claim_token = str(uuid.uuid4())
        await self._begin_command_claim(robot_id, claim_token)
        try:
            return await self._acquire_claimed(
                driver_id,
                robot_id,
                principal_id,
                client_id,
                claim_token,
                descriptor,
            )
        finally:
            claim_retained = any(
                token == claim_token
                for _claimed_driver_id, token in self._command_claims.values()
            ) or any(
                token == claim_token
                for _recovered_session_id, token in self._recovery_claims.values()
            )
            if not claim_retained:
                cleanup = asyncio.create_task(
                    self.command_broker.release_authority(robot_id, claim_token),
                    name=f'teleop-unreserved-command-claim-{robot_id}',
                )
                await self._await_safety_completion(cleanup)

    async def _acquire_claimed(
        self,
        driver_id: str,
        robot_id: str,
        principal_id: str,
        client_id: str,
        claim_token: str,
        descriptor: dict,
    ) -> AcquireResult:
        if descriptor['mode'] == LIVE_MODE:
            try:
                session = await self.manager.reserve(
                    robot_id,
                    principal_id,
                    driver_id=driver_id,
                    boot_id='',
                    capability_digest=descriptor['capability_digest'],
                    mode=LIVE_MODE,
                    profile_id=descriptor['profile_id'],
                    capabilities=descriptor['capabilities'],
                    effectors=descriptor['capabilities']['effectors'],
                    signaling_audience=descriptor['signaling']['audience'],
                    defer_identity=True,
                    client_id=client_id,
                    lease_seconds=MIN_LEASE_SECONDS,
                )
            except EpochExhausted:
                raise TeleopServiceError('driver_epoch_exhausted', 409) from None
            self._command_claims[session.id] = (robot_id, claim_token)
            await self._update_command_claim(session, 'awaiting_confirmation')
            await audit.emit(
                'teleop.session.live_confirmation_required',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=principal_id,
                source='api',
                decision='reserved',
                reason='operator_acquire_live',
                details={
                    'mode': LIVE_MODE,
                    'profile_id': session.profile_id,
                    'hardware_output': False,
                    'lease_seconds': session.lease_seconds,
                },
            )
            return AcquireResult(session, 'confirmation_required')

        pinned_target = None
        if self._uses_pinned_targets:
            try:
                pinned_target = await mcp_client.resolve_trusted_shadow_target(
                    driver_id,
                    timeout_seconds=_DEFAULT_CONTROL_TIMEOUT_SECONDS,
                )
            except mcp_client.TrustedShadowTransportError as error:
                raise self._transport_service_error(error) from None
            if pinned_target.capability_digest != descriptor['capability_digest']:
                raise TeleopServiceError('driver_not_ready', 503)
        digest = descriptor['capability_digest']
        try:
            raw_status = await self._call(
                driver_id,
                'status',
                pinned_target=pinned_target,
            )
        except mcp_client.TrustedShadowTransportError as error:
            raise self._transport_service_error(error) from None
        status, _ = _project_driver_snapshot(
            raw_status,
            driver_id=driver_id,
            robot_id=descriptor['robot_id'],
            capability_digest=digest,
            action='status',
            expected_mode=descriptor['mode'],
            expected_profile_id=descriptor['profile_id'],
            expected_capabilities=descriptor['capabilities'],
        )
        if status['authority_valid'] is True or status['session_id'] is not None:
            raise TeleopServiceError('driver_authority_busy', 409)
        minimum_epoch = status['epoch'] + 1
        try:
            session = await self.manager.reserve(
                robot_id,
                principal_id,
                driver_id=driver_id,
                boot_id=status['boot_id'],
                capability_digest=digest,
                mode=descriptor['mode'],
                profile_id=descriptor['profile_id'],
                capabilities=descriptor['capabilities'],
                effectors=descriptor['capabilities']['effectors'],
                signaling_audience=descriptor['signaling']['audience'],
                client_id=client_id,
                minimum_epoch=minimum_epoch,
            )
        except EpochExhausted:
            raise TeleopServiceError('driver_epoch_exhausted', 409) from None
        generation = session.operation_generation
        self._command_claims[session.id] = (robot_id, claim_token)
        guard_created = False
        prepare_started = False
        try:
            self._store_driver_snapshot(session, status)
            if pinned_target is not None:
                self._pinned_targets[session.id] = pinned_target
            # The durable deny record is committed before the first request that
            # can install a Driver fence. A failed/uncertain write cannot reach
            # prepare_shadow.
            await self._create_authority_guard(session, status, pinned_target)
            guard_created = True
            await self._update_command_claim(session, 'preparing')
            # Publish the lock in the same event-loop turn as the reservation.
            # Owner Release may observe a preparing session, but it cannot pass a
            # delayed prepare/heartbeat and allow authority to reappear afterward.
            operation_lock = self._session_lock(session.id)
            async with operation_lock:
                prepare_started = True
                result = await self._activate_reserved_session(
                    session,
                    generation=generation,
                    digest=digest,
                    principal_id=principal_id,
                )
            await self._update_command_claim(result.session, 'active')
        except asyncio.CancelledError:
            committed = self._authority_guards.get(session.robot_id)
            guard_created = guard_created or (
                committed is not None and committed.session_id == session.id
            )
            cleanup = asyncio.create_task(
                self._fail_prepare_locked(session, generation, 'request_cancelled')
                if guard_created and prepare_started
                else self._discard_guard_before_prepare(
                    session,
                    generation,
                    claim_token,
                    'request_cancelled',
                )
                if guard_created
                else self._abandon_before_prepare(
                    session,
                    generation,
                    'request_cancelled',
                ),
                name=f'teleop-cancelled-command-claim-{session.id}',
            )
            await self._await_safety_completion(cleanup)
            raise
        except Exception:
            committed = self._authority_guards.get(session.robot_id)
            guard_created = guard_created or (
                committed is not None and committed.session_id == session.id
            )
            if guard_created and prepare_started:
                await self._fail_prepare_locked(session, generation, 'prepare_failed')
            elif guard_created:
                await self._discard_guard_before_prepare(
                    session,
                    generation,
                    claim_token,
                    'prepare_not_started',
                )
            else:
                await self._abandon_before_prepare(
                    session,
                    generation,
                    'authority_guard_not_committed',
                )
            raise
        return result

    async def _discard_guard_before_prepare(
        self,
        session: ShadowSession,
        generation: int,
        claim_token: str,
        reason: str,
    ) -> None:
        """Remove a committed guard when no Driver fence request was sent."""

        current = self._authority_guards.get(session.robot_id)
        if current is None or current.session_id != session.id:
            await self._abandon_before_prepare(session, generation, reason)
            return

        deletion_confirmed = False
        try:
            deletion_confirmed = await asyncio.to_thread(
                authority_guard.delete_guard,
                session.robot_id,
                session.id,
            )
            if not deletion_confirmed:
                deletion_confirmed = await asyncio.to_thread(
                    authority_guard.get_guard,
                    session.robot_id,
                ) is None
        except Exception:  # noqa: BLE001 -- uncertain storage becomes recovery lock
            try:
                deletion_confirmed = await asyncio.to_thread(
                    authority_guard.get_guard,
                    session.robot_id,
                ) is None
            except Exception:  # noqa: BLE001 -- preserve the deny-only claim
                deletion_confirmed = False

        if deletion_confirmed:
            self._authority_guards.pop(session.robot_id, None)
            await self._abandon_before_prepare(session, generation, reason)
            return

        try:
            recovered = await asyncio.to_thread(
                authority_guard.update_guard,
                session.robot_id,
                session.id,
                phase='recovery_required',
                dispatch_generation=current.dispatch_generation,
            )
        except Exception:  # noqa: BLE001 -- in-memory quarantine remains fail-closed
            recovered = replace(
                current,
                phase='recovery_required',
                updated_at=max(current.updated_at, self._wall_clock()),
            )
        self._authority_guards[session.robot_id] = recovered
        self._command_claims.pop(session.id, None)
        self._recovery_claims[session.robot_id] = (session.id, claim_token)
        await self.command_broker.update_authority(
            session.robot_id,
            claim_token,
            session_id=session.id,
            principal_id='core:recovery',
            state='recovery_required',
        )
        await self.manager.fail_reservation(session.id, generation)
        self._pinned_targets.pop(session.id, None)
        self._driver_snapshots.pop(session.id, None)
        self._driver_snapshot_revisions.pop(session.id, None)
        self._terminal_cleanup_done.add(session.id)

    async def _abandon_before_prepare(
        self,
        session: ShadowSession,
        generation: int,
        reason: str,
    ) -> None:
        """Discard a reservation without contacting a Driver that was never prepared."""

        await self.manager.fail_reservation(session.id, generation)
        claim = self._command_claims.pop(session.id, None)
        self._pinned_targets.pop(session.id, None)
        self._driver_snapshots.pop(session.id, None)
        self._driver_snapshot_revisions.pop(session.id, None)
        self._terminal_cleanup_done.add(session.id)
        if claim is not None:
            await self.command_broker.release_authority(claim[0], claim[1])
        try:
            await audit.emit(
                'teleop.session.prepare_rejected',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=session.principal_id,
                source='core',
                decision='rejected',
                reason=reason,
            )
        except Exception:  # noqa: BLE001, S110 -- admission result is already closed
            pass

    async def _activate_reserved_session(
        self,
        session: ShadowSession,
        *,
        generation: int,
        digest: str,
        principal_id: str,
    ) -> AcquireResult:
        try:
            await audit.emit(
                'teleop.session.reserved',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=principal_id,
                source='api',
                decision='reserved',
                reason='operator_acquire',
                details={
                    'mode': session.mode,
                    'profile_id': session.profile_id,
                    'lease_seconds': session.lease_seconds,
                },
            )
            baseline_generation = self._driver_snapshots[
                session.id
            ]['dispatch']['generation']
            raw_prepared = await self._prepare_with_reconcile(session)
            projected, lease_seconds = _project_driver_snapshot(
                raw_prepared,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=digest,
                action=PREPARE_ACTION_BY_MODE[session.mode],
                session=session,
            )
            self._store_driver_snapshot(
                session,
                projected,
                minimum_generation_exclusive=baseline_generation,
            )
            self._driver_lease_seconds[session.id] = lease_seconds
            raw_heartbeat = await self._call(
                session.driver_id,
                'heartbeat',
                self._identity(session),
                timeout_seconds=self._heartbeat_timeout(lease_seconds),
            )
            projected, _ = _project_driver_snapshot(
                raw_heartbeat,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=digest,
                action='heartbeat',
                session=session,
            )
            self._store_driver_snapshot(session, projected)
            activated = await self.manager.activate(session.id, generation)
            now_mono = self._monotonic()
            self._heartbeat_health[session.id] = _HeartbeatHealth(
                last_confirmed_monotonic=now_mono,
                last_confirmed_at=self._wall_clock(),
            )
            self._start_heartbeat(activated)
            try:
                await self._advance_authority_guard(
                    session,
                    phase='active',
                    dispatch_generation=projected['dispatch']['generation'],
                )
            except TeleopServiceError:
                # The committed guard already fails closed. Phase metadata must
                # never delay or tear down a safely heartbeating Driver lease.
                pass
            await audit.emit(
                'teleop.session.activated',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=principal_id,
                source='core',
                decision='activated',
                reason='driver_prepared',
                details={
                    'mode': session.mode,
                    'profile_id': session.profile_id,
                    'state': 'active',
                },
            )
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._fail_prepare_locked(session, generation, 'request_cancelled'),
                name=f'teleop-cancelled-prepare-{session.id}',
            )
            await self._await_safety_completion(cleanup)
            raise
        except mcp_client.TrustedShadowTransportError as error:
            await self._fail_prepare_locked(session, generation, error.code)
            raise self._transport_service_error(error) from None
        except (TeleopServiceError, SessionNotFound, SessionStateConflict) as error:
            code = error.code if isinstance(error, TeleopServiceError) else 'stale_prepare'
            await self._fail_prepare_locked(session, generation, code)
            if isinstance(error, TeleopServiceError):
                raise
            raise TeleopServiceError('session_prepare_stale', 409) from None
        except Exception:  # noqa: BLE001 -- any post-fence fault must revoke authority
            await self._fail_prepare_locked(
                session,
                generation,
                'driver_response_invalid',
            )
            raise TeleopServiceError('driver_response_invalid', 502) from None

        return AcquireResult(activated, 'created')

    async def _fail_prepare_locked(
        self,
        session: ShadowSession,
        generation: int,
        reason: str,
    ) -> None:
        failed = await self.manager.fail_reservation(session.id, generation)
        if failed is None:
            current = await self.manager.get_current(session.id)
            if current is None or current.operation_generation != generation:
                return
            try:
                failed = await self.manager.release(
                    session.id,
                    'core:cancelled-acquire',
                    session.client_id,
                    owner=True,
                )
            except (SessionNotFound, SessionStateConflict):
                return
        await self._cancel_heartbeat(session.id, origin_task=asyncio.current_task())
        await self._update_command_claim(failed, 'releasing')
        acknowledged = await self._best_effort_release(failed)
        await self._audit_prepare_failure(session, reason)
        await self._finalize_or_quarantine_release(failed, acknowledged)

    def _store_driver_snapshot(
        self,
        session: ShadowSession,
        projected: dict,
        *,
        allow_restart: bool = False,
        stale_after_revision: int | None = None,
        minimum_generation_exclusive: int | None = None,
    ) -> bool:
        previous = self._driver_snapshots.get(session.id)
        current_dispatch = projected.get('dispatch')
        previous_dispatch = (
            previous.get('dispatch') if isinstance(previous, dict) else None
        )
        if (
            isinstance(current_dispatch, dict)
            and minimum_generation_exclusive is not None
            and current_dispatch['generation'] <= minimum_generation_exclusive
        ):
            raise TeleopServiceError('driver_response_invalid', 502)

        def superseded_while_inflight() -> bool:
            return (
                stale_after_revision is not None
                and self._driver_snapshot_revisions.get(session.id, 0)
                > stale_after_revision
            )

        if isinstance(current_dispatch, dict) and isinstance(previous_dispatch, dict):
            same_boot = projected['boot_id'] == previous['boot_id']
            if not same_boot and not allow_restart:
                raise TeleopServiceError('driver_identity_changed', 409)
            if same_boot:
                current_generation = current_dispatch['generation']
                previous_generation = previous_dispatch['generation']
                if current_generation < previous_generation:
                    if superseded_while_inflight():
                        return False
                    raise TeleopServiceError('driver_response_invalid', 502)
                stop_state_changed = (
                    current_dispatch['state'] in _STOP_DISPATCH_STATES
                    and (
                        projected['state'],
                        projected['reason'],
                    ) != (
                        previous['state'],
                        previous['reason'],
                    )
                )
                if stop_state_changed and current_generation <= previous_generation:
                    if superseded_while_inflight():
                        return False
                    raise TeleopServiceError('driver_response_invalid', 502)
                if current_generation == previous_generation:
                    progression_regressed = (
                        previous_dispatch['stop_acknowledged'] is True
                        and current_dispatch['stop_acknowledged'] is False
                    ) or (
                        previous_dispatch['stop_queue_depth'] == 0
                        and current_dispatch['stop_queue_depth'] > 0
                    )
                    evidence_key = (
                        'last_published_sequence'
                        if current_dispatch.get('kind') == 'hardware'
                        else 'last_would_apply_sequence'
                    )
                    for key in ('last_admitted_sequence', evidence_key):
                        before = previous_dispatch[key]
                        after = current_dispatch[key]
                        progression_regressed = progression_regressed or (
                            before is not None and (after is None or after < before)
                    )
                    if progression_regressed:
                        if superseded_while_inflight():
                            return False
                        raise TeleopServiceError('driver_response_invalid', 502)
        self._driver_snapshots[session.id] = projected
        self._driver_snapshot_revisions[session.id] = (
            self._driver_snapshot_revisions.get(session.id, 0) + 1
        )
        return True

    def _terminal_status_proves_safe(
        self,
        session: ShadowSession,
        projected: dict,
    ) -> bool:
        if (
            projected['authority_valid'] is not False
            or projected['session_id'] is not None
        ):
            return False
        dispatch = projected.get('dispatch')
        if (
            not isinstance(dispatch, dict)
            or dispatch['stop_acknowledged'] is not True
            or dispatch['stop_queue_depth'] != 0
            or dispatch['io_inflight'] is not None
        ):
            return False
        if projected['boot_id'] == session.boot_id:
            return (
                projected['epoch'] == session.epoch
                and dispatch['state'] == 'safe_revoked'
            )
        return (
            projected['state'] == 'idle'
            and dispatch['state'] == 'safe_unarmed'
        )

    async def _cleanup_terminal_runtime(self, session_id: str) -> None:
        if session_id not in self._terminal_cleanup_done:
            raise RuntimeError('terminal cleanup requires acknowledged safety proof')
        if any(
            guard.session_id == session_id
            for guard in self._authority_guards.values()
        ):
            raise RuntimeError('terminal cleanup requires durable guard deletion')
        self._pinned_targets.pop(session_id, None)
        self._signaling_sources.pop(session_id, None)
        self._driver_lease_seconds.pop(session_id, None)
        retained = await self.manager.retained_session_ids()
        retained.update(self._command_claims)
        retained.update(self._release_reconcile_tasks)
        for mapping in (
            self._driver_snapshots,
            self._driver_snapshot_revisions,
            self._heartbeat_health,
            self._release_results,
        ):
            for stale_id in set(mapping) - retained:
                mapping.pop(stale_id, None)
        self._terminal_cleanup_done.intersection_update(retained)
        claim = self._command_claims.get(session_id)
        if (
            claim is not None
            and await self.command_broker.release_authority(claim[0], claim[1])
        ):
            self._command_claims.pop(session_id, None)
        reconcile = self._release_reconcile_tasks.pop(session_id, None)
        if reconcile is not None and reconcile is not asyncio.current_task():
            reconcile.cancel()
            await asyncio.gather(reconcile, return_exceptions=True)

    def _schedule_release_reconcile(self, session: ShadowSession) -> None:
        if self._stopping:
            return
        existing = self._release_reconcile_tasks.get(session.id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._reconcile_unconfirmed_release(session),
            name=f'teleop-release-reconcile-{session.id}',
        )
        self._release_reconcile_tasks[session.id] = task

        def forget(done: asyncio.Task) -> None:
            if self._release_reconcile_tasks.get(session.id) is done:
                self._release_reconcile_tasks.pop(session.id, None)

        task.add_done_callback(forget)

    async def _finalize_or_quarantine_release(
        self,
        session: ShadowSession,
        acknowledged: bool,
    ) -> None:
        if acknowledged and await self._delete_authority_guard_after_safe(session):
            self._terminal_cleanup_done.add(session.id)
            await self._cleanup_terminal_runtime(session.id)
        else:
            self._schedule_release_reconcile(session)

    async def _reconcile_unconfirmed_release(self, session: ShadowSession) -> None:
        lease_seconds = self._driver_lease_seconds.get(
            session.id,
            _MAX_DRIVER_LEASE_SECONDS,
        )
        retry_seconds = min(
            _MAX_DRIVER_LEASE_SECONDS,
            max(_MIN_DRIVER_LEASE_SECONDS, lease_seconds),
        ) + _MAX_HEARTBEAT_INTERVAL_SECONDS
        while not self._stopping and session.id in self._command_claims:
            await asyncio.sleep(retry_seconds)
            lock = self._session_lock(session.id)
            async with lock:
                if self._stopping or session.id not in self._command_claims:
                    return
                if await self._best_effort_release(session):
                    self._release_results[session.id] = True
                    if not await self._delete_authority_guard_after_safe(session):
                        continue
                    self._terminal_cleanup_done.add(session.id)
                    await audit.emit(
                        'teleop.session.release_reconciled',
                        session_id=session.id,
                        robot_id=session.robot_id,
                        principal_id=session.principal_id,
                        source='core',
                        decision='released',
                        reason='driver_release_retry_acknowledged',
                    )
                    await self._cleanup_terminal_runtime(session.id)
                    return
                try:
                    raw = await self._call(
                        session.driver_id,
                        'status',
                        timeout_seconds=0.5,
                        pinned_target=self._pinned_targets.get(session.id),
                    )
                    projected, _ = _project_driver_snapshot(
                        raw,
                        driver_id=session.driver_id,
                        robot_id=session.robot_id,
                        capability_digest=session.capability_digest,
                        action='status',
                        expected_mode=session.mode,
                        expected_profile_id=session.profile_id,
                        expected_capabilities=session.capabilities,
                        allow_stop_pending=True,
                    )
                except Exception:  # noqa: BLE001, S112 -- quarantine stays closed
                    continue
                if self._terminal_status_proves_safe(session, projected):
                    try:
                        self._store_driver_snapshot(
                            session,
                            projected,
                            allow_restart=True,
                        )
                    except TeleopServiceError:
                        continue
                    self._release_results[session.id] = True
                    if not await self._delete_authority_guard_after_safe(session):
                        continue
                    self._terminal_cleanup_done.add(session.id)
                    await audit.emit(
                        'teleop.session.release_reconciled',
                        session_id=session.id,
                        robot_id=session.robot_id,
                        principal_id=session.principal_id,
                        source='core',
                        decision='released',
                        reason='driver_authority_observed_inactive',
                    )
                    await self._cleanup_terminal_runtime(session.id)
                    return

    async def _cancel_heartbeat(
        self,
        session_id: str,
        *,
        origin_task: asyncio.Task | None,
    ) -> None:
        task = self._heartbeat_tasks.pop(session_id, None)
        if task is not None and task is not origin_task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _await_safety_completion(self, task: asyncio.Task):
        """Finish Driver revocation even when the initiating request is cancelled."""

        self._safety_tasks.add(task)
        cancelled = False
        try:
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if task.done():
                        result = task.result()
                        break
            if cancelled:
                raise asyncio.CancelledError
            return result
        finally:
            if task.done():
                self._safety_tasks.discard(task)

    async def _drain_safety_tasks(self) -> None:
        current = asyncio.current_task()
        while True:
            pending = [
                task for task in self._safety_tasks
                if task is not current and not task.done()
            ]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def _prepare_with_reconcile(self, session: ShadowSession) -> dict:
        prepare_action = PREPARE_ACTION_BY_MODE[session.mode]
        try:
            return await self._call(
                session.driver_id,
                prepare_action,
                {
                    'session_id': session.id,
                    'epoch': session.epoch,
                    'fence': session.fence,
                },
            )
        except mcp_client.TrustedShadowTransportError as error:
            ambiguous = error.code in {'timeout', 'network_error'} or (
                error.code == 'http_error'
                and error.http_status is not None
                and 500 <= error.http_status < 600
            )
            if not ambiguous:
                raise
            raw_status = await self._call(
                session.driver_id,
                'status',
                pinned_target=self._pinned_targets.get(session.id),
            )
            _project_driver_snapshot(
                raw_status,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=session.capability_digest,
                action=prepare_action,
                session=session,
            )
            return raw_status

    async def _audit_prepare_failure(self, session: ShadowSession, reason: str) -> None:
        await audit.emit(
            'teleop.session.prepare_failed',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=session.principal_id,
            source='core',
            decision='faulted',
            reason=reason,
        )

    @staticmethod
    def _identity(session: ShadowSession) -> dict:
        return {
            'boot_id': session.boot_id,
            'session_id': session.id,
            'epoch': session.epoch,
            'fence': session.fence,
        }

    @staticmethod
    def _heartbeat_timeout(driver_lease_seconds: float) -> float:
        del driver_lease_seconds
        return 0.25

    def _start_heartbeat(self, session: ShadowSession) -> None:
        old = self._heartbeat_tasks.pop(session.id, None)
        if old is not None:
            old.cancel()
        task = asyncio.create_task(
            self._heartbeat_loop(session.id, session.operation_generation),
            name=f'teleop-heartbeat-{session.id}',
        )
        self._heartbeat_tasks[session.id] = task
        task.add_done_callback(
            lambda completed, session_id=session.id, generation=session.operation_generation: (
                self._heartbeat_done(session_id, generation, completed)
            )
        )

    def _heartbeat_done(
        self,
        session_id: str,
        generation: int,
        task: asyncio.Task,
    ) -> None:
        if self._heartbeat_tasks.get(session_id) is task:
            self._heartbeat_tasks.pop(session_id, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None and not self._stopping:
                fault_task = asyncio.create_task(
                    self._fault_crashed_worker(session_id, generation),
                    name=f'teleop-heartbeat-fault-{session_id}',
                )
                self._supervisor_tasks.add(fault_task)
                fault_task.add_done_callback(self._supervisor_done)

    def _supervisor_done(self, task: asyncio.Task) -> None:
        self._supervisor_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _fault_crashed_worker(self, session_id: str, generation: int) -> None:
        session = await self.manager.get_current(session_id)
        if session is not None and session.operation_generation == generation:
            await self._fault_session(session, 'heartbeat_supervisor_error')

    async def _heartbeat_loop(self, session_id: str, generation: int) -> None:
        lease_seconds = self._driver_lease_seconds[session_id]
        interval = min(_MAX_HEARTBEAT_INTERVAL_SECONDS, lease_seconds / 4.0)
        # Schedule transport work against the event loop's clock.  The
        # injectable monotonic clock remains the authority for lease-age
        # decisions, but it may be deliberately frozen or advanced in tests.
        # Mixing that clock with ``asyncio.sleep`` makes the second and third
        # heartbeat drift to 2x/3x the intended interval when it is frozen.
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + interval
        while not self._stopping:
            await asyncio.sleep(max(0.0, next_tick - loop.time()))
            next_tick += interval
            session = await self.manager.get_current(session_id)
            if session is None or session.operation_generation != generation:
                return
            health = self._heartbeat_health.get(session_id)
            if (
                health is None
                or self._monotonic() - health.last_confirmed_monotonic
                >= lease_seconds * 0.75
            ):
                await self._fault_session(session, 'driver_heartbeat_lost')
                return
            try:
                # Authority heartbeats deliberately do not share the operator
                # operation lock.  Slow read-only status, Pause, or HOLD calls
                # must never consume the Driver's short watchdog budget.
                current = await self.manager.get_current(session_id)
                if current is None or current.operation_generation != generation:
                    return
                snapshot_revision = self._driver_snapshot_revisions.get(
                    session_id,
                    0,
                )
                raw = await self._call(
                    current.driver_id,
                    'heartbeat',
                    self._identity(current),
                    timeout_seconds=self._heartbeat_timeout(lease_seconds),
                )
                projected, _ = _project_driver_snapshot(
                    raw,
                    driver_id=current.driver_id,
                    robot_id=current.robot_id,
                    capability_digest=current.capability_digest,
                    action='heartbeat',
                    session=current,
                    allow_stop_pending=True,
                )
                self._store_driver_snapshot(
                    current,
                    projected,
                    stale_after_revision=snapshot_revision,
                )
                if projected['state'] == 'fault':
                    await self._fault_session(current, 'driver_dispatch_fault')
                    return
                if projected['dispatch']['stop_acknowledged'] is not True:
                    # A pending final-output stop is valid transitional data,
                    # not evidence that the safety path has completed.
                    continue
                # A concurrently completed Status/Pause may make this snapshot
                # stale for display, but the validated heartbeat still renewed
                # the Driver authority lease and therefore proves liveness.
                await self._record_heartbeat_success(current)
            except mcp_client.TrustedShadowTransportError as error:
                if (
                    error.code == 'rpc_error'
                    and error.rpc_data_code in _TERMINAL_RPC_CODES
                    and await self._capture_terminal_fault_after_heartbeat(
                        current,
                        snapshot_revision=snapshot_revision,
                    )
                ):
                    return
                if await self._record_heartbeat_failure(current, error):
                    return
            except TeleopServiceError as error:
                await self._fault_session(session, error.code)
                return

    async def _record_heartbeat_success(self, session: ShadowSession) -> None:
        health = self._heartbeat_health[session.id]
        recovered = health.state == 'degraded'
        health.last_confirmed_monotonic = self._monotonic()
        health.last_confirmed_at = self._wall_clock()
        health.consecutive_failures = 0
        health.state = 'healthy'
        if recovered:
            await audit.emit(
                'teleop.driver.recovered',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=session.principal_id,
                source='core',
                decision='healthy',
                reason='heartbeat_recovered',
            )

    async def _capture_terminal_fault_after_heartbeat(
        self,
        session: ShadowSession,
        *,
        snapshot_revision: int,
    ) -> bool:
        """Preserve one strict terminal fault snapshot before fail-safe cleanup."""

        target = self._pinned_targets.get(session.id)
        if target is None:
            return False
        try:
            raw = await self._call(
                session.driver_id,
                'status',
                timeout_seconds=0.5,
                pinned_target=target,
            )
            projected, _ = _project_driver_snapshot(
                raw,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=session.capability_digest,
                action='status',
                session=session,
                allow_stop_pending=True,
            )
        except Exception:  # noqa: BLE001 -- fallback keeps the original fail-closed path
            return False
        if projected['state'] != 'fault':
            return False

        lock = self._session_lock(session.id)
        async with lock:
            current = await self.manager.get_current(session.id)
            if (
                current is None
                or current is not session
                or current.operation_generation != session.operation_generation
            ):
                return True
            try:
                stored = self._store_driver_snapshot(
                    session,
                    projected,
                    stale_after_revision=snapshot_revision,
                )
            except TeleopServiceError:
                return False
            if not stored:
                return False
            await self._fault_session_locked(session, 'driver_dispatch_fault')
            return True

    async def _record_heartbeat_failure(
        self,
        session: ShadowSession,
        error: mcp_client.TrustedShadowTransportError,
    ) -> bool:
        health = self._heartbeat_health[session.id]
        retryable = error.code in _RETRYABLE_TRANSPORT_CODES or (
            error.code == 'http_error'
            and error.http_status is not None
            and 500 <= error.http_status < 600
        )
        if error.code == 'rpc_error' and error.rpc_data_code in _TERMINAL_RPC_CODES:
            retryable = False
        if not retryable:
            await self._fault_session(session, self._transport_service_error(error).code)
            return True

        health.consecutive_failures += 1
        if health.state != 'degraded':
            health.state = 'degraded'
            await audit.emit(
                'teleop.driver.degraded',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=session.principal_id,
                source='core',
                decision='retrying',
                reason=error.code,
            )
        lease_seconds = self._driver_lease_seconds[session.id]
        acknowledgement_age = self._monotonic() - health.last_confirmed_monotonic
        if health.consecutive_failures >= 3 or acknowledgement_age >= lease_seconds * 0.75:
            await self._fault_session(session, 'driver_heartbeat_lost')
            return True
        return False

    async def _fault_session(
        self,
        session: ShadowSession,
        reason: str,
        *,
        operation_locked: bool = False,
    ) -> None:
        if operation_locked:
            await self._fault_session_locked(session, reason)
            return
        lock = self._session_lock(session.id)
        async with lock:
            await self._fault_session_locked(session, reason)

    async def _fault_session_locked(self, session: ShadowSession, reason: str) -> None:
        faulted = await self.manager.fault(session.id, session.operation_generation)
        if faulted is None:
            return
        await self.capture_manager.revoke_for_session(session.id, 'session_faulted')
        origin_task = asyncio.current_task()
        cleanup = asyncio.create_task(
            self._complete_fault_cleanup(faulted, reason, origin_task=origin_task),
            name=f'teleop-fault-cleanup-{session.id}',
        )
        await self._await_safety_completion(cleanup)

    async def _complete_fault_cleanup(
        self,
        session: ShadowSession,
        reason: str,
        *,
        origin_task: asyncio.Task | None,
    ) -> None:
        await self._cancel_heartbeat(session.id, origin_task=origin_task)
        await self._update_command_claim(session, 'faulted_cleanup')
        health = self._heartbeat_health.get(session.id)
        if health is not None:
            health.state = 'faulted'
        acknowledged = await self._best_effort_soft_stop_and_release(session)
        await audit.emit(
            'teleop.session.faulted',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=session.principal_id,
            source='core',
            decision='faulted',
            reason=reason,
        )
        await self._finalize_or_quarantine_release(session, acknowledged)

    async def _best_effort_soft_stop_and_release(self, session: ShadowSession) -> bool:
        try:
            await self._call(
                session.driver_id,
                'soft_stop',
                self._identity(session),
                timeout_seconds=0.25,
            )
        except Exception:  # noqa: BLE001, S110 -- cleanup must be fail-safe
            pass
        return await self._best_effort_release(session)

    async def _best_effort_release(self, session: ShadowSession) -> bool:
        try:
            previous = self._driver_snapshots.get(session.id)
            baseline_generation = (
                previous['dispatch']['generation']
                if isinstance(previous, dict)
                else None
            )
            raw = await self._call(
                session.driver_id,
                'release',
                self._identity(session),
                timeout_seconds=0.5,
            )
            projected, _ = _project_driver_snapshot(
                raw,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=session.capability_digest,
                action='release',
                session=session,
            )
            self._store_driver_snapshot(
                session,
                projected,
                minimum_generation_exclusive=baseline_generation,
            )
            return True
        except Exception:  # noqa: BLE001 -- cleanup reports only acknowledgement
            return False

    async def create_capture_pairing(
        self,
        principal_id: str,
        client_id: str,
        *,
        label: str,
    ) -> CapturePairingResult:
        pairing = await self.capture_manager.create_pairing(
            principal_id,
            client_id,
            label=label,
        )
        await audit.emit(
            'teleop.capture.pairing_created',
            principal_id=principal_id,
            source='api',
            decision='created',
            reason='operator_pairing_request',
            details={'pairing_id': pairing.pairing_id},
        )
        return pairing

    async def list_captures(self, principal_id: str) -> list[dict]:
        return await self.capture_manager.list_for_supervisor(principal_id)

    async def revoke_capture(self, capture_id: str, principal_id: str) -> dict:
        capture = await self.capture_manager.revoke_capture(capture_id, principal_id)
        await audit.emit(
            'teleop.capture.revoked',
            session_id=(capture.get('assignment') or {}).get('session_id', ''),
            principal_id=principal_id,
            source='api',
            decision='revoked',
            reason='operator_revoked',
            details={'capture_id': capture_id},
        )
        return capture

    async def attach_capture(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        capture_id: str,
        mode: str,
        profile_id: str,
        capability_digest: str,
    ) -> CaptureAssignment:
        lock = self._session_lock(session_id)
        async with lock:
            session = await self.manager.get_authorized(
                session_id,
                principal_id,
                owner=False,
                client_id=client_id,
                require_client=True,
            )
            if (
                session.state != 'active'
                or (session.mode == LIVE_MODE and not session.live_confirmed)
            ):
                raise SessionStateConflict(session_id)
            if (
                mode != session.mode
                or profile_id != session.profile_id
                or capability_digest != session.capability_digest
            ):
                raise TeleopServiceError('capture_contract_mismatch', 409)
            source = self._signaling_sources.get(session.id)
            expected_source = ('capture', capture_id)
            if source is not None and source != expected_source:
                raise TeleopServiceError('signaling_source_conflict', 409)
            assignment = await self.capture_manager.attach(
                capture_id=capture_id,
                principal_id=principal_id,
                session_id=session.id,
                operation_generation=session.operation_generation,
                mode=session.mode,
                profile_id=session.profile_id,
                capability_digest=session.capability_digest,
                capabilities=session.capabilities,
                effectors=session.effectors,
            )
            # Source ownership belongs to the enrolled capture device, not to
            # one disposable assignment. A signaling failure fails this whole
            # session closed; the stable capture id only prevents another
            # device or a direct browser from racing the current assignment.
            if source is not None and source != expected_source:
                await self.capture_manager.revoke_for_session(
                    session.id,
                    'signaling_source_conflict',
                )
                raise TeleopServiceError('signaling_source_conflict', 409)
            self._signaling_sources[session.id] = expected_source
        await audit.emit(
            'teleop.capture.attached',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=principal_id,
            source='api',
            decision='attached',
            reason='operator_start_capture',
            details={
                'capture_id': capture_id,
                'assignment_id': assignment.id,
                'mode': assignment.mode,
            },
        )
        return assignment

    async def connect_capture_with_pairing(
        self,
        pairing_id: str,
        pairing_code: str,
        *,
        capture_protocol: str,
        frame_protocol: str,
        client_kind: str,
        app_version: str,
    ) -> CaptureConnection:
        return await self.capture_manager.connect_with_pairing(
            pairing_id,
            pairing_code,
            capture_protocol=capture_protocol,
            frame_protocol=frame_protocol,
            client_kind=client_kind,
            app_version=app_version,
        )

    async def connect_capture_with_credential(
        self,
        capture_id: str,
        capture_credential: str,
        *,
        capture_protocol: str,
        frame_protocol: str,
        client_kind: str,
        app_version: str,
    ) -> CaptureConnection:
        return await self.capture_manager.connect_with_credential(
            capture_id,
            capture_credential,
            capture_protocol=capture_protocol,
            frame_protocol=frame_protocol,
            client_kind=client_kind,
            app_version=app_version,
        )

    async def disconnect_capture(self, capture_id: str, connection_id: str) -> None:
        lost_assignment = await self.capture_manager.disconnect(
            capture_id,
            connection_id,
        )
        await self._fail_close_capture_assignment_loss(lost_assignment)

    async def capture_presence(
        self,
        capture_id: str,
        connection_id: str,
        *,
        state: str,
        assignment_id: str | None,
    ) -> dict:
        presence, lost_assignment = await self.capture_manager.update_presence(
            capture_id,
            connection_id,
            state=state,
            assignment_id=assignment_id,
        )
        await self._fail_close_capture_assignment_loss(lost_assignment)
        return presence

    async def expire_capture_connection(
        self,
        capture_id: str,
        connection_id: str,
    ) -> bool:
        stale, lost_assignment = await self.capture_manager.expire_stale_connection(
            capture_id,
            connection_id,
        )
        await self._fail_close_capture_assignment_loss(lost_assignment)
        return stale

    async def _fail_close_capture_assignment_loss(
        self,
        lost_assignment: CaptureAssignment | None,
    ) -> None:
        if lost_assignment is None:
            return
        cleanup = asyncio.create_task(
            self._complete_capture_assignment_loss(lost_assignment),
            name=(
                'teleop-capture-loss-'
                f'{lost_assignment.session_id}-{lost_assignment.generation}'
            ),
        )
        await self._await_safety_completion(cleanup)

    async def _complete_capture_assignment_loss(
        self,
        lost_assignment: CaptureAssignment,
    ) -> None:
        reason = lost_assignment.failure_code
        if (
            lost_assignment.state != 'revoked'
            or reason not in _CAPTURE_FAIL_CLOSE_REASONS
        ):
            return

        session: ShadowSession | None = None
        held: ShadowSession | None = None
        lock = self._session_lock(lost_assignment.session_id)
        try:
            async with lock:
                if not await self.capture_manager.assignment_loss_is_pending(
                    lost_assignment,
                ):
                    return
                session = await self.manager.get_current(lost_assignment.session_id)
                if (
                    session is None
                    or session.state != 'active'
                    or session.operation_generation
                    != lost_assignment.operation_generation
                    or session.mode != lost_assignment.mode
                    or session.profile_id != lost_assignment.profile_id
                    or session.capability_digest != lost_assignment.capability_digest
                    or self._signaling_sources.get(session.id)
                    != ('capture', lost_assignment.capture_id)
                ):
                    return
                try:
                    held = await self._soft_stop_locked(
                        session,
                        session.principal_id,
                        session.client_id,
                        owner=False,
                    )
                except (TeleopServiceError, SessionStateConflict):
                    # `_soft_stop_locked` faults the still-current session whenever
                    # Driver HOLD cannot be proven. Do not turn a safety-complete
                    # WSS disconnect into a second external error path.
                    return
        finally:
            await self.capture_manager.complete_assignment_loss(lost_assignment)

        assert session is not None and held is not None
        await audit.emit(
            'teleop.capture.assignment_lost',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=session.principal_id,
            source='capture-wss',
            decision='hold',
            reason=reason,
            details={
                'assignment_id': lost_assignment.id,
                'assignment_generation': lost_assignment.generation,
                'capture_id': lost_assignment.capture_id,
                'state': held.state,
            },
        )

    async def capture_signaling_offer(
        self,
        capture_id: str,
        connection_id: str,
        assignment_id: str,
        offer: dict,
    ) -> dict:
        """Consume one capture offer without giving the device session authority."""

        assignment = await self.capture_manager.claim_offer(
            capture_id,
            connection_id,
            assignment_id,
        )
        session: ShadowSession | None = None
        try:
            signaling_lock = self._signaling_lock(assignment.session_id)
            async with signaling_lock:
                lock = self._session_lock(assignment.session_id)
                async with lock:
                    session = await self.manager.get_current(assignment.session_id)
                    if (
                        session is None
                        or session.state not in _LIVE_SIGNALING_STATES
                        or session.operation_generation != assignment.operation_generation
                        or session.mode != assignment.mode
                        or session.profile_id != assignment.profile_id
                        or session.capability_digest != assignment.capability_digest
                        or (session.mode == LIVE_MODE and not session.live_confirmed)
                    ):
                        raise SessionStateConflict(assignment.session_id)
                    if self._signaling_sources.get(session.id) != (
                        'capture', capture_id,
                    ):
                        raise TeleopServiceError('signaling_source_conflict', 409)
                    offer_sdp = _validated_signaling_offer_sdp(offer)
                    target = self._pinned_targets.get(session.id)
                    if (
                        target is None
                        or target.mcp_id != session.driver_id
                        or target.capability_digest != session.capability_digest
                    ):
                        raise TeleopServiceError('driver_not_ready', 503)
                    operation_generation = session.operation_generation
                    ticket = _signaling_ticket(
                        session,
                        offer_sdp,
                        wall_now=self._wall_clock(),
                    )

                try:
                    raw_answer = await self._signal_offer(
                        session,
                        offer,
                        ticket,
                        target,
                    )
                except mcp_client.TrustedShadowTransportError as error:
                    raise self._transport_service_error(error) from None
                answer = _project_signaling_answer(raw_answer, session, ticket)

                async with lock:
                    current = await self.manager.get_current(session.id)
                    assignment_current = await self.capture_manager.assignment_is_current(
                        capture_id,
                        connection_id,
                        assignment.id,
                        session_id=session.id,
                        operation_generation=operation_generation,
                    )
                    if (
                        current is not session
                        or current.operation_generation != operation_generation
                        or current.state not in _LIVE_SIGNALING_STATES
                        or self._pinned_targets.get(session.id) != target
                        or not assignment_current
                    ):
                        raise SessionStateConflict(session.id)
                    if not await self.capture_manager.complete_offer(
                        capture_id,
                        connection_id,
                        assignment.id,
                    ):
                        raise SessionStateConflict(session.id)
        except BaseException as error:
            lost_assignment = await self.capture_manager.fail_offer(
                capture_id,
                assignment.id,
            )
            await self._fail_close_capture_assignment_loss(lost_assignment)
            if isinstance(error, (CaptureError, TeleopServiceError, SessionStateConflict)):
                raise
            if isinstance(error, asyncio.CancelledError):
                raise
            raise TeleopServiceError('capture_signaling_failed', 502) from None

        assert session is not None
        await audit.emit(
            'teleop.capture.signaling_offer_accepted',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=session.principal_id,
            source='capture-wss',
            decision='accepted',
            reason='capture_assignment_authorized',
            details={
                'capture_id': capture_id,
                'assignment_id': assignment.id,
                'mode': session.mode,
            },
        )
        return answer

    async def signaling_offer(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        offer: dict,
    ) -> dict:
        """Negotiate RTC without exposing or renewing session authority."""

        session: ShadowSession | None = None
        failure: Exception | None = None
        try:
            # Serialize competing offers without making a network round-trip
            # part of the safety-operation critical section. Pause/HOLD/release
            # must always be able to preempt a slow or abandoned offer.
            signaling_lock = self._signaling_lock(session_id)
            async with signaling_lock:
                lock = self._session_lock(session_id)
                async with lock:
                    # Signaling has no owner override: both the human principal
                    # and per-tab client UUID must match the reservation.
                    session = await self.manager.get_authorized(
                        session_id,
                        principal_id,
                        owner=False,
                        client_id=client_id,
                        require_client=True,
                    )
                    if (
                        session.mode not in {SHADOW_MODE, LIVE_MODE}
                        or session.state not in _LIVE_SIGNALING_STATES
                        or (session.mode == LIVE_MODE and not session.live_confirmed)
                    ):
                        raise SessionStateConflict(session_id)
                    source = self._signaling_sources.get(session.id)
                    expected_source = ('direct', client_id)
                    if source is not None and source != expected_source:
                        raise TeleopServiceError('signaling_source_conflict', 409)
                    offer_sdp = _validated_signaling_offer_sdp(offer)
                    target = self._pinned_targets.get(session.id)
                    if (
                        target is None
                        or target.mcp_id != session.driver_id
                        or target.capability_digest != session.capability_digest
                    ):
                        raise TeleopServiceError('driver_not_ready', 503)
                    operation_generation = session.operation_generation
                    ticket = _signaling_ticket(
                        session,
                        offer_sdp,
                        wall_now=self._wall_clock(),
                    )
                    self._signaling_sources[session.id] = expected_source

                try:
                    raw_answer = await self._signal_offer(
                        session,
                        offer,
                        ticket,
                        target,
                    )
                except mcp_client.TrustedShadowTransportError as error:
                    raise self._transport_service_error(error) from None
                answer = _project_signaling_answer(raw_answer, session, ticket)

                # Re-enter the safety lock before returning the answer. This
                # read does not renew the browser lease and rejects any offer
                # preempted by expiry, Pause, HOLD, release, or target cleanup.
                async with lock:
                    current = await self.manager.get_current(session.id)
                    if (
                        current is not session
                        or current.operation_generation != operation_generation
                        or self._pinned_targets.get(session.id) != target
                    ):
                        raise SessionNotFound(session.id)
                    if current.state not in _LIVE_SIGNALING_STATES:
                        raise SessionStateConflict(session.id)
        except (
            SessionClientMismatch,
            SessionForbidden,
            SessionNotFound,
            SessionStateConflict,
            TeleopServiceError,
        ) as error:
            failure = error

        if failure is not None:
            reason = (
                failure.code
                if isinstance(failure, TeleopServiceError)
                else 'session_client_mismatch'
                if isinstance(failure, SessionClientMismatch)
                else 'session_forbidden'
                if isinstance(failure, SessionForbidden)
                else 'session_state_conflict'
                if isinstance(failure, SessionStateConflict)
                else 'session_not_found'
                if isinstance(failure, SessionNotFound)
                else 'signaling_failed'
            )
            await audit.emit(
                'teleop.signaling.offer.rejected',
                session_id=session_id,
                robot_id=session.robot_id if session is not None else '',
                principal_id=principal_id,
                source='api',
                decision='rejected',
                reason=reason,
            )
            raise failure

        assert session is not None
        await audit.emit(
            'teleop.signaling.offer.accepted',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=principal_id,
            source='api',
            decision='accepted',
            reason='rtc_offer_authorized',
            details={'mode': session.mode, 'answer_type': 'answer'},
        )
        return answer

    async def heartbeat(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
    ) -> ShadowSession:
        """Renew only the browser → Core ownership lease.

        Owners deliberately cannot renew another operator's session.
        """

        return await self.manager.heartbeat(
            session_id,
            principal_id,
            client_id,
            owner=False,
        )

    async def pause(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> ShadowSession:
        lock = self._session_lock(session_id)
        async with lock:
            session = await self.manager.get_authorized(
                session_id,
                principal_id,
                owner=owner,
                client_id=client_id,
                require_client=True,
            )
            if session.state == 'awaiting_confirmation':
                raise SessionStateConflict(session_id)
            await self.capture_manager.revoke_for_session(session.id, 'operator_pause')
            if session.state == 'paused':
                return session
            baseline_generation = self._driver_snapshots[
                session.id
            ]['dispatch']['generation']
            try:
                raw = await self._call(
                    session.driver_id,
                    'pause',
                    self._identity(session),
                    timeout_seconds=0.5,
                )
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(
                    self._fault_session(
                        session,
                        'driver_pause_unconfirmed',
                        operation_locked=True,
                    ),
                    name=f'teleop-cancelled-pause-{session.id}',
                )
                await self._await_safety_completion(cleanup)
                raise
            except mcp_client.TrustedShadowTransportError as error:
                service_error = self._transport_service_error(error)
                await self._fault_session(
                    session,
                    'driver_pause_unconfirmed',
                    operation_locked=True,
                )
                raise service_error from None
            try:
                projected, _ = _project_driver_snapshot(
                    raw,
                    driver_id=session.driver_id,
                    robot_id=session.robot_id,
                    capability_digest=session.capability_digest,
                    action='pause',
                    session=session,
                )
                self._store_driver_snapshot(
                    session,
                    projected,
                    minimum_generation_exclusive=baseline_generation,
                )
            except Exception:  # noqa: BLE001 -- an unconfirmed Pause must revoke
                await self._fault_session(
                    session,
                    'driver_pause_unconfirmed',
                    operation_locked=True,
                )
                raise TeleopServiceError('driver_pause_rejected', 502) from None
            paused = await self.manager.pause(
                session.id,
                principal_id,
                client_id,
                owner=owner,
            )
            await self._update_command_claim(paused, 'paused')
        await audit.emit(
            'teleop.session.paused',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=principal_id,
            source='api',
            decision='paused',
            reason='operator_pause',
        )
        return paused

    async def soft_stop(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> ShadowSession:
        lock = self._session_lock(session_id)
        async with lock:
            session = await self.manager.get_authorized(
                session_id,
                principal_id,
                owner=owner,
                client_id=client_id,
                require_client=True,
            )
            if session.state == 'awaiting_confirmation':
                raise SessionStateConflict(session_id)
            await self.capture_manager.revoke_for_session(session.id, 'operator_soft_stop')
            if session.state == 'hold':
                return session
            held = await self._soft_stop_locked(
                session,
                principal_id,
                client_id,
                owner=owner,
            )
        await audit.emit(
            'teleop.session.held',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=principal_id,
            source='api',
            decision='hold',
            reason='operator_soft_stop',
        )
        return held

    async def _soft_stop_locked(
        self,
        session: ShadowSession,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> ShadowSession:
        """Prove Driver HOLD, or fault the still-current session fail-closed."""

        try:
            baseline_generation = self._driver_snapshots[
                session.id
            ]['dispatch']['generation']
        except (KeyError, TypeError):
            await self._fault_session(
                session,
                'driver_soft_stop_unconfirmed',
                operation_locked=True,
            )
            raise TeleopServiceError('driver_soft_stop_rejected', 502) from None
        try:
            raw = await self._call(
                session.driver_id,
                'soft_stop',
                self._identity(session),
                timeout_seconds=0.5,
            )
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._fault_session(
                    session,
                    'driver_soft_stop_unconfirmed',
                    operation_locked=True,
                ),
                name=f'teleop-cancelled-soft-stop-{session.id}',
            )
            await self._await_safety_completion(cleanup)
            raise
        except mcp_client.TrustedShadowTransportError as error:
            service_error = self._transport_service_error(error)
            await self._fault_session(
                session,
                'driver_soft_stop_unconfirmed',
                operation_locked=True,
            )
            raise service_error from None
        try:
            projected, _ = _project_driver_snapshot(
                raw,
                driver_id=session.driver_id,
                robot_id=session.robot_id,
                capability_digest=session.capability_digest,
                action='soft_stop',
                session=session,
            )
            self._store_driver_snapshot(
                session,
                projected,
                minimum_generation_exclusive=baseline_generation,
            )
        except Exception:  # noqa: BLE001 -- an unconfirmed HOLD must revoke
            await self._fault_session(
                session,
                'driver_soft_stop_unconfirmed',
                operation_locked=True,
            )
            raise TeleopServiceError('driver_soft_stop_rejected', 502) from None
        held = await self.manager.soft_stop(
            session.id,
            principal_id,
            client_id,
            owner=owner,
        )
        await self._update_command_claim(held, 'hold')
        return held

    async def release(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        owner: bool,
    ) -> tuple[ShadowSession, bool]:
        lock = self._session_lock(session_id)
        async with lock:
            existing = await self.manager.get_authorized(
                session_id,
                principal_id,
                owner=owner,
                include_terminal=True,
                client_id=client_id,
                require_client=True,
            )
            await self.capture_manager.revoke_for_session(existing.id, 'operator_release')
            if existing.state in {'expired', 'faulted'}:
                return existing, False
            if existing.state == 'released':
                if self._release_results.get(existing.id) is True:
                    return existing, self._release_results[existing.id]
                # Recover an interrupted local terminalization.  The Driver's
                # exact identity tuple makes this bounded retry idempotent.
                released = existing
            else:
                released = await self.manager.release(
                    session_id,
                    principal_id,
                    client_id,
                    owner=owner,
                )
            if released.mode == LIVE_MODE and not released.live_confirmed:
                await self._update_command_claim(released, 'releasing')
                self._release_results[released.id] = True
                self._terminal_cleanup_done.add(released.id)
                await audit.emit(
                    'teleop.session.released',
                    session_id=released.id,
                    robot_id=released.robot_id,
                    principal_id=principal_id,
                    source='api',
                    decision='released',
                    reason='unconfirmed_live_reservation_released',
                    details={
                        'state': 'released',
                        'driver_contacted': False,
                        'hardware_output': False,
                    },
                )
                await self._cleanup_terminal_runtime(released.id)
                return released, True
            await self._update_command_claim(released, 'releasing')
            origin_task = asyncio.current_task()
            cleanup = asyncio.create_task(
                self._complete_release_cleanup(
                    released,
                    principal_id,
                    origin_task=origin_task,
                ),
                name=f'teleop-release-cleanup-{session_id}',
            )
            acknowledged = await self._await_safety_completion(cleanup)
            return released, acknowledged

    async def _complete_release_cleanup(
        self,
        released: ShadowSession,
        principal_id: str,
        *,
        origin_task: asyncio.Task | None,
    ) -> bool:
        await self._cancel_heartbeat(released.id, origin_task=origin_task)
        acknowledged = await self._best_effort_release(released)
        self._release_results[released.id] = acknowledged
        await audit.emit(
            'teleop.session.released',
            session_id=released.id,
            robot_id=released.robot_id,
            principal_id=principal_id,
            source='api',
            decision='released',
            reason='operator_release',
            details={'state': 'released', 'driver_acknowledged': acknowledged},
        )
        await self._finalize_or_quarantine_release(released, acknowledged)
        return acknowledged

    async def status(self, session_id: str, principal_id: str, *, owner: bool) -> dict:
        lock = self._session_lock(session_id)
        async with lock:
            session = await self.manager.get_authorized(session_id, principal_id, owner=owner)
            if session.state == 'awaiting_confirmation':
                raise SessionStateConflict(session_id)
            snapshot_revision = self._driver_snapshot_revisions.get(session.id, 0)
            try:
                raw = await self._call(
                    session.driver_id,
                    'status',
                    timeout_seconds=0.5,
                    pinned_target=self._pinned_targets.get(session.id),
                )
            except mcp_client.TrustedShadowTransportError as error:
                raise self._transport_service_error(error) from None
            try:
                projected, lease_seconds = _project_driver_snapshot(
                    raw,
                    driver_id=session.driver_id,
                    robot_id=session.robot_id,
                    capability_digest=session.capability_digest,
                    action='status',
                    session=session,
                    allow_stop_pending=True,
                )
            except Exception:  # noqa: BLE001 -- invalid authority status must revoke
                await self._fault_session(
                    session,
                    'driver_session_lost',
                    operation_locked=True,
                )
                raise TeleopServiceError('driver_session_lost', 409) from None
            if projected['state'] == 'fault':
                current = await self.manager.get_current(session.id)
                if (
                    current is not session
                    or current.operation_generation != session.operation_generation
                ):
                    raise SessionNotFound(session.id)
                try:
                    self._store_driver_snapshot(
                        session,
                        projected,
                        stale_after_revision=snapshot_revision,
                    )
                except TeleopServiceError:
                    await self._fault_session(
                        session,
                        'driver_session_lost',
                        operation_locked=True,
                    )
                    raise TeleopServiceError('driver_session_lost', 409) from None
                self._driver_lease_seconds[session.id] = lease_seconds
                await self._fault_session(
                    session,
                    'driver_dispatch_fault',
                    operation_locked=True,
                )
                return self.public_session(session)
            if (
                projected['authority_valid'] is not True
                or projected['session_id'] != session.id
                or projected['epoch'] != session.epoch
                or projected['boot_id'] != session.boot_id
            ):
                await self._fault_session(
                    session,
                    'driver_session_lost',
                    operation_locked=True,
                )
                raise TeleopServiceError('driver_session_lost', 409)
            current = await self.manager.get_current(session.id)
            if (
                current is not session
                or current.operation_generation != session.operation_generation
            ):
                raise SessionNotFound(session.id)
            try:
                self._store_driver_snapshot(
                    session,
                    projected,
                    stale_after_revision=snapshot_revision,
                )
            except TeleopServiceError:
                await self._fault_session(
                    session,
                    'driver_session_lost',
                    operation_locked=True,
                )
                raise TeleopServiceError('driver_session_lost', 409) from None
            self._driver_lease_seconds[session.id] = lease_seconds
            return self.public_session(session)

    async def sessions_for(self, principal_id: str, *, owner: bool) -> list[dict]:
        sessions = await self.manager.list_visible(principal_id, owner=owner)
        return [self.public_session(session) for session in sessions]

    async def session_for(
        self,
        session_id: str,
        principal_id: str,
        *,
        owner: bool,
        include_terminal: bool = True,
    ) -> dict:
        session = await self.manager.get_authorized(
            session_id,
            principal_id,
            owner=owner,
            include_terminal=include_terminal,
        )
        return self.public_session(session)

    def public_session(self, session: ShadowSession) -> dict:
        public = self.manager.public_dict(session)
        public['driver'] = self._driver_snapshots.get(session.id)
        health = self._heartbeat_health.get(session.id)
        public['driver_heartbeat'] = health.public_dict() if health else {
            'state': 'pending' if session.state == 'preparing' else 'stopped',
            'last_confirmed_at': None,
            'consecutive_failures': 0,
        }
        if session.fence in repr(public):
            raise RuntimeError('teleop public projection contained private authority')
        return public

    async def _reaper_loop(self) -> None:
        while not self._stopping:
            try:
                expired = await self.manager.expire_due()
                for session in expired:
                    lock = self._session_lock(session.id)
                    async with lock:
                        if session.id in self._terminal_cleanup_done:
                            continue
                        origin_task = asyncio.current_task()
                        cleanup = asyncio.create_task(
                            self._complete_expiry_cleanup(
                                session,
                                origin_task=origin_task,
                            ),
                            name=f'teleop-expiry-cleanup-{session.id}',
                        )
                        await self._await_safety_completion(cleanup)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001, S110 -- keep the safety reaper alive
                pass
            await asyncio.sleep(_REAPER_INTERVAL_SECONDS)

    async def _complete_expiry_cleanup(
        self,
        session: ShadowSession,
        *,
        origin_task: asyncio.Task | None,
    ) -> None:
        await self.capture_manager.revoke_for_session(session.id, 'session_expired')
        await self._cancel_heartbeat(session.id, origin_task=origin_task)
        await self._update_command_claim(session, 'expired_cleanup')
        if session.mode == LIVE_MODE and not session.live_confirmed:
            await audit.emit(
                'teleop.session.expired',
                session_id=session.id,
                robot_id=session.robot_id,
                principal_id=session.principal_id,
                source='core',
                decision='expired',
                reason='unconfirmed_live_reservation_expired',
                details={'driver_contacted': False, 'hardware_output': False},
            )
            self._terminal_cleanup_done.add(session.id)
            await self._cleanup_terminal_runtime(session.id)
            return
        acknowledged = await self._best_effort_soft_stop_and_release(session)
        await audit.emit(
            'teleop.session.expired',
            session_id=session.id,
            robot_id=session.robot_id,
            principal_id=session.principal_id,
            source='core',
            decision='expired',
            reason='browser_lease_expired',
        )
        await self._finalize_or_quarantine_release(session, acknowledged)


coordinator = TeleopCoordinator()


__all__ = [
    'AcquireResult',
    'SessionConflict',
    'SessionForbidden',
    'SessionNotFound',
    'SessionStateConflict',
    'TeleopCoordinator',
    'TeleopServiceError',
    'coordinator',
]
