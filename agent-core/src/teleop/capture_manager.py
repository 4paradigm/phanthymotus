from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import re
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PAIRING_TTL_SECONDS = 60.0
MAX_PAIRINGS_PER_SUPERVISOR = 4
MAX_PAIRING_RECORDS = 256
MAX_CAPTURES_PER_SUPERVISOR = 4
MAX_CAPTURE_RECORDS = 256
CAPTURE_EVENT_QUEUE_SIZE = 8
CAPTURE_PRESENCE_TIMEOUT_SECONDS = 5.0

_CAPTURE_PRESENCE_STATES = frozenset({
    'browser_ready',
    'error',
    'rtc_connecting',
    'streaming',
    'xr_ended',
    'xr_standby',
})
_ASSIGNMENT_PRESENCE_STATES = frozenset({'error', 'rtc_connecting', 'streaming'})
_CAPTURE_PROTOCOL = 'motus.teleop.capture.v1'
_FRAME_PROTOCOL = 'motus.teleop.rtc-frame.v1'
_CLIENT_KINDS = frozenset({'browser_webxr', 'native_openxr'})
_APP_VERSION_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]{0,31}$')
_ACTIVE_ASSIGNMENT_STATES = frozenset({
    'issued',
    'negotiated',
    'offer_consumed',
})


class CaptureError(RuntimeError):
    """Stable, secret-free capture contract failure."""

    def __init__(self, code: str, status_code: int):
        self.code = code
        self.status_code = status_code
        super().__init__(f'capture error: {code}')


@dataclass(frozen=True)
class CapturePairingResult:
    pairing_id: str
    pairing_code: str = field(repr=False)
    expires_at: float


@dataclass(frozen=True)
class CaptureConnection:
    capture_id: str
    connection_id: str
    events: asyncio.Queue
    capture_credential: str | None = field(default=None, repr=False)


@dataclass
class _Pairing:
    id: str
    principal_id: str = field(repr=False)
    client_id: str = field(repr=False)
    label: str
    code_digest: bytes = field(repr=False)
    created_at: float
    expires_at: float


@dataclass
class _Capture:
    id: str
    principal_id: str = field(repr=False)
    label: str
    capture_protocol: str
    frame_protocol: str
    client_kind: str
    app_version: str
    credential_digest: bytes = field(repr=False)
    created_at: float
    connected_at: float
    last_seen_at: float
    last_seen_monotonic: float = field(repr=False)
    observed_state: str = 'browser_ready'
    connection_id: str | None = field(default=None, repr=False)
    events: asyncio.Queue | None = field(default=None, repr=False)
    revoked: bool = False


@dataclass
class CaptureAssignment:
    id: str
    generation: int
    capture_id: str
    session_id: str
    operation_generation: int
    mode: str
    profile_id: str
    capability_digest: str
    capabilities: dict[str, Any]
    effectors: list[str]
    state: str
    created_at: float
    updated_at: float
    failure_code: str | None = None


def _token_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode('utf-8')).digest()


def _bounded_label(value: object) -> str:
    if not isinstance(value, str):
        raise CaptureError('capture_label_invalid', 400)
    label = value.strip()
    if not label or len(label) > 64 or any(ord(character) < 0x20 for character in label):
        raise CaptureError('capture_label_invalid', 400)
    return label


def _validate_client_metadata(
    capture_protocol: object,
    frame_protocol: object,
    client_kind: object,
    app_version: object,
) -> None:
    if (
        capture_protocol != _CAPTURE_PROTOCOL
        or frame_protocol != _FRAME_PROTOCOL
        or client_kind not in _CLIENT_KINDS
        or not isinstance(app_version, str)
        or not _APP_VERSION_RE.fullmatch(app_version)
    ):
        raise CaptureError('capture_protocol_mismatch', 409)


class CaptureManager:
    """Own ephemeral capture pairing, presence, and per-session assignments.

    Capture credentials never confer a human role or a robot lease.  The
    coordinator remains responsible for checking the original supervisor tab
    and the current robot session before creating or consuming an assignment.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self._lock = asyncio.Lock()
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._pairings: dict[str, _Pairing] = {}
        self._captures: dict[str, _Capture] = {}
        self._assignments: dict[str, CaptureAssignment] = {}
        self._assignment_by_capture: dict[str, str] = {}
        self._assignment_by_session: dict[str, str] = {}
        self._loss_pending_by_session: dict[str, tuple[str, int]] = {}
        self._assignment_sequence = 0

    async def create_pairing(
        self,
        principal_id: str,
        client_id: str,
        *,
        label: str,
    ) -> CapturePairingResult:
        if not principal_id or not client_id:
            raise CaptureError('capture_supervisor_required', 400)
        bounded_label = _bounded_label(label)
        pairing_code = secrets.token_urlsafe(32)
        async with self._lock:
            now_monotonic = self._monotonic()
            now_wall = self._wall_clock()
            self._prune_pairings_locked(now_monotonic)
            supervisor_pairings = sum(
                pairing.principal_id == principal_id
                for pairing in self._pairings.values()
            )
            if (
                supervisor_pairings >= MAX_PAIRINGS_PER_SUPERVISOR
                or len(self._pairings) >= MAX_PAIRING_RECORDS
            ):
                raise CaptureError('capture_pairing_limit', 409)
            pairing_id = str(uuid.uuid4())
            pairing = _Pairing(
                id=pairing_id,
                principal_id=principal_id,
                client_id=client_id,
                label=bounded_label,
                code_digest=_token_digest(pairing_code),
                created_at=now_wall,
                expires_at=now_monotonic + PAIRING_TTL_SECONDS,
            )
            self._pairings[pairing_id] = pairing
            return CapturePairingResult(
                pairing_id=pairing.id,
                pairing_code=pairing_code,
                expires_at=now_wall + PAIRING_TTL_SECONDS,
            )

    async def connect_with_pairing(
        self,
        pairing_id: str,
        pairing_code: str,
        *,
        capture_protocol: str,
        frame_protocol: str,
        client_kind: str,
        app_version: str,
    ) -> CaptureConnection:
        _validate_client_metadata(
            capture_protocol,
            frame_protocol,
            client_kind,
            app_version,
        )
        if not isinstance(pairing_code, str) or not 32 <= len(pairing_code) <= 128:
            raise CaptureError('capture_pairing_invalid', 401)
        async with self._lock:
            now_monotonic = self._monotonic()
            now_wall = self._wall_clock()
            self._prune_pairings_locked(now_monotonic)
            pairing = self._pairings.get(pairing_id)
            if pairing is None or not hmac.compare_digest(
                pairing.code_digest,
                _token_digest(pairing_code),
            ):
                raise CaptureError('capture_pairing_invalid', 401)
            supervisor_captures = sum(
                not capture.revoked
                and capture.principal_id == pairing.principal_id
                for capture in self._captures.values()
            )
            if (
                supervisor_captures >= MAX_CAPTURES_PER_SUPERVISOR
                or len(self._captures) >= MAX_CAPTURE_RECORDS
            ):
                raise CaptureError('capture_device_limit', 409)

            # Consume the code before producing any credential. A cancelled or
            # disconnected redemption therefore cannot be replayed.
            self._pairings.pop(pairing.id, None)
            credential = secrets.token_urlsafe(32)
            capture_id = str(uuid.uuid4())
            connection_id = str(uuid.uuid4())
            events: asyncio.Queue = asyncio.Queue(maxsize=CAPTURE_EVENT_QUEUE_SIZE)
            self._captures[capture_id] = _Capture(
                id=capture_id,
                principal_id=pairing.principal_id,
                label=pairing.label,
                capture_protocol=capture_protocol,
                frame_protocol=frame_protocol,
                client_kind=client_kind,
                app_version=app_version,
                credential_digest=_token_digest(credential),
                created_at=now_wall,
                connected_at=now_wall,
                last_seen_at=now_wall,
                last_seen_monotonic=now_monotonic,
                connection_id=connection_id,
                events=events,
            )
            return CaptureConnection(
                capture_id=capture_id,
                connection_id=connection_id,
                events=events,
                capture_credential=credential,
            )

    async def connect_with_credential(
        self,
        capture_id: str,
        capture_credential: str,
        *,
        capture_protocol: str,
        frame_protocol: str,
        client_kind: str,
        app_version: str,
    ) -> CaptureConnection:
        _validate_client_metadata(
            capture_protocol,
            frame_protocol,
            client_kind,
            app_version,
        )
        if (
            not isinstance(capture_credential, str)
            or not 32 <= len(capture_credential) <= 128
        ):
            raise CaptureError('capture_auth_invalid', 401)
        async with self._lock:
            capture = self._captures.get(capture_id)
            if (
                capture is None
                or capture.revoked
                or not hmac.compare_digest(
                    capture.credential_digest,
                    _token_digest(capture_credential),
                )
            ):
                raise CaptureError('capture_auth_invalid', 401)
            if capture.connection_id is not None:
                raise CaptureError('capture_already_connected', 409)
            if (
                capture.capture_protocol != capture_protocol
                or capture.frame_protocol != frame_protocol
                or capture.client_kind != client_kind
            ):
                raise CaptureError('capture_protocol_mismatch', 409)
            connection_id = str(uuid.uuid4())
            events: asyncio.Queue = asyncio.Queue(maxsize=CAPTURE_EVENT_QUEUE_SIZE)
            now_wall = self._wall_clock()
            capture.connection_id = connection_id
            capture.events = events
            capture.connected_at = now_wall
            capture.last_seen_at = now_wall
            capture.last_seen_monotonic = self._monotonic()
            capture.observed_state = 'browser_ready'
            capture.app_version = app_version
            return CaptureConnection(
                capture_id=capture.id,
                connection_id=connection_id,
                events=events,
            )

    async def disconnect(
        self,
        capture_id: str,
        connection_id: str,
    ) -> CaptureAssignment | None:
        async with self._lock:
            capture = self._captures.get(capture_id)
            if capture is None or capture.connection_id != connection_id:
                return None
            capture.connection_id = None
            capture.events = None
            capture.observed_state = 'offline'
            capture.last_seen_at = self._wall_clock()
            return self._lose_for_capture_locked(
                capture_id,
                'capture_disconnected',
            )

    async def update_presence(
        self,
        capture_id: str,
        connection_id: str,
        *,
        state: str,
        assignment_id: str | None,
    ) -> tuple[dict[str, Any], CaptureAssignment | None]:
        if state not in _CAPTURE_PRESENCE_STATES:
            raise CaptureError('capture_presence_invalid', 400)
        async with self._lock:
            capture = self._require_connection_locked(capture_id, connection_id)
            assignment = self._active_assignment_for_capture_locked(capture_id)
            if state in _ASSIGNMENT_PRESENCE_STATES:
                if assignment is None or assignment.id != assignment_id:
                    raise CaptureError('capture_assignment_mismatch', 409)
            elif assignment_id is not None:
                raise CaptureError('capture_presence_invalid', 400)
            capture.observed_state = state
            capture.last_seen_at = self._wall_clock()
            capture.last_seen_monotonic = self._monotonic()
            lost_assignment = None
            if assignment is not None and state in {
                'browser_ready', 'error', 'xr_ended', 'xr_standby',
            }:
                lost_assignment = self._lose_assignment_locked(
                    assignment,
                    f'capture_{state}',
                )
            return self._public_capture_locked(capture), lost_assignment

    async def list_for_supervisor(
        self,
        principal_id: str,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            captures = [
                capture
                for capture in self._captures.values()
                if not capture.revoked
                and capture.principal_id == principal_id
            ]
            return [
                self._public_capture_locked(capture)
                for capture in sorted(captures, key=lambda item: (item.label, item.id))
            ]

    async def expire_stale_connection(
        self,
        capture_id: str,
        connection_id: str,
    ) -> tuple[bool, CaptureAssignment | None]:
        async with self._lock:
            capture = self._captures.get(capture_id)
            if capture is None or capture.connection_id != connection_id:
                return True, None
            if (
                self._monotonic() - capture.last_seen_monotonic
                < CAPTURE_PRESENCE_TIMEOUT_SECONDS
            ):
                return False, None
            return True, self._expire_capture_connection_locked(capture)

    async def assignment_loss_is_pending(
        self,
        lost_assignment: CaptureAssignment,
    ) -> bool:
        """Return whether this exact passive loss still requires fail-close."""

        async with self._lock:
            return self._loss_pending_by_session.get(
                lost_assignment.session_id,
            ) == (lost_assignment.id, lost_assignment.generation)

    async def complete_assignment_loss(
        self,
        lost_assignment: CaptureAssignment,
    ) -> bool:
        """Clear one exact loss fence after Core proves HOLD or terminal state."""

        async with self._lock:
            expected = (lost_assignment.id, lost_assignment.generation)
            if self._loss_pending_by_session.get(
                lost_assignment.session_id,
            ) != expected:
                return False
            replacement = self._active_assignment_for_session_locked(
                lost_assignment.session_id,
            )
            if replacement is not None:
                self._revoke_assignment_locked(
                    replacement,
                    'capture_loss_fail_closed',
                )
            self._loss_pending_by_session.pop(lost_assignment.session_id, None)
            return True

    async def revoke_capture(
        self,
        capture_id: str,
        principal_id: str,
    ) -> dict[str, Any]:
        async with self._lock:
            capture = self._require_supervisor_capture_locked(
                capture_id,
                principal_id,
            )
            if self._active_assignment_for_capture_locked(capture.id) is not None:
                # Deleting enrollment does not terminate an already negotiated
                # capture-to-Driver peer connection. Require the operator to
                # enter Pause/HOLD first so the Driver safety path is explicit.
                raise CaptureError('capture_attached', 409)
            if any(
                assignment.capture_id == capture.id
                and self._loss_pending_by_session.get(assignment.session_id)
                == (assignment.id, assignment.generation)
                for assignment in self._assignments.values()
            ):
                raise CaptureError('capture_loss_pending', 409)
            public = self._public_capture_locked(capture)
            capture.revoked = True
            self._publish_locked(capture, {
                'type': 'capture_revoked',
                'reason': 'operator_revoked',
            })
            self._captures.pop(capture.id, None)
            self._assignment_by_capture.pop(capture.id, None)
            for assignment_id, assignment in list(self._assignments.items()):
                if assignment.capture_id != capture.id:
                    continue
                if self._assignment_by_session.get(assignment.session_id) == assignment_id:
                    self._assignment_by_session.pop(assignment.session_id, None)
                self._assignments.pop(assignment_id, None)
            return {**public, 'revoked': True, 'assignment': None}

    async def attach(
        self,
        *,
        capture_id: str,
        principal_id: str,
        session_id: str,
        operation_generation: int,
        mode: str,
        profile_id: str,
        capability_digest: str,
        capabilities: dict[str, Any],
        effectors: list[str],
    ) -> CaptureAssignment:
        async with self._lock:
            capture = self._require_supervisor_capture_locked(
                capture_id,
                principal_id,
            )
            if session_id in self._loss_pending_by_session:
                raise CaptureError('capture_loss_pending', 409)
            if capture.connection_id is None or capture.observed_state != 'xr_standby':
                raise CaptureError('capture_not_ready', 409)
            if capture.frame_protocol != 'motus.teleop.rtc-frame.v1':
                raise CaptureError('capture_protocol_mismatch', 409)

            session_assignment = self._active_assignment_for_session_locked(session_id)
            capture_assignment = self._active_assignment_for_capture_locked(capture_id)
            existing = session_assignment or capture_assignment
            if existing is not None:
                same_binding = (
                    existing.capture_id == capture_id
                    and existing.session_id == session_id
                    and existing.operation_generation == operation_generation
                    and existing.mode == mode
                    and existing.profile_id == profile_id
                    and existing.capability_digest == capability_digest
                )
                if same_binding:
                    return copy.deepcopy(existing)
                raise CaptureError('capture_assignment_conflict', 409)

            self._assignment_sequence += 1
            now = self._wall_clock()
            assignment = CaptureAssignment(
                id=str(uuid.uuid4()),
                generation=self._assignment_sequence,
                capture_id=capture.id,
                session_id=session_id,
                operation_generation=operation_generation,
                mode=mode,
                profile_id=profile_id,
                capability_digest=capability_digest,
                capabilities=copy.deepcopy(capabilities),
                effectors=copy.deepcopy(effectors),
                state='issued',
                created_at=now,
                updated_at=now,
            )
            self._assignments[assignment.id] = assignment
            self._assignment_by_capture[capture.id] = assignment.id
            self._assignment_by_session[session_id] = assignment.id
            self._publish_locked(capture, self._assignment_event_locked(assignment))
            return copy.deepcopy(assignment)

    async def claim_offer(
        self,
        capture_id: str,
        connection_id: str,
        assignment_id: str,
    ) -> CaptureAssignment:
        async with self._lock:
            self._require_connection_locked(capture_id, connection_id)
            assignment = self._active_assignment_for_capture_locked(capture_id)
            if assignment is None or assignment.id != assignment_id:
                raise CaptureError('capture_assignment_mismatch', 409)
            if assignment.state != 'issued':
                raise CaptureError('capture_offer_already_consumed', 409)
            assignment.state = 'offer_consumed'
            assignment.updated_at = self._wall_clock()
            return copy.deepcopy(assignment)

    async def complete_offer(
        self,
        capture_id: str,
        connection_id: str,
        assignment_id: str,
    ) -> bool:
        async with self._lock:
            try:
                self._require_connection_locked(capture_id, connection_id)
            except CaptureError:
                return False
            assignment = self._active_assignment_for_capture_locked(capture_id)
            if (
                assignment is None
                or assignment.id != assignment_id
                or assignment.state != 'offer_consumed'
            ):
                return False
            assignment.state = 'negotiated'
            assignment.updated_at = self._wall_clock()
            return True

    async def fail_offer(
        self,
        capture_id: str,
        assignment_id: str,
    ) -> CaptureAssignment | None:
        async with self._lock:
            assignment = self._active_assignment_for_capture_locked(capture_id)
            if assignment is None or assignment.id != assignment_id:
                return None
            if assignment.state not in {'issued', 'offer_consumed'}:
                return None
            return self._lose_assignment_locked(
                assignment,
                'capture_signaling_failed',
            )

    async def assignment_is_current(
        self,
        capture_id: str,
        connection_id: str,
        assignment_id: str,
        *,
        session_id: str,
        operation_generation: int,
    ) -> bool:
        async with self._lock:
            try:
                self._require_connection_locked(capture_id, connection_id)
            except CaptureError:
                return False
            assignment = self._active_assignment_for_capture_locked(capture_id)
            return bool(
                assignment is not None
                and assignment.id == assignment_id
                and assignment.session_id == session_id
                and assignment.operation_generation == operation_generation
                and assignment.state == 'offer_consumed'
            )

    async def revoke_for_session(self, session_id: str, reason: str) -> None:
        async with self._lock:
            assignment = self._active_assignment_for_session_locked(session_id)
            if assignment is not None:
                self._revoke_assignment_locked(assignment, reason)

    async def reset(self, reason: str = 'core_stopped') -> None:
        async with self._lock:
            for assignment in list(self._assignments.values()):
                if assignment.state in _ACTIVE_ASSIGNMENT_STATES:
                    self._revoke_assignment_locked(assignment, reason)
            for capture in self._captures.values():
                self._publish_locked(capture, {'type': 'capture_revoked', 'reason': reason})
                capture.revoked = True
                capture.connection_id = None
                capture.events = None
            self._pairings.clear()
            self._captures.clear()
            self._assignments.clear()
            self._assignment_by_capture.clear()
            self._assignment_by_session.clear()
            self._loss_pending_by_session.clear()

    def public_assignment(self, assignment: CaptureAssignment) -> dict[str, Any]:
        return {
            'id': assignment.id,
            'generation': assignment.generation,
            'session_id': assignment.session_id,
            'mode': assignment.mode,
            'profile_id': assignment.profile_id,
            'capability_digest': assignment.capability_digest,
            'capabilities': copy.deepcopy(assignment.capabilities),
            'effectors': copy.deepcopy(assignment.effectors),
            'state': assignment.state,
            'created_at': assignment.created_at,
            'updated_at': assignment.updated_at,
            'failure_code': assignment.failure_code,
        }

    def _prune_pairings_locked(self, now_monotonic: float) -> None:
        expired = [
            pairing_id
            for pairing_id, pairing in self._pairings.items()
            if pairing.expires_at <= now_monotonic
        ]
        for pairing_id in expired:
            self._pairings.pop(pairing_id, None)

    def _expire_capture_connection_locked(
        self,
        capture: _Capture,
    ) -> CaptureAssignment | None:
        lost_assignment = self._lose_for_capture_locked(
            capture.id,
            'capture_presence_timeout',
        )
        self._publish_locked(capture, {
            'type': 'capture_stale',
            'reason': 'capture_presence_timeout',
        })
        capture.connection_id = None
        capture.events = None
        capture.observed_state = 'offline'
        return lost_assignment

    def _require_supervisor_capture_locked(
        self,
        capture_id: str,
        principal_id: str,
    ) -> _Capture:
        capture = self._captures.get(capture_id)
        if capture is None or capture.revoked:
            raise CaptureError('capture_not_found', 404)
        if capture.principal_id != principal_id:
            raise CaptureError('capture_forbidden', 403)
        return capture

    def _require_connection_locked(
        self,
        capture_id: str,
        connection_id: str,
    ) -> _Capture:
        capture = self._captures.get(capture_id)
        if (
            capture is None
            or capture.revoked
            or capture.connection_id != connection_id
        ):
            raise CaptureError('capture_auth_invalid', 401)
        return capture

    def _active_assignment_for_capture_locked(
        self,
        capture_id: str,
    ) -> CaptureAssignment | None:
        assignment_id = self._assignment_by_capture.get(capture_id)
        assignment = self._assignments.get(assignment_id) if assignment_id else None
        return (
            assignment
            if assignment is not None and assignment.state in _ACTIVE_ASSIGNMENT_STATES
            else None
        )

    def _active_assignment_for_session_locked(
        self,
        session_id: str,
    ) -> CaptureAssignment | None:
        assignment_id = self._assignment_by_session.get(session_id)
        assignment = self._assignments.get(assignment_id) if assignment_id else None
        return (
            assignment
            if assignment is not None and assignment.state in _ACTIVE_ASSIGNMENT_STATES
            else None
        )

    def _public_capture_locked(self, capture: _Capture) -> dict[str, Any]:
        assignment = self._active_assignment_for_capture_locked(capture.id)
        return {
            'id': capture.id,
            'label': capture.label,
            'capture_protocol': capture.capture_protocol,
            'frame_protocol': capture.frame_protocol,
            'client_kind': capture.client_kind,
            'app_version': capture.app_version,
            'connected': capture.connection_id is not None,
            'observed_state': capture.observed_state,
            'created_at': capture.created_at,
            'connected_at': capture.connected_at,
            'last_seen_at': capture.last_seen_at,
            'assignment': self.public_assignment(assignment) if assignment else None,
        }

    def _assignment_event_locked(
        self,
        assignment: CaptureAssignment,
    ) -> dict[str, Any]:
        return {
            'type': 'assignment',
            'assignment': self.public_assignment(assignment),
        }

    def _revoke_for_capture_locked(
        self,
        capture_id: str,
        reason: str,
    ) -> CaptureAssignment | None:
        assignment = self._active_assignment_for_capture_locked(capture_id)
        if assignment is not None:
            return self._revoke_assignment_locked(assignment, reason)
        return None

    def _lose_for_capture_locked(
        self,
        capture_id: str,
        reason: str,
    ) -> CaptureAssignment | None:
        assignment = self._active_assignment_for_capture_locked(capture_id)
        if assignment is None:
            return None
        return self._lose_assignment_locked(assignment, reason)

    def _lose_assignment_locked(
        self,
        assignment: CaptureAssignment,
        reason: str,
    ) -> CaptureAssignment | None:
        lost_assignment = self._revoke_assignment_locked(assignment, reason)
        if lost_assignment is not None:
            self._loss_pending_by_session[lost_assignment.session_id] = (
                lost_assignment.id,
                lost_assignment.generation,
            )
        return lost_assignment

    def _revoke_assignment_locked(
        self,
        assignment: CaptureAssignment,
        reason: str,
    ) -> CaptureAssignment | None:
        if assignment.state not in _ACTIVE_ASSIGNMENT_STATES:
            return None
        assignment.state = 'revoked'
        assignment.failure_code = reason
        assignment.updated_at = self._wall_clock()
        if self._assignment_by_capture.get(assignment.capture_id) == assignment.id:
            self._assignment_by_capture.pop(assignment.capture_id, None)
        if self._assignment_by_session.get(assignment.session_id) == assignment.id:
            self._assignment_by_session.pop(assignment.session_id, None)
        capture = self._captures.get(assignment.capture_id)
        if capture is not None:
            self._publish_locked(capture, {
                'type': 'assignment_revoked',
                'assignment_id': assignment.id,
                'reason': reason,
            })
        return copy.deepcopy(assignment)

    @staticmethod
    def _publish_locked(capture: _Capture, event: dict[str, Any]) -> None:
        queue = capture.events
        if queue is None:
            return
        try:
            queue.put_nowait(copy.deepcopy(event))
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(copy.deepcopy(event))
            except asyncio.QueueFull:
                pass


__all__ = [
    'CaptureAssignment',
    'CaptureConnection',
    'CaptureError',
    'CaptureManager',
    'CapturePairingResult',
]
