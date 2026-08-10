from __future__ import annotations

import asyncio
import copy
import math
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable

import config
from teleop.models import ShadowSession

MIN_LEASE_SECONDS = 15.0
MAX_LEASE_SECONDS = 120.0
DEFAULT_LEASE_SECONDS = 15.0
MAX_TERMINAL_SESSIONS = 256
MAX_DRIVER_EPOCH = (1 << 63) - 2
_TERMINAL_STATES = frozenset({'released', 'expired', 'faulted'})
_PROFILE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
_epoch_process_lock = threading.Lock()


class SessionConflict(Exception):
    def __init__(self, session: ShadowSession):
        super().__init__(f'robot {session.robot_id} is controlled by {session.principal_id}')
        self.session = session


class SessionNotFound(Exception):
    pass


class SessionForbidden(Exception):
    pass


class SessionClientMismatch(SessionForbidden):
    pass


class SessionStateConflict(Exception):
    pass


class StaleSessionOperation(SessionStateConflict):
    pass


class EpochExhausted(SessionStateConflict):
    pass


def _ensure_epoch_table(conn) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS teleop_driver_epochs (
            driver_id TEXT PRIMARY KEY,
            epoch INTEGER NOT NULL CHECK (epoch >= 1)
        )
    ''')
    conn.commit()


def _allocate_driver_epoch(driver_id: str, minimum_epoch: int = 1) -> int:
    """Atomically allocate an epoch that survives manager/Core restarts.

    ``minimum_epoch`` is an inclusive floor supplied from the Driver status. It
    lets Core recover safely if its database was restored or the Driver was
    prepared by another Core instance.
    """

    if not driver_id:
        raise ValueError('driver_id is required')
    if isinstance(minimum_epoch, bool):
        raise TypeError('minimum_epoch must be a positive integer')
    try:
        minimum = int(minimum_epoch)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('minimum_epoch must be a positive integer') from exc
    if minimum < 1 or minimum != minimum_epoch:
        raise ValueError('minimum_epoch must be a positive integer')
    if minimum > MAX_DRIVER_EPOCH:
        raise EpochExhausted(driver_id)

    # SQLite is the cross-process authority.  The process lock also prevents
    # same-process threads from racing in config._get_conn's schema bootstrap.
    with _epoch_process_lock, config._get_conn() as conn:
        _ensure_epoch_table(conn)
        try:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute(
                'SELECT epoch FROM teleop_driver_epochs WHERE driver_id=?',
                (driver_id,),
            ).fetchone()
            stored = int(row[0]) if row else 0
            if stored >= MAX_DRIVER_EPOCH:
                raise EpochExhausted(driver_id)
            stored_next = stored + 1
            epoch = max(stored_next, minimum)
            if epoch > MAX_DRIVER_EPOCH:
                raise EpochExhausted(driver_id)
            if row:
                conn.execute(
                    'UPDATE teleop_driver_epochs SET epoch=? WHERE driver_id=?',
                    (epoch, driver_id),
                )
            else:
                conn.execute(
                    'INSERT INTO teleop_driver_epochs (driver_id, epoch) VALUES (?, ?)',
                    (driver_id, epoch),
                )
            conn.commit()
            return epoch
        except Exception:
            conn.rollback()
            raise


class ShadowSessionManager:
    """Own one expiring Shadow session per robot.

    The in-memory map is authoritative only for the current Core process. Driver
    epochs are persisted separately so a restarted Core can never issue an older
    fence epoch to the same driver.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self._lock = asyncio.Lock()
        self._sessions: dict[str, ShadowSession] = {}
        self._robot_sessions: dict[str, str] = {}
        self._pending_expired: dict[str, ShadowSession] = {}
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._operation_sequence = 0

    @staticmethod
    def _bounded_lease(value: object, fallback: float = DEFAULT_LEASE_SECONDS) -> float:
        try:
            lease = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(lease):
            return fallback
        return min(MAX_LEASE_SECONDS, max(MIN_LEASE_SECONDS, lease))

    @classmethod
    def default_lease_seconds(cls) -> float:
        return cls._bounded_lease(
            os.environ.get('MOTUS_TELEOP_LEASE_SECONDS', str(DEFAULT_LEASE_SECONDS))
        )

    def _next_operation_generation_locked(self) -> int:
        self._operation_sequence += 1
        return self._operation_sequence

    def _is_current_locked(self, session: ShadowSession) -> bool:
        return (
            self._sessions.get(session.id) is session
            and self._robot_sessions.get(session.robot_id) == session.id
        )

    def _unbind_robot_locked(self, session: ShadowSession) -> bool:
        """Remove ownership only if this exact session still owns the mapping."""

        if self._robot_sessions.get(session.robot_id) != session.id:
            return False
        self._robot_sessions.pop(session.robot_id, None)
        return True

    def _expire_due_locked(self, now_monotonic: float) -> list[ShadowSession]:
        expired: list[ShadowSession] = []
        # Authority work is O(live robots), never O(unbounded history).
        live_ids = list(dict.fromkeys(self._robot_sessions.values()))
        for session_id in live_ids:
            session = self._sessions.get(session_id)
            if session is None:
                continue
            if session.state in _TERMINAL_STATES:
                continue
            if session.deadline_monotonic > now_monotonic:
                continue
            session.state = 'expired'
            session.operation_state = 'expired'
            self._unbind_robot_locked(session)
            self._pending_expired.setdefault(session.id, session)
            expired.append(session)
        return expired

    def _prune_terminal_locked(self) -> None:
        terminal_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if session.state in _TERMINAL_STATES
            and session_id not in self._pending_expired
        ]
        excess = len(terminal_ids) - MAX_TERMINAL_SESSIONS
        for session_id in terminal_ids[:max(0, excess)]:
            self._sessions.pop(session_id, None)

    async def reserve(
        self,
        robot_id: str,
        principal_id: str,
        *,
        driver_id: str,
        boot_id: str,
        capability_digest: str,
        client_id: str,
        mode: str = 'shadow',
        profile_id: str = 'recording',
        capabilities: dict | None = None,
        effectors: list[str] | None = None,
        signaling_audience: str = 'teleop-shadow-rtc',
        defer_identity: bool = False,
        dry_run_profile: str | None = None,
        lease_seconds: float | None = None,
        minimum_epoch: int = 1,
    ) -> ShadowSession:
        if not robot_id or not principal_id or not capability_digest or not client_id:
            raise ValueError(
                'robot_id, principal_id, capability_digest and client_id are required'
            )
        if dry_run_profile is not None:
            if dry_run_profile != 'recording' or profile_id != 'recording':
                raise ValueError('dry_run_profile only supports legacy recording')
        if not isinstance(profile_id, str) or not _PROFILE_ID_RE.fullmatch(profile_id):
            raise ValueError('profile_id is not supported')
        if mode not in {'shadow', 'live'}:
            raise ValueError('mode is not supported')
        if defer_identity is not (mode == 'live'):
            raise ValueError('only live reservations defer Driver identity')
        if not defer_identity and not boot_id:
            raise ValueError('boot_id is required')
        expected_audiences = {
            'shadow': {'teleop-shadow-rtc', 'motus-teleop-rtc'},
            'live': {'motus-teleop-rtc'},
        }
        if signaling_audience not in expected_audiences[mode]:
            raise ValueError('signaling_audience is not supported')
        if not isinstance(capabilities or {}, dict):
            raise ValueError('capabilities must be an object')
        if not isinstance(effectors or [], list):
            raise ValueError('effectors must be a list')
        if capabilities:
            if capabilities.get('profile_id') != profile_id:
                raise ValueError('capabilities.profile_id does not match profile_id')
            capability_effectors = capabilities.get('effectors')
            if not isinstance(capability_effectors, list):
                raise ValueError('capabilities.effectors must be a list')
            if capability_effectors != (effectors or []):
                raise ValueError('effectors do not match capabilities.effectors')
        lease = (
            self.default_lease_seconds()
            if lease_seconds is None
            else self._bounded_lease(lease_seconds)
        )
        # Epoch persistence is SQLite-backed and may wait on a writer.  Allocate
        # outside both the event loop and the manager lock: wasting an epoch on
        # a concurrent reservation conflict is safe, while delaying existing
        # session heartbeats behind database I/O is not.
        epoch = 0
        if not defer_identity:
            epoch = await asyncio.to_thread(
                _allocate_driver_epoch,
                driver_id,
                minimum_epoch,
            )
        async with self._lock:
            now_monotonic = self._monotonic()
            now_wall = self._wall_clock()
            self._expire_due_locked(now_monotonic)
            existing_id = self._robot_sessions.get(robot_id)
            if existing_id:
                existing = self._sessions.get(existing_id)
                if existing is not None and existing.state not in _TERMINAL_STATES:
                    raise SessionConflict(existing)
                # Repair an impossible/stale map entry conditionally.
                if self._robot_sessions.get(robot_id) == existing_id:
                    self._robot_sessions.pop(robot_id, None)

            session = ShadowSession(
                id=str(uuid.uuid4()),
                robot_id=robot_id,
                driver_id=driver_id,
                principal_id=principal_id,
                boot_id=boot_id,
                epoch=epoch,
                capability_digest=capability_digest,
                mode=mode,
                profile_id=profile_id,
                capabilities=copy.deepcopy(capabilities or {}),
                effectors=copy.deepcopy(effectors or []),
                signaling_audience=signaling_audience,
                live_confirmed=mode == 'shadow',
                client_id=client_id,
                fence=secrets.token_urlsafe(32),
                state='awaiting_confirmation' if defer_identity else 'preparing',
                operation_generation=self._next_operation_generation_locked(),
                operation_state='pending',
                created_at=now_wall,
                lease_seconds=lease,
                deadline_monotonic=now_monotonic + lease,
            )
            self._sessions[session.id] = session
            self._robot_sessions[robot_id] = session.id
            return session

    async def confirm_live_identity(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        *,
        boot_id: str,
        minimum_epoch: int,
    ) -> ShadowSession:
        """Bind Driver identity only after the owning tab explicitly confirms live."""

        if not boot_id:
            raise ValueError('boot_id is required')
        candidate = await self.get_authorized(
            session_id,
            principal_id,
            owner=False,
            client_id=client_id,
            require_client=True,
        )
        if (
            candidate.mode != 'live'
            or candidate.state != 'awaiting_confirmation'
            or candidate.live_confirmed
        ):
            raise SessionStateConflict(
                f'session {session_id} cannot confirm live from {candidate.state}'
            )
        epoch = await asyncio.to_thread(
            _allocate_driver_epoch,
            candidate.driver_id,
            minimum_epoch,
        )
        async with self._lock:
            now = self._monotonic()
            self._expire_due_locked(now)
            session = self._require_current_locked(session_id)
            self._authorize_locked(session, principal_id, False)
            self._authorize_client_locked(session, client_id, False)
            if session.mode != 'live' or session.state != 'awaiting_confirmation':
                raise SessionStateConflict(
                    f'session {session_id} cannot confirm live from {session.state}'
                )
            session.boot_id = boot_id
            session.epoch = epoch
            session.live_confirmed = True
            session.state = 'preparing'
            session.operation_state = 'pending'
            session.deadline_monotonic = now + session.lease_seconds
            return session

    async def activate(
        self,
        session_id: str,
        operation_generation: int,
    ) -> ShadowSession:
        async with self._lock:
            now = self._monotonic()
            self._expire_due_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(session_id)
            if session.operation_generation != operation_generation:
                raise StaleSessionOperation(session_id)
            if session.state != 'preparing' or not self._is_current_locked(session):
                raise SessionStateConflict(
                    f'session {session_id} cannot activate from {session.state}'
                )
            session.state = 'active'
            session.operation_state = 'succeeded'
            return session

    async def heartbeat(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        owner: bool = False,
    ) -> ShadowSession:
        async with self._lock:
            now = self._monotonic()
            self._expire_due_locked(now)
            session = self._require_current_locked(session_id)
            self._authorize_locked(session, principal_id, owner)
            self._authorize_client_locked(session, client_id, owner)
            if session.state not in ('active', 'paused', 'hold'):
                raise SessionStateConflict(
                    f'session {session_id} cannot heartbeat from {session.state}'
                )
            session.deadline_monotonic = now + session.lease_seconds
            return session

    async def pause(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        owner: bool = False,
    ) -> ShadowSession:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session = self._require_current_locked(session_id)
            self._authorize_locked(session, principal_id, owner)
            self._authorize_client_locked(session, client_id, owner)
            if session.state not in ('active', 'paused', 'hold'):
                raise SessionStateConflict(
                    f'session {session_id} cannot pause from {session.state}'
                )
            session.state = 'paused'
            return session

    async def soft_stop(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        owner: bool = False,
    ) -> ShadowSession:
        """Enter the visible HOLD state; repeated HOLD requests are idempotent."""

        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session = self._require_current_locked(session_id)
            self._authorize_locked(session, principal_id, owner)
            self._authorize_client_locked(session, client_id, owner)
            if session.state not in ('active', 'hold'):
                raise SessionStateConflict(
                    f'session {session_id} cannot soft-stop from {session.state}'
                )
            session.state = 'hold'
            return session

    async def release(
        self,
        session_id: str,
        principal_id: str,
        client_id: str,
        owner: bool = False,
    ) -> ShadowSession:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session = self._require_current_locked(session_id)
            self._authorize_locked(session, principal_id, owner)
            self._authorize_client_locked(session, client_id, owner)
            session.state = 'released'
            session.operation_state = 'cancelled'
            self._unbind_robot_locked(session)
            self._prune_terminal_locked()
            return session

    async def fail_reservation(
        self,
        session_id: str,
        operation_generation: int,
    ) -> ShadowSession | None:
        """Fail only the still-current prepare operation.

        A delayed Driver error must not fault an activated session or unbind a
        replacement session for the same robot.
        """

        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if (
                session.operation_generation != operation_generation
                or session.state != 'preparing'
                or not self._is_current_locked(session)
            ):
                return None
            session.state = 'faulted'
            session.operation_state = 'failed'
            self._unbind_robot_locked(session)
            self._prune_terminal_locked()
            return session

    async def fault(
        self,
        session_id: str,
        operation_generation: int | None = None,
    ) -> ShadowSession | None:
        """Conditionally fault and release any current, non-terminal session.

        The heartbeat worker uses this after a Driver failure. Supplying an
        operation generation additionally protects against a delayed worker from
        an obsolete reservation.
        """

        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session = self._sessions.get(session_id)
            if session is None or not self._is_current_locked(session):
                return None
            if session.state in _TERMINAL_STATES:
                return None
            if (
                operation_generation is not None
                and session.operation_generation != operation_generation
            ):
                return None
            session.state = 'faulted'
            session.operation_state = 'failed'
            self._unbind_robot_locked(session)
            self._prune_terminal_locked()
            return session

    async def get(self, session_id: str) -> ShadowSession | None:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            return self._sessions.get(session_id)

    async def get_current(self, session_id: str) -> ShadowSession | None:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session = self._sessions.get(session_id)
            if session is None or not self._is_current_locked(session):
                return None
            if session.state in _TERMINAL_STATES:
                return None
            return session

    async def authorize(
        self,
        session_id: str,
        principal_id: str,
        owner: bool = False,
    ) -> ShadowSession:
        return await self.get_authorized(session_id, principal_id, owner=owner)

    async def get_authorized(
        self,
        session_id: str,
        principal_id: str,
        owner: bool = False,
        include_terminal: bool = False,
        *,
        client_id: str = '',
        require_client: bool = False,
    ) -> ShadowSession:
        """Read an authorized session without renewing or changing its lease."""

        async with self._lock:
            self._expire_due_locked(self._monotonic())
            if include_terminal:
                session = self._sessions.get(session_id)
                if session is None:
                    raise SessionNotFound(session_id)
                if session.state not in _TERMINAL_STATES and not self._is_current_locked(session):
                    raise SessionNotFound(session_id)
            else:
                session = self._require_current_locked(session_id)
            self._authorize_locked(session, principal_id, owner)
            if require_client:
                self._authorize_client_locked(session, client_id, owner)
            return session

    async def active_for_robot(self, robot_id: str) -> ShadowSession | None:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            session_id = self._robot_sessions.get(robot_id)
            session = self._sessions.get(session_id) if session_id else None
            if session is None or session.state in _TERMINAL_STATES:
                if self._robot_sessions.get(robot_id) == session_id:
                    self._robot_sessions.pop(robot_id, None)
                return None
            return session

    async def list_visible(
        self,
        principal_id: str,
        owner: bool = False,
    ) -> list[ShadowSession]:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            sessions = list(self._sessions.values())
            if owner:
                return sessions
            return [s for s in sessions if s.principal_id == principal_id]

    async def expire_due(self) -> list[ShadowSession]:
        async with self._lock:
            self._expire_due_locked(self._monotonic())
            expired = list(self._pending_expired.values())
            self._pending_expired.clear()
            self._prune_terminal_locked()
            return expired

    async def retained_session_ids(self) -> set[str]:
        async with self._lock:
            self._prune_terminal_locked()
            return set(self._sessions)

    async def reset(self) -> None:
        """Clear ephemeral ownership; persisted driver epochs are untouched."""

        async with self._lock:
            self._sessions.clear()
            self._robot_sessions.clear()
            self._pending_expired.clear()
            self._operation_sequence = 0

    def public_dict(self, session: ShadowSession) -> dict:
        return session.public_dict(
            monotonic_now=self._monotonic(),
            wall_now=self._wall_clock(),
        )

    def _require_current_locked(self, session_id: str) -> ShadowSession:
        session = self._sessions.get(session_id)
        if (
            session is None
            or session.state in _TERMINAL_STATES
            or not self._is_current_locked(session)
        ):
            raise SessionNotFound(session_id)
        return session

    @staticmethod
    def _authorize_locked(
        session: ShadowSession,
        principal_id: str,
        owner: bool,
    ) -> None:
        if not owner and session.principal_id != principal_id:
            raise SessionForbidden(session.id)

    @staticmethod
    def _authorize_client_locked(
        session: ShadowSession,
        client_id: str,
        owner: bool,
    ) -> None:
        if not owner and session.client_id != client_id:
            raise SessionClientMismatch(session.id)


manager = ShadowSessionManager()
