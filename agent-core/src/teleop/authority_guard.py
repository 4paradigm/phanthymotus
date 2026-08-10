from __future__ import annotations

import asyncio
import math
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass

import config

MAX_SQLITE_INTEGER = (1 << 63) - 1
MAX_DRIVER_EPOCH = MAX_SQLITE_INTEGER - 1
AUTHORITY_GUARD_PHASES = frozenset(
    {
        "preparing",
        "active",
        "stopping",
        "recovery_required",
        "reconciling",
    }
)

_AUTHORITY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TABLE_NAME = "teleop_authority_guards"
_COLUMNS = (
    "robot_id",
    "driver_id",
    "session_id",
    "boot_id",
    "epoch",
    "capability_digest",
    "target_fingerprint",
    "dispatch_generation",
    "phase",
    "created_at",
    "updated_at",
)

# One Core process serializes safety-relevant target writes with the transition
# that makes a persistent guard visible. SQLite remains the durable boundary;
# this lock closes the in-process check/write race around config.main.
target_mutation_lock = asyncio.Lock()


class AuthorityGuardError(RuntimeError):
    """Base error for a persistent authority guard operation."""


class AuthorityGuardSchemaError(AuthorityGuardError):
    """The guard table does not have the fail-closed schema we require."""


class AuthorityGuardStateConflict(AuthorityGuardError):
    """A requested state transition would move persisted safety state backwards."""


class AuthorityGuardNotFound(AuthorityGuardStateConflict):
    """The exact robot/session pair does not own a persistent guard."""


def _authority_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _AUTHORITY_ID_RE.fullmatch(value):
        raise ValueError(f"{field} must be a 1-64 character authority identifier")
    return value


def _canonical_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError):
        raise ValueError(f"{field} must be a canonical UUID") from None
    if parsed.int == 0 or str(parsed) != value:
        raise ValueError(f"{field} must be a canonical non-nil UUID")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _timestamp(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative timestamp")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative timestamp")
    return result


def _phase(value: object) -> str:
    if not isinstance(value, str) or value not in AUTHORITY_GUARD_PHASES:
        raise ValueError("phase is not an allowed authority guard phase")
    return value


@dataclass(frozen=True, slots=True)
class AuthorityGuard:
    """Secret-free evidence that a robot must remain authority-quarantined.

    A guard deliberately contains no fence, bearer token, browser/client identity,
    or human principal.  Such credentials remain process-local and cannot be
    recovered after a Core restart.
    """

    robot_id: str
    driver_id: str
    session_id: str
    boot_id: str
    epoch: int
    capability_digest: str
    target_fingerprint: str
    dispatch_generation: int
    phase: str
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        _authority_id(self.robot_id, "robot_id")
        _authority_id(self.driver_id, "driver_id")
        _canonical_uuid(self.session_id, "session_id")
        _canonical_uuid(self.boot_id, "boot_id")
        _integer(self.epoch, "epoch", minimum=1, maximum=MAX_DRIVER_EPOCH)
        _digest(self.capability_digest, "capability_digest")
        _digest(self.target_fingerprint, "target_fingerprint")
        _integer(
            self.dispatch_generation,
            "dispatch_generation",
            minimum=0,
            maximum=MAX_SQLITE_INTEGER,
        )
        _phase(self.phase)
        created_at = _timestamp(self.created_at, "created_at")
        updated_at = _timestamp(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at cannot precede created_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)


def _ensure_table(conn: sqlite3.Connection) -> None:
    phases = ", ".join(repr(item) for item in sorted(AUTHORITY_GUARD_PHASES))
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
            robot_id TEXT NOT NULL PRIMARY KEY,
            driver_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL UNIQUE,
            boot_id TEXT NOT NULL,
            epoch INTEGER NOT NULL CHECK (
                typeof(epoch) = 'integer' AND epoch BETWEEN 1 AND {MAX_DRIVER_EPOCH}
            ),
            capability_digest TEXT NOT NULL,
            target_fingerprint TEXT NOT NULL,
            dispatch_generation INTEGER NOT NULL CHECK (
                typeof(dispatch_generation) = 'integer'
                AND dispatch_generation BETWEEN 0 AND {MAX_SQLITE_INTEGER}
            ),
            phase TEXT NOT NULL CHECK (phase IN ({phases})),
            created_at REAL NOT NULL CHECK (created_at >= 0),
            updated_at REAL NOT NULL CHECK (
                updated_at >= created_at AND updated_at >= 0
            )
        )
    """)
    _verify_schema(conn)


def _verify_schema(conn: sqlite3.Connection) -> None:
    rows = conn.execute(f"PRAGMA table_info({_TABLE_NAME})").fetchall()
    expected = (
        ("robot_id", "TEXT", 1, 1),
        ("driver_id", "TEXT", 1, 0),
        ("session_id", "TEXT", 1, 0),
        ("boot_id", "TEXT", 1, 0),
        ("epoch", "INTEGER", 1, 0),
        ("capability_digest", "TEXT", 1, 0),
        ("target_fingerprint", "TEXT", 1, 0),
        ("dispatch_generation", "INTEGER", 1, 0),
        ("phase", "TEXT", 1, 0),
        ("created_at", "REAL", 1, 0),
        ("updated_at", "REAL", 1, 0),
    )
    actual = tuple((row[1], row[2].upper(), row[3], row[5]) for row in rows)
    if actual != expected:
        raise AuthorityGuardSchemaError(
            "authority guard table columns are incompatible"
        )

    unique_columns: set[tuple[str, ...]] = set()
    for index in conn.execute(f"PRAGMA index_list({_TABLE_NAME})").fetchall():
        if not index[2]:
            continue
        columns = tuple(
            row[2]
            for row in conn.execute(f'PRAGMA index_info("{index[1]}")').fetchall()
        )
        unique_columns.add(columns)
    required_unique = {("robot_id",), ("driver_id",), ("session_id",)}
    if unique_columns != required_unique:
        raise AuthorityGuardSchemaError("authority guard uniqueness is incompatible")


def _row_to_guard(row: tuple[object, ...]) -> AuthorityGuard:
    if len(row) != len(_COLUMNS):
        raise AuthorityGuardSchemaError("authority guard row has incompatible shape")
    return AuthorityGuard(**dict(zip(_COLUMNS, row, strict=True)))


def _select_guard(
    conn: sqlite3.Connection,
    robot_id: str,
) -> AuthorityGuard | None:
    row = conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE_NAME} WHERE robot_id=?",
        (robot_id,),
    ).fetchone()
    return None if row is None else _row_to_guard(row)


def _begin(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    _ensure_table(conn)


def create_guard(guard: AuthorityGuard) -> AuthorityGuard:
    """Insert a new guard without ever replacing an existing authority claim."""

    if not isinstance(guard, AuthorityGuard):
        raise TypeError("guard must be an AuthorityGuard")
    # Reconstruct to prevent a caller from bypassing frozen dataclass validation
    # through object.__setattr__ before crossing the storage boundary.
    validated = AuthorityGuard(**{field: getattr(guard, field) for field in _COLUMNS})
    with config._get_conn() as conn:
        try:
            _begin(conn)
            conn.execute(
                f"""INSERT INTO {_TABLE_NAME} ({", ".join(_COLUMNS)})
                    VALUES ({", ".join("?" for _ in _COLUMNS)})""",
                tuple(getattr(validated, field) for field in _COLUMNS),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return validated


def get_guard(robot_id: str) -> AuthorityGuard | None:
    """Return one validated guard; corrupted storage raises instead of unlocking."""

    validated_robot_id = _authority_id(robot_id, "robot_id")
    with config._get_conn() as conn:
        try:
            _begin(conn)
            guard = _select_guard(conn, validated_robot_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return guard


def list_guards() -> list[AuthorityGuard]:
    """Return every validated guard in deterministic robot-id order."""

    with config._get_conn() as conn:
        try:
            _begin(conn)
            rows = conn.execute(
                f"""SELECT {", ".join(_COLUMNS)} FROM {_TABLE_NAME}
                    ORDER BY robot_id"""
            ).fetchall()
            guards = [_row_to_guard(row) for row in rows]
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return guards


def update_guard(
    robot_id: str,
    session_id: str,
    *,
    phase: str,
    dispatch_generation: int,
    updated_at: float | None = None,
) -> AuthorityGuard:
    """Conditionally advance one exact robot/session guard in one transaction.

    An exact-pair miss, storage error, or backwards generation/time transition
    raises, so callers must keep their in-memory authority lock when the
    persistent result is uncertain.  ``updated_at`` defaults to the wall clock
    but never moves an existing timestamp backwards after a clock rollback.
    """

    validated_robot_id = _authority_id(robot_id, "robot_id")
    validated_session_id = _canonical_uuid(session_id, "session_id")
    validated_phase = _phase(phase)
    validated_generation = _integer(
        dispatch_generation,
        "dispatch_generation",
        minimum=0,
        maximum=MAX_SQLITE_INTEGER,
    )
    validated_updated_at = (
        None if updated_at is None else _timestamp(updated_at, "updated_at")
    )

    with config._get_conn() as conn:
        try:
            _begin(conn)
            current = _select_guard(conn, validated_robot_id)
            if current is None or current.session_id != validated_session_id:
                raise AuthorityGuardNotFound("exact authority guard pair was not found")
            if validated_generation < current.dispatch_generation:
                raise AuthorityGuardStateConflict("dispatch_generation cannot regress")
            next_updated_at = (
                max(_timestamp(time.time(), "updated_at"), current.updated_at)
                if validated_updated_at is None
                else validated_updated_at
            )
            if next_updated_at < current.updated_at:
                raise AuthorityGuardStateConflict("updated_at cannot regress")
            cursor = conn.execute(
                f"""UPDATE {_TABLE_NAME}
                    SET phase=?, dispatch_generation=?, updated_at=?
                    WHERE robot_id=? AND session_id=?""",
                (
                    validated_phase,
                    validated_generation,
                    next_updated_at,
                    validated_robot_id,
                    validated_session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise AuthorityGuardStateConflict(
                    "authority guard changed concurrently"
                )
            updated = _select_guard(conn, validated_robot_id)
            if updated is None:
                raise AuthorityGuardStateConflict(
                    "authority guard disappeared during update"
                )
            conn.commit()
            return updated
        except Exception:
            conn.rollback()
            raise


def delete_guard(robot_id: str, session_id: str) -> bool:
    """Delete only the exact robot/session guard and report whether it existed."""

    validated_robot_id = _authority_id(robot_id, "robot_id")
    validated_session_id = _canonical_uuid(session_id, "session_id")
    with config._get_conn() as conn:
        try:
            _begin(conn)
            cursor = conn.execute(
                f"DELETE FROM {_TABLE_NAME} WHERE robot_id=? AND session_id=?",
                (validated_robot_id, validated_session_id),
            )
            if cursor.rowcount not in {0, 1}:
                raise AuthorityGuardStateConflict(
                    "authority guard delete was not singular"
                )
            deleted = cursor.rowcount == 1
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            raise
