"""SQLite connection policy, migration runner, and storage diagnostics."""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import NoReturn

from .errors import (
    InvalidMemoryError,
    MigrationError,
    StorageBusyError,
    StorageDamagedError,
    UnsupportedSchemaVersionError,
)
from .migrations import APPLICATION_ID, LATEST_SCHEMA_VERSION, MIGRATIONS
from .models import MAX_EPOCH_MS

Clock = Callable[[], int]

MAX_BUSY_TIMEOUT_MS = (1 << 31) - 1
_SQLITE_BUSY = getattr(sqlite3, "SQLITE_BUSY", 5)
_SQLITE_LOCKED = getattr(sqlite3, "SQLITE_LOCKED", 6)


def epoch_ms() -> int:
    """Return the current Unix epoch time in milliseconds."""

    return time.time_ns() // 1_000_000


def _sqlite_primary_code(exc: sqlite3.Error) -> int | None:
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return code & 0xFF
    return None


def _is_busy_error(exc: sqlite3.Error) -> bool:
    code = _sqlite_primary_code(exc)
    if code in {_SQLITE_BUSY, _SQLITE_LOCKED}:
        return True
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _raise_storage_error(message: str, exc: sqlite3.Error) -> NoReturn:
    if _is_busy_error(exc):
        raise StorageBusyError(message) from exc
    raise StorageDamagedError(message) from exc


class MemoryDatabase:
    """Own one on-disk SQLite database; every operation gets its own connection."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        clock: Clock = epoch_ms,
    ) -> None:
        try:
            self.path = Path(path)
        except (TypeError, ValueError) as exc:
            raise InvalidMemoryError("database path must be a filesystem path") from exc
        if str(self.path) == ":memory:" or str(self.path).startswith("file:"):
            raise InvalidMemoryError("Memory Core requires a real filesystem database")
        if (
            isinstance(busy_timeout_ms, bool)
            or not isinstance(busy_timeout_ms, int)
            or not 1 <= busy_timeout_ms <= MAX_BUSY_TIMEOUT_MS
        ):
            raise InvalidMemoryError(
                f"busy_timeout_ms must be an integer between 1 and {MAX_BUSY_TIMEOUT_MS}"
            )
        if not callable(clock):
            raise InvalidMemoryError("clock must be callable")
        self.busy_timeout_ms = busy_timeout_ms
        self.clock = clock

    def _prepare_path(self, *, enforce_permissions: bool) -> None:
        try:
            parent = self.path.parent
            parent_existed = parent.exists()
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not parent_existed:
                parent.chmod(0o700)
            if self.path.is_symlink():
                raise StorageDamagedError("database path must not be a symbolic link")
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if not self.path.is_file():
                    raise StorageDamagedError("database path is not a regular file") from None
            else:
                os.close(fd)
            if enforce_permissions and stat.S_IMODE(self.path.stat().st_mode) != 0o600:
                self.path.chmod(0o600)
        except StorageDamagedError:
            raise
        except OSError as exc:
            raise StorageDamagedError("failed to prepare the database path") from exc

    @staticmethod
    def _application_id(connection: sqlite3.Connection) -> int:
        row = connection.execute("PRAGMA application_id").fetchone()
        if row is None:
            raise StorageDamagedError("SQLite did not return an application_id")
        return int(row[0])

    @classmethod
    def _check_application_id(cls, connection: sqlite3.Connection) -> int:
        application_id = cls._application_id(connection)
        if application_id not in {0, APPLICATION_ID}:
            raise StorageDamagedError("database application_id belongs to a different application")
        return application_id

    @staticmethod
    def _configure_owned_connection(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA recursive_triggers = ON")
        connection.execute("PRAGMA synchronous = NORMAL")

    def connect(
        self,
        *,
        enforce_permissions: bool = True,
        configure_owned: bool = True,
    ) -> sqlite3.Connection:
        """Open and configure one operation-local SQLite connection."""

        self._prepare_path(enforce_permissions=enforce_permissions)
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1_000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            _raise_storage_error("failed to open the Memory Core database", exc)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            self._check_application_id(connection)
            if configure_owned:
                self._configure_owned_connection(connection)
        except sqlite3.Error as exc:
            connection.close()
            _raise_storage_error("failed to configure the Memory Core database", exc)
        except Exception:
            connection.close()
            raise
        return connection

    def _clock_ms(self) -> int:
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_EPOCH_MS:
            raise InvalidMemoryError(
                f"clock must return an epoch millisecond between 0 and {MAX_EPOCH_MS}"
            )
        return value

    def _retry_locked(self, operation: Callable[[], None], message: str) -> None:
        deadline = time.monotonic() + self.busy_timeout_ms / 1_000
        while True:
            try:
                operation()
                return
            except sqlite3.Error as exc:
                if not _is_busy_error(exc):
                    _raise_storage_error(message, exc)
                if time.monotonic() >= deadline:
                    raise StorageBusyError(message) from exc
                time.sleep(0.01)

    def _enable_wal(self, connection: sqlite3.Connection) -> None:
        journal_mode: str | None = None

        def enable() -> None:
            nonlocal journal_mode
            row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            journal_mode = None if row is None else str(row[0]).lower()

        self._retry_locked(enable, "timed out while enabling SQLite WAL mode")
        if journal_mode != "wal":
            raise StorageDamagedError(f"SQLite refused WAL mode: {journal_mode}")

    def _begin(self, connection: sqlite3.Connection, mode: str) -> None:
        self._retry_locked(
            lambda: connection.execute(f"BEGIN {mode}"),
            f"timed out while starting a {mode.lower()} transaction",
        )

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        if not connection.in_transaction:
            return
        with suppress(sqlite3.Error):
            connection.execute("ROLLBACK")

    @staticmethod
    def _validate_migration_plan() -> None:
        if not MIGRATIONS:
            raise MigrationError("this build contains no schema migrations")
        expected_version = 1
        for migration in MIGRATIONS:
            if migration.version != expected_version:
                raise MigrationError("migrations must be an ordered continuous sequence")
            if not migration.name.strip() or not migration.statements:
                raise MigrationError(f"migration {migration.version} is incomplete")
            expected_version += 1
        if MIGRATIONS[-1].version != LATEST_SCHEMA_VERSION:
            raise MigrationError("LATEST_SCHEMA_VERSION does not match the migration plan")

    def initialize(self) -> None:
        """Claim the database file and apply every pending migration atomically."""

        self._validate_migration_plan()
        connection = self.connect(enforce_permissions=False, configure_owned=False)
        try:
            self._begin(connection, "EXCLUSIVE")

            application_id = self._check_application_id(connection)
            if application_id == 0:
                existing_objects = tuple(
                    (row[0], row[1])
                    for row in connection.execute(
                        """
                        SELECT type, name
                        FROM sqlite_schema
                        WHERE name NOT LIKE 'sqlite_%'
                        ORDER BY type, name
                        """
                    )
                )
                if existing_objects:
                    raise StorageDamagedError(
                        "unclaimed database already contains application objects"
                    )
                connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            if self._application_id(connection) != APPLICATION_ID:
                raise StorageDamagedError("failed to claim the database application_id")
            connection.execute("COMMIT")

            # Only an empty database, or one already carrying our application id,
            # reaches this point. Persistent file policy and WAL are safe now.
            self._prepare_path(enforce_permissions=True)
            self._configure_owned_connection(connection)
            self._enable_wal(connection)
            self._begin(connection, "EXCLUSIVE")
            if self._application_id(connection) != APPLICATION_ID:
                raise StorageDamagedError("database application_id changed during initialization")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version       INTEGER PRIMARY KEY,
                    name          TEXT NOT NULL,
                    checksum      TEXT NOT NULL,
                    applied_at_ms INTEGER NOT NULL CHECK (applied_at_ms >= 0)
                )
                """
            )
            rows = {
                int(row["version"]): row
                for row in connection.execute(
                    "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
                )
            }
            if rows and max(rows) > LATEST_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"database schema {max(rows)} is newer than supported {LATEST_SCHEMA_VERSION}"
                )
            versions = sorted(rows)
            if versions and versions != list(range(1, versions[-1] + 1)):
                raise MigrationError("applied migrations are not a continuous prefix")

            for migration in MIGRATIONS:
                applied = rows.get(migration.version)
                if applied is not None:
                    if (
                        applied["name"] != migration.name
                        or applied["checksum"] != migration.checksum
                    ):
                        raise MigrationError(
                            f"migration {migration.version} checksum/name does not match this build"
                        )
                    continue
                try:
                    for statement in migration.statements:
                        connection.execute(statement)
                except sqlite3.Error as exc:
                    if _is_busy_error(exc):
                        raise StorageBusyError(
                            f"timed out while applying migration {migration.version}"
                        ) from exc
                    raise MigrationError(
                        f"failed to apply migration {migration.version} ({migration.name})"
                    ) from exc
                connection.execute(
                    """
                    INSERT INTO schema_migrations(version, name, checksum, applied_at_ms)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        migration.version,
                        migration.name,
                        migration.checksum,
                        self._clock_ms(),
                    ),
                )
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(connection)
            _raise_storage_error("failed to initialize the Memory Core database", exc)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection for one read operation."""

        connection = self.connect()
        try:
            self._begin(connection, "DEFERRED")
            yield connection
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(connection)
            _raise_storage_error("failed to read the Memory Core database", exc)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield one immediate transaction and commit or roll it back as a unit."""

        connection = self.connect()
        try:
            self._begin(connection, "IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(connection)
            _raise_storage_error("failed to write the Memory Core database", exc)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
            return int(row[0])

    def migration_rows(self) -> tuple[tuple[int, str, str], ...]:
        with self.read_connection() as connection:
            return tuple(
                (int(row["version"]), str(row["name"]), str(row["checksum"]))
                for row in connection.execute(
                    "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
                )
            )

    def rebuild_search_index(self) -> None:
        """Rebuild FTS from active entries only."""

        try:
            with self.write_transaction() as connection:
                connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('delete-all')")
                connection.execute(
                    """
                    INSERT INTO memory_fts(rowid, title, body, tags_text)
                    SELECT row_id, title, body, tags_text
                    FROM memory_entries
                    WHERE state = 'active'
                    """
                )
        except StorageBusyError:
            raise
        except StorageDamagedError as exc:
            raise StorageDamagedError("failed to rebuild the Memory Core search index") from exc

    def integrity_report(self) -> dict[str, object]:
        """Validate SQLite, foreign keys, and the active-only FTS projection."""

        try:
            with self.read_connection() as connection:
                integrity = tuple(row[0] for row in connection.execute("PRAGMA integrity_check"))
                foreign_keys = tuple(
                    tuple(row) for row in connection.execute("PRAGMA foreign_key_check")
                )
                # Rank 0 checks the FTS index itself. Rank 1 would require every
                # external-content row to be indexed, while deleted entries are
                # deliberately omitted and checked separately below.
                connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('integrity-check')")
                entry_count = int(
                    connection.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
                )
                active_entry_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM memory_entries WHERE state = 'active'"
                    ).fetchone()[0]
                )
                deleted_entry_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM memory_entries WHERE state = 'deleted'"
                    ).fetchone()[0]
                )
                change_count = int(
                    connection.execute("SELECT COUNT(*) FROM memory_changes").fetchone()[0]
                )
                request_count = int(
                    connection.execute("SELECT COUNT(*) FROM memory_requests").fetchone()[0]
                )
                fts_count = int(
                    connection.execute("SELECT COUNT(*) FROM memory_fts_docsize").fetchone()[0]
                )
                fts_missing_active = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM memory_entries AS entry
                        LEFT JOIN memory_fts_docsize AS indexed ON indexed.id = entry.row_id
                        WHERE entry.state = 'active' AND indexed.id IS NULL
                        """
                    ).fetchone()[0]
                )
                fts_unexpected_rows = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM memory_fts_docsize AS indexed
                        LEFT JOIN memory_entries AS entry ON entry.row_id = indexed.id
                        WHERE entry.row_id IS NULL OR entry.state <> 'active'
                        """
                    ).fetchone()[0]
                )
                schema_version = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
                    ).fetchone()[0]
                )
                pragmas = {
                    "application_id": self._application_id(connection),
                    "journal_mode": str(
                        connection.execute("PRAGMA journal_mode").fetchone()[0]
                    ).lower(),
                    "foreign_keys": int(connection.execute("PRAGMA foreign_keys").fetchone()[0]),
                    "busy_timeout": int(connection.execute("PRAGMA busy_timeout").fetchone()[0]),
                    "recursive_triggers": int(
                        connection.execute("PRAGMA recursive_triggers").fetchone()[0]
                    ),
                }
        except StorageBusyError:
            raise
        except StorageDamagedError as exc:
            raise StorageDamagedError("failed to inspect Memory Core integrity") from exc

        problems: list[str] = []
        if integrity != ("ok",):
            problems.append("SQLite integrity_check failed")
        if foreign_keys:
            problems.append("foreign key violations exist")
        if fts_count != active_entry_count or fts_missing_active or fts_unexpected_rows:
            problems.append("search index does not match active entries")
        if pragmas["application_id"] != APPLICATION_ID:
            problems.append("application_id does not match")
        if problems:
            raise StorageDamagedError("Memory Core integrity check failed: " + "; ".join(problems))

        return {
            "integrity": integrity,
            "foreign_key_violations": foreign_keys,
            "schema_version": schema_version,
            "entry_count": entry_count,
            "active_entry_count": active_entry_count,
            "deleted_entry_count": deleted_entry_count,
            "change_count": change_count,
            "request_count": request_count,
            "fts_count": fts_count,
            "fts_missing_active": fts_missing_active,
            "fts_unexpected_rows": fts_unexpected_rows,
            "pragmas": pragmas,
        }
