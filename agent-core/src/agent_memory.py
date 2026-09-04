"""Agent-owned long-term memory backed by the standalone Memory Core package.

This adapter deliberately exposes one canonical prompt document.  Agent Core
does not yet have a trustworthy user identity shared by every input channel, so
PR2 must not turn channel payload fields into Memory Core ownership keys.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from memory_core import (
    AccessContext,
    InvalidMemoryError,
    ListQuery,
    MemoryCoreError,
    MemoryDraft,
    MemoryPatch,
    MemoryPlace,
    MemoryRecord,
    MemoryStore,
    MigrationError,
    RevisionConflictError,
    StorageBusyError,
    StorageDamagedError,
    UnsupportedSchemaVersionError,
)

_OWNER_KEY = 'agent:main'
_PROMPT_KIND = 'agent_prompt'
_PROMPT_TITLE = 'Agent long-term memory'
_BOOTSTRAP_OPERATION = 'agent-prompt-bootstrap-v1'
_FALSE_VALUES = {'0', 'false', 'no', 'off'}
_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_REFRESH_POLL_SECONDS = 0.25
_HANDOFF_LOCK_TIMEOUT_SECONDS = 5.0
_COMPATIBILITY_STATE_VERSION = 2
COMPATIBILITY_WARNING = (
    '长期记忆已保存，但兼容副本未同步；切换存储后端前请先检查并处理兼容文件。'
)
DEGRADED_WARNING = '长期记忆存储当前处于只读降级状态；请检查存储后重启服务。'


class AgentMemoryError(RuntimeError):
    """Base error exposed to Agent Core callers without storage internals."""


class AgentMemoryValidationError(AgentMemoryError):
    """The requested prompt cannot be stored as long-term memory."""


class AgentMemoryUnavailableError(AgentMemoryError):
    """The configured durable backend cannot currently accept a write."""


class AgentMemoryCommitUncertainError(AgentMemoryUnavailableError):
    """A file-backed write became visible but durability was not confirmed."""


class _CompatibilityConflictError(AgentMemoryUnavailableError):
    """A direct file edit conflicts with the last confirmed Core mirror."""


class _HandoffLockTimeout(AgentMemoryUnavailableError):
    """A cooperating process held the memory hand-off lock too long."""


class _AtomicWritePostCommitError(OSError):
    """A replacement became visible but its directory sync was not confirmed."""


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    text: str
    revision: int | None
    cache_key: str
    backend: str
    fallback_ready: bool = True


@dataclass(frozen=True, slots=True)
class _CompatibilityState:
    source: str
    body: str
    core_revision: int | None
    mirror_synced: bool
    mirror_digest: str
    mirror_signature: tuple[int, int, int, int, int] | None


StoreFactory = Callable[[str | pathlib.Path], MemoryStore]

_PROCESS_LOCKS: dict[pathlib.Path, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()
_NO_MIRROR_PRECONDITION = object()


def _normalized_prompt(value: object) -> str:
    if not isinstance(value, str):
        raise AgentMemoryValidationError('long-term memory is invalid')
    try:
        patch = MemoryPatch(body=value)
    except InvalidMemoryError as error:
        raise AgentMemoryValidationError('long-term memory is invalid') from error
    assert patch.body is not None
    return patch.body


def _prompt_matches(candidate: str | None, normalized: str) -> bool:
    if candidate is None:
        return False
    try:
        return _normalized_prompt(candidate) == normalized
    except AgentMemoryValidationError:
        return False


def _mirror_digest(text: str | None) -> str:
    """Fingerprint exact mirror bytes while distinguishing a missing file."""

    prefix = b'missing\0' if text is None else b'present\0'
    payload = b'' if text is None else text.encode('utf-8')
    return hashlib.sha256(prefix + payload).hexdigest()


def _compatibility_state_checksum(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _text_snapshot(
    text: str,
    *,
    backend: str,
    fallback_ready: bool = True,
) -> MemorySnapshot:
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return MemorySnapshot(
        text=text,
        revision=None,
        cache_key=f'{backend}:{digest}',
        backend=backend,
        fallback_ready=fallback_ready,
    )


def _file_snapshot(path: pathlib.Path, *, backend: str) -> MemorySnapshot:
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        text = ''
    return _text_snapshot(text, backend=backend)


def _record_snapshot(record: MemoryRecord) -> MemorySnapshot:
    return MemorySnapshot(
        text=record.body,
        revision=record.revision,
        cache_key=f'core:{record.memory_id}:{record.revision}',
        backend='core',
        fallback_ready=True,
    )


def _atomic_write(path: pathlib.Path, text: str) -> None:
    """Replace a compatibility file without exposing a partially written prompt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    target_owner: tuple[int, int] | None = None
    try:
        target_metadata = path.lstat()
    except FileNotFoundError:
        # The operator-facing compatibility document remains readable with the
        # same policy as a normal checked-out file. Hidden journals default to
        # owner-only because they contain the same long-term-memory body.
        target_mode = 0o600 if path.name.startswith('.') else 0o644
    else:
        if not stat.S_ISREG(target_metadata.st_mode):
            raise OSError('atomic-write target must be a regular file')
        target_mode = stat.S_IMODE(target_metadata.st_mode)
        target_owner = (target_metadata.st_uid, target_metadata.st_gid)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            if target_owner is not None:
                os.fchown(temporary.fileno(), *target_owner)
            os.fchmod(temporary.fileno(), target_mode)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
            directory_fd = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise _AtomicWritePostCommitError(
                'replacement is visible but its directory sync failed'
            ) from error
    finally:
        if temporary_name is not None:
            # Preserve the original write error; a stale temporary file is
            # less important than reporting that the mirror was not replaced.
            with contextlib.suppress(OSError):
                pathlib.Path(temporary_name).unlink(missing_ok=True)


class AgentMemory:
    """Own one canonical Agent prompt record and its compatibility mirror."""

    def __init__(
        self,
        *,
        enabled: bool,
        database_path: str | pathlib.Path,
        legacy_path: str | pathlib.Path,
        store_factory: StoreFactory = MemoryStore.open,
    ) -> None:
        self._enabled = enabled
        self._database_path = pathlib.Path(database_path)
        self._legacy_path = pathlib.Path(legacy_path)
        try:
            legacy_is_symlink = stat.S_ISLNK(self._legacy_path.lstat().st_mode)
        except FileNotFoundError:
            legacy_is_symlink = False
        except OSError:
            # Initialization maps inaccessible paths to a read-only degraded
            # snapshot; do not turn a transient probe error into construction.
            legacy_is_symlink = False
        self._legacy_path_is_symlink = legacy_is_symlink
        try:
            database_identity = self._database_path.resolve(strict=False)
            legacy_identity = self._legacy_path.resolve(strict=False)
        except (OSError, RuntimeError):
            database_identity = self._database_path.absolute()
            legacy_identity = self._legacy_path.absolute()
        if database_identity == legacy_identity and not legacy_is_symlink:
            raise AgentMemoryValidationError(
                'memory database and compatibility file must use different paths'
            )
        self._state_path = self._legacy_path.with_name(
            f'.{self._legacy_path.name}.state.json'
        )
        self._store_factory = store_factory
        self._lock = threading.RLock()
        self._initialized = False
        self._backend = 'uninitialized'
        self._file_snapshot_pinned = False
        self._store: MemoryStore | None = None
        self._record: MemoryRecord | None = None
        self._last_core_revision: int | None = None
        self._observable_signature: tuple[object, ...] | None = None
        self._file_mirror_signature: object = _NO_MIRROR_PRECONDITION
        self._pending_file_state: _CompatibilityState | None = None
        self._mirror_precondition_failed = False
        self._mirror_postcommit_visible = False
        self._repairable_core_publication: (
            tuple[int, str, tuple[object, ...]] | None
        ) = None
        self._compatibility_durability_uncertain = False
        self._refresh_guard = threading.Lock()
        self._refresh_running = False
        self._next_refresh_check = 0.0
        self._snapshot = _text_snapshot('', backend='uninitialized')

    @property
    def database_path(self) -> pathlib.Path:
        return self._database_path

    def seed_legacy_if_cold(self, text: str) -> bool:
        """Seed the compatibility file once under the shared outer lock."""

        with self._lock:
            if self._initialized:
                return False
            legacy_lock = self._legacy_path.with_name(
                f'.{self._legacy_path.name}.handoff.lock'
            )
            try:
                normalized = _normalized_prompt(text)
                with self._exclusive_file_lock(legacy_lock):
                    return self._seed_legacy_if_cold_under_handoff(normalized)
            except (AgentMemoryError, OSError, UnicodeError):
                # Initialization will select the safest available fallback and
                # report its status; seeding must never prevent startup.
                return False

    def _seed_legacy_if_cold_under_handoff(self, normalized: str) -> bool:
        """Seed a pristine deployment while the compatibility lock is held."""

        # Rollback mode must neither inspect the durable database nor fabricate
        # a replacement when its compatibility evidence is missing. Normal new
        # deployments start enabled and perform the one-time seed there.
        if not self._enabled:
            return False
        if self._path_signature(self._state_path) is not None:
            return False
        legacy_signature = self._path_signature(self._legacy_path)
        if self._read_legacy_text() is not None:
            return False
        if self._path_signature(self._database_path) is not None:
            return False
        return self._mirror(
            normalized,
            expected_signature=legacy_signature,
        )

    @contextlib.contextmanager
    def _exclusive_file_lock(
        self,
        lock_path: pathlib.Path,
        *,
        deadline: float | None = None,
    ):
        lock_key = lock_path.absolute()
        with _PROCESS_LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(lock_key, threading.RLock())
        if deadline is None:
            deadline = time.monotonic() + _HANDOFF_LOCK_TIMEOUT_SECONDS
        remaining = max(0.0, deadline - time.monotonic())
        if not process_lock.acquire(timeout=remaining):
            raise _HandoffLockTimeout('timed out waiting for the memory hand-off lock')
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('a+b') as lock_file:
                while True:
                    try:
                        fcntl.flock(
                            lock_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                        break
                    except OSError as error:
                        if error.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        if time.monotonic() >= deadline:
                            raise _HandoffLockTimeout(
                                'timed out waiting for the memory hand-off lock'
                            ) from error
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            process_lock.release()

    @contextlib.contextmanager
    def _handoff_lock(self):
        """Serialize a durable write and its compatibility publication."""

        database_lock = self._database_path.with_name(
            f'.{self._database_path.name}.handoff.lock'
        )
        legacy_lock = self._legacy_path.with_name(
            f'.{self._legacy_path.name}.handoff.lock'
        )
        deadline = time.monotonic() + _HANDOFF_LOCK_TIMEOUT_SECONDS
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(
                    self._exclusive_file_lock(database_lock, deadline=deadline)
                )
            except OSError:
                if self._enabled:
                    raise
                # Disabled mode must remain a usable rollback path even when
                # the configured database directory itself is unavailable.
            # Every normal path also takes this second lock, in a fixed order.
            # A disabled process falling back from the database lock therefore
            # remains mutually exclusive with an enabled writer.
            stack.enter_context(
                self._exclusive_file_lock(legacy_lock, deadline=deadline)
            )
            yield

    def _context(self, actor_key: str) -> AccessContext:
        return AccessContext(owner_key=_OWNER_KEY, actor_key=actor_key)

    def _read_legacy_raw(self) -> str | None:
        try:
            return self._legacy_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return None

    def _read_legacy_text(self) -> str | None:
        text = self._read_legacy_raw()
        return text if text is not None and text.strip() else None

    @staticmethod
    def _path_signature(path: pathlib.Path) -> tuple[int, int, int, int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _handoff_signature(self) -> tuple[object, ...]:
        return (
            self._path_signature(self._state_path),
            self._path_signature(self._legacy_path),
        )

    def _stable_legacy_observation(
        self,
        expected_signature: object = _NO_MIRROR_PRECONDITION,
    ) -> tuple[tuple[int, int, int, int, int] | None, str, str | None] | None:
        """Capture exact content and generation from one stable mirror read."""

        for _ in range(3):
            before = self._path_signature(self._legacy_path)
            if (
                expected_signature is not _NO_MIRROR_PRECONDITION
                and before != expected_signature
            ):
                return None
            text = self._read_legacy_raw()
            after = self._path_signature(self._legacy_path)
            if before == after:
                return before, _mirror_digest(text), text
        return None

    def _current_observable_signature(self) -> tuple[object, ...]:
        wal_path = pathlib.Path(f'{self._database_path}-wal')
        return (
            *self._handoff_signature(),
            self._path_signature(self._database_path),
            self._path_signature(wal_path),
        )

    def _capture_observable_signature(self) -> None:
        try:
            self._observable_signature = self._current_observable_signature()
        except OSError:
            self._observable_signature = None

    def _set_core_record(
        self,
        record: MemoryRecord,
        *,
        fallback_ready: bool = True,
    ) -> MemorySnapshot:
        self._record = record
        self._last_core_revision = record.revision
        snapshot = _record_snapshot(record)
        if not fallback_ready:
            snapshot = MemorySnapshot(
                text=snapshot.text,
                revision=snapshot.revision,
                cache_key=snapshot.cache_key,
                backend=snapshot.backend,
                fallback_ready=False,
            )
        self._snapshot = snapshot
        return snapshot

    def _mirror(
        self,
        text: str,
        *,
        expected_signature: object = _NO_MIRROR_PRECONDITION,
    ) -> bool:
        self._mirror_precondition_failed = False
        self._mirror_postcommit_visible = False
        read_failed = False
        try:
            current = self._legacy_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            current = None
        except (OSError, UnicodeError):
            current = None
            read_failed = True
        if _prompt_matches(current, text):
            return True
        if expected_signature is not _NO_MIRROR_PRECONDITION:
            try:
                current_signature = self._path_signature(self._legacy_path)
            except OSError:
                current_signature = _NO_MIRROR_PRECONDITION
            if read_failed or current_signature != expected_signature:
                self._mirror_precondition_failed = True
                print('[memory] compatibility mirror changed before publication')
                return False
        try:
            _atomic_write(self._legacy_path, text)
        except _AtomicWritePostCommitError as error:
            self._compatibility_durability_uncertain = True
            self._mirror_postcommit_visible = True
            print(
                '[memory] compatibility mirror durability is unconfirmed: '
                f'{type(error.__cause__).__name__}'
            )
            return False
        except OSError as error:
            print(f'[memory] compatibility mirror failed: {type(error).__name__}')
            return False
        self._compatibility_durability_uncertain = False
        return True

    def _confirm_compatibility_directory(self) -> bool:
        """Confirm visible journal/mirror renames before a file-backed success."""

        directory_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        for _ in range(3):
            try:
                directory_fd = os.open(self._legacy_path.parent, directory_flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as error:
                last_error = error
                continue
            self._compatibility_durability_uncertain = False
            return True
        print(
            '[memory] compatibility directory durability is unconfirmed: '
            f'{type(last_error).__name__}'
        )
        return False

    def _legacy_snapshot(self, *, backend: str) -> MemorySnapshot:
        try:
            return _file_snapshot(self._legacy_path, backend=backend)
        except (OSError, UnicodeError) as error:
            print(f'[memory] compatibility file unavailable: {type(error).__name__}')
            return _text_snapshot(self._snapshot.text, backend=backend)

    def _finalize_file_snapshot(self, snapshot: MemorySnapshot) -> MemorySnapshot:
        """Publish a file-backed snapshot with truthful mirror readiness."""

        mirrored = True
        state_ready = snapshot.fallback_ready
        self._mirror_precondition_failed = False
        if not self._file_snapshot_pinned:
            mirrored = self._mirror(
                snapshot.text,
                expected_signature=self._file_mirror_signature,
            )
        pending_state = self._pending_file_state
        self._pending_file_state = None
        if mirrored and pending_state is not None:
            state_ready = self._write_synced_compatibility_state(
                source=pending_state.source,
                body=pending_state.body,
                core_revision=pending_state.core_revision,
            )
        self._snapshot = _text_snapshot(
            snapshot.text,
            backend=snapshot.backend,
            fallback_ready=(
                state_ready
                and mirrored
                and not self._compatibility_durability_uncertain
            ),
        )
        if self._mirror_precondition_failed:
            self._observable_signature = None
        else:
            self._capture_observable_signature()
        return self._snapshot

    def _read_stable_compatibility(
        self,
    ) -> tuple[_CompatibilityState | None, str | None, tuple[object, ...]]:
        """Read journal and mirror from one observable generation."""

        for _ in range(3):
            before = self._handoff_signature()
            state = self._read_compatibility_state()
            legacy_text = self._read_legacy_text()
            after = self._handoff_signature()
            if before == after:
                return state, legacy_text, after
        raise _CompatibilityConflictError(
            'compatibility files changed repeatedly while being read'
        )

    def _read_compatibility_state(self) -> _CompatibilityState | None:
        try:
            raw = self._state_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise AgentMemoryValidationError(
                'compatibility state is invalid'
            ) from error
        if not isinstance(payload, dict):
            raise AgentMemoryValidationError('compatibility state is invalid')
        expected_keys = {
            'body',
            'checksum',
            'core_revision',
            'mirror_digest',
            'mirror_signature',
            'mirror_synced',
            'source',
            'version',
        }
        if set(payload) != expected_keys:
            raise AgentMemoryValidationError('compatibility state is invalid')
        checksum = payload.get('checksum')
        checksummed = {
            key: value for key, value in payload.items() if key != 'checksum'
        }
        if (
            payload.get('version') != _COMPATIBILITY_STATE_VERSION
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or checksum != _compatibility_state_checksum(checksummed)
        ):
            raise AgentMemoryValidationError('compatibility state is invalid')
        source = payload.get('source')
        revision = payload.get('core_revision')
        mirror_synced = payload.get('mirror_synced')
        mirror_digest = payload.get('mirror_digest')
        mirror_signature = payload.get('mirror_signature')
        if source not in {'core', 'file'}:
            raise AgentMemoryValidationError('compatibility state is invalid')
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
        ):
            raise AgentMemoryValidationError('compatibility state is invalid')
        if not isinstance(mirror_synced, bool):
            raise AgentMemoryValidationError('compatibility state is invalid')
        if (
            not isinstance(mirror_digest, str)
            or len(mirror_digest) != 64
            or any(character not in '0123456789abcdef' for character in mirror_digest)
        ):
            raise AgentMemoryValidationError('compatibility state is invalid')
        if mirror_signature is not None and (
            not isinstance(mirror_signature, list)
            or len(mirror_signature) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in mirror_signature
            )
        ):
            raise AgentMemoryValidationError('compatibility state is invalid')
        return _CompatibilityState(
            source=source,
            body=_normalized_prompt(payload.get('body')),
            core_revision=revision,
            mirror_synced=mirror_synced,
            mirror_digest=mirror_digest,
            mirror_signature=(
                tuple(mirror_signature) if mirror_signature is not None else None
            ),
        )

    def _write_compatibility_state(self, state: _CompatibilityState) -> bool:
        state_payload: dict[str, object] = {
            'body': state.body,
            'core_revision': state.core_revision,
            'mirror_digest': state.mirror_digest,
            'mirror_signature': (
                list(state.mirror_signature)
                if state.mirror_signature is not None
                else None
            ),
            'mirror_synced': state.mirror_synced,
            'source': state.source,
            'version': _COMPATIBILITY_STATE_VERSION,
        }
        state_payload['checksum'] = _compatibility_state_checksum(state_payload)
        payload = json.dumps(
            state_payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            _atomic_write(self._state_path, payload)
        except _AtomicWritePostCommitError as error:
            # The new journal is already observable. Treating the request as
            # rejected would allow that supposedly failed intent to take effect
            # during a later refresh. Continue publication, but retain an
            # unconfirmed-durability signal until a directory sync succeeds.
            self._compatibility_durability_uncertain = True
            print(
                '[memory] compatibility state durability is unconfirmed: '
                f'{type(error.__cause__).__name__}'
            )
            return True
        except OSError as error:
            print(f'[memory] compatibility state write failed: {type(error).__name__}')
            return False
        self._compatibility_durability_uncertain = False
        return True

    def _synced_compatibility_state(
        self,
        *,
        source: str,
        body: str,
        core_revision: int | None,
    ) -> _CompatibilityState | None:
        try:
            observation = self._stable_legacy_observation()
        except (OSError, UnicodeError) as error:
            print(f'[memory] compatibility mirror proof failed: {type(error).__name__}')
            return None
        if observation is None or not _prompt_matches(observation[2], body):
            print('[memory] compatibility mirror changed before confirmation')
            return None
        signature, digest, _ = observation
        return _CompatibilityState(
            source=source,
            body=body,
            core_revision=core_revision,
            mirror_synced=True,
            mirror_digest=digest,
            mirror_signature=signature,
        )

    def _write_synced_compatibility_state(
        self,
        *,
        source: str,
        body: str,
        core_revision: int | None,
    ) -> bool:
        state = self._synced_compatibility_state(
            source=source,
            body=body,
            core_revision=core_revision,
        )
        return (
            state is not None
            and self._write_compatibility_state(state)
            and not self._compatibility_durability_uncertain
        )

    def _publish_core_compatibility(
        self,
        record: MemoryRecord,
        *,
        expected_legacy_signature: object = _NO_MIRROR_PRECONDITION,
        previous_mirror_digest: str,
        previous_mirror_signature: tuple[int, int, int, int, int] | None,
    ) -> bool:
        unsynced = _CompatibilityState(
            source='core',
            body=record.body,
            core_revision=record.revision,
            mirror_synced=False,
            mirror_digest=previous_mirror_digest,
            mirror_signature=previous_mirror_signature,
        )
        state_written = self._write_compatibility_state(unsynced)
        mirrored = self._mirror(
            record.body,
            expected_signature=expected_legacy_signature,
        )
        synced_state_written = False
        if mirrored:
            synced_state_written = self._write_synced_compatibility_state(
                source='core',
                body=record.body,
                core_revision=record.revision,
            )
        elif not state_written:
            self._write_compatibility_state(unsynced)
        ready = (
            mirrored
            and synced_state_written
            and not self._compatibility_durability_uncertain
        )
        if not ready:
            print('[memory] compatibility hand-off is unavailable')
        if self._mirror_precondition_failed:
            self._observable_signature = None
        else:
            self._capture_observable_signature()
        if ready:
            self._repairable_core_publication = None
        elif not self._mirror_precondition_failed:
            try:
                repair_signature = self._handoff_signature()
            except OSError:
                self._observable_signature = None
                self._repairable_core_publication = None
            else:
                self._repairable_core_publication = (
                    record.revision,
                    record.body,
                    repair_signature,
                )
        else:
            self._repairable_core_publication = None
        return ready

    def _publish_core_if_handoff_unchanged(
        self,
        record: MemoryRecord,
        expected_signature: tuple[object, ...],
    ) -> bool:
        try:
            unchanged = self._handoff_signature() == expected_signature
            previous_observation = self._stable_legacy_observation(
                expected_signature[1]
            )
            unchanged = unchanged and self._handoff_signature() == expected_signature
        except (OSError, UnicodeError):
            unchanged = False
            previous_observation = None
        if not unchanged or previous_observation is None:
            self._repairable_core_publication = None
            self._observable_signature = None
            print('[memory] compatibility hand-off changed during durable write')
            return False
        previous_signature, previous_digest, _ = previous_observation
        return self._publish_core_compatibility(
            record,
            expected_legacy_signature=expected_signature[1],
            previous_mirror_digest=previous_digest,
            previous_mirror_signature=previous_signature,
        )

    def _compatibility_conflict_snapshot(self, text: str) -> MemorySnapshot:
        print('[memory] compatibility hand-off needs manual reconciliation')
        snapshot = _text_snapshot(text, backend='degraded', fallback_ready=False)
        self._snapshot = snapshot
        self._file_snapshot_pinned = True
        self._backend = 'degraded'
        return snapshot

    def _file_backend_snapshot(
        self,
        *,
        backend: str,
    ) -> MemorySnapshot:
        state, legacy_text, handoff_signature = self._read_stable_compatibility()
        self._file_mirror_signature = handoff_signature[1]
        self._pending_file_state = None

        if state is None:
            self._last_core_revision = None
            if legacy_text is None:
                return self._compatibility_conflict_snapshot('')
            return _text_snapshot(legacy_text or '', backend=backend)

        self._last_core_revision = state.core_revision

        if (
            state.mirror_synced
            and legacy_text is not None
            and not _prompt_matches(legacy_text, state.body)
        ):
            # The file changed after a confirmed mirror.  In explicitly
            # disabled mode the operator-selected file is authoritative.
            body = _normalized_prompt(legacy_text)
            journaled = self._write_compatibility_state(
                file_state := _CompatibilityState(
                    source='file',
                    body=body,
                    core_revision=state.core_revision,
                    mirror_synced=True,
                    mirror_digest=_mirror_digest(legacy_text),
                    mirror_signature=handoff_signature[1],
                )
            )
            state_ready = journaled and not self._compatibility_durability_uncertain
            if not state_ready:
                self._pending_file_state = file_state
            return _text_snapshot(body, backend=backend, fallback_ready=state_ready)

        if (
            not state.mirror_synced
            and not _prompt_matches(legacy_text, state.body)
            and not self._legacy_matches_observation(state)
        ):
            # The file no longer matches the predecessor captured before the
            # interrupted publication, so it may be a later edit.
            return self._compatibility_conflict_snapshot(state.body)

        if not state.mirror_synced:
            self._pending_file_state = state

        return _text_snapshot(
            state.body,
            backend=backend,
            fallback_ready=state.mirror_synced,
        )

    def _legacy_matches_observation(self, state: _CompatibilityState) -> bool:
        observation = self._stable_legacy_observation()
        if observation is None:
            return False
        signature, digest, _ = observation
        return signature == state.mirror_signature and digest == state.mirror_digest

    def _fallback_snapshot(self, *, backend: str) -> MemorySnapshot:
        try:
            state = self._read_compatibility_state()
        except (AgentMemoryError, OSError, UnicodeError) as error:
            print(f'[memory] compatibility state unavailable: {type(error).__name__}')
        else:
            if state is not None:
                return _text_snapshot(state.body, backend=backend)
        return self._legacy_snapshot(backend=backend)

    def _update_record(
        self,
        store: MemoryStore,
        record: MemoryRecord,
        text: str,
        *,
        actor_key: str,
        reason: str,
    ) -> MemoryRecord:
        transition = (
            f'{record.memory_id}\0{record.revision}\0{actor_key}\0{reason}\0{text}'
        ).encode()
        operation = 'agent-prompt-replace:' + hashlib.sha256(transition).hexdigest()
        return store.update(
            self._context(actor_key),
            record.memory_id,
            MemoryPatch(body=text),
            expected_revision=record.revision,
            op_key=operation,
            note=reason,
        ).record

    def _assert_compatibility_safe_for_core_write(self, record: MemoryRecord) -> None:
        """Refuse a Core write when another mode has unpublished intent."""

        state = self._read_compatibility_state()
        legacy_text = self._read_legacy_text()

        if state is None:
            if not _prompt_matches(legacy_text, record.body):
                raise _CompatibilityConflictError(
                    'compatibility file diverged without a synchronization journal'
                )
            return

        if state.source == 'file':
            if state.body != record.body:
                raise _CompatibilityConflictError(
                    'file-backed memory has a pending durable synchronization'
                )
            if _prompt_matches(legacy_text, state.body):
                return
            if not state.mirror_synced and self._legacy_matches_observation(state):
                return
            raise _CompatibilityConflictError(
                'file-backed compatibility publication conflicts with a later edit'
            )

        state_revision = state.core_revision
        if state_revision is None or record.revision < state_revision:
            raise _CompatibilityConflictError(
                'compatibility state is newer than the durable revision'
            )
        if record.revision == state_revision and record.body != state.body:
            raise _CompatibilityConflictError(
                'compatibility state conflicts with the durable revision'
            )
        if record.revision > state_revision:
            if _prompt_matches(legacy_text, record.body):
                return
            if (
                state.mirror_synced
                and _prompt_matches(legacy_text, state.body)
                and self._legacy_matches_observation(state)
            ):
                return
            raise _CompatibilityConflictError(
                'compatibility file conflicts with a newer durable revision'
            )
        if _prompt_matches(legacy_text, state.body):
            return
        if not state.mirror_synced and self._legacy_matches_observation(state):
            return
        raise _CompatibilityConflictError(
            'compatibility mirror is not confirmed for the durable revision'
        )

    def _bootstrap_text(
        self,
        state: _CompatibilityState | None,
        legacy_text: str | None,
    ) -> str:
        if state is None:
            return _normalized_prompt(legacy_text or '')

        if _prompt_matches(legacy_text, state.body):
            return state.body
        if not state.mirror_synced and self._legacy_matches_observation(state):
            return state.body
        if state.source == 'file' and state.mirror_synced:
            body = _normalized_prompt(legacy_text)
            file_state = self._synced_compatibility_state(
                source='file',
                body=body,
                core_revision=state.core_revision,
            )
            if file_state is not None and self._write_compatibility_state(file_state):
                return body
            raise _CompatibilityConflictError(
                'latest file-backed memory could not be journaled'
            )
        raise _CompatibilityConflictError(
            'compatibility state and file conflict while rebuilding the durable record'
        )

    def _degrade_for_compatibility_conflict(
        self,
        error: _CompatibilityConflictError,
    ) -> None:
        self._snapshot = MemorySnapshot(
            text=self._snapshot.text,
            revision=self._snapshot.revision,
            cache_key=self._snapshot.cache_key,
            backend=self._snapshot.backend,
            fallback_ready=False,
        )
        self._degrade(error)

    def _preferred_conflict_text(self, record: MemoryRecord) -> str:
        try:
            state = self._read_compatibility_state()
            legacy_text = self._read_legacy_text()
        except (AgentMemoryError, OSError, UnicodeError):
            return record.body

        candidate = record.body
        if state is None:
            candidate = legacy_text or record.body
        elif state.source == 'file':
            candidate = state.body
            if (
                state.mirror_synced
                and legacy_text is not None
                and not _prompt_matches(legacy_text, state.body)
            ):
                candidate = legacy_text
        elif legacy_text is not None and not _prompt_matches(legacy_text, state.body):
            candidate = legacy_text
        elif state.core_revision is not None and state.core_revision >= record.revision:
            candidate = state.body
        try:
            return _normalized_prompt(candidate)
        except AgentMemoryValidationError:
            return record.body

    def _degrade_to_compatibility_conflict(
        self,
        record: MemoryRecord,
        error: _CompatibilityConflictError,
    ) -> MemorySnapshot:
        snapshot = _text_snapshot(
            self._preferred_conflict_text(record),
            backend='degraded',
            fallback_ready=False,
        )
        self._store = None
        self._record = None
        self._file_snapshot_pinned = True
        self._snapshot = snapshot
        self._backend = 'degraded'
        print(
            '[memory] backend unavailable; using read-only snapshot: '
            f'{type(error).__name__}'
        )
        return self._snapshot

    def _refresh_core_snapshot_under_handoff(self) -> MemorySnapshot:
        store = self._store
        cached = self._record
        if store is None or cached is None:
            return self._snapshot
        handoff_signature = self._handoff_signature()
        try:
            current = store.read(
                self._context('system:snapshot_refresh'),
                cached.memory_id,
            )
            self._assert_compatibility_safe_for_core_write(current)
        except _CompatibilityConflictError as error:
            return self._degrade_to_compatibility_conflict(current, error)
        except StorageBusyError:
            # A contended read is transient.  Keep the complete cached Core
            # state and leave its observable signature stale so a later
            # snapshot retries the refresh.
            return self._snapshot
        except (
            StorageDamagedError,
            MigrationError,
            UnsupportedSchemaVersionError,
        ) as error:
            return self._degrade(error)
        except MemoryCoreError as error:
            return self._degrade(error)

        if current.revision < cached.revision or (
            current.revision == cached.revision and current.body != cached.body
        ):
            error = _CompatibilityConflictError(
                'durable long-term memory moved to an incompatible revision'
            )
            return self._degrade_to_compatibility_conflict(current, error)

        ready = self._publish_core_if_handoff_unchanged(
            current,
            handoff_signature,
        )
        return self._set_core_record(current, fallback_ready=ready)

    def _degrade(
        self,
        error: BaseException,
        *,
        prefer_legacy: bool = False,
    ) -> MemorySnapshot:
        had_core_snapshot = self._backend == 'core'
        if had_core_snapshot:
            # Once a durable record has been observed it remains more
            # authoritative than a mirror that may have failed to refresh.
            snapshot = _text_snapshot(
                self._snapshot.text,
                backend='degraded',
                fallback_ready=self._snapshot.fallback_ready,
            )
        else:
            snapshot = (
                self._legacy_snapshot(backend='degraded')
                if prefer_legacy
                else self._fallback_snapshot(backend='degraded')
            )
        if isinstance(error, _CompatibilityConflictError):
            snapshot = MemorySnapshot(
                text=snapshot.text,
                revision=snapshot.revision,
                cache_key=snapshot.cache_key,
                backend=snapshot.backend,
                fallback_ready=False,
            )
        self._store = None
        self._record = None
        self._snapshot = snapshot
        self._backend = 'degraded'
        print(
            f'[memory] backend unavailable; using read-only snapshot: {type(error).__name__}'
        )
        return self._snapshot

    def initialize(self, *, seed_text: str | None = None) -> MemorySnapshot:
        """Open/bootstrap the store once; failures leave Agent Core operational."""

        with self._lock:
            if self._initialized:
                return self._snapshot
            self._initialized = True
            if self._legacy_path_is_symlink:
                error = AgentMemoryValidationError(
                    'memory compatibility file must not be a symbolic link'
                )
                snapshot = _text_snapshot('', backend='degraded', fallback_ready=False)
                self._file_snapshot_pinned = True
                self._snapshot = snapshot
                self._backend = 'degraded'
                print(
                    '[memory] backend unavailable; using read-only snapshot: '
                    f'{type(error).__name__}'
                )
                return snapshot
            normalized_seed = None
            if seed_text is not None:
                try:
                    normalized_seed = _normalized_prompt(seed_text)
                except AgentMemoryValidationError:
                    # A packaged-template problem must not stop the rest of the
                    # control plane. Initialization below will expose the safest
                    # available read-only state.
                    pass
            completed_snapshot: MemorySnapshot | None = None
            try:
                with self._handoff_lock():
                    if normalized_seed is not None:
                        try:
                            self._seed_legacy_if_cold_under_handoff(normalized_seed)
                        except (AgentMemoryError, OSError, UnicodeError) as error:
                            print(
                                '[memory] packaged seed unavailable; continuing '
                                f'initialization: {type(error).__name__}'
                            )
                    completed_snapshot = self._initialize_under_handoff()
                return completed_snapshot
            except (AgentMemoryError, MemoryCoreError, OSError, UnicodeError) as error:
                if completed_snapshot is not None:
                    print(
                        '[memory] hand-off lock cleanup failed after initialization: '
                        f'{type(error).__name__}'
                    )
                    return completed_snapshot
                return self._degrade(error)

    def _initialize_under_handoff(self) -> MemorySnapshot:
        if not self._enabled:
            try:
                candidate = self._file_backend_snapshot(
                    backend='file',
                )
            except (AgentMemoryError, OSError, UnicodeError) as error:
                return self._degrade(error)
            if candidate.backend == 'degraded':
                self._backend = 'degraded'
                return candidate
            snapshot = self._finalize_file_snapshot(candidate)
            # Publish the backend last so concurrent readers either wait for
            # initialization or observe the complete file publication.
            self._backend = 'file'
            return snapshot

        try:
            store = self._store_factory(self._database_path)
            context = self._context('system:bootstrap')
            state, legacy_text, publication_signature = (
                self._read_stable_compatibility()
            )
            records = store.list(context, ListQuery(limit=2, kinds=(_PROMPT_KIND,)))
            if len(records) > 1:
                raise AgentMemoryUnavailableError(
                    'durable backend contains multiple canonical prompt records'
                )
            if records:
                record = records[0]
                self._snapshot = _record_snapshot(record)
                if state is not None and state.source == 'file':
                    file_state = state
                    if (
                        state.mirror_synced
                        and legacy_text is not None
                        and not _prompt_matches(legacy_text, state.body)
                    ):
                        file_state = self._synced_compatibility_state(
                            source='file',
                            body=_normalized_prompt(legacy_text),
                            core_revision=state.core_revision,
                        )
                        if file_state is None or not self._write_compatibility_state(
                            file_state
                        ):
                            raise _CompatibilityConflictError(
                                'latest file-backed memory could not be journaled'
                            )
                        publication_signature = (
                            self._path_signature(self._state_path),
                            publication_signature[1],
                        )
                    elif (
                        not state.mirror_synced
                        and not _prompt_matches(legacy_text, state.body)
                        and not self._legacy_matches_observation(state)
                    ):
                        raise _CompatibilityConflictError(
                            'file-backed memory has an ambiguous unpublished edit'
                        )
                    if record.body != file_state.body:
                        if (
                            file_state.core_revision is None
                            or file_state.core_revision != record.revision
                        ):
                            raise _CompatibilityConflictError(
                                'compatibility update conflicts with the durable revision'
                            )
                        record = self._update_record(
                            store,
                            record,
                            file_state.body,
                            actor_key='system:compatibility_sync',
                            reason='compatibility_sync',
                        )
                else:
                    publication_signature = self._handoff_signature()
                    self._assert_compatibility_safe_for_core_write(record)
            else:
                initial = self._bootstrap_text(state, legacy_text)
                publication_signature = (
                    self._path_signature(self._state_path),
                    publication_signature[1],
                )
                record = store.create(
                    context,
                    MemoryPlace.private(),
                    MemoryDraft(
                        title=_PROMPT_TITLE,
                        body=initial,
                        kind=_PROMPT_KIND,
                        tags=('agent', 'prompt'),
                        metadata={'format': 'prompt_memory', 'version': 1},
                    ),
                    op_key=_BOOTSTRAP_OPERATION,
                ).record
        except _CompatibilityConflictError as error:
            return self._degrade(error, prefer_legacy=True)
        except (AgentMemoryError, MemoryCoreError, OSError, UnicodeError) as error:
            return self._degrade(error)

        self._store = store
        ready = self._publish_core_if_handoff_unchanged(
            record,
            publication_signature,
        )
        snapshot = self._set_core_record(record, fallback_ready=ready)
        # Publish the backend last so lock-free readers cannot observe a
        # partially initialized Core state.
        self._backend = 'core'
        return snapshot

    def _refresh_snapshot_worker(self) -> None:
        try:
            try:
                changed = (
                    self._current_observable_signature() != self._observable_signature
                )
            except OSError:
                changed = True
            if not changed:
                return
            with self._lock:
                try:
                    if (
                        self._current_observable_signature()
                        == self._observable_signature
                    ):
                        return
                except OSError:
                    pass
                if self._backend == 'core':
                    with self._handoff_lock():
                        self._refresh_core_snapshot_under_handoff()
                elif self._backend == 'file':
                    with self._handoff_lock():
                        candidate = self._file_backend_snapshot(
                            backend='file',
                        )
                        if candidate.backend != 'degraded':
                            self._finalize_file_snapshot(candidate)
        except _HandoffLockTimeout:
            # Contention is transient; preserve the complete last snapshot and
            # retry on a later poll instead of permanently disabling writes.
            self._observable_signature = None
        except (AgentMemoryError, MemoryCoreError, OSError, UnicodeError) as error:
            with self._lock:
                self._degrade(error)
        finally:
            with self._refresh_guard:
                self._refresh_running = False

    def _schedule_snapshot_refresh(self) -> None:
        with self._refresh_guard:
            now = time.monotonic()
            if self._refresh_running or now < self._next_refresh_check:
                return
            self._refresh_running = True
            self._next_refresh_check = now + _REFRESH_POLL_SECONDS
        threading.Thread(
            target=self._refresh_snapshot_worker,
            name='agent-memory-refresh',
            daemon=True,
        ).start()

    def snapshot(self) -> MemorySnapshot:
        # Reads never wait behind SQLite, fsync, or the cross-process handoff
        # lock.  Observable changes schedule one background reconciliation;
        # this call returns the last complete immutable snapshot immediately.
        if not self._initialized or self._backend == 'uninitialized':
            with self._lock:
                if not self._initialized:
                    self.initialize()
        if self._backend in {'core', 'file'} and not self._file_snapshot_pinned:
            self._schedule_snapshot_refresh()
        return self._snapshot

    def replace(self, text: str, *, actor_key: str, reason: str) -> MemorySnapshot:
        """Atomically replace the canonical prompt and then refresh its mirror."""

        normalized = _normalized_prompt(text)
        with self._lock:
            if not self._initialized:
                self.initialize()
            completed_snapshot: MemorySnapshot | None = None
            body_error: AgentMemoryError | None = None
            try:
                with self._handoff_lock():
                    try:
                        completed_snapshot = self._replace_under_handoff(
                            normalized,
                            actor_key=actor_key,
                            reason=reason,
                        )
                    except AgentMemoryError as error:
                        # Preserve the semantic write outcome even if lock
                        # cleanup subsequently loses its acknowledgement.
                        body_error = error
                if body_error is not None:
                    raise body_error
                return completed_snapshot
            except AgentMemoryError:
                raise
            except UnicodeError as error:
                self._degrade(error)
                raise AgentMemoryUnavailableError(
                    'long-term memory compatibility data is invalid; existing memory was kept'
                ) from error
            except OSError as error:
                if body_error is not None:
                    raise body_error from error
                if completed_snapshot is not None:
                    print(
                        '[memory] hand-off lock cleanup failed after publication: '
                        f'{type(error).__name__}'
                    )
                    return completed_snapshot
                raise AgentMemoryUnavailableError(
                    'long-term memory hand-off lock is unavailable; existing memory was kept'
                ) from error

    def _replace_under_handoff(
        self,
        normalized: str,
        *,
        actor_key: str,
        reason: str,
    ) -> MemorySnapshot:
        if self._backend == 'file':
            # Another process may have advanced Core since this disabled
            # instance last observed it.  Reconcile inside the shared handoff
            # lock before journaling a new file-backed generation.
            try:
                latest = self._file_backend_snapshot(
                    backend='file',
                )
            except (AgentMemoryError, OSError, UnicodeError) as error:
                self._degrade(error)
                raise AgentMemoryUnavailableError(
                    'long-term memory could not reconcile the latest generation'
                ) from error
            if latest.backend == 'degraded':
                self._backend = 'degraded'
                raise AgentMemoryUnavailableError(
                    'long-term memory generations conflict; existing memory was kept'
                )

            try:
                previous_observation = self._stable_legacy_observation(
                    self._file_mirror_signature
                )
            except (OSError, UnicodeError) as error:
                raise AgentMemoryUnavailableError(
                    'long-term memory file changed while preparing the update'
                ) from error
            if previous_observation is None:
                raise AgentMemoryUnavailableError(
                    'long-term memory file changed while preparing the update'
                )
            previous_signature, previous_digest, _ = previous_observation
            unsynced = _CompatibilityState(
                source='file',
                body=normalized,
                core_revision=self._last_core_revision,
                mirror_synced=False,
                mirror_digest=previous_digest,
                mirror_signature=previous_signature,
            )
            if not self._write_compatibility_state(unsynced):
                raise AgentMemoryUnavailableError(
                    'long-term memory file could not be updated'
                )
            self._file_snapshot_pinned = False
            mirrored = self._mirror(
                normalized,
                expected_signature=self._file_mirror_signature,
            )
            if self._compatibility_durability_uncertain:
                mirror_became_visible = self._mirror_postcommit_visible
                if not self._confirm_compatibility_directory():
                    self._compatibility_conflict_snapshot(normalized)
                    raise AgentMemoryCommitUncertainError(
                        'file-backed long-term memory commit is unconfirmed'
                    )
                if mirror_became_visible:
                    mirrored = True
            state_synced = False
            if mirrored:
                state_synced = self._write_synced_compatibility_state(
                    source='file',
                    body=normalized,
                    core_revision=self._last_core_revision,
                )
            self._snapshot = _text_snapshot(
                normalized,
                backend='file',
                fallback_ready=(
                    mirrored
                    and state_synced
                    and not self._compatibility_durability_uncertain
                ),
            )
            if self._mirror_precondition_failed:
                self._observable_signature = None
            else:
                self._capture_observable_signature()
            return self._snapshot

        if self._backend != 'core' or self._store is None or self._record is None:
            raise AgentMemoryUnavailableError(
                'durable long-term memory is unavailable; existing memory was kept'
            )

        cached = self._record
        try:
            current = self._store.read(
                self._context('system:write_refresh'),
                cached.memory_id,
            )
        except (
            StorageDamagedError,
            MigrationError,
            UnsupportedSchemaVersionError,
        ) as error:
            self._degrade(error)
            raise AgentMemoryUnavailableError(
                'durable long-term memory became unavailable; existing memory was kept'
            ) from error
        except MemoryCoreError as error:
            raise AgentMemoryUnavailableError(
                'long-term memory could not be refreshed; existing memory was kept'
            ) from error

        handoff_signature = self._handoff_signature()
        repairable = self._repairable_core_publication
        if (
            repairable is not None
            and repairable == (current.revision, current.body, handoff_signature)
            and current.body == normalized
        ):
            ready = self._publish_core_if_handoff_unchanged(
                current,
                handoff_signature,
            )
            return self._set_core_record(current, fallback_ready=ready)
        try:
            self._assert_compatibility_safe_for_core_write(current)
        except _CompatibilityConflictError as error:
            self._degrade_for_compatibility_conflict(error)
            raise AgentMemoryUnavailableError(
                'long-term memory modes conflict; existing memory was kept'
            ) from error

        if current.revision < cached.revision or (
            current.revision == cached.revision and current.body != cached.body
        ):
            self._degrade(
                AgentMemoryUnavailableError(
                    'durable long-term memory moved to an incompatible revision'
                )
            )
            raise AgentMemoryUnavailableError(
                'long-term memory revision requires manual reconciliation'
            )

        if current.revision > cached.revision:
            ready = self._publish_core_if_handoff_unchanged(
                current,
                handoff_signature,
            )
            self._set_core_record(current, fallback_ready=ready)
            if current.body == normalized:
                return self._snapshot
            raise AgentMemoryUnavailableError(
                'long-term memory changed concurrently; retry the update'
            )

        if current.body == normalized:
            ready = self._publish_core_if_handoff_unchanged(
                current,
                handoff_signature,
            )
            return self._set_core_record(current, fallback_ready=ready)

        try:
            record = self._update_record(
                self._store,
                current,
                normalized,
                actor_key=actor_key,
                reason=reason,
            )
        except RevisionConflictError as error:
            try:
                latest = self._store.read(
                    self._context('system:conflict_refresh'),
                    current.memory_id,
                )
            except (
                StorageDamagedError,
                MigrationError,
                UnsupportedSchemaVersionError,
            ) as refresh_error:
                self._degrade(refresh_error)
                raise AgentMemoryUnavailableError(
                    'durable long-term memory became unavailable; existing memory was kept'
                ) from refresh_error
            except MemoryCoreError as refresh_error:
                raise AgentMemoryUnavailableError(
                    'long-term memory changed concurrently and could not be refreshed'
                ) from refresh_error
            ready = self._publish_core_if_handoff_unchanged(
                latest,
                handoff_signature,
            )
            self._set_core_record(latest, fallback_ready=ready)
            raise AgentMemoryUnavailableError(
                'long-term memory changed concurrently; retry the update'
            ) from error
        except (
            StorageDamagedError,
            MigrationError,
            UnsupportedSchemaVersionError,
        ) as error:
            self._degrade(error)
            raise AgentMemoryUnavailableError(
                'durable long-term memory became unavailable; existing memory was kept'
            ) from error
        except MemoryCoreError as error:
            raise AgentMemoryUnavailableError(
                'long-term memory could not be updated; existing memory was kept'
            ) from error

        ready = self._publish_core_if_handoff_unchanged(
            record,
            handoff_signature,
        )
        return self._set_core_record(record, fallback_ready=ready)

    def status(
        self,
        snapshot: MemorySnapshot | None = None,
    ) -> dict[str, object]:
        if snapshot is None:
            snapshot = self.snapshot()
        warning = None
        if snapshot.backend == 'degraded':
            warning = DEGRADED_WARNING
        elif not snapshot.fallback_ready:
            warning = COMPATIBILITY_WARNING
        return {
            'backend': snapshot.backend,
            'revision': snapshot.revision,
            'enabled': self._enabled,
            'fallback_ready': snapshot.fallback_ready,
            'warning': warning,
        }


def _enabled_from_environment() -> bool:
    raw = os.environ.get('MEMORY_CORE_ENABLED', '1').strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise AgentMemoryValidationError(
        'MEMORY_CORE_ENABLED must be one of 1/0, true/false, yes/no, or on/off'
    )


def _legacy_memory_path() -> pathlib.Path:
    import config

    configured = (
        config.main.get('event', {})
        .get('llm', {})
        .get('prompt_memory', './resource/memory/prompt_memory.md')
    )
    return pathlib.Path(configured)


_service: AgentMemory | None = None
_service_lock = threading.Lock()


def _get_service() -> AgentMemory:
    global _service
    with _service_lock:
        if _service is None:
            _service = AgentMemory(
                enabled=_enabled_from_environment(),
                database_path=os.environ.get('MEMORY_DB_PATH', './resource/memory.db'),
                legacy_path=_legacy_memory_path(),
            )
        return _service


def initialize(*, seed_text: str | None = None) -> MemorySnapshot:
    return _get_service().initialize(seed_text=seed_text)


def seed_legacy_if_cold(text: str) -> bool:
    return _get_service().seed_legacy_if_cold(text)


def snapshot() -> MemorySnapshot:
    return _get_service().snapshot()


async def replace(text: str, *, actor_key: str, reason: str) -> MemorySnapshot:
    return await asyncio.to_thread(
        _get_service().replace,
        text,
        actor_key=actor_key,
        reason=reason,
    )


def status(snapshot: MemorySnapshot | None = None) -> dict[str, object]:
    return _get_service().status(snapshot)


__all__ = [
    'COMPATIBILITY_WARNING',
    'DEGRADED_WARNING',
    'AgentMemory',
    'AgentMemoryCommitUncertainError',
    'AgentMemoryError',
    'AgentMemoryUnavailableError',
    'AgentMemoryValidationError',
    'MemorySnapshot',
    'initialize',
    'replace',
    'seed_legacy_if_cold',
    'snapshot',
    'status',
]
