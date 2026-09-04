import asyncio
import concurrent.futures
import contextlib
import json
import pathlib
import sqlite3
import stat
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from memory_core import (
    AccessContext,
    ListQuery,
    MemoryStore,
    StorageBusyError,
    StorageDamagedError,
)

import agent_memory


def _runtime(tmp_path, *, enabled=True):
    legacy = tmp_path / 'prompt_memory.md'
    legacy.write_text('initial memory', encoding='utf-8')
    runtime = agent_memory.AgentMemory(
        enabled=enabled,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
    )
    return runtime, legacy


def test_database_and_compatibility_paths_must_differ(tmp_path):
    path = tmp_path / 'memory'

    with pytest.raises(agent_memory.AgentMemoryValidationError):
        agent_memory.AgentMemory(
            enabled=True,
            database_path=path,
            legacy_path=path,
        )


def test_database_and_compatibility_symlink_alias_degrades_without_touching_target(
    tmp_path,
):
    database = tmp_path / 'memory.db'
    alias = tmp_path / 'prompt_memory.md'
    alias.symlink_to(database)

    runtime = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=alias,
    )

    snapshot = runtime.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == ''
    assert snapshot.fallback_ready is False
    assert database.exists() is False
    assert alias.is_symlink()


def test_compatibility_file_symlink_is_rejected_without_replacing_target(tmp_path):
    target = tmp_path / 'operator-memory.md'
    target.write_text('operator memory', encoding='utf-8')
    legacy = tmp_path / 'prompt_memory.md'
    legacy.symlink_to(target)

    runtime = agent_memory.AgentMemory(
        enabled=False,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
    )

    snapshot = runtime.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == ''
    assert snapshot.fallback_ready is False
    with pytest.raises(OSError):
        agent_memory._atomic_write(legacy, 'must not replace the link')

    assert legacy.is_symlink()
    assert target.read_text(encoding='utf-8') == 'operator memory'


def _rewrite_state(runtime, **changes):
    state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    state.update(changes)
    state.pop('checksum', None)
    state['checksum'] = agent_memory._compatibility_state_checksum(state)
    runtime._state_path.write_text(json.dumps(state), encoding='utf-8')


def _eventually_snapshot(runtime, predicate, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    snapshot = runtime.snapshot()
    while not predicate(snapshot) and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = runtime.snapshot()
    return snapshot


def test_atomic_compatibility_write_syncs_file_and_parent_directory(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / 'prompt_memory.md'
    real_fsync = agent_memory.os.fsync
    synced_modes = []

    def observe_fsync(file_descriptor):
        synced_modes.append(agent_memory.os.fstat(file_descriptor).st_mode)
        real_fsync(file_descriptor)

    monkeypatch.setattr(agent_memory.os, 'fsync', observe_fsync)

    agent_memory._atomic_write(target, 'durable memory')

    assert target.read_text(encoding='utf-8') == 'durable memory'
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert len(synced_modes) == 2
    assert stat.S_ISREG(synced_modes[0])
    assert stat.S_ISDIR(synced_modes[1])


def test_atomic_compatibility_write_preserves_existing_metadata(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / 'prompt_memory.md'
    target.write_text('old memory', encoding='utf-8')
    target.chmod(0o640)
    before = target.stat()
    real_fchown = agent_memory.os.fchown
    ownership_calls = []

    def observe_fchown(file_descriptor, uid, gid):
        ownership_calls.append((uid, gid))
        real_fchown(file_descriptor, uid, gid)

    monkeypatch.setattr(agent_memory.os, 'fchown', observe_fchown)

    agent_memory._atomic_write(target, 'replacement memory')

    after = target.stat()
    assert target.read_text(encoding='utf-8') == 'replacement memory'
    assert ownership_calls == [(before.st_uid, before.st_gid)]
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert stat.S_IMODE(after.st_mode) == 0o640


def test_new_compatibility_state_is_owner_only(tmp_path):
    target = tmp_path / '.prompt_memory.md.state.json'

    agent_memory._atomic_write(target, '{}')

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_file_write_reports_visible_intent_when_directory_sync_fails(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    original = runtime.initialize()
    real_fsync = agent_memory.os.fsync

    def fail_directory_sync(file_descriptor):
        mode = agent_memory.os.fstat(file_descriptor).st_mode
        if stat.S_ISDIR(mode):
            raise OSError('directory sync unavailable')
        real_fsync(file_descriptor)

    monkeypatch.setattr(agent_memory.os, 'fsync', fail_directory_sync)

    with pytest.raises(agent_memory.AgentMemoryCommitUncertainError):
        runtime.replace(
            'visible despite uncertain directory sync',
            actor_key='agent:main',
            reason='llm_update',
        )

    state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    snapshot = runtime._snapshot
    assert snapshot.backend == 'degraded'
    assert snapshot.text == 'visible despite uncertain directory sync'
    assert snapshot.fallback_ready is False
    assert snapshot != original
    assert state['body'] == snapshot.text
    assert state['mirror_synced'] is False
    assert legacy.read_text(encoding='utf-8') == snapshot.text


def test_uncertain_file_commit_survives_a_second_lock_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    runtime, _ = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    real_fsync = agent_memory.os.fsync
    real_handoff = runtime._handoff_lock

    def fail_directory_sync(file_descriptor):
        mode = agent_memory.os.fstat(file_descriptor).st_mode
        if stat.S_ISDIR(mode):
            raise OSError('directory sync unavailable')
        real_fsync(file_descriptor)

    @contextlib.contextmanager
    def fail_cleanup_too():
        try:
            with real_handoff():
                yield
        finally:
            raise OSError('unlock acknowledgement lost')

    monkeypatch.setattr(agent_memory.os, 'fsync', fail_directory_sync)
    monkeypatch.setattr(runtime, '_handoff_lock', fail_cleanup_too)

    with pytest.raises(agent_memory.AgentMemoryCommitUncertainError) as raised:
        runtime.replace(
            'uncertain value with cleanup failure',
            actor_key='agent:main',
            reason='llm_update',
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert runtime._snapshot.backend == 'degraded'
    assert runtime._snapshot.text == 'uncertain value with cleanup failure'


def test_file_write_confirms_visible_renames_before_reporting_success(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    real_fsync = agent_memory.os.fsync
    remaining_directory_failures = 2

    def fail_first_directory_syncs(file_descriptor):
        nonlocal remaining_directory_failures
        mode = agent_memory.os.fstat(file_descriptor).st_mode
        if stat.S_ISDIR(mode) and remaining_directory_failures:
            remaining_directory_failures -= 1
            raise OSError('transient directory sync failure')
        real_fsync(file_descriptor)

    monkeypatch.setattr(agent_memory.os, 'fsync', fail_first_directory_syncs)

    updated = runtime.replace(
        'confirmed after explicit directory sync',
        actor_key='agent:main',
        reason='llm_update',
    )

    state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    assert remaining_directory_failures == 0
    assert updated.backend == 'file'
    assert updated.text == 'confirmed after explicit directory sync'
    assert updated.fallback_ready is True
    assert state['mirror_synced'] is True
    assert legacy.read_text(encoding='utf-8') == updated.text


def test_bootstrap_update_and_restart_use_one_canonical_record(tmp_path):
    runtime, legacy = _runtime(tmp_path)

    first = runtime.initialize()
    assert first.backend == 'core'
    assert first.text == 'initial memory'
    assert first.revision == 1

    second = runtime.replace(
        'remember the charging station',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert second.revision == 2
    assert second.text == 'remember the charging station'
    assert legacy.read_text(encoding='utf-8') == second.text

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
    )
    recovered = restarted.initialize()

    assert recovered.text == second.text
    assert recovered.revision == 2
    assert legacy.read_text(encoding='utf-8') == second.text

    store = MemoryStore.open(tmp_path / 'memory.db')
    records = store.list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )
    assert len(records) == 1
    assert records[0].owner_key == 'agent:main'
    changes = store.changes(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        records[0].memory_id,
    )
    assert [(change.actor_key, change.reason) for change in changes] == [
        ('system:bootstrap', ''),
        ('agent:main', 'llm_update'),
    ]


def test_replace_returns_committed_snapshot_when_lock_cleanup_fails(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    real_handoff = runtime._handoff_lock

    @contextlib.contextmanager
    def fail_after_release():
        with real_handoff():
            yield
        raise OSError('unlock acknowledgement lost')

    monkeypatch.setattr(runtime, '_handoff_lock', fail_after_release)

    updated = runtime.replace(
        'committed before cleanup failure',
        actor_key='agent:main',
        reason='llm_update',
    )

    assert updated.revision == original.revision + 1
    assert updated.text == 'committed before cleanup failure'
    assert updated.fallback_ready is True
    assert runtime.snapshot() == updated
    assert legacy.read_text(encoding='utf-8') == updated.text
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == updated.text


def test_legacy_formatting_whitespace_does_not_dirty_or_conflict_on_restart(tmp_path):
    legacy = tmp_path / 'prompt_memory.md'
    legacy.write_text('  initial memory\r\n', encoding='utf-8')
    database = tmp_path / 'memory.db'

    first = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=legacy,
    ).initialize()
    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=legacy,
    ).initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=database,
        legacy_path=legacy,
    ).initialize()

    assert first.backend == 'core'
    assert first.text == 'initial memory'
    assert restarted.backend == 'core'
    assert disabled.backend == 'file'
    assert legacy.read_bytes() == b'  initial memory\r\n'


def test_same_content_is_a_noop_without_revision_churn(tmp_path):
    runtime, _ = _runtime(tmp_path)
    original = runtime.initialize()

    replay = runtime.replace(
        original.text,
        actor_key='api:agent_definition',
        reason='api_edit',
    )

    assert replay == original


def test_disabled_mode_keeps_file_backend_and_creates_no_database(tmp_path):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    database = runtime.database_path

    assert runtime.initialize().backend == 'file'
    updated = runtime.replace(
        'file-only memory',
        actor_key='agent:main',
        reason='llm_update',
    )

    assert updated.backend == 'file'
    assert updated.revision is None
    assert legacy.read_text(encoding='utf-8') == 'file-only memory'
    state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    assert state['source'] == 'file'
    assert state['body'] == 'file-only memory'
    assert state['core_revision'] is None
    assert not database.exists()


def test_empty_disabled_cold_start_is_degraded_and_creates_no_empty_mirror(tmp_path):
    legacy = tmp_path / 'prompt_memory.md'
    runtime = agent_memory.AgentMemory(
        enabled=False,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
    )

    snapshot = runtime.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == ''
    assert snapshot.fallback_ready is False
    assert legacy.exists() is False
    assert runtime.status()['warning'] == agent_memory.DEGRADED_WARNING


def test_disabled_mode_uses_legacy_lock_when_database_directory_is_unavailable(
    tmp_path,
):
    blocked_parent = tmp_path / 'database-parent'
    blocked_parent.write_text('not a directory', encoding='utf-8')
    legacy_directory = tmp_path / 'legacy'
    legacy_directory.mkdir()
    legacy = legacy_directory / 'prompt_memory.md'
    legacy.write_text('initial memory', encoding='utf-8')
    runtime = agent_memory.AgentMemory(
        enabled=False,
        database_path=blocked_parent / 'memory.db',
        legacy_path=legacy,
    )

    assert runtime.initialize().backend == 'file'
    updated = runtime.replace(
        'file-only recovery',
        actor_key='api:agent_definition',
        reason='api_edit',
    )

    assert updated.backend == 'file'
    assert updated.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == 'file-only recovery'
    assert (legacy_directory / '.prompt_memory.md.handoff.lock').is_file()


def test_disabled_fallback_lock_still_blocks_enabled_writer(tmp_path, monkeypatch):
    enabled, legacy = _runtime(tmp_path)
    enabled.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=enabled.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    database_lock = enabled.database_path.with_name(
        f'.{enabled.database_path.name}.handoff.lock'
    )
    real_lock = disabled._exclusive_file_lock

    @contextlib.contextmanager
    def fail_disabled_database_lock(path, **kwargs):
        if path == database_lock:
            raise OSError('database lock unavailable to disabled process')
        with real_lock(path, **kwargs):
            yield

    monkeypatch.setattr(disabled, '_exclusive_file_lock', fail_disabled_database_lock)
    fallback_entered = threading.Event()
    release_fallback = threading.Event()
    writer_finished = threading.Event()

    def hold_fallback_lock():
        with disabled._handoff_lock():
            fallback_entered.set()
            assert release_fallback.wait(timeout=2)

    def write_core():
        enabled.replace(
            'written after fallback lock',
            actor_key='api:agent_definition',
            reason='api_edit',
        )
        writer_finished.set()

    holder = threading.Thread(target=hold_fallback_lock)
    holder.start()
    assert fallback_entered.wait(timeout=2)
    writer = threading.Thread(target=write_core)
    writer.start()

    assert writer_finished.wait(timeout=0.1) is False
    release_fallback.set()
    holder.join(timeout=2)
    writer.join(timeout=2)

    assert holder.is_alive() is False
    assert writer.is_alive() is False
    assert writer_finished.is_set()
    assert enabled.snapshot().text == 'written after fallback lock'


def test_handoff_lock_contention_has_a_bounded_startup_failure(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    contender = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    database_lock = runtime.database_path.with_name(
        f'.{runtime.database_path.name}.handoff.lock'
    )
    entered = threading.Event()
    release = threading.Event()

    def hold_database_lock():
        with runtime._exclusive_file_lock(database_lock):
            entered.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(agent_memory, '_HANDOFF_LOCK_TIMEOUT_SECONDS', 0.05)
    holder = threading.Thread(target=hold_database_lock)
    holder.start()
    assert entered.wait(timeout=2)
    started = time.monotonic()
    try:
        snapshot = contender.initialize()
    finally:
        release.set()
        holder.join(timeout=2)

    assert time.monotonic() - started < 0.5
    assert snapshot.backend == 'degraded'
    assert contender.status()['warning'] == agent_memory.DEGRADED_WARNING


def test_cross_process_handoff_lock_has_a_bounded_startup_failure(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    contender = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    database_lock = runtime.database_path.with_name(
        f'.{runtime.database_path.name}.handoff.lock'
    )
    script = (
        'import fcntl, pathlib, sys\n'
        'path = pathlib.Path(sys.argv[1])\n'
        'path.parent.mkdir(parents=True, exist_ok=True)\n'
        "with path.open('a+b') as handle:\n"
        '    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n'
        "    print('locked', flush=True)\n"
        '    sys.stdin.readline()\n'
    )
    process = subprocess.Popen(
        [sys.executable, '-c', script, str(database_lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monkeypatch.setattr(agent_memory, '_HANDOFF_LOCK_TIMEOUT_SECONDS', 0.05)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == 'locked'
        started = time.monotonic()
        snapshot = contender.initialize()
        assert time.monotonic() - started < 0.5
    finally:
        process.communicate(input='release\n', timeout=2)

    assert process.returncode == 0
    assert snapshot.backend == 'degraded'


def test_background_handoff_timeout_is_transient(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    database_lock = runtime.database_path.with_name(
        f'.{runtime.database_path.name}.handoff.lock'
    )
    entered = threading.Event()
    release = threading.Event()

    def hold_database_lock():
        with runtime._exclusive_file_lock(database_lock):
            entered.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(agent_memory, '_HANDOFF_LOCK_TIMEOUT_SECONDS', 0.05)
    legacy.write_text(f' {original.text} ', encoding='utf-8')
    holder = threading.Thread(target=hold_database_lock)
    holder.start()
    assert entered.wait(timeout=2)
    try:
        runtime.snapshot()
        deadline = time.monotonic() + 1
        while runtime._refresh_running and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        release.set()
        holder.join(timeout=2)

    assert holder.is_alive() is False
    assert runtime._refresh_running is False
    assert runtime._snapshot.backend == 'core'
    assert runtime._snapshot == original

    updated = runtime.replace(
        'update after transient lock contention',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.backend == 'core'
    assert updated.text == 'update after transient lock contention'


def test_disabled_mode_ignores_unavailable_database_probe(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    database_lock = runtime.database_path.with_name(
        f'.{runtime.database_path.name}.handoff.lock'
    )
    real_lock = runtime._exclusive_file_lock
    real_stat = pathlib.Path.stat

    @contextlib.contextmanager
    def fail_database_lock(path, **kwargs):
        if path == database_lock:
            raise PermissionError('database directory unavailable')
        with real_lock(path, **kwargs):
            yield

    def fail_database_probe(path, *args, **kwargs):
        if path == runtime.database_path:
            raise PermissionError('database path unavailable')
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(runtime, '_exclusive_file_lock', fail_database_lock)
    monkeypatch.setattr(pathlib.Path, 'stat', fail_database_probe)

    assert runtime.initialize(seed_text='factory memory').backend == 'file'
    updated = runtime.replace(
        'file survives unavailable database path',
        actor_key='api:agent_definition',
        reason='api_edit',
    )

    assert updated.backend == 'file'
    assert updated.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == updated.text


def test_cold_seed_is_blocked_by_recovery_state_or_database(tmp_path):
    database = tmp_path / 'memory.db'
    legacy = tmp_path / 'prompt_memory.md'
    state = tmp_path / '.prompt_memory.md.state.json'
    state.write_text('{}', encoding='utf-8')
    with_state = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=legacy,
    )

    assert with_state.seed_legacy_if_cold('must not overwrite recovery') is False
    assert legacy.exists() is False

    state.unlink()
    database.touch()
    with_database = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=legacy,
    )
    assert with_database.seed_legacy_if_cold('must not overwrite database') is False
    assert legacy.exists() is False


def test_default_seed_probe_error_fails_closed(tmp_path, monkeypatch):
    database = tmp_path / 'memory.db'
    legacy = tmp_path / 'prompt_memory.md'
    runtime = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=legacy,
    )
    real_stat = pathlib.Path.stat

    def fail_database_probe(path, *args, **kwargs):
        if path == database:
            raise PermissionError('database stat unavailable')
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, 'stat', fail_database_probe)

    assert runtime.seed_legacy_if_cold('must not seed') is False
    assert legacy.exists() is False


def test_cold_start_seed_is_serialized_across_instances(tmp_path):
    database = tmp_path / 'memory.db'
    legacy = tmp_path / 'prompt_memory.md'
    runtimes = [
        agent_memory.AgentMemory(
            enabled=True,
            database_path=database,
            legacy_path=legacy,
        )
        for _ in range(2)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda runtime: runtime.seed_legacy_if_cold('seed'), runtimes)
        )

    assert sorted(results) == [False, True]
    assert legacy.read_text(encoding='utf-8') == 'seed'


def test_initialize_seeds_and_bootstraps_under_one_handoff_lock(
    tmp_path,
    monkeypatch,
):
    legacy = tmp_path / 'prompt_memory.md'
    runtime = agent_memory.AgentMemory(
        enabled=True,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
    )
    real_handoff = runtime._handoff_lock
    entries = 0

    @contextlib.contextmanager
    def observed_handoff():
        nonlocal entries
        entries += 1
        with real_handoff():
            yield

    monkeypatch.setattr(runtime, '_handoff_lock', observed_handoff)

    snapshot = runtime.initialize(seed_text='factory memory')

    assert entries == 1
    assert snapshot.backend == 'core'
    assert snapshot.text == 'factory memory'
    assert legacy.read_text(encoding='utf-8') == 'factory memory'


def test_initialize_returns_completed_snapshot_when_lock_cleanup_fails(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    real_handoff = runtime._handoff_lock

    @contextlib.contextmanager
    def fail_after_release():
        with real_handoff():
            yield
        raise OSError('unlock acknowledgement lost')

    monkeypatch.setattr(runtime, '_handoff_lock', fail_after_release)

    snapshot = runtime.initialize()

    assert snapshot.backend == 'core'
    assert snapshot.text == 'initial memory'
    assert snapshot.fallback_ready is True
    assert runtime.snapshot() == snapshot
    assert legacy.read_text(encoding='utf-8') == snapshot.text


def test_cold_seed_lock_timeout_never_blocks_startup(tmp_path, monkeypatch):
    database = tmp_path / 'memory.db'
    legacy = tmp_path / 'prompt_memory.md'
    runtime = agent_memory.AgentMemory(
        enabled=True,
        database_path=database,
        legacy_path=legacy,
    )
    legacy_lock = legacy.with_name(f'.{legacy.name}.handoff.lock')
    entered = threading.Event()
    release = threading.Event()

    def hold_legacy_lock():
        with runtime._exclusive_file_lock(legacy_lock):
            entered.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(agent_memory, '_HANDOFF_LOCK_TIMEOUT_SECONDS', 0.05)
    holder = threading.Thread(target=hold_legacy_lock)
    holder.start()
    assert entered.wait(timeout=2)
    started = time.monotonic()
    try:
        assert runtime.seed_legacy_if_cold('seed') is False
    finally:
        release.set()
        holder.join(timeout=2)

    assert time.monotonic() - started < 0.5
    assert holder.is_alive() is False
    assert legacy.exists() is False
    assert runtime.seed_legacy_if_cold('seed') is True
    assert legacy.read_text(encoding='utf-8') == 'seed'


def test_invalid_cold_seed_never_prevents_later_initialization(tmp_path):
    legacy = tmp_path / 'prompt_memory.md'
    runtime = agent_memory.AgentMemory(
        enabled=True,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
    )

    assert runtime.seed_legacy_if_cold('') is False

    legacy.write_text('valid fallback', encoding='utf-8')
    assert runtime.initialize().text == 'valid fallback'


def test_disabled_mode_does_not_initialize_or_migrate_database(tmp_path):
    database = tmp_path / 'memory.db'
    connection = sqlite3.connect(database)
    connection.close()
    before = database.read_bytes()
    legacy = tmp_path / 'prompt_memory.md'
    legacy.write_text('rollback memory', encoding='utf-8')
    runtime = agent_memory.AgentMemory(
        enabled=False,
        database_path=database,
        legacy_path=legacy,
    )

    snapshot = runtime.initialize()

    assert snapshot.backend == 'file'
    assert database.read_bytes() == before
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    try:
        objects = connection.execute(
            "SELECT name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        application_id = connection.execute('PRAGMA application_id').fetchone()[0]
    finally:
        connection.close()
    assert objects == []
    assert application_id == 0
    assert not pathlib.Path(f'{database}-wal').exists()
    assert not pathlib.Path(f'{database}-shm').exists()


def test_disabled_seed_does_not_mask_missing_compatibility_for_existing_database(
    tmp_path,
):
    enabled, legacy = _runtime(tmp_path)
    enabled.initialize()
    durable = enabled.replace(
        'durable user memory',
        actor_key='agent:main',
        reason='llm_update',
    )
    enabled._state_path.unlink()
    legacy.unlink()

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=enabled.database_path,
        legacy_path=legacy,
    )
    snapshot = disabled.initialize(seed_text='factory memory')

    assert snapshot.backend == 'degraded'
    assert snapshot.text == ''
    assert snapshot.fallback_ready is False
    assert legacy.exists() is False
    stored = MemoryStore.open(enabled.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == durable.text


def test_disabled_file_write_reports_failed_compatibility_mirror(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    snapshot = runtime.replace(
        'journaled but not mirrored',
        actor_key='agent:main',
        reason='llm_update',
    )

    assert snapshot.backend == 'file'
    assert snapshot.text == 'journaled but not mirrored'
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'initial memory'


def test_file_replace_publishes_one_complete_snapshot(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    original = runtime.initialize()
    real_mirror = runtime._mirror
    entered = threading.Event()
    release = threading.Event()

    def slow_mirror(text, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_mirror(text, **kwargs)

    monkeypatch.setattr(runtime, '_mirror', slow_mirror)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(
            runtime.replace,
            'new complete file snapshot',
            actor_key='agent:main',
            reason='llm_update',
        )
        assert entered.wait(timeout=2)
        during = runtime.snapshot()
        state_during = json.loads(runtime._state_path.read_text(encoding='utf-8'))
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                writing.result(timeout=0.1)
        finally:
            release.set()
        final = writing.result(timeout=2)

    assert during == original
    assert state_during['mirror_synced'] is False
    assert legacy.read_text(encoding='utf-8') == final.text
    assert final.text == 'new complete file snapshot'
    assert final.fallback_ready is True
    assert runtime.snapshot() == final


def test_disabled_restart_repairs_interrupted_file_mirror_with_proven_predecessor(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    updated = runtime.replace(
        'journaled but not mirrored',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'initial memory'

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    restarted = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    recovered = restarted.initialize()

    assert recovered.backend == 'file'
    assert recovered.text == updated.text
    assert recovered.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == updated.text
    repaired_state = json.loads(restarted._state_path.read_text(encoding='utf-8'))
    assert repaired_state['source'] == 'file'
    assert repaired_state['mirror_synced'] is True
    assert len(repaired_state['mirror_digest']) == 64


def test_disabled_restart_reports_failed_missing_mirror_recovery(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    expected = runtime.replace(
        'journaled file memory',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    legacy.unlink()
    restarted = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    recovered = restarted.initialize()

    assert recovered.backend == 'file'
    assert recovered.text == expected.text
    assert recovered.fallback_ready is False
    assert legacy.exists() is False


def test_unsynced_file_state_never_overwrites_a_third_legacy_value(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    pending = runtime.replace(
        'journaled pending value',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    assert pending.fallback_ready is False
    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    legacy.write_text('third operator value', encoding='utf-8')

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    ).initialize()
    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    ).initialize()

    assert disabled.backend == 'degraded'
    assert disabled.fallback_ready is False
    assert reenabled.backend == 'degraded'
    assert reenabled.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'third operator value'


def test_disabled_write_is_reconciled_when_core_is_reenabled(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    durable = runtime.replace(
        'durable revision two',
        actor_key='agent:main',
        reason='llm_update',
    )

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    assert disabled.initialize().text == durable.text
    file_update = disabled.replace(
        'updated while disabled',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    disabled_state = json.loads(disabled._state_path.read_text(encoding='utf-8'))
    assert disabled_state['source'] == 'file'
    assert disabled_state['core_revision'] == 2

    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    reconciled = reenabled.initialize()

    assert reconciled.backend == 'core'
    assert reconciled.text == file_update.text
    assert reconciled.revision == 3
    assert legacy.read_text(encoding='utf-8') == file_update.text
    reconciled_state = json.loads(reenabled._state_path.read_text(encoding='utf-8'))
    assert reconciled_state['source'] == 'core'
    assert reconciled_state['core_revision'] == 3
    assert reconciled_state['mirror_synced'] is True

    store = MemoryStore.open(runtime.database_path)
    record = store.list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    changes = store.changes(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        record.memory_id,
    )
    assert (changes[-1].actor_key, changes[-1].reason) == (
        'system:compatibility_sync',
        'compatibility_sync',
    )


def test_reenable_imports_direct_edit_after_confirmed_file_write(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    disabled.replace(
        'journaled file value',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    legacy.write_text('newest direct file value', encoding='utf-8')

    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = reenabled.initialize()

    assert snapshot.backend == 'core'
    assert snapshot.text == 'newest direct file value'
    assert snapshot.revision == 2
    state = json.loads(reenabled._state_path.read_text(encoding='utf-8'))
    assert state['source'] == 'core'
    assert state['body'] == snapshot.text
    assert state['core_revision'] == snapshot.revision


def test_operator_file_edit_is_honored_after_explicit_disable(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    legacy.write_text('operator rollback value', encoding='utf-8')

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    file_snapshot = disabled.initialize()

    assert file_snapshot.backend == 'file'
    assert file_snapshot.text == 'operator rollback value'
    state = json.loads(disabled._state_path.read_text(encoding='utf-8'))
    assert state['source'] == 'file'
    assert state['core_revision'] == original.revision

    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    reconciled = reenabled.initialize()
    assert reconciled.text == file_snapshot.text
    assert reconciled.revision == 2


def test_unjournaled_file_divergence_enters_read_only_mode(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    legacy.write_text('external file edit', encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == 'external file edit'
    assert snapshot.fallback_ready is False
    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        restarted.replace(
            'must not choose one side implicitly',
            actor_key='agent:main',
            reason='llm_update',
        )

    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == original.text


def test_explicit_file_rollback_never_changes_a_newer_database(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    original_state = runtime._state_path.read_text(encoding='utf-8')
    real_atomic_write = agent_memory._atomic_write

    def fail_all_writes(*args, **kwargs):
        raise OSError('read only')

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_all_writes)
    updated = runtime.replace(
        'database revision two',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.revision == 2
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == original.text
    assert runtime._state_path.read_text(encoding='utf-8') == original_state

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    disabled_snapshot = disabled.initialize()
    assert disabled_snapshot.backend == 'file'
    assert disabled_snapshot.text == original.text
    assert disabled_snapshot.fallback_ready is True

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    recovered = restarted.initialize()

    assert recovered.backend == 'core'
    assert recovered.text == updated.text
    assert recovered.revision == updated.revision
    assert legacy.read_text(encoding='utf-8') == updated.text
    refreshed_state = json.loads(restarted._state_path.read_text(encoding='utf-8'))
    assert refreshed_state['body'] == updated.text
    assert refreshed_state['core_revision'] == updated.revision


def test_same_content_retry_repairs_stale_compatibility_state(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    original_state = runtime._state_path.read_text(encoding='utf-8')
    real_atomic_write = agent_memory._atomic_write

    def fail_state_only(path, text):
        if path == runtime._state_path:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_state_only)
    updated = runtime.replace(
        'new durable value',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == updated.text
    assert runtime._state_path.read_text(encoding='utf-8') == original_state

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    retried = runtime.replace(
        updated.text,
        actor_key='agent:main',
        reason='llm_update',
    )
    assert retried.revision == updated.revision
    assert retried.fallback_ready is True
    repaired_state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    assert repaired_state['body'] == updated.text
    assert repaired_state['core_revision'] == updated.revision

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    assert disabled.initialize().text == updated.text


def test_same_process_retry_repairs_state_before_disabled_handoff(
    tmp_path, monkeypatch
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_state_only(path, text):
        if path == runtime._state_path:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_state_only)
    durable = runtime.replace(
        'durable revision two',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert durable.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == durable.text

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    repaired = runtime.replace(
        durable.text,
        actor_key='agent:main',
        reason='llm_update',
    )
    assert repaired.fallback_ready is True
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    recovered = disabled.initialize()
    repaired_state = json.loads(disabled._state_path.read_text(encoding='utf-8'))
    assert recovered.text == durable.text
    assert repaired_state['source'] == 'core'
    assert repaired_state['core_revision'] == durable.revision

    offline = disabled.replace(
        'updated while disabled',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    ).initialize()
    assert reenabled.backend == 'core'
    assert reenabled.text == offline.text
    assert reenabled.revision == 3


def test_disabled_rollback_accepts_file_but_reenable_detects_newer_database(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_state_only(path, text):
        if path == runtime._state_path:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_state_only)
    durable = runtime.replace(
        'newer database value',
        actor_key='agent:main',
        reason='llm_update',
    )
    legacy.write_text('third operator value', encoding='utf-8')
    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = disabled.initialize()

    assert snapshot.backend == 'file'
    assert snapshot.text == 'third operator value'
    assert snapshot.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == 'third operator value'
    state = json.loads(disabled._state_path.read_text(encoding='utf-8'))
    assert state['source'] == 'file'
    assert state['body'] == 'third operator value'
    assert state['core_revision'] == 1

    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    ).initialize()
    assert reenabled.backend == 'degraded'
    assert reenabled.fallback_ready is False
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == durable.text


def test_enabled_restart_refuses_newer_database_with_third_file_value(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_state_only(path, text):
        if path == runtime._state_path:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_state_only)
    durable = runtime.replace(
        'newer database value',
        actor_key='agent:main',
        reason='llm_update',
    )
    legacy.write_text('third operator value', encoding='utf-8')
    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == 'third operator value'
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'third operator value'
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == durable.text


def test_missing_state_allows_explicit_rollback_but_reenable_requires_reconciliation(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    runtime._state_path.unlink()
    real_atomic_write = agent_memory._atomic_write

    def fail_all_writes(*args, **kwargs):
        raise OSError('read only')

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_all_writes)
    durable = runtime.replace(
        'durable without compatibility files',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert durable.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'initial memory'

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = disabled.initialize()

    assert snapshot.backend == 'file'
    assert snapshot.text == 'initial memory'
    assert snapshot.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == 'initial memory'
    assert not disabled._state_path.exists()

    reenabled = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    ).initialize()
    assert reenabled.backend == 'degraded'
    assert reenabled.fallback_ready is False


@pytest.mark.parametrize(
    ('state_revision', 'state_body'),
    [(2, 'newer compatibility value'), (1, 'conflicting compatibility value')],
)
def test_core_state_direction_conflict_requires_manual_reconciliation(
    tmp_path,
    state_revision,
    state_body,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    _rewrite_state(
        runtime,
        body=state_body,
        core_revision=state_revision,
        mirror_synced=True,
        source='core',
    )
    legacy.write_text(state_body, encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == state_body
    assert snapshot.fallback_ready is False
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == 'initial memory'
    assert legacy.read_text(encoding='utf-8') == state_body


def test_empty_database_rebuild_refuses_state_and_file_conflict(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    legacy.write_text('newer direct file value', encoding='utf-8')
    rebuilt_database = tmp_path / 'rebuilt.db'

    rebuilt = agent_memory.AgentMemory(
        enabled=True,
        database_path=rebuilt_database,
        legacy_path=legacy,
    )
    snapshot = rebuilt.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == 'newer direct file value'
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'newer direct file value'
    records = MemoryStore.open(rebuilt_database).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )
    assert records == ()


def test_unsynced_core_state_and_changed_file_degrades_without_overwriting_either(
    tmp_path,
):
    runtime, legacy = _runtime(tmp_path)
    durable = runtime.initialize()
    _rewrite_state(runtime, mirror_synced=False)
    legacy.write_text('operator file value', encoding='utf-8')

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = disabled.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == durable.text
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'operator file value'
    state = json.loads(disabled._state_path.read_text(encoding='utf-8'))
    assert state['mirror_synced'] is False


def test_runtime_file_conflict_switches_service_to_read_only(tmp_path):
    core, legacy = _runtime(tmp_path)
    core.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=core.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    _rewrite_state(disabled, mirror_synced=False)
    legacy.write_text('ambiguous file value', encoding='utf-8')

    snapshot = _eventually_snapshot(
        disabled,
        lambda candidate: candidate.backend == 'degraded',
    )

    assert snapshot.backend == 'degraded'
    assert snapshot.fallback_ready is False
    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        disabled.replace(
            'must not write through a conflict',
            actor_key='agent:main',
            reason='llm_update',
        )


def test_background_file_refresh_reports_failed_mirror_recovery(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    expected = runtime.replace(
        'file-backed memory',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    real_atomic_write = agent_memory._atomic_write
    legacy.unlink()

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    observed = _eventually_snapshot(
        runtime,
        lambda candidate: candidate.fallback_ready is False,
    )

    assert observed.backend == 'file'
    assert observed.text == expected.text
    assert legacy.exists() is False


def test_background_file_refresh_never_overwrites_a_later_direct_edit(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    runtime.replace(
        'file-backed memory',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    _rewrite_state(runtime, mirror_synced=True)
    real_mirror = runtime._mirror

    def edit_then_mirror(text, **kwargs):
        legacy.write_text('direct edit after refresh read', encoding='utf-8')
        return real_mirror(text, **kwargs)

    monkeypatch.setattr(runtime, '_mirror', edit_then_mirror)
    observed = _eventually_snapshot(
        runtime,
        lambda candidate: candidate.text == 'direct edit after refresh read',
    )

    assert observed.backend == 'file'
    assert observed.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == 'direct edit after refresh read'


def test_file_replace_never_overwrites_an_edit_after_reconciliation(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path, enabled=False)
    runtime.initialize()
    runtime.replace(
        'first file-backed value',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    real_mirror = runtime._mirror

    def edit_then_mirror(text, **kwargs):
        legacy.write_text('operator edit during replace', encoding='utf-8')
        return real_mirror(text, **kwargs)

    monkeypatch.setattr(runtime, '_mirror', edit_then_mirror)
    updated = runtime.replace(
        'second file-backed value',
        actor_key='api:agent_definition',
        reason='api_edit',
    )

    assert updated.text == 'second file-backed value'
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'operator edit during replace'


def test_stale_instance_cannot_publish_a_false_noop_over_newer_core_value(tmp_path):
    first, legacy = _runtime(tmp_path)
    first.initialize()
    stale = agent_memory.AgentMemory(
        enabled=True,
        database_path=first.database_path,
        legacy_path=legacy,
    )
    stale.initialize()
    current = first.replace(
        'newer durable value',
        actor_key='agent:main',
        reason='llm_update',
    )

    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        stale.replace(
            'initial memory',
            actor_key='agent:main',
            reason='llm_update',
        )

    assert stale.snapshot().text == current.text
    assert stale.snapshot().revision == current.revision
    assert legacy.read_text(encoding='utf-8') == current.text
    state = json.loads(stale._state_path.read_text(encoding='utf-8'))
    assert state['body'] == current.text
    assert state['core_revision'] == current.revision


def test_enabled_instance_eventually_observes_another_core_writer(tmp_path):
    writer, legacy = _runtime(tmp_path)
    writer.initialize()
    reader = agent_memory.AgentMemory(
        enabled=True,
        database_path=writer.database_path,
        legacy_path=legacy,
    )
    reader.initialize()
    current = writer.replace(
        'written by another core instance',
        actor_key='agent:main',
        reason='llm_update',
    )

    observed = _eventually_snapshot(
        reader,
        lambda candidate: candidate.revision == current.revision,
    )

    assert observed.backend == 'core'
    assert observed.text == current.text


def test_stale_disabled_writer_rebases_on_latest_core_generation(tmp_path):
    enabled, legacy = _runtime(tmp_path)
    enabled.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=enabled.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    core_update = enabled.replace(
        'new durable generation',
        actor_key='agent:main',
        reason='llm_update',
    )

    file_update = disabled.replace(
        'file update after durable generation',
        actor_key='api:agent_definition',
        reason='api_edit',
    )
    state = json.loads(disabled._state_path.read_text(encoding='utf-8'))

    assert file_update.backend == 'file'
    assert state['source'] == 'file'
    assert state['core_revision'] == core_update.revision

    reconciled = agent_memory.AgentMemory(
        enabled=True,
        database_path=enabled.database_path,
        legacy_path=legacy,
    ).initialize()

    assert reconciled.backend == 'core'
    assert reconciled.text == file_update.text
    assert reconciled.revision == core_update.revision + 1


def test_enabled_instance_detects_live_file_mode_update(tmp_path):
    enabled, legacy = _runtime(tmp_path)
    enabled.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=enabled.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    pending = disabled.replace(
        'new file-mode value',
        actor_key='api:agent_definition',
        reason='api_edit',
    )

    observed = _eventually_snapshot(
        enabled,
        lambda candidate: candidate.backend == 'degraded',
    )

    assert observed.text == pending.text
    assert observed.fallback_ready is False


def test_live_core_writer_refuses_pending_disabled_mode_update(tmp_path):
    enabled, legacy = _runtime(tmp_path)
    original = enabled.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=enabled.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    pending = disabled.replace(
        'pending file-backed value',
        actor_key='api:agent_definition',
        reason='api_edit',
    )

    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        enabled.replace(
            'competing core value',
            actor_key='agent:main',
            reason='llm_update',
        )

    assert enabled.snapshot().backend == 'degraded'
    assert enabled.snapshot().fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == pending.text
    state = json.loads(enabled._state_path.read_text(encoding='utf-8'))
    assert state['source'] == 'file'
    assert state['body'] == pending.text
    stored = MemoryStore.open(enabled.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == original.text

    reconciled = agent_memory.AgentMemory(
        enabled=True,
        database_path=enabled.database_path,
        legacy_path=legacy,
    ).initialize()
    assert reconciled.backend == 'core'
    assert reconciled.text == pending.text
    assert reconciled.revision == 2


def test_live_core_writer_refuses_direct_compatibility_file_edit(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    legacy.write_text('direct file edit', encoding='utf-8')

    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        runtime.replace(
            'must not overwrite the file edit',
            actor_key='agent:main',
            reason='llm_update',
        )

    assert runtime.snapshot().backend == 'degraded'
    assert runtime.snapshot().text == original.text
    assert runtime.snapshot().fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'direct file edit'


def test_disabled_mode_uses_proven_core_value_when_mirror_publish_failed(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    updated = runtime.replace(
        'latest durable value',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'initial memory'

    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    fallback = disabled.initialize()

    assert fallback.backend == 'file'
    assert fallback.text == updated.text
    assert fallback.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'initial memory'


def test_core_restart_repairs_interrupted_mirror_with_proven_predecessor(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    updated = runtime.replace(
        'latest durable value',
        actor_key='agent:main',
        reason='llm_update',
    )
    interrupted_state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    assert interrupted_state['mirror_synced'] is False
    assert len(interrupted_state['mirror_digest']) == 64
    assert legacy.read_text(encoding='utf-8') == 'initial memory'

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    recovered = restarted.initialize()

    assert recovered.backend == 'core'
    assert recovered.text == updated.text
    assert recovered.revision == updated.revision
    assert recovered.fallback_ready is True
    assert legacy.read_text(encoding='utf-8') == updated.text
    repaired_state = json.loads(restarted._state_path.read_text(encoding='utf-8'))
    assert repaired_state['mirror_synced'] is True
    assert len(repaired_state['mirror_digest']) == 64


def test_core_restart_never_repairs_over_a_later_direct_edit(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    updated = runtime.replace(
        'latest durable value',
        actor_key='agent:main',
        reason='llm_update',
    )
    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    legacy.write_text('later operator value', encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == 'later operator value'
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'later operator value'
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == updated.text


@pytest.mark.parametrize('mutation', ['delete', 'truncate'])
def test_core_restart_never_repairs_over_a_later_missing_mirror(
    tmp_path,
    monkeypatch,
    mutation,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_legacy_only(path, text):
        if path == legacy:
            raise OSError('read only')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_legacy_only)
    updated = runtime.replace(
        'latest durable value',
        actor_key='agent:main',
        reason='llm_update',
    )
    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    if mutation == 'delete':
        legacy.unlink()
    else:
        legacy.write_text('', encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == ''
    assert snapshot.fallback_ready is False
    if mutation == 'delete':
        assert legacy.exists() is False
    else:
        assert legacy.read_text(encoding='utf-8') == ''
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == updated.text


@pytest.mark.parametrize('mutation', ['delete', 'truncate'])
def test_newer_core_without_unsynced_journal_never_overwrites_missing_mirror(
    tmp_path,
    monkeypatch,
    mutation,
):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_state_and_mirror(path, text):
        if path in {runtime._state_path, legacy}:
            raise OSError('compatibility publication unavailable')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_state_and_mirror)
    updated = runtime.replace(
        'durable revision without a new journal',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.revision == original.revision + 1
    persisted_state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    assert persisted_state['core_revision'] == original.revision

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    if mutation == 'delete':
        legacy.unlink()
    else:
        legacy.write_text('', encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.fallback_ready is False
    if mutation == 'delete':
        assert legacy.exists() is False
    else:
        assert legacy.read_text(encoding='utf-8') == ''
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == updated.text


def test_unsynced_predecessor_generation_rejects_an_aba_file_edit(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    real_atomic_write = agent_memory._atomic_write

    def fail_final_confirmation(path, text):
        if path == runtime._state_path:
            payload = json.loads(text)
            if payload['body'] == 'durable revision two' and payload['mirror_synced']:
                raise OSError('final confirmation unavailable')
        return real_atomic_write(path, text)

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_final_confirmation)
    updated = runtime.replace(
        'durable revision two',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == updated.text
    interrupted_state = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    assert interrupted_state['mirror_synced'] is False

    monkeypatch.setattr(agent_memory, '_atomic_write', real_atomic_write)
    legacy.write_text(original.text, encoding='utf-8')
    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == original.text
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == original.text


def test_unsynced_missing_predecessor_does_not_treat_empty_file_as_same_generation(
    tmp_path,
):
    runtime, legacy = _runtime(tmp_path)
    record = runtime.initialize()
    legacy.unlink()
    missing_observation = runtime._stable_legacy_observation()
    assert missing_observation is not None
    signature, digest, raw = missing_observation
    assert signature is None
    assert raw is None
    runtime._write_compatibility_state(
        agent_memory._CompatibilityState(
            source='core',
            body=record.text,
            core_revision=record.revision,
            mirror_synced=False,
            mirror_digest=digest,
            mirror_signature=signature,
        )
    )
    legacy.write_text('', encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == ''
    assert snapshot.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == ''


def test_compatibility_state_checksum_rejects_valid_json_body_tampering(tmp_path):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    payload = json.loads(runtime._state_path.read_text(encoding='utf-8'))
    payload['body'] = 'tampered but syntactically valid memory'
    runtime._state_path.write_text(json.dumps(payload), encoding='utf-8')

    restarted = agent_memory.AgentMemory(
        enabled=True,
        database_path=runtime.database_path,
        legacy_path=legacy,
    )
    snapshot = restarted.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == original.text
    assert legacy.read_text(encoding='utf-8') == original.text
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == original.text


def test_corrupt_database_degrades_reads_but_refuses_split_brain_write(
    tmp_path, capsys
):
    runtime, legacy = _runtime(tmp_path)
    runtime.database_path.write_bytes(b'not a sqlite database')

    snapshot = runtime.initialize()

    assert snapshot.backend == 'degraded'
    assert snapshot.text == 'initial memory'
    assert runtime.status()['warning'] == agent_memory.DEGRADED_WARNING
    assert 'initial memory' not in capsys.readouterr().out

    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        runtime.replace(
            'must not be written only to the fallback',
            actor_key='agent:main',
            reason='llm_update',
        )
    assert legacy.read_text(encoding='utf-8') == 'initial memory'


def test_transient_store_error_keeps_cache_and_mirror_unchanged(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()

    def fail_update(*args, **kwargs):
        raise StorageBusyError('busy')

    monkeypatch.setattr(runtime._store, 'update', fail_update)

    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        runtime.replace(
            'new value',
            actor_key='agent:main',
            reason='llm_update',
        )

    assert runtime.snapshot() == original
    assert legacy.read_text(encoding='utf-8') == original.text


def test_transient_background_read_contention_keeps_core_writable(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    real_read = runtime._store.read
    attempted = threading.Event()
    calls = 0

    def busy_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            attempted.set()
            raise StorageBusyError('busy')
        return real_read(*args, **kwargs)

    monkeypatch.setattr(runtime._store, 'read', busy_once)
    legacy.touch()
    runtime.snapshot()
    assert attempted.wait(timeout=2)
    deadline = time.monotonic() + 2
    while runtime._refresh_running and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runtime.snapshot().backend == 'core'
    assert runtime.snapshot().text == original.text
    updated = runtime.replace(
        'write after transient contention',
        actor_key='agent:main',
        reason='llm_update',
    )
    assert updated.backend == 'core'
    assert updated.text == 'write after transient contention'


def test_snapshot_does_not_wait_for_slow_durable_write(tmp_path, monkeypatch):
    runtime, _ = _runtime(tmp_path)
    original = runtime.initialize()
    entered = threading.Event()
    release = threading.Event()
    real_update = runtime._store.update

    def slow_update(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(runtime._store, 'update', slow_update)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        write = pool.submit(
            runtime.replace,
            'new memory',
            actor_key='agent:main',
            reason='llm_update',
        )
        assert entered.wait(timeout=2)
        read = pool.submit(runtime.snapshot)
        try:
            observed = read.result(timeout=0.5)
        finally:
            release.set()
        updated = write.result(timeout=2)

    assert observed == original
    assert updated.text == 'new memory'


def test_snapshot_does_not_wait_for_slow_compatibility_publish(tmp_path, monkeypatch):
    runtime, _ = _runtime(tmp_path)
    original = runtime.initialize()
    entered = threading.Event()
    release = threading.Event()
    real_mirror = runtime._mirror

    def slow_mirror(text, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_mirror(text, **kwargs)

    monkeypatch.setattr(runtime, '_mirror', slow_mirror)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        write = pool.submit(
            runtime.replace,
            'new memory',
            actor_key='agent:main',
            reason='llm_update',
        )
        assert entered.wait(timeout=2)
        started = time.monotonic()
        observed = runtime.snapshot()
        elapsed = time.monotonic() - started
        release.set()
        updated = write.result(timeout=2)

    assert elapsed < 0.2
    assert observed == original
    assert updated.text == 'new memory'


def test_post_commit_file_change_is_not_overwritten(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_update = runtime._update_record

    def update_then_edit_file(*args, **kwargs):
        record = real_update(*args, **kwargs)
        legacy.write_text('concurrent file intent', encoding='utf-8')
        return record

    monkeypatch.setattr(runtime, '_update_record', update_then_edit_file)
    committed = runtime.replace(
        'durable committed value',
        actor_key='agent:main',
        reason='llm_update',
    )

    assert committed.text == 'durable committed value'
    assert committed.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'concurrent file intent'
    observed = _eventually_snapshot(
        runtime,
        lambda candidate: candidate.backend == 'degraded',
    )
    assert observed.text == 'concurrent file intent'


def test_direct_edit_after_publication_guard_is_not_overwritten(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()
    real_mirror = runtime._mirror

    def edit_then_mirror(text, **kwargs):
        legacy.write_text('operator value after guard', encoding='utf-8')
        return real_mirror(text, **kwargs)

    monkeypatch.setattr(runtime, '_mirror', edit_then_mirror)
    committed = runtime.replace(
        'durable committed value',
        actor_key='agent:main',
        reason='llm_update',
    )

    assert committed.text == 'durable committed value'
    assert committed.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'operator value after guard'


def test_file_snapshot_journal_holds_handoff_against_core_writer(tmp_path, monkeypatch):
    core, legacy = _runtime(tmp_path)
    core.initialize()
    disabled = agent_memory.AgentMemory(
        enabled=False,
        database_path=core.database_path,
        legacy_path=legacy,
    )
    disabled.initialize()
    legacy.write_text('file intent', encoding='utf-8')
    entered = threading.Event()
    release = threading.Event()
    real_read_state = disabled._read_compatibility_state

    def slow_read_state():
        state = real_read_state()
        entered.set()
        assert release.wait(timeout=2)
        return state

    monkeypatch.setattr(disabled, '_read_compatibility_state', slow_read_state)
    disabled.snapshot()
    assert entered.wait(timeout=2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        write = pool.submit(
            core.replace,
            'core value must not win',
            actor_key='agent:main',
            reason='llm_update',
        )
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                write.result(timeout=0.1)
        finally:
            release.set()
        with pytest.raises(agent_memory.AgentMemoryUnavailableError):
            write.result(timeout=2)

    state = json.loads(disabled._state_path.read_text(encoding='utf-8'))
    assert state['source'] == 'file'
    assert state['body'] == 'file intent'
    assert legacy.read_text(encoding='utf-8') == 'file intent'


def test_snapshot_cannot_observe_partially_initialized_core_state(
    tmp_path, monkeypatch
):
    runtime, _ = _runtime(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    real_publish = runtime._publish_core_compatibility

    def slow_publish(record, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_publish(record, **kwargs)

    monkeypatch.setattr(runtime, '_publish_core_compatibility', slow_publish)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        initializing = pool.submit(runtime.initialize)
        assert entered.wait(timeout=2)
        reading = pool.submit(runtime.snapshot)
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                reading.result(timeout=0.1)
        finally:
            release.set()
        initialized = initializing.result(timeout=2)
        observed = reading.result(timeout=2)

    assert observed == initialized
    assert observed.backend == 'core'
    assert observed.text == 'initial memory'


def test_snapshot_cannot_observe_partially_initialized_degraded_state(
    tmp_path,
    monkeypatch,
):
    legacy = tmp_path / 'prompt_memory.md'
    legacy.write_text('safe compatibility memory', encoding='utf-8')

    def fail_store(_path):
        raise StorageDamagedError('damaged')

    runtime = agent_memory.AgentMemory(
        enabled=True,
        database_path=tmp_path / 'memory.db',
        legacy_path=legacy,
        store_factory=fail_store,
    )
    entered = threading.Event()
    release = threading.Event()
    real_fallback = runtime._fallback_snapshot

    def slow_fallback(*, backend):
        entered.set()
        assert release.wait(timeout=2)
        return real_fallback(backend=backend)

    monkeypatch.setattr(runtime, '_fallback_snapshot', slow_fallback)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        initializing = pool.submit(runtime.initialize)
        assert entered.wait(timeout=2)
        reading = pool.submit(runtime.snapshot)
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                reading.result(timeout=0.1)
        finally:
            release.set()
        initialized = initializing.result(timeout=2)
        observed = reading.result(timeout=2)

    assert observed == initialized
    assert observed.backend == 'degraded'
    assert observed.text == 'safe compatibility memory'


def test_async_facade_runs_write_outside_event_loop_thread(tmp_path, monkeypatch):
    runtime, _ = _runtime(tmp_path)
    runtime.initialize()
    event_loop_thread = threading.get_ident()
    write_threads = []
    real_replace = runtime.replace

    def observed_replace(*args, **kwargs):
        write_threads.append(threading.get_ident())
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(runtime, 'replace', observed_replace)
    monkeypatch.setattr(agent_memory, '_service', runtime)

    updated = asyncio.run(
        agent_memory.replace(
            'written in worker',
            actor_key='agent:main',
            reason='llm_update',
        )
    )

    assert updated.text == 'written in worker'
    assert write_threads and write_threads[0] != event_loop_thread


def test_mirror_failure_does_not_roll_back_committed_memory(
    tmp_path, monkeypatch, capsys
):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()

    def fail_mirror(path, text):
        raise OSError('read only')

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_mirror)
    updated = runtime.replace(
        'durably committed',
        actor_key='api:solutions',
        reason='solution_apply',
    )

    assert updated.text == 'durably committed'
    assert updated.revision == 2
    assert updated.fallback_ready is False
    assert legacy.read_text(encoding='utf-8') == 'initial memory'
    output = capsys.readouterr().out
    assert 'compatibility mirror failed' in output
    assert 'durably committed' not in output

    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )
    assert stored[0].body == 'durably committed'


def test_stat_failure_after_durable_commit_returns_the_committed_value(
    tmp_path,
    monkeypatch,
):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    real_update = runtime._update_record
    real_handoff_signature = runtime._handoff_signature
    committed = False

    def observed_update(*args, **kwargs):
        nonlocal committed
        record = real_update(*args, **kwargs)
        committed = True
        return record

    def fail_post_commit_stat():
        if committed:
            raise PermissionError('stat unavailable after commit')
        return real_handoff_signature()

    monkeypatch.setattr(runtime, '_update_record', observed_update)
    monkeypatch.setattr(runtime, '_handoff_signature', fail_post_commit_stat)

    updated = runtime.replace(
        'durable despite publication stat failure',
        actor_key='agent:main',
        reason='llm_update',
    )

    assert updated.backend == 'core'
    assert updated.revision == original.revision + 1
    assert updated.text == 'durable despite publication stat failure'
    assert updated.fallback_ready is False
    assert runtime._snapshot == updated
    assert legacy.read_text(encoding='utf-8') == original.text
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == updated.text


def test_invalid_runtime_compatibility_encoding_is_a_storage_error(
    tmp_path,
):
    runtime, legacy = _runtime(tmp_path)
    original = runtime.initialize()
    legacy.write_bytes(b'\xff\xfe')

    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        runtime.replace(
            'must not commit over invalid compatibility data',
            actor_key='agent:main',
            reason='llm_update',
        )

    assert runtime._snapshot.backend == 'degraded'
    assert runtime._snapshot.text == original.text
    stored = MemoryStore.open(runtime.database_path).list(
        AccessContext(owner_key='agent:main', actor_key='test:reader'),
        ListQuery(kinds=('agent_prompt',)),
    )[0]
    assert stored.body == original.text


def test_degrade_after_core_read_keeps_newer_cached_value(tmp_path, monkeypatch):
    runtime, legacy = _runtime(tmp_path)
    runtime.initialize()

    monkeypatch.setattr(agent_memory, '_atomic_write', lambda *args: None)
    committed = runtime.replace(
        'newer than mirror',
        actor_key='api:solutions',
        reason='solution_apply',
    )
    assert legacy.read_text(encoding='utf-8') == 'initial memory'

    def fail_update(*args, **kwargs):
        raise StorageDamagedError('damaged')

    monkeypatch.setattr(runtime._store, 'update', fail_update)
    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        runtime.replace(
            'not committed',
            actor_key='agent:main',
            reason='llm_update',
        )

    snapshot = runtime.snapshot()
    assert snapshot.backend == 'degraded'
    assert snapshot.text == committed.text


def test_degrade_preserves_unsynchronized_fallback_status(tmp_path, monkeypatch):
    runtime, _ = _runtime(tmp_path)
    runtime.initialize()

    def fail_all_compatibility_writes(*args, **kwargs):
        raise OSError('read only')

    monkeypatch.setattr(agent_memory, '_atomic_write', fail_all_compatibility_writes)
    committed = runtime.replace(
        'durable but not mirrored',
        actor_key='api:solutions',
        reason='solution_apply',
    )
    assert committed.fallback_ready is False

    def fail_update(*args, **kwargs):
        raise StorageDamagedError('damaged')

    monkeypatch.setattr(runtime._store, 'update', fail_update)
    with pytest.raises(agent_memory.AgentMemoryUnavailableError):
        runtime.replace(
            'not committed',
            actor_key='agent:main',
            reason='llm_update',
        )

    snapshot = runtime.snapshot()
    assert snapshot.backend == 'degraded'
    assert snapshot.text == committed.text
    assert snapshot.fallback_ready is False


@pytest.mark.parametrize('invalid', ['', '  ', 'nul\0byte', '\ud800'])
def test_invalid_prompt_is_rejected_consistently(tmp_path, invalid):
    runtime, _ = _runtime(tmp_path, enabled=False)

    with pytest.raises(agent_memory.AgentMemoryValidationError):
        runtime.replace(
            invalid,
            actor_key='agent:main',
            reason='llm_update',
        )


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('1', True),
        ('true', True),
        ('ON', True),
        ('0', False),
        ('false', False),
        ('No', False),
    ],
)
def test_environment_flag_is_strict(monkeypatch, raw, expected):
    monkeypatch.setenv('MEMORY_CORE_ENABLED', raw)
    assert agent_memory._enabled_from_environment() is expected


def test_invalid_environment_flag_is_rejected(monkeypatch):
    monkeypatch.setenv('MEMORY_CORE_ENABLED', 'sometimes')
    with pytest.raises(agent_memory.AgentMemoryValidationError):
        agent_memory._enabled_from_environment()
