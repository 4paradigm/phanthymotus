"""On-disk compatibility, application identity, and FTS recovery contracts."""

from __future__ import annotations

import sqlite3
import stat

import pytest

from memory_core import (
    ListQuery,
    MemoryStore,
    MigrationError,
    SearchQuery,
    StorageDamagedError,
    UnsupportedSchemaVersionError,
)
from memory_core.migrations import APPLICATION_ID, LATEST_SCHEMA_VERSION, MIGRATIONS
from memory_core_test_helpers import create_private


def _migration_rows(database_path):
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            """
            SELECT version, name, checksum, applied_at_ms
            FROM schema_migrations ORDER BY version
            """
        ).fetchall()


def test_reopening_is_migration_idempotent_and_keeps_data(db_path, clock, owner_a):
    first = MemoryStore(db_path, clock=clock)
    record = create_private(first, owner_a).record
    before = _migration_rows(db_path)

    second = MemoryStore(db_path, clock=clock)

    assert _migration_rows(db_path) == before
    assert [row[0] for row in before] == list(range(1, LATEST_SCHEMA_VERSION + 1))
    assert second.read(owner_a, record.memory_id) == record


def test_recorded_migration_checksums_match_code(db_path, clock):
    MemoryStore(db_path, clock=clock)

    recorded = [
        (version, name, checksum) for version, name, checksum, _ in _migration_rows(db_path)
    ]
    assert recorded == [
        (migration.version, migration.name, migration.checksum) for migration in MIGRATIONS
    ]


def test_checksum_drift_fails_closed(db_path, clock):
    MemoryStore(db_path, clock=clock)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = 'sha256:tampered' WHERE version = 1"
        )

    with pytest.raises(MigrationError):
        MemoryStore(db_path, clock=clock)


def test_newer_schema_version_fails_closed(db_path, clock):
    MemoryStore(db_path, clock=clock)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, checksum, applied_at_ms)
            VALUES (?, 'future', 'sha256:future', ?)
            """,
            (LATEST_SCHEMA_VERSION + 1, clock.now_ms),
        )

    with pytest.raises(UnsupportedSchemaVersionError):
        MemoryStore(db_path, clock=clock)


def test_database_sets_and_verifies_application_id(db_path, clock):
    MemoryStore(db_path, clock=clock)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        connection.execute("PRAGMA application_id = 1")

    with pytest.raises(StorageDamagedError):
        MemoryStore(db_path, clock=clock)


def test_non_memory_database_is_not_adopted(tmp_path, clock):
    path = tmp_path / "other-application.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA application_id = 12345")
        connection.execute("CREATE TABLE unrelated(value TEXT)")

    with pytest.raises(StorageDamagedError):
        MemoryStore(path, clock=clock)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'unrelated'"
        ).fetchone() == ("unrelated",)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'memory_entries'"
            ).fetchone()
            is None
        )


def test_zero_application_id_database_with_a_user_table_is_not_adopted(tmp_path, clock):
    path = tmp_path / "unclaimed-application.db"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('keep-me')")
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    path.chmod(0o644)
    original_mode = stat.S_IMODE(path.stat().st_mode)

    with pytest.raises(StorageDamagedError):
        MemoryStore(path, clock=clock)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == journal_mode
        assert connection.execute("SELECT value FROM unrelated").fetchall() == [("keep-me",)]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'memory_entries'"
            ).fetchone()
            is None
        )
    assert stat.S_IMODE(path.stat().st_mode) == original_mode


def test_zero_application_id_database_with_only_a_view_is_not_adopted(tmp_path, clock):
    path = tmp_path / "view-only-application.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE VIEW unrelated_view AS SELECT 'keep-me' AS value")
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    path.chmod(0o644)
    original_mode = stat.S_IMODE(path.stat().st_mode)

    with pytest.raises(StorageDamagedError):
        MemoryStore(path, clock=clock)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == journal_mode
        assert connection.execute("SELECT value FROM unrelated_view").fetchall() == [("keep-me",)]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'memory_entries'"
            ).fetchone()
            is None
        )
    assert stat.S_IMODE(path.stat().st_mode) == original_mode


def test_search_treats_fts_syntax_as_plain_user_text(store, owner_a):
    target = create_private(
        store,
        owner_a,
        title="C++ parser (v2)",
        body='Literal foo:bar and an "unfinished quote marker.',
        tags=("query-safe",),
    ).record

    for text in ("C++", "foo:bar", '"unfinished', "(v2)", "OR NOT NEAR", "* - : ( )"):
        hits = store.search(owner_a, SearchQuery(text=text))
        assert isinstance(hits, tuple)

    assert target.memory_id in {
        hit.record.memory_id for hit in store.search(owner_a, SearchQuery(text="foo:bar"))
    }


def test_search_supports_chinese_text(store, owner_a):
    target = create_private(
        store,
        owner_a,
        title="云南咖啡",
        body="云南咖啡豆 水洗处理 柑橘香气。",
        tags=("咖啡", "云南"),
    ).record

    hits = store.search(owner_a, SearchQuery(text="云南咖啡豆"))

    assert target.memory_id in {hit.record.memory_id for hit in hits}


def test_rebuild_restores_only_active_rows(store, db_path, owner_a):
    active = create_private(
        store,
        owner_a,
        op_key="active-index-row",
        body="active-index-token",
    ).record
    removed = create_private(
        store,
        owner_a,
        op_key="deleted-index-row",
        body="deleted-index-token",
    ).record
    store.delete(
        owner_a,
        removed.memory_id,
        expected_revision=1,
        op_key="delete-index-row",
        reason="cleanup",
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('delete-all')")

    with pytest.raises(StorageDamagedError):
        store.self_check()

    store.rebuild_search_index()

    assert tuple(
        hit.record.memory_id
        for hit in store.search(owner_a, SearchQuery(text="active-index-token"))
    ) == (active.memory_id,)
    assert store.search(owner_a, SearchQuery(text="deleted-index-token")) == ()
    assert store.list(owner_a, ListQuery(include_deleted=True))
    assert store.self_check()["status"] == "ok"
