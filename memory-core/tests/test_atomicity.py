"""Injected SQLite failures prove each public write commits as one transaction."""

from __future__ import annotations

import sqlite3

import pytest

from memory_core import (
    MemoryPatch,
    SearchQuery,
    StorageBusyError,
    StorageDamagedError,
)
from memory_core_test_helpers import create_private, make_draft


def _install_change_failure(database_path, change_kind: str) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"""
            CREATE TRIGGER fail_{change_kind}_change
            BEFORE INSERT ON memory_changes
            WHEN new.change_kind = '{change_kind}'
            BEGIN
                SELECT RAISE(ABORT, 'injected change failure');
            END
            """
        )


def _counts(database_path) -> tuple[int, int, int]:
    with sqlite3.connect(database_path) as connection:
        return tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("memory_entries", "memory_changes", "memory_requests")
        )


def test_create_rolls_back_entry_request_and_change(store, db_path, owner_a):
    _install_change_failure(db_path, "create")

    with pytest.raises(StorageDamagedError):
        create_private(store, owner_a)

    assert _counts(db_path) == (0, 0, 0)
    assert store.list(owner_a) == ()


def test_update_failure_restores_entry_index_and_request(store, db_path, owner_a):
    original = create_private(store, owner_a, body="oldmarkerqz").record
    before = _counts(db_path)
    _install_change_failure(db_path, "update")

    with pytest.raises(StorageDamagedError):
        store.update(
            owner_a,
            original.memory_id,
            MemoryPatch(body="newmarkerqz"),
            expected_revision=1,
            op_key="failing-update",
        )

    assert _counts(db_path) == before
    assert store.read(owner_a, original.memory_id) == original
    assert tuple(hit.record for hit in store.search(owner_a, SearchQuery(text="oldmarkerqz"))) == (
        original,
    )
    assert store.search(owner_a, SearchQuery(text="newmarkerqz")) == ()


def test_correction_failure_restores_entry_index_and_request(store, db_path, owner_a):
    original = create_private(store, owner_a, body="wrongmarkerqz").record
    before = _counts(db_path)
    _install_change_failure(db_path, "correct")

    with pytest.raises(StorageDamagedError):
        store.correct(
            owner_a,
            original.memory_id,
            make_draft(body="fixedmarkerqz"),
            expected_revision=1,
            op_key="failing-correct",
            reason="wrong_fact",
        )

    assert _counts(db_path) == before
    assert store.read(owner_a, original.memory_id) == original
    hits = store.search(owner_a, SearchQuery(text="wrongmarkerqz"))
    assert tuple(hit.record for hit in hits) == (original,)
    assert store.search(owner_a, SearchQuery(text="fixedmarkerqz")) == ()


def test_delete_failure_keeps_active_entry_and_index(store, db_path, owner_a):
    original = create_private(store, owner_a, body="must remain searchable").record
    before = _counts(db_path)
    _install_change_failure(db_path, "delete")

    with pytest.raises(StorageDamagedError):
        store.delete(
            owner_a,
            original.memory_id,
            expected_revision=1,
            op_key="failing-delete",
            reason="injected",
        )

    assert _counts(db_path) == before
    assert store.read(owner_a, original.memory_id) == original
    assert tuple(hit.record for hit in store.search(owner_a, SearchQuery(text="remain"))) == (
        original,
    )


def test_lock_timeout_is_reported_without_partial_write(db_path, clock, owner_a):
    from memory_core import MemoryStore

    store = MemoryStore(db_path, clock=clock, busy_timeout_ms=25)
    blocker = sqlite3.connect(db_path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(StorageBusyError):
            create_private(store, owner_a, op_key="blocked-create")
    finally:
        blocker.rollback()
        blocker.close()

    assert _counts(db_path) == (0, 0, 0)
    assert store.list(owner_a) == ()
