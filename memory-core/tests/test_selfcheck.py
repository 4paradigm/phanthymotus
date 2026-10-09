"""The public self-check validates a real database, not only object shape."""

from __future__ import annotations

import sqlite3

import pytest

from memory_core import AccessContext, MemoryPatch, StorageDamagedError
from memory_core_test_helpers import create_private


def test_self_check_accepts_consistent_active_and_deleted_data(store, owner_a):
    active = create_private(store, owner_a, op_key="active").record
    changed = create_private(store, owner_a, op_key="changed").record
    changed = store.update(
        owner_a,
        changed.memory_id,
        MemoryPatch(body="updated body"),
        expected_revision=1,
        op_key="update",
    ).record
    removed = create_private(store, owner_a, op_key="removed").record
    store.delete(
        owner_a,
        removed.memory_id,
        expected_revision=1,
        op_key="delete",
        reason="cleanup",
    )
    create_private(
        store,
        AccessContext(owner_key="other-owner", actor_key="other-actor"),
        op_key="other-private-entry",
    )

    report = store.self_check()

    assert report == {
        "status": "ok",
        "schema_version": 2,
        "journal_mode": "wal",
    }
    assert store.read(owner_a, active.memory_id) == active
    assert store.read(owner_a, changed.memory_id) == changed


def test_self_check_rejects_foreign_key_damage(store, db_path, owner_a):
    create_private(store, owner_a)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO memory_changes(
                memory_id, actor_key, change_kind, from_revision,
                to_revision, reason, operation_ref, changed_ms
            ) VALUES ('missing-memory', 'actor', 'create', NULL, 1, '', 'orphan', 1)
            """
        )

    with pytest.raises(StorageDamagedError):
        store.self_check()
