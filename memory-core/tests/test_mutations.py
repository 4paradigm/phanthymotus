"""Revision, idempotency, logical deletion, and change-ledger contracts."""

from __future__ import annotations

import sqlite3

import pytest

from memory_core import (
    ChangeKind,
    EntryState,
    IdempotencyConflictError,
    InvalidMemoryError,
    ListQuery,
    MemoryNotFoundError,
    MemoryPatch,
    RevisionConflictError,
    SearchQuery,
)
from memory_core_test_helpers import create_private, make_draft


def test_update_applies_only_patch_fields_and_appends_change(store, owner_a, clock):
    original = create_private(store, owner_a).record
    clock.advance(25)

    result = store.update(
        owner_a,
        original.memory_id,
        MemoryPatch(body="Now prefers natural Yunnan coffee.", tags=("coffee", "natural")),
        expected_revision=1,
        op_key="update-1",
        note="preference_changed",
    )

    updated = result.record
    assert result.replayed is False
    assert updated.memory_id == original.memory_id
    assert updated.revision == 2
    assert updated.title == original.title
    assert updated.body == "Now prefers natural Yunnan coffee."
    assert updated.kind == original.kind
    assert updated.tags == ("coffee", "natural")
    assert updated.metadata == original.metadata
    assert updated.created_ms == original.created_ms
    assert updated.updated_ms == clock.now_ms
    assert updated.deleted_ms is None
    assert updated.state is EntryState.ACTIVE

    change = store.changes(owner_a, original.memory_id)[-1]
    assert change.change_seq == result.change_seq
    assert change.memory_id == original.memory_id
    assert change.actor_key == owner_a.actor_key
    assert change.change_kind is ChangeKind.UPDATE
    assert change.from_revision == 1
    assert change.to_revision == 2
    assert change.reason == "preference_changed"
    assert change.operation_ref.startswith("op-sha256:")
    assert "update-1" not in change.operation_ref
    assert change.changed_ms == clock.now_ms


def test_correct_replaces_all_editable_fields(store, owner_a, clock):
    original = create_private(store, owner_a).record
    clock.advance()
    updated = store.update(
        owner_a,
        original.memory_id,
        MemoryPatch(body="intermediate"),
        expected_revision=1,
        op_key="update-first",
    ).record
    clock.advance()
    replacement = make_draft(
        title="Corrected preference",
        body="Prefers light-roast tea, not coffee.",
        kind="correction",
        tags=("tea",),
        metadata={"confirmed": True},
    )

    result = store.correct(
        owner_a,
        original.memory_id,
        replacement,
        expected_revision=updated.revision,
        op_key="correct-1",
        reason="wrong_fact",
    )

    corrected = result.record
    assert corrected.memory_id == original.memory_id
    assert corrected.revision == 3
    assert corrected.title == replacement.title
    assert corrected.body == replacement.body
    assert corrected.kind == replacement.kind
    assert corrected.tags == replacement.tags
    assert corrected.metadata == replacement.metadata
    assert corrected.created_ms == original.created_ms
    assert corrected.updated_ms == clock.now_ms
    assert corrected.state is EntryState.ACTIVE
    change = store.changes(owner_a, original.memory_id)[-1]
    assert change.change_kind is ChangeKind.CORRECT
    assert (change.from_revision, change.to_revision) == (2, 3)
    assert change.reason == "wrong_fact"


def test_delete_is_logical_and_removes_record_from_default_reads(store, owner_a, clock):
    original = create_private(store, owner_a, body="erase-me unique marker").record
    clock.advance(50)

    result = store.delete(
        owner_a,
        original.memory_id,
        expected_revision=1,
        op_key="delete-1",
        reason="user_request",
    )

    deleted = result.record
    assert deleted.memory_id == original.memory_id
    assert deleted.revision == 2
    assert deleted.state is EntryState.DELETED
    assert deleted.deleted_ms == clock.now_ms
    assert deleted.updated_ms == clock.now_ms
    with pytest.raises(MemoryNotFoundError):
        store.read(owner_a, original.memory_id)
    assert store.read(owner_a, original.memory_id, include_deleted=True) == deleted
    assert store.list(owner_a) == ()
    assert store.list(owner_a, ListQuery(include_deleted=True)) == (deleted,)
    assert store.search(owner_a, SearchQuery(text="erase-me")) == ()

    change = store.changes(owner_a, original.memory_id)[-1]
    assert change.change_kind is ChangeKind.DELETE
    assert (change.from_revision, change.to_revision) == (1, 2)
    assert change.reason == "user_request"


def test_change_history_is_ordered_and_pageable(store, owner_a):
    created = create_private(store, owner_a)
    updated = store.update(
        owner_a,
        created.record.memory_id,
        MemoryPatch(title="Changed title"),
        expected_revision=1,
        op_key="history-update",
    )
    corrected = store.correct(
        owner_a,
        created.record.memory_id,
        make_draft(title="Replacement"),
        expected_revision=2,
        op_key="history-correct",
        reason="wrong_fact",
    )
    deleted = store.delete(
        owner_a,
        created.record.memory_id,
        expected_revision=3,
        op_key="history-delete",
        reason="cleanup",
    )

    all_changes = store.changes(owner_a, created.record.memory_id)

    assert tuple(change.change_seq for change in all_changes) == (
        created.change_seq,
        updated.change_seq,
        corrected.change_seq,
        deleted.change_seq,
    )
    assert tuple(change.change_kind for change in all_changes) == tuple(ChangeKind)
    assert (
        store.changes(
            owner_a,
            created.record.memory_id,
            after_seq=created.change_seq,
            limit=2,
        )
        == all_changes[1:3]
    )


def test_stale_revision_rejects_every_mutation_without_side_effect(store, owner_a):
    original = create_private(store, owner_a).record
    current = store.update(
        owner_a,
        original.memory_id,
        MemoryPatch(body="revision two"),
        expected_revision=1,
        op_key="winning-update",
    ).record
    before_changes = store.changes(owner_a, original.memory_id)

    operations = (
        lambda: store.update(
            owner_a,
            original.memory_id,
            MemoryPatch(body="stale update"),
            expected_revision=1,
            op_key="stale-update",
        ),
        lambda: store.correct(
            owner_a,
            original.memory_id,
            make_draft(body="stale correction"),
            expected_revision=1,
            op_key="stale-correct",
            reason="stale",
        ),
        lambda: store.delete(
            owner_a,
            original.memory_id,
            expected_revision=1,
            op_key="stale-delete",
            reason="stale",
        ),
    )

    for operation in operations:
        with pytest.raises(RevisionConflictError):
            operation()
    assert store.read(owner_a, original.memory_id) == current
    assert store.changes(owner_a, original.memory_id) == before_changes


def test_create_replays_same_payload_and_rejects_key_reuse(store, owner_a):
    first = create_private(store, owner_a, op_key="create-once")
    replay = create_private(store, owner_a, op_key="create-once")

    assert replay.replayed is True
    assert replay.record == first.record
    assert replay.change_seq == first.change_seq
    with pytest.raises(IdempotencyConflictError):
        create_private(store, owner_a, op_key="create-once", body="different")
    assert store.list(owner_a) == (first.record,)
    assert store.changes(owner_a, first.record.memory_id) == (
        store.changes(owner_a, first.record.memory_id)[0],
    )


def test_update_replay_precedes_revision_check_and_detects_payload_change(store, owner_a):
    original = create_private(store, owner_a).record
    arguments = {
        "expected_revision": 1,
        "op_key": "update-once",
        "note": "confirmed",
    }
    first = store.update(
        owner_a,
        original.memory_id,
        MemoryPatch(body="revision two"),
        **arguments,
    )
    replay = store.update(
        owner_a,
        original.memory_id,
        MemoryPatch(body="revision two"),
        **arguments,
    )

    assert replay.replayed is True
    assert replay.record == first.record
    assert replay.change_seq == first.change_seq
    with pytest.raises(IdempotencyConflictError):
        store.update(
            owner_a,
            original.memory_id,
            MemoryPatch(body="different revision two"),
            **arguments,
        )
    assert len(store.changes(owner_a, original.memory_id)) == 2


def test_update_replay_returns_first_result_after_later_revisions(store, owner_a):
    original = create_private(store, owner_a).record
    patch = MemoryPatch(body="revision two")
    first = store.update(
        owner_a,
        original.memory_id,
        patch,
        expected_revision=1,
        op_key="durable-update-replay",
    )
    store.update(
        owner_a,
        original.memory_id,
        MemoryPatch(title="revision three title"),
        expected_revision=2,
        op_key="later-update",
    )

    replay = store.update(
        owner_a,
        original.memory_id,
        patch,
        expected_revision=1,
        op_key="durable-update-replay",
    )

    assert replay.replayed is True
    assert replay.record == first.record
    assert replay.change_seq == first.change_seq
    assert store.read(owner_a, original.memory_id).revision == 3


def test_correct_replay_precedes_revision_check_and_detects_payload_change(store, owner_a):
    original = create_private(store, owner_a).record
    arguments = {
        "expected_revision": 1,
        "op_key": "correct-once",
        "reason": "wrong_fact",
    }
    replacement = make_draft(title="Corrected")
    first = store.correct(owner_a, original.memory_id, replacement, **arguments)
    replay = store.correct(owner_a, original.memory_id, replacement, **arguments)

    assert replay.replayed is True
    assert replay.record == first.record
    assert replay.change_seq == first.change_seq
    with pytest.raises(IdempotencyConflictError):
        store.correct(
            owner_a,
            original.memory_id,
            make_draft(title="Different"),
            **arguments,
        )
    assert len(store.changes(owner_a, original.memory_id)) == 2


def test_delete_replay_precedes_revision_check_and_detects_payload_change(store, owner_a):
    original = create_private(store, owner_a).record
    arguments = {
        "expected_revision": 1,
        "op_key": "delete-once",
        "reason": "cleanup",
    }
    first = store.delete(owner_a, original.memory_id, **arguments)
    replay = store.delete(owner_a, original.memory_id, **arguments)

    assert replay.replayed is True
    assert replay.record == first.record
    assert replay.change_seq == first.change_seq
    with pytest.raises(IdempotencyConflictError):
        store.delete(
            owner_a,
            original.memory_id,
            expected_revision=1,
            op_key="delete-once",
            reason="different_reason",
        )
    assert len(store.changes(owner_a, original.memory_id)) == 2


def test_change_ledger_contains_no_memory_body(store, db_path, owner_a):
    marker = "BODY-MUST-NOT-ENTER-LEDGER-8fddde"
    record = create_private(store, owner_a, body=marker).record
    store.correct(
        owner_a,
        record.memory_id,
        make_draft(body=f"corrected-{marker}"),
        expected_revision=1,
        op_key="body-free-correct",
        reason="wrong_fact",
    )

    with sqlite3.connect(db_path) as connection:
        serialized = "\n".join(
            str(value)
            for row in connection.execute("SELECT * FROM memory_changes")
            for value in row
            if value is not None
        )

    assert marker not in serialized


def test_audit_fields_reject_body_text_and_hash_raw_operation_keys(store, db_path, owner_a):
    marker = "body text must never become an audit reason"
    record = create_private(store, owner_a, body=marker).record

    with pytest.raises(InvalidMemoryError):
        store.correct(
            owner_a,
            record.memory_id,
            make_draft(body="replacement"),
            expected_revision=1,
            op_key="rejected-reason-operation",
            reason=marker,
        )

    raw_operation_key = "raw operation key must not be persisted"
    created = create_private(
        store,
        owner_a,
        op_key=raw_operation_key,
        body="separate entry",
    )
    with sqlite3.connect(db_path) as connection:
        serialized = "\n".join(
            str(value)
            for table in ("memory_changes", "memory_requests")
            for row in connection.execute(f"SELECT * FROM {table}")
            for value in row
            if value is not None
        )

    assert raw_operation_key not in serialized
    assert created.record.memory_id in serialized
    assert len(store.changes(owner_a, record.memory_id)) == 1


def test_change_cursor_rejects_values_outside_sqlite_integer_range(store, owner_a):
    record = create_private(store, owner_a).record

    for value in (-1, 1 << 63, True, 1.5, "1"):
        with pytest.raises(InvalidMemoryError):
            store.changes(owner_a, record.memory_id, after_seq=value)


def test_change_ledger_rejects_update_delete_and_insert_or_replace(store, db_path, owner_a):
    result = create_private(store, owner_a)

    with sqlite3.connect(db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE memory_changes SET reason = 'tampered' WHERE change_seq = ?",
                (result.change_seq,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM memory_changes WHERE change_seq = ?",
                (result.change_seq,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_changes
                SELECT * FROM memory_changes WHERE change_seq = ?
                """,
                (result.change_seq,),
            )

    assert len(store.changes(owner_a, result.record.memory_id)) == 1
