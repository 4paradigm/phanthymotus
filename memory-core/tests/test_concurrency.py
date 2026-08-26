"""Concurrent callers preserve uniqueness, CAS, and cold-start safety."""

from __future__ import annotations

from memory_core import MemoryPatch, MemoryStore, RevisionConflictError
from memory_core_test_helpers import create_private, run_threads


def test_concurrent_distinct_creates_lose_no_records(store, owner_a):
    results, errors = run_threads(
        12,
        lambda index: create_private(
            store,
            owner_a,
            op_key=f"parallel-create-{index}",
            title=f"Parallel {index}",
            body=f"parallel body {index}",
        ),
    )

    assert errors == []
    assert len(results) == 12
    assert len({result.record.memory_id for result in results}) == 12
    assert len({result.change_seq for result in results}) == 12
    assert len(store.list(owner_a)) == 12
    assert all(len(store.changes(owner_a, result.record.memory_id)) == 1 for result in results)


def test_concurrent_same_create_operation_commits_once(store, owner_a):
    results, errors = run_threads(
        10,
        lambda _index: create_private(store, owner_a, op_key="one-operation"),
    )

    assert errors == []
    assert len(results) == 10
    assert len({result.record.memory_id for result in results}) == 1
    assert len({result.change_seq for result in results}) == 1
    assert sum(not result.replayed for result in results) == 1
    assert sum(result.replayed for result in results) == 9
    assert len(store.list(owner_a)) == 1
    assert len(store.changes(owner_a, results[0].record.memory_id)) == 1


def test_concurrent_updates_from_same_revision_allow_one_winner(store, owner_a):
    original = create_private(store, owner_a).record

    results, errors = run_threads(
        2,
        lambda index: store.update(
            owner_a,
            original.memory_id,
            MemoryPatch(body=f"winner candidate {index}"),
            expected_revision=1,
            op_key=f"parallel-update-{index}",
        ),
    )

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RevisionConflictError)
    assert results[0].record.revision == 2
    assert store.read(owner_a, original.memory_id) == results[0].record
    assert len(store.changes(owner_a, original.memory_id)) == 2


def test_concurrent_first_initialization_is_repeatable(tmp_path, clock):
    for iteration in range(5):
        database_path = tmp_path / f"cold-start-{iteration}.db"

        results, errors = run_threads(
            10,
            lambda _index, path=database_path: MemoryStore(path, clock=clock),
        )

        assert errors == []
        assert len(results) == 10
        assert MemoryStore(database_path, clock=clock).self_check()["status"] == "ok"
