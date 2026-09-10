"""Authorization is enforced before any record existence is disclosed."""

from __future__ import annotations

import pytest

from memory_core import (
    AccessContext,
    MemoryNotFoundError,
    MemoryPatch,
    MemoryPlace,
    MemoryStore,
    SearchQuery,
    SharedPlaceDeniedError,
    ShareMode,
)
from memory_core_test_helpers import create_private, create_shared, make_draft


def test_private_record_is_visible_only_to_its_owner(store, owner_a, owner_b):
    record = create_private(store, owner_a, body="owner-a-only marker").record

    assert store.read(owner_a, record.memory_id) == record
    assert store.list(owner_b) == ()
    assert store.search(owner_b, SearchQuery(text="owner-a-only")) == ()
    with pytest.raises(MemoryNotFoundError):
        store.read(owner_b, record.memory_id)
    with pytest.raises(MemoryNotFoundError):
        store.changes(owner_b, record.memory_id)


def test_hidden_and_absent_ids_have_identical_read_failure(
    store,
    db_path,
    tmp_path,
    clock,
    owner_a,
    owner_b,
):
    memory_id = create_private(store, owner_a).record.memory_id
    empty_store = MemoryStore(tmp_path / "empty.db", clock=clock)

    with pytest.raises(MemoryNotFoundError) as hidden:
        store.read(owner_b, memory_id)
    with pytest.raises(MemoryNotFoundError) as absent:
        empty_store.read(owner_b, memory_id)

    assert type(hidden.value) is type(absent.value)
    assert str(hidden.value) == str(absent.value)


def test_private_mutations_by_another_owner_look_absent(store, owner_a, owner_b):
    record = create_private(store, owner_a).record

    operations = (
        lambda: store.update(
            owner_b,
            record.memory_id,
            MemoryPatch(body="not allowed"),
            expected_revision=1,
            op_key="hidden-update",
        ),
        lambda: store.correct(
            owner_b,
            record.memory_id,
            make_draft(body="not allowed"),
            expected_revision=1,
            op_key="hidden-correct",
            reason="attempt",
        ),
        lambda: store.delete(
            owner_b,
            record.memory_id,
            expected_revision=1,
            op_key="hidden-delete",
            reason="attempt",
        ),
    )

    for operation in operations:
        with pytest.raises(MemoryNotFoundError):
            operation()
    assert store.read(owner_a, record.memory_id) == record


def test_same_private_operation_key_is_independent_per_owner(store, owner_a, owner_b):
    first = create_private(store, owner_a, op_key="same-key").record
    second = create_private(store, owner_b, op_key="same-key").record

    assert first.memory_id != second.memory_id
    assert store.list(owner_a) == (first,)
    assert store.list(owner_b) == (second,)


def test_shared_read_key_grants_read_list_search_and_changes(
    store,
    shared_writer,
    shared_reader,
):
    created = create_shared(store, shared_writer, body="blue team launch plan")
    record = created.record

    assert record.share_mode is ShareMode.SHARED
    assert record.place_key == "team-blue"
    assert record.owner_key == shared_writer.owner_key
    assert store.read(shared_reader, record.memory_id) == record
    assert store.list(shared_reader) == (record,)
    assert tuple(hit.record for hit in store.search(shared_reader, SearchQuery(text="launch"))) == (
        record,
    )
    assert store.changes(shared_reader, record.memory_id)[0].change_seq == created.change_seq


def test_shared_read_without_write_cannot_create_or_mutate(
    store,
    shared_writer,
    shared_reader,
):
    record = create_shared(store, shared_writer).record

    operations = (
        lambda: create_shared(store, shared_reader, op_key="reader-create"),
        lambda: store.update(
            shared_reader,
            record.memory_id,
            MemoryPatch(body="reader edit"),
            expected_revision=1,
            op_key="reader-update",
        ),
        lambda: store.correct(
            shared_reader,
            record.memory_id,
            make_draft(body="reader correction"),
            expected_revision=1,
            op_key="reader-correct",
            reason="reader_attempt",
        ),
        lambda: store.delete(
            shared_reader,
            record.memory_id,
            expected_revision=1,
            op_key="reader-delete",
            reason="reader_attempt",
        ),
    )

    for operation in operations:
        with pytest.raises(SharedPlaceDeniedError):
            operation()
    assert store.read(shared_reader, record.memory_id) == record


def test_shared_record_without_read_key_is_hidden(store, shared_writer):
    record = create_shared(store, shared_writer, body="hidden shared marker").record
    blocked = AccessContext(owner_key="owner-c", actor_key="actor-c")

    with pytest.raises(MemoryNotFoundError):
        store.read(blocked, record.memory_id)
    with pytest.raises(MemoryNotFoundError):
        store.update(
            blocked,
            record.memory_id,
            MemoryPatch(body="blocked"),
            expected_revision=1,
            op_key="blocked-update",
        )
    assert store.list(blocked) == ()
    assert store.search(blocked, SearchQuery(text="hidden")) == ()


def test_invisible_places_never_add_search_hits(store, owner_a, owner_b, shared_writer):
    visible = create_private(
        store,
        owner_a,
        op_key="visible",
        body="common-token visible",
    ).record
    for index in range(5):
        create_private(
            store,
            owner_b,
            op_key=f"hidden-private-{index}",
            body=f"common-token hidden-private-{index}",
        )
    create_shared(
        store,
        shared_writer,
        op_key="hidden-shared",
        body="common-token hidden-shared",
    )

    hits = store.search(owner_a, SearchQuery(text="common-token"))

    assert tuple(hit.record.memory_id for hit in hits) == (visible.memory_id,)


def test_missing_shared_write_key_rejects_explicit_place(store, owner_a):
    with pytest.raises(SharedPlaceDeniedError):
        store.create(
            owner_a,
            MemoryPlace.shared("team-blue"),
            make_draft(),
            op_key="no-shared-key",
        )
