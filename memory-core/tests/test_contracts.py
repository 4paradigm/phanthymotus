"""Public object and persistence contracts, independent of implementation helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from memory_core import (
    AccessContext,
    ChangeKind,
    EntryState,
    InvalidMemoryError,
    ListQuery,
    MemoryDraft,
    MemoryPatch,
    MemoryPlace,
    MemoryStore,
    SearchQuery,
    ShareMode,
)
from memory_core_test_helpers import create_private, make_draft


def test_public_enums_have_stable_storage_values():
    assert {item.name: item.value for item in ShareMode} == {
        "PRIVATE": "private",
        "SHARED": "shared",
    }
    assert {item.name: item.value for item in EntryState} == {
        "ACTIVE": "active",
        "DELETED": "deleted",
    }
    assert {item.name: item.value for item in ChangeKind} == {
        "CREATE": "create",
        "UPDATE": "update",
        "CORRECT": "correct",
        "DELETE": "delete",
    }


def test_create_returns_complete_record_and_change(store, owner_a, clock):
    result = create_private(store, owner_a)
    record = result.record

    assert result.replayed is False
    assert result.change_seq == 1
    assert isinstance(record.memory_id, str) and record.memory_id
    assert record.share_mode is ShareMode.PRIVATE
    assert record.place_key == "owner-a"
    assert record.owner_key == "owner-a"
    assert record.title == "Coffee preference"
    assert record.body == "Prefers washed Yunnan coffee."
    assert record.kind == "preference"
    assert record.tags == ("coffee", "yunnan")
    assert record.metadata == {"confidence": 0.9}
    assert record.revision == 1
    assert record.state is EntryState.ACTIVE
    assert record.created_ms == clock.now_ms
    assert record.updated_ms == clock.now_ms
    assert record.deleted_ms is None


def test_restart_preserves_reads_lists_search_and_changes(db_path, clock, owner_a):
    first = MemoryStore(db_path, clock=clock)
    created = create_private(first, owner_a, body="Durable Yunnan coffee note")

    reopened = MemoryStore(db_path, clock=clock)

    assert reopened.read(owner_a, created.record.memory_id) == created.record
    assert reopened.list(owner_a) == (created.record,)
    hits = reopened.search(owner_a, SearchQuery(text="Yunnan"))
    assert tuple(hit.record for hit in hits) == (created.record,)
    assert all(isinstance(hit.score, float) for hit in hits)
    changes = reopened.changes(owner_a, created.record.memory_id)
    assert len(changes) == 1
    assert changes[0].change_seq == created.change_seq
    assert changes[0].change_kind is ChangeKind.CREATE


def test_list_filters_by_kind_and_limit(store, owner_a):
    first = create_private(store, owner_a, op_key="one", title="One", kind="note").record
    create_private(store, owner_a, op_key="two", title="Two", kind="decision")
    third = create_private(store, owner_a, op_key="three", title="Three", kind="note").record

    records = store.list(owner_a, ListQuery(kinds=("note",), limit=2))

    assert len(records) == 2
    assert {record.memory_id for record in records} == {first.memory_id, third.memory_id}
    assert all(record.kind == "note" for record in records)


def test_default_query_objects_are_accepted(store, owner_a):
    record = create_private(store, owner_a).record

    assert store.list(owner_a, ListQuery()) == (record,)
    assert tuple(hit.record for hit in store.search(owner_a, SearchQuery(text="coffee"))) == (
        record,
    )


def test_patch_requires_at_least_one_changed_field():
    with pytest.raises(InvalidMemoryError):
        MemoryPatch()


def test_context_rejects_write_keys_without_matching_read_keys():
    with pytest.raises(InvalidMemoryError):
        AccessContext(
            owner_key="owner-a",
            actor_key="actor-a",
            shared_write_keys=frozenset({"team-blue"}),
        )


@pytest.mark.parametrize("bad_text", ["bad\0text", "bad\ud800text"])
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda value: AccessContext(value, "actor"), id="context-owner"),
        pytest.param(lambda value: AccessContext("owner", value), id="context-actor"),
        pytest.param(lambda value: MemoryPlace.shared(value), id="shared-key"),
        pytest.param(lambda value: MemoryDraft(value, "body"), id="draft-title"),
        pytest.param(lambda value: MemoryDraft("title", value), id="draft-body"),
        pytest.param(
            lambda value: MemoryDraft("title", "body", kind=value),
            id="draft-kind",
        ),
        pytest.param(
            lambda value: MemoryDraft("title", "body", tags=(value,)),
            id="draft-tag",
        ),
        pytest.param(
            lambda value: MemoryDraft("title", "body", metadata={"nested": value}),
            id="draft-metadata",
        ),
        pytest.param(lambda value: SearchQuery(text=value), id="search-text"),
    ],
)
def test_text_contract_rejects_nul_and_unpaired_surrogates(
    bad_text: str,
    build: Callable[[str], object],
):
    with pytest.raises(InvalidMemoryError):
        build(bad_text)


@pytest.mark.parametrize("bad_text", ["bad\0key", "bad\ud800key"])
def test_operation_key_rejects_invalid_text(store, owner_a, bad_text):
    with pytest.raises(InvalidMemoryError):
        store.create(owner_a, MemoryPlace.private(), make_draft(), op_key=bad_text)


@pytest.mark.parametrize("bad_epoch", [-1, 1 << 63, True, 1.5, "1700000000000"])
def test_clock_must_return_a_signed_64_bit_epoch(tmp_path, bad_epoch):
    with pytest.raises(InvalidMemoryError):
        MemoryStore(tmp_path / f"bad-clock-{type(bad_epoch).__name__}.db", clock=lambda: bad_epoch)


@pytest.mark.parametrize("bad_timeout", [-1, 0, True, 1.5, "10"])
def test_busy_timeout_requires_a_positive_integer(tmp_path, bad_timeout):
    with pytest.raises(InvalidMemoryError):
        MemoryStore(tmp_path / "bad-timeout.db", busy_timeout_ms=bad_timeout)
