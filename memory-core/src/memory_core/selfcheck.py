"""Run a deterministic local smoke scenario for the Memory Core kernel."""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path

from .models import AccessContext, MemoryDraft, MemoryPlace, SearchQuery
from .repository import MemoryStore


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Memory Core self-check failed: {message}")


def _run(database_path: Path) -> dict[str, object]:
    run_id = uuid.uuid4().hex
    owner_a = AccessContext(
        owner_key=f"selfcheck-owner-a-{run_id}",
        actor_key=f"selfcheck-actor-a-{run_id}",
    )
    owner_b = AccessContext(
        owner_key=f"selfcheck-owner-b-{run_id}",
        actor_key=f"selfcheck-actor-b-{run_id}",
    )

    store = MemoryStore(database_path)
    draft = MemoryDraft(
        title="饮品偏好",
        body="喝咖啡不加糖",
        kind="preference",
        tags=("咖啡",),
    )
    created = store.create(
        owner_a,
        MemoryPlace.private(),
        draft,
        op_key=f"{run_id}:create",
    )
    replay = store.create(
        owner_a,
        MemoryPlace.private(),
        draft,
        op_key=f"{run_id}:create",
    )
    _require(
        replay.replayed
        and replay.record == created.record
        and replay.change_seq == created.change_seq,
        "idempotent replay",
    )
    _require(
        [hit.record.memory_id for hit in store.search(owner_a, SearchQuery("咖啡"))]
        == [created.record.memory_id],
        "owner recall",
    )
    _require(store.search(owner_b, SearchQuery("咖啡")) == (), "owner isolation")

    # A new store instance proves the state is on disk rather than process-local.
    store = MemoryStore(database_path)
    _require(store.read(owner_a, created.record.memory_id) == created.record, "restart recovery")
    corrected = store.correct(
        owner_a,
        created.record.memory_id,
        MemoryDraft(
            title="饮品偏好",
            body="最近改喝茶",
            kind="preference",
            tags=("茶",),
        ),
        expected_revision=created.record.revision,
        op_key=f"{run_id}:correct",
        reason="manual_fix",
    )
    _require(store.search(owner_a, SearchQuery("咖啡")) == (), "old text exclusion")
    _require(
        [hit.record.memory_id for hit in store.search(owner_a, SearchQuery("喝茶"))]
        == [corrected.record.memory_id],
        "corrected recall",
    )

    deleted = store.delete(
        owner_a,
        corrected.record.memory_id,
        expected_revision=corrected.record.revision,
        op_key=f"{run_id}:delete",
        reason="user_request",
    )
    _require(store.search(owner_a, SearchQuery("喝茶")) == (), "logical deletion")
    _require(
        store.read(owner_a, deleted.record.memory_id, include_deleted=True) == deleted.record,
        "deleted record inspection",
    )
    changes = store.changes(owner_a, deleted.record.memory_id)
    _require(len(changes) == 3, "change history completeness")
    integrity = store.self_check()
    return {
        "status": "ok",
        "schema_version": integrity["schema_version"],
        "changes": len(changes),
        "journal_mode": integrity["journal_mode"],
        "owner_isolation": "ok",
        "restart_recovery": "ok",
        "correction": "ok",
        "logical_deletion": "ok",
        "idempotency": "ok",
    }


def run_selfcheck(database_path: str | Path | None = None) -> dict[str, object]:
    if database_path is not None:
        return _run(Path(database_path))
    with tempfile.TemporaryDirectory(prefix="phanthy-memory-selfcheck-") as directory:
        return _run(Path(directory) / "memory.db")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="optional persistent self-check database")
    args = parser.parse_args(argv)
    print(json.dumps(run_selfcheck(args.db), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
