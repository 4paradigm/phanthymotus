"""Small black-box helpers shared by MemoryStore contract tests."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from memory_core import MemoryDraft, MemoryPlace


def make_draft(**changes: Any) -> MemoryDraft:
    values: dict[str, Any] = {
        "title": "Coffee preference",
        "body": "Prefers washed Yunnan coffee.",
        "kind": "preference",
        "tags": ("coffee", "yunnan"),
        "metadata": {"confidence": 0.9},
    }
    values.update(changes)
    return MemoryDraft(**values)


def create_private(store, context, *, op_key: str = "create-1", **changes: Any):
    return store.create(
        context,
        MemoryPlace.private(),
        make_draft(**changes),
        op_key=op_key,
    )


def create_shared(
    store,
    context,
    *,
    key: str = "team-blue",
    op_key: str = "create-1",
    **changes: Any,
):
    return store.create(
        context,
        MemoryPlace.shared(key),
        make_draft(**changes),
        op_key=op_key,
    )


def run_threads(count: int, operation: Callable[[int], Any]):
    barrier = threading.Barrier(count)
    results: list[Any] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def run(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            value = operation(index)
            with result_lock:
                results.append(value)
        except BaseException as exc:  # noqa: BLE001 - returned for explicit assertions
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not any(thread.is_alive() for thread in threads)
    return results, errors
