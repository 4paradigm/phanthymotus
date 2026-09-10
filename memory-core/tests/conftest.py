"""Shared deterministic fixtures for the public MemoryStore contract."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from memory_core import AccessContext, MemoryStore


@dataclass
class ManualClock:
    now_ms: int = 1_700_000_000_000

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, milliseconds: int = 1) -> int:
        self.now_ms += milliseconds
        return self.now_ms


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory.db"


@pytest.fixture
def store(db_path, clock):
    return MemoryStore(db_path, clock=clock)


@pytest.fixture
def owner_a() -> AccessContext:
    return AccessContext(owner_key="owner-a", actor_key="actor-a")


@pytest.fixture
def owner_b() -> AccessContext:
    return AccessContext(owner_key="owner-b", actor_key="actor-b")


@pytest.fixture
def shared_writer() -> AccessContext:
    keys = frozenset({"team-blue"})
    return AccessContext(
        owner_key="owner-a",
        actor_key="writer-a",
        shared_read_keys=keys,
        shared_write_keys=keys,
    )


@pytest.fixture
def shared_reader() -> AccessContext:
    return AccessContext(
        owner_key="owner-b",
        actor_key="reader-b",
        shared_read_keys=frozenset({"team-blue"}),
    )
