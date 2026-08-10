from __future__ import annotations

import asyncio
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import pytest

import config
from teleop import authority_guard
from teleop.authority_guard import (
    MAX_DRIVER_EPOCH,
    MAX_SQLITE_INTEGER,
    AuthorityGuard,
    AuthorityGuardNotFound,
    AuthorityGuardSchemaError,
    AuthorityGuardStateConflict,
)

SESSION_A = "11111111-1111-4111-8111-111111111111"
SESSION_B = "22222222-2222-4222-8222-222222222222"
BOOT_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def guard_values(index: int = 1, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "robot_id": f"robot-{index}",
        "driver_id": f"driver-{index}",
        "session_id": SESSION_A if index == 1 else SESSION_B,
        "boot_id": BOOT_A,
        "epoch": index,
        "capability_digest": "a" * 64,
        "target_fingerprint": "b" * 64,
        "dispatch_generation": index,
        "phase": "preparing",
        "created_at": 100.0 + index,
        "updated_at": 100.0 + index,
    }
    values.update(overrides)
    return values


def make_guard(index: int = 1, **overrides: object) -> AuthorityGuard:
    return AuthorityGuard(**guard_values(index, **overrides))


def test_guard_round_trip_list_order_and_thread_entrypoints():
    async def scenario():
        second = make_guard(2)
        first = make_guard(1)
        await asyncio.to_thread(authority_guard.create_guard, second)
        await asyncio.to_thread(authority_guard.create_guard, first)

        loaded = await asyncio.to_thread(authority_guard.get_guard, first.robot_id)
        listed = await asyncio.to_thread(authority_guard.list_guards)
        missing = await asyncio.to_thread(authority_guard.get_guard, "robot-missing")
        return loaded, listed, missing

    loaded, listed, missing = asyncio.run(scenario())

    assert loaded == make_guard(1)
    assert listed == [make_guard(1), make_guard(2)]
    assert missing is None


def test_insert_never_replaces_robot_driver_or_session_aliases():
    original = authority_guard.create_guard(make_guard())

    aliases = [
        make_guard(
            2,
            robot_id=original.robot_id,
            session_id=SESSION_B,
        ),
        make_guard(
            2,
            driver_id=original.driver_id,
            session_id=SESSION_B,
        ),
        make_guard(
            2,
            session_id=original.session_id,
        ),
    ]
    for alias in aliases:
        with pytest.raises(sqlite3.IntegrityError):
            authority_guard.create_guard(alias)

    assert authority_guard.list_guards() == [original]


def test_concurrent_alias_inserts_have_one_winner_and_do_not_replace():
    candidates = [
        make_guard(
            1,
            robot_id="shared-robot",
            driver_id=f"driver-{index}",
            session_id=f"00000000-0000-4000-8000-{index:012d}",
            epoch=index,
        )
        for index in range(1, 9)
    ]

    def insert(candidate: AuthorityGuard) -> tuple[str, AuthorityGuard | None]:
        try:
            return "created", authority_guard.create_guard(candidate)
        except sqlite3.IntegrityError:
            return "conflict", None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(insert, candidates))

    assert [result[0] for result in results].count("created") == 1
    assert [result[0] for result in results].count("conflict") == 7
    stored = authority_guard.list_guards()
    assert len(stored) == 1
    assert stored[0] in candidates


def test_update_and_delete_compare_the_exact_robot_session_pair(monkeypatch):
    first = authority_guard.create_guard(make_guard(1))
    second = authority_guard.create_guard(make_guard(2))

    with pytest.raises(AuthorityGuardNotFound):
        authority_guard.update_guard(
            first.robot_id,
            second.session_id,
            phase="active",
            dispatch_generation=2,
        )
    assert authority_guard.delete_guard(first.robot_id, second.session_id) is False
    assert authority_guard.get_guard(first.robot_id) == first
    assert authority_guard.get_guard(second.robot_id) == second

    updated = authority_guard.update_guard(
        first.robot_id,
        first.session_id,
        phase="active",
        dispatch_generation=2,
        updated_at=110.0,
    )
    assert updated.phase == "active"
    assert updated.dispatch_generation == 2
    assert updated.updated_at == 110.0
    assert updated.created_at == first.created_at

    monkeypatch.setattr(authority_guard.time, "time", lambda: 1.0)
    after_clock_rollback = authority_guard.update_guard(
        first.robot_id,
        first.session_id,
        phase="stopping",
        dispatch_generation=3,
    )
    assert after_clock_rollback.updated_at == 110.0

    assert authority_guard.delete_guard(first.robot_id, first.session_id) is True
    assert authority_guard.delete_guard(first.robot_id, first.session_id) is False
    assert authority_guard.list_guards() == [second]


def test_update_rejects_generation_and_timestamp_regression_atomically():
    original = authority_guard.create_guard(
        make_guard(dispatch_generation=10, updated_at=200.0)
    )

    with pytest.raises(AuthorityGuardStateConflict, match="dispatch_generation"):
        authority_guard.update_guard(
            original.robot_id,
            original.session_id,
            phase="active",
            dispatch_generation=9,
            updated_at=201.0,
        )
    with pytest.raises(AuthorityGuardStateConflict, match="updated_at"):
        authority_guard.update_guard(
            original.robot_id,
            original.session_id,
            phase="active",
            dispatch_generation=11,
            updated_at=199.0,
        )

    assert authority_guard.get_guard(original.robot_id) == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"robot_id": ""},
        {"robot_id": "r" * 65},
        {"robot_id": "robot/unsafe"},
        {"driver_id": ""},
        {"driver_id": " driver"},
        {"session_id": "not-a-uuid"},
        {"session_id": "00000000-0000-0000-0000-000000000000"},
        {"session_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"},
        {"boot_id": "not-a-uuid"},
        {"capability_digest": "A" * 64},
        {"capability_digest": "a" * 63},
        {"target_fingerprint": "g" * 64},
        {"epoch": True},
        {"epoch": 0},
        {"epoch": MAX_DRIVER_EPOCH + 1},
        {"dispatch_generation": False},
        {"dispatch_generation": -1},
        {"dispatch_generation": MAX_SQLITE_INTEGER + 1},
        {"phase": "released"},
        {"created_at": math.nan},
        {"created_at": math.inf},
        {"created_at": -1},
        {"updated_at": 0},
    ],
)
def test_guard_rejects_invalid_or_out_of_range_fields(overrides):
    with pytest.raises((TypeError, ValueError)):
        make_guard(**overrides)


def test_mutation_rolls_back_when_sqlite_rejects_the_write():
    original = authority_guard.create_guard(make_guard())
    with config._get_conn() as conn:
        conn.execute("""
            CREATE TRIGGER reject_guard_updates
            BEFORE UPDATE ON teleop_authority_guards
            BEGIN
                SELECT RAISE(ABORT, 'simulated durable write failure');
            END
        """)
        conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="durable write failure"):
        authority_guard.update_guard(
            original.robot_id,
            original.session_id,
            phase="active",
            dispatch_generation=2,
            updated_at=150.0,
        )

    assert authority_guard.get_guard(original.robot_id) == original


@pytest.mark.parametrize(
    "operation",
    [
        lambda guard: authority_guard.create_guard(guard),
        lambda guard: authority_guard.get_guard(guard.robot_id),
        lambda guard: authority_guard.list_guards(),
        lambda guard: authority_guard.update_guard(
            guard.robot_id,
            guard.session_id,
            phase="active",
            dispatch_generation=2,
        ),
        lambda guard: authority_guard.delete_guard(guard.robot_id, guard.session_id),
    ],
)
def test_database_open_failures_propagate_fail_closed(monkeypatch, operation):
    def unavailable():
        raise OSError("database unavailable")

    monkeypatch.setattr(config, "_get_conn", unavailable)
    with pytest.raises(OSError, match="database unavailable"):
        operation(make_guard())


def test_corrupt_raw_row_causes_reads_to_fail_instead_of_appearing_unlocked():
    guard = authority_guard.create_guard(make_guard())
    with config._get_conn() as conn:
        conn.execute(
            """UPDATE teleop_authority_guards
               SET target_fingerprint=? WHERE robot_id=?""",
            ("not-a-fingerprint", guard.robot_id),
        )
        conn.commit()

    with pytest.raises(ValueError, match="target_fingerprint"):
        authority_guard.get_guard(guard.robot_id)
    with pytest.raises(ValueError, match="target_fingerprint"):
        authority_guard.list_guards()


def test_incompatible_preexisting_schema_is_rejected():
    with config._get_conn() as conn:
        conn.execute("""
            CREATE TABLE teleop_authority_guards (
                robot_id TEXT PRIMARY KEY,
                fence TEXT
            )
        """)
        conn.commit()

    with pytest.raises(AuthorityGuardSchemaError):
        authority_guard.list_guards()


def test_raw_schema_and_dataclass_contain_only_secret_free_guard_evidence():
    authority_guard.create_guard(make_guard())

    expected_fields = [
        "robot_id",
        "driver_id",
        "session_id",
        "boot_id",
        "epoch",
        "capability_digest",
        "target_fingerprint",
        "dispatch_generation",
        "phase",
        "created_at",
        "updated_at",
    ]
    assert [field.name for field in fields(AuthorityGuard)] == expected_fields

    with config._get_conn() as conn:
        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(teleop_authority_guards)"
            ).fetchall()
        ]
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            ("teleop_authority_guards",),
        ).fetchone()[0]
        row = conn.execute("SELECT * FROM teleop_authority_guards").fetchone()

    assert columns == expected_fields
    normalized = " ".join(columns).lower()
    schema_lower = create_sql.lower()
    for secret_name in (
        "fence",
        "token",
        "client",
        "principal",
        "credential",
        "password",
    ):
        assert secret_name not in normalized
        assert secret_name not in schema_lower
    assert len(row) == len(expected_fields)


def test_raw_schema_enforces_phase_and_integer_ranges():
    authority_guard.create_guard(make_guard())
    with config._get_conn() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """UPDATE teleop_authority_guards SET phase='released'
                   WHERE robot_id='robot-1' """
            )
        conn.rollback()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """UPDATE teleop_authority_guards SET dispatch_generation=-1
                   WHERE robot_id='robot-1' """
            )
        conn.rollback()

    assert authority_guard.get_guard("robot-1") == make_guard()
