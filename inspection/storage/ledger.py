from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import SegmentRecord, SegmentState


class SegmentLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    object_key TEXT NOT NULL DEFAULT '',
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_retry_at_ns INTEGER NOT NULL DEFAULT 0,
                    created_at_ns INTEGER NOT NULL,
                    updated_at_ns INTEGER NOT NULL,
                    uploaded_at_ns INTEGER,
                    verified_at_ns INTEGER,
                    multipart_upload_id TEXT NOT NULL DEFAULT '',
                    multipart_started_at_ns INTEGER,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {str(row["name"]) for row in self._db.execute("PRAGMA table_info(segments)")}
            if "next_retry_at_ns" not in columns:
                self._db.execute(
                    "ALTER TABLE segments ADD COLUMN next_retry_at_ns INTEGER NOT NULL DEFAULT 0"
                )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_upload_ready "
                "ON segments(state, next_retry_at_ns, created_at_ns)"
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS instance_state (
                    card_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    input_topic TEXT NOT NULL,
                    desired_state TEXT NOT NULL,
                    auto_resume INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    updated_at_ns INTEGER NOT NULL,
                    PRIMARY KEY(card_id, instance_id)
                )
                """
            )
            instance_columns = {
                str(row["name"]) for row in self._db.execute("PRAGMA table_info(instance_state)")
            }
            for name, definition in (
                ("runtime_state", "TEXT NOT NULL DEFAULT ''"),
                ("last_error", "TEXT NOT NULL DEFAULT ''"),
                ("error_kind", "TEXT NOT NULL DEFAULT ''"),
                ("error_at_ns", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in instance_columns:
                    self._db.execute(f"ALTER TABLE instance_state ADD COLUMN {name} {definition}")

    @property
    def journal_mode(self) -> str:
        with self._lock:
            return str(self._db.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    @property
    def synchronous(self) -> int:
        with self._lock:
            return int(self._db.execute("PRAGMA synchronous").fetchone()[0])

    def upsert_finalized(self, record: SegmentRecord) -> None:
        now = time.time_ns()
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO segments (
                    segment_id, kind, card_id, instance_id, local_path, metadata_path,
                    object_key, size, sha256, state, created_at_ns, updated_at_ns, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    local_path=excluded.local_path,
                    metadata_path=excluded.metadata_path,
                    size=excluded.size,
                    sha256=excluded.sha256,
                    updated_at_ns=excluded.updated_at_ns,
                    last_error=''
                """,
                (
                    record.segment_id, record.kind, record.card_id, record.instance_id,
                    record.local_path, record.metadata_path, record.object_key,
                    record.size, record.sha256, SegmentState.FINALIZED.value,
                    record.created_at_ns or now, now, record.last_error,
                ),
            )

    def get(self, segment_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM segments WHERE segment_id=?", (segment_id,)).fetchone()
            return dict(row) if row else None

    def count(self, *states: SegmentState) -> int:
        with self._lock:
            if not states:
                return int(self._db.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
            placeholders = ",".join("?" for _ in states)
            values = tuple(state.value for state in states)
            return int(self._db.execute(f"SELECT COUNT(*) FROM segments WHERE state IN ({placeholders})", values).fetchone()[0])

    def list_by_state(self, *states: SegmentState) -> list[dict]:
        if not states:
            return []
        placeholders = ",".join("?" for _ in states)
        values = tuple(state.value for state in states)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM segments WHERE state IN ({placeholders}) ORDER BY created_at_ns, segment_id",
                values,
            ).fetchall()
            return [dict(row) for row in rows]

    def summary(self, *, card_id: str | None = None, instance_id: str | None = None) -> dict[str, int]:
        clauses: list[str] = []
        values_list: list[str] = []
        if card_id is not None:
            clauses.append("card_id=?")
            values_list.append(card_id)
        if instance_id is not None:
            clauses.append("instance_id=?")
            values_list.append(instance_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        values = tuple(values_list)
        with self._lock:
            rows = self._db.execute(
                f"SELECT state, COUNT(*) AS count, COALESCE(SUM(size), 0) AS bytes FROM segments{where} GROUP BY state",
                values,
            ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        bytes_by_state = {str(row["state"]): int(row["bytes"]) for row in rows}
        local_bytes = sum(int(row["bytes"]) for row in rows if row["state"] != SegmentState.PURGED_LOCAL.value)
        upload_states = (SegmentState.FINALIZED.value, SegmentState.UPLOADING.value)
        return {
            "local_bytes": local_bytes,
            "finalized_segments": sum(counts.values()),
            "upload_backlog": sum(counts.get(state, 0) for state in upload_states),
            "upload_backlog_bytes": sum(bytes_by_state.get(state, 0) for state in upload_states),
            "uploaded_verified": counts.get(SegmentState.UPLOADED_VERIFIED.value, 0),
        }

    def set_instance_state(
        self,
        *,
        card_id: str,
        instance_id: str,
        input_topic: str,
        desired_state: str,
        auto_resume: bool,
        session_id: str,
        config: dict,
    ) -> None:
        config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO instance_state (
                    card_id, instance_id, input_topic, desired_state, auto_resume,
                    session_id, config_json, updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id, instance_id) DO UPDATE SET
                    input_topic=excluded.input_topic,
                    desired_state=excluded.desired_state,
                    auto_resume=excluded.auto_resume,
                    session_id=excluded.session_id,
                    config_json=excluded.config_json,
                    updated_at_ns=excluded.updated_at_ns
                """,
                (
                    card_id, instance_id, input_topic, desired_state, int(auto_resume),
                    session_id, config_json, time.time_ns(),
                ),
            )

    def list_desired_recording(self, *, card_id: str) -> list[dict]:
        return [row for row in self.list_instance_states(card_id=card_id) if row["desired_state"] == "recording"]

    def set_instance_error(
        self,
        *,
        card_id: str,
        instance_id: str,
        runtime_state: str,
        last_error: str,
        error_kind: str,
    ) -> None:
        error_at_ns = time.time_ns() if last_error else 0
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO instance_state (
                    card_id, instance_id, input_topic, desired_state, auto_resume,
                    session_id, config_json, updated_at_ns,
                    runtime_state, last_error, error_kind, error_at_ns
                ) VALUES (?, ?, '', 'idle', 0, '', '{}', ?, ?, ?, ?, ?)
                ON CONFLICT(card_id, instance_id) DO UPDATE SET
                    runtime_state=excluded.runtime_state,
                    last_error=excluded.last_error,
                    error_kind=excluded.error_kind,
                    error_at_ns=excluded.error_at_ns,
                    updated_at_ns=excluded.updated_at_ns
                """,
                (
                    card_id,
                    instance_id,
                    time.time_ns(),
                    runtime_state,
                    last_error,
                    error_kind,
                    error_at_ns,
                ),
            )

    def clear_instance_error(self, *, card_id: str, instance_id: str) -> None:
        self.set_instance_error(
            card_id=card_id,
            instance_id=instance_id,
            runtime_state="",
            last_error="",
            error_kind="",
        )

    def list_instance_states(self, *, card_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM instance_state WHERE card_id=? ORDER BY updated_at_ns",
                (card_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["config"] = json.loads(item.pop("config_json"))
            except (TypeError, json.JSONDecodeError):
                item["config"] = {}
                item.pop("config_json", None)
            item["auto_resume"] = bool(item["auto_resume"])
            result.append(item)
        return result

    def claim_next_upload(self, *, card_id: str) -> dict | None:
        now = time.time_ns()
        with self._lock, self._db:
            row = self._db.execute(
                """
                SELECT * FROM segments
                WHERE card_id=? AND state=? AND next_retry_at_ns<=?
                ORDER BY created_at_ns, segment_id
                LIMIT 1
                """,
                (card_id, SegmentState.FINALIZED.value, now),
            ).fetchone()
            if row is None:
                return None
            cursor = self._db.execute(
                """
                UPDATE segments
                SET state=?, attempts=attempts+1, next_retry_at_ns=0,
                    updated_at_ns=?, last_error=''
                WHERE segment_id=? AND state=? AND next_retry_at_ns<=?
                """,
                (
                    SegmentState.UPLOADING.value, now, row["segment_id"],
                    SegmentState.FINALIZED.value, now,
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = dict(row)
            claimed["state"] = SegmentState.UPLOADING.value
            claimed["attempts"] = int(claimed["attempts"]) + 1
            return claimed

    def mark_upload_verified(self, segment_id: str, *, object_key: str) -> None:
        now = time.time_ns()
        with self._lock, self._db:
            cursor = self._db.execute(
                """
                UPDATE segments
                SET state=?, object_key=?, uploaded_at_ns=COALESCE(uploaded_at_ns, ?),
                    verified_at_ns=?, next_retry_at_ns=0, updated_at_ns=?, last_error=''
                WHERE segment_id=?
                """,
                (SegmentState.UPLOADED_VERIFIED.value, object_key, now, now, now, segment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(segment_id)

    def mark_upload_retry(self, segment_id: str, *, error: str, retry_after_seconds: float) -> None:
        now = time.time_ns()
        retry_at = now + int(max(0.0, float(retry_after_seconds)) * 1_000_000_000)
        with self._lock, self._db:
            cursor = self._db.execute(
                """
                UPDATE segments
                SET state=?, next_retry_at_ns=?, last_error=?, updated_at_ns=?
                WHERE segment_id=?
                """,
                (SegmentState.FINALIZED.value, retry_at, error, now, segment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(segment_id)

    def mark_conflict(self, segment_id: str, *, error: str) -> None:
        self.transition(segment_id, SegmentState.CONFLICT, error=error)

    def list_cleanup_candidates(
        self,
        *,
        card_id: str,
        instance_id: str,
        include_unuploaded: bool = False,
    ) -> list[dict]:
        states = [SegmentState.RETENTION_ELIGIBLE.value, SegmentState.UPLOADED_VERIFIED.value]
        if include_unuploaded:
            states.append(SegmentState.FINALIZED.value)
        placeholders = ", ".join("?" for _ in states)
        with self._lock:
            rows = self._db.execute(
                f"""
                SELECT * FROM segments
                WHERE card_id=? AND instance_id=? AND state IN ({placeholders})
                ORDER BY created_at_ns, segment_id
                """,
                (card_id, instance_id, *states),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_retention_eligible(self, segment_id: str) -> None:
        self.transition(segment_id, SegmentState.RETENTION_ELIGIBLE)

    def mark_purged_local(self, segment_id: str) -> None:
        self.transition(segment_id, SegmentState.PURGED_LOCAL)

    def transition(self, segment_id: str, state: SegmentState, *, error: str = "") -> None:
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE segments SET state=?, last_error=?, updated_at_ns=? WHERE segment_id=?",
                (state.value, error, time.time_ns(), segment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(segment_id)

    def reset_uploading_for_recovery(self, *, card_id: str | None = None) -> int:
        where = "state=?"
        values = (
            SegmentState.FINALIZED.value,
            time.time_ns(),
            SegmentState.UPLOADING.value,
        )
        if card_id is not None:
            where += " AND card_id=?"
            values += (card_id,)
        with self._lock, self._db:
            cursor = self._db.execute(
                f"UPDATE segments SET state=?, next_retry_at_ns=0, updated_at_ns=? WHERE {where}",
                values,
            )
            return int(cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._db.close()
