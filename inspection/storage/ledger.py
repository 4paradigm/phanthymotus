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
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_segments_state_created ON segments(state, created_at_ns)")
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

    def summary(self, *, instance_id: str | None = None) -> dict[str, int]:
        where = ""
        values: tuple[str, ...] = ()
        if instance_id is not None:
            where = " WHERE instance_id=?"
            values = (instance_id,)
        with self._lock:
            rows = self._db.execute(
                f"SELECT state, COUNT(*) AS count, COALESCE(SUM(size), 0) AS bytes FROM segments{where} GROUP BY state",
                values,
            ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        local_bytes = sum(int(row["bytes"]) for row in rows if row["state"] != SegmentState.PURGED_LOCAL.value)
        return {
            "local_bytes": local_bytes,
            "finalized_segments": sum(counts.values()),
            "upload_backlog": counts.get(SegmentState.FINALIZED.value, 0) + counts.get(SegmentState.UPLOADING.value, 0),
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
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM instance_state WHERE card_id=? AND desired_state='recording' ORDER BY updated_at_ns",
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

    def transition(self, segment_id: str, state: SegmentState, *, error: str = "") -> None:
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE segments SET state=?, last_error=?, updated_at_ns=? WHERE segment_id=?",
                (state.value, error, time.time_ns(), segment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(segment_id)

    def reset_uploading_for_recovery(self) -> int:
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE segments SET state=?, updated_at_ns=? WHERE state=?",
                (SegmentState.FINALIZED.value, time.time_ns(), SegmentState.UPLOADING.value),
            )
            return int(cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._db.close()
