"""Scope-safe SQLite repository for Phanthymotus memory entries."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

from .database import Clock, MemoryDatabase, epoch_ms
from .errors import (
    IdempotencyConflictError,
    InvalidMemoryError,
    MemoryCoreError,
    MemoryNotFoundError,
    RevisionConflictError,
    SearchUnavailableError,
    SharedPlaceDeniedError,
    StorageDamagedError,
)
from .models import (
    MAX_EPOCH_MS,
    AccessContext,
    ChangeKind,
    EntryState,
    ListQuery,
    MemoryChange,
    MemoryDraft,
    MemoryPatch,
    MemoryPlace,
    MemoryRecord,
    SearchHit,
    SearchQuery,
    ShareMode,
    WriteResult,
    canonical_json,
    normalize_audit_label,
    normalize_identifier,
    validate_epoch_ms,
    validate_limit,
    validate_positive_int,
)

_ENTRY_FIELD_NAMES = (
    "memory_id",
    "share_mode",
    "place_key",
    "owner_key",
    "title",
    "body",
    "kind",
    "tags_json",
    "metadata_json",
    "revision",
    "state",
    "created_ms",
    "updated_ms",
    "deleted_ms",
)


def _entry_columns(alias: str = "entry") -> str:
    return ", ".join(f"{alias}.{name} AS {name}" for name in _ENTRY_FIELD_NAMES)


def _request_hash(operation: ChangeKind, payload: dict[str, Any]) -> str:
    encoded = canonical_json({"contract": 1, "operation": operation.value, **payload})
    return f"sha256:{sha256(encoded.encode('utf-8')).hexdigest()}"


def _new_memory_id() -> str:
    return f"memory_{uuid.uuid4().hex}"


def _operation_ref(op_key: object) -> str:
    normalized = normalize_identifier(op_key, "op_key")
    return f"op-sha256:{sha256(normalized.encode('utf-8')).hexdigest()}"


def _draft_payload(draft: MemoryDraft) -> dict[str, Any]:
    return {
        "body": draft.body,
        "kind": draft.kind,
        "metadata": draft.metadata,
        "tags": list(draft.tags),
        "title": draft.title,
    }


def _parse_json(value: object, field_name: str, expected_type: type[Any]) -> Any:
    if not isinstance(value, str):
        raise StorageDamagedError(f"database contains a non-text {field_name}")
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StorageDamagedError(f"database contains invalid {field_name}") from exc
    if not isinstance(decoded, expected_type):
        raise StorageDamagedError(f"database contains an invalid {field_name} shape")
    return decoded


def _record_to_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "body": record.body,
        "created_ms": record.created_ms,
        "deleted_ms": record.deleted_ms,
        "kind": record.kind,
        "memory_id": record.memory_id,
        "metadata": record.metadata,
        "owner_key": record.owner_key,
        "place_key": record.place_key,
        "revision": record.revision,
        "share_mode": record.share_mode.value,
        "state": record.state.value,
        "tags": list(record.tags),
        "title": record.title,
        "updated_ms": record.updated_ms,
    }


def _record_from_payload(payload: object) -> MemoryRecord:
    if not isinstance(payload, dict):
        raise StorageDamagedError("database contains an invalid idempotency result")
    try:
        return MemoryRecord(
            memory_id=payload["memory_id"],
            share_mode=payload["share_mode"],
            place_key=payload["place_key"],
            owner_key=payload["owner_key"],
            title=payload["title"],
            body=payload["body"],
            kind=payload["kind"],
            tags=tuple(payload["tags"]),
            metadata=payload["metadata"],
            revision=payload["revision"],
            state=payload["state"],
            created_ms=payload["created_ms"],
            updated_ms=payload["updated_ms"],
            deleted_ms=payload["deleted_ms"],
        )
    except (KeyError, TypeError, InvalidMemoryError) as exc:
        raise StorageDamagedError("database contains an invalid idempotency result") from exc


class MemoryStore:
    """Durable memory store with ownership checks in every record query."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        clock: Clock = epoch_ms,
    ) -> None:
        self._database = MemoryDatabase(
            path,
            busy_timeout_ms=busy_timeout_ms,
            clock=clock,
        )
        self._clock = clock
        try:
            self._database.initialize()
        except MemoryCoreError:
            raise
        except sqlite3.Error as exc:  # Defensive boundary for custom SQLite builds.
            raise StorageDamagedError("failed to initialize Memory Core") from exc

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        clock: Clock = epoch_ms,
    ) -> MemoryStore:
        return cls(path, busy_timeout_ms=busy_timeout_ms, clock=clock)

    @property
    def path(self) -> Path:
        return self._database.path

    @staticmethod
    def _require_context(context: AccessContext) -> None:
        if not isinstance(context, AccessContext):
            raise InvalidMemoryError("context must be an AccessContext")

    @staticmethod
    def _require_place(place: MemoryPlace) -> None:
        if not isinstance(place, MemoryPlace):
            raise InvalidMemoryError("place must be a MemoryPlace")

    @staticmethod
    def _require_draft(draft: MemoryDraft) -> None:
        if not isinstance(draft, MemoryDraft):
            raise InvalidMemoryError("draft must be a MemoryDraft")

    @staticmethod
    def _require_patch(patch: MemoryPatch) -> None:
        if not isinstance(patch, MemoryPatch):
            raise InvalidMemoryError("patch must be a MemoryPatch")

    def _now_ms(self) -> int:
        value = validate_epoch_ms(self._clock(), "clock")
        assert isinstance(value, int)
        return value

    def _next_ms(self, previous: int) -> int:
        current = self._now_ms()
        if current > previous:
            return current
        if previous == MAX_EPOCH_MS:
            raise InvalidMemoryError("clock cannot advance beyond the signed 64-bit epoch range")
        return previous + 1

    @staticmethod
    def _visibility_sql(context: AccessContext, alias: str = "entry") -> tuple[str, list[Any]]:
        private_clause = f"({alias}.share_mode = 'private' AND {alias}.owner_key = ?)"
        parameters: list[Any] = [context.owner_key]
        shared_keys = sorted(context.shared_read_keys)
        if not shared_keys:
            return private_clause, parameters
        placeholders = ", ".join("?" for _ in shared_keys)
        shared_clause = f"({alias}.share_mode = 'shared' AND {alias}.place_key IN ({placeholders}))"
        parameters.extend(shared_keys)
        return f"({private_clause} OR {shared_clause})", parameters

    @staticmethod
    def _resolved_place(
        context: AccessContext,
        place: MemoryPlace,
    ) -> tuple[ShareMode, str, str]:
        if place.share_mode is ShareMode.PRIVATE:
            return ShareMode.PRIVATE, context.owner_key, context.owner_key
        if place.place_key not in context.shared_write_keys:
            raise SharedPlaceDeniedError("shared place write access is required")
        return ShareMode.SHARED, place.place_key, context.owner_key

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
        tags = _parse_json(row["tags_json"], "tags_json", list)
        metadata = _parse_json(row["metadata_json"], "metadata_json", dict)
        if not all(isinstance(tag, str) for tag in tags):
            raise StorageDamagedError("database contains invalid tags_json values")
        try:
            return MemoryRecord(
                memory_id=row["memory_id"],
                share_mode=row["share_mode"],
                place_key=row["place_key"],
                owner_key=row["owner_key"],
                title=row["title"],
                body=row["body"],
                kind=row["kind"],
                tags=tuple(tags),
                metadata=metadata,
                revision=row["revision"],
                state=row["state"],
                created_ms=row["created_ms"],
                updated_ms=row["updated_ms"],
                deleted_ms=row["deleted_ms"],
            )
        except InvalidMemoryError as exc:
            raise StorageDamagedError("database contains an invalid memory entry") from exc

    @staticmethod
    def _change_from_row(row: sqlite3.Row) -> MemoryChange:
        try:
            return MemoryChange(
                change_seq=row["change_seq"],
                memory_id=row["memory_id"],
                actor_key=row["actor_key"],
                change_kind=row["change_kind"],
                from_revision=row["from_revision"],
                to_revision=row["to_revision"],
                reason=row["reason"],
                operation_ref=row["operation_ref"],
                changed_ms=row["changed_ms"],
            )
        except InvalidMemoryError as exc:
            raise StorageDamagedError("database contains an invalid memory change") from exc

    def _visible_row(
        self,
        connection: sqlite3.Connection,
        context: AccessContext,
        memory_id: str,
        *,
        include_deleted: bool,
    ) -> sqlite3.Row:
        visibility, parameters = self._visibility_sql(context)
        state_clause = "" if include_deleted else "AND entry.state = 'active'"
        row = connection.execute(
            f"""
            SELECT entry.row_id, {_entry_columns()}
            FROM memory_entries AS entry
            WHERE entry.memory_id = ?
              AND {visibility}
              {state_clause}
            """,
            [memory_id, *parameters],
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError("memory was not found")
        return row

    def _writable_row(
        self,
        connection: sqlite3.Connection,
        context: AccessContext,
        memory_id: str,
    ) -> sqlite3.Row:
        row = self._visible_row(
            connection,
            context,
            memory_id,
            include_deleted=True,
        )
        if (
            row["share_mode"] == ShareMode.SHARED.value
            and row["place_key"] not in context.shared_write_keys
        ):
            raise SharedPlaceDeniedError("shared place write access is required")
        return row

    @staticmethod
    def _receipt(
        connection: sqlite3.Connection,
        *,
        share_mode: ShareMode,
        place_key: str,
        operation_ref: str,
        operation: ChangeKind,
        request_hash: str,
    ) -> WriteResult | None:
        row = connection.execute(
            """
            SELECT request_hash, operation, result_memory_id, result_revision,
                   result_change_seq, result_json
            FROM memory_requests
            WHERE share_mode = ? AND place_key = ? AND operation_ref = ?
            """,
            (share_mode.value, place_key, operation_ref),
        ).fetchone()
        if row is None:
            return None
        if row["operation"] != operation.value or row["request_hash"] != request_hash:
            raise IdempotencyConflictError("operation key was reused with a different request")
        payload = _parse_json(row["result_json"], "result_json", dict)
        record = _record_from_payload(payload)
        if record.memory_id != row["result_memory_id"] or record.revision != row["result_revision"]:
            raise StorageDamagedError("idempotency receipt does not match its result")
        change_seq = int(row["result_change_seq"])
        return WriteResult(record=record, change_seq=change_seq, replayed=True)

    @staticmethod
    def _insert_change(
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        context: AccessContext,
        change_kind: ChangeKind,
        from_revision: int | None,
        to_revision: int,
        reason: str,
        operation_ref: str,
        changed_ms: int,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO memory_changes(
                memory_id, actor_key, change_kind, from_revision, to_revision,
                reason, operation_ref, changed_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                context.actor_key,
                change_kind.value,
                from_revision,
                to_revision,
                reason,
                operation_ref,
                changed_ms,
            ),
        )
        if cursor.lastrowid is None:
            raise StorageDamagedError("SQLite did not return a change sequence")
        return int(cursor.lastrowid)

    @staticmethod
    def _insert_receipt(
        connection: sqlite3.Connection,
        *,
        share_mode: ShareMode,
        place_key: str,
        operation_ref: str,
        request_hash: str,
        operation: ChangeKind,
        record: MemoryRecord,
        change_seq: int,
        created_ms: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_requests(
                share_mode, place_key, operation_ref, request_hash, operation,
                result_memory_id, result_revision, result_change_seq,
                result_json, created_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                share_mode.value,
                place_key,
                operation_ref,
                request_hash,
                operation.value,
                record.memory_id,
                record.revision,
                change_seq,
                canonical_json(_record_to_payload(record)),
                created_ms,
            ),
        )

    def create(
        self,
        context: AccessContext,
        place: MemoryPlace,
        draft: MemoryDraft,
        *,
        op_key: str,
    ) -> WriteResult:
        self._require_context(context)
        self._require_place(place)
        self._require_draft(draft)
        operation_ref = _operation_ref(op_key)
        share_mode, place_key, owner_key = self._resolved_place(context, place)
        fingerprint = _request_hash(
            ChangeKind.CREATE,
            {
                "actor_key": context.actor_key,
                "draft": _draft_payload(draft),
                "owner_key": owner_key,
                "place_key": place_key,
                "share_mode": share_mode.value,
            },
        )
        with self._database.write_transaction() as connection:
            replay = self._receipt(
                connection,
                share_mode=share_mode,
                place_key=place_key,
                operation_ref=operation_ref,
                operation=ChangeKind.CREATE,
                request_hash=fingerprint,
            )
            if replay is not None:
                return replay
            now_ms = self._now_ms()
            record = MemoryRecord(
                memory_id=_new_memory_id(),
                share_mode=share_mode,
                place_key=place_key,
                owner_key=owner_key,
                title=draft.title,
                body=draft.body,
                kind=draft.kind,
                tags=draft.tags,
                metadata=draft.metadata,
                revision=1,
                state=EntryState.ACTIVE,
                created_ms=now_ms,
                updated_ms=now_ms,
                deleted_ms=None,
            )
            connection.execute(
                """
                INSERT INTO memory_entries(
                    memory_id, share_mode, place_key, owner_key, title, body, kind,
                    tags_json, tags_text, metadata_json, revision, state,
                    created_ms, updated_ms, deleted_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.share_mode.value,
                    record.place_key,
                    record.owner_key,
                    record.title,
                    record.body,
                    record.kind,
                    canonical_json(list(record.tags)),
                    " ".join(record.tags),
                    canonical_json(record.metadata),
                    record.revision,
                    record.state.value,
                    record.created_ms,
                    record.updated_ms,
                    record.deleted_ms,
                ),
            )
            change_seq = self._insert_change(
                connection,
                memory_id=record.memory_id,
                context=context,
                change_kind=ChangeKind.CREATE,
                from_revision=None,
                to_revision=1,
                reason="",
                operation_ref=operation_ref,
                changed_ms=now_ms,
            )
            self._insert_receipt(
                connection,
                share_mode=share_mode,
                place_key=place_key,
                operation_ref=operation_ref,
                request_hash=fingerprint,
                operation=ChangeKind.CREATE,
                record=record,
                change_seq=change_seq,
                created_ms=now_ms,
            )
            return WriteResult(record=record, change_seq=change_seq, replayed=False)

    def read(
        self,
        context: AccessContext,
        memory_id: str,
        *,
        include_deleted: bool = False,
    ) -> MemoryRecord:
        self._require_context(context)
        memory_id = normalize_identifier(memory_id, "memory_id")
        if not isinstance(include_deleted, bool):
            raise InvalidMemoryError("include_deleted must be a bool")
        with self._database.read_connection() as connection:
            row = self._visible_row(
                connection,
                context,
                memory_id,
                include_deleted=include_deleted,
            )
            return self._record_from_row(row)

    def list(
        self,
        context: AccessContext,
        query: ListQuery | None = None,
    ) -> tuple[MemoryRecord, ...]:
        self._require_context(context)
        if query is None:
            query = ListQuery()
        if not isinstance(query, ListQuery):
            raise InvalidMemoryError("query must be a ListQuery")
        visibility, parameters = self._visibility_sql(context)
        conditions = [visibility]
        if not query.include_deleted:
            conditions.append("entry.state = 'active'")
        if query.kinds:
            placeholders = ", ".join("?" for _ in query.kinds)
            conditions.append(f"entry.kind IN ({placeholders})")
            parameters.extend(query.kinds)
        parameters.append(query.limit)
        with self._database.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_entry_columns()}
                FROM memory_entries AS entry
                WHERE {" AND ".join(conditions)}
                ORDER BY entry.updated_ms DESC, entry.row_id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return tuple(self._record_from_row(row) for row in rows)

    @staticmethod
    def _fts_expression(text: str) -> str:
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    @staticmethod
    def _like_pattern(text: str) -> str:
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    @staticmethod
    def _search_score(record: MemoryRecord, text: str) -> float:
        needle = text.casefold()
        title = record.title.casefold()
        body = record.body.casefold()
        tags = " ".join(record.tags).casefold()
        score = title.count(needle) * 3 + body.count(needle) + tags.count(needle) * 2
        if score:
            return float(score)
        tokens = [token.casefold() for token in re.findall(r"\w+", text, flags=re.UNICODE)]
        return float(
            sum(
                title.count(token) * 3 + body.count(token) + tags.count(token) * 2
                for token in tokens
            )
        )

    def search(
        self,
        context: AccessContext,
        query: SearchQuery,
    ) -> tuple[SearchHit, ...]:
        self._require_context(context)
        if not isinstance(query, SearchQuery):
            raise InvalidMemoryError("query must be a SearchQuery")
        visibility, visibility_parameters = self._visibility_sql(context)
        kind_condition = ""
        kind_parameters: list[Any] = []
        if query.kinds:
            placeholders = ", ".join("?" for _ in query.kinds)
            kind_condition = f"AND entry.kind IN ({placeholders})"
            kind_parameters.extend(query.kinds)
        candidate_limit = min(max(query.limit * 20, 100), 1_000)
        expression = self._fts_expression(query.text)
        safe_expression = expression or '"__memory_core_no_fts_token__"'
        pattern = self._like_pattern(query.text)
        try:
            with self._database.read_connection() as connection:
                rows = connection.execute(
                    f"""
                    WITH visible AS (
                        SELECT entry.*
                        FROM memory_entries AS entry
                        WHERE entry.state = 'active'
                          AND {visibility}
                          {kind_condition}
                    ),
                    matched AS (
                        SELECT visible.row_id
                        FROM visible
                        JOIN memory_fts ON memory_fts.rowid = visible.row_id
                        WHERE ? = 1 AND memory_fts MATCH ?

                        UNION

                        SELECT visible.row_id
                        FROM visible
                        WHERE visible.title LIKE ? ESCAPE '\\'
                           OR visible.body LIKE ? ESCAPE '\\'
                           OR visible.tags_text LIKE ? ESCAPE '\\'
                    )
                    SELECT {_entry_columns("visible")}
                    FROM visible
                    JOIN matched ON matched.row_id = visible.row_id
                    ORDER BY visible.updated_ms DESC, visible.row_id DESC
                    LIMIT ?
                    """,
                    [
                        *visibility_parameters,
                        *kind_parameters,
                        int(bool(expression)),
                        safe_expression,
                        pattern,
                        pattern,
                        pattern,
                        candidate_limit,
                    ],
                ).fetchall()
        except StorageDamagedError as exc:
            raise SearchUnavailableError("full-text memory search is unavailable") from exc

        hits = [
            SearchHit(record=record, score=self._search_score(record, query.text))
            for record in (self._record_from_row(row) for row in rows)
        ]
        hits.sort(
            key=lambda hit: (
                -hit.score,
                -hit.record.updated_ms,
                hit.record.memory_id,
            )
        )
        return tuple(hits[: query.limit])

    def _replace(
        self,
        context: AccessContext,
        memory_id: str,
        draft: MemoryDraft,
        *,
        expected_revision: int,
        op_key: str,
        reason: str,
        change_kind: ChangeKind,
        request_body: dict[str, Any],
    ) -> WriteResult:
        memory_id = normalize_identifier(memory_id, "memory_id")
        expected_revision = validate_positive_int(expected_revision, "expected_revision")
        operation_ref = _operation_ref(op_key)
        reason = normalize_audit_label(
            reason,
            required=change_kind is ChangeKind.CORRECT,
        )
        with self._database.write_transaction() as connection:
            row = self._writable_row(connection, context, memory_id)
            old = self._record_from_row(row)
            fingerprint = _request_hash(
                change_kind,
                {
                    "actor_key": context.actor_key,
                    "expected_revision": expected_revision,
                    "memory_id": memory_id,
                    "reason": reason,
                    "request": request_body,
                },
            )
            replay = self._receipt(
                connection,
                share_mode=old.share_mode,
                place_key=old.place_key,
                operation_ref=operation_ref,
                operation=change_kind,
                request_hash=fingerprint,
            )
            if replay is not None:
                return replay
            if old.state is not EntryState.ACTIVE:
                raise MemoryNotFoundError("memory was not found")
            if old.revision != expected_revision:
                raise RevisionConflictError("memory revision does not match expected_revision")
            if (
                old.title == draft.title
                and old.body == draft.body
                and old.kind == draft.kind
                and old.tags == draft.tags
                and old.metadata == draft.metadata
            ):
                raise InvalidMemoryError("replacement does not change the memory")
            changed_ms = self._next_ms(old.updated_ms)
            record = MemoryRecord(
                memory_id=old.memory_id,
                share_mode=old.share_mode,
                place_key=old.place_key,
                owner_key=old.owner_key,
                title=draft.title,
                body=draft.body,
                kind=draft.kind,
                tags=draft.tags,
                metadata=draft.metadata,
                revision=old.revision + 1,
                state=EntryState.ACTIVE,
                created_ms=old.created_ms,
                updated_ms=changed_ms,
                deleted_ms=None,
            )
            cursor = connection.execute(
                """
                UPDATE memory_entries
                SET title = ?, body = ?, kind = ?, tags_json = ?, tags_text = ?,
                    metadata_json = ?, revision = ?, updated_ms = ?
                WHERE memory_id = ? AND revision = ? AND state = 'active'
                """,
                (
                    record.title,
                    record.body,
                    record.kind,
                    canonical_json(list(record.tags)),
                    " ".join(record.tags),
                    canonical_json(record.metadata),
                    record.revision,
                    record.updated_ms,
                    record.memory_id,
                    old.revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError("memory revision changed during the write")
            change_seq = self._insert_change(
                connection,
                memory_id=record.memory_id,
                context=context,
                change_kind=change_kind,
                from_revision=old.revision,
                to_revision=record.revision,
                reason=reason,
                operation_ref=operation_ref,
                changed_ms=changed_ms,
            )
            self._insert_receipt(
                connection,
                share_mode=record.share_mode,
                place_key=record.place_key,
                operation_ref=operation_ref,
                request_hash=fingerprint,
                operation=change_kind,
                record=record,
                change_seq=change_seq,
                created_ms=changed_ms,
            )
            return WriteResult(record=record, change_seq=change_seq, replayed=False)

    def update(
        self,
        context: AccessContext,
        memory_id: str,
        patch: MemoryPatch,
        *,
        expected_revision: int,
        op_key: str,
        note: str = "",
    ) -> WriteResult:
        self._require_context(context)
        self._require_patch(patch)
        memory_id = normalize_identifier(memory_id, "memory_id")
        with self._database.read_connection() as connection:
            row = self._writable_row(connection, context, memory_id)
            old = self._record_from_row(row)
        draft = MemoryDraft(
            title=old.title if patch.title is None else patch.title,
            body=old.body if patch.body is None else patch.body,
            kind=old.kind if patch.kind is None else patch.kind,
            tags=old.tags if patch.tags is None else patch.tags,
            metadata=old.metadata if patch.metadata is None else patch.metadata,
        )
        patch_payload: dict[str, Any] = {}
        for field_name in ("title", "body", "kind", "tags", "metadata"):
            value = getattr(patch, field_name)
            if value is not None:
                patch_payload[field_name] = list(value) if field_name == "tags" else value
        return self._replace(
            context,
            memory_id,
            draft,
            expected_revision=expected_revision,
            op_key=op_key,
            reason=note,
            change_kind=ChangeKind.UPDATE,
            request_body={"patch": patch_payload},
        )

    def correct(
        self,
        context: AccessContext,
        memory_id: str,
        replacement: MemoryDraft,
        *,
        expected_revision: int,
        op_key: str,
        reason: str,
    ) -> WriteResult:
        self._require_context(context)
        self._require_draft(replacement)
        return self._replace(
            context,
            memory_id,
            replacement,
            expected_revision=expected_revision,
            op_key=op_key,
            reason=reason,
            change_kind=ChangeKind.CORRECT,
            request_body={"replacement": _draft_payload(replacement)},
        )

    def delete(
        self,
        context: AccessContext,
        memory_id: str,
        *,
        expected_revision: int,
        op_key: str,
        reason: str,
    ) -> WriteResult:
        self._require_context(context)
        memory_id = normalize_identifier(memory_id, "memory_id")
        expected_revision = validate_positive_int(expected_revision, "expected_revision")
        operation_ref = _operation_ref(op_key)
        reason = normalize_audit_label(reason, required=True)
        with self._database.write_transaction() as connection:
            row = self._writable_row(connection, context, memory_id)
            old = self._record_from_row(row)
            fingerprint = _request_hash(
                ChangeKind.DELETE,
                {
                    "actor_key": context.actor_key,
                    "expected_revision": expected_revision,
                    "memory_id": memory_id,
                    "reason": reason,
                },
            )
            replay = self._receipt(
                connection,
                share_mode=old.share_mode,
                place_key=old.place_key,
                operation_ref=operation_ref,
                operation=ChangeKind.DELETE,
                request_hash=fingerprint,
            )
            if replay is not None:
                return replay
            if old.state is not EntryState.ACTIVE:
                raise MemoryNotFoundError("memory was not found")
            if old.revision != expected_revision:
                raise RevisionConflictError("memory revision does not match expected_revision")
            changed_ms = self._next_ms(old.updated_ms)
            record = MemoryRecord(
                memory_id=old.memory_id,
                share_mode=old.share_mode,
                place_key=old.place_key,
                owner_key=old.owner_key,
                title=old.title,
                body=old.body,
                kind=old.kind,
                tags=old.tags,
                metadata=old.metadata,
                revision=old.revision + 1,
                state=EntryState.DELETED,
                created_ms=old.created_ms,
                updated_ms=changed_ms,
                deleted_ms=changed_ms,
            )
            cursor = connection.execute(
                """
                UPDATE memory_entries
                SET revision = ?, state = 'deleted', updated_ms = ?, deleted_ms = ?
                WHERE memory_id = ? AND revision = ? AND state = 'active'
                """,
                (
                    record.revision,
                    record.updated_ms,
                    record.deleted_ms,
                    record.memory_id,
                    old.revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError("memory revision changed during the write")
            change_seq = self._insert_change(
                connection,
                memory_id=record.memory_id,
                context=context,
                change_kind=ChangeKind.DELETE,
                from_revision=old.revision,
                to_revision=record.revision,
                reason=reason,
                operation_ref=operation_ref,
                changed_ms=changed_ms,
            )
            self._insert_receipt(
                connection,
                share_mode=record.share_mode,
                place_key=record.place_key,
                operation_ref=operation_ref,
                request_hash=fingerprint,
                operation=ChangeKind.DELETE,
                record=record,
                change_seq=change_seq,
                created_ms=changed_ms,
            )
            return WriteResult(record=record, change_seq=change_seq, replayed=False)

    def changes(
        self,
        context: AccessContext,
        memory_id: str,
        *,
        after_seq: int = 0,
        limit: int = 100,
    ) -> tuple[MemoryChange, ...]:
        self._require_context(context)
        memory_id = normalize_identifier(memory_id, "memory_id")
        if (
            isinstance(after_seq, bool)
            or not isinstance(after_seq, int)
            or not 0 <= after_seq <= MAX_EPOCH_MS
        ):
            raise InvalidMemoryError(f"after_seq must be an integer between 0 and {MAX_EPOCH_MS}")
        limit = validate_limit(limit)
        with self._database.read_connection() as connection:
            self._visible_row(
                connection,
                context,
                memory_id,
                include_deleted=True,
            )
            rows = connection.execute(
                """
                SELECT change_seq, memory_id, actor_key, change_kind,
                       from_revision, to_revision, reason, operation_ref, changed_ms
                FROM memory_changes
                WHERE memory_id = ? AND change_seq > ?
                ORDER BY change_seq ASC
                LIMIT ?
                """,
                (memory_id, after_seq, limit),
            ).fetchall()
            return tuple(self._change_from_row(row) for row in rows)

    def rebuild_search_index(self) -> None:
        self._database.rebuild_search_index()

    def self_check(self) -> dict[str, object]:
        report = self._database.integrity_report()
        pragmas = report["pragmas"]
        if not isinstance(pragmas, dict):
            raise StorageDamagedError("integrity report contains invalid pragmas")
        return {
            "status": "ok",
            "schema_version": report["schema_version"],
            "journal_mode": pragmas["journal_mode"],
        }


__all__ = ["MemoryStore"]
