"""Ordered, checksummed SQLite migrations for Memory Core."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

# ASCII "MCOR", stored in SQLite's application_id header field.
APPLICATION_ID = 0x4D434F52


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable schema migration."""

    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        canonical = "\n-- statement --\n".join(statement.strip() for statement in self.statements)
        return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="independent_memory",
        statements=(
            """
            CREATE TABLE memory_entries (
                row_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id     TEXT NOT NULL UNIQUE
                              CHECK (length(trim(memory_id)) > 0),
                share_mode    TEXT NOT NULL
                              CHECK (share_mode IN ('private', 'shared')),
                place_key     TEXT NOT NULL,
                owner_key     TEXT NOT NULL,
                title         TEXT NOT NULL,
                body          TEXT NOT NULL CHECK (length(trim(body)) > 0),
                kind          TEXT NOT NULL CHECK (length(trim(kind)) > 0),
                tags_json     TEXT NOT NULL
                              CHECK (json_valid(tags_json) AND json_type(tags_json) = 'array'),
                tags_text     TEXT NOT NULL,
                metadata_json TEXT NOT NULL CHECK (
                    json_valid(metadata_json) AND json_type(metadata_json) = 'object'
                ),
                revision      INTEGER NOT NULL CHECK (revision >= 1),
                state         TEXT NOT NULL CHECK (state IN ('active', 'deleted')),
                created_ms    INTEGER NOT NULL CHECK (created_ms >= 0),
                updated_ms    INTEGER NOT NULL CHECK (updated_ms >= created_ms),
                deleted_ms    INTEGER,
                CHECK (
                    (
                        share_mode = 'private'
                        AND length(trim(owner_key)) > 0
                        AND place_key = owner_key
                    )
                    OR
                    (
                        share_mode = 'shared'
                        AND length(trim(place_key)) > 0
                        AND length(trim(owner_key)) > 0
                    )
                ),
                CHECK (
                    (state = 'active' AND deleted_ms IS NULL)
                    OR
                    (
                        state = 'deleted'
                        AND deleted_ms IS NOT NULL
                        AND deleted_ms >= updated_ms
                    )
                )
            )
            """,
            """
            CREATE INDEX idx_memory_entries_private_visible
            ON memory_entries(owner_key, updated_ms DESC, row_id DESC)
            WHERE share_mode = 'private' AND state = 'active'
            """,
            """
            CREATE INDEX idx_memory_entries_shared_visible
            ON memory_entries(place_key, updated_ms DESC, row_id DESC)
            WHERE share_mode = 'shared' AND state = 'active'
            """,
            """
            CREATE TABLE memory_changes (
                change_seq    INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id     TEXT NOT NULL REFERENCES memory_entries(memory_id)
                              ON UPDATE RESTRICT ON DELETE RESTRICT,
                actor_key     TEXT NOT NULL CHECK (length(trim(actor_key)) > 0),
                change_kind   TEXT NOT NULL CHECK (
                    change_kind IN ('create', 'update', 'correct', 'delete')
                ),
                from_revision INTEGER,
                to_revision   INTEGER NOT NULL CHECK (to_revision >= 1),
                reason        TEXT NOT NULL CHECK (
                    (change_kind = 'create' AND reason = '')
                    OR
                    (
                        change_kind = 'update'
                        AND (
                            reason = ''
                            OR (
                                length(reason) BETWEEN 1 AND 64
                                AND substr(reason, 1, 1) GLOB '[a-z0-9]'
                                AND reason NOT GLOB '*[^a-z0-9_.:-]*'
                            )
                        )
                    )
                    OR
                    (
                        change_kind IN ('correct', 'delete')
                        AND length(reason) BETWEEN 1 AND 64
                        AND substr(reason, 1, 1) GLOB '[a-z0-9]'
                        AND reason NOT GLOB '*[^a-z0-9_.:-]*'
                    )
                ),
                operation_ref TEXT NOT NULL CHECK (length(trim(operation_ref)) > 0),
                changed_ms    INTEGER NOT NULL CHECK (changed_ms >= 0),
                CHECK (
                    (
                        change_kind = 'create'
                        AND from_revision IS NULL
                        AND to_revision = 1
                    )
                    OR
                    (
                        change_kind IN ('update', 'correct', 'delete')
                        AND from_revision >= 1
                        AND to_revision = from_revision + 1
                    )
                ),
                UNIQUE (memory_id, to_revision)
            )
            """,
            """
            CREATE TRIGGER memory_changes_no_replace
            BEFORE INSERT ON memory_changes
            WHEN EXISTS (
                SELECT 1
                FROM memory_changes AS existing
                WHERE existing.change_seq = new.change_seq
                   OR (
                       existing.memory_id = new.memory_id
                       AND existing.to_revision = new.to_revision
                   )
            )
            BEGIN
                SELECT RAISE(ABORT, 'memory_changes is append-only');
            END
            """,
            """
            CREATE TRIGGER memory_changes_no_update
            BEFORE UPDATE ON memory_changes
            BEGIN
                SELECT RAISE(ABORT, 'memory_changes is append-only');
            END
            """,
            """
            CREATE TRIGGER memory_changes_no_delete
            BEFORE DELETE ON memory_changes
            BEGIN
                SELECT RAISE(ABORT, 'memory_changes is append-only');
            END
            """,
            """
            CREATE INDEX idx_memory_changes_memory
            ON memory_changes(memory_id, change_seq DESC)
            """,
            """
            CREATE INDEX idx_memory_changes_changed
            ON memory_changes(changed_ms DESC, change_seq DESC)
            """,
            """
            CREATE TABLE memory_requests (
                share_mode        TEXT NOT NULL
                                  CHECK (share_mode IN ('private', 'shared')),
                place_key         TEXT NOT NULL CHECK (length(trim(place_key)) > 0),
                operation_ref     TEXT NOT NULL CHECK (length(trim(operation_ref)) > 0),
                request_hash      TEXT NOT NULL CHECK (length(trim(request_hash)) > 0),
                operation         TEXT NOT NULL CHECK (
                    operation IN ('create', 'update', 'correct', 'delete')
                ),
                result_json       TEXT NOT NULL CHECK (
                    json_valid(result_json) AND json_type(result_json) = 'object'
                ),
                result_memory_id  TEXT NOT NULL REFERENCES memory_entries(memory_id)
                                  ON UPDATE RESTRICT ON DELETE RESTRICT,
                result_revision   INTEGER NOT NULL CHECK (result_revision >= 1),
                result_change_seq INTEGER NOT NULL REFERENCES memory_changes(change_seq)
                                  ON UPDATE RESTRICT ON DELETE RESTRICT,
                created_ms        INTEGER NOT NULL CHECK (created_ms >= 0),
                UNIQUE (share_mode, place_key, operation_ref)
            )
            """,
            """
            CREATE INDEX idx_memory_requests_result
            ON memory_requests(result_memory_id, result_revision)
            """,
        ),
    ),
    Migration(
        version=2,
        name="active_memory_search",
        statements=(
            """
            CREATE VIRTUAL TABLE memory_fts USING fts5(
                title,
                body,
                tags_text,
                content = 'memory_entries',
                content_rowid = 'row_id',
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """,
            """
            CREATE TRIGGER memory_entries_fts_insert
            AFTER INSERT ON memory_entries
            WHEN new.state = 'active'
            BEGIN
                INSERT INTO memory_fts(rowid, title, body, tags_text)
                VALUES (new.row_id, new.title, new.body, new.tags_text);
            END
            """,
            """
            CREATE TRIGGER memory_entries_fts_delete
            AFTER DELETE ON memory_entries
            WHEN old.state = 'active'
            BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, title, body, tags_text)
                VALUES ('delete', old.row_id, old.title, old.body, old.tags_text);
            END
            """,
            """
            CREATE TRIGGER memory_entries_fts_update
            AFTER UPDATE ON memory_entries
            BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, title, body, tags_text)
                SELECT 'delete', old.row_id, old.title, old.body, old.tags_text
                WHERE old.state = 'active';

                INSERT INTO memory_fts(rowid, title, body, tags_text)
                SELECT new.row_id, new.title, new.body, new.tags_text
                WHERE new.state = 'active';
            END
            """,
            """
            INSERT INTO memory_fts(rowid, title, body, tags_text)
            SELECT row_id, title, body, tags_text
            FROM memory_entries
            WHERE state = 'active'
            """,
        ),
    ),
)


LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version
