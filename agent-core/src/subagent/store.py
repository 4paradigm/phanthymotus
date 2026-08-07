"""
store.py — Subagent state persistence via SQLite.

Provides checkpoint/resume for subagent lifecycle:
- Save full state (spec, turns, status) at regular intervals
- Restore active subagents on startup
- Clean up old terminal subagents
"""

import json
import sqlite3
import pathlib
import time

import config
from .protocol import (
    SubagentSpec, SubagentResult, SubagentStatus,
    ACTIVE_STATUSES, TERMINAL_STATUSES,
)


def _get_conn() -> sqlite3.Connection:
    db_path = pathlib.Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute('''
        CREATE TABLE IF NOT EXISTS subagents (
            id TEXT PRIMARY KEY,
            spec TEXT NOT NULL,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 2,
            turns TEXT DEFAULT '[]',
            summary TEXT DEFAULT '',
            rounds_completed INTEGER DEFAULT 0,
            result TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            checkpoint_at REAL
        )
    ''')
    conn.commit()
    return conn


class SubagentStore:
    """SQLite-backed persistence for subagent state."""

    def save_checkpoint(self, agent_id: str, spec: SubagentSpec, status: str,
                        turns: list, summary: str, rounds_completed: int,
                        created_at: float, updated_at: float) -> None:
        """Persist full subagent state."""
        now = time.time()
        with _get_conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO subagents
                (id, spec, status, priority, turns, summary, rounds_completed,
                 result, created_at, updated_at, checkpoint_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_id,
                json.dumps(spec.to_dict(), ensure_ascii=False),
                status,
                spec.priority,
                json.dumps(turns, ensure_ascii=False),
                summary,
                rounds_completed,
                '',  # result empty until terminal
                created_at,
                updated_at,
                now,
            ))
            conn.commit()

    def save_result(self, agent_id: str, result: SubagentResult) -> None:
        """Mark terminal state and save result."""
        now = time.time()
        with _get_conn() as conn:
            conn.execute('''
                UPDATE subagents
                SET status = ?, result = ?, updated_at = ?
                WHERE id = ?
            ''', (
                result.status,
                json.dumps(result.to_dict(), ensure_ascii=False),
                now,
                agent_id,
            ))
            conn.commit()

    def load_active(self) -> list[dict]:
        """Load all non-terminal subagents for restore on startup."""
        with _get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM subagents WHERE status IN (?, ?, ?, ?)',
                tuple(ACTIVE_STATUSES),
            ).fetchall()
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'spec': SubagentSpec.from_dict(json.loads(row['spec'])),
                'status': row['status'],
                'turns': json.loads(row['turns']),
                'summary': row['summary'],
                'rounds_completed': row['rounds_completed'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
            })
        return results

    def delete(self, agent_id: str) -> None:
        """Remove a subagent record."""
        with _get_conn() as conn:
            conn.execute('DELETE FROM subagents WHERE id = ?', (agent_id,))
            conn.commit()

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Remove completed/failed subagents older than max_age."""
        cutoff = time.time() - max_age_hours * 3600
        with _get_conn() as conn:
            cursor = conn.execute(
                'DELETE FROM subagents WHERE status IN (?, ?, ?, ?) AND updated_at < ?',
                (*TERMINAL_STATUSES, cutoff),
            )
            conn.commit()
            return cursor.rowcount
