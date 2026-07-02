"""Persistent chat history storage."""

import json
import time
import uuid

from config import _get_conn


def create_session() -> str:
    """Create a new chat session, return its ID."""
    sid = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            'INSERT INTO chat_sessions (id, started_at) VALUES (?, ?)',
            (sid, time.time())
        )
        conn.commit()
    return sid


def save_turn(session_id: str, turn_index: int, turn_messages: list[dict]):
    """Persist a single turn (list of messages) to the database."""
    now = time.time()
    with _get_conn() as conn:
        conn.execute(
            'INSERT INTO chat_messages (session_id, turn_index, messages, created_at) '
            'VALUES (?, ?, ?, ?)',
            (session_id, turn_index, json.dumps(turn_messages, ensure_ascii=False, default=str), now)
        )
        conn.execute(
            'UPDATE chat_sessions SET ended_at = ?, turn_count = turn_count + 1 WHERE id = ?',
            (now, session_id)
        )
        conn.commit()


def update_summary(session_id: str, text: str):
    """Set session summary (first user trigger text)."""
    # Truncate to 100 chars for display
    summary = (text[:100] + '…') if len(text) > 100 else text
    with _get_conn() as conn:
        conn.execute(
            'UPDATE chat_sessions SET summary = ? WHERE id = ? AND summary = \'\'',
            (summary, session_id)
        )
        conn.commit()


def list_sessions(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """Return recent sessions (newest first) and total count. Excludes empty (0-turn) sessions."""
    with _get_conn() as conn:
        conn.row_factory = None
        total = conn.execute('SELECT COUNT(*) FROM chat_sessions WHERE turn_count > 0').fetchone()[0]
        rows = conn.execute(
            'SELECT id, started_at, ended_at, summary, turn_count '
            'FROM chat_sessions WHERE turn_count > 0 ORDER BY started_at DESC LIMIT ? OFFSET ?',
            (limit, offset)
        ).fetchall()
    sessions = [
        {'id': r[0], 'started_at': r[1], 'ended_at': r[2], 'summary': r[3], 'turn_count': r[4]}
        for r in rows
    ]
    return sessions, total


def get_session_messages(session_id: str) -> list[list[dict]]:
    """Return all turns for a session, ordered by turn_index."""
    with _get_conn() as conn:
        conn.row_factory = None
        rows = conn.execute(
            'SELECT messages FROM chat_messages WHERE session_id = ? ORDER BY turn_index',
            (session_id,)
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def delete_session(session_id: str):
    """Delete a session and all its messages."""
    with _get_conn() as conn:
        conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
        conn.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        conn.commit()


def delete_sessions(session_ids: list[str]):
    """Delete multiple sessions."""
    if not session_ids:
        return
    placeholders = ','.join('?' * len(session_ids))
    with _get_conn() as conn:
        conn.execute(f'DELETE FROM chat_messages WHERE session_id IN ({placeholders})', session_ids)
        conn.execute(f'DELETE FROM chat_sessions WHERE id IN ({placeholders})', session_ids)
        conn.commit()


def clear_all():
    """Delete all sessions and messages."""
    with _get_conn() as conn:
        conn.execute('DELETE FROM chat_messages')
        conn.execute('DELETE FROM chat_sessions')
        conn.commit()
