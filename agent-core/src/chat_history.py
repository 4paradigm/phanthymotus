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
        # Sync FTS index
        try:
            _ensure_fts(conn)
            cursor = conn.execute('SELECT last_insert_rowid()')
            rowid = cursor.fetchone()[0]
            conn.execute(
                'INSERT INTO chat_messages_fts(rowid, content) VALUES (?, ?)',
                (rowid, json.dumps(turn_messages, ensure_ascii=False, default=str))
            )
        except Exception:
            pass  # FTS is non-critical
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
        try:
            conn.execute('DELETE FROM chat_messages_fts')
        except Exception:
            pass
        conn.commit()


# ── FTS5 全文搜索 ─────────────────────────────────────────────────────────────

def _ensure_fts(conn=None):
    """确保 FTS5 虚拟表存在。"""
    if conn is None:
        conn = _get_conn()
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chat_messages_fts
        USING fts5(content, content_rowid='id', tokenize='unicode61')
    ''')


def rebuild_fts():
    """重建 FTS 索引（首次或数据修复时调用）。"""
    with _get_conn() as conn:
        _ensure_fts(conn)
        conn.execute('DELETE FROM chat_messages_fts')
        conn.execute('''
            INSERT INTO chat_messages_fts(rowid, content)
            SELECT id, messages FROM chat_messages
        ''')
        conn.commit()


def search(query: str, limit: int = 10) -> list[dict]:
    """全文搜索历史对话。"""
    with _get_conn() as conn:
        _ensure_fts(conn)
        try:
            rows = conn.execute('''
                SELECT cm.session_id, cm.turn_index, cm.messages, cm.created_at
                FROM chat_messages_fts fts
                JOIN chat_messages cm ON cm.id = fts.rowid
                WHERE fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (query, limit)).fetchall()
        except Exception:
            return []
    results = []
    for session_id, turn_index, messages_json, ts in rows:
        messages = json.loads(messages_json)
        texts = []
        for m in messages:
            content = m.get('content', '')
            if isinstance(content, str) and content:
                texts.append(content[:200])
        results.append({
            'session_id': session_id,
            'turn_index': turn_index,
            'ts': ts,
            'preview': ' | '.join(texts)[:300],
        })
    return results


# ── 重启续跑 ──────────────────────────────────────────────────────────────────

def get_last_session_turns(limit: int = 10) -> dict | None:
    """获取最近一个 session 的最后 N 轮，用于重启续跑。"""
    with _get_conn() as conn:
        row = conn.execute(
            'SELECT id FROM chat_sessions WHERE turn_count > 0 ORDER BY started_at DESC LIMIT 1'
        ).fetchone()
        if not row:
            return None
        session_id = row[0]
        rows = conn.execute(
            'SELECT messages FROM chat_messages WHERE session_id=? ORDER BY turn_index DESC LIMIT ?',
            (session_id, limit)
        ).fetchall()
    if not rows:
        return None
    turns = [json.loads(r[0]) for r in reversed(rows)]
    return {'session_id': session_id, 'turns': turns}
