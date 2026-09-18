"""Persistent chat history storage."""

import json
import re
import time
import uuid

from config import _get_conn


KIND_MAIN = 'main'
KIND_SUBAGENT = 'subagent'
KIND_BG_SUBAGENT = 'bg_subagent'


def create_session(kind: str = KIND_MAIN) -> str:
    """Create a new chat session, return its ID.

    `kind` separates the main agent's own conversation from delegated ones, so the
    history UI can group them without parsing the summary text.
    """
    sid = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            'INSERT INTO chat_sessions (id, started_at, kind) VALUES (?, ?, ?)',
            (sid, time.time(), kind)
        )
        conn.commit()
    return sid


def save_turn(session_id: str, turn_index: int, turn_messages: list[dict]):
    """Persist a turn, overwriting any previous write at the same index.

    Callers write a turn repeatedly while it is still running — once per LLM round —
    so the history UI can show a turn in progress instead of only after it ends.
    That makes this an upsert, and makes turn_count a recount rather than a
    counter increment.
    """
    now = time.time()
    blob = json.dumps(turn_messages, ensure_ascii=False, default=str)
    with _get_conn() as conn:
        row = conn.execute(
            'SELECT id, created_at FROM chat_messages WHERE session_id = ? AND turn_index = ?',
            (session_id, turn_index)
        ).fetchone()
        if row:
            rowid = row[0]
            conn.execute(
                'UPDATE chat_messages SET messages = ?, updated_at = ? WHERE id = ?',
                (blob, now, rowid)
            )
        else:
            cursor = conn.execute(
                'INSERT INTO chat_messages (session_id, turn_index, messages, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (session_id, turn_index, blob, now, now)
            )
            rowid = cursor.lastrowid
        conn.execute(
            'UPDATE chat_sessions SET ended_at = ?, '
            'turn_count = (SELECT COUNT(*) FROM chat_messages WHERE session_id = ?) '
            'WHERE id = ?',
            (now, session_id, session_id)
        )
        # Sync FTS index
        try:
            _ensure_fts(conn)
            conn.execute('DELETE FROM chat_messages_fts WHERE rowid = ?', (rowid,))
            conn.execute(
                'INSERT INTO chat_messages_fts(rowid, content) VALUES (?, ?)',
                (rowid, blob)
            )
        except Exception:
            pass  # FTS is non-critical
        conn.commit()


_EVENT_TAG_RE = re.compile(r'</?event\b[^>]*>')


def summary_text(text: str) -> str:
    """把一条 trigger 压成人能读的一行，供会话列表当标题用。

    trigger 的文本常常是 `<event source="dds:/…" channel="…" ts="…">\\n{"text": "早上好。"}`
    这种信封。直接截 100 字的结果是整行都是 source/channel/ts，真正说了什么被截在
    外面 —— 列表里那一条长这样：`<event source="dds:/remote_control/message"
    channel="remote_web" ts="2026-09-14T11:05:38">\\n{"text": …`，看不出是哪次对话。
    """
    body = _EVENT_TAG_RE.sub('', text).strip()
    if body.startswith('{'):
        try:
            data = json.loads(body)
            if isinstance(data, dict) and isinstance(data.get('text'), str):
                body = data['text'].strip()
        except (ValueError, TypeError):
            pass
    return body or text.strip()


def update_summary(session_id: str, text: str):
    """Set session summary (first user trigger text)."""
    text = summary_text(text)
    # Truncate to 100 chars for display
    summary = (text[:100] + '…') if len(text) > 100 else text
    with _get_conn() as conn:
        conn.execute(
            'UPDATE chat_sessions SET summary = ? WHERE id = ? AND summary = \'\'',
            (summary, session_id)
        )
        conn.commit()


def list_sessions(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """Return recent sessions and total count. Excludes empty (0-turn) sessions.

    Ordered by last activity, not by start: a long-running session that just spoke
    belongs at the top, and ordering by `started_at` buried it under subagents that
    started later but finished long ago.
    """
    with _get_conn() as conn:
        conn.row_factory = None
        total = conn.execute('SELECT COUNT(*) FROM chat_sessions WHERE turn_count > 0').fetchone()[0]
        rows = conn.execute(
            'SELECT id, started_at, ended_at, summary, turn_count, kind '
            'FROM chat_sessions WHERE turn_count > 0 '
            'ORDER BY COALESCE(ended_at, started_at) DESC LIMIT ? OFFSET ?',
            (limit, offset)
        ).fetchall()
    sessions = [
        {'id': r[0], 'started_at': r[1], 'ended_at': r[2], 'summary': r[3],
         'turn_count': r[4], 'kind': r[5] or KIND_MAIN,
         'last_at': r[2] or r[1]}
        for r in rows
    ]
    return sessions, total


def get_session_messages(session_id: str) -> list[list[dict]]:
    """Return all turns for a session, ordered by turn_index."""
    return [t['messages'] for t in get_session_turns(session_id)]


def get_session_turns(session_id: str) -> list[dict]:
    """Return all turns with their timestamps, ordered by turn_index.

    `started_at` is when the turn's first message was written, `updated_at` when it
    was last rewritten — for a turn still running they differ, and the UI labels
    each turn with the latter.
    """
    with _get_conn() as conn:
        conn.row_factory = None
        rows = conn.execute(
            'SELECT messages, created_at, updated_at FROM chat_messages '
            'WHERE session_id = ? ORDER BY turn_index',
            (session_id,)
        ).fetchall()
    return [
        {'messages': json.loads(r[0]), 'started_at': r[1], 'updated_at': r[2] or r[1]}
        for r in rows
    ]


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
    """获取主代理最近一个 session 的最后 N 轮，用于重启续跑。

    只认 main：子代理的对话也存在同一张表里，而"最近的一个 session"现在几乎总是
    某个后台子代理的 —— 它们每轮都落盘。续跑到那上面，主代理会把一段后台监控的
    transcript 当成自己的历史接着往下写，并且之后每个 turn 都追加进那个会话。
    老记录的 kind 列是迁移时统一填的 'main'，所以还要排掉带 `[subagent:` 前缀的。
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM chat_sessions WHERE turn_count > 0 "
            "AND kind = 'main' AND summary NOT LIKE '[subagent:%' "
            'ORDER BY COALESCE(ended_at, started_at) DESC LIMIT 1'
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
