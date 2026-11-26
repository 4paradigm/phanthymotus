"""Persistent chat history storage."""

import json
import re
import time
import uuid
from contextvars import ContextVar

from config import _get_conn


KIND_MAIN = 'main'
KIND_SUBAGENT = 'subagent'
KIND_BG_SUBAGENT = 'bg_subagent'
KIND_CHANNEL = 'channel'

# None is shared main memory; an empty key is an unidentifiable Channel input.
# Set by the dispatcher, never supplied by a model tool argument.
current_channel_context: ContextVar[str | None] = ContextVar('channel_history_context', default=None)


def create_session(kind: str = KIND_MAIN, *, context_key: str = '') -> str:
    """Create a new chat session, return its ID.

    `kind` separates the main agent's own conversation from delegated ones, so the
    history UI can group them without parsing the summary text.
    """
    sid = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            'INSERT INTO chat_sessions (id, started_at, kind, context_key) VALUES (?, ?, ?, ?)',
            (sid, time.time(), kind, context_key)
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
                'SELECT ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM chat_sessions WHERE id = ?)',
                (session_id, turn_index, blob, now, now, session_id)
            )
            if not cursor.rowcount:
                return  # A concurrent history deletion must not leave orphan messages.
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
    delete_sessions([session_id])


def _delete_channel_recaps(conn, session_ids: list[str], placeholders: str):
    """Remove the legacy recap for each deleted human Channel identity."""
    contexts = conn.execute(
        f'SELECT context_key FROM chat_sessions WHERE id IN ({placeholders}) AND kind = ?',
        [*session_ids, KIND_CHANNEL],
    ).fetchall()
    if not contexts:
        return
    row = conn.execute("SELECT value FROM config WHERE key = 'channel_chat_history'").fetchone()
    if not row:
        return
    history = json.loads(row[0])
    if not isinstance(history, dict):
        return
    for (context_key,) in contexts:
        try:
            scope = json.loads(context_key)
        except (ValueError, TypeError):
            continue
        if not isinstance(scope, list) or len(scope) != 6:
            continue
        _, channel_id, chat_id, user_id, user_role, _ = scope
        channel = history.get(channel_id, {})
        if not isinstance(channel, dict) or not isinstance(channel.get(chat_id), list):
            continue
        channel[chat_id] = [entry for entry in channel[chat_id]
                            if not (isinstance(entry, dict) and entry.get('user_id') == user_id
                                    and entry.get('user_role') == user_role
                                    and entry.get('sender_type') == 'user')]
    conn.execute("UPDATE config SET value = ? WHERE key = 'channel_chat_history'",
                 (json.dumps(history, ensure_ascii=False),))


def delete_sessions(session_ids: list[str]):
    """Delete multiple sessions."""
    if not session_ids:
        return
    placeholders = ','.join('?' * len(session_ids))
    with _get_conn() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _delete_channel_recaps(conn, session_ids, placeholders)
        conn.execute(f'DELETE FROM chat_messages WHERE session_id IN ({placeholders})', session_ids)
        conn.execute(f'DELETE FROM chat_sessions WHERE id IN ({placeholders})', session_ids)
        conn.commit()


def clear_all():
    """Delete all sessions and messages, and the conclusions drawn from them.

    `subagent_conclusions` is written by `subagent/agent.py` and
    `subagent/manager.py` and read back by `memory_recall`
    (`event/llm.py`, `source in ('all', 'subagent')`). It therefore holds the
    *same* memory as the chat tables, one abstraction layer up — a background
    monitor's summary of what was said, a daily digest of it.

    Clearing the chat tables alone left that behind, so after "清除历史" the
    agent could still recall conversations the operator had deleted and answer
    out of them. Measured on Orin5 2026-10-17: `chat_messages` started at
    17:13 while `subagent_conclusions` still held 1048 rows going back to
    09-04 — six weeks of conclusions about deleted conversations. The symptom
    is not a visible row anywhere; it is the agent simply still knowing
    something, which is why it took a report of "感觉清除不干净" rather than a
    bug to find.

    **Only the wholesale clear can do this.** A conclusion carries `agent_id`,
    `goal` and `created_at` but no `session_id` (see `config.py`'s DDL), so
    `delete_session` / `delete_sessions` have nothing to join on and leave
    conclusions behind. That gap is real and deliberate rather than
    overlooked: guessing by timestamp would delete another session's
    conclusions.
    """
    with _get_conn() as conn:
        conn.execute('DELETE FROM chat_messages')
        conn.execute('DELETE FROM chat_sessions')
        conn.execute("DELETE FROM config WHERE key IN ('channel_chat_history', 'channel_history_migrations')")
        try:
            conn.execute('DELETE FROM chat_messages_fts')
        except Exception:
            pass
        # Separate try: a deployment predating the conclusions table must still
        # get its chat tables cleared rather than failing the whole reset.
        try:
            conn.execute('DELETE FROM subagent_conclusions')
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


def search(query: str, limit: int = 10, *, context_key: str | None = None,
           include_channels: bool = True) -> list[dict]:
    """全文搜索历史对话。"""
    if context_key == '':
        return []  # An unidentified Channel input has no authorized history.
    with _get_conn() as conn:
        _ensure_fts(conn)
        try:
            where = ''
            params = [query]
            if context_key is not None:
                where = ' AND cs.kind = ? AND cs.context_key = ?'
                params.extend([KIND_CHANNEL, context_key])
            elif not include_channels:
                where = ' AND cs.kind != ?'
                params.append(KIND_CHANNEL)
            params.append(limit)
            rows = conn.execute('''
                SELECT cm.session_id, cm.turn_index, cm.messages, cm.created_at
                FROM chat_messages_fts fts
                JOIN chat_messages cm ON cm.id = fts.rowid
                JOIN chat_sessions cs ON cs.id = cm.session_id
                WHERE fts.content MATCH ?
            ''' + where + '''
                ORDER BY rank
                LIMIT ?
            ''', params).fetchall()
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

def get_channel_session(context_key: str, limit: int | None = 18, *,
                        session_id: str | None = None) -> dict | None:
    """Restore only this sender's same-platform, same-role Channel conversation."""
    if not context_key:
        return None
    with _get_conn() as conn:
        query = ('SELECT id, rolling_summary, summarized_through FROM chat_sessions '
                 'WHERE kind=? AND context_key=?')
        params = [KIND_CHANNEL, context_key]
        if session_id is not None:
            query += ' AND id=?'
            params.append(session_id)
        row = conn.execute(query + ' ORDER BY COALESCE(ended_at, started_at) DESC LIMIT 1', params).fetchone()
    if not row:
        return None
    sid, summary, through = row
    with _get_conn() as conn:
        next_index = conn.execute(
            'SELECT COALESCE(MAX(turn_index), -1)+1 FROM chat_messages WHERE session_id=?', (sid,),
        ).fetchone()[0]
        query = ('SELECT turn_index, messages FROM chat_messages WHERE session_id=? AND turn_index>? '
                 'ORDER BY turn_index DESC')
        params = [sid, through]
        if limit is not None:
            query += ' LIMIT ?'
            params.append(limit)
        rows = conn.execute(
            query, params,
        ).fetchall()
    return {'session_id': sid, 'next_turn_index': next_index,
            'summary': summary or '', 'summarized_through': through,
            'turn_indexes': [row[0] for row in reversed(rows)],
            'turns': [json.loads(row[1]) for row in reversed(rows)]}


def save_channel_summary(session_id: str, context_key: str, summary: str, *,
                         through: int, expected_through: int) -> bool:
    """Advance a scoped summary without recreating deleted or changed sessions."""
    with _get_conn() as conn:
        cursor = conn.execute(
            'UPDATE chat_sessions SET rolling_summary=?, summarized_through=? '
            'WHERE id=? AND kind=? AND context_key=? AND summarized_through=?',
            (summary, through, session_id, KIND_CHANNEL, context_key, expected_through),
        )
        conn.commit()
        return cursor.rowcount == 1


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
