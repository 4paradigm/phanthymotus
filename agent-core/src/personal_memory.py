"""Opt-in personal memory. Operators and the configured model provider are trusted.

Perception identifies a candidate; this feature deliberately accepts face/voice-only
unlock. It is not replay-resistant authentication. No identities are linked by name.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import sqlite3
import time
from uuid import uuid4

import config

_user = ContextVar('personal_memory_user', default=None)
_turn = ContextVar('personal_memory_turn', default='')
_runtime_settings: dict | None = None
_revision = ContextVar('personal_memory_revision', default=None)
_block_turn_save = ContextVar('personal_memory_block_save', default=False)


def configured_settings() -> dict:
    return config.main.get('personal_memory', {})


def settings() -> dict:
    return _runtime_settings if _runtime_settings is not None else configured_settings()


def pin_settings() -> None:
    """Snapshot mode settings at initialization or a stopped project startup boundary."""
    global _runtime_settings
    import copy
    _runtime_settings = copy.deepcopy(configured_settings())


def update_runtime_routes(routes: dict) -> None:
    if _runtime_settings is not None:
        _runtime_settings.update(routes)
    global resolver
    resolver = IdentityResolver()


def enabled() -> bool:
    return settings().get('enabled', False) is True


def current_user() -> str | None:
    return _user.get()


def bind_context(user_id: str | None):
    return _user.set(user_id)


def reset_context(token):
    _user.reset(token)


@contextmanager
def _db():
    conn = config._get_conn()
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS pm_users (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                remember INTEGER NOT NULL, onboarding INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS pm_bindings (
                modality TEXT NOT NULL, external_id TEXT NOT NULL,
                user_id TEXT NOT NULL, PRIMARY KEY (modality, external_id));
            CREATE TABLE IF NOT EXISTS pm_memories (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
                content TEXT NOT NULL, source_turn TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS pm_memories_owner ON pm_memories(owner, created_at);
            CREATE TABLE IF NOT EXISTS pm_turns (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, messages TEXT NOT NULL,
                created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS pm_turns_owner ON pm_turns(owner, created_at);
            CREATE TABLE IF NOT EXISTS pm_public (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, content TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
        ''')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(pm_users)')}
        for column in ('paused', 'revision'):
            if column not in columns:
                conn.execute(f'ALTER TABLE pm_users ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0')
        yield conn
        conn.commit()
    finally:
        conn.close()


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError('内容不能为空，且不得超过 4000 字。')
    return value.strip()


def create_user(name: str, remember: bool = False, onboarding: bool = False) -> dict:
    user_id = str(uuid4())
    with _db() as conn:
        conn.execute('INSERT INTO pm_users (id, name, remember, onboarding) VALUES (?, ?, ?, ?)',
                     (user_id, _text(name), int(remember), int(onboarding)))
    return get_user(user_id)


def get_user(user_id: str | None) -> dict | None:
    if not user_id:
        return None
    with _db() as conn:
        row = conn.execute('SELECT * FROM pm_users WHERE id=?', (user_id,)).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    with _db() as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM pm_users ORDER BY name')]


def update_user(user_id: str, remember: bool, onboarding: bool) -> dict:
    with _db() as conn:
        if not conn.execute('SELECT 1 FROM pm_users WHERE id=?', (user_id,)).fetchone():
            raise ValueError('用户不存在。')
        conn.execute('UPDATE pm_users SET remember=?, onboarding=?, paused=?, revision=revision+1 WHERE id=?',
                     (int(remember), int(onboarding), int(not remember), user_id))
        if not remember:
            for table in ('pm_memories', 'pm_turns'):
                conn.execute(f'DELETE FROM {table} WHERE owner=?', (user_id,))
            conn.execute('DELETE FROM pm_public WHERE owner=? AND approved=0', (user_id,))
    return get_user(user_id)


def bind_identity(user_id: str, modality: str, external_id: str) -> None:
    if modality not in ('face', 'voice'):
        raise ValueError('modality 必须是 face 或 voice。')
    external_id = _text(external_id)
    with _db() as conn:
        if not conn.execute('SELECT 1 FROM pm_users WHERE id=?', (user_id,)).fetchone():
            raise ValueError('用户不存在。')
        row = conn.execute('SELECT user_id FROM pm_bindings WHERE modality=? AND external_id=?',
                           (modality, external_id)).fetchone()
        if row and row['user_id'] != user_id:
            raise ValueError('该生物特征身份已绑定其他用户，请先解除绑定。')
        conn.execute('INSERT OR IGNORE INTO pm_bindings VALUES (?, ?, ?)',
                     (modality, external_id, user_id))


def unbind_identity(modality: str, external_id: str) -> None:
    with _db() as conn:
        conn.execute('DELETE FROM pm_bindings WHERE modality=? AND external_id=?',
                     (modality, external_id))


def _lookup(modality: str, external_id) -> str | None:
    if not isinstance(external_id, str) or not external_id:
        return None
    with _db() as conn:
        row = conn.execute('SELECT b.user_id FROM pm_bindings b JOIN pm_users u ON u.id=b.user_id '
                           'WHERE modality=? AND external_id=?', (modality, external_id)).fetchone()
    return row['user_id'] if row else None


class IdentityResolver:
    """Only configured DDS perception sources count; fresh faces never unlock chats."""
    def __init__(self):
        self.face_ids: list[str | None] = []
        self.face_ts = 0.0

    def resolve(self, event: dict) -> str | None:
        cfg = settings()
        source = event.get('source', '')
        if source not in cfg.get('voice_sources', []) + cfg.get('face_sources', []):
            return None
        try:
            data = json.loads(event.get('text', '{}'))
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        now = time.time()
        ts = event.get('ts')
        if not isinstance(ts, (int, float)) or not 0 <= now - ts <= cfg.get('face_max_age_s', 3):
            return None
        if source in cfg.get('face_sources', []):
            faces = data.get('faces', [])
            self.face_ts = ts
            self.face_ids = [f.get('person_id') if isinstance(f, dict) and f.get('quality') == 'ok'
                             else None for f in faces] if isinstance(faces, list) else []
            return _lookup('face', self.face_ids[0]) if len(self.face_ids) == 1 else None
        fresh_faces = self.face_ids if 0 <= now - self.face_ts <= cfg.get('face_max_age_s', 3) else []
        if data.get('multi_speaker') or len(fresh_faces) > 1:
            return None
        face = _lookup('face', fresh_faces[0]) if fresh_faces else None
        if data.get('speaker_id'):
            voice = _lookup('voice', data['speaker_id'])
            if not voice or (face and face != voice):
                return None
            return voice
        return face


@dataclass(frozen=True)
class IdentitySnapshot:
    user_id: str | None


resolver = IdentityResolver()


def _remembering_user() -> dict:
    user = get_user(current_user())
    if not user or not user['remember']:
        raise PermissionError('未识别到已授权保存记忆的用户。')
    return user


def _identified_user() -> dict:
    user = get_user(current_user())
    if not user:
        raise PermissionError('需要先识别并匹配本人，不能操作其他人的资料。')
    return user


def context_changed() -> bool:
    if _revision.get() is None:
        return False
    user = get_user(current_user())
    return not user or user['revision'] != _revision.get()


def _writing_user() -> dict:
    user = _remembering_user()
    if user['paused'] or _block_turn_save.get() or context_changed():
        raise PermissionError('记录已暂停或本轮资料发生更改；本轮不再新增记忆。')
    return user


def save_memory(content: str, kind: str, source_turn: str) -> dict:
    user = _writing_user()
    if kind not in ('fact', 'preference', 'event', 'analysis'):
        raise ValueError('记忆类型必须是 fact/preference/event/analysis；分析不是事实。')
    record = {'id': str(uuid4()), 'owner': user['id'], 'kind': kind,
              'content': _text(content), 'source_turn': source_turn, 'created_at': time.time()}
    with _db() as conn:
        saved = conn.execute('INSERT INTO pm_memories (id, owner, kind, content, source_turn, created_at) '
                             'SELECT :id, :owner, :kind, :content, :source_turn, :created_at FROM pm_users '
                             'WHERE id=:owner AND remember=1 AND paused=0 AND revision=:expected_revision',
                             {**record, 'expected_revision': user['revision']}).rowcount
        if not saved:
            raise PermissionError('保存前授权已变更，未写入记忆。')
    return record


def recall(query: str, limit: int = 20, offset: int = 0) -> list[dict]:
    user = get_user(current_user())
    owner = user['id'] if user and user['remember'] else ''
    # instr is literal substring search (also works for Chinese, '%' and '_').
    with _db() as conn:
        rows = conn.execute('''
            SELECT id, owner, kind, content, source_turn, created_at FROM pm_memories WHERE owner=? AND instr(content, ?) > 0
            UNION ALL SELECT id, '', 'knowledge', content, '', created_at FROM pm_public WHERE approved=1 AND instr(content, ?) > 0
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        ''', (owner, query, query, max(1, min(limit, 100)), max(0, offset))).fetchall()
    return [dict(r) for r in rows]


def delete_memory(memory_id: str) -> bool:
    user = _identified_user()
    with _db() as conn:
        deleted = conn.execute('DELETE FROM pm_memories WHERE id=? AND owner=?',
                               (memory_id, user['id'])).rowcount > 0
        if deleted:
            # Raw history may repeat or paraphrase it; retain only other structured records.
            conn.execute('DELETE FROM pm_turns WHERE owner=?', (user['id'],))
            conn.execute('DELETE FROM pm_public WHERE owner=? AND approved=0', (user['id'],))
            conn.execute('UPDATE pm_users SET revision=revision+1 WHERE id=?', (user['id'],))
        return deleted


def propose_public(content: str) -> dict:
    user = _writing_user()
    record = {'id': str(uuid4()), 'owner': user['id'], 'content': _text(content),
              'approved': 0, 'created_at': time.time()}
    with _db() as conn:
        saved = conn.execute('INSERT INTO pm_public (id, owner, content, approved, created_at) '
                             'SELECT :id, :owner, :content, :approved, :created_at FROM pm_users '
                             'WHERE id=:owner AND remember=1 AND paused=0 AND revision=:expected_revision',
                             {**record, 'expected_revision': user['revision']}).rowcount
        if not saved:
            raise PermissionError('保存前授权已变更，未提交公共候选。')
    return record


def public_candidates() -> list[dict]:
    with _db() as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM pm_public WHERE approved=0')]


def approve_public(memory_id: str) -> None:
    with _db() as conn:
        if not conn.execute('UPDATE pm_public SET approved=1 WHERE id=?', (memory_id,)).rowcount:
            raise ValueError('公共知识候选不存在。')


def delete_public(memory_id: str) -> None:
    with _db() as conn:
        conn.execute('DELETE FROM pm_public WHERE id=?', (memory_id,))


def save_turn(turn_id: str, messages: list[dict]) -> None:
    user = get_user(current_user())
    if not user or not user['remember'] or user['paused'] or _block_turn_save.get() or context_changed():
        return
    with _db() as conn:
        conn.execute('INSERT INTO pm_turns (id, owner, messages, created_at) '
                     'SELECT ?, ?, ?, ? FROM pm_users WHERE id=? AND remember=1 AND paused=0 AND revision=? '
                     'ON CONFLICT(id) DO UPDATE SET messages=excluded.messages WHERE pm_turns.owner=excluded.owner',
                     (turn_id, user['id'], json.dumps(messages, ensure_ascii=False), time.time(), user['id'], user['revision']))


def load_turns(limit: int = 18) -> list[list[dict]]:
    user = get_user(current_user())
    if not user or not user['remember']:
        return []
    with _db() as conn:
        rows = conn.execute('SELECT messages FROM pm_turns WHERE owner=? ORDER BY created_at DESC LIMIT ?',
                            (user['id'], max(1, min(limit, 100)))).fetchall()
    return [json.loads(r['messages']) for r in reversed(rows)]


def search_turns(query: str, limit: int = 5) -> list[dict]:
    user = get_user(current_user())
    if not user or not user['remember']:
        return []
    with _db() as conn:
        rows = conn.execute('SELECT id, messages, created_at FROM pm_turns WHERE owner=? AND instr(messages, ?) > 0 '
                            'ORDER BY created_at DESC LIMIT ?', (user['id'], query, max(1, min(limit, 20)))).fetchall()
    return [dict(r) for r in rows]


def delete_user(user_id: str) -> None:
    with _db() as conn:
        for table in ('pm_memories', 'pm_turns', 'pm_public'):
            conn.execute(f'DELETE FROM {table} WHERE owner=?', (user_id,))
        conn.execute('DELETE FROM pm_bindings WHERE user_id=?', (user_id,))
        conn.execute('DELETE FROM pm_users WHERE id=?', (user_id,))


def tool_allowed(name: str) -> bool:
    if not enabled():
        return True
    return name in {'finish', 'memory_recall', 'remember_personal', 'forget_personal',
                    'propose_public_memory', 'set_memory_preferences', 'set_memory_collection',
                    'export_personal_data', 'correct_personal', 'withdraw_public_memory', 'delete_personal_data'} | set(settings().get('output_tools', []))


def set_collection(enabled: bool) -> None:
    user = _identified_user()
    with _db() as conn:
        conn.execute('UPDATE pm_users SET paused=?, remember=?, revision=revision+1 WHERE id=?',
                     (int(not enabled), int(bool(enabled or user['remember'])), user['id']))
        if not enabled:
            conn.execute('DELETE FROM pm_turns WHERE id=? AND owner=?', (_turn.get(), user['id']))


def correct_memory(memory_id: str, content: str, kind: str) -> dict:
    user = _identified_user()
    if kind not in ('fact', 'preference', 'event', 'analysis'):
        raise ValueError('记忆类型必须是 fact/preference/event/analysis。')
    with _db() as conn:
        changed = conn.execute('UPDATE pm_memories SET content=?, kind=?, source_turn=?, created_at=? '
                               'WHERE id=? AND owner=?',
                               (_text(content), kind, _turn.get(), time.time(), memory_id, user['id'])).rowcount
        if not changed:
            raise PermissionError('未找到当前用户的这条记忆。')
        conn.execute('DELETE FROM pm_turns WHERE owner=?', (user['id'],))
        conn.execute('DELETE FROM pm_public WHERE owner=? AND approved=0', (user['id'],))
        conn.execute('UPDATE pm_users SET revision=revision+1 WHERE id=?', (user['id'],))
        return dict(conn.execute('SELECT * FROM pm_memories WHERE id=?', (memory_id,)).fetchone())


def export_data(collection: str, limit: int = 20, offset: int = 0) -> dict:
    user = _identified_user()
    tables = {'memories': ('pm_memories', 'owner'), 'conversations': ('pm_turns', 'owner'),
              'public_contributions': ('pm_public', 'owner'), 'bindings': ('pm_bindings', 'user_id')}
    limit, offset = max(1, min(limit, 100)), max(0, offset)
    if collection == 'profile':
        rows, total = [user], 1
        rows = rows[offset:offset + limit]
    elif collection in tables:
        table, owner_column = tables[collection]
        with _db() as conn:
            total = conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {owner_column}=?', (user['id'],)).fetchone()[0]
            rows = [dict(row) for row in conn.execute(
                f'SELECT * FROM {table} WHERE {owner_column}=? ORDER BY rowid LIMIT ? OFFSET ?',
                (user['id'], limit, offset))]
        if collection == 'conversations':
            for row in rows:
                row['messages'] = json.loads(row['messages'])
    else:
        raise ValueError('collection 必须是 profile/memories/conversations/public_contributions/bindings。')
    return {'collection': collection, 'items': rows, 'total': total,
            'next_offset': offset + len(rows) if offset + len(rows) < total else None}


def withdraw_public(memory_id: str) -> bool:
    user = _identified_user()
    with _db() as conn:
        deleted = conn.execute('DELETE FROM pm_public WHERE id=? AND owner=?', (memory_id, user['id'])).rowcount > 0
        if deleted:
            conn.execute('UPDATE pm_users SET revision=revision+1 WHERE id=?', (user['id'],))
        return deleted


def clear_personal_data(delete_profile: bool = False) -> None:
    user = _identified_user()
    if delete_profile:
        delete_user(user['id'])
        return
    with _db() as conn:
        for table in ('pm_memories', 'pm_turns', 'pm_public'):
            conn.execute(f'DELETE FROM {table} WHERE owner=?', (user['id'],))
        conn.execute('UPDATE pm_users SET remember=0, paused=1, revision=revision+1 WHERE id=?', (user['id'],))
