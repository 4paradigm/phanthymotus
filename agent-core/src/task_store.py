"""
task_store.py — 结构化任务存储。

提供长时间任务的生命周期管理：创建、更新、完成、失败。
数据持久化到 SQLite（与 config.py 共用 DB_PATH），重启后恢复活跃任务。
"""

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field

import config


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ''
    status: str = 'running'       # pending | running | paused | done | failed
    progress: str = ''
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    check_cron: str = ''          # 空则不自动检查
    metadata: dict = field(default_factory=dict)


# ── 内存缓存 ──────────────────────────────────────────────────────────────────

_tasks: dict[str, Task] = {}


# ── SQLite 操作 ───────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    import pathlib
    db_path = pathlib.Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


def _persist(task: Task) -> None:
    with _get_conn() as conn:
        conn.execute(
            'INSERT OR REPLACE INTO tasks (id, data) VALUES (?, ?)',
            (task.id, json.dumps(asdict(task), ensure_ascii=False))
        )
        conn.commit()


def _delete_from_db(task_id: str) -> None:
    with _get_conn() as conn:
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
        conn.commit()


def load_all() -> None:
    """启动时从 SQLite 加载所有活跃任务到内存。"""
    with _get_conn() as conn:
        rows = conn.execute('SELECT id, data FROM tasks').fetchall()
    for _, data_str in rows:
        data = json.loads(data_str)
        task = Task(**data)
        # 只恢复活跃任务
        if task.status in ('running', 'paused', 'pending'):
            _tasks[task.id] = task
        else:
            # 已完成/失败的从 DB 清除
            _delete_from_db(task.id)


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def create(goal: str, check_cron: str = '', metadata: dict | None = None) -> Task:
    task = Task(
        goal=goal,
        check_cron=check_cron,
        metadata=metadata or {},
    )
    _tasks[task.id] = task
    _persist(task)
    return task


def update(task_id: str, **kwargs) -> Task | None:
    task = _tasks.get(task_id)
    if not task:
        return None
    for k, v in kwargs.items():
        if hasattr(task, k) and v is not None:
            setattr(task, k, v)
    task.updated_at = time.time()
    _persist(task)
    return task


def done(task_id: str, summary: str = '') -> Task | None:
    task = _tasks.get(task_id)
    if not task:
        return None
    task.status = 'done'
    task.progress = summary or task.progress
    task.updated_at = time.time()
    _persist(task)
    del _tasks[task_id]
    return task


def fail(task_id: str, reason: str = '') -> Task | None:
    task = _tasks.get(task_id)
    if not task:
        return None
    task.status = 'failed'
    task.progress = reason or task.progress
    task.updated_at = time.time()
    _persist(task)
    del _tasks[task_id]
    return task


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)


def active_tasks() -> list[Task]:
    """返回所有活跃任务（running/paused/pending）。"""
    return list(_tasks.values())
