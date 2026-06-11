
import os
import sqlite3
import json
import pathlib


# ── .env 加载 ─────────────────────────────────────────────────────────────────

def _load_dotenv():
    env_file = pathlib.Path('.env')
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()


# ── 部署级配置（env）──────────────────────────────────────────────────────────

DB_PATH = os.environ.get('DB_PATH', './resource/data.db')


# ── SQLite 配置存储 ───────────────────────────────────────────────────────────

_DB_DEFAULTS = {
    'core': {
        'main_loop_enable': True,
        'configured': False,
    },
    'services': {
        'llm': {'url': '', 'key': '', 'model': ''},
        'tts': {'url': ''},
        'asr': {'provider': 'openai', 'url': '', 'key': '', 'model': '',
                'app_key': '', 'ak_id': '', 'ak_secret': '', 'api_secret': '', 'language': 'zh-CN'},
        'mcp': [],
        'inspector': {'url': 'http://localhost:15671'},
        'resource_center': {'url': 'https://motus.phanthy.com'},
    },
    'client': {
        'llm': [],
    },
    'event': {
        'llm': {
            'memory_count_limit': 50,
            'prompt_system': './resource/memory/prompt_system.md',
            'prompt_memory':  './resource/memory/prompt_memory.md',
            'trigger_interval_ms': 1000,
            'collector_max_window': 20,
            'history_turns': 30,
            'compress_threshold_chars': 80000,  # 约 20K tokens，超过此字符数触发压缩
            'compress_keep_recent': 6,          # 压缩时保留最近 N 轮不动
        },
        'subscribe_topics': [],  # DDS topics core subscribes to directly (e.g. ["/robot/mic/audio/asr_event"])
    },
    'scheduler': [],
    'skills': {'installed': []},
}


def _get_conn() -> sqlite3.Connection:
    db_path = pathlib.Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE IF NOT EXISTS config '
        '(key TEXT PRIMARY KEY, value TEXT NOT NULL)'
    )
    conn.execute(
        'CREATE TABLE IF NOT EXISTS chat_sessions '
        '(id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, '
        'summary TEXT DEFAULT \'\', turn_count INTEGER DEFAULT 0)'
    )
    conn.execute(
        'CREATE TABLE IF NOT EXISTS chat_messages '
        '(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, '
        'turn_index INTEGER NOT NULL, messages TEXT NOT NULL, created_at REAL NOT NULL)'
    )
    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_cm_session ON chat_messages(session_id, turn_index)'
    )
    conn.commit()
    return conn


def _seed_defaults():
    with _get_conn() as conn:
        for k, v in _DB_DEFAULTS.items():
            conn.execute(
                'INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)',
                (k, json.dumps(v))
            )
        conn.commit()

_seed_defaults()


def _migrate():
    """One-time data migrations to fix stale values from previous versions."""
    with _get_conn() as conn:
        # Dedup MCP list by id (keep last occurrence)
        row_svc = conn.execute("SELECT value FROM config WHERE key='services'").fetchone()
        if row_svc:
            svc = json.loads(row_svc[0])
            mcp_list = svc.get('mcp', [])
            seen_ids: dict = {}
            for m in mcp_list:
                seen_ids[m['id']] = m
            deduped = list(seen_ids.values())
            # Also dedup by URL — keep the entry with tools, else keep last
            seen_urls: dict = {}
            for m in deduped:
                url = m.get('url', '')
                if not url:
                    seen_urls[f'__no_url_{id(m)}'] = m
                    continue
                prev = seen_urls.get(url)
                if prev is None or (not prev.get('tools') and m.get('tools')):
                    seen_urls[url] = m
            deduped = list(seen_urls.values())
            if len(deduped) < len(mcp_list):
                svc['mcp'] = deduped
                conn.execute("UPDATE config SET value=? WHERE key='services'", (json.dumps(svc),))
                conn.commit()
                print(f'[config] deduped {len(mcp_list) - len(deduped)} duplicate MCP entries')

_migrate()


class ConfigDB:
    def __getitem__(self, key: str):
        with _get_conn() as conn:
            row = conn.execute('SELECT value FROM config WHERE key = ?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key: str, value):
        with _get_conn() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
                (key, json.dumps(value))
            )
            conn.commit()

    def __contains__(self, key: str) -> bool:
        with _get_conn() as conn:
            row = conn.execute('SELECT 1 FROM config WHERE key = ?', (key,)).fetchone()
        return row is not None

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


main = ConfigDB()


# ── 读取文件内容（保持原接口）─────────────────────────────────────────────────

def load(key_chain):
    value = main
    for key in key_chain.split('.'):
        value = value[key]

    path = pathlib.Path(value)
    match path.suffix.lower():
        case '.json':
            value = path.read_text()
            value = json.loads(value)
        case '.txt' | '.md':
            value = path.read_text()
        case _:
            return ''
    return value
