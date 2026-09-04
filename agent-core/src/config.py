
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
        'update_channel': 'ga',  # preview | release | ga
        'auto_start': False,
    },
    'services': {
        'llm': {'url': '', 'key': '', 'model': ''},
        'tts': {'url': ''},
        'asr': {'provider': 'openai', 'url': '', 'key': '', 'model': '',
                'app_key': '', 'ak_id': '', 'ak_secret': '', 'api_secret': '', 'language': 'zh-CN'},
        'mcp': [],
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
            'max_rounds': 100,                  # 单 turn 触发截断续跑的轮数阈值
            'truncate_keep_rounds': 50,         # 截断时保留最新消息条数
            'compress_threshold_chars': 80000,  # 约 20K tokens，超过此字符数触发压缩（兜底）
            'compress_keep_recent': 6,          # 压缩时保留最近 N 轮不动（旧逻辑兼容）
            'tier1_turns': 6,                   # tiered retention: 全量保留最近 N 轮
            'tier2_turns': 8,                   # tiered retention: 降质保留再往前 N 轮
            'summary_max_chars': 5000,          # rolling summary 最大字符数
            'turn_compact_threshold': 20,       # turn 内消息超过此数触发 compaction
            'turn_compact_keep_recent': 12,     # turn 内 compaction 保留最近 N 条完整
            'save_compact_chars': 500,          # turn 保存时 tool result 截断到此长度
            'source_ring_size': 50,             # per-source ring buffer 大小（供 raw_input_info 查询）
            'interrupt_mode': 'steer',          # 打断模式: steer | interrupt | followup
            'barge_in_threshold_ms': 500,       # 语音 barge-in 阈值（ms），低于此值视为 backchannel
        },
        'subscribe_topics': [],  # DDS topics core subscribes to directly (e.g. ["/robot/mic/audio/asr_event"])
    },
    'scheduler': [],
    'skills': {'installed': []},
    'channel_configs': [],
    'channel_settings': {
        'default_role': 'viewer',
        'auto_approve': True,
        'require_actuator_confirm': True,
    },
    'peer_settings': {
        'enabled': False,
        # 广播给同网段的展示名。空则用 hostname。
        'display_name': '',
        # 本机对外可达的地址，供 peer 回连；空则由 mDNS 用网卡地址填。
        'advertise_url': '',
        'discovery': {'mdns': True, 'static': []},
        # 新配对的 peer 默认角色。刻意不提供 auto_approve —— 配对必须有人确认。
        'default_role': 'viewer',
        # 签名的时间窗（秒）。离网机器人时钟可能漂移，必要时放宽。
        'clock_skew_s': 120,
    },
    'subagent': {
        'max_concurrent': 2,
        'max_total': 10,
        'default_max_rounds': 50,
        'default_timeout_s': 300,
        'preemption_enabled': True,
        'checkpoint_interval': 5,
        'compress_threshold_chars': 20000,
        'cleanup_age_hours': 24,
        'bg_route_enabled': True,
        'bg_model': None,  # None = use main model; or specify e.g. 'qwen-turbo'
    },
    'desktop_tools': {
        'enabled': True,
        'allowed_dirs': ['/work', '/tmp'],
        'bash_blocked_patterns': ['rm -rf /', 'rm -rf /*', 'mkfs', 'reboot', 'shutdown', 'poweroff'],
        'python_allowed_modules': ['math', 'json', 're', 'datetime', 'collections', 'itertools',
                                   'struct', 'pathlib', 'numpy', 'hashlib', 'base64', 'urllib.parse'],
        'max_output_bytes': 51200,
        'search': {
            'type': 'none',       # 'none' | 'baidu_search'
            'base_url': '',
            'api_key': '',
        },
    },
    'llm_logger': {
        'enabled': True,
        'data_dir': './resource/llm_data',
        'recent_dir': './resource/llm_recent_request',
        'batch_size': 500,
        'max_records': 50000,
        'recent_max_per_dir': 100,
    },
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
    conn.execute('''
        CREATE TABLE IF NOT EXISTS channel_users (
            platform TEXT NOT NULL,
            platform_user_id TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            role TEXT DEFAULT 'viewer',
            tool_filter TEXT DEFAULT '*',
            alert_subscriptions TEXT DEFAULT '[]',
            created_at REAL,
            UNIQUE(platform, platform_user_id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS perf_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            source TEXT DEFAULT '',
            trigger_text TEXT DEFAULT '',
            total_duration_ms INTEGER
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_perf_created ON perf_turns(created_at)')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS perf_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            span TEXT NOT NULL,
            component TEXT NOT NULL,
            start_ts REAL NOT NULL,
            end_ts REAL,
            duration_ms INTEGER,
            meta TEXT DEFAULT '{}',
            created_at REAL NOT NULL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_spans_trace ON perf_spans(trace_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_spans_created ON perf_spans(created_at)')
    # ── subagent 结论存储（memory_recall 检索用）──────────────────────────────
    conn.execute('''
        CREATE TABLE IF NOT EXISTS subagent_conclusions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            goal TEXT DEFAULT '',
            conclusion TEXT NOT NULL,
            source_type TEXT DEFAULT 'bg_monitor',
            created_at REAL NOT NULL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_conclusions_ts ON subagent_conclusions(created_at)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_conclusions_type ON subagent_conclusions(source_type)')
    # ── 已配对的 peer（另一台 Agent Core）─────────────────────────────────────
    # peer_id 是 Ed25519 公钥指纹，不是 IP，也不是平台账号 —— 同一个 peer 从
    # mDNS / 云名册多条路径被发现时仍是同一行，这是链路降级能成立的前提。
    # role / tool_filter 与 channel_users 共用 acl.py 的那套取值。
    conn.execute('''
        CREATE TABLE IF NOT EXISTS peers (
            peer_id TEXT PRIMARY KEY,
            display_name TEXT DEFAULT '',
            public_key TEXT NOT NULL,
            role TEXT DEFAULT 'viewer',
            tool_filter TEXT DEFAULT '*',
            endpoints TEXT DEFAULT '[]',
            capabilities TEXT DEFAULT '[]',
            paired_at REAL,
            last_seen REAL,
            -- When we last had evidence that the peer has *us* in its own table.
            -- Pairing is per-direction, so confirming here proves nothing about the
            -- other side: without this, a half-finished pairing looked complete on
            -- the side that confirmed, and the failure only surfaced later as 403s.
            mutual_at REAL
        )
    ''')
    # Added after the table shipped; an existing database must not be discarded
    # just because it predates the column.
    cols = {r[1] for r in conn.execute('PRAGMA table_info(peers)')}
    if 'mutual_at' not in cols:
        conn.execute('ALTER TABLE peers ADD COLUMN mutual_at REAL')
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
