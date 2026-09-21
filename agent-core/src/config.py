
import os
import sqlite3
import json
import pathlib
from contextlib import closing


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
            # 主动播报：距上次面向用户的输出多久没动静就自动生成并播报一句进展汇报。
            # 0 = 关闭。播放期间不计时（正在说话时根本没有计时器）。
            'auto_narration': True,             # 长时间不出声时由系统代为播报进展的总开关
            'narration_silence_seconds': 15,
            'narration_context_chars': 6000,    # 喂给汇报调用的上下文预算（取 turn 尾部）
            'narration_timeout_s': 20,          # 汇报调用硬超时，超时视为本次放弃
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
        # ble 默认关闭：它要主机侧先解 rfkill、开 bluetoothd，还要 dbus socket 挂进容器。
        # 默认开启会让 provider 常态报错，而这类"红着也没人管"的告警很快就没人看了。
        'discovery': {'mdns': True, 'static': [], 'ble': False},
        # 新配对的 peer 默认角色。刻意不提供 auto_approve —— 配对必须有人确认。
        'default_role': 'viewer',
        # 签名的时间窗（秒）。离网机器人时钟可能漂移，必要时放宽。
        'clock_skew_s': 120,
    },
    'subagent': {
        'max_concurrent': 2,
        'max_total': 10,
        'default_max_rounds': 50,
        'default_timeout_s': 600,
        'preemption_enabled': True,
        'checkpoint_interval': 5,
        'compress_threshold_chars': 40000,
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
        'summary TEXT DEFAULT \'\', turn_count INTEGER DEFAULT 0, '
        'kind TEXT DEFAULT \'main\')'
    )
    conn.execute(
        'CREATE TABLE IF NOT EXISTS chat_messages '
        '(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, '
        'turn_index INTEGER NOT NULL, messages TEXT NOT NULL, created_at REAL NOT NULL, '
        'updated_at REAL)'
    )
    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_cm_session ON chat_messages(session_id, turn_index)'
    )
    # Columns added after the tables shipped; existing DBs need them backfilled.
    for table, column, ddl in (
        ('chat_sessions', 'kind', 'kind TEXT DEFAULT \'main\''),
        ('chat_messages', 'updated_at', 'updated_at REAL'),
    ):
        try:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {ddl}')
        except sqlite3.OperationalError:
            pass  # already present
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
    # ── peer 关系变更的审计轨迹 ───────────────────────────────────────────────
    # `peers` 只记录「现在是什么」。删掉一行之后，那段关系就不再有任何痕迹 ——
    # 有人在天轶上解除了和 Orin5 的配对，一周后想查是谁、什么时候干的，
    # `docker logs` 早已轮换过去，活动流是内存里的，数据库里只剩一张空表。
    # 这张表存的是「发生过什么」，所以它只增不改。
    conn.execute('''
        CREATE TABLE IF NOT EXISTS peer_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            peer_id TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            -- paired | unpaired_local | unpaired_by_peer | role_changed
            event TEXT NOT NULL,
            -- 'local'（这台机器上的操作员）或对方的 peer_id
            actor TEXT DEFAULT '',
            detail TEXT DEFAULT ''
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_peer_audit_ts ON peer_audit(ts)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_peer_audit_peer ON peer_audit(peer_id)')
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

        # subagent 的两个旧默认值：压缩阈值 20000 → 40000，空转超时 300 → 600。
        #
        # _seed_defaults 用的是 INSERT OR IGNORE，整行粒度：已部署机器上的 'subagent'
        # 行早就存在，新默认值永远进不去。Orin5 实测读出来就是 20000/300。
        #
        # 20000 的代价：一次 WebSearch 的结果就超过它，压缩几乎每轮触发、只留最近两轮，
        # 子代理因此忘掉自己刚查到的东西并反复重查（同一个问题搜了 round 2、6、9）。
        # 300 的代价：轮次预算放到 50 之后，先撞上的会是这个空转超时，而超时是 cancel，
        # cancel 不会走收尾调用 —— 又变回什么都拿不回来。
        #
        # 只改还停在旧默认值上的键；有人手工调过就不动。
        _stale_subagent_defaults = {
            'compress_threshold_chars': (20000, 40000),
            'default_timeout_s': (300, 600),
        }
        row_sa = conn.execute("SELECT value FROM config WHERE key='subagent'").fetchone()
        if row_sa:
            sa = json.loads(row_sa[0])
            changed = []
            for key, (old, new) in _stale_subagent_defaults.items():
                if sa.get(key) == old:
                    sa[key] = new
                    changed.append(f'{key} {old} -> {new}')
            if changed:
                conn.execute("UPDATE config SET value=? WHERE key='subagent'",
                             (json.dumps(sa),))
                conn.commit()
                print(f'[config] subagent: {", ".join(changed)}')

        # event.llm 新增的主动播报键。同上面那段：_seed_defaults 是整行粒度的
        # INSERT OR IGNORE，已部署机器上的 'event' 行早就存在，新默认值永远进不去 ——
        # 结果会是阈值读成 0，功能静默不生效。只补缺失的键，手工调过的值不动。
        _narration_defaults = {
            'auto_narration': True,
            'narration_silence_seconds': 15,
            'narration_context_chars': 6000,
            'narration_timeout_s': 20,
        }
        # 已删除的键。轮数维度取消后（纯时间触发，定时器全局负责），这个键没有任何读者，
        # 留在库里只会让人对着设置页猜"它还管不管用"。和上面的补种合并成一次
        # read-modify-write —— 拆成两段就要对同一行读写两次，中间还多一个失败窗口。
        # auto_notify 换名成 auto_narration，**不继承旧值**。
        #
        # 旧键的含义是"把模型写的 content 自动念出来"。有人（比如 Tianyi）因为那功能念的是
        # 内部推理而把它关掉了 —— 一个完全合理的决定。现在 content 自动播报已经废除，同一个
        # 键被重新定义成"框架进度播报的总开关"，于是那个旧决定会静默地把一个它从没评价过的
        # 新功能也关死，而设置页上看不出任何异常（Tianyi 就是人工打开才恢复的）。
        #
        # 语义变了就换键：新键按默认值 True 生效，旧键删掉。这是替操作员重新做决定，但
        # 他当初拒绝的那个东西已经不存在了，让一个作废的决定继续生效更糟。
        _narration_removed = ('narration_silence_rounds', 'auto_notify')

        # 改过的默认值：沉默阈值 25 → 15 秒。25 是几小时前由上面这段自己种进去的，不是
        # 谁选的，所以停在 25 的机器要跟着改；手工调过（比如 90）的不动。同 subagent 那段
        # 的做法，(旧默认, 新默认)。
        _stale_narration_defaults = {
            'narration_silence_seconds': (25, 15),
        }

        row_ev = conn.execute("SELECT value FROM config WHERE key='event'").fetchone()
        if row_ev:
            ev = json.loads(row_ev[0])
            llm_cfg = ev.setdefault('llm', {})
            added = [k for k in _narration_defaults if k not in llm_cfg]
            for k in added:
                llm_cfg[k] = _narration_defaults[k]
            dropped = [k for k in _narration_removed if k in llm_cfg]
            for k in dropped:
                llm_cfg.pop(k, None)
            retuned = []
            for k, (old_v, new_v) in _stale_narration_defaults.items():
                if llm_cfg.get(k) == old_v:
                    llm_cfg[k] = new_v
                    retuned.append(f'{k} {old_v} -> {new_v}')
            if added or dropped or retuned:
                conn.execute("UPDATE config SET value=? WHERE key='event'", (json.dumps(ev),))
                conn.commit()
                _msg = []
                if added:
                    _msg.append(f'seeded {", ".join(added)}')
                if dropped:
                    _msg.append(f'dropped {", ".join(dropped)}')
                if retuned:
                    _msg.append(f'retuned {", ".join(retuned)}')
                print(f'[config] event.llm: {"; ".join(_msg)}')

_migrate()


class ConfigDB:
    def update_atomic(self, values, *, delete_keys=(), delete_prefix=None):
        """Commit related config rows together; any failure rolls back all rows."""
        removed = 0
        with closing(_get_conn()) as conn, conn:
            for key in delete_keys:
                removed += conn.execute('DELETE FROM config WHERE key = ?', (key,)).rowcount
            if delete_prefix is not None:
                removed += conn.execute('DELETE FROM config WHERE substr(key, 1, ?) = ?',
                                        (len(delete_prefix), delete_prefix)).rowcount
            for key, value in values.items():
                conn.execute('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
                             (key, json.dumps(value)))
        return removed

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
