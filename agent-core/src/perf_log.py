"""
perf_log.py — 开放 Span 式性能追踪。

每个组件（perception、core、driver）上报命名 span，
agent-core 收集后按 trace_id（turn_id）关联存储到 SQLite。

Span 格式：
  {"span": "asr_inference", "component": "perception",
   "start_ts": float, "end_ts": float, "meta": {...}}
"""

import json
import time
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import config


def _get_conn():
    conn = config._get_conn()
    # Ensure token_usage table exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cached_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_token_usage_created ON token_usage(created_at)')
    return conn


def commit_spans(trace_id: str, spans: list[dict], source: str = '', trigger_text: str = ''):
    """写入一组 spans 到 perf_spans 表，同时写/更新 perf_turns 索引。"""
    if not spans:
        return
    now = time.time()
    conn = _get_conn()

    # 检查 turn 是否已存在（TTS 等异步 span 会后到）
    existing = conn.execute(
        'SELECT id FROM perf_turns WHERE turn_id=?', (trace_id,)
    ).fetchone()

    if not existing:
        # 新建 perf_turns 记录（total_duration_ms 在写完 spans 后统一算）
        conn.execute(
            '''INSERT INTO perf_turns (turn_id, created_at, source, trigger_text, total_duration_ms)
               VALUES (?, ?, ?, ?, ?)''',
            (trace_id, now, source, trigger_text[:200], None),
        )

    # 写 perf_spans
    for s in spans:
        start_ts = s.get('start_ts')
        end_ts = s.get('end_ts')
        dur = None
        if start_ts and end_ts and start_ts > 1e9 and end_ts > 1e9:
            dur = int((end_ts - start_ts) * 1000)
        conn.execute(
            '''INSERT INTO perf_spans (trace_id, span, component, start_ts, end_ts, duration_ms, meta, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                trace_id,
                s.get('span', ''),
                s.get('component', ''),
                start_ts,
                end_ts,
                dur,
                json.dumps(s.get('meta', {}), ensure_ascii=False),
                now,
            ),
        )

    # 总时长从库里所有 span 重算 —— turn 跑到一半就会提交一批 spans（让性能面板能看到
    # 正在跑的 turn），后到的那批不能让已经写下的总时长停在第一批的值上。
    conn.execute(
        '''UPDATE perf_turns SET total_duration_ms = (
               SELECT CAST((MAX(end_ts) - MIN(start_ts)) * 1000 AS INTEGER)
               FROM perf_spans WHERE trace_id = ? AND start_ts > 1e9 AND end_ts > 1e9
           ) WHERE turn_id = ?''',
        (trace_id, trace_id),
    )

    conn.commit()
    conn.close()


def query_latest(n: int = 20) -> list:
    """返回最近 N 个 turn，每个 turn 附带其 spans。"""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row

    turns = conn.execute(
        'SELECT * FROM perf_turns ORDER BY created_at DESC LIMIT ?', (n,)
    ).fetchall()

    result = []
    for t in turns:
        td = dict(t)
        trace_id = td['turn_id']
        spans = conn.execute(
            'SELECT span, component, start_ts, end_ts, duration_ms, meta FROM perf_spans WHERE trace_id=? ORDER BY start_ts',
            (trace_id,),
        ).fetchall()
        td['spans'] = []
        for s in spans:
            sd = dict(s)
            try:
                sd['meta'] = json.loads(sd['meta'])
            except (json.JSONDecodeError, TypeError):
                sd['meta'] = {}
            td['spans'].append(sd)
        result.append(td)

    conn.close()
    return result


def turns_between(start: float, end: float) -> list:
    """一个时间窗内的 turn 及其 spans，按时间正序。

    给基准测试的「运行详情」用：spans 里有每次工具调用的**真实起止时刻**，
    而会话历史只有一轮被写完的时刻。两者相差一整轮的时长 —— 左右两栏于是对不上。
    """
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    turns = conn.execute(
        'SELECT * FROM perf_turns WHERE created_at BETWEEN ? AND ? ORDER BY created_at',
        (float(start), float(end))).fetchall()
    result = []
    for t in turns:
        td = dict(t)
        td['spans'] = [dict(s) for s in conn.execute(
            'SELECT span, component, start_ts, end_ts, duration_ms FROM perf_spans '
            'WHERE trace_id=? ORDER BY start_ts', (td['turn_id'],)).fetchall()]
        result.append(td)
    conn.close()
    return result


def spans_between(start: float, end: float) -> list:
    """一个时间窗内的 spans，不分 turn。

    和 `turns_between` 的区别是**不按轮分组**：基准测试的指标要算的是「这段时间里推理
    一共占了多久、工具重叠了多少」，分组反而要再摊平一次。

    按 `start_ts` 筛而不是 `created_at` —— 后者是写入时刻，一轮结束才写，用它筛会把
    运行末尾那几轮整个漏掉。
    """
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT span, component, start_ts, end_ts, duration_ms FROM perf_spans '
        'WHERE start_ts BETWEEN ? AND ? ORDER BY start_ts',
        (float(start), float(end))).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def query_spans(trace_id: str) -> list:
    """返回单个 turn 的全部 spans。"""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    spans = conn.execute(
        'SELECT * FROM perf_spans WHERE trace_id=? ORDER BY start_ts', (trace_id,)
    ).fetchall()
    result = []
    for s in spans:
        sd = dict(s)
        try:
            sd['meta'] = json.loads(sd['meta'])
        except (json.JSONDecodeError, TypeError):
            sd['meta'] = {}
        result.append(sd)
    conn.close()
    return result


def aggregate(start: float = 0, end: float = 0) -> dict:
    """按 span 名称聚合 avg/p95。"""
    conn = _get_conn()

    where = 'WHERE duration_ms IS NOT NULL'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    # 总 turn 数
    turn_count = conn.execute(
        f'SELECT COUNT(DISTINCT trace_id) FROM perf_spans {where}', params
    ).fetchone()[0]

    # 按 span 名称聚合
    rows = conn.execute(
        f'''SELECT span, COUNT(*) as cnt, AVG(duration_ms) as avg_ms
            FROM perf_spans {where}
            GROUP BY span ORDER BY avg_ms DESC''',
        params,
    ).fetchall()

    by_span = {}
    for row in rows:
        span_name = row[0]
        cnt = row[1]
        avg_ms = int(row[2]) if row[2] else 0

        # P95: index of the 95th percentile value in ascending order
        import math
        p95_offset = min(cnt - 1, math.ceil(cnt * 0.95) - 1)
        p95_row = conn.execute(
            f'SELECT duration_ms FROM perf_spans {where} AND span=? ORDER BY duration_ms ASC LIMIT 1 OFFSET ?',
            params + [span_name, p95_offset],
        ).fetchone()
        p95_ms = p95_row[0] if p95_row else avg_ms

        # 平均开始偏移（相对于每个 trace 中最早 span 的 start_ts）
        offset_row = conn.execute(
            f'''SELECT AVG((s.start_ts - t_min.min_start) * 1000) FROM perf_spans s
                INNER JOIN (SELECT trace_id, MIN(start_ts) as min_start FROM perf_spans GROUP BY trace_id) t_min
                ON s.trace_id = t_min.trace_id
                {where} AND s.span = ?''',
            params + [span_name],
        ).fetchone()
        avg_offset_ms = int(offset_row[0]) if offset_row and offset_row[0] else 0

        by_span[span_name] = {'avg_ms': avg_ms, 'p95_ms': p95_ms, 'count': cnt, 'avg_offset_ms': avg_offset_ms}

    conn.close()
    return {'count': turn_count, 'by_span': by_span}


def prune(days: int = 90):
    """清理过期记录。"""
    cutoff = time.time() - days * 86400
    conn = _get_conn()
    conn.execute('DELETE FROM perf_spans WHERE created_at < ?', (cutoff,))
    conn.execute('DELETE FROM perf_turns WHERE created_at < ?', (cutoff,))
    conn.execute('DELETE FROM token_usage WHERE created_at < ?', (cutoff,))
    conn.commit()
    conn.close()


# ── Token Usage ───────────────────────────────────────────────────────────────

def record_usage(trace_id: str, usage: dict):
    """Record token usage for a single LLM call."""
    if not usage:
        return
    conn = _get_conn()
    conn.execute(
        '''INSERT INTO token_usage (trace_id, prompt_tokens, completion_tokens, cached_tokens, total_tokens, created_at)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (
            trace_id,
            usage.get('prompt_tokens', 0),
            usage.get('completion_tokens', 0),
            usage.get('cached_tokens', 0),
            usage.get('total_tokens', 0),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def query_usage_summary(start: float = 0, end: float = 0) -> dict:
    """Aggregate total usage within time range."""
    conn = _get_conn()
    where = 'WHERE 1=1'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    row = conn.execute(
        f'''SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),
                   COALESCE(SUM(cached_tokens),0), COALESCE(SUM(total_tokens),0),
                   COUNT(*)
            FROM token_usage {where}''',
        params,
    ).fetchone()
    conn.close()
    return {
        'prompt_tokens': row[0],
        'completion_tokens': row[1],
        'cached_tokens': row[2],
        'total_tokens': row[3],
        'call_count': row[4],
    }


def query_usage_daily(start: float = 0, end: float = 0) -> list:
    """Aggregate usage grouped by day (UTC+8)."""
    conn = _get_conn()
    where = 'WHERE 1=1'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    # Group by date (UTC+8: +28800 seconds offset)
    rows = conn.execute(
        f'''SELECT date(created_at + 28800, 'unixepoch') as day,
                   SUM(prompt_tokens), SUM(completion_tokens),
                   SUM(cached_tokens), SUM(total_tokens), COUNT(*)
            FROM token_usage {where}
            GROUP BY day ORDER BY day DESC''',
        params,
    ).fetchall()
    conn.close()
    return [
        {
            'date': row[0],
            'prompt_tokens': row[1] or 0,
            'completion_tokens': row[2] or 0,
            'cached_tokens': row[3] or 0,
            'total_tokens': row[4] or 0,
            'call_count': row[5],
        }
        for row in rows
    ]


def query_usage_hourly(start: float = 0, end: float = 0) -> list:
    """Aggregate usage grouped by hour (UTC+8)."""
    conn = _get_conn()
    where = 'WHERE 1=1'
    params = []
    if start:
        where += ' AND created_at >= ?'
        params.append(start)
    if end:
        where += ' AND created_at <= ?'
        params.append(end)

    rows = conn.execute(
        f'''SELECT strftime('%Y-%m-%d %H', created_at + 28800, 'unixepoch') as hour,
                   SUM(prompt_tokens), SUM(completion_tokens),
                   SUM(cached_tokens), SUM(total_tokens), COUNT(*)
            FROM token_usage {where}
            GROUP BY hour ORDER BY hour DESC''',
        params,
    ).fetchall()
    conn.close()
    return [
        {
            'date': row[0],  # "2026-07-29 14"
            'prompt_tokens': row[1] or 0,
            'completion_tokens': row[2] or 0,
            'cached_tokens': row[3] or 0,
            'total_tokens': row[4] or 0,
            'call_count': row[5],
        }
        for row in rows
    ]


# 模块加载时自动清理
try:
    prune()
except Exception:
    pass
