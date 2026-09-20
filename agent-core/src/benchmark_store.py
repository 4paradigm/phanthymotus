"""benchmark_store.py — 一次 benchmark 运行的结果存储。

这不是仿真器的一部分：仿真器负责跑和判定，这里只负责**记下来、能回看、能比较**。
裁判与被测系统保持不相交，所以 agent-core 不参与评分。

## 为什么每条记录都必须带「被测配置」

一个不绑定配置的分数是噪音。真正有人关心的坐标轴是「**换了模型 / 改了 prompt /
换了镜像之后，分数动没动**」，而这个问题在缺少 llm_model、镜像 tag 与 git sha 时
根本无法回答。所以这些字段在写入时就一并落盘，而不是事后去猜。

## 为什么 n 是一等字段

LLM 是随机的，单次运行的分数是分布里的一个样本。`n=1` 的分数看起来像结论，其实
是一次抛硬币 —— 这是 benchmark 和演示之间的分界线。所以 `n_repeats` 与 `stdev`
和 `mean` 一起存，UI 也一起显示。

表建在访问时（`CREATE TABLE IF NOT EXISTS`），与 `perf_log.py` 同样的做法。
"""

import json
import time
import uuid

import config

_SCHEMA = (
    '''
    CREATE TABLE IF NOT EXISTS benchmark_run (
        id           TEXT PRIMARY KEY,
        suite        TEXT NOT NULL,
        tier         TEXT NOT NULL DEFAULT 'fidelity',
        status       TEXT NOT NULL DEFAULT 'running',
        started_at   REAL NOT NULL,
        ended_at     REAL,
        n_repeats    INTEGER NOT NULL DEFAULT 1,
        llm_model    TEXT DEFAULT '',
        llm_provider TEXT DEFAULT '',
        host         TEXT DEFAULT '',
        image_tags   TEXT DEFAULT '{}',
        git_shas     TEXT DEFAULT '{}',
        score_total  REAL,
        score_stdev  REAL,
        scores_by_dim TEXT DEFAULT '{}',
        detail       TEXT DEFAULT ''
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS benchmark_case (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL,
        scenario      TEXT NOT NULL,
        repeat_idx    INTEGER NOT NULL DEFAULT 0,
        seed          INTEGER DEFAULT 0,
        ok            INTEGER DEFAULT 0,
        outcome       TEXT DEFAULT '',
        score         REAL,
        elapsed_ms    INTEGER,
        assertions    TEXT DEFAULT '[]',
        artifacts_ref TEXT DEFAULT ''
    )
    ''',
    # 本机用例库。表名**不是** benchmark_case —— 那个名字已经被「一次运行里的一个
    # repeat」占了。同名不同义在这一轮已经坑过三次（kind / result / state），不再
    # 来第四次：这里存的是「要跑什么」，上面那张存的是「跑出了什么」。
    '''
    CREATE TABLE IF NOT EXISTS case_library (
        id         TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        origin     TEXT DEFAULT '',
        updated_at REAL NOT NULL,
        payload    TEXT NOT NULL
    )
    ''',
    'CREATE INDEX IF NOT EXISTS idx_benchmark_run_started ON benchmark_run(started_at)',
    'CREATE INDEX IF NOT EXISTS idx_benchmark_case_run ON benchmark_case(run_id)',
)


# 加列用 ALTER，不能靠 CREATE TABLE IF NOT EXISTS —— 表已经存在的机器上，新字段
# 不会自己长出来。重复执行会报 duplicate column，吞掉即可。
_ADDITIONS = (
    'ALTER TABLE benchmark_case ADD COLUMN facts TEXT',
    'ALTER TABLE benchmark_run ADD COLUMN session_id TEXT',
    'ALTER TABLE benchmark_run ADD COLUMN agent_track TEXT',
)


def _get_conn():
    conn = config._get_conn()
    for statement in _SCHEMA:
        conn.execute(statement)
    for statement in _ADDITIONS:
        try:
            conn.execute(statement)
        except Exception:
            pass        # 已经有这一列
    return conn


def _dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return '{}'


def _loads(raw, fallback):
    try:
        return json.loads(raw) if raw else fallback
    except (TypeError, ValueError):
        return fallback


def create_run(suite: str, *, tier: str = 'fidelity', n_repeats: int = 1,
               llm_model: str = '', llm_provider: str = '', host: str = '',
               image_tags: dict | None = None, git_shas: dict | None = None,
               session_id: str = '') -> str:
    run_id = f'bm-{int(time.time())}-{uuid.uuid4().hex[:6]}'
    conn = _get_conn()
    conn.execute(
        'INSERT INTO benchmark_run (id, suite, tier, status, started_at, n_repeats, '
        'llm_model, llm_provider, host, image_tags, git_shas, session_id) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        (run_id, suite, tier, 'running', time.time(), max(1, int(n_repeats)),
         llm_model, llm_provider, host, _dumps(image_tags or {}), _dumps(git_shas or {}),
         session_id))
    conn.commit()
    return run_id


def add_case(run_id: str, *, scenario: str, repeat_idx: int = 0, seed: int = 0,
             ok: bool = False, outcome: str = '', score: float | None = None,
             elapsed_ms: int | None = None, assertions: list | None = None,
             artifacts_ref: str = '', facts: dict | None = None) -> None:
    """记一次 repeat 的结果。

    `facts` 是驱动那一侧的完整事实（事件流、播报记录、ACP 上报）。存下来，是因为
    仿真器的世界**下一次运行一开始就被重置**了 —— 不在这里留一份，一次运行结束之后
    就再也没法回看它到底发生了什么，而「分数为什么是这个」恰恰只能从那里回答。
    """
    conn = _get_conn()
    conn.execute(
        'INSERT INTO benchmark_case (run_id, scenario, repeat_idx, seed, ok, outcome, '
        'score, elapsed_ms, assertions, artifacts_ref, facts) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (run_id, scenario, int(repeat_idx), int(seed), 1 if ok else 0, outcome,
         score, elapsed_ms, _dumps(assertions or []), artifacts_ref,
         _dumps(facts or {})))
    conn.commit()


def finish_run(run_id: str, *, status: str = 'done', score_total: float | None = None,
               score_stdev: float | None = None, scores_by_dim: dict | None = None,
               detail: str = '', agent_track: list | None = None) -> None:
    """收尾一次运行。

    `agent_track` 在这里定格。会话里的轮次是**活的** —— 运行结束之后 agent 继续工作，
    同一批行还会被接着改写，`updated_at` 一路往后走。等到有人回头看这次运行，读到的
    就不是它当时的样子了：真机上先是排出了正确的 +32.8s，几分钟后同一条记录变成了
    「时间落在本轮之外」。驱动那侧的事实早就是这么存的，agent 这侧同理。
    """
    conn = _get_conn()
    conn.execute(
        'UPDATE benchmark_run SET status=?, ended_at=?, score_total=?, score_stdev=?, '
        'scores_by_dim=?, detail=?, agent_track=? WHERE id=?',
        (status, time.time(), score_total, score_stdev,
         _dumps(scores_by_dim or {}), detail, _dumps(agent_track or []), run_id))
    conn.commit()


def mark_stale_runs() -> int:
    """启动时把还标着 `running` 的运行记为中断。

    agent-core 重启会带走那个在跑的 asyncio task，而记录留在原地 —— 于是它永远停在
    「进行中」，面板每次打开都报一次并不存在的运行。
    """
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE benchmark_run SET status='interrupted', ended_at=?, "
        "detail='agent-core 重启，运行被中断' WHERE status='running'",
        (time.time(),))
    conn.commit()
    return cursor.rowcount


def _run_row(row) -> dict:
    return {
        'id': row[0], 'suite': row[1], 'tier': row[2], 'status': row[3],
        'started_at': row[4], 'ended_at': row[5],
        # `n` travels with every score. A number shown without its sample size
        # reads as a conclusion when it is one sample of a distribution.
        'n_repeats': row[6],
        'llm_model': row[7], 'llm_provider': row[8], 'host': row[9],
        'image_tags': _loads(row[10], {}), 'git_shas': _loads(row[11], {}),
        'score_total': row[12], 'score_stdev': row[13],
        'scores_by_dim': _loads(row[14], {}), 'detail': row[15],
        'session_id': row[16] if len(row) > 16 else '',
        'agent_track': _loads(row[17], []) if len(row) > 17 else [],
    }


_RUN_COLUMNS = ('id, suite, tier, status, started_at, ended_at, n_repeats, llm_model, '
                'llm_provider, host, image_tags, git_shas, score_total, score_stdev, '
                'scores_by_dim, detail, session_id, agent_track')


def list_runs(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        f'SELECT {_RUN_COLUMNS} FROM benchmark_run ORDER BY started_at DESC LIMIT ?',
        (max(1, min(int(limit), 500)),)).fetchall()
    return [_run_row(row) for row in rows]


def get_run(run_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(f'SELECT {_RUN_COLUMNS} FROM benchmark_run WHERE id=?',
                       (run_id,)).fetchone()
    if row is None:
        return None
    run = _run_row(row)
    run['cases'] = [
        {'scenario': c[0], 'repeat_idx': c[1], 'seed': c[2], 'ok': bool(c[3]),
         'outcome': c[4], 'score': c[5], 'elapsed_ms': c[6],
         'assertions': _loads(c[7], []), 'artifacts_ref': c[8],
         'facts': _loads(c[9], {})}
        for c in conn.execute(
            'SELECT scenario, repeat_idx, seed, ok, outcome, score, elapsed_ms, '
            'assertions, artifacts_ref, facts FROM benchmark_case WHERE run_id=? ORDER BY id',
            (run_id,)).fetchall()
    ]
    return run


def delete_run(run_id: str) -> bool:
    conn = _get_conn()
    conn.execute('DELETE FROM benchmark_case WHERE run_id=?', (run_id,))
    cursor = conn.execute('DELETE FROM benchmark_run WHERE id=?', (run_id,))
    conn.commit()
    return cursor.rowcount > 0


def trend(suite: str = '', limit: int = 30) -> list[dict]:
    """按时间排列的分数，供对比「改了什么之后分数动没动」。

    `llm_model` 与镜像 tag 一并返回 —— 两次运行之间分数变了，第一个问题永远是
    「变的是代码还是模型」，没有这两个字段就答不了。
    """
    runs = [run for run in list_runs(limit=limit * 2)
            if run['status'] == 'done' and (not suite or run['suite'] == suite)]
    return [{
        'id': run['id'], 'suite': run['suite'], 'started_at': run['started_at'],
        'score_total': run['score_total'], 'score_stdev': run['score_stdev'],
        'n_repeats': run['n_repeats'], 'llm_model': run['llm_model'],
        'image_tags': run['image_tags'], 'tier': run['tier'],
    } for run in runs[:limit]][::-1]


# ── 本机用例库 ────────────────────────────────────────────────────────────────
#
# 存的是**整个解决方案包体**，不是只有 `test` 段。用例是解决方案 + 执行方案 +
# 评估方案；只存后两段，载入时就拿不出画布，「用例自带画布」这条也就不成立了。

def save_case(payload: dict, *, name: str = '', origin: str = '',
              case_id: str = '') -> str:
    """新建或覆盖一条用例。返回它的 id。"""
    case_id = case_id or f'case-{int(time.time())}-{uuid.uuid4().hex[:6]}'
    conn = _get_conn()
    conn.execute(
        'INSERT INTO case_library (id, name, origin, updated_at, payload) '
        'VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
        'name=excluded.name, origin=excluded.origin, updated_at=excluded.updated_at, '
        'payload=excluded.payload',
        (case_id, name or '未命名用例', origin, time.time(), _dumps(payload)))
    conn.commit()
    return case_id


def _case_row(row) -> dict:
    return {'id': row[0], 'name': row[1], 'origin': row[2],
            'updated_at': row[3], 'payload': _loads(row[4], {})}


def get_case(case_id: str) -> dict | None:
    row = _get_conn().execute(
        'SELECT id, name, origin, updated_at, payload FROM case_library WHERE id=?',
        (case_id,)).fetchone()
    return _case_row(row) if row else None


def list_cases(limit: int = 100) -> list[dict]:
    rows = _get_conn().execute(
        'SELECT id, name, origin, updated_at, payload FROM case_library '
        'ORDER BY updated_at DESC LIMIT ?', (limit,)).fetchall()
    return [_case_row(row) for row in rows]


def delete_case(case_id: str) -> bool:
    """删用例。**不动 benchmark_run** —— 跑过的分数是已经发生的事实，用例被删掉
    不会让它没发生过，而历史里那一行仍然带着当时的模型与镜像 tag。"""
    conn = _get_conn()
    cursor = conn.execute('DELETE FROM case_library WHERE id=?', (case_id,))
    conn.commit()
    return cursor.rowcount > 0


# ── 两次运行之间，分数真的动了吗 ──────────────────────────────────────────────
#
# 原先这里只有 mean ± stdev，而面板照着它说「涨了」是没有依据的：LLM 是随机的，
# 一次运行的分数是分布里的一个样本，两个样本均值不同不等于有差别。
#
# `tools/llm_bench` 已经把这件事做对过一次，它的 README 记着当初为什么必须这么做：
# 「显著性用统计检验，不靠重复测量……任一条不过，报告就判『测不出显著差异』，
# 不排名、不给推荐。」这里照搬，不重写。

def _stats():
    """`llm_bench.stats`，拿不到就返回 None。

    它在 `tools/` 下，不是 `src/` 的一部分。**必须走包命名空间** —— 扁平的
    `import config` 会被 `src/config.py` 顶掉（`tests/test_llm_bench.py` 开头记着
    这个坑）。拿不到就老实说算不了，而不是退回自己手搓一个检验。
    """
    import pathlib
    import sys
    tools = str(pathlib.Path(__file__).resolve().parents[1] / 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    try:
        from llm_bench import stats
        return stats
    except Exception:
        return None


def compare_runs(baseline_id: str, current_id: str) -> dict:
    """两次跑同一个用例，分数的差异站不站得住。

    **按重复序号配对**，因为第 i 次重复两边用的是同一个 seed（`seed + index`）——
    seed 存在的理由就是让两次运行之间有东西可以配对。样本不足（n<3）时返回
    `available: False` 并说明原因，而不是给一个看起来很确定的结论。
    """
    stats = _stats()
    if stats is None:
        return {'available': False, 'reason': '取不到 llm_bench.stats，算不了显著性'}

    def scores(run_id):
        run = get_run(run_id) or {}
        return {c.get('repeat_idx'): c.get('score') for c in (run.get('cases') or [])
                if c.get('score') is not None}

    before, after = scores(baseline_id), scores(current_id)
    shared = sorted(set(before) & set(after))
    deltas = [float(after[i]) - float(before[i]) for i in shared]
    result = stats.significance(deltas)
    result['paired'] = len(shared)
    if not result.get('available'):
        return result
    # 「测不出显著差异」是一个结论，不是缺省值 —— 面板据此**不**给涨跌箭头。
    result['verdict'] = ('更好' if result['significant'] and result['median_delta'] > 0
                         else '更差' if result['significant']
                         else '测不出显著差异')
    return result
