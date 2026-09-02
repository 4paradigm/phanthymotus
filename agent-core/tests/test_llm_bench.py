"""llm_bench 离线测试：不发任何真实请求。

重点不在「函数能跑」，而在**那些防止误判的约束真的生效**。这些约束每一条都对应
一次真实的错误结论（见 tools/llm_bench/README.md「为什么有这些多余步骤」），
所以它们必须有测试兜着，否则以后被人当冗余优化掉，工具就退化成了它要取代的那种
临时脚本。
"""
import json
import os
import pathlib
import sqlite3
import sys
import time

import pytest

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(_TOOLS))

# 必须走 llm_bench.* 包命名空间：agent-core 自己有 src/config.py，扁平的
# `import config` 在整套测试跑在一起时会被它顶掉（sys.modules 抢名字）。
from llm_bench import config as cfgmod  # noqa: E402
from llm_bench import corpus as corpusmod  # noqa: E402
from llm_bench import report as reportmod  # noqa: E402
from llm_bench import runner  # noqa: E402
from llm_bench import stats as st  # noqa: E402
from llm_bench.transport import Kind, Transport, classify  # noqa: E402


# ── 语料 ──────────────────────────────────────────────────────────────────────

def _rec(i, n_msgs=4, trace=None):
    return {'request_id': f'req-{i}', 'trace_id': trace or f'tr-{i}',
            'model': 'glm-5.2',
            'messages': [{'role': 'user', 'content': f'q{i}-{j}'} for j in range(n_msgs)],
            'tools': [{'type': 'function', 'function': {'name': 'finish'}}],
            'ts': 1788000000 + i}


def _fake_ok(elapsed=1.0):
    return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': elapsed,
            'prompt': 10, 'completion': 5, 'cached': 0, 'finish': 'stop',
            'err': None, 'detail': None}


def _write_jsonl(path, records, tail: bytes = b''):
    with path.open('wb') as f:
        for r in records:
            f.write((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8'))
        if tail:
            f.write(tail)


def test_truncated_tail_does_not_raise(tmp_path):
    """半条记录（甚至截在 UTF-8 中间）不能让整个文件不可用。

    真实故障：非正常关机让 llm_request_*.jsonl 停在 0xe5 上。text-mode 迭代会抛
    UnicodeDecodeError，于是最后一行的损坏变成整个语料不可读。
    """
    f = tmp_path / 'llm_request_1.jsonl'
    # 截断的多字节序列 + 一条语法不完整的 JSON
    _write_jsonl(f, [_rec(i) for i in range(5)], tail=b'{"messages": [{"role": "\xe5')

    records, meta = corpusmod.load_records(str(tmp_path))
    assert len(records) == 5
    assert meta['bad_lines'] == 1, '坏行必须被计数并上报，不能静默吞掉'


def test_bad_lines_at_eof_are_counted(tmp_path):
    """损坏最常出现在文件末尾——计数不能漏掉最后一批。"""
    f = tmp_path / 'llm_request_1.jsonl'
    _write_jsonl(f, [_rec(i) for i in range(3)], tail=b'garbage1\ngarbage2\ngarbage3\n')
    _, meta = corpusmod.load_records(str(tmp_path))
    assert meta['bad_lines'] == 3


def test_min_messages_filters_degenerate(tmp_path):
    f = tmp_path / 'llm_request_1.jsonl'
    _write_jsonl(f, [_rec(0, n_msgs=1), _rec(1, n_msgs=5), _rec(2, n_msgs=2)])
    records, _ = corpusmod.load_records(str(tmp_path), min_messages=2)
    assert [r['request_id'] for r in records] == ['req-1', 'req-2']


def test_even_sampling_spans_whole_file():
    """均匀抽样必须覆盖整个文件，不能退化成只取开头。

    朴素的 `records[::len//count]` 在 count 接近 len 时正是这个毛病。
    """
    records = [_rec(i) for i in range(97)]
    picked = corpusmod.sample(records, 50, 'even')
    assert len(picked) == 50
    ids = [int(r['request_id'].split('-')[1]) for r in picked]
    assert ids == sorted(ids)
    assert ids[0] == 0
    assert ids[-1] > 90, f'最后一个样本 idx={ids[-1]}，没覆盖到文件尾部'


def test_even_sampling_is_deterministic():
    records = [_rec(i) for i in range(97)]
    assert corpusmod.sample(records, 20, 'even') == corpusmod.sample(records, 20, 'even')


def test_fingerprint_detects_different_payload_set():
    a = corpusmod.fingerprint([_rec(i) for i in range(10)])
    assert a == corpusmod.fingerprint([_rec(i) for i in range(10)])
    assert a != corpusmod.fingerprint([_rec(i) for i in range(1, 11)])


# ── trace 抽样 ────────────────────────────────────────────────────────────────

def test_trace_sampling_is_the_default():
    """默认必须是 trace —— 它同时给出真实缓存率和连续的轮次形态。"""
    assert cfgmod.DEFAULTS['corpus']['sampling'] == 'trace'


def test_trace_sampling_keeps_turns_contiguous_and_ordered():
    """同一条 trace 的记录必须连续且保持原顺序。

    生产里 agent 一轮接一轮，每轮 prompt 在上一轮基础上追加。打散顺序就测不出
    真实缓存率：实测同 trace 相邻请求前缀重叠 93.3%，打散后只剩 40.0%。
    """
    records = []
    for t in range(6):
        for k in range(4):
            records.append(_rec(t * 10 + k, trace=f'T{t}'))

    picked = corpusmod.sample(records, 8, 'trace')
    assert len(picked) == 8
    tids = [r['trace_id'] for r in picked]
    # 连续成段
    assert tids == sorted(tids, key=lambda t: tids.index(t)), '同 trace 记录必须连续'
    # 每段内部保持原顺序
    for t in set(tids):
        seg = [r['request_id'] for r in picked if r['trace_id'] == t]
        assert seg == sorted(seg, key=lambda x: int(x.split('-')[1])), f'{t} 顺序被打乱'


def test_trace_sampling_prefers_multi_turn_but_falls_back_to_singles():
    """单条 trace 不产生前缀增长，优先多轮；但凑不够时仍要用上。"""
    records = [_rec(0, trace='SOLO1'), _rec(1, trace='SOLO2')]
    records += [_rec(10, trace='M'), _rec(11, trace='M'), _rec(12, trace='M')]

    # 只要 3 条 → 全部来自多轮 trace
    assert {r['trace_id'] for r in corpusmod.sample(records, 3, 'trace')} == {'M'}
    # 要 5 条 → 单条 trace 也得用上，不能凭空少给
    picked = corpusmod.sample(records, 5, 'trace')
    assert len(picked) == 5
    assert {'SOLO1', 'SOLO2'} <= {r['trace_id'] for r in picked}


def test_trace_sampling_is_not_biased_toward_long_traces():
    """按时间顺序取，不按大小降序 —— 后者会过度偏向长对话，拉偏 prompt 规模分布。"""
    records = []
    # 前面若干条短 trace（各 2 条），最后一条超长 trace（20 条）
    for t in range(5):
        records += [_rec(t * 10, trace=f'S{t}'), _rec(t * 10 + 1, trace=f'S{t}')]
    records += [_rec(900 + k, trace='LONG') for k in range(20)]

    picked = corpusmod.sample(records, 10, 'trace')
    tids = {r['trace_id'] for r in picked}
    assert 'LONG' not in tids, '不该因为 LONG 最长就优先取它'
    assert tids == {f'S{t}' for t in range(5)}


def test_trace_sampling_truncation_keeps_a_prefix():
    """最后一条 trace 被截断时保留前缀，剩下的仍是连续轮次。"""
    records = [_rec(k, trace='T') for k in range(10)]
    picked = corpusmod.sample(records, 4, 'trace')
    assert [r['request_id'] for r in picked] == ['req-0', 'req-1', 'req-2', 'req-3']


# ── 配置 ──────────────────────────────────────────────────────────────────────

def _yaml(tmp_path, body: str) -> str:
    p = tmp_path / 'bench.yaml'
    p.write_text(body, encoding='utf-8')
    return str(p)


def test_models_expand_into_groups(tmp_path):
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - name: router
    url: https://r.example.com/v1
    key: sk-secret-aaaa
    models: [glm-5.2, zai-org/glm-5.3]
"""))
    groups = cfgmod.build_groups(cfg)
    assert [g.name for g in groups] == ['router/glm-5.2', 'router/zai-org/glm-5.3']
    assert all(g.key == 'sk-secret-aaaa' for g in groups)


def test_single_model_keeps_plain_name(tmp_path):
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: solo, url: https://a/v1, key: sk-x, model: glm-5.3}
"""))
    assert [g.name for g in cfgmod.build_groups(cfg)] == ['solo']


def test_key_in_yaml_is_used(tmp_path):
    """唯一的密钥来源：YAML 里的 key。"""
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: g, url: https://u/v1, key: sk-inline-key, model: glm-5.3}
"""))
    assert cfgmod.build_groups(cfg, str(tmp_path / 'none.db'))[0].key == 'sk-inline-key'


def test_missing_key_points_at_the_template(tmp_path):
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: g, url: https://u/v1, model: m}
"""))
    with pytest.raises(cfgmod.ConfigError, match='bench.yaml.example'):
        cfgmod.build_groups(cfg, str(tmp_path / 'none.db'))


@pytest.mark.parametrize('field', ['key_env', 'key_file', 'key_from_current'])
def test_removed_key_fields_error_without_echoing_the_value(tmp_path, field):
    """旧写法要给明确指引，且**绝不回显值**。

    这些字段里往往填的就是密钥本身（照着 key_env 的位置直接贴 sk-... 是常见笔误），
    把值打进报错就等于把密钥写进日志 —— 一个笔误变成一次泄露。
    """
    secret = 'sk-RjAbwdQbzU8eURU2ikUJJaXfB0wn50jPD54ssxU8fx1ojpt5'
    cfg = cfgmod.load(_yaml(tmp_path, f"""
groups:
  - {{name: g, url: https://u/v1, {field}: {secret}, model: m}}
"""))
    with pytest.raises(cfgmod.ConfigError) as ei:
        cfgmod.build_groups(cfg, str(tmp_path / 'none.db'))
    msg = str(ei.value)
    assert secret not in msg, f'{field} 的报错泄露了密钥原文'
    assert field in msg and 'key:' in msg


def test_duplicate_group_names_are_disambiguated(tmp_path):
    """重名会让报告里两行无法区分，也会让 resume 的三元组键冲突。"""
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: same, url: https://a/v1, key: sk-x, model: m}
  - {name: same, url: https://b/v1, key: sk-x, model: m}
"""))
    assert [g.name for g in cfgmod.build_groups(cfg)] == ['same', 'same#2']


def test_include_current_reads_configdb(tmp_path):
    db = tmp_path / 'data.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO config VALUES ('client', ?)", (json.dumps(
        {'llm': [{'url': 'https://prod/v1', 'key': 'sk-prod-key',
                  'model': 'glm-5.3', 'think_mode': False}]}),))
    conn.commit()
    conn.close()

    cfg = cfgmod.load(None, {'include_current': True})
    cfg['include_current'] = True
    groups = cfgmod.build_groups(cfg, str(db))
    assert len(groups) == 1
    g = groups[0]
    assert g.name == 'current/glm-5.3'
    assert g.source == 'configdb'
    # think_mode=false 时生产会带这个 extra_body（src/client/llm.py:136-139）
    assert g.extra_body['chat_template_kwargs'] == {'enable_thinking': False}


def test_configdb_absent_is_not_fatal_when_groups_exist(tmp_path):
    cfg = cfgmod.load(_yaml(tmp_path, """
include_current: true
groups:
  - {name: g, url: https://u/v1, key: sk-x, model: m}
"""))
    groups = cfgmod.build_groups(cfg, str(tmp_path / 'missing.db'))
    assert [g.name for g in groups] == ['g']


def test_cli_overrides_only_apply_when_given(tmp_path):
    """命令行不给的项必须沿用 YAML，不能被 None 覆盖成空。

    这是「所有参数 YAML 里都有、命令行只是临时覆盖」的前提。
    """
    body = """
corpus: {count: 50, sampling: trace}
run: {order: rotate}
request: {max_tokens: 4096}
groups:
  - {name: g, url: https://u/v1, key: sk-x, model: m}
"""
    # 全部不指定
    cfg = cfgmod.load(_yaml(tmp_path, body), {
        'corpus': {'count': None, 'sampling': None},
        'run': {},
        'request': {'max_tokens': None},
        'include_current': None,
    })
    assert cfg['corpus']['count'] == 50
    assert cfg['corpus']['sampling'] == 'trace'
    assert cfg['request']['max_tokens'] == 4096

    # 指定的才覆盖
    cfg2 = cfgmod.load(_yaml(tmp_path, body), {
        'corpus': {'count': 3, 'sampling': None, 'min_messages': 0},
        'request': {'max_tokens': None},
    })
    assert cfg2['corpus']['count'] == 3
    assert cfg2['corpus']['sampling'] == 'trace'    # 没给，保持 YAML
    assert cfg2['request']['max_tokens'] == 4096    # 没给，保持 YAML
    assert cfg2['corpus']['min_messages'] == 0      # 给了 0，必须生效（假值陷阱）


# ── 配置键校验 ────────────────────────────────────────────────────────────────

def test_removed_key_is_reported(tmp_path):
    """已移除的键必须给提示。

    YAML 里多一个键默认静默忽略 —— 那意味着过期的键和写错的键名都没有任何反馈，
    你以为设了其实没生效。刚移除 repeats 之后这一点尤其要紧。
    """
    warnings = []
    cfgmod.load(_yaml(tmp_path, """
run: {warmup: 1, repeats: 3}
groups:
  - {name: g, url: https://u/v1, key: sk-x, model: m}
"""), warn=warnings.append)
    assert any('repeats' in w for w in warnings)
    assert any('置信区间' in w for w in warnings), '要说清为什么移除了'


def test_typo_in_key_is_reported(tmp_path):
    warnings = []
    cfgmod.load(_yaml(tmp_path, """
corpus: {conut: 50}
groups:
  - {name: g, url: https://u/v1, key: sk-x, model: m}
"""), warn=warnings.append)
    assert any('conut' in w for w in warnings)


def test_unknown_group_field_is_reported(tmp_path):
    warnings = []
    cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: g, url: https://u/v1, key: sk-x, model: m, modle: typo}
"""), warn=warnings.append)
    assert any('modle' in w for w in warnings)


def test_valid_config_is_silent(tmp_path):
    """正常配置不能刷警告，否则真警告会被淹没。"""
    warnings = []
    cfgmod.load(_yaml(tmp_path, """
corpus: {dir: x, count: 10, sampling: trace, seed: 1, min_messages: 2}
request: {max_tokens: 100, timeout_s: 10, extra_body: {}}
run: {order: rotate, stop_after_consecutive_failures: 0}
include_current: false
groups:
  - {name: g, url: https://u/v1, key: sk-x, models: [a, b], extra_body: {}}
"""), warn=warnings.append)
    assert warnings == [], warnings


# ── baseline 开关 ─────────────────────────────────────────────────────────────

def _db_with_client(tmp_path) -> str:
    db = tmp_path / 'data.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO config VALUES ('client', ?)", (json.dumps(
        {'llm': [{'url': 'https://prod/v1', 'key': 'sk-prod',
                  'model': 'glm-5.3'}]}),))
    conn.commit()
    conn.close()
    return str(db)


def _cfg_with_switch(tmp_path, yaml_value: str, override):
    """override 模拟命令行：None=没指定，True/False=显式覆盖。"""
    cfg = cfgmod.load(_yaml(tmp_path, f"""
include_current: {yaml_value}
groups:
  - {{name: g, url: https://u/v1, key: sk-x, model: m}}
"""), {'include_current': override})
    return cfg


def test_cli_switch_can_enable_baseline_over_yaml(tmp_path):
    cfg = _cfg_with_switch(tmp_path, 'false', True)
    names = [g.name for g in cfgmod.build_groups(cfg, _db_with_client(tmp_path))]
    assert 'current/glm-5.3' in names


def test_cli_switch_can_disable_baseline_over_yaml(tmp_path):
    """--no-include-current 必须能关掉 YAML 里的 true。

    False 是个假值，用 `if override:` 之类的真值判断会把它当成「没指定」，
    于是这个开关在关闭方向上完全失效 —— 而关闭恰恰是它更常用的方向。
    """
    cfg = _cfg_with_switch(tmp_path, 'true', False)
    assert cfg['include_current'] is False
    names = [g.name for g in cfgmod.build_groups(cfg, _db_with_client(tmp_path))]
    assert names == ['g']


def test_no_cli_switch_falls_back_to_yaml(tmp_path):
    cfg = _cfg_with_switch(tmp_path, 'true', None)
    names = [g.name for g in cfgmod.build_groups(cfg, _db_with_client(tmp_path))]
    assert 'current/glm-5.3' in names


def test_cli_switch_parses_both_directions():
    from llm_bench.__main__ import parse_args
    assert parse_args([]).include_current is None
    assert parse_args(['--include-current']).include_current is True
    assert parse_args(['--no-include-current']).include_current is False


# ── 密钥打码 ──────────────────────────────────────────────────────────────────

SECRET = 'sk-RjAbwdQbzU8eURU2ikUJJaXfB0wn50jPD54ssxU8fx1ojpt5'


def test_mask_keeps_ends_only():
    m = cfgmod.mask(SECRET)
    assert SECRET not in m
    assert m.startswith('sk-RjA') and m.endswith('jpt5')


def test_plaintext_key_never_reaches_serialized_output(tmp_path):
    """明文密钥不允许出现在 run.json / report.md 的任何位置。"""
    g = cfgmod.Group('g', 'https://u/v1', SECRET, 'glm-5.3')
    assert SECRET not in json.dumps(g.public(), ensure_ascii=False)

    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    meta = _meta(groups=[g.public()])
    reportmod.write(run_dir, meta, [_res('g', i, ok=True) for i in range(4)])

    for name in ('run.json', 'report.md'):
        text = (run_dir / name).read_text(encoding='utf-8')
        assert SECRET not in text, f'{name} 泄露了明文 key'


# ── 传输层错误分类 ────────────────────────────────────────────────────────────

def test_model_unavailable_is_not_lumped_into_server_error():
    """「无可用渠道」必须和普通 5xx 分开：前者重试无意义。

    router 对 glm-5.3 连续返回 503「无可用渠道」，归成 server_error 会被误读成过载。
    """
    body = '{"error":{"code":"model_not_found","message":"分组 default 下模型 glm-5.3 无可用渠道（distributor）"}}'
    assert classify(503, body) == Kind.MODEL_UNAVAILABLE
    assert classify(503, 'upstream overloaded') == Kind.SERVER_ERROR


@pytest.mark.parametrize('status,body,expect', [
    (429, '', Kind.RATE_LIMIT),
    (402, '', Kind.BILLING),
    (401, '', Kind.AUTH),
    (403, '', Kind.AUTH),
    (400, 'messages 参数非法', Kind.BAD_REQUEST),
    (400, 'maximum context length exceeded', Kind.CONTEXT_OVERFLOW),
    (502, '', Kind.SERVER_ERROR),
])
def test_classify_matches_production_vocabulary(status, body, expect):
    assert classify(status, body) == expect


def test_connection_and_timeout_from_exception():
    assert classify(None, '', TimeoutError('read timeout')) == Kind.TIMEOUT
    assert classify(None, '', OSError('connection refused')) == Kind.CONNECTION


def test_200_with_error_body_counts_as_failure():
    """网关把错误塞进 200 体里。当成功会凭空拉高成功率，且这些"成功"极快，
    还会把延迟中位数拉低。"""
    r = Transport._fail(200, '{"message": "quota exceeded"}', None, 0.05,
                        note='200 响应体内含 error')
    assert r['ok'] is False
    assert '200 响应体内含 error' in r['detail']


# ── 统计约束 ──────────────────────────────────────────────────────────────────

def _res(group, idx, ok=True, elapsed=5.0, repeat=0, phase='measure',
         prompt=1000, cached=900, completion=100, kind='ok', status=200):
    return {'group': group, 'payload_idx': idx, 'repeat': repeat, 'phase': phase,
            'ok': ok, 'kind': kind if ok else kind, 'status': status,
            'elapsed': elapsed, 'prompt': prompt if ok else 0,
            'completion': completion if ok else 0, 'cached': cached if ok else 0,
            'finish': 'stop' if ok else None,
            'err': None if ok else f'HTTP {status}',
            'detail': None if ok else 'boom', 'ts': 1788000000 + idx}


def _meta(groups=None, count=4):
    return {
        'run_id': 'test', 'mode': 'bench', 'config': {},
        'groups': groups or [], 'hostname': 'test-host', 'image_tag': 'test',
        'transport': 'urllib', 'started_at': 'now', 'finished_at': 'now',
        'corpus': {'source': 'x', 'count': count, 'available': 100,
                   'sampling': 'even', 'fingerprint': 'abc123', 'bad_lines': 0},
        'request': {'max_tokens': 10240},
        'run': {'order': 'rotate'},
    }


def test_significance_needs_ci_and_sign_test():
    """显著性要两条证据都过：CI 不跨 0 且符号检验 p<0.05。

    只看「中位差 0.4s」这种点估计不够 —— 手工评测时正是这种点估计让 0.39s 的
    差异被当成了结论，而它完全在波动范围内。
    """
    # A 稳定快 2s：方向一致、赢 12:0
    clear = [-2.0, -1.9, -2.1, -2.0, -1.8, -2.2, -2.0, -1.9, -2.1, -2.0, -2.0, -1.95]
    r = st.significance(clear)
    assert r['significant'] is True
    assert r['ci95'][1] < 0, 'CI 应完全在 0 以下'
    assert r['p_sign'] < 0.05

    # 有正有负、中位接近 0：不显著
    noisy = [-2.0, 2.1, -0.3, 0.4, -1.8, 1.9, 0.2, -0.4, 1.1, -1.2, 0.6, -0.5]
    r2 = st.significance(noisy)
    assert r2['significant'] is False


def test_significance_rejects_sign_conflict():
    """中位与均值符号相反：赢在多数、输在长尾，不构成差异。"""
    # 多数略负（A 快），但一条极端长尾把均值拉正
    deltas = [-0.3] * 11 + [40.0]
    r = st.significance(deltas)
    assert r['sign_conflict'] is True
    assert r['significant'] is False


def test_significance_needs_enough_samples():
    r = st.significance([-1.0, -1.0])
    assert r['available'] is False
    assert '2' in r['reason']


def test_verdict_refuses_to_rank_when_not_significant():
    """核心约束：未达显著就不排名、不给推荐。"""
    by_group = {
        'A': [_res('A', i, elapsed=e) for i, e in enumerate(
            [6.0, 5.0, 7.0, 4.0, 8.0, 6.5, 5.5, 7.5, 6.2, 5.8, 6.9, 4.9])],
        'B': [_res('B', i, elapsed=e) for i, e in enumerate(
            [5.9, 5.4, 6.7, 4.6, 7.6, 6.8, 5.2, 7.1, 6.6, 5.4, 7.2, 4.6])],
    }
    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    v = st.verdict(summaries, st.paired(by_group))
    assert v.get('indistinguishable') is True
    assert v['recommend'] is None


def test_verdict_ranks_when_significant():
    by_group = {
        'slow': [_res('slow', i, elapsed=20.0 + i * 0.1) for i in range(12)],
        'fast': [_res('fast', i, elapsed=2.0 + i * 0.1) for i in range(12)],
    }
    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    v = st.verdict(summaries, st.paired(by_group))
    assert v.get('indistinguishable') is not True
    assert v['recommend'] == 'fast'


def test_verdict_puts_availability_before_latency():
    """一个入口再快，成功率不满就不是候选。

    真实数据：router/glm-5.3 延迟和对手持平，但 35/50 成功。
    """
    by_group = {
        'flaky_fast': [_res('flaky_fast', i, ok=(i < 3), elapsed=1.0, status=503,
                            kind='model_unavailable') for i in range(10)],
        'solid_slow': [_res('solid_slow', i, elapsed=9.0) for i in range(10)],
    }
    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    v = st.verdict(summaries, st.paired(by_group))
    assert v['primary'] == 'availability'
    assert v['recommend'] == 'solid_slow', '不能推荐一个成功率不满的组'


def test_single_pass_is_the_default():
    """默认只跑一轮 —— warmup 已从配置里移除。"""
    assert 'warmup' not in cfgmod.DEFAULTS['run']
    assert 'repeats' not in cfgmod.DEFAULTS['run']


def test_warmup_key_is_reported_as_removed(tmp_path):
    warnings = []
    cfgmod.load(_yaml(tmp_path, """
run: {warmup: 1}
groups:
  - {name: g, url: https://u/v1, key: sk-x, model: m}
"""), warn=warnings.append)
    assert any('warmup' in w and '只跑一轮' in w for w in warnings)


def test_runner_makes_exactly_one_request_per_group_and_payload():
    """总请求数必须是 组数 × payload 数，一次不多。"""
    calls = []

    class T:
        def post_json(self, url, key, payload):
            calls.append((url, payload['messages'][0]['content']))
            return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': 1.0,
                    'prompt': 10, 'completion': 5, 'cached': 0,
                    'finish': 'stop', 'err': None, 'detail': None}

    class Sink:
        def write(self, rec): pass

    groups = [cfgmod.Group('A', 'https://a/v1', 'k', 'm'),
              cfgmod.Group('B', 'https://b/v1', 'k', 'm')]
    runner.run(groups, [_rec(i) for i in range(5)], T(),
               {'request': {'max_tokens': 8}, 'run': {}}, Sink(),
               log=lambda *a, **k: None)
    assert len(calls) == 10, f'应为 2×5=10 次，实际 {len(calls)}'


def test_group_order_rotates_across_payloads():
    """轮换要真的换：否则固定排后面的组总吃到前面那组刚热好的缓存。"""
    order = []

    class T:
        def post_json(self, url, key, payload):
            order.append(url)
            return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': 1.0,
                    'prompt': 10, 'completion': 5, 'cached': 0,
                    'finish': 'stop', 'err': None, 'detail': None}

    class Sink:
        def write(self, rec): pass

    groups = [cfgmod.Group('A', 'https://a/v1', 'k', 'm'),
              cfgmod.Group('B', 'https://b/v1', 'k', 'm'),
              cfgmod.Group('C', 'https://c/v1', 'k', 'm')]
    runner.run(groups, [_rec(i) for i in range(3)], T(),
               {'request': {'max_tokens': 8}, 'run': {'order': 'rotate'}}, Sink(),
               log=lambda *a, **k: None)
    firsts = [order[0], order[3], order[6]]
    assert len(set(firsts)) == 3, f'每条 payload 的首发组应轮换，实际 {firsts}'


def test_failure_shape_detects_clustering():
    """连续 14 次 503 是渠道断供；散布的 14 次是限流。聚合失败率区分不了。"""
    clustered = [_res('A', i, ok=(i < 36)) for i in range(50)]
    shape = st.failure_shape(clustered)
    assert shape['longest_consecutive_failures'] == 14
    assert shape['first_failure_idx'] == 36
    assert shape['clustered'] is True

    scattered = [_res('A', i, ok=(i % 4 != 0)) for i in range(50)]
    assert st.failure_shape(scattered)['clustered'] is False


def test_summarize_separates_failure_latency():
    """秒拒的失败不能混进延迟分位，否则失败更多的组反而显得更快。"""
    recs = [_res('A', i, ok=True, elapsed=10.0) for i in range(5)] + \
           [_res('A', i + 5, ok=False, elapsed=0.06) for i in range(5)]
    s = st.summarize_group(recs)
    assert s['ok'] == 5 and s['failed'] == 5
    assert s['complete'] is False
    assert s['p50'] == pytest.approx(10.0), '失败样本不应拉低延迟分位'
    assert s['failure_elapsed_median'] == pytest.approx(0.06)


def test_paired_uses_only_payloads_all_groups_succeeded():
    by_group = {
        'A': [_res('A', i, elapsed=5.0) for i in range(5)],
        # B 在 payload 3、4 上失败 → 配对集只剩 0,1,2
        'B': [_res('B', i, ok=(i < 3), elapsed=6.0) for i in range(5)],
    }
    p = st.paired(by_group)
    assert p['n_common'] == 3
    assert p['pairs']['A vs B']['median_delta'] == pytest.approx(-1.0)


# ── 并行 ──────────────────────────────────────────────────────────────────────

def test_parallel_defaults_to_off():
    assert cfgmod.DEFAULTS['run']['parallel'] is False


def test_parallel_switch_parses_both_directions():
    from llm_bench.__main__ import parse_args
    assert parse_args([]).parallel is None
    assert parse_args(['--parallel']).parallel is True
    assert parse_args(['--no-parallel']).parallel is False


def test_parallel_runs_groups_concurrently():
    """同一条 payload 的各组必须真的同时在飞。"""
    import threading
    inflight = {'now': 0, 'peak': 0}
    lk = threading.Lock()

    class T:
        def post_json(self, url, key, payload):
            with lk:
                inflight['now'] += 1
                inflight['peak'] = max(inflight['peak'], inflight['now'])
            time.sleep(0.05)
            with lk:
                inflight['now'] -= 1
            return _fake_ok()

    class Sink:
        def write(self, rec): pass

    groups = [cfgmod.Group(n, f'https://{n}/v1', 'k', 'm') for n in ('A', 'B', 'C')]
    runner.run(groups, [_rec(i) for i in range(2)], T(),
               {'request': {'max_tokens': 8}, 'run': {'parallel': True}},
               Sink(), log=lambda *a, **k: None)
    assert inflight['peak'] == 3, f'应有 3 组同时在飞，实际峰值 {inflight["peak"]}'


def test_parallel_keeps_payloads_strictly_sequential():
    """payload 之间绝不能并发。

    payload 顺序承载前缀增长的语义（第 i+1 条的 prompt 由第 i 条追加而来），
    跨 payload 并发会让缓存命中变成不可解释的竞态，把 trace 抽样保住的真实性
    又毁掉。
    """
    import threading
    lk = threading.Lock()
    active_payloads = {'now': set(), 'peak': 0}

    class T:
        def post_json(self, url, key, payload):
            idx = payload['messages'][0]['content']
            with lk:
                active_payloads['now'].add(idx)
                active_payloads['peak'] = max(
                    active_payloads['peak'], len(active_payloads['now']))
            time.sleep(0.03)
            with lk:
                active_payloads['now'].discard(idx)
            return _fake_ok()

    class Sink:
        def write(self, rec): pass

    groups = [cfgmod.Group(n, f'https://{n}/v1', 'k', 'm') for n in ('A', 'B')]
    runner.run(groups, [_rec(i) for i in range(4)], T(),
               {'request': {'max_tokens': 8}, 'run': {'parallel': True}},
               Sink(), log=lambda *a, **k: None)
    assert active_payloads['peak'] == 1, \
        f'同时只能有一条 payload 在飞，实际 {active_payloads["peak"]}'


def test_parallel_makes_the_same_number_of_requests():
    calls = []

    class T:
        def post_json(self, url, key, payload):
            calls.append(url)
            return _fake_ok()

    class Sink:
        def write(self, rec): pass

    groups = [cfgmod.Group(n, f'https://{n}/v1', 'k', 'm') for n in ('A', 'B', 'C')]
    runner.run(groups, [_rec(i) for i in range(5)], T(),
               {'request': {'max_tokens': 8}, 'run': {'parallel': True}},
               Sink(), log=lambda *a, **k: None)
    assert len(calls) == 15


def test_result_sink_is_thread_safe(tmp_path):
    """并行下多线程同写：一条记录必须是一整行。

    交错写入会产出既非上一条也非下一条的坏行，而坏行是静默跳过的 —— 那等于
    悄悄丢结果。
    """
    import threading
    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    n_per_thread, n_threads = 40, 8

    def worker(t):
        for i in range(n_per_thread):
            sink.write(_res(f'g{t}', i, elapsed=1.0))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    sink.close()

    # 每行都必须是完整可解析的 JSON，且总数不能少
    raw = p.read_text(encoding='utf-8').splitlines()
    assert len(raw) == n_per_thread * n_threads
    for line in raw:
        json.loads(line)          # 抛异常就说明出现了交错写入
    assert len(runner.load_results(p)) == n_per_thread * n_threads


def test_parallel_survives_a_worker_exception():
    """某组抛异常不能中断整轮 —— 其余组必须照常完成。"""
    class T:
        def post_json(self, url, key, payload):
            if 'boom' in url:
                raise RuntimeError('transport exploded')
            return _fake_ok()

    written = []

    class Sink:
        def write(self, rec): written.append(rec)

    groups = [cfgmod.Group('boom', 'https://boom/v1', 'k', 'm'),
              cfgmod.Group('ok', 'https://ok/v1', 'k', 'm')]
    runner.run(groups, [_rec(i) for i in range(3)], T(),
               {'request': {'max_tokens': 8}, 'run': {'parallel': True}},
               Sink(), log=lambda *a, **k: None)
    ok_recs = [r for r in written if r['group'] == 'ok']
    assert len(ok_recs) == 3 and all(r['ok'] for r in ok_recs)


# ── 报告 ──────────────────────────────────────────────────────────────────────

def test_report_flags_survivor_bias(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res('A', i, ok=(i < 7)) for i in range(10)] + \
           [_res('B', i) for i in range(10)]
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '幸存者偏差' in md
    assert '(7/10)' in md, '成功率不满时延迟分位必须带样本量'


def test_report_warns_when_sampling_breaks_cache_realism(tmp_path):
    """even 抽样打散了 trace 连续性，缓存率不代表生产 —— 必须警告。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res(g, i) for g in ('A', 'B') for i in range(5)]
    meta = _meta()
    meta['corpus']['sampling'] = 'even'
    reportmod.write(run_dir, meta, recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '93.3%' in md and '40.0%' in md
    assert 'sampling: trace' in md


def test_report_trusts_cache_under_trace_sampling(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res(g, i) for g in ('A', 'B') for i in range(5)]
    meta = _meta()
    meta['corpus']['sampling'] = 'trace'
    reportmod.write(run_dir, meta, recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '可以当' in md and '生产缓存率' in md


def test_report_marks_sign_conflict_per_pair(tmp_path):
    """符号冲突是逐对判定，不能对所有组对一概而论。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = []
    for i in range(12):
        recs.append(_res('A', i, elapsed=5.0))
        # B 多数略快，但一条极端长尾把均值拉过去
        recs.append(_res('B', i, elapsed=4.7 if i < 11 else 45.0))
    reportmod.write(run_dir, _meta(), recs)
    data = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    pair = data['paired']['pairs']['A vs B']
    assert pair['sign_conflict'] is True
    assert pair['significant'] is False
    assert '符号冲突' in (run_dir / 'report.md').read_text(encoding='utf-8')


def test_report_shows_sign_flip_note_when_it_happens(tmp_path):
    """一方多数请求略快但吃到更重长尾时，必须提示。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    # A 在多数 payload 上略慢（中位 A-B > 0），但 B 有一个极端长尾拉高均值
    recs = []
    for i in range(6):
        recs.append(_res('A', i, elapsed=5.0))
        recs.append(_res('B', i, elapsed=4.0 if i < 5 else 40.0))
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '符号相反' in md


def test_report_separates_request_and_payload_units(tmp_path):
    """失败计数混用「请求」和「payload」两种单位会让人读错严重程度。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = []
    for i in range(10):
        recs.append(_res('A', i, ok=(i < 7)))
        recs.append(_res('B', i))
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '失败 3/10 次请求' in md
    assert '受影响 payload 3 条' in md


def test_report_survives_a_totally_dead_group(tmp_path):
    """某组全挂不能影响其他组出报告——否则一次评测的全部开销都白费。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res('dead', i, ok=False, kind='model_unavailable', status=503)
            for i in range(5)] + [_res('alive', i) for i in range(5)]
    path = reportmod.write(run_dir, _meta(), recs)
    md = path.read_text(encoding='utf-8')
    assert 'dead' in md and 'alive' in md
    data = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    assert data['summaries']['dead']['ok'] == 0
    assert data['verdict']['recommend'] == 'alive'


def test_report_only_recomputes_from_results(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    sink = runner.ResultSink(run_dir / 'results.jsonl')
    for i in range(4):
        sink.write(_res('A', i))
    sink.close()
    recs = runner.load_results(run_dir / 'results.jsonl')
    assert len(recs) == 4
    assert reportmod.write(run_dir, _meta(), recs).is_file()


# ── 续跑 ──────────────────────────────────────────────────────────────────────

def test_results_are_flushed_immediately(tmp_path):
    """不 flush 就没有续跑：被 Ctrl-C 时缓冲区里的结果连同花掉的钱一起丢。"""
    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    sink.write(_res('A', 0))
    assert len(runner.load_results(p)) == 1, '写入后未落盘'
    sink.close()


def test_resume_skips_completed_triples(tmp_path):
    """续跑只补缺失的 (组, payload, 轮次, 阶段)。"""
    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    for i in range(3):
        sink.write(_res('A', i, phase='measure', repeat=0))
    sink.close()

    done = {(r['group'], r['payload_idx'], r['repeat'], r['phase'])
            for r in runner.load_results(p)}

    calls = []

    class FakeTransport:
        def post_json(self, url, key, payload):
            calls.append(payload['messages'][0]['content'])
            return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': 1.0,
                    'prompt': 10, 'completion': 5, 'cached': 0,
                    'finish': 'stop', 'err': None, 'detail': None}

    g = cfgmod.Group('A', 'https://u/v1', 'sk-x', 'm')
    payloads = [_rec(i) for i in range(5)]
    sink = runner.ResultSink(p)
    runner.run([g], payloads, FakeTransport(),
               {'request': {'max_tokens': 8}, 'run': {'warmup': 0}},
               sink, done, log=lambda *a, **k: None)
    sink.close()

    assert len(calls) == 2, f'应只补 2 条，实际发了 {len(calls)} 条'
    assert len(runner.load_results(p)) == 5


def test_meta_is_written_before_any_request(tmp_path):
    """run.json 必须在开跑前就存在。

    真实故障：一次 450 请求的 run 被 kill -9 后，目录里只剩 results.jsonl。
    --resume 和 --report-only 都从 run.json 取 meta，于是这次 run 既不能续跑也
    不能出报告 —— 恰好是续跑存在的唯一场景。
    """
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    meta = _meta()
    reportmod.write_meta(run_dir, meta)

    assert (run_dir / 'run.json').is_file()
    loaded = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    assert loaded['meta']['corpus']['fingerprint'] == 'abc123'


def test_partial_run_can_still_produce_a_report(tmp_path):
    """被打断的 run 用 meta + 部分 results 就能出报告，并标注不完整。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    meta = _meta()
    meta['incomplete'] = '172/450 次请求'
    reportmod.write_meta(run_dir, meta)

    sink = runner.ResultSink(run_dir / 'results.jsonl')
    for g in ('A', 'B'):
        for i in range(4):
            sink.write(_res(g, i))
    sink.close()

    recs = runner.load_results(run_dir / 'results.jsonl')
    md = reportmod.write(run_dir, meta, recs).read_text(encoding='utf-8')
    assert '本次结果不完整' in md
    assert '172/450' in md


def test_resume_reselects_exact_payloads_from_grown_corpus():
    """语料是活的：agent-core 边跑边追加日志。

    续跑必须按 request_id 精确复原原样本集。如果重新抽样，一轮几十分钟的评测被
    中断后语料几乎必然已经变了，`even` 会选出另一批 payload —— 那时指纹校验会
    正确地拒绝，但「续跑」在它唯一有用的场景里就永远不可用了。
    """
    original = [_rec(i) for i in range(20)]
    picked = corpusmod.sample(original, 5, 'even')
    ids = corpusmod.request_ids(picked)
    fp = corpusmod.fingerprint(picked)

    # 评测跑到一半，agent-core 又写进来 30 条新记录
    grown = original + [_rec(i) for i in range(100, 130)]

    # 重新抽样会选出完全不同的一批 —— 这正是不能重新抽样的原因
    assert corpusmod.request_ids(corpusmod.sample(grown, 5, 'even')) != ids

    restored = corpusmod.reselect(grown, ids)
    assert corpusmod.request_ids(restored) == ids
    assert corpusmod.fingerprint(restored) == fp


def test_reselect_rejects_when_records_rotated_away():
    """日志被轮转/清理后原样本找不回来，必须明确报错而不是悄悄换一批。"""
    ids = corpusmod.request_ids([_rec(i) for i in range(5)])
    with pytest.raises(corpusmod.CorpusError, match='找不到'):
        corpusmod.reselect([_rec(i) for i in range(2)], ids)


def test_reselect_rejects_records_without_request_id():
    records = [{'messages': [{'role': 'user', 'content': 'x'}]}]
    with pytest.raises(corpusmod.CorpusError, match='request_id'):
        corpusmod.reselect(records, [None])


def test_resume_load_tolerates_corrupt_line(tmp_path):
    p = tmp_path / 'results.jsonl'
    with p.open('wb') as f:
        f.write((json.dumps(_res('A', 0)) + '\n').encode())
        f.write(b'{"group": "A", broken\n')
        f.write((json.dumps(_res('A', 1)) + '\n').encode())
    assert len(runner.load_results(p)) == 2


def test_run_continues_after_a_failure(tmp_path):
    """单次失败不能中断整批——否则既没结果也没失败样本可分析。"""
    class FlakyTransport:
        def __init__(self):
            self.n = 0

        def post_json(self, url, key, payload):
            self.n += 1
            if self.n == 2:
                return {'ok': False, 'kind': 'server_error', 'status': 503,
                        'elapsed': 0.06, 'prompt': 0, 'completion': 0, 'cached': 0,
                        'finish': None, 'err': 'HTTP 503', 'detail': 'boom'}
            return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': 1.0,
                    'prompt': 10, 'completion': 5, 'cached': 0,
                    'finish': 'stop', 'err': None, 'detail': None}

    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    runner.run([cfgmod.Group('A', 'https://u/v1', 'sk-x', 'm')],
               [_rec(i) for i in range(4)], FlakyTransport(),
               {'request': {'max_tokens': 8}, 'run': {'warmup': 0}},
               sink, log=lambda *a, **k: None)
    sink.close()
    recs = runner.load_results(p)
    assert len(recs) == 4
    assert sum(1 for r in recs if not r['ok']) == 1


def test_stop_after_consecutive_failures_only_skips_that_group():
    """一个入口挂了不该毁掉整轮评测。"""
    class DeadForA:
        def post_json(self, url, key, payload):
            dead = 'dead' in url
            return {'ok': not dead, 'kind': 'model_unavailable' if dead else 'ok',
                    'status': 503 if dead else 200, 'elapsed': 0.05,
                    'prompt': 0 if dead else 10, 'completion': 0 if dead else 5,
                    'cached': 0, 'finish': None if dead else 'stop',
                    'err': 'HTTP 503' if dead else None,
                    'detail': 'no channel' if dead else None}

    written = []

    class Sink:
        def write(self, rec):
            written.append(rec)

    groups = [cfgmod.Group('dead', 'https://dead/v1', 'k', 'm'),
              cfgmod.Group('ok', 'https://ok/v1', 'k', 'm')]
    runner.run(groups, [_rec(i) for i in range(10)], DeadForA(),
               {'request': {'max_tokens': 8},
                'run': {'warmup': 0,
                        'stop_after_consecutive_failures': 3}},
               Sink(), log=lambda *a, **k: None)

    dead = [r for r in written if r['group'] == 'dead']
    alive = [r for r in written if r['group'] == 'ok']
    assert len(dead) == 3, f'挂掉的组应在 3 次后停，实际 {len(dead)} 次'
    assert len(alive) == 10, '健康的组必须跑完'
