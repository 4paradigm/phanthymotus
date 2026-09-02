"""报告：Markdown 输出。

结论段由数据生成，不是模板套话——未达统计显著时明说「测不出差异」，有组成功率
不满时把可用性而不是延迟摆在最前面，冷热两态结论不一致时点出来。
"""
import datetime
import json
import pathlib

from llm_bench import stats as st


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def build(run_meta: dict, summaries: dict, paired: dict,
          shapes: dict, verdict: dict) -> str:
    L = []
    A = L.append

    A(f'# LLM 入口评测报告')
    A('')
    A(f'- run_id: `{run_meta["run_id"]}`')
    A(f'- 测量位置: `{run_meta.get("hostname", "?")}`'
      f'（镜像 `{run_meta.get("image_tag", "?")}`）')
    A(f'- 时间: {run_meta.get("started_at", "?")} → {run_meta.get("finished_at", "?")}')
    A(f'- 语料: {run_meta["corpus"]["count"]} 条，指纹 `{run_meta["corpus"]["fingerprint"]}`')
    A(f'- 抽样 `{run_meta["corpus"].get("sampling")}`，'
      f'组顺序 {run_meta["run"].get("order", "rotate")}，单轮，'
      f'{"组间并行" if run_meta["run"].get("parallel") else "串行"}')
    A(f'- HTTP 后端: {run_meta.get("transport", "?")}')
    if run_meta.get('incomplete'):
        A('')
        A(f'> **本次结果不完整**：缺失 {run_meta["incomplete"]}，'
          f'下面的数字只覆盖已完成部分。')
    A('')

    # ── 结论 ──────────────────────────────────────────────────────────────
    A('## 结论')
    A('')
    if verdict.get('primary') == 'availability':
        A('**首要问题是可用性，不是延迟。**')
        A('')
    if verdict.get('indistinguishable'):
        A('**各组之间测不出显著差异——不要据此排名。**')
        A('')
    for r in verdict.get('reasons', []):
        A(f'- {r}')
    if verdict.get('recommend'):
        A('')
        A(f'**推荐：`{verdict["recommend"]}`**')
    elif not verdict.get('reasons'):
        A('- 数据不足，无结论')
    A('')

    # ── 总表 ──────────────────────────────────────────────────────────────
    A('## 总表')
    A('')
    rows = []
    for name in sorted(summaries):
        s = summaries[name]
        ok_note = f'{s["ok"]}/{s["total"]}'
        if not s.get('complete'):
            ok_note = f'**{ok_note}**'
        if not s.get('ok'):
            rows.append([name, ok_note] + ['—'] * 8)
            continue
        # 成功率不满时，延迟数字后面强制带上样本量——避免幸存者偏差被读成事实
        suffix = f' ({s["ok"]}/{s["total"]})' if not s.get('complete') else ''
        rows.append([
            name, ok_note,
            f'{s["p50"]:.2f}s{suffix}', f'{s["p90"]:.2f}s', f'{s["p95"]:.2f}s',
            f'{s["max"]:.2f}s', f'{s["cv"]:.2f}',
            _fmt_pct(s['cache_overall']), _fmt_pct(s['cache_median']),
            f'{s["throughput"]:.1f}',
        ])
    A(_table(rows, ['组', '成功', 'p50', 'p90', 'p95', 'max', 'cv',
                    '缓存(总体)', '缓存(中位)', 'tok/s']))
    A('')
    if any(not s.get('complete') for s in summaries.values()):
        A('> 加粗的成功率表示该组有失败样本。**延迟分位只统计成功请求**，'
          '存在幸存者偏差：秒拒的失败（如 0.06s 的 503）不计入，'
          '否则会让失败更多的组显得更快。')
        A('')

    # ── 配对比较 ──────────────────────────────────────────────────────────
    A('## 配对比较')
    A('')
    if paired.get('n_common', 0) < 1 or not paired.get('pairs'):
        A('配对样本不足（需要至少一条所有组都成功的 payload）。')
    else:
        A(f'同一条 payload 逐条比较，消除 payload 难度差异。'
          f'共 {paired["n_common"]} 条所有组都成功。')
        A('')
        rows = []
        for pair, v in sorted(paired['pairs'].items()):
            ci = v.get('ci95') or (None, None)
            ci_s = (f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'
                    if ci[0] is not None else '—')
            if v.get('significant'):
                verdict_s = '**显著**'
            elif not v.get('available'):
                verdict_s = '样本不足'
            elif v.get('sign_conflict'):
                verdict_s = '符号冲突'
            else:
                verdict_s = '不显著'
            rows.append([pair, f'{v.get("median_delta", 0):+.2f}s',
                         f'{v.get("mean_delta", 0):+.2f}s', ci_s,
                         f'{v.get("p_sign", 1):.3f}',
                         f'{v.get("a_faster", 0)} / {v.get("b_faster", 0)}',
                         verdict_s])
        A(_table(rows, ['组对 (A vs B)', 'A−B 中位', 'A−B 均值',
                        '中位差 95% CI', '符号检验 p',
                        'A 更快 / B 更快', '判定']))
        A('')
        A('判定要两条证据都过：**中位差的 95% bootstrap 置信区间不跨 0**'
          '（方向稳定）**且符号检验 p < 0.05**（赢的次数不像抛硬币）。'
          '只看「中位差 0.4s」这种点估计不够——手工评测时正是这种点估计'
          '让 0.39s 的差异被当成了结论。')
        A('')
        A('「符号冲突」指中位与均值符号相反：一方在多数请求上略快但吃到更重的'
          '长尾，不构成差异。')
    A('')

    # ── 失败分析 ──────────────────────────────────────────────────────────
    A('## 失败分析')
    A('')
    any_fail = False
    for name in sorted(summaries):
        s = summaries[name]
        if s.get('complete'):
            continue
        any_fail = True
        shape = shapes.get(name, {})
        A(f'### {name}')
        A('')
        A(f'- 失败 {s["failed"]}/{s["total"]} 次请求，'
          f'类型 {", ".join(f"{k}×{v}" for k, v in sorted(s["failure_kinds"].items()))}')
        A(f'- 受影响 payload {shape.get("failed_payloads", 0)} 条，'
          f'其中最长连续 {shape.get("longest_consecutive_failures", 0)} 条，'
          f'首次失败于 payload #{shape.get("first_failure_idx")}')
        if s.get('failure_elapsed_median') is not None:
            A(f'- 失败样本耗时中位 {s["failure_elapsed_median"]:.2f}s'
              + ('（秒拒，不是超时）' if s['failure_elapsed_median'] < 1 else ''))
        if shape.get('clustered'):
            A('- **失败高度聚集**：形态像渠道断供/服务下线，而不是限流或随机抖动。'
              '这两者的处置完全不同，聚合失败率区分不了。')
        else:
            A('- 失败分散分布：更像限流或偶发抖动。')
        A('')
        for f in s.get('failure_samples', []):
            A(f'  - `#{f["payload_idx"]}` {f["kind"]} (HTTP {f["status"]}): '
              f'{f["detail"][:160]}')
        A('')
    if not any_fail:
        A('所有组全部成功，无失败样本。')
        A('')

    # ── 缓存 ──────────────────────────────────────────────────────────────
    A('## 缓存')
    A('')
    rows = []
    for name in sorted(summaries):
        s_ = summaries[name]
        if not s_.get('ok'):
            continue
        rows.append([name, _fmt_pct(s_['cache_overall']),
                     _fmt_pct(s_['cache_median']),
                     f'{s_["cache_hit_count"]}/{s_["ok"]}',
                     f'{s_["cached_tokens"]}/{s_["prompt_tokens"]}'])
    A(_table(rows, ['组', '总体命中(按token加权)', '逐条命中中位',
                    '有命中的请求', 'cached/prompt tokens']))
    A('')

    sampling = run_meta['corpus'].get('sampling')
    if sampling == 'trace':
        A('抽样用的是 `trace`（按整条 trace 取、保留轮次顺序），且**只跑一轮**，'
          '所以每条请求的命中只能来自它和前一条 payload 的公共前缀——'
          '这正是生产里「每轮在上一轮 prompt 上追加」的形态，这些数字可以当'
          '生产缓存率读。')
    else:
        A(f'> ⚠ 抽样用的是 `{sampling}`，它打散了 trace 内的连续性。实测语料里'
          '同 trace 相邻请求的消息前缀重叠中位是 **93.3%**，打散后只剩 **40.0%**'
          '——**这种抽样下的缓存率不代表生产**。要量缓存请用 `sampling: trace`'
          '（默认就是它）。')
    A('')
    A('两个口径都要看：总体命中反映实际省下的钱和 prefill 时间；逐条中位不被'
      '个别超长 prompt 带偏。')
    A('')

    # ── 方法与局限 ────────────────────────────────────────────────────────
    A('## 方法与局限')
    A('')
    A(f'- payload 取自本机真实请求日志 `{run_meta["corpus"]["source"]}`，'
      f'`messages` + `tools` 原样重放，模型名由评测组决定。')
    A(f'- 抽样方式 `{run_meta["corpus"]["sampling"]}`，'
      f'从 {run_meta["corpus"]["available"]} 条可用记录中取 '
      f'{run_meta["corpus"]["count"]} 条'
      + (f'，跳过坏行 {run_meta["corpus"]["bad_lines"]} 条'
         if run_meta['corpus'].get('bad_lines') else '') + '。')
    A(f'- `max_tokens={run_meta["request"]["max_tokens"]}`。')
    if run_meta['run'].get('parallel'):
        A('- **组间并行**：同一条 payload 的各组同时发出，payload 之间仍严格串行'
          '（payload 顺序承载前缀增长的语义，跨 payload 并发会让缓存命中变成'
          '不可解释的竞态）。并行消除了时段漂移——各组拿到同一条 payload 的时刻'
          '完全相同；但如果各组共用同一条上行链路或同一个供应商的配额，'
          '它们会互相争资源，**绝对延迟会比串行时偏高**。组间相对比较仍然有效，'
          '绝对值不要拿去和串行的结果对比。')
    else:
        A('- **串行发送**：各组互不干扰，绝对延迟最接近单请求真实水平。'
          '代价是墙钟时间是并行的 N 倍（N = 组数）。')
    A('- **只跑一轮**，每条 payload 每组各请求一次。不做预热轮也不做重复轮：'
      '重放同一条 payload 量的是「同一个请求重发」，前缀缓存必然命中，'
      '既不代表真实负载，也会把缓存数字污染成接近 100%。'
      '谁先跑谁吃冷缓存的不对称由组顺序轮换消除。')
    A('- 显著性由配对差的 95% bootstrap 置信区间 + 符号检验给出，'
      '不靠重复测量估噪声。')
    if run_meta['run']['order'] == 'rotate':
        A('- 组顺序逐条轮换，抵消时段漂移。')
    else:
        A('- **未轮换组顺序**：组间差异里可能混入时段差异。')
    A(f'- 单次运行、单一测量位置。同一入口在办公网和在机器人上的表现可以完全不同；'
      '不同时段也会不同。跨时段复跑一轮再下定论更稳。')
    A('- 只测延迟/可用性/缓存，**不评估回答质量**。')
    A('')
    return '\n'.join(L)


def write_meta(run_dir: pathlib.Path, run_meta: dict) -> pathlib.Path:
    """在开跑之前就把 run.json 落盘（只有 meta，没有统计量）。

    --resume 和 --report-only 都靠 run.json 取 meta。如果只在跑完时才写，被
    kill -9 / 断电 / 容器重启打断的 run 就只剩 results.jsonl，既不能续跑也不能
    出报告 —— 恰好是续跑存在的唯一场景。所以先写一份，收尾时再用完整版覆盖。
    """
    path = run_dir / 'run.json'
    path.write_text(json.dumps({'meta': run_meta}, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    return path


def write(run_dir: pathlib.Path, run_meta: dict, records: list[dict]) -> pathlib.Path:
    """从原始结果算出全部统计并落盘报告。"""
    measured = [r for r in records if r.get('phase') == 'measure']
    if not measured:
        # 只跑了探活，或旧 run 目录里还有 warmup 阶段的记录
        measured = [r for r in records
                    if r.get('phase') in ('probe', 'warmup')]

    by_group: dict[str, list[dict]] = {}
    for r in measured:
        by_group.setdefault(r['group'], []).append(r)

    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    shapes = {n: st.failure_shape(rs) for n, rs in by_group.items()}
    paired = st.paired(by_group)
    verdict = st.verdict(summaries, paired)

    md = build(run_meta, summaries, paired, shapes, verdict)
    (run_dir / 'report.md').write_text(md, encoding='utf-8')

    (run_dir / 'run.json').write_text(json.dumps({
        'meta': run_meta, 'summaries': summaries, 'paired': paired,
        'failure_shapes': shapes, 'verdict': verdict,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return run_dir / 'report.md'


def now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()
