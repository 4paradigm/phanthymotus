"""统计：配对比较、显著性、幸存者偏差、失败形态、冷热两态。

这个模块的职责一半是算数字，一半是**阻止过度解读**。手工评测时我犯过的每个
解读错误都在这里有一条对应的约束。

显著性用配对差的 bootstrap 置信区间 + 符号检验，不靠重复测量估噪声——重复重放
同一条 payload 量的是「同一个请求重发」，服务端前缀缓存必然命中，用它去校准
真实负载的波动逻辑上站不住。
"""
import math
import random
import statistics


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def _safe_stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def summarize_group(records: list[dict]) -> dict:
    """单组汇总。失败样本单独统计，不混进延迟分位。

    为什么必须分开：很多失败是秒拒（0.06s 的 503），混进平均值会让**失败更多的
    组看起来更快**。手工评测时一个组 15/50 失败，若不分开统计，它的均值反而漂亮。
    """
    ok = [r for r in records if r.get('ok')]
    bad = [r for r in records if not r.get('ok')]
    lat = [r['elapsed'] for r in ok]

    prompt = sum(r.get('prompt', 0) for r in ok)
    cached = sum(r.get('cached', 0) for r in ok)
    completion = sum(r.get('completion', 0) for r in ok)
    per_cache = [r['cached'] / r['prompt'] for r in ok if r.get('prompt')]

    kinds: dict[str, int] = {}
    for r in bad:
        k = r.get('kind') or 'unknown'
        kinds[k] = kinds.get(k, 0) + 1

    out = {
        'total': len(records),
        'ok': len(ok),
        'failed': len(bad),
        'success_rate': len(ok) / len(records) if records else 0.0,
        'complete': len(bad) == 0,
        'failure_kinds': kinds,
        'failure_samples': [
            {'payload_idx': r.get('payload_idx'), 'kind': r.get('kind'),
             'status': r.get('status'), 'detail': (r.get('detail') or '')[:200]}
            for r in bad[:3]
        ],
        # 失败样本自己的耗时分布：秒拒和超时是两种完全不同的故障
        'failure_elapsed_median': statistics.median(
            [r['elapsed'] for r in bad]) if bad else None,
    }

    if lat:
        out.update({
            'p50': pct(lat, .5), 'p90': pct(lat, .9), 'p95': pct(lat, .95),
            'min': min(lat), 'max': max(lat),
            'mean': statistics.mean(lat), 'stdev': _safe_stdev(lat),
            'cv': _safe_stdev(lat) / statistics.mean(lat) if statistics.mean(lat) else 0,
            'prompt_tokens': prompt, 'completion_tokens': completion,
            'cached_tokens': cached,
            'cache_overall': cached / prompt if prompt else 0.0,
            'cache_median': statistics.median(per_cache) if per_cache else 0.0,
            'cache_hit_count': sum(1 for c in per_cache if c > 0),
            'throughput': completion / sum(lat) if sum(lat) else 0.0,
        })
    return out


def failure_shape(records: list[dict]) -> dict:
    """失败是连续的还是散布的。

    聚合失败率区分不了这两种情况，但处置完全不同：连续 14 次 503 是渠道断供，
    散布的 14 次是限流或抖动。手工评测时正是「全部集中在 idx36→49」这个形态
    才让我确认是断供而不是偶发。
    """
    by_idx = {}
    for r in records:
        by_idx.setdefault(r.get('payload_idx', 0), []).append(r)

    seq = [all(x.get('ok') for x in by_idx[i]) for i in sorted(by_idx)]
    longest = cur = 0
    first_fail = None
    for i, ok in enumerate(seq):
        if ok:
            cur = 0
        else:
            cur += 1
            longest = max(longest, cur)
            if first_fail is None:
                first_fail = sorted(by_idx)[i]
    n_fail = sum(1 for ok in seq if not ok)
    return {
        'longest_consecutive_failures': longest,
        'first_failure_idx': first_fail,
        'failed_payloads': n_fail,
        # 连续段占了绝大多数失败 → 断供形态而非随机抖动
        'clustered': bool(n_fail and longest >= max(3, n_fail * 0.6)),
    }


def _binom_two_sided_p(k: int, n: int) -> float:
    """符号检验的双侧 p 值：n 次配对里 A 赢 k 次，零假设 p=0.5。"""
    if n == 0:
        return 1.0
    total = 2.0 ** n
    def tail(m):
        return sum(math.comb(n, i) for i in range(0, m + 1)) / total
    lo = min(k, n - k)
    return min(1.0, 2.0 * tail(lo))


def _bootstrap_median_ci(deltas: list[float], seed: int = 42,
                         iters: int = 2000, alpha: float = 0.05) -> tuple:
    """配对差中位数的 bootstrap 置信区间。

    固定 seed，保证同一份结果反复算出同样的区间——报告要可复现，
    不能每次 --report-only 都给出不同的置信区间。
    """
    n = len(deltas)
    if n < 3:
        return (None, None)
    rnd = random.Random(seed)
    meds = []
    for _ in range(iters):
        s = [deltas[rnd.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(s))
    meds.sort()
    lo = meds[int(iters * alpha / 2)]
    hi = meds[min(iters - 1, int(iters * (1 - alpha / 2)))]
    return (lo, hi)


def significance(deltas: list[float], seed: int = 42) -> dict:
    """判断一组配对差是否构成真实差异。

    两条独立证据都要过：
      1. 中位数的 95% bootstrap 置信区间不跨 0 —— 差异的方向稳定
      2. 符号检验双侧 p < 0.05 —— 赢的次数不像抛硬币

    只看「中位差 0.4s」这种点估计是不够的：手工评测时正是这种点估计让我把
    0.39s 的差异当成了结论，而它其实完全在波动范围内。
    """
    n = len(deltas)
    out = {'n': n}
    if n == 0:
        return {**out, 'available': False, 'reason': '没有配对样本'}

    out['median_delta'] = statistics.median(deltas)
    out['mean_delta'] = statistics.mean(deltas)
    a_faster = sum(1 for d in deltas if d < 0)
    b_faster = sum(1 for d in deltas if d > 0)
    out['a_faster'], out['b_faster'] = a_faster, b_faster

    if n < 3:
        return {**out, 'available': False,
                'reason': f'配对样本只有 {n} 条，无法判断显著性'}

    lo, hi = _bootstrap_median_ci(deltas, seed)
    out['ci95'] = (lo, hi)
    out['p_sign'] = _binom_two_sided_p(min(a_faster, b_faster), a_faster + b_faster)
    ci_excludes_zero = lo is not None and (lo > 0 or hi < 0)
    out['available'] = True
    out['significant'] = bool(ci_excludes_zero and out['p_sign'] < 0.05)
    # 符号相反说明一方赢在多数、输在长尾，不该视为有差别
    out['sign_conflict'] = out['median_delta'] * out['mean_delta'] < 0
    if out['sign_conflict']:
        out['significant'] = False
    return out


def paired(by_group: dict[str, list[dict]]) -> dict:
    """配对比较：只用**所有组都成功**的 payload。

    消除 payload 难度差异——不同 payload 的 prompt 长度和生成长度差一个数量级，
    直接比两组的 p50 会被抽到的样本构成左右。
    """
    names = sorted(by_group)
    # payload_idx -> group -> 延迟
    per_idx: dict[int, dict[str, float]] = {}
    for name, recs in by_group.items():
        acc: dict[int, list[float]] = {}
        for r in recs:
            if r.get('ok'):
                acc.setdefault(r['payload_idx'], []).append(r['elapsed'])
        for idx, vals in acc.items():
            # 正常每个 (组, payload) 只有一条；万一有多条（续跑重复写入）取中位
            per_idx.setdefault(idx, {})[name] = statistics.median(vals)

    common = sorted(i for i, d in per_idx.items() if len(d) == len(names))
    out = {'n_common': len(common), 'groups': names, 'pairs': {}}
    if not common or len(names) < 2:
        return out

    out['per_group_median_on_common'] = {
        n: statistics.median([per_idx[i][n] for i in common]) for n in names
    }

    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            d = [per_idx[i][a] - per_idx[i][b] for i in common]
            out['pairs'][f'{a} vs {b}'] = significance(d)
    return out


def verdict(summaries: dict, paired_res: dict) -> dict:
    """生成结论。这段刻意保守——宁可说「测不出差别」也不编排名。

    优先级：可用性 > 延迟。一个入口再快，成功率不满就不是候选；这正是
    router/glm-5.3 那次的情况（35/50 成功，延迟却和对手持平）。
    """
    incomplete = {n: s for n, s in summaries.items() if not s.get('complete')}
    usable = {n: s for n, s in summaries.items() if s.get('complete') and s.get('ok')}

    res = {'incomplete': sorted(incomplete), 'reasons': []}

    if incomplete:
        res['primary'] = 'availability'
        for n, s in sorted(incomplete.items()):
            kinds = ', '.join(f'{k}×{v}' for k, v in sorted(s['failure_kinds'].items()))
            res['reasons'].append(
                f'{n} 成功率 {s["ok"]}/{s["total"]}（{kinds}）——可用性问题优先于延迟')
        if usable:
            res['recommend'] = sorted(
                usable, key=lambda n: usable[n].get('p50', float('inf')))[0]
            res['reasons'].append(
                f'建议 {res["recommend"]}：本轮全部成功')
        return res

    res['primary'] = 'latency'
    pairs = paired_res.get('pairs') or {}
    if not pairs:
        res['recommend'] = None
        res['reasons'].append('配对样本不足，无法比较')
        return res

    sig = {k: v for k, v in pairs.items() if v.get('significant')}
    if not sig:
        res['recommend'] = None
        res['indistinguishable'] = True
        unavailable = [k for k, v in pairs.items() if not v.get('available')]
        if unavailable and len(unavailable) == len(pairs):
            res['reasons'].append(
                f'配对样本太少（{pairs[unavailable[0]].get("n", 0)} 条），'
                '无法判断显著性——加大 --count 再看')
        else:
            worst = max(pairs.values(), key=lambda v: abs(v.get('median_delta', 0)))
            ci = worst.get('ci95') or (None, None)
            ci_s = (f'95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]s'
                    if ci[0] is not None else 'CI 不可用')
            res['reasons'].append(
                f'没有一对达到显著：最大中位差 {worst["median_delta"]:+.2f}s，'
                f'{ci_s} 跨 0（符号检验 p={worst.get("p_sign", 1):.2f}）')
            flipped = [k for k, v in pairs.items() if v.get('sign_conflict')]
            if flipped:
                res['reasons'].append(
                    '以下组对中位与均值符号相反（赢在多数、输在长尾），'
                    '不构成差异: ' + '; '.join(sorted(flipped)))
        return res

    ranked = sorted(usable, key=lambda n: usable[n].get('p50', float('inf')))
    res['recommend'] = ranked[0] if ranked else None
    res['ranking'] = ranked
    for k, v in sorted(sig.items()):
        ci = v.get('ci95') or (None, None)
        res['reasons'].append(
            f'{k}: 中位差 {v["median_delta"]:+.2f}s，'
            f'95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]s 不跨 0，'
            f'符号检验 p={v["p_sign"]:.3f}（{v["a_faster"]}:{v["b_faster"]}）')
    insig = [k for k in pairs if k not in sig]
    if insig:
        res['reasons'].append(
            '以下组对未达显著，不应视为有差别: ' + '; '.join(sorted(insig)))
    return res
