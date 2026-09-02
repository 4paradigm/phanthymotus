"""执行：预检 → 预热 → 计分轮，逐条落盘，可续跑。

这里的每个默认行为都对应一次真实的误判，注释里写清来由，否则后人会把它们当
多余步骤删掉：

- 预检：曾经把 `zai-org/glm-5.2` 配到一个根本没有 GLM 的网关上，50×2 次请求
  全部 404 才发现。
- 预热：冷缓存让一个入口的缓存命中中位只有 48.8%，被判「慢 2.5 倍」；喂热后
  实际只慢 1.8 倍。不预热就是在测「谁的缓存先被我喂热」。
- 轮换：先把 N 条全发给 A 再全发给 B，两组之间的差异里会混进时段差异。
"""
import concurrent.futures
import json
import pathlib
import threading
import time

from llm_bench.transport import Kind, Transport


class Phase:
    PREFLIGHT = 'preflight'
    WARMUP = 'warmup'
    MEASURE = 'measure'
    PROBE = 'probe'


TINY_MESSAGES = [{'role': 'user', 'content': 'hi'}]


def _key_of(rec: dict) -> tuple:
    return (rec['group'], rec['payload_idx'], rec['repeat'], rec['phase'])


class ResultSink:
    """逐条 append 到 jsonl。

    立即落盘 + flush 是续跑的前提：进程被 Ctrl-C、SSH 断线或容器重启打断时，
    已经花掉的请求不能白花。这类评测一轮就是几十分钟和真金白银。
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._fh = path.open('a', encoding='utf-8')
        # 并行模式下多个线程会同时写。一条记录必须是一整行，两个线程交错写入
        # 会产出既非上一条也非下一条的坏行，而坏行是静默跳过的 —— 那等于悄悄
        # 丢结果。
        self._lock = threading.Lock()

    def write(self, rec: dict):
        line = json.dumps(rec, ensure_ascii=False) + '\n'
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self):
        self._fh.close()


def load_results(path: pathlib.Path) -> list[dict]:
    """读回已有结果。坏行跳过——半条记录不该让整次续跑失败。"""
    if not path.is_file():
        return []
    out = []
    with path.open('rb') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw.decode('utf-8')))
            except (UnicodeDecodeError, ValueError):
                continue
    return out


def preflight(groups, transport: Transport, request_cfg: dict, log=print) -> dict:
    """开跑前确认每组真的可用，并把模型名对不上的组挑出来。

    /models 只能证明「名字在列表里」，不能证明「能用」——orin6 上 router 的
    glm-5.3 就长期在列表里但返回「无可用渠道」。所以列表检查之后一定要再发一次
    真实的最小请求。
    """
    report = {}
    for g in groups:
        entry = {'group': g.name, 'model': g.model, 'host': g.host}
        models = transport.list_models(g.url, g.key)
        if models is None:
            entry['model_listed'] = 'unknown'
        else:
            entry['model_listed'] = g.model in models
            if not entry['model_listed']:
                near = [m for m in models if g.model.split('/')[-1] in m][:5]
                entry['similar'] = near

        r = transport.post_json(f'{g.url}/chat/completions', g.key, {
            'model': g.model, 'messages': TINY_MESSAGES,
            'max_tokens': 8, 'stream': False,
        })
        entry['ok'] = r['ok']
        entry['kind'] = r['kind']
        entry['elapsed'] = round(r['elapsed'], 3)
        if not r['ok']:
            entry['detail'] = r['detail']
        report[g.name] = entry

        status = 'OK' if r['ok'] else f'FAIL {r["kind"]}'
        listed = entry['model_listed']
        listed_s = {True: 'listed', False: 'NOT LISTED', 'unknown': 'list?'}[listed]
        log(f'  [preflight] {g.name:<28} {status:<24} {listed_s}  {r["elapsed"]:.2f}s')
        if not r['ok']:
            log(f'              {r["detail"][:160]}')
        if listed is False and entry.get('similar'):
            log(f'              相近的模型名: {entry["similar"]}')
    return report


def probe(groups, transport: Transport, times: int, interval: float = 2.0,
          sink: ResultSink | None = None, log=print) -> None:
    """轻量探活：只发最小请求，测「现在能不能用」。

    比整轮回测早得多、便宜得多地给出决定性证据——router 的 glm-5.3 那次
    0/30 全 503，用这个模式一分钟就能看清，不必先烧掉 50 条完整 payload。
    """
    for g in groups:
        seq = []
        for i in range(times):
            r = transport.post_json(f'{g.url}/chat/completions', g.key, {
                'model': g.model, 'messages': TINY_MESSAGES,
                'max_tokens': 8, 'stream': False,
            })
            seq.append(r)
            if sink:
                sink.write({'group': g.name, 'payload_idx': i, 'repeat': 0,
                            'phase': Phase.PROBE, 'ts': time.time(),
                            **{k: r[k] for k in
                               ('ok', 'kind', 'status', 'elapsed', 'prompt',
                                'completion', 'cached', 'finish', 'err', 'detail')}})
            if i < times - 1:
                time.sleep(interval)
        ok = sum(1 for r in seq if r['ok'])
        marks = ''.join('.' if r['ok'] else 'X' for r in seq)
        kinds = sorted({r['kind'] for r in seq if not r['ok']})
        log(f'  [probe] {g.name:<28} ok={ok}/{times}  {marks}'
            + (f'  {kinds}' if kinds else ''))


def run(groups, payloads, transport: Transport, cfg: dict,
        sink: ResultSink, done: set | None = None, log=print) -> None:
    """跑一轮。done 里的四元组会被跳过（续跑）。"""
    done = done or set()
    req = cfg['request']
    run_cfg = cfg['run']
    rotate = run_cfg.get('order', 'rotate') == 'rotate'
    stop_after = int(run_cfg.get('stop_after_consecutive_failures', 0) or 0)
    parallel = bool(run_cfg.get('parallel', False))

    # 只跑一轮，每条 payload 每组各请求一次。
    #
    # 不做预热轮：预热会把每条 payload 变成「原样重放」，缓存命中必然接近 100%，
    # 那量的是「同一个请求重发」而不是生产的前缀增长，反而污染缓存测量。谁先跑
    # 谁吃冷缓存的不对称由组顺序轮换消除（见下面的 rotate）。
    #
    # 也不做重复轮：同理，重复重放不代表真实负载。显著性由配对差的置信区间 +
    # 符号检验给出（stats.significance），不需要靠重复测量估噪声。
    total = len(payloads) * len(groups)
    counter = {'n': 0}
    consecutive = {g.name: 0 for g in groups}
    skipped = {g.name: False for g in groups}
    phase, rep = Phase.MEASURE, 0

    def build_payload(g, rec):
        payload = {
            'model': g.model,
            'messages': rec['messages'],
            'max_tokens': req['max_tokens'],
            'stream': False,
            **g.extra_body,
        }
        if rec.get('tools'):
            payload['tools'] = rec['tools']
        return payload

    def fire(g, rec, idx):
        """发一次请求并落盘。返回结果，供调用方更新失败计数。"""
        r = transport.post_json(f'{g.url}/chat/completions', g.key,
                                build_payload(g, rec))
        sink.write({'group': g.name, 'payload_idx': idx, 'repeat': rep,
                    'phase': phase, 'ts': time.time(),
                    **{k: r[k] for k in
                       ('ok', 'kind', 'status', 'elapsed', 'prompt',
                        'completion', 'cached', 'finish', 'err', 'detail')}})
        return r

    def report(g, r):
        """打印一条结果并更新连续失败计数。只在主线程调用，无需加锁。"""
        counter['n'] += 1
        n = counter['n']
        if r['ok']:
            tps = r['completion'] / r['elapsed'] if r['elapsed'] else 0
            log(f'  [{n}/{total}] {g.name:<24} ok  {r["elapsed"]:6.2f}s '
                f'p={r["prompt"]:6d} c={r["completion"]:5d} '
                f'cache={r["cached"]:6d} {tps:5.1f} tok/s')
            consecutive[g.name] = 0
        else:
            log(f'  [{n}/{total}] {g.name:<24} FAIL {r["elapsed"]:6.2f}s '
                f'{r["kind"]} {(r["detail"] or "")[:90]}')
            consecutive[g.name] += 1
            if stop_after and consecutive[g.name] >= stop_after:
                # 只跳过这一组，其余组继续——一个入口挂了不该毁掉整轮评测
                skipped[g.name] = True
                log(f'  [!] {g.name} 连续失败 {stop_after} 次，跳过该组剩余请求')

    for idx, rec in enumerate(payloads):
        # 轮换组顺序，抵消时段漂移；两组时退化成 A/B 交替。
        # 并行模式下各组本来就同时发出，顺序只影响提交次序。
        order = groups[idx % len(groups):] + groups[:idx % len(groups)] \
            if rotate else list(groups)
        todo = [g for g in order
                if (g.name, idx, rep, phase) not in done and not skipped[g.name]]
        counter['n'] += len(order) - len(todo)   # 跳过的也占进度
        if not todo:
            continue

        if not parallel:
            for g in todo:
                report(g, fire(g, rec, idx))
            continue

        # 并行：**只在组之间并行，payload 之间仍然严格串行**。
        #
        # payload 的先后顺序承载着前缀增长的语义（第 i+1 条的 prompt 是第 i 条
        # 追加而来），跨 payload 并发会让缓存命中变成不可解释的竞态，把 trace
        # 抽样辛苦保住的真实性又毁掉。
        #
        # 副产品：各组拿到同一条 payload 的时刻完全相同，时段漂移被彻底消除。
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(todo)) as ex:
            futs = {ex.submit(fire, g, rec, idx): g for g in todo}
            results = {}
            for fut in concurrent.futures.as_completed(futs):
                g = futs[fut]
                try:
                    results[g.name] = fut.result()
                except Exception as e:  # 兜底：fire 本身不该抛，但绝不能让它中断整轮
                    results[g.name] = {
                        'ok': False, 'kind': 'unknown', 'status': None,
                        'elapsed': 0.0, 'prompt': 0, 'completion': 0, 'cached': 0,
                        'finish': None, 'err': type(e).__name__,
                        'detail': str(e)[:200]}
        # 按固定顺序打印，日志才可读、可比对
        for g in todo:
            report(g, results[g.name])


def run_dir_for(base: pathlib.Path, run_id: str) -> pathlib.Path:
    d = base / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
