"""语料：从 agent-core 本机的真实请求日志里取回放样本。

日志由 `src/llm_logger.py` 写出，每行一条记录（`llm_logger.py:145-163`）：

    {request_id, trace_id, agent_type, model, messages, tools, ts}

回放时只用 `messages` + `tools`，模型名由评测组决定而不是沿用记录里的。

解析必须容错。`llm_logger` 的 `_scan_and_repair`（`llm_logger.py:60-114`）记录了
一个真实故障：非正常关机会让文件停在半条记录上，甚至停在 UTF-8 多字节序列中间。
那边的做法是截断修复，这里**只读不写**——评测工具没有资格改生产数据——所以坏行
跳过并计数，让调用方知道丢了多少。
"""
import hashlib
import json
import pathlib
import random


class CorpusError(Exception):
    pass


def _read_file(path: pathlib.Path, min_messages: int) -> tuple[list[dict], int]:
    """读一个 jsonl，返回 (合格记录, 坏行数)。

    按字节读再手工解码：一条被截断的多字节字符会让整个文件的 text-mode 迭代抛
    UnicodeDecodeError，那样最后一行的损坏就变成了整个文件不可用。

    坏行计数在函数返回值里而不是边产出边报——损坏最常出现在**文件末尾**（写到
    一半断电），生成器式的计数会恰好漏掉这批，正是最该被看见的那批。
    """
    kept: list[dict] = []
    bad = 0
    with path.open('rb') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError):
                bad += 1
                continue
            if not isinstance(rec, dict):
                bad += 1
                continue
            msgs = rec.get('messages')
            if isinstance(msgs, list) and len(msgs) >= min_messages:
                kept.append(rec)
    return kept, bad


def load_records(source: str, min_messages: int = 2) -> tuple[list[dict], dict]:
    """读入语料。source 可以是单个 jsonl 文件，也可以是包含它们的目录。

    返回 (records, meta)。meta 记录扫了哪些文件、跳过多少坏行——报告里要写清楚，
    否则「50 条样本」这个数字背后有多少损耗是不可见的。
    """
    p = pathlib.Path(source)
    if p.is_dir():
        files = sorted(p.glob('llm_request_*.jsonl'))
    elif p.is_file():
        files = [p]
    else:
        raise CorpusError(f'语料路径不存在: {source}')

    if not files:
        raise CorpusError(f'{source} 下没有 llm_request_*.jsonl')

    records: list[dict] = []
    bad_total = 0
    per_file = []
    for f in files:
        kept, bad = _read_file(f, min_messages)
        records.extend(kept)
        bad_total += bad
        per_file.append({'file': f.name, 'kept': len(kept), 'bad_lines': bad})

    if not records:
        raise CorpusError(
            f'{source} 里没有可用记录（min_messages={min_messages}，跳过坏行 {bad_total}）')

    return records, {'files': per_file, 'bad_lines': bad_total, 'total': len(records)}


def sample(records: list[dict], count: int, strategy: str = 'even',
           seed: int = 42) -> list[dict]:
    """抽样。

    even 是延迟对比的默认：跨越整个文件均匀取，覆盖不同 prompt 规模和会话阶段。
    只取开头会让样本集中在某一段会话上，那段的 prompt 长度和缓存形态都不具代表性。

    trace 是**缓存对比**该用的：按整条 trace 取，保留 trace 内的原始顺序。
    生产里 agent 是一轮接一轮，每轮 prompt 在上一轮基础上追加，同 trace 相邻请求
    的消息前缀重叠中位是 93.3%；而 even 抽样打散了这个连续性，抽完后相邻样本的
    重叠只有 40.0%（实测于 orin6 的 133 条记录 / 51 条 trace）。用 even 测出来的
    缓存率不代表生产。
    """
    if count >= len(records):
        return list(records)

    if strategy == 'recent':
        return records[-count:]
    if strategy == 'random':
        return random.Random(seed).sample(records, count)
    if strategy == 'trace':
        return _sample_traces(records, count)
    if strategy != 'even':
        raise CorpusError(f'未知抽样策略: {strategy}')

    # 均匀跨越：按位置取,而不是按固定 step 切片——len//count 的整数除法在
    # count 接近 len 时会退化成只取前面一段。
    n = len(records)
    return [records[i * n // count] for i in range(count)]


def _sample_traces(records: list[dict], count: int) -> list[dict]:
    """按整条 trace 取，保留 trace 内顺序，直到凑够 count 条。

    按 trace 在文件里首次出现的顺序取（即时间顺序），不按大小排序——按大小降序
    会过度偏向长对话，把 prompt 规模分布拉偏；时间顺序拿到的是一个跨时段的
    trace 样本，每条内部又是连续的，两个性质都要。

    多轮 trace 优先：只有一条记录的 trace 根本不产生前缀增长，对缓存测量没有
    贡献（实测语料里 51 条 trace 有 15 条是单条的）。凑不够时才用单条 trace 补，
    因为它们也是真实生产请求，不该完全排除。
    """
    order: dict = {}
    first_pos: dict = {}
    for i, r in enumerate(records):
        tid = r.get('trace_id') or f'_no_trace_{i}'
        if tid not in order:
            order[tid] = []
            first_pos[tid] = i
        order[tid].append(r)

    multi = [t for t in order if len(order[t]) > 1]
    single = [t for t in order if len(order[t]) == 1]
    ranked = sorted(multi, key=lambda t: first_pos[t]) + \
        sorted(single, key=lambda t: first_pos[t])

    picked: list[dict] = []
    for tid in ranked:
        if len(picked) >= count:
            break
        rs = order[tid]
        # 整条收进来；最后一条 trace 允许截断以精确凑够 count（截断保留前缀，
        # 所以剩下的部分仍然是一段连续的轮次）
        room = count - len(picked)
        picked.extend(rs[:room] if len(rs) > room else rs)
    return picked


def request_ids(records: list[dict]) -> list:
    """选中样本的 request_id 列表（缺失的位置为 None）。"""
    return [r.get('request_id') for r in records]


def reselect(records: list[dict], ids: list) -> list[dict]:
    """按 request_id 精确重建原样本集，供续跑使用。

    续跑**不能重新抽样**：agent-core 在跑的时候一直在往 llm_data 里追加记录，
    语料是活的。一轮 50×N 组的评测要几十分钟，中断后语料几乎必然已经变了，
    `even` 抽样会选出另一批 payload —— 那时指纹校验会正确地拒绝续跑，但结果是
    「续跑」在它唯一有用的场景里永远不可用。所以按 id 精确复原。
    """
    if not ids or any(i is None for i in ids):
        raise CorpusError(
            '原始 run 的样本缺少 request_id（可能来自旧版日志），无法精确复原，'
            '不能续跑。请开一次新的 run。')
    index = {r.get('request_id'): r for r in records if r.get('request_id')}
    missing = [i for i in ids if i not in index]
    if missing:
        raise CorpusError(
            f'原样本中有 {len(missing)}/{len(ids)} 条在当前语料里找不到'
            f'（首个: {missing[0]}）。日志可能已被轮转或清理，无法续跑。')
    return [index[i] for i in ids]


def fingerprint(records: list[dict]) -> str:
    """样本集指纹，用于 resume 时确认是同一批 payload。

    续跑最危险的失败模式不是崩溃，是**静默地把两批不同 payload 的结果混在一起**
    ——那样报告照常生成，数字却毫无意义。指纹让这种情况变成一个显式错误。
    """
    h = hashlib.sha256()
    for r in records:
        rid = r.get('request_id')
        # 老日志可能没有 request_id，退化为按内容摘要
        h.update((rid or json.dumps(r.get('messages'), sort_keys=True,
                                    ensure_ascii=False)[:4096]).encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()[:16]
