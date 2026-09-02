"""llm_bench —— LLM 入口回测。

用法见 README.md。典型：

    python3 /work/tools/llm_bench --config bench.yaml --probe 10
    python3 /work/tools/llm_bench --config bench.yaml --count 50
    python3 /work/tools/llm_bench --resume resource/llm_bench/20260902_143000
    python3 /work/tools/llm_bench --report-only resource/llm_bench/20260902_143000
"""
import argparse
import datetime
import json
import os
import pathlib
import platform
import sys

# 把 tools/ 加进 sys.path，然后一律用 `llm_bench.*` 绝对导入。
#
# 这里必须是包命名空间而不是扁平的 `import config`：agent-core 自己就有
# src/config.py，扁平名字在同一个进程里会互相抢 sys.modules —— 整个测试套件
# 跑在一起时，llm_bench 的 config/report/runner/stats 会被 src/ 的同名模块顶掉。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm_bench import config as cfgmod  # noqa: E402
from llm_bench import corpus as corpusmod  # noqa: E402
from llm_bench import report as reportmod  # noqa: E402
from llm_bench import runner  # noqa: E402
from llm_bench.transport import Transport  # noqa: E402


def _image_tag() -> str:
    p = pathlib.Path('/work/VERSION')
    try:
        return p.read_text(encoding='utf-8').strip()
    except OSError:
        return 'n/a'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog='llm_bench', description='LLM 入口回测：延迟 / 可用性 / 缓存')
    ap.add_argument('--config', help='YAML 配置文件')
    ap.add_argument('--corpus', help='语料目录或单个 jsonl（覆盖配置）')
    ap.add_argument('--count', type=int, help='样本数（覆盖配置）')
    ap.add_argument('--sampling', choices=['even', 'trace', 'recent', 'random'],
                    help='抽样方式（覆盖配置）')
    ap.add_argument('--max-tokens', type=int, help='max_tokens（覆盖配置）')
    ap.add_argument('--timeout', type=float, help='单请求超时秒数（覆盖配置）')
    ap.add_argument('--out', default='resource/llm_bench', help='输出根目录')
    ap.add_argument('--db', help='ConfigDB 路径（include_current 用），默认取 $DB_PATH')
    # 当前生产配置这一组测不测的开关。命令行优先于 YAML 的 include_current：
    # 「跟现在比一比」和「只比几个候选」是两种不同的问题，切换它不该改配置文件。
    cur = ap.add_mutually_exclusive_group()
    cur.add_argument('--include-current', dest='include_current',
                     action='store_true', default=None,
                     help='把 ConfigDB 当前生产配置加为一组 baseline（覆盖配置文件）')
    cur.add_argument('--no-include-current', dest='include_current',
                     action='store_false',
                     help='不测当前生产配置这一组（覆盖配置文件）')
    # 组间并行开关。默认串行（保守），命令行优先于 YAML。
    par = ap.add_mutually_exclusive_group()
    par.add_argument('--parallel', dest='parallel', action='store_true', default=None,
                     help='同一条 payload 的各组并行请求（payload 之间仍串行）')
    par.add_argument('--no-parallel', dest='parallel', action='store_false',
                     help='各组串行请求（覆盖配置文件）')
    ap.add_argument('--probe', type=int, metavar='N',
                    help='只做探活：每组发 N 次最小请求，不跑语料')
    ap.add_argument('--probe-interval', type=float, default=2.0,
                    help='探活间隔秒数（默认 2）')
    ap.add_argument('--resume', metavar='RUN_DIR', help='续跑一个已有 run 目录')
    ap.add_argument('--report-only', metavar='RUN_DIR',
                    help='只对已有结果重新生成报告，不发请求')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='跳过预检（不建议：预检能挡住模型名写错导致的整轮浪费）')
    ap.add_argument('--yes', '-y', action='store_true',
                    help='预检失败时不询问，直接继续')
    return ap.parse_args(argv)


def _overrides(args) -> dict:
    return {
        'corpus': {'dir': args.corpus, 'count': args.count,
                   'sampling': args.sampling},
        'request': {'max_tokens': args.max_tokens, 'timeout_s': args.timeout},
        'run': {'parallel': args.parallel},
        # None 表示命令行没指定 → 沿用配置文件；True/False 都是显式覆盖，
        # 所以这里不能用 `if args.include_current` 之类的真值判断。
        'include_current': args.include_current,
    }


def _load_meta(run_dir: pathlib.Path) -> dict:
    f = run_dir / 'run.json'
    if not f.is_file():
        raise SystemExit(f'{run_dir} 里没有 run.json，无法续跑/复算')
    return json.loads(f.read_text(encoding='utf-8'))['meta']


def main(argv=None) -> int:
    args = parse_args(argv)

    # ── 只复算报告 ────────────────────────────────────────────────────────
    if args.report_only:
        run_dir = pathlib.Path(args.report_only)
        meta = _load_meta(run_dir)
        records = runner.load_results(run_dir / 'results.jsonl')
        if not records:
            raise SystemExit(f'{run_dir}/results.jsonl 为空')
        path = reportmod.write(run_dir, meta, records)
        print(f'报告已更新: {path}')
        return 0

    # ── 配置与组 ──────────────────────────────────────────────────────────
    resume_dir = pathlib.Path(args.resume) if args.resume else None
    if resume_dir:
        meta = _load_meta(resume_dir)
        cfg = meta['config']
    else:
        cfg = cfgmod.load(args.config, _overrides(args))
        meta = None

    try:
        groups = cfgmod.build_groups(cfg, args.db)
    except cfgmod.ConfigError as e:
        raise SystemExit(f'配置错误: {e}')

    print(f'评测组 {len(groups)} 个:')
    for g in groups:
        print(f'  - {g.name:<28} {g.host:<28} {g.model}  key={cfgmod.mask(g.key)}')
    # 显式要了 baseline 却一组都没读到，必须说出来。静默跳过会让人以为
    # 「跟现在比过了」，而报告里其实根本没有那一行。
    if cfg.get('include_current') and not any(g.source == 'configdb' for g in groups):
        print('  [!] 已要求测当前生产配置(baseline)，但 ConfigDB 里没读到 LLM 配置'
              f'（{args.db or os.environ.get("DB_PATH", "resource/data.db")}）'
              '—— 本轮不含 baseline 组')
    print()

    transport = Transport(timeout_s=cfg['request']['timeout_s'])
    try:
        return _run(args, cfg, groups, transport, resume_dir, meta)
    finally:
        transport.close()


def _run(args, cfg, groups, transport, resume_dir, meta) -> int:
    out_root = pathlib.Path(args.out)

    # ── 探活 ──────────────────────────────────────────────────────────────
    if args.probe:
        print(f'探活：每组 {args.probe} 次最小请求，间隔 {args.probe_interval}s')
        run_dir = runner.run_dir_for(
            out_root, datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_probe')
        sink = runner.ResultSink(run_dir / 'results.jsonl')
        run_meta = {
            'run_id': run_dir.name, 'mode': 'probe', 'config': cfg,
            'groups': [g.public() for g in groups],
            'hostname': platform.node(), 'image_tag': _image_tag(),
            'transport': transport.backend,
            'started_at': reportmod.now_iso(),
            'corpus': {'source': '(probe)', 'count': args.probe, 'available': 0,
                       'sampling': 'n/a', 'fingerprint': 'n/a', 'bad_lines': 0},
            'request': cfg['request'], 'run': cfg['run'],
        }
        reportmod.write_meta(run_dir, run_meta)
        try:
            runner.probe(groups, transport, args.probe, args.probe_interval, sink)
        except KeyboardInterrupt:
            print('\n[中断] 已完成的探活结果已落盘。')
        finally:
            sink.close()
        run_meta['finished_at'] = reportmod.now_iso()
        records = runner.load_results(run_dir / 'results.jsonl')
        path = reportmod.write(run_dir, run_meta, records)
        print(f'\n报告: {path}')
        return 0

    # ── 语料 ──────────────────────────────────────────────────────────────
    ccfg = cfg['corpus']
    try:
        records, cmeta = corpusmod.load_records(ccfg['dir'], ccfg['min_messages'])
    except corpusmod.CorpusError as e:
        raise SystemExit(f'语料错误: {e}')
    payloads = corpusmod.sample(records, ccfg['count'], ccfg['sampling'], ccfg['seed'])
    fp = corpusmod.fingerprint(payloads)
    print(f'语料: {cmeta["total"]} 条可用 → 抽样 {len(payloads)} 条'
          f'（{ccfg["sampling"]}），指纹 {fp}'
          + (f'，跳过坏行 {cmeta["bad_lines"]}' if cmeta['bad_lines'] else ''))

    # ── 续跑校验 ──────────────────────────────────────────────────────────
    done = set()
    if resume_dir:
        # 按 request_id 精确复原原样本集，而不是重新抽样：语料是活的（agent-core
        # 一直在追加日志），重新抽样几乎必然选出另一批 payload。
        old = meta.get('corpus', {})
        try:
            payloads = corpusmod.reselect(records, old.get('request_ids') or [])
        except corpusmod.CorpusError as e:
            raise SystemExit(f'无法续跑: {e}')
        fp = corpusmod.fingerprint(payloads)
        if old.get('fingerprint') != fp:
            raise SystemExit(
                f'语料指纹不匹配（原 {old.get("fingerprint")}，现 {fp}），拒绝续跑。\n'
                '两批不同的 payload 混在一起会让报告数字失去意义。\n'
                '若确实要用新语料，请开一次新的 run。')
        print(f'已按 request_id 复原原样本 {len(payloads)} 条，指纹校验通过')
        prev = runner.load_results(resume_dir / 'results.jsonl')
        done = {(r['group'], r['payload_idx'], r['repeat'], r['phase'])
                for r in prev}
        run_dir = resume_dir
        print(f'续跑 {run_dir}：已有 {len(done)} 条结果，只补缺失部分')
        run_meta = meta
    else:
        run_dir = runner.run_dir_for(
            out_root, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        run_meta = {
            'run_id': run_dir.name, 'mode': 'bench', 'config': cfg,
            'groups': [g.public() for g in groups],
            'hostname': platform.node(), 'image_tag': _image_tag(),
            'transport': transport.backend,
            'started_at': reportmod.now_iso(),
            'corpus': {'source': ccfg['dir'], 'count': len(payloads),
                       'available': cmeta['total'], 'sampling': ccfg['sampling'],
                       'fingerprint': fp, 'bad_lines': cmeta['bad_lines'],
                       'files': cmeta['files'],
                       # 续跑要靠这个精确复原样本集，不能重新抽样
                       'request_ids': corpusmod.request_ids(payloads)},
            'request': cfg['request'], 'run': cfg['run'],
        }

    # ── 预检 ──────────────────────────────────────────────────────────────
    if not args.skip_preflight:
        print('\n预检：')
        pf = runner.preflight(groups, transport, cfg['request'])
        run_meta['preflight'] = pf
        broken = [n for n, v in pf.items() if not v['ok']]
        if broken:
            print(f'\n以下组预检失败: {", ".join(broken)}')
            print('继续跑会在这些组上浪费全部请求。')
            if not args.yes:
                # 后台跑（docker exec -d / nohup）时 stdin 是关的，问不出来。
                # 默认取「不继续」——白跑一轮几十分钟和真金白银，比多打一个
                # 参数贵得多。但必须说清为什么停了，否则看起来像是启动就没动。
                if not sys.stdin.isatty():
                    print('当前是非交互环境（后台运行），无法询问 —— 已中止。')
                    print('确认要带着失败的组继续，请加 --yes。')
                    return 2
                try:
                    ans = input('仍要继续吗？[y/N] ').strip().lower()
                except EOFError:
                    ans = 'n'
                if ans != 'y':
                    print('已中止。修正配置后重试。')
                    return 2

    # ── 正式跑 ────────────────────────────────────────────────────────────
    total = len(payloads) * len(groups)
    par = cfg['run'].get('parallel')
    print(f'\n开始：{len(groups)} 组 × {len(payloads)} 条 = {total} 次请求'
          f'（{"组间并行" if par else "串行"}）')
    if par:
        # 同一个 endpoint+model 出现在两个组里时，并行会让它们真的互相争资源，
        # 而且共享同一份前缀缓存 —— 那时的对比结果无法解释。
        seen: dict = {}
        for g in groups:
            seen.setdefault((g.host, g.model), []).append(g.name)
        dup = {k: v for k, v in seen.items() if len(v) > 1}
        for (host, model), names in dup.items():
            print(f'  [!] {", ".join(names)} 指向同一个 {host} 的 {model}，'
                  '并行下它们会互相争资源并共享同一份前缀缓存，'
                  '结果不可解释 —— 建议对这种组合用 --no-parallel')
    print(f'输出目录 {run_dir}\n')

    # 先把 meta 落盘：被 kill 打断时 run.json 必须已经存在，否则这次 run 既不能
    # --resume 也不能 --report-only，只剩一堆无从解释的 results.jsonl。
    reportmod.write_meta(run_dir, run_meta)

    sink = runner.ResultSink(run_dir / 'results.jsonl')
    interrupted = False
    try:
        runner.run(groups, payloads, transport, cfg, sink, done)
    except KeyboardInterrupt:
        interrupted = True
        print('\n[中断] 已完成的结果已落盘，可用 --resume 续跑。')
    finally:
        sink.close()

    run_meta['finished_at'] = reportmod.now_iso()
    all_records = runner.load_results(run_dir / 'results.jsonl')
    expected = len(payloads) * len(groups)
    got = len([r for r in all_records if r.get('phase') in ('warmup', 'measure')])
    if got < expected:
        run_meta['incomplete'] = f'{got}/{expected} 次请求'

    path = reportmod.write(run_dir, run_meta, all_records)
    print(f'\n报告: {path}')
    print(f'原始结果: {run_dir / "results.jsonl"}')
    return 130 if interrupted else 0


if __name__ == '__main__':
    sys.exit(main())
