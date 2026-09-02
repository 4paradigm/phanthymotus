"""配置：把 YAML 展开成一组扁平的评测组，并保证密钥不落明文。

三种组来源可混用：
  1. YAML 里显式列举 groups（一个 url+key 可带多个 model，自动展开成多组）
  2. include_current — 从 ConfigDB 读当前生产配置，回答「换完会不会比现在差」
  3. （两者的并集，组名冲突时自动加后缀区分）
"""
import json
import os
import pathlib
import sqlite3

try:
    import yaml
except ImportError:  # pragma: no cover - 部署环境一定有 pyyaml
    yaml = None


class ConfigError(Exception):
    pass


DEFAULTS = {
    # sampling 默认 trace：它同时给出真实缓存率和连续的轮次形态。even 只在
    # 「只关心延迟、想覆盖尽可能杂的 prompt 规模」时才更合适。
    'corpus': {'dir': 'resource/llm_data', 'count': 50, 'sampling': 'trace',
               'seed': 42, 'min_messages': 2},
    'request': {'max_tokens': 10240, 'timeout_s': 180, 'extra_body': {}},
    # parallel 默认 false：串行是保守选择，各组互不干扰。开了更快，代价见
    # README「并行」一节。
    'run': {'order': 'rotate', 'parallel': False,
            'stop_after_consecutive_failures': 0},
    'include_current': False,
    'groups': [],
}


def mask(secret: str) -> str:
    """密钥打码。报告和 run.json 里只允许出现这个形式。

    保留头尾是为了让人能认出「是哪一把 key」而不泄露它——排查配置错误时，
    知道用的是 sk-RjA…jpt5 还是 sk_-8Sg…nLZ4 往往就够了。
    """
    if not secret:
        return '<empty>'
    if len(secret) <= 12:
        return secret[:2] + '…' + secret[-2:]
    return f'{secret[:6]}…{secret[-4:]}'


_OTHER_KEY_FIELDS = ('key_env', 'key_file', 'key_from_current')


def _resolve_key(spec: dict, name: str) -> str:
    """取密钥。只有一种方式：YAML 里直接写 `key:`。

    配置文件放在机器人的 /opt/phanthy-motus/data/bench.yaml —— 宿主机的运行时
    数据目录，不在仓库里，`.gitignore` 也挡了 bench.yaml，所以明文放这儿是合适的。
    刻意不支持环境变量/外部文件：多一种取密钥的方式就多一类「为什么没生效」的
    排查，而这个工具的配置本来就该一眼看完。
    """
    key = spec.get('key')
    if key:
        return str(key).strip()

    # 用了已经废弃的写法时给出明确指引，且**绝不回显值** —— 那些字段里往往就是
    # 密钥本身，打进报错就等于写进日志。
    used = [f for f in _OTHER_KEY_FIELDS if spec.get(f)]
    if used:
        raise ConfigError(
            f'组 {name}: 不再支持 {"/".join(used)}，密钥直接写在 bench.yaml 里：\n'
            '    key: sk-...\n'
            '  （bench.yaml 已被 .gitignore 挡住，模板见 bench.yaml.example）')

    raise ConfigError(
        f'组 {name}: 缺密钥。在 bench.yaml 里写 `key: sk-...`'
        '（模板见 bench.yaml.example）。')


class Group:
    """一个评测组 = 一个 (url, key, model) 三元组。"""

    def __init__(self, name: str, url: str, key: str, model: str,
                 extra_body: dict | None = None, source: str = 'yaml'):
        self.name = name
        self.url = url.rstrip('/')
        self.key = key
        self.model = model
        self.extra_body = extra_body or {}
        self.source = source

    @property
    def host(self) -> str:
        return self.url.split('//')[-1].split('/')[0]

    def public(self) -> dict:
        """可安全序列化的形式——key 已打码。"""
        return {'name': self.name, 'url': self.url, 'model': self.model,
                'key': mask(self.key), 'source': self.source,
                'extra_body': self.extra_body}

    def __repr__(self):
        return f'<Group {self.name} {self.host} {self.model}>'


def _current_from_db(db_path: str) -> list[dict]:
    """从 ConfigDB 读当前生产配置的 LLM 入口。

    直接开 sqlite 只读，**不 import src/config.py**：那个模块在 import 时就会建库
    并跑端口迁移，一个评测工具不该有这种副作用。
    """
    p = pathlib.Path(db_path)
    if not p.is_file():
        return []
    uri = f'file:{p}?mode=ro'
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        raise ConfigError(f'读 ConfigDB 失败 ({db_path}): {e}')
    try:
        row = conn.execute("SELECT value FROM config WHERE key='client'").fetchone()
    except sqlite3.Error as e:
        raise ConfigError(f'ConfigDB 无 config 表或不可读: {e}')
    finally:
        conn.close()
    if not row:
        return []
    try:
        return json.loads(row[0]).get('llm', []) or []
    except (ValueError, AttributeError):
        return []


_REMOVED_KEYS = {
    'repeats': '已移除：不再做重复轮（重复重放同一条 payload 量的是「同一个请求'
               '重发」，前缀缓存必然命中，不代表真实负载）。显著性改用配对差的'
               '置信区间 + 符号检验，不需要重复测量。',
    'warmup': '已移除：只跑一轮。预热轮会把每条 payload 变成「原样重放」，'
              '缓存命中必然接近 100%，反而污染了缓存测量；而组顺序轮换'
              '（run.order: rotate）已经消除了谁先跑谁吃冷缓存的不对称。',
}


def _check_keys(cfg: dict, loaded: dict, warn) -> None:
    """对未知/已移除的配置键给出提示。

    YAML 里多一个键默认是静默忽略的，那意味着写错的键名和过期的键都得不到任何
    反馈 —— 你以为设了，其实没生效。刚移除 repeats 之后这一点尤其要紧。
    """
    for section in ('corpus', 'request', 'run'):
        known = set(DEFAULTS[section])
        for k in (loaded.get(section) or {}):
            if k in _REMOVED_KEYS:
                warn(f'[!] {section}.{k} {_REMOVED_KEYS[k]}')
            elif k not in known:
                warn(f'[!] {section}.{k} 不是已知配置项（可用: '
                     f'{", ".join(sorted(known))}）—— 已忽略')

    top_known = set(DEFAULTS) | {'groups', 'include_current'}
    for k in loaded:
        if k not in top_known:
            warn(f'[!] 顶层配置项 {k} 未知 —— 已忽略')

    group_known = {'name', 'url', 'key', 'model', 'models', 'extra_body'}
    for g in (loaded.get('groups') or []):
        if not isinstance(g, dict):
            continue
        for k in g:
            if k not in group_known and k not in _OTHER_KEY_FIELDS:
                warn(f'[!] 组 {g.get("name", "?")} 的 {k} 未知 —— 已忽略')


def load(path: str | None, overrides: dict | None = None, warn=print) -> dict:
    """读 YAML 并与默认值合并（按段落深合并，不是整段覆盖）。"""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if path:
        if yaml is None:
            raise ConfigError('缺少 pyyaml，无法读配置文件')
        p = pathlib.Path(path)
        if not p.is_file():
            raise ConfigError(f'配置文件不存在: {path}')
        loaded = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f'{path} 顶层必须是一个 mapping')
        _check_keys(cfg, loaded, warn)
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    for section, kv in (overrides or {}).items():
        if kv is None:
            continue
        if isinstance(cfg.get(section), dict) and isinstance(kv, dict):
            cfg[section].update({k: v for k, v in kv.items() if v is not None})
        else:
            cfg[section] = kv
    return cfg


def build_groups(cfg: dict, db_path: str | None = None) -> list[Group]:
    """展开成扁平的组列表。"""
    groups: list[Group] = []
    default_extra = cfg.get('request', {}).get('extra_body') or {}


    for spec in cfg.get('groups') or []:
        if not isinstance(spec, dict):
            raise ConfigError(f'groups 里出现非 mapping 条目: {spec!r}')
        name = spec.get('name') or spec.get('url', 'unnamed')
        url = spec.get('url')
        if not url:
            raise ConfigError(f'组 {name}: 缺 url')

        models = spec.get('models') or ([spec['model']] if spec.get('model') else [])
        if not models:
            raise ConfigError(f'组 {name}: 必须提供 model 或 models')

        key = _resolve_key(spec, name)
        extra = {**default_extra, **(spec.get('extra_body') or {})}
        for m in models:
            # 一个入口配多个模型时组名必须带上模型，否则报告里两行同名
            gname = f'{name}/{m}' if len(models) > 1 else name
            groups.append(Group(gname, url, key, m, extra, source='yaml'))

    if cfg.get('include_current'):
        # 只在真的要 baseline 时才碰 ConfigDB：$DB_PATH 可能指向一个没有 config
        # 表的库，不该让「压根不关心生产配置」的评测也被它挡死。
        db = db_path or os.environ.get('DB_PATH', 'resource/data.db')
        for item in _current_from_db(db):
            url, key, model = item.get('url'), item.get('key'), item.get('model')
            if not (url and key and model):
                continue
            extra = dict(default_extra)
            if not item.get('think_mode', False):
                # 与 src/client/llm.py:136-139 的生产行为一致
                extra.setdefault('chat_template_kwargs', {'enable_thinking': False})
            groups.append(Group(f'current/{model}', url, key, model, extra,
                                source='configdb'))

    if not groups:
        raise ConfigError('没有可评测的组：groups 为空且 include_current 没读到配置')

    # 组名去重——重名会让报告里两行无法区分，也会让 resume 的三元组键冲突
    seen: dict[str, int] = {}
    for g in groups:
        if g.name in seen:
            seen[g.name] += 1
            g.name = f'{g.name}#{seen[g.name]}'
        else:
            seen[g.name] = 1
    return groups
