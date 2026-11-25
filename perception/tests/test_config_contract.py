"""configSchema 往返契约：卡片声明的每个配置键，config 动作必须消费并回显。

画布弹窗按 configSchema 渲染；没有任何机制保证插件的 config 动作真的应用了
弹窗发来的键。这类失败天然静默——pose 曾带着 28 个 schema 字段上线，其中 3 个
被丢在地板上而调用仍回答 configured（plugins/pose.py 里那条注释）；battery_monitor
v0 也是只回 configured 不应用。本文件把「消费」变成接口层可观察的事实：

  契约：config 动作的 ack 必须回显生效后的合并配置（`config` 键）。
  没有这个回显，接口层就没有任何读回通道。

覆盖方式（棘轮，不静默降级）：

  ROUND_TRIP  这些插件在本测试里逐键强制往返。新加 schema 字段必须跟着过。
  PENDING     显式登记的缓办项，必须带理由；测试输出会打印它们。
  其余        能离机构造、有非空 schema、又不在两张名单里的插件 → 测试失败，
              强迫做一次显式决定（修好进 ROUND_TRIP，或带理由进 PENDING）。

构造不了的插件自动进 PENDING（理由是构造异常），import 失败同理。子包插件
（plugins/measure/）暂不在扫描范围内。array/object 类型的键暂不发哨兵值。
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import traceback

import pytest

import vision_stubs  # noqa: F401  (安装 ROS 桩、摆好 sys.path)
from vision_stubs import _FakeExecutor

PLUGINS_DIR = pathlib.Path(vision_stubs.PERCEPTION_ROOT) / 'plugins'

# 逐键强制往返的插件。加名字之前先让它的 config ack 回显合并配置。
ROUND_TRIP = {'battery_monitor', 'echo_cancellation', 'ocr', 'barcode', 'face'}

# 显式缓办：constructible 但还没接契约。理由必须写实，空字符串按失败处理。
PENDING = {
    'asr': 'config 动作把键逐个吃进属性（kws/trigger_mode 有联动），全键消费与 ack 回显待办',
    'audio_mixer': 'config 动作待核对是否全键消费；ack 未回显',
    'tts': 'schema 大（10 处声明，含 engine 切换副作用），且构造即加载模型，离机无法往返；ack 回显待办',
    'speaker': 'config 动作待核对；ack 未回显',
    'phanthy_music': 'config 动作待核对；ack 未回显',
    'vits2_tts': 'tts 卡的 TensorRT 换装变体（构建期替换 plugins/tts，同一 PREFIX），契约跟随 plugins/tts',
}


def _facade_classes(module):
    """模块里长得像 facade 的类：有 PREFIX / dispatch / get_tools。"""
    found = []
    for name in getattr(module, '__dict__', {}):
        obj = getattr(module, name, None)
        if not inspect.isclass(obj) or obj.__module__ != module.__name__:
            continue
        if hasattr(obj, 'PREFIX') and hasattr(obj, 'dispatch') and hasattr(obj, 'get_tools'):
            found.append(obj)
    return found


def _constructible(cls) -> bool:
    """(plugin_cfg, executor) 能不能造出来：多余的参数必须有默认值。"""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())[1:]  # 去掉 self
    for param in params[2:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            return False
    return True


def _schema_properties(tool) -> dict:
    props = ((tool or {}).get('configSchema') or {}).get('properties') or {}
    return props if isinstance(props, dict) else {}


def _sentinel_for(prop: dict, key: str):
    """能从 default 区分出来的哨兵值；None 表示这个键没法安全区分（跳过）。"""
    values = None
    if 'enum' in prop:
        values = prop['enum']
    elif 'oneOf' in prop:
        values = [item.get('const') if isinstance(item, dict) else item
                  for item in (prop['oneOf'] or [])]
    if values is not None:
        choices = [v for v in values if v is not None and v != prop.get('default')]
        if choices:
            return choices[0]
        if prop.get('type') != 'string':
            return None
        # 枚举只剩 default（如 echo 的 oneOf 由 get_tools 按请求填充，离机时
        # 只剩 auto）：插件端的契约是「发来的键要被消费」，string 键发裸哨兵
        # 一样能证明。画布只会发合法选项，但存档行里可能是什么都行。
    kind = prop.get('type')
    if kind == 'boolean':
        return not prop.get('default', False)
    if kind in ('number', 'integer'):
        base = prop.get('default')
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            base = 0
        return base + 7
    if kind in ('array', 'object'):
        return None
    return f'__contract__:{key}'


def _round_trip(plugin, prefix: str, props: dict, instance_id: str = ''):
    """发一轮哨兵，断言 ack 回显。返回 (发出哨兵, 跳过键)。"""
    sentinels, skipped = {}, []
    for key, prop in props.items():
        value = _sentinel_for(prop, key)
        if value is None:
            skipped.append(key)
        else:
            sentinels[key] = value
    if not sentinels:
        return sentinels, skipped

    args = {'action': 'config', **sentinels}
    if instance_id:
        args['instance_id'] = instance_id
    result = plugin.dispatch(prefix, args)
    assert isinstance(result, dict), f'{prefix}: config 动作没有应答（返回 {result!r}）'
    assert result.get('status') == 'configured', f'{prefix}: ack 非 configured：{result!r}'
    echoed = result.get('config')
    assert isinstance(echoed, dict), (
        f'{prefix}: config ack 不回显生效配置（无 config 键）——往返契约没有读回通道：{result!r}')
    for key, value in sentinels.items():
        assert echoed.get(key) == value, (
            f'{prefix}: schema 键 {key!r} 发了哨兵 {value!r}，ack 回显 '
            f'{echoed.get(key)!r} —— 该键没有被消费（pose/battery 踩过的那类静默丢失）')
    return sentinels, skipped


def test_config_round_trip_contract():
    checked, pending, notes = [], dict(PENDING), []
    for path in sorted(PLUGINS_DIR.glob('*.py')):
        stem = path.stem
        if stem.startswith('_'):
            continue
        try:
            module = importlib.import_module(f'plugins.{stem}')
        except Exception as error:  # 重依赖（torch/cv2 变体）缺失：登记，不静默
            pending.setdefault(stem, f'import 失败: {type(error).__name__}: {error}')
            continue

        for cls in _facade_classes(module):
            prefix = getattr(cls, 'PREFIX', stem)
            if not _constructible(cls):
                sig = ''
                try:
                    sig = str(inspect.signature(cls.__init__))
                except (TypeError, ValueError):
                    pass
                pending.setdefault(stem, f'离机不可构造：__init__{sig}')
                continue
            try:
                plugin = cls({}, _FakeExecutor())
            except Exception as error:
                pending.setdefault(stem, f'构造失败: {type(error).__name__}: {error}')
                continue

            schemas = [(tool, _schema_properties(tool))
                       for tool in plugin.get_tools() if _schema_properties(tool)]
            if not schemas:
                continue  # 没有非空 configSchema，往返无从谈起

            if stem not in ROUND_TRIP:
                assert stem in pending, (
                    f'{stem} 可离机构造且声明了 configSchema，但既不在 ROUND_TRIP '
                    f'也不在 PENDING——请让它通过往返后加入 ROUND_TRIP，或带着写实理由登记 PENDING')
                continue

            for tool, props in schemas:
                sentinels, skipped = _round_trip(plugin, prefix, props)
                notes.append(f'{stem}: {len(sentinels)} 键往返通过'
                             + (f'（跳过 {skipped}）' if skipped else ''))
                if any(prop.get('scope') == 'instance' for prop in props.values()):
                    inst_sentinels, inst_skipped = _round_trip(
                        plugin, prefix, {k: v for k, v in props.items()
                                         if v.get('scope') == 'instance'},
                        instance_id='contract')
                    notes.append(f'{stem}: instance 寻址往返 {len(inst_sentinels)} 键'
                                 + (f'（跳过 {inst_skipped}）' if inst_skipped else ''))

            checked.append(stem)

    assert 'battery_monitor' in checked, '往返覆盖退化为空——契约测试名存实亡'
    for name, reason in pending.items():
        assert isinstance(reason, str) and reason.strip(), f'PENDING[{name}] 理由不能为空'
    print('\n[config contract] enforced:', sorted(checked))
    print('[config contract] pending:', len(pending))
    for line in notes:
        print('[config contract]', line)


def test_round_trip_list_and_pending_list_do_not_overlap():
    assert not (ROUND_TRIP & set(PENDING)), '同一插件不能同时在 ROUND_TRIP 和 PENDING'


def test_pending_reasons_survive_refactor_of_this_file():
    """PENDING 里的名字必须真实存在于插件目录——防止名单腐烂成文学创作。"""
    for name in PENDING:
        assert (PLUGINS_DIR / f'{name}.py').exists(), f'PENDING 里的 {name} 不是 plugins/ 下的模块'
