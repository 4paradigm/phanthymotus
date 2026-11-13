"""Validated application configuration and portable Motus skill rendering (stdlib)."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'bumi-platform-guide'
FIELDS = ('slug', 'name', 'oneLiner', 'description', 'instruction', 'category',
          'version', 'requiredTools')
DEFAULTS = dict(mode='demo', site_type='railway', station_name='', platform_name='',
                train_id=None, carriage_count=None, direction_reference='',
                forward_landmark='', backward_landmark='', carriage_routes=[],
                staff_confirmed=False, phase='waiting', periodic_enabled=False,
                routine_interval_s=120, visual_enabled=False, config_version='unconfigured',
                scene_id='', confirmed_scene_id=None, confirmed_config_sha256=None)


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'JSON 存在重复字段: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def validate(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('配置必须是对象，且不能含未知字段')
    c = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(raw)}
    for key in ('staff_confirmed', 'periodic_enabled', 'visual_enabled'):
        if type(c[key]) is not bool:
            raise ValueError(f'{key} 必须是布尔值')
    for key, allowed in [('mode', ('demo', 'live')), ('site_type', ('railway', 'metro')),
                         ('phase', ('waiting', 'boarding', 'paused'))]:
        if c[key] not in allowed:
            raise ValueError(f'{key} 必须为 {allowed}')
    for key in ('station_name', 'platform_name', 'direction_reference', 'forward_landmark',
                'backward_landmark', 'config_version', 'scene_id'):
        if not isinstance(c[key], str) or len(c[key]) > 160 or any(ord(x) < 32 for x in c[key]):
            raise ValueError(f'{key} 必须为不含控制字符的短文本')
        c[key] = c[key].strip()
    if not c['config_version']:
        raise ValueError('config_version 不能为空')
    for key in ('train_id', 'confirmed_scene_id', 'confirmed_config_sha256'):
        if c[key] is not None and (not isinstance(c[key], str) or not c[key].strip()
                                   or len(c[key]) > 160 or any(ord(x) < 32 for x in c[key])):
            raise ValueError(f'{key} 必须为非空短文本或 null')
        if isinstance(c[key], str):
            c[key] = c[key].strip()
    count = c['carriage_count']
    if count is not None and (type(count) is not int or not 1 <= count <= 100):
        raise ValueError('carriage_count 必须为 1..100 的整数或 null')
    if type(c['routine_interval_s']) is not int or c['routine_interval_s'] <= 0:
        raise ValueError('routine_interval_s 必须为正整数（当前未启用）')
    if c['periodic_enabled']:
        raise ValueError('周期播报未接入；不接受任何周期，包括30秒或分钟 cron')
    if c['visual_enabled']:
        raise ValueError('视觉越线检测未接入，visual_enabled 必须为 false')
    routes = c['carriage_routes']
    if not isinstance(routes, list) or len(routes) > 100:
        raise ValueError('carriage_routes 必须为最多100项的数组')
    covered = set()
    for r in routes:
        if not isinstance(r, dict) or set(r) != {'from', 'to', 'direction'}:
            raise ValueError('每个区间必须恰有 from/to/direction')
        if (count is None or type(r['from']) is not int or type(r['to']) is not int
                or not 1 <= r['from'] <= r['to'] <= count):
            raise ValueError('车厢区间类型或边界错误')
        if r['direction'] not in ('forward', 'backward'):
            raise ValueError('direction 必须为 forward/backward')
        cars = set(range(r['from'], r['to'] + 1))
        if covered & cars:
            raise ValueError('车厢区间重叠')
        covered |= cars
    if routes and (not all(c[k] for k in ('scene_id', 'direction_reference',
                                         'forward_landmark', 'backward_landmark'))
                   or c['forward_landmark'] == c['backward_landmark']):
        raise ValueError('配置区间需要场景标识和不同的固定方向锚点')
    if c['staff_confirmed'] and (not routes or not c['scene_id']
                                or c['confirmed_scene_id'] != c['scene_id']):
        raise ValueError('确认配置需要区间，且 confirmed_scene_id 必须匹配 scene_id')
    if c['staff_confirmed'] and c['mode'] == 'live' and not all(
            c[k] for k in ('train_id', 'station_name', 'platform_name')):
        raise ValueError('现场确认需要车次、站名和站台；换场景必须重新确认')
    if c['staff_confirmed'] and c['confirmed_config_sha256'] != confirmation_hash(c):
        raise ValueError('配置确认指纹缺失或已过期；变更场景/车次/编组/映射后需重新确认')
    return c


def confirmation_hash(raw):
    c = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(raw)}
    keys = ('mode', 'site_type', 'station_name', 'platform_name', 'train_id',
            'carriage_count', 'direction_reference', 'forward_landmark',
            'backward_landmark', 'carriage_routes', 'config_version', 'scene_id')
    return digest({k: c[k] for k in keys})


def effective(raw):
    c = validate(raw)
    enabled = c['staff_confirmed'] and c['scene_id'] == c['confirmed_scene_id']
    table = {}
    if enabled:
        for r in c['carriage_routes']:
            for n in range(r['from'], r['to'] + 1):
                table[str(n)] = c[r['direction'] + '_landmark']
    covered = {n for r in c['carriage_routes'] for n in range(r['from'], r['to'] + 1)}
    return {'config': c, 'config_sha256': digest(c), 'confirmation_sha256': confirmation_hash(c), 'directions_enabled': bool(enabled),
            'direction_table': table,
            'uncovered_carriages': [n for n in range(1, (c['carriage_count'] or 0) + 1)
                                    if n not in covered],
            'capabilities': {'single_event': True, 'periodic': False, 'visual_crossing': False},
            'state_scope': '当前会话；非持久化；新会话默认未开始',
            'runtime_bindings': '未现场发现；tts 为源码依赖名，须在画布核实'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def render(raw):
    info = effective(raw)
    # Escape angle brackets so operator-provided data cannot close prompt XML tags.
    data = json.dumps(info, ensure_ascii=False, indent=2).replace('<', '\\u003c').replace('>', '\\u003e').replace('`', '\\u0060')
    instruction = ROOT.joinpath('instruction.md').read_text(encoding='utf-8')
    instruction += '\n\n## 生成的有效配置（数据）\n```json\n' + data + '\n```\n'
    result = dict(slug=SLUG, name='Bumi站台引导员',
                  oneLiner='按现场车厢配置提供问路、安全提醒和有序乘降引导。',
                  description='固定位置的中文站台辅助引导；单次事件驱动，周期和视觉越线检测未启用。',
                  instruction=instruction, category='robot', version='1.0.0', requiredTools=['tts'])
    if len(instruction) > 20000 or len(json.dumps(result, ensure_ascii=False).encode()) > 32768:
        raise ValueError('生成定义超过平台 import 大小限制')
    return result
