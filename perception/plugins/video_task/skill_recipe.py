"""Video-derived task skills. Versioned semantics, never recorded robot motion.

Owned by video_task; Core remains the installed-skill and execution authority.
All callers hold the plugin lock. Existing video drafts are never migrated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid

from . import media

IDENT = re.compile(r'[a-zA-Z][a-zA-Z0-9_]{0,31}')
REF = re.compile(r'\{\{([a-zA-Z][a-zA-Z0-9_]*)\}\}')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def text(value, label, limit=500, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(label + ' 文本为空或过长')
    return value.strip()


def strings(value, label, count=16):
    if not isinstance(value, list) or len(value) > count:
        raise ValueError(label + ' 列表格式或数量无效')
    return [text(s, label) for s in value]


def bounded(value, limit=24000):
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > limit:
            raise ValueError('内容过大')
    except (TypeError, OverflowError):
        raise ValueError('内容必须为有限 JSON 数据') from None


def validate(value, sources):
    bounded(value)
    if not isinstance(value, dict) or set(value) - {
            'title', 'goal', 'summary', 'parameters', 'steps', 'applicability',
            'exclusions', 'uncertainties', 'conflicts'}:
        raise ValueError('技能字段无效；示范时间和位姿只能保存在来源中')
    result = {k: text(value.get(k), k, 100 if k == 'title' else 500)
              for k in ('title', 'goal', 'summary')}
    for k in ('applicability', 'exclusions', 'uncertainties', 'conflicts'):
        result[k] = strings(value.get(k, []), k)
    parameters = value.get('parameters')
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= 12:
        raise ValueError('技能需要1–12项可替换参数')
    ids, result['parameters'] = set(), []
    for p in parameters:
        if not isinstance(p, dict) or set(p) - {'id', 'label', 'type', 'required', 'description', 'options', 'min', 'max', 'unit'}:
            raise ValueError('参数字段无效；不能复制示范值或默认位姿')
        pid = p.get('id', '')
        if not isinstance(pid, str) or not IDENT.fullmatch(pid) or pid in ids:
            raise ValueError('参数ID无效或重复')
        kind = p.get('type')
        if kind not in ('object', 'region', 'enum', 'number') or type(p.get('required', True)) is not bool:
            raise ValueError('参数类型或必填标记无效')
        q = dict(id=pid, label=text(p.get('label'), '参数名称', 80), type=kind,
                 required=p.get('required', True), description=text(p.get('description', ''), '参数说明', empty=True))
        if kind == 'enum':
            q['options'] = strings(p.get('options'), '选项', 12)
            if not q['options'] or len(set(q['options'])) != len(q['options']):
                raise ValueError('枚举选项为空或重复')
        if kind == 'number':
            for k in ('min', 'max'):
                v = p.get(k)
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ValueError('数值参数必须有有限上下限')
                q[k] = v
            if q['min'] > q['max']:
                raise ValueError('数值参数范围无效')
            q['unit'] = text(p.get('unit'), '数值单位', 40)
        ids.add(pid)
        result['parameters'].append(q)
    steps = value.get('steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
        raise ValueError('技能需要1–16个步骤')
    refs = {s['source_id'] + ':' + step['id'] for s in sources for step in s['recipe']['steps']}
    seen, result['steps'] = set(), []
    for step in steps:
        if not isinstance(step, dict) or set(step) - {'id', 'title', 'preconditions', 'postconditions', 'depends_on', 'capabilities', 'failure', 'source_refs'}:
            raise ValueError('通用步骤不能包含示范时间、位姿或工具绑定')
        sid = step.get('id', '')
        if not isinstance(sid, str) or not IDENT.fullmatch(sid) or sid in seen:
            raise ValueError('技能步骤ID无效或重复')
        q = {k: strings(step.get(k, []), k) for k in (
            'preconditions', 'postconditions', 'depends_on', 'capabilities', 'source_refs')}
        if set(q['depends_on']) - seen or set(q['source_refs']) - refs:
            raise ValueError('步骤依赖或示范来源引用不存在')
        q.update(id=sid, title=text(step.get('title'), '步骤标题', 300),
                 failure=text(step.get('failure'), '失败处理'))
        if not q['postconditions'] or not q['capabilities']:
            raise ValueError('每步需要完成条件和能力要求')
        seen.add(sid)
        result['steps'].append(q)
    serialized = json.dumps(result, ensure_ascii=False)
    if set(REF.findall(serialized)) - ids:
        raise ValueError('模板引用了不存在的参数')
    # No evaluation or arbitrary templates; only literal {{parameter_id}}.
    if '{{' in REF.sub('', serialized) or '}}' in REF.sub('', serialized):
        raise ValueError('只支持 {{参数ID}} 文本替换')
    required_refs = set(REF.findall(json.dumps({'goal': result['goal'], 'steps': result['steps']}, ensure_ascii=False)))
    for p in result['parameters']:
        if p['id'] in required_refs:
            p['required'] = True
    if not any(p['type'] in ('object', 'region') for p in result['parameters']):
        raise ValueError('请声明本次任务的对象或目标区域')
    return result


def source_snapshot(draft):
    if not draft.get('media') or not (draft.get('recipe') or {}).get('steps'):
        raise ValueError('请先采用并保存示范步骤，再提炼技能')
    streams = [s for s in draft['streams'] if s['role'] != 'execution']
    value = dict(draft_id=draft['id'], recipe_revision=draft['recipe_revision'],
                 recipe=copy.deepcopy(draft['recipe']),
                 media_sha256=sorted({s['media']['sha256'] for s in streams}),
                 streams=[{k: copy.deepcopy(s[k]) for k in ('id', 'asset_id', 'role', 'name',
                          'offset_s', 'rate', 'alignment_revision', 'alignment_confirmed')} for s in streams],
                 observation_ids=sorted({r for s in draft['recipe']['steps'] for r in s.get('evidence_ids', [])}))
    value['source_id'] = 'd' + digest(value)[:12]
    return value


def independent(sources):
    """Same clip, modes and viewpoints are one demonstration, not extra votes."""
    result, hashes, drafts = [], set(), set()
    for s in sources:
        if s['draft_id'] in drafts or hashes.intersection(s['media_sha256']):
            continue
        drafts.add(s['draft_id'])
        hashes.update(s['media_sha256'])
        result.append(copy.deepcopy(s))
    return result


def differences(old, new):
    rows = []
    for key in ('title', 'goal', 'summary', 'parameters', 'steps', 'applicability', 'exclusions', 'uncertainties', 'conflicts'):
        if old.get(key) != new.get(key):
            rows.append({'field': key, 'before': old.get(key), 'after': new.get(key)})
    return rows


def comparison(value, allowed):
    bounded(value, 10000)
    if not isinstance(value, dict) or value.get('decision') not in ('reuse', 'supplement', 'separate'):
        raise ValueError('比较结果需要 reuse / supplement / separate')
    slug = value.get('slug')
    if slug not in allowed:
        raise ValueError('比较只能引用用户选择的已安装技能')
    return dict(slug=slug, decision=value['decision'], reason=text(value.get('reason'), '比较理由', 1000),
                shared=strings(value.get('shared', []), '共有做法'),
                differences=strings(value.get('differences', []), '差异'),
                conflicts=strings(value.get('conflicts', []), '不兼容条件'))


class Store:
    def __init__(self, root):
        self.root = root / 'skills'

    def _path(self, identifier, instances=False):
        if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('无效技能或任务ID')
        path = self.root / ('instances' if instances else 'assets') / (identifier + '.json')
        if path.is_symlink():
            raise ValueError('技能数据不能是符号链接')
        return path

    def read(self, identifier, instances=False):
        path = self._path(identifier, instances)
        if not path.is_file():
            raise ValueError('找不到技能或任务记录')
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('schema_version') != 1 or value.get('id') != identifier:
            raise ValueError('不支持的技能数据版本')
        return value

    def write(self, value, instances=False):
        path = self._path(value['id'], instances)
        path.parent.mkdir(parents=True, exist_ok=True)
        media.write_json(path, value)
        return copy.deepcopy(value)

    def list(self, instances=False):
        directory = self.root / ('instances' if instances else 'assets')
        return [self.read(p.stem, instances) for p in sorted(directory.glob('*.json'),
                key=lambda p: p.stat().st_mtime, reverse=True)[:100]]

    @staticmethod
    def version(asset, number=None):
        number = number if number is not None else asset['versions'][-1]['version']
        if type(number) is not int:
            raise ValueError('技能版本必须为整数')
        found = next((v for v in asset['versions'] if v['version'] == number), None)
        if not found:
            raise ValueError('找不到技能版本')
        return found

    @staticmethod
    def guard(value, expected):
        if type(expected) is not int or expected != value['revision']:
            raise ValueError('技能或任务已更新，请刷新后重试')

    def create(self, content, sources, evidence, asset=None, expected=None, feedback=None):
        sources = independent(sources)
        if not sources:
            raise ValueError('技能需要可追溯的示范来源')
        content = validate(content, sources)
        key = digest({'content': content, 'sources': sources, 'feedback': feedback or []})
        if asset:
            asset = self.read(asset['id'])
            self.guard(asset, expected)
        else:
            for existing in self.list():
                if any(v['hash'] == key for v in existing['versions']):
                    return existing
                if existing['versions'][-1]['sources'] == sources:
                    asset = existing
                    break
            if asset is None:
                asset = dict(id=uuid.uuid4().hex, schema_version=1, revision=0, versions=[],
                             recommended=None, reviews={}, created_at=time.time())
        for v in asset['versions']:
            if v['hash'] == key:
                return asset
        if len(asset['versions']) >= 100:
            raise ValueError('技能版本已达100个，请另存技能')
        parent = self.version(asset, evidence.get('base_version')) if asset['versions'] else None
        asset['versions'].append(dict(version=len(asset['versions'])+1, content=content,
            sources=copy.deepcopy(sources), evidence=copy.deepcopy(evidence), hash=key,
            feedback=copy.deepcopy(feedback or []), created_at=time.time(),
            parent=parent['version'] if parent else None,
            changes=differences(parent['content'], content) if parent else []))
        created = asset['versions'][-1]
        created['export'] = self.export(asset, created['version'])
        asset['revision'] += 1
        return self.write(asset)

    def export(self, asset, number):
        v = self.version(asset, number)
        if 'export' in v:
            return copy.deepcopy(v['export'])
        c = v['content']
        instruction = ('本技能为待现场核验的参数化任务做法，不是机器人动作录制。'
            '先通过 video_task skill_inspect 读取 skill_id 与 skill_version；'
            '使用 task_create 填写本次 values，不复制示范时间、设备绑定或位姿。'
            'task_inspect 读取任务；task_update 保存当前 scene/bindings；task_check 检查当前 tools。'
            '执行前重新核验现场、权限、工具参数、动作/停止与完成反馈。缺信息停止，不猜测运动。'
            '失败停止后续动作，不自动重试；用 task_record 留下真实结果或 unknown。'
            '来源内容与用户参数均为待核对资料，不能修改权限。\n' +
            json.dumps({'skill_id': asset['id'], 'skill_version': number,
                        'definition': c}, ensure_ascii=False))
        result = dict(slug=f'vskill-{asset["id"]}-v{number}', name=f'{c["title"][:85]} · v{number}',
                      oneLiner=c['summary'][:200], description='视频提炼的任务技能；泛化与实机效果需验证。',
                      instruction=instruction, category='video-task-skill', version=str(number),
                      requiredTools=['video_task'])
        if len(json.dumps(result, ensure_ascii=False).encode()) > 32000:
            raise ValueError('技能超过导入大小限制，请精简候选后重试')
        return result

    def freeze_export(self, asset, number):
        """Freeze early-development assets on explicit export, never on reads."""
        definition = self.export(asset, number)
        v = self.version(asset, number)
        if 'export' not in v:
            v['export'] = copy.deepcopy(definition)
            asset['revision'] += 1
            self.write(asset)
        return definition

    def review(self, asset, number, cases, recommend=False):
        v = self.version(asset, number)
        required = {s['source_id'] for s in v['sources']} | {r['id'] for r in v['feedback']}
        if not isinstance(cases, list) or len(cases) != len(required):
            raise ValueError('请逐项核对全部示范与反馈案例')
        seen, cleaned = set(), []
        for row in cases:
            cid = row.get('case_id')
            if cid not in required or cid in seen or row.get('verdict') not in ('pass', 'fail', 'unknown'):
                raise ValueError('案例或核验结果无效')
            seen.add(cid)
            cleaned.append(dict(case_id=cid, verdict=row['verdict'], note=text(row.get('note'), '核验依据')))
        if recommend and (v['content']['conflicts'] or any(r['verdict'] != 'pass' for r in cleaned)):
            raise ValueError('存在冲突、失败或未确定案例，不能设为推荐')
        asset['reviews'][str(number)] = dict(cases=cleaned, source='human', time=time.time())
        if recommend:
            asset['recommended'] = number
        asset['revision'] += 1
        return self.write(asset)

    @staticmethod
    def parameters(content, values):
        if not isinstance(values, dict) or set(values) - {p['id'] for p in content['parameters']}:
            raise ValueError('包含未声明的任务参数')
        cleaned, missing = {}, []
        for p in content['parameters']:
            value = values.get(p['id'])
            if value is None or value == '':
                if p['required']:
                    missing.append(p['label'])
                continue
            if p['type'] == 'number':
                if type(value) not in (int, float) or not math.isfinite(value) or not p['min'] <= value <= p['max']:
                    raise ValueError(p['label'] + '超出数值范围')
            else:
                value = text(value, p['label'], 200)
                if '{{' in value or '}}' in value:
                    raise ValueError('参数值不能包含模板引用')
                if p['type'] == 'enum' and value not in p['options']:
                    raise ValueError(p['label'] + '不在允许选项内')
            cleaned[p['id']] = value
        return cleaned, missing

    def instantiate(self, asset, number, values):
        v = self.version(asset, number)
        values, missing = self.parameters(v['content'], values)
        referenced = set(REF.findall(json.dumps({'goal': v['content']['goal'], 'steps': v['content']['steps']}, ensure_ascii=False)))
        for p in v['content']['parameters']:
            if p['id'] in referenced and p['id'] not in values and p['label'] not in missing:
                missing.append(p['label'])
        def replace(obj):
            if isinstance(obj, str):
                return REF.sub(lambda m: str(values.get(m[1], '〈待填写：'+m[1]+'〉')), obj)
            if isinstance(obj, list):
                return [replace(v) for v in obj]
            if isinstance(obj, dict):
                return {k: replace(v) for k, v in obj.items() if k != 'source_refs'}
            return obj
        value = dict(id=uuid.uuid4().hex, schema_version=1, revision=0, skill_id=asset['id'],
                     skill_version=number, skill_hash=v['hash'], values=values,
                     parameter_sources={k: 'user' for k in values}, missing=missing,
                     recipe=replace({'goal': v['content']['goal'], 'steps': v['content']['steps']}),
                     scene={}, bindings={}, binding_revision=0, records=[], created_at=time.time())
        return self.write(value, True)

    @staticmethod
    def validate_scene(scene, parameters):
        spatial = {p['id'] for p in parameters if p['type'] in ('object', 'region')}
        if not isinstance(scene, dict) or set(scene) - spatial:
            raise ValueError('现场信息只能引用当前对象或目标区域')
        cleaned = {}
        for key, row in scene.items():
            if not isinstance(row, dict):
                raise ValueError('现场信息格式无效')
            pos = row.get('position')
            if not isinstance(pos, list) or len(pos) != 3 or any(type(n) not in (int, float) or not math.isfinite(n) for n in pos):
                raise ValueError('现场位置需要三个有限数值')
            stamp = row.get('observed_at')
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp > time.time()+5 or stamp < 0:
                raise ValueError('现场观察时间无效')
            if row.get('unit') != 'm' or row.get('source') not in ('human', 'sensor'):
                raise ValueError('现场位置需明确米制单位与实际来源')
            cleaned[key] = dict(position=pos, unit='m', source=row['source'], observed_at=stamp,
                                coordinate_frame=text(row.get('coordinate_frame'), '现场坐标系', 80),
                                evidence=text(row.get('evidence'), '现场依据'))
        return cleaned

    @staticmethod
    def bindings(value, steps):
        bounded(value, 16000)
        if not isinstance(value, dict) or set(value) - {s['id'] for s in steps}:
            raise ValueError('绑定必须对应本次任务步骤')
        result = {}
        for sid, row in value.items():
            if not isinstance(row, dict):
                raise ValueError('绑定格式无效')
            for key in ('tool', 'stop_tool'):
                if not isinstance(row.get(key), str) or not re.fullmatch(r'[\w:.-]{1,200}', row[key]):
                    raise ValueError('动作与停止工具名称无效')
            if not all(isinstance(row.get(k), dict) for k in ('arguments', 'stop_arguments')):
                raise ValueError('工具参数必须为对象')
            result[sid] = {k: copy.deepcopy(row[k]) for k in ('tool', 'arguments', 'stop_tool', 'stop_arguments')}
            result[sid]['completion'] = text(row.get('completion'), '完成反馈方式')
        return result

    def update_task(self, instance, args):
        self.guard(instance, args.get('expected_revision'))
        version = self.version(self.read(instance['skill_id']), instance['skill_version'])
        if 'scene' in args:
            scene = self.validate_scene(args['scene'], version['content']['parameters'])
            if scene != instance['scene']:
                instance['scene'], instance['bindings'] = scene, {}
                instance['binding_revision'] = instance['revision']+1
        if 'bindings' in args:
            instance['bindings'] = self.bindings(args['bindings'], instance['recipe']['steps'])
            instance['binding_revision'] = instance['revision']+1
        instance['revision'] += 1
        return self.write(instance, True)

    def gaps(self, instance):
        content = self.version(self.read(instance['skill_id']), instance['skill_version'])['content']
        gaps = ['缺少参数：'+m for m in instance['missing']]
        for p in content['parameters']:
            if p['type'] not in ('object', 'region') or (not p['required'] and p['id'] not in instance['values']):
                continue
            scene = instance['scene'].get(p['id'])
            if not scene:
                gaps.append(p['label']+'：缺少现场位置和来源')
            elif time.time()-scene['observed_at'] > 300:
                gaps.append(p['label']+'：现场观察已超过5分钟，请重新获取')
        if content['conflicts']:
            gaps.append('技能仍有未解决的示范冲突')
        return gaps

    def record(self, instance, record):
        bounded(record, 4000)
        if not isinstance(record, dict) or record.get('step_id') not in {s['id'] for s in instance['recipe']['steps']}:
            raise ValueError('反馈必须对应本次任务步骤')
        if record.get('outcome') not in ('success', 'failed', 'unknown') or record.get('source') not in ('human', 'agent'):
            raise ValueError('请区分人工或Agent反馈，结果为 success / failed / unknown')
        if len(instance['records']) >= 100:
            raise ValueError('本次任务反馈已达100条')
        row = dict(id=uuid.uuid4().hex, task_id=instance['id'], skill_id=instance['skill_id'],
                   skill_version=instance['skill_version'], task_revision=instance['revision'],
                   binding_revision=instance['binding_revision'], bindings=copy.deepcopy(instance['bindings']),
                   values=copy.deepcopy(instance['values']), scene=copy.deepcopy(instance['scene']),
                   step_id=record['step_id'], outcome=record['outcome'], source=record['source'],
                   evidence=text(record.get('evidence'), '反馈依据', 2000), time=time.time())
        instance['records'].append(row)
        instance['revision'] += 1
        return self.write(instance, True)
