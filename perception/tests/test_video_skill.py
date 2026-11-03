"""Task-skill contracts with real persistence/media and explicit model doubles."""
import copy
import json
import time

import pytest

from test_video_task import ffmpeg, video, plugin, model_server, call, completed, imported, recipe
from plugins.video_task import skill_recipe


def definition(source=None):
    return dict(title='移动物体', goal='将{{object}}移动到{{target}}', summary='移动对象并核对目标位置',
        applicability=['可夹持且路径畅通'], exclusions=['固定在桌面的物体'], uncertainties=['未验证实机'], conflicts=[],
        parameters=[dict(id='object', label='操作对象', type='object', required=True),
                    dict(id='target', label='放置区域', type='region', required=True)],
        steps=[dict(id='s1', title='将{{object}}移动到{{target}}', preconditions=['路径畅通'],
                    postconditions=['{{object}}位于{{target}}'], capabilities=['移动', '停止', '完成反馈'],
                    depends_on=[], failure='停止并报告，禁止自动重试', source_refs=[source+':s1'] if source else [])])


def adopted(plugin, video):
    d = imported(plugin, video)
    return call(plugin, 'update', draft_id=d['id'], expected_revision=d['revision'], recipe=recipe())['draft']


def extracted(plugin, video, model_server):
    d = adopted(plugin, video)
    source = skill_recipe.source_snapshot(d)
    model_server['recipe'] = definition(source['source_id'])
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-test-only')
    call(plugin, 'skill_extract', draft_ids=[d['id']])
    completed(plugin)
    return d, call(plugin, 'skill_inspect', **plugin.job['result'])


def test_extract_provenance_idempotent_export_and_legacy(plugin, video, model_server):
    d, result = extracted(plugin, video, model_server)
    aid = result['asset']['id']
    v = result['version']
    assert v['sources'][0]['recipe_revision'] == d['recipe_revision']
    assert v['sources'][0]['media_sha256'] == [d['media']['sha256']]
    assert v['content']['steps'][0]['source_refs']
    assert 'start_s' not in json.dumps(v['content'])
    req = model_server['requests'][-1]
    assert len(req['messages'][1]['content']) == 1
    assert req['messages'][1]['content'][0]['type'] == 'text'
    exported = call(plugin, 'save', skill_id=aid, skill_version=1, confirmed=True)['skill']
    assert plugin.skills.read(aid)['versions'][0]['export'] == exported
    assert exported == call(plugin, 'save', skill_id=aid, skill_version=1, confirmed=True)['skill']
    assert 'task_create' in exported['instruction']
    assert 'start_s' not in exported['instruction']
    assert 'confirmed' in plugin.dispatch('video_task', dict(action='save', skill_id=aid))['error'] or '核对' in plugin.dispatch('video_task', dict(action='save', skill_id=aid))['error']
    renamed = call(plugin, 'rename', draft_id=d['id'], expected_revision=d['revision'], name='改名')['draft']
    call(plugin, 'skill_extract', draft_ids=[renamed['id']])
    completed(plugin)
    assert plugin.job['result']['skill_id'] == aid
    assert len(call(plugin, 'skill_list')['skills']) == 1
    assert call(plugin, 'save', draft_id=d['id'])['skill']['slug'].startswith('video-')


def test_duplicate_video_and_multiview_do_not_count_as_independent(plugin, video, model_server):
    d, result = extracted(plugin, video, model_server)
    other = adopted(plugin, video)
    call(plugin, 'skill_extract', draft_ids=[d['id'], other['id'], d['id']])
    completed(plugin)
    result = call(plugin, 'skill_inspect', **plugin.job['result'])
    assert len(result['version']['sources']) == 1
    sources = copy.deepcopy(result['version']['sources'])
    different = copy.deepcopy(sources[0])
    different.update(draft_id='a'*32, media_sha256=['b'*64], source_id='different')
    assert len(skill_recipe.independent(sources+[different])) == 2


def test_revisions_frozen_conflicts_reviews_and_old_export(plugin, video, model_server):
    _, r = extracted(plugin, video, model_server)
    aid = r['asset']['id']
    original = call(plugin, 'save', skill_id=aid, skill_version=1, confirmed=True)['skill']
    content = copy.deepcopy(r['version']['content'])
    content['exclusions'].append('螺纹盖不能用拔盖方式')
    r = call(plugin, 'skill_edit', skill_id=aid, skill_version=1, expected_revision=1, skill=content)
    assert r['version']['version'] == 2 and r['version']['changes']
    assert call(plugin, 'save', skill_id=aid, skill_version=1, confirmed=True)['skill'] == original
    assert '已更新' in plugin.dispatch('video_task', dict(action='skill_edit', skill_id=aid,
        skill_version=1, expected_revision=1, skill=content))['error']
    cases = [dict(case_id=s['source_id'], verdict='unknown', note='缺少反馈') for s in r['version']['sources']]
    assert '不能设为推荐' in plugin.dispatch('video_task', dict(action='skill_review', skill_id=aid,
        skill_version=2, expected_revision=2, cases=cases, recommend=True))['error']
    cases[0].update(verdict='pass', note='人工核对适用范围；不是实机完成证据')
    r = call(plugin, 'skill_review', skill_id=aid, skill_version=2, expected_revision=2, cases=cases, recommend=True)
    assert r['asset']['recommended'] == 2 and r['review']['source'] == 'human'


@pytest.mark.parametrize('damage', ['pose', 'time', 'bad_ref', 'template', 'cycle', 'default'])
def test_reject_physical_copy_invalid_source_and_code_templates(plugin, video, damage):
    source = skill_recipe.source_snapshot(adopted(plugin, video))
    c = definition(source['source_id'])
    if damage == 'pose': c['steps'][0]['pose'] = [0, 0, 0]
    if damage == 'time': c['steps'][0]['start_s'] = 1
    if damage == 'bad_ref': c['steps'][0]['source_refs'] = ['forged:s1']
    if damage == 'template': c['goal'] = '{{object.__class__}}'
    if damage == 'cycle': c['steps'][0]['depends_on'] = ['s1']
    if damage == 'default': c['parameters'][0]['default'] = 'old demonstration object'
    with pytest.raises(ValueError):
        skill_recipe.validate(c, [source])


@pytest.mark.parametrize('tool_name', ['mcp__arm_a__move', 'mcp__generic_b__act'])
def test_video_free_instance_preflight_and_feedback(plugin, video, model_server, tool_name):
    _, r = extracted(plugin, video, model_server)
    aid = r['asset']['id']
    task = call(plugin, 'task_create', skill_id=aid, skill_version=1,
                values={'object': '蓝色杯子', 'target': '右侧托盘'})['task']
    assert '蓝色杯子' in task['recipe']['steps'][0]['title']
    assert 'media' not in task and 'start_s' not in json.dumps(task['recipe'])
    assert task['scene'] == task['bindings'] == {}
    check = call(plugin, 'task_check', task_id=task['id'], tools=[])
    assert len(check['gaps']) == 2 and not check['ready_for_review']
    assert '缺口' in plugin.dispatch('video_task', dict(action='task_agent', task_id=task['id'], expected_revision=0, tools=[]))['error']
    tools = [dict(name=tool_name, instance_id='arm', inputSchema={'type': 'object',
                 'properties': {'action': {'type': 'string', 'enum': ['move', 'stop']}}, 'required': ['action']})]
    binding = {'s1': dict(tool=tool_name, arguments={'action': 'move'}, stop_tool=tool_name,
                         stop_arguments={'action': 'stop'}, completion='根据实际位置传感器反馈核对')}
    scene = {p: dict(position=[.1, .2, .3], unit='m', source='human', observed_at=time.time(),
                     coordinate_frame='robot_world', evidence='人工现场测量') for p in ('object', 'target')}
    task = call(plugin, 'task_update', task_id=task['id'], expected_revision=0, scene=scene, bindings=binding)['task']
    assert call(plugin, 'task_check', task_id=task['id'], tools=tools)['ready_for_review']
    scene['object']['observed_at'] -= 600
    task = call(plugin, 'task_update', task_id=task['id'], expected_revision=task['revision'], scene=scene)['task']
    assert not task['bindings']
    assert '超过5分钟' in ' '.join(call(plugin, 'task_check', task_id=task['id'], tools=tools)['gaps'])
    task = call(plugin, 'task_record', task_id=task['id'], expected_revision=task['revision'],
                record=dict(step_id='s1', source='human', outcome='unknown', evidence='人工观察不足，未实际执行'))['task']
    record = task['records'][0]
    assert record['skill_version'] == 1 and record['values']['object'] == '蓝色杯子'
    assert record['source'] == 'human' and record['outcome'] == 'unknown'
    model_server['recipe'] = definition(r['version']['sources'][0]['source_id'])
    model_server['recipe']['uncertainties'].append('必须补充现场位置反馈')
    call(plugin, 'skill_improve', skill_id=aid, skill_version=1, expected_revision=r['asset']['revision'], task_ids=[task['id']])
    completed(plugin)
    updated = call(plugin, 'skill_inspect', **plugin.job['result'])
    assert updated['version']['version'] == 2 and updated['asset']['recommended'] is None
    assert updated['version']['feedback'][0]['id'] == record['id']


def test_compare_counterexample_and_no_automatic_overwrite(plugin, video, model_server):
    _, r = extracted(plugin, video, model_server)
    aid = r['asset']['id']
    model_server['recipe'] = dict(slug='screw-lid', decision='separate', reason='旋转和拔出动作不同',
        shared=['打开容器'], differences=['螺纹盖需要旋转'], conflicts=['拔出可能损坏螺纹'])
    call(plugin, 'skill_compare', skill_id=aid, skill_version=1, expected_revision=1,
         installed=[dict(slug='screw-lid', name='旋盖', instruction='旋转打开螺纹盖')])
    completed(plugin)
    r = call(plugin, 'skill_inspect', skill_id=aid, skill_version=1)
    assert r['comparison']['decision'] == 'separate'
    assert len(r['history']) == 1
    model_server['recipe'].pop('slug')
    call(plugin, 'skill_compare', skill_id=aid, skill_version=1, expected_revision=r['asset']['revision'],
         installed=[dict(slug='screw-lid', name='旋盖', instruction='旋转打开螺纹盖')])
    completed(plugin)
    r = call(plugin, 'skill_inspect', skill_id=aid, skill_version=1)
    assert r['comparison']['slug'] == 'screw-lid'
    model_server['recipe']['slug'] = 'not-selected'
    call(plugin, 'skill_compare', skill_id=aid, skill_version=1, expected_revision=r['asset']['revision'],
         installed=[dict(slug='screw-lid', name='旋盖', instruction='旋转打开螺纹盖')])
    plugin.worker.join(15)
    assert plugin.job['state'] == 'failed'
    assert call(plugin, 'skill_inspect', skill_id=aid, skill_version=1)['comparison']['slug'] == 'screw-lid'


def test_cancel_late_skill_result_leaves_no_asset(plugin, video, model_server):
    d = adopted(plugin, video)
    model_server.update(recipe=definition(), wait=True)
    call(plugin, 'config', endpoint=model_server['endpoint'], model='contract-test-only')
    call(plugin, 'skill_extract', draft_ids=[d['id']])
    assert model_server['entered'].wait(5)
    call(plugin, 'cancel')
    model_server['release'].set()
    plugin.worker.join(10)
    assert plugin.job['state'] == 'cancelled'
    assert call(plugin, 'skill_list')['skills'] == []


def test_multiple_sources_supplement_and_stale_guard(plugin, video, model_server):
    import cv2
    import numpy as np
    first, original = extracted(plugin, video, model_server)
    other_video = video.parent/'second.mp4'
    writer = cv2.VideoWriter(str(other_video), cv2.VideoWriter_fourcc(*'mp4v'), 12, (320, 240))
    assert writer.isOpened()
    for i in range(48):
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(frame, (40+i*3, 120), 14, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    second = adopted(plugin, other_video)
    s2 = skill_recipe.source_snapshot(second)
    model_server['recipe'] = definition(s2['source_id'])
    call(plugin, 'skill_extract', draft_ids=[second['id']]);completed(plugin)
    candidate = call(plugin, 'skill_inspect', **plugin.job['result'])
    assert candidate['asset']['id'] != original['asset']['id']
    model_server['recipe']['steps'][0]['source_refs'].append(original['version']['sources'][0]['source_id']+':s1')
    model_server['recipe']['uncertainties'].append('两段合成测试素材，非真实泛化证据')
    call(plugin, 'skill_supplement', skill_id=candidate['asset']['id'], skill_version=1,
         target_skill_id=original['asset']['id'], target_version=1, expected_revision=1)
    completed(plugin)
    result = call(plugin, 'skill_inspect', **plugin.job['result'])
    assert result['asset']['id']==original['asset']['id']
    assert len(result['version']['sources'])==2 and result['version']['version']==2
    assert result['asset']['recommended'] is None
    # Model result cannot overwrite a newer saved candidate, even with an out-of-process edit.
    model_server.update(wait=True)
    model_server['entered'].clear();model_server['release'].clear()
    call(plugin, 'skill_compare', skill_id=result['asset']['id'], skill_version=2, expected_revision=2,
         installed=[dict(slug='reference',name='Reference',instruction='Reference instruction')])
    assert model_server['entered'].wait(5)
    with plugin.lock:
        asset=plugin.skills.read(result['asset']['id']);asset['revision']+=1;plugin.skills.write(asset)
    model_server['release'].set();plugin.worker.join(10)
    assert plugin.job['state']=='failed' and '已更新' in plugin.job['error']
    assert call(plugin, 'skill_inspect', skill_id=result['asset']['id'], skill_version=2)['comparison'] is None


def test_numeric_parameters_missing_reference_and_reextract_family(plugin, video, model_server):
    d, result = extracted(plugin, video, model_server)
    c = copy.deepcopy(result['version']['content'])
    c['parameters'].append(dict(id='count',label='次数',type='number',required=False,min=1,max=3,unit='次'))
    c['goal'] += '，共{{count}}次'
    model_server['recipe'] = c
    call(plugin, 'skill_extract', draft_ids=[d['id']]);completed(plugin)
    updated = call(plugin, 'skill_inspect', **plugin.job['result'])
    assert updated['asset']['id']==result['asset']['id'] and updated['version']['version']==2
    args=dict(skill_id=result['asset']['id'], skill_version=2)
    for invalid in (True, float('nan'), 4, '2'):
        assert 'error' in plugin.dispatch('video_task', dict(action='task_create', **args,
            values={'object':'new','target':'new place','count':invalid}))
    task=call(plugin,'task_create',**args,values={'object':'new','target':'new place'})['task']
    assert '次数' in task['missing']
    assert call(plugin,'task_check',task_id=task['id'],tools=[])['ready_for_review'] is False
    assert 'error' in plugin.dispatch('video_task', dict(action='task_record', task_id=task['id'],expected_revision=-1,
        record=dict(step_id='s1',source='human',outcome='unknown',evidence='stale')))
