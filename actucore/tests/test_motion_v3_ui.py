"""V3 layouts and real-sample transport behavior in the sandboxed workbench."""
import base64
import io

import pytest

from test_motion_training_ui import training_browser
from test_motion_unified_ui import unified_browser


def inside(frame, selector):
    return frame.locator(selector).evaluate('''node => {
      const box=node.getBoundingClientRect();return box.width>0&&box.height>0&&
        box.left>=0&&box.top>=0&&box.right<=innerWidth+1&&box.bottom<=innerHeight+1;
    }''')


@pytest.mark.parametrize('height', [660, 517])
def test_default_monitor_sample_deduplication_replay_and_fixed_actions(training_browser, tmp_path, height):
    page, frame, state, expect_module = training_browser
    expect = expect_module.expect
    page.locator('iframe').evaluate('(iframe,height)=>{iframe.style.width="960px";iframe.style.height=height+"px"}', height)
    image = pytest.importorskip('PIL.Image')
    images = {}
    sources = [{'rank': 0, 'env_id': index} for index in range(8)]

    def preview(args):
        index = args.get('frame', 2)
        sample = args.get('sample')
        key = (args['width'], args['height'], index)
        if key not in images:
            output = io.BytesIO()
            image.new('RGB', key[:2], (30 + index * 10, 50, 60)).save(output, format='JPEG')
            images[key] = base64.b64encode(output.getvalue()).decode()
        shown = sources[args.get('sample_start', 0):args.get('sample_start', 0) + args.get('sample_limit', 8)] if sample is None else [sources[sample]]
        return {'available': True, 'frame_id': f'captured-{index}-sample-{sample}-group-{args.get("sample_start", 0)}',
                'captured_at': 1700000000 + index * 2, 'age_s': 2.5,
                'sample_hz': 4, 'actual_sample_hz': .5, 'render_ms': 4,
                'sample_count': 8, 'selected_sample': sample, 'sample_sources': sources,
                'total_envs': 64, 'world_size': 1, 'stale': False,
                'image': {'mime': 'image/jpeg', 'width': args['width'], 'height': args['height'], 'data': images[key]},
                'samples': [{**source, 'iteration': index, 'time': index / 50,
                             'captured_at': 1700000000 + index * 2} for source in shown],
                'playback': {'frame_count': 3, 'selected_frame': index,
                             'first_captured_at': 1700000000, 'last_captured_at': 1700000004,
                             'mode': 'replay' if 'frame' in args else 'live'}}

    frame.locator('body').evaluate('''()=>{
      window.imageDecodes=0;const original=Image.prototype.decode;
      Image.prototype.decode=function(){window.imageDecodes++;return original.call(this);};
    }''')
    state['preview'] = preview
    frame.locator('#trainingAdvanced > summary').click()
    assert inside(frame, '#train')
    page.screenshot(path=str(tmp_path / f'training-setup-{height}.png'))
    frame.locator('#train').click()
    state['jobs'][-1].update(state='running', progress={'iteration': 2, 'total_iterations': 24, 'elapsed_s': 6})
    expect(frame.locator('#trainingImage')).to_be_visible()
    expect(frame.locator('#watchTraining')).to_be_hidden()
    expect(frame.locator('#monitorOverlay span')).to_have_count(4 if height == 517 else 8)
    if height == 517:
        frame.locator('#monitorNextGroup').click()
        expect(frame.locator('#monitorOverlay')).to_contain_text('实例 4')
        frame.locator('#monitorNextGroup').click()
    expect(frame.locator('#monitorRate')).to_have_text('0.5 帧/秒')
    expect(frame.locator('#monitorAge')).to_have_text('画面采样 2.5 秒前')
    page.wait_for_timeout(900)
    decoded = frame.locator('body').evaluate('()=>window.imageDecodes')
    page.wait_for_timeout(600)
    assert frame.locator('body').evaluate('()=>window.imageDecodes') == decoded
    assert inside(frame, '#cancel') and inside(frame, '#monitorPause')
    assert inside(frame, '#trainingStage')
    assert frame.locator('#trainingStage').evaluate('node=>node.getBoundingClientRect().width') > 850
    page.screenshot(path=str(tmp_path / f'training-live-{height}.png'))
    frame.locator('#monitorPause').click()
    page.wait_for_timeout(100)
    calls = len([args for action, args in state['calls'] if action == 'training_preview'])
    page.wait_for_timeout(700)
    assert len([args for action, args in state['calls'] if action == 'training_preview']) == calls
    assert not any(action == 'cancel' for action, _ in state['calls'])
    frame.locator('#monitorSingle').click()
    expect(frame.locator('#monitorOverlay span')).to_have_count(1)
    frame.locator('#monitorSample').select_option('3')
    expect(frame.locator('#monitorOverlay')).to_have_text('进程 0 · 实例 3')
    frame.locator('#monitorHistory').click()
    frame.locator('#monitorFrame').evaluate('node=>{node.value="0";node.dispatchEvent(new Event("input"))}')
    expect(frame.locator('#previewStatus')).to_have_text('回放')
    expect(frame.locator('#monitorFrameLabel')).to_have_text('1 / 3')
    assert any(args.get('sample') == 3 and args.get('frame') == 0 for action, args in state['calls'] if action == 'training_preview')
    frame.locator('#monitorLive').click()
    expect(frame.locator('#previewStatus')).to_have_text('实时采样')
    frame.locator('#monitorGrid').click()
    expect(frame.locator('#monitorOverlay span')).to_have_count(4 if height == 517 else 8)
    state['jobs'][-1].update(state='succeeded', model_id='model-v3')
    state['models'] = [{'id': 'model-v3', 'profile_id': 'g1', 'profile_hash': 'profile-g1',
                        'reference_sha256': 'walk-sha', 'dataset_id': 'walk', 'qualification': 'unverified'}]
    expect(frame.locator('#jobStatus')).to_have_text('已完成', timeout=7000)
    expect(frame.locator('#previewStatus')).to_have_text('已结束', timeout=7000)
    assert inside(frame, '#jobTrial') and inside(frame, '#jobModel')
    expect(frame.locator('#cancel')).to_be_hidden()
    expect(frame.locator('#trainingOutcome')).to_have_text('本次训练结果 · 未检查')
    page.screenshot(path=str(tmp_path / f'training-result-{height}.png'))
    assert sum(action == 'train' for action, _ in state['calls']) == 1


@pytest.mark.parametrize('height', [660, 517])
def test_three_column_actions_reference_aspect_and_merged_model_details(unified_browser, tmp_path, height):
    page, frame, state, _, mount, expect_module = unified_browser
    expect = expect_module.expect
    page.locator('iframe').evaluate('(iframe,height)=>{iframe.style.width="960px";iframe.style.height=height+"px"}', height)
    expect(frame.locator('#referenceImage')).to_be_visible()
    page.wait_for_timeout(450)
    widths = frame.locator('.motion-layout').evaluate('node=>[...node.children].map(child=>child.getBoundingClientRect().width)')
    total = sum(widths)
    assert widths[0] >= 150 and widths[1] > 400 and widths[2] >= 230
    assert total <= 960
    frame_args = [args for action, args in state['calls'] if action == 'dataset_render'][-1]
    aspect = frame.locator('#datasetPreview').evaluate('node=>node.clientWidth/node.clientHeight')
    assert abs(frame_args['width'] / frame_args['height'] - aspect) < .02
    assert inside(frame, '#motionPrimary') and inside(frame, '#referencePlay')
    assert frame.locator('header').inner_text().count('动作工作台') == 0
    page.screenshot(path=str(tmp_path / f'actions-{height}.png'))
    frame.locator('#modelHistory').evaluate('node=>node.open=true')
    frame.locator('#openEvaluation').click()
    expect(frame.locator('#modelDetails')).to_be_visible()
    expect(frame.locator('#evaluation')).to_be_visible()
    expect(frame.locator('#publish')).to_be_visible()
    assert inside(frame, '#modelTrial') and inside(frame, '#export') and inside(frame, '#evaluate')
    assert not any(action in {'evaluate', 'simulation_create', 'train'} for action, _ in state['calls'])
    page.screenshot(path=str(tmp_path / f'model-details-{height}.png'))


def test_model_switch_drops_late_evaluation_from_previous_model(unified_browser):
    page, frame, state, _, _, pw = unified_browser
    state['jobs'].append({'id': 'old-check', 'kind': 'evaluate', 'state': 'running',
        'model_id': 'walk-old', 'profile_id': 'g1', 'dataset_id': 'walk',
        'evaluation': {'joint_rmse_rad': .01, 'qualification': 'evaluated'}})
    frame.locator('#refresh').click()
    frame.locator('#modelHistory').evaluate('node=>node.open=true')
    frame.locator('#actionVersions .version-row').last.click()
    page.evaluate('''()=>{
      const original=window.hostCall;window.hostCall=async message=>{
        const result=await original(message);
        if(message.args.mode==='job'&&message.args.job_id==='old-check'){
          document.body.dataset.oldCheckHeld='true';await new Promise(resolve=>window.releaseOldCheck=resolve);
        }return result;
      };
    }''')
    frame.locator('#openEvaluation').click()
    pw.expect(page.locator('body')).to_have_attribute('data-old-check-held', 'true')
    frame.locator('#modelVersions').locator('..').locator('summary').click()
    frame.locator('#modelVersions .item').first.click()
    pw.expect(frame.locator('#modelDetailsState')).to_have_text('未检查')
    page.evaluate('()=>window.releaseOldCheck()')
    page.wait_for_timeout(300)
    pw.expect(frame.locator('#evalResult')).to_have_text('尚未检查')
    pw.expect(frame.locator('#evalReplay')).to_be_hidden()
    assert not any(action in {'train', 'evaluate', 'simulation_create'} for action, _ in state['calls'])


@pytest.mark.parametrize('new_artifact', [False, True])
def test_model_details_does_not_wait_for_gpu_resource_refresh(unified_browser, new_artifact):
    page, frame, state, _, _, pw = unified_browser
    model_id = 'fresh-model' if new_artifact else 'walk-new'
    state['jobs'].append({'id': 'completed-training', 'kind': 'train', 'state': 'succeeded',
                         'model_id': model_id, 'profile_id': 'g1', 'dataset_id': 'walk'})
    frame.locator('#refresh').click()
    if new_artifact:
        frame.locator('#actionVersions .version-row').filter(has_text='已完成').first.click()
    else:
        frame.locator('#openTraining').click()
    pw.expect(frame.locator('#jobModel')).to_be_visible()
    if new_artifact:
        state['models'].append({'id': model_id, 'profile_id': 'g1', 'profile_hash': 'g1-hash',
                                'dataset_id': 'walk', 'reference_sha256': 'walk-hash'})
    page.evaluate('''()=>{
      const original=window.hostCall;window.hostCall=async message=>{
        if(message.args.mode==='resources'){
          document.body.dataset.resourceRefreshHeld='true';
          await new Promise(resolve=>window.releaseResourceRefresh=resolve);
        }return original(message);
      };
    }''')
    before = len(state['calls'])
    frame.locator('#jobModel').click()
    pw.expect(frame.locator('#modelDetails')).to_be_visible(timeout=1500)
    pw.expect(frame.locator('#exportModel')).to_have_value(model_id)
    calls = state['calls'][before:]
    assert sum(action == 'catalog' for action, _ in calls) == int(new_artifact)
    assert not any(action in {'resources', 'jobs', 'plan'} for action, _ in calls)
    assert page.locator('body').get_attribute('data-resource-refresh-held') is None


def test_old_version_returns_to_its_own_training_after_evaluation(unified_browser):
    _, frame, state, _, _, pw = unified_browser
    state['jobs'].extend([
        {'id': 'older-training', 'kind': 'train', 'state': 'succeeded', 'created_at': 100,
         'model_id': 'walk-old', 'profile_id': 'g1', 'dataset_id': 'walk'},
        {'id': 'latest-training', 'kind': 'train', 'state': 'succeeded', 'created_at': 900,
         'model_id': 'walk-new', 'profile_id': 'g1', 'dataset_id': 'walk'},
        {'id': 'old-evaluation', 'kind': 'evaluate', 'state': 'succeeded', 'created_at': 1000,
         'model_id': 'walk-old', 'profile_id': 'g1', 'dataset_id': 'walk'}])
    frame.locator('#refresh').click()
    frame.locator('#actionVersions .version-row').last.click()
    frame.locator('#openEvaluation').click()
    pw.expect(frame.locator('#modelBack')).to_have_text('返回此版本训练')
    pw.expect(frame.locator('#modelDetailsTitle')).to_contain_text('版本 1')
    before = len(state['calls'])
    frame.locator('#modelBack').click()
    pw.expect(frame.locator('#training')).to_be_visible()
    pw.expect(frame.locator('#jobInfo')).to_contain_text('older-training')
    requested = [args['job_id'] for action, args in state['calls'][before:] if action == 'job']
    assert requested and set(requested) == {'older-training'}


@pytest.mark.parametrize('height', [660, 517])
def test_many_versions_scroll_without_clipping_action_buttons(unified_browser, tmp_path, height):
    page, frame, state, _, _, pw = unified_browser
    for index in range(6):
        state['models'].append({'id': f'additional-{index}', 'profile_id': 'g1',
            'profile_hash': 'g1-hash', 'dataset_id': 'walk', 'reference_sha256': 'walk-hash',
            'created_at': 300 + index, 'qualification': 'unverified'})
    page.locator('iframe').evaluate('(node,height)=>node.style.height=height+"px"', height)
    frame.locator('#refresh').click()
    pw.expect(frame.locator('#actionVersions .version-row')).to_have_count(8)
    assert frame.locator('#actionVersions').evaluate('node=>node.scrollHeight>node.clientHeight')
    assert frame.locator('#actionVersions').evaluate('node=>{const chosen=node.querySelector(".selected");const a=chosen.getBoundingClientRect(),b=node.getBoundingClientRect();return a.top>=b.top-1&&a.bottom<=b.bottom+1;}')
    for selector in ('#retrain', '#openTraining', '#openEvaluation', '#motionPrimary'):
        assert inside(frame, selector)
        assert frame.locator(selector).evaluate('''node=>{
          const r=node.getBoundingClientRect();
          return document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===node;
        }''')
    assert frame.locator('.action-rail h3').evaluate('node=>node.getBoundingClientRect().left>=8')
    assert '动作学习' not in frame.locator('header').inner_text()
    page.screenshot(path=str(tmp_path / f'many-versions-{height}.png'))
    frame.locator('#retrain').click()
    pw.expect(frame.locator('.train-setup')).to_be_visible()
    pw.expect(frame.locator('.training-layout .action-rail')).to_be_visible()
