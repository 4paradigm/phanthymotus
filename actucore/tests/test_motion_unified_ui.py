"""Unified workbench: real sandbox/MessagePort/plugin with deterministic service responses."""
import base64
import io
from pathlib import Path
import sys

import pytest
from motion_ui_helpers import chromium_options

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.motion import MotionPlugin
from plugins.motion.client import MotionClient


@pytest.fixture
def unified_browser(tmp_path, monkeypatch):
    playwright = pytest.importorskip('playwright.sync_api')
    image = pytest.importorskip('PIL.Image')
    jpeg = io.BytesIO()
    image.new('RGB', (960, 540), (43, 55, 65)).save(jpeg, format='JPEG')
    encoded = base64.b64encode(jpeg.getvalue()).decode()
    profile = {'id': 'g1', 'label': 'G1', 'profile_hash': 'g1-hash', 'joint_names': ['knee']}
    walk = {'id': 'walk', 'name': '行走', 'profile_id': 'g1', 'profile_hash': 'g1-hash', 'sha256': 'walk-hash', 'frames': 100, 'fps': 25}
    wave = {**walk, 'id': 'wave', 'name': '挥手', 'sha256': 'wave-hash'}
    sample = {**walk, 'id': 'sample', 'name': '样例', 'sha256': 'sample-hash',
              'provenance': {'kind': 'integration_fixture', 'not_recorded_motion': True}}
    models = [dict(id='walk-old', name='SuperMimic old', profile_id='g1', profile_hash='g1-hash',
                   dataset_id='walk', reference_sha256='walk-hash', created_at=100, qualification='evaluated'),
              dict(id='walk-new', name='SuperMimic new', profile_id='g1', profile_hash='g1-hash',
                   dataset_id='walk', reference_sha256='walk-hash', created_at=200, qualification='unverified'),
              dict(id='wrong-reference', name='Wrong reference', profile_id='g1', profile_hash='g1-hash',
                   dataset_id='walk', reference_sha256='bad-hash', created_at=300),
              dict(id='wrong-profile', name='Wrong profile', profile_id='g1', profile_hash='bad-hash',
                   dataset_id='walk', reference_sha256='walk-hash', created_at=400)]
    state = {'calls': [], 'jobs': [], 'models': models, 'sessions': {}, 'manager_jobs': {},
             'saved': {'endpoint': 'http://motion:15732', 'api_key': 'motion-secret', 'environment_job': ''},
             'running': False, 'config_changes': [], 'errors': []}
    foreign = {'id': 'another-card', 'profile_id': 'g1', 'model_id': 'other', 'state': 'paused'}
    snapshot = {'time': 0.0, 'root_pose': {'pos': [0, 0, .8], 'quat': [1, 0, 0, 0]},
                'joint_names': ['knee'], 'joint_pos': [.1], 'contact_count': 0, 'bodies': [{'id': 1, 'name': 'pelvis'}]}
    image_data = {'mime': 'image/jpeg', 'width': 960, 'height': 540, 'data': encoded}

    rendered_images = {}

    def rendered_image(args):
        size = (args.get('width', 960), args.get('height', 540))
        if size not in rendered_images:
            output = io.BytesIO()
            image.new('RGB', size, (43, 55, 65)).save(output, format='JPEG')
            rendered_images[size] = {'mime': 'image/jpeg', 'width': size[0], 'height': size[1],
                                     'data': base64.b64encode(output.getvalue()).decode()}
        return rendered_images[size]

    def remote(client, action, **args):
        state['calls'].append((action, dict(args)))
        if client.endpoint.startswith('http://environment'):
            if action == 'targets':
                latest = state['manager_jobs'].get(state.get('latest_manager_job_id', 'install-B'))
                return {'targets': [{'id': 'gpu-server', 'label': 'GPU 服务器', 'host': 'gpu-server',
                    'capabilities': {'check': True, 'isolated': True, 'reuse': True},
                    'existing_service': {'endpoint': 'http://motion:15732'}, 'latest_job': latest}]}
            if action in {'check', 'install'}:
                job_id = 'check-B' if action == 'check' else 'install-B'
                job = {'id': job_id, 'target_id': 'gpu-server', 'action': action, 'mode': args.get('mode'),
                       'state': 'running', 'stage': 'checking', 'checks': []}
                state['manager_jobs'][job_id] = job
                return {'job': job}
            if action == 'status':
                if not args.get('job_id'):
                    if state.get('history_unavailable'):
                        raise RuntimeError('environment history is not supported')
                    return {'jobs': list(state['manager_jobs'].values())}
                return {'job': state['manager_jobs'][args['job_id']],
                        'events': [{'stage': 'checking', 'message': '真实状态来自测试服务'}] if not args.get('cursor') else [],
                        'next_cursor': 10}
            if action == 'cancel':
                state['manager_jobs'][args['job_id']]['state'] = 'cancelled'
                return {'job': state['manager_jobs'][args['job_id']]}
            if action == 'connection':
                job = state['manager_jobs'][args['job_id']]
                assert job['action'] == 'install' and job['state'] == 'succeeded'
                return {'connection': {'endpoint': 'http://motion:15732', 'api_key': 'backend-only-secret'}}
            raise AssertionError(action)
        if action == 'catalog':
            return {'profiles': [profile], 'datasets': [walk, wave, sample], 'models': state['models']}
        if action == 'resources':
            return {'gpus': [{'id': 2, 'name': 'A100', 'memory_total_mib': 81920, 'memory_used_mib': 1000,
                             'utilization_percent': 0, 'selectable': True, 'shared': True}],
                    'training': {'available': True, 'multi_gpu_supported': True, 'max_gpus': 8,
                                 'max_num_envs': 8192, 'max_iterations': 1000000}}
        if action == 'plan':
            return {'gpu_mode': args['gpu_mode'], 'ready': True, 'queueable': True, 'reason': '',
                    'gpu_ids': [2], 'world_size': 1, 'num_envs': args['num_envs'], 'envs_per_rank': args['num_envs']}
        if action == 'dataset_render':
            assert args['dataset_id'] in {'walk', 'wave', 'sample'}
            return {'available': True, 'dataset_id': args['dataset_id'], 'profile_id': 'g1',
                    'frame_index': args['offset'], 'frames': 100, 'fps': 25, 'time': args['offset'] / 25,
                    'duration_s': 4, 'camera': {'azimuth': 135, 'elevation': -20, 'distance': 3, 'lookat': [0, 0, .8]},
                    'image': rendered_image(args)}
        if action == 'jobs':
            return {'jobs': state['jobs']}
        if action == 'job':
            job = next(job for job in state['jobs'] if job['id'] == args['job_id'])
            return {'job': job, 'log': '', 'evaluation': job.get('evaluation')}
        if action == 'train':
            assert args['gpu_mode'] == 'auto' and 'gpu_ids' not in args
            job = {'id': 'new-training', 'kind': 'train', 'state': 'queued', 'dataset_id': args['dataset_id'],
                   'profile_id': args['profile_id'], 'created_at': 500, 'config': args}
            state['jobs'].append(job)
            return {'job': job}
        if action == 'simulation_list':
            return {'sessions': [foreign, *[value for value in state['sessions'].values() if value['state'] != 'closed']]}
        if action == 'simulation_create':
            assert args['model_id'] in {model['id'] for model in state['models']}
            session = {'id': f'owned-{len(state["sessions"])}', 'model_id': args['model_id'], 'profile_id': args['profile_id'],
                       'state': 'paused', 'control_mode': 'policy', 'engine': 'native_mujoco',
                       'recording': False, 'recorded_frames': 0,
                       'control': {'owner_view_id': args.get('view_id'), 'epoch': 1,
                                   'is_controller': True, 'lease_seconds': 20}}
            state['sessions'][session['id']] = session
            return {'session': dict(session), 'snapshot': dict(snapshot)}
        if action.startswith('simulation_'):
            assert args['session_id'] != foreign['id'], 'must not operate another card session'
            session = state['sessions'][args['session_id']]
            if action == 'simulation_run':
                session['state'] = 'running'
            elif action in {'simulation_pause', 'simulation_reset'}:
                session['state'] = 'paused'
            elif action == 'simulation_close':
                session['state'] = 'closed'
            elif action == 'simulation_record':
                session.update(recording=True, recorded_frames=2)
            elif action == 'simulation_record_stop':
                session['recording'] = False
            elif action == 'simulation_release':
                session['state'] = 'paused'
                session['control'].update(owner_view_id=None, is_controller=False)
            elif action == 'simulation_claim':
                session['control'].update(owner_view_id=args['view_id'], is_controller=True,
                                          epoch=session['control']['epoch'] + 1)
            session['control']['is_controller'] = session['control']['owner_view_id'] == args.get('view_id')
            result = {'session': dict(session), 'snapshot': dict(snapshot)}
            if action == 'simulation_render':
                result['image'] = image_data
            if action == 'simulation_replay':
                result['frames'] = [dict(snapshot)]
            return result
        raise AssertionError(action)

    monkeypatch.setattr(MotionClient, 'call', remote)
    monkeypatch.setattr(MotionClient, 'health', lambda client: {'state': 'ready'})
    plugin = MotionPlugin({'endpoint': 'http://motion:15732', 'api_key': 'motion-secret',
                           'environment_endpoint': 'http://environment:15734', 'environment_token': 'manager-secret',
                           'state_dir': str(tmp_path / 'state')})

    def host(message):
        args = message.get('args', {})
        if message['method'] == 'config':
            if 'values' in args:
                if state['running']:
                    raise RuntimeError('项目运行中不能修改配置')
                next_saved = {**state['saved'], **args['values']}
                result = plugin.dispatch('motion', {'action': 'config', **next_saved})
                if result.get('error'):
                    raise RuntimeError(result['error'])
                state['saved'] = next_saved
                state['config_changes'].append(args['values'])
            return {'values': {key: value for key, value in state['saved'].items() if key != 'api_key'},
                    'runtime_secret_set': {'api_key': True}}
        # Model the existing Core's replay of saved card configuration on every call.
        configured = plugin.dispatch('motion', {'action': 'config', **state['saved']})
        if configured.get('error'):
            return configured
        return plugin.dispatch('motion', args)

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(**chromium_options(), headless=True)
        page = browser.new_page(viewport={'width': 1000, 'height': 720})
        page.on('pageerror', lambda error: state['errors'].append(str(error)))
        page.expose_function('hostCall', host)
        page.route('http://localhost/workbench', lambda route: route.fulfill(body='<iframe sandbox="allow-scripts" style="width:958px;height:664px;border:0"></iframe>', content_type='text/html'))
        page.goto('http://localhost/workbench')

        def mount():
            page.evaluate('''html=>{const frame=document.querySelector('iframe');frame.onload=()=>{
              const channel=new MessageChannel();channel.port1.onmessage=async event=>{
                try{channel.port1.postMessage({id:event.data.id,result:await window.hostCall(event.data)});}
                catch(error){channel.port1.postMessage({id:event.data.id,error:error.message});}};
              frame.contentWindow.postMessage({protocol:'motus.workbench/1'},'*',[channel.port2]);};
              frame.srcdoc=`<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'">${html}`;
            }''', plugin.dispatch('motion', {'action': 'ui'})['html'])
            playwright.expect(page.frame_locator('iframe').locator('#status')).to_have_text('已连接')
        mount()
        yield page, page.frame_locator('iframe'), state, plugin, mount, playwright
        assert not state['errors']
        assert 'backend-only-secret' not in page.frame_locator('iframe').locator('body').inner_text()
        browser.close()


def test_reference_selection_existing_model_and_owned_simulation_restore(unified_browser, tmp_path):
    page, frame, state, plugin, mount, pw = unified_browser
    assert frame.locator('.breadcrumb button').all_text_contents() == ['动作']
    pw.expect(frame.locator('#referenceImage')).to_be_visible()
    pw.expect(frame.locator('#motionPrimary')).to_have_text('在仿真里试跑')
    assert frame.locator('#motionModel option').count() == 3  # placeholder plus two exact compatible versions
    pw.expect(frame.locator('#motionModel')).to_have_value('walk-new')
    pw.expect(frame.locator('#selectedModelState')).to_have_text('未检查')
    assert not any(action in {'train', 'simulation_create', 'simulation_run'} for action, _ in state['calls'])
    page.screenshot(path=str(tmp_path / 'unified-action.png'))
    frame.locator('#referencePlay').click()
    page.wait_for_timeout(850)
    frames = [args['offset'] for action, args in state['calls'] if action == 'dataset_render']
    assert any(offset > 0 for offset in frames) and len(frames) <= 7
    assert all('lookat' not in args for action, args in state['calls'] if action == 'dataset_render')
    frame.locator('#referencePlay').click()
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#simulation')).to_be_visible()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    frame.locator('#sim-run').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('运行中')
    pw.expect(frame.locator('#sim-renderedView')).to_be_visible()
    assert [args['model_id'] for action, args in state['calls'] if action == 'simulation_create'] == ['walk-new']
    assert not any(action == 'train' for action, _ in state['calls'])
    frame.locator('#sim-pause').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    page.screenshot(path=str(tmp_path / 'unified-simulation.png'))
    before = sum(action == 'simulation_create' for action, _ in state['calls'])
    mount()
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-renderedView')).to_be_visible()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    assert sum(action == 'simulation_create' for action, _ in state['calls']) == before
    assert all(args.get('session_id') != 'another-card' for action, args in state['calls'] if action.startswith('simulation_'))
    assert plugin._session.id == 'owned-0'


def test_missing_model_select_once_train_then_trial(unified_browser, tmp_path):
    page, frame, state, _, mount, pw = unified_browser
    frame.locator('#datasets .item').filter(has_text='挥手').click()
    pw.expect(frame.locator('#motionPrimary')).to_have_text('训练这个动作')
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#training')).to_be_visible()
    pw.expect(frame.locator('#trainProfile')).to_have_value('g1')
    pw.expect(frame.locator('#trainDataset')).to_have_value('wave')
    pw.expect(frame.locator('#numEnvs')).to_be_hidden()
    frame.locator('#train').click()
    pw.expect(frame.locator('#jobStatus')).to_have_text('排队中')
    request = next(args for action, args in state['calls'] if action == 'train')
    assert request['dataset_id'] == 'wave' and request['profile_id'] == 'g1' and request['gpu_mode'] == 'auto'
    model = {'id': 'wave-result', 'dataset_id': 'wave', 'profile_id': 'g1', 'profile_hash': 'g1-hash',
             'reference_sha256': 'wave-hash', 'created_at': 500, 'qualification': 'unverified'}
    state['models'].append(model)
    state['jobs'][-1].update(state='succeeded', model_id='wave-result')
    pw.expect(frame.locator('#jobTrial')).to_be_visible(timeout=10000)
    frame.locator('#jobTrial').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    frame.locator('#sim-run').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('运行中')
    assert next(args for action, args in state['calls'] if action == 'simulation_create')['model_id'] == 'wave-result'
    assert not any(action == 'evaluate' for action, _ in state['calls'])
    page.screenshot(path=str(tmp_path / 'unified-training-result.png'))
    frame.locator('#sim-pause').click()
    mount()
    pw.expect(frame.locator('#motionTitle')).to_have_text('挥手')
    pw.expect(frame.locator('#motionModel')).to_have_value('wave-result')
    assert sum(action == 'train' for action, _ in state['calls']) == 1
    assert sum(action == 'simulation_create' for action, _ in state['calls']) == 1


@pytest.mark.parametrize('saved_job', ['', 'install-A'])
def test_environment_host_config_survives_replay_without_exposing_credentials(unified_browser, saved_job):
    _, frame, state, plugin, _, pw = unified_browser
    state['manager_jobs'].update({job_id: {'id': job_id, 'target_id': 'gpu-server', 'action': 'install',
        'state': 'succeeded', 'stage': 'ready', 'checks': []} for job_id in ('install-A', 'install-B')})
    state['saved']['environment_job'] = saved_job
    frame.locator('#environmentButton').click()
    pw.expect(frame.locator('#environmentPrimary')).to_have_text('连接此环境')
    frame.locator('#environmentPrimary').click()
    pw.expect(frame.locator('#data')).to_be_visible()
    pw.expect(frame.locator('#datasetPreview')).to_be_visible()
    pw.expect(frame.locator('#referenceImage')).to_be_visible()
    assert state['config_changes'][-1] == {'environment_job': 'install-B'}
    frame.locator('#refresh').click()
    pw.expect(frame.locator('#status')).to_have_text('已刷新')
    assert plugin._environment_job == 'install-B' and state['saved']['environment_job'] == 'install-B'
    assert not any(action == 'env_connect' for action, _ in state['calls'])
    assert sum(action == 'connection' and args.get('job_id') == 'install-B' for action, args in state['calls']) == 1


def test_environment_checks_do_not_connect_or_install_without_click(unified_browser):
    _, frame, state, plugin, _, pw = unified_browser
    frame.locator('#environmentButton').click()
    frame.locator('#environmentPrimary').click()
    pw.expect(frame.locator('#environmentState')).to_have_text('运行中')
    pw.expect(frame.locator('#environmentPrimary')).to_be_disabled()
    state['manager_jobs']['check-B'].update(state='succeeded', stage='ready', checks=[{'name': 'gpu_runtime', 'state': 'passed', 'detail': 'CUDA OK'}])
    pw.expect(frame.locator('#environmentPrimary')).to_have_text('接入已有环境', timeout=7000)
    assert not any(action in {'install', 'connection'} for action, _ in state['calls'])
    frame.locator('#environmentPrimary').click()
    pw.expect(frame.locator('#environmentState')).to_have_text('运行中')
    assert next(args for action, args in state['calls'] if action == 'install')['mode'] == 'reuse'
    state['manager_jobs']['install-B'].update(state='succeeded', stage='ready')
    pw.expect(frame.locator('#environmentPrimary')).to_have_text('连接此环境', timeout=7000)
    state['running'] = True
    frame.locator('#environmentPrimary').click()
    pw.expect(frame.locator('#status')).to_have_text('项目运行中不能修改配置')
    assert plugin._environment_job == ''
    assert not any(action == 'connection' for action, _ in state['calls'])


def test_short_main_modal_keeps_primary_and_model_history_visible(unified_browser, tmp_path):
    page, frame, _, _, _, pw = unified_browser
    page.set_viewport_size({'width': 1000, 'height': 610})
    page.locator('iframe').evaluate("element=>element.style.height='520px'")
    pw.expect(frame.locator('#referenceImage')).to_be_visible()
    pw.expect(frame.locator('#selectedModelName')).not_to_contain_text('未检查')
    pw.expect(frame.locator('#selectedModelState')).to_have_text('未检查')
    assert frame.locator('body').evaluate('''()=>{
      const bounds=document.querySelector('main').getBoundingClientRect();
      return ['#motionPrimary','#referencePlay','#modelHistory>h4'].every(selector=>{
        const r=document.querySelector(selector).getBoundingClientRect();
        return r.height>12&&r.top>=bounds.top&&r.bottom<=bounds.bottom;
      });
    }''')
    pw.expect(frame.locator('#actionVersions')).to_be_visible()
    page.screenshot(path=str(tmp_path / 'short-main-modal.png'))


def test_policy_joint_limit_failure_is_clear_and_reset_stays_available(unified_browser):
    _, frame, state, _, _, pw = unified_browser
    frame.locator('#motionPrimary').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('已暂停')
    frame.locator('#sim-run').click()
    pw.expect(frame.locator('#sim-sessionState')).to_have_text('运行中')
    state['sessions']['owned-0'].update(state='error', error='policy target exceeds robot joint limits')
    frame.locator('#sim-refreshView').click()
    pw.expect(frame.locator('#sim-status')).to_have_text('模型输出超出关节限位，试跑已停止；请更换训练结果或重新训练。')
    pw.expect(frame.locator('#sim-reset')).to_be_enabled()
    pw.expect(frame.locator('#sim-run')).to_be_disabled()
    assert state['sessions']['owned-0']['error'] == 'policy target exceeds robot joint limits'
    frame.locator('#breadcrumbActions').click()
    pw.expect(frame.locator('#selectedModelState')).to_have_text('未检查')


@pytest.mark.parametrize('history_unavailable', [False, True])
def test_connection_dialog_reuses_verified_environments_and_only_switches_on_click(unified_browser, history_unavailable):
    _, frame, state, plugin, _, pw = unified_browser
    state['manager_jobs'].update({
        'install-A': {'id': 'install-A', 'target_id': 'gpu-server', 'action': 'install',
                      'state': 'succeeded', 'mode': 'reuse', 'created_at': 100},
        'install-B': {'id': 'install-B', 'target_id': 'gpu-server', 'action': 'install',
                      'state': 'succeeded', 'mode': 'isolated', 'created_at': 200},
        'check-new': {'id': 'check-new', 'target_id': 'gpu-server', 'action': 'check',
                      'state': 'succeeded', 'created_at': 300},
        'install-failed': {'id': 'install-failed', 'target_id': 'gpu-server', 'action': 'install',
                           'state': 'failed', 'created_at': 400}})
    state['saved']['environment_job'] = 'install-A'
    state['history_unavailable'] = history_unavailable
    state['latest_manager_job_id'] = 'install-B' if history_unavailable else 'check-new'
    frame.locator('#connection').click()
    pw.expect(frame.locator('#environmentRef')).to_have_value('install-A')
    pw.expect(frame.locator('#environmentRef option')).to_have_count(3)
    assert set(frame.locator('#environmentRef option').evaluate_all('nodes=>nodes.map(n=>n.value)')) == {'', 'install-A', 'install-B'}
    pw.expect(frame.locator('#connectionCurrent')).to_contain_text('GPU 服务器 · 已有环境')
    pw.expect(frame.locator('#environmentButton')).to_contain_text('GPU 服务器 · 已有环境')
    assert plugin._environment_job == 'install-A' and not state['config_changes']
    assert frame.locator('#apiKey').input_value() == ''
    assert 'backend-only-secret' not in frame.locator('body').inner_html()
    frame.locator('#environmentRef').select_option('install-B')
    assert plugin._environment_job == 'install-A'
    frame.locator('#useEnvironment').click()
    pw.expect(frame.locator('#settings')).to_be_hidden()
    pw.expect(frame.locator('#environmentButton')).to_contain_text('GPU 服务器 · 独立环境')
    assert state['config_changes'][-1] == {'environment_job': 'install-B'}
    assert state['saved']['environment_job'] == plugin._environment_job == 'install-B'
    assert frame.locator('#apiKey').input_value() == ''
    assert not any(action == 'env_connect' for action, _ in state['calls'])
