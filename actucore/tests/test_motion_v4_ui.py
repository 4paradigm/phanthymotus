"""Sandbox UI contracts: explicit ownership, responsive grouping and shared artifacts."""
import base64
import io
from pathlib import Path

import pytest
from motion_ui_helpers import chromium_options

from tools.build_motion_workbenches import bundle


@pytest.mark.parametrize('height', [660, 517])
def test_simulation_overview_observe_takeover_and_recording(tmp_path, height):
    pw = pytest.importorskip('playwright.sync_api')
    image = pytest.importorskip('PIL.Image')
    session = {'id': 'learning-trial', 'profile_id': 'g1', 'model_id': 'trained-version',
               'state': 'paused', 'engine': 'native_mujoco', 'control_mode': 'policy',
               'recorded_frames': 0, 'recording': False,
               'control': {'owner_view_id': 'learning-window', 'epoch': 3, 'lease_seconds': 20}}
    snapshot = {'time': 0.0, 'root_pose': {'pos': [0, 0, .8]}, 'joint_names': ['knee'],
                'joint_pos': [0.0], 'bodies': [{'id': 1, 'name': 'pelvis'}]}
    calls, errors = [], []

    def host(message):
        args = message['args']
        action = args['action']
        calls.append((action, dict(args)))
        if action == 'info':
            return {'state': 'ready'}
        if action == 'catalog':
            return {'profiles': [{'id': 'g1', 'label': 'G1'}],
                    'datasets': [{'id': 'walk', 'name': '行走'}],
                    'models': [{'id': 'trained-version', 'dataset_id': 'walk', 'profile_id': 'g1'}]}
        control = {**session['control'], 'is_controller': session['control']['owner_view_id'] == args.get('view_id')}
        if action == 'simulation_list':
            return {'sessions': [{**session, 'control': control}]}
        if action == 'simulation_claim':
            assert args['control_epoch'] == 3 and args['takeover'] is True
            session['control'].update(owner_view_id=args['view_id'], epoch=4)
        elif action == 'simulation_release':
            assert args['control_epoch'] == 4
            session['state'] = 'paused'
            session['control']['owner_view_id'] = None
        elif action == 'simulation_run':
            assert args['control_epoch'] == 4
            session['state'] = 'running'
        elif action == 'simulation_record':
            session.update(recording=True, recorded_frames=2)
        elif action == 'simulation_record_stop':
            session['recording'] = False
        elif action == 'simulation_detach':
            return {'state': 'ready'}
        control = {**session['control'], 'is_controller': session['control']['owner_view_id'] == args.get('view_id')}
        result = {'session': {**session, 'control': control}, 'snapshot': dict(snapshot)}
        if action == 'inspect' and args['mode'] == 'frame':
            output = io.BytesIO()
            image.new('RGB', (args['width'], args['height']), (28, 38, 51)).save(output, format='JPEG')
            result['image'] = {'mime': 'image/jpeg', 'width': args['width'], 'height': args['height'],
                               'data': base64.b64encode(output.getvalue()).decode()}
        return result

    with pw.sync_playwright() as playwright:
        browser = playwright.chromium.launch(**chromium_options(), headless=True)
        page = browser.new_page(viewport={'width': 1000, 'height': height + 40})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.expose_function('hostCall', host)
        page.route('http://localhost/workbench', lambda route: route.fulfill(body=f'<iframe sandbox="allow-scripts" style="width:960px;height:{height}px;border:0"></iframe>', content_type='text/html'))
        page.goto('http://localhost/workbench')
        page.evaluate('''html=>{const frame=document.querySelector('iframe');frame.onload=()=>{
          const channel=new MessageChannel();channel.port1.onmessage=async event=>{try{
            channel.port1.postMessage({id:event.data.id,result:await window.hostCall(event.data)});
          }catch(error){channel.port1.postMessage({id:event.data.id,error:error.message});}};
          frame.contentWindow.postMessage({protocol:'motus.workbench/1'},'*',[channel.port2]);};frame.srcdoc=html;}''', bundle('simulation.html'))
        frame = page.frame_locator('iframe')
        pw.expect(frame.locator('.session-row')).to_have_count(1)
        page.screenshot(path=str(tmp_path / 'session-overview.png'))
        frame.locator('.session-row button').click()
        pw.expect(frame.locator('#renderedView')).to_be_visible()
        pw.expect(frame.locator('#controlState')).to_have_text('观看中')
        pw.expect(frame.locator('#run')).to_be_disabled()
        assert not any(action in {'simulation_run', 'simulation_claim'} for action, _ in calls)
        frame.locator('#claim').click()
        pw.expect(frame.locator('#controlState')).to_have_text('控制中')
        frame.locator('#run').click()
        pw.expect(frame.locator('#pause')).to_be_enabled()
        frame.locator('[data-debug="recording"]').click()
        frame.locator('#record').click()
        pw.expect(frame.locator('#recordStop')).to_be_enabled()
        frame.locator('#recordStop').click()
        pw.expect(frame.locator('#replay')).to_be_enabled()
        assert frame.locator('#viewer').evaluate('node=>{const r=node.getBoundingClientRect();return r.height>=150&&r.bottom<=innerHeight;}')
        assert frame.locator('#renderedView').evaluate('node=>getComputedStyle(node).objectFit') == 'contain'
        page.screenshot(path=str(tmp_path / 'session-debug.png'))
        frame.locator('#backSessions').click()
        pw.expect(frame.locator('#sessionsView')).to_be_visible()
        assert session['control']['owner_view_id'] is None and session['state'] == 'paused'
        assert not errors
        browser.close()


def test_both_bundles_embed_identical_simulation_source():
    source = (Path(__file__).parents[1] / 'plugins/motion/ui/simulation.js').read_text()
    for name in ('motion.html', 'simulation.html'):
        artifact = bundle(name)
        assert source in artifact
        assert len(artifact) < 250_000
        assert '@include' not in artifact
        assert 'object-fit:cover' not in artifact


@pytest.mark.parametrize('catalog_fails', [False, True])
def test_simulation_create_waits_for_catalog_and_keeps_startup_failure(catalog_fails):
    pw = pytest.importorskip('playwright.sync_api')
    calls = []
    session = {}

    def host(message):
        args = message['args']
        action = args['action']
        calls.append((action, dict(args)))
        if action == 'info':
            return {'state': 'ready'}
        if action == 'catalog':
            if catalog_fails:
                raise RuntimeError('无法读取机器人配置')
            return {'profiles': [{'id': 'g1', 'label': 'G1'}], 'models': [], 'datasets': []}
        if action == 'simulation_list':
            return {'sessions': []}
        if action == 'simulation_create':
            assert args['profile_id'] == 'g1'
            session.update(id='new-scene', profile_id='g1', state='paused', engine='native_mujoco',
                control_mode='pd_hold', control={'owner_view_id': args['view_id'], 'epoch': 1,
                                               'is_controller': True, 'lease_seconds': 20})
        return {'session': dict(session), 'snapshot': {'time': 0.0, 'root_pose': {'pos': [0, 0, .8]}}}

    with pw.sync_playwright() as playwright:
        browser = playwright.chromium.launch(**chromium_options(), headless=True)
        page = browser.new_page(viewport={'width': 1000, 'height': 720})
        page.expose_function('hostCall', host)
        page.route('http://localhost/workbench', lambda route: route.fulfill(
            body='<iframe sandbox="allow-scripts" style="width:960px;height:660px;border:0"></iframe>', content_type='text/html'))
        page.goto('http://localhost/workbench')
        page.evaluate('''html=>{
          window.catalogGate=new Promise(resolve=>window.releaseCatalog=resolve);
          const frame=document.querySelector('iframe');frame.onload=()=>{
            const channel=new MessageChannel();channel.port1.onmessage=async event=>{try{
              if(event.data.args.action==='catalog'){document.body.dataset.catalogPending='true';await window.catalogGate;}
              channel.port1.postMessage({id:event.data.id,result:await window.hostCall(event.data)});
            }catch(error){channel.port1.postMessage({id:event.data.id,error:error.message});}};
            frame.contentWindow.postMessage({protocol:'motus.workbench/1'},'*',[channel.port2]);};frame.srcdoc=html;
        }''', bundle('simulation.html'))
        frame = page.frame_locator('iframe')
        pw.expect(page.locator('body')).to_have_attribute('data-catalog-pending', 'true')
        pw.expect(frame.locator('#create')).to_be_disabled()
        bounds = frame.locator('#create').bounding_box()
        page.mouse.click(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
        assert not any(action == 'simulation_create' for action, _ in calls)
        page.evaluate('()=>window.releaseCatalog()')
        if catalog_fails:
            pw.expect(frame.locator('#appStatus')).to_have_text('无法读取机器人配置')
            pw.expect(frame.locator('#create')).to_be_disabled()
            page.wait_for_timeout(300)
            pw.expect(frame.locator('#appStatus')).to_have_text('无法读取机器人配置')
            assert not any(action == 'simulation_create' for action, _ in calls)
        else:
            pw.expect(frame.locator('#profile')).to_have_value('g1')
            pw.expect(frame.locator('#create')).to_be_enabled()
            frame.locator('#create').click()
            pw.expect(frame.locator('#debugView')).to_be_visible()
            assert len([action for action, _ in calls if action == 'simulation_create']) == 1
        browser.close()
