"""Browser → existing Core routes → real Perception HTTP → codec → skill import.

ROS is stubbed in a separate process. No model or robot is simulated as real.
Optional host dependencies: opencv-python-headless<5, imageio-ffmpeg, playwright,
uvicorn. The actual Canvas case also loads Core's dependencies, including
aiofiles and cryptography. Run: python -m pytest agent-core/tests/test_video_workbench.py -q
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import fastapi
from fastapi.testclient import TestClient
import pytest

import config
from api import mcp_manage, skills

ROOT = Path(__file__).resolve().parents[2]


CONFIG_PROBE = """window.addEventListener('message', event => {
    if (!event.ports[0]) return;
    const port = event.ports[0], pending = new Map(); let seq = 0;
    port.addEventListener('message', event => {
        const request = pending.get(event.data.id); if (!request) return;
        pending.delete(event.data.id); clearTimeout(request.timeout);
        event.data.error ? request.reject(Error(event.data.error)) : request.resolve(event.data.result);
    });
    window.hostProbe = (method,args={}) => new Promise((resolve, reject) => {
        const id = 'config-probe-' + (++seq), timeout = setTimeout(() => reject(Error('probe timeout')), 10000);
        pending.set(id, {resolve, reject, timeout}); port.postMessage({id, method, args});
    });
    window.configProbe = args => window.hostProbe('config',args);
});"""


def catalog_fixture(page):
    """Explicit browser double; public catalogue HTTP is tested in Perception."""
    def route(request):
        body = request.request.post_data_json
        if (body or {}).get('arguments', {}).get('action') != 'model_catalog':
            request.continue_(); return
        result = {'models': ['PR-G-Flash-3.8', 'fixture-gemini'], 'provider': 'gemini',
                  'source': 'fixture://catalog', 'note': 'Explicit browser fixture'}
        request.fulfill(json={'code': 200, 'data': [{'type': 'text', 'text': json.dumps(result)}]})
    page.route('**/api/mcp/local-video/call*', route)


def screenshot(page, **kwargs):
    # Let ordinary notifications expire so screenshots describe the current page,
    # not an earlier permission-denial check. Do not alter the rendered UI.
    page.locator('#canvas-toast').wait_for(state='detached', timeout=6000)
    page.screenshot(**kwargs)


def test_capture_card_in_original_canvas_and_handoff(stack, tmp_path, monkeypatch):
    """Real Canvas/iframe/RPC/codecs, synthetic ROS JPEG+state; no model/hardware."""
    import base64
    import cv2
    sync = pytest.importorskip('playwright.sync_api')
    uvicorn = pytest.importorskip('uvicorn')
    _, video, registry = stack
    monkeypatch.chdir(tmp_path)
    (tmp_path/'web').symlink_to(ROOT/'agent-core/web', target_is_directory=True)
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path/'capture-canvas.db'))
    config._seed_defaults()
    monkeypatch.setattr(config, 'main', config.ConfigDB())
    import start
    from api import canvas as canvas_api
    monkeypatch.setattr(canvas_api, '_editor_session', None)
    monkeypatch.setattr(canvas_api, '_editor_last_seen', 0.)
    monkeypatch.setattr(canvas_api, '_live_sessions', {})
    config.main['core'] = {**config.main['core'], 'main_loop_enable': False, 'auto_start': False}
    (tmp_path/'expose-fixture-source').touch()
    tool_request = urllib.request.Request(registry[0]['url'], data=json.dumps({'jsonrpc':'2.0','id':1,'method':'tools/list'}).encode(), headers={'Content-Type':'application/json'})
    registry[0]['tools'] = json.load(urllib.request.urlopen(tool_request))['result']['tools']
    mcp_manage._save_mcp_list(registry)
    start._register_core_mcp(silent=True)
    config.main['canvas_layout'] = {'cards': [
        {'id': 'core', 'mcpId': 'agentcore', 'toolName': 'decision_core', 'driverName': 'AgentCore', 'x': 30, 'y': 60},
        {'id': 'capture', 'mcpId': 'local-video', 'toolName': 'demo_capture', 'driverName': 'Perception', 'x': 400, 'y': 60},
        {'id': 'fixture-source', 'mcpId': 'local-video', 'toolName': 'fixture_source', 'driverName':'Synthetic fixture', 'x':30, 'y':330},
        {'id': 'video', 'mcpId': 'local-video', 'toolName': 'video_task', 'driverName': 'Perception', 'x': 780, 'y': 60}],
        'connections': [{'id':f'capture-input-{i}','fromCardId':'fixture-source','fromPortIdx':i,'toCardId':'capture','toPortIdx':0 if i<2 else 2,'format':'image/jpeg' if i<2 else 'data/json'} for i in range(3)], 'execConnections': [
            {'id': 'exec-'+cid, 'fromCardId': 'core', 'toCardId': cid, 'toMcpId': 'local-video', 'toToolName': tool}
            for cid, tool in [('capture', 'demo_capture'), ('video', 'video_task')]]}
    def backend(tool, action, **args):
        body = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': tool, 'arguments': {'action': action, **args}}}
        request = urllib.request.Request(registry[0]['url'], data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        value = json.load(urllib.request.urlopen(request))
        result = json.loads(value['result']['content'][0]['text'])
        assert 'error' not in result, result
        return result
    feeding = threading.Event()
    feed_errors = []
    def feed():
        cap = cv2.VideoCapture(str(video));i = 0
        try:
            while not feeding.is_set():
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0);continue
                payload = {'jpeg': base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode(),
                           'state': {'gripper_width': .01+(i % 20)*.002, 'position': [i*.001], 'units': {'gripper_width': 'm', 'position': 'rad'}}}
                req = urllib.request.Request(registry[0]['url'].replace('/mcp', '/test-capture-feed'), data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(req, timeout=3).close();i += 1
                feeding.wait(.12)
        except Exception as error:
            feed_errors.append(str(error))
        finally:
            cap.release()
    feeder = threading.Thread(target=feed, daemon=True);feeder.start()
    sock = socket.socket();sock.bind(('127.0.0.1', 0));sock.listen()
    server = uvicorn.Server(uvicorn.Config(start.app, lifespan='off', log_level='error'))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True);thread.start()
    try:
        with sync.sync_playwright() as p:
            chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser = p.chromium.launch(headless=True, **({'executable_path': str(chrome)} if chrome.exists() else {}))
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = [];page.on('pageerror', lambda e: errors.append(str(e)))
            page.add_init_script(CONFIG_PROBE)
            page.goto(f'http://127.0.0.1:{sock.getsockname()[1]}')
            page.locator('[data-card-id="capture"]').wait_for()
            page.locator('#canvas-claim-btn').click()
            artifacts = Path(os.environ.get('VIDEO_WORKBENCH_ARTIFACT_DIR', tmp_path))
            artifacts.mkdir(parents=True, exist_ok=True)
            screenshot(page, path=str(artifacts/'demo-canvas.png'))
            card = page.locator('[data-card-id="capture"]')
            assert card.locator('[data-key="action"]').is_hidden()
            assert card.locator('[data-key="action"] option').all_text_contents() == ['ui']
            card.locator('.canvas-exec-btn').click()
            frame = page.frame_locator('[data-workbench] iframe')
            assert '示范采集' in card.locator('.canvas-card-tool').inner_text()
            frame.get_by_role('button', name='开启采集预览', exact=True).wait_for()
            assert backend('demo_capture', 'info')['state'] == 'idle'
            frame.locator('#primary').click()
            frame.get_by_role('button', name='开始录制', exact=True).wait_for()
            assert backend('demo_capture', 'info')['session'] is None
            assert page.get_by_role('button', name='开启智能控制', exact=True).is_visible()
            # An iframe-supplied topic is overridden with this card's inputs.
            result = frame.locator('body').evaluate("async()=>await window.hostProbe('call',{action:'start',input_topics:['/other-card/private'],input_topic:'/injected'})")
            assert result['state'] == 'running'
            assert {s['topic'] for s in backend('demo_capture','info')['sources']} == {'/fixture/ego','/fixture/exo','/fixture/state'}

            denied = frame.locator('body').evaluate("async()=>{try{await window.configProbe();return 'unexpected';}catch(e){return e.message;}}")
            assert '未声明配置能力' in denied
            assert frame.locator('#mainView img').get_attribute('src').startswith('data:image/jpeg')
            frame.locator('#setup summary').click()
            frame.locator('#demoName').fill('浏览器验收：合成录像与状态')
            frame.locator('#setup summary').click()
            frame.locator('#primary').click()
            frame.get_by_role('heading', name='正在记录示范').wait_for()
            frame.get_by_role('button', name='标记人工干预').click()
            artifacts = Path(os.environ.get('VIDEO_WORKBENCH_ARTIFACT_DIR', tmp_path));artifacts.mkdir(parents=True, exist_ok=True)
            frame.locator('body').evaluate('()=>window.scrollTo(0,0)')
            screenshot(page, path=str(artifacts/'v5-capture-recording.png'))
            # Closing the iframe leaves the collection running.
            page.locator('[data-workbench] button[aria-label="关闭工作台"]').click()
            assert backend('demo_capture', 'info')['session']['state'] == 'recording'
            card.locator('.canvas-exec-btn').click()
            frame.get_by_role('button', name='结束并保存', exact=True).wait_for()
            frame.locator('#primary').click()
            frame.get_by_role('heading', name='示范已保存').wait_for(timeout=30000)
            info = backend('demo_capture', 'info')
            did = info['session']['draft_id']
            assert info['session']['marks']
            screenshot(page, path=str(artifacts/'v5-capture-saved.png'))
            page.set_viewport_size({'width': 420, 'height': 860})
            assert frame.locator('html').evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
            screenshot(page, path=str(artifacts/'v5-capture-small.png'))
            page.locator('[data-workbench] button[aria-label="关闭工作台"]').click()
            page.set_viewport_size({'width': 1440, 'height': 1000})
            video_card = page.locator('[data-card-id="video"]')
            assert video_card.locator('[data-key="action"]').is_hidden()
            assert video_card.locator('[data-key="action"] option').all_text_contents() == ['ui']
            video_card.locator('.canvas-exec-btn').click()
            frame.locator('#status').filter(has_text='就绪').wait_for()
            frame.locator('#headerDraft').select_option(did)
            frame.locator('#observationToggle').click()
            frame.locator('[data-observe="motion"]').click()
            frame.locator('#motionSummary').filter(has_text='1路数值状态').wait_for()
            assert frame.get_by_role('button', name='结合动作分析', exact=True).is_visible()
            assert frame.locator('#observeDock .primary:visible').count() == 1
            assert 'gripper_width' in frame.locator('#motionSignals').inner_text()
            frame.locator('#motionEvidence summary').click()
            assert '未做硬件同步' in frame.locator('#captureQuality').inner_text()
            screenshot(page, path=str(artifacts/'v5-captured-motion.png'))
            assert not errors and not feed_errors
            page.locator('[data-workbench] button[aria-label="关闭工作台"]').click()
            page.locator('#canvas-release-btn').click()
            browser.close()
    finally:
        feeding.set();feeder.join(5)
        backend('demo_capture', 'stop')
        server.should_exit = True;thread.join(10);sock.close()


@pytest.fixture
def stack(tmp_path, monkeypatch):
    ffmpeg = pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    pytest.importorskip('cv2')
    # Keep perception's `main`, `utils`, and `plugins` out of Core's namespace.
    script = '''
import sys, json
from pathlib import Path
root, work, ffmpeg = map(str, sys.argv[1:])
sys.path[:0] = [root+'/perception/tests', root+'/perception']
import vision_stubs
import main
from utils import file_intake
from http.server import ThreadingHTTPServer
cfg = {'plugins': {'video_task': {'enabled': True, 'data_dir': work+'/drafts',
       'upload_roots': [work+'/uploads'], 'ffmpeg': ffmpeg},
       'demo_capture': {'enabled': True},
       'echo_cancellation': {'enabled': False}, 'audio_mixer': {'enabled': False}},
       'file_intake': {'dir': work+'/uploads'}}
main._bundle = main.PerceptionBundle(cfg, None)
# Fixture-only ROS ingress: never shipped by the production HTTP handler.
from types import SimpleNamespace
import base64
capture = main._bundle._plugin_for('demo_capture')[0]
capture.executor = SimpleNamespace(add_node=lambda n: None, remove_node=lambda n: None)
vision_stubs._FakeNode.get_topic_names_and_types = lambda self: [
    ('/fixture/ego', ['sensor_msgs/msg/CompressedImage']),
    ('/fixture/exo', ['sensor_msgs/msg/CompressedImage']),
    ('/fixture/state', ['std_msgs/msg/String'])]
# Opt-in synthetic source tool for the actual wiring/preview-start test only.
source_ports = [{'topic': '/fixture/ego', 'format': 'image/jpeg'},
                {'topic': '/fixture/exo', 'format': 'image/jpeg'},
                {'topic': '/fixture/state', 'format': 'data/json'}]
source_tool = {'name': 'fixture_source', 'type': 'sensor', 'description': 'Synthetic test source; no hardware',
               'inputSchema': {'type': 'object', 'properties': {'action': {'type': 'string', 'enum': ['info','start','stop']}}},
               'topic_out': source_ports}
original_tools, original_dispatch = main._bundle.get_all_tools, main._bundle.dispatch
source_enabled = lambda: Path(work+'/expose-fixture-source').exists()
main._bundle.get_all_tools = lambda: original_tools() + ([source_tool] if source_enabled() else [])
main._bundle.dispatch = lambda name, args: ({'state': 'running', 'topic_out': source_ports}
    if name == 'fixture_source' and source_enabled() else original_dispatch(name, args))
main._load_config = lambda: cfg
file_intake.access_token = lambda: 'workbench-test-token'
class FixtureHandler(main.make_handler()):
    def do_POST(self):
        if self.path == '/test-capture-feed':
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            capture.receive('/fixture/ego', SimpleNamespace(data=base64.b64decode(data['jpeg'])))
            capture.receive('/fixture/exo', SimpleNamespace(data=base64.b64decode(data['jpeg'])))
            capture.receive('/fixture/state', SimpleNamespace(data=json.dumps(data['state'])))
            self._send(200, '{}')
        else:
            super().do_POST()
server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
Path(work+'/ready.json').write_text(json.dumps({'port':server.server_port,'tools':main._bundle.get_all_tools()}))
server.serve_forever()
'''
    with (tmp_path/'perception.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', script, str(ROOT), str(tmp_path), ffmpeg], stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+20
            while not (tmp_path/'ready.json').exists():
                assert process.poll() is None, (tmp_path/'perception.log').read_text()
                assert time.monotonic() < deadline, 'Perception did not start'
                time.sleep(.05)
            ready = json.loads((tmp_path/'ready.json').read_text())
            registry = [{'id': 'local-video', 'category': 'perception', 'online': True, 'transport': 'http', 'url': f'http://127.0.0.1:{ready["port"]}/mcp', 'tools': ready['tools']}]
            monkeypatch.setattr(config, 'main', {'services': {'mcp': registry}, 'skills': {'installed': []}})
            import auth
            monkeypatch.setattr(auth, 'get_token', lambda: 'workbench-test-token')
            async def ignore_topics(*args, **kwargs):
                pass
            monkeypatch.setattr(mcp_manage, '_notify_inspector', ignore_topics)
            app = fastapi.FastAPI()
            app.include_router(mcp_manage.router, prefix='/api')
            app.include_router(skills.router, prefix='/api')
            app.state.model_requests = []
            app.state.skill_reply = None
            @app.post('/test-model/chat/completions')
            async def model(request: fastapi.Request):
                payload = await request.json()
                app.state.model_requests.append(payload)
                recipe = {'goal': '测试替身生成的候选', 'objects': [], 'uncertainties': ['固定测试输出，不是模型理解结果'],
                          'steps': [{'id': 's1', 'title': '测试候选步骤', 'start_s': 1, 'end_s': 2,
                                     'basis': 'unknown', 'preconditions': [], 'postconditions': [],
                                     'depends_on': [], 'notes': '仅验证候选采用流程'}]}
                if app.state.skill_reply is not None:
                    recipe = app.state.skill_reply
                return {'model': 'contract-test-only',
                        'choices': [{'message': {'content': json.dumps(recipe)}, 'finish_reason': 'stop'}],
                        'usage': {'total_tokens': 46}}
            layout = {'cards': [{'id': 'core', 'mcpId': 'agentcore'}, {'id': 'video', 'mcpId': 'local-video', 'toolName': 'video_task'}],
                      'execConnections': [{'id': 'exec-video', 'fromCardId': 'core', 'toCardId': 'video', 'toMcpId': 'local-video', 'toToolName': 'video_task'}]}
            @app.get('/api/canvas/layout')
            def canvas():
                return {'code': 200, 'data': layout}
            from api.canvas import router as canvas_router
            app.include_router(canvas_router, prefix='/api')
            @app.get('/workbench.js')
            def host_script():
                return fastapi.responses.FileResponse(ROOT/'agent-core/web/js/workbench.js', media_type='text/javascript')
            @app.get('/')
            def page():
                return fastapi.responses.HTMLResponse('''<!doctype html><style>
                body{margin:0;background:#0c121c;color:white;font:14px system-ui}.modal-overlay{display:flex;justify-content:center;padding:12px}.modal{display:flex;flex-direction:column;max-width:98vw}.modal-header{display:flex;justify-content:space-between;padding:12px}.btn-icon{cursor:pointer}</style>
                <script type="module">import {openWorkbench} from '/workbench.js';
                const response=await fetch('/api/mcp/local-video/tools');const tools=(await response.json()).data;
                const result=await fetch('/api/mcp/local-video/call',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tool:'video_task',arguments:{action:'ui'}})});
                const ui=JSON.parse((await result.json()).data[0].text);
                const tool=tools.find(t=>t.name==='video_task');
                await openWorkbench({mcp:'local-video',tool:'video_task',instanceId:'video'},tool.inputSchema['x-workbench'],ui);
                </script>''')
            video = tmp_path/'fixture.mp4'
            subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=duration=4:size=320x240:rate=12',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
            yield app, video, registry
        finally:
            process.terminate()
            process.wait(timeout=10)


def test_proxy_upload_observe_media_limits_and_skill_import(stack):
    app, video, registry = stack
    video_tool = next(t for t in registry[0]['tools'] if t['name'] == 'video_task')
    with TestClient(app) as client:
        def call(action, **args):
            r = client.post('/api/mcp/local-video/call', json={'tool':'video_task', 'arguments': dict(action=action, **args)})
            assert r.json()['code'] == 200, r.text
            return r.json()['data']
        def text(action, **args):
            r = json.loads(call(action, **args)[0]['text'])
            assert 'error' not in r, r
            return r
        draft = text('new', goal='查看测试图案')['draft']
        upload = client.post('/api/mcp/local-video/file/upload', files={'file': ('test.mp4', video.read_bytes(), 'video/mp4')})
        assert upload.json()['code'] == 200
        text('import', draft_id=draft['id'], video_path=upload.json()['data']['path'])
        deadline=time.monotonic()+15
        while text('info')['busy']:
            assert time.monotonic() < deadline
            time.sleep(.05)
        assert text('info')['job']['state'] == 'done'
        content=call('observe', draft_id=draft['id'], frame_count=2)
        assert [c['type'] for c in content] == ['text', 'image', 'image']
        body={'tool':'video_task','asset_id':draft['id']}
        r=client.post('/api/mcp/local-video/workbench/media',json=body)
        assert r.status_code==200 and r.content[4:8]==b'ftyp'
        assert r.headers['content-type']=='video/mp4'
        for changed, code in [({'asset_id':'../escape'},400),({'tool':'arbitrary'},403),({'kind':'config'},400)]:
            assert client.post('/api/mcp/local-video/workbench/media',json={**body,**changed}).status_code==code
        video_tool['inputSchema']['x-workbench']['media']=False
        assert client.post('/api/mcp/local-video/workbench/media',json=body).status_code==403
        video_tool['inputSchema']['x-workbench']['media']=True
        recipe={'goal':'查看测试图案','steps':[{'id':'s1','title':'观察变化','start_s':0,'end_s':3,'basis':'observed'}]}
        text('update',draft_id=draft['id'],expected_revision=1,recipe=recipe)
        skill=text('save',draft_id=draft['id'])['skill']
        assert client.post('/api/skills/import',json=skill).json()['code']==200
        assert config.main['skills']['installed'][0]['slug']==skill['slug']
        from mcp_client import _to_openai_schema
        converted=_to_openai_schema('local-video',video_tool)
        update=next(t for t in converted if t['name'].endswith('__update'))
        assert {'draft_id','recipe','expected_revision'} <= set(update['parameters']['required'])


def create_demo(frame, name):
    frame.locator('.topbar .menu > summary').click()
    frame.locator('#newDraft').click()
    frame.locator('#dialogBody input').fill(name)
    frame.locator('#dialogOk').click()
    frame.locator('#dataWorkspace:not(.hidden)').wait_for()


def upload_video(frame, video, name='测试视频', role='external'):
    if not frame.locator('#dataAdvanced').is_visible():
        frame.locator('#observationToggle').click()
    frame.locator('#addVideo').click()
    frame.locator('#dialogBody input').nth(0).fill(name)
    frame.locator('#dialogBody select').select_option(role)
    frame.locator('#dialogOk').click()
    frame.locator('#videoFile').set_input_files(str(video))
    frame.locator('#status').filter(has_text='处理完成').wait_for(timeout=20000)
    frame.locator('#mainVideo').evaluate('(v)=>v.readyState>=1 || new Promise(r=>v.addEventListener("loadedmetadata",r,{once:true}))')


def task_from_demo(frame):
    frame.locator('[data-tab="task"]').click()
    frame.locator('#manualTask > summary').click()
    frame.locator('#newGoal').fill('测试图案回看与条件编辑')
    frame.locator('#createTask').click()
    frame.locator('#status').filter(has_text='任务已创建').wait_for()
    frame.locator('#addStep').click()
    frame.locator('#stepTitle').fill('核对图案位置')
    frame.locator('#stepDetails > summary').click()
    frame.locator('#post').fill('位置可在画面中核对')
    frame.locator('#observeStep').click()
    frame.locator('#frames img').nth(5).wait_for()
    frame.locator('#attachEvidence').click()
    frame.locator('#basis').select_option('observed')
    frame.locator('#saveRecipe').click()
    frame.locator('#status').filter(has_text='任务已保存').wait_for()


def exercise_skill_flow(page, frame, origin, app, artifacts):
    """Real Canvas/HTTP/persistence; explicitly fixed model outputs, no robot."""
    def call(action, **args):
        response = page.request.post(origin+'/api/mcp/local-video/call', data={
            'tool': 'video_task', 'arguments': {'action': action, **args}})
        result = json.loads(response.json()['data'][0]['text'])
        assert 'error' not in result, result
        return result
    skill = dict(title='测试：把对象放到目标区域', goal='把{{object}}放到{{target}}', summary='参数化任务合同测试，非模型质量证据',
        parameters=[dict(id='object', label='操作对象', type='object', required=True),
                    dict(id='target', label='目标区域', type='region', required=True)],
        applicability=['场景已核对'], exclusions=['无法夹持的对象'], uncertainties=['固定测试响应'], conflicts=[],
        steps=[dict(id='s1', title='把{{object}}放到{{target}}', preconditions=['确认位置'],
                    postconditions=['对象到达目标'], capabilities=['移动', '停止'], depends_on=[],
                    source_refs=[], failure='停止并报告')])
    app.state.skill_reply = skill
    call('config', model='contract-test-only', endpoint=origin+'/test-model')
    frame.locator('[data-tab=task]').click()
    frame.locator('#extractSkill').click()
    frame.locator('#skill-candidate:not(.hidden), #status.error').wait_for(timeout=20000)
    assert '参数化任务合同测试' in frame.locator('#skillDescription').inner_text(), frame.locator('#status').inner_text()
    assert frame.locator('nav button').all_text_contents() == ['视频', '步骤', '技能', '复盘']
    assert frame.locator('#skillPrimary').inner_text() == '确认并入库'
    assert frame.locator('#skillDescription input:visible').count() == 0
    asset = call('skill_list')['skills'][0]
    before = call('skill_inspect', skill_id=asset['id'], skill_version=1)
    assert len(before['version']['sources']) == 1
    frame.locator('#skillVideo').evaluate('(v)=>v.readyState>=2 || new Promise(r=>v.addEventListener("loadeddata",r,{once:true}))')
    screenshot(page, path=str(artifacts/'v4-skill-extraction.png'))
    # An incompatible plain instruction skill is a reference, never an inferred parameterized asset.
    existing = {'slug': 'screw-lid-test', 'name': '旋开螺纹盖', 'oneLiner': '螺纹盖需要旋转',
                'instruction': '固定瓶身并旋转螺纹盖；拔出不适用。'}
    assert page.request.post(origin+'/api/skills/import', data=existing).json()['code'] == 200
    app.state.skill_reply = dict(slug='screw-lid-test', decision='reuse', reason='测试普通指令技能不会被误当作参数化技能',
                                shared=[], differences=[], conflicts=[])
    frame.locator('#skillCompare').click()
    frame.locator('#skillMatches button').filter(has_text='旋开螺纹盖').click()
    frame.locator('#skillPrimary').click()
    frame.locator('#skillPrimary').filter(has_text='查看原技能').wait_for(timeout=20000)
    frame.locator('#skillPrimary').click()
    frame.locator('#dialogBody').filter(has_text='没有它的参数化结构').wait_for()
    assert '没有它的参数化结构' in frame.locator('#dialogBody').inner_text()
    frame.locator('#dialogOk').click()
    app.state.skill_reply = dict(slug='screw-lid-test', decision='separate', reason='旋转与移动的操作条件不同',
                                shared=[], differences=['螺纹结构'], conflicts=['不能按名字合并'])
    current=call('skill_inspect',skill_id=asset['id'],skill_version=1)
    call('skill_compare',skill_id=asset['id'],skill_version=1,expected_revision=current['asset']['revision'],installed=[existing])
    deadline=time.monotonic()+10
    while call('info')['busy']:
        assert time.monotonic()<deadline
        time.sleep(.05)
    frame.locator('#skillRefresh').click()
    frame.locator('#skillComparison').filter(has_text='不能按名字合并').wait_for(timeout=20000)
    screenshot(page, path=str(artifacts/'v4-skill-comparison.png'))
    frame.locator('#skillPrimary').click()
    # Import failure must keep the candidate and must not pretend installation succeeded.
    page.route('**/api/skills/import', lambda route: route.fulfill(status=503, content_type='application/json',
               body=json.dumps({'code':503,'error':'测试导入失败，可重试'})), times=1)
    frame.locator('#skillPrimary').click()
    frame.locator('#status.error').filter(has_text='测试导入失败').wait_for()
    assert frame.locator('#skillPrimary').inner_text() == '确认并入库'
    frame.locator('#skillPrimary').click()
    frame.locator('#skillPrimary').filter(has_text='用于新任务').wait_for()
    exported=call('save',skill_id=asset['id'],skill_version=1,confirmed=True)['skill']
    assert page.request.post(origin+'/api/skills/import',data=exported).json()['data']['already']
    assert len([s for s in config.main['skills']['installed'] if s['slug']==exported['slug']]) == 1
    frame.locator('#skillPrimary').click()
    frame.locator('[data-parameter=object]').fill('本次蓝杯')
    frame.locator('[data-parameter=target]').fill('右侧托盘')
    frame.locator('#skillPrimary').click()
    frame.locator('#skillTaskPreview').filter(has_text='把本次蓝杯放到右侧托盘').wait_for()
    frame.locator('#skillPrimary').click()
    frame.locator('#skillTaskPreview').filter(has_text='缺少现场位置').wait_for()
    assert frame.locator('#skillPrimary').inner_text() == '补充现场与工具'
    assert frame.locator('button.primary:visible').count() == 1
    task=call('task_list')['tasks'][0]
    detail=call('task_inspect',task_id=task['id'])['task']
    assert detail['scene']==detail['bindings']=={}
    assert 'start_s' not in json.dumps(detail['recipe'])
    screenshot(page, path=str(artifacts/'v4-skill-new-task.png'))
    # A new task can be reviewed as unknown without fabricating an execution.
    frame.locator('#skillTaskPreview button').filter(has_text='复盘这次任务').click()
    frame.locator('[data-task-outcome=unknown]').click()
    frame.locator('#skillReviewPrimary').click()
    frame.locator('#skillRunRecords').filter(has_text='人工标记').wait_for()
    app.state.skill_reply={**skill,'uncertainties':['补充现场反馈后再核验，尚未验证改进']}
    frame.locator('#skillReviewPrimary').click()
    frame.locator('#skillSubtitle').filter(has_text='v2').wait_for(timeout=20000)
    assert frame.locator('#skillPrimary').inner_text()=='确认并入库'
    revised=call('skill_inspect',skill_id=asset['id'],skill_version=2)
    assert revised['asset']['recommended'] is None
    assert revised['version']['feedback'][0]['outcome']=='unknown'
    assert call('save',skill_id=asset['id'],skill_version=1,confirmed=True)['skill']==exported
    screenshot(page, path=str(artifacts/'v4-skill-feedback-candidate.png'))
    assert '本次候选改了什么' in frame.locator('#skillDescription').inner_text()
    frame.locator('#skillTools .menu > summary').click()
    frame.locator('#skillEdit').click()
    frame.locator('#dialogBody details > summary').first.click()
    frame.locator('#dialogBody [aria-label="所需能力"]').fill('固定\n移动\n停止与完成反馈')
    frame.locator('#dialogOk').click()
    frame.locator('#skillSubtitle').filter(has_text='v3').wait_for()
    assert call('skill_inspect',skill_id=asset['id'],skill_version=3)['version']['content']['steps'][0]['capabilities']==['固定','移动','停止与完成反馈']
    assert call('save',skill_id=asset['id'],skill_version=1,confirmed=True)['skill']==exported
    # Reopening and the skill tab do not require choosing/uploading a video.
    page.get_by_role('button',name='关闭工作台').click()
    page.locator('[data-card-id="video"] .canvas-exec-btn').click()
    frame.locator('#status').filter(has_text='就绪').wait_for()
    assert frame.locator('#headerDraft').input_value()==''
    frame.locator('[data-tab=skill]').click()
    frame.locator('#skillLibrary button').first.click()
    frame.locator('#skillSubtitle').filter(has_text='v3').wait_for()
    page.set_viewport_size({'width':420,'height':780})
    assert frame.locator('body').evaluate('()=>document.documentElement.scrollWidth<=innerWidth+1')
    assert frame.locator('#skillPrimary').evaluate('(e)=>e.getBoundingClientRect().bottom<=innerHeight')
    assert frame.locator('button.primary:visible').count()==1
    screenshot(page, path=str(artifacts/'v4-skill-small-canvas.png'))
    page.set_viewport_size({'width':1440,'height':1000})
    app.state.skill_reply=None


def test_browser_upload_preview_edit_save_and_native_mode(stack, tmp_path):
    sync = pytest.importorskip('playwright.sync_api')
    uvicorn = pytest.importorskip('uvicorn')
    app, video, _ = stack
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen()
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        with sync.sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
            page.goto(origin)
            frame=page.frame_locator('iframe')
            frame.locator('#status').filter(has_text='就绪').wait_for()
            create_demo(frame, '测试素材：无任务也能导入')
            upload_video(frame, video)
            assert frame.locator('#dataTaskState').inner_text() == '尚未关联任务'
            assert frame.locator('#mainVideo').evaluate('(v)=>v.duration') == pytest.approx(4, abs=.1)
            task_from_demo(frame)
            frame.locator('#openTaskMore').click()
            frame.locator('#saveSkill').click()
            frame.locator('#status').filter(has_text='存入现有技能库').wait_for()
            assert len(config.main['skills']['installed']) == 1
            frame.locator('#closeTools').click()
            frame.locator('#openAnalysis').click()
            frame.locator('#mode').select_option('native')
            frame.locator('#analyze').click()
            frame.locator('#status.error').filter(has_text='配置模型').wait_for(timeout=20000)
            assert '核对图案位置' in frame.locator('#steps').inner_text()
            frame.locator('#editCurrent').click()
            frame.locator('#stepTitle').fill('未保存编辑')
            frame.locator('.topbar .menu > summary').click()
            frame.locator('#refresh').click()
            frame.locator('#dialog[open]').wait_for()
            frame.locator('#dialogCancel').click()
            assert frame.locator('#stepTitle').input_value() == '未保存编辑'
            frame.locator('#saveRecipe').click()
            frame.locator('#status').filter(has_text='任务已保存').wait_for()
            # Wire-contract double; these checks do not claim model quality.
            response=page.request.post(origin+'/api/mcp/local-video/call',data={'tool':'video_task','arguments':{
                'action':'config','model':'contract-test-only','endpoint':origin+'/test-model'}})
            assert response.json()['code']==200
            frame.locator('#openAnalysis').click()
            frame.locator('#mode').select_option('frames')
            frame.locator('#analyze').click()
            frame.locator('#candidates .candidate').filter(has_text='测试候选步骤').wait_for(timeout=20000)
            assert '未保存编辑' in frame.locator('#steps').inner_text()
            assert len(app.state.model_requests)==1
            frame.locator('#adoptAnalysis').click()
            frame.locator('#steps').filter(has_text='测试候选步骤').wait_for()
            page.reload()
            frame.locator('#status').filter(has_text='就绪').wait_for()
            draft_id=frame.locator('#headerDraft option').evaluate_all('(items)=>items.map(o=>o.value).filter(Boolean)[0]')
            frame.locator('#headerDraft').select_option(draft_id)
            frame.locator('[data-tab="task"]').click()
            frame.locator('#editCurrent').click()
            frame.locator('#steps').filter(has_text='测试候选步骤').wait_for()
            # Conflict: a later server edit cannot be overwritten by dirty UI.
            frame.locator('#steps button').first.click()
            frame.locator('#stepTitle').fill('本地编辑')
            def api(action, **args):
                result=page.request.post(origin+'/api/mcp/local-video/call',data={'tool':'video_task','arguments':{'action':action,'draft_id':draft_id,**args}})
                return json.loads(result.json()['data'][0]['text'])
            d=api('inspect')['draft']; value=d['recipe'];value['goal']='并发保存的新目标'
            api('update',expected_revision=d['revision'],recipe=value)
            page.wait_for_timeout(5500)
            frame.locator('#saveRecipe').click()
            frame.locator('#status.error').filter(has_text='未覆盖新版本').wait_for()
            assert api('inspect')['draft']['recipe']['goal']=='并发保存的新目标'
            assert frame.locator('body').evaluate("()=>{try{parent.document.body;return false}catch(e){return true}}")
            assert not errors, errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10);sock.close()


def test_browser_enters_workbench_from_real_canvas(stack, tmp_path, monkeypatch):
    """Exercise the actual start.app, index.html and Canvas, not the harness page.

    The robot lifespan stays off on development hosts. Perception/codec HTTP
    comes from stack; this test does not claim ROS or model integration.
    """
    sync = pytest.importorskip('playwright.sync_api')
    uvicorn = pytest.importorskip('uvicorn')
    pytest.importorskip('cryptography')
    model_app, video, registry = stack
    monkeypatch.chdir(tmp_path)
    (tmp_path/'web').symlink_to(ROOT/'agent-core/web', target_is_directory=True)
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path/'canvas.db'))
    config._seed_defaults()
    monkeypatch.setattr(config, 'main', config.ConfigDB())
    import start
    monkeypatch.setattr(start.app.router, 'routes', [
                        *[r for r in model_app.routes if getattr(r, 'path', '') == '/test-model/chat/completions'],
                        *start.app.router.routes])
    config.main['core'] = {**config.main['core'], 'main_loop_enable': False, 'auto_start': False}
    mcp_manage._save_mcp_list(registry)
    start._register_core_mcp(silent=True)
    config.main['canvas_layout'] = {
        'cards': [
            {'id': 'core', 'mcpId': 'agentcore', 'toolName': 'decision_core',
             'driverName': 'AgentCore', 'x': 60, 'y': 80},
            {'id': 'video', 'mcpId': 'local-video', 'toolName': 'video_task',
             'driverName': 'Perception', 'x': 510, 'y': 80},
            {'id': 'message', 'mcpId': 'agentcore', 'toolName': 'remote_message',
             'driverName': 'AgentCore', 'x': 60, 'y': 300}],
        'connections': [],
        'execConnections': [{'id': 'exec-video', 'fromCardId': 'core', 'toCardId': 'video',
                             'toMcpId': 'local-video', 'toToolName': 'video_task'}]}
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0)); sock.listen()
    server = uvicorn.Server(uvicorn.Config(start.app, lifespan='off', log_level='error'))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        with sync.sync_playwright() as p:
            chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser = p.chromium.launch(headless=True, **(
                {'executable_path': str(chrome)} if chrome.exists() else {}))
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            origin = f'http://127.0.0.1:{sock.getsockname()[1]}'
            page.add_init_script(CONFIG_PROBE)
            catalog_fixture(page)
            response = page.goto(origin)
            assert response.body() == (ROOT/'agent-core/web/index.html').read_bytes()
            card = page.locator('[data-card-id="video"]')
            card.wait_for()
            assert page.locator('#canvas-area').is_visible()
            assert page.locator('[data-workbench]').count() == 0
            assert card.locator('[data-key="action"]').input_value() == 'ui'
            page.locator('#canvas-claim-btn').click()
            action = card.locator('[data-key="action"]')
            button = card.locator('.canvas-exec-btn')
            assert button.inner_text() == '打开工作台'
            assert 'locked' not in button.get_attribute('class')
            assert 'locked' in page.locator('[data-card-id="message"] .canvas-exec-btn').get_attribute('class')
            assert action.is_hidden()
            assert action.locator('option').all_text_contents() == ['ui']
            assert card.locator('.canvas-field:visible').count() == 0
            # Ordinary cards retain the stopped-project guard.
            calls = []
            page.on('request', lambda request: calls.append(request.url)
                    if '/api/mcp/' in request.url and '/call' in request.url else None)
            page.locator('[data-card-id="message"] .canvas-exec-btn').evaluate('(button)=>button.click()')
            page.locator('#canvas-toast').filter(has_text='请先开启智能控制').wait_for()
            assert not calls
            card.locator('.canvas-exec-btn').click()
            frame = page.frame_locator('[data-workbench] iframe')
            frame.locator('#status').filter(has_text='就绪').wait_for()
            artifacts = Path(os.environ.get('VIDEO_WORKBENCH_ARTIFACT_DIR', tmp_path))
            artifacts.mkdir(parents=True, exist_ok=True)
            screenshot(page, path=str(artifacts/'demo-video-entry.png'))
            # Configure from the actual iframe and persist through existing Canvas API.
            frame.locator('#openModelSettings').click()
            frame.locator('#modelDialog[open]').wait_for()
            assert frame.locator('#modelService').input_value() == 'router'
            frame.locator('#modelChoice option[value="fixture-gemini"]').wait_for(state='attached')
            assert frame.locator('#modelName').is_hidden()
            assert frame.locator('#modelChoice').input_value() == 'PR-G-Flash-3.8'
            frame.locator('#modelChoice').select_option('fixture-gemini')
            assert frame.locator('#modelName').input_value() == 'fixture-gemini'
            assert '示范学习' in card.locator('.canvas-card-tool').inner_text()
            sidebar_card = page.locator('.sidebar-tool-card[data-tool-name="video_task"]')
            sidebar_card.wait_for()
            assert '示范学习' in sidebar_card.locator('.tool-card-name').inner_text()
            assert sidebar_card.locator('.tool-card-config-btn').count() == 0
            assert page.locator('[data-tool-name="decision_core"] .chip-config-btn, [data-tool-name="decision_core"] .tool-card-config-btn').count() == 1
            assert frame.locator('#modelAdvanced').get_attribute('open') is None
            assert frame.locator('#modelEndpoint').input_value() == 'https://cloud.phanthy.com/phanrouterapi/v1beta'
            screenshot(page, path=str(artifacts/'demo-model-settings.png'))
            frame.locator('#modelService').select_option('compatible')
            frame.locator('#modelEndpoint').fill(origin+'/test-model')
            frame.locator('#modelName').fill('contract-test-only')
            frame.locator('#modelKey').fill('fixture-key-never-return')
            frame.locator('#saveModelSettings').click()
            frame.locator('#status').filter(has_text='模型设置已保存').wait_for()
            assert frame.locator('#modelDialog').is_hidden()
            saved = config.main['tool_config:local-video:video_task']
            assert saved['api_key'] == 'fixture-key-never-return'
            assert saved['endpoint'] == origin+'/test-model'
            assert config.ConfigDB()['tool_config:local-video:video_task'] == saved
            public = frame.locator('body').evaluate("async()=>await window.configProbe()")
            assert public['secret_set']['api_key'] is True
            assert 'fixture-key-never-return' not in json.dumps(public)
            invalid = frame.locator('body').evaluate("""async()=>{const errors=[];
                for(const args of [{tool:'other'},{values:{data_dir:'/tmp/no'}},{values:{provider:'invalid'}}]){
                    try{await window.configProbe(args);errors.push('unexpected success');}catch(e){errors.push(e.message);}
                }return errors;}""")
            assert all('无效' in message for message in invalid)
            assert not model_app.state.model_requests  # Saving doesn't spend model tokens.
            page.reload()
            card.locator('.canvas-exec-btn').click()
            frame.locator('#status').filter(has_text='就绪').wait_for()
            frame.locator('#openModelSettings').click()
            frame.locator('#modelDialog[open]').wait_for()
            assert frame.locator('#modelName').input_value() == 'contract-test-only'
            assert frame.locator('#modelKey').input_value() == ''
            assert '留空保留' in frame.locator('#modelKey').get_attribute('placeholder')
            frame.locator('#saveModelSettings').click()
            frame.locator('#status').filter(has_text='模型设置已保存').wait_for()
            assert config.main['tool_config:local-video:video_task']['api_key'] == 'fixture-key-never-return'
            create_demo(frame, '浏览器验收：双视角测试图案')
            upload_video(frame, video, '主视角（合成测试片）')
            upload_video(frame, video, 'Ego视角（同片验证同步）', 'ego')
            frame.locator('#syncDrag').focus()
            frame.locator('#syncDrag').press('ArrowRight')
            assert float(frame.locator('#offset').input_value())==pytest.approx(.01)
            frame.locator('#observe-sync details > summary').click()
            frame.locator('#offset').fill('0.25')
            frame.locator('#syncConfirmed').check()
            frame.locator('#saveSync').click()
            frame.locator('#status').filter(has_text='同步已保存').wait_for()
            assert frame.locator('#thumbs video').count() == 1
            screenshot(page, path=str(artifacts/'demo-multiview.png'))
            frame.locator('[data-observe=points]').click()
            frame.locator('#editPoints').check()
            frame.locator('#pointName').fill('人工测试点')
            rect=frame.locator('#overlay').bounding_box()
            frame.locator('#overlay').click(position={'x':rect['width']/2,'y':rect['height']/2})
            frame.locator('#savePoints').click()
            frame.locator('#trackChoice option').nth(1).wait_for(state='attached')
            artifacts=Path(os.environ.get('VIDEO_WORKBENCH_ARTIFACT_DIR',str(tmp_path)))
            artifacts.mkdir(parents=True,exist_ok=True)
            frame.locator('main').evaluate('(e)=>e.scrollTop=0')
            screenshot(page, path=str(artifacts/'v3-data.png'),full_page=True)
            # Actual numeric upload and trajectory render, explicitly synthetic.
            import numpy as np
            data_file=tmp_path/'umi-fixture.npz'
            np.savez(data_file, robot0_eef_pos=np.array([[.1,0,0],[.2,.1,0],[.3,.1,.1]]),
                     robot0_eef_rot_axis_angle=np.zeros((3,3)), robot0_gripper_width=np.array([.05,.07,.03]))
            frame.locator('[data-observe=motion]').click()
            with page.expect_file_chooser() as chooser:
                frame.locator('#importData').click()
            chooser.value.set_files(str(data_file))
            assert '已识别为 UMI' in frame.locator('#dialogBody').inner_text()
            frame.locator('#dialogBody input').nth(0).fill('12')
            frame.locator('#dialogBody input').nth(1).fill('synthetic_test_world')
            frame.locator('#dialogOk').click()
            frame.locator('#trajectoryView:not(.hidden)').wait_for(timeout=20000)
            assert 'synthetic_test_world' in frame.locator('#trackSource').inner_text()
            frame.locator('#trajectory').scroll_into_view_if_needed()
            screenshot(page, path=str(artifacts/'v3-trajectory.png'),full_page=True)
            frame.locator('#exportTrack').click()
            exported=json.loads(frame.locator('#dialogBody textarea').input_value())
            assert exported['frames'][0]['widths'] == [.05]
            frame.locator('#dialogOk').click()
            # Select primary before creating task to keep step 0 in range.
            frame.locator('#streams button').first.click()
            task_from_demo(frame)
            frame.locator('#editCurrent').click()
            frame.locator('#stepDetails > summary').click()
            frame.locator('main').evaluate('(e)=>e.scrollTop=0')
            frame.locator('#resultVideo').evaluate('(v)=>v.readyState>=2 || new Promise(r=>v.addEventListener("loadeddata",r,{once:true}))')
            screenshot(page, path=str(artifacts/'v3-task.png'),full_page=True)
            frame.locator('#openDevice').click()
            frame.locator('#bindingEditor > summary').click()
            frame.locator('#loadTools').click()
            frame.locator('#status').filter(has_text='已读取 1 项').wait_for()
            frame.locator('#profileName').fill('仅测试声明，不运行设备')
            frame.locator('#actionTool').select_option('0')
            frame.locator('#actionParams [data-key="action"]').select_option('info')
            frame.locator('#stopTool').select_option('0')
            frame.locator('#stopParams [data-key="action"]').select_option('cancel')
            frame.locator('#completion').fill('人工核对测试图案')
            frame.locator('#saveBinding').click()
            frame.locator('#status').filter(has_text='已保存').wait_for()
            frame.locator('#checkBinding').click()
            frame.locator('#readinessSummary').filter(has_text='声明检查通过').wait_for()
            frame.locator('#bindingEditor > summary').click()
            frame.locator('#toolsDialog').evaluate('(e)=>e.scrollTop=0')
            screenshot(page, path=str(artifacts/'v3-device.png'),full_page=True)
            assert config.main['canvas_layout']['cards'][1]['toolName'] == 'video_task'
            frame.locator('#closeTools').click()
            frame.locator('[data-tab="review"]').click()
            assert frame.locator('#reviewEmpty').is_visible()
            with page.expect_file_chooser() as chooser:
                frame.locator('#addExecution').click()
            chooser.value.set_files(str(video))
            frame.locator('#status').filter(has_text='处理完成').wait_for(timeout=20000)
            frame.locator('#executionVideo').evaluate('(v)=>v.readyState>=1 || new Promise(r=>v.addEventListener("loadedmetadata",r,{once:true}))')
            assert frame.locator('#recordStream').input_value()
            assert frame.locator('#executionVideo').evaluate('(v)=>v.duration')==pytest.approx(4,abs=.1)
            assert not frame.locator('#recordProfile').is_visible()
            frame.locator('[data-outcome=unknown]').click()
            frame.locator('#saveRecord').click()
            frame.locator('#records .feedback').wait_for(state='attached')
            assert '不确定' in frame.locator('#records').inner_text()
            assert '未提供补充说明' in frame.locator('#records').inner_text()
            assert '人工' in frame.locator('#records').inner_text()
            frame.locator('main').evaluate('(e)=>e.scrollTop=0')
            screenshot(page, path=str(artifacts/'v3-review-overview.png'),full_page=True)
            frame.locator('#openRegression').click()
            frame.locator('#manualImprovement').click()
            frame.locator('#status').filter(has_text='编辑已保存为候选').wait_for()
            candidate=frame.locator('#reviewCandidate option').evaluate_all('(items)=>items.map(o=>o.value).filter(Boolean)[0]')
            frame.locator('#reviewCandidate').select_option(candidate)
            frame.locator('#adoptImprovement').click()
            frame.locator('#status.error').filter(has_text='尚未通过回归').wait_for()
            for index in range(2):
                frame.locator('.case-status').nth(index).select_option('passed')
                frame.locator('.case-evidence').nth(index).fill('仅验证UI状态转换，不是机器人成功')
            frame.locator('#saveCases').click()
            frame.locator('#status').filter(has_text='回归证据已记录').wait_for()
            screenshot(page, path=str(artifacts/'v3-review.png'),full_page=True)
            frame.locator('#adoptImprovement').click()
            frame.locator('#status').filter(has_text='候选已采用').wait_for()
            page.get_by_role('button', name='关闭工作台').click()
            page.locator('[data-workbench]').wait_for(state='detached')
            assert card.is_visible()
            screenshot(page, path=str(artifacts/'v3-canvas.png'),full_page=True)
            page.reload()
            card.locator('.canvas-exec-btn').click()
            frame.locator('#status').filter(has_text='就绪').wait_for()
            draft_id = frame.locator('#headerDraft option').evaluate_all('(items)=>items.map(o=>o.value).filter(Boolean)[0]')
            frame.locator('#headerDraft').select_option(draft_id)
            frame.locator('#observationToggle').click()
            frame.locator('#observe-sync details > summary').click()
            frame.locator('#streams button').nth(1).wait_for()
            assert frame.locator('#streams button').count() == 3
            frame.locator('#streams button').nth(1).click()
            assert frame.locator('#offset').input_value() == '0.25'
            assert frame.locator('#syncConfirmed').is_checked()
            exercise_skill_flow(page, frame, origin, model_app, artifacts)
            assert not errors, errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.mark.parametrize("native", [False, True])
def test_quick_import_analyze_and_read_steps_without_forms(stack, native):
    """Zero required text fields through upload, model request and readable result."""
    sync = pytest.importorskip('playwright.sync_api')
    uvicorn = pytest.importorskip('uvicorn')
    app, video, _ = stack
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen()
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        with sync.sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1000})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
            catalog_fixture(page)
            page.goto(origin);frame=page.frame_locator('iframe')
            frame.locator('#status').filter(has_text='就绪').wait_for()
            assert frame.locator('#welcomeNew').is_visible()
            assert not frame.locator('#offset').is_visible()
            with page.expect_file_chooser() as chooser:
                frame.locator('#welcomeNew').click()
            chooser.value.set_files(str(video))
            frame.locator('#status').filter(has_text='视频已就绪').wait_for(timeout=20000)
            assert frame.locator('#headerDraft option:checked').inner_text()=='fixture'
            assert not frame.locator('#dialog').is_visible()
            assert not frame.locator('#offset').is_visible()
            assert not frame.locator('#pointName').is_visible()
            assert not frame.locator('#quickGoal').is_visible()
            assert frame.locator('#quickAnalyze').inner_text()=='分析这个视频'
            # Missing model is actionable and makes no paid request.
            frame.locator('#quickAnalyze').click()
            frame.locator('#modelDialog[open]').wait_for()
            assert frame.locator('#modelService').input_value() == 'router'
            frame.locator('#closeModelSettings').click()
            assert len(app.state.model_requests)==0
            response=page.request.post(origin+'/api/mcp/local-video/call',data={'tool':'video_task','arguments':{
                'action':'config','model':'contract-test-only','endpoint':origin+'/test-model',
                'native_protocol':'qwen_video_url' if native else 'disabled'}})
            assert response.json()['code']==200
            frame.locator('#quickAnalyze').click()
            frame.locator('#candidates .watch-step').first.wait_for(timeout=20000)
            assert '测试候选步骤' in frame.locator('#candidates').inner_text()
            assert len(app.state.model_requests)==1
            parts=app.state.model_requests[0]['messages'][1]['content']
            assert any(item['type']=='video_url' for item in parts)==native
            assert any(item['type']=='image_url' for item in parts)==(not native)
            assert not frame.locator('#mode').is_visible()
            assert not frame.locator('#taskGoal').is_visible()
            assert not frame.locator('#candidates pre').is_visible()
            frame.locator('#candidates .watch-step').first.click()
            assert frame.locator('#resultVideo').is_visible()
            assert '测试候选步骤' in frame.locator('#resultCaption').inner_text()
            assert frame.locator('#resultVideo').evaluate('(v)=>v.currentTime')==pytest.approx(1,abs=.1)
            frame.locator('[data-tab="data"]').click()
            assert frame.locator('#quickAnalyze').inner_text()=='查看分析结果'
            frame.locator('#quickAnalyze').click()
            assert len(app.state.model_requests)==1, 'Reading an existing result must not call the model again'
            frame.locator('#adoptAnalysis').click()
            frame.locator('#stepTitle').wait_for()
            assert frame.locator('#stepTitle').input_value()=='测试候选步骤'
            assert frame.locator('#taskEditor').is_visible()
            assert not errors,errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10);sock.close()


def test_history_current_recipe_inline_edit_and_small_embed(stack, tmp_path):
    """History is read-only; current means saved recipe, including later edits.

    Real codec and plugin persistence, manual candidates (no model or device).
    Exercise keyboard/pointer timeline editing and the actual iframe at 420px.
    """
    sync = pytest.importorskip('playwright.sync_api')
    uvicorn = pytest.importorskip('uvicorn')
    app, video, _ = stack
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen()
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        with sync.sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            page=browser.new_page(viewport={'width':1200,'height':850})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
            catalog_fixture(page)
            page.goto(origin);frame=page.frame_locator('iframe')
            frame.locator('#status').filter(has_text='就绪').wait_for()
            # Drag/drop follows the same no-required-form file flow.
            frame.locator('#dataWelcome').evaluate("""(el,bytes)=>{const d=new DataTransfer();
                d.items.add(new File([new Uint8Array(bytes)],'拖入的示范.mp4',{type:'video/mp4'}));
                el.dispatchEvent(new DragEvent('drop',{dataTransfer:d,bubbles:true,cancelable:true}));}""",list(video.read_bytes()))
            frame.locator('#status').filter(has_text='视频已就绪').wait_for(timeout=20000)
            draft_id=frame.locator('#headerDraft').input_value()
            def api(action, **args):
                result=page.request.post(origin+'/api/mcp/local-video/call',data={'tool':'video_task','arguments':{'action':action,'draft_id':draft_id,**args}})
                value=json.loads(result.json()['data'][0]['text'])
                assert 'error' not in value,value
                return value
            d=api('inspect')['draft']
            d=api('update',expected_revision=d['revision'],recipe={'goal':'历史隔离验收','steps':[]})['draft']
            for i in range(5):
                d=api('propose',expected_revision=d['revision'],source='human',recipe={
                    'goal':f'第{i+1}次人工建议', 'uncertainties':['这是测试建议，需要核对'],
                    'steps':[{'id':'s1','title':f'建议{i+1}中的动作','start_s':.5,'end_s':3,'basis':'unknown'}]})['draft']
            frame.locator('.topbar .menu > summary').click();frame.locator('#refresh').click()
            frame.locator('#status').filter(has_text='已刷新').wait_for()
            frame.locator('[data-tab=task]').click()
            assert frame.locator('#candidates .candidate').count()==1
            assert frame.locator('#resultTitle').inner_text()=='第5次人工建议'
            assert '待采用' in frame.locator('#resultIdentity').inner_text()
            assert frame.locator('#adoptAnalysis').evaluate('(e)=>e.getBoundingClientRect().bottom<=innerHeight')
            before=api('inspect')['draft']
            frame.locator('#openHistory').click()
            assert frame.locator('.history-item').count()==5
            assert all('时间未记录' not in s for s in frame.locator('.history-item time').all_text_contents())
            frame.locator('[data-candidate-id="'+d['analyses'][0]['id']+'"]').click()
            assert frame.locator('#resultTitle').inner_text()=='第1次人工建议'
            assert '历史分析' in frame.locator('#resultIdentity').inner_text()
            assert api('inspect')['draft']==before
            frame.locator('#openHistory').click();frame.locator('#returnCurrent').click()
            assert frame.locator('#resultTitle').inner_text()=='第5次人工建议'
            frame.locator('#adoptAnalysis').click()
            frame.locator('#stepTitle').wait_for()
            frame.locator('#stepTitle').fill('我修改后保存的动作')
            frame.locator('#rangeIn').focus();frame.locator('#rangeIn').press('ArrowRight')
            assert float(frame.locator('#stepStart').input_value())==pytest.approx(.6,abs=.001)
            handle=frame.locator('#rangeOut').bounding_box();track=frame.locator('#rangeTimeline').bounding_box()
            page.mouse.move(handle['x']+handle['width']/2,handle['y']+handle['height']/2)
            page.mouse.down();page.mouse.move(track['x']+track['width']*.625,handle['y']+handle['height']/2,steps=5);page.mouse.up()
            assert float(frame.locator('#stepEnd').input_value())==pytest.approx(2.5,abs=.03)
            # Reading another result never silently discards dirty edits.
            frame.locator('#openHistory').click();frame.locator('[data-candidate-id="'+d['analyses'][0]['id']+'"]').click()
            frame.locator('#dialog[open]').wait_for();frame.locator('#dialogCancel').click()
            assert frame.locator('#historyDrawer').is_visible()
            frame.locator('#closeHistory').click()
            assert frame.locator('#stepTitle').input_value()=='我修改后保存的动作'
            frame.locator('#saveRecipe').click();frame.locator('#status').filter(has_text='任务已保存').wait_for()
            assert '当前任务 · 已保存'==frame.locator('#resultIdentity').inner_text()
            assert '我修改后保存的动作' in frame.locator('#candidates').inner_text()
            saved=api('inspect')['draft']
            assert saved['recipe']['steps'][0]['start_s']==pytest.approx(.6,abs=.001)
            # A previously adopted result is historical, not the edited task.
            frame.locator('#openHistory').click()
            assert '曾采用' in frame.locator('.history-item').first.inner_text()
            frame.locator('.history-item button').first.click()
            frame.locator('#adoptAnalysis').click()
            frame.locator('#status.error').wait_for()
            assert api('inspect')['draft']==saved, 'Stale adoption must retain the edited saved recipe'
            frame.locator('#openHistory').click();frame.locator('#returnCurrent').click()
            assert '我修改后保存的动作' in frame.locator('#candidates').inner_text()
            # Real embed remains usable on a small viewport; one main action.
            page.set_viewport_size({'width':420,'height':780})
            assert frame.locator('body').evaluate('()=>document.documentElement.scrollWidth<=innerWidth+1')
            assert frame.locator('#editCurrent').evaluate('(e)=>{const r=e.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight;}')
            assert frame.locator('button.primary:visible').count()==1
            frame.locator('#editCurrent').click()
            assert frame.locator('#saveRecipe').evaluate('(e)=>e.getBoundingClientRect().bottom<=innerHeight')
            frame.locator('#finishEdit').click()
            frame.locator('#openHistory').click()
            assert frame.locator('#historyDrawer').evaluate('(e)=>e.getBoundingClientRect().width<=innerWidth')
            frame.locator('#closeHistory').click()
            artifacts=Path(os.environ.get('VIDEO_WORKBENCH_ARTIFACT_DIR',str(tmp_path)))
            artifacts.mkdir(parents=True,exist_ok=True)
            screenshot(page, path=str(artifacts/'small-embed.png'),full_page=True)
            page.reload();frame.locator('#status').filter(has_text='就绪').wait_for()
            frame.locator('#headerDraft').select_option(draft_id);frame.locator('[data-tab=task]').click()
            assert '当前任务 · 已保存'==frame.locator('#resultIdentity').inner_text()
            assert '我修改后保存的动作' in frame.locator('#candidates').inner_text()
            assert app.state.model_requests==[]
            assert not errors,errors
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10);sock.close()


@pytest.mark.parametrize('allowed', [False, True])
def test_skill_read_bridge_is_declared_bounded_and_read_only(stack, allowed):
    sync = pytest.importorskip('playwright.sync_api')
    uvicorn = pytest.importorskip('uvicorn')
    app, _, _ = stack
    config.main['skills'] = {'installed': [dict(slug='plain-test', name='Example', oneLiner='Summary',
        instruction='Selected instruction', category='test', version='1', active=True, internal='not exposed')]}
    probe = '''<body><pre id="result">pending</pre><script>
    window.addEventListener('message',async e=>{if(!e.ports[0])return;const p=e.ports[0];let n=0;
    const pending=new Map();p.onmessage=e=>{const resolve=pending.get(e.data.id);if(resolve)resolve(e.data);};
    const ask=args=>new Promise(resolve=>{const id=String(++n);pending.set(id,resolve);p.postMessage({id,method:'skills',args});});
    const results=[];for(const args of [{},{slug:'plain-test'},{url:'/api/config'},{slug:'../config'},{limit:31},{cursor:-1},{slug:'plain-test',query:'ignored'}])results.push(await ask(args));
    document.getElementById('result').textContent=JSON.stringify(results);});
    </script></body>'''
    @app.get('/bridge-probe')
    def bridge_probe():
        spec = dict(protocol='motus.workbench/1', kind='skill-authoring', actions=['info','stop'], skills={'read':allowed})
        return fastapi.responses.HTMLResponse('<script type="module">import {openWorkbench} from "/workbench.js";'
            +'await openWorkbench({mcp:"local-video",tool:"video_task",instanceId:"video"},'
            +json.dumps(spec)+','+json.dumps({'protocol':'motus.workbench/1','html':probe}).replace('</','<\\/')+');</script>')
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen()
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    try:
        with sync.sync_playwright() as p:
            browser=p.chromium.launch(headless=True);page=browser.new_page()
            requests=[];page.on('request',lambda r:requests.append((r.method,r.url)))
            page.goto(f'http://127.0.0.1:{sock.getsockname()[1]}/bridge-probe')
            result=page.frame_locator('iframe').locator('#result')
            result.filter(has_text='plain-test' if allowed else '未声明').wait_for()
            rows=json.loads(result.inner_text())
            if allowed:
                assert 'instruction' not in rows[0]['result']['skills'][0]
                assert rows[1]['result']['skill']['instruction']=='Selected instruction'
                assert 'internal' not in rows[1]['result']['skill']
                assert all('error' in r for r in rows[2:])
                assert len([r for r in requests if '/api/skills' in r[1]])==2
                assert all(method=='GET' for method,url in requests if '/api/skills' in url)
            else:
                assert all('未声明' in r['error'] for r in rows)
                assert not any('/api/skills' in url for _,url in requests)
            assert len(config.main['skills']['installed'])==1
            browser.close()
    finally:
        server.should_exit=True;thread.join(timeout=10);sock.close()
