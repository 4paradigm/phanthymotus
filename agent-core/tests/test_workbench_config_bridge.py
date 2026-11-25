"""Exercise the real opaque-iframe configuration bridge against bounded APIs.

Run with Playwright Chromium installed, or set PLAYWRIGHT_CHROMIUM_EXECUTABLE.
No live browser, server, credentials, or robot connections are used.
"""
import copy
import json
import os
from pathlib import Path
import sys

import pytest


CORE = Path(__file__).resolve().parents[1]
HOST = 'http://workbench.test'
SECRET = 'fixture-secret-never-return'
PROPS = {
    'endpoint': {'type': 'string', 'scope': 'shared'},
    'api_key': {'type': 'string', 'scope': 'shared', 'format': 'password'},
    'environment_job': {'type': 'string', 'scope': 'shared'},
    'instance_setting': {'type': 'string', 'scope': 'instance'},
    'unadvertised': {'type': 'string', 'scope': 'shared'},
    'count': {'type': 'integer', 'scope': 'shared'},
    'provider': {'type': 'string', 'scope': 'shared', 'enum': ['local', 'remote']},
}
FIELDS = ['endpoint', 'api_key', 'environment_job', 'instance_setting', 'count', 'provider']
PROBE = '''<script>
let port, sequence=0;const pending=new Map();
window.addEventListener('message',event=>{
  if(event.source!==parent||event.data?.protocol!=='motus.workbench/1'||!event.ports[0])return;
  port=event.ports[0];port.onmessage=event=>{const done=pending.get(event.data.id);if(done){pending.delete(event.data.id);done(event.data);}};
  window.probe=(method,args={})=>new Promise(resolve=>{const id=String(++sequence);pending.set(id,resolve);port.postMessage({id,method,args});});
  window.probeReady=true;
});
</script>'''


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    pw = pytest.importorskip('playwright.sync_api')
    monkeypatch.setenv('DB_PATH', str(tmp_path/'canvas.db'))
    saved = {'endpoint': 'http://old:15732', 'api_key': SECRET,
             'environment_job': '', 'provider': 'local'}
    state = {
        'saved': dict(saved), 'runtime': dict(saved), 'cache': {}, 'calls': [], 'puts': [],
        'project_running': False, 'editor': True, 'busy': False, 'put_mode': 'ok',
        'rollback_fails': False, 'reads': 0,
    }

    def route(request_route):
        request = request_route.request
        path = request.url.removeprefix(HOST).split('?', 1)[0]
        content_type = 'application/json'
        status, value = 200, {'code': 200, 'data': None}
        if path == '/':
            content_type, value = 'text/html', '<html><body></body></html>'
        elif path == '/workbench.js':
            content_type, value = 'text/javascript', (CORE/'web/js/workbench.js').read_text()
        elif path == '/canvas.js':
            content_type = 'text/javascript'
            value = 'export const isProjectRunning=()=>window.testFlags.project_running;export const ensureEdit=async()=>window.testFlags.editor;'
        elif path == '/sidebar.js':
            content_type, value = 'text/javascript', 'export const getToolConfigs=()=>window.testCache;'
        elif path == '/api/mcp/selected/tools':
            value['data'] = [{'name': 'motion', 'configSchema': {'properties': PROPS}}]
        elif path == '/api/canvas/tool-config/selected/motion':
            if request.method == 'PUT':
                submitted = request.post_data_json
                state['puts'].append(submitted)
                if state['put_mode'] in ('ok', 'committed_error'):
                    state['saved'] = copy.deepcopy(submitted)
                elif state['put_mode'] == 'concurrent_write':
                    state['saved'] = {**state['saved'], 'endpoint': 'http://other-editor:15732'}
                if state['put_mode'] != 'ok':
                    status, value = 503, {'code': 503, 'message': 'simulated write response failure'}
            else:
                state['reads'] += 1
                if state['put_mode'] == 'unreadable' and state['puts']:
                    status, value = 503, {'code': 503, 'message': 'simulated read failure'}
                else:
                    value['data'] = copy.deepcopy(state['saved'])
        elif path == '/api/mcp/selected/call':
            body = request.post_data_json
            assert body['tool'] == 'motion'
            args = body['arguments']
            state['calls'].append(copy.deepcopy(args))
            if args['action'] == 'config':
                if args.get('endpoint') == 'invalid' or (state['rollback_fails'] and len(state['puts']) and args.get('endpoint') == 'http://old:15732'):
                    value['data'] = {'error': 'invalid connection'}
                else:
                    state['runtime'].update({k: v for k, v in args.items() if k not in ('action', 'instance_id')})
                    value['data'] = {'state': 'ready'}
            elif args['action'] == 'catalog':
                # Use Core's real replay planner, not an invented scope merge.
                from tool_config import plan_config_calls
                for config, extra in plan_config_calls({'configSchema': {'properties': PROPS}}, state['saved'], {}, args.get('instance_id', '')):
                    state['runtime'].update(config)
                value['data'] = {'profiles': [], 'datasets': [], 'models': []}
            elif args['action'] == 'env_targets':
                value['data'] = {'targets': []}
            elif args['action'] == 'inspect':
                value['data'] = ({'gpus': [], 'training': {
                    'available': False, 'multi_gpu_supported': False,
                    'max_gpus': 0, 'max_num_envs': 0, 'max_iterations': 0,
                }} if args.get('mode') == 'resources' else {'jobs': []})
            else:
                value['data'] = {
                    'state': 'ready', 'busy': state['busy'],
                    'config_values': {k: v for k, v in state['runtime'].items() if k != 'api_key'},
                    'config_secrets': {'api_key': bool(state['runtime'].get('api_key'))},
                }
        else:
            raise AssertionError('Unexpected host route: '+path)
        request_route.fulfill(status=status, content_type=content_type,
                              body=value if isinstance(value, str) else json.dumps(value))

    with pw.sync_playwright() as playwright:
        executable = os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE')
        if not executable and sys.platform == 'darwin':
            candidate = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            if candidate.is_file():
                executable = str(candidate)
        browser = playwright.chromium.launch(**({'executable_path': executable} if executable else {}))
        page = browser.new_page()
        page.route('**/*', route)
        page.goto(HOST)
        page.evaluate('flags=>{window.testFlags=flags;window.testCache={};}',
                      {k: state[k] for k in ('project_running', 'editor')})
        page.evaluate('async()=>{window.host=await import("/workbench.js");}')
        opened = [False]

        def open_probe(fields=FIELDS, actions=None, html=PROBE):
            spec = {'protocol': 'motus.workbench/1', 'kind': 'motion-training',
                    'actions': actions or ['info', 'stop', 'config', 'catalog'], 'config': fields}
            page.evaluate('async args=>await window.host.openWorkbench(...args)', [
                {'mcp': 'selected', 'tool': 'motion', 'instanceId': 'selected-card'}, spec,
                {'protocol': 'motus.workbench/1', 'html': html}, None,
            ])
            iframe = page.frames[1]
            if html == PROBE:
                iframe.wait_for_function('()=>window.probeReady')
            else:
                iframe.wait_for_function("()=>document.getElementById('status')?.textContent==='已连接'")
            opened[0] = True
            return iframe

        def ask(args=None, method='config'):
            if not opened[0]:
                open_probe()
            page.evaluate('flags=>{window.testFlags=flags;}',
                          {k: state[k] for k in ('project_running', 'editor')})
            return page.frames[1].evaluate('async value=>await window.probe(value.method,value.args)',
                                           {'method': method, 'args': args or {}})

        yield state, page, ask, open_probe
        browser.close()


def test_read_filters_secrets_and_instance_fields(bridge):
    state, page, ask, _ = bridge
    state['saved']['instance_setting'] = 'legacy-instance-value'
    result = ask()['result']
    assert result['values'] == {'endpoint': 'http://old:15732', 'environment_job': '', 'provider': 'local'}
    assert result['secret_set'] == result['runtime_secret_set'] == {'api_key': True}
    assert SECRET not in json.dumps(result)
    assert not state['puts']
    assert page.locator('iframe').get_attribute('sandbox') == 'allow-scripts'
    assert 'connect-src \'none\'' in page.locator('iframe').get_attribute('srcdoc')


@pytest.mark.parametrize('args', [
    {'tool': 'other'}, {'values': {'unknown': 'x'}}, {'values': {'unadvertised': 'x'}},
    {'values': {'instance_setting': 'x'}}, {'values': {'endpoint': 12}},
    {'values': {'provider': 'unknown'}}, {'values': {'count': 4}}, {'values': []},
])
def test_rejects_undeclared_cross_scope_or_invalid_values(bridge, args):
    state, _, ask, _ = bridge
    assert '无效' in ask(args)['error']
    assert not state['puts']
    assert not any(call['action'] == 'config' for call in state['calls'])


def test_opted_in_workbench_cannot_bypass_configuration_bridge(bridge):
    state, _, ask, _ = bridge
    result = ask({'action': 'config', 'unadvertised': 'bypass'}, method='call')
    assert '配置接口' in result['error']
    assert not state['calls'] and not state['puts']


def test_config_requires_explicit_capability(bridge):
    state, _, ask, open_probe = bridge
    open_probe(fields=[], actions=['info', 'stop', 'catalog'])
    assert '未声明' in ask()['error']
    assert not state['puts']


@pytest.mark.parametrize('flag', ['project_running', 'editor', 'busy'])
def test_project_edit_and_busy_gates_precede_mutation(bridge, flag):
    state, _, ask, _ = bridge
    state[flag] = False if flag == 'editor' else True
    assert 'error' in ask({'values': {'endpoint': 'http://new:15732'}})
    assert not state['puts']
    assert not any(call['action'] == 'config' for call in state['calls'])


def test_saved_connection_retains_secret_and_survives_core_replay(bridge):
    state, page, ask, _ = bridge
    result = ask({'values': {'endpoint': 'http://new:15732', 'environment_job': 'ready-job'}})
    assert result['result'] == {'saved': True}
    assert state['saved']['api_key'] == SECRET
    applied = [call for call in state['calls'] if call['action'] == 'config']
    assert len(applied) == 1 and not applied[0]['instance_id']
    assert 'instance_setting' not in applied[0]
    assert page.evaluate('window.testCache["selected:motion"]') == state['saved']
    ask({'action': 'catalog'}, method='call')
    assert state['runtime']['endpoint'] == 'http://new:15732'
    assert state['runtime']['environment_job'] == 'ready-job'
    assert SECRET not in json.dumps(ask()['result'])


def test_shared_save_preserves_existing_fields_and_core_instance_scope(bridge):
    state, _, ask, _ = bridge
    state['saved'].update(instance_setting='legacy-default', unadvertised='existing-setting')
    assert ask({'values': {'endpoint': 'http://new:15732'}})['result'] == {'saved': True}
    applied = [call for call in state['calls'] if call['action'] == 'config'][0]
    assert not applied['instance_id'] and 'instance_setting' not in applied and 'unadvertised' not in applied
    assert state['saved']['instance_setting'] == 'legacy-default'
    assert state['saved']['unadvertised'] == 'existing-setting'
    from tool_config import plan_config_calls
    shared, instance = plan_config_calls({'configSchema': {'properties': PROPS}}, state['saved'],
                                         {'instance_setting': 'this-card-only'}, 'selected-card')
    assert shared[1] == {} and shared[0]['endpoint'] == 'http://new:15732'
    assert 'instance_setting' not in shared[0]
    assert instance == ({'instance_setting': 'this-card-only'}, {'instance_id': 'selected-card'})


def test_rejected_connection_never_changes_saved_configuration(bridge):
    state, _, ask, _ = bridge
    old = copy.deepcopy(state['saved'])
    assert 'error' in ask({'values': {'endpoint': 'invalid'}})
    assert state['saved'] == state['runtime'] == old
    assert not state['puts']


def test_lost_write_response_is_confirmed_by_readback(bridge):
    state, page, ask, _ = bridge
    state['put_mode'] = 'committed_error'
    result = ask({'values': {'endpoint': 'http://new:15732'}})
    assert result['result'] == {'saved': True}
    assert state['reads'] == 2
    assert state['saved']['endpoint'] == state['runtime']['endpoint'] == 'http://new:15732'
    assert page.evaluate('window.testCache["selected:motion"]') == state['saved']


def test_write_failure_restores_complete_previous_connection(bridge):
    state, page, ask, _ = bridge
    old = copy.deepcopy(state['saved'])
    state['put_mode'] = 'rejected'
    result = ask({'values': {'endpoint': 'http://new:15732', 'api_key': 'new-fixture-secret'}})
    assert '已恢复原连接' in result['error']
    assert SECRET not in result['error'] and 'new-fixture-secret' not in result['error']
    assert state['runtime'] == state['saved'] == old
    assert page.evaluate('window.testCache') == {}


@pytest.mark.parametrize('mode', ['missing_secret', 'unreadable', 'concurrent_write', 'rollback_failure'])
def test_unrecoverable_write_failure_is_explicit_and_never_claims_success(bridge, mode):
    state, page, ask, _ = bridge
    state['put_mode'] = 'rejected'
    if mode == 'missing_secret':
        del state['saved']['api_key']  # Runtime credential came from the environment.
    elif mode == 'rollback_failure':
        state['rollback_fails'] = True
    else:
        state['put_mode'] = mode
    result = ask({'values': {'endpoint': 'http://new:15732'}})
    assert 'error' in result and 'result' not in result
    assert '请重试保存' in result['error'] and '已恢复' not in result['error']
    assert state['runtime']['endpoint'] == 'http://new:15732'
    assert page.evaluate('window.testCache') == {}
    assert SECRET not in result['error']


def test_actual_motion_html_connects_saves_reopens_and_keeps_core_replay(bridge):
    state, page, _, open_probe = bridge
    pw = pytest.importorskip('playwright.sync_api')
    # Keep the exact checked-in HTML and host module. Only HTTP dependencies
    # are simulated; the opaque iframe and MessagePort handshake are real.
    html = (CORE.parent/'actucore/plugins/motion/workbench.html').read_text()
    actions = ['info', 'stop', 'config', 'catalog', 'inspect', 'env_targets']
    fields = ['endpoint', 'api_key', 'environment_job']
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    frame = open_probe(fields=fields, actions=actions, html=html)
    frame.locator('#connection').click()
    pw.expect(frame.locator('#settings')).to_be_visible()
    frame.locator('#manualConnection > summary').click()
    pw.expect(frame.locator('#endpoint')).to_have_value('http://old:15732')
    pw.expect(frame.locator('#apiKey')).to_have_value('')
    frame.locator('#endpoint').fill('http://new:15732')
    frame.locator('#apiKey').fill('new-fixture-secret')
    before_catalogs = sum(call['action'] == 'catalog' for call in state['calls'])
    frame.locator('#saveConnection').click()
    pw.expect(frame.locator('#settings')).not_to_be_visible()
    assert state['saved']['endpoint'] == state['runtime']['endpoint'] == 'http://new:15732'
    assert state['saved']['api_key'] == state['runtime']['api_key'] == 'new-fixture-secret'
    assert sum(call['action'] == 'catalog' for call in state['calls']) > before_catalogs
    assert page.evaluate('window.testCache["selected:motion"]') == state['saved']
    page.get_by_role('button', name='关闭工作台', exact=True).click()
    pw.expect(page.locator('[data-workbench]')).to_have_count(0)
    frame = open_probe(fields=fields, actions=actions, html=html)
    frame.locator('#connection').click()
    pw.expect(frame.locator('#settings')).to_be_visible()
    pw.expect(frame.locator('#endpoint')).to_have_value('http://new:15732')
    pw.expect(frame.locator('#apiKey')).to_have_value('')
    assert 'new-fixture-secret' not in frame.locator('body').inner_text()
    assert not errors
