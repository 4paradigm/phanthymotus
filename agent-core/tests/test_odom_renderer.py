"""Exercise the real renderer with a tiny DOM in JavaScriptCore."""
import pathlib
import shutil
import subprocess
import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / 'web' / 'js'
JSC = '/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc'


def test_odom_renderer_displays_pose_and_clears_it_when_lost(tmp_path):
    runtime = shutil.which('node') or (JSC if pathlib.Path(JSC).exists() else None)
    if runtime is None:
        pytest.skip('JavaScript runtime is unavailable')
    source = (WEB / 'renderers' / 'odom.js').read_text().replace('export const OdomRenderer', 'const OdomRenderer').replace('export function odomPreview', 'function odomPreview')
    harness = r'''
class Element {
  constructor() { this.children=[]; this.style={}; this._text=''; this.title=''; }
  set textContent(value) { this._text=value; this.children=[]; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  appendChild(e) { this.children.push(e); }
  remove() {}
}
globalThis.document={createElement: () => new Element()};
let timer; let now=0;
globalThis.performance={now:()=>now};
globalThis.setInterval=(fn)=>{timer=fn;return 1}; globalThis.clearInterval=()=>{};
globalThis.TextDecoder=class { decode(x) { return x; } };
const box=new Element();
OdomRenderer.mount(box);
const send=(x)=>OdomRenderer.onData(JSON.stringify(x));
const sample={schema:'motus.odom/1', stamp_ms:1234, twist:[1,null,null,null,null,0.5],
 pose:{position:[1,2,3],quaternion:[0,0,0,1],frame_id:'odom',child_frame_id:'base'},
 vendor:{turn_angle_deg:90,displacement_m:2.123456,pose_reference:'body_axes',reason:'ok',
 motus_slam:{status:'tracking',segment_id:'run-1'}}};
send(sample);
let text=box.children[0].textContent;
if (!text.includes('1.000') || !text.includes('odom') || !text.includes('run-1')) throw Error(text);
if (!text.includes('90.0') || !text.includes('2.123') || !text.includes('body_axes')) throw Error(text);
const findTitle=(el, title)=>el.title===title || el.children.some(c=>findTitle(c,title));
if (!findTitle(box, 'run-1')) throw Error('full segment must remain accessible');
sample.vendor.motus_slam.status='degraded'; send(sample);
text=box.children[0].textContent;
if (text.includes('2.123') || text.includes('90.0')) throw Error('untrusted summary remains visible: '+text);
if (text.includes('1.000, 2.000, 3.000')) throw Error('untrusted pose remains visible: '+text);
sample.vendor.motus_slam.status='unknown'; send(sample);
if (box.children[0].textContent.includes('1.000, 2.000, 3.000')) throw Error('unknown pose remains visible');
sample.pose=null; sample.vendor.motus_slam.status='lost'; send(sample);
text=box.children[0].textContent;
if (!text.includes('lost') || !text.includes('位置') || !text.includes('—')) throw Error(text);
sample.twist={}; send(sample);
now=6001; timer();
if (!box.children[0].textContent.includes('已暂停')) throw Error('missing stale notice');
OdomRenderer.unmount();
'''
    script = tmp_path / 'renderer.cjs'
    script.write_text(source + '\n' + harness)
    result = subprocess.run([runtime, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_views_select_odom_before_fallback():
    for view in ('detail-panel.js', 'dashboard.js', 'monitor-dashboard.js'):
        source = (WEB / view).read_text()
        assert "renderers/odom.js" in source
        renderers = source.split('const RENDERERS = [', 1)[1].split(']', 1)[0]
        assert renderers.index('OdomRenderer') < renderers.index('ActivityRenderer')


def test_dashboard_prefers_odom_topic_without_changing_port_order(tmp_path):
    runtime = shutil.which('node') or (JSC if pathlib.Path(JSC).exists() else None)
    if runtime is None:
        pytest.skip('JavaScript runtime is unavailable')
    source = (WEB / 'renderers' / 'odom.js').read_text().replace('export const OdomRenderer', 'const OdomRenderer').replace('export function odomPreview', 'function odomPreview')
    script = tmp_path / 'preview.cjs'
    script.write_text(source + r'''
const skill={ws_path:'/ws/bus/status',render_hint:'data/json',topic_out:[
 {topic:'/status',format:'data/json'}, {topic:'/odom',format:'state/odom'}]};
const result=odomPreview(skill);
if (result.path !== '/ws/bus/odom' || result.hint !== 'state/odom') throw Error(JSON.stringify(result));
if (skill.topic_out[0].topic !== '/status') throw Error('changed port order');
const fallback=odomPreview({ws_path:'/ws/bus/other',render_hint:'data/json',topic_out:[]});
if (fallback.path !== '/ws/bus/other' || fallback.hint !== 'data/json') throw Error('changed fallback');
''')
    result = subprocess.run([runtime, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
