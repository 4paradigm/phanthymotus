"""The monitor renders odometry through the same JSON route as `data/json`.

方案 A：`state/odom` 与 `data/json` 共用通用 JSON 渲染器。monitor 不再有 odom
专用渲染器、也没有「原地 setMode / 缩放不重挂」的特例；模式切换和缩放后的重建
与普通 JSON 完全一致。这个测试沿用旧做法：把 monitor-dashboard.js 的 import 去掉、
用桩替换渲染器后在 node 里跑真实的 `_createRenderer` / `_switchMode` / `_onPointerEnd`。

Run: cd agent-core && uv run --no-project --python 3.12 --with pytest \
     python -m pytest tests/test_monitor_odom_renderer.py -q
"""
import os
import pathlib
import re
import shutil
import subprocess

import pytest


def test_monitor_routes_odometry_through_the_generic_json_renderers(tmp_path):
    runtime = os.environ.get('NODE') or shutil.which('node')
    if runtime is None:
        pytest.skip('Node.js is required for browser behavior tests')
    path = pathlib.Path(__file__).resolve().parents[1] / 'web/js/monitor-dashboard.js'
    source = re.sub(r'^import\s[\s\S]*?;\n', '', path.read_text(), flags=re.M).replace('export ', '')
    preamble = '''
const noop=()=>{};
globalThis.window={addEventListener:noop};
globalThis.document={};
const localStorage={getItem:()=>null,setItem:noop};
const ActivityRenderer={name:'activity',canRender:()=>true};
const TextRenderer={name:'text',canRender:f=>f==='data/json'||(f||'').startsWith('text/')};
const KvLatestRenderer={name:'kv-latest'};
const VideoRenderer={},ImageRenderer={},AudioRenderer={},LidarRenderer={},ControlRenderer={},
PointCloudRenderer={},MappingRenderer={},SkeletonRenderer={},Pose2dRenderer={},
TactileRenderer={},CameraRenderer={},DepthRenderer={},DepthZlibRenderer={};
for (const renderer of [VideoRenderer,ImageRenderer,AudioRenderer,LidarRenderer,ControlRenderer,
PointCloudRenderer,MappingRenderer,SkeletonRenderer,Pose2dRenderer,TactileRenderer,
CameraRenderer,DepthRenderer,DepthZlibRenderer]) renderer.canRender=()=>false;
const assert=(yes,message)=>{if(!yes)throw Error(message)};
'''
    checks = '''
assert(_rendererHint('state/odom')==='data/json','odom hint was not normalized to data/json');
assert(_rendererHint('data/json')==='data/json','JSON hint changed');
assert(_isJsonFormat('state/odom')&&_isJsonFormat('data/json'),'odom must keep the log/latest modes');
assert(!_isJsonFormat('image/jpeg'),'non-JSON formats gained modes');
assert(_createRenderer('state/odom','kv').name==='kv-latest','odom latest route changed');
assert(_createRenderer('state/odom','log').name==='text','odom log route changed');
assert(_createRenderer('data/json','log').name==='text','JSON log changed');
assert(_createRenderer('data/json','kv').name==='kv-latest','JSON latest changed');

// 模式切换：odom 现在和 data/json 一样重建渲染器（通用渲染器没有原地 setMode）
let remounts=0;
_refreshRenderer=()=>{remounts++};
_hideGridOverlay=noop;_saveLayout=noop;
document.removeEventListener=noop;
_cards.set('/odom',{format:'state/odom',mode:'kv',renderer:{name:'kv-latest'},col:0,row:0,colSpan:1,rowSpan:2});
_switchMode('/odom','log',[]);
assert(remounts===1,'odom mode switch must remount like ordinary JSON');
assert(_cards.get('/odom').mode==='log','odom mode switch discarded the new mode');

// 缩放：odom 与普通 JSON 一样重建渲染器，不再有按 format 的特例
let resized=0;
_refreshRenderer=()=>{resized++};
const odomCard={format:'state/odom',el:{classList:{remove:noop}}};
_drag={card:odomCard,topicPath:'/odom',kind:'resize',moved:true,valid:true};
_onPointerEnd();
assert(resized===1,'resizing odom must refresh the generic renderer');
const jsonCard={format:'data/json',el:{classList:{remove:noop}}};
_drag={card:jsonCard,topicPath:'/json',kind:'resize',moved:true,valid:true};
_onPointerEnd();
assert(resized===2,'ordinary JSON resize behavior changed');
'''
    script = tmp_path / 'monitor.cjs'
    script.write_text(preamble + source + '\n' + checks)
    result = subprocess.run([runtime, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
