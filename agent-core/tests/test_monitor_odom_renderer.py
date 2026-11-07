"""The monitor changes only its odometry display route."""
import os
import pathlib
import re
import shutil
import subprocess

import pytest


def test_monitor_keeps_json_route_and_switches_odom_without_remount(tmp_path):
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
const TextRenderer={name:'text',canRender:f=>f==='data/json'};
const KvLatestRenderer={name:'kv-latest'};
const OdomRenderer={name:'odom',canRender:f=>f==='state/odom'};
const VideoRenderer={},ImageRenderer={},AudioRenderer={},LidarRenderer={},ControlRenderer={},
PointCloudRenderer={},MappingRenderer={},SlamPreviewRenderer={},SkeletonRenderer={},Pose2dRenderer={},
TactileRenderer={},CameraRenderer={},DepthRenderer={},DepthZlibRenderer={};
for (const renderer of [VideoRenderer,ImageRenderer,AudioRenderer,LidarRenderer,ControlRenderer,
PointCloudRenderer,MappingRenderer,SlamPreviewRenderer,SkeletonRenderer,Pose2dRenderer,TactileRenderer,
CameraRenderer,DepthRenderer,DepthZlibRenderer]) renderer.canRender=()=>false;
const assert=(yes,message)=>{if(!yes)throw Error(message)};
'''
    checks = '''
assert(_createRenderer('state/odom','kv').name==='odom','odom latest route');
assert(_createRenderer('state/odom','log').name==='odom','odom log route');
assert(_createRenderer('data/json','log').name==='text','JSON log changed');
assert(_createRenderer('data/json','kv').name==='kv-latest','JSON latest changed');
let selected;
_cards.set('/odom',{format:'state/odom',mode:'kv',renderer:{name:'odom',setMode:m=>{selected=m}},col:0,row:0,colSpan:1,rowSpan:2});
_switchMode('/odom','log',[]);
assert(selected==='log','odom mode switch discarded its current data');
let remounts=0;
_refreshRenderer=()=>{remounts++};_hideGridOverlay=noop;_saveLayout=noop;
document.removeEventListener=noop;
const card={format:'state/odom',el:{classList:{remove:noop}}};
_drag={card,topicPath:'/odom',kind:'resize',moved:true,valid:true};
_onPointerEnd();
assert(remounts===0,'resizing discarded odom history');
card.format='data/json';
_drag={card,topicPath:'/json',kind:'resize',moved:true,valid:true};
_onPointerEnd();
assert(remounts===1,'ordinary JSON resize behavior changed');
'''
    script = tmp_path / 'monitor.cjs'
    script.write_text(preamble + source + '\n' + checks)
    result = subprocess.run([runtime, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
