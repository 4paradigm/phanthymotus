import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {createRequire} from 'node:module';
const xmldom=createRequire(import.meta.url)('@xmldom/xmldom');
const base = new URL('../../', import.meta.url);
const read = name => fs.readFile(new URL(name, base), 'utf8');
const strip = s => s.replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
const topic = '/rokae_helios/state/joints';
const id = 'mcp-1794972666';

// Exercise the actual canvas event handler's live-info and cached fallback paths.
const canvas = await read('canvas.js');
const start = canvas.indexOf('async function _openTopicDetailFor(');
const end = canvas.indexOf('\n/**', start);
for (const live of [true, false]) {
  let called;
  const el = {querySelectorAll: () => [{dataset:{topic,format:'sensor/skeleton'}}]};
  const c = vm.createContext({console, _cards:[{el,id:'card',toolName:'joints'}],_allMcps:[],
    fetch:async()=>({json:async()=>({topic_out:live?[{topic,format:'sensor/skeleton'}]:[]})}),
    _parseMcpCallResult:x=>x, inputArgs:()=>({}),_inputTopicsFor:()=>[],
    showTopicDetail:(...args)=>called=args,_showToast:()=>assert.fail('unexpected toast')});
  vm.runInContext(canvas.slice(start,end),c);
  await c._openTopicDetailFor(el,id,[]);
  assert.deepEqual(Array.from(called),[topic,'sensor/skeleton',id]);
}

// Execute actual detail-panel with DOM and renderers stubbed only at the UI boundary.
let mountedId;
const element = () => ({style:{},classList:{remove(){},add(){}},appendChild(){},addEventListener(){},remove(){}});
const elements = Object.fromEntries(['detail-panel','detail-close','detail-title','detail-subtitle','detail-body'].map(n=>[n,element()]));
const globals = {console,TextEncoder,TextDecoder,clearTimeout,setTimeout,
  document:{getElementById:n=>elements[n],createElement:element},
  openDetailPanelMobile(){},closeDetailPanelMobile(){}};
for(const name of ['Activity','Text','Video','Image','Audio','Lidar','Control','Skeleton','Pose2d','Tactile','Camera','Depth','DepthZlib']) {
 globals[name+'Renderer']={canRender:f=>name==='Skeleton'&&f==='sensor/skeleton',mount:(_,driver)=>mountedId=driver,unmount(){}};
}
const panel=vm.createContext(globals);
vm.runInContext(strip(await read('detail-panel.js')),panel);
vm.runInContext('_connect = () => {};',panel);
panel.initDetailPanel();panel.showTopicDetail(topic,'sensor/skeleton',id);
assert.equal(mountedId,id);

// Run real Three.js kinematics without a WebGL/browser dependency.
function decorate(node){
 node.querySelectorAll = name => Array.from(node.getElementsByTagName(name));
 node.querySelector = name => node.querySelectorAll(name)[0] || null;
 for(let c=node.firstChild;c;c=c.nextSibling) if(c.nodeType===1)decorate(c);
 return node;
}
class DOMParser {parseFromString(s,type){return decorate(new xmldom.DOMParser().parseFromString(s,type));}}
const xml = await fs.readFile(new URL('./fixtures/helios-kinematics.urdf', import.meta.url),'utf8');
let requested;
const ctx = vm.createContext({THREE,DOMParser,TextDecoder,console,fetch:async(url)=>{
 requested=url;return {json:async()=>({data:[{text:JSON.stringify({urdf:xml})}]})};}});
vm.runInContext(strip(await read('renderers/skeleton.js')).replace('const SkeletonRenderer =','globalThis.SkeletonRenderer ='),ctx);
const renderer=ctx.SkeletonRenderer;
renderer._scene=new THREE.Scene();renderer._joints={};renderer._links={};
renderer._buildFallbackSkeleton=()=>assert.fail('must not use humanoid fallback');
await renderer._loadModel(mountedId);
assert.equal(requested,'/api/mcp/'+id+'/call');
assert.equal(Object.keys(renderer._joints).length,20);
assert.equal(Object.keys(renderer._links).length,34);
assert.equal(Object.keys(renderer._joints).some(n=>n.includes('knee')),false);
const positions={left:[2.617994,1.570711,-1.570857,1.745331,0,0,0],right:[-2.617992,1.570712,1.570854,1.745330,0,0,0],body:[-1.169371,2.356204,-1.221732,0],head:[-0.004003,0.15363]};
const joints=Object.entries(positions).flatMap(([g,values])=>values.map((q,i)=>({name:g+'_joint_'+(i+1),q})));
renderer.onData(new TextEncoder().encode(JSON.stringify({joints})).buffer,'sensor/skeleton');
renderer._scene.updateMatrixWorld(true);
const doc=new DOMParser().parseFromString(xml,'text/xml');
const expected={base_link:new THREE.Matrix4().makeRotationX(-Math.PI/2)};
const qmap=Object.fromEntries(joints.map(j=>[j.name,j.q]));
let pending=doc.querySelectorAll('joint');
while(pending.length){
 const before=pending.length;
 pending=pending.filter(j=>{
  const parent=j.querySelector('parent').getAttribute('link');if(!expected[parent])return true;
  const child=j.querySelector('child').getAttribute('link');
  const origin=j.querySelector('origin');
  const xyz=(origin?.getAttribute('xyz')||'0 0 0').trim().split(/\s+/).map(Number);
  const [r,p,y]=(origin?.getAttribute('rpy')||'0 0 0').trim().split(/\s+/).map(Number);
  // Independent URDF definition: translation * Rz * Ry * Rx * joint rotation.
  const matrix=new THREE.Matrix4().makeTranslation(...xyz)
    .multiply(new THREE.Matrix4().makeRotationZ(y)).multiply(new THREE.Matrix4().makeRotationY(p)).multiply(new THREE.Matrix4().makeRotationX(r));
  const axis=(j.querySelector('axis')?.getAttribute('xyz')||'0 0 1').trim().split(/\s+/).map(Number);
  matrix.multiply(new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(...axis).normalize(),qmap[j.getAttribute('name')]||0));
  expected[child]=expected[parent].clone().multiply(matrix);
  const actual=renderer._links[child].matrixWorld.elements;
  const error=Math.max(...actual.map((v,i)=>Math.abs(v-expected[child].elements[i])));
  assert.ok(error<1e-10,child+' transform error: '+error);
  return false;
 });
 assert.ok(pending.length<before,'model is not a tree');
}
console.log('PASS: live + cached canvas paths, detail-panel driver ID, actual model fetch, 20 joints/34 links, all URDF transforms vs independent Rz*Ry*Rx FK');

// A narrow monitoring card must frame the entire real robot. Replay data that
// arrived while the asynchronous model request was still in flight.
for (const aspect of [0.5, 2]) {
  const instance = Object.create(ctx.SkeletonRenderer);
  instance._scene = new THREE.Scene(); instance._joints = {}; instance._links = {};
  instance._camera = new THREE.PerspectiveCamera(50, aspect, 0.01, 50);
  instance._controls = {target:new THREE.Vector3(),update(){
    instance._camera.lookAt(this.target); instance._camera.updateMatrixWorld(true);
  }};
  instance._loaded = false; instance._needsFit = true;
  instance.onData(new TextEncoder().encode(JSON.stringify({joints})).buffer);
  await instance._loadModel(id);
  assert.equal(instance._needsFit, false);
  instance._scene.updateMatrixWorld(true);
  for (const joint of joints) {
    const expected = new THREE.Quaternion().setFromAxisAngle(instance._joints[joint.name].userData.axis, joint.q);
    assert.ok(instance._joints[joint.name].quaternion.angleTo(expected) < 1e-7);
  }
  const box = new THREE.Box3().setFromObject(instance._rootGroup);
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
    const projected = new THREE.Vector3(x,y,z).project(instance._camera);
    assert.ok(Math.abs(projected.x) < 1 && Math.abs(projected.y) < 1 && Math.abs(projected.z) < 1);
  }
  const camera = instance._camera.position.clone();
  instance.onData(new TextEncoder().encode(JSON.stringify({joints:joints.map(j=>({...j,q:j.q+0.01}))})).buffer);
  assert.ok(camera.equals(instance._camera.position), 'live updates must preserve user camera');
}
console.log('PASS: early joint frames replayed, full model fits narrow/wide cards, live frames preserve camera');

// Helios-only display simplification must preserve all joint/sensor transforms.
const build=(source,model)=>{
 const context=vm.createContext({THREE,DOMParser,TextDecoder,console});
 vm.runInContext(strip(source).replace('const SkeletonRenderer =','globalThis.SkeletonRenderer ='),context);
 const r=context.SkeletonRenderer;r._scene=new THREE.Scene();r._joints={};r._links={};
 r._parseAndBuild(model);r._applyState({joints});r._scene.updateMatrixWorld(true);return r;
};
const current=await read('renderers/skeleton.js');
const meshes=obj=>obj.children.filter(c=>c.isMesh);
const cylinders=obj=>meshes(obj).filter(m=>m.geometry.type==='CylinderGeometry');
const upper=renderer._heliosUpperBody;
assert.equal(upper.head.geometry.parameters.radius,0.04);
assert.equal(upper.neck.scale.y,0.10);
assert.equal(cylinders(renderer._links.body_link_4).length,0,'no V-shaped mount or chest-camera bones');
assert.equal(cylinders(renderer._links.head_link_1).length,0,'no separate long neck');
assert.equal(cylinders(renderer._links.left_link_1).length,0,'shoulder mount becomes transverse shoulder line');
const point=(mesh,local)=>mesh.localToWorld(local.clone());
const world=link=>link.getWorldPosition(new THREE.Vector3());
const checkUpper=()=>{
 renderer._scene.updateMatrixWorld(true);
 const left=world(renderer._links.left_link_2),right=world(renderer._links.right_link_2);
 assert.ok(point(upper.shoulderLine,new THREE.Vector3()).distanceTo(left)<1e-10);
 assert.ok(point(upper.shoulderLine,new THREE.Vector3(0,1,0)).distanceTo(right)<1e-10);
 const center=left.clone().add(right).multiplyScalar(0.5);
 assert.ok(world(upper.shoulder).distanceTo(center)<1e-10,'spine connects to shoulder midpoint');
 assert.ok(point(upper.spine,new THREE.Vector3(0,1,0)).distanceTo(center)<1e-10);
 assert.ok(world(upper.headFrame).distanceTo(center)<1e-10);
 assert.ok(upper.headFrame.getWorldQuaternion(new THREE.Quaternion()).angleTo(renderer._links.head_link_2.getWorldQuaternion(new THREE.Quaternion()))<1e-7);
 assert.ok(Math.abs(world(upper.head).distanceTo(center)-0.10)<1e-10,'10cm neck above the shoulder center');
};
checkUpper();
assert.equal(cylinders(renderer._links.head_link_2).length,0,'head camera branches are hidden');
for(const name of ['ChestRGBD_link','HeadRGBD_link','RGB1_link','RGB2_link','RGB3_link']){
 assert.ok(renderer._links[name],'sensor frame retained');
 assert.equal(meshes(renderer._links[name]).length,0,'sensor endpoint dot hidden');
}
// Asymmetric live updates move shoulder endpoints; all real link transforms remain exact.
renderer._applyState({joints:joints.map(j=>({...j,q:j.q+0.15}))});
checkUpper();
renderer._applyState({joints});renderer._scene.updateMatrixWorld(true);
// Even matching joint/link names on another robot must retain its previous style.
const otherXml=xml.replace('name="Rokae_Robot_2.1_AR5_v1.1"','name="other_robot"');
const other=build(current,otherXml);
assert.equal(other._heliosUpperBody,null);
for(const name of ['head_joint_1','head_joint_2','left_joint_1','right_joint_1']) {
 assert.ok(meshes(other._joints[name]).some(mesh=>mesh.geometry.type==='SphereGeometry'));
}
for(const name of ['ChestRGBD_link','HeadRGBD_link','RGB1_link','RGB2_link','RGB3_link']) {
 assert.ok(meshes(other._links[name]).length>0,'other robot sensor markers stay visible');
}
console.log('PASS: G1-style transverse shoulders, midpoint spine and short head, live shoulder/head tracking, no V-shaped mount bones, exact real transforms and other robot rendering retained');
