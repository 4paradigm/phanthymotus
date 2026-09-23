import test from 'node:test';
import assert from 'node:assert/strict';
import {FrameEncoder,poseOf,controllerOf,frameSample,validateAssignment} from '../plugins/teleop/webxr/frame.mjs';
import {CaptureClient} from '../plugins/teleop/webxr/capture.mjs';
import {panelPoint,panelAction} from '../plugins/teleop/webxr/view.mjs';

const pose={transform:{position:{x:0,y:1,z:0},orientation:{x:0,y:0,z:0,w:1}},emulatedPosition:false};
const input=(hand='left',grip=0,trigger=0)=>({handedness:hand,targetRayMode:'tracked-pointer',gripSpace:{},
  gamepad:{connected:true,mapping:'xr-standard',axes:[0,0,0,0],buttons:[{value:trigger,pressed:trigger>.5},{value:grip,pressed:grip>=.75}]}});
const sample=(grip=0,trigger=0)=>({head:poseOf(pose),left:controllerOf(input('left',grip,trigger),pose,'left'),right:controllerOf(input('right',grip,trigger),pose,'right')});
const assignment=()=>({id:'11111111-1111-4111-8111-111111111111',session_id:'22222222-2222-4222-8222-222222222222',
  generation:1,mode:'shadow',state:'issued',profile_id:'test',capability_digest:'a'.repeat(64),effectors:['end_effectors'],
  capabilities:{profile_id:'test',effectors:['end_effectors'],outputs:{end_effectors:{enabled:true,mode:'eef_pose'}},
    input_bindings:{head:{required:true,role:'reference'},left_controller:{required:true,role:'left_end_effector'},right_controller:{required:true,role:'right_end_effector'}}}});

test('initial held grips, neutral release, trigger rearm and tracking recovery',()=>{
  const e=new FrameEncoder();
  assert.equal(e.encode(sample(1),'shadow',1).deadman,false);
  assert.equal(e.encode(sample(),'shadow',2).deadman,false);
  assert.equal(e.encode(sample(1),'shadow',3).deadman,true);
  assert.equal(e.encode(sample(1,.8),'shadow',4).deadman,true);
  assert.equal(e.encode(sample(0,.8),'shadow',5).deadman,false);
  assert.equal(e.encode(sample(1),'shadow',6).deadman,false);
  e.encode(sample(),'shadow',7);const enabled=e.encode(sample(1),'shadow',8);
  assert.equal(enabled.deadman,true);assert.equal(enabled.clutch_sequence,2);
  assert.equal(e.encode({...sample(1),left:null},'shadow',9).deadman,false);
  assert.equal(e.encode(sample(1),'shadow',10).deadman,false);
  e.encode(sample(),'shadow',11);assert.equal(e.encode(sample(1),'shadow',12).deadman,true);
  e.release();assert.equal(e.encode(sample(1),'shadow',13).deadman,false);
});
test('stopped input cannot rearm until an explicit start and neutral release',()=>{
  const e=new FrameEncoder();e.encode(sample(),'live',1);
  assert.equal(e.encode(sample(1),'live',2).deadman,true);
  e.encode(sample(),'live',3,false);assert.equal(e.encode(sample(1),'live',4).deadman,false);
  e.encode(sample(),'live',5);assert.equal(e.encode(sample(1),'live',6).deadman,true);
  assert.equal(e.encode(sample(),'live',0).client_monotonic_ns,6000001);
});
test('emulated, untracked, wrong-profile, bare-hand and duplicate inputs are invalid',()=>{
  assert.equal(poseOf({...pose,emulatedPosition:true}),null);
  assert.equal(poseOf({transform:{...pose.transform,position:{x:NaN,y:0,z:0}}}),null);
  for(const mutate of [s=>s.gamepad.mapping='',s=>s.hand={},s=>s.gripSpace=null,s=>s.gamepad.buttons[0].value=NaN,
    s=>s.gamepad.axes[0]=2,s=>s.handedness='none']){
    const s=input();mutate(s);assert.equal(controllerOf(s,pose,'left'),null);
  }
  const frame={getViewerPose:()=>pose,getPose:()=>pose};
  const result=frameSample(frame,{}, {inputSources:[input(),input(),input('right')]});assert.equal(result.left,null);
});
test('assignment accepts current EEF producer and refuses mobile/unknown modes',()=>{
  assert.equal(validateAssignment(assignment()).mode,'shadow');
  for(const change of [a=>a.mode='automatic',a=>a.capabilities.outputs.base={enabled:true},
    a=>a.capabilities.input_bindings.base_twist={},a=>a.effectors=['wrong'],a=>a.generation=0]){
    const a=assignment();change(a);assert.throws(()=>validateAssignment(a));
  }
});

test('PICO six-button xr-standard input matches the native trigger/squeeze contract',()=>{
  const source=input('left',1,.2);
  source.gamepad.axes=[0,0,.1,-.2];
  source.gamepad.buttons.push(...Array.from({length:4},()=>({value:1,pressed:true})));
  const normalized=controllerOf(source,pose,'left');
  assert.deepEqual(normalized.wire,{axes:[0,0,.1,-.2],buttons:[.2,1]});
  assert.equal(normalized.squeeze,'pressed');
  assert.equal(source.gamepad.buttons.length,6);
});

class Socket {
  readyState=1;bufferedAmount=0;messages=[];
  send(raw){this.messages.push(JSON.parse(raw));}
  close(){this.readyState=3;}
}
function client() {
  let now=100;const losses=[];
  const c=new CaptureClient({url:'wss://robot.test/ws/teleop-capture',Socket,now:()=>now,onLoss:()=>losses.push(true)});
  c.connect({type:'credential'});c.authenticated=true;c.lastAck=now;c.focused=true;c.lastFrame=now;
  return {c,losses,setNow:n=>{now=n;}};
}
test('focus loss fences output before asynchronous socket close',()=>{
  const {c,losses}=client();const socket=c.socket;c.allowed=true;c.focus(false);
  assert.equal(c.allowed,false);assert.equal(c.socket,null);assert.equal(socket.readyState,3);assert.equal(losses.length,1);
});
test('pose backpressure closes connection without queueing a stale frame',()=>{
  const {c}=client();let sends=0;c.assignment=assignment();c.allowed=true;
  c.pose={readyState:'open',bufferedAmount:1,send:()=>sends++};c.control={readyState:'open'};
  c.submit(sample(1));assert.equal(sends,0);assert.equal(c.socket,null);
});
test('XR frame stall fences output even with fresh presence acknowledgements',()=>{
  const {c,setNow}=client();setNow(351);c.lastAck=350;c.tick();assert.equal(c.socket,null);
});
test('start requires fresh armed status; stop remains available during a pending start',()=>{
  const {c}=client();c.operatorControl={connection_id:assignment().id};c.operator={armed:false};c.operatorAt=100;
  assert.equal(c.command('start'),false);c.operator={armed:true};
  assert.equal(c.command('start'),true);const first=c.pending.id;
  assert.equal(c.command('stop'),true);assert.notEqual(c.pending.id,first);assert.equal(c.allowed,false);
  c.fail();
});
test('delayed offer after disconnect cannot send on a later connection',async()=>{
  let resolve;const gate=new Promise(r=>resolve=r);
  class Peer {
    iceGatheringState='complete';localDescription={sdp:'fixture'};
    createDataChannel(){return {readyState:'connecting'};}
    createOffer(){return gate;}
    async setLocalDescription(){}
    close(){}
  }
  const {c}=client();c.Peer=Peer;const socket=c.socket;
  const running=c.message({type:'assignment',assignment:assignment()});c.fail();resolve({type:'offer',sdp:'fixture'});await running;
  assert.equal(socket.messages.some(m=>m.type==='signaling_offer'),false);
});
test('assignment revocation and stale operator status never preserve local enable',async()=>{
  const {c}=client();c.assignment=assignment();c.allowed=true;
  await c.message({type:'assignment_revoked',assignment_id:c.assignment.id});assert.equal(c.allowed,false);
  c.operatorControl={connection_id:assignment().id};c.allowed=true;
  await c.message({type:'visualization',visualization:{operator:{armed:false}}});assert.equal(c.allowed,false);c.fail();
});
test('head-relative panel selects only visible button bounds',()=>{
  const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  const ray=[...identity];ray[12]=-.35;ray[13]=-.1;
  const point=panelPoint({transform:{matrix:ray}}, {transform:{inverse:{matrix:identity}}});
  assert.equal(panelAction(point),'start');assert.equal(panelAction({x:10,y:200}),null);
  ray[10]=-1;assert.equal(panelPoint({transform:{matrix:ray}}, {transform:{inverse:{matrix:identity}}}),null);
});

test('first frame after an event-loop stall cannot clear the stall watchdog',()=>{
  const {c,setNow}=client();setNow(351);c.submit(sample(1));assert.equal(c.socket,null);
});
