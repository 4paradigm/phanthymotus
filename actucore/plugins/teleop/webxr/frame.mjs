/* Public RTC Frame v1 only. No lease, fence, robot joints or private authority. */
const finite = (n, min, max) => Number.isFinite(n) && n >= min && n <= max;

export function poseOf(pose) {
  if (!pose || pose.emulatedPosition || !pose.transform) return null;
  const {position:p, orientation:q} = pose.transform;
  if (!p || !q) return null;
  const position = [p.x,p.y,p.z], orientation = [q.x,q.y,q.z,q.w];
  if (!position.every(n=>finite(n,-100,100)) || !orientation.every(n=>finite(n,-1.000001,1.000001))) return null;
  const norm = Math.hypot(...orientation);
  if (!finite(norm,.5,1.5)) return null;
  return {position,orientation:orientation.map(n=>n/norm)};
}

export function controllerOf(source, pose, hand) {
  const gamepad=source?.gamepad;
  if (!source || source.handedness!==hand || source.hand || source.targetRayMode!=='tracked-pointer' ||
      !source.gripSpace || !gamepad || gamepad.mapping!=='xr-standard' || gamepad.connected===false ||
      gamepad.buttons.length<2 || gamepad.buttons.length>16 || gamepad.axes.length>8) return null;
  const axes=Array.from(gamepad.axes), buttons=Array.from(gamepad.buttons,b=>b.value), normalized=poseOf(pose);
  if (!normalized || !axes.every(n=>finite(n,-1,1)) || !buttons.every(n=>finite(n,0,1))) return null;
  const pressed=gamepad.buttons[1].pressed===true;
  // A changing/inconsistent squeeze is neither a confirmed release nor a press.
  const squeeze=pressed && buttons[1]>=.75 ? 'pressed' : !pressed && buttons[1]<.75 ? 'released' : 'transition';
  // RTC Frame v1's native producer sends trigger and squeeze only. PICO also
  // exposes thumbstick and face buttons through xr-standard; omit those extras.
  return {pose:normalized,wire:{axes,buttons:buttons.slice(0,2)},squeeze};
}

export function frameSample(frame, space, session) {
  const sources=Array.from(session.inputSources);
  const left=sources.filter(s=>s.handedness==='left'), right=sources.filter(s=>s.handedness==='right');
  const read=(items,hand)=>items.length===1 ? controllerOf(items[0],
    items[0].gripSpace ? frame.getPose(items[0].gripSpace,space) : null,hand) : null;
  return {head:poseOf(frame.getViewerPose(space)),left:read(left,'left'),right:read(right,'right')};
}

export function validateAssignment(a) {
  const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
  if (!a || !uuid.test(a.id) || !uuid.test(a.session_id) || !Number.isSafeInteger(a.generation) || a.generation<1 ||
      !['shadow','live'].includes(a.mode) || a.state!=='issued' || !/^[0-9a-f]{64}$/.test(a.capability_digest)) throw Error('任务资料无效');
  const c=a.capabilities;
  if (!c || c.profile_id!==a.profile_id || !c.outputs || !c.input_bindings || !Array.isArray(c.effectors) ||
      !Array.isArray(a.effectors) || JSON.stringify(c.effectors)!==JSON.stringify(a.effectors)) throw Error('任务能力不匹配');
  const enabled=[];
  for (const [name,output] of Object.entries(c.outputs)) {
    if (typeof output?.enabled!=='boolean') throw Error('任务输出无效');
    if (output.enabled) {
      if (!['dual_arm','hands','end_effectors'].includes(name) ||
          (name==='end_effectors' && output.mode!=='eef_pose')) throw Error('网页版当前仅支持双臂和手部遥操');
      enabled.push(name);
    }
  }
  if (c.input_bindings.base_twist || new Set(c.effectors).size!==c.effectors.length ||
      enabled.sort().join(',')!==[...c.effectors].sort().join(',')) throw Error('任务能力不支持');
  for (const [name,binding] of Object.entries(c.input_bindings)) {
    if (!['head','left_controller','right_controller'].includes(name) || typeof binding?.required!=='boolean') throw Error('输入能力不支持');
  }
  return a;
}

export class FrameEncoder {
  constructor() { this.sequence=0;this.clutch=0;this.ns=0;this.release(); }
  release() { this.rearm=true;this.deadman=false; }
  encode(sample, mode, now, allowed=true) {
    if (!['shadow','live'].includes(mode) || !Number.isFinite(now) || now<0) throw Error('输入时间或模式无效');
    const {head,left,right}=sample, tracked=!!(head&&left&&right);
    const pressed=tracked && left.squeeze==='pressed' && right.squeeze==='pressed';
    const released=tracked && (left.squeeze==='released' || right.squeeze==='released');
    const neutral=tracked && left.wire.buttons[0]<=.05 && right.wire.buttons[0]<=.05;
    if (!allowed || !tracked) this.rearm=true;
    else if (released) this.rearm=!neutral;
    else if ((this.deadman&&!pressed) || (!this.deadman&&pressed&&!neutral)) this.rearm=true;
    const deadman=allowed&&tracked&&pressed&&!this.rearm;
    if (deadman&&!this.deadman) this.clutch++;
    this.deadman=!!deadman;this.ns=Math.max(this.ns+1,Math.floor(now*1e6));
    if (![this.sequence+1,this.clutch,this.ns].every(Number.isSafeInteger)) throw Error('输入计数已耗尽，请结束会话');
    return {schema_version:1,sequence:this.sequence++,client_monotonic_ns:this.ns,mode,
      deadman:this.deadman,clutch_sequence:this.clutch,
      tracking:{head:!!head,left_controller:!!left,right_controller:!!right},
      head:head||null,left_controller:left?.pose||null,right_controller:right?.pose||null,
      controllers:{left:left?.wire||{axes:[],buttons:[]},right:right?.wire||{axes:[],buttons:[]}}};
  }
}
