import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {XrPanel} from '../plugins/teleop/webxr/view.mjs';

// Exercise app.mjs's actual entry/visibility handlers. TLS, DOM, XR and GL are
// fixtures here; the PICO session transition was observed in a USB device test.
test('PICO entry visibility transitions wait for tracking; subsequent focus loss closes input',async()=>{
  const originals=new Map();
  const install=(key,value)=>{originals.set(key,Object.getOwnPropertyDescriptor(globalThis,key));Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});};
  const events=new EventTarget(),elements=new Map();
  const element=id=>{
    if(!elements.has(id))elements.set(id,Object.assign(new EventTarget(),{hidden:false,disabled:false,textContent:'',getContext:kind=>kind==='webgl'?gl:{}}));
    return elements.get(id);
  };
  let compatible,compatibilityRequested=false;
  const gate=new Promise(resolve=>compatible=resolve);
  const gl=new Proxy({}, {get:(_,key)=>{
    if(key==='makeXRCompatible')return()=>{compatibilityRequested=true;return gate;};
    if(key==='getShaderParameter'||key==='getProgramParameter')return()=>true;
    if(String(key)===String(key).toUpperCase())return 1;
    return()=>({});
  }});
  const doc=Object.assign(new EventTarget(),{hidden:false,getElementById:element,createElement:()=>element('offscreen')});
  const certificate=Buffer.from('fixture certificate identity'),device=createHash('sha256').update(certificate).digest('hex');
  const saved={device_id:device,credentials:{type:'credential',capture_id:'11111111-1111-4111-8111-111111111111',capture_credential:'a'.repeat(64)}};
  const sockets=[];
  class Socket {
    readyState=1;bufferedAmount=0;messages=[];
    constructor(){sockets.push(this);}
    send(raw){this.messages.push(JSON.parse(raw));}
    close(){this.readyState=3;}
  }
  const session=Object.assign(new EventTarget(),{
    environmentBlendMode:'alpha-blend',visibilityState:'visible',inputSources:[],ended:false,
    renderState:{},updateRenderState(value){this.renderState=value;},
    async requestReferenceSpace(){return new EventTarget();},
    requestAnimationFrame(callback){this.frame=callback;},
    async end(){this.ended=true;this.dispatchEvent(new Event('end'));},
  });
  const visibility=state=>{session.visibilityState=state;session.dispatchEvent(new Event('visibilitychange'));};
  const settle=async(predicate)=>{
    for(let n=0;n<100;n++){if(predicate())return;await new Promise(resolve=>setTimeout(resolve,5));}
    assert.fail('app did not reach expected state');
  };
  const draw=XrPanel.prototype.draw;
  try {
    install('document',doc);install('addEventListener',events.addEventListener.bind(events));
    install('location',new URL('https://robot.test/webxr/'));install('isSecureContext',true);
    install('localStorage',{getItem:()=>JSON.stringify(saved),setItem(){},removeItem(){}});
    install('WebSocket',Socket);
    install('navigator',{xr:{isSessionSupported:async()=>true,requestSession:async()=>session}});
    install('XRWebGLLayer',class {constructor(){this.framebuffer=null;}});
    install('fetch',async()=>({ok:true,json:async()=>({schema:'motus.teleop.webxr.v1',origin:'https://robot.test',
      wss_url:'wss://robot.test/ws/teleop-capture',device_id:device,certificate_der_base64:certificate.toString('base64')})}));
    XrPanel.prototype.draw=()=>{};
    await import('../plugins/teleop/webxr/app.mjs');
    await settle(()=>element('support').textContent.includes('浏览器支持')&&!element('reconnect').disabled);
    element('reconnect').onclick();const socket=sockets[0];socket.onopen();
    socket.onmessage({data:JSON.stringify({type:'connected',capture_protocol:'motus.teleop.capture.v1',frame_protocol:'motus.teleop.rtc-frame.v1',
      capture_id:saved.credentials.capture_id,presence_interval_ms:250,presence_timeout_ms:1000})});
    assert.equal(element('enter').disabled,false);
    const entering=element('enter').onclick();
    await settle(()=>compatibilityRequested);
    visibility('hidden');visibility('visible-blurred');
    assert.equal(socket.readyState,1,'entry transition must not close an unused capture');
    assert.equal(session.ended,false);
    compatible();await entering;
    const staleFrame={getViewerPose(){throw Error('hidden frame must not be sampled');}};
    const first=session.frame;session.frame=null;first(performance.now(),staleFrame);
    assert.equal(typeof session.frame,'function','wait for the next visible frame');
    assert.equal(socket.messages.some(m=>m.type==='presence'&&m.state==='xr_standby'),false);
    visibility('visible');
    const pose={emulatedPosition:false,transform:{position:{x:0,y:1,z:0},orientation:{x:0,y:0,z:0,w:1}}};
    session.inputSources=['left','right'].map(handedness=>({handedness,targetRayMode:'tracked-pointer',gripSpace:{},targetRaySpace:{},
      gamepad:{mapping:'xr-standard',connected:true,axes:[],buttons:[{value:0,pressed:false},{value:0,pressed:false}]}}));
    session.frame(performance.now(),{getViewerPose:()=>pose,getPose:space=>space===session.inputSources[0].targetRaySpace||space===session.inputSources[1].targetRaySpace?null:pose});
    assert.equal(socket.messages.some(m=>m.type==='presence'&&m.state==='xr_standby'),true);
    assert.equal(socket.messages.some(m=>m.type==='operator_command'),false,'tracking does not implicitly start motion');
    visibility('visible-blurred');
    assert.equal(socket.readyState,3,'after tracking starts, blur must close input');
    assert.equal(session.ended,true);
  }finally{
    events.dispatchEvent(new Event('pagehide'));compatible();XrPanel.prototype.draw=draw;
    for(const [key,descriptor] of originals){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}
  }
});
