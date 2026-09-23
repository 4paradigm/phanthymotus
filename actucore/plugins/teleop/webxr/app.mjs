import {CaptureClient} from './capture.mjs';
import {frameSample} from './frame.mjs';
import {XrPanel,panelPoint,panelAction} from './view.mjs';

const $=id=>document.getElementById(id), storageKey='motus.webxr.capture.v1';
let config,client,saved,session,panel,space,lastHead,pairEpoch=0,confirmed=false,pairing=false,supported=false;
const status=text=>{$('status').textContent=text;};
const bytes=value=>Uint8Array.from(atob(value),c=>c.charCodeAt(0));
const base64=value=>btoa(String.fromCharCode(...new Uint8Array(value)));
const digest=async value=>new Uint8Array(await crypto.subtle.digest('SHA-256',value));
const hex=value=>Array.from(value,n=>n.toString(16).padStart(2,'0')).join('');
const update=()=>{
  $('pair').disabled=!config||!supported||pairing||!!saved||!!client?.socket;
  $('reconnect').hidden=!saved;$('reconnect').disabled=!supported||!!client?.socket;
  $('enter').disabled=!supported||!client?.authenticated||!!session;
  $('disconnect').disabled=!client?.socket&&!pairing;
};
async function post(operation,body) {
  const abort=new AbortController(),timer=setTimeout(()=>abort.abort(),5000);
  try {
    const response=await fetch('/pairing/'+operation,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body),credentials:'omit',cache:'no-store',redirect:'error',signal:abort.signal});
    if(!response.ok)throw Error(response.status===403?'配对窗口已关闭或邀请失效，请在电脑端重新允许配对':'配对未完成，请检查电脑端是否已有配对或正在连接');
    return await response.json();
  }finally{clearTimeout(timer);}
}
function endXr() {
  const current=session;session=null;space=null;lastHead=null;
  if(current)current.end().catch(()=>{});
  if(panel){panel.dispose();panel=null;}update();
}
function connect(credentials) {
  if(client?.socket)throw Error('请先断开已有连接');
  client=new CaptureClient({url:config.wss_url,onState:c=>{status(c.status);update();},onLoss:endXr,
    onCredentials:credentials=>{
      saved={device_id:config.device_id,credentials};
      try{localStorage.setItem(storageKey,JSON.stringify(saved));}
      catch{status('当前浏览器无法保存配对，请勿关闭本页；下次需重新配对。');}
      update();
    }});
  client.connect(credentials);update();
}
function stopPairing(){pairEpoch++;pairing=false;confirmed=false;$('fingerprint').hidden=true;$('confirm').hidden=true;update();}
async function pair() {
  if(pairing||saved||client?.socket)return;
  const epoch=++pairEpoch;pairing=true;confirmed=false;update();status('正在申请配对，请保持电脑端配对窗口打开…');
  try {
    const key=await crypto.subtle.generateKey({name:'ECDSA',namedCurve:'P-256'},true,['sign','verify']);
    const publicKey=await crypto.subtle.exportKey('spki',key.publicKey),nonce=crypto.getRandomValues(new Uint8Array(32));
    const pending=await post('request',{device_name:'PICO WebXR',public_key:base64(publicKey),nonce:base64(nonce)});
    if(epoch!==pairEpoch)return;
    const serverNonce=bytes(pending.server_nonce);if(serverNonce.length!==32)throw Error('配对响应无效');
    const parts=[new TextEncoder().encode('motus-enrollment-v1\0'),await digest(bytes(config.certificate_der_base64)),await digest(publicKey),nonce,serverNonce];
    const transcript=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let offset=0;for(const p of parts){transcript.set(p,offset);offset+=p.length;}
    const fingerprint=hex(await digest(transcript)).slice(0,32).toUpperCase();
    if(fingerprint!==pending.fingerprint)throw Error('配对身份校验失败');
    if(epoch!==pairEpoch)return;
    $('fingerprint').textContent=fingerprint.match(/.{8}/g).join(' ');$('fingerprint').hidden=false;$('confirm').hidden=false;
    status('请核对电脑端配对码，两端一致后确认。');
    const deadline=performance.now()+120000;
    while(epoch===pairEpoch&&performance.now()<deadline) {
      const result=await post('poll',{request_id:pending.request_id,ticket:pending.ticket,confirm:confirmed,fingerprint});
      if(epoch!==pairEpoch)return;
      if(result.state==='approved') {
        if(result.wss_url!==config.wss_url)throw Error('配对地址不匹配');
        connect({type:'pair',pairing_id:result.pairing_id,pairing_code:result.pairing_code});return;
      }
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
    if(epoch===pairEpoch)throw Error('配对已超时，请在电脑端重新允许配对');
  }catch(error){if(epoch===pairEpoch)status(error.message);}
  finally{if(epoch===pairEpoch)stopPairing();}
}

$('pair').onclick=pair;
$('confirm').onclick=()=>{confirmed=true;$('confirm').hidden=true;status('已确认，等待电脑端批准…');};
$('reconnect').onclick=()=>{try{connect(saved.credentials);}catch(error){status(error.message);}};
$('disconnect').onclick=()=>{stopPairing();client?.fail('已断开连接，输入已停止');status('已断开连接');};
$('forget').onclick=()=>{
  if(!globalThis.confirm('忘记本机配对？电脑端也需要撤销旧配对，之后才能重新申请。'))return;
  stopPairing();client?.fail('已忘记配对');saved=null;try{localStorage.removeItem(storageKey);}catch{}update();
};
$('enter').onclick=async()=>{
  const activeClient=client,epoch=client?.epoch;
  try {
    // Request inside the user's click; do not await network before this call.
    const pending=navigator.xr.requestSession('immersive-ar',{requiredFeatures:['local-floor']});
    $('enter').disabled=true;
    const current=await pending;
    if(client!==activeClient||client.epoch!==epoch||!client.authenticated){await current.end();return;}
    if(current.environmentBlendMode==='opaque'){await current.end();throw Error('透视不可用');}
    session=current;
    current.addEventListener('end',()=>{if(session===current)client.fail('已退出透视，输入已停止');});
    current.addEventListener('inputsourceschange',()=>{if(session===current&&client.focused)client.fail('手柄输入源已变化，请重新进入并开始');});
    current.addEventListener('visibilitychange',()=>{if(session===current&&current.visibilityState!=='visible')client.focus(false);});
    panel=new XrPanel($('xr'));await panel.gl.makeXRCompatible();
    if(session!==current)return;
    current.updateRenderState({baseLayer:new XRWebGLLayer(current,panel.gl,{alpha:true})});
    space=await current.requestReferenceSpace('local-floor');
    if(session!==current)return;
    space.addEventListener('reset',()=>{if(session===current)client.fail('空间坐标已重置，请重新进入并开始');});
    current.addEventListener('select',event=>{
      if(session!==current||current.visibilityState!=='visible'||!lastHead||!space)return;
      try{
        const ray=event.frame.getPose(event.inputSource.targetRaySpace,space),action=panelAction(panelPoint(ray,lastHead));
        if(!action)return;
        if(action==='start'&&Array.from(current.inputSources).some(s=>s.gamepad?.buttons[1]?.value>=.75))return;
        client.command(action);
      }catch{client.fail('手柄输入异常，请重新进入');}
    });
    const loop=(time,frame)=>{
      if(session!==current)return;
      try {
        if(current.visibilityState!=='visible'){client.focus(false);return;}
        const sample=frameSample(frame,space,current);lastHead=frame.getViewerPose(space);
        if(!client.focused&&sample.head&&sample.left&&sample.right)client.focus(true);
        if(client.focused)client.submit(sample);
        if(session!==current)return;
        const pointers=Array.from(current.inputSources).map(s=>panelPoint(frame.getPose(s.targetRaySpace,space),lastHead)).filter(Boolean);
        panel.draw(current,lastHead,client,pointers);current.requestAnimationFrame(loop);
      }catch{client.fail('头显跟踪或显示异常，输入已停止');}
    };
    status('请拿起双手柄，在透视面板点击开始遥操');update();current.requestAnimationFrame(loop);
  }catch(error){client?.fail('未能进入透视，请检查浏览器权限和设备支持');endXr();}
};
$('xr').addEventListener('webglcontextlost',event=>{event.preventDefault();client?.fail('透视显示中断，输入已停止');});
addEventListener('pagehide',()=>{stopPairing();client?.fail('页面已关闭，输入已停止');});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopPairing();client?.fail('页面已切换，输入已停止');}});

async function initialize() {
  if(!isSecureContext||location.protocol!=='https:'){status('请使用维护者提供的可信 HTTPS 地址打开');$('support').textContent='当前页面无法安全启用头显跟踪';return;}
  try {
    const response=await fetch('/webxr/config',{credentials:'omit',cache:'no-store',redirect:'error'});
    if(!response.ok)throw Error('无法读取机器人连接信息');
    const value=await response.json(),url=new URL(value.wss_url);
    if(value.schema!=='motus.teleop.webxr.v1'||value.origin!==location.origin||
       url.protocol!=='wss:'||url.host!==location.host||url.pathname!=='/ws/teleop-capture'||url.search||url.hash||url.username||url.password||
       typeof value.certificate_der_base64!=='string'||value.certificate_der_base64.length>43692||
       hex(await digest(bytes(value.certificate_der_base64)))!==value.device_id)throw Error('机器人地址或身份不匹配，请使用维护者提供的地址');
    config=value;
    try{saved=JSON.parse(localStorage.getItem(storageKey)||'null');}catch{saved=null;}
    if(saved&&(saved.device_id!==config.device_id||saved.credentials?.type!=='credential'||
       typeof saved.credentials.capture_credential!=='string'||saved.credentials.capture_credential.length<32)){
      saved=null;config=null;throw Error('机器人身份或本机记录已变化，请确认后忘记旧配对，再刷新页面');
    }
    status(saved?'已保存配对，点击连接后进入透视':'请先在电脑端允许新设备配对，再点击申请配对');
    supported=!!navigator.xr&&await navigator.xr.isSessionSupported('immersive-ar');
    $('support').textContent=supported?'浏览器支持透视；进入后检查双手柄跟踪':'当前浏览器不支持透视遥操，请使用支持 WebXR 的 PICO 浏览器';update();
  }catch(error){status(error.message);update();}
}
initialize();
