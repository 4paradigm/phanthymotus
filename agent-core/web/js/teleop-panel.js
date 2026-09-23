/** Teleop controls use the registered MCP only. No robot addresses or secrets. */
const STATES = {idle:'未开始',armed:'等待 PICO 开始',ready:'等待双握把',prepared:'等待双握把',prepared_shadow:'等待双握把',active:'遥操中',active_shadow:'模拟中',hold:'保持',fault:'故障',returning:'正在收臂',starting:'正在准备',stopping:'正在停止'};
const REASONS = {deadman_released:'握把已松开',operator_pause:'已暂停',ik_target_unreachable:'目标不可达，保持最后有效位置',ik_timeout:'求解超时，等待有效输入',command_timeout:'指令中断，已保持',command_expired:'指令过期，已保持',power_ns_stale:'电源状态过期',arm_ns_stale:'关节反馈过期',stop_unconfirmed:'停止尚未确认',invalid_lease:'控制权失效',stop_confirmed:'已确认停止',operator_release:'会话已结束'};
export function viewState(info, fresh=true, projectManaged=false) {
  const d=info?.driver_feedback, op=info?.operator || {}, project=info?.project || {};
  const armed=project.armed ?? info?.armed ?? false;
  const driverFresh=fresh && info?.driver_feedback_fresh===true;
  const executing=driverFresh && d?.output_active===true;
  const state=project.stopping?'returning':project.error?'fault':op.state==='error'?'fault':['starting','returning','stopping'].includes(op.state)?op.state:armed && !info?.authority_valid?'armed':info?.state;
  const reason=project.error || op.error || info?.host_error || (state==='fault'?d?.reason:null) || info?.reason || d?.reason;
  return {state, armed, executing, driverFresh, connected:fresh && info?.capture?.connected===true,
    title:!fresh?'状态不可用':STATES[state] || state || '等待状态',
    mode:info?.mode==='live'?'Live · 真机':info?.mode==='shadow'?'Shadow · 模拟':'模式未知',
    execution:!driverFresh?'执行状态未知':executing?'硬件正在执行':d?.ownership_held?'持有控制权 · 未输出':'无硬件输出',
    reason:!fresh?'状态读取失败，不能确认机器人是否停止':REASONS[reason] || reason || (projectManaged?(armed?'请在 PICO 点击开始遥操':'连好 Driver 并开启智能控制，之后在 PICO 点击开始'):'独立卡片模式：未支持项目托管收臂；先松开双握把再开始'),
    busy:project.stopping===true || ['starting','returning','stopping'].includes(op.state)};
}
export function mountTeleopPanel(host,{call,actions=[],configSchema={},loadConfig,saveConfig,loadTargets,buildTemplate,prepareInstallation,createInvitation,prepareWebxr}) {
  const supported=new Set(actions), buttons=new Map();
  let info=null, fresh=false, pending=null, busy=false, timer=null, disposed=false, error='', readError='', notice='', request=0, operationPending=null;
  const panel=document.createElement('section');panel.className='teleop-panel';panel.setAttribute('aria-label','遥操控制');
  panel.innerHTML=`<style>
.teleop-panel{font:12px/1.5 system-ui,sans-serif;color:var(--text-primary,#e4e8f1);padding:12px;border:1px solid #53607844;border-radius:12px;background:#1922310a;min-width:260px;max-width:360px;box-sizing:border-box}
.teleop-panel *{box-sizing:border-box}.teleop-panel header{display:flex;align-items:center;justify-content:space-between;gap:10px}.teleop-panel h3{margin:0;font-size:16px}.teleop-panel .tp-mode{font-size:10px;padding:3px 7px;border-radius:12px;background:#73839922}.teleop-panel[data-live=true] .tp-mode{color:#e5a43e;background:#e5a43e18}
.teleop-panel .tp-install-link{color:#8cb9ff}.teleop-panel .tp-package{font-size:11px;overflow-wrap:anywhere}.teleop-panel .tp-topology select{font:inherit;max-width:60%;padding:6px;color:inherit;background:#273247;border:1px solid #73839955;border-radius:6px}.teleop-panel .tp-install-qr[hidden]{display:none!important}
.teleop-panel .tp-execution{margin:8px 0;font-weight:650}.teleop-panel[data-executing=true] .tp-execution{color:#35b884}.teleop-panel .tp-reason{color:var(--text-secondary,#8997ac);min-height:36px;overflow-wrap:anywhere}.teleop-panel .tp-actions{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0}.teleop-panel button{font:inherit;cursor:pointer;border:1px solid #73839955;border-radius:7px;padding:7px 9px;background:#73839910;color:inherit}.teleop-panel button:hover:not(:disabled){background:#73839930}.teleop-panel button:disabled{opacity:.4;cursor:not-allowed}.teleop-panel [data-action=start]{background:#3478f6;color:white;border-color:#3478f6}.teleop-panel [data-action=stop]{color:#ef7272;border-color:#ef727277;grid-column:1/-1}.teleop-panel dl{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin:10px 0}.teleop-panel dt{color:var(--text-secondary,#8997ac)}.teleop-panel dd{margin:0;text-align:right;overflow-wrap:anywhere}.teleop-panel details{border-top:1px solid #73839933;padding-top:9px;margin-top:9px}.teleop-panel summary{cursor:pointer;font-weight:600}.teleop-panel .tp-pair-buttons{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.teleop-panel .tp-fingerprint{font:12px/1.7 monospace;white-space:pre-wrap;word-break:break-all}.teleop-panel .tp-error{color:#ef7272;overflow-wrap:anywhere}.teleop-panel .tp-notice{color:#8997ac;overflow-wrap:anywhere}.teleop-panel pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.5 monospace;max-height:200px;overflow:auto}
</style><header><h3 data-field="title">读取状态…</h3><span class="tp-mode" data-field="mode"></span></header>
<div class="tp-execution" data-field="execution"></div><div class="tp-reason" data-field="reason"></div>
<div class="tp-actions"></div><div class="tp-error" role="alert"></div><div class="tp-notice" role="status"></div>
<dl><dt>连接 Driver</dt><dd data-field="driver"></dd><dt>智能控制</dt><dd data-field="armed"></dd><dt>PICO 连接</dt><dd data-field="connected"></dd><dt>跟踪</dt><dd data-field="tracking"></dd><dt>输入时效</dt><dd data-field="age"></dd><dt>标定</dt><dd data-field="calibration"></dd></dl>
<details class="tp-topology" hidden><summary>连接机器人卡片</summary><p>遥操 → 运动控制 → 双臂执行；虚线自动返回求解与执行反馈。</p><select aria-label="目标机器人"></select><button type="button">建立三段连线</button><p class="tp-topology-result" role="status"></p></details>
<details class="tp-install" hidden><summary>安装 PICO 与一键连接</summary><p>在 PICO 浏览器打开短地址下载应用；生成连接邀请后，资料会自动预填。</p><button type="button" class="tp-download">生成安装链接</button> <button type="button" class="tp-invite" disabled>生成一次性连接邀请</button> <button type="button" class="tp-revoke-invite" hidden>撤销邀请</button><p class="tp-package"></p><a class="tp-install-link" target="_blank" rel="noreferrer" style="overflow-wrap:anywhere"></a><img class="tp-install-qr" alt="PICO 安装与连接二维码" hidden width="220" height="220" style="display:block;background:white;margin:8px auto"><p class="tp-install-result" role="status"></p></details>
<details class="tp-pair"><summary>连接与配对</summary><p data-field="pairing"></p><div class="tp-fingerprint"></div><div class="tp-pair-buttons"></div></details>
<details class="tp-config"><summary>服务配置</summary><p>结束会话后保存并应用。保存不启动机器人；重新开始前需标定。</p><form><div class="tp-config-fields"></div><button type="submit">保存并应用配置</button><p class="tp-config-result" role="status"></p></form></details>
<details><summary>诊断与维护</summary><div class="tp-maintenance tp-pair-buttons"></div><pre></pre></details>`;
  host.append(panel);
  const field=(key,value)=>{panel.querySelector(`[data-field="${key}"]`).textContent=value;};
  const has=a=>supported.has(a);
  const projectManaged=has('project_start') && has('project_stop');
  panel.querySelector('.tp-pair').insertAdjacentHTML('beforebegin',`<details class="tp-webxr" hidden><summary>浏览器遥操 · 试验版</summary><p>在 PICO 浏览器打开入口，无需 APK。需要可信 HTTPS 证书；进入后在下方允许新设备配对。</p><button type="button">显示浏览器入口</button><a target="_blank" rel="noreferrer" style="display:block;overflow-wrap:anywhere"></a><img alt="浏览器遥操入口二维码" hidden width="220" height="220"><p role="status"></p></details>`);
  const browserEntry=panel.querySelector('.tp-webxr');
  if(prepareWebxr && has('installation_info')){
    browserEntry.hidden=false;
    browserEntry.querySelector('button').onclick=async()=>{
      const button=browserEntry.querySelector('button');button.disabled=true;
      try{const value=await prepareWebxr(),url=new URL(value.url);
        if(url.protocol!=='https:'||url.username||url.password||url.pathname!=='/webxr/'||url.search||url.hash)throw Error('浏览器入口地址无效');
        const link=browserEntry.querySelector('a');link.href=url.href;link.textContent=url.href;
        const image=browserEntry.querySelector('img');image.src='data:image/svg+xml;base64,'+btoa(value.qr_svg);image.hidden=false;
        browserEntry.querySelector('p[role=status]').textContent='在头显浏览器打开；此链接不授予配对或运动权限。';
      }catch(e){browserEntry.querySelector('p[role=status]').textContent=e.message;}
      finally{if(!disposed)button.disabled=false;}
    };
  }
  let installation=null;
  const install=panel.querySelector('.tp-install'),template=panel.querySelector('.tp-topology');
  if(prepareInstallation && has('installation_info'))install.hidden=false;
  if(loadTargets && buildTemplate){
    template.hidden=false;
    loadTargets().then(targets=>{if(disposed)return;const select=template.querySelector('select');
      for(const t of targets){const opt=document.createElement('option');opt.value=t.mcp_id;opt.textContent=`${t.label} · ${t.robot_profile}`;select.append(opt);}
      if(!targets.length){template.querySelector('button').disabled=true;template.querySelector('p:last-child').textContent='未发现支持三段控制的 Driver';}
    }).catch(e=>{if(!disposed)template.querySelector('p:last-child').textContent=e.message;});
    template.querySelector('button').onclick=async()=>{const button=template.querySelector('button');button.disabled=true;
      try{await buildTemplate(template.querySelector('select').value);}
      catch(e){template.querySelector('p:last-child').textContent=e.message;}
      finally{if(!disposed)button.disabled=false;}};
  }
  function showInstallation(value){
    const url=new URL(value.url,location.href);
    if(url.origin!==location.origin || !url.pathname.startsWith('/pico/'))throw Error('安装链接来源不匹配');
    const link=install.querySelector('a');link.href=url.href;link.textContent=url.origin+url.pathname;
    const image=install.querySelector('img');image.src='data:image/svg+xml;base64,'+btoa(value.qr_svg);image.hidden=false;
    install.querySelector('.tp-install-result').textContent=value.deep_link?'邀请仅可使用一次，15 分钟内有效；连接不会开始运动。':'安装链接 15 分钟有效；仅提供安装包下载。';
  }
  install.querySelector('.tp-download').onclick=async()=>{const b=install.querySelector('.tp-download');b.disabled=true;
    try{installation=await prepareInstallation();showInstallation(installation);
      const p=installation.package;install.querySelector('.tp-package').textContent=`版本 ${p.version} · ${Math.round(p.size_bytes/1024/1024)} MB · SHA256 ${p.sha256}`;
      install.querySelector('.tp-invite').disabled=!createInvitation || !has('create_invitation');}
    catch(e){install.querySelector('.tp-install-result').textContent=e.message;}
    finally{if(!disposed)b.disabled=false;}};
  install.querySelector('.tp-invite').onclick=async()=>{const b=install.querySelector('.tp-invite');b.disabled=true;
    try{const result=await createInvitation(installation.ticket);showInstallation(result);install.querySelector('.tp-revoke-invite').hidden=!has('revoke_invitation');}
    catch(e){install.querySelector('.tp-install-result').textContent=e.message;}
    finally{if(!disposed)b.disabled=false;}};
  install.querySelector('.tp-revoke-invite').onclick=async()=>{
    try{await call('revoke_invitation');showInstallation(installation);install.querySelector('.tp-revoke-invite').hidden=true;install.querySelector('.tp-install-result').textContent='邀请已撤销；安装链接仍可用于下载。';}
    catch(e){install.querySelector('.tp-install-result').textContent=e.message;}};
  const configFields=new Map(),configForm=panel.querySelector('.tp-config form');
  let configDirty=false,configLoaded=false,savedConfig={};
  const labels={mode:'运行模式',robot_profile:'机器人机型',mapping_version:'位姿映射',position_scale:'位移比例',shadow_feedback_source:'模拟反馈来源',namespace:'机器人命名空间',driver_mcp_url:'Driver MCP 地址',calibration_path:'标定文件路径',controller_to_palm:'左右手柄至手掌变换'};
  for(const [key,def] of Object.entries(configSchema.properties||{})){
    if(projectManaged && ['namespace','driver_mcp_url'].includes(key))continue;
    const label=document.createElement('label');label.style.display='block';label.textContent=labels[key]||def.title||key;
    const structured=['object','array'].includes(def.type);
    const input=document.createElement(def.enum?'select':structured?'textarea':'input');input.dataset.configKey=key;input.style.cssText='display:block;width:100%;margin:4px 0 8px;color:inherit;background:transparent;border:1px solid #73839966;padding:6px';
    if(def.enum)for(const value of def.enum){const opt=document.createElement('option');opt.value=value;opt.textContent=value==='live'?'Live · 真机':value==='shadow'?'Shadow · 模拟':value;input.append(opt);}
    else if(structured){input.rows=8;input.spellcheck=false;}
    else {input.type=def.type==='number'?'number':'text';if(def.type==='number'){input.step='any';if(def.minimum!=null)input.min=def.minimum;if(def.maximum!=null)input.max=def.maximum;}}
    input.addEventListener('input',()=>{configDirty=true;});label.append(input);
    if(key==='controller_to_palm'){const hint=document.createElement('small');hint.textContent='JSON：left / right 分别包含 position [x,y,z]（米）与 orientation [x,y,z,w]（单位四元数）。';label.append(hint);}
    configFields.set(key,input);panel.querySelector('.tp-config-fields').append(label);
  }
  if(!saveConfig || !configFields.size)panel.querySelector('.tp-config').hidden=true;
  function showConfig(){
    if(configDirty)return;
    const values={...savedConfig,...info?.configuration};
    for(const [key,input] of configFields){const value=values[key]??configSchema.properties[key].default;
      input.value=value==null?'':['object','array'].includes(configSchema.properties[key].type)?JSON.stringify(value,null,2):value;}
  }
  configForm.addEventListener('submit',async e=>{
    e.preventDefault();if(configForm.querySelector('button').disabled || !configForm.reportValidity())return;
    const values={};
    try{for(const [key,input] of configFields){if(!input.value.trim())continue;const type=configSchema.properties[key].type;
      if(['object','array'].includes(type)){const value=JSON.parse(input.value);
        if(value===null || (type==='array'?!Array.isArray(value):typeof value!=='object'||Array.isArray(value)))throw Error((labels[key]||key)+' 必须为 JSON '+type);
        values[key]=value;}else values[key]=type==='number'?Number(input.value):input.value.trim();}}
    catch(err){panel.querySelector('.tp-config-result').textContent='配置未保存：'+err.message;return;}
    if(values.mode==='live' && info?.mode!=='live' && !window.confirm('保存真机模式；开始并握住双握把后将驱动机器人。继续？'))return;
    busy=true;render();const result=panel.querySelector('.tp-config-result');result.textContent='正在保存并应用…';
    try{await saveConfig(values);savedConfig=await loadConfig();configDirty=false;result.textContent='配置已保存并应用；请重新标定。';}
    catch(e){result.textContent='配置未确认：'+e.message;}
    finally{busy=false;await refresh();showConfig();}
  });
  function render(){
    const v=viewState(info,fresh,projectManaged), d=info?.driver_feedback, pose=info?.pose, track=pose?.latest?.tracking;
    const configDisabled=busy || v.busy || !!operationPending || !fresh || !configLoaded || !!info?.authority_valid || !!d?.ownership_held || v.armed===true;
    for(const el of configForm.querySelectorAll('input,select,textarea,button'))el.disabled=configDisabled;
    panel.dataset.live=info?.mode==='live';panel.dataset.executing=v.executing;
    for(const k of ['title','mode','execution','reason'])field(k,v[k]);
    const binding=info?.project?.driver_binding || info?.driver_binding;
    field('driver',binding?`${binding.robot_profile} · ${binding.mcp_id}`:'等待 Canvas 命令连线');
    if(binding?.protocol_version===2)field('driver',`${binding.robot_profile} · ${binding.tool} → ${binding.execution_binding?.tool || '执行卡'}`);
    template.querySelector('select').disabled=busy || v.busy || v.armed;
    if(template.querySelector('select').options.length)template.querySelector('button').disabled=busy || v.busy || v.armed;
    field('armed',!projectManaged?'未支持项目托管收臂':!fresh?'未知':v.armed?'已开启 · 等待或接受 PICO 操作':'未开启');
    field('connected',!fresh?'未知':v.connected?'已连接':'未连接');
    field('tracking',!fresh||!pose?.fresh?'无新鲜输入':track && Object.values(track).every(Boolean)?'头显与双手有效':'跟踪缺失');
    field('age',fresh && Number.isFinite(pose?.age_ms)?`${Math.round(pose.age_ms)} ms`:'—');
    field('calibration',!fresh?'未知':info?.calibrated?'已加载':'未标定');
    const e=info?.enrollment;
    field('pairing',!fresh?'连接状态不可用':e?.window_open?`允许配对 · 剩余 ${Math.max(0,Math.ceil(e.expires_in_seconds||0))} 秒`:`配对窗口关闭 · 已配对 ${info?.capture?.paired_devices||0} 台`);
    panel.querySelector('.tp-fingerprint').textContent=pending?`${pending.device_name || '新头显'}\n${String(pending.fingerprint||'').match(/.{1,8}/g)?.join(' ') || '指纹缺失'}\n请与 PICO 指纹逐组核对`:'暂无新配对申请';
    panel.querySelector('.tp-error').textContent=error || readError;panel.querySelector('.tp-notice').textContent=notice;
    panel.querySelector('pre').textContent=JSON.stringify({reason:info?.reason,operator:info?.operator,driver_state:d?.state,driver_reason:d?.reason,stop_confirmed:v.driverFresh?d?.stop_confirmed:null,calibration:info?.dispatch?.adapter?.diagnostics?.calibration_version,ik:info?.diagnostics?.latency_ms?.ik},null,2);
    for(const [a,b] of buttons){
      let disabled=busy || v.busy || !!operationPending || !fresh;
      if(['start','resume','finish'].includes(a))disabled ||= !v.connected;
      if(projectManaged && ['start','resume'].includes(a))disabled ||= !v.armed;
      if(a==='start')disabled ||= !!info?.authority_valid || !info?.calibrated;
      if(a==='resume')disabled ||= !['hold','fault'].includes(info?.state);
      if(a==='pause')disabled ||= !info?.authority_valid && !d?.ownership_held;
      if(a==='config')b.textContent=info?.mode==='live'?'切换为模拟':'切换为真机';
      if(['calibrate','self_test','config'].includes(a))disabled ||= !!info?.authority_valid || !!d?.ownership_held;
      if(['approve_pairing','reject_pairing'].includes(a))disabled ||= !pending?.request_id || !pending?.fingerprint;
      if(a==='stop')disabled=false; // Stop stays available even during stale/pending UI.
      b.disabled=disabled;
    }
  }
  async function refresh(){
    const generation=request;
    try {const value=await call('info');if(disposed||generation!==request)return;info=value;fresh=true;readError='';pending=value.enrollment?.pending||null;
      if(operationPending && value.operator?.action===operationPending && !['starting','returning','stopping'].includes(value.operator.state)){
        notice=value.operator.error?'操作未完成，请查看故障原因':'操作状态已更新，请以上方执行反馈为准';operationPending=null;
      }showConfig();}
    catch(e){if(disposed||generation!==request)return;fresh=false;pending=null;readError=e.message||'读取失败';}
    render();
  }
  async function execute(action){
    if(disposed || buttons.get(action)?.disabled)return;
    if(action==='revoke_headset' && !window.confirm('撤销后需要在 PICO 重新配对，继续？'))return;
    if(info?.mode==='live' && ['start','resume','finish'].includes(action) && !window.confirm(action==='finish'?'双臂将缓慢回到自然下垂位置。确认结束并收臂？':'将准备真机遥操。请先松开双握把；之后双握把使能真实运动。继续？'))return;
    if(action==='config' && info?.mode!=='live' && !window.confirm('切换为真机模式。之后开始遥操并握住双握把会驱动机器人，继续？'))return;
    const args=action==='config'?{mode:info?.mode==='live'?'shadow':'live'}:['approve_pairing','reject_pairing'].includes(action)?{request_id:pending.request_id,fingerprint:pending.fingerprint}:action==='finish'?{request_id:crypto.randomUUID()}:{};
    const generation=++request;if(action==='stop')operationPending=null;busy=true;error='';notice='请求处理中…';render();
    try {const result=await call(action,args);if(disposed||generation!==request)return;operationPending=result.state==='accepted'?action:null;notice=result.state==='accepted'?'请求已接收，等待执行结果':'操作已返回，正在核对实际状态';}
    catch(e){if(disposed||generation!==request)return;error=e.message||'操作失败';notice='';}
    finally{if(!disposed&&generation===request){busy=false;await refresh();}}
  }
  for(const [label,action,group] of [['开始遥操','start','.tp-actions'],['暂停保持','pause','.tp-actions'],['恢复遥操','resume','.tp-actions'],['结束并收臂','finish','.tp-actions'],['立即停止','stop','.tp-actions'],['允许新设备配对','open_pairing','.tp-pair-buttons'],['指纹一致，批准','approve_pairing','.tp-pair-buttons'],['拒绝申请','reject_pairing','.tp-pair-buttons'],['断开头显','disconnect_headset','.tp-pair-buttons'],['撤销配对','revoke_headset','.tp-pair-buttons'],['重新标定','calibrate','.tp-maintenance'],['零输出自检','self_test','.tp-maintenance'],['切换模式','config','.tp-maintenance']]){
    if(!has(action) || (projectManaged && action==='start'))continue;
    const b=document.createElement('button');b.type='button';b.textContent=label;b.dataset.action=action;b.onclick=()=>execute(action);buttons.set(action,b);panel.querySelector(group).append(b);
  }
  panel.addEventListener('pointerdown',e=>e.stopPropagation());panel.addEventListener('click',e=>e.stopPropagation());
  async function poll(){if(disposed)return;if(!host.isConnected){dispose();return;}await refresh();if(!disposed)timer=setTimeout(poll,1000);}
  function dispose(){disposed=true;clearTimeout(timer);observer.disconnect();panel.remove();}
  const observer=new MutationObserver(()=>{if(!host.isConnected)dispose();});observer.observe(document.body,{childList:true,subtree:true});
  render();timer=setTimeout(poll,0);
  if(loadConfig)loadConfig().then(value=>{if(!disposed){savedConfig=value||{};configLoaded=true;showConfig();render();}}).catch(e=>{if(!disposed)panel.querySelector('.tp-config-result').textContent='读取配置失败：'+e.message;});
  return {dispose,refresh};
}
