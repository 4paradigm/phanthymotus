function workbenchUuid(){const b=crypto.getRandomValues(new Uint8Array(16));b[6]=(b[6]&15)|64;b[8]=(b[8]&63)|128;const h=[...b].map(x=>x.toString(16).padStart(2,"0")).join("");return [h.slice(0,8),h.slice(8,12),h.slice(12,16),h.slice(16,20),h.slice(20)].join("-");}
function createSimulationWorkbench(config){
 const hostCall=config.call, catalog=config.catalog, modelLabel=config.modelLabel, friendlyError=config.friendlyError||String;
 const viewId=config.viewId;
 const $=id=>document.getElementById(config.prefix+id),canvas=$('view'),ctx=canvas.getContext('2d');
 let session=null,snapshot=null,playback=false,replayTotal=0,replayOffset=0,active=false,heartbeatBusy=false,createKey='',createRequestId='';
 const owned=()=>session?.control?.is_controller===true;
 function viewportSize(box,moving=false){const width=Math.max(160,box.width||640),height=Math.max(120,box.height||360),scale=Math.min(moving?1:1.5,(moving?640:960)/width,(moving?480:640)/height);return {width:Math.round(width*scale),height:Math.round(height*scale)};}
 async function call(action,args={}){args={view_id:viewId,...args};if(action.startsWith('simulation_')&&!['simulation_create','simulation_attach','simulation_list','simulation_replay'].includes(action))args.control_epoch=session?.control?.epoch??0;return config.transport(action,args);}
 async function renew(){if(!active||document.hidden||!owned()||heartbeatBusy||!session)return;heartbeatBusy=true;try{inspect(await call('simulation_heartbeat',{session_id:session.id}));}catch(error){if(session?.control)session.control.is_controller=false;controls();status('控制已断开，请重新接管',true);}finally{heartbeatBusy=false;}}
 setInterval(renew,5000);

 function controls(){const exists=session&&session.state!=='closed',running=exists&&session.state==='running',error=exists&&session.state==='error',control=exists&&owned();$('closeSession').disabled=!control;$('run').disabled=!control||running||error;$('run').classList.toggle('hidden',!!running);$('run').textContent=session?.state==='completed'?'再次运行':'开始运行';$('pause').disabled=!control||!running;$('pause').classList.toggle('hidden',!running);$('step').disabled=!control||running||error||session.state==='completed';$('reset').disabled=!control;$('record').disabled=!control||session.recording||error;$('recordStop').disabled=!control||!session.recording;$('replay').disabled=!exists||!session.recorded_frames;$('perturb').disabled=!control||!$('body').value||error;$('controlState').textContent=!exists?'':control?'控制中':session.control?.owner_view_id?'观看中':'无人控制';$('claim').classList.toggle('hidden',!exists||control);$('claim').textContent=session?.control?.owner_view_id?'接管控制':'取得控制';$('release').classList.toggle('hidden',!control);for(const id of ['run','pause','step','reset','record','recordStop','perturb','closeSession'])$(id).title=exists&&!control?'先取得控制': '';for(const button of inFlight)button.disabled=true;config.onChange?.(session);}


let viewBusy=false,viewTimer=null,viewPendingMode=null,viewEpoch=0,lastViewAt=0;
let interactionUntil=0,settleTimer=null,cameraVersion=0,lastFrameCamera=-1,lastFrameWidth=0,renderRetryAt=0;
let lastPanelAt=0,panelTimer=null,bodyKey='',jointKey='',jointCells=[],stateCells=null;
const FRAME_INTERVAL=50,INTERACTION_SETTLE=450,PANEL_INTERVAL=250;
const camera={azimuth:135,elevation:-20,distance:3,lookat:[0,0,.8]};
const inFlight=new Set();
let yaw=-.75,pitch=.32,zoom=170,target=[0,0,.65],drag=null;
const states={paused:'已暂停',running:'运行中',completed:'动作完成',closed:'已关闭',error:'异常'};
function status(message,error=false){$('status').textContent=friendlyError(message);$('status').classList.toggle('error',error);}
function perform(fn){return async event=>{const button=event?.currentTarget;if(button?.disabled||inFlight.has(button))return;if(button){inFlight.add(button);button.disabled=true;}clearTimeout(viewTimer);try{await fn();}catch(e){status(e.message,true);}finally{if(button){inFlight.delete(button);button.disabled=false;}controls();if(viewPendingMode)scheduleView(viewPendingMode==='inspect'?2000:0,viewPendingMode==='inspect');}};}
function options(id,items,empty){const select=$(id),old=select.value;select.replaceChildren();const first=document.createElement('option');first.value='';first.textContent=empty;select.append(first);for(const item of items){const o=document.createElement('option');o.value=item.id;o.textContent=item.label||item.name||item.id;select.append(o);}if(items.some(i=>String(i.id)===old))select.value=old;}
function inspect(r,frame=false){if(r.session){session=r.session;}if(r.control&&session)session.control=r.control;if(r.snapshot&&(!playback||frame))snapshot=r.snapshot;renderState();if(!realView())draw();controls();}
function textValue(element,value){const text=String(value??'—');if(element.textContent!==text)element.textContent=text;}
function renderState(){
 if(!session||document.hidden||!active)return;
 textValue($('sessionState'),states[session.state]||session.state);textValue($('modeLabel'),session.control_mode==='pd_hold'?'物理调试 · PD 保持':session.control_mode==='free_physics'?'物理调试':session.control_mode==='policy'?'策略运行':session.control_mode||'');
 textValue($('recordCount'),session.recording?`录制中 · ${session.recorded_frames||0} 帧`:session.recorded_frames?`${session.recorded_frames} 帧`:'');
 if(!snapshot)return;if(!realView())$('empty').classList.add('hidden');textValue($('simTime'),`${Number(snapshot.time||0).toFixed(2)} s`);
 const remaining=PANEL_INTERVAL-(performance.now()-lastPanelAt);if(remaining>0){if(!panelTimer)panelTimer=setTimeout(()=>{panelTimer=null;renderState();},remaining+1);return;}
 clearTimeout(panelTimer);panelTimer=null;lastPanelAt=performance.now();
 if(!stateCells){stateCells={};for(const [key,label] of [['engine','引擎'],['contacts','接触点'],['height','机身高度'],['error','状态']]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;stateCells[key]=dd;$('stateInfo').append(dt,dd);}}
 textValue(stateCells.engine,session.engine);textValue(stateCells.contacts,snapshot.contact_count??snapshot.contacts?.length);textValue(stateCells.height,snapshot.root_pose?.pos?.[2]?.toFixed(3));textValue(stateCells.error,friendlyError(session.error)||'—');
 const bodies=(snapshot.bodies||[]).filter(b=>b.id!==0),nextBodyKey=JSON.stringify(bodies.map(b=>[b.id,b.name]));if(nextBodyKey!==bodyKey){bodyKey=nextBodyKey;options('body',bodies,'选择刚体');if(!$('body').value&&bodies.length)$('body').value=bodies[0].id;}
 const names=snapshot.joint_names||[],nextJointKey=JSON.stringify(names);if(nextJointKey!==jointKey){jointKey=nextJointKey;jointCells=[];const table=document.createElement('table'),head=document.createElement('tr');for(const label of ['关节','rad']){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);for(const name of names){const tr=document.createElement('tr'),a=document.createElement('td'),b=document.createElement('td');a.textContent=name;a.title=name;tr.append(a,b);table.append(tr);jointCells.push(b);}$('jointTable').replaceChildren(table);}
 jointCells.forEach((cell,i)=>textValue(cell,Number.isFinite(snapshot.joint_pos?.[i])?snapshot.joint_pos[i].toFixed(3):'—'));controls();
}
function clearSession(){viewEpoch++;clearTimeout(viewTimer);clearTimeout(settleTimer);clearTimeout(panelTimer);panelTimer=null;viewPendingMode=null;stateCells=null;bodyKey='';jointKey='';jointCells=[];lastPanelAt=0;lastFrameWidth=0;lastFrameCamera=-1;renderRetryAt=0;interactionUntil=0;drag=null;$('renderedView').classList.add('hidden');$('renderedView').removeAttribute('src');$('empty').textContent='未加载机器人';$('empty').classList.remove('error');session=null;snapshot=null;playback=false;$('replayBar').classList.add('hidden');$('sessionState').textContent='未创建';$('modeLabel').textContent='';$('simTime').textContent='0.00 s';$('recordCount').textContent='';$('empty').classList.remove('hidden');$('stateInfo').replaceChildren();$('jointTable').replaceChildren();$('body').replaceChildren();draw();controls();}
async function simulation(action,args={}){if(!session)throw Error('请先创建仿真环境');viewEpoch++;playback=false;$('replayBar').classList.add('hidden');const r=await call(`simulation_${action}`,{session_id:session.id,...args});inspect(r);status(session.error||states[session.state]||'已更新',session.state==='error');if(action!=='close')scheduleView();return r;}
function project(point){const x=point[0]-target[0],y=point[1]-target[1],z=point[2]-target[2],side=x*Math.cos(yaw)-y*Math.sin(yaw),depth=x*Math.sin(yaw)+y*Math.cos(yaw);return [canvas.clientWidth/2+side*zoom,canvas.clientHeight/2-(z*Math.cos(pitch)-depth*Math.sin(pitch))*zoom,depth*Math.cos(pitch)+z*Math.sin(pitch)];}
function line(a,b,color,width=1){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();}
function dot(pos,r,color){const p=project(pos);ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fill();}
function corner(g,x,y,z){const m=g.mat||[1,0,0,0,1,0,0,0,1],s=g.size||[.03,.03,.03],v=[x*s[0],y*s[1],z*s[2]];return [0,1,2].map(i=>g.pos[i]+m[i*3]*v[0]+m[i*3+1]*v[1]+m[i*3+2]*v[2]);}
function draw(){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);if($('viewMode').value==='render')return;ctx.lineCap='round';const tx=Math.round(target[0]),ty=Math.round(target[1]);for(let i=-8;i<=8;i++){line([tx-4,ty+i*.5,0],[tx+4,ty+i*.5,0],'#dad4c9');line([tx+i*.5,ty-4,0],[tx+i*.5,ty+4,0],'#dad4c9');}if(!snapshot)return;if($('viewMode').value==='geometry'){const geoms=[...(snapshot.geoms||[])].sort((a,b)=>project(b.pos)[2]-project(a.pos)[2]);for(const g of geoms){if(g.type==='plane')continue;const color=g.rgba?`rgba(${g.rgba.slice(0,3).map(v=>Math.round(v*220)).join(',')},${Math.max(.3,g.rgba[3]??1)})`:'#54666b';if(g.type==='sphere'){dot(g.pos,Math.max(2,g.size[0]*zoom),color);}else if(g.type==='capsule'||g.type==='cylinder'){const m=g.mat||[1,0,0,0,1,0,0,0,1],half=g.size[1];line(g.pos.map((v,i)=>v-m[i*3+2]*half),g.pos.map((v,i)=>v+m[i*3+2]*half),color,Math.max(3,g.size[0]*2*zoom));}else{const pts=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(v=>corner(g,...v));for(const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])line(pts[a],pts[b],color,2);}}}else{const bodies=(snapshot.bodies||[]).filter(b=>Array.isArray(b.pos)),byId=new Map(bodies.map(b=>[b.id,b]));for(const b of bodies){const p=byId.get(b.parent);if(p&&p.id!==0&&b.id!==0)line(b.pos,p.pos,'#40545c',6);}for(const b of bodies)if(b.id!==0)dot(b.pos,3.8,'#b55b36');}for(const c of snapshot.contacts||[])dot(c.pos,3,'#ca6b35');}
function realView(){return $('viewMode').value==='render';}
function invalidateView(){viewEpoch++;scheduleView();}
function scheduleView(delay=0,inspectOnly=false){if(!inspectOnly||viewPendingMode===null)viewPendingMode=inspectOnly?'inspect':'render';if(viewBusy||inFlight.size||document.hidden||!active)return;clearTimeout(viewTimer);viewTimer=setTimeout(refreshView,Math.max(delay,FRAME_INTERVAL-(performance.now()-lastViewAt),viewPendingMode==='render'&&realView()?renderRetryAt-performance.now():0));}
function poseKey(value){return JSON.stringify([value?.time,value?.root_pose,value?.joint_pos]);}
function imageError(message){$('renderedView').classList.add('hidden');$('empty').textContent=message;$('empty').classList.add('error');$('empty').classList.remove('hidden');status(message,true);}
async function refreshView(){
 if(viewBusy||inFlight.size||document.hidden||!active)return;if(!session||session.state==='closed'){viewPendingMode=null;return;}
 const inspectOnly=viewPendingMode==='inspect';viewBusy=true;viewPendingMode=null;lastViewAt=performance.now();const epoch=viewEpoch,id=session.id,mode=$('viewMode').value,offset=playback?replayOffset:null,frameCamera=cameraVersion;let failed=false;
 try{
  let result;
  if(mode==='render'&&!inspectOnly){
   const moving=session.state==='running'||!!drag||performance.now()<interactionUntil;const args={session_id:id,...viewportSize($('viewer').getBoundingClientRect(),moving),jpeg_quality:moving?70:88,detail:'compact',azimuth:camera.azimuth,elevation:camera.elevation,distance:camera.distance,lookat:[...camera.lookat]};if(offset!==null)args.offset=offset;
   result=await call('inspect',{mode:'frame',...args});
   if(epoch!==viewEpoch||session?.id!==id)return;
   const image=result.image;if(image?.mime!=='image/jpeg'||typeof image.data!=='string'||image.data.length>3000000||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<1||image.width>960||image.height<1||image.height>640)throw Error('服务器返回了无效 MuJoCo 图像');
   const decoded=new Image();decoded.src='data:image/jpeg;base64,'+image.data;try{await decoded.decode();}catch{throw Error('MuJoCo 图像解码失败');}
   if(epoch!==viewEpoch||session?.id!==id)return;
   if(decoded.naturalWidth!==image.width||decoded.naturalHeight!==image.height)throw Error('MuJoCo 图像尺寸不一致');
   renderRetryAt=0;lastFrameWidth=image.width;lastFrameCamera=frameCamera;$('renderedView').src=decoded.src;$('renderedView').classList.remove('hidden');$('empty').classList.add('hidden');$('empty').classList.remove('error');
  }else if(offset!==null){
   const replay=await call('simulation_replay',{session_id:id,offset,limit:1});if(!replay.frames?.length)throw Error('没有可回放的帧');result={snapshot:replay.frames[0]};
  }else result=await call('inspect',{mode:'state',session_id:id,detail:mode==='render'?'compact':'full'});
  if(epoch!==viewEpoch||session?.id!==id)return;
  const changed=poseKey(result.snapshot)!==poseKey(snapshot);inspect(result,true);if(inspectOnly&&mode==='render'&&(changed||session.state==='running'))scheduleView();if(!inFlight.size&&(!inspectOnly||!$('empty').classList.contains('error')))status(playback?'回放':session.error||states[session.state]||'已更新',session.state==='error');
 }catch(e){failed=true;if(epoch===viewEpoch&&session?.id===id){if(mode==='render'){if(!inspectOnly)renderRetryAt=performance.now()+2000;imageError(e.message);}else status(e.message,true);}}
 finally{viewBusy=false;if(session&&session.state!=='closed'){if(viewPendingMode)scheduleView(0,viewPendingMode==='inspect');else if(!playback){if(session.state==='running')scheduleView(failed?2000:0);else scheduleView(2000,true);}}}
}
function scheduleSharpFrame(){clearTimeout(settleTimer);settleTimer=setTimeout(()=>{if(drag)return;if(performance.now()<interactionUntil){scheduleSharpFrame();return;}if(session&&session.state!=='running'&&realView()&&(viewBusy||lastFrameWidth!==viewportSize($('viewer').getBoundingClientRect()).width||lastFrameCamera!==cameraVersion))scheduleView();},Math.max(1,interactionUntil-performance.now()));}
function cameraChanged(){cameraVersion++;interactionUntil=performance.now()+INTERACTION_SETTLE;scheduleSharpFrame();scheduleView();}
function center(){if(snapshot?.root_pose?.pos?.length===3&&snapshot.root_pose.pos.every(Number.isFinite)){target=[...snapshot.root_pose.pos];camera.lookat=[...target];cameraVersion++;}draw();scheduleView();}
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch,azimuth:camera.azimuth,elevation:camera.elevation};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;if(realView()){camera.azimuth=((drag.azimuth+(e.clientX-drag.x)*.3)%360+360)%360;camera.elevation=Math.max(-85,Math.min(85,drag.elevation+(e.clientY-drag.y)*.3));cameraChanged();}else{yaw=drag.yaw+(e.clientX-drag.x)*.008;pitch=Math.max(-.15,Math.min(1.4,drag.pitch+(e.clientY-drag.y)*.008));draw();}};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;interactionUntil=performance.now()+INTERACTION_SETTLE;scheduleSharpFrame();};
canvas.addEventListener('wheel',e=>{e.preventDefault();if(realView()){camera.distance=Math.max(.5,Math.min(12,camera.distance*Math.exp(e.deltaY*.001)));cameraChanged();}else{zoom=Math.max(25,Math.min(800,zoom*Math.exp(-e.deltaY*.001)));draw();}},{passive:false});
canvas.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();if(realView()){if(e.key==='ArrowLeft')camera.azimuth=(camera.azimuth+355)%360;else if(e.key==='ArrowRight')camera.azimuth=(camera.azimuth+5)%360;else if(e.key==='ArrowUp')camera.elevation=Math.min(85,camera.elevation+5);else camera.elevation=Math.max(-85,camera.elevation-5);cameraChanged();}else{if(e.key==='ArrowLeft')yaw-=.1;else if(e.key==='ArrowRight')yaw+=.1;else if(e.key==='ArrowUp')pitch=Math.min(1.4,pitch+.1);else pitch=Math.max(-.15,pitch-.1);draw();}};
let lastCanvasSize='';new ResizeObserver(()=>{draw();const size=canvas.clientWidth+'x'+canvas.clientHeight;if(size!==lastCanvasSize){lastCanvasSize=size;viewEpoch++;scheduleView();}}).observe(canvas);
document.addEventListener('visibilitychange',()=>{viewEpoch++;clearTimeout(viewTimer);clearTimeout(settleTimer);clearTimeout(panelTimer);panelTimer=null;drag=null;interactionUntil=0;if(!document.hidden&&active&&session){renderState();scheduleView();}});
$('viewMode').onchange=()=>{$('viewer').classList.toggle('diagnostic',!realView());$('renderedView').classList.add('hidden');$('empty').classList.remove('error');$('empty').textContent=session?'正在加载画面':'未加载机器人';$('empty').classList.toggle('hidden',!!snapshot&&!realView());draw();invalidateView();};
$('center').onclick=center;$('refreshView').onclick=()=>{renderRetryAt=0;invalidateView();};
$('closeSession').onclick=perform(async()=>{await simulation('close');clearSession();status('仿真已关闭');config.onClosed?.();});
for(const action of ['pause','step','reset','record'])$(action).onclick=perform(async()=>{playback=false;$('replayBar').classList.add('hidden');await simulation(action,action==='step'?{steps:1}:{});if(action==='reset')center();});$('run').onclick=perform(run);
$('recordStop').onclick=perform(()=>simulation('record_stop'));
$('perturb').onclick=perform(async()=>{const force=[Number($('fx').value),Number($('fy').value),0];if(force.some(v=>!Number.isFinite(v)||Math.abs(v)>200))throw Error('扰动力须在 -200 至 200 N 之间');await simulation('perturb',{body_id:Number($('body').value),force,duration_s:.1});});
async function replayFrame(offset){if(!session?.recorded_frames)throw Error('没有可回放的帧');replayTotal=session.recorded_frames;if(!Number.isInteger(offset)||offset<0||offset>=replayTotal)throw Error('回放帧超出范围');playback=true;replayOffset=offset;$('replayFrame').max=replayTotal-1;$('replayFrame').value=offset;$('replayLabel').textContent=`${offset+1} / ${replayTotal}`;$('replayBar').classList.remove('hidden');invalidateView();}
$('replay').onclick=perform(()=>replayFrame(0));$('replayFrame').onchange=perform(()=>replayFrame(Number($('replayFrame').value)));$('live').onclick=perform(async()=>{playback=false;$('replayBar').classList.add('hidden');invalidateView();});
 async function run(){if(!session)throw Error('请先选择模型');if(session.state==='running')return;if(session.state==='completed')await simulation('reset');await simulation('run');}
 async function create(args){if(session&&session.state!=='closed')throw Error('请先离开当前会话');const key=JSON.stringify([args.profile_id,args.model_id,args.control_mode]);if(key!==createKey){createKey=key;createRequestId=args.request_id||workbenchUuid();}inspect(await call('simulation_create',{...args,request_id:createRequestId}));createKey='';createRequestId='';center();return session;}
 async function detach(){
  if(!session)return;const id=session.id;
  async function refreshControl(){
   const result=await call('inspect',{mode:'state',session_id:id,detail:'compact'});
   if(session?.id!==id||result.session?.id!==id)throw Error('仿真会话已变化，请重新选择');
   inspect(result);return session.control;
  }
  // The action page stops the preview heartbeat. Its cached ownership can be
  // expired or transferred when a user comes back to try another version.
  await refreshControl();
  if(owned()){
   try{inspect(await call('simulation_release',{session_id:id}));}
   catch(error){
    if(!String(error.message).includes('控制权已变化'))throw error;
    // Only confirmed loss of ownership lets us leave after a release race.
    // Network errors and an unconfirmed release must keep the binding intact.
    const control=await refreshControl();
    if(control?.is_controller!==false)throw error;
   }
  }
  await call('simulation_detach',{session_id:id});clearSession();
 }
 async function prepare(model){if(session&&session.state!=='closed'&&(session.model_id!==model.id||session.profile_id!==model.profile_id))await detach();if(!session||session.state==='closed')await create({profile_id:model.profile_id,model_id:model.id});$('title').textContent=modelLabel(model);center();controls();}
 $('claim').onclick=perform(async()=>{try{inspect(await call('simulation_claim',{session_id:session.id,takeover:!!session.control?.owner_view_id}));status('已取得控制');}catch(error){try{inspect(await call('inspect',{mode:'state',session_id:session.id,detail:'compact'}));}catch{}throw error;}scheduleView();});
 $('release').onclick=perform(async()=>{inspect(await call('simulation_release',{session_id:session.id}));status('已暂停，当前仅观看');});
 return {prepare,run,create,detach,clear:clearSession,getSession:()=>session,async attach(id,binding={}){inspect(await call('simulation_attach',{session_id:id,...binding}));center();return session;},async restore(id){if(!id)return;inspect(await call('inspect',{mode:'state',session_id:id,detail:'compact'}));config.onRestore?.(session);center();},visible(value){active=value;viewEpoch++;clearTimeout(viewTimer);clearTimeout(settleTimer);if(value&&session){renderState();scheduleView();renew();}draw();}};
}
