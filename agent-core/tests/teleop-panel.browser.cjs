// Run with a host Playwright installation; all MCP responses are local fixtures.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../web/js/teleop-panel.js'));
const renderer=fs.readFileSync(path.join(__dirname,'../web/js/renderers/control.js'));
const server=http.createServer((req,res)=>{
 if(req.url==='/teleop-panel.js'){res.setHeader('Content-Type','text/javascript');res.end(source);return;}
 if(req.url==='/control.js'){res.setHeader('Content-Type','text/javascript');res.end(renderer);return;}
 res.end(`<html><meta charset="utf-8"><body style="background:#111827;padding:32px"><div id="host"></div><script type="module">
 import {mountTeleopPanel,viewState} from '/teleop-panel.js';
 import {ControlRenderer} from '/control.js';window.ControlRenderer=ControlRenderer;
 window.mountTeleopPanel=mountTeleopPanel;
 window.calls=[];window.fail=false;window.wait=false;window.releases=[];window.saved={mode:"shadow",position_scale:.5,controller_to_palm:{left:{position:[0,0,0],orientation:[0,0,0,1]},right:{position:[0,0,0],orientation:[0,0,0,1]}}};window.rejectConfig=false;window.saves=[];
 window.sample={mode:'shadow',state:'idle',calibrated:true,capture:{connected:true,paired_devices:1},driver_feedback_fresh:true,driver_feedback:{state:'idle',ownership_held:false,output_active:false},pose:{fresh:true,age_ms:12,latest:{tracking:{head:true,left:true,right:true}}},enrollment:{window_open:false},operator:{state:'idle'}};
 window.ui=mountTeleopPanel(document.querySelector('#host'),{configSchema:{properties:{mode:{type:'string',enum:['shadow','live']},position_scale:{type:'number',minimum:.01,maximum:1},controller_to_palm:{type:'object'}}},loadConfig:async()=>structuredClone(saved),saveConfig:async(v)=>{if(rejectConfig)throw Error('release_before_config');saves.push(v);saved=v;sample.configuration=v;},actions:['start','pause','resume','finish','stop','open_pairing','approve_pairing','reject_pairing','disconnect_headset','revoke_headset','calibrate','self_test','config'],call:async(a,args)=>{calls.push({a,args});if(a==='info'){if(fail)throw Error('network unavailable');return structuredClone(sample);}if(wait&&a!=='stop')await new Promise(r=>releases.push(r));sample.operator={state:'idle',action:a};return {state:'accepted'};}});window.viewState=viewState;
 </script></body></html>`);
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH});
 try{
  const page=await browser.newPage({viewport:{width:960,height:900}});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${server.address().port}`);await page.waitForFunction(()=>window.ui && document.querySelector('[data-action=start]')?.disabled===false);
  assert.equal(await page.locator('[data-field=execution]').innerText(),'无硬件输出');
  await page.locator('[data-action=start]').click();await page.waitForFunction(()=>calls.some(x=>x.a==='start'));
  await page.evaluate(async()=>{sample.mode='live';sample.driver_feedback.output_active=true;sample.driver_feedback.ownership_held=true;sample.state='active';await ui.refresh();});
  assert.equal(await page.locator('[data-field=execution]').innerText(),'硬件正在执行');
  await page.evaluate(async()=>{sample.enrollment.pending={device_name:'<img src=x onerror=alert(1)>',request_id:'req1',fingerprint:'1234567890ABCDEF'};await ui.refresh();});
  await page.locator('.tp-pair summary').click();assert.equal(await page.locator('.tp-fingerprint img').count(),0);
  await page.locator('[data-action=approve_pairing]').click();await page.waitForFunction(()=>calls.some(x=>x.a==='approve_pairing'));
  assert.deepEqual(await page.evaluate(()=>calls.find(x=>x.a==='approve_pairing').args),{request_id:'req1',fingerprint:'1234567890ABCDEF'});
  await page.evaluate(async()=>{fail=true;await ui.refresh();});
  assert.equal(await page.locator('[data-field=execution]').innerText(),'执行状态未知');assert(await page.locator('[data-action=approve_pairing]').isDisabled());assert(await page.locator('[data-action=stop]').isEnabled());
  await page.locator('[data-action=stop]').click();await page.waitForFunction(()=>calls.some(x=>x.a==='stop'));
  await page.evaluate(async()=>{fail=false;sample.state='idle';sample.driver_feedback.output_active=false;sample.driver_feedback.ownership_held=false;sample.enrollment.pending=null;await ui.refresh();});
  // Accepted is not completion. A stop can overtake a pending finish request.
  page.on('dialog',d=>d.accept());await page.evaluate(()=>{wait=true;});
  await page.locator('[data-action=finish]').click();await page.waitForFunction(()=>releases.length===1);
  const n=await page.evaluate(()=>calls.filter(x=>x.a==='stop').length);
  await page.locator('[data-action=stop]').click();await page.waitForFunction(n=>calls.filter(x=>x.a==='stop').length>n,n);
  await page.evaluate(()=>releases.splice(0).forEach(r=>r()));
  assert.match(await page.evaluate(()=>calls.find(x=>x.a==='finish').args.request_id),/^[0-9a-f-]{36}$/);
  await page.evaluate(()=>{wait=false;});
  await page.locator('details').last().locator('summary').click();
  await page.locator('[data-action=config]').click();await page.waitForFunction(()=>calls.some(x=>x.a==='config'));
  assert.deepEqual(await page.evaluate(()=>calls.find(x=>x.a==='config').args),{mode:'shadow'});
  await page.locator('details').last().locator('summary').click();
  await page.locator('.tp-config summary').click();
  const mapping=page.locator('[data-config-key=controller_to_palm]');
  const originalMapping=JSON.parse(await mapping.inputValue());
  assert.deepEqual(originalMapping.left.position,[0,0,0]);
  await mapping.fill('{bad');await page.locator('.tp-config button').click();
  assert.match(await page.locator('.tp-config-result').innerText(),/配置未保存/);
  assert.equal(await page.evaluate(()=>saves.length),0);
  assert.deepEqual(await page.evaluate(()=>saved.controller_to_palm),originalMapping);
  await mapping.fill('[]');await page.locator('.tp-config button').click();
  assert.match(await page.locator('.tp-config-result').innerText(),/必须为 JSON object/);
  originalMapping.left.position[2]=.04;await mapping.fill(JSON.stringify(originalMapping));
  await page.locator('[data-config-key=position_scale]').fill('.75');await page.evaluate(()=>ui.refresh());
  assert.equal(await page.locator('[data-config-key=position_scale]').inputValue(),'.75');
  await page.evaluate(()=>{rejectConfig=true;});await page.locator('.tp-config button').click();
  await page.waitForFunction(()=>document.querySelector('.tp-config-result').textContent.includes('release_before_config'));
  assert.equal(await page.evaluate(()=>saved.position_scale),.5);
  await page.evaluate(()=>{rejectConfig=false;});await page.locator('.tp-config button').click();await page.waitForFunction(()=>saves.length===1);
  assert.equal(await page.evaluate(()=>saved.position_scale),.75);
  assert.deepEqual(await page.evaluate(()=>saved.controller_to_palm),originalMapping);
  await page.evaluate(async()=>{sample.authority_valid=true;await ui.refresh();});assert(await page.locator('.tp-config button').isDisabled());assert(await mapping.isDisabled());
  await page.evaluate(async()=>{sample.authority_valid=false;await ui.refresh();});
  await page.screenshot({path:process.env.TELEOP_SCREENSHOT || path.join(require('node:os').tmpdir(),'teleop-card.png'),fullPage:true});
  const before=await page.evaluate(()=>calls.length);await page.evaluate(()=>document.querySelector('#host').remove());await page.waitForTimeout(1200);assert.equal(await page.evaluate(()=>calls.length),before);
  // Project-managed Tianyi: Canvas only arms; PICO Start is the session entry.
  await page.evaluate(async()=>{
   const host=document.createElement('div');host.id='managed';document.body.append(host);
   sample={mode:'live',state:'hold',calibrated:true,capture:{connected:true},driver_feedback_fresh:true,driver_feedback:{state:'idle',output_active:false,ownership_held:false},project:{armed:false,stopping:false,error:'',driver_binding:{robot_profile:'tianyi2',mcp_id:'registered-driver'}}};
   ui=mountTeleopPanel(host,{actions:['project_start','project_stop','start','finish','pause','resume','stop','config'],configSchema:{properties:{mode:{type:'string',enum:['shadow','live']},namespace:{type:'string'},driver_mcp_url:{type:'string'}}},loadConfig:async()=>({mode:'live'}),saveConfig:async()=>{},call:async(a)=>{calls.push({a});return structuredClone(sample);}});await ui.refresh();
  });
  const managed=page.locator('#managed');
  assert.equal(await managed.locator('[data-action=start]').count(),0);
  assert.equal(await managed.locator('[data-config-key=namespace]').count(),0);
  assert.equal(await managed.locator('[data-config-key=driver_mcp_url]').count(),0);
  assert.equal(await managed.locator('[data-field=driver]').innerText(),'tianyi2 · registered-driver');
  assert.equal(await managed.locator('[data-field=armed]').innerText(),'未开启');
  assert(await managed.locator('[data-action=resume]').isDisabled());
  await page.evaluate(async()=>{sample.project.armed=true;sample.state='idle';await ui.refresh();});
  assert.equal(await managed.locator('[data-field=title]').innerText(),'等待 PICO 开始');
  assert.match(await managed.locator('[data-field=reason]').innerText(),/PICO 点击开始/);
  assert(await managed.locator('.tp-config button').isDisabled());
  await page.evaluate(async()=>{sample.state='hold';sample.authority_valid=true;await ui.refresh();});
  assert(await managed.locator('[data-action=resume]').isEnabled());
  await page.evaluate(async()=>{sample.project.stopping=true;await ui.refresh();});
  assert.equal(await managed.locator('[data-field=title]').innerText(),'正在收臂');
  assert(await managed.locator('[data-action=finish]').isDisabled());
  assert(await managed.locator('[data-action=stop]').isEnabled());
  await page.evaluate(async()=>{sample.project.stopping=false;sample.project.error='return_collision';await ui.refresh();});
  assert.equal(await managed.locator('[data-field=title]').innerText(),'故障');
  assert.equal(await managed.locator('[data-field=reason]').innerText(),'return_collision');
  await page.evaluate(()=>{ui.dispose();document.querySelector('#managed').remove();});
  // G1 advertises no project/finish capability; standalone Shadow stays usable.
  await page.evaluate(async()=>{
   const host=document.createElement('div');host.id='g1';document.body.append(host);
   sample={mode:'shadow',state:'idle',calibrated:true,capture:{connected:true},driver_feedback_fresh:true,driver_feedback:{state:'idle',output_active:false,ownership_held:false}};
   ui=mountTeleopPanel(host,{actions:['start','pause','resume','stop','config'],configSchema:{properties:{namespace:{type:'string'},driver_mcp_url:{type:'string'}}},loadConfig:async()=>({namespace:'g1',driver_mcp_url:'http://127.0.0.1:15701/mcp'}),saveConfig:async()=>{},call:async()=>structuredClone(sample)});await ui.refresh();
  });
  const g1=page.locator('#g1');
  assert(await g1.locator('[data-action=start]').isEnabled());
  assert.equal(await g1.locator('[data-action=finish]').count(),0);
  assert.equal(await g1.locator('[data-config-key=namespace]').count(),1);
  assert.equal(await g1.locator('[data-field=armed]').innerText(),'未支持项目托管收臂');
  await page.evaluate(async()=>{sample.state='hold';await ui.refresh();});
  assert(await g1.locator('[data-action=resume]').isEnabled());
  await page.evaluate(()=>ui.dispose());
  await page.evaluate(()=>{
   const host=document.createElement('div');host.id='control-v2';document.body.append(host);
   const renderer=Object.create(ControlRenderer);renderer.mount(host);
   renderer.onData(new TextEncoder().encode(JSON.stringify({schema:'motus.control/2',boot_id:'boot',session_id:'session',seq:4,source_seq:10,mapping_epoch:1,mode:'eef_pose',frame:'chest',generated_ns:1000000000,values:Array(14).fill(0)})));
  });
  assert.match(await page.locator('#control-v2').innerText(),/机器人单调时钟；见 Driver 反馈/);
  assert.match(await page.locator('#control-v2').innerText(),/末端1.x/);
  assert.match(await page.locator('#control-v2').innerText(),/4 \/ 10/);
  // New three-stage card: template only edits graph; installer/invitation do
  // not call start, claim, or project_start. QR never uses an external service.
  await page.evaluate(async()=>{
   const host=document.createElement('div');host.id='three-stage';document.body.append(host);
   window.templates=[];window.installations=0;window.invitations=[];
   const qr='<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="white"/></svg>';
   sample={mode:'shadow',state:'idle',capture:{connected:false},project:{armed:false,driver_binding:{protocol_version:2,robot_profile:'tianyi2',tool:'motion_control',mcp_id:'driver',execution_binding:{tool:'arm'}}}};
   ui=mountTeleopPanel(host,{actions:['info','project_start','project_stop','installation_info','create_invitation','revoke_invitation'],
    loadTargets:async()=>[{mcp_id:'driver',label:'展示机器人',robot_profile:'tianyi2'}],buildTemplate:async(id)=>templates.push(id),
    prepareInstallation:async()=>{installations++;return {ticket:'download-only',url:location.origin+'/pico/download-only',qr_svg:qr,package:{version:'fixture',size_bytes:100,sha256:'f'.repeat(64)}};},
    createInvitation:async(ticket)=>{invitations.push(ticket);return {url:location.origin+'/pico/download-only#c2VjcmV0',deep_link:'motus-teleop://connect#c2VjcmV0',qr_svg:qr};},
    call:async(a)=>{calls.push({a});return structuredClone(sample);}});await ui.refresh();
  });
  const three=page.locator('#three-stage');
  assert.equal(await three.locator('[data-field=driver]').innerText(),'tianyi2 · motion_control → arm');
  await three.locator('.tp-topology summary').click();await three.locator('.tp-topology button').click();
  assert.deepEqual(await page.evaluate(()=>templates),['driver']);
  await three.locator('.tp-install summary').click();await three.locator('.tp-download').click();
  await page.waitForFunction(()=>document.querySelector('#three-stage .tp-invite').disabled===false);
  assert.match(await three.locator('.tp-install-qr').getAttribute('src'),/^data:image\/svg\+xml;base64,/);
  await three.locator('.tp-invite').click();await page.waitForFunction(()=>invitations.length===1);
  assert.deepEqual(await page.evaluate(()=>invitations),['download-only']);
  assert.match(await three.locator('.tp-install-link').getAttribute('href'),/#c2VjcmV0$/);
  assert(!((await three.locator('.tp-install-link').innerText()).includes('c2VjcmV0')));
  await three.screenshot({path:process.env.TELEOP_NEW_SCREENSHOT || path.join(require('node:os').tmpdir(),'teleop-three-stage-card.png')});
  await three.locator('.tp-revoke-invite').click();await page.waitForFunction(()=>calls.some(c=>c.a==='revoke_invitation'));
  assert(!(await three.locator('.tp-install-link').getAttribute('href')).includes('#'));
  await page.evaluate(()=>ui.dispose());
  assert.deepEqual(errors,[]);console.log('PASS: rendering, freshness, pairing/XSS, actions, stop concurrency, disposal, Tianyi project lifecycle and G1 capabilities; no robot calls');
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
