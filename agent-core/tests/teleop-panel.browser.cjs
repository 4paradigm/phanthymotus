// Run with a host Playwright installation; all MCP responses are local fixtures.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../web/js/teleop-panel.js'));
const server=http.createServer((req,res)=>{
 if(req.url==='/teleop-panel.js'){res.setHeader('Content-Type','text/javascript');res.end(source);return;}
 res.end(`<html><meta charset="utf-8"><body style="background:#111827;padding:32px"><div id="host"></div><script type="module">
 import {mountTeleopPanel,viewState} from '/teleop-panel.js';
 window.calls=[];window.fail=false;window.wait=false;window.releases=[];window.saved={mode:"shadow",position_scale:.5};window.rejectConfig=false;window.saves=[];
 window.sample={mode:'shadow',state:'idle',calibrated:true,capture:{connected:true,paired_devices:1},driver_feedback_fresh:true,driver_feedback:{state:'idle',ownership_held:false,output_active:false},pose:{fresh:true,age_ms:12,latest:{tracking:{head:true,left:true,right:true}}},enrollment:{window_open:false},operator:{state:'idle'}};
 window.ui=mountTeleopPanel(document.querySelector('#host'),{configSchema:{properties:{mode:{type:'string',enum:['shadow','live']},position_scale:{type:'number',minimum:.01,maximum:1}}},loadConfig:async()=>structuredClone(saved),saveConfig:async(v)=>{if(rejectConfig)throw Error('release_before_config');saves.push(v);saved=v;sample.configuration=v;},actions:['start','pause','resume','finish','stop','open_pairing','approve_pairing','reject_pairing','disconnect_headset','revoke_headset','calibrate','self_test','config'],call:async(a,args)=>{calls.push({a,args});if(a==='info'){if(fail)throw Error('network unavailable');return structuredClone(sample);}if(wait&&a!=='stop')await new Promise(r=>releases.push(r));sample.operator={state:'idle',action:a};return {state:'accepted'};}});window.viewState=viewState;
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
  await page.locator('[data-config-key=position_scale]').fill('.75');await page.evaluate(()=>ui.refresh());
  assert.equal(await page.locator('[data-config-key=position_scale]').inputValue(),'.75');
  await page.evaluate(()=>{rejectConfig=true;});await page.locator('.tp-config button').click();
  await page.waitForFunction(()=>document.querySelector('.tp-config-result').textContent.includes('release_before_config'));
  assert.equal(await page.evaluate(()=>saved.position_scale),.5);
  await page.evaluate(()=>{rejectConfig=false;});await page.locator('.tp-config button').click();await page.waitForFunction(()=>saves.length===1);
  assert.equal(await page.evaluate(()=>saved.position_scale),.75);
  await page.evaluate(async()=>{sample.authority_valid=true;await ui.refresh();});assert(await page.locator('.tp-config button').isDisabled());
  await page.evaluate(async()=>{sample.authority_valid=false;await ui.refresh();});
  await page.screenshot({path:process.env.TELEOP_SCREENSHOT || path.join(require('node:os').tmpdir(),'teleop-card.png'),fullPage:true});
  const before=await page.evaluate(()=>calls.length);await page.evaluate(()=>document.querySelector('#host').remove());await page.waitForTimeout(1200);assert.equal(await page.evaluate(()=>calls.length),before);
  assert.deepEqual(errors,[]);console.log('PASS: rendering, freshness, pairing/XSS, actions, stop concurrency, disposal; no robot calls');
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
