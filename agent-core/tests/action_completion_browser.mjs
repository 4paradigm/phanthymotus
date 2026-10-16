/** Real Canvas functions and completion module; localhost fake device only. */
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.CORE_TEST_PLAYWRIGHT || 'playwright');
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../web');
const out=process.env.CORE_TEST_OUTPUT;await mkdir(out,{recursive:true});
const tools=JSON.parse(await readFile(process.env.CORE_TEST_TOOLS,'utf8'));
const source=await readFile(path.join(web,'js/canvas.js'),'utf8');
function fn(name){
 const start=source.indexOf(`function ${name}(`);assert.ok(start>=0,name);
 return (source.slice(start-6,start)==='async '?'async ':'')+source.slice(start,source.indexOf('\n}\n',start)+3);
}
const entry=`
import {completionRequest,watchCompletion} from '/action-completion.js';
import {claimCompletionActivity,preferManualCompletionActivity} from '/completion-activity.js';
const _cardExecutions=new WeakMap();
const _allMcps=[{id:'sim-arm',tools:${JSON.stringify(tools)}}];
const _cards=[];
const _projectRunning=true;
const _logActivity=(type,text)=>window.logs.push({type,text});
const _esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
${['_buildCardEl','_fmtColorClass','_executeCard','_formatCallResult','_truncate','_showResult'].map(fn).join('\n')}
window.logs=[];
for(const toolName of ['joints','piper_camera_live']){
 const el=_buildCardEl({id:toolName,mcpId:'sim-arm',toolName,driverName:'PiPER simulated',x:0,y:0});
 document.querySelector('#sensors').appendChild(el);
}
window.run=(action='move',el=document.querySelector('#arm'))=>{
 el.querySelector('input').value=action;return _executeCard(el,'sim-arm','arm_control','test-card');
};
window.ready=true;
`;
let kind='completed', serial=0, receiptGate;
const receipts=new Map(),requests=[];
const server=createServer(async(req,res)=>{
 try{
  if(req.url==='/'){
   res.setHeader('Content-Type','text/html');res.end(`<meta charset="utf-8"><title>PiPER simulated acceptance</title><style>body{font:16px system-ui;background:#eee;padding:20px}#sensors{display:flex;gap:20px}.canvas-card{background:white;padding:16px;border:1px solid #aaa}pre{white-space:pre-wrap}.error{color:#b00}#arm{margin-top:20px}</style><h2>Simulated only — no hardware</h2><div id="activity-log"></div><div id="sensors"></div><div id="arm" class="canvas-card"><h3>Arm control</h3><input class="canvas-field-input" data-key="action" value="move"><button class="canvas-exec-btn">Execute</button></div><script type="module" src="/entry.js"></script>`);return;
  }
  if(req.url==='/entry.js'){res.setHeader('Content-Type','application/javascript');res.end(entry);return;}
  if(['/action-completion.js','/completion-activity.js'].includes(req.url)){res.setHeader('Content-Type','application/javascript');res.end(await readFile(path.join(web,'js',req.url.slice(1))));return;}
  let data;
  if(req.url==='/api/mcp/sim-arm/call'){
   let raw='';for await(const chunk of req)raw+=chunk;
   const body=JSON.parse(raw);requests.push(body);
   if(body.arguments.action==='get_status')data={code:200,data:[{type:'text',text:'{"state":"idle","snapshot":{"fresh":true},"task":{"state":"idle"}}'}]};
   else{
    const action_id=`sim-${++serial}`;
    receipts.set(action_id,{action_id,state:'settled',status:kind,result:{actual:[1,2,3],error:kind==='error'?'<b>Rejected</b>':undefined}});
    data={code:200,data:[{type:'text',text:JSON.stringify({state:'running',action_id})}]};
   }
  }else if(req.url.startsWith('/api/mcp/actions/')){
   const id=decodeURIComponent(req.url.split('/').at(-1));
   const gate=receiptGate;
   if(gate){gate.seen(id);await gate.promise;}
   else await new Promise(r=>setTimeout(r,120));
   data={code:200,data:receipts.get(id)};
  }
  if(data){res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));return;}
  res.statusCode=404;res.end();
 }catch(e){res.statusCode=500;res.end(String(e));}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:process.env.CORE_TEST_BROWSER});
const page=await browser.newPage({viewport:{width:1100,height:800}});const errors=[];
page.on('pageerror',e=>errors.push(String(e)));const passed=[];
try{
 await page.goto(`http://127.0.0.1:${server.address().port}`);await page.waitForFunction(()=>window.ready);
 for(const name of ['joints','piper_camera_live']){
  const sensor=page.locator(`[data-card-id="${name}"]`);
  assert.equal(await sensor.locator('.canvas-field-input,.canvas-exec-btn').count(),0);
  assert.equal(await sensor.locator('.canvas-view-btn').count(),1);
 }
 assert.equal(await page.locator('[data-card-id="piper_camera_live"] .canvas-port.out').getAttribute('data-format'),'image/jpeg');
 passed.push('Both sensor cards use the actual Canvas read-only layout');
 for(const [status,state] of [['completed','completed'],['error','failed'],['cancelled','cancelled']]){
  kind=status;await page.evaluate(()=>window.run());
  assert.match(await page.locator('#arm pre').innerText(),/running/);
  assert.equal(await page.locator('#arm button').isEnabled(),true);
  await page.waitForFunction(state=>JSON.parse(document.querySelector('#arm pre').textContent).state===state,state);
  assert.equal(await page.locator('#arm pre b').count(),0);
  passed.push(`running changes to ${state} with safely rendered actual result`);
 }
 for(const [status,state] of [['completed','completed'],['error','failed'],['cancelled','cancelled']]){
  kind=status;
  let release,seen;
  const received=new Promise(resolve=>{seen=resolve;});
  receiptGate={seen,promise:new Promise(resolve=>{release=resolve;})};
  await page.evaluate(()=>window.run());
  const actionId=JSON.parse(await page.locator('#arm pre').innerText()).action_id;
  assert.equal(await received,actionId); // Hold the actual fetch callback until the next call finishes.
  await page.evaluate(()=>window.run('get_status'));
  const statusText=await page.locator('#arm pre').innerText();
  assert.equal(JSON.parse(statusText).state,'idle');
  receiptGate=null;release();
  await page.waitForFunction(({id,state})=>window.logs.some(log=>
   log.type==='trigger' && log.text.includes(`[${id}] → ${state}:`)),{id:actionId,state});
  assert.equal(await page.locator('#arm pre').innerText(),statusText);
  const completionLogs=await page.evaluate(id=>window.logs.filter(log=>
   log.type==='trigger' && log.text.includes(`[${id}]`)),actionId);
  assert.equal(completionLogs.length,1);
  passed.push(`Late ${state} preserves the newer Get status result and still emits one trigger`);
 }
 await page.screenshot({path:path.join(out,'three-card-completion.png'),fullPage:true});
 assert.deepEqual(errors,[]);assert.equal(requests.length,9);
 await writeFile(path.join(out,'browser-result.json'),JSON.stringify({passed,requests,errors},null,2));
 console.log(JSON.stringify({passed:passed.length,errors}));
}finally{await browser.close();await new Promise(r=>server.close(r));}
