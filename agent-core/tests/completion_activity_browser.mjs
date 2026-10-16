/** Actual Canvas + activity-strip handlers; simulated ACP, no hardware. */
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.CORE_TEST_PLAYWRIGHT || 'playwright');
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../web');
const canvas=await readFile(path.join(web,'js/canvas.js'),'utf8');
const activity=await readFile(path.join(web,'js/activity-log.js'),'utf8');
function fn(source,name){
 const start=source.indexOf(`function ${name}(`);assert.ok(start>=0,name);
 return (source.slice(start-6,start)==='async '?'async ':'')+source.slice(start,source.indexOf('\n}\n',start)+3);
}
const entry=`
import {completionRequest,watchCompletion} from '/action-completion.js';
import {claimCompletionActivity,preferManualCompletionActivity,activityCompletionId} from '/completion-activity.js';
import {summarizeEvent} from '/activity-summary.js';
const _cardExecutions=new WeakMap();
const _allMcps=[{id:'sim-arm',tools:[{name:'arm_control',inputSchema:{properties:{},'x-completion':{actions:['move'],timeout:3}}}]}];
const _esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
${['_executeCard','_formatCallResult','_truncate','_showResult','_logActivity'].map(n=>fn(canvas,n)).join('\n')}
${fn(activity,'_append')}
window.run=()=>_executeCard(document.querySelector('#arm'),'sim-arm','arm_control');
window.collect=id=>_append({type:'trigger',ts:Date.now()/1000,mcp_id:'collector',payload:{text:
 '<event source="acp:arm_control" channel="sensor" ts="2026-10-16 12:04:40">\\n'+JSON.stringify({type:'action_complete',action_id:id,status:'completed',silent:false})+'\\n</event>'}});
window.ready=true;
`;
let scenario;
const server=createServer(async(req,res)=>{
 try{
  if(req.url==='/'){
   res.setHeader('Content-Type','text/html');res.end('<meta charset="utf-8"><div id="activity-strip"><div id="activity-latest"></div><div id="activity-log"></div></div><div id="arm"><input class="canvas-field-input" data-key="action" value="move"><button class="canvas-exec-btn">Execute</button></div><script type="module" src="/entry.js"></script>');return;
  }
  if(req.url==='/entry.js'){res.setHeader('Content-Type','application/javascript');res.end(entry);return;}
  if(['/action-completion.js','/completion-activity.js','/activity-summary.js'].includes(req.url)){
   res.setHeader('Content-Type','application/javascript');res.end(await readFile(path.join(web,'js',req.url.slice(1))));return;
  }
  let data;
  if(req.url==='/api/mcp/sim-arm/call'){
   for await(const chunk of req){};
   scenario.called=true;await scenario.response;
   data={code:200,data:[{type:'text',text:JSON.stringify({state:'running',action_id:scenario.id})}]};
  }else if(req.url.startsWith('/api/mcp/actions/')){
   scenario.polling=true;await scenario.receipt;
   data={code:200,data:{action_id:scenario.id,state:'settled',status:'completed',result:{executed:true,actual:[1,2,3]}}};
  }
  if(data){res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));return;}
  res.statusCode=404;res.end();
 }catch(error){res.statusCode=500;res.end(String(error));}
});
const deferred=()=>{let resolve;return {promise:new Promise(r=>{resolve=r;}),resolve:()=>resolve()};};
async function until(test){for(let n=0;n<200;n++){if(test())return;await new Promise(r=>setTimeout(r,10));}throw Error('Simulated request not received');}
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:process.env.CORE_TEST_BROWSER});
const page=await browser.newPage();const errors=[],passed=[];
page.on('pageerror',error=>errors.push(String(error)));
try{
 for(const order of ['collector-before-response','collector-before-poll','collector-after-poll','no-collector']){
  const response=deferred(),receipt=deferred();
  scenario={id:'arm_control_'+order,response:response.promise,receipt:receipt.promise};
  await page.goto(`http://127.0.0.1:${server.address().port}`);await page.waitForFunction(()=>window.ready);
  await page.evaluate(()=>{window.pending=window.run();});await until(()=>scenario.called);
  if(order==='collector-before-response'){
   await page.evaluate(id=>window.collect(id),scenario.id);
   assert.equal(await page.locator('.log-type.trigger').count(),1);
  }
  response.resolve();await page.evaluate(()=>window.pending);await until(()=>scenario.polling);
  assert.equal(await page.locator('.log-type.trigger').count(),0);
  if(order==='collector-before-poll')await page.evaluate(id=>window.collect(id),scenario.id);
  assert.equal(await page.locator('.log-type.trigger').count(),0);
  receipt.resolve();await page.waitForFunction(()=>document.querySelector('#arm pre')?.textContent.includes('"completed"'));
  if(order==='collector-after-poll')await page.evaluate(id=>window.collect(id),scenario.id);
  const rows=page.locator('.log-entry').filter({has:page.locator('.log-type.trigger')});
  assert.equal(await rows.count(),1);
  assert.match(await rows.innerText(),new RegExp(`arm_control \\[${scenario.id}\\] → completed:`));
  assert.equal(await page.locator('#activity-log').getByText('[collector]',{exact:false}).count(),0);
  assert.match(await page.locator('#activity-latest').innerText(),/arm_control \[.*\] → completed:/);
  // A separate LLM-only action still needs its collector activity entry.
  await page.evaluate(()=>window.collect('llm-only'));
  assert.equal(await page.locator('.log-type.trigger').count(),2);
  passed.push(order);
 }
 assert.deepEqual(errors,[]);
 if(process.env.CORE_TEST_OUTPUT){
  await mkdir(process.env.CORE_TEST_OUTPUT,{recursive:true});
  await writeFile(path.join(process.env.CORE_TEST_OUTPUT,'activity-browser-result.json'),JSON.stringify({passed,errors,hardware:false},null,2));
 }
 console.log(JSON.stringify({passed:passed.length,errors,hardware:false}));
}finally{await browser.close();await new Promise(r=>server.close(r));}
