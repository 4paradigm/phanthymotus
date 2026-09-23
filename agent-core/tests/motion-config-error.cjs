// Execute the existing shared-config save handler with bounded DOM/HTTP doubles.
// No browser network, ROS or robot is used. Run: node tests/motion-config-error.cjs
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../web/js/sidebar.js'),'utf8');
const begin=source.indexOf('  const save = async () => {');
const end=source.indexOf('\n  };',begin);
assert(begin>=0 && end>begin,'shared config save handler not found');
const save=source.slice(begin,end+5);
(async()=>{
 for(const [toolName,message] of [['motion_control','配置保存失败：Driver 配置未确认：configuration_requires_idle'],['camera','配置保存失败 (HTTP 409)']]){
  const alerts=[],cache={value:'unchanged'};let closed=false,parsed=0;
  const context={ensureEdit:async()=>true,bodyEl:{querySelectorAll:()=>[]},props:{},mcpId:'driver',toolName,
   alert:value=>alerts.push(value),fetch:async()=>({ok:false,status:409,json:async()=>{parsed++;return {detail:'Driver 配置未确认：configuration_requires_idle'};}}),
   _toolConfigs:cache,configKey:'value',close:()=>{closed=true;},console:{error:()=>{}}};
  await vm.runInNewContext(save+'\nsave();',context);
  assert.deepEqual(alerts,[message]);assert.deepEqual(cache,{value:'unchanged'});assert.equal(closed,false);
  assert.equal(parsed,toolName==='motion_control'?1:0);
 }
 console.log('PASS: motion-control config rejection is visible; modal/cache and ordinary-card semantics preserved');
})().catch(e=>{console.error(e);process.exitCode=1;});
