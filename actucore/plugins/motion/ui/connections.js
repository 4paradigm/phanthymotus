/* Saved environment references and their public labels; credentials stay in the host. */
function createConnectionDirectory({call,onChange}){
 let values={endpoint:'',environment_job:''},targets=[],jobs=[];
 const targetFor=job=>targets.find(target=>target.id===job?.target_id);
 function jobLabel(job){const target=targetFor(job),name=target?.label||target?.host||job.target_id||'环境';const mode=job.mode==='reuse'?'已有环境':job.mode==='isolated'?'独立环境':'已准备环境';const stamp=Number(job.created_at);const date=Number.isFinite(stamp)&&stamp>0?new Date(stamp*1000).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}):'';return [name,mode,date].filter(Boolean).join(' · ');}
 function currentLabel(){const job=jobs.find(job=>job.id===values.environment_job);if(job)return jobLabel(job);if(values.environment_job)return '已连接环境 '+values.environment_job.slice(0,8);const target=targets.find(target=>target.existing_service?.endpoint===values.endpoint);if(target)return target.label||target.host||target.id;try{return new URL(values.endpoint).host;}catch{return '未连接';}}
 function setValues(next){values={endpoint:next.endpoint??values.endpoint,environment_job:next.environment_job??values.environment_job};onChange?.();}
 async function load(){
  try{const result=await call('env_targets');if(Array.isArray(result.targets))targets=result.targets;}catch{}
  let history=[];try{const result=await call('inspect',{mode:'env_status'});if(Array.isArray(result.jobs))history=result.jobs;}catch{}
  const merged=new Map([...jobs,...targets.map(target=>target.latest_job).filter(Boolean),...history].map(job=>[job.id,job]));
  if(values.environment_job&&!merged.has(values.environment_job)){try{const result=await call('inspect',{mode:'env_status',job_id:values.environment_job});if(result.job)merged.set(result.job.id,result.job);}catch{}}
  jobs=[...merged.values()];onChange?.();return ready();
 }
 function ready(){const result=jobs.filter(job=>job.state==='succeeded'&&job.action==='install').sort((a,b)=>Number(b.created_at||0)-Number(a.created_at||0)).map(job=>({id:job.id,label:jobLabel(job)+(job.id===values.environment_job?' · 当前连接':'')}));if(values.environment_job&&!result.some(item=>item.id===values.environment_job))result.unshift({id:values.environment_job,label:currentLabel()+' · 当前连接'});return result;}
 return {setValues,load,ready,currentLabel,jobLabel,get currentRef(){return values.environment_job;},get endpoint(){return values.endpoint;},targets:()=>targets};
}
