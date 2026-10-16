import test from 'node:test';
import assert from 'node:assert/strict';
import { completionRequest, watchCompletion } from './action-completion.js';

const schema = {'x-completion': {actions: ['move'], timeout: 60}};
const envelope = value => [{type: 'text', text: JSON.stringify(value)}];
const response = value => ({ok: true, json: async () => ({code: 200, data: value})});
const settled = (status, id = 'a') => ({action_id: id, state: 'settled', status, result: {actual: [1, 2, 3]}});

test('only declared async actions with a nonterminal correlated ID are watched', () => {
  assert.deepEqual(completionRequest(envelope({state:'running',action_id:'a'}),schema,'move'),
    {actionId:'a',timeoutMs:65000});
  for (const value of [{state:'completed',action_id:'a'}, {state:'running'}, {action_id:12}, {action_id:''}])
    assert.equal(completionRequest(envelope(value),schema,'move'),null);
  assert.equal(completionRequest(envelope({action_id:'a'}),schema,'info'),null);
  assert.equal(completionRequest(envelope({action_id:'a'}),{},'move'),null);
  assert.equal(completionRequest([{type:'text',text:'not JSON'}],schema,'move'),null);
});

for (const [status,state] of [['completed','completed'],['error','failed'],['failed','failed'],['cancelled','cancelled']]) {
  test(`pending then ${status} produces one real terminal, only read requests`, async () => {
    const calls=[], results=[];let count=0;
    await watchCompletion({actionId:'a',timeoutMs:1000,isCurrent:()=>true,sleep:async()=>{},
      fetchStatus:async(url,options)=>{
        calls.push([url,options]);
        return response(count++ ? settled(status) : {action_id:'a',state:'pending'});
      }, onResult:(...args)=>results.push(args)});
    assert.equal(calls.length,2);
    assert.ok(calls.every(([url,options])=>url==='/api/mcp/actions/a' && !options.method && !options.body));
    assert.deepEqual(results,[[{state,action_id:'a',result:{actual:[1,2,3]}},state!=='completed']]);
  });
}

test('early completion is consumed by the first read',async()=>{
  let calls=0, result;
  await watchCompletion({actionId:'early/id',timeoutMs:1000,isCurrent:()=>true,
    fetchStatus:async url=>{assert.equal(url,'/api/mcp/actions/early%2Fid');calls++;return response(settled('completed','early/id'));},
    onResult:r=>result=r});
  assert.equal(calls,1);assert.equal(result.state,'completed');
});

test('newer call or removed card suppresses late completion',async()=>{
  let current=true,resolve;const results=[];
  const promise=watchCompletion({actionId:'a',timeoutMs:1000,isCurrent:()=>current,
    fetchStatus:()=>new Promise(r=>resolve=r),onResult:r=>results.push(r)});
  current=false;resolve(response(settled('completed')));await promise;
  assert.deepEqual(results,[]);
});

test('wrong IDs, malformed responses and connection failures never manufacture success',async()=>{
  let tick=0;const results=[];
  await watchCompletion({actionId:'a',timeoutMs:10,isCurrent:()=>true,now:()=>tick,
    sleep:async()=>{tick+=5;},fetchStatus:async()=>{
      if(tick===0)throw Error('offline');return response(settled('completed','another-arm'));},
    onResult:r=>results.push(r)});
  assert.equal(results.length,1);assert.equal(results[0].state,'unconfirmed');
});

for(const state of ['unknown','unconfirmed'])test(`${state} record stops polling without claiming robot success`,async()=>{
  const results=[];let calls=0;
  await watchCompletion({actionId:'a',timeoutMs:1000,isCurrent:()=>true,
    fetchStatus:async()=>{calls++;return response({action_id:'a',state,status:'timeout'});},
    onResult:r=>results.push(r)});
  assert.equal(calls,1);assert.equal(results[0].state,'unconfirmed');
});
