import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
const source = readFileSync(new URL('./monitor-dashboard.js', import.meta.url), 'utf8');
function fixture() {
 const cards=new Map(), timers=new Map(); let next=0;
 const ctx={_cards:cards, ArrayBuffer, Blob, createImageBitmap:async blob=>{
  if ((await blob.arrayBuffer()).byteLength !== 2) throw Error('bad image');
  return {close(){}};
 },setTimeout:fn=>{timers.set(++next,fn);return next;},clearTimeout:id=>timers.delete(id),
 _handleWsMessage:()=>true};
 const code=source.slice(source.indexOf('function _wsHandler('),source.indexOf('function _refreshRenderer('));
 const handler=runInNewContext(code+';_wsHandler',ctx);
 function card(key) {
  const status={hidden:true}; const c={ws:{},el:{classList:{add(){}},querySelector:()=>status},renderer:{async onData(data, format, ready){ await Promise.resolve(); if(data.byteLength===2) ready(); }}};cards.set(key,c);
  return {c,status,send:handler(key,()=>c.renderer,'image/jpeg',c.ws)};
 }
 return {cards,timers,card};
}
test('no timer before first valid frame; nonframes and bad images ignored',async()=>{
 const f=fixture(), a=f.card('/a');
 for(const data of ['{"type":"ping"}','{"type":"meta"}','{"type":"error"}',new ArrayBuffer(0),new ArrayBuffer(1)]) await a.send({data});
 assert.equal(f.timers.size,0);assert.equal(a.c.gotData,undefined);
});
test('cards expire independently; valid recovery only resets its own notice',async()=>{
 const f=fixture(),a=f.card('/a'),b=f.card('/b');
 await a.send({data:new ArrayBuffer(2)});await b.send({data:new ArrayBuffer(2)});
 for(const callback of f.timers.values())callback();
 assert.equal(a.status.hidden,false);assert.equal(b.status.hidden,false);
 await a.send({data:new ArrayBuffer(2)});assert.equal(a.status.hidden,true);assert.equal(b.status.hidden,false);
});
test('old socket cannot update replacement card',async()=>{
 const f=fixture(),a=f.card('/a'),replacement=f.card('/a');
 await a.send({data:new ArrayBuffer(2)});assert.equal(replacement.c.gotData,undefined);assert.equal(f.timers.size,0);
});
test('in-flight decode cannot update a removed card',async()=>{
 const f=fixture(),a=f.card('/a');const pending=a.send({data:new ArrayBuffer(2)});f.cards.clear();await pending;
 assert.equal(f.timers.size,0);assert.equal(a.c.gotData,undefined);
});
test('renderer rebuild invalidates in-flight decode',async()=>{
 const f=fixture(),a=f.card('/a');const pending=a.send({data:new ArrayBuffer(2)});a.c.generation=1;await pending;
 assert.equal(f.timers.size,0);
});
