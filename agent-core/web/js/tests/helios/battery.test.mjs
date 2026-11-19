import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const base=new URL('../../',import.meta.url);
const read=name=>fs.readFile(new URL(name,base),'utf8');
const strip=s=>s.replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
class Element {
 constructor(){this.children=[];this.style={};this.classList={add(){},remove(){}};this.textContent='';this.scrollTop=0;}
 appendChild(c){c.parent=this;this.children.push(c);return c;}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
 removeChild(c){c.remove();}
 get firstChild(){return this.children[0];}
 set innerHTML(s){this.html=s;this.children=[];for(const _ of s.matchAll(/<details>/g)){const d=new Element();d.tagName='details';d.open=false;this.appendChild(d);}}
 get innerHTML(){return this.html;}
 querySelector(selector){for(const c of this.children){if(selector==='details[open]'&&c.tagName==='details'&&c.open)return c;const found=c.querySelector(selector);if(found)return found;}return null;}
 get scrollHeight(){return this.children.length*100;}
 getBoundingClientRect(){return {top:this.parent?(this.parent.getBoundingClientRect().top+this.parent.children.indexOf(this)*100-this.parent.scrollTop):0};}
}
let now=0,wall=0;const timers=new Set();
class Clock extends Date {toLocaleTimeString(){return `10:00:${String(wall).padStart(2,'0')}`;}}
const c=vm.createContext({Date:Clock,TextDecoder,console,performance:{now:()=>now},
 document:{createElement:()=>new Element()},
 setInterval:fn=>{timers.add(fn);return fn;},clearInterval:fn=>timers.delete(fn)});
vm.runInContext(strip(await read('renderers/json-util.js')),c);
vm.runInContext(strip(await read('renderers/battery-view.js')),c);
vm.runInContext('{'+strip(await read('renderers/text.js'))+';globalThis.text=TextRenderer;}',c);
vm.runInContext('{'+strip(await read('renderers/kv-latest.js'))+';globalThis.kv=KvLatestRenderer;}',c);
const live=JSON.parse(await fs.readFile(new URL('./fixtures/battery.json',import.meta.url),'utf8'));
const frame=structuredClone(live.frames[0]);
const encode=o=>new TextEncoder().encode(JSON.stringify(o));
const entries=c.batteryEntries(frame);
assert.equal(entries.length,2);
for(let i=0;i<2;i++){
 const pack=frame.batteries.find(p=>p.id===String(i+1)), row=entries[i];
 assert.equal(row.values.battery_id,pack.id);
 assert.equal(row.values.soc,pack.soc_pct);
 assert.ok(!('soh' in row.values));
 assert.equal(row.values.voltage,Math.round(pack.voltage_v*1000));
 assert.ok(!('current' in row.values));
 assert.ok(!('current' in row.details.units));
 assert.equal(row.values.cell_vol_min,Math.round(pack.cell_voltage_min_v*1000));
 assert.equal(row.values.cell_vol_max,Math.round(pack.cell_voltage_max_v*1000));
 assert.equal(row.values.temperature,pack.temperature_max_c);
 assert.equal(row.details.units.temperature,'°C');
 assert.ok(!('temperature_range' in row.values));
 assert.ok(!('charging' in row.values));
 assert.ok(!('charging_source' in row.details));
 assert.equal(row.values.cycle,pack.cycle_count);
 assert.equal(row.details.remaining_capacity,pack.remaining_capacity_mah);
 assert.equal(row.details.bms_hardware_version,pack.bms_hardware_version);
 assert.equal(row.details.units.voltage,'mV');
 assert.ok(!('cell_vol' in row.values),'a range must not masquerade as individual cells');
}
const text=c.text; text.mount(new Element());text.onData(encode(frame));
assert.equal(text._textEl.children.length,1);
assert.equal((text._textEl.children[0].html.match(/class="log-time"/g)||[]).length,1);
assert.equal((text._textEl.children[0].html.match(/<b>battery_id<\/b>/g)||[]).length,2);
assert.doesNotMatch(text._textEl.children[0].html,/<b>soh<\/b>/);
assert.doesNotMatch(text._textEl.children[0].html,/<b>current<\/b>|temperature_range/);
assert.match(text._textEl.children[0].html,/<b>temperature<\/b>/);
assert.doesNotMatch(text._textEl.children[0].html,/charging/);
for(const row of text._textEl.children){
 assert.equal(row.className,'log-entry');
 assert.match(row.html,/<span class="log-time">10:00:00<\/span>/);
 assert.match(row.html, /class="log-msg" style="color:var\(--text-muted\);font-size:12px;white-space:pre-wrap"/);
 assert.match(row.html,/<b>soc<\/b>/);assert.match(row.html,/<b>voltage<\/b>/);
 assert.match(row.html,/<details><summary>Details<\/summary>/);
 assert.doesNotMatch(row.html,/[\u4e00-\u9fff]/);
}
wall=1;text.onData(encode(frame));
assert.match(text._textEl.children[1].html,/10:00:01/);
assert.match(text._textEl.children[0].html,/10:00:00/,'history keeps original timestamps');
for(let i=0;i<100;i++)text.onData(encode(frame));assert.equal(text._textEl.children.length,50);
// Reading either battery's details must survive ongoing frames and history trimming.
const container=text._textEl, inspected=container.children[10];
const detail=inspected.children[1];detail.open=true;
container.scrollTop=1000;const originalTop=detail.getBoundingClientRect().top;
const originalHtml=inspected.html;
for(let i=0;i<120;i++)text.onData(encode(frame));
assert.equal(container.children.length,50);
assert.ok(container.children.includes(inspected),'open history must not be evicted');
assert.equal(detail.open,true);assert.equal(inspected.html,originalHtml,'historical values and timestamp stay intact');
assert.equal(detail.getBoundingClientRect().top,originalTop,'reading position survives trimming above the open record');
detail.open=false;text.onData(encode(frame));
assert.equal(container.scrollTop,container.scrollHeight,'closing all details resumes following the latest frames');
assert.ok(!container.children.includes(inspected),'closed old history can be trimmed again');
// Bound memory even if the user opens every retained frame.
for(const row of container.children)row.children[0].open=true;
for(let i=0;i<100;i++)text.onData(encode(frame));
assert.equal(container.children.length,50);
for(const row of container.children)row.children[0].open=false;
const unsafe=structuredClone(frame);unsafe.batteries[0].bms_hardware_version='<img src=x onerror=alert(1)>';
text.onData(encode(unsafe));assert.match(text._textEl.children.at(-1).html,/&lt;img/);assert.doesNotMatch(text._textEl.children.at(-1).html,/<img/);
// Native G1 payloads follow byte-identical rendering, including timestamp/CSS.
const g1={soc:91,soh:91,current:-1898,voltage:[52465,52500],cell_vol:[4041,4043],temperature:[29,28,27,29],cycle:112};
text.onData(encode(g1));
const g1Snapshot=await fs.readFile(new URL('./fixtures/g1-log.html',import.meta.url),'utf8');
assert.equal(text._textEl.children.at(-1).html,g1Snapshot);
const zero=structuredClone(frame);zero.batteries[0].soc_pct=0;zero.batteries[0].temperature_max_c=0;
assert.equal(c.batteryEntries(zero)[0].values.soc,0);assert.equal(c.batteryEntries(zero)[0].values.temperature,0);
const warmer=structuredClone(frame);warmer.batteries[0].temperature_min_c=15;warmer.batteries[0].temperature_max_c=29;
assert.equal(c.batteryEntries(warmer)[0].values.temperature,29);
assert.equal(c.batteryEntries(warmer)[1].values.temperature,entries[1].values.temperature);
const missingTemp=structuredClone(frame);delete missingTemp.batteries[0].temperature_max_c;
assert.equal(c.batteryEntries(missingTemp)[0].values.temperature,null,'never substitute another pack or an aggregate temperature');
for(const current of [0,120,-120,null]){
 const unverified=structuredClone(frame);unverified.batteries[0].current_ma=current;unverified.charge_arrive=1;
 assert.ok(!('charging' in c.batteryEntries(unverified)[0].values),'current or dock sensor cannot establish charging');
}
const reversed=structuredClone(frame);reversed.batteries.reverse();assert.equal(c.batteryEntries(reversed)[0].values.battery_id,'1');
const partial=structuredClone(frame);partial.batteries[1].available=false;
assert.equal(c.batteryEntries(partial)[1].values.voltage,null);
assert.equal(c.batteryEntries(partial)[0].values.voltage,entries[0].values.voltage);
const kv=c.kv;kv.mount(new Element());kv.onData(encode(frame));
assert.equal(kv._cells['battery_1.voltage'].valueEl.textContent,String(entries[0].values.voltage));
assert.equal(kv._cells['battery_1.soh'],undefined);
assert.equal(kv._cells['battery_1.current'],undefined);
assert.equal(kv._cells['battery_1.temperature_range'],undefined);
assert.equal(kv._cells['battery_1.temperature'].valueEl.textContent,String(entries[0].values.temperature));
assert.equal(kv._cells['battery_1.charging'],undefined);
assert.equal(kv._cells.updated_at.valueEl.textContent,'10:00:01');
assert.equal(timers.size,1);
wall=2;kv.onData(encode(frame));assert.equal(kv._cells.updated_at.valueEl.textContent,'10:00:02');
now=4000;for(const fn of timers)fn();
assert.equal(kv._cells['battery_1.voltage'].valueEl.textContent,'—');
assert.equal(kv._cells['battery_1.temperature'].valueEl.textContent,'—');
assert.equal(kv._cells['battery_1.charging'],undefined);
assert.equal(kv._cells['battery_1.state'].valueEl.textContent,'stale');
assert.equal(kv._cells.updated_at.valueEl.textContent,'10:00:02','silence must not fabricate timestamp updates');
kv.onData(encode({state:'online',value:42}));assert.equal(timers.size,0);assert.equal(kv._cells.value.valueEl.textContent,'42');
kv.onData(encode({state:'online',angles_rad:{left_joint_1:0.25}}));
assert.equal(Number(kv._cells['angles_rad.left_joint_1'].valueEl.textContent),0.25);
kv.onData(encode({state:'stale',angles_rad:{}}));
assert.equal(kv._cells['angles_rad.left_joint_1'],undefined,'an empty joint snapshot removes old angles');
kv.onData(encode(frame));kv.unmount();assert.equal(timers.size,0);
console.log('PASS: Details stays open and anchored across 120 frames, pinned history and bounded 50-row retention, resume following on close, native log CSS/time, English fields, current/SOH/charging hidden, realtime temperature, escaping, latest stale cleanup, unchanged G1 logs');
