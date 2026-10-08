// DOM-level regression using the actual sidebar rendering and Save handler.
// Run directly with Node 18+, or via test_sidebar_hotwords.py. No npm dependencies.
const fs=require('fs'), vm=require('vm'), assert=require('assert');
class El {
 constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.style={};this.value='';this.listeners={};this.classList={add(){},remove(){}};}
 set innerHTML(v){this.children=[];}
 appendChild(e){e.parentNode=this;this.children.push(e);return e;}
 replaceChildren(){this.children=[];}
 setAttribute(k,v){this[k]=v;}
 focus(){}
 cloneNode(){return new El(this.tagName);}
 replaceChild(n,o){this.children[this.children.indexOf(o)]=n;n.parentNode=this;this.replaced=n;}
 addEventListener(k,f){this.listeners[k]=f;}
 querySelectorAll(s){let a=this.children.flatMap(c=>[c,...c.querySelectorAll('*')]);if(s==='*')return a;if(s==='[data-key]')return a.filter(e=>e.dataset.key);if(s.includes('data-show-when'))return a.filter(e=>e.dataset.showWhen);if(s.includes('data-hide-when'))return a.filter(e=>e.dataset.hideWhen);return [];}
 querySelector(s){const m=s.match(/data-key="([^"]+)"/);return m?this.querySelectorAll('[data-key]').find(e=>e.dataset.key===m[1]):null;}
 closest(){return this.parentNode;}
}
const ids={};for(const id of ['tool-config-overlay','tool-config-title','tool-config-body','tool-config-save'])ids[id]=new El();
const parent=new El();parent.appendChild(ids['tool-config-save']);let saved;
const ctx={document:{getElementById:id=>ids[id]||null,createElement:t=>new El(t)},console,ensureEdit:async()=>true,isProjectRunning:()=>false,isDisplayOnly:()=>false,fetch:async(u,o)=>{saved=JSON.parse(o.body);return {ok:true}},alert:m=>{throw Error(m)}};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(require('path').join(__dirname, '../web/js/sidebar.js'),'utf8').replace(/^import .*;$/gm,'').replace(/\bexport /g,''),ctx);
ctx.schema = {properties: {
 asr_model: {type:'string', scope:'shared', oneOf:[{const:'x-asr-zh-en',title:'X-ASR（中英文）'}]},
 device: {type:'string', scope:'shared', enum:['cpu','gpu'], default:'cpu'},
 trigger_mode: {type:'string', scope:'shared', oneOf:[{const:'asr_kws',title:'唤醒词触发'}], default:'asr_kws'},
 asr_kws_keyword: {type:'string', scope:'shared', 'x-allow-empty':true},
 asr_hotwords: {type:'string', scope:'shared', format:'hotwords', default:'', 'x-allow-empty':true,
               'x-show-when':{asr_model:'x-asr-zh-en'}},
}};
(async()=>{
 vm.runInContext("_scroll={querySelector:()=>null}; _toolConfigs={'p:asr':{asr_model:'x-asr-zh-en',device:'gpu',asr_hotwords:'旧词',asr_kws_keyword:'旧唤醒'}}",ctx);
 await vm.runInContext("openToolConfigModal('p','asr',schema)",ctx);
 const body=ids['tool-config-body'],field=k=>body.querySelector(`[data-key="${k}"]`);
 assert.equal(body.querySelectorAll('[data-key]').length,5);
 const hotwords=field('asr_hotwords'), tags=hotwords.children[0], entry=hotwords.children[1];
 assert.equal(entry.tagName,'input');
 assert.equal(field('trigger_mode').value,'asr_kws');
 assert.equal(field('asr_model').children[1].textContent,'X-ASR（中英文）');
 assert.equal(tags.children[0].children[0].textContent,'旧词');
 const enter=()=>entry.listeners.keydown({key:'Enter',preventDefault(){}});
 entry.value='Fancy Robot';enter();
 entry.value='Fancy Robot';enter();
 assert.equal(hotwords.value,'旧词\nFancy Robot');
 entry.value='星河展厅';
 entry.listeners.keydown({key:'Enter',isComposing:true,preventDefault(){throw Error('IME was intercepted')}});
 assert.equal(tags.children.length,2);
 enter();
 entry.value='';entry.selectionStart=0;entry.selectionEnd=0;
 entry.listeners.paste({clipboardData:{getData:()=> '万神殿\r\nFancy Robot\r\n小星小星'},preventDefault(){}});
 assert.equal(tags.children.length,5);
 tags.children[0].children[1].listeners.click();
 assert(!hotwords.value.includes('旧词'));
 entry.value='尚未按回车';
 await parent.replaced.listeners.click();
 assert.equal(saved.asr_hotwords,'Fancy Robot\n星河展厅\n万神殿\n小星小星\n尚未按回车');
 entry.value='';
 while(tags.children.length)tags.children[0].children[1].listeners.click();
 field('asr_kws_keyword').value='';
 await parent.replaced.listeners.click();
 assert.equal(saved.asr_hotwords,'');assert.equal(saved.asr_kws_keyword,'');
 console.log('Tag card passed: Enter, IME, paste, deduplication, English spaces, removal, pending input save, clear.');
})().catch(e=>{console.error(e);process.exitCode=1});
