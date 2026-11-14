import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('./renderers/camera.js',import.meta.url),'utf8');
function fixture(){
 const images=[], live=new Set(), shown=[], accepted=[];let id=0;
 class Element {
  constructor(){this.src='';this.style={};}
  appendChild(){} remove(){} removeAttribute(){this.src='';}
  cloneNode(){const e=new Element();images.push(e);return e;}
  replaceWith(e){shown.push(e.src);}
 }
 const ctx={document:{createElement:()=>new Element()},Blob,performance:{now:()=>2000},URL:{createObjectURL:()=>{const url='blob:'+ ++id;live.add(url);return url;},revokeObjectURL:u=>live.delete(u)}};
 const renderer=runInNewContext(source.replaceAll('export ', '')+';CameraRenderer',ctx);
 renderer.mount(new Element());
 const send=(name,valid=true)=>renderer.onData(new Uint8Array([1]).buffer,'image/jpeg',()=>{if(valid)accepted.push(name);return valid;});
 return {renderer,images,live,shown,accepted,send};
}
test('slow A finishes despite B C D; only D starts next',()=>{
 const f=fixture();f.send('A');const a=f.images[0];const load=a.onload;
 f.send('B');f.send('C');f.send('D');
 assert.equal(a.onload,load);assert.equal(f.images.length,1);assert.equal(f.live.size,1);
 load();assert.deepEqual(f.accepted,['A']);assert.equal(f.shown.length,1);assert.equal(f.images.length,2);
 f.images[1].onload();assert.deepEqual(f.accepted,['A','D']);assert.equal(f.live.size,1);
 f.renderer.unmount();assert.equal(f.live.size,0);
});
test('error retains good frame and advances only latest pending',()=>{
 const f=fixture();f.send('good');f.images[0].onload();const good=f.renderer._img;
 f.send('bad');const bad=f.images[1];f.send('B');f.send('C');bad.onerror();
 assert.equal(f.renderer._img,good);assert.deepEqual(f.accepted,['good']);assert.equal(f.images.length,3);
 f.images[2].onload();assert.deepEqual(f.accepted,['good','C']);
});
test('unmount cancels load and queued frame; saved late callbacks do nothing',()=>{
 const f=fixture();f.send('A');const late=f.images[0].onload;f.send('B');f.renderer.unmount();late();
 assert.equal(f.live.size,0);assert.equal(f.shown.length,0);assert.equal(f.accepted.length,0);
});
test('obsolete connection callback cannot swap image',()=>{
 const f=fixture();f.send('old',false);f.images[0].onload();assert.equal(f.shown.length,0);assert.equal(f.live.size,0);
});
