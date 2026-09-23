// Run like teleop-panel.browser.cjs with host Playwright and Chrome.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const web=path.join(__dirname,'../web'), token='ABCDEFGHJKLM';
let expired=false;const downloads=[];
const server=http.createServer((req,res)=>{
  if(req.url==='/js/pico-install.js'){res.setHeader('Content-Type','text/javascript');res.end(fs.readFileSync(path.join(web,'js/pico-install.js')));return;}
  if(req.url.endsWith('/package')){res.setHeader('Content-Type','application/json');res.statusCode=expired?410:200;res.end(JSON.stringify({version:'0.3.test',size_bytes:2400000,sha256:'a'.repeat(64)}));return;}
  if(req.url.endsWith('/apk')){downloads.push(req.url);res.setHeader('Content-Disposition','attachment; filename="motus-pico.apk"');res.end('fixture');return;}
  res.setHeader('Content-Type','text/html; charset=utf-8');res.end(fs.readFileSync(path.join(web,'pico-install.html')));
});
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH});
  try{
    const page=await browser.newPage({viewport:{width:800,height:1050}}),errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    const origin=`http://127.0.0.1:${server.address().port}`;
    await page.goto(origin+'/pico');
    await page.locator('#code').fill('bad');await page.locator('button').click();
    assert.match(await page.locator('#error').innerText(),/核对安装码/);
    await page.locator('#code').fill('abcd-efgh-jklm');await page.locator('button').click();
    await page.waitForURL(origin+'/pico/'+token);
    await page.locator('#apk').waitFor({state:'visible'});
    assert.match(await page.locator('#package').innerText(),/0.3.test/);
    assert.match(await page.locator('#installation').innerText(),/直接点击打开/);
    const download=page.waitForEvent('download');await page.locator('#apk').click();await download;
    assert.deepEqual(downloads,['/pico/'+token+'/apk']);
    await page.screenshot({path:'/tmp/motus-pico-install.png',fullPage:true});
    await page.goto(origin+'/pico/'+token+'#QUJD');
    await page.waitForFunction(()=>document.querySelector('#connect').getAttribute('href')?.includes('QUJD'));
    assert.equal(await page.locator('#connect').getAttribute('href'),'motus-teleop://connect#QUJD');
    expired=true;await page.goto(origin+'/pico/'+token);await page.reload();
    await page.waitForFunction(()=>document.querySelector('#error').textContent.includes('过期'));
    assert(await page.locator('#apk').isHidden());
    await page.locator('#entry-link').click();await page.locator('#code').waitFor({state:'visible'});
    assert.deepEqual(errors,[]);console.log('PICO install browser flow passed: code, download, invitation, expiry recovery.');
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
