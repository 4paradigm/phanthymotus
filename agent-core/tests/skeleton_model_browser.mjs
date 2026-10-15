/** Real Core browser modules with fake MCP/DDS; never connects to a robot.
 * CORE_TEST_THREE: directory containing three.module.js and OrbitControls.js
 * CORE_TEST_PLAYWRIGHT: installed playwright module; CORE_TEST_BROWSER: binary
 * CORE_TEST_URDF: PiPER visualization URDF; CORE_TEST_OUTPUT: evidence directory
 */
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
const {chromium} = require(process.env.CORE_TEST_PLAYWRIGHT || 'playwright');
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../web');
const output = process.env.CORE_TEST_OUTPUT;
await mkdir(output, {recursive: true});
const urdf = await readFile(process.env.CORE_TEST_URDF, 'utf8');
const requests = [];
const topic = '/piper_arm1/piper/state/joints';
const mcp = {id: 'piper-arm1', name: 'PiPER', online: true, tools: [{name: 'joints',
  topic_out: [{topic, format: 'sensor/skeleton'}]}]};

// Exercise the actual Canvas handler's live-info and cached-topic branches,
// without booting unrelated project execution/editor-lock services.
const canvasSource = await readFile(path.join(web, 'js/canvas.js'), 'utf8');
function canvasFunction(name) {
  const start = canvasSource.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `Missing Canvas function ${name}`);
  const declaration = canvasSource.slice(start, canvasSource.indexOf('\n}\n', start) + 3);
  return name === '_openTopicDetailFor' ? `async ${declaration}` : declaration;
}
const canvasEntry = `
import {showTopicDetail} from '/js/detail-panel.js';
let _cards = [];
const _allMcps = [${JSON.stringify(mcp)}];
const _inputTopicsFor = () => [];
const inputArgs = () => ({});
const _showToast = message => { throw new Error(message); };
${canvasFunction('_parseMcpCallResult')}
${canvasFunction('_openTopicDetailFor')}
export async function open(live) {
  const el = document.createElement('div');
  const port = document.createElement('span');
  port.className = 'canvas-port out';
  port.dataset.topic = ${JSON.stringify(topic)};
  port.dataset.format = 'sensor/skeleton';
  el.appendChild(port);
  _cards = [{el, id: 'arm1-joints', toolName: live ? 'joints' : 'unavailable_info'}];
  await _openTopicDetailFor(el, ${JSON.stringify(mcp.id)}, []);
}
`;

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let data;
    if (url.pathname === '/api/topics') data = {data: [{topic, format: 'sensor/skeleton', mcp_id: mcp.id}]};
    else if (url.pathname === '/api/mcp') data = {data: [mcp]};
    else if (url.pathname === '/api/canvas/layout') data = {data: {cards: [
      {id: 'arm1-joints', mcpId: mcp.id, toolName: 'joints'}], connections: []}};
    else if (url.pathname.startsWith('/api/mcp/') && url.pathname.endsWith('/call')) {
      let body = ''; for await (const part of req) body += part;
      const id = decodeURIComponent(url.pathname.split('/')[3]), call = JSON.parse(body);
      requests.push({id, ...call});
      if (call.tool === 'joints') data = {code: 200, data: [{type: 'text', text: JSON.stringify({
        topic_out: [{topic, format: 'sensor/skeleton'}]})}]};
      else if (call.tool === 'unavailable_info' || id === 'failed-arm') {
        res.statusCode = 503; res.end('Unavailable'); return;
      } else {
        assert.equal(call.tool, 'model');
        if (id === 'slow-arm') await new Promise(resolve => setTimeout(resolve, 300));
        if (id === 'invalid-json') {
          res.setHeader('Content-Type', 'application/json'); res.end('{'); return;
        }
        const model = id === 'quadruped' ? {type: 'quadruped'}
          : id === 'empty-model' ? {}
          : {urdf: id === 'invalid-arm' ? '<robot name="empty"></robot>'
            : id === 'malformed-arm' ? '<robot><link></robot>' : urdf};
        data = id === 'missing-content' ? {code: 200, data: []}
          : {code: 200, data: [{type: 'text', text: JSON.stringify(model)}]};
      }
    }
    if (data) {res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(data)); return;}
    if (url.pathname === '/canvas-entry.js') {
      res.setHeader('Content-Type', 'application/javascript'); res.end(canvasEntry); return;
    }
    if (url.pathname === '/') {
      res.setHeader('Content-Type', 'text/html');
      res.end(`<meta charset="utf-8"><title>Robot model regression — simulated</title>
        <style>body{background:#eee;font:16px system-ui}.renderer-skeleton{width:100%;height:100%}
        #detail-panel{width:650px;height:520px}#detail-body{height:440px}.hidden{display:none}
        #monitor-dashboard-grid{width:650px;height:520px}.monitor-card-body{height:420px}
        #direct{width:600px;height:440px}</style>
        <div id="detail-panel"><button id="detail-close">Close</button><div id="detail-title"></div>
        <div id="detail-subtitle"></div><div id="detail-body"></div></div>
        <div id="monitor-dashboard-grid"></div><div id="direct"></div>
        <script type="importmap">{"imports":{"three":"/three.module.js","three/addons/controls/OrbitControls.js":"/OrbitControls.js"}}</script>
        <script type="module">
        window.models=[];window.sockets=[];window.direct=null;
        window.WebSocket=class {constructor(url){this.url=url;window.sockets.push(this);}close(){}};
        const {SkeletonRenderer}=await import('/js/renderers/skeleton.js');window.Skeleton=SkeletonRenderer;
        const mount=SkeletonRenderer.mount;
        SkeletonRenderer.mount=function(...args){window.models.push(this);return mount.apply(this,args);};
        window.detail=await import('/js/detail-panel.js');window.detail.initDetailPanel();
        window.canvasEntry=await import('/canvas-entry.js');
        window.dashboard=await import('/js/monitor-dashboard.js');window.ready=true;
        </script>`); return;
    }
    const file = ['/three.module.js', '/OrbitControls.js'].includes(url.pathname)
      ? path.join(process.env.CORE_TEST_THREE, path.basename(url.pathname)) : path.join(web, url.pathname);
    res.setHeader('Content-Type', file.endsWith('.js') ? 'application/javascript' : 'text/plain');
    res.end(await readFile(file));
  } catch (error) {res.statusCode = 500; res.end(String(error));}
});

await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({headless: true, executablePath: process.env.CORE_TEST_BROWSER,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport: {width: 1400, height: 1100}});
const errors = []; page.on('pageerror', error => errors.push(String(error)));
const passed = [];
try {
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(() => window.ready);
  for (const live of [true, false]) {
    const count = await page.evaluate(() => window.models.length);
    await page.evaluate(live => window.canvasEntry.open(live), live);
    await page.waitForFunction(count => window.models[count]?._loaded, count);
    assert.equal(requests.at(-1).id, mcp.id);
    assert.equal(requests.at(-1).tool, 'model');
    assert.deepEqual(await page.evaluate(() => Object.keys(window.models.at(-1)._joints)),
      ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']);
    passed.push(live ? 'Canvas live-info forwards device ID' : 'Canvas cached topic forwards device ID');
  }
  await page.evaluate(() => {
    window.sockets.at(-1).onmessage({data: new TextEncoder().encode(JSON.stringify({
      joints: [{name: 'joint1', q: .3}] })).buffer});
  });
  assert.ok(Math.abs(await page.evaluate(() => window.models.at(-1)._joints.joint1.quaternion.z) - Math.sin(.15)) < 1e-8);
  passed.push('PiPER joint animation');

  const count = await page.evaluate(() => window.models.length);
  await page.evaluate(() => window.dashboard.activate());
  await page.waitForFunction(count => window.models[count]?._loaded, count);
  assert.equal(requests.at(-1).id, mcp.id);
  await page.evaluate(() => {
    window.sockets.at(-1).onmessage({data: new TextEncoder().encode(JSON.stringify({
      joints: [{name: 'joint1', q: .3}] })).buffer});
  });
  await page.screenshot({path: path.join(output, 'piper-detail-and-monitor.png'), fullPage: true});
  passed.push('Top Monitor still loads PiPER');

  for (const id of ['failed-arm', 'invalid-arm', 'malformed-arm', 'empty-model',
    'missing-content', 'invalid-json', 'detail', 'dashboard', null]) {
    await page.evaluate(id => {
      window.direct?.unmount();
      window.direct = Object.create(window.Skeleton);
      window.direct.mount(document.getElementById('direct'), id);
    }, id);
    await page.waitForFunction(() => window.direct._modelNotice?.textContent &&
      !window.direct._modelNotice.textContent.includes('正在加载'));
    assert.equal(await page.evaluate(() => window.direct._loaded), false);
    assert.deepEqual(await page.evaluate(() => Object.keys(window.direct._joints)), []);
    assert.equal(await page.evaluate(() => window.direct._rootGroup), null);
    // Only the two lights and ground grid may remain, with no fallback humanoid.
    assert.equal(await page.evaluate(() => window.direct._scene.children.length), 3);
    passed.push(`Visible model error: ${id}`);
  }
  // The old two-argument public call remains valid but must not invent an ID.
  await page.evaluate(topic => window.detail.showTopicDetail(topic, 'sensor/skeleton'), topic);
  assert.equal(await page.evaluate(() => window.models.at(-1)._loaded), false);
  assert.ok(await page.evaluate(() => window.models.at(-1)._modelNotice.textContent.includes('无法确定设备')));
  await page.evaluate(() => {
    window.sockets.at(-1).onmessage({data: new TextEncoder().encode(JSON.stringify({joints: []})).buffer});
  });
  await page.screenshot({path: path.join(output, 'model-identity-error.png'), fullPage: true});
  passed.push('Missing owner is reported instead of showing a humanoid');

  await page.evaluate(async () => {
    await window.direct._loadModel('quadruped');
    window.direct.onData(new TextEncoder().encode(JSON.stringify({joints: [{name: 'FR_hip', q: .2}]})).buffer);
  });
  assert.equal(await page.evaluate(() => window.direct._loaded), true);
  assert.equal(await page.evaluate(() => Object.keys(window.direct._joints).length), 12);
  assert.ok(Math.abs(await page.evaluate(() => window.direct._joints.FR_hip.quaternion.y) - Math.sin(.1)) < 1e-8);
  await page.evaluate(async () => {await window.direct._loadModel('failed-arm');});
  assert.equal(await page.evaluate(() => window.direct._scene.children.length), 3);
  passed.push('Explicit quadruped model and animation remain supported');
  passed.push('Replacing a model releases the previous model');

  // Ignore AbortController deliberately: the generation guard must still stop
  // a late request from replacing the next selected model or reviving a panel.
  await page.evaluate(async () => {
    const original = window.fetch;
    window.fetch = (url, options) => original(url, String(url).includes('slow-arm') ? {...options, signal: undefined} : options);
    window.direct._loadModel('slow-arm');
    await window.direct._loadModel('piper/arm two');
  });
  await page.waitForFunction(() => window.direct._loaded);
  await new Promise(resolve => setTimeout(resolve, 400));
  assert.deepEqual(await page.evaluate(() => Object.keys(window.direct._joints)),
    ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']);
  assert.equal(await page.evaluate(() => window.direct._scene.children.filter(obj => obj === window.direct._rootGroup).length), 1);
  assert.ok(requests.some(request => request.id === 'piper/arm two'));
  passed.push('Encoded device IDs');
  passed.push('Late model response cannot replace current model');
  await page.evaluate(() => {window.direct._loadModel('slow-arm'); window.direct.unmount();});
  await new Promise(resolve => setTimeout(resolve, 400));
  assert.equal(await page.evaluate(() => window.direct._loaded), false);
  assert.equal(await page.evaluate(() => window.direct._scene), null);
  passed.push('Unmount aborts loading and prevents model resurrection');
  assert.deepEqual(errors, []);
  assert.ok(!requests.some(request => ['detail', 'dashboard'].includes(request.id)));
  await writeFile(path.join(output, 'model-browser-result.json'), JSON.stringify({requests, script_errors: errors, passed}, null, 2));
  console.log(`Robot model browser regression passed: ${passed.length} scenarios.`);
} finally {
  await page.evaluate(() => {
    window.direct?.unmount?.(); window.dashboard?.deactivate();
    document.getElementById('detail-close')?.click();
  }).catch(() => {});
  await browser.close(); server.close();
}
