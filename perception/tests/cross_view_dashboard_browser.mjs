// Run against isolated Core and the cross_view_dashboard_fixture MCP service.
// Requires Node 22+ and a local Chromium binary.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [coreUrl, fixtureUrl, chrome, output] = process.argv.slice(2);
if (!output || !chrome || !fixtureUrl || !coreUrl) {
  throw new Error('Usage: node cross_view_dashboard_browser.mjs CORE_URL FIXTURE_URL CHROME SCREENSHOT');
}

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, description, timeout = 20000) {
  const end = Date.now() + timeout;
  let last;
  do {
    try {
      const result = await fn();
      if (result) return result;
    } catch (error) {
      last = error;
    }
    await wait(250);
  } while (Date.now() < end);
  throw new Error(description, { cause: last });
}

const profile = await mkdtemp(join(tmpdir(), 'cross-view-chrome-'));
const browser = spawn(chrome, ['--headless', '--disable-gpu', '--disable-background-networking',
  '--no-first-run', '--ignore-certificate-errors', '--remote-debugging-port=0',
  '--window-size=1600,1100', '--user-data-dir=' + profile, 'about:blank'],
  { windowsHide: true, stdio: 'ignore' });
const closed = new Promise(resolve => browser.once('close', resolve));
let ws;

try {
  const endpoint = await until(async () => {
    const raw = await readFile(join(profile, 'DevToolsActivePort'), 'utf8');
    const pages = await (await fetch('http://127.0.0.1:' + raw.split('\n')[0] + '/json/list')).json();
    return pages.find(page => page.type === 'page')?.webSocketDebuggerUrl;
  }, 'Chromium did not start');
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });
  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    const callback = pending.get(message.id);
    if (callback) {
      pending.delete(message.id);
      callback(message);
    }
  });
  const cdp = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++id, message => message.error ? reject(new Error(message.error.message)) : resolve(message.result));
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const { result, exceptionDetails } = await cdp('Runtime.evaluate',
      { expression, awaitPromise: true, returnByValue: true });
    if (exceptionDetails) throw new Error(result.description);
    return result.value;
  };
  const api = async (path, method = 'GET', payload) => {
    const args = JSON.stringify({ method, headers: { 'Content-Type': 'application/json' },
      body: payload === undefined ? undefined : JSON.stringify(payload) });
    const result = await evaluate(`fetch(${JSON.stringify('/api' + path)}, ${args}).then(r => r.json())`);
    if (result.ok !== undefined) {
      assert.equal(result.ok, true, JSON.stringify(result));
      return result;
    }
    assert.equal(result.code, 200, JSON.stringify(result));
    return result.data;
  };
  await cdp('Page.navigate', { url: coreUrl });
  await until(() => evaluate('!!document.querySelector("#canvas-area")'), 'Dashboard did not load');

  const { id: mcp } = await api('/mcp', 'POST', {
    name: 'Cross-view acceptance', url: fixtureUrl, category: 'perception',
  });
  const name = 'cross_view_object_consistency';
  const fixture = 'cross_view_fixture_';
  const mcpCall = async (tool, args) => {
    const data = await api(`/mcp/${mcp}/call`, 'POST', { tool, arguments: args });
    return JSON.parse((data.content || data)[0].text);
  };
  const discovered = await until(async () => {
    const rows = await api('/mcp');
    return rows.find(row => row.id === mcp)?.tools?.find(tool => tool.name === name);
  }, 'Core did not discover the Perception Card');
  assert.equal(discovered.topic_in.length, 6);
  const roles = ['rgb', 'objects', 'depth'];
  const formats = ['image/jpeg', 'data/json', 'image/depth-zlib'];
  const topology = (withDepth) => {
    const cards = [{id: 'cross-view-e2e', mcpId: mcp, toolName: name, driverName: 'Cross-view acceptance', x: 540, y: 170}];
    const connections = [];
    const topics = { a: {}, b: {} };
    for (const [sideIndex, side] of ['a', 'b'].entries()) {
      for (const [roleIndex, role] of roles.entries()) {
        if (role === 'depth' && !withDepth) continue;
        const id = `fixture-${side}-${role}`;
        const tool = fixture + side + '_' + role;
        const topic = `/cross_view_fixture/${side}/${role}`;
        const format = formats[roleIndex];
        topics[side][role] = topic;
        cards.push({id, mcpId: mcp, toolName: tool, driverName: 'Fixture',
          x: 40 + sideIndex * 245, y: 100 + roleIndex * 135, topicOut: [{topic, format}]});
        connections.push({id: 'link-' + id, fromCardId: id, toCardId: 'cross-view-e2e',
          fromPortIdx: '0', toPortIdx: String(sideIndex * 3 + roleIndex), format, fromTopic: topic});
      }
    }
    return { cards, connections, topics };
  };
  const cal = (camera, focal, offset, device) => ({
    camera_id: camera, world_frame: 'virtual', rectified: true, depth_registered: true,
    direction: 'camera_to_world', size: [640, 480], error_m: .003,
    K: [focal, 0, 320, 0, focal, 240, 0, 0, 1],
    T_world_camera: [1, 0, 0, offset, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    calibration_version: 'synthetic', device_id: device,
  });
  const session_id = 'cross-view-dashboard-acceptance';
  await api('/canvas/claim-edit', 'POST', {session_id});
  const configure = async withDepth => {
    const { cards, connections, topics } = topology(withDepth);
    const devices = ['a'.repeat(32), 'b'.repeat(32)];
    const rig = {camera_ids: ['a', 'b'], device_ids: devices, topics, time_error_ms: 5};
    if (withDepth) rig.calibration = {a: cal('a', 420, 0, devices[0]), b: cal('b', 510, .2, devices[1])};
    await api('/canvas/claim-edit', 'POST', {session_id});
    await api('/canvas/layout', 'POST', {session_id, cards, connections});
    await api(`/canvas/tool-config/${mcp}/${name}/cross-view-e2e`, 'PUT', {rig_json: JSON.stringify(rig)});
    await cdp('Page.reload');
    const card = await until(() => evaluate(`(() => {
      const card = [...document.querySelectorAll('.canvas-card')].find(
        card => card.querySelector('.canvas-card-title')?.textContent.includes(${JSON.stringify(name)})
          || card.dataset.cardId === 'cross-view-e2e');
      return card && card.querySelectorAll('.canvas-port.in').length;
    })()`), 'The six input ports did not render');
    assert.equal(card, 6);
    await api('/config/start-project', 'POST', {});
  };
  const diagnostic = async state => {
    return until(async () => {
      const result = await mcpCall(name, {action: 'info'});
      return result.diagnostic?.reason === state && result.diagnostic;
    }, 'Missing diagnostic: ' + state);
  };

  await configure(false);
  await diagnostic('missing_calibration');
  console.log('Chromium canvas displayed six ports and the RGB-only project reported missing calibration');
  await api('/config/stop-project', 'POST', {});
  await until(async () => (await mcpCall(name, {action: 'info'})).state === 'idle', 'RGB-only Card did not stop');

  await configure(true);
  await diagnostic('geometry_agrees');
  await evaluate(`document.querySelector('[data-mode="monitor"]').click()`);
  await until(() => evaluate(`document.querySelector('#monitor-dashboard-grid').textContent.includes('geometry_agrees')`),
    'The real Monitor did not render the ROS result');
  const screenshot = await cdp('Page.captureScreenshot', { format: 'png' });
  await writeFile(output, Buffer.from(screenshot.data, 'base64'));
  await mcpCall(fixture + 'b_rgb', {action: 'stop'});
  await diagnostic('camera_b_rgb_stream_stale');
  await until(() => evaluate(`document.querySelector('#monitor-dashboard-grid').textContent.includes('camera_b_rgb_stream_stale')`),
    'The real Monitor did not render the stale state');
  await api('/config/stop-project', 'POST', {});
  await until(async () => (await mcpCall(name, {action: 'info'})).state === 'idle', 'RGB-D Card did not stop');
  console.log('Chromium Dashboard/Monitor RGB-D, stale update, and project stop passed');
} finally {
  ws?.close();
  browser.kill();
  await closed;
  await rm(profile, { recursive: true, force: true, maxRetries: 5 }).catch(() => {});
}
