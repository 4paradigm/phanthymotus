import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const read = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const source = read('./detail-panel.js');

// The real canRender of every renderer, so match order is tested against the
// matchers that ship rather than against stubs that only Activity can pass.
// Module bodies are evaluated without their imports; mount/onData, where the
// real work happens, are replaced by the doubles below.
const real = {};
for (const file of ['activity', 'text', 'video', 'image', 'audio', 'lidar', 'control', 'pointcloud',
                    'mapping', 'skeleton', 'pose2d', 'tactile', 'camera']) {
  const body = read(`./renderers/${file}.js`)
    .replace(/^import[\s\S]*?;\r?$/gm, '').replace(/^export /gm, '');
  const names = [...body.matchAll(/^const (\w+Renderer) = /gm)].map(m => m[1]);
  // Some renderers build THREE objects in their literal (mapping's _robotPos),
  // so give them classes that construct to nothing.
  const THREE = new Proxy({}, { get: () => class {} });
  Object.assign(real, runInNewContext(`${body}\n({ ${names.join(', ')} })`,
    { console, THREE, OrbitControls: class {} }));
}

function harness() {
  const mounted = [], unmounted = [], sockets = [], timers = new Map();
  // One double per renderer, shared by every import site so identity checks
  // (detail-panel's TextRenderer/ActivityRenderer comparison) still hold.
  const doubles = Object.fromEntries(Object.entries(real).map(([name, renderer]) => [name, {
    name,
    canRender: renderer.canRender,
    mount(body, mcpId) { this.body = body; this.mcpId = mcpId; this.received = []; mounted.push(this); },
    onData(buffer, hint) { this.received.push({ buffer, hint }); },
    unmount() { unmounted.push(this); },
  }]));
  const fromDoubles = bindings => Object.fromEntries(
    bindings.replace(/[{}\s]/g, '').split(',').filter(Boolean).map(name => [name, doubles[name]]));
  const asConsts = (text, resolve) => text.replace(/^import (\{[^}]+\})\s+from '([^']+)';\r?$/gm,
    (_, bindings, file) => `const ${bindings} = imports[${JSON.stringify(resolve(file))}];`);
  const imports = {};
  // registry.js, evaluated for real so its order is the order under test.
  const registrySource = read('./renderers/registry.js');
  for (const [, bindings, file] of registrySource.matchAll(/^import (\{[^}]+\})\s+from '([^']+)';/gm)) {
    imports['./renderers/' + file.slice(2)] = fromDoubles(bindings);
  }
  imports['./renderers/registry.js'] = runInNewContext(
    asConsts(registrySource, file => './renderers/' + file.slice(2)).replace(/^export /gm, '')
      + '\n({ RENDERERS });', { imports });
  imports['./renderers/activity.js'] = { ActivityRenderer: doubles.ActivityRenderer };
  imports['./renderers/text.js'] = { TextRenderer: doubles.TextRenderer };
  imports['./mobile.js'] = { openDetailPanelMobile() {}, closeDetailPanelMobile() {} };
  const elements = Object.fromEntries(
    ['detail-panel', 'detail-close', 'detail-title', 'detail-subtitle', 'detail-body']
      .map(id => [id, {
        style: {}, classList: { add() {}, remove() {} }, children: [], listeners: {},
        addEventListener(type, callback) { this.listeners[type] = callback; },
        appendChild(child) { this.children.push(child); },
      }]),
  );
  let timerId = 0;
  const api = runInNewContext(
    asConsts(source, file => file).replace(/^export /gm, '') + '\n({ initDetailPanel, showTopicDetail });',
    {
      imports, ArrayBuffer, TextEncoder, console,
      location: { protocol: 'https:', host: 'robot.local:15678' },
      document: {
        getElementById: id => elements[id],
        createElement: () => ({ style: {}, remove() {} }),
      },
      WebSocket: class {
        constructor(url) { this.url = url; sockets.push(this); }
        close() { this.closed = true; }
      },
      setTimeout(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; },
      clearTimeout(id) { timers.delete(id); },
    },
  );
  api.initDetailPanel();
  return { ...api, mounted, unmounted, sockets, timers, elements };
}

function frame() {
  const buffer = new ArrayBuffer(8 + 2 * 12);
  const view = new DataView(buffer);
  view.setUint32(0, 12, true);
  view.setUint32(4, 2, true);
  [1, 2, 3, 4, 5, 6].forEach((value, index) => view.setFloat32(8 + index * 4, value, true));
  return buffer;
}

test('point-cloud details hand binary frames to the point-cloud renderer, not the JSON log', () => {
  const h = harness();
  h.showTopicDetail('/nvidia_desktop/lidar/cloud', 'sensor/pointcloud', 'as2w-driver');
  const renderer = h.mounted[0], ws = h.sockets[0];
  assert.equal(renderer.name, 'PointCloudRenderer');
  assert.equal(ws.url, 'wss://robot.local:15678/ws/bus/nvidia_desktop/lidar/cloud');
  assert.equal(ws.binaryType, 'arraybuffer');
  ws.onmessage({ data: JSON.stringify({ type: 'meta' }) });
  assert.equal(renderer.received.length, 0);
  const payload = frame();
  ws.onmessage({ data: payload });
  assert.equal(renderer.received[0].buffer, payload);
  assert.equal(renderer.received[0].hint, 'sensor/pointcloud');
  const stale = [...h.timers.values()].find(timer => timer.delay === 10000);
  assert.ok(stale, 'point clouds must retain the visual-stream stale notice');
  stale.callback();
  assert.match(h.elements['detail-body'].children.at(-1).textContent, /画面是最后一帧/);
});

test('the renderer is mounted with the owning device, so per-device settings apply', () => {
  const h = harness();
  h.showTopicDetail('/lidar/cloud', 'sensor/pointcloud', 'as2w-driver');
  assert.equal(h.mounted[0].mcpId, 'as2w-driver');
  // No device known: the monitor page's own "use the defaults" value.
  h.showTopicDetail('/lidar/cloud', 'sensor/pointcloud');
  assert.equal(h.mounted[1].mcpId, 'dashboard');
});

test('mapping topics get the mapping renderer, as on the monitor page', () => {
  const h = harness();
  h.showTopicDetail('/slam/map', 'sensor/mapping', 'slam');
  assert.equal(h.mounted[0].name, 'MappingRenderer');
});

test('the monitor page and the detail panel share one renderer list', () => {
  for (const file of ['./detail-panel.js', './monitor-dashboard.js']) {
    const text = read(file);
    assert.match(text, /^import \{ RENDERERS \}\s+from '\.\/renderers\/registry\.js';$/m, file);
    assert.doesNotMatch(text, /^const RENDERERS\b/m, `${file} must not keep its own copy`);
  }
});

test('switching and closing point-cloud details release the old renderer and subscription', () => {
  const h = harness();
  h.showTopicDetail('/lidar/first', 'sensor/pointcloud');
  const first = h.mounted[0], ws = h.sockets[0];
  ws.onmessage({ data: frame() });
  h.showTopicDetail('/lidar/second', 'sensor/pointcloud');
  assert.equal(h.unmounted[0], first);
  assert.equal(ws.closed, true);
  assert.equal(ws.onmessage, null);
  assert.notEqual(h.mounted[1], first);
  assert.equal(h.mounted[1].received.length, 0);
  h.elements['detail-close'].listeners.click();
  assert.equal(h.unmounted[1], h.mounted[1]);
  assert.equal(h.sockets[1].closed, true);
  assert.equal(h.timers.size, 0);
});

test('unrecognized sensor formats still use the activity renderer', () => {
  const h = harness();
  h.showTopicDetail('/sensor/state', 'sensor/vendor-data');
  assert.equal(h.mounted[0].name, 'ActivityRenderer');
  h.sockets[0].onmessage({ data: '{"state":"healthy"}' });
  assert.equal(new TextDecoder().decode(h.mounted[0].received[0].buffer), '{"state":"healthy"}');
  assert.equal(h.timers.size, 0);
});
