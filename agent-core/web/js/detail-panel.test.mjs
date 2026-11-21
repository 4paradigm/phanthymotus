import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('./detail-panel.js', import.meta.url), 'utf8');
// Use the real point-cloud matcher and decoder; only mounting WebGL needs a stub.
const cloudSource = readFileSync(new URL('./renderers/pointcloud.js', import.meta.url), 'utf8')
  .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
const cloud = runInNewContext(cloudSource + '\nPointCloudRenderer;', {
  ArrayBuffer, DataView, Float32Array,
});

function harness() {
  const mounted = [], unmounted = [], sockets = [], timers = new Map();
  const imports = {};
  for (const [file, names] of [
    ['activity', ['ActivityRenderer']], ['text', ['TextRenderer']],
    ['video', ['VideoRenderer']], ['image', ['ImageRenderer']],
    ['audio', ['AudioRenderer']], ['lidar', ['LidarRenderer']],
    ['pointcloud', ['PointCloudRenderer']], ['control', ['ControlRenderer']],
    ['skeleton', ['SkeletonRenderer']], ['pose2d', ['Pose2dRenderer']],
    ['tactile', ['TactileRenderer']],
    ['camera', ['CameraRenderer', 'DepthRenderer', 'DepthZlibRenderer']],
  ]) {
    imports['./renderers/' + file + '.js'] = Object.fromEntries(names.map(name => {
      const renderer = {
        name: file,
        canRender: hint => name === 'ActivityRenderer'
          && (!hint || /^(sensor|control|state|data)\//.test(hint)),
        mount(body) { this.body = body; this.received = []; mounted.push(this); },
        onData(buffer, hint) { this.received.push({ buffer, hint }); },
        unmount() { unmounted.push(this); },
      };
      if (name === 'PointCloudRenderer') {
        renderer.canRender = cloud.canRender;
        renderer.mount = function(body) {
          this.body = body; this.received = []; this._frames = [];
          cloud._buildAxisMap.call(this, null);
          mounted.push(this);
        };
        renderer.onData = function(buffer, hint) {
          this.received.push({ buffer, hint });
          cloud.onData.call(this, buffer);
        };
      }
      return [name, renderer];
    }));
  }
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
    source.replace(/^import (\{[^}]+\})\s+from '([^']+)';\r?$/gm,
      (_, bindings, file) => 'const ' + bindings + ' = imports[' + JSON.stringify(file) + '];')
      .replace(/^export /gm, '') + '\n({ initDetailPanel, showTopicDetail });',
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

test('point-cloud details decode binary frames instead of falling through to the JSON log', () => {
  const h = harness();
  h.showTopicDetail('/nvidia_desktop/lidar/cloud', 'sensor/pointcloud');
  const renderer = h.mounted[0], ws = h.sockets[0];
  assert.equal(renderer.name, 'pointcloud');
  assert.equal(ws.url, 'wss://robot.local:15678/ws/bus/nvidia_desktop/lidar/cloud');
  assert.equal(ws.binaryType, 'arraybuffer');
  ws.onmessage({ data: JSON.stringify({ type: 'meta' }) });
  assert.equal(renderer.received.length, 0);
  const payload = frame();
  ws.onmessage({ data: payload });
  assert.equal(renderer.received[0].buffer, payload);
  assert.equal(renderer.received[0].hint, 'sensor/pointcloud');
  assert.equal(renderer._frames[0].count, 2);
  assert.deepEqual(Array.from(renderer._frames[0].pos), [2, -3, -1, 5, -6, -4]);
  const stale = [...h.timers.values()].find(timer => timer.delay === 10000);
  assert.ok(stale, 'point clouds must retain the visual-stream stale notice');
  stale.callback();
  assert.match(h.elements['detail-body'].children.at(-1).textContent, /画面是最后一帧/);
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
  assert.equal(h.mounted[0].name, 'activity');
  h.sockets[0].onmessage({ data: '{"state":"healthy"}' });
  assert.equal(new TextDecoder().decode(h.mounted[0].received[0].buffer), '{"state":"healthy"}');
  assert.equal(h.timers.size, 0);
});
