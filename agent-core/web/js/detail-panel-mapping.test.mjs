import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const panelSource = readFileSync(new URL('./detail-panel.js', import.meta.url), 'utf8');

// Run the real mapping mount, decoder and teardown with in-memory DOM/WebGL stubs.
function harness() {
  function element(tag = 'div') {
    return {
      tagName: tag.toUpperCase(), style: {}, children: [], listeners: {},
      clientWidth: 400, clientHeight: 300, classList: { add() {}, remove() {} },
      set innerHTML(value) { this.html = value; this.children = []; },
      get innerHTML() { return this.html || ''; },
      appendChild(child) { child.parent = this; this.children.push(child); },
      addEventListener(type, callback) { this.listeners[type] = callback; },
      remove() {
        const index = this.parent?.children.indexOf(this);
        if (index >= 0) this.parent.children.splice(index, 1);
      },
    };
  }
  class Vector3 {
    constructor(x = 0, y = 0, z = 0) { this.set(x, y, z); }
    set(x, y, z) { Object.assign(this, { x, y, z }); return this; }
    lerp() { return this; }
  }
  class Disposable { dispose() { this.disposed = true; } }
  class Geometry extends Disposable {
    constructor() { super(); this.attributes = {}; }
    setAttribute(name, attribute) { this.attributes[name] = attribute; }
    setDrawRange(start, count) { this.drawRange = { start, count }; }
    rotateZ() {}
  }
  class Points {
    constructor(geometry, material) { Object.assign(this, { geometry, material }); }
  }
  const THREE = {
    Vector3, BufferGeometry: Geometry, ConeGeometry: Geometry,
    PointsMaterial: Disposable, MeshBasicMaterial: Disposable, Points,
    Scene: class { add() {} }, Color: class {},
    PerspectiveCamera: class {
      constructor() { this.position = new Vector3(); }
      lookAt() {}
      updateProjectionMatrix() {}
    },
    WebGLRenderer: class extends Disposable {
      constructor() { super(); this.domElement = element('canvas'); }
      setSize() {}
      setPixelRatio() {}
      render() {}
    },
    BufferAttribute: class {
      constructor(array, itemSize) { Object.assign(this, { array, itemSize }); }
    },
    Mesh: class extends Points {
      constructor(...args) {
        super(...args); this.position = new Vector3(); this.rotation = new Vector3();
      }
    },
  };
  const elements = Object.fromEntries(
    ['detail-panel', 'detail-close', 'detail-title', 'detail-subtitle', 'detail-body']
      .map(id => [id, element()]),
  );
  const sockets = [], timers = new Map(), animations = new Map();
  let nextId = 0;
  const scope = {
    THREE, ArrayBuffer, DataView, Float32Array, TextEncoder, TextDecoder, console,
    window: { devicePixelRatio: 1 },
    location: { protocol: 'https:', host: 'robot.local:15678' },
    document: { getElementById: id => elements[id], createElement: element },
    OrbitControls: class extends Disposable {
      constructor() { super(); this.target = new Vector3(); }
      addEventListener() {}
      update() {}
    },
    ResizeObserver: class {
      observe() {}
      disconnect() { this.disconnected = true; }
    },
    requestAnimationFrame(callback) {
      animations.set(++nextId, callback); return nextId;
    },
    cancelAnimationFrame(id) { animations.delete(id); },
    setTimeout(callback, delay) {
      timers.set(++nextId, { callback, delay }); return nextId;
    },
    clearTimeout(id) { timers.delete(id); },
    WebSocket: class {
      constructor(url) { this.url = url; sockets.push(this); }
      close() { this.closed = true; }
    },
  };
  function loadRenderer(file, name) {
    const source = readFileSync(new URL('./renderers/' + file + '.js', import.meta.url), 'utf8')
      .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
    return runInNewContext(source + '\n' + name + ';', scope);
  }
  const imports = {
    './renderers/mapping.js': { MappingRenderer: loadRenderer('mapping', 'MappingRenderer') },
    './renderers/activity.js': { ActivityRenderer: loadRenderer('activity', 'ActivityRenderer') },
    './mobile.js': { openDetailPanelMobile() {}, closeDetailPanelMobile() {} },
  };
  for (const [file, names] of [
    ['text', ['TextRenderer']], ['video', ['VideoRenderer']], ['image', ['ImageRenderer']],
    ['audio', ['AudioRenderer']], ['lidar', ['LidarRenderer']],
    ['pointcloud', ['PointCloudRenderer']], ['control', ['ControlRenderer']],
    ['skeleton', ['SkeletonRenderer']], ['pose2d', ['Pose2dRenderer']],
    ['tactile', ['TactileRenderer']],
    ['camera', ['CameraRenderer', 'DepthRenderer', 'DepthZlibRenderer']],
  ]) {
    imports['./renderers/' + file + '.js'] = Object.fromEntries(
      names.map(name => [name, { canRender: () => false }]),
    );
  }
  const api = runInNewContext(
    panelSource.replace(/^import (\{[^}]+\})\s+from '([^']+)';\r?$/gm,
      (_, bindings, file) => 'const ' + bindings + ' = imports[' + JSON.stringify(file) + '];')
      .replace(/^export /gm, '')
      + '\n({ initDetailPanel, showTopicDetail, renderer: () => _renderer });',
    { ...scope, imports },
  );
  api.initDetailPanel();
  return { ...api, sockets, timers, animations, elements };
}

function snapshot() {
  // AS2W: little-endian <fffBI> header, flags=full_map|has_z, then XYZ float32.
  const buffer = new ArrayBuffer(17 + 2 * 12);
  const view = new DataView(buffer);
  [10, 20, 0.5].forEach((value, index) => view.setFloat32(index * 4, value, true));
  view.setUint8(12, 0x03);
  view.setUint32(13, 2, true);
  [1, 2, 3, 4, 5, 6].forEach((value, index) => view.setFloat32(17 + index * 4, value, true));
  return buffer;
}

test('mapping details mount a canvas and decode the AS2W map snapshot', () => {
  const h = harness();
  h.showTopicDetail('/nvidia_desktop/spatial/mapping', 'sensor/mapping');
  const renderer = h.renderer(), ws = h.sockets[0];
  assert.equal(renderer.name, 'mapping');
  assert.equal(renderer._el.className, 'renderer-mapping');
  assert.equal(renderer._renderer.domElement.tagName, 'CANVAS');
  assert.equal(ws.url, 'wss://robot.local:15678/ws/bus/nvidia_desktop/spatial/mapping');
  assert.equal(ws.binaryType, 'arraybuffer');
  ws.onmessage({ data: JSON.stringify({ type: 'meta' }) });
  assert.equal(renderer._points.geometry.drawRange.count, 0);
  ws.onmessage({ data: snapshot() });
  assert.equal(renderer._points.geometry.drawRange.count, 2);
  assert.deepEqual(Array.from(renderer._positions.subarray(0, 6)), [1, 3, -2, 4, 6, -5]);
  assert.equal(renderer._robotMesh.position.x, 10);
  assert.equal(renderer._robotMesh.position.z, -20);
  assert.equal(renderer._robotMesh.rotation.y, -0.5);
  const stale = [...h.timers.values()].find(timer => timer.delay === 10000);
  assert.ok(stale, 'maps must keep the visual-stream stale notice');
  stale.callback();
  assert.match(h.elements['detail-body'].children.at(-1).textContent, /画面是最后一帧/);
});

test('switching and closing mapping details dispose graphics and stop subscriptions', () => {
  const h = harness();
  h.showTopicDetail('/mapping/first', 'sensor/mapping');
  const first = h.renderer();
  assert.equal(first.name, 'mapping');
  h.sockets[0].onmessage({ data: snapshot() });
  const resources = [
    first._renderer, first._controls, first._points.geometry, first._points.material,
    first._robotMesh.geometry, first._robotMesh.material,
  ];
  const observer = first._ro;
  h.showTopicDetail('/mapping/second', 'sensor/mapping');
  assert.ok(resources.every(resource => resource.disposed));
  assert.equal(observer.disconnected, true);
  assert.equal(first._positions, null);
  assert.equal(h.sockets[0].closed, true);
  assert.equal(h.sockets[0].onmessage, null);
  assert.notEqual(h.renderer(), first);
  assert.equal(h.renderer()._points.geometry.drawRange.count, 0);
  const secondWebGL = h.renderer()._renderer;
  h.elements['detail-close'].listeners.click();
  assert.equal(secondWebGL.disposed, true);
  assert.equal(h.sockets[1].closed, true);
  assert.equal(h.renderer(), null);
  assert.equal(h.timers.size, 0);
  assert.equal(h.animations.size, 0);
});

test('unknown sensor formats still display JSON through the activity renderer', () => {
  const h = harness();
  h.showTopicDetail('/sensor/state', 'sensor/vendor-data');
  assert.equal(h.renderer().name, 'activity');
  h.sockets[0].onmessage({ data: '{"state":"healthy"}' });
  assert.equal(h.renderer()._el.children.length, 1);
  assert.match(h.renderer()._el.children[0].innerHTML, /healthy/);
  assert.equal(h.timers.size, 0);
});
