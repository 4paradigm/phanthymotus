import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const canvas = readFileSync(new URL('../web/js/canvas.js', import.meta.url), 'utf8');
const skeleton = readFileSync(new URL('../web/js/renderers/skeleton.js', import.meta.url), 'utf8');
const detailPanel = readFileSync(new URL('../web/js/detail-panel.js', import.meta.url), 'utf8');

function realTopicDetailHandoff() {
  const mounts = [], connections = [];
  const renderer = { canRender: () => true, mount: (_body, id) => mounts.push(id) };
  const body = { style: {}, appendChild() {} };
  const context = {
    _cleanup() {}, _panel: { classList: { remove() {} } },
    openDetailPanelMobile() {},
    document: { getElementById: id => id === 'detail-body' ? body : {},
      createElement: () => ({ style: {} }) },
    RENDERERS: [renderer], ActivityRenderer: {}, TextRenderer: {},
    _textLike: false, _renderer: null, _frames: 0, _status: null,
    _CENTERED: '', _setStatus() {}, _session: 1,
    _connect: (...args) => connections.push(args), console,
  };
  const start = detailPanel.indexOf('export function showTopicDetail(');
  assert.ok(start >= 0);
  const source = detailPanel.slice(start, detailPanel.indexOf('\n}\n', start) + 2)
    .replace('export function', 'function');
  const show = vm.runInNewContext(`(${source})`, context);
  return { show, mounts, connections };
}

function functionSource(name) {
  const start = canvas.indexOf(`async function ${name}(`);
  assert.ok(start >= 0);
  return canvas.slice(start, canvas.indexOf('\n}\n', start) + 2);
}

test('live stream details use the producing driver to fetch its model', async () => {
  const el = {};
  const panel = realTopicDetailHandoff();
  const context = {
    _cards: [{ el, id: 'card-left', toolName: 'glove_teleop' }],
    inputArgs: () => ({}), _inputTopicsFor: () => [],
    fetch: async () => ({ json: async () => ({ topic_out: [{ topic: '/hands', format: 'sensor/skeleton' }] }) }),
    _parseMcpCallResult: value => value,
    showTopicDetail: panel.show,
    console,
  };
  const open = vm.runInNewContext(`(${functionSource('_openTopicDetailFor')})`, context);
  await open(el, 'mcp-glove', []);
  assert.deepEqual(panel.mounts, ['mcp-glove']);
  assert.deepEqual(panel.connections, [['/hands', 'sensor/skeleton', 1]]);
});

test('cached stream details retain the same model provider', async () => {
  const panel = realTopicDetailHandoff();
  const context = {
    _cards: [], _allMcps: [], console,
    showTopicDetail: panel.show,
    _showToast: () => assert.fail('a cached topic exists'),
  };
  const el = { querySelectorAll: () => [] };
  const open = vm.runInNewContext(`(${functionSource('_openTopicDetailFor')})`, context);
  await open(el, 'mcp-glove', [{ topic: '/hands', format: 'sensor/skeleton' }]);
  assert.deepEqual(panel.mounts, ['mcp-glove']);
  assert.deepEqual(panel.connections, [['/hands', 'sensor/skeleton', 1]]);
});

test('topic detail remains compatible with callers without a model provider', () => {
  const panel = realTopicDetailHandoff();
  panel.show('/log', 'text');
  assert.deepEqual(panel.mounts, ['detail']);
});

test('generic execute button sends the bound card identity', async () => {
  const start = canvas.indexOf("    const execBtn = el.querySelector('.canvas-exec-btn');");
  const end = canvas.indexOf('    const viewBtn =', start);
  assert.ok(start >= 0 && end > start);
  let click;
  const calls = [];
  const context = {
    el: { querySelector: () => ({ addEventListener: (_event, callback) => { click = callback; } }) },
    id: 'left-instance', mcpId: 'hands', toolName: 'hand',
    _ensureEdit: async () => true,
    _executeCard: (...args) => calls.push(args),
  };
  vm.runInNewContext(canvas.slice(start, end), context);
  await click({ stopPropagation() {} });
  assert.equal(calls.length, 1);
  assert.equal(calls[0][3], 'left-instance');
});

function rotate(vector, axis, angle) {
  const [x, y, z] = vector;
  const c = Math.cos(angle), s = Math.sin(angle);
  if (axis === 'X') return [x, c*y-s*z, s*y+c*z];
  if (axis === 'Y') return [c*x+s*z, y, -s*x+c*z];
  return [c*x-s*y, s*x+c*y, z];
}

function previewRenderer() {
  const context = { TextDecoder };
  const source = skeleton.replace(/^import .*;\s*$/gm, '')
    .replace('export const SkeletonRenderer =', 'globalThis.renderer =');
  vm.runInNewContext(source, context);
  const renderer = context.renderer;
  renderer._loaded = true;
  renderer._gloveTargets = {};
  renderer._lastAnimationAt = null;
  const angles = [];
  renderer._joints = { finger: { userData: { axis: {} },
    quaternion: { setFromAxisAngle: (_axis, value) => angles.push(value) } } };
  const send = (q, glove = true) => renderer.onData(Buffer.from(JSON.stringify({
    preview_is_glove_target: glove, joints: [{ name: 'finger', q }],
  })));
  return { renderer, angles, send };
}

test('glove display interpolates to the latest target without overshoot or extrapolation', () => {
  const { renderer, angles, send } = previewRenderer();
  send(0);
  renderer._advanceGlovePreview(0);
  send(1);
  renderer._advanceGlovePreview(16);
  assert.ok(angles.at(-1) > 0 && angles.at(-1) < 1);
  send(0.2); // A newer frame replaces the previous target, without queuing it.
  for (let t = 32; t < 1000; t += 16) renderer._advanceGlovePreview(t);
  assert.ok(Math.abs(angles.at(-1) - 0.2) < 1e-9);
  assert.ok(angles.every(a => a >= 0 && a <= 1));
});

test('glove interpolation is independent of display refresh rate', () => {
  const valueAfter = step => {
    const { renderer, angles, send } = previewRenderer();
    send(0); renderer._advanceGlovePreview(0); send(1);
    for (let t = step; t <= 160; t += step) renderer._advanceGlovePreview(t);
    return angles.at(-1);
  };
  assert.ok(Math.abs(valueAfter(8) - valueAfter(16)) < 1e-10);
});

test('other robot streams retain direct joint updates and cancel glove smoothing', () => {
  const { renderer, angles, send } = previewRenderer();
  send(0); send(1);
  send(-0.5, false);
  assert.equal(angles.at(-1), -0.5);
  renderer._advanceGlovePreview(100);
  assert.equal(angles.at(-1), -0.5);
});

for (const yaw of [1.8064, -1.8076]) {
  test(`URDF thumb origin uses fixed-axis RPY, yaw=${yaw}`, () => {
    class Group {
      constructor() {
        this.children = [];
        this.position = { set() {} };
        this.rotation = { copy(euler) { this.euler = euler; } };
      }
      add(child) { child.parent = this; this.children.push(child); }
    }
    class Euler {
      constructor(x, y, z, order) { Object.assign(this, { x, y, z, order }); }
    }
    const attributes = map => ({ getAttribute: key => map[key] });
    const joint = {
      getAttribute: key => ({ name: 'thumb_origin', type: 'fixed' })[key],
      querySelector: tag => ({
        parent: attributes({ link: 'palm' }), child: attributes({ link: 'thumb' }),
        origin: attributes({ xyz: '0 0 0', rpy: `0 -0.9163 ${yaw}` }),
        axis: attributes({ xyz: '0 1 0' }),
      })[tag],
    };
    class DOMParser {
      parseFromString() {
        return { querySelectorAll: tag => tag === 'link'
          ? [attributes({ name: 'palm' }), attributes({ name: 'thumb' })] : [joint] };
      }
    }
    const context = {
      DOMParser,
      THREE: { Group, Euler, Vector3: class {}, Mesh: Group,
        SphereGeometry: class {}, MeshPhongMaterial: class {} },
    };
    const source = skeleton.replace(/^import .*;\s*$/gm, '')
      .replace('export const SkeletonRenderer =', 'globalThis.renderer =');
    vm.runInNewContext(source, context);
    const renderer = context.renderer;
    renderer._scene = { add() {} };
    renderer._joints = {}; renderer._links = {};
    renderer._addBoneVisual = () => {};
    renderer._parseAndBuild('<synthetic-urdf/>');
    const euler = renderer._links.thumb.parent.rotation.euler;
    let actual = [1, 0, 0];
    for (const axis of [...euler.order].reverse()) actual = rotate(actual, axis, euler[axis.toLowerCase()]);
    const expected = [Math.cos(yaw)*Math.cos(-0.9163), Math.sin(yaw)*Math.cos(-0.9163), -Math.sin(-0.9163)];
    actual.forEach((value, index) => assert.ok(Math.abs(value-expected[index]) < 1e-10));
  });
}
